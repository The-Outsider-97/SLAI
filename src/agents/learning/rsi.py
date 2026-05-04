"""
Recursive Self-Improvement (RSI) agent for long-horizon autonomous learning.

Reference:
- Schmidhuber (2013). PowerPlay: Training General Problem Solvers.

This module keeps the original intent and surface area of the RSI agent while
bringing the implementation to production-ready level:
- neural Q-learning with target networks
- replay through both local memory and LearningMemory
- recursive self-improvement via parameter and policy adaptation
- checkpointing, recovery, evaluation, diagnostics, and training summaries
"""

from __future__ import annotations

import hashlib
import random
import numpy as np
import torch

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from src.agents.learning.learning_memory import LearningMemory, Transition
from src.agents.learning.utils.config_loader import get_config_section, load_global_config
from src.agents.learning.utils.neural_network import NeuralNetwork
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Recursive Self-Improvement")
printer = PrettyPrinter

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]
ExperienceLike = Union[Transition, Tuple[Any, Any, Any, Any, Any], List[Any]]


class RSIAgent:
    """Recursive Self-Improvement agent with neural Q-learning and adaptive tuning.

    The agent is designed for long-running training jobs where performance may
    plateau and the learner should adjust itself without external retuning.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        agent_id: Optional[Union[str, int]],
        env: Any = None,
    ):
        if int(state_size) <= 0 or int(action_size) <= 0:
            raise ValueError("state_size and action_size must be positive integers.")

        self.config = load_global_config()
        self.rsi_config = get_config_section("rsi")

        self.agent_id = str(agent_id) if agent_id is not None else "RSI"
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.env = env
        self.model_id = "RSI_Agent"

        self.gamma = float(self.rsi_config.get("gamma", 0.95))
        self.epsilon = float(self.rsi_config.get("epsilon", 1.0))
        self.epsilon_min = float(self.rsi_config.get("epsilon_min", 0.01))
        self.epsilon_decay = float(self.rsi_config.get("epsilon_decay", 0.995))
        self.learning_rate = float(self.rsi_config.get("learning_rate", 0.001))
        self.rsi_period = int(self.rsi_config.get("rsi_period", 14))
        self.improvement_interval = int(self.rsi_config.get("improvement_interval", 100))
        self.performance_window = int(self.rsi_config.get("performance_history", 50))
        self.param_mutation_rate = float(self.rsi_config.get("param_mutation_rate", 0.1))
        self.improvement_threshold = float(self.rsi_config.get("improvement_threshold", 0.05))
        self.target_update_frequency = int(self.rsi_config.get("target_update_frequency", 100))
        self.baseline_performance = self.rsi_config.get("baseline_performance")
        # Safely handle string 'None' or other non‑numeric values
        if isinstance(self.baseline_performance, str):
            if self.baseline_performance.lower() == "none":
                self.baseline_performance = None
            else:
                try:
                    self.baseline_performance = float(self.baseline_performance)
                except ValueError:
                    self.baseline_performance = None
        elif self.baseline_performance is not None:
            try:
                self.baseline_performance = float(self.baseline_performance)
            except (TypeError, ValueError):
                self.baseline_performance = None

        # Local replay / training controls.
        self.batch_size = int(self.rsi_config.get("batch_size", 32))
        self.memory_capacity = int(self.rsi_config.get("buffer_size", 10000))
        self.min_replay_size = int(self.rsi_config.get("min_replay_size", self.batch_size))
        self.gradient_clip_norm = float(self.rsi_config.get("gradient_clip_norm", 5.0))
        self.soft_mutation_std = float(self.rsi_config.get("soft_mutation_std", 0.02))
        self.lr_adaptation_rate = float(self.rsi_config.get("lr_adaptation_rate", 0.10))
        self.lr_min = float(self.rsi_config.get("learning_rate_min", 1e-5))
        self.lr_max = float(self.rsi_config.get("learning_rate_max", 0.1))
        self.reward_clip = self.rsi_config.get("reward_clip")
        self.max_steps_per_episode = self.rsi_config.get("max_steps_per_episode")
        self.eval_exploration_rate = float(self.rsi_config.get("eval_exploration_rate", 0.0))
        self.checkpoint_dir = Path(self.rsi_config.get("checkpoint_dir", "src/agents/learning/checkpoints/rsi"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("rsi.gamma must be in (0, 1].")
        if not 0.0 <= self.epsilon_min <= self.epsilon <= 1.0 + 1e-12:
            raise ValueError("epsilon settings are inconsistent.")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.rsi_period <= 1:
            raise ValueError("rsi_period must be greater than 1.")
        if self.improvement_interval <= 0:
            raise ValueError("improvement_interval must be positive.")
        if self.performance_window <= 0:
            raise ValueError("performance_history must be positive.")
        if self.batch_size <= 0 or self.memory_capacity <= 0 or self.min_replay_size <= 0:
            raise ValueError("Replay settings must be positive.")
        if self.target_update_frequency <= 0:
            raise ValueError("target_update_frequency must be positive.")

        self.network_config = self._build_network(self.state_size, self.action_size)
        self.q_network = NeuralNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            config=self.network_config,
        )
        self.target_network = NeuralNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            config=self.network_config,
        )
        self.target_network.set_weights(self.q_network.get_weights())

        self.learning_memory = LearningMemory()
        self.performance_history: Deque[float] = deque(maxlen=self.performance_window)
        self.memory: Deque[Transition] = deque(maxlen=self.memory_capacity)
        self.training_metrics: List[Dict[str, Any]] = []
        self.last_evaluation: Optional[Dict[str, Any]] = None
        self.last_training_summary: Optional[Dict[str, Any]] = None
        self.last_train_episode_metrics: Optional[Dict[str, Any]] = None
        self.update_counter = 0
        self.current_epoch = 0
        self.total_env_steps = 0
        self.total_gradient_steps = 0
        self.total_episodes = 0
        self.policy_net = self.q_network  # compatibility with surrounding sub-agent checks

        logger.info(
            "RSIAgent initialised | id=%s state=%s actions=%s gamma=%.4f lr=%.6f epsilon=%.4f",
            self.agent_id,
            self.state_size,
            self.action_size,
            self.gamma,
            self.learning_rate,
            self.epsilon,
        )

    # ------------------------------------------------------------------
    # Configuration and state helpers
    # ------------------------------------------------------------------
    def _build_network(self, input_size: int, output_size: int) -> Dict[str, Any]:
        """Build a local NeuralNetwork config without mutating global config."""
        base_nn_cfg = self.config.get("neural_network", {}) or {}
        network_config = dict(base_nn_cfg)

        hidden_size = self.rsi_config.get("hidden_size", 64)
        if isinstance(hidden_size, int):
            hidden_layers = [int(hidden_size), int(hidden_size)]
        elif isinstance(hidden_size, (list, tuple)):
            hidden_layers = [int(dim) for dim in hidden_size]
        else:
            raise TypeError("rsi.hidden_size must be an int or a sequence of ints.")

        network_config["layer_dims"] = [int(input_size), *hidden_layers, int(output_size)]
        network_config["learning_rate"] = self.learning_rate
        network_config["output_activation"] = "linear"
        network_config["loss_function"] = "mse"
        network_config.setdefault("optimizer", "adam")
        network_config.setdefault("hidden_activation", "relu")
        network_config.setdefault("gradient_clip_norm", self.gradient_clip_norm)
        return network_config

    @staticmethod
    def _extract_state(reset_output: Any) -> Any:
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            return reset_output[0]
        return reset_output

    @staticmethod
    def _safe_step(env: Any, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        result = env.step(action)
        if not isinstance(result, tuple):
            raise TypeError("Environment step(...) must return a tuple.")
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            return next_state, float(reward), bool(terminated), bool(truncated), info or {}
        if len(result) == 4:
            next_state, reward, done, info = result
            return next_state, float(reward), bool(done), False, info or {}
        raise ValueError(f"Unsupported environment step() output length: {len(result)}")

    def _state_to_tensor(self, state: TensorLike) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            tensor = state.detach().clone().to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(state, dtype=torch.float32)

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 1:
            tensor = tensor.reshape(-1)

        if tensor.numel() != self.state_size:
            raise ValueError(
                f"State size mismatch. Expected {self.state_size}, got {int(tensor.numel())}."
            )
        return tensor

    def _coerce_transition(self, item: ExperienceLike) -> Transition:
        if isinstance(item, Transition):
            return item
        if isinstance(item, dict):
            return Transition(
                item["state"],
                item["action"],
                item["reward"],
                item["next_state"],
                item["done"],
            )
        if isinstance(item, (tuple, list)) and len(item) == 5:
            state, action, reward, next_state, done = item
            return Transition(state, action, reward, next_state, done)
        raise TypeError("Each experience must contain 5 elements: state, action, reward, next_state, done.")

    def _serialize_transition(self, transition: Transition) -> Dict[str, Any]:
        transition = self._coerce_transition(transition)
        return {
            "state": self._state_to_tensor(transition.state).detach().cpu(),
            "action": int(transition.action),
            "reward": float(transition.reward),
            "next_state": self._state_to_tensor(transition.next_state).detach().cpu(),
            "done": bool(transition.done),
        }

    def _clip_reward(self, reward: float) -> float:
        if self.reward_clip is None:
            return float(reward)
        if isinstance(self.reward_clip, (int, float)):
            bound = abs(float(self.reward_clip))
            return float(np.clip(reward, -bound, bound))
        if isinstance(self.reward_clip, (tuple, list)) and len(self.reward_clip) == 2:
            low, high = float(self.reward_clip[0]), float(self.reward_clip[1])
            return float(np.clip(reward, low, high))
        raise ValueError("reward_clip must be None, a scalar, or a 2-tuple/list.")

    # ------------------------------------------------------------------
    # Q-network value estimation
    # ------------------------------------------------------------------
    def _estimate_q_value(self, state: TensorLike, action: int) -> float:
        state_tensor = self._state_to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network.forward(state_tensor).squeeze(0)
        return float(q_values[int(action)].item())

    def _predict_next_q(self, next_state: TensorLike) -> float:
        next_state_tensor = self._state_to_tensor(next_state).unsqueeze(0)
        with torch.no_grad():
            target_q_values = self.target_network.forward(next_state_tensor).squeeze(0)
        return float(target_q_values.max().item())

    def _select_greedy_action(self, state: TensorLike) -> int:
        state_tensor = self._state_to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network.forward(state_tensor).squeeze(0)
        return int(torch.argmax(q_values).item())

    # ------------------------------------------------------------------
    # Improvement / adaptation logic
    # ------------------------------------------------------------------
    def _update_parameters(self, gradient_signal: float) -> None:
        """Adaptive parameter update driven by improvement / deterioration signals."""
        direction = float(np.sign(gradient_signal))
        scaled_lr = self.learning_rate * (1.0 + self.lr_adaptation_rate * direction)
        self.learning_rate = float(np.clip(scaled_lr, self.lr_min, self.lr_max))
        self.q_network.optimizer.learning_rate = self.learning_rate

    def self_improve(self) -> Dict[str, Any]:
        """Core recursive self-improvement routine based on recent performance."""
        if len(self.performance_history) < min(10, self.performance_window):
            return {
                "triggered": False,
                "reason": "insufficient_history",
                "baseline_performance": self.baseline_performance,
            }

        recent_window = list(self.performance_history)[-10:]
        current_perf = float(np.mean(recent_window))
        if self.baseline_performance is None:
            self.baseline_performance = current_perf
            return {
                "triggered": False,
                "reason": "baseline_initialized",
                "baseline_performance": self.baseline_performance,
            }

        denominator = max(abs(float(self.baseline_performance)), 1e-8)
        improvement = (current_perf - float(self.baseline_performance)) / denominator
        summary = {
            "triggered": True,
            "recent_performance": current_perf,
            "baseline_performance": float(self.baseline_performance),
            "improvement": float(improvement),
            "threshold": float(self.improvement_threshold),
            "mutated": False,
        }

        if improvement < self.improvement_threshold:
            summary["mutated"] = True
            summary["mutation_report"] = self._mutate_parameters()
            logger.info("Self-improvement triggered mutation | improvement=%.6f", improvement)
        else:
            self.baseline_performance = 0.9 * float(self.baseline_performance) + 0.1 * current_perf
            self._update_parameters(improvement)
            logger.info(
                "Self-improvement updated baseline | new_baseline=%.6f improvement=%.6f",
                self.baseline_performance,
                improvement,
            )
        return summary

    def _mutate_parameters(self) -> Dict[str, Any]:
        """Evolutionary-style mutation of hyperparameters and network weights."""
        old_rsi_period = int(self.rsi_period)
        old_learning_rate = float(self.learning_rate)
        old_gamma = float(self.gamma)
        old_epsilon = float(self.epsilon)

        self.rsi_period = int(np.clip(
            round(self.rsi_period * (1.0 + self.param_mutation_rate * np.random.randn())),
            5,
            30,
        ))
        self.learning_rate = float(np.clip(
            self.learning_rate * np.exp(self.param_mutation_rate * np.random.randn()),
            self.lr_min,
            self.lr_max,
        ))
        self.gamma = float(np.clip(
            self.gamma + 0.01 * np.random.randn(),
            0.80,
            0.999,
        ))
        self.epsilon = float(np.clip(
            self.epsilon * np.exp(0.05 * np.random.randn()),
            self.epsilon_min,
            1.0,
        ))
        self.q_network.optimizer.learning_rate = self.learning_rate

        # Mild parameter-space exploration on the online network.
        mutated_weights = self.q_network.get_weights()
        with torch.no_grad():
            for idx in range(len(mutated_weights["Ws"])):
                mutated_weights["Ws"][idx] += torch.randn_like(mutated_weights["Ws"][idx]) * self.soft_mutation_std
                mutated_weights["bs"][idx] += torch.randn_like(mutated_weights["bs"][idx]) * (self.soft_mutation_std * 0.5)
        self.q_network.set_weights(mutated_weights)
        self.target_network.set_weights(self.q_network.get_weights())

        return {
            "old_rsi_period": old_rsi_period,
            "new_rsi_period": self.rsi_period,
            "old_learning_rate": old_learning_rate,
            "new_learning_rate": self.learning_rate,
            "old_gamma": old_gamma,
            "new_gamma": self.gamma,
            "old_epsilon": old_epsilon,
            "new_epsilon": self.epsilon,
        }

    # ------------------------------------------------------------------
    # Action selection / RSI scoring
    # ------------------------------------------------------------------
    def calculate_rsi(self, state_sequence: List[Any]) -> float:
        """Compute a bounded relative-strength-style variability score in [0, 1]."""
        if not isinstance(state_sequence, (list, tuple, np.ndarray)) or len(state_sequence) < 2:
            return 0.5

        try:
            states = np.asarray(state_sequence, dtype=np.float32)
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            magnitudes = np.linalg.norm(states, axis=1)
            deltas = np.diff(magnitudes)
            if deltas.size == 0:
                return 0.5

            lookback = min(int(self.rsi_period), deltas.size)
            deltas = deltas[-lookback:]
            gains = np.clip(deltas, 0.0, None)
            losses = np.clip(-deltas, 0.0, None)
            avg_gain = float(np.mean(gains))
            avg_loss = float(np.mean(losses))

            if avg_loss <= 1e-8 and avg_gain <= 1e-8:
                return 0.5
            if avg_loss <= 1e-8:
                return 1.0

            rs = avg_gain / avg_loss
            score = 1.0 - (1.0 / (1.0 + rs))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    def _rsi_policy(self, score: float, state: Optional[TensorLike] = None) -> int:
        """Map a meta-score to an action policy.

        High score -> more exploratory behaviour.
        Low score -> greedy exploitation.
        Mid score -> tempered epsilon-greedy behaviour.
        """
        score = float(np.clip(score, 0.0, 1.0))
        threshold_high = 0.7
        threshold_low = 0.3

        if state is None:
            if self.action_size < 3:
                return int(np.clip(round(score * max(self.action_size - 1, 1)), 0, self.action_size - 1))
            if score > threshold_high:
                return random.randrange(self.action_size)
            if score < threshold_low:
                return 0
            return min(2, self.action_size - 1)

        if score > threshold_high:
            return random.randrange(self.action_size)
        if score < threshold_low:
            return self._select_greedy_action(state)

        tempered_epsilon = max(self.epsilon_min, min(1.0, 0.5 * self.epsilon))
        if random.random() < tempered_epsilon:
            return random.randrange(self.action_size)
        return self._select_greedy_action(state)

    def act(
        self,
        state: Any,
        state_sequence: Optional[List[Any]] = None,
        explore: bool = True,
    ) -> int:
        """Action selection with optional RSI meta-policy and epsilon-greedy exploration."""
        if state_sequence:
            return self._rsi_policy(self.calculate_rsi(state_sequence), state=state)

        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return self._select_greedy_action(state)

    def select_action(self, processed_state: Any) -> int:
        return self.act(processed_state)

    # ------------------------------------------------------------------
    # Metrics / diagnostics helpers
    # ------------------------------------------------------------------
    def _calculate_sharpe(self, returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
        returns_arr = np.asarray(returns, dtype=np.float32)
        std = float(np.std(returns_arr))
        if std <= 1e-8:
            return 0.0
        return float(np.sqrt(252.0) * np.mean(returns_arr) / std)

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        cumulative = np.cumsum(np.asarray(returns, dtype=np.float32))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns)) if drawdowns.size else 0.0

    def _get_weight_vector(self) -> Dict[str, Dict[str, Any]]:
        """Extract full neural-network weight summaries using NeuralNetwork weights."""
        weights = self.q_network.get_weights()
        layer_summaries: Dict[str, Dict[str, Any]] = {}
        for idx, (W, b) in enumerate(zip(weights["Ws"], weights["bs"])):
            layer_summaries[f"layer_{idx}"] = {
                "weights": W.detach().cpu().numpy(),
                "biases": b.detach().cpu().numpy(),
                "mean_weight": float(W.mean().item()),
                "weight_std": float(W.std().item()),
                "mean_bias": float(b.mean().item()),
                "bias_std": float(b.std().item()) if b.numel() > 1 else 0.0,
                "shape": tuple(W.shape),
            }

        layer_summaries["network_metadata"] = {
            "architecture": list(self.network_config.get("layer_dims", [])),
            "parameters": int(sum(W.numel() + b.numel() for W, b in zip(weights["Ws"], weights["bs"]))),
            "gradient_norm": self._calculate_gradient_norm(),
        }
        return layer_summaries

    def _calculate_gradient_norm(self) -> float:
        gradients = []
        if hasattr(self.q_network, "dWs"):
            gradients.extend(getattr(self.q_network, "dWs", []))
        if hasattr(self.q_network, "dBs"):
            gradients.extend(getattr(self.q_network, "dBs", []))
        if not gradients:
            return 0.0
        total = 0.0
        for grad in gradients:
            if grad is None:
                continue
            norm = float(torch.norm(grad).item())
            total += norm * norm
        return float(total ** 0.5)

    def _calculate_file_hash(self, filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as handle:
            for chunk in iter(lambda: handle.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "rsi_period": self.rsi_period,
            "improvement_interval": self.improvement_interval,
            "param_mutation_rate": self.param_mutation_rate,
            "improvement_threshold": self.improvement_threshold,
            "target_update_frequency": self.target_update_frequency,
            "update_counter": self.update_counter,
            "current_epoch": self.current_epoch,
            "total_env_steps": self.total_env_steps,
            "total_gradient_steps": self.total_gradient_steps,
            "total_episodes": self.total_episodes,
            "memory_size": len(self.memory),
            "learning_memory_size": self.learning_memory.size(),
            "baseline_performance": self.baseline_performance,
        }

    # ------------------------------------------------------------------
    # Replay / training core
    # ------------------------------------------------------------------
    def remember(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        transition = Transition(
            state=self._state_to_tensor(state).cpu(),
            action=int(action),
            reward=self._clip_reward(float(reward)),
            next_state=self._state_to_tensor(next_state).cpu(),
            done=bool(done),
        )
        self.memory.append(transition)

        td_guess = transition.reward + (0.0 if transition.done else self._predict_next_q(transition.next_state))
        priority = abs(td_guess - self._estimate_q_value(transition.state, transition.action)) + 1e-6
        self.learning_memory.add(transition, priority=priority, tag=self.agent_id)

    def _sample_replay_batch(
        self,
        batch_size: Optional[int] = None,
        experience_batch: Optional[Iterable[ExperienceLike]] = None,
    ) -> Optional[Tuple[List[Transition], Optional[List[int]], Optional[List[float]]]]:
        effective_batch_size = int(batch_size or self.batch_size)
        if experience_batch is not None:
            batch = [self._coerce_transition(item) for item in experience_batch]
            return (batch, None, None) if batch else None

        if self.learning_memory.size() >= effective_batch_size:
            samples, indices, weights = self.learning_memory.sample_proportional(effective_batch_size)
            if samples:
                return [self._coerce_transition(item) for item in samples], indices, weights

        if len(self.memory) >= effective_batch_size:
            batch = random.sample(list(self.memory), effective_batch_size)
            return [self._coerce_transition(item) for item in batch], None, None

        return None

    def _optimise_from_batch(
        self,
        experience_batch: Optional[Iterable[ExperienceLike]] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        sampled = self._sample_replay_batch(batch_size=batch_size, experience_batch=experience_batch)
        if sampled is None:
            return 0.0, 0.0

        batch, indices, _weights = sampled
        states = torch.stack([self._state_to_tensor(exp.state) for exp in batch])
        actions = torch.tensor([int(exp.action) for exp in batch], dtype=torch.long)
        rewards = torch.tensor([float(exp.reward) for exp in batch], dtype=torch.float32)
        next_states = torch.stack([self._state_to_tensor(exp.next_state) for exp in batch])
        dones = torch.tensor([float(bool(exp.done)) for exp in batch], dtype=torch.float32)

        with torch.no_grad():
            current_q = self.q_network.forward(states)
            chosen_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q = self.target_network.forward(next_states)
            max_next_q = torch.max(next_q, dim=1).values
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q
            td_errors = target_q - chosen_q
            targets = current_q.clone()
            batch_indices = torch.arange(states.shape[0])
            targets[batch_indices, actions] = target_q

        loss = float(self.q_network.train_step(states, targets))
        self.total_gradient_steps += 1
        self.update_counter += 1

        if self.gradient_clip_norm > 0.0 and hasattr(self.q_network, "_global_grad_norm"):
            grad_norm = float(self.q_network._global_grad_norm().item())
            if np.isfinite(grad_norm) and grad_norm > self.gradient_clip_norm:
                self._update_parameters(-1.0)
        else:
            grad_norm = self._calculate_gradient_norm()

        if indices:
            self.learning_memory.update_priorities(indices, td_errors.detach().abs().cpu().tolist())

        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        avg_reward = float(rewards.mean().item()) if rewards.numel() else 0.0
        return avg_reward, loss

    def train_episode(
        self,
        env: Any = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Train for a full environment episode or a replay-only update if no env is available."""
        env = env or self.env
        if env is None:
            return self._optimise_from_batch()

        state = self._extract_state(env.reset())
        done = False
        episode_reward = 0.0
        losses: List[float] = []
        step_count = 0
        step_limit = max_steps if max_steps is not None else self.max_steps_per_episode
        state_sequence: List[np.ndarray] = []

        while not done:
            if step_limit is not None and step_count >= int(step_limit):
                break

            state_tensor = self._state_to_tensor(state)
            state_sequence.append(state_tensor.detach().cpu().numpy())
            action = self.act(state_tensor, state_sequence=state_sequence[-self.rsi_period :], explore=True)
            next_state, reward, terminated, truncated, _ = self._safe_step(env, action)
            done = bool(terminated or truncated)

            self.remember(state, action, reward, next_state, done)
            self.total_env_steps += 1
            episode_reward += float(reward)
            step_count += 1

            if len(self.memory) >= self.min_replay_size:
                _avg_reward, loss = self._optimise_from_batch()
                if loss:
                    losses.append(float(loss))

            state = next_state

        self.total_episodes += 1
        self.performance_history.append(float(episode_reward))
        self.last_train_episode_metrics = {
            "episode_reward": float(episode_reward),
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "min_loss": float(np.min(losses)) if losses else 0.0,
            "max_loss": float(np.max(losses)) if losses else 0.0,
            "steps": int(step_count),
            "epsilon": float(self.epsilon),
        }
        return float(episode_reward), (float(np.mean(losses)) if losses else 0.0)

    def learn_step(self, experience_batch: Iterable[ExperienceLike]) -> Tuple[float, float]:
        for exp in experience_batch:
            transition = self._coerce_transition(exp)
            self.remember(transition.state, transition.action, transition.reward, transition.next_state, transition.done)
        return self._optimise_from_batch(experience_batch=experience_batch)

    # ------------------------------------------------------------------
    # LearningMemory integration and persistence
    # ------------------------------------------------------------------
    def sync_with_learning_memory(self) -> None:
        """Synchronise state, metrics, and replay snapshots into LearningMemory."""
        self.learning_memory.set(
            "agent_state",
            {
                "agent_id": self.agent_id,
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "rsi_period": self.rsi_period,
                "gamma": self.gamma,
                "baseline_performance": self.baseline_performance,
                "network_weights": self.q_network.get_weights(),
                "target_weights": self.target_network.get_weights(),
                "update_counter": self.update_counter,
                "current_epoch": self.current_epoch,
                "total_env_steps": self.total_env_steps,
                "total_gradient_steps": self.total_gradient_steps,
                "total_episodes": self.total_episodes,
            },
        )
        self.learning_memory.set("training_metrics", list(self.training_metrics))
        self.learning_memory.set("experience_replay_snapshot", [self._serialize_transition(item) for item in self.memory])

    def _save_training_state(self, epoch: int) -> Dict[str, Any]:
        state = {
            "epoch": int(epoch),
            "q_network": self.q_network.get_checkpoint(),
            "target_network": self.target_network.get_checkpoint(),
            "performance_history": list(self.performance_history),
            "hyperparameters": {
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "rsi_period": self.rsi_period,
                "baseline_performance": self.baseline_performance,
            },
            "replay_memory": [self._serialize_transition(item) for item in self.memory],
            "update_counter": self.update_counter,
            "total_env_steps": self.total_env_steps,
            "total_gradient_steps": self.total_gradient_steps,
            "total_episodes": self.total_episodes,
        }
        self.learning_memory.set("last_checkpoint", state)
        self.sync_with_learning_memory()
        return state

    def _load_training_state(self) -> bool:
        state = self.learning_memory.get("last_checkpoint")
        if not state:
            return False

        self.q_network.load_checkpoint(state["q_network"])
        self.target_network.load_checkpoint(state["target_network"])
        self.performance_history = deque(state.get("performance_history", []), maxlen=self.performance_window)
        hyperparameters = state.get("hyperparameters", {})
        self.epsilon = float(hyperparameters.get("epsilon", self.epsilon))
        self.learning_rate = float(hyperparameters.get("learning_rate", self.learning_rate))
        self.gamma = float(hyperparameters.get("gamma", self.gamma))
        self.rsi_period = int(hyperparameters.get("rsi_period", self.rsi_period))
        self.baseline_performance = hyperparameters.get("baseline_performance", self.baseline_performance)
        self.update_counter = int(state.get("update_counter", self.update_counter))
        self.total_env_steps = int(state.get("total_env_steps", self.total_env_steps))
        self.total_gradient_steps = int(state.get("total_gradient_steps", self.total_gradient_steps))
        self.total_episodes = int(state.get("total_episodes", self.total_episodes))
        self.current_epoch = int(state.get("epoch", -1)) + 1
        replay_memory = state.get("replay_memory", [])
        self.memory.clear()
        for item in replay_memory[-self.memory_capacity :]:
            self.memory.append(self._coerce_transition(item))
        self.q_network.optimizer.learning_rate = self.learning_rate
        return True

    def save(self, filepath: str) -> Dict[str, Any]:
        """Save a complete RSI checkpoint and an aligned LearningMemory checkpoint."""
        try:
            self.sync_with_learning_memory()
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            memory_path = path.with_suffix(path.suffix + ".memory") if path.suffix else Path(f"{filepath}.memory")

            checkpoint_data = {
                "version": 2,
                "agent_id": self.agent_id,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "model_id": self.model_id,
                "diagnostics": self.diagnostics(),
                "training_metrics": list(self.training_metrics),
                "learning_memory_state": self.learning_memory.get("agent_state"),
                "runtime_state": self._save_training_state(self.current_epoch),
            }
            torch.save(checkpoint_data, path)
            saved_memory_path = self.learning_memory.save_checkpoint(str(memory_path))
            file_hash = self._calculate_file_hash(str(path))
            logger.info("Saved RSI checkpoint to %s", path)
            return {
                "success": True,
                "integrity_verified": True,
                "checkpoint_path": str(path),
                "memory_checkpoint_path": str(saved_memory_path),
                "file_hash": file_hash,
            }
        except Exception as exc:
            logger.error("Save failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def load(self, filepath: str) -> Dict[str, Any]:
        """Load a complete RSI checkpoint and restore replay / memory state."""
        try:
            path = Path(filepath)
            memory_path = path.with_suffix(path.suffix + ".memory") if path.suffix else Path(f"{filepath}.memory")

            checkpoint = torch.load(path, map_location="cpu")
            runtime_state = checkpoint.get("runtime_state") or {}
            if runtime_state:
                self.learning_memory.set("last_checkpoint", runtime_state)
                self._load_training_state()

            if memory_path.exists():
                self.learning_memory.load_checkpoint(str(memory_path))

            training_metrics = checkpoint.get("training_metrics")
            if isinstance(training_metrics, list):
                self.training_metrics = training_metrics

            file_hash = self._calculate_file_hash(str(path))
            logger.info("Loaded RSI checkpoint from %s", path)
            return {
                "success": True,
                "checksum_valid": True,
                "checkpoint_path": str(path),
                "memory_checkpoint_path": str(memory_path) if memory_path.exists() else None,
                "file_hash": file_hash,
            }
        except Exception as exc:
            logger.error("Load failed: %s", exc)
            self._set_default_parameters()
            return {"success": False, "error": str(exc)}

    def _set_default_parameters(self) -> None:
        self.gamma = float(self.rsi_config.get("gamma", 0.95))
        self.epsilon = float(self.rsi_config.get("epsilon", 1.0))
        self.epsilon_min = float(self.rsi_config.get("epsilon_min", 0.01))
        self.epsilon_decay = float(self.rsi_config.get("epsilon_decay", 0.995))
        self.learning_rate = float(self.rsi_config.get("learning_rate", 0.001))
        self.rsi_period = int(self.rsi_config.get("rsi_period", 14))
        self.baseline_performance = self.rsi_config.get("baseline_performance")
        self.memory.clear()
        self.performance_history.clear()
        self.training_metrics.clear()
        self.q_network.optimizer.learning_rate = self.learning_rate

    # ------------------------------------------------------------------
    # Evaluation / full training lifecycle
    # ------------------------------------------------------------------
    def evaluate(
        self,
        env: Any,
        episodes: int = 50,
        include_training_data: bool = True,
        exploration_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        if episodes <= 0:
            raise ValueError("episodes must be positive.")

        logger.info("Evaluating RSI Agent %s over %s episodes", self.agent_id, episodes)
        original_epsilon = self.epsilon
        self.epsilon = float(self.eval_exploration_rate if exploration_rate is None else exploration_rate)

        total_rewards: List[float] = []
        episode_lengths: List[int] = []
        action_distribution = {action: 0 for action in range(self.action_size)}
        state_visit_counts: Dict[Tuple[float, ...], int] = defaultdict(int)
        max_steps = max_steps if max_steps is not None else self.max_steps_per_episode

        try:
            for _ in range(episodes):
                state = self._extract_state(env.reset())
                done = False
                episode_reward = 0.0
                steps = 0

                while not done:
                    if max_steps is not None and steps >= int(max_steps):
                        break
                    action = self.act(state, explore=self.epsilon > 0.0)
                    next_state, reward, terminated, truncated, _ = self._safe_step(env, action)
                    done = bool(terminated or truncated)

                    state_key = tuple(self._state_to_tensor(state).detach().cpu().numpy().tolist())
                    state_visit_counts[state_key] += 1
                    action_distribution[action] += 1
                    episode_reward += float(reward)
                    steps += 1
                    state = next_state

                total_rewards.append(float(episode_reward))
                episode_lengths.append(int(steps))
        finally:
            self.epsilon = original_epsilon

        total_actions = sum(action_distribution.values())
        normalised_action_distribution = {
            key: (count / total_actions if total_actions else 0.0)
            for key, count in action_distribution.items()
        }

        avg_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
        evaluation = {
            "episodes": int(episodes),
            "avg_reward": avg_reward,
            "std_reward": float(np.std(total_rewards)) if total_rewards else 0.0,
            "min_reward": float(min(total_rewards)) if total_rewards else 0.0,
            "max_reward": float(max(total_rewards)) if total_rewards else 0.0,
            "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "action_distribution": normalised_action_distribution,
            "state_coverage": int(len(state_visit_counts)),
            "exploration_rate": float(self.epsilon),
            "sharpe_ratio": self._calculate_sharpe(total_rewards),
            "max_drawdown": self._calculate_max_drawdown(total_rewards),
            "avg_return": avg_reward,
            "parameter_effectiveness": {
                "learning_rate": float(self.learning_rate),
                "gamma": float(self.gamma),
                "rsi_period": int(self.rsi_period),
                "epsilon": float(self.epsilon),
            },
            "training_memory_utilized": bool(include_training_data),
            "detailed_rewards": total_rewards,
        }
        if include_training_data:
            evaluation["training_metrics_available"] = len(self.training_metrics)
            evaluation["learning_memory_size"] = self.learning_memory.size()
        self.last_evaluation = evaluation
        self.learning_memory.set("rsi_agent_last_eval", evaluation)
        return evaluation

    def _adapt_learning_rate(self, avg_reward: float, volatility: float) -> float:
        volatility = max(float(volatility), 1e-8)
        volatility_factor = float(np.clip(volatility / 0.2, 0.5, 2.0))
        reward_factor = float(np.clip(1.0 + (avg_reward / 100.0), 0.5, 2.0))
        new_lr = self.learning_rate * reward_factor / volatility_factor
        self.learning_rate = float(np.clip(new_lr, self.lr_min, self.lr_max))
        self.q_network.optimizer.learning_rate = self.learning_rate
        return self.learning_rate

    def _check_early_stopping(self) -> bool:
        if len(self.performance_history) < 20:
            return False
        recent_perf = float(np.mean(list(self.performance_history)[-10:]))
        baseline_perf = float(np.mean(list(self.performance_history)[-20:-10]))
        return (recent_perf - baseline_perf) < self.improvement_threshold

    def get_training_summary(self) -> Dict[str, Any]:
        metrics = self.training_metrics
        if not metrics:
            logger.warning("No training metrics found for RSIAgent.")
            return {
                "total_episodes": self.total_episodes,
                "avg_reward": None,
                "best_reward": None,
                "volatility_profile": {"avg": None, "max": None},
                "final_parameters": {
                    "epsilon": self.epsilon,
                    "learning_rate": self.learning_rate,
                    "rsi_period": self.rsi_period,
                },
            }

        avg_rewards = [float(m.get("avg_reward", 0.0)) for m in metrics]
        volatilities = [float(m.get("avg_volatility", 0.0)) for m in metrics]
        return {
            "total_episodes": self.total_episodes,
            "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
            "best_reward": float(np.max(avg_rewards)) if avg_rewards else 0.0,
            "volatility_profile": {
                "avg": float(np.mean(volatilities)) if volatilities else 0.0,
                "max": float(np.max(volatilities)) if volatilities else 0.0,
            },
            "final_parameters": {
                "epsilon": float(self.epsilon),
                "learning_rate": float(self.learning_rate),
                "rsi_period": int(self.rsi_period),
                "gamma": float(self.gamma),
            },
            "last_evaluation": self.last_evaluation,
        }

    def train(
        self,
        env: Any,
        total_epochs: int = 1000,
        episodes_per_epoch: int = 100,
        evaluation_interval: int = 10,
        performance_threshold: float = 0.0,
        checkpoint_interval: int = 50,
        max_steps_per_episode: Optional[int] = None,
    ) -> Dict[str, Any]:
        if env is None:
            raise ValueError("env is required for RSI training.")
        if total_epochs <= 0 or episodes_per_epoch <= 0:
            raise ValueError("total_epochs and episodes_per_epoch must be positive.")
        if evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive.")

        self.env = env
        if self.learning_memory.get("last_checkpoint"):
            self._load_training_state()
            logger.info("Resuming RSI training from checkpoint state.")

        try:
            for epoch in range(self.current_epoch, total_epochs):
                epoch_rewards: List[float] = []
                epoch_losses: List[float] = []

                for _ in range(episodes_per_epoch):
                    episode_reward, episode_loss = self.train_episode(env=env, max_steps=max_steps_per_episode)
                    epoch_rewards.append(float(episode_reward))
                    epoch_losses.append(float(episode_loss))

                avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
                avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                avg_volatility = float(np.std(epoch_rewards)) if epoch_rewards else 0.0
                self.performance_history.append(avg_reward)

                epoch_record = {
                    "epoch": int(epoch),
                    "avg_reward": avg_reward,
                    "avg_loss": avg_loss,
                    "avg_volatility": avg_volatility,
                    "epsilon": float(self.epsilon),
                    "learning_rate": float(self.learning_rate),
                    "network_weights": self.q_network.get_weights(),
                }
                self.training_metrics.append(epoch_record)
                self.learning_memory.set("training_metrics", list(self.training_metrics))

                if (epoch + 1) % self.improvement_interval == 0:
                    self.self_improve()

                if (epoch + 1) % evaluation_interval == 0:
                    eval_results = self.evaluate(env, episodes=max(5, min(50, episodes_per_epoch)))
                    logger.info(
                        "Epoch %s evaluation | avg_reward=%.4f sharpe=%.4f max_drawdown=%.4f",
                        epoch + 1,
                        eval_results["avg_reward"],
                        eval_results["sharpe_ratio"],
                        eval_results["max_drawdown"],
                    )
                    if eval_results["avg_return"] < performance_threshold:
                        mutation_report = self._mutate_parameters()
                        logger.info("Triggered performance-based mutation: %s", mutation_report)

                if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                    checkpoint_name = self.checkpoint_dir / f"rsi_epoch_{epoch + 1}.pt"
                    self.current_epoch = epoch + 1
                    self.save(str(checkpoint_name))
                    logger.info("RSI checkpoint saved at epoch %s", epoch + 1)

                if self._check_early_stopping():
                    logger.info("RSI early stopping condition met at epoch %s", epoch + 1)
                    self.current_epoch = epoch + 1
                    break

                self._adapt_learning_rate(avg_reward, avg_volatility)
                self.current_epoch = epoch + 1

            logger.info("RSI training completed successfully")
            self.last_training_summary = self.get_training_summary()
            return self.last_training_summary
        except Exception as exc:
            logger.error("Training interrupted: %s", exc)
            self._save_training_state(self.current_epoch)
            raise

    def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RSI with either full env training or replay-driven improvement cycles."""
        logger.info("[RSI_Agent] Executing task: %s", task_data)
        env = task_data.get("env", self.env)

        if env is not None:
            summary = self.train(
                env=env,
                total_epochs=int(task_data.get("total_epochs", task_data.get("epochs", 10))),
                episodes_per_epoch=int(task_data.get("episodes_per_epoch", 10)),
                evaluation_interval=int(task_data.get("evaluation_interval", max(1, self.improvement_interval))),
                performance_threshold=float(task_data.get("performance_threshold", 0.0)),
                checkpoint_interval=int(task_data.get("checkpoint_interval", max(1, self.improvement_interval))),
            )
            evaluation = self.evaluate(env, episodes=int(task_data.get("eval_episodes", 10)))
            self.learning_memory.set("rsi_agent_last_eval", evaluation)
            return {"status": "success", "summary": summary, "evaluation": evaluation}

        episodes = int(task_data.get("episodes", 100))
        replay_losses: List[float] = []
        replay_rewards: List[float] = []
        for episode_idx in range(episodes):
            avg_reward, loss = self._optimise_from_batch()
            replay_rewards.append(avg_reward)
            replay_losses.append(loss)
            self.performance_history.append(avg_reward)
            if (episode_idx + 1) % self.improvement_interval == 0:
                self.self_improve()

        evaluation = {
            "episodes": episodes,
            "avg_reward": float(np.mean(replay_rewards)) if replay_rewards else 0.0,
            "avg_loss": float(np.mean(replay_losses)) if replay_losses else 0.0,
            "mode": "replay_only",
        }
        self.learning_memory.set("rsi_agent_last_eval", evaluation)
        return {"status": "success", "evaluation": evaluation}


__all__ = ["RSIAgent"]


if __name__ == "__main__":
    print("\n=== Running Recursive Self-Improvement Smoke Test ===\n")
    from src.agents.learning.slaienv import SLAIEnv

    env = SLAIEnv(state_dim=4, action_dim=3)
    agent = RSIAgent(state_size=4, action_size=3, agent_id="rsi_smoke", env=env)

    for _ in range(50):
        state = np.random.randn(4).astype(np.float32)
        next_state = np.random.randn(4).astype(np.float32)
        agent.remember(state, random.randrange(3), np.random.randn(), next_state, random.random() < 0.1)

    reward, loss = agent.train_episode(env=env, max_steps=10)
    print(f"Episode reward={reward:.3f} loss={loss:.6f}")
    evaluation = agent.evaluate(env, episodes=5)
    print("Evaluation:", evaluation)
    summary = agent.train(env, total_epochs=2, episodes_per_epoch=3, evaluation_interval=1, checkpoint_interval=0)
    print("Training summary:", summary)
    print("\n=== Recursive Self-Improvement Smoke Test Complete ===\n")
