"""Production-ready Deep Q-Network (DQN) agent with optional evolutionary tuning.

Key references:
1. Mnih et al. (2015). Human-level control through deep reinforcement learning.
2. Mnih et al. (2015). Target networks and experience replay.
3. Salimans et al. (2017). Evolution Strategies as a Scalable Alternative to RL.

This module provides:
- A robust DQN agent with replay, target networks, checkpointing, and evaluation.
- Optional prioritized replay integration via ``LearningMemory``.
- An evolutionary hyperparameter search utility for DQN configurations.
- A unified interface that supports both standard and evolutionary workflows.
"""

from __future__ import annotations

import copy
import math
import os
import random
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from src.agents.learning.learning_memory import LearningMemory, Transition
from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.neural_network import NeuralNetwork
from src.utils.buffer.replay_buffer import ReplayBuffer
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Deep-Q Network Agent")
printer = PrettyPrinter

TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]
TransitionLike = Union[Transition, Tuple[Any, Any, Any, Any, Any], List[Any]]


def _extract_state(reset_output: Any) -> Any:
    """Normalize Gym/Gymnasium reset outputs to a raw observation."""
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        return reset_output[0]
    return reset_output


def _step_environment(env: Any, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """Normalize Gym/Gymnasium step outputs to (state, reward, done, info)."""
    step_output = env.step(action)
    if not isinstance(step_output, tuple):
        raise TypeError("Environment step(...) must return a tuple.")

    if len(step_output) == 5:
        next_state, reward, terminated, truncated, info = step_output
        done = bool(terminated or truncated)
        return next_state, float(reward), done, info or {}

    if len(step_output) == 4:
        next_state, reward, done, info = step_output
        return next_state, float(reward), bool(done), info or {}

    raise ValueError(f"Unsupported environment step() output length: {len(step_output)}")


def _state_to_tensor(state: TensorLike, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert a state-like input into a 1D float tensor."""
    if isinstance(state, torch.Tensor):
        tensor = state.detach().clone().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(state, dtype=torch.float32)

    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 1:
        tensor = tensor.reshape(-1)

    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _coerce_transition(transition: TransitionLike) -> Transition:
    """Convert tuples/lists to ``Transition`` while preserving existing namedtuples."""
    if isinstance(transition, Transition):
        return transition

    if isinstance(transition, (tuple, list)) and len(transition) == 5:
        state, action, reward, next_state, done = transition
        return Transition(state, action, reward, next_state, done)

    raise TypeError("Each transition must contain exactly 5 elements: state, action, reward, next_state, done.")


def _weight_distance(model_a: NeuralNetwork, model_b: NeuralNetwork) -> float:
    """Compute mean Euclidean distance across corresponding parameters."""
    weights_a = model_a.get_weights()
    weights_b = model_b.get_weights()
    distances: List[float] = []

    for key in ("Ws", "bs"):
        for tensor_a, tensor_b in zip(weights_a[key], weights_b[key]):
            diff = tensor_a.detach().cpu().numpy() - tensor_b.detach().cpu().numpy()
            distances.append(float(np.linalg.norm(diff)))

    return float(np.mean(distances)) if distances else 0.0


class DQNAgent:
    """Deep Q-Network agent for discrete action spaces.

    Parameters
    ----------
    agent_id:
        Logical identifier used in logging and checkpoints.
    state_dim:
        Flat state dimension.
    action_dim:
        Number of discrete actions.
    config:
        Optional configuration override merged on top of the global config.
    device:
        Optional torch device for the policy / target networks.
    """

    def __init__(
        self,
        agent_id: Optional[str],
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if int(state_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("state_dim and action_dim must be positive integers.")

        self.agent_id = str(agent_id) if agent_id is not None else "DQN"
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.config = load_global_config()
        self.dqn_config = get_config_section('dqn')
        self.model_id = "DQN_Agent"

        self.hidden_dim = self.dqn_config.get("hidden_size", 128)
        self.gamma = float(self.dqn_config.get("gamma", 0.99))
        self.epsilon = float(self.dqn_config.get("epsilon", 1.0))
        self.epsilon_min = float(self.dqn_config.get("epsilon_min", 0.01))
        self.epsilon_decay = float(self.dqn_config.get("epsilon_decay", 0.995))
        self.lr = float(self.dqn_config.get("learning_rate", 0.001))
        self.batch_size = int(self.dqn_config.get("batch_size", 64))
        self.target_update = int(self.dqn_config.get("target_update_frequency", 100))
        self.buffer_size = int(self.dqn_config.get("buffer_size", 10000))
        self.warmup_steps = int(self.dqn_config.get("warmup_steps", self.batch_size))
        self.max_steps_per_episode = self.dqn_config.get("max_steps_per_episode")
        self.target_soft_update_tau = self.dqn_config.get("target_soft_update_tau")
        self.target_soft_update_tau = (
            float(self.target_soft_update_tau) if self.target_soft_update_tau is not None else None
        )
        self.use_double_dqn = bool(self.dqn_config.get("double_dqn", False))
        self.replay_backend = str(self.dqn_config.get("replay_backend", "uniform")).strip().lower()

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if self.target_update <= 0:
            raise ValueError("target_update_frequency must be positive.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if self.target_soft_update_tau is not None and not 0.0 < self.target_soft_update_tau <= 1.0:
            raise ValueError("target_soft_update_tau must be in (0, 1].")

        self.learning_memory: Optional[LearningMemory] = None
        self.last_train_metrics: Optional[Dict[str, float]] = None
        self.total_env_steps = 0
        self.total_gradient_steps = 0
        self.episodes_completed = 0

        network_config = self._build_network_config()
        self.policy_net = NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim,
            config=network_config,
            device=self.device,
        )
        self.target_net = NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim,
            config=network_config,
            device=self.device,
        )
        self.update_target_net(force=True)
        self._init_buffer()

        logger.info(
            "Deep-Q Network Agent initialised | id=%s state_dim=%s action_dim=%s replay=%s double_dqn=%s",
            self.agent_id,
            self.state_dim,
            self.action_dim,
            self.replay_backend,
            self.use_double_dqn,
        )

    def _build_network_config(self) -> Dict[str, Any]:
        """Build an explicit network config aligned with the DQN config."""
        neural_cfg = copy.deepcopy(self.config.get("neural_network", {}))
        hidden_size = self.hidden_dim

        if "layer_dims" not in neural_cfg:
            if isinstance(hidden_size, int):
                hidden_layers = [hidden_size]
            elif isinstance(hidden_size, (list, tuple)):
                hidden_layers = [int(dim) for dim in hidden_size]
            else:
                raise TypeError("dqn.hidden_size must be an int or a sequence of ints.")

            if any(dim <= 0 for dim in hidden_layers):
                raise ValueError("All hidden layer sizes must be positive.")

            neural_cfg["layer_dims"] = [self.state_dim, *hidden_layers, self.action_dim]

        neural_cfg.setdefault("hidden_activation", "relu")
        neural_cfg["output_activation"] = "linear"
        neural_cfg["loss_function"] = "mse"
        neural_cfg.setdefault("optimizer", "adam")
        neural_cfg["learning_rate"] = self.lr
        return neural_cfg

    def _init_buffer(self) -> None:
        """Initialise the selected replay backend."""
        if self.replay_backend in {"prioritized", "per", "learning_memory"}:
            self.learning_memory = LearningMemory()
            self.learning_memory.memory_config["max_size"] = self.buffer_size
            self.learning_memory.clear()
            self.memory = self.learning_memory
            self.replay_backend = "prioritized"
        else:
            self.memory = ReplayBuffer(self.buffer_size)
            self.replay_backend = "uniform"

    def replay_size(self) -> int:
        """Return the current replay size regardless of backend."""
        if self.replay_backend == "prioritized":
            return int(self.memory.size())
        return len(self.memory)

    def update_target_net(self, force: bool = False) -> None:
        """Update target network weights.

        Hard updates are the default and match the canonical DQN algorithm.
        If ``target_soft_update_tau`` is configured, soft updates are used on
        non-forced calls.
        """
        if force or self.target_soft_update_tau is None:
            self.target_net.set_weights(self.policy_net.get_weights())
            return

        tau = self.target_soft_update_tau
        with torch.no_grad():
            for target_W, policy_W in zip(self.target_net.Ws, self.policy_net.Ws):
                target_W.mul_(1.0 - tau).add_(policy_W, alpha=tau)
            for target_b, policy_b in zip(self.target_net.bs, self.policy_net.bs):
                target_b.mul_(1.0 - tau).add_(policy_b, alpha=tau)

    def select_action(self, processed_state: TensorLike, explore: bool = True) -> int:
        """Select an action with epsilon-greedy exploration."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = _state_to_tensor(processed_state, device=self.device)
        if state_tensor.numel() != self.state_dim:
            raise ValueError(
                f"State dimension mismatch. Expected {self.state_dim} values, got {state_tensor.numel()}."
            )

        with torch.no_grad():
            q_values = self.policy_net.forward(state_tensor.unsqueeze(0)).squeeze(0)
        return int(torch.argmax(q_values).item())

    def store_transition(self, *transition: Any) -> None:
        """Store a transition in the configured replay backend."""
        transition_obj = _coerce_transition(transition if len(transition) != 1 else transition[0])
        stored_transition = Transition(
            _state_to_tensor(transition_obj.state).cpu(),
            int(transition_obj.action),
            float(transition_obj.reward),
            _state_to_tensor(transition_obj.next_state).cpu(),
            bool(transition_obj.done),
        )

        if stored_transition.state.numel() != self.state_dim or stored_transition.next_state.numel() != self.state_dim:
            raise ValueError("Stored state dimensions must match the configured state_dim.")

        if self.replay_backend == "prioritized":
            self.memory.add(stored_transition)
        else:
            self.memory.push(stored_transition)

        self.total_env_steps += 1

    def _sample_replay_batch(
        self,
        experience_batch: Optional[Iterable[TransitionLike]] = None,
        indices: Optional[Sequence[int]] = None,
        importance_weights: Optional[Sequence[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Prepare a replay batch for learning."""
        batch: List[Transition]
        sampled_indices = list(indices) if indices is not None else None

        if experience_batch is None:
            if self.replay_size() < self.batch_size:
                return None

            if self.replay_backend == "prioritized":
                batch_raw, sampled_indices, sampled_weights = self.memory.sample_proportional(self.batch_size)
                batch = [_coerce_transition(item) for item in batch_raw]
                weights = torch.as_tensor(sampled_weights, dtype=torch.float32, device=self.device)
            else:
                batch_raw = self.memory.sample(self.batch_size)
                batch = [_coerce_transition(item) for item in batch_raw]
                weights = torch.ones(len(batch), dtype=torch.float32, device=self.device)
        else:
            batch = [_coerce_transition(item) for item in experience_batch]
            if len(batch) == 0:
                return None
            if importance_weights is None:
                weights = torch.ones(len(batch), dtype=torch.float32, device=self.device)
            else:
                weights = torch.as_tensor(importance_weights, dtype=torch.float32, device=self.device)
                if weights.ndim != 1 or weights.shape[0] != len(batch):
                    raise ValueError("importance_weights must be a 1D sequence aligned with the experience batch.")

        states = torch.stack([_state_to_tensor(t.state, device=self.device) for t in batch])
        actions = torch.as_tensor([int(t.action) for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([float(t.reward) for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([_state_to_tensor(t.next_state, device=self.device) for t in batch])
        dones = torch.as_tensor([float(bool(t.done)) for t in batch], dtype=torch.float32, device=self.device)

        return {
            "transitions": batch,
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "indices": sampled_indices,
            "weights": weights.clamp_min(1e-8),
        }

    def learn_step(
        self,
        experience_batch: Optional[Iterable[TransitionLike]] = None,
        indices: Optional[Sequence[int]] = None,
        importance_weights: Optional[Sequence[float]] = None,
    ) -> Optional[Dict[str, float]]:
        """Run one real DQN optimisation step.

        When using prioritized replay, TD-error priorities are updated after the
        optimisation step and the loss is importance-weighted by scaling the
        Bellman residual for the selected action.
        """
        batch = self._sample_replay_batch(
            experience_batch=experience_batch,
            indices=indices,
            importance_weights=importance_weights,
        )
        if batch is None:
            return None

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        weights = batch["weights"]
        sampled_indices = batch["indices"]

        with torch.no_grad():
            current_q = self.policy_net.forward(states)
            chosen_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

            if self.use_double_dqn:
                next_policy_q = self.policy_net.forward(next_states)
                next_actions = torch.argmax(next_policy_q, dim=1)
                next_target_q = self.target_net.forward(next_states)
                next_q_values = next_target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_target_q = self.target_net.forward(next_states)
                next_q_values = torch.max(next_target_q, dim=1).values

            td_target = rewards + (1.0 - dones) * self.gamma * next_q_values
            td_errors = td_target - chosen_q

            target = current_q.clone()
            batch_indices = torch.arange(states.shape[0], device=self.device)
            weighted_residual = td_errors * torch.sqrt(weights)
            target[batch_indices, actions] = chosen_q + weighted_residual

        loss = float(self.policy_net.train_step(states, target))

        if self.replay_backend == "prioritized" and sampled_indices is not None:
            self.memory.update_priorities(sampled_indices, td_errors.detach().abs().cpu().tolist())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_gradient_steps += 1

        if self.target_soft_update_tau is None:
            if self.total_gradient_steps % self.target_update == 0:
                self.update_target_net(force=True)
        else:
            self.update_target_net(force=False)

        metrics = {
            "loss": loss,
            "epsilon": float(self.epsilon),
            "avg_q_value": float(chosen_q.mean().item()),
            "avg_target_q": float(td_target.mean().item()),
            "avg_td_error": float(td_errors.abs().mean().item()),
            "batch_size": float(states.shape[0]),
            "replay_size": float(self.replay_size()),
        }
        self.last_train_metrics = metrics
        return metrics

    def train(self) -> Optional[float]:
        """Backward-compatible alias that returns only the scalar loss."""
        if self.replay_size() < max(self.batch_size, self.warmup_steps):
            return None
        metrics = self.learn_step()
        return None if metrics is None else float(metrics["loss"])

    def train_episode(
        self,
        env: Any,
        max_steps: Optional[int] = None,
        explore: bool = True,
        learn: bool = True,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Run one full environment episode and optionally learn online from replay."""
        state = _extract_state(env.reset())
        done = False
        step_limit = max_steps if max_steps is not None else self.max_steps_per_episode
        total_reward = 0.0
        losses: List[float] = []
        q_values_seen: List[float] = []
        steps = 0

        while not done:
            if step_limit is not None and steps >= int(step_limit):
                break
            if render:
                env.render()

            with torch.no_grad():
                state_tensor = _state_to_tensor(state, device=self.device)
                q_values = self.policy_net.forward(state_tensor.unsqueeze(0)).squeeze(0)
                q_values_seen.append(float(q_values.max().item()))

            action = self.select_action(state, explore=explore)
            next_state, reward, done, _ = _step_environment(env, action)
            self.store_transition(state, action, reward, next_state, done)

            if learn and self.replay_size() >= max(self.batch_size, self.warmup_steps):
                learn_metrics = self.learn_step()
                if learn_metrics is not None:
                    losses.append(float(learn_metrics["loss"]))

            total_reward += float(reward)
            state = next_state
            steps += 1

        self.episodes_completed += 1
        return {
            "reward": float(total_reward),
            "length": int(steps),
            "avg_loss": float(np.mean(losses)) if losses else None,
            "min_loss": float(np.min(losses)) if losses else None,
            "max_loss": float(np.max(losses)) if losses else None,
            "loss_history": losses,
            "avg_max_q_value": float(np.mean(q_values_seen)) if q_values_seen else 0.0,
            "epsilon": float(self.epsilon),
        }

    def evaluate(
        self,
        env: Any,
        episodes: int = 20,
        exploration_rate: float = 0.05,
        visualize: bool = False,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate the agent with multiple behavioural and value statistics."""
        if episodes <= 0:
            raise ValueError("episodes must be positive.")

        logger.info("Evaluating DQN Agent %s over %s episodes", self.agent_id, episodes)

        total_rewards: List[float] = []
        episode_lengths: List[int] = []
        q_value_means: List[float] = []
        q_value_stds: List[float] = []
        action_distribution = {a: 0 for a in range(self.action_dim)}

        original_epsilon = self.epsilon
        self.epsilon = float(exploration_rate)

        try:
            for episode_idx in range(episodes):
                state = _extract_state(env.reset())
                done = False
                steps = 0
                episode_reward = 0.0
                step_limit = max_steps if max_steps is not None else self.max_steps_per_episode

                while not done:
                    if step_limit is not None and steps >= int(step_limit):
                        break
                    if visualize:
                        env.render()

                    action = self.select_action(state, explore=True)
                    state_tensor = _state_to_tensor(state, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        q_values = self.policy_net.forward(state_tensor).squeeze(0).detach().cpu().numpy()

                    next_state, reward, done, _ = _step_environment(env, action)
                    q_value_means.append(float(np.mean(q_values)))
                    q_value_stds.append(float(np.std(q_values)))
                    action_distribution[action] += 1
                    episode_reward += float(reward)
                    steps += 1
                    state = next_state

                total_rewards.append(float(episode_reward))
                episode_lengths.append(int(steps))
                logger.debug(
                    "Evaluation episode %s/%s | reward=%.3f steps=%s",
                    episode_idx + 1,
                    episodes,
                    episode_reward,
                    steps,
                )
        finally:
            self.epsilon = original_epsilon

        try:
            reward_threshold = getattr(getattr(env, "spec", None), "reward_threshold")
        except Exception:  # pragma: no cover - defensive against exotic env wrappers
            reward_threshold = None

        if reward_threshold is None:
            reward_threshold = max(total_rewards) * 0.9 if total_rewards else 0.0

        total_actions = sum(action_distribution.values())
        normalised_action_distribution = {
            action: (count / total_actions if total_actions else 0.0)
            for action, count in action_distribution.items()
        }

        return {
            "episodes": int(episodes),
            "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
            "std_reward": float(np.std(total_rewards)) if total_rewards else 0.0,
            "min_reward": float(min(total_rewards)) if total_rewards else 0.0,
            "max_reward": float(max(total_rewards)) if total_rewards else 0.0,
            "success_rate": float((np.array(total_rewards) >= reward_threshold).mean()) if total_rewards else 0.0,
            "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "action_distribution": normalised_action_distribution,
            "avg_q_value": float(np.mean(q_value_means)) if q_value_means else 0.0,
            "q_value_std": float(np.mean(q_value_stds)) if q_value_stds else 0.0,
            "target_network_divergence": _weight_distance(self.policy_net, self.target_net),
            "replay_buffer_utilization": float(self.replay_size() / self.buffer_size),
            "exploration_rate": float(exploration_rate),
            "reward_threshold": float(reward_threshold),
            "detailed_rewards": total_rewards,
            "episode_lengths": episode_lengths,
        }

    def diagnostics(self) -> Dict[str, Any]:
        """Return compact runtime diagnostics for observability and debugging."""
        diagnostics = {
            "agent_id": self.agent_id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "epsilon": float(self.epsilon),
            "gamma": float(self.gamma),
            "learning_rate": float(self.lr),
            "batch_size": int(self.batch_size),
            "buffer_size": int(self.buffer_size),
            "replay_size": int(self.replay_size()),
            "replay_backend": self.replay_backend,
            "double_dqn": bool(self.use_double_dqn),
            "target_update_frequency": int(self.target_update),
            "target_soft_update_tau": self.target_soft_update_tau,
            "total_env_steps": int(self.total_env_steps),
            "total_gradient_steps": int(self.total_gradient_steps),
            "episodes_completed": int(self.episodes_completed),
            "target_network_divergence": _weight_distance(self.policy_net, self.target_net),
        }
        if self.last_train_metrics is not None:
            diagnostics["last_train_metrics"] = copy.deepcopy(self.last_train_metrics)
        return diagnostics

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Persist agent state and model checkpoints."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "version": 2,
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "config": copy.deepcopy(self.config),
            "epsilon": float(self.epsilon),
            "total_env_steps": int(self.total_env_steps),
            "total_gradient_steps": int(self.total_gradient_steps),
            "episodes_completed": int(self.episodes_completed),
            "replay_backend": self.replay_backend,
            "policy_net": self.policy_net.get_checkpoint(),
            "target_net": self.target_net.get_checkpoint(),
        }

        if bool(self.config.get("dqn", {}).get("save_replay_buffer", False)):
            if self.replay_backend == "prioritized":
                checkpoint["replay_buffer"] = self.memory.get()
            else:
                checkpoint["replay_buffer"] = list(self.memory.buffer)

        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved DQN model checkpoint to %s", checkpoint_path)

    def load(self, path: Union[str, os.PathLike[str]]) -> None:
        """Load agent state from a checkpoint created by :meth:`save`."""
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if int(checkpoint.get("state_dim", self.state_dim)) != self.state_dim:
            raise ValueError("Checkpoint state_dim does not match the current agent.")
        if int(checkpoint.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("Checkpoint action_dim does not match the current agent.")

        self.config = _deep_merge(self.config, checkpoint.get("config", {}))
        self.policy_net.load_checkpoint(checkpoint["policy_net"])
        self.target_net.load_checkpoint(checkpoint["target_net"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.total_env_steps = int(checkpoint.get("total_env_steps", self.total_env_steps))
        self.total_gradient_steps = int(checkpoint.get("total_gradient_steps", self.total_gradient_steps))
        self.episodes_completed = int(checkpoint.get("episodes_completed", self.episodes_completed))

        replay_buffer = checkpoint.get("replay_buffer")
        if replay_buffer:
            if self.replay_backend == "prioritized":
                self.memory.clear()
                for item in replay_buffer:
                    self.memory.add(_coerce_transition(item))
            else:
                self.memory = ReplayBuffer(self.buffer_size)
                for item in replay_buffer:
                    self.memory.push(_coerce_transition(item))

        logger.info("Loaded DQN model checkpoint from %s", checkpoint_path)


class EvolutionaryTrainer:
    """Evolutionary hyperparameter optimisation for DQN agents.

    The trainer evaluates candidate DQN configurations by training each one for
    a limited number of episodes and ranking them by evaluation reward.
    """

    def __init__(
        self,
        env: Any,
        state_dim: int,
        action_dim: int,
        agent_id_prefix: str = "EvoDQN",
    ):
        if env is None:
            raise ValueError("env is required for evolutionary training.")
        if int(state_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("state_dim and action_dim must be positive integers.")

        self.env = env
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.agent_id_prefix = str(agent_id_prefix)
        self.evolutionary_config = get_config_section("evolutionary")
        self.dqn_config = get_config_section('dqn')

        self.pop_size = int(self.evolutionary_config.get("population_size", 10))
        self.generations = int(self.evolutionary_config.get("generations", 20))
        self.mutation_rate = float(self.evolutionary_config.get("mutation_rate", 0.2))
        self.evaluation_episodes = int(self.evolutionary_config.get("evaluation_episodes", 3))
        self.elite_ratio = float(self.evolutionary_config.get("elite_ratio", 0.3))
        self.candidate_training_episodes = int(
            self.evolutionary_config.get("candidate_training_episodes", max(5, self.evaluation_episodes * 2))
        )
        self.validation_exploration_rate = float(self.evolutionary_config.get("validation_exploration_rate", 0.05))
        self.max_steps_per_episode = self.evolutionary_config.get(
            "max_steps_per_episode",
            self.dqn_config.get("max_steps_per_episode"),
        )
        self.include_baseline = bool(self.evolutionary_config.get("include_baseline", True))

        self.population: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.best_agent: Optional[DQNAgent] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_fitness = -math.inf

        logger.info(
            "Evolutionary Trainer initialised | population=%s generations=%s mutation_rate=%.3f",
            self.pop_size,
            self.generations,
            self.mutation_rate,
        )

    def _candidate_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Return a full DQN config (top‑level 'dqn' key) with overrides merged."""
        # Start with the global DQN section as base
        base = dict(self.dqn_config)  # shallow copy
        # Apply overrides
        base.update(overrides)
        # Wrap under the 'dqn' key as DQNAgent expects
        return {"dqn": base}

    def _random_config(self) -> Dict[str, Any]:
        """Create a fully numeric, serialisable candidate config."""
        hidden_size = random.choice([64, 128, 256, [128, 128], [256, 128]])
        overrides = {
            "gamma": round(random.uniform(0.90, 0.999), 6),
            "epsilon": 1.0,
            "epsilon_min": round(random.uniform(0.01, 0.10), 4),
            "epsilon_decay": round(random.uniform(0.990, 0.9995), 6),
            "learning_rate": float(10 ** random.uniform(-4.2, -2.3)),
            "hidden_size": hidden_size,
            "batch_size": random.choice([32, 64, 128]),
            "target_update_frequency": random.choice([50, 100, 200, 500]),
            "buffer_size": random.choice([5000, 10000, 20000]),
            "double_dqn": random.choice([False, True]),
            "replay_backend": random.choice(["uniform", "uniform", "prioritized"]),
        }
        return self._candidate_config(overrides)

    def _mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a candidate config while keeping parameters in valid ranges."""
        # config is full config with top‑level 'dqn'
        mutated = copy.deepcopy(config)
        dqn_config = mutated.setdefault("dqn", {})

        if random.random() < self.mutation_rate:
            dqn_config["gamma"] = float(min(0.999, max(0.90, dqn_config.get("gamma", 0.99) + random.gauss(0.0, 0.01))))

        if random.random() < self.mutation_rate:
            current_lr = float(dqn_config.get("learning_rate", 0.001))
            dqn_config["learning_rate"] = float(min(1e-2, max(1e-4, current_lr * math.exp(random.gauss(0.0, 0.25)))))

        if random.random() < self.mutation_rate:
            decay = float(dqn_config.get("epsilon_decay", 0.995))
            dqn_config["epsilon_decay"] = float(min(0.9999, max(0.985, decay + random.gauss(0.0, 0.002))))

        if random.random() < self.mutation_rate:
            dqn_config["epsilon_min"] = float(min(0.20, max(0.001, dqn_config.get("epsilon_min", 0.01) + random.gauss(0.0, 0.01))))

        if random.random() < self.mutation_rate:
            dqn_config["hidden_size"] = random.choice([64, 128, 256, [128, 128], [256, 128]])

        if random.random() < self.mutation_rate:
            dqn_config["batch_size"] = random.choice([32, 64, 128])

        if random.random() < self.mutation_rate:
            dqn_config["target_update_frequency"] = random.choice([50, 100, 200, 500])

        if random.random() < self.mutation_rate:
            dqn_config["buffer_size"] = random.choice([5000, 10000, 20000])

        if random.random() < self.mutation_rate:
            dqn_config["double_dqn"] = not bool(dqn_config.get("double_dqn", False))

        if random.random() < self.mutation_rate:
            dqn_config["replay_backend"] = random.choice(["uniform", "prioritized"])

        return mutated

    def _make_agent(self, config: Dict[str, Any], agent_suffix: str) -> DQNAgent:
        """Create a DQNAgent with the given configuration."""
        return DQNAgent(
            agent_id=f"{self.agent_id_prefix}_{agent_suffix}",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,          # Now config is properly structured
        )

    def _train_candidate(self, agent: DQNAgent) -> Dict[str, Any]:
        """Train a candidate for a fixed number of episodes and return metrics."""
        rewards: List[float] = []
        losses: List[float] = []

        for _ in range(self.candidate_training_episodes):
            metrics = agent.train_episode(
                self.env,
                max_steps=self.max_steps_per_episode,
                explore=True,
                learn=True,
            )
            rewards.append(float(metrics["reward"]))
            if metrics["avg_loss"] is not None:
                losses.append(float(metrics["avg_loss"]))

        return {
            "avg_training_reward": float(np.mean(rewards)) if rewards else 0.0,
            "avg_training_loss": float(np.mean(losses)) if losses else None,
            "training_rewards": rewards,
        }

    def _evaluate(self, agent: DQNAgent, episodes: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a candidate using the same metric surface as the base agent."""
        return agent.evaluate(
            self.env,
            episodes=episodes or self.evaluation_episodes,
            exploration_rate=self.validation_exploration_rate,
            max_steps=self.max_steps_per_episode,
        )

    def evolve(self) -> DQNAgent:
        """Run the evolutionary search and return the best DQN agent."""
        # Initialise population
        self.population = [self._random_config() for _ in range(self.pop_size)]
        self.history = []
        self.best_agent = None
        self.best_config = None
        self.best_fitness = -math.inf

        for gen in range(self.generations):
            fitness_scores = []  # (fitness, config, agent)

            # Train and evaluate each candidate
            for idx, config in enumerate(self.population):
                agent = self._make_agent(config, f"gen{gen}_c{idx}")
                train_results = self._train_candidate(agent)
                eval_results = self._evaluate(agent, episodes=self.evaluation_episodes)
                fitness = float(eval_results["avg_reward"])
                fitness_scores.append((fitness, config, agent))
                logger.debug(
                    f"Gen {gen} candidate {idx}: train_reward={train_results['avg_training_reward']:.2f} "
                    f"eval_reward={fitness:.2f}"
                )

            # Sort by fitness descending
            fitness_scores.sort(key=lambda x: x[0], reverse=True)

            # Update best overall
            if fitness_scores[0][0] > self.best_fitness:
                self.best_fitness = fitness_scores[0][0]
                self.best_config = copy.deepcopy(fitness_scores[0][1])
                self.best_agent = fitness_scores[0][2]   # keep the agent object
                logger.info(f"New best fitness: {self.best_fitness:.2f}")

            # Record history
            avg_fitness = float(np.mean([s[0] for s in fitness_scores]))
            self.history.append({
                "generation": gen,
                "best_fitness": fitness_scores[0][0],
                "avg_fitness": avg_fitness,
                "best_config": copy.deepcopy(fitness_scores[0][1]),
            })

            # Selection: keep top elite_ratio as elites
            num_elites = max(1, int(self.pop_size * self.elite_ratio))
            elites = [fitness_scores[i][1] for i in range(num_elites)]

            # Create next generation
            next_population = []
            # Preserve elites
            next_population.extend(elites)
            # Fill the rest with mutations of elites (or random if no elite)
            while len(next_population) < self.pop_size:
                parent = random.choice(elites)
                child = self._mutate(parent)
                next_population.append(child)

            self.population = next_population

        # Final safety: if no best agent was stored (e.g., population empty), create one from best config
        if self.best_agent is None:
            if self.best_config is not None:
                self.best_agent = self._make_agent(self.best_config, "final_best")
            else:
                # Fallback: use the first configuration of the last generation
                self.best_agent = self._make_agent(self.population[0], "fallback_best")

        return self.best_agent


class UnifiedDQNAgent:
    """Unified interface for standard and evolutionary DQN workflows."""

    def __init__(
        self,
        mode: str = "standard",
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        env: Any = None,
        agent_id: str = "Unified",
    ):
        self.mode = str(mode).strip().lower()
        self.config = config or {}
        self.env = env
        self.agent_id = agent_id
        self.agent: Optional[DQNAgent] = None
        self.trainer: Optional[EvolutionaryTrainer] = None

        if self.mode == "standard":
            if state_dim is None or action_dim is None:
                raise ValueError("State and action dimensions are required for standard mode.")
            self.agent = DQNAgent(
                agent_id=self.agent_id,
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                config=self.config,
            )
        elif self.mode == "evolutionary":
            if env is None or state_dim is None or action_dim is None:
                raise ValueError("Evolutionary mode requires env, state_dim, and action_dim.")
            self.trainer = EvolutionaryTrainer(
                env=env,
                state_dim=int(state_dim),
                action_dim=int(action_dim),
                base_config=self.config,
                agent_id_prefix=self.agent_id,
            )
        else:
            raise ValueError("Invalid mode. Choose 'standard' or 'evolutionary'.")

        logger.info("Unified Deep-Q Network Agent initialised in %s mode", self.mode)

    def _require_agent(self) -> DQNAgent:
        if self.agent is None:
            raise RuntimeError("No active DQN agent is available. Train or evolve the agent first.")
        return self.agent

    def _run_validation(self, episodes: int = 5) -> Dict[str, Any]:
        if self.env is None:
            raise ValueError("Environment not provided for validation.")
        return self._require_agent().evaluate(self.env, episodes=episodes, exploration_rate=0.05)

    def train(
        self,
        episodes: int = 1000,
        validation_freq: int = 50,
        validation_episodes: int = 5,
        checkpoint_dir: Union[str, os.PathLike[str]] = "checkpoints",
        early_stop_patience: Optional[int] = 20,
        target_reward: Optional[float] = None,
        max_steps_per_episode: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train in standard mode or run evolutionary search in evolutionary mode."""
        if self.mode == "evolutionary":
            self.agent = self.trainer.evolve()
            return {
                "mode": "evolutionary",
                "best_fitness": float(self.trainer.best_fitness),
                "best_config": copy.deepcopy(self.trainer.best_config),
                "history": copy.deepcopy(self.trainer.history),
            }

        if self.env is None:
            raise ValueError("Environment not provided for standard training.")
        if episodes <= 0:
            raise ValueError("episodes must be positive.")
        if validation_freq <= 0:
            raise ValueError("validation_freq must be positive.")

        agent = self._require_agent()
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        episode_rewards: List[float] = []
        episode_losses: List[float] = []
        episode_lengths: List[int] = []
        validation_history: List[Dict[str, Any]] = []
        best_val_reward = -math.inf
        best_checkpoint_path: Optional[str] = None
        early_stop_counter = 0

        for episode in range(episodes):
            train_metrics = agent.train_episode(
                self.env,
                max_steps=max_steps_per_episode,
                explore=True,
                learn=True,
            )
            reward = float(train_metrics["reward"])
            avg_loss = train_metrics["avg_loss"]
            length = int(train_metrics["length"])

            episode_rewards.append(reward)
            episode_losses.append(float(avg_loss) if avg_loss is not None else float("nan"))
            episode_lengths.append(length)

            reward_window = episode_rewards[-10:]
            loss_window = [loss for loss in episode_losses[-10:] if not math.isnan(loss)]
            avg_reward_10 = float(np.mean(reward_window)) if reward_window else reward
            avg_loss_10 = float(np.mean(loss_window)) if loss_window else float("nan")

            logger.info(
                "Episode %s/%s | reward=%.3f avg10_reward=%.3f avg_loss=%s avg10_loss=%s steps=%s epsilon=%.4f",
                episode + 1,
                episodes,
                reward,
                avg_reward_10,
                f"{avg_loss:.6f}" if avg_loss is not None else "n/a",
                f"{avg_loss_10:.6f}" if not math.isnan(avg_loss_10) else "n/a",
                length,
                agent.epsilon,
            )

            if (episode + 1) % validation_freq == 0:
                validation_metrics = self._run_validation(validation_episodes)
                validation_metrics["episode"] = episode + 1
                validation_history.append(validation_metrics)
                val_reward = float(validation_metrics["avg_reward"])

                checkpoint_path = checkpoint_dir / f"best_ep{episode + 1}.pt"
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    best_checkpoint_path = str(checkpoint_path)
                    self.save(checkpoint_path)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                logger.info(
                    "Validation after episode %s | avg_reward=%.3f best=%.3f patience=%s/%s",
                    episode + 1,
                    val_reward,
                    best_val_reward,
                    early_stop_counter,
                    early_stop_patience,
                )

                if early_stop_patience is not None and early_stop_counter >= early_stop_patience:
                    logger.info("Early stopping triggered at episode %s", episode + 1)
                    break

            if target_reward is not None and avg_reward_10 >= float(target_reward):
                logger.info("Target reward %.3f achieved at episode %s", target_reward, episode + 1)
                break

        return {
            "mode": "standard",
            "rewards": episode_rewards,
            "losses": episode_losses,
            "lengths": episode_lengths,
            "validation_history": validation_history,
            "best_validation_reward": None if best_val_reward == -math.inf else float(best_val_reward),
            "best_checkpoint_path": best_checkpoint_path,
            "final_epsilon": float(agent.epsilon),
            "diagnostics": agent.diagnostics(),
        }

    def save(self, path: Union[str, os.PathLike[str]]) -> None:
        """Save the current agent checkpoint."""
        self._require_agent().save(path)

    def load(self, path: Union[str, os.PathLike[str]]) -> None:
        """Load a checkpoint into the current standard-mode agent."""
        self._require_agent().load(path)

    def act(self, state: TensorLike, explore: bool = False) -> int:
        """Return an action from the active agent."""
        return self._require_agent().select_action(state, explore=explore)


if __name__ == "__main__":
    print("\n=== Running Deep-Q Network Agent Smoke Test ===\n")
    from src.agents.learning.slaienv import SLAIEnv

    env = SLAIEnv()
    config = {
        "dqn": {
            "hidden_size": [128, 64],
            "batch_size": 16,
            "buffer_size": 512,
            "target_update_frequency": 20,
            "epsilon_decay": 0.99,
            "replay_backend": "uniform",
        }
    }

    agent = DQNAgent(agent_id="smoke_test", state_dim=4, action_dim=2, config=config)
    for _ in range(5):
        metrics = agent.train_episode(env, max_steps=20)
        print(f"Episode reward={metrics['reward']:.2f} loss={metrics['avg_loss']}")

    evaluation = agent.evaluate(env, episodes=3)
    print("Evaluation summary:", evaluation)

    trainer = EvolutionaryTrainer(env=env, state_dim=4, action_dim=2, base_config=config)
    trainer.generations = 2
    trainer.pop_size = 3
    trainer.candidate_training_episodes = 2
    best_agent = trainer.evolve()
    print("Best evolutionary agent diagnostics:", best_agent.diagnostics())

    unified = UnifiedDQNAgent(mode="standard", state_dim=4, action_dim=2, config=config, env=env)
    summary = unified.train(episodes=3, validation_freq=1, validation_episodes=1, early_stop_patience=2)
    print("Unified training summary:", summary)
    print("\n=== Deep-Q Network Agent Smoke Test Complete ===\n")
