from __future__ import annotations

__version__ = "2.1.0"

"""
SLAI Learning Agent: Core Component for Autonomous Learning & Improvement

Academic References:
1. DQN & RL: Mnih et al. (2015). Human-level control through deep RL. Nature.
2. MAML: Finn et al. (2017). Model-Agnostic Meta-Learning. PMLR.
3. RSI: Schmidhuber (2013). PowerPlay: Training General Problem Solvers.
4. Continual Learning: Parisi et al. (2019). Continual Learning Survey. IEEE TCDS.

Academic Foundations:
- Catastrophic Forgetting: Kirkpatrick et al. (2017) EWC
- Concept Drift: Gama et al. (2014) Survey on Concept Drift
- Meta-Learning: Finn et al. (2017) MAML
- Evolutionary Strategies: Salimans et al. (2017)
"""
import hashlib
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.agents.base.utils.main_config_loader import get_config_section, load_global_config
from src.agents.base_agent import BaseAgent
from src.agents.learning.learning_factory import LearningFactory
from src.agents.learning.slaienv import SLAIEnv
from src.agents.learning.strategy_selector import StrategySelector
from src.agents.learning.utils.error_calls import InvalidActionError, InvalidConfigError, NaNException
from src.agents.learning.utils.learning_calculations import LearningCalculations
from src.agents.learning.utils.multi_task_learner import MultiTaskLearner
from src.agents.learning.utils.policy_network import PolicyNetwork, create_policy_optimizer
from src.agents.learning.utils.recovery_system import RecoverySystem
from src.agents.learning.utils.state_processor import StateProcessor
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Learning Agent")
printer = PrettyPrinter


class LearningAgent(BaseAgent):
    """Production orchestration layer for SLAI's lifelong learning subsystem."""

    DEFAULT_TASK_IDS: Sequence[str] = ("dqn", "maml", "rsi", "rl")

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)

        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.learning_config = get_config_section("learning_agent") or {}
        if isinstance(config, dict):
            self.learning_config.update(config)

        self._validate_config()

        self.task_ids = list(self.learning_config.get("task_ids", self.DEFAULT_TASK_IDS))
        self.strategy_index = {name: idx for idx, name in enumerate(self.task_ids)}

        self.batch_size = int(self.learning_config.get("batch_size", 32))
        self.max_episode_steps = int(self.learning_config.get("max_episode_steps", 128))
        self.max_eval_episodes = int(self.learning_config.get("max_eval_episodes", 3))
        self.recovery_trigger_threshold = int(self.learning_config.get("recovery_trigger_threshold", 3))

        self.strategy_weights = np.asarray(self.learning_config.get("strategy_weights", [0.25] * 4), dtype=np.float32)
        self.prediction_weights = np.asarray(self.learning_config.get("prediction_weights", [0.25] * 4), dtype=np.float32)
        self.maml_task_pool_size = int(self.learning_config.get("maml_task_pool_size", 100))
        self.rsi_improvement_cycle = int(self.learning_config.get("rsi_improvement_cycle", 50))
        self.performance_threshold = float(self.learning_config.get("performance_threshold", 0.7))
        self.data_change_threshold = float(self.learning_config.get("data_change_threshold", 0.15))
        self.retraining_interval = timedelta(hours=int(self.learning_config.get("retraining_interval_hours", 24)))
        self.novelty_threshold = float(self.learning_config.get("novelty_threshold", 0.3))
        self.uncertainty_threshold = float(self.learning_config.get("uncertainty_threshold", 0.25))
        self.task_embedding_dim = int(self.learning_config.get("task_embedding_dim", 256))

        self.embedding_buffer: Deque[Tuple[torch.Tensor, int]] = deque(maxlen=int(self.learning_config.get("embedding_buffer_size", 512)))
        self.performance_history = deque(maxlen=int(self.learning_config.get("performance_history_size", 1000)))
        self.state_recency = deque(maxlen=int(self.learning_config.get("state_recency_size", 1000)))
        self.architecture_history = deque(maxlen=int(self.learning_config.get("architecture_history_size", 10)))
        self.error_history = deque(maxlen=int(self.learning_config.get("error_history_size", 100)))

        self.performance_metrics = {
            # These are useful across all agent types:
            'scenario_rewards': defaultdict(float),        # Mean reward per scenario or task
            'success_rate': defaultdict(float),            # Percentage of episodes meeting success criteria
            'episode_length': defaultdict(list),           # Track duration of episodes
            'q_value_mean': defaultdict(list),             # Mean Q-value per evaluation run
            'q_value_std': defaultdict(list),              # Std of Q-values (to track learning stability)

            # Per agent strategy (DQN, MAML, RSI, RL):
            'strategy_selection_count': defaultdict(int),  # How often each strategy is selected
            'strategy_accuracy': defaultdict(float),       # Meta-controller prediction accuracy
            'strategy_loss': defaultdict(float),           # Loss during meta-controller training

            # For adaptive tuning and monitoring drift or stagnation:
            'novelty_score': defaultdict(float),           # From novelty detector (e.g., for RSI)
            'uncertainty_estimate': defaultdict(float),    # Optional: from model prediction variance
            'catastrophic_forgetting': defaultdict(float), # Score or heuristic for forgetting events
            'concept_drift_detected': defaultdict(bool),   # Boolean or counter per task/episode

            # For experience buffer and MAML/meta-training:
            'embedding_buffer_size': lambda: len(self.embedding_buffer),
            'replay_buffer_usage': defaultdict(int),       # Transitions used per training step
            'performance_history_stats': lambda: {
                'mean': np.mean(self.performance_history) if self.performance_history else 0,
                'std': np.std(self.performance_history) if self.performance_history else 0,
            },

            # Specific to LearningFactory and RSI:
            'param_mutation_rate': {},                     # Track over time for RSI
            'checkpoint_quality': defaultdict(float),      # Loaded from model checkpoints
            'agent_fitness_score': defaultdict(float),     # For evolutionary scoring of agents

            'plot_tags': ['average_reward', 'success_rate', 'strategy_selection_count', 'novelty_score'],
        }
        self.env = SLAIEnv() or kwargs.get("env")
        self.state_processor = StateProcessor(env=self.env)
        self.learning_calculations = LearningCalculations()
        self.learning_factory = LearningFactory(env=self.env, performance_metrics=self.performance_metrics)
        self.agents = self.learning_factory.agents

        self.multi_task_learner = MultiTaskLearner(task_ids=self.task_ids)
        self.strategy_selector = StrategySelector()
        self._initialize_strategy_selector()

        self.recovery_system = RecoverySystem(learning_agent=self)

        self.observation_count = 0
        self.training_iterations = 0
        self.last_training_time = datetime.now(timezone.utc) - self.retraining_interval

        self._init_shared_memory_keys()
        logger.info("LearningAgent initialized with task_ids=%s", self.task_ids)

    def _validate_config(self) -> None:
        for key in ("task_embedding_dim", "batch_size", "max_episode_steps", "recovery_trigger_threshold"):
            value = int(self.learning_config.get(key, 1))
            if value <= 0:
                raise InvalidConfigError(f"learning_agent.{key} must be > 0")

    def _init_shared_memory_keys(self) -> None:
        self.sm_keys = {
            "decision_trace": f"learning:decision_trace:{self.name}",
            "metrics": f"learning:metrics:{self.name}",
            "strategies": f"learning:strategies:{self.name}",
            "episodes": f"learning:episodes:{self.name}",
            "health": f"learning:health:{self.name}",
        }

    def _initialize_strategy_selector(self) -> None:
        self.strategy_selector.set_agent_strategies_map(self.strategy_index)

        embedder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.task_embedding_dim),
        )
        self.strategy_selector.set_state_embedder(embedder)

        policy_network = PolicyNetwork(
            input_dim=self.task_embedding_dim,
            output_dim=len(self.task_ids),
            hidden_sizes=[128, 64],
            hidden_activation="relu",
            output_activation="linear",
        )
        optimizer = create_policy_optimizer(policy_network)
        self.strategy_selector.set_policy_network(
            policy_net=policy_network,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )

    @staticmethod
    def _extract_state(reset_output: Any) -> Any:
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            return reset_output[0]
        return reset_output

    @staticmethod
    def _safe_step(env: Any, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        result = env.step(action)
        if not isinstance(result, tuple):
            raise RuntimeError("Environment step(...) must return tuple")
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            return next_state, float(reward), bool(terminated or truncated), info or {}
        if len(result) == 4:
            next_state, reward, done, info = result
            return next_state, float(reward), bool(done), info or {}
        raise RuntimeError(f"Unsupported step output length: {len(result)}")

    @staticmethod
    def _align_vectors(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        max_dim = max(len(a), len(b))
        if len(a) < max_dim:
            a = np.pad(a, (0, max_dim - len(a)))
        if len(b) < max_dim:
            b = np.pad(b, (0, max_dim - len(b)))
        return a, b

    def _prepare_state_array(self, state: Any) -> np.ndarray:
        processed = self.state_processor.process(state)
        if processed.numel() == 0:
            return np.zeros(1, dtype=np.float32)
        array = processed.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise NaNException("State contains non-finite values")
        return array

    def _compute_novelty(self, state: np.ndarray) -> float:
        if not self.state_recency:
            return 1.0
        prev = self.state_recency[-1]
        prev, state = self._align_vectors(prev, state)
        denom = max(1e-6, float(np.linalg.norm(prev)) + float(np.linalg.norm(state)))
        return float(np.linalg.norm(state - prev) / denom)

    def _estimate_uncertainty(self, embedding: torch.Tensor) -> float:
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        with torch.no_grad():
            logits = self.strategy_selector.policy_net(embedding)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        max_entropy = float(np.log(max(2, len(self.task_ids))))
        return float(entropy / max_entropy)

    def _normalized_strategy_weights(self) -> Dict[str, float]:
        weights = self.strategy_weights
        if len(weights) != len(self.task_ids):
            weights = np.ones(len(self.task_ids), dtype=np.float32)
        weights = np.clip(weights, 1e-6, None)
        weights = weights / float(weights.sum())
        return {task_id: float(weights[idx]) for idx, task_id in enumerate(self.task_ids)}

    def _strategy_scores(self, uncertainty: float, novelty: float, task_metadata: Mapping[str, Any]) -> Dict[str, float]:
        priors = self._normalized_strategy_weights()
        scores = {task: priors.get(task, 0.0) for task in self.task_ids}

        for task_id in self.task_ids:
            perf_hist = self.performance_metrics.get(task_id, deque(maxlen=100))
            perf = float(np.mean(perf_hist)) if perf_hist else 0.0
            scores[task_id] += 0.45 * perf

        if uncertainty > self.uncertainty_threshold or novelty > self.novelty_threshold:
            if "maml" in scores:
                scores["maml"] += 0.20
            if "rsi" in scores:
                scores["rsi"] += 0.15

        if self._performance_trend() < 0.0 and "rsi" in scores:
            scores["rsi"] += 0.10

        preferred = str(task_metadata.get("preferred_strategy", "")).strip().lower()
        if preferred in scores:
            scores[preferred] += 0.20

        return scores

    def _performance_trend(self) -> float:
        if len(self.performance_history) < 10:
            return 0.0
        recent = float(np.mean(list(self.performance_history)[-10:]))
        baseline = float(np.mean(list(self.performance_history)[:10]))
        return (recent - baseline) / (abs(baseline) + 1e-6)

    def _resolve_strategy_name(self, agent: Any) -> str:
        for name, known_agent in self.learning_factory.agents.items():
            if known_agent is agent:
                return name
        return getattr(agent, "agent_id", "dqn").split("_")[0].lower()

    def _select_strategy(self, state: Any, task_metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, Any, Dict[str, Any]]:
        metadata = task_metadata or {}
        state_array = self._prepare_state_array(state)
        embedding = self.strategy_selector.generate_task_embedding(state_array)

        novelty = self._compute_novelty(state_array)
        uncertainty = self._estimate_uncertainty(embedding)
        policy_pick = self.strategy_selector.select_strategy(embedding)
        heuristic_scores = self._strategy_scores(uncertainty, novelty, metadata)
        heuristic_pick = max(heuristic_scores, key=heuristic_scores.get)

        factory_agent = self.learning_factory.select_agent(
            {
                "novelty": novelty,
                "volatility": uncertainty,
                "preferred_agent": metadata.get("preferred_strategy"),
                "complexity": float(metadata.get("complexity", 0.5)),
                "compute_budget": float(metadata.get("compute_budget", 0.5)),
                "training_budget": float(metadata.get("training_budget", 0.5)),
            }
        )
        factory_pick = self._resolve_strategy_name(factory_agent)

        selected = heuristic_pick if uncertainty > self.uncertainty_threshold else policy_pick
        if metadata.get("selector_mode") == "factory":
            selected = factory_pick

        selected_agent = self.learning_factory.agents.get(selected, factory_agent)

        trace = {
            "policy_pick": policy_pick,
            "heuristic_pick": heuristic_pick,
            "factory_pick": factory_pick,
            "selected": selected,
            "uncertainty": uncertainty,
            "novelty": novelty,
            "scores": heuristic_scores,
            "state_hash": hashlib.sha256(state_array.tobytes()).hexdigest(),
        }

        self.state_recency.append(state_array)
        return selected, selected_agent, trace

    def _agent_action(self, agent: Any, state: np.ndarray, explore: bool = True) -> int:
        try:
            if hasattr(agent, "act") and callable(agent.act):
                return int(agent.act(state, explore=explore))
        except TypeError:
            return int(agent.act(state))

        if hasattr(agent, "select_action") and callable(agent.select_action):
            try:
                return int(agent.select_action(state, explore=explore))
            except TypeError:
                return int(agent.select_action(state))

        if hasattr(agent, "get_action") and callable(agent.get_action):
            action = agent.get_action(state)
            if isinstance(action, tuple):
                action = action[0]
            return int(action)

        raise InvalidActionError("agent has no compatible action API")

    def _agent_learn_step(self, strategy: str, agent: Any, transition: Tuple[Any, int, float, Any, bool]) -> Optional[float]:
        try:
            if hasattr(agent, "store_transition"):
                agent.store_transition(*transition)
            elif hasattr(agent, "remember"):
                agent.remember(*transition)

            if hasattr(agent, "train") and callable(agent.train):
                train_result = agent.train()
                if isinstance(train_result, (int, float)) and np.isfinite(train_result):
                    self.multi_task_learner.update_loss(strategy, float(train_result))
                    return float(train_result)
                if isinstance(train_result, dict):
                    loss = train_result.get("loss") or train_result.get("avg_loss")
                    if isinstance(loss, (int, float)) and np.isfinite(loss):
                        self.multi_task_learner.update_loss(strategy, float(loss))
                        return float(loss)
        except Exception as exc:
            logger.warning("Learning step failed for strategy=%s: %s", strategy, exc)
            self.error_history.append({"strategy": strategy, "error": str(exc), "timestamp": time.time()})
        return None

    def _run_episode(self, strategy: str, agent: Any, max_steps: int, seed: Optional[int], train: bool = True) -> Dict[str, Any]:
        reset_output = self.env.reset(seed=seed)
        state = self._extract_state(reset_output)

        total_reward = 0.0
        losses: List[float] = []

        for step_idx in range(max_steps):
            state_arr = self._prepare_state_array(state)
            action = self._agent_action(agent, state_arr, explore=train)
            if action < 0:
                raise InvalidActionError(action)

            next_state, reward, done, _ = self._safe_step(self.env, action)
            if not np.isfinite(reward):
                raise NaNException("Reward became non-finite")

            next_arr = self._prepare_state_array(next_state)
            if train:
                loss = self._agent_learn_step(strategy, agent, (state_arr, action, float(reward), next_arr, bool(done)))
                if loss is not None:
                    losses.append(loss)

            total_reward += float(reward)
            self.observation_count += 1
            state = next_state
            if done:
                break

        self.performance_history.append(total_reward)
        if strategy not in self.performance_metrics:
            self.performance_metrics[strategy] = deque(maxlen=100)
        self.performance_metrics[strategy].append(float(total_reward))
        self.learning_factory.record_performance(strategy, float(total_reward))

        if train:
            self.multi_task_learner.rebalance()
            final_embedding = self.strategy_selector.generate_task_embedding(self._prepare_state_array(state))
            self.strategy_selector.observe(final_embedding, strategy)
            self.strategy_selector.train_from_embeddings()

        return {
            "status": "ok",
            "strategy": strategy,
            "episode_reward": float(total_reward),
            "steps": step_idx + 1,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "loss_samples": len(losses),
            "train_mode": train,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _record_shared_memory(self, episode_result: Dict[str, Any], trace: Dict[str, Any]) -> None:
        metric_key = self.sm_keys["metrics"]
        strategy_key = self.sm_keys["strategies"]
        episodes_key = self.sm_keys["episodes"]
        trace_key = self.sm_keys["decision_trace"]

        metrics = self.shared_memory.get(metric_key) or {}
        metrics.update(
            {
                "last_reward": episode_result["episode_reward"],
                "last_steps": episode_result["steps"],
                "last_strategy": episode_result["strategy"],
                "train_mode": episode_result["train_mode"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.shared_memory.set(metric_key, metrics)

        strategy_history = self.shared_memory.get(strategy_key) or []
        strategy_history.append(
            {
                "strategy": episode_result["strategy"],
                "reward": episode_result["episode_reward"],
                "timestamp": time.time(),
            }
        )
        self.shared_memory.set(strategy_key, strategy_history[-500:])

        episodes = self.shared_memory.get(episodes_key) or []
        episodes.append(episode_result)
        self.shared_memory.set(episodes_key, episodes[-500:])

        traces = self.shared_memory.get(trace_key) or []
        traces.append(trace)
        self.shared_memory.set(trace_key, traces[-500:])

    def _should_retrain(self) -> bool:
        return datetime.now(timezone.utc) - self.last_training_time >= self.retraining_interval

    def _train_cycles(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        episodes = int(task_data.get("episodes", 1))
        max_steps = int(task_data.get("max_steps", self.max_episode_steps))
        seed = task_data.get("seed")
        metadata = task_data.get("task_metadata") or {}

        if episodes <= 0 or max_steps <= 0:
            raise InvalidConfigError("episodes and max_steps must be positive")

        episode_reports = []
        for episode_idx in range(episodes):
            probe_state = self._extract_state(self.env.reset(seed=seed))
            strategy, agent, trace = self._select_strategy(probe_state, metadata)
            report = self._run_episode(strategy=strategy, agent=agent, max_steps=max_steps, seed=seed, train=True)
            report["episode_index"] = episode_idx
            report["selection_trace"] = trace
            episode_reports.append(report)
            self._record_shared_memory(report, trace)

        self.training_iterations += episodes
        self.last_training_time = datetime.now(timezone.utc)

        avg_reward = float(np.mean([x["episode_reward"] for x in episode_reports])) if episode_reports else 0.0
        return {
            "status": "ok",
            "mode": "train",
            "episodes": episodes,
            "avg_reward": avg_reward,
            "reports": episode_reports,
            "trend": self._performance_trend(),
            "task_weights": self.multi_task_learner.get_weights(),
        }

    def _evaluate(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        episodes = int(task_data.get("episodes", self.max_eval_episodes))
        max_steps = int(task_data.get("max_steps", self.max_episode_steps))
        metadata = task_data.get("task_metadata") or {}

        rewards = []
        steps = []
        strategies = []
        traces = []

        for _ in range(episodes):
            state = self._extract_state(self.env.reset())
            strategy, agent, trace = self._select_strategy(state, metadata)
            report = self._run_episode(strategy=strategy, agent=agent, max_steps=max_steps, seed=None, train=False)
            rewards.append(report["episode_reward"])
            steps.append(report["steps"])
            strategies.append(strategy)
            traces.append(trace)

        result = {
            "status": "ok",
            "mode": "evaluate",
            "episodes": episodes,
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "avg_steps": float(np.mean(steps)) if steps else 0.0,
            "strategy_distribution": {k: strategies.count(k) for k in sorted(set(strategies))},
            "traces": traces,
        }
        self.shared_memory.set(self.sm_keys["metrics"], {**(self.shared_memory.get(self.sm_keys["metrics"]) or {}), "last_evaluation": result})
        return result

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        if task_data is None:
            task_data = {}
        if not isinstance(task_data, dict):
            task_data = {"payload": task_data}

        mode = str(task_data.get("mode", "train")).strip().lower()
        if mode == "health":
            report = self.health_report()
            self.shared_memory.set(self.sm_keys["health"], report)
            return {"status": "ok", "mode": "health", "report": report}

        if mode == "evaluate":
            return self._evaluate(task_data)

        if mode == "train":
            train_result = self._train_cycles(task_data)
            if len(self.error_history) >= self.recovery_trigger_threshold:
                train_result["recovery"] = self.recovery_system.execute_recovery()
            return train_result

        if mode == "auto":
            if self._should_retrain():
                return self._train_cycles(task_data)
            return self._evaluate(task_data)

        raise InvalidConfigError(f"Unsupported perform_task mode: {mode}")

    def alternative_execute(self, task_data, original_error=None):
        recovery_result = self.recovery_system.execute_recovery(error=original_error)
        safe_task_data = dict(task_data) if isinstance(task_data, dict) else {"payload": task_data}
        safe_task_data["mode"] = "evaluate"
        safe_task_data["episodes"] = min(2, int(safe_task_data.get("episodes", 1)))
        safe_task_data["max_steps"] = min(32, int(safe_task_data.get("max_steps", 32)))
        try:
            eval_result = self._evaluate(safe_task_data)
        except Exception:
            eval_result = super().alternative_execute(safe_task_data, original_error=original_error)
        return {
            "status": "recovered_fallback",
            "recovery": recovery_result,
            "fallback_result": eval_result,
        }

    def extract_performance_metrics(self, result: Any) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        return {
            "avg_reward": float(result.get("avg_reward", result.get("episode_reward", 0.0))),
            "episodes": int(result.get("episodes", 1)),
            "steps": float(result.get("avg_steps", result.get("steps", 0))),
            "trend": float(result.get("trend", self._performance_trend())),
        }

    def health_report(self) -> Dict[str, Any]:
        strategy_health = {}
        for strategy in self.task_ids:
            history = self.performance_metrics.get(strategy, [])
            strategy_health[strategy] = {
                "samples": len(history),
                "mean_reward": float(np.mean(history)) if history else 0.0,
                "std_reward": float(np.std(history)) if history else 0.0,
            }

        return {
            "name": self.name,
            "task_ids": list(self.task_ids),
            "observations": self.observation_count,
            "training_iterations": self.training_iterations,
            "trend": self._performance_trend(),
            "last_training_time": self.last_training_time.isoformat(),
            "recovery_error_count": self.recovery_system.error_count,
            "task_weights": self.multi_task_learner.get_weights(),
            "strategy_health": strategy_health,
        }
    

if __name__ == "__main__":
    print("\n=== Running Learning Agent ===\n")
    printer.status("TEST", "Learning Agent initialized", "info")
    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory

    memory = SharedMemory()
    factory = AgentFactory()
    learning_config = get_config_section("learning_agent")

    agent = LearningAgent(shared_memory=memory, agent_factory=factory, config=learning_config)
    print(agent)

    print("\n=== All tests completed successfully! ===\n")
