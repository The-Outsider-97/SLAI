"""
Tabular recursive learning agent for small discrete domains.

This module intentionally keeps implementation lightweight and dependency-free
(for the learning algorithm itself) while exposing a practical API used across
SLAI.
"""

from __future__ import annotations

import pickle
import random
import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from src.agents.learning.learning_memory import LearningMemory, Transition
from src.agents.learning.utils.config_loader import get_config_section, load_global_config
from src.agents.learning.utils.rl_engine import ExplorationStrategies, QTableOptimizer, StateProcessor
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Recursive Learning")
printer = PrettyPrinter

State = Tuple[Any, ...]
QKey = Tuple[State, Any]


@dataclass(frozen=True)
class RLHyperparameters:
    """Core parameters for tabular Q(λ) learning and RL-engine integration."""

    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1
    trace_decay: float = 0.7
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    replay_batch_size: int = 32
    replay_interval: int = 20
    replay_updates: int = 1
    replay_priority_epsilon: float = 1e-6
    max_steps_per_episode: Optional[int] = None
    default_q_value: float = 0.0
    exploration_strategy: str = "epsilon_greedy"
    temperature: float = 1.0
    ucb_c: float = 2.0
    q_optimizer_batch_size: int = 32
    q_optimizer_momentum: float = 0.9
    q_optimizer_cache_size: int = 1000
    q_optimizer_compression: bool = True
    q_optimizer_update_frequency: int = 100


class RLAgent:
    """Recursive learning agent for simple finite state/action tasks.

    Best used when:
    - You need explainability and transparent value estimates.
    - The environment is small and fast to simulate.
    - You are prototyping exploration strategies.

    Mathematical basis:
    - Tabular Q-learning
    - Eligibility traces (Q(λ), accumulating traces)
    - Epsilon-greedy exploration
    """

    def __init__(self, agent_id: str, possible_actions: List[Any], state_size: int):
        if not possible_actions:
            raise ValueError("possible_actions must contain at least one action")
        if int(state_size) <= 0:
            raise ValueError("state_size must be a positive integer")

        self.config = load_global_config()
        rl_config = self.config.get("rl") or {}
        rl_engine_config = get_config_section("rl_engine") or {}
        exploration_cfg = rl_engine_config.get("exploration_strategies", {})
        state_processor_cfg = rl_engine_config.get("state_processor", {})
        q_table_cfg = rl_engine_config.get("q_table_optimizer", {})

        self.agent_id = str(agent_id) if agent_id is not None else "RL"
        self.possible_actions = list(possible_actions)
        self.state_size = int(state_size)

        self.hparams = RLHyperparameters(
            learning_rate=float(rl_config.get("learning_rate", 0.1)),
            discount_factor=float(rl_config.get("discount_factor", 0.9)),
            epsilon=float(rl_config.get("epsilon", 0.1)),
            trace_decay=float(rl_config.get("trace_decay", 0.7)),
            epsilon_decay=float(exploration_cfg.get("epsilon_decay", 0.995)),
            min_epsilon=float(exploration_cfg.get("min_epsilon", 0.01)),
            replay_batch_size=int(q_table_cfg.get("batch_size", rl_config.get("replay_batch_size", 32))),
            replay_interval=int(q_table_cfg.get("update_frequency", rl_config.get("replay_interval", 20))),
            replay_updates=int(rl_config.get("replay_updates", 1)),
            replay_priority_epsilon=float(rl_config.get("replay_priority_epsilon", 1e-6)),
            max_steps_per_episode=rl_config.get("max_steps_per_episode"),
            default_q_value=float(q_table_cfg.get("default_value", 0.0)),
            exploration_strategy=str(exploration_cfg.get("strategy", "epsilon_greedy")),
            temperature=float(exploration_cfg.get("temperature", 1.0)),
            ucb_c=float(exploration_cfg.get("ucb_c", 2.0)),
            q_optimizer_batch_size=int(q_table_cfg.get("batch_size", 32)),
            q_optimizer_momentum=float(q_table_cfg.get("momentum", 0.9)),
            q_optimizer_cache_size=int(q_table_cfg.get("cache_size", 1000)),
            q_optimizer_compression=bool(q_table_cfg.get("compression", True)),
            q_optimizer_update_frequency=int(q_table_cfg.get("update_frequency", 100)),
        )

        if not 0.0 < self.hparams.learning_rate <= 1.0:
            raise ValueError("rl.learning_rate must be in (0, 1].")
        if not 0.0 <= self.hparams.discount_factor <= 1.0:
            raise ValueError("rl.discount_factor must be in [0, 1].")
        if not 0.0 <= self.hparams.epsilon <= 1.0:
            raise ValueError("rl.epsilon must be in [0, 1].")
        if not 0.0 <= self.hparams.trace_decay <= 1.0:
            raise ValueError("rl.trace_decay must be in [0, 1].")
        if not 0.0 < self.hparams.epsilon_decay <= 1.0:
            raise ValueError("rl_engine.exploration_strategies.epsilon_decay must be in (0, 1].")
        if not 0.0 <= self.hparams.min_epsilon <= 1.0:
            raise ValueError("rl_engine.exploration_strategies.min_epsilon must be in [0, 1].")
        if self.hparams.replay_batch_size <= 0:
            raise ValueError("replay_batch_size must be positive.")
        if self.hparams.replay_interval < 0:
            raise ValueError("replay_interval must be >= 0.")
        if self.hparams.replay_updates <= 0:
            raise ValueError("replay_updates must be positive.")

        # Mutable runtime parameters kept for compatibility with the wider SLAI stack.
        self.learning_rate = self.hparams.learning_rate
        self.discount_factor = self.hparams.discount_factor
        self.epsilon = self.hparams.epsilon
        self.trace_decay = self.hparams.trace_decay
        self.epsilon_decay = self.hparams.epsilon_decay
        self.min_epsilon = self.hparams.min_epsilon
        self.replay_batch_size = self.hparams.replay_batch_size
        self.replay_interval = self.hparams.replay_interval
        self.replay_updates = self.hparams.replay_updates
        self.default_q_value = self.hparams.default_q_value
        self.exploration_strategy = self.hparams.exploration_strategy.strip().lower()
        self.temperature = max(self.hparams.temperature, 1e-6)
        self.ucb_c = self.hparams.ucb_c
        self.replay_priority_epsilon = self.hparams.replay_priority_epsilon
        self.max_steps_per_episode = self.hparams.max_steps_per_episode
        self.q_optimizer_update_frequency = self.hparams.q_optimizer_update_frequency
        self.q_optimizer_compression = self.hparams.q_optimizer_compression

        # Shared learning utilities used by the Learning Agent subagents.
        self.state_processor = StateProcessor(
            state_size=self.state_size,
            tiling_resolution=state_processor_cfg.get("tiling_resolution"),
            num_tilings=state_processor_cfg.get("num_tilings"),
            feature_engineering=state_processor_cfg.get("feature_engineering"),
        )
        self.exploration = ExplorationStrategies(
            action_space=self.possible_actions,
            strategy=self.exploration_strategy,
            temperature=self.temperature,
            ucb_c=self.ucb_c,
        )
        self.learning_memory = LearningMemory()
        self.q_optimizer = QTableOptimizer(
            batch_size=self.hparams.q_optimizer_batch_size,
            momentum=self.hparams.q_optimizer_momentum,
            cache_size=self.hparams.q_optimizer_cache_size,
            learning_rate=self.learning_rate,
        )

        # Tabular structures
        self.q_table: Dict[QKey, float] = {}
        self.eligibility_traces: DefaultDict[QKey, float] = defaultdict(float)
        self.state_action_counts: DefaultDict[QKey, int] = defaultdict(int)

        # Episode buffers
        self.state_history: List[State] = []
        self.action_history: List[Any] = []
        self.reward_history: List[float] = []

        # Runtime metrics
        self.episode_count = 0
        self.total_steps = 0
        self.total_learning_updates = 0
        self.total_replay_updates = 0
        self.completed_episode_rewards: List[float] = []
        self.completed_episode_lengths: List[int] = []
        self.last_learning_metrics: Optional[Dict[str, float]] = None
        self.last_episode_report: Dict[str, Any] = {}
        self.last_training_report: Dict[str, Any] = {}
        self.model_id = "RL_Agent"
        self.policy_net = None  # kept for compatibility with external checks

        logger.info(
            "RLAgent '%s' initialized | actions=%d strategy=%s lr=%.4f gamma=%.4f epsilon=%.4f",
            self.agent_id,
            len(self.possible_actions),
            self.exploration_strategy,
            self.learning_rate,
            self.discount_factor,
            self.epsilon,
        )

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_reset(env: Any) -> Tuple[Any, Dict[str, Any]]:
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, info = reset_result
            return state, info if isinstance(info, dict) else {}
        return reset_result, {}

    @staticmethod
    def _safe_step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        step_result = env.step(action)
        if not isinstance(step_result, tuple):
            raise TypeError("Environment step(...) must return a tuple.")
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            return next_state, float(reward), bool(terminated), bool(truncated), info if isinstance(info, dict) else {}
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
            return next_state, float(reward), bool(done), False, info if isinstance(info, dict) else {}
        raise ValueError(f"Unsupported environment step() output length: {len(step_result)}")

    # ------------------------------------------------------------------
    # State / value helpers
    # ------------------------------------------------------------------
    def _to_numeric_array(self, raw_state: Any) -> Optional[np.ndarray]:
        if isinstance(raw_state, np.ndarray):
            arr = raw_state.astype(np.float32, copy=False).reshape(-1)
            return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        if isinstance(raw_state, (int, float, np.number, bool)):
            return np.asarray([raw_state], dtype=np.float32)
        if isinstance(raw_state, (list, tuple)):
            try:
                arr = np.asarray(raw_state, dtype=np.float32).reshape(-1)
                return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            except (TypeError, ValueError):
                return None
        if isinstance(raw_state, dict):
            numeric_parts: List[float] = []
            for key in sorted(raw_state.keys(), key=str):
                child = self._to_numeric_array(raw_state[key])
                if child is None:
                    return None
                numeric_parts.extend(child.tolist())
            return np.asarray(numeric_parts, dtype=np.float32)
        if hasattr(raw_state, "detach") and hasattr(raw_state, "cpu") and hasattr(raw_state, "numpy"):
            arr = np.asarray(raw_state.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        if hasattr(raw_state, "tolist"):
            try:
                arr = np.asarray(raw_state.tolist(), dtype=np.float32).reshape(-1)
                return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            except (TypeError, ValueError):
                return None
        return None

    def _hashable_state(self, raw_state: Any) -> State:
        if isinstance(raw_state, tuple):
            return tuple(self._hashable_state(item) if isinstance(item, (list, tuple, dict)) else item for item in raw_state)
        if isinstance(raw_state, list):
            return tuple(self._hashable_state(item) if isinstance(item, (list, tuple, dict)) else item for item in raw_state)
        if isinstance(raw_state, dict):
            return tuple((str(key), self._hashable_state(value) if isinstance(value, (list, tuple, dict)) else value) for key, value in sorted(raw_state.items(), key=lambda kv: str(kv[0])))
        if isinstance(raw_state, np.ndarray):
            return tuple(np.asarray(raw_state).reshape(-1).tolist())
        return (raw_state,)

    def _process_state(self, raw_state: Any) -> State:
        """Convert raw observations into a stable, hashable tabular state."""
        numeric = self._to_numeric_array(raw_state)
        if numeric is not None:
            processed = numeric
            if self.state_processor.feature_engineering:
                processed = self.state_processor.extract_features(processed)
            discrete = self.state_processor.discretize(processed, self.state_processor.num_tilings)
            return tuple(int(x) for x in discrete)
        return self._hashable_state(raw_state)

    def _get_q_value(self, state: State, action: Any) -> float:
        key = (state, action)
        if key in self.q_table:
            return float(self.q_table[key])
        mirrored = self.q_optimizer._get_q_value(state, action)
        return float(mirrored if mirrored is not None else self.default_q_value)

    def _set_q_value(self, state: State, action: Any, value: float) -> None:
        key = (state, action)
        float_value = float(value)
        self.q_table[key] = float_value
        self.q_optimizer._set_q_value(state, action, float_value)

    def _compress_q_value(self, state: State, action: Any, value: float) -> None:
        if not self.q_optimizer_compression:
            return
        if self.total_learning_updates <= 0:
            return
        if self.total_learning_updates % self.q_optimizer_update_frequency != 0:
            return
        self.q_optimizer.compressed_store(state=state, action=action, value=value)

    def get_action_values(self, state: Any) -> Dict[Any, float]:
        processed = self._process_state(state)
        return {action: self._get_q_value(processed, action) for action in self.possible_actions}

    def get_state_value(self, state: Any) -> float:
        values = list(self.get_action_values(state).values())
        return float(max(values)) if values else self.default_q_value

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def _greedy_action(self, state: State) -> Any:
        q_values = np.asarray([self._get_q_value(state, action) for action in self.possible_actions], dtype=np.float64)
        max_q = float(np.max(q_values)) if q_values.size else 0.0
        best_actions = [action for action, q in zip(self.possible_actions, q_values) if float(q) == max_q]
        return random.choice(best_actions)

    def _strategy_action(self, state: State) -> Any:
        q_values = np.asarray([self._get_q_value(state, action) for action in self.possible_actions], dtype=np.float64)

        if self.exploration_strategy == "boltzmann":
            stabilized = q_values - float(np.max(q_values)) if q_values.size else q_values
            self.exploration.temperature = max(self.temperature, 1e-6)
            return self.exploration.boltzmann(stabilized)

        if self.exploration_strategy == "ucb":
            self.exploration.q_table = self.q_table
            self.exploration.state_history = [state]
            return self.exploration.ucb(self.state_action_counts, c=self.ucb_c)

        return self._greedy_action(state)

    def _epsilon_greedy(self, state: State) -> Any:
        if random.random() < self.epsilon:
            return random.choice(self.possible_actions)
        return self._strategy_action(state)

    def select_action(self, state: Any) -> Any:
        return self._epsilon_greedy(self._process_state(state))

    def act(self, state: Any) -> Any:
        return self.select_action(state)

    def step(self, state: Any) -> Any:
        processed = self._process_state(state)
        action = self._epsilon_greedy(processed)
        self.state_history.append(processed)
        self.action_history.append(action)
        return action

    def receive_reward(self, reward: float, state: Any = None, action: Any = None) -> None:
        """Record reward for the latest transition.

        `state`/`action` are accepted for API compatibility and may be used to
        backfill a transition when external callers provide explicit values.
        """
        if state is not None and action is not None:
            processed = self._process_state(state)
            if not self.state_history or self.state_history[-1] != processed:
                self.state_history.append(processed)
            if not self.action_history or self.action_history[-1] != action:
                self.action_history.append(action)
        self.reward_history.append(float(reward))

    # ------------------------------------------------------------------
    # Learning updates
    # ------------------------------------------------------------------
    def _update_eligibility(self, state: State, action: Any) -> None:
        self.eligibility_traces[(state, action)] += 1.0

    def _decay_eligibility(self) -> None:
        decay = self.discount_factor * self.trace_decay
        expired: List[QKey] = []
        for key in list(self.eligibility_traces.keys()):
            self.eligibility_traces[key] *= decay
            if self.eligibility_traces[key] < 1e-8:
                expired.append(key)
        for key in expired:
            del self.eligibility_traces[key]

    def learn_transition(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> Dict[str, float]:
        processed_state = self._process_state(state)
        processed_next_state = self._process_state(next_state)
        reward_value = float(reward)
        done_flag = bool(done)

        current_q = self._get_q_value(processed_state, action)
        next_best = 0.0 if done_flag else max(
            self._get_q_value(processed_next_state, next_action) for next_action in self.possible_actions
        )
        td_target = reward_value + self.discount_factor * next_best
        td_error = td_target - current_q

        transition = Transition(
            state=processed_state,
            action=action,
            reward=reward_value,
            next_state=processed_next_state,
            done=done_flag,
        )
        priority = abs(td_error) + self.replay_priority_epsilon
        self.learning_memory.add(transition, priority=priority, tag=self.agent_id)

        self._update_eligibility(processed_state, action)

        updated_pairs: List[Tuple[State, Any, float]] = []
        for key, eligibility in list(self.eligibility_traces.items()):
            state_key, action_key = key
            updated_value = self._get_q_value(state_key, action_key) + self.learning_rate * td_error * eligibility
            self._set_q_value(state_key, action_key, updated_value)
            updated_pairs.append((state_key, action_key, updated_value))

        self._decay_eligibility()
        self.state_action_counts[(processed_state, action)] += 1
        self.total_steps += 1
        self.total_learning_updates += 1

        for state_key, action_key, value in updated_pairs:
            self._compress_q_value(state_key, action_key, value)

        self._replay_from_memory()

        metrics = {
            "td_error": float(td_error),
            "td_target": float(td_target),
            "current_q": float(current_q),
            "next_best_q": float(next_best),
            "epsilon": float(self.epsilon),
            "q_table_size": float(len(self.q_table)),
            "memory_size": float(self.learning_memory.size()),
        }
        self.last_learning_metrics = metrics

        if done_flag:
            self.end_episode(processed_next_state, done=True)

        return metrics

    def learn(self, next_state: Any, reward: float, done: bool) -> Optional[Dict[str, float]]:
        """Apply one Q(λ) update from the most recent (state, action)."""
        if not self.state_history or not self.action_history:
            logger.warning("learn() skipped: no state/action history")
            return None

        if len(self.reward_history) < len(self.action_history):
            self.reward_history.append(float(reward))

        return self.learn_transition(
            state=self.state_history[-1],
            action=self.action_history[-1],
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def end_episode(self, final_state: Any, done: bool) -> Dict[str, Any]:
        final_processed = self._process_state(final_state)
        episode_length = len(self.action_history)
        episode_reward = float(np.sum(self.reward_history)) if self.reward_history else 0.0

        if episode_length > 0:
            self.episode_count += 1
            self.completed_episode_rewards.append(episode_reward)
            self.completed_episode_lengths.append(episode_length)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.eligibility_traces.clear()
        self.last_episode_report = {
            "episode": int(self.episode_count),
            "done": bool(done),
            "final_state": final_processed,
            "episode_reward": episode_reward,
            "episode_length": int(episode_length),
            "epsilon": float(self.epsilon),
            "q_table_size": int(len(self.q_table)),
        }
        self.reset_history()
        return dict(self.last_episode_report)

    def _replay_from_memory(self) -> None:
        """Replay prioritized experiences from shared learning memory."""
        if self.replay_interval <= 0:
            return
        if self.total_learning_updates <= 0 or self.total_learning_updates % self.replay_interval != 0:
            return
        if self.learning_memory.size() < self.replay_batch_size:
            return

        for _ in range(self.replay_updates):
            samples, indices, weights = self.learning_memory.sample_proportional(self.replay_batch_size)
            if not samples:
                return

            updates: List[Tuple[State, Any, float]] = []
            updated_priorities: List[float] = []
            for exp, weight in zip(samples, weights):
                current_q = self._get_q_value(exp.state, exp.action)
                next_best = 0.0 if exp.done else max(
                    self._get_q_value(exp.next_state, action) for action in self.possible_actions
                )
                td_target = float(exp.reward) + self.discount_factor * next_best
                td_error = td_target - current_q
                weighted_delta = float(weight) * td_error
                updates.append((exp.state, exp.action, weighted_delta))
                updated_priorities.append(abs(td_error) + self.replay_priority_epsilon)

            self.q_optimizer.batch_update(
                updates=updates,
                batch_size=self.hparams.q_optimizer_batch_size,
                momentum=self.hparams.q_optimizer_momentum,
            )
            for state, action, _ in updates:
                refreshed = self.q_optimizer._get_q_value(state, action)
                self.q_table[(state, action)] = float(refreshed)
                self._compress_q_value(state, action, float(refreshed))

            self.learning_memory.update_priorities(indices, updated_priorities)
            self.total_replay_updates += 1

    # ------------------------------------------------------------------
    # Public table / policy accessors
    # ------------------------------------------------------------------
    def reset_history(self) -> None:
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()

    def reset_learning_state(self, clear_memory: bool = False) -> None:
        self.q_table.clear()
        self.eligibility_traces.clear()
        self.state_action_counts.clear()
        self.reset_history()
        self.q_optimizer = QTableOptimizer(
            batch_size=self.hparams.q_optimizer_batch_size,
            momentum=self.hparams.q_optimizer_momentum,
            cache_size=self.hparams.q_optimizer_cache_size,
            learning_rate=self.learning_rate,
        )
        self.episode_count = 0
        self.total_steps = 0
        self.total_learning_updates = 0
        self.total_replay_updates = 0
        self.completed_episode_rewards.clear()
        self.completed_episode_lengths.clear()
        self.last_learning_metrics = None
        self.last_episode_report = {}
        self.last_training_report = {}
        self.epsilon = self.hparams.epsilon
        if clear_memory:
            self.learning_memory.clear()

    def get_q_table(self) -> Dict[QKey, float]:
        return dict(self.q_table)

    def get_policy(self) -> Dict[State, Any]:
        """Derive deterministic greedy policy from learned Q-values."""
        states = {state for state, _ in self.q_table.keys()}
        policy: Dict[State, Any] = {}
        for state in states:
            q_values = [self._get_q_value(state, action) for action in self.possible_actions]
            max_q = max(q_values) if q_values else self.default_q_value
            best_actions = [action for action, q in zip(self.possible_actions, q_values) if q == max_q]
            policy[state] = random.choice(best_actions)
        return policy

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "state_size": self.state_size,
            "num_actions": len(self.possible_actions),
            "epsilon": float(self.epsilon),
            "learning_rate": float(self.learning_rate),
            "discount_factor": float(self.discount_factor),
            "trace_decay": float(self.trace_decay),
            "exploration_strategy": self.exploration_strategy,
            "temperature": float(self.temperature),
            "ucb_c": float(self.ucb_c),
            "episode_count": int(self.episode_count),
            "total_steps": int(self.total_steps),
            "total_learning_updates": int(self.total_learning_updates),
            "total_replay_updates": int(self.total_replay_updates),
            "q_table_size": int(len(self.q_table)),
            "memory_size": int(self.learning_memory.size()),
            "avg_episode_reward": float(np.mean(self.completed_episode_rewards)) if self.completed_episode_rewards else 0.0,
            "avg_episode_length": float(np.mean(self.completed_episode_lengths)) if self.completed_episode_lengths else 0.0,
            "last_learning_metrics": dict(self.last_learning_metrics) if self.last_learning_metrics else None,
            "last_episode_report": dict(self.last_episode_report),
        }

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        env: Any,
        episodes: int = 20,
        exploration_rate: float = 0.05,
        visualize: bool = False,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        original_epsilon = self.epsilon
        self.epsilon = float(exploration_rate)

        rewards: List[float] = []
        lengths: List[int] = []
        action_distribution = {action: 0 for action in self.possible_actions}

        try:
            for _ in range(episodes):
                state, _ = self._safe_reset(env)
                done = False
                total_reward = 0.0
                steps = 0
                step_limit = max_steps if max_steps is not None else self.max_steps_per_episode

                while not done:
                    if step_limit is not None and steps >= int(step_limit):
                        break
                    if visualize:
                        env.render()
                    action = self.select_action(state)
                    next_state, reward, terminated, truncated, _ = self._safe_step(env, action)
                    done = terminated or truncated
                    total_reward += float(reward)
                    steps += 1
                    action_distribution[action] = action_distribution.get(action, 0) + 1
                    state = next_state

                rewards.append(total_reward)
                lengths.append(steps)
        finally:
            self.epsilon = original_epsilon

        total_actions = sum(action_distribution.values())
        normalized_distribution = {
            action: (count / total_actions if total_actions else 0.0)
            for action, count in action_distribution.items()
        }

        try:
            reward_threshold = getattr(getattr(env, "spec", None), "reward_threshold")
        except Exception:
            reward_threshold = None
        if reward_threshold is None:
            reward_threshold = max(rewards) * 0.9 if rewards else 0.0

        return {
            "episodes": int(episodes),
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(min(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "success_rate": float((np.asarray(rewards) >= reward_threshold).mean()) if rewards else 0.0,
            "avg_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "q_table_size": int(len(self.q_table)),
            "memory_size": int(self.learning_memory.size()),
            "action_distribution": normalized_distribution,
            "exploration_rate": float(exploration_rate),
            "reward_threshold": float(reward_threshold),
            "detailed_rewards": rewards,
            "episode_lengths": lengths,
        }

    def train(
        self,
        env: Any,
        num_tasks: int = 3,
        episodes_per_task: int = 5,
        max_steps_per_episode: Optional[int] = None,
        visualize: bool = False,
    ) -> Dict[State, Any]:
        """Task-oriented training loop with per-task and global summaries."""
        if num_tasks <= 0 or episodes_per_task <= 0:
            raise ValueError("num_tasks and episodes_per_task must both be positive")

        task_summaries: List[Dict[str, float]] = []
        global_rewards: List[float] = []
        global_lengths: List[int] = []
        step_limit = max_steps_per_episode if max_steps_per_episode is not None else self.max_steps_per_episode

        try:
            reward_threshold = getattr(getattr(env, "spec", None), "reward_threshold")
        except Exception:
            reward_threshold = None

        for task_idx in range(num_tasks):
            task_rewards: List[float] = []
            task_lengths: List[int] = []
            task_successes = 0

            for episode_idx in range(episodes_per_task):
                state, _ = self._safe_reset(env)
                done = False
                episode_reward = 0.0
                episode_steps = 0

                while not done:
                    if step_limit is not None and episode_steps >= int(step_limit):
                        break
                    if visualize:
                        env.render()
                    action = self.step(state)
                    next_state, reward, terminated, truncated, _ = self._safe_step(env, action)
                    done = terminated or truncated
                    self.receive_reward(reward, state=state, action=action)
                    self.learn(next_state=next_state, reward=reward, done=done)
                    state = next_state
                    episode_reward += float(reward)
                    episode_steps += 1

                if not done:
                    self.end_episode(state, done=False)

                if reward_threshold is not None and episode_reward >= reward_threshold:
                    task_successes += 1
                task_rewards.append(episode_reward)
                task_lengths.append(episode_steps)

                logger.debug(
                    "Task %s/%s | Episode %s/%s | reward=%.4f steps=%s epsilon=%.4f",
                    task_idx + 1,
                    num_tasks,
                    episode_idx + 1,
                    episodes_per_task,
                    episode_reward,
                    episode_steps,
                    self.epsilon,
                )

            task_summary = {
                "avg_reward": float(np.mean(task_rewards)) if task_rewards else 0.0,
                "max_reward": float(max(task_rewards)) if task_rewards else 0.0,
                "min_reward": float(min(task_rewards)) if task_rewards else 0.0,
                "avg_length": float(np.mean(task_lengths)) if task_lengths else 0.0,
                "success_rate": float(task_successes / episodes_per_task),
            }
            task_summaries.append(task_summary)
            global_rewards.extend(task_rewards)
            global_lengths.extend(task_lengths)

        self.last_training_report = {
            "num_tasks": int(num_tasks),
            "episodes_per_task": int(episodes_per_task),
            "total_episodes": int(num_tasks * episodes_per_task),
            "global_avg_reward": float(np.mean(global_rewards)) if global_rewards else 0.0,
            "global_std_reward": float(np.std(global_rewards)) if global_rewards else 0.0,
            "global_avg_length": float(np.mean(global_lengths)) if global_lengths else 0.0,
            "q_table_size": int(len(self.q_table)),
            "memory_size": int(self.learning_memory.size()),
            "epsilon": float(self.epsilon),
            "task_summaries": task_summaries,
            "diagnostics": self.diagnostics(),
        }

        logger.info(
            "RLAgent '%s' training complete | episodes=%s avg_reward=%.4f q_table=%s memory=%s",
            self.agent_id,
            num_tasks * episodes_per_task,
            self.last_training_report["global_avg_reward"],
            self.last_training_report["q_table_size"],
            self.last_training_report["memory_size"],
        )

        return self.get_policy()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _q_optimizer_state(self) -> Dict[str, Any]:
        return {
            "batch_size": self.q_optimizer.batch_size,
            "momentum": self.q_optimizer.momentum,
            "cache_size": self.q_optimizer.cache_size,
            "learning_rate": self.q_optimizer.learning_rate,
            "state_index": dict(self.q_optimizer.state_index),
            "action_index": dict(self.q_optimizer.action_index),
            "next_state_idx": int(self.q_optimizer.next_state_idx),
            "next_action_idx": int(self.q_optimizer.next_action_idx),
            "state_action_matrix": {state: dict(actions) for state, actions in self.q_optimizer.state_action_matrix.items()},
            "lru_cache": list(self.q_optimizer.lru_cache.items()),
            "sparse_matrix": list(self.q_optimizer.sparse_matrix),
            "update_momentum": dict(self.q_optimizer.update_momentum),
            "default_value": float(self.q_optimizer.DEFAULT_VALUE),
            "huffman_codes": dict(self.q_optimizer.huffman_codes),
            "inverse_huffman": dict(self.q_optimizer.inverse_huffman),
            "code_counter": int(self.q_optimizer.code_counter),
        }

    def _restore_q_optimizer_state(self, state: Dict[str, Any]) -> None:
        self.q_optimizer = QTableOptimizer(
            batch_size=int(state.get("batch_size", self.hparams.q_optimizer_batch_size)),
            momentum=float(state.get("momentum", self.hparams.q_optimizer_momentum)),
            cache_size=int(state.get("cache_size", self.hparams.q_optimizer_cache_size)),
            learning_rate=float(state.get("learning_rate", self.learning_rate)),
        )
        self.q_optimizer.state_index = defaultdict(int, state.get("state_index", {}))
        self.q_optimizer.action_index = defaultdict(int, state.get("action_index", {}))
        self.q_optimizer.next_state_idx = int(state.get("next_state_idx", 0))
        self.q_optimizer.next_action_idx = int(state.get("next_action_idx", 0))
        self.q_optimizer.state_action_matrix = defaultdict(dict, {
            tuple(state_key) if isinstance(state_key, list) else state_key: dict(action_map)
            for state_key, action_map in state.get("state_action_matrix", {}).items()
        })
        self.q_optimizer.lru_cache.clear()
        for key, value in state.get("lru_cache", []):
            normalized_key = tuple(key) if isinstance(key, list) else key
            self.q_optimizer.lru_cache[normalized_key] = float(value)
        self.q_optimizer.sparse_matrix = list(state.get("sparse_matrix", []))
        self.q_optimizer.update_momentum = defaultdict(float, {
            tuple(key) if isinstance(key, list) else key: float(value)
            for key, value in state.get("update_momentum", {}).items()
        })
        self.q_optimizer.DEFAULT_VALUE = float(state.get("default_value", self.default_q_value))
        self.q_optimizer.huffman_codes = dict(state.get("huffman_codes", {}))
        self.q_optimizer.inverse_huffman = dict(state.get("inverse_huffman", {}))
        self.q_optimizer.code_counter = int(state.get("code_counter", 0))

    def save_checkpoint(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        memory_path = path_obj.with_name(f"{path_obj.stem}_memory.pt")

        payload = {
            "version": 2,
            "agent_id": self.agent_id,
            "possible_actions": self.possible_actions,
            "state_size": self.state_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "trace_decay": self.trace_decay,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "replay_batch_size": self.replay_batch_size,
            "replay_interval": self.replay_interval,
            "replay_updates": self.replay_updates,
            "default_q_value": self.default_q_value,
            "exploration_strategy": self.exploration_strategy,
            "temperature": self.temperature,
            "ucb_c": self.ucb_c,
            "q_table": dict(self.q_table),
            "eligibility_traces": dict(self.eligibility_traces),
            "state_action_counts": dict(self.state_action_counts),
            "state_history": list(self.state_history),
            "action_history": list(self.action_history),
            "reward_history": list(self.reward_history),
            "episode_count": int(self.episode_count),
            "total_steps": int(self.total_steps),
            "total_learning_updates": int(self.total_learning_updates),
            "total_replay_updates": int(self.total_replay_updates),
            "completed_episode_rewards": list(self.completed_episode_rewards),
            "completed_episode_lengths": list(self.completed_episode_lengths),
            "last_learning_metrics": dict(self.last_learning_metrics) if self.last_learning_metrics else None,
            "last_episode_report": dict(self.last_episode_report),
            "last_training_report": dict(self.last_training_report),
            "q_optimizer_state": self._q_optimizer_state(),
            "learning_memory_checkpoint": str(memory_path),
        }

        with path_obj.open("wb") as handle:
            pickle.dump(payload, handle)
        self.learning_memory.save_checkpoint(str(memory_path))
        logger.info("Saved RLAgent checkpoint to %s", path_obj)

    def load_checkpoint(self, path: str) -> None:
        path_obj = Path(path)
        with path_obj.open("rb") as handle:
            payload = pickle.load(handle)

        restored_q_table: Dict[QKey, float] = {}
        for key, value in payload.get("q_table", {}).items():
            state_key, action_key = key
            normalized_state = tuple(state_key) if isinstance(state_key, list) else state_key
            restored_q_table[(normalized_state, action_key)] = float(value)
        self.q_table = restored_q_table

        self.eligibility_traces = defaultdict(float)
        for key, value in payload.get("eligibility_traces", {}).items():
            state_key, action_key = key
            normalized_state = tuple(state_key) if isinstance(state_key, list) else state_key
            self.eligibility_traces[(normalized_state, action_key)] = float(value)

        self.state_action_counts = defaultdict(int)
        for key, value in payload.get("state_action_counts", {}).items():
            state_key, action_key = key
            normalized_state = tuple(state_key) if isinstance(state_key, list) else state_key
            self.state_action_counts[(normalized_state, action_key)] = int(value)

        self.state_history = [tuple(state) if isinstance(state, list) else state for state in payload.get("state_history", [])]
        self.action_history = list(payload.get("action_history", []))
        self.reward_history = [float(reward) for reward in payload.get("reward_history", [])]
        self.episode_count = int(payload.get("episode_count", 0))
        self.total_steps = int(payload.get("total_steps", 0))
        self.total_learning_updates = int(payload.get("total_learning_updates", 0))
        self.total_replay_updates = int(payload.get("total_replay_updates", 0))
        self.completed_episode_rewards = [float(value) for value in payload.get("completed_episode_rewards", [])]
        self.completed_episode_lengths = [int(value) for value in payload.get("completed_episode_lengths", [])]
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.discount_factor = float(payload.get("discount_factor", self.discount_factor))
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.trace_decay = float(payload.get("trace_decay", self.trace_decay))
        self.epsilon_decay = float(payload.get("epsilon_decay", self.epsilon_decay))
        self.min_epsilon = float(payload.get("min_epsilon", self.min_epsilon))
        self.replay_batch_size = int(payload.get("replay_batch_size", self.replay_batch_size))
        self.replay_interval = int(payload.get("replay_interval", self.replay_interval))
        self.replay_updates = int(payload.get("replay_updates", self.replay_updates))
        self.default_q_value = float(payload.get("default_q_value", self.default_q_value))
        self.exploration_strategy = str(payload.get("exploration_strategy", self.exploration_strategy))
        self.temperature = float(payload.get("temperature", self.temperature))
        self.ucb_c = float(payload.get("ucb_c", self.ucb_c))
        self.last_learning_metrics = payload.get("last_learning_metrics")
        self.last_episode_report = dict(payload.get("last_episode_report", {}))
        self.last_training_report = dict(payload.get("last_training_report", {}))

        self._restore_q_optimizer_state(payload.get("q_optimizer_state", {}))
        for (state, action), value in self.q_table.items():
            self.q_optimizer._set_q_value(state, action, value)

        memory_checkpoint = payload.get("learning_memory_checkpoint")
        if memory_checkpoint and Path(memory_checkpoint).exists():
            self.learning_memory.load_checkpoint(str(memory_checkpoint))

        logger.info("Loaded RLAgent checkpoint from %s", path_obj)

    def save(self, path: str) -> None:
        self.save_checkpoint(path)

    def load(self, path: str) -> None:
        self.load_checkpoint(path)


__all__ = ["RLAgent", "RLHyperparameters"]


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Recursive Learning ===\n")
    from src.agents.learning.slaienv import SLAIEnv

    env = SLAIEnv(state_dim=4, action_dim=3)
    possible_actions = list(range(env.action_space.n)) if hasattr(env.action_space, "n") else [0, 1, 2]
    state_size = int(env.observation_space.shape[0]) if hasattr(env.observation_space, "shape") else 4

    agent = RLAgent(agent_id="rl_main_test", possible_actions=possible_actions, state_size=state_size)
    policy = agent.train(env=env, num_tasks=3, episodes_per_task=10)
    report = getattr(agent, "last_training_report", {})
    evaluation = agent.evaluate(env=env, episodes=5, exploration_rate=0.0, visualize=False)

    print("Training Report:", report)
    print("Learned Policy States:", len(policy))
    print("Evaluation:", evaluation)
    print("\n=== Recursive Learning Complete ===")
