"""Tabular recursive learning agent for small discrete domains.

This module intentionally keeps implementation lightweight and dependency-free
(for the learning algorithm itself) while exposing a practical API used across
SLAI.
"""

from __future__ import annotations

import pickle
import random

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np

from src.agents.learning.learning_memory import LearningMemory, Transition
from src.agents.learning.utils.rl_engine import ExplorationStrategies, StateProcessor
from src.agents.learning.utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Recursive Learning")
printer = PrettyPrinter


State = Tuple[Any, ...]
QKey = Tuple[State, Any]


@dataclass(frozen=True)
class RLHyperparameters:
    """Core parameters for tabular Q(λ) learning."""

    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1
    trace_decay: float = 0.7
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    replay_batch_size: int = 32
    replay_interval: int = 20
    replay_updates: int = 1


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

        self.config = load_global_config()
        self.agent_id = agent_id
        self.possible_actions = list(possible_actions)
        self.state_size = state_size

        rl_config = self.config.get("rl", {})
        exploration_cfg = get_config_section("rl_engine").get("exploration_strategies", {})
        self.hparams = RLHyperparameters(
            learning_rate=float(rl_config.get("learning_rate", 0.1)),
            discount_factor=float(rl_config.get("discount_factor", 0.9)),
            epsilon=float(rl_config.get("epsilon", 0.1)),
            trace_decay=float(rl_config.get("trace_decay", 0.7)),
            epsilon_decay=float(exploration_cfg.get("epsilon_decay", 0.995)),
            min_epsilon=float(exploration_cfg.get("min_epsilon", 0.01)),
            replay_batch_size=int(rl_config.get("replay_batch_size", 32)),
            replay_interval=int(rl_config.get("replay_interval", 20)),
            replay_updates=int(rl_config.get("replay_updates", 1)),
        )

        # Mutable runtime parameters
        self.learning_rate = self.hparams.learning_rate
        self.discount_factor = self.hparams.discount_factor
        self.epsilon = self.hparams.epsilon
        self.trace_decay = self.hparams.trace_decay
        self.epsilon_decay = self.hparams.epsilon_decay
        self.min_epsilon = self.hparams.min_epsilon
        self.replay_batch_size = self.hparams.replay_batch_size
        self.replay_interval = self.hparams.replay_interval
        self.replay_updates = self.hparams.replay_updates

        # Shared learning utilities used by the Learning Agent subagents.
        self.state_processor = StateProcessor(state_size=state_size)
        self.exploration = ExplorationStrategies(
            action_space=self.possible_actions,
            strategy=exploration_cfg.get("strategy", "epsilon_greedy"),
            temperature=float(exploration_cfg.get("temperature", 1.0)),
            ucb_c=float(exploration_cfg.get("ucb_c", 2.0)),
        )
        self.learning_memory = LearningMemory()

        # Tabular structures
        self.q_table: Dict[QKey, float] = {}
        self.eligibility_traces: DefaultDict[QKey, float] = defaultdict(float)
        self.state_action_counts: DefaultDict[QKey, int] = defaultdict(int)

        # Episode buffers
        self.state_history: List[State] = []
        self.action_history: List[Any] = []
        self.reward_history: List[float] = []

        self.episode_count = 0
        self.model_id = "RL_Agent"
        self.policy_net = None  # kept for compatibility with external checks

        logger.info("RLAgent '%s' initialized with %d actions", self.agent_id, len(self.possible_actions))

    def _process_state(self, raw_state: Any) -> State:
        """Convert raw observations into a stable hashable tuple state."""
        if isinstance(raw_state, np.ndarray):
            if raw_state.ndim == 0:
                return (float(raw_state),)
            flat_state = raw_state.astype(float).reshape(-1)
            if self.state_processor.feature_engineering:
                engineered = self.state_processor.extract_features(flat_state)
                flat_state = engineered
            discrete = self.state_processor.discretize(flat_state, self.state_processor.num_tilings)
            return tuple(discrete)
        if isinstance(raw_state, list):
            return self._process_state(np.asarray(raw_state, dtype=float))
        if isinstance(raw_state, tuple):
            return raw_state
        return (raw_state,)

    def _get_q_value(self, state: State, action: Any) -> float:
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state: Any) -> Any:
        return self._epsilon_greedy(self._process_state(state))

    def _epsilon_greedy(self, state: State) -> Any:
        if random.random() < self.epsilon:
            return random.choice(self.possible_actions)

        q_values = [self._get_q_value(state, action) for action in self.possible_actions]

        if self.exploration.strategy == "boltzmann":
            return self.exploration.boltzmann(np.array(q_values, dtype=float))

        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.possible_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def step(self, state: Any) -> Any:
        processed = self._process_state(state)
        action = self._epsilon_greedy(processed)
        self.state_history.append(processed)
        self.action_history.append(action)
        return action

    def receive_reward(self, reward: float, state: Any = None, action: Any = None) -> None:
        """Record reward for the latest transition.

        `state`/`action` are accepted for API compatibility.
        """
        self.reward_history.append(float(reward))

    def _update_eligibility(self, state: State, action: Any) -> None:
        self.eligibility_traces[(state, action)] += 1.0

    def _decay_eligibility(self) -> None:
        decay = self.discount_factor * self.trace_decay
        to_delete: List[QKey] = []
        for key in self.eligibility_traces:
            self.eligibility_traces[key] *= decay
            if self.eligibility_traces[key] < 1e-8:
                to_delete.append(key)
        for key in to_delete:
            del self.eligibility_traces[key]

    def learn(self, next_state: Any, reward: float, done: bool) -> None:
        """Apply one Q(λ) update from the most recent (state, action)."""
        if not self.state_history or not self.action_history:
            logger.warning("learn() skipped: no state/action history")
            return

        state = self.state_history[-1]
        action = self.action_history[-1]
        next_processed = self._process_state(next_state)

        current_q = self._get_q_value(state, action)
        next_best = 0.0 if done else max(self._get_q_value(next_processed, a) for a in self.possible_actions)
        td_error = float(reward) + self.discount_factor * next_best - current_q

        transition = Transition(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_processed,
            done=bool(done),
        )
        priority = abs(td_error) + 1e-6
        self.learning_memory.add(transition, priority=priority, tag=self.agent_id)

        self._update_eligibility(state, action)

        for key, eligibility in list(self.eligibility_traces.items()):
            self.q_table[key] = self.q_table.get(key, 0.0) + self.learning_rate * td_error * eligibility

        self._decay_eligibility()
        self.state_action_counts[(state, action)] += 1
        self.episode_count += 1

        self._replay_from_memory()

        if done:
            self.end_episode(next_processed, done=True)

    def end_episode(self, final_state: Any, done: bool) -> None:
        _ = final_state, done
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.eligibility_traces.clear()
        self.reset_history()

    def _replay_from_memory(self) -> None:
        """Replay recent experiences from shared learning memory."""
        if self.replay_interval <= 0 or self.episode_count % self.replay_interval != 0:
            return
        if self.learning_memory.size() < self.replay_batch_size:
            return

        for _ in range(self.replay_updates):
            samples, indices, _weights = self.learning_memory.sample_proportional(self.replay_batch_size)
            if not samples:
                return

            updated_priorities: List[float] = []
            for exp in samples:
                current_q = self._get_q_value(exp.state, exp.action)
                next_best = 0.0 if exp.done else max(
                    self._get_q_value(exp.next_state, a) for a in self.possible_actions
                )
                td_error = exp.reward + self.discount_factor * next_best - current_q
                self.q_table[(exp.state, exp.action)] = current_q + self.learning_rate * td_error
                updated_priorities.append(abs(td_error) + 1e-6)

            self.learning_memory.update_priorities(indices, updated_priorities)

    def reset_history(self) -> None:
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()

    def get_q_table(self) -> Dict[QKey, float]:
        return dict(self.q_table)

    def get_policy(self) -> Dict[State, Any]:
        """Derive deterministic greedy policy from learned Q-values."""
        states = {state for state, _ in self.q_table.keys()}
        policy: Dict[State, Any] = {}
        for state in states:
            q_values = [self._get_q_value(state, action) for action in self.possible_actions]
            best_action = self.possible_actions[int(np.argmax(q_values))]
            policy[state] = best_action
        return policy

    def evaluate(
        self,
        env: Any,
        episodes: int = 20,
        exploration_rate: float = 0.05,
        visualize: bool = False,
    ) -> Dict[str, Any]:
        original_epsilon = self.epsilon
        self.epsilon = exploration_rate

        rewards: List[float] = []
        lengths: List[int] = []

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                if visualize:
                    env.render()
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                steps += 1
                state = next_state

            rewards.append(total_reward)
            lengths.append(steps)

        self.epsilon = original_epsilon

        return {
            "episodes": episodes,
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(min(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "avg_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            "q_table_size": len(self.q_table),
            "exploration_rate": exploration_rate,
            "detailed_rewards": rewards,
        }

    def train(self, env: Any, num_tasks: int = 3, episodes_per_task: int = 5) -> Dict[State, Any]:
        """Task-oriented training loop with per-task and global summaries."""
        task_summaries: List[Dict[str, float]] = []
        global_rewards: List[float] = []

        for _ in range(num_tasks):
            task_rewards: List[float] = []
            task_lengths: List[int] = []

            for _ in range(episodes_per_task):
                state, _ = env.reset()
                done = False
                episode_reward = 0.0
                episode_steps = 0
                while not done:
                    action = self.step(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    self.receive_reward(reward)
                    self.learn(next_state=next_state, reward=reward, done=done)
                    state = next_state
                    episode_reward += float(reward)
                    episode_steps += 1
                if not done:
                    self.end_episode(state, done=False)
                task_rewards.append(episode_reward)
                task_lengths.append(episode_steps)

            task_summary = {
                "avg_reward": float(np.mean(task_rewards)) if task_rewards else 0.0,
                "max_reward": float(max(task_rewards)) if task_rewards else 0.0,
                "min_reward": float(min(task_rewards)) if task_rewards else 0.0,
                "avg_length": float(np.mean(task_lengths)) if task_lengths else 0.0,
            }
            task_summaries.append(task_summary)
            global_rewards.extend(task_rewards)

        self.last_training_report = {
            "num_tasks": num_tasks,
            "episodes_per_task": episodes_per_task,
            "total_episodes": num_tasks * episodes_per_task,
            "global_avg_reward": float(np.mean(global_rewards)) if global_rewards else 0.0,
            "q_table_size": len(self.q_table),
            "memory_size": self.learning_memory.size(),
            "task_summaries": task_summaries,
        }

        return self.get_policy()

    def save_checkpoint(self, path: str) -> None:
        payload = {
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
            "q_table": self.q_table,
            "episode_count": self.episode_count,
        }
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as handle:
            pickle.dump(payload, handle)
        self.learning_memory.save_checkpoint()

    def load_checkpoint(self, path: str) -> None:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        self.q_table = dict(payload.get("q_table", {}))
        self.episode_count = int(payload.get("episode_count", 0))
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.replay_batch_size = int(payload.get("replay_batch_size", self.replay_batch_size))
        self.replay_interval = int(payload.get("replay_interval", self.replay_interval))
        self.replay_updates = int(payload.get("replay_updates", self.replay_updates))


__all__ = ["RLAgent", "RLHyperparameters"]

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Recursive Learning ===\n")
    class _SimpleFiniteEnv:
        """Small deterministic environment for local RLAgent smoke testing."""

        def __init__(self):
            self.goal = 4
            self.max_steps = 12
            self.state = 0
            self.steps = 0

        def reset(self):
            self.state = 0
            self.steps = 0
            return np.array([self.state], dtype=float), {}

        def step(self, action: int):
            self.steps += 1
            if action == 1:
                self.state = min(self.goal, self.state + 1)
            else:
                self.state = max(0, self.state - 1)

            done = self.state == self.goal or self.steps >= self.max_steps
            reward = 1.0 if self.state == self.goal else -0.05
            return np.array([self.state], dtype=float), reward, done, False, {}

        def render(self):
            return None
    
    env = _SimpleFiniteEnv()
    agent = RLAgent(agent_id="rl_main_test", possible_actions=[0, 1], state_size=1)

    policy = agent.train(env=env, num_tasks=3, episodes_per_task=20)
    report = getattr(agent, "last_training_report", {})
    evaluation = agent.evaluate(env=env, episodes=10, exploration_rate=0.0, visualize=False)

    print("Training Report:", report)
    print("Learned Policy States:", len(policy))
    print("Evaluation:", evaluation)
    print("\n=== Recursive Learning Complete ===")
