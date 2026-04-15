import heapq
import random
import time
import numpy as np

from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import RLock
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from ...utils.metrics_utils import FairnessMetrics, MetricSummarizer, PerformanceMetrics
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Distributed Replay Buffer")
printer = PrettyPrinter

PUSH_TIMEOUT_SECONDS = 5.0
MIN_PRIORITY = 1e-5

Experience = Tuple[Any, Any, Any, float, Any, bool]


class DistributedReplayBuffer:
    """Production-grade distributed replay buffer supporting multiple sampling strategies."""

    def __init__(self, user_config: Optional[Dict[str, Any]] = None):
        self.config = load_global_config()
        dist_config = dict(get_config_section("distributed") or {})
        if user_config:
            dist_config.update(user_config)
        self.buffer_config = dist_config

        self.capacity = int(dist_config.get("capacity", 100_000))
        self.staleness_threshold = timedelta(days=float(dist_config.get("staleness_threshold_days", 1)))
        self.alpha = float(dist_config.get("prioritization_alpha", 0.6))
        self.default_beta = float(dist_config.get("importance_sampling_beta", 0.4))
        seed = dist_config.get("seed")

        self.buffer: Deque[Experience] = deque(maxlen=self.capacity)
        self.timestamps: Deque[datetime] = deque(maxlen=self.capacity)
        self.priorities: List[Tuple[float, int]] = []  # max-heap through negative values
        self.agent_experience_map: Dict[Any, int] = defaultdict(int)
        self.agent_rewards: Dict[Any, List[float]] = defaultdict(list)
        self.reward_stats = {"sum": 0.0, "max": -np.inf, "min": np.inf}
        self.fairness_stats = {
            "demographic_parity_violations": 0,
            "fair_batches_checked": 0,
            "last_violation_message": "",
        }
        self.metric_provenance = {
            "framework": "Mitchell model cards + Barocas fairness checks",
            "created_at": datetime.utcnow().isoformat(),
        }

        self.lock = RLock()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        logger.info(
            "Initialized distributed replay buffer (capacity=%s, alpha=%s, staleness=%s)",
            self.capacity,
            self.alpha,
            self.staleness_threshold,
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return (
            "DistributedReplayBuffer("
            f"size={len(self.buffer)}, capacity={self.capacity}, alpha={self.alpha}, "
            f"staleness_threshold={self.staleness_threshold})"
        )

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            size = len(self.buffer)
            return {
                "total_experiences": size,
                "avg_reward": self.reward_stats["sum"] / size if size else 0.0,
                "max_reward": self.reward_stats["max"] if size else None,
                "min_reward": self.reward_stats["min"] if size else None,
                "active_agents": len(self.agent_experience_map),
                "staleness_threshold": str(self.staleness_threshold),
                "priority_alpha": self.alpha,
            }

    def push(
        self,
        agent_id: Any,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        done: Any,
        priority: Optional[float] = None,
    ) -> bool:
        start_time = time.perf_counter()
        printer.status("PUSH", "Attempting to push experience...", "info")

        acquired = self.lock.acquire(timeout=PUSH_TIMEOUT_SECONDS)
        if not acquired:
            logger.error("[ReplayBuffer] Lock acquisition failed — timeout exceeded")
            printer.status("PUSH", "❌ Lock timeout — skipping push", "error")
            return False

        try:
            timestamp = datetime.utcnow()
            normalized_reward = float(reward) if isinstance(reward, (int, float, np.number)) else 0.0
            if normalized_reward != reward:
                logger.warning("[ReplayBuffer] Invalid reward type %s. Defaulting to 0.0", type(reward))

            normalized_done = bool(done)
            resolved_priority = self._resolve_priority(priority=priority, reward=normalized_reward)

            evicted_experience = self.buffer[0] if len(self.buffer) == self.capacity else None
            self.buffer.append((agent_id, state, action, normalized_reward, next_state, normalized_done))
            self.timestamps.append(timestamp)

            if evicted_experience is not None:
                evicted_agent_id = evicted_experience[0]
                if self.agent_experience_map[evicted_agent_id] > 0:
                    self.agent_experience_map[evicted_agent_id] -= 1
                    if self.agent_experience_map[evicted_agent_id] == 0:
                        del self.agent_experience_map[evicted_agent_id]

            index = len(self.buffer) - 1
            heapq.heappush(self.priorities, (-resolved_priority, index))
            self.agent_experience_map[agent_id] += 1
            self._track_reward_stats(agent_id=agent_id, reward=normalized_reward)

            duration = time.perf_counter() - start_time
            printer.status(
                "PUSH",
                f"✔ Pushed | Agent: {agent_id} | Priority: {resolved_priority:.4f} | Time: {duration:.4f}s",
                "success",
            )
            return True
        except Exception:
            logger.exception("[ReplayBuffer] Exception in push()")
            printer.status("PUSH", "❌ Exception during push", "error")
            return False
        finally:
            self.lock.release()

    def _resolve_priority(self, priority: Optional[float], reward: float) -> float:
        if priority is None or not isinstance(priority, (int, float, np.number)) or float(priority) < 0:
            priority = abs(float(reward)) + MIN_PRIORITY
        return max(MIN_PRIORITY, float(priority)) ** self.alpha

    def _track_reward_stats(self, agent_id: Any, reward: float) -> None:
        self.reward_stats["sum"] += reward
        self.reward_stats["max"] = max(self.reward_stats["max"], reward)
        self.reward_stats["min"] = min(self.reward_stats["min"], reward)
        self.agent_rewards[agent_id].append(reward)

    def _remove_stale_experiences(self) -> None:
        now = datetime.utcnow()
        filtered_buffer: Deque[Experience] = deque(maxlen=self.capacity)
        filtered_timestamps: Deque[datetime] = deque(maxlen=self.capacity)

        removed_count = 0
        for exp, ts in zip(self.buffer, self.timestamps):
            if now - ts <= self.staleness_threshold:
                filtered_buffer.append(exp)
                filtered_timestamps.append(ts)
            else:
                removed_count += 1

        if removed_count:
            logger.info("Removed %s stale experiences", removed_count)
            self.buffer = filtered_buffer
            self.timestamps = filtered_timestamps
            self._rebuild_statistics()

    def _rebuild_statistics(self) -> None:
        self.agent_experience_map = defaultdict(int)
        self.agent_rewards = defaultdict(list)
        self.reward_stats = {"sum": 0.0, "max": -np.inf, "min": np.inf}
        self.priorities = []

        for idx, exp in enumerate(self.buffer):
            agent_id, _, _, reward, _, _ = exp
            self.agent_experience_map[agent_id] += 1
            self.agent_rewards[agent_id].append(reward)
            self.reward_stats["sum"] += reward
            self.reward_stats["max"] = max(self.reward_stats["max"], reward)
            self.reward_stats["min"] = min(self.reward_stats["min"], reward)
            heapq.heappush(self.priorities, (-self._resolve_priority(priority=None, reward=reward), idx))

    def sample(
        self,
        batch_size: int,
        strategy: str = "uniform",
        beta: Optional[float] = None,
        agent_distribution: Optional[Dict[Any, float]] = None,
    ):
        with self.lock:
            self._remove_stale_experiences()

            if batch_size <= 0:
                raise ValueError("batch_size must be greater than zero")
            if len(self.buffer) < batch_size:
                raise ValueError(f"Insufficient samples ({len(self.buffer)} available, requested {batch_size})")

            if beta is None:
                beta = self.default_beta

            strategy = strategy.lower().strip()
            if strategy == "prioritized":
                processed_batch, selected_indices, weights = self._prioritized_sample(batch_size=batch_size, beta=beta)
                self._check_fairness(processed_batch, strategy)
                return processed_batch, selected_indices, weights
            if strategy == "reward":
                processed_batch = self._reward_based_sample(batch_size)
            elif strategy == "agent_balanced":
                processed_batch = self._agent_balanced_sample(batch_size, agent_distribution)
            else:
                processed_batch = self._uniform_sample(batch_size)

            self._check_fairness(processed_batch, strategy)
            self._log_calibration(processed_batch)
            return processed_batch

    def _prioritized_sample(self, batch_size: int, beta: float):
        valid_entries: List[Tuple[float, int]] = [(-p, idx) for p, idx in self.priorities if idx < len(self.buffer)]
        if len(valid_entries) < batch_size:
            raise ValueError("Not enough valid priority entries for prioritized sampling")

        priority_values = np.array([entry[0] for entry in valid_entries], dtype=np.float64)
        value_sum = priority_values.sum()
        probabilities = (
            np.ones_like(priority_values) / len(priority_values)
            if value_sum <= 0
            else priority_values / value_sum
        )

        sampled_positions = np.random.choice(len(valid_entries), size=batch_size, replace=False, p=probabilities)
        selected_indices = np.array([valid_entries[pos][1] for pos in sampled_positions], dtype=np.int64)
        sampled_probabilities = probabilities[sampled_positions]

        experiences = [self.buffer[idx] for idx in selected_indices]

        weights = (len(self.buffer) * sampled_probabilities) ** (-float(beta))
        weights /= weights.max() if weights.size else 1.0
        return self._process_batch(experiences), selected_indices, weights.astype(np.float32)

    def _reward_based_sample(self, batch_size: int):
        rewards = np.array([exp[3] for exp in self.buffer], dtype=np.float64)
        shifted = rewards - rewards.min() + MIN_PRIORITY
        probs = shifted / shifted.sum() if shifted.sum() > 0 else np.ones_like(shifted) / len(shifted)
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)
        return self._process_batch([self.buffer[idx] for idx in indices])

    def _agent_balanced_sample(self, batch_size: int, distribution: Optional[Dict[Any, float]]):
        if not distribution:
            total = sum(self.agent_experience_map.values())
            if total == 0:
                return self._uniform_sample(batch_size)
            distribution = {agent_id: count / total for agent_id, count in self.agent_experience_map.items()}

        samples: List[Experience] = []
        for agent_id, proportion in distribution.items():
            requested = max(0, int(round(batch_size * float(proportion))))
            agent_samples = [exp for exp in self.buffer if exp[0] == agent_id]
            if not agent_samples:
                continue
            take = min(requested, len(agent_samples))
            samples.extend(random.sample(agent_samples, take))

        if len(samples) < batch_size:
            remaining = batch_size - len(samples)
            samples.extend(random.sample(list(self.buffer), remaining))
        elif len(samples) > batch_size:
            samples = samples[:batch_size]

        return self._process_batch(samples)

    def _uniform_sample(self, batch_size: int):
        indices = random.sample(range(len(self.buffer)), batch_size)
        return self._process_batch([self.buffer[idx] for idx in indices])

    def _process_batch(self, batch: Sequence[Experience]):
        agent_ids, states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(agent_ids, dtype=object),
            np.array(states, dtype=object),
            np.array(actions, dtype=object),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=object),
            np.array(dones, dtype=np.bool_),
        )

    def _check_fairness(self, batch, strategy: str) -> None:
        agent_ids = batch[0]
        if len(agent_ids) == 0:
            return

        unique, counts = np.unique(agent_ids, return_counts=True)
        selection_rates = {aid: count / len(agent_ids) for aid, count in zip(unique, counts)}
        self.fairness_stats["fair_batches_checked"] += 1

        violation, msg = FairnessMetrics.demographic_parity(
            sensitive_groups=list(selection_rates.keys()),
            positive_rates=selection_rates,
            threshold=0.1,
        )
        if violation:
            self.fairness_stats["demographic_parity_violations"] += 1
            self.fairness_stats["last_violation_message"] = f"[{strategy}] {msg}"
            logger.warning("Fairness Alert (%s): %s", strategy, msg)

    def _log_calibration(self, batch) -> None:
        rewards = batch[3]
        if rewards.size == 0:
            return

        probs = np.abs(rewards) + MIN_PRIORITY
        try:
            calibration = PerformanceMetrics.calibration_error(y_true=rewards, probs=probs)
            logger.info("Reward calibration error: %.6f", calibration)
        except Exception as exc:
            logger.warning("Failed to compute calibration error: %s", exc)

    def update_priorities(self, indices: Sequence[int], new_priorities: Sequence[float]) -> None:
        with self.lock:
            for idx, priority in zip(indices, new_priorities):
                if 0 <= idx < len(self.buffer):
                    heapq.heappush(self.priorities, (-self._resolve_priority(priority, self.buffer[idx][3]), idx))

    def apply_augmentation(self, batch, augment_fn: Callable):
        return augment_fn(batch)

    def get_agent_statistics(self) -> Dict[Any, int]:
        with self.lock:
            return dict(self.agent_experience_map)

    def get_reward_statistics(self) -> Dict[str, float]:
        with self.lock:
            size = len(self.buffer)
            return {
                **self.reward_stats,
                "mean": self.reward_stats["sum"] / size if size else 0.0,
            }

    def get_all(self):
        with self.lock:
            if not self.buffer:
                return [], [], [], [], [], []
            agent_ids, states, actions, rewards, next_states, dones = zip(*self.buffer)
            return list(agent_ids), list(states), list(actions), list(rewards), list(next_states), list(dones)

    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()
            self.timestamps.clear()
            self.priorities.clear()
            self.agent_experience_map.clear()
            self.agent_rewards.clear()
            self.reward_stats = {"sum": 0.0, "max": -np.inf, "min": np.inf}
            logger.info("Replay buffer cleared.")

    def save(self, filepath: str) -> None:
        with self.lock:
            meta = {
                "capacity": self.capacity,
                "prioritization_alpha": self.alpha,
                "staleness_threshold": self.staleness_threshold.total_seconds(),
                "reward_stats": self.reward_stats,
                "fairness_stats": self.fairness_stats,
                "metric_provenance": self.metric_provenance,
            }
            np.savez_compressed(
                filepath,
                buffer=np.array(self.buffer, dtype=object),
                timestamps=np.array(self.timestamps, dtype=object),
                priorities=np.array([-p[0] for p in self.priorities], dtype=np.float64),
                meta=meta,
            )

    def load(self, filepath: str) -> None:
        data = np.load(filepath, allow_pickle=True)
        meta = data["meta"].item()

        with self.lock:
            self.capacity = int(meta["capacity"])
            self.alpha = float(meta["prioritization_alpha"])
            self.staleness_threshold = timedelta(seconds=float(meta["staleness_threshold"]))
            self.reward_stats = meta.get("reward_stats", {"sum": 0.0, "max": -np.inf, "min": np.inf})
            self.fairness_stats = meta.get("fairness_stats", self.fairness_stats)
            self.metric_provenance = meta.get("metric_provenance", self.metric_provenance)

            self.buffer = deque(data["buffer"].tolist(), maxlen=self.capacity)
            self.timestamps = deque(data["timestamps"].tolist(), maxlen=self.capacity)

            self.priorities = []
            for idx, priority in enumerate(data["priorities"]):
                if idx < len(self.buffer):
                    heapq.heappush(self.priorities, (-float(priority), idx))

            self._rebuild_statistics()

    def generate_health_report(self) -> Dict[str, Any]:
        with self.lock:
            rewards = [exp[3] for exp in self.buffer]
            reward_mean = float(np.mean(rewards)) if rewards else 0.0
            reward_variance = float(np.var(rewards)) if rewards else 0.0
            return MetricSummarizer.create_model_card(
                metrics={
                    "fairness": self.fairness_stats,
                    "performance": {
                        "reward_mean": reward_mean,
                        "reward_variance": reward_variance,
                    },
                },
                references=self.metric_provenance,
            )


if __name__ == "__main__":
    print("\n=== Running Distributed Replay Buffer ===\n")
    printer.status("TEST", "Distributed Replay Buffer initialized", "info")

    try:
        config = load_global_config()
        printer.status(
            "CONFIG",
            f"Loaded config from {config.get('__config_path__', 'unknown')}",
            "success",
        )

        buffer = DistributedReplayBuffer()
        printer.status("INIT", "Buffer object created", "success")

        for index in range(128):
            agent_id = f"agent_{index % 4}"
            state = np.array([index, index + 1], dtype=np.float32)
            action = index % 3
            reward = float((index % 7) - 3)
            next_state = state + 0.5
            done = (index % 11 == 0)
            assert buffer.push(agent_id, state, action, reward, next_state, done)

        stats = buffer.stats()
        assert stats["total_experiences"] == 128
        printer.status("STATS", f"Buffer stats: {stats}", "success")

        uniform_batch = buffer.sample(batch_size=16, strategy="uniform")
        assert uniform_batch[0].shape[0] == 16
        printer.status("SAMPLE", "Uniform sampling passed", "success")

        prioritized_batch, prioritized_indices, importance_weights = buffer.sample(
            batch_size=16,
            strategy="prioritized",
            beta=0.6,
        )
        assert prioritized_batch[0].shape[0] == 16
        assert len(prioritized_indices) == 16
        assert len(importance_weights) == 16
        printer.status("SAMPLE", "Prioritized sampling passed", "success")

        reward_batch = buffer.sample(batch_size=16, strategy="reward")
        assert reward_batch[0].shape[0] == 16
        printer.status("SAMPLE", "Reward sampling passed", "success")

        agent_balanced_batch = buffer.sample(batch_size=16, strategy="agent_balanced")
        assert agent_balanced_batch[0].shape[0] == 16
        printer.status("SAMPLE", "Agent-balanced sampling passed", "success")

        health_report = buffer.generate_health_report()
        assert isinstance(health_report, dict)
        printer.status("HEALTH", "Health report generated", "success")

        buffer.clear()
        assert len(buffer) == 0
        printer.status("CLEAR", "Buffer cleared", "success")

        print("\n=== Test ran successfully ===\n")
    except Exception as exc:
        logger.exception("Distributed replay buffer self-test failed")
        printer.status("TEST", f"Distributed replay buffer self-test failed: {exc}", "error")
        raise