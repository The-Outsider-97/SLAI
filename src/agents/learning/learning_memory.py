"""Production-ready prioritized replay memory for the SLAI learning subsystem.

This module keeps the original public surface used across the learning stack:
- ``Transition`` namedtuple for RL-style transitions.
- ``SumTree`` for proportional prioritized replay.
- ``LearningMemory`` with add/sample/update/tag/checkpoint APIs.

The implementation is hardened for production use while retaining the existing
configuration flow through ``learning_config.yaml`` and ``get_config_section``.
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import torch # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]

from pathlib import Path
from threading import RLock
from datetime import datetime
from collections import namedtuple, defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.learning_error import *
from .utils.learning_calculations import *
from .utils.learning_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Learning Memory")
printer = PrettyPrinter()

# Keep compatibility with checkpoints produced by older numpy/torch stacks.
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class SumTree:
    """Priority sum tree for proportional replay sampling.

    Leaves hold priorities and the parallel ``data`` array holds experiences.
    ``add`` overwrites circularly; ``sample`` maps a cumulative priority value to
    a leaf in O(log n).
    """

    def __init__(self, capacity: int):
        validate_positive(capacity, "sum_tree.capacity", strict=True)
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = np.empty(self.capacity, dtype=object)
        self.data[:] = None
        self.size = 0
        self.write_ptr = 0
        self.max_priority = 1.0

    def _propagate(self, idx: int, delta: float) -> None:
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += float(delta)

    def _retrieve(self, idx: int, value: float) -> int:
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left] or right >= len(self.tree):
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx

    def total(self) -> float:
        total = float(self.tree[0])
        if not np.isfinite(total):
            raise NumericalInstabilityError("SumTree total priority is non-finite", metric_name="priority_total", observed_value=total)
        return total

    def add(self, priority: float, data: Any) -> int:
        validate_finite(priority, "priority")
        priority = max(float(priority), 0.0)
        idx = self.write_ptr + self.capacity - 1
        data_idx = self.write_ptr
        self.data[data_idx] = data
        delta = priority - float(self.tree[idx])
        self.tree[idx] = priority
        self._propagate(idx, delta)
        self.max_priority = max(self.max_priority, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_idx

    def update(self, data_idx: int, priority: float) -> None:
        if not 0 <= int(data_idx) < self.capacity:
            raise ReplayBufferError("SumTree update index out of range", context={"index": data_idx, "capacity": self.capacity})
        validate_finite(priority, "priority")
        tree_idx = int(data_idx) + self.capacity - 1
        priority = max(float(priority), 0.0)
        delta = priority - float(self.tree[tree_idx])
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, value: float) -> Tuple[int, float, Any]:
        total = self.total()
        if self.size <= 0 or total <= 0.0:
            raise BufferUnderflowError(requested=1, available=self.size)
        value = clamp(float(value), 0.0, max(np.nextafter(total, 0.0), 0.0))
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return data_idx, float(self.tree[tree_idx]), self.data[data_idx]

    def non_empty_indices(self) -> List[int]:
        return [idx for idx in range(self.capacity) if self.data[idx] is not None]

    def __len__(self) -> int:
        return int(self.size)


class LearningMemory:
    """Thread-safe prioritized replay memory with tags, metadata, and checkpoints."""

    CHECKPOINT_VERSION = 2

    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section("learning_memory") or {}

        self.memory_config.setdefault("max_size", 10000)
        self.memory_config.setdefault("checkpoint_dir", "src/agents/learning/checkpoints/memory")
        self.memory_config.setdefault("checkpoint_freq", 1000)
        self.memory_config.setdefault("auto_save", True)
        self.memory_config.setdefault("alpha", 0.6)
        self.memory_config.setdefault("beta", 0.4)
        self.memory_config.setdefault("beta_end", 1.0)
        self.memory_config.setdefault("epsilon", 1e-5)
        self.memory_config.setdefault("beta_annealing_steps", 100000)
        self.memory_config.setdefault("allow_empty_sample", True)
        self.memory_config.setdefault("allow_partial_batch", True)
        self.memory_config.setdefault("uniform_without_replacement", True)
        self.memory_config.setdefault("max_recent_rewards", 1000)
        self.memory_config.setdefault("safe_checkpoint_load", True)
        self.memory_config.setdefault("checkpoint_pickle_protocol", 2)

        self.capacity = coerce_int(self.memory_config.get("max_size"), default=10000, minimum=1)
        self.checkpoint_freq = coerce_int(self.memory_config.get("checkpoint_freq"), default=1000, minimum=1)
        self.alpha = coerce_float(self.memory_config.get("alpha"), default=0.6, minimum=0.0, maximum=1.0)
        self.beta_start = coerce_float(self.memory_config.get("beta"), default=0.4, minimum=0.0, maximum=1.0)
        self.beta_end = coerce_float(self.memory_config.get("beta_end"), default=1.0, minimum=0.0, maximum=1.0)
        self.priority_epsilon = coerce_float(self.memory_config.get("epsilon"), default=1e-5, minimum=0.0)
        self.beta_annealing_steps = coerce_int(self.memory_config.get("beta_annealing_steps"), default=100000, minimum=1)
        self.auto_save = coerce_bool(self.memory_config.get("auto_save"), default=True)
        self.allow_empty_sample = coerce_bool(self.memory_config.get("allow_empty_sample"), default=True)
        self.allow_partial_batch = coerce_bool(self.memory_config.get("allow_partial_batch"), default=True)
        self.uniform_without_replacement = coerce_bool(self.memory_config.get("uniform_without_replacement"), default=True)
        self.safe_checkpoint_load = coerce_bool(self.memory_config.get("safe_checkpoint_load"), default=True)

        self.tree = SumTree(self.capacity)
        self.tag_index: Dict[str, set[int]] = defaultdict(set)
        self.index_to_tags: Dict[int, set[str]] = defaultdict(set)
        self.lock = RLock()
        self.access_counter = 0
        self.sample_counter = 0
        self.max_priority = 1.0
        self.key_value_store: Dict[str, Any] = {}
        self.reward_history: Deque[float] = deque(maxlen=coerce_int(self.memory_config.get("max_recent_rewards"), default=1000, minimum=1))
        self.priority_stats = RunningStats()
        self.sample_weight_stats = RunningStats()
        self.calculations = LearningCalculations()
        logger.info("LearningMemory initialized with capacity=%s alpha=%.3f beta=%.3f", self.capacity, self.alpha, self.beta_start)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _current_beta(self) -> float:
        progress = clamp(self.access_counter / max(1, self.beta_annealing_steps), 0.0, 1.0)
        return float(self.beta_start + (self.beta_end - self.beta_start) * progress)

    def _priority_from_raw(self, raw_priority: Optional[float]) -> float:
        if raw_priority is None:
            raw_priority = self.max_priority
        validate_finite(raw_priority, "raw_priority")
        priority = (abs(float(raw_priority)) + self.priority_epsilon) ** self.alpha
        validate_finite(priority, "priority")
        return max(float(priority), self.priority_epsilon)

    @staticmethod
    def _experience_reward(experience: Any) -> Optional[float]:
        if hasattr(experience, "reward"):
            try:
                return float(experience.reward)
            except (TypeError, ValueError):
                return None
        if isinstance(experience, (tuple, list)) and len(experience) >= 3:
            try:
                return float(experience[2])
            except (TypeError, ValueError):
                return None
        return None

    def _remove_tags_for_index(self, data_idx: int) -> None:
        old_tags = list(self.index_to_tags.get(int(data_idx), set()))
        for old_tag in old_tags:
            self.tag_index[old_tag].discard(int(data_idx))
            if not self.tag_index[old_tag]:
                del self.tag_index[old_tag]
        self.index_to_tags.pop(int(data_idx), None)

    def _attach_tags(self, data_idx: int, tags: Optional[Union[str, Sequence[str]]]) -> None:
        if tags is None:
            return
        tag_values = [tags] if isinstance(tags, str) else list(tags)
        for tag in tag_values:
            clean = str(tag).strip()
            if not clean:
                continue
            self.tag_index[clean].add(int(data_idx))
            self.index_to_tags[int(data_idx)].add(clean)

    def _require_batch_size(self, batch_size: int, available: int) -> int:
        validate_positive(batch_size, "batch_size", strict=True)
        requested = int(batch_size)
        if available <= 0:
            if self.allow_empty_sample:
                return 0
            raise BufferUnderflowError(requested=requested, available=available)
        if requested > available and not self.allow_partial_batch:
            raise BufferUnderflowError(requested=requested, available=available)
        return min(requested, available) if self.allow_partial_batch else requested

    def _normalise_indices_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> List[Tuple[int, float]]:
        if len(indices) != len(priorities):
            raise InvalidConfigError("indices and priorities must have the same length", context={"indices": len(indices), "priorities": len(priorities)})
        pairs: List[Tuple[int, float]] = []
        for idx, raw_priority in zip(indices, priorities):
            if not 0 <= int(idx) < self.tree.capacity:
                raise ReplayBufferError("Priority index out of range", context={"index": idx, "capacity": self.tree.capacity})
            pairs.append((int(idx), self._priority_from_raw(float(raw_priority))))
        return pairs

    # ------------------------------------------------------------------
    # Core experience management
    # ------------------------------------------------------------------
    def size(self) -> int:
        with self.lock:
            return len(self.tree)

    def __len__(self) -> int:
        return self.size()

    def add(self, experience: Any, priority: Optional[float] = None, tag: Optional[Union[str, Sequence[str]]] = None) -> int:
        """Add one experience and return its circular-buffer data index."""
        with self.lock:
            write_idx = int(self.tree.write_ptr)
            if self.tree.size == self.tree.capacity:
                self._remove_tags_for_index(write_idx)
            transformed_priority = self._priority_from_raw(priority)
            data_idx = self.tree.add(transformed_priority, experience)
            self._attach_tags(data_idx, tag)
            self.access_counter += 1
            self.max_priority = max(self.max_priority, transformed_priority)
            self.priority_stats.update(transformed_priority)
            reward = self._experience_reward(experience)
            if reward is not None and np.isfinite(reward):
                self.reward_history.append(reward)
                self.calculations.update_performance(reward)
            if self.auto_save and self.access_counter % self.checkpoint_freq == 0:
                self.save_checkpoint()
            return data_idx

    def add_batch(self, experiences: List[Any], tag: Optional[Union[str, Sequence[str]]] = None) -> List[int]:
        validate_type(experiences, "experiences", list)
        return [self.add(exp, tag=tag) for exp in experiences]

    def sample_proportional(self, batch_size: int) -> Tuple[List[Any], List[int], List[float]]:
        """Sample by PER priority and return (samples, indices, IS weights)."""
        with self.lock:
            available = len(self.tree)
            actual_batch = self._require_batch_size(batch_size, available)
            if actual_batch == 0:
                return [], [], []
            total_priority = self.tree.total()
            if total_priority <= 0.0:
                return self._uniform_sample_with_indices(actual_batch)
            beta = self._current_beta()
            segment = total_priority / actual_batch
            samples: List[Any] = []
            indices: List[int] = []
            weights: List[float] = []
            max_weight = 0.0
            for i in range(actual_batch):
                low = segment * i
                high = segment * (i + 1)
                value = random.uniform(low, high)
                idx, priority, data = self.tree.sample(value)
                if data is None:
                    continue
                prob = max(float(priority) / total_priority, 1e-12)
                weight = (available * prob) ** (-beta)
                max_weight = max(max_weight, weight)
                samples.append(data)
                indices.append(idx)
                weights.append(weight)
            if max_weight > 0.0:
                weights = [float(w / max_weight) for w in weights]
            for weight in weights:
                self.sample_weight_stats.update(weight)
            self.sample_counter += len(samples)
            return samples, indices, weights

    def _uniform_sample_with_indices(self, batch_size: int) -> Tuple[List[Any], List[int], List[float]]:
        indices = self.tree.non_empty_indices()
        if not indices:
            return [], [], []
        k = min(int(batch_size), len(indices)) if self.allow_partial_batch else int(batch_size)
        if self.uniform_without_replacement and k <= len(indices):
            chosen = random.sample(indices, k)
        else:
            chosen = [random.choice(indices) for _ in range(k)]
        self.sample_counter += len(chosen)
        return [self.tree.data[i] for i in chosen], chosen, [1.0 for _ in chosen]

    def sample(self, batch_size: int) -> List[Any]:
        with self.lock:
            samples, _, _ = self._uniform_sample_with_indices(self._require_batch_size(batch_size, len(self.tree)))
            return samples

    def sample_with_indices(self, batch_size: int, prioritized: bool = True) -> Tuple[List[Any], List[int], List[float]]:
        return self.sample_proportional(batch_size) if prioritized else self._uniform_sample_with_indices(batch_size)

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        with self.lock:
            for idx, priority in self._normalise_indices_priorities(indices, priorities):
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)
                self.priority_stats.update(priority)

    def update_priorities_from_model(self, indices: List[int], model: torch.nn.Module,
                                     loss_fn: Callable, gamma: float = 0.99,
                                     device: torch.device = torch.device("cpu")) -> None:
        """Compute one-step TD errors for stored ``Transition`` objects and update PER priorities."""
        validate_probability(gamma, "gamma")
        with self.lock:
            transitions = [self.tree.data[int(idx)] for idx in indices if 0 <= int(idx) < self.tree.capacity and self.tree.data[int(idx)] is not None]
        if not transitions:
            return
        for transition in transitions:
            if not all(hasattr(transition, attr) for attr in Transition._fields):
                raise ReplayBufferError("update_priorities_from_model expects Transition-like experiences")
        states = torch.stack([torch.as_tensor(t.state, dtype=torch.float32) for t in transitions]).to(device)
        actions = torch.as_tensor([t.action for t in transitions], dtype=torch.long, device=device)
        rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
        next_states = torch.stack([torch.as_tensor(t.next_state, dtype=torch.float32) for t in transitions]).to(device)
        dones = torch.as_tensor([t.done for t in transitions], dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = model(next_states).max(1)[0]
            target = rewards + float(gamma) * next_q_values * (1.0 - dones)
            td_errors = (target - q_values).abs().detach().cpu().numpy().tolist()
        self.update_priorities(indices[: len(td_errors)], td_errors)

    # ------------------------------------------------------------------
    # Tag management
    # ------------------------------------------------------------------
    def get_by_tag(self, tag: str) -> List[Any]:
        with self.lock:
            return [self.tree.data[idx] for idx in sorted(self.tag_index.get(str(tag), set())) if self.tree.data[idx] is not None]

    def delete_tag(self, tag: str) -> None:
        with self.lock:
            tag = str(tag)
            if tag not in self.tag_index:
                return
            for idx in list(self.tag_index[tag]):
                self.index_to_tags[idx].discard(tag)
                if not self.index_to_tags[idx]:
                    del self.index_to_tags[idx]
            del self.tag_index[tag]

    def add_tag(self, index: int, tag: str) -> None:
        with self.lock:
            if not 0 <= int(index) < self.tree.capacity or self.tree.data[int(index)] is None:
                raise ReplayBufferError("Cannot tag missing experience", context={"index": index})
            self._attach_tags(int(index), str(tag))

    def remove(self, index: int) -> None:
        """Remove one experience and clear its priority/tag state."""
        with self.lock:
            if not 0 <= int(index) < self.tree.capacity:
                raise ReplayBufferError("Remove index out of range", context={"index": index})
            self.tree.data[int(index)] = None
            self.tree.update(int(index), 0.0)
            self._remove_tags_for_index(int(index))
            self.tree.size = len(self.tree.non_empty_indices())

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_recent_states(self, num_states: int = 100) -> List[Any]:
        validate_positive(num_states, "num_states", strict=True)
        with self.lock:
            recent: List[Any] = []
            start = (self.tree.write_ptr - 1) % self.tree.capacity
            for i in range(self.tree.capacity):
                idx = (start - i) % self.tree.capacity
                exp = self.tree.data[idx]
                if exp is not None and hasattr(exp, "state"):
                    recent.append(exp.state)
                    if len(recent) >= int(num_states):
                        break
            return recent

    def get(self, key: Optional[Union[int, str]] = None, default=None):
        with self.lock:
            if key is None:
                return [self.tree.data[i] for i in self.tree.non_empty_indices()]
            if isinstance(key, int):
                return self.tree.data[key] if 0 <= key < self.tree.capacity and self.tree.data[key] is not None else default
            return self.key_value_store.get(str(key), default)

    def set(self, key: Union[int, str], value: Any) -> None:
        with self.lock:
            if isinstance(key, int):
                if not 0 <= key < self.tree.capacity:
                    raise ReplayBufferError("Set index out of range", context={"index": key})
                self.tree.data[key] = value
            else:
                self.key_value_store[str(key)] = value

    def clear(self) -> None:
        with self.lock:
            self.tree = SumTree(self.capacity)
            self.tag_index.clear()
            self.index_to_tags.clear()
            self.access_counter = 0
            self.sample_counter = 0
            self.max_priority = 1.0
            self.key_value_store.clear()
            self.reward_history.clear()
            self.priority_stats = RunningStats()
            self.sample_weight_stats = RunningStats()

    def metrics(self) -> Dict[str, Any]:
        with self.lock:
            priorities = self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + self.tree.capacity]
            active_priorities = [float(priorities[i]) for i in self.tree.non_empty_indices()]
            priority_summary = self.calculations.summarize_rewards(active_priorities)
            reward_summary = self.calculations.summarize_rewards(list(self.reward_history))
            uniform = np.full(max(len(active_priorities), 1), 1.0 / max(len(active_priorities), 1), dtype=np.float64)
            if active_priorities:
                probs = np.asarray(normalize_probabilities(active_priorities), dtype=np.float64)
                drift = self.calculations.calculate_distribution_drift(probs, uniform)
                priority_drift = to_json_safe(drift)
            else:
                priority_drift = {}
            return {
                "size": len(self.tree),
                "capacity": self.tree.capacity,
                "access_counter": self.access_counter,
                "sample_counter": self.sample_counter,
                "tags": sorted(self.tag_index.keys()),
                "total_priority": round_float(self.tree.total()),
                "avg_priority": round_float(safe_divide(self.tree.total(), len(self.tree), default=0.0)),
                "max_priority": round_float(self.max_priority),
                "beta_current": round_float(self._current_beta()),
                "priority_summary": priority_summary,
                "reward_summary": reward_summary,
                "priority_drift": priority_drift,
                "priority_stats": to_json_safe(self.priority_stats.snapshot()),
                "sample_weight_stats": to_json_safe(self.sample_weight_stats.snapshot()),
            }

    def diagnostics(self) -> Dict[str, Any]:
        return self.metrics()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "tree_data": self.tree.data,
            "tree_structure": self.tree.tree,
            "tree_write_ptr": self.tree.write_ptr,
            "tree_size": self.tree.size,
            "tree_capacity": self.tree.capacity,
            "tree_max_priority": self.tree.max_priority,
            "tag_index": {k: sorted(v) for k, v in self.tag_index.items()},
            "index_to_tags": {int(k): sorted(v) for k, v in self.index_to_tags.items()},
            "access_counter": self.access_counter,
            "sample_counter": self.sample_counter,
            "max_priority": self.max_priority,
            "key_value_store": self.key_value_store,
            "reward_history": list(self.reward_history),
            "config": dict(self.memory_config),
            "metrics": self.metrics(),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        validate_required_keys(checkpoint, ["tree_data", "tree_structure", "tree_write_ptr", "tree_size"], name="learning_memory_checkpoint")
        capacity = int(checkpoint.get("tree_capacity", len(checkpoint["tree_data"])))
        if capacity != self.tree.capacity:
            self.capacity = capacity
            self.tree = SumTree(capacity)
        self.tree.data = np.asarray(checkpoint["tree_data"], dtype=object)
        self.tree.tree = np.asarray(checkpoint["tree_structure"], dtype=np.float64)
        self.tree.write_ptr = int(checkpoint["tree_write_ptr"])
        self.tree.size = int(checkpoint["tree_size"])
        self.tree.max_priority = float(checkpoint.get("tree_max_priority", self.tree.max_priority))
        self.tag_index = defaultdict(set, {str(k): set(int(i) for i in v) for k, v in checkpoint.get("tag_index", {}).items()})
        self.index_to_tags = defaultdict(set, {int(k): set(str(tag) for tag in v) for k, v in checkpoint.get("index_to_tags", {}).items()})
        self.access_counter = int(checkpoint.get("access_counter", 0))
        self.sample_counter = int(checkpoint.get("sample_counter", 0))
        self.max_priority = float(checkpoint.get("max_priority", self.tree.max_priority))
        self.key_value_store = dict(checkpoint.get("key_value_store", {}))
        self.reward_history = deque([float(v) for v in checkpoint.get("reward_history", [])], maxlen=self.reward_history.maxlen)
        self.priority_stats = RunningStats()
        for value in self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + self.tree.capacity]:
            if value > 0:
                self.priority_stats.update(float(value))

    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> str:
        with self.lock:
            checkpoint_dir = Path(self.memory_config.get("checkpoint_dir", "src/agents/learning/checkpoints/memory"))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = checkpoint_dir / f"memory_{timestamp}.pt"
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_fd, temp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_memory_", suffix=".pt")
            try:
                with os.fdopen(temp_fd, "wb") as handle:
                    torch.save(self.snapshot(), handle, pickle_protocol=coerce_int(self.memory_config.get("checkpoint_pickle_protocol"), default=2, minimum=2))
                shutil.move(temp_path, target)
                logger.info("Memory checkpoint saved to %s", target)
                return str(target)
            except Exception as exc:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise CheckpointError(str(target), operation="save", cause=exc) from exc

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        source = Path(path)
        if not source.exists():
            raise CheckpointError(str(source), operation="load", message="Learning memory checkpoint does not exist")
        with self.lock:
            try:
                if self.safe_checkpoint_load:
                    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
                else:
                    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            except Exception:
                checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, Mapping):
                raise CheckpointError(str(source), operation="load", message="Learning memory checkpoint is not a mapping")
            self.restore(checkpoint)
            logger.info("Loaded memory checkpoint from %s", source)


if __name__ == "__main__":
    print("\n=== Running Learning Memory ===\n")
    printer.status("TEST", "Learning Memory initialized", "info")
    torch.manual_seed(7); random.seed(7); np.random.seed(7)
    memory = LearningMemory()
    memory.memory_config["auto_save"] = False
    memory.auto_save = False
    for i in range(24):
        exp = Transition(torch.randn(4), i % 3, float(i % 5 - 2), torch.randn(4), i % 7 == 0)
        memory.add(exp, priority=abs(exp.reward) + 0.1, tag=["main", f"a{exp.action}"])
    batch, idx, w = memory.sample_proportional(8)
    assert len(batch) == len(idx) == len(w) == 8
    memory.update_priorities(idx, [0.5 + i for i in range(len(idx))])
    assert len(memory.sample(5)) == 5
    assert len(memory.get_by_tag("main")) == memory.size()
    assert len(memory.get_recent_states(3)) == 3
    m = memory.metrics()
    assert m["size"] == 24 and m["total_priority"] > 0
    ckpt = Path("learning_memory_test.pt")
    memory.save_checkpoint(ckpt)
    restored = LearningMemory(); restored.load_checkpoint(ckpt)
    assert restored.size() == memory.size()
    assert len(restored.get_by_tag("main")) == memory.size()
    ckpt.unlink(missing_ok=True)
    memory.delete_tag("main")
    assert memory.get_by_tag("main") == []
    printer.status("TEST", "Learning Memory PER, tags, metrics, and checkpoints verified", "success")
    print("\n=== Test ran successfully ===\n")
