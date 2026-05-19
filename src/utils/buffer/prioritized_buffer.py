from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np  # type: ignore

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from .buffer_persistence import BufferCheckpointIO, build_checkpoint_io
from .buffer_telemetry import BufferTelemetry
from .buffer_validation import *
from .segment_tree import MinSegmentTree, SumSegmentTree
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Prioritized Replay Buffer")
printer = PrettyPrinter()

PER_SCHEMA_VERSION = "prioritized_replay_buffer.v1"
PER_COMPONENT_NAME = "prioritized_replay_buffer"


@dataclass(frozen=True)
class PrioritizedSampleBatch:
    """Structured sample result for single-node Prioritized Experience Replay."""

    agent_ids: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    probabilities: np.ndarray
    priorities: np.ndarray
    beta: float

    def as_tuple(self) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """Compatibility bridge: ((batch fields), indices, weights)."""
        return (
            (self.agent_ids, self.states, self.actions, self.rewards, self.next_states, self.dones),
            self.indices,
            self.weights,
        )

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "agent_ids": self.agent_ids,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "indices": self.indices,
            "weights": self.weights,
            "probabilities": self.probabilities,
            "priorities": self.priorities,
        }


@dataclass
class PriorityUpdateReport:
    """Diagnostics for an explicit TD-error priority update pass."""

    requested: int = 0
    updated: int = 0
    rejected: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.rejected == 0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.requested if self.requested else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested,
            "updated": self.updated,
            "rejected": self.rejected,
            "rejection_rate": self.rejection_rate,
            "errors": list(self.errors),
        }


def _error_cls(name: str, fallback: Any) -> Any:
    return globals().get(name, fallback)


def _priority_update_error(reason: str, indices: Optional[Sequence[int]] = None) -> BaseException:
    cls = _error_cls("PriorityUpdateError", PrioritySamplingError)
    if cls is PrioritySamplingError:
        return cls(f"Priority update failed: {reason}")
    return cls(reason, indices=indices)


def _priority_mass_error(total_mass: Any, reason: str) -> BaseException:
    cls = _error_cls("PriorityMassError", PrioritySamplingError)
    if cls is PrioritySamplingError:
        return cls(f"Invalid priority mass {total_mass}: {reason}")
    return cls(total_mass, reason=reason)


def _lock_timeout_error(operation: str, timeout_seconds: float) -> BaseException:
    cls = _error_cls("BufferLockTimeoutError", None)
    if cls is not None:
        return cls(operation=operation, timeout_seconds=timeout_seconds)
    return BufferOperationError(f"Timed out acquiring lock for {operation}", [])


def _mutation_error(operation: str, reason: str) -> BaseException:
    cls = _error_cls("BufferMutationError", None)
    if cls is not None:
        return cls(operation=operation, reason=reason)
    return BufferOperationError(f"Buffer mutation failed during {operation}: {reason}", [])


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValueError(name, value, "positive integer") from exc
    if result <= 0:
        raise ConfigValueError(name, value, "positive integer")
    return result


def _finite_float(value: Any, name: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValueError(name, value, "finite float") from exc
    if not np.isfinite(result):
        raise ConfigValueError(name, value, "finite float")
    if minimum is not None and result < minimum:
        raise ConfigValueError(name, value, f">= {minimum}")
    if maximum is not None and result > maximum:
        raise ConfigValueError(name, value, f"<= {maximum}")
    return result


def _telemetry_increment(telemetry: Any, name: str, amount: float = 1.0) -> None:
    if telemetry is not None and hasattr(telemetry, "increment"):
        telemetry.increment(name, amount)


def _telemetry_observe(telemetry: Any, name: str, value: float) -> None:
    if telemetry is not None and hasattr(telemetry, "observe"):
        telemetry.observe(name, value)


def _tree_update(tree: Any, idx: int, value: float) -> None:
    if hasattr(tree, "update"):
        tree.update(idx, value)
        return
    tree.set(idx, value)


def _tree_prefix_index(tree: Any, mass: float) -> int:
    if hasattr(tree, "prefix_sum_index"):
        return int(tree.prefix_sum_index(mass))
    return int(tree.find_prefixsum_idx(mass))


def _tree_total(tree: Any) -> float:
    if hasattr(tree, "total_mass"):
        return float(tree.total_mass)
    return float(tree.sum())


def _tree_min(tree: Any, size: int) -> float:
    if hasattr(tree, "min"):
        try:
            return float(tree.min(0, size))
        except TypeError:
            return float(tree.min())
    return float("inf")


class PrioritizedReplayBuffer:
    """
    Single-node Prioritized Experience Replay buffer.

    This module owns the non-distributed PER specialization:
    - O(log N) priority updates through Sum/Min segment trees.
    - Stratified proportional sampling.
    - Explicit TD-error based priority update loop.
    - Importance-sampling weights for bias correction.
    - Shared validation, telemetry, error handling, and persistence.
    """

    def __init__(
        self,
        user_config: Optional[Mapping[str, Any]] = None,
        validator: Optional[TransitionValidator] = None,
        telemetry: Optional[BufferTelemetry] = None,
        checkpoint_io: Optional[BufferCheckpointIO] = None,
    ) -> None:
        self.config = load_global_config()
        self.prioritized_cfg = dict(get_config_section("prioritized_replay") or {})
        if user_config:
            overlay = user_config.get("prioritized_replay", {}) if isinstance(user_config, Mapping) else {}
            if isinstance(overlay, Mapping):
                self.prioritized_cfg.update(dict(overlay))

        self.capacity = _positive_int(self.prioritized_cfg.get("capacity", 100_000), "prioritized_replay.capacity")
        self.alpha = _finite_float(self.prioritized_cfg.get("alpha", 0.6), "prioritized_replay.alpha", minimum=0.0)
        self.default_beta = _finite_float(
            self.prioritized_cfg.get("beta", 0.4), "prioritized_replay.beta", minimum=0.0, maximum=1.0
        )
        self.beta_increment_per_sample = _finite_float(
            self.prioritized_cfg.get("beta_increment_per_sample", 0.0),
            "prioritized_replay.beta_increment_per_sample",
            minimum=0.0,
        )
        self.epsilon = _finite_float(self.prioritized_cfg.get("epsilon", 1e-6), "prioritized_replay.epsilon", minimum=0.0)
        self.initial_priority = _finite_float(
            self.prioritized_cfg.get("initial_priority", 1.0),
            "prioritized_replay.initial_priority",
            minimum=0.0,
        )
        self.priority_clip_min = _finite_float(
            self.prioritized_cfg.get("priority_clip_min", self.epsilon),
            "prioritized_replay.priority_clip_min",
            minimum=0.0,
        )
        clip_max_raw = self.prioritized_cfg.get("priority_clip_max")
        self.priority_clip_max = (
            None
            if clip_max_raw is None
            else _finite_float(clip_max_raw, "prioritized_replay.priority_clip_max", minimum=self.priority_clip_min)
        )
        self.sample_with_replacement = bool(self.prioritized_cfg.get("sample_with_replacement", True))
        self.return_dict = bool(self.prioritized_cfg.get("return_dict", True))
        self.lock_timeout_seconds = _finite_float(
            self.prioritized_cfg.get("lock_timeout_seconds", 5.0),
            "prioritized_replay.lock_timeout_seconds",
            minimum=0.0,
        )
        self.persistence_enabled = bool(self.prioritized_cfg.get("persistence_enabled", True))
        self.checkpoint_schema_version = str(
            self.prioritized_cfg.get("checkpoint_schema_version", PER_SCHEMA_VERSION)
        ).strip() or PER_SCHEMA_VERSION
        self.checkpoint_component_name = str(
            self.prioritized_cfg.get("checkpoint_component_name", PER_COMPONENT_NAME)
        ).strip() or PER_COMPONENT_NAME

        seed = self.prioritized_cfg.get("seed")
        self._rng = random.Random(seed)
        self._lock = RLock()
        self.validator = validator or TransitionValidator()
        self.telemetry = telemetry or BufferTelemetry(component_name=PER_COMPONENT_NAME)
        self.checkpoint_io = checkpoint_io or build_checkpoint_io(user_config=user_config, telemetry=self.telemetry)

        self.buffer: List[Optional[Transition]] = [None] * self.capacity
        self.metadata: List[Dict[str, Any]] = [{} for _ in range(self.capacity)]
        self.raw_priorities = np.zeros(self.capacity, dtype=np.float64)
        self.scaled_priorities = np.zeros(self.capacity, dtype=np.float64)
        self.td_errors = np.zeros(self.capacity, dtype=np.float64)

        self.sum_tree = SumSegmentTree(self.capacity)
        self.min_tree = MinSegmentTree(self.capacity)

        self.position = 0
        self.size = 0
        self.max_raw_priority = max(self.initial_priority, self.epsilon)
        self.total_pushed = 0
        self.total_sampled = 0
        self.total_priority_updates = 0
        self.total_overwrites = 0
        self.total_rejections = 0

    def __len__(self) -> int:
        with self._locked("len"):
            return self.size

    @contextmanager
    def _locked(self, operation: str) -> Iterator[None]:
        start = time.perf_counter()
        acquired = self._lock.acquire(timeout=self.lock_timeout_seconds) if self.lock_timeout_seconds > 0 else self._lock.acquire(blocking=False)
        waited = time.perf_counter() - start
        if hasattr(self.telemetry, "record_lock_contention"):
            self.telemetry.record_lock_contention(operation, waited, acquired=acquired)
        else:
            _telemetry_observe(self.telemetry, "lock_wait_seconds", waited)
            if waited > 0 or not acquired:
                _telemetry_increment(self.telemetry, "lock_contention_count")
        if not acquired:
            raise _lock_timeout_error(operation, self.lock_timeout_seconds)
        try:
            yield
        finally:
            self._lock.release()

    def _scaled_priority(self, priority_or_td_error: Any, *, from_td_error: bool = False) -> Tuple[float, float]:
        value = float(priority_or_td_error)
        if not np.isfinite(value):
            raise PrioritySamplingError("priority/td_error must be finite")
        raw = abs(value) + self.epsilon if from_td_error else value
        raw = max(float(raw), self.priority_clip_min)
        if self.priority_clip_max is not None:
            raw = min(raw, self.priority_clip_max)
        scaled = raw ** self.alpha if self.alpha > 0 else 1.0
        if scaled < 0 or not np.isfinite(scaled):
            raise PrioritySamplingError("scaled priority must be finite and non-negative")
        return raw, float(scaled)

    def _new_item_priority(self, priority: Optional[float], td_error: Optional[float]) -> Tuple[float, float]:
        if priority is not None:
            return self._scaled_priority(priority, from_td_error=False)
        if td_error is not None:
            return self._scaled_priority(td_error, from_td_error=True)
        return self._scaled_priority(max(self.max_raw_priority, self.initial_priority), from_td_error=False)

    def _set_slot(
        self,
        index: int,
        transition: Transition,
        *,
        raw_priority: float,
        scaled_priority: float,
        td_error: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if index < 0 or index >= self.capacity:
            raise IndexOutOfBoundsError(index, self.capacity, "push")
        self.buffer[index] = transition
        self.metadata[index] = dict(metadata or {})
        self.raw_priorities[index] = raw_priority
        self.scaled_priorities[index] = scaled_priority
        self.td_errors[index] = float(td_error)
        _tree_update(self.sum_tree, index, scaled_priority)
        _tree_update(self.min_tree, index, scaled_priority)
        self.max_raw_priority = max(self.max_raw_priority, raw_priority)

    def push(
        self,
        agent_id: Any,
        state: Any = None,
        action: Any = None,
        reward: Any = None,
        next_state: Any = None,
        done: Any = None,
        *,
        priority: Optional[float] = None,
        td_error: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        started = time.perf_counter()
        try:
            transition = (
                self.validator.sanitize_transition(agent_id)
                if state is None and isinstance(agent_id, (tuple, list, dict))
                else self.validator.sanitize_transition((agent_id, state, action, reward, next_state, done))
            )
            raw, scaled = self._new_item_priority(priority=priority, td_error=td_error)
            with self._locked("push"):
                index = self.position
                if self.size == self.capacity and self.buffer[index] is not None:
                    self.total_overwrites += 1
                self._set_slot(
                    index,
                    transition,
                    raw_priority=raw,
                    scaled_priority=scaled,
                    td_error=0.0 if td_error is None else float(td_error),
                    metadata=metadata,
                )
                self.position = (self.position + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)
                self.total_pushed += 1
                _telemetry_increment(self.telemetry, "push_count")
                return index
        except BufferError:
            self.total_rejections += 1
            if hasattr(self.telemetry, "record_rejection"):
                self.telemetry.record_rejection("push", "buffer_error")
            else:
                _telemetry_increment(self.telemetry, "push_rejection_count")
            raise
        finally:
            elapsed = time.perf_counter() - started
            if hasattr(self.telemetry, "record_push_latency"):
                self.telemetry.record_push_latency(elapsed)
            else:
                _telemetry_observe(self.telemetry, "push_latency_seconds", elapsed)

    def extend(self, transitions: Iterable[Sequence[Any]], *, fail_fast: bool = True) -> ValidationReport:
        report = ValidationReport()
        for idx, transition in enumerate(transitions):
            try:
                self.push(transition)
                if hasattr(report, "add_success"):
                    report.add_success(idx)
                else:
                    report.valid += 1
            except Exception as exc:
                if hasattr(report, "add_error"):
                    report.add_error(idx, exc)
                else:
                    report.invalid += 1
                    report.errors.append(f"index={idx}: {exc}")
                if fail_fast:
                    raise
        return report

    def _assert_sample_ready(self, batch_size: int) -> None:
        if int(batch_size) <= 0:
            cls = _error_cls("InvalidBatchSizeError", SamplingError)
            if cls is SamplingError:
                raise cls("batch_size must be > 0")
            raise cls(batch_size)
        if self.size <= 0:
            raise BufferEmptyError("sample")
        if not self.sample_with_replacement and batch_size > self.size:
            raise InsufficientSamplesError(batch_size, self.size, replace=False)
        total = _tree_total(self.sum_tree)
        if total <= 0 or not np.isfinite(total):
            raise _priority_mass_error(total, "cannot sample without positive finite priority mass")

    def sample(self, batch_size: int, beta: Optional[float] = None) -> Any:
        started = time.perf_counter()
        try:
            batch_n = int(batch_size)
            beta_value = self.default_beta if beta is None else float(beta)
            if not (0.0 <= beta_value <= 1.0):
                raise ConfigValueError("beta", beta_value, "float in [0, 1]")

            with self._locked("sample"):
                self._assert_sample_ready(batch_n)
                total = _tree_total(self.sum_tree)
                segment = total / batch_n
                indices: List[int] = []

                if self.sample_with_replacement:
                    for i in range(batch_n):
                        left = segment * i
                        right = segment * (i + 1)
                        mass = self._rng.uniform(left, right)
                        idx = _tree_prefix_index(self.sum_tree, mass)
                        while idx >= self.size or self.buffer[idx] is None:
                            mass = self._rng.uniform(0.0, total)
                            idx = _tree_prefix_index(self.sum_tree, mass)
                        indices.append(idx)
                else:
                    seen = set()
                    attempts = 0
                    while len(indices) < batch_n and attempts < batch_n * 20:
                        attempts += 1
                        idx = _tree_prefix_index(self.sum_tree, self._rng.uniform(0.0, total))
                        if idx < self.size and self.buffer[idx] is not None and idx not in seen:
                            seen.add(idx)
                            indices.append(idx)
                    if len(indices) < batch_n:
                        raise PrioritySamplingError("could not draw enough unique prioritized samples")

                transitions = [self.buffer[i] for i in indices]
                if any(t is None for t in transitions):
                    raise _mutation_error("sample", "sampled index without transition payload")

                scaled_priorities = self.scaled_priorities[np.asarray(indices, dtype=np.int64)]
                probabilities = scaled_priorities / max(total, self.epsilon)

                min_priority = _tree_min(self.min_tree, self.size)
                if not np.isfinite(min_priority) or min_priority <= 0:
                    min_priority = max(float(np.min(scaled_priorities)), self.epsilon)
                min_probability = max(min_priority / max(total, self.epsilon), self.epsilon)
                max_weight = (min_probability * self.size) ** (-beta_value) if beta_value > 0 else 1.0

                weights = (probabilities * self.size) ** (-beta_value) if beta_value > 0 else np.ones_like(probabilities)
                weights = weights / max(max_weight, self.epsilon)

                agent_ids, states, actions, rewards, next_states, dones = zip(*transitions)  # type: ignore[arg-type]
                result = PrioritizedSampleBatch(
                    agent_ids=np.asarray(agent_ids, dtype=object),
                    states=np.asarray(states, dtype=object),
                    actions=np.asarray(actions, dtype=object),
                    rewards=np.asarray(rewards, dtype=np.float32),
                    next_states=np.asarray(next_states, dtype=object),
                    dones=np.asarray(dones, dtype=np.bool_),
                    indices=np.asarray(indices, dtype=np.int64),
                    weights=weights.astype(np.float32),
                    probabilities=probabilities.astype(np.float32),
                    priorities=scaled_priorities.astype(np.float32),
                    beta=float(beta_value),
                )

                self.total_sampled += batch_n
                if beta is None and self.beta_increment_per_sample > 0:
                    self.default_beta = min(1.0, self.default_beta + self.beta_increment_per_sample)
                _telemetry_increment(self.telemetry, "sample_count")
                _telemetry_observe(self.telemetry, "last_batch_size", batch_n)
                return result.to_dict() if self.return_dict else result
        except BufferError:
            if hasattr(self.telemetry, "record_rejection"):
                self.telemetry.record_rejection("sample", "buffer_error")
            else:
                _telemetry_increment(self.telemetry, "sample_rejection_count")
            raise
        finally:
            elapsed = time.perf_counter() - started
            if hasattr(self.telemetry, "record_sample_latency"):
                self.telemetry.record_sample_latency(elapsed)
            else:
                _telemetry_observe(self.telemetry, "sample_latency_seconds", elapsed)

    def sample_tuple(self, batch_size: int, beta: Optional[float] = None) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        result = self.sample(batch_size=batch_size, beta=beta)
        if isinstance(result, PrioritizedSampleBatch):
            return result.as_tuple()
        batch = (
            result["agent_ids"], result["states"], result["actions"],
            result["rewards"], result["next_states"], result["dones"],
        )
        return batch, result["indices"], result["weights"]

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> PriorityUpdateReport:
        report = PriorityUpdateReport(requested=len(indices))
        if len(indices) != len(td_errors):
            raise _priority_update_error("indices and td_errors must have the same length", indices=indices)

        with self._locked("update_priorities"):
            for index, td_error in zip(indices, td_errors):
                try:
                    idx = int(index)
                    if idx < 0 or idx >= self.size or self.buffer[idx] is None:
                        raise IndexOutOfBoundsError(idx, self.size, "update_priorities")
                    raw, scaled = self._scaled_priority(td_error, from_td_error=True)
                    self.raw_priorities[idx] = raw
                    self.scaled_priorities[idx] = scaled
                    self.td_errors[idx] = float(td_error)
                    _tree_update(self.sum_tree, idx, scaled)
                    _tree_update(self.min_tree, idx, scaled)
                    self.max_raw_priority = max(self.max_raw_priority, raw)
                    report.updated += 1
                except Exception as exc:
                    report.rejected += 1
                    report.errors.append(f"index={index}: {exc}")
            if report.rejected:
                if hasattr(self.telemetry, "record_rejection"):
                    self.telemetry.record_rejection("update_priorities", "invalid_priority_update")
                else:
                    _telemetry_increment(self.telemetry, "priority_update_rejection_count", report.rejected)
                raise _priority_update_error("one or more priority updates failed", indices=indices)
            self.total_priority_updates += report.updated
            _telemetry_increment(self.telemetry, "priority_update_count", report.updated)
            return report

    def get(self, index: int) -> Transition:
        with self._locked("get"):
            idx = int(index)
            if idx < 0 or idx >= self.size or self.buffer[idx] is None:
                raise IndexOutOfBoundsError(idx, self.size, "get")
            return self.buffer[idx]  # type: ignore[return-value]

    def get_all(self) -> List[Transition]:
        with self._locked("get_all"):
            return [t for t in self.buffer[: self.size] if t is not None]

    def clear(self) -> None:
        with self._locked("clear"):
            self.buffer = [None] * self.capacity
            self.metadata = [{} for _ in range(self.capacity)]
            self.raw_priorities.fill(0.0)
            self.scaled_priorities.fill(0.0)
            self.td_errors.fill(0.0)
            self.sum_tree = SumSegmentTree(self.capacity)
            self.min_tree = MinSegmentTree(self.capacity)
            self.position = 0
            self.size = 0
            self.max_raw_priority = max(self.initial_priority, self.epsilon)

    def stats(self) -> Dict[str, Any]:
        with self._locked("stats"):
            total_mass = _tree_total(self.sum_tree)
            min_priority = _tree_min(self.min_tree, self.size) if self.size else 0.0
            return {
                "size": self.size,
                "capacity": self.capacity,
                "fill_ratio": self.size / self.capacity,
                "position": self.position,
                "alpha": self.alpha,
                "beta": self.default_beta,
                "epsilon": self.epsilon,
                "total_priority_mass": total_mass,
                "min_scaled_priority": 0.0 if not np.isfinite(min_priority) else min_priority,
                "max_raw_priority": self.max_raw_priority,
                "total_pushed": self.total_pushed,
                "total_sampled": self.total_sampled,
                "total_priority_updates": self.total_priority_updates,
                "total_overwrites": self.total_overwrites,
                "total_rejections": self.total_rejections,
            }

    def state_dict(self) -> Dict[str, Any]:
        with self._locked("state_dict"):
            return {
                "buffer": list(self.buffer),
                "metadata": list(self.metadata),
                "raw_priorities": self.raw_priorities.copy(),
                "scaled_priorities": self.scaled_priorities.copy(),
                "td_errors": self.td_errors.copy(),
                "position": self.position,
                "size": self.size,
                "max_raw_priority": self.max_raw_priority,
                "total_pushed": self.total_pushed,
                "total_sampled": self.total_sampled,
                "total_priority_updates": self.total_priority_updates,
                "total_overwrites": self.total_overwrites,
                "total_rejections": self.total_rejections,
                "rng_state": self._rng.getstate(),
                "stats": self.stats(),
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with self._locked("load_state_dict"):
            buffer = list(state.get("buffer", []))
            if len(buffer) > self.capacity:
                raise _mutation_error("load_state_dict", "checkpoint capacity exceeds current buffer capacity")
            self.clear()
            for i, transition in enumerate(buffer[: self.capacity]):
                self.buffer[i] = transition
            self.metadata = list(state.get("metadata", [{} for _ in range(self.capacity)]))[: self.capacity]
            while len(self.metadata) < self.capacity:
                self.metadata.append({})
            self.raw_priorities[:] = np.asarray(state.get("raw_priorities", np.zeros(self.capacity)), dtype=np.float64)[: self.capacity]
            self.scaled_priorities[:] = np.asarray(state.get("scaled_priorities", np.zeros(self.capacity)), dtype=np.float64)[: self.capacity]
            self.td_errors[:] = np.asarray(state.get("td_errors", np.zeros(self.capacity)), dtype=np.float64)[: self.capacity]
            self.position = int(state.get("position", 0)) % self.capacity
            self.size = min(int(state.get("size", len([x for x in buffer if x is not None]))), self.capacity)
            self.max_raw_priority = float(state.get("max_raw_priority", max(self.initial_priority, self.epsilon)))
            self.total_pushed = int(state.get("total_pushed", self.size))
            self.total_sampled = int(state.get("total_sampled", 0))
            self.total_priority_updates = int(state.get("total_priority_updates", 0))
            self.total_overwrites = int(state.get("total_overwrites", 0))
            self.total_rejections = int(state.get("total_rejections", 0))
            rng_state = state.get("rng_state")
            if rng_state is not None:
                self._rng.setstate(rng_state)
            self.sum_tree = SumSegmentTree(self.capacity)
            self.min_tree = MinSegmentTree(self.capacity)
            for idx in range(self.size):
                if self.buffer[idx] is not None:
                    _tree_update(self.sum_tree, idx, float(self.scaled_priorities[idx]))
                    _tree_update(self.min_tree, idx, float(self.scaled_priorities[idx]))

    def save(self, filepath: str) -> Any:
        if not self.persistence_enabled:
            raise BufferOperationError("PrioritizedReplayBuffer persistence is disabled", [])
        return self.checkpoint_io.save_checkpoint(
            self.state_dict(),
            filepath,
            component_name=self.checkpoint_component_name,
            schema_version=self.checkpoint_schema_version,
            metadata={"buffer_type": "prioritized_replay", "capacity": self.capacity},
            telemetry=self.telemetry,
            lock=self._lock,
        )

    def load(self, filepath: str) -> None:
        checkpoint = self.checkpoint_io.load_checkpoint(
            filepath,
            expected_component=self.checkpoint_component_name,
            telemetry=self.telemetry,
            lock=self._lock,
        )
        self.load_state_dict(checkpoint.state)


if __name__ == "__main__":
    print("\n=== Running  Prioritized Buffer ===\n")
    printer.status("TEST", " Prioritized Buffer initialized", "info")

    buf = PrioritizedReplayBuffer(user_config={"prioritized_replay": {"capacity": 8, "seed": 7, "return_dict": True}})
    for i in range(6):
        buf.push("agent", f"s{i}", i, float(i), f"s{i+1}", False, td_error=float(i + 1))
    assert len(buf) == 6

    batch = buf.sample(4, beta=0.5)
    assert batch["indices"].shape == (4,)
    assert batch["weights"].shape == (4,)
    assert np.all(batch["weights"] > 0)

    report = buf.update_priorities(batch["indices"], np.ones(4) * 2.5)
    assert report.updated == 4
    stats = buf.stats()
    assert stats["size"] == 6 and stats["total_priority_updates"] >= 4

    path = "/tmp/prioritized_replay_test.slai-buffer"
    buf.save(path)
    restored = PrioritizedReplayBuffer(user_config={"prioritized_replay": {"capacity": 8, "seed": 7}})
    restored.load(path)
    assert len(restored) == len(buf)
    assert restored.sample(2)["indices"].shape == (2,)

    buf.clear()
    try:
        buf.sample(1)
        raise AssertionError("expected BufferEmptyError")
    except BufferEmptyError:
        pass

    printer.status("TEST", " Prioritized Buffer checks passed", "success")
    print("\n=== Test ran successfully ===\n")
