from __future__ import annotations

import math
import random
import time
import numpy as np  # type: ignore

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from .buffer_persistence import BufferCheckpointIO, build_checkpoint_io
from .buffer_telemetry import BufferTelemetry
from .buffer_validation import Transition, TransitionValidator, ValidationReport
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Reservoir Buffer")
printer = PrettyPrinter()

TransitionBatch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# -----------------------------------------------------------------------------
# Configuration and reports
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReservoirConfig:
    """Configuration contract for unbiased streaming reservoir replay.

    Config is loaded from the shared ``reservoir`` section in buffer_config.yaml.
    User overrides may be supplied as ``{"reservoir": {...}}`` or as a direct
    mapping of reservoir fields, matching the existing buffer config pattern.
    """

    capacity: int = 100_000
    seed: Optional[int] = None
    default_replace: bool = False
    lock_timeout_seconds: float = 5.0
    track_agent_stats: bool = True
    checkpoint_schema_version: str = "reservoir_replay_buffer.v1"
    checkpoint_component_name: str = "reservoir_replay_buffer"
    persistence_enabled: bool = True
    max_extend_errors: int = 100

    @classmethod
    def from_config(cls, user_config: Optional[Mapping[str, Any]] = None) -> "ReservoirConfig":
        load_global_config()
        config: Dict[str, Any] = dict(get_config_section("reservoir") or {})
        if user_config:
            override = user_config.get("reservoir", user_config) if isinstance(user_config, Mapping) else {}
            if isinstance(override, Mapping):
                config.update(dict(override))

        capacity = positive_int(config.get("capacity", 100_000), "capacity")
        lock_timeout = non_negative_float(config.get("lock_timeout_seconds", 5.0), "lock_timeout_seconds")
        max_extend_errors = positive_int(config.get("max_extend_errors", 100), "max_extend_errors")
        seed = config.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ConfigValueError("seed", seed, "integer or null", section="reservoir") from exc

        return cls(
            capacity=capacity,
            seed=seed,
            default_replace=bool(config.get("default_replace", False)),
            lock_timeout_seconds=lock_timeout,
            track_agent_stats=bool(config.get("track_agent_stats", True)),
            checkpoint_schema_version=str(config.get("checkpoint_schema_version", "reservoir_replay_buffer.v1")),
            checkpoint_component_name=str(config.get("checkpoint_component_name", "reservoir_replay_buffer")),
            persistence_enabled=bool(config.get("persistence_enabled", True)),
            max_extend_errors=max_extend_errors,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "default_replace": self.default_replace,
            "lock_timeout_seconds": self.lock_timeout_seconds,
            "track_agent_stats": self.track_agent_stats,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "checkpoint_component_name": self.checkpoint_component_name,
            "persistence_enabled": self.persistence_enabled,
            "max_extend_errors": self.max_extend_errors,
        }


@dataclass(frozen=True)
class ReservoirSample:
    """Structured sample result for callers that need selected indices."""

    batch: TransitionBatch
    indices: np.ndarray
    replace: bool
    batch_size: int

    def as_tuple(self) -> TransitionBatch:
        return self.batch


@dataclass
class ReservoirIngestReport:
    """Bulk ingestion result for streaming producers."""

    attempted: int = 0
    accepted: int = 0
    retained: int = 0
    replaced: int = 0
    skipped: int = 0
    rejected: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.rejected == 0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.attempted if self.attempted else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "retained": self.retained,
            "replaced": self.replaced,
            "skipped": self.skipped,
            "rejected": self.rejected,
            "rejection_rate": self.rejection_rate,
            "errors": list(self.errors),
        }


# -----------------------------------------------------------------------------
# Reservoir replay buffer
# -----------------------------------------------------------------------------

class ReservoirReplayBuffer:
    """Unbiased fixed-memory replay buffer for unbounded streams.

    Algorithm:
        For the Nth accepted transition, keep it automatically while N <= capacity.
        Once full, draw j uniformly from [0, N). If j < capacity, replace slot j;
        otherwise the new transition is counted in the stream but not retained.

    This preserves the standard reservoir guarantee: every accepted transition has
    equal probability ``min(1, capacity / total_seen)`` of being retained.
    """

    def __init__(
        self,
        user_config: Optional[Mapping[str, Any]] = None,
        validator: Optional[TransitionValidator] = None,
        telemetry: Optional[BufferTelemetry] = None,
        checkpoint_io: Optional[BufferCheckpointIO] = None,
    ) -> None:
        self.config = ReservoirConfig.from_config(user_config=user_config)
        self.capacity = self.config.capacity
        self.validator = validator or TransitionValidator()
        self.telemetry = telemetry or BufferTelemetry(component_name="reservoir_replay_buffer")
        self.checkpoint_io = checkpoint_io or build_checkpoint_io(user_config=user_config, telemetry=self.telemetry)

        self._rng = random.Random(self.config.seed)
        self._lock = RLock()
        self._closed = False

        self.buffer: List[Transition] = []
        self._sequence_numbers: List[int] = []
        self._inserted_at: List[float] = []
        self._agent_counts: Dict[Any, int] = {}
        self._reward_sum = 0.0
        self._reward_min: Optional[float] = None
        self._reward_max: Optional[float] = None

        self.total_seen = 0
        self.total_retained = 0
        self.total_replaced = 0
        self.total_skipped = 0
        self.total_rejected = 0

    # ------------------------------------------------------------------ #
    # Core protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __repr__(self) -> str:
        return (
            "ReservoirReplayBuffer("
            f"size={len(self)}, capacity={self.capacity}, total_seen={self.total_seen}, "
            f"total_replaced={self.total_replaced})"
        )

    @contextmanager
    def _locked(self, operation: str) -> Iterator[None]:
        start = time.perf_counter()
        acquired = self._lock.acquire(timeout=self.config.lock_timeout_seconds)
        waited = time.perf_counter() - start
        telemetry_call(self.telemetry, "record_lock_wait", operation, waited, acquired=acquired)
        if not acquired:
            raise BufferLockTimeoutError(operation=operation, timeout_seconds=self.config.lock_timeout_seconds)
        try:
            if self._closed:
                raise ReservoirBufferError(f"Cannot {operation}: reservoir buffer is closed")
            yield
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def push(self, transition: Sequence[Any]) -> Optional[int]:
        """Validate and ingest one transition.

        Returns the retained slot index when the transition is stored/replaced, or
        ``None`` when the transition was valid but skipped by reservoir sampling.
        """
        start = time.perf_counter()
        try:
            with self._locked("push"):
                normalized = self.validator.sanitize_transition(transition)
                slot = self._append_or_sample_locked(normalized)
                telemetry_call(self.telemetry, "record_acceptance", "push")
                telemetry_increment(self.telemetry, "push_count", 1)
                if slot is None:
                    telemetry_increment(self.telemetry, "reservoir_skip_count", 1)
                else:
                    telemetry_increment(self.telemetry, "reservoir_retained_count", 1)
                return slot
        except TransitionValidationError:
            self.total_rejected += 1
            telemetry_call(self.telemetry, "record_rejection", "push", "transition_validation")
            raise
        finally:
            elapsed = time.perf_counter() - start
            telemetry_call(self.telemetry, "record_push_latency", elapsed)
            telemetry_observe(self.telemetry, "push_latency_seconds", elapsed)

    def push_components(
        self,
        agent_id: Any,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        done: Any,
    ) -> Optional[int]:
        return self.push((agent_id, state, action, reward, next_state, done))

    def add(self, transition: Sequence[Any]) -> Optional[int]:
        """Alias for push(), useful for producer-style call sites."""
        return self.push(transition)

    def extend(self, transitions: Iterable[Sequence[Any]], *, fail_fast: bool = True) -> ReservoirIngestReport:
        """Ingest multiple transitions and return a structured report."""
        report = ReservoirIngestReport()
        errors: List[BaseException] = []
        for idx, transition in enumerate(transitions):
            report.attempted += 1
            try:
                before_replaced = self.total_replaced
                retained_slot = self.push(transition)
                report.accepted += 1
                if retained_slot is None:
                    report.skipped += 1
                else:
                    report.retained += 1
                if self.total_replaced > before_replaced:
                    report.replaced += 1
            except TransitionValidationError as exc:
                report.rejected += 1
                errors.append(exc)
                if len(report.errors) < self.config.max_extend_errors:
                    report.errors.append(f"index={idx}: {exc}")
                if fail_fast:
                    raise
        if errors and not fail_fast:
            telemetry_call(self.telemetry, "record_rejection", "extend", "partial_validation_failure")
            logger.warning("Reservoir extend completed with %s rejected transitions", len(errors))
        return report

    def _append_or_sample_locked(self, transition: Transition) -> Optional[int]:
        self.total_seen += 1
        sequence_no = self.total_seen
        now = time.time()

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self._sequence_numbers.append(sequence_no)
            self._inserted_at.append(now)
            slot = len(self.buffer) - 1
            self.total_retained += 1
            self._add_stats(transition)
            return slot

        candidate = self._rng.randrange(sequence_no)
        if candidate < self.capacity:
            old = self.buffer[candidate]
            self._remove_stats(old)
            self.buffer[candidate] = transition
            self._sequence_numbers[candidate] = sequence_no
            self._inserted_at[candidate] = now
            self.total_retained += 1
            self.total_replaced += 1
            self._add_stats(transition)
            return candidate

        self.total_skipped += 1
        return None

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int, replace: Optional[bool] = None) -> TransitionBatch:
        return self.sample_with_indices(batch_size=batch_size, replace=replace).batch

    def sample_with_indices(self, batch_size: int, replace: Optional[bool] = None) -> ReservoirSample:
        started = time.perf_counter()
        resolved_size = validate_batch_size(batch_size)
        resolved_replace = self.config.default_replace if replace is None else bool(replace)
        try:
            with self._locked("sample"):
                size = len(self.buffer)
                if size == 0:
                    telemetry_call(self.telemetry, "record_rejection", "sample", "empty_buffer")
                    raise BufferEmptyError(operation="sample")
                if not resolved_replace and resolved_size > size:
                    telemetry_call(self.telemetry, "record_rejection", "sample", "insufficient_samples")
                    raise InsufficientSamplesError(requested=resolved_size, available=size, replace=False)

                indices = (
                    [self._rng.randrange(size) for _ in range(resolved_size)]
                    if resolved_replace
                    else self._rng.sample(range(size), resolved_size)
                )
                batch = self._format_batch([self.buffer[i] for i in indices])
                telemetry_call(self.telemetry, "record_acceptance", "sample")
                telemetry_increment(self.telemetry, "sample_count", 1)
                telemetry_observe(self.telemetry, "last_batch_size", float(resolved_size))
                return ReservoirSample(
                    batch=batch,
                    indices=np.asarray(indices, dtype=np.int64),
                    replace=resolved_replace,
                    batch_size=resolved_size,
                )
        finally:
            elapsed = time.perf_counter() - started
            telemetry_call(self.telemetry, "record_sample_latency", elapsed)
            telemetry_observe(self.telemetry, "sample_latency_seconds", elapsed)

    def _format_batch(self, batch: Sequence[Transition]) -> TransitionBatch:
        if not batch:
            raise BufferEmptyError(operation="format_batch")
        agent_ids, states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(agent_ids, dtype=object),
            np.asarray(states, dtype=object),
            np.asarray(actions, dtype=object),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=object),
            np.asarray(dones, dtype=np.bool_),
        )

    # ------------------------------------------------------------------ #
    # Introspection and state
    # ------------------------------------------------------------------ #
    def get_all(self) -> Tuple[List[Any], List[Any], List[Any], List[float], List[Any], List[bool]]:
        with self._locked("get_all"):
            if not self.buffer:
                return [], [], [], [], [], []
            agent_ids, states, actions, rewards, next_states, dones = zip(*self.buffer)
            return list(agent_ids), list(states), list(actions), list(rewards), list(next_states), list(dones)

    def stats(self) -> Dict[str, Any]:
        with self._locked("stats"):
            size = len(self.buffer)
            retention_probability = min(1.0, self.capacity / self.total_seen) if self.total_seen else 0.0
            return {
                "size": size,
                "capacity": self.capacity,
                "fill_ratio": size / self.capacity if self.capacity else 0.0,
                "total_seen": self.total_seen,
                "total_retained": self.total_retained,
                "total_replaced": self.total_replaced,
                "total_skipped": self.total_skipped,
                "total_rejected": self.total_rejected,
                "replacement_rate": self.total_replaced / self.total_seen if self.total_seen else 0.0,
                "skip_rate": self.total_skipped / self.total_seen if self.total_seen else 0.0,
                "stream_retention_probability": retention_probability,
                "avg_reward": self._reward_sum / size if size else 0.0,
                "max_reward": self._reward_max if size else None,
                "min_reward": self._reward_min if size else None,
                "active_agents": len(self._agent_counts),
                "agent_counts": dict(self._agent_counts) if self.config.track_agent_stats else {},
                "closed": self._closed,
            }

    def snapshot(self, *, include_telemetry: bool = True) -> Dict[str, Any]:
        with self._locked("snapshot"):
            payload = self.state_dict(include_rng_state=False)
            payload["stats"] = self.stats()
            if include_telemetry:
                snap = getattr(self.telemetry, "snapshot", None)
                payload["telemetry"] = snap() if callable(snap) else {}
            return payload

    def state_dict(self, *, include_rng_state: bool = True) -> Dict[str, Any]:
        with self._lock:
            state = {
                "schema_version": self.config.checkpoint_schema_version,
                "config": self.config.to_dict(),
                "buffer": list(self.buffer),
                "sequence_numbers": list(self._sequence_numbers),
                "inserted_at": list(self._inserted_at),
                "total_seen": self.total_seen,
                "total_retained": self.total_retained,
                "total_replaced": self.total_replaced,
                "total_skipped": self.total_skipped,
                "total_rejected": self.total_rejected,
                "agent_counts": dict(self._agent_counts),
                "reward_sum": self._reward_sum,
                "reward_min": self._reward_min,
                "reward_max": self._reward_max,
                "closed": self._closed,
            }
            if include_rng_state:
                state["rng_state"] = self._rng.getstate()
            return state

    def load_state_dict(self, state: Mapping[str, Any], *, restore_rng: bool = True) -> None:
        with self._locked("load_state_dict"):
            buffer = list(state.get("buffer", []))
            if len(buffer) > self.capacity:
                raise ReservoirBufferError(
                    f"Loaded reservoir state has {len(buffer)} items but configured capacity is {self.capacity}"
                )
            self.buffer = [self.validator.sanitize_transition(item) for item in buffer]
            self._sequence_numbers = list(state.get("sequence_numbers", range(1, len(self.buffer) + 1)))[: len(self.buffer)]
            self._inserted_at = list(state.get("inserted_at", [time.time()] * len(self.buffer)))[: len(self.buffer)]
            self.total_seen = int(state.get("total_seen", len(self.buffer)))
            self.total_retained = int(state.get("total_retained", len(self.buffer)))
            self.total_replaced = int(state.get("total_replaced", 0))
            self.total_skipped = int(state.get("total_skipped", max(0, self.total_seen - self.total_retained)))
            self.total_rejected = int(state.get("total_rejected", 0))
            self._closed = bool(state.get("closed", False))
            if restore_rng and state.get("rng_state") is not None:
                self._rng.setstate(state["rng_state"])
            self._rebuild_statistics_locked()
            self.validate_invariants()

    def validate_invariants(self) -> None:
        with self._lock:
            size = len(self.buffer)
            if size > self.capacity:
                raise BufferMutationError("validate_invariants", "reservoir size exceeds capacity")
            if len(self._sequence_numbers) != size or len(self._inserted_at) != size:
                raise BufferMutationError("validate_invariants", "metadata length does not match buffer size")
            if self.total_seen < size:
                raise BufferMutationError("validate_invariants", "total_seen cannot be lower than retained size")

    def clear(self) -> None:
        with self._locked("clear"):
            self.buffer.clear()
            self._sequence_numbers.clear()
            self._inserted_at.clear()
            self._agent_counts.clear()
            self._reward_sum = 0.0
            self._reward_min = None
            self._reward_max = None
            self.total_seen = 0
            self.total_retained = 0
            self.total_replaced = 0
            self.total_skipped = 0
            self.total_rejected = 0
            telemetry_increment(self.telemetry, "clear_count", 1)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, filepath: Union[str, Path]) -> Path:
        if not self.config.persistence_enabled:
            raise BufferSaveError(str(filepath), "reservoir persistence is disabled")
        state = self.state_dict(include_rng_state=True)
        return self.checkpoint_io.save_checkpoint(
            state,
            filepath,
            component_name=self.config.checkpoint_component_name,
            schema_version=self.config.checkpoint_schema_version,
            metadata={"buffer_type": "reservoir", "capacity": self.capacity},
            telemetry=self.telemetry,
        )

    def load(self, filepath: Union[str, Path]) -> None:
        checkpoint = self.checkpoint_io.load_checkpoint(
            filepath,
            expected_component=self.config.checkpoint_component_name,
            telemetry=self.telemetry,
        )
        state = checkpoint.state
        if not isinstance(state, Mapping):
            raise ReservoirBufferError(f"Reservoir checkpoint state must be a mapping, got {type(state).__name__}")
        self.load_state_dict(state)

    # ------------------------------------------------------------------ #
    # Internal statistics
    # ------------------------------------------------------------------ #
    def _add_stats(self, transition: Transition) -> None:
        reward = float(transition[3])
        self._reward_sum += reward
        self._reward_min = reward if self._reward_min is None else min(self._reward_min, reward)
        self._reward_max = reward if self._reward_max is None else max(self._reward_max, reward)
        if self.config.track_agent_stats:
            agent = transition[0]
            self._agent_counts[agent] = self._agent_counts.get(agent, 0) + 1

    def _remove_stats(self, transition: Transition) -> None:
        reward = float(transition[3])
        self._reward_sum -= reward
        if self.config.track_agent_stats:
            agent = transition[0]
            count = self._agent_counts.get(agent, 0) - 1
            if count > 0:
                self._agent_counts[agent] = count
            else:
                self._agent_counts.pop(agent, None)
        # Min/max require a rebuild when the removed item may have been an extreme.
        if reward == self._reward_min or reward == self._reward_max:
            self._rebuild_statistics_locked()

    def _rebuild_statistics_locked(self) -> None:
        self._agent_counts = {}
        self._reward_sum = 0.0
        self._reward_min = None
        self._reward_max = None
        for transition in self.buffer:
            self._add_stats(transition)


__all__ = [
    "TransitionBatch",
    "ReservoirIngestReport",
    "ReservoirReplayBuffer",
]


if __name__ == "__main__":
    print("\n=== Running  Reservoir Buffer ===\n")
    printer.status("TEST", " Reservoir Buffer initialized", "info")
    cfg = {"reservoir": {"capacity": 10, "seed": 7, "lock_timeout_seconds": 1.0}}
    buf = ReservoirReplayBuffer(user_config=cfg)
    for i in range(50):
        buf.push_components(f"agent_{i % 3}", np.array([i], dtype=np.float32), i % 2, float(i), np.array([i + 1], dtype=np.float32), i % 17 == 0)
    assert len(buf) == 10
    stats = buf.stats()
    assert stats["total_seen"] == 50 and stats["size"] == 10
    sample = buf.sample_with_indices(5)
    assert sample.batch[0].shape[0] == 5 and len(sample.indices) == 5
    report = buf.extend([(0, [1], 1, 1.0, [2], False), (1, [2], 0, 2.0, [3], True)])
    assert report.accepted == 2 and len(buf) == 10
    state = buf.state_dict()
    clone = ReservoirReplayBuffer(user_config=cfg)
    clone.load_state_dict(state)
    assert clone.stats()["total_seen"] == buf.stats()["total_seen"]
    all_data = clone.get_all()
    assert len(all_data[0]) == 10
    clone.clear()
    assert len(clone) == 0
    printer.status("TEST", "Reservoir Buffer checks passed", "success")
    print("\n=== Test ran successfully ===\n")
