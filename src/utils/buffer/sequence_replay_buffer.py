from __future__ import annotations

import random
import time
import numpy as np  # type: ignore

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from .buffer_persistence import BufferCheckpointIO, build_checkpoint_io
from .buffer_telemetry import BufferTelemetry
from .buffer_validation import TransitionValidator
from .eviction_policies import EvictionContext, build_eviction_policy
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Sequence Replay Buffer")
printer = PrettyPrinter()

Transition = Tuple[Any, Any, Any, float, Any, bool]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SequenceWindow:
    """Resolved contiguous sampling window inside an episode."""

    episode_index: int
    start_index: int
    valid_length: int
    episode_length: int
    is_current_episode: bool = False


@dataclass
class SequenceIngestReport:
    """Compact batch-ingestion report for sequence replay."""

    accepted: int = 0
    rejected: int = 0
    closed_episodes: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.total if self.total else 0.0

    @property
    def is_clean(self) -> bool:
        return self.rejected == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "closed_episodes": self.closed_episodes,
            "total": self.total,
            "rejection_rate": self.rejection_rate,
            "errors": list(self.errors),
            "elapsed_seconds": self.elapsed_seconds,
        }


class SequenceReplayBuffer:
    """Replay buffer for contiguous sequence sampling.

    Config handling intentionally stays in ``__init__`` using the existing
    project pattern:
        self.config = load_global_config()
        self.sequence_cfg = get_config_section("sequence_replay") or {}

    The buffer stores validated transitions grouped by episode and returns
    padded sequence batches with masks for recurrent/Transformer training.
    """

    def __init__(
        self,
        user_config: Optional[Mapping[str, Any]] = None,
        validator: Optional[TransitionValidator] = None,
        telemetry: Optional[BufferTelemetry] = None,
        checkpoint_io: Optional[BufferCheckpointIO] = None,
    ) -> None:
        self.config = load_global_config()
        self.sequence_cfg = get_config_section("sequence_replay") or {}
        if user_config and isinstance(user_config.get("sequence_replay"), Mapping):
            self.sequence_cfg = dict(self.sequence_cfg)
            self.sequence_cfg.update(dict(user_config["sequence_replay"]))

        self.capacity = int(self.sequence_cfg.get("capacity", 100_000))
        self.sequence_length = int(self.sequence_cfg.get("sequence_length", 16))
        self.default_burn_in = int(self.sequence_cfg.get("default_burn_in", self.sequence_cfg.get("burn_in", 4)))
        self.burn_in = self.default_burn_in
        self.min_episode_length = int(self.sequence_cfg.get("min_episode_length", 2))
        self.pad_value = self.sequence_cfg.get("pad_value", 0.0)
        self.include_current_episode = bool(self.sequence_cfg.get("include_current_episode", True))
        self.allow_partial_sequences = bool(self.sequence_cfg.get("allow_partial_sequences", False))
        self.sample_with_replacement = bool(self.sequence_cfg.get("sample_with_replacement", True))
        self.lock_timeout_seconds = float(self.sequence_cfg.get("lock_timeout_seconds", 5.0))
        self.max_extend_errors = max(0, int(self.sequence_cfg.get("max_extend_errors", 100)))
        self.track_episode_stats = bool(self.sequence_cfg.get("track_episode_stats", True))
        self.persistence_enabled = bool(self.sequence_cfg.get("persistence_enabled", True))
        self.checkpoint_schema_version = str(self.sequence_cfg.get("checkpoint_schema_version", "sequence_replay_buffer.v1"))
        self.checkpoint_component_name = str(self.sequence_cfg.get("checkpoint_component_name", "sequence_replay_buffer"))
        self._validate_config_values()

        self.lock = RLock()
        self.validator = validator or TransitionValidator()
        self.telemetry = telemetry or BufferTelemetry(component_name=self.checkpoint_component_name)
        self.checkpoint_io = checkpoint_io or build_checkpoint_io(user_config=user_config, telemetry=self.telemetry)
        self.eviction_policy = build_eviction_policy(user_config=dict(user_config or {}))
        self._rng = random.Random(self.sequence_cfg.get("seed"))

        self.episodes: Deque[List[Transition]] = deque()
        self.current_episode: List[Transition] = []
        self.total_transitions = 0
        self.total_pushed = 0
        self.total_sampled_sequences = 0
        self.total_evicted_episodes = 0
        self.total_evicted_transitions = 0
        self.total_rejected = 0
        self.total_closed_episodes = 0
        self._created_at = utcnow_iso()

    def __len__(self) -> int:
        with self._locked("len"):
            return self.total_transitions

    @property
    def default_window(self) -> int:
        return self.sequence_length + self.default_burn_in

    def _validate_config_values(self) -> None:
        if self.capacity <= 0:
            raise ConfigValueError("sequence_replay.capacity", self.capacity, "> 0")
        if self.sequence_length <= 0:
            raise InvalidSequenceLengthError(self.sequence_length, 1)
        if self.default_burn_in < 0:
            raise ConfigValueError("sequence_replay.burn_in", self.default_burn_in, ">= 0")
        if self.min_episode_length <= 0:
            raise ConfigValueError("sequence_replay.min_episode_length", self.min_episode_length, "> 0")
        if self.lock_timeout_seconds < 0:
            raise ConfigValueError("sequence_replay.lock_timeout_seconds", self.lock_timeout_seconds, ">= 0")
        if not self.checkpoint_schema_version.strip():
            raise ConfigValueError("sequence_replay.checkpoint_schema_version", self.checkpoint_schema_version, "non-empty string")
        if not self.checkpoint_component_name.strip():
            raise ConfigValueError("sequence_replay.checkpoint_component_name", self.checkpoint_component_name, "non-empty string")

    def _record_latency(self, name: str, elapsed: float) -> None:
        if name == "push_latency_seconds" and hasattr(self.telemetry, "record_push_latency"):
            self.telemetry.record_push_latency(elapsed)
        elif name == "sample_latency_seconds" and hasattr(self.telemetry, "record_sample_latency"):
            self.telemetry.record_sample_latency(elapsed)
        else:
            self.telemetry.observe(name, elapsed)

    def _record_rejection(self, operation: str, reason: str) -> None:
        self.total_rejected += 1
        if hasattr(self.telemetry, "record_rejection"):
            self.telemetry.record_rejection(operation, reason)
        else:
            self.telemetry.increment(f"{operation}_rejection_count", 1)
            self.telemetry.increment("rejection_count", 1)

    def _record_lock(self, operation: str, waited: float, acquired: bool) -> None:
        if hasattr(self.telemetry, "record_lock_contention"):
            self.telemetry.record_lock_contention(operation, waited, acquired=acquired)
        else:
            self.telemetry.observe("lock_wait_seconds", waited)
            if not acquired:
                self.telemetry.increment("lock_timeout_count", 1)
            elif waited > 0.0:
                self.telemetry.increment("lock_contention_count", 1)

    @contextmanager
    def _locked(self, operation: str) -> Iterator[None]:
        started = time.perf_counter()
        acquired = self.lock.acquire(timeout=self.lock_timeout_seconds) if self.lock_timeout_seconds > 0 else self.lock.acquire(blocking=False)
        waited = time.perf_counter() - started
        self._record_lock(operation, waited, acquired)
        if not acquired:
            self._record_rejection(operation, "lock_timeout")
            raise BufferLockTimeoutError(operation=operation, timeout_seconds=self.lock_timeout_seconds)
        try:
            yield
        finally:
            self.lock.release()

    def push(self, agent_id: Any, state: Any, action: Any, reward: Any, next_state: Any, done: Any) -> None:
        started = time.perf_counter()
        try:
            with self._locked("push"):
                transition = self.validator.sanitize_transition((agent_id, state, action, reward, next_state, done))
                self.current_episode.append(transition)
                self.total_transitions += 1
                self.total_pushed += 1
                self.telemetry.increment("push_count", 1)
                if bool(transition[-1]):
                    self._close_current_episode_locked(reason="terminal")
                self._enforce_capacity_locked()
        except BufferError:
            self._record_rejection("push", "buffer_error")
            raise
        except Exception as exc:
            self._record_rejection("push", "unexpected_error")
            raise SequenceReplayError(f"Failed to push transition into sequence replay buffer: {exc}") from exc
        finally:
            self._record_latency("push_latency_seconds", time.perf_counter() - started)

    def push_transition(self, transition: Sequence[Any]) -> None:
        if len(transition) != 6:
            raise TransitionLengthError(expected=6, actual=len(transition))
        self.push(*transition)

    def extend(self, transitions: Iterable[Sequence[Any]], *, fail_fast: bool = True) -> SequenceIngestReport:
        started = time.perf_counter()
        report = SequenceIngestReport()
        errors: List[BaseException] = []
        for idx, transition in enumerate(transitions):
            before_closed = self.total_closed_episodes
            try:
                self.push_transition(transition)
                report.accepted += 1
                report.closed_episodes += max(0, self.total_closed_episodes - before_closed)
            except Exception as exc:
                report.rejected += 1
                self._record_rejection("extend", type(exc).__name__)
                if len(report.errors) < self.max_extend_errors:
                    report.errors.append(f"index={idx}: {type(exc).__name__}: {exc}")
                errors.append(exc)
                if fail_fast:
                    report.elapsed_seconds = time.perf_counter() - started
                    raise BufferOperationError("Sequence replay extend failed", errors) from exc
        report.elapsed_seconds = time.perf_counter() - started
        self.telemetry.observe("extend_latency_seconds", report.elapsed_seconds)
        self.telemetry.observe("extend_rejection_rate", report.rejection_rate)
        return report

    def flush_current_episode(self) -> bool:
        with self._locked("flush_current_episode"):
            if not self.current_episode:
                return False
            return self._close_current_episode_locked(reason="manual_flush")

    def clear(self) -> None:
        with self._locked("clear"):
            self.episodes.clear()
            self.current_episode = []
            self.total_transitions = 0
            self.total_sampled_sequences = 0
            self.telemetry.increment("clear_count", 1)

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: Optional[int] = None,
        burn_in: Optional[int] = None,
        *,
        replace: Optional[bool] = None,
        include_current_episode: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:
        started = time.perf_counter()
        try:
            with self._locked("sample_sequences"):
                batch_size = self._validate_batch_size(batch_size)
                seq_len, burn, window = self._resolve_window(sequence_length, burn_in)
                include_current = self.include_current_episode if include_current_episode is None else bool(include_current_episode)
                use_replace = self.sample_with_replacement if replace is None else bool(replace)

                windows = self._candidate_windows(window=window, include_current_episode=include_current)
                if not windows:
                    self._record_rejection("sample", "no_eligible_windows")
                    if self.total_transitions == 0:
                        raise BufferEmptyError("sample_sequences")
                    raise InsufficientSamplesError(requested=batch_size, available=0, replace=use_replace)
                if not use_replace and batch_size > len(windows):
                    self._record_rejection("sample", "insufficient_windows")
                    raise InsufficientSamplesError(requested=batch_size, available=len(windows), replace=False)

                chosen = [self._rng.choice(windows) for _ in range(batch_size)] if use_replace else self._rng.sample(windows, batch_size)
                payload = self._assemble_batch(chosen, window=window, sequence_length=seq_len, burn_in=burn)
                self.total_sampled_sequences += batch_size
                self.telemetry.increment("sample_count", 1)
                self.telemetry.observe("sampled_sequence_count", float(batch_size))
                self.telemetry.observe("eligible_window_count", float(len(windows)))
                self.telemetry.observe("last_batch_size", float(batch_size))
                return payload
        except BufferError:
            raise
        except Exception as exc:
            self._record_rejection("sample", "unexpected_error")
            raise SequenceReplayError(f"Failed to sample sequence batch: {exc}") from exc
        finally:
            self._record_latency("sample_latency_seconds", time.perf_counter() - started)

    def sample(self, batch_size: int, **kwargs: Any) -> Dict[str, np.ndarray]:
        return self.sample_sequences(batch_size=batch_size, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        with self._locked("state_dict"):
            return {
                "episodes": [list(ep) for ep in self.episodes],
                "current_episode": list(self.current_episode),
                "total_transitions": self.total_transitions,
                "total_pushed": self.total_pushed,
                "total_sampled_sequences": self.total_sampled_sequences,
                "total_evicted_episodes": self.total_evicted_episodes,
                "total_evicted_transitions": self.total_evicted_transitions,
                "total_rejected": self.total_rejected,
                "total_closed_episodes": self.total_closed_episodes,
                "created_at": self._created_at,
                "sequence_cfg": dict(self.sequence_cfg),
                "rng_state": self._rng.getstate(),
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with self._locked("load_state_dict"):
            episodes = [list(ep) for ep in state.get("episodes", [])]
            current = list(state.get("current_episode", []))
            self.episodes = deque(episodes)
            self.current_episode = current
            self.total_transitions = int(state.get("total_transitions", sum(len(ep) for ep in episodes) + len(current)))
            self.total_pushed = int(state.get("total_pushed", self.total_transitions))
            self.total_sampled_sequences = int(state.get("total_sampled_sequences", 0))
            self.total_evicted_episodes = int(state.get("total_evicted_episodes", 0))
            self.total_evicted_transitions = int(state.get("total_evicted_transitions", 0))
            self.total_rejected = int(state.get("total_rejected", 0))
            self.total_closed_episodes = int(state.get("total_closed_episodes", len(episodes)))
            self._created_at = str(state.get("created_at", utcnow_iso()))
            rng_state = state.get("rng_state")
            if rng_state is not None:
                self._rng.setstate(rng_state)
            self._validate_invariants_locked()

    def save(self, filepath: Union[str, Path]) -> Path:
        if not self.persistence_enabled:
            raise SequenceReplayError("Sequence replay persistence is disabled by config")
        return self.checkpoint_io.save_checkpoint(
            self.state_dict(),
            filepath,
            component_name=self.checkpoint_component_name,
            schema_version=self.checkpoint_schema_version,
            metadata={"buffer_type": "SequenceReplayBuffer", "created_at": self._created_at},
            telemetry=self.telemetry,
            lock=self.lock,
        )

    def load(self, filepath: Union[str, Path]) -> None:
        checkpoint = self.checkpoint_io.load_checkpoint(
            filepath,
            expected_component=self.checkpoint_component_name,
            telemetry=self.telemetry,
            lock=self.lock,
        )
        self.load_state_dict(checkpoint.state)

    def stats(self) -> Dict[str, Any]:
        with self._locked("stats"):
            episode_lengths = [len(ep) for ep in self.episodes]
            if self.current_episode:
                episode_lengths.append(len(self.current_episode))
            closed = len(self.episodes)
            return {
                "total_transitions": self.total_transitions,
                "capacity": self.capacity,
                "num_episodes": closed + (1 if self.current_episode else 0),
                "closed_episodes": closed,
                "current_episode_length": len(self.current_episode),
                "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "min_episode_length": int(np.min(episode_lengths)) if episode_lengths else 0,
                "max_episode_length": int(np.max(episode_lengths)) if episode_lengths else 0,
                "sequence_length": self.sequence_length,
                "burn_in": self.burn_in,
                "default_window": self.default_window,
                "total_pushed": self.total_pushed,
                "total_sampled_sequences": self.total_sampled_sequences,
                "total_evicted_episodes": self.total_evicted_episodes,
                "total_evicted_transitions": self.total_evicted_transitions,
                "total_rejected": self.total_rejected,
                "created_at": self._created_at,
            }

    def get_all_episodes(self, *, include_current: bool = True) -> List[List[Transition]]:
        with self._locked("get_all_episodes"):
            episodes = [list(ep) for ep in self.episodes]
            if include_current and self.current_episode:
                episodes.append(list(self.current_episode))
            return episodes

    def _close_current_episode_locked(self, *, reason: str) -> bool:
        length = len(self.current_episode)
        if length <= 0:
            return False
        if length >= self.min_episode_length:
            self.episodes.append(list(self.current_episode))
            self.total_closed_episodes += 1
            self.telemetry.increment("closed_episode_count", 1)
            self.telemetry.observe("closed_episode_length", float(length))
            accepted = True
        else:
            self.total_transitions -= length
            self.telemetry.increment("dropped_short_episode_count", 1)
            self.telemetry.observe("dropped_short_episode_length", float(length))
            accepted = False
        self.current_episode = []
        self.telemetry.increment(f"episode_close_reason_{reason}", 1)
        self._enforce_capacity_locked()
        return accepted

    def _enforce_capacity_locked(self) -> None:
        while self.total_transitions > self.capacity:
            overflow = self.total_transitions - self.capacity
            if self.episodes:
                context = EvictionContext(
                    overflow=1,
                    total_items=self.total_transitions,
                    metadata=self._eviction_metadata_locked(),
                )
                selected = (
                    list(self.eviction_policy.select_indices(self.episodes, context=context))[0]
                    if hasattr(self.eviction_policy, "select_indices")
                    else self.eviction_policy.select_index(self.episodes, context=context)
                )
                index = int(selected)
                if index < 0 or index >= len(self.episodes):
                    raise EvictionContextError(context, reason=f"policy selected invalid episode index {index}")
                removed = self.episodes[index]
                removed_len = len(removed)
                del self.episodes[index]
                self.total_transitions -= removed_len
                self.total_evicted_episodes += 1
                self.total_evicted_transitions += removed_len
                self.telemetry.increment("evictions", 1)
                self.telemetry.increment("evicted_episode_count", 1)
                self.telemetry.observe("evicted_transition_count", float(removed_len))
                continue

            trim = min(overflow, max(0, len(self.current_episode) - max(1, self.min_episode_length)))
            if trim <= 0:
                raise SequenceReplayError("Sequence replay capacity exceeded but no evictable episode is available")
            del self.current_episode[:trim]
            self.total_transitions -= trim
            self.total_evicted_transitions += trim
            self.telemetry.increment("current_episode_trim_count", 1)
            self.telemetry.observe("current_episode_trimmed_transitions", float(trim))

    def _eviction_metadata_locked(self) -> Dict[str, Any]:
        rewards = []
        terminal_flags = []
        lengths = []
        for episode in self.episodes:
            lengths.append(len(episode))
            rewards.append(float(np.mean([abs(float(t[3])) for t in episode])) if episode else 0.0)
            terminal_flags.append(bool(episode[-1][5]) if episode else False)
        return {"num_episodes": len(self.episodes), "episode_lengths": lengths, "rewards": rewards, "terminal_flags": terminal_flags}

    def _candidate_windows(self, *, window: int, include_current_episode: bool) -> List[SequenceWindow]:
        sources: List[Tuple[int, List[Transition], bool]] = [(idx, ep, False) for idx, ep in enumerate(self.episodes)]
        if include_current_episode and self.current_episode:
            sources.append((len(self.episodes), self.current_episode, True))

        windows: List[SequenceWindow] = []
        for episode_index, episode, is_current in sources:
            ep_len = len(episode)
            if ep_len >= window:
                for start in range(0, ep_len - window + 1):
                    windows.append(SequenceWindow(episode_index, start, window, ep_len, is_current))
            elif self.allow_partial_sequences and ep_len >= self.min_episode_length:
                windows.append(SequenceWindow(episode_index, 0, ep_len, ep_len, is_current))
        return windows

    def _episode_for_window(self, window: SequenceWindow) -> List[Transition]:
        return self.current_episode if window.is_current_episode else self.episodes[window.episode_index]

    def _pad_sequence(self, seq: List[Transition], target_len: int) -> Tuple[List[Transition], np.ndarray]:
        if target_len <= 0:
            raise InvalidSequenceLengthError(target_len, 1)
        if len(seq) > target_len:
            raise SequencePaddingError(target_len, f"source length {len(seq)} exceeds target length")
        mask = np.zeros(target_len, dtype=np.bool_)
        mask[: len(seq)] = True
        if len(seq) == target_len:
            return seq, mask
        template = seq[-1] if seq else (None, self.pad_value, self.pad_value, 0.0, self.pad_value, True)
        pad_transition: Transition = (template[0], self.pad_value, self.pad_value, 0.0, self.pad_value, True)
        return seq + [pad_transition for _ in range(target_len - len(seq))], mask

    def _assemble_batch(self, windows: Sequence[SequenceWindow], *, window: int, sequence_length: int, burn_in: int) -> Dict[str, np.ndarray]:
        sampled: List[List[Transition]] = []
        masks: List[np.ndarray] = []
        for spec in windows:
            episode = self._episode_for_window(spec)
            raw = list(episode[spec.start_index : spec.start_index + min(window, spec.valid_length)])
            padded, mask = self._pad_sequence(raw, target_len=window)
            sampled.append(padded)
            masks.append(mask)

        mask = np.array(masks, dtype=np.bool_)
        burn_mask = np.zeros_like(mask, dtype=np.bool_)
        if burn_in > 0:
            burn_mask[:, :burn_in] = mask[:, :burn_in]
        learning_mask = np.zeros_like(mask, dtype=np.bool_)
        learning_mask[:, burn_in:] = mask[:, burn_in:]

        return {
            "agent_ids": np.array([[t[0] for t in seq] for seq in sampled], dtype=object),
            "states": np.array([[t[1] for t in seq] for seq in sampled], dtype=object),
            "actions": np.array([[t[2] for t in seq] for seq in sampled], dtype=object),
            "rewards": np.array([[t[3] for t in seq] for seq in sampled], dtype=np.float32),
            "next_states": np.array([[t[4] for t in seq] for seq in sampled], dtype=object),
            "dones": np.array([[t[5] for t in seq] for seq in sampled], dtype=np.bool_),
            "mask": mask,
            "burn_in_mask": burn_mask,
            "learning_mask": learning_mask,
            "episode_indices": np.array([w.episode_index for w in windows], dtype=np.int64),
            "start_indices": np.array([w.start_index for w in windows], dtype=np.int64),
            "valid_lengths": np.array([w.valid_length for w in windows], dtype=np.int32),
            "episode_lengths": np.array([w.episode_length for w in windows], dtype=np.int32),
            "is_current_episode": np.array([w.is_current_episode for w in windows], dtype=np.bool_),
            "burn_in": np.array(burn_in, dtype=np.int32),
            "sequence_length": np.array(sequence_length, dtype=np.int32),
            "window_length": np.array(window, dtype=np.int32),
        }

    def _validate_batch_size(self, batch_size: int) -> int:
        try:
            size = int(batch_size)
        except Exception as exc:
            raise InvalidBatchSizeError(batch_size, reason="batch_size must be integer-like") from exc
        if size <= 0:
            raise InvalidBatchSizeError(batch_size, reason="batch_size must be > 0")
        return size

    def _resolve_window(self, sequence_length: Optional[int], burn_in: Optional[int]) -> Tuple[int, int, int]:
        seq_len = int(self.sequence_length if sequence_length is None else sequence_length)
        burn = int(self.default_burn_in if burn_in is None else burn_in)
        if seq_len <= 0:
            raise InvalidSequenceLengthError(seq_len, 1)
        if burn < 0:
            raise ConfigValueError("burn_in", burn, ">= 0")
        return seq_len, burn, seq_len + burn

    def _validate_invariants_locked(self) -> None:
        actual = sum(len(ep) for ep in self.episodes) + len(self.current_episode)
        if actual != self.total_transitions:
            raise SequenceReplayError(f"Invalid sequence replay state: total_transitions={self.total_transitions}, actual={actual}")
        if self.total_transitions > self.capacity:
            self._enforce_capacity_locked()


if __name__ == "__main__":
    print("\n=== Running  Sequence Replay Buffer ===\n")
    printer.status("TEST", " Sequence Replay Buffer initialized", "info")

    cfg = {"sequence_replay": {"capacity": 32, "sequence_length": 3, "burn_in": 1, "min_episode_length": 2, "seed": 7, "persistence_enabled": False}}
    buf = SequenceReplayBuffer(user_config=cfg)
    for ep in range(3):
        for step in range(5):
            buf.push(f"agent-{ep}", {"s": step}, step, float(step), {"s": step + 1}, step == 4)
    assert len(buf) == 15
    batch = buf.sample_sequences(batch_size=4)
    assert batch["states"].shape == (4, 4)
    assert batch["mask"].all()
    report = buf.extend([("a", 1, 1, 1.0, 2, False), ("a", 2, 2, 1.0, 3, True)], fail_fast=False)
    assert report.accepted == 2
    assert buf.flush_current_episode() is False
    stats = buf.stats()
    assert stats["total_transitions"] <= stats["capacity"]
    assert stats["closed_episodes"] >= 3
    buf.clear()
    assert len(buf) == 0

    print("\n=== Test ran successfully ===\n")
