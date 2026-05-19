from __future__ import annotations

import time

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from .buffer_validation import *
from .buffer_telemetry import BufferTelemetry
from .buffer_persistence import BufferCheckpointIO, build_checkpoint_io
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("NStep Buffer")
printer = PrettyPrinter()


@dataclass(frozen=True)
class NStepOutput:
    """Traceable n-step output emitted from the pending transition queue."""

    transition: Transition
    window_size: int
    truncated: bool = False
    terminal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition": self.transition,
            "window_size": self.window_size,
            "truncated": self.truncated,
            "terminal": self.terminal,
        }


@dataclass
class NStepIngestReport:
    """Batch ingestion report for compact diagnostics and telemetry."""

    accepted: int = 0
    rejected: int = 0
    emitted: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "emitted": self.emitted,
            "total": self.total,
            "rejection_rate": self.rejection_rate,
            "errors": list(self.errors),
        }


class NStepBuffer:
    """Transforms validated 1-step replay transitions into n-step returns.

    Config is intentionally loaded directly in ``__init__`` through the existing
    buffer config loader. No NStepConfig wrapper is used.

    Input/output transition shape:
        (agent_id, state, action, reward, next_state, done)
    """

    def __init__(
        self,
        user_config: Optional[Mapping[str, Any]] = None,
        validator: Optional[TransitionValidator] = None,
        telemetry: Optional[BufferTelemetry] = None,
        checkpoint_io: Optional[BufferCheckpointIO] = None,
    ) -> None:
        self.config = load_global_config()
        self.nstep_cfg = dict(get_config_section("nstep") or {})
        if user_config:
            overlay = user_config.get("nstep", {}) if isinstance(user_config.get("nstep"), Mapping) else {}
            if not overlay:
                allowed = {
                    "n_step", "gamma", "clear_on_terminal", "flush_on_terminal",
                    "lock_timeout_seconds", "max_report_errors", "persistence_enabled",
                    "checkpoint_schema_version", "checkpoint_component_name",
                }
                overlay = {key: value for key, value in user_config.items() if key in allowed}
            self.nstep_cfg.update(dict(overlay))

        self.n_step = self._int_config("n_step", 3, minimum=1)
        self.gamma = self._float_config("gamma", 0.99, minimum=0.0, maximum=1.0)
        self.clear_on_terminal = bool(self.nstep_cfg.get("clear_on_terminal", True))
        self.flush_on_terminal = bool(self.nstep_cfg.get("flush_on_terminal", True))
        self.lock_timeout_seconds = self._float_config("lock_timeout_seconds", 5.0, minimum=0.0)
        self.max_report_errors = self._int_config("max_report_errors", 100, minimum=0)
        self.persistence_enabled = bool(self.nstep_cfg.get("persistence_enabled", True))
        self.checkpoint_schema_version = str(self.nstep_cfg.get("checkpoint_schema_version", "nstep_buffer.v1"))
        self.checkpoint_component_name = str(self.nstep_cfg.get("checkpoint_component_name", "nstep_buffer"))

        self.validator = validator or TransitionValidator()
        self.telemetry = telemetry or BufferTelemetry(component_name="nstep_buffer")
        self.checkpoint_io = checkpoint_io or build_checkpoint_io(user_config=user_config, telemetry=self.telemetry)

        self._queue: Deque[Transition] = deque()
        self._ready_outputs: Deque[NStepOutput] = deque()
        self._lock = RLock()
        self._stats: Dict[str, float] = {
            "accepted": 0.0,
            "rejected": 0.0,
            "emitted": 0.0,
            "terminal_flushes": 0.0,
            "manual_flushes": 0.0,
            "clears": 0.0,
        }

    # ------------------------------------------------------------------
    # Config and telemetry helpers
    # ------------------------------------------------------------------
    def _int_config(self, key: str, default: int, *, minimum: Optional[int] = None) -> int:
        value = int(self.nstep_cfg.get(key, default))
        if minimum is not None and value < minimum:
            raise ConfigValueError(f"nstep.{key}", value, f">= {minimum}")
        return value

    def _float_config(
        self,
        key: str,
        default: float,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        value = float(self.nstep_cfg.get(key, default))
        if minimum is not None and value < minimum:
            raise ConfigValueError(f"nstep.{key}", value, f">= {minimum}")
        if maximum is not None and value > maximum:
            raise ConfigValueError(f"nstep.{key}", value, f"<= {maximum}")
        return value

    def _observe(self, name: str, value: float) -> None:
        if hasattr(self.telemetry, "observe"):
            self.telemetry.observe(name, float(value))

    def _increment(self, name: str, amount: float = 1.0) -> None:
        if hasattr(self.telemetry, "increment"):
            self.telemetry.increment(name, float(amount))

    def _record_rejection(self, operation: str, reason: str) -> None:
        self._stats["rejected"] += 1.0
        if hasattr(self.telemetry, "record_rejection"):
            self.telemetry.record_rejection(operation, reason)
        else:
            self._increment(f"{operation}_rejection_count", 1.0)
            self._increment("rejection_count", 1.0)

    @contextmanager
    def _locked(self, operation: str) -> Iterator[None]:
        start = time.perf_counter()
        acquired = self._lock.acquire(timeout=self.lock_timeout_seconds) if self.lock_timeout_seconds > 0 else self._lock.acquire(blocking=False)
        waited = time.perf_counter() - start
        self._observe("lock_wait_seconds", waited)
        if hasattr(self.telemetry, "record_lock_contention"):
            self.telemetry.record_lock_contention(operation, waited, acquired=acquired)
        if not acquired:
            cls = globals().get("BufferLockTimeoutError")
            raise cls(operation=operation, timeout_seconds=self.lock_timeout_seconds) if cls else NStepBufferError(
                f"Timed out acquiring n-step buffer lock for {operation} after {self.lock_timeout_seconds:.3f}s"
            )
        try:
            yield
        finally:
            self._lock.release()

    # ------------------------------------------------------------------
    # Core transformation
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        with self._locked("len"):
            return len(self._queue)

    def _build_nstep_transition(self, window: Sequence[Transition], *, truncated: bool = False) -> NStepOutput:
        if not window:
            cls = globals().get("NStepWindowError")
            raise cls(0, self.n_step, "empty window") if cls else NStepBufferError("Invalid n-step window: empty window")

        try:
            first = window[0]
            agent_id, state, action = first[0], first[1], first[2]
            discounted_reward = 0.0
            terminal = False
            final_next_state = window[-1][4]

            for idx, transition in enumerate(window):
                discounted_reward += (self.gamma ** idx) * float(transition[3])
                final_next_state = transition[4]
                terminal = bool(transition[5])
                if terminal:
                    truncated = idx + 1 < self.n_step
                    break

            output: Transition = (agent_id, state, action, float(discounted_reward), final_next_state, bool(terminal))
            return NStepOutput(output, window_size=len(window), truncated=bool(truncated), terminal=bool(terminal))
        except BufferError:
            raise
        except Exception as exc:
            cls = globals().get("NStepComputationError")
            raise cls(str(exc)) if cls else NStepBufferError(f"N-step computation failed: {exc}")

    def _ready(self) -> bool:
        return bool(self._queue) and (len(self._queue) >= self.n_step or bool(self._queue[-1][5]))

    def _materialize_ready_locked(self) -> None:
        while self._ready():
            terminal_at_tail = bool(self._queue[-1][5])
            window_size = min(self.n_step, len(self._queue))
            truncated = window_size < self.n_step
            window = [self._queue[i] for i in range(window_size)]
            output = self._build_nstep_transition(window, truncated=truncated)
            self._ready_outputs.append(output)
            self._queue.popleft()
            self._stats["emitted"] += 1.0
            self._increment("nstep_emit_count", 1.0)

            if terminal_at_tail and self.flush_on_terminal:
                while self._queue:
                    window_size = min(self.n_step, len(self._queue))
                    window = [self._queue[i] for i in range(window_size)]
                    self._ready_outputs.append(self._build_nstep_transition(window, truncated=True))
                    self._queue.popleft()
                    self._stats["emitted"] += 1.0
                    self._increment("nstep_emit_count", 1.0)
                self._stats["terminal_flushes"] += 1.0
                break

            if output.terminal and self.clear_on_terminal:
                self._queue.clear()
                break

    def add(self, transition: Sequence[Any]) -> Optional[Transition]:
        """Append one transition and return the next available n-step output, if any."""
        with self.telemetry.time_block("push_latency_seconds"):
            with self._locked("add"):
                try:
                    normalized = self.validator.sanitize_transition(tuple(transition))
                    self._queue.append(normalized)
                    self._stats["accepted"] += 1.0
                    self._increment("push_count", 1.0)
                    self._materialize_ready_locked()
                    ready = self._ready_outputs.popleft() if self._ready_outputs else None
                    return ready.transition if ready else None
                except BufferError:
                    self._record_rejection("nstep_add", "buffer_error")
                    raise
                except Exception as exc:
                    self._record_rejection("nstep_add", type(exc).__name__)
                    raise NStepBufferError(f"Failed to add transition to n-step buffer: {exc}") from exc

    def add_components(self, agent_id: Any, state: Any, action: Any, reward: Any, next_state: Any, done: Any) -> Optional[Transition]:
        return self.add((agent_id, state, action, reward, next_state, done))

    def extend(self, transitions: Iterable[Sequence[Any]], *, fail_fast: bool = True) -> NStepIngestReport:
        report = NStepIngestReport()
        for idx, transition in enumerate(transitions):
            try:
                before = len(self._ready_outputs)
                result = self.add(transition)
                report.accepted += 1
                report.emitted += 1 if result is not None else 0
                report.emitted += max(0, len(self._ready_outputs) - before)
            except Exception as exc:
                report.rejected += 1
                if len(report.errors) < self.max_report_errors:
                    report.errors.append(f"index={idx}: {type(exc).__name__}: {exc}")
                if fail_fast:
                    raise
        return report

    def drain_ready(self, max_items: Optional[int] = None) -> List[Transition]:
        with self._locked("drain_ready"):
            limit = len(self._ready_outputs) if max_items is None else max(0, int(max_items))
            outputs: List[Transition] = []
            while self._ready_outputs and len(outputs) < limit:
                outputs.append(self._ready_outputs.popleft().transition)
            return outputs

    def flush(self) -> List[Transition]:
        """Flush all ready and truncated n-step outputs from the queue."""
        with self.telemetry.time_block("flush_latency_seconds"):
            with self._locked("flush"):
                outputs = [item.transition for item in self._ready_outputs]
                self._ready_outputs.clear()
                while self._queue:
                    window_size = min(self.n_step, len(self._queue))
                    window = [self._queue[i] for i in range(window_size)]
                    outputs.append(self._build_nstep_transition(window, truncated=window_size < self.n_step).transition)
                    self._queue.popleft()
                    self._stats["emitted"] += 1.0
                self._stats["manual_flushes"] += 1.0
                self._increment("nstep_flush_count", 1.0)
                return outputs

    def clear(self) -> None:
        with self._locked("clear"):
            self._queue.clear()
            self._ready_outputs.clear()
            self._stats["clears"] += 1.0

    # ------------------------------------------------------------------
    # Persistence and diagnostics
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        with self._locked("state_dict"):
            return {
                "nstep_cfg": dict(self.nstep_cfg),
                "n_step": self.n_step,
                "gamma": self.gamma,
                "clear_on_terminal": self.clear_on_terminal,
                "flush_on_terminal": self.flush_on_terminal,
                "queue": list(self._queue),
                "ready_outputs": [item.to_dict() for item in self._ready_outputs],
                "stats": dict(self._stats),
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with self._locked("load_state_dict"):
            self._queue = deque(tuple(item) for item in state.get("queue", []))
            self._ready_outputs = deque(
                NStepOutput(tuple(item["transition"]), int(item.get("window_size", self.n_step)), bool(item.get("truncated", False)), bool(item.get("terminal", False)))
                for item in state.get("ready_outputs", [])
            )
            self._stats.update({str(k): float(v) for k, v in dict(state.get("stats", {})).items()})

    def save(self, filepath: str) -> str:
        if not self.persistence_enabled:
            raise NStepBufferError("N-step persistence is disabled")
        path = self.checkpoint_io.save_checkpoint(
            self.state_dict(),
            filepath,
            component_name=self.checkpoint_component_name,
            schema_version=self.checkpoint_schema_version,
            metadata={"n_step": self.n_step, "gamma": self.gamma},
            telemetry=self.telemetry,
        )
        return str(path)

    def load(self, filepath: str) -> None:
        checkpoint = self.checkpoint_io.load_checkpoint(
            filepath,
            expected_component=self.checkpoint_component_name,
            telemetry=self.telemetry,
        )
        self.load_state_dict(checkpoint.state)

    def stats(self) -> Dict[str, Any]:
        with self._locked("stats"):
            return {
                "pending": len(self._queue),
                "ready_outputs": len(self._ready_outputs),
                "n_step": self.n_step,
                "gamma": self.gamma,
                "clear_on_terminal": self.clear_on_terminal,
                "flush_on_terminal": self.flush_on_terminal,
                **dict(self._stats),
            }

    def snapshot(self) -> Dict[str, Any]:
        return self.stats()


__all__ = ["NStepOutput", "NStepIngestReport", "NStepBuffer"]


if __name__ == "__main__":
    print("\n=== Running  NStep Buffer ===\n")
    printer.status("TEST", " NStep Buffer initialized", "info")

    cfg = {"nstep": {"n_step": 3, "gamma": 0.9, "clear_on_terminal": True, "flush_on_terminal": True}}
    buf = NStepBuffer(user_config=cfg)
    assert buf.add_components("a", "s0", 0, 1.0, "s1", False) is None
    assert buf.add_components("a", "s1", 1, 2.0, "s2", False) is None
    out = buf.add_components("a", "s2", 2, 3.0, "s3", False)
    assert out == ("a", "s0", 0, 1.0 + 0.9 * 2.0 + 0.81 * 3.0, "s3", False)
    terminal = buf.add_components("a", "s3", 3, 4.0, "s4", True)
    tail = buf.drain_ready()
    assert terminal is not None and tail and len(buf) == 0
    report = buf.extend([("b", i, i, 1.0, i + 1, i == 2) for i in range(3)], fail_fast=True)
    assert report.accepted == 3
    flushed = buf.flush()
    assert flushed
    stats = buf.stats()
    assert stats["emitted"] >= 4
    buf.clear()
    assert len(buf) == 0

    print("\n=== Test ran successfully ===\n")
