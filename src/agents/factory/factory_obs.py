"""Factory observability helpers.

This module provides a lightweight, dependency-free observability layer for
factory internals. It supports:

- monotonic counters for factory events and orchestration outcomes;
- gauges for current runtime values such as registry size or cache size;
- timings with aggregate duration statistics;
- a bounded recent-event log for diagnostics;
- structured snapshots that can be consumed by tests, health checks, logs,
  dashboards, or factory orchestration reports.

The goal is production-friendly visibility without introducing heavyweight
telemetry dependencies in core factory paths. The implementation stays
in-memory, thread-safe, and intentionally independent of concrete agent classes.
"""

from __future__ import annotations

import json

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Deque, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.factory_errors import *
from .utils.factory_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Factory Observability")
printer = PrettyPrinter()


@dataclass(slots=True)
class TimingStats:
    """Aggregated statistics for a named timed operation."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0

    def add(self, duration_ms: float) -> None:
        """Record one duration sample."""
        duration = validate_timing_duration(duration_ms)
        self.count += 1
        self.total_ms += duration
        self.last_ms = duration
        if self.count == 1:
            self.min_ms = duration
            self.max_ms = duration
            return
        if duration < self.min_ms:
            self.min_ms = duration
        if duration > self.max_ms:
            self.max_ms = duration

    def merge(self, other: "TimingStats") -> None:
        """Merge another timing aggregate into this aggregate."""
        if other.count <= 0:
            return
        if self.count <= 0:
            self.count = other.count
            self.total_ms = other.total_ms
            self.min_ms = other.min_ms
            self.max_ms = other.max_ms
            self.last_ms = other.last_ms
            return
        self.count += other.count
        self.total_ms += other.total_ms
        self.min_ms = min(self.min_ms, other.min_ms)
        self.max_ms = max(self.max_ms, other.max_ms)
        self.last_ms = other.last_ms

    def as_dict(self) -> Dict[str, float]:
        """Return a JSON-friendly timing summary."""
        avg_ms = (self.total_ms / self.count) if self.count else 0.0
        return {
            "count": float(self.count),
            "total_ms": self.total_ms,
            "avg_ms": avg_ms,
            "min_ms": self.min_ms if self.count else 0.0,
            "max_ms": self.max_ms if self.count else 0.0,
            "last_ms": self.last_ms if self.count else 0.0,
        }


@dataclass(slots=True)
class EventRecord:
    """Bounded diagnostic event stored by ``FactoryObservability``."""

    sequence: int
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=now_epoch_seconds)
    monotonic_ms: float = field(default_factory=monotonic_ms)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "ts": self.ts,
            "monotonic_ms": self.monotonic_ms,
            "event_type": self.event_type,
            "payload": normalize_payload(self.payload),
        }


class FactoryObservability:
    """In-memory observability collector for factory orchestration internals.

    Configuration is loaded from ``factory_config.yaml`` through the existing
    factory config loader. Constructor arguments are optional runtime overrides;
    when omitted, the values from ``factory_obs`` are used.
    """

    def __init__(self, event_buffer_size: Optional[int] = None, *,
        enabled: Optional[bool] = None,
        record_events: Optional[bool] = None,
        record_timings: Optional[bool] = None,
        record_counters: Optional[bool] = None,
        record_gauges: Optional[bool] = None,
        include_events_in_snapshot: Optional[bool] = None,
    ) -> None:
        self.config = load_global_config()
        self.obs_config = get_config_section("factory_obs")

        configured_buffer_size = self.obs_config.get("event_buffer_size", 500)
        self.event_buffer_size = validate_event_buffer_size(
            configured_buffer_size if event_buffer_size is None else event_buffer_size
        )

        self.enabled = bool(self.obs_config.get("enabled", True) if enabled is None else enabled)
        self.record_events_enabled = bool(self.obs_config.get("record_events", True) if record_events is None else record_events)
        self.record_timings_enabled = bool(self.obs_config.get("record_timings", True) if record_timings is None else record_timings)
        self.record_counters_enabled = bool(self.obs_config.get("record_counters", True) if record_counters is None else record_counters)
        self.record_gauges_enabled = bool(self.obs_config.get("record_gauges", True) if record_gauges is None else record_gauges)
        self.include_events_in_snapshot = bool(
            self.obs_config.get("include_events_in_snapshot", False)
            if include_events_in_snapshot is None
            else include_events_in_snapshot
        )
        self.auto_counter_events = bool(self.obs_config.get("auto_counter_events", False))
        self.auto_gauge_events = bool(self.obs_config.get("auto_gauge_events", False))
        self.auto_timing_events = bool(self.obs_config.get("auto_timing_events", False))

        self.created_at = now_epoch_seconds()
        self._last_updated_at = self.created_at
        self._lock = RLock()
        self._counters: DefaultDict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._timings: DefaultDict[str, TimingStats] = defaultdict(TimingStats)
        self._events: Deque[EventRecord] = deque(maxlen=self.event_buffer_size)
        self._sequence = 0

        logger.info("Factory Observability successfully initialized")

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _touch(self) -> None:
        self._last_updated_at = now_epoch_seconds()

    def _event_unlocked(self, event_type: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        """Record an event while the caller already holds ``self._lock``."""
        if not self.enabled or not self.record_events_enabled:
            return
        name, safe_payload = validate_event_payload(event_type, payload)
        self._events.append(EventRecord(sequence=self._next_sequence(), event_type=name, payload=safe_payload))
        self._touch()

    def inc(self, name: str, value: int = 1) -> int:
        """Increment a named monotonic counter and return the updated value."""
        counter_name = validate_observability_name(name, field_name="counter_name")
        increment = validate_counter_increment(value)
        with self._lock:
            if not self.enabled or not self.record_counters_enabled:
                return self._counters.get(counter_name, 0)
            self._counters[counter_name] += increment
            updated = self._counters[counter_name]
            self._touch()
            if self.auto_counter_events:
                self._event_unlocked("counter.incremented", {"name": counter_name, "value": increment, "current": updated})
            return updated

    def increment_many(self, counters: Mapping[str, int]) -> Dict[str, int]:
        """Increment several counters atomically."""
        payload = normalize_payload(counters)
        updates: Dict[str, int] = {}
        with self._lock:
            for raw_name, raw_value in payload.items():
                counter_name = validate_observability_name(raw_name, field_name="counter_name")
                increment = validate_counter_increment(raw_value)
                if self.enabled and self.record_counters_enabled:
                    self._counters[counter_name] += increment
                updates[counter_name] = self._counters.get(counter_name, 0)
            self._touch()
            return updates

    def set_gauge(self, name: str, value: float) -> None:
        """Set a named gauge to its latest numeric value."""
        gauge_name = validate_observability_name(name, field_name="gauge_name")
        gauge_value = validate_gauge_value(value)
        with self._lock:
            if not self.enabled or not self.record_gauges_enabled:
                return
            self._gauges[gauge_name] = gauge_value
            self._touch()
            if self.auto_gauge_events:
                self._event_unlocked("gauge.updated", {"name": gauge_name, "value": gauge_value})

    def set_gauges(self, gauges: Mapping[str, float]) -> None:
        """Set several gauges atomically."""
        payload = normalize_payload(gauges)
        with self._lock:
            for raw_name, raw_value in payload.items():
                gauge_name = validate_observability_name(raw_name, field_name="gauge_name")
                gauge_value = validate_gauge_value(raw_value)
                if self.enabled and self.record_gauges_enabled:
                    self._gauges[gauge_name] = gauge_value
            self._touch()

    def observe_timing(self, name: str, duration_ms: float) -> None:
        """Record a measured operation duration in milliseconds."""
        timing_name = validate_observability_name(name, field_name="timing_name")
        duration = validate_timing_duration(duration_ms)
        with self._lock:
            if not self.enabled or not self.record_timings_enabled:
                return
            self._timings[timing_name].add(duration)
            self._touch()
            if self.auto_timing_events:
                self._event_unlocked("timing.observed", {"name": timing_name, "duration_ms": duration})

    @contextmanager
    def timer(self, name: str, *, record_event: bool = False, payload: Optional[Mapping[str, Any]] = None) -> Iterator[None]:
        """Context manager for measuring operation duration.

        If the wrapped block raises, the original exception is preserved. The
        duration is still observed, and an optional ``timer.failed`` event is
        recorded for diagnostics.
        """
        timing_name = validate_observability_name(name, field_name="timing_name")
        start_ms = monotonic_ms()
        failed = False
        error_payload: Dict[str, Any] = {}
        try:
            yield
        except Exception as exc:
            failed = True
            error_payload = {"error_type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            elapsed_ms = monotonic_ms() - start_ms
            self.observe_timing(timing_name, elapsed_ms)
            if record_event:
                event_type = "timer.failed" if failed else "timer.completed"
                event_payload = normalize_payload(payload)
                event_payload.update({"name": timing_name, "duration_ms": elapsed_ms})
                if failed:
                    event_payload.update(error_payload)
                self.record_event(event_type, event_payload)

    def record_event(self, event_type: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        """Append a timestamped diagnostic event into the bounded event buffer."""
        name, safe_payload = validate_event_payload(event_type, payload)
        with self._lock:
            if not self.enabled or not self.record_events_enabled:
                return
            self._events.append(EventRecord(sequence=self._next_sequence(), event_type=name, payload=safe_payload))
            self._touch()

    def record_error(self, error: FactoryError, *, event_type: str = "factory.error") -> None:
        """Record a factory error as an observability event."""
        if not isinstance(error, FactoryError):
            raise EventRecordingError("record_error expects a FactoryError instance", context={"actual_type": type(error).__name__})
        self.record_event(
            event_type,
            {
                "code": error.code,
                "error_type": error.error_type.value,
                "message": error.message,
                "severity": error.severity,
                "retryable": error.retryable,
                "component": error.component,
                "operation": error.operation,
            },
        )

    def get_counter(self, name: str, default: int = 0) -> int:
        counter_name = validate_observability_name(name, field_name="counter_name")
        with self._lock:
            return self._counters.get(counter_name, default)

    def get_gauge(self, name: str, default: Optional[float] = None) -> Optional[float]:
        gauge_name = validate_observability_name(name, field_name="gauge_name")
        with self._lock:
            return self._gauges.get(gauge_name, default)

    def get_timing(self, name: str) -> Optional[Dict[str, float]]:
        timing_name = validate_observability_name(name, field_name="timing_name")
        with self._lock:
            stats = self._timings.get(timing_name)
            return None if stats is None else stats.as_dict()

    def get_recent_events(self, limit: int = 50, *, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        limit_value = validate_recent_event_limit(limit)
        if limit_value == 0:
            return []
        normalized_event_type = validate_observability_name(event_type, field_name="event_type") if event_type else None
        with self._lock:
            events: Iterable[EventRecord] = list(self._events)
            if normalized_event_type is not None:
                events = [event for event in events if event.event_type == normalized_event_type]
            return [event.as_dict() for event in list(events)[-limit_value:]]

    def counter_names(self) -> List[str]:
        with self._lock:
            return sorted(self._counters.keys())

    def gauge_names(self) -> List[str]:
        with self._lock:
            return sorted(self._gauges.keys())

    def timing_names(self) -> List[str]:
        with self._lock:
            return sorted(self._timings.keys())

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def clear_events(self) -> None:
        with self._lock:
            self._events.clear()
            self._touch()

    def clear_counters(self) -> None:
        with self._lock:
            self._counters.clear()
            self._touch()

    def clear_gauges(self) -> None:
        with self._lock:
            self._gauges.clear()
            self._touch()

    def clear_timings(self) -> None:
        with self._lock:
            self._timings.clear()
            self._touch()

    def snapshot(
        self,
        *,
        include_events: Optional[bool] = None,
        event_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a point-in-time snapshot of all collected observability data."""
        try:
            include_event_records = self.include_events_in_snapshot if include_events is None else bool(include_events)
            limit = validate_recent_event_limit(event_limit if event_limit is not None else self.obs_config.get("snapshot_event_limit", 50))
            with self._lock:
                snapshot: Dict[str, Any] = {
                    "enabled": self.enabled,
                    "created_at": self.created_at,
                    "last_updated_at": self._last_updated_at,
                    "uptime_seconds": max(0.0, now_epoch_seconds() - self.created_at),
                    "counters": dict(self._counters),
                    "gauges": dict(self._gauges),
                    "timings": {name: stats.as_dict() for name, stats in self._timings.items()},
                    "event_count": len(self._events),
                    "event_buffer_size": self.event_buffer_size,
                    "sequence": self._sequence,
                }
                if include_event_records:
                    snapshot["events"] = [event.as_dict() for event in list(self._events)[-limit:]] if limit else []
                return snapshot
        except FactoryError:
            raise
        except Exception as exc:
            raise observability_snapshot_failed(cause=exc) from exc

    def to_json(self, *, include_events: Optional[bool] = None, indent: Optional[int] = 2) -> str:
        """Return a JSON snapshot string."""
        return json.dumps(self.snapshot(include_events=include_events), indent=indent, sort_keys=True, default=str)

    def health(self) -> Dict[str, Any]:
        """Return a compact health payload for factory diagnostics."""
        with self._lock:
            return {
                "status": "ok" if self.enabled else "disabled",
                "event_buffer_size": self.event_buffer_size,
                "event_count": len(self._events),
                "counter_count": len(self._counters),
                "gauge_count": len(self._gauges),
                "timing_count": len(self._timings),
                "uptime_seconds": max(0.0, now_epoch_seconds() - self.created_at),
            }

    def reset(self) -> None:
        """Clear all in-memory observability state."""
        try:
            with self._lock:
                self._counters.clear()
                self._gauges.clear()
                self._timings.clear()
                self._events.clear()
                self._sequence = 0
                self._last_updated_at = now_epoch_seconds()
        except Exception as exc:
            raise observability_reset_failed(cause=exc) from exc


_default_observability: Optional[FactoryObservability] = None
_default_lock = RLock()


def get_factory_observability() -> FactoryObservability:
    """Return a process-local default observability collector."""
    global _default_observability
    with _default_lock:
        if _default_observability is None:
            _default_observability = FactoryObservability()
        return _default_observability


def reset_factory_observability() -> FactoryObservability:
    """Reset and return the process-local default observability collector."""
    global _default_observability
    with _default_lock:
        _default_observability = FactoryObservability()
        return _default_observability


if __name__ == "__main__":
    print("\n=== Running Factory Observability ===\n")
    printer.status("TEST", "Factory Observability initialized", "info")

    obs = FactoryObservability(event_buffer_size=10, include_events_in_snapshot=True)
    assert obs.inc("registry.registered", 2) == 2
    assert obs.inc("registry.registered") == 3
    obs.set_gauge("registry.size", 3)
    obs.observe_timing("registry.resolve", 12.5)

    with obs.timer("factory.bootstrap", record_event=True, payload={"phase": "test"}):
        _ = sum(range(10))

    obs.record_event("agent.created", {"agent": "adaptive", "version": "1.0.0"})
    snapshot = obs.snapshot(include_events=True)

    assert snapshot["counters"]["registry.registered"] == 3
    assert snapshot["gauges"]["registry.size"] == 3.0
    assert snapshot["timings"]["registry.resolve"]["count"] == 1.0
    assert snapshot["event_count"] >= 2
    assert obs.get_counter("registry.registered") == 3
    assert obs.get_gauge("registry.size") == 3.0
    assert obs.get_recent_events(limit=2)

    health = obs.health()
    assert health["status"] == "ok"
    assert health["event_buffer_size"] == 10

    obs.reset()
    assert obs.snapshot()["event_count"] == 0
    assert obs.get_counter("registry.registered") == 0

    print("\n=== Test ran successfully ===\n")
