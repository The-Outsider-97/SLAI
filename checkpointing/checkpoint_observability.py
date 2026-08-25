"""Structured events, metrics, spans, and health reports for checkpointing.

The module is intentionally transport-neutral.  It never configures handlers,
creates log files, changes the root logger, or starts background threads at
import time.  Applications inject event sinks; ``LoggingEventSink`` forwards
events through the already-configured SLAI/stdlib logging pipeline using the
``event``, ``trace_id``, ``component``, and ``metadata`` fields understood by
SLAI's structured formatters.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .checkpoint_errors import *
from .checkpoint_types import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Observability")
printer = PrettyPrinter()


class CheckpointEventKind(str, Enum):
    """Stable event names emitted by the checkpoint subsystem."""

    OPERATION_STARTED = "checkpoint.operation.started"
    OPERATION_COMPLETED = "checkpoint.operation.completed"
    OPERATION_FAILED = "checkpoint.operation.failed"
    SELECTION_DECIDED = "checkpoint.selection.decided"
    ROLLBACK_PLANNED = "checkpoint.rollback.planned"
    RETENTION_PLANNED = "checkpoint.retention.planned"
    HEALTH_REPORTED = "checkpoint.health.reported"


class EventSeverity(str, Enum):
    """Transport-neutral event severity aligned with logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def _enum_string(value: str | Enum | None) -> str | None:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value) if value is not None else None


def _freeze_attributes(value: Mapping[str, Any] | None) -> Mapping[str, JSONValue]:
    frozen = freeze_json(dict(value or {}), _path="$.checkpoint_event.attributes")
    if not isinstance(frozen, Mapping):  # Defensive: input is always a mapping.
        raise TypeError("checkpoint event attributes must freeze to a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class CheckpointEvent:
    """Immutable SLAI-compatible checkpoint event."""

    kind: CheckpointEventKind
    severity: EventSeverity
    message: str
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=utc_now_iso)
    operation: str | None = None
    stage: str | None = None
    checkpoint_id: str | None = None
    version: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    success: bool | None = None
    committed: bool | None = None
    retryable: bool | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    component_count: int | None = None
    health: CheckpointHealth | None = None
    attributes: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CheckpointEventKind(self.kind))
        object.__setattr__(self, "severity", EventSeverity(self.severity))
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("checkpoint event message must be non-empty")
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(
            self, "event_id", validate_identifier(self.event_id, field_name="event_id")
        )
        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))
        for name in ("checkpoint_id", "agent_id", "run_id", "trace_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_identifier(value, field_name=name))
        if self.version is not None:
            object.__setattr__(self, "version", validate_version(self.version))
        object.__setattr__(self, "operation", _enum_string(self.operation))
        object.__setattr__(self, "stage", _enum_string(self.stage))
        for name in ("operation", "stage"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"event {name} cannot be empty")
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    validate_identifier(value, field_name=name),
                )
        operation_kinds = {
            CheckpointEventKind.OPERATION_STARTED,
            CheckpointEventKind.OPERATION_COMPLETED,
            CheckpointEventKind.OPERATION_FAILED,
        }
        if self.kind in operation_kinds and self.operation is None:
            raise ValueError("operation lifecycle events require an operation")
        if self.kind is CheckpointEventKind.OPERATION_STARTED and self.success is not None:
            raise ValueError("operation-started events cannot declare success")
        if (
            self.kind is CheckpointEventKind.OPERATION_STARTED
            and self.duration_seconds is not None
        ):
            raise ValueError("operation-started events cannot declare a duration")
        if self.kind is CheckpointEventKind.OPERATION_COMPLETED and self.success is not True:
            raise ValueError("operation-completed events must declare success=True")
        if self.kind is CheckpointEventKind.OPERATION_FAILED and self.success is not False:
            raise ValueError("operation-failed events must declare success=False")
        for name in ("success", "committed", "retryable"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"event {name} must be a boolean when supplied")
        if self.duration_seconds is not None:
            if (
                isinstance(self.duration_seconds, bool)
                or not isinstance(self.duration_seconds, (int, float))
                or not math.isfinite(self.duration_seconds)
                or self.duration_seconds < 0
            ):
                raise ValueError("event duration_seconds must be finite and non-negative")
        for name in ("size_bytes", "component_count"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"event {name} must be a non-negative integer")
        if self.health is not None:
            object.__setattr__(self, "health", CheckpointHealth(self.health))
        object.__setattr__(self, "attributes", _freeze_attributes(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "message": self.message,
            "attributes": thaw_json(self.attributes),
        }
        for name in (
            "operation",
            "stage",
            "checkpoint_id",
            "version",
            "agent_id",
            "run_id",
            "trace_id",
            "success",
            "committed",
            "retryable",
            "duration_seconds",
            "size_bytes",
            "component_count",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.health is not None:
            result["health"] = self.health.value
        return result

    def to_slai_event(self) -> dict[str, Any]:
        """Return the schema used by ``logs.standards.StandardLogEvent``."""

        payload = self.to_dict()
        payload.pop("timestamp", None)
        payload.pop("trace_id", None)
        payload.pop("kind", None)
        payload.pop("severity", None)
        return {
            "timestamp": self.timestamp,
            "agent": "checkpointing",
            "trace_id": self.trace_id or "unknown",
            "event": self.kind.value,
            "severity": self.severity.value,
            "payload": payload,
        }


@runtime_checkable
class CheckpointEventSink(Protocol):
    """Synchronous event transport contract."""

    def emit(self, event: CheckpointEvent) -> None: ...


class NullEventSink:
    """Event sink that intentionally discards every event."""

    def emit(self, event: CheckpointEvent) -> None:
        del event


class InMemoryEventSink:
    """Thread-safe bounded sink suitable for tests and local inspection."""

    def __init__(self, max_events: int = 1024) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._events: deque[CheckpointEvent] = deque(maxlen=max_events)
        self._lock = threading.RLock()

    def emit(self, event: CheckpointEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> tuple[CheckpointEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class LoggingEventSink:
    """Adapter for SLAI's explicitly configured logging pipeline.

    Constructing this sink obtains a named logger but does not add handlers or
    configure global logging.
    """

    _LEVELS = {
        EventSeverity.DEBUG: logging.DEBUG,
        EventSeverity.INFO: logging.INFO,
        EventSeverity.WARNING: logging.WARNING,
        EventSeverity.ERROR: logging.ERROR,
        EventSeverity.CRITICAL: logging.CRITICAL,
    }

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        logger_name: str = "SLAI Checkpointing",
    ) -> None:
        self.logger = logger or logging.getLogger(logger_name)

    def emit(self, event: CheckpointEvent) -> None:
        data = event.to_dict()
        self.logger.log(
            self._LEVELS[event.severity],
            event.message,
            extra={
                "event": event.kind.value,
                "trace_id": event.trace_id,
                "component": "checkpointing",
                "metadata": data,
            },
        )


@dataclass(frozen=True, slots=True)
class TelemetryDelivery:
    """Per-event delivery outcome across all configured sinks."""

    event_id: str
    attempted_sinks: int
    successful_sinks: int
    failed_sinks: int
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "event_id", validate_identifier(self.event_id, field_name="event_id")
        )
        for name in ("attempted_sinks", "successful_sinks", "failed_sinks"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.successful_sinks + self.failed_sinks != self.attempted_sinks:
            raise ValueError("telemetry delivery sink counts are inconsistent")
        object.__setattr__(self, "errors", tuple(self.errors))
        if len(self.errors) != self.failed_sinks:
            raise ValueError("telemetry delivery errors must account for failed sinks")
        if any(not isinstance(error, str) or not error for error in self.errors):
            raise ValueError("telemetry delivery errors must be non-empty strings")

    @property
    def degraded(self) -> bool:
        return self.failed_sinks > 0


@dataclass(frozen=True, slots=True)
class OperationMetricsSnapshot:
    """Immutable counters and bounded-window latency statistics for one operation."""

    operation: str
    started: int
    succeeded: int
    failed: int
    success_rate: float | None
    total_bytes: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    maximum_latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("metrics operation cannot be empty")
        object.__setattr__(self, "operation", self.operation.strip())
        for name in ("started", "succeeded", "failed", "total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"metrics {name} must be a non-negative integer")
        if self.success_rate is not None and (
            isinstance(self.success_rate, bool)
            or not isinstance(self.success_rate, (int, float))
            or not math.isfinite(self.success_rate)
            or not 0.0 <= self.success_rate <= 1.0
        ):
            raise ValueError("metrics success_rate must be finite and between zero and one")
        for name in (
            "average_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "maximum_latency_ms",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"metrics {name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CheckpointMetricsSnapshot:
    """Immutable point-in-time view of in-process checkpoint metrics."""

    timestamp: str
    total_events: int
    telemetry_failures: int
    operations: Mapping[str, OperationMetricsSnapshot]
    health_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", validate_utc_timestamp(self.timestamp))
        for name in ("total_events", "telemetry_failures"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"metrics {name} must be a non-negative integer")
        object.__setattr__(self, "operations", MappingProxyType(dict(self.operations)))
        object.__setattr__(self, "health_counts", MappingProxyType(dict(self.health_counts)))
        for name, snapshot in self.operations.items():
            if name != snapshot.operation:
                raise ValueError("operation metrics mapping key does not match snapshot")
        for health, count in self.health_counts.items():
            CheckpointHealth(health)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("health metric counts must be non-negative integers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_events": self.total_events,
            "telemetry_failures": self.telemetry_failures,
            "operations": {
                name: {
                    "started": value.started,
                    "succeeded": value.succeeded,
                    "failed": value.failed,
                    "success_rate": value.success_rate,
                    "total_bytes": value.total_bytes,
                    "average_latency_ms": value.average_latency_ms,
                    "p50_latency_ms": value.p50_latency_ms,
                    "p95_latency_ms": value.p95_latency_ms,
                    "p99_latency_ms": value.p99_latency_ms,
                    "maximum_latency_ms": value.maximum_latency_ms,
                }
                for name, value in self.operations.items()
            },
            "health_counts": dict(self.health_counts),
            "tracked_health_identities": sum(self.health_counts.values()),
        }


class _OperationAccumulator:
    __slots__ = ("started", "succeeded", "failed", "total_bytes", "latencies_ms")

    def __init__(self, window_size: int) -> None:
        self.started = 0
        self.succeeded = 0
        self.failed = 0
        self.total_bytes = 0
        self.latencies_ms: deque[float] = deque(maxlen=window_size)


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile, defined for every non-empty finite sample."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


class CheckpointMetrics:
    """Thread-safe bounded in-process checkpoint metrics aggregator.

    Latency samples and per-identity health state are bounded independently.
    Health counts therefore describe the most recently observed identities,
    not an unbounded historical inventory; durable inventory health belongs in
    ``CheckpointHealthReport``.
    """

    def __init__(
        self,
        latency_window_size: int = 512,
        health_identity_capacity: int = 4096,
    ) -> None:
        if (
            isinstance(latency_window_size, bool)
            or not isinstance(latency_window_size, int)
            or latency_window_size <= 0
        ):
            raise ValueError("latency_window_size must be positive")
        if (
            isinstance(health_identity_capacity, bool)
            or not isinstance(health_identity_capacity, int)
            or health_identity_capacity <= 0
        ):
            raise ValueError("health_identity_capacity must be positive")
        self._window_size = latency_window_size
        self._health_identity_capacity = health_identity_capacity
        self._operations: dict[str, _OperationAccumulator] = {}
        self._health_by_checkpoint: OrderedDict[
            tuple[str, str | None], CheckpointHealth
        ] = OrderedDict()
        self._total_events = 0
        self._telemetry_failures = 0
        self._lock = threading.RLock()

    def _accumulator(self, operation: str) -> _OperationAccumulator:
        accumulator = self._operations.get(operation)
        if accumulator is None:
            accumulator = _OperationAccumulator(self._window_size)
            self._operations[operation] = accumulator
        return accumulator

    def record(self, event: CheckpointEvent) -> None:
        with self._lock:
            self._total_events += 1
            if event.operation is not None:
                accumulator = self._accumulator(event.operation)
                if event.kind is CheckpointEventKind.OPERATION_STARTED:
                    accumulator.started += 1
                elif event.kind is CheckpointEventKind.OPERATION_COMPLETED:
                    accumulator.succeeded += 1
                elif event.kind is CheckpointEventKind.OPERATION_FAILED:
                    accumulator.failed += 1
                if event.size_bytes is not None and event.kind in {
                    CheckpointEventKind.OPERATION_COMPLETED,
                    CheckpointEventKind.OPERATION_FAILED,
                }:
                    accumulator.total_bytes += event.size_bytes
                if event.duration_seconds is not None:
                    accumulator.latencies_ms.append(event.duration_seconds * 1000.0)
            if event.checkpoint_id is not None and event.health is not None:
                identity = (event.checkpoint_id, event.version)
                self._health_by_checkpoint.pop(identity, None)
                self._health_by_checkpoint[identity] = event.health
                while (
                    len(self._health_by_checkpoint) > self._health_identity_capacity
                ):
                    self._health_by_checkpoint.popitem(last=False)

    def record_delivery_failures(self, count: int) -> None:
        if count < 0:
            raise ValueError("delivery failure count cannot be negative")
        with self._lock:
            self._telemetry_failures += count

    def snapshot(self) -> CheckpointMetricsSnapshot:
        with self._lock:
            operations: dict[str, OperationMetricsSnapshot] = {}
            for name, accumulator in sorted(self._operations.items()):
                completed = accumulator.succeeded + accumulator.failed
                latencies = tuple(accumulator.latencies_ms)
                operations[name] = OperationMetricsSnapshot(
                    operation=name,
                    started=accumulator.started,
                    succeeded=accumulator.succeeded,
                    failed=accumulator.failed,
                    success_rate=(
                        accumulator.succeeded / completed if completed else None
                    ),
                    total_bytes=accumulator.total_bytes,
                    average_latency_ms=(sum(latencies) / len(latencies) if latencies else 0.0),
                    p50_latency_ms=_percentile(latencies, 0.50),
                    p95_latency_ms=_percentile(latencies, 0.95),
                    p99_latency_ms=_percentile(latencies, 0.99),
                    maximum_latency_ms=max(latencies, default=0.0),
                )
            health_counts = {health.value: 0 for health in CheckpointHealth}
            for health in self._health_by_checkpoint.values():
                health_counts[health.value] += 1
            return CheckpointMetricsSnapshot(
                timestamp=utc_now_iso(),
                total_events=self._total_events,
                telemetry_failures=self._telemetry_failures,
                operations=operations,
                health_counts=health_counts,
            )

    def reset(self) -> None:
        with self._lock:
            self._operations.clear()
            self._health_by_checkpoint.clear()
            self._total_events = 0
            self._telemetry_failures = 0


@dataclass(frozen=True, slots=True)
class HealthFinding:
    """Evidence-backed issue included in a checkpoint health report."""

    code: str
    severity: EventSeverity
    message: str
    checkpoint_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not self.code.strip()
            or not isinstance(self.message, str)
            or not self.message.strip()
        ):
            raise ValueError("health finding code and message must be non-empty")
        object.__setattr__(self, "code", self.code.strip())
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(self, "severity", EventSeverity(self.severity))
        object.__setattr__(
            self,
            "checkpoint_ids",
            tuple(
                sorted(
                    {
                        validate_identifier(value, field_name="checkpoint_id")
                        for value in self.checkpoint_ids
                    }
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckpointHealthReport:
    """Inventory-level health summary derived from checkpoint records."""

    generated_at: str
    overall_health: CheckpointHealth
    checkpoint_count: int
    verified_count: int
    total_size_bytes: int
    health_counts: Mapping[str, int]
    findings: tuple[HealthFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_at", validate_utc_timestamp(self.generated_at))
        object.__setattr__(self, "overall_health", CheckpointHealth(self.overall_health))
        object.__setattr__(self, "health_counts", MappingProxyType(dict(self.health_counts)))
        object.__setattr__(self, "findings", tuple(self.findings))
        if any(not isinstance(finding, HealthFinding) for finding in self.findings):
            raise TypeError("findings must contain HealthFinding values")
        for name in ("checkpoint_count", "verified_count", "total_size_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.verified_count > self.checkpoint_count:
            raise ValueError("verified_count cannot exceed checkpoint_count")
        expected_healths = {health.value for health in CheckpointHealth}
        if set(self.health_counts) != expected_healths:
            raise ValueError("health_counts must contain every checkpoint health exactly once")
        for count in self.health_counts.values():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("health_counts values must be non-negative integers")
        if sum(self.health_counts.values()) != self.checkpoint_count:
            raise ValueError("health_counts must sum to checkpoint_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "overall_health": self.overall_health.value,
            "checkpoint_count": self.checkpoint_count,
            "verified_count": self.verified_count,
            "total_size_bytes": self.total_size_bytes,
            "health_counts": dict(self.health_counts),
            "findings": [
                {
                    "code": finding.code,
                    "severity": finding.severity.value,
                    "message": finding.message,
                    "checkpoint_ids": list(finding.checkpoint_ids),
                }
                for finding in self.findings
            ],
        }


def build_health_report(
    records: Sequence[CheckpointRecord],
) -> CheckpointHealthReport:
    """Build an evidence-only health report without probing storage."""

    health_counts = {health.value: 0 for health in CheckpointHealth}
    by_health: dict[CheckpointHealth, list[str]] = {
        health: [] for health in CheckpointHealth
    }
    evidence_by_health: dict[CheckpointHealth, list[str]] = {
        health: [] for health in CheckpointHealth
    }
    verified_count = 0
    total_size = 0
    unverified: list[str] = []
    failed_verification: list[str] = []
    health_mismatch: list[str] = []
    id_counts: dict[str, int] = {}
    version_counts: dict[str, int] = {}
    for record in records:
        id_counts[record.checkpoint_id] = id_counts.get(record.checkpoint_id, 0) + 1
        version_counts[record.version] = version_counts.get(record.version, 0) + 1
        health_counts[record.health.value] += 1
        by_health[record.health].append(record.checkpoint_id)
        evidence_by_health[record.health].append(record.checkpoint_id)
        total_size += sum(
            artifact.size_bytes for artifact in record.manifest.artifacts
        )
        if (
            record.verification is not None
            and record.verification.status is VerificationStatus.PASSED
        ):
            verified_count += 1
        else:
            unverified.append(record.checkpoint_id)
        if record.verification is not None:
            evidence_by_health[record.verification.health].append(record.checkpoint_id)
            if record.verification.status is VerificationStatus.FAILED:
                failed_verification.append(record.checkpoint_id)
            if record.verification.health is not record.health:
                health_mismatch.append(record.checkpoint_id)

    findings: list[HealthFinding] = []
    severity_by_health = {
        CheckpointHealth.QUARANTINED: EventSeverity.CRITICAL,
        CheckpointHealth.CORRUPT: EventSeverity.CRITICAL,
        CheckpointHealth.INCOMPLETE: EventSeverity.ERROR,
        CheckpointHealth.INCOMPATIBLE: EventSeverity.WARNING,
        CheckpointHealth.DEGRADED: EventSeverity.WARNING,
        CheckpointHealth.UNKNOWN: EventSeverity.WARNING,
    }
    for health, severity in severity_by_health.items():
        identifiers = tuple(sorted(by_health[health]))
        if identifiers:
            findings.append(
                HealthFinding(
                    code=f"checkpoint_{health.value}",
                    severity=severity,
                    message=f"one or more checkpoints report {health.value} health",
                    checkpoint_ids=identifiers,
                )
            )
    if unverified:
        findings.append(
            HealthFinding(
                code="verification_evidence_missing",
                severity=EventSeverity.WARNING,
                message="one or more checkpoints lack passing verification evidence",
                checkpoint_ids=tuple(sorted(unverified)),
            )
        )
    if failed_verification:
        findings.append(
            HealthFinding(
                code="verification_failed",
                severity=EventSeverity.ERROR,
                message="one or more checkpoints have failed integrity verification",
                checkpoint_ids=tuple(failed_verification),
            )
        )
    if health_mismatch:
        findings.append(
            HealthFinding(
                code="verification_health_mismatch",
                severity=EventSeverity.ERROR,
                message="record health disagrees with its verification verdict",
                checkpoint_ids=tuple(health_mismatch),
            )
        )

    duplicate_ids = tuple(
        sorted(checkpoint_id for checkpoint_id, count in id_counts.items() if count > 1)
    )
    if duplicate_ids:
        findings.append(
            HealthFinding(
                code="duplicate_checkpoint_identity",
                severity=EventSeverity.ERROR,
                message="checkpoint identities are not unique",
                checkpoint_ids=duplicate_ids,
            )
        )
    duplicate_versions = sorted(
        version for version, count in version_counts.items() if count > 1
    )
    if duplicate_versions:
        affected = tuple(
            record.checkpoint_id
            for record in records
            if record.version in duplicate_versions
        )
        findings.append(
            HealthFinding(
                code="duplicate_checkpoint_version",
                severity=EventSeverity.ERROR,
                message="checkpoint versions are not unique",
                checkpoint_ids=affected,
            )
        )

    if not records:
        overall = CheckpointHealth.UNKNOWN
        findings.append(
            HealthFinding(
                code="no_checkpoints_available",
                severity=EventSeverity.WARNING,
                message="no checkpoint records were available for health assessment",
            )
        )
    else:
        precedence = (
            CheckpointHealth.QUARANTINED,
            CheckpointHealth.CORRUPT,
            CheckpointHealth.INCOMPLETE,
            CheckpointHealth.INCOMPATIBLE,
            CheckpointHealth.DEGRADED,
            CheckpointHealth.UNKNOWN,
        )
        overall = next(
            (health for health in precedence if evidence_by_health[health]),
            CheckpointHealth.HEALTHY,
        )
        if (
            unverified
            or health_mismatch
            or duplicate_ids
            or duplicate_versions
        ) and overall is CheckpointHealth.HEALTHY:
            overall = CheckpointHealth.DEGRADED

    return CheckpointHealthReport(
        generated_at=utc_now_iso(),
        overall_health=overall,
        checkpoint_count=len(records),
        verified_count=verified_count,
        total_size_bytes=total_size,
        health_counts=health_counts,
        findings=tuple(findings),
    )

class CheckpointTelemetry:
    """Coordinates event delivery and in-process metrics without hiding failure."""

    def __init__(
        self,
        sinks: Iterable[CheckpointEventSink] = (),
        *,
        metrics: CheckpointMetrics | None = None,
        strict_delivery: bool = False,
    ) -> None:
        self._sinks = tuple(sinks)
        if any(not isinstance(sink, CheckpointEventSink) for sink in self._sinks):
            raise TypeError("every checkpoint telemetry sink must implement emit(event)")
        self.metrics = metrics or CheckpointMetrics()
        if not isinstance(self.metrics, CheckpointMetrics):
            raise TypeError("metrics must be a CheckpointMetrics instance")
        if not isinstance(strict_delivery, bool):
            raise TypeError("strict_delivery must be a boolean")
        self.strict_delivery = strict_delivery

    @property
    def sinks(self) -> tuple[CheckpointEventSink, ...]:
        return self._sinks

    def emit(
        self,
        event: CheckpointEvent,
        *,
        strict: bool | None = None,
    ) -> TelemetryDelivery:
        if not isinstance(event, CheckpointEvent):
            raise TypeError("event must be a CheckpointEvent")
        if strict is not None and not isinstance(strict, bool):
            raise TypeError("strict must be a boolean when supplied")
        self.metrics.record(event)
        succeeded = 0
        errors: list[str] = []
        for sink in self._sinks:
            try:
                sink.emit(event)
                succeeded += 1
            except Exception as exc:  # Sink failures must not stop other sinks.
                errors.append(f"{type(exc).__name__}: {exc}")
        failed = len(errors)
        if failed:
            self.metrics.record_delivery_failures(failed)
        delivery = TelemetryDelivery(
            event_id=event.event_id,
            attempted_sinks=len(self._sinks),
            successful_sinks=succeeded,
            failed_sinks=failed,
            errors=tuple(errors),
        )
        enforce = self.strict_delivery if strict is None else strict
        if failed and enforce:
            raise CheckpointObservabilityError(
                "one or more checkpoint telemetry sinks failed",
                operation=event.operation,
                stage=event.stage,
                version=event.version,
                checkpoint_id=event.checkpoint_id,
                retryable=True,
                committed=event.committed,
                details={
                    "event_id": event.event_id,
                    "failed_sinks": failed,
                    "attempted_sinks": len(self._sinks),
                    "errors": errors,
                },
            )
        return delivery

    def operation(
        self,
        operation: str | CheckpointOperation,
        *,
        stage: str | CheckpointStage | None = None,
        version: str | None = None,
        checkpoint_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> "CheckpointOperationSpan":
        return CheckpointOperationSpan(
            telemetry=self,
            operation=operation,
            stage=stage,
            version=version,
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            run_id=run_id,
            trace_id=trace_id,
            attributes=attributes,
        )

    def emit_health_report(
        self,
        report: CheckpointHealthReport,
        *,
        trace_id: str | None = None,
    ) -> TelemetryDelivery:
        severity = (
            EventSeverity.INFO
            if report.overall_health is CheckpointHealth.HEALTHY
            else EventSeverity.WARNING
            if report.overall_health
            in {
                CheckpointHealth.UNKNOWN,
                CheckpointHealth.DEGRADED,
                CheckpointHealth.INCOMPATIBLE,
            }
            else EventSeverity.ERROR
        )
        return self.emit(
            CheckpointEvent(
                kind=CheckpointEventKind.HEALTH_REPORTED,
                severity=severity,
                message="checkpoint health report generated",
                trace_id=trace_id,
                health=report.overall_health,
                attributes=report.to_dict(),
            )
        )


class CheckpointOperationSpan:
    """Context manager that emits paired start/outcome events."""

    def __init__(
        self,
        *,
        telemetry: CheckpointTelemetry,
        operation: str | CheckpointOperation,
        stage: str | CheckpointStage | None = None,
        version: str | None = None,
        checkpoint_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.telemetry = telemetry
        self.operation = _enum_string(operation)
        if self.operation is None or not self.operation.strip():
            raise ValueError("checkpoint operation cannot be empty")
        self.stage = _enum_string(stage)
        self.version = version
        self.checkpoint_id = checkpoint_id
        self.agent_id = agent_id
        self.run_id = run_id
        self.trace_id = trace_id
        self.attributes = dict(attributes or {})
        self.size_bytes: int | None = None
        self.component_count: int | None = None
        self.health: CheckpointHealth | None = None
        self.committed: bool | None = None
        self._started_at: float | None = None

    def set_result(
        self,
        *,
        checkpoint_id: str | None = None,
        size_bytes: int | None = None,
        component_count: int | None = None,
        health: CheckpointHealth | None = None,
        committed: bool | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if checkpoint_id is not None:
            self.checkpoint_id = checkpoint_id
        if size_bytes is not None:
            if size_bytes < 0:
                raise ValueError("size_bytes cannot be negative")
            self.size_bytes = size_bytes
        if component_count is not None:
            if component_count < 0:
                raise ValueError("component_count cannot be negative")
            self.component_count = component_count
        if health is not None:
            self.health = CheckpointHealth(health)
        if committed is not None:
            self.committed = committed
        if attributes:
            self.attributes.update(attributes)

    def __enter__(self) -> "CheckpointOperationSpan":
        if self._started_at is not None:
            raise RuntimeError("checkpoint operation span cannot be entered twice")
        self._started_at = time.monotonic()
        self.telemetry.emit(
            CheckpointEvent(
                kind=CheckpointEventKind.OPERATION_STARTED,
                severity=EventSeverity.INFO,
                message=f"checkpoint {self.operation} started",
                operation=self.operation,
                stage=self.stage,
                checkpoint_id=self.checkpoint_id,
                version=self.version,
                agent_id=self.agent_id,
                run_id=self.run_id,
                trace_id=self.trace_id,
                attributes=self.attributes,
            )
        )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, traceback
        if self._started_at is None:
            raise RuntimeError("checkpoint operation span was not entered")
        duration = time.monotonic() - self._started_at
        if exc is None:
            self.telemetry.emit(
                CheckpointEvent(
                    kind=CheckpointEventKind.OPERATION_COMPLETED,
                    severity=EventSeverity.INFO,
                    message=f"checkpoint {self.operation} completed",
                    operation=self.operation,
                    stage=self.stage,
                    checkpoint_id=self.checkpoint_id,
                    version=self.version,
                    agent_id=self.agent_id,
                    run_id=self.run_id,
                    trace_id=self.trace_id,
                    success=True,
                    committed=self.committed,
                    duration_seconds=duration,
                    size_bytes=self.size_bytes,
                    component_count=self.component_count,
                    health=self.health,
                    attributes=self.attributes,
                )
            )
            return False

        committed = self.committed
        retryable: bool | None = None
        error_attributes = dict(self.attributes)
        error_attributes["error_type"] = type(exc).__name__
        if isinstance(exc, CheckpointError):
            committed = exc.committed if exc.committed is not None else committed
            retryable = exc.retryable
            error_attributes["error_code"] = exc.code
        # Never let telemetry mask the checkpoint-domain exception already in
        # flight, even when strict delivery is enabled.
        try:
            self.telemetry.emit(
                CheckpointEvent(
                    kind=CheckpointEventKind.OPERATION_FAILED,
                    severity=EventSeverity.ERROR,
                    message=f"checkpoint {self.operation} failed",
                    operation=self.operation,
                    stage=self.stage,
                    checkpoint_id=self.checkpoint_id,
                    version=self.version,
                    agent_id=self.agent_id,
                    run_id=self.run_id,
                    trace_id=self.trace_id,
                    success=False,
                    committed=committed,
                    retryable=retryable,
                    duration_seconds=duration,
                    size_bytes=self.size_bytes,
                    component_count=self.component_count,
                    health=self.health,
                    attributes=error_attributes,
                ),
                strict=False,
            )
        except Exception:
            pass
        return False


__all__ = [
    "CheckpointEvent",
    "CheckpointEventKind",
    "CheckpointEventSink",
    "CheckpointHealthReport",
    "CheckpointMetrics",
    "CheckpointMetricsSnapshot",
    "CheckpointOperationSpan",
    "CheckpointTelemetry",
    "EventSeverity",
    "HealthFinding",
    "InMemoryEventSink",
    "LoggingEventSink",
    "NullEventSink",
    "OperationMetricsSnapshot",
    "TelemetryDelivery",
    "build_health_report",
]

if __name__ == "__main__":
    print("\n=== Running Checkpoint Observability Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting observability tests", "info")

    sink = InMemoryEventSink()
    telemetry = CheckpointTelemetry(sinks=[sink])
    event = CheckpointEvent(
        kind=CheckpointEventKind.OPERATION_STARTED,
        severity=EventSeverity.INFO,
        message="test event",
        operation="save",
    )
    telemetry.emit(event)
    assert len(sink.snapshot()) == 1
    # Metrics
    metrics = CheckpointMetrics()
    metrics.record(event)
    snap = metrics.snapshot()
    assert snap.total_events == 1
    printer.status("EVENTS", "event emission and metrics work", "success")

    print("\n=== All observability tests passed ===\n")