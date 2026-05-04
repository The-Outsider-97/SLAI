"""Production-grade error model for the Observability subsystem.

This module centralizes runtime failure classification for tracing, metrics,
alerting, incident synthesis, and observability memory operations.

Design goals:
- Stable, explicit error taxonomy with deterministic error codes.
- Structured payloads suitable for logs, metrics, alerts, and incident briefs.
- Low-risk serialization that prevents arbitrary context objects from breaking
  downstream sinks.
- Backward-compatible sink registration while supporting richer payloads.
- Practical normalization helpers so unexpected exceptions can still be turned
  into actionable observability failures.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


class ObservabilitySeverity(str, Enum):
    """Severity scale used across the observability subsystem."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {
            ObservabilitySeverity.LOW: 10,
            ObservabilitySeverity.MEDIUM: 20,
            ObservabilitySeverity.HIGH: 30,
            ObservabilitySeverity.CRITICAL: 40,
        }[self]

    def is_at_least(self, other: "ObservabilitySeverity") -> bool:
        return self.rank >= other.rank

    def to_incident_level(self) -> str:
        return {
            ObservabilitySeverity.LOW: "info",
            ObservabilitySeverity.MEDIUM: "warning",
            ObservabilitySeverity.HIGH: "warning",
            ObservabilitySeverity.CRITICAL: "critical",
        }[self]


class ObservabilityErrorType(str, Enum):
    """Canonical error taxonomy for observability failures and degradations."""

    UNKNOWN = "unknown"

    # Tracing
    TRACE_COLLECTION_FAILED = "trace_collection_failed"
    TRACE_CONTEXT_MISSING = "trace_context_missing"
    TRACE_SERIALIZATION_FAILED = "trace_serialization_failed"
    TRACE_CORRUPTION_DETECTED = "trace_corruption_detected"

    # Metrics and performance telemetry
    METRIC_PIPELINE_FAILED = "metric_pipeline_failed"
    METRIC_EMISSION_FAILED = "metric_emission_failed"
    METRIC_CARDINALITY_EXPLODED = "metric_cardinality_exploded"
    METRIC_BACKPRESSURE = "metric_backpressure"
    LATENCY_REGRESSION_DETECTED = "latency_regression_detected"
    THROUGHPUT_REGRESSION_DETECTED = "throughput_regression_detected"
    QUEUE_BACKLOG_GROWTH = "queue_backlog_growth"
    RESOURCE_SATURATION_DETECTED = "resource_saturation_detected"

    # Alerting and incident intelligence
    ALERT_DISPATCH_FAILED = "alert_dispatch_failed"
    ALERT_FATIGUE_DETECTED = "alert_fatigue_detected"
    ALERT_SUPPRESSION_MISCONFIGURED = "alert_suppression_misconfigured"
    INCIDENT_CLUSTERING_FAILED = "incident_clustering_failed"
    INCIDENT_CLASSIFICATION_AMBIGUOUS = "incident_classification_ambiguous"
    INCIDENT_BRIEF_GENERATION_FAILED = "incident_brief_generation_failed"
    RUNBOOK_LOOKUP_FAILED = "runbook_lookup_failed"
    RCA_GENERATION_FAILED = "rca_generation_failed"

    # Memory and contracts
    OBSERVABILITY_MEMORY_UNAVAILABLE = "observability_memory_unavailable"
    OBSERVABILITY_MEMORY_CORRUPTED = "observability_memory_corrupted"
    TELEMETRY_CONTRACT_VIOLATION = "telemetry_contract_violation"

    # Service objectives
    SLO_BREACH = "slo_breach"
    SLA_BREACH = "sla_breach"


OBSERVABILITY_ERROR_CODES: Dict[ObservabilityErrorType, str] = {
    ObservabilityErrorType.UNKNOWN: "OBS-2000",
    ObservabilityErrorType.TRACE_COLLECTION_FAILED: "OBS-2001",
    ObservabilityErrorType.TRACE_CONTEXT_MISSING: "OBS-2002",
    ObservabilityErrorType.METRIC_PIPELINE_FAILED: "OBS-2003",
    ObservabilityErrorType.METRIC_CARDINALITY_EXPLODED: "OBS-2004",
    ObservabilityErrorType.METRIC_BACKPRESSURE: "OBS-2005",
    ObservabilityErrorType.ALERT_DISPATCH_FAILED: "OBS-2006",
    ObservabilityErrorType.ALERT_FATIGUE_DETECTED: "OBS-2007",
    ObservabilityErrorType.INCIDENT_CLUSTERING_FAILED: "OBS-2008",
    ObservabilityErrorType.INCIDENT_CLASSIFICATION_AMBIGUOUS: "OBS-2009",
    ObservabilityErrorType.RUNBOOK_LOOKUP_FAILED: "OBS-2010",
    ObservabilityErrorType.RCA_GENERATION_FAILED: "OBS-2011",
    ObservabilityErrorType.OBSERVABILITY_MEMORY_UNAVAILABLE: "OBS-2012",
    ObservabilityErrorType.TELEMETRY_CONTRACT_VIOLATION: "OBS-2013",
    ObservabilityErrorType.SLO_BREACH: "OBS-2014",
    ObservabilityErrorType.TRACE_SERIALIZATION_FAILED: "OBS-2015",
    ObservabilityErrorType.TRACE_CORRUPTION_DETECTED: "OBS-2016",
    ObservabilityErrorType.METRIC_EMISSION_FAILED: "OBS-2017",
    ObservabilityErrorType.LATENCY_REGRESSION_DETECTED: "OBS-2018",
    ObservabilityErrorType.THROUGHPUT_REGRESSION_DETECTED: "OBS-2019",
    ObservabilityErrorType.QUEUE_BACKLOG_GROWTH: "OBS-2020",
    ObservabilityErrorType.RESOURCE_SATURATION_DETECTED: "OBS-2021",
    ObservabilityErrorType.INCIDENT_BRIEF_GENERATION_FAILED: "OBS-2022",
    ObservabilityErrorType.OBSERVABILITY_MEMORY_CORRUPTED: "OBS-2023",
    ObservabilityErrorType.SLA_BREACH: "OBS-2024",
    ObservabilityErrorType.ALERT_SUPPRESSION_MISCONFIGURED: "OBS-2025",
}


_AUDIT_SINK: Optional[Callable[..., None]] = None
_METRICS_SINK: Optional[Callable[..., None]] = None
_SINK_LOCK = RLock()


_STAGE_DEFAULTS: Dict[str, tuple[ObservabilityErrorType, ObservabilitySeverity, bool]] = {
    "tracing.collect": (
        ObservabilityErrorType.TRACE_COLLECTION_FAILED,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "tracing.context": (
        ObservabilityErrorType.TRACE_CONTEXT_MISSING,
        ObservabilitySeverity.HIGH,
        False,
    ),
    "tracing.serialize": (
        ObservabilityErrorType.TRACE_SERIALIZATION_FAILED,
        ObservabilitySeverity.MEDIUM,
        True,
    ),
    "tracing.integrity": (
        ObservabilityErrorType.TRACE_CORRUPTION_DETECTED,
        ObservabilitySeverity.HIGH,
        False,
    ),
    "metrics.pipeline": (
        ObservabilityErrorType.METRIC_PIPELINE_FAILED,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "metrics.emit": (
        ObservabilityErrorType.METRIC_EMISSION_FAILED,
        ObservabilitySeverity.MEDIUM,
        True,
    ),
    "metrics.cardinality": (
        ObservabilityErrorType.METRIC_CARDINALITY_EXPLODED,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "metrics.backpressure": (
        ObservabilityErrorType.METRIC_BACKPRESSURE,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "performance.latency": (
        ObservabilityErrorType.LATENCY_REGRESSION_DETECTED,
        ObservabilitySeverity.HIGH,
        False,
    ),
    "performance.throughput": (
        ObservabilityErrorType.THROUGHPUT_REGRESSION_DETECTED,
        ObservabilitySeverity.HIGH,
        False,
    ),
    "capacity.queue": (
        ObservabilityErrorType.QUEUE_BACKLOG_GROWTH,
        ObservabilitySeverity.HIGH,
        False,
    ),
    "capacity.resource": (
        ObservabilityErrorType.RESOURCE_SATURATION_DETECTED,
        ObservabilitySeverity.HIGH,
        False,
    ),
    "alerts.dispatch": (
        ObservabilityErrorType.ALERT_DISPATCH_FAILED,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "alerts.fatigue": (
        ObservabilityErrorType.ALERT_FATIGUE_DETECTED,
        ObservabilitySeverity.MEDIUM,
        False,
    ),
    "alerts.suppression": (
        ObservabilityErrorType.ALERT_SUPPRESSION_MISCONFIGURED,
        ObservabilitySeverity.HIGH,
        False,
    ),
    "incident.cluster": (
        ObservabilityErrorType.INCIDENT_CLUSTERING_FAILED,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "incident.classify": (
        ObservabilityErrorType.INCIDENT_CLASSIFICATION_AMBIGUOUS,
        ObservabilitySeverity.MEDIUM,
        True,
    ),
    "incident.brief": (
        ObservabilityErrorType.INCIDENT_BRIEF_GENERATION_FAILED,
        ObservabilitySeverity.MEDIUM,
        True,
    ),
    "incident.runbook": (
        ObservabilityErrorType.RUNBOOK_LOOKUP_FAILED,
        ObservabilitySeverity.MEDIUM,
        True,
    ),
    "incident.rca": (
        ObservabilityErrorType.RCA_GENERATION_FAILED,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "memory.read": (
        ObservabilityErrorType.OBSERVABILITY_MEMORY_UNAVAILABLE,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "memory.write": (
        ObservabilityErrorType.OBSERVABILITY_MEMORY_UNAVAILABLE,
        ObservabilitySeverity.HIGH,
        True,
    ),
    "memory.index": (
        ObservabilityErrorType.OBSERVABILITY_MEMORY_CORRUPTED,
        ObservabilitySeverity.CRITICAL,
        False,
    ),
    "contracts.validate": (
        ObservabilityErrorType.TELEMETRY_CONTRACT_VIOLATION,
        ObservabilitySeverity.HIGH,
        False,
    ),
    "slo.evaluate": (
        ObservabilityErrorType.SLO_BREACH,
        ObservabilitySeverity.CRITICAL,
        False,
    ),
    "sla.evaluate": (
        ObservabilityErrorType.SLA_BREACH,
        ObservabilitySeverity.CRITICAL,
        False,
    ),
}


def _resolve_stage_defaults(
    stage: str,
    default_error_type: Optional[ObservabilityErrorType],
    default_severity: Optional[ObservabilitySeverity],
    retryable: Optional[bool],
) -> tuple[ObservabilityErrorType, ObservabilitySeverity, bool]:
    if stage in _STAGE_DEFAULTS:
        stage_error_type, stage_severity, stage_retryable = _STAGE_DEFAULTS[stage]
    else:
        stage_error_type, stage_severity, stage_retryable = (
            ObservabilityErrorType.UNKNOWN,
            ObservabilitySeverity.MEDIUM,
            True,
        )
        for prefix, defaults in _STAGE_DEFAULTS.items():
            if stage.startswith(prefix):
                stage_error_type, stage_severity, stage_retryable = defaults
                break

    return (
        default_error_type or stage_error_type,
        default_severity or stage_severity,
        stage_retryable if retryable is None else retryable,
    )


def _truncate_text(value: str, *, limit: int = 1024) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _normalize_message_template(message: str) -> str:
    collapsed = re.sub(r"\d+", "#", message)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return _truncate_text(collapsed, limit=256)


def _safe_serialize(value: Any, *, depth: int = 0, max_depth: int = 4) -> Any:
    if depth >= max_depth:
        return "<max_depth_exceeded>"

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            return _truncate_text(value, limit=4096)
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Exception):
        return {
            "type": value.__class__.__name__,
            "message": _truncate_text(str(value), limit=1024),
        }

    if isinstance(value, Mapping):
        serialized: Dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= 100:
                serialized["__truncated__"] = True
                break
            serialized[str(k)] = _safe_serialize(v, depth=depth + 1, max_depth=max_depth)
        return serialized

    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        serialized_items = [
            _safe_serialize(item, depth=depth + 1, max_depth=max_depth)
            for item in items[:100]
        ]
        if len(items) > 100:
            serialized_items.append("<truncated>")
        return serialized_items

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _safe_serialize(value.to_dict(), depth=depth + 1, max_depth=max_depth)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return _safe_serialize(vars(value), depth=depth + 1, max_depth=max_depth)
        except Exception:
            pass

    return _truncate_text(repr(value), limit=1024)


def _sanitize_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not context:
        return {}
    return _safe_serialize(dict(context))


def _extract_traceback_excerpt(exc: Exception, *, limit: int = 4096) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _truncate_text(tb, limit=limit)


def set_observability_audit_sink(callback: Optional[Callable[..., None]]) -> None:
    """Register or clear the audit sink.

    The sink may accept either:
    - sink(payload)
    - sink(event_type="observability_error", payload=payload)
    """

    global _AUDIT_SINK
    with _SINK_LOCK:
        _AUDIT_SINK = callback


def set_observability_metrics_sink(callback: Optional[Callable[..., None]]) -> None:
    """Register or clear the metrics sink.

    Backward-compatible signatures supported during emission:
    - sink(error_code, severity, count)
    - sink(error_code, severity, count, tags)
    - sink(metric_name=..., severity=..., count=..., tags=...)
    """

    global _METRICS_SINK
    with _SINK_LOCK:
        _METRICS_SINK = callback


def clear_observability_sinks() -> None:
    with _SINK_LOCK:
        global _AUDIT_SINK, _METRICS_SINK
        _AUDIT_SINK = None
        _METRICS_SINK = None


@dataclass
class ObservabilityError(Exception):
    """Structured exception used for all observability-domain failures."""

    message: str
    error_type: ObservabilityErrorType
    severity: ObservabilitySeverity = ObservabilitySeverity.MEDIUM
    retryable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    component: str = "observability"
    operation: Optional[str] = None
    agent_name: Optional[str] = None
    trace_id: Optional[str] = None
    incident_id: Optional[str] = None
    service: Optional[str] = None
    cause: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)
        self.context = _sanitize_context(self.context)
        self.trace_id = self.trace_id or self._context_str("trace_id") or self._context_str("observability.trace_id")
        self.incident_id = self.incident_id or self._context_str("incident_id")
        self.agent_name = self.agent_name or self._context_str("agent_name")
        self.service = self.service or self._context_str("service")
        self.operation = self.operation or self._context_str("operation")
        self.component = self.component or "observability"
        self.tags = self._build_tags(self.tags)
        if self.cause is not None:
            self.cause = _truncate_text(str(self.cause), limit=2048)

    def __str__(self) -> str:
        return self.message

    def _context_str(self, key: str) -> Optional[str]:
        value = self.context.get(key)
        if value is None:
            return None
        return str(value)

    def _build_tags(self, custom_tags: Optional[Mapping[str, Any]]) -> Dict[str, str]:
        tags: Dict[str, str] = {
            "error_code": self.error_code,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "component": self.component,
            "incident_level": self.incident_level,
        }
        if self.operation:
            tags["operation"] = self.operation
        if self.agent_name:
            tags["agent_name"] = self.agent_name
        if self.service:
            tags["service"] = self.service
        if self.trace_id:
            tags["trace_id"] = self.trace_id
        if self.incident_id:
            tags["incident_id"] = self.incident_id

        for key, value in (custom_tags or {}).items():
            if value is None:
                continue
            tags[str(key)] = _truncate_text(str(value), limit=256)

        return tags

    @property
    def error_code(self) -> str:
        return OBSERVABILITY_ERROR_CODES.get(self.error_type, OBSERVABILITY_ERROR_CODES[ObservabilityErrorType.UNKNOWN])

    @property
    def timestamp_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))

    @property
    def incident_level(self) -> str:
        return self.severity.to_incident_level()

    @property
    def fingerprint(self) -> str:
        material = {
            "error_type": self.error_type.value,
            "component": self.component,
            "operation": self.operation,
            "agent_name": self.agent_name,
            "service": self.service,
            "message_template": _normalize_message_template(self.message),
            "context_keys": sorted(self.context.keys()),
        }
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:16]

    @property
    def dedupe_key(self) -> str:
        return f"{self.error_code}:{self.fingerprint}"

    @property
    def should_page(self) -> bool:
        pageworthy_types = {
            ObservabilityErrorType.SLO_BREACH,
            ObservabilityErrorType.SLA_BREACH,
            ObservabilityErrorType.RESOURCE_SATURATION_DETECTED,
            ObservabilityErrorType.QUEUE_BACKLOG_GROWTH,
            ObservabilityErrorType.OBSERVABILITY_MEMORY_CORRUPTED,
        }
        return self.severity.is_at_least(ObservabilitySeverity.HIGH) or self.error_type in pageworthy_types

    def with_context(self, **additional_context: Any) -> "ObservabilityError":
        self.context.update(_sanitize_context(additional_context))
        self.tags = self._build_tags(self.tags)
        return self

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "error_code": self.error_code,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "incident_level": self.incident_level,
            "retryable": self.retryable,
            "should_page": self.should_page,
            "message": self.message,
            "component": self.component,
            "operation": self.operation,
            "service": self.service,
            "agent_name": self.agent_name,
            "trace_id": self.trace_id,
            "incident_id": self.incident_id,
            "fingerprint": self.fingerprint,
            "dedupe_key": self.dedupe_key,
            "context": self.context,
            "remediation": self.remediation,
            "cause": self.cause,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
        }
        return {k: v for k, v in payload.items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    def report(self) -> Dict[str, Any]:
        payload = self.to_dict()
        self._emit_audit(payload)
        self._emit_metrics(payload)
        return payload

    def _emit_audit(self, payload: Dict[str, Any]) -> None:
        with _SINK_LOCK:
            sink = _AUDIT_SINK
        if sink is None:
            return
        try:
            sink(payload)
        except TypeError:
            try:
                sink(event_type="observability_error", payload=payload)
            except Exception:
                pass
        except Exception:
            pass

    def _emit_metrics(self, payload: Dict[str, Any]) -> None:
        with _SINK_LOCK:
            sink = _METRICS_SINK
        if sink is None:
            return

        tags = payload.get("tags", {})
        try:
            sink(self.error_code, self.severity.value, 1, tags)
            return
        except TypeError:
            pass
        except Exception:
            return

        try:
            sink(self.error_code, self.severity.value, 1)
            return
        except TypeError:
            pass
        except Exception:
            return

        try:
            sink(metric_name=self.error_code, severity=self.severity.value, count=1, tags=tags)
        except Exception:
            pass


class TraceCollectionError(ObservabilityError):
    def __init__(self, source: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Trace collection failed for source '{source}': {details}",
            error_type=ObservabilityErrorType.TRACE_COLLECTION_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=True,
            operation="trace_collection",
            context={"source": source, "details": details, **(context or {})},
            remediation="Validate span emitters, restore collector connectivity, and replay buffered spans.",
        )


class TraceContextMissingError(ObservabilityError):
    def __init__(self, agent_name: str, missing_fields: Iterable[str], context: Optional[Dict[str, Any]] = None):
        missing = sorted({str(field) for field in missing_fields})
        super().__init__(
            message=f"Trace context missing for '{agent_name}', required fields: {missing}",
            error_type=ObservabilityErrorType.TRACE_CONTEXT_MISSING,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            agent_name=agent_name,
            operation="trace_context_validation",
            context={"agent_name": agent_name, "missing_fields": missing, **(context or {})},
            remediation="Propagate trace context through BaseAgent hooks before executing downstream spans.",
        )


class TelemetryContractError(ObservabilityError):
    def __init__(self, agent_name: str, missing_fields: list[str], context: Optional[Dict[str, Any]] = None):
        missing = sorted({str(field) for field in missing_fields})
        super().__init__(
            message=f"Telemetry contract violation in '{agent_name}', missing fields: {missing}",
            error_type=ObservabilityErrorType.TELEMETRY_CONTRACT_VIOLATION,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            agent_name=agent_name,
            operation="telemetry_contract_validation",
            context={"agent_name": agent_name, "missing_fields": missing, **(context or {})},
            remediation="Backfill required telemetry fields, validate event schemas, and rerun the contract validator.",
        )


class ObservabilityMemoryError(ObservabilityError):
    def __init__(
        self,
        operation: str,
        details: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        backend: Optional[str] = None,
        retryable: bool = True,
        corrupted: bool = False,
    ):
        error_type = (
            ObservabilityErrorType.OBSERVABILITY_MEMORY_CORRUPTED
            if corrupted
            else ObservabilityErrorType.OBSERVABILITY_MEMORY_UNAVAILABLE
        )
        severity = ObservabilitySeverity.CRITICAL if corrupted else ObservabilitySeverity.HIGH
        remediation = (
            "Quarantine the corrupted memory partition, rebuild affected indexes, and restore from the last healthy snapshot."
            if corrupted
            else "Switch to degraded local buffering, protect write-ahead state, and trigger memory-backend recovery."
        )
        super().__init__(
            message=f"observability_memory operation '{operation}' failed: {details}",
            error_type=error_type,
            severity=severity,
            retryable=retryable,
            component="observability_memory",
            operation=operation,
            context={"operation": operation, "details": details, "backend": backend, **(context or {})},
            remediation=remediation,
        )


class SLOBreachError(ObservabilityError):
    def __init__(
        self,
        *,
        service: str,
        slo_name: str,
        observed: float,
        target: float,
        context: Optional[Dict[str, Any]] = None,
        error_budget_remaining: Optional[float] = None,
    ):
        budget_fragment = (
            f", error_budget_remaining={error_budget_remaining}"
            if error_budget_remaining is not None
            else ""
        )
        super().__init__(
            message=(
                f"SLO breach for '{service}' ({slo_name}): observed={observed}, "
                f"target={target}{budget_fragment}"
            ),
            error_type=ObservabilityErrorType.SLO_BREACH,
            severity=ObservabilitySeverity.CRITICAL,
            retryable=False,
            service=service,
            operation="slo_evaluation",
            context={
                "service": service,
                "slo_name": slo_name,
                "observed": observed,
                "target": target,
                "error_budget_remaining": error_budget_remaining,
                **(context or {}),
            },
            remediation="Trigger degraded-mode policy, page incident response, and apply the mapped recovery runbook.",
        )


class SLABreachError(ObservabilityError):
    def __init__(
        self,
        *,
        service: str,
        sla_name: str,
        observed: float,
        target: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"SLA breach for '{service}' ({sla_name}): observed={observed}, target={target}",
            error_type=ObservabilityErrorType.SLA_BREACH,
            severity=ObservabilitySeverity.CRITICAL,
            retryable=False,
            service=service,
            operation="sla_evaluation",
            context={
                "service": service,
                "sla_name": sla_name,
                "observed": observed,
                "target": target,
                **(context or {}),
            },
            remediation="Escalate incident command, notify affected stakeholders, and apply the contractual remediation workflow.",
        )


class MetricCardinalityError(ObservabilityError):
    def __init__(self, metric_name: str, cardinality: int, limit: int, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Metric '{metric_name}' exceeded cardinality limit: {cardinality}>{limit}",
            error_type=ObservabilityErrorType.METRIC_CARDINALITY_EXPLODED,
            severity=ObservabilitySeverity.HIGH,
            retryable=True,
            operation="metric_cardinality_control",
            context={
                "metric_name": metric_name,
                "cardinality": cardinality,
                "limit": limit,
                **(context or {}),
            },
            remediation="Apply bounded labels, bucket untrusted dimensions, and restart ingestion workers only after reducing key-space growth.",
        )


class MetricPipelineError(ObservabilityError):
    def __init__(self, pipeline: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Metric pipeline '{pipeline}' failed: {details}",
            error_type=ObservabilityErrorType.METRIC_PIPELINE_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=True,
            operation="metric_pipeline",
            context={"pipeline": pipeline, "details": details, **(context or {})},
            remediation="Retry the failing pipeline stage, flush partial buffers safely, and fail over to degraded aggregation if the error persists.",
        )


class MetricBackpressureError(ObservabilityError):
    def __init__(
        self,
        pipeline: str,
        queue_depth: int,
        threshold: int,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Metric backpressure detected in '{pipeline}': queue_depth={queue_depth}, threshold={threshold}",
            error_type=ObservabilityErrorType.METRIC_BACKPRESSURE,
            severity=ObservabilitySeverity.HIGH,
            retryable=True,
            operation="metric_backpressure_control",
            context={
                "pipeline": pipeline,
                "queue_depth": queue_depth,
                "threshold": threshold,
                **(context or {}),
            },
            remediation="Throttle producers, raise buffer pressure alerts, and scale ingestion or reduce sample rate until backlog stabilizes.",
        )


class AlertDispatchError(ObservabilityError):
    def __init__(self, channel: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Alert dispatch failed for channel '{channel}': {details}",
            error_type=ObservabilityErrorType.ALERT_DISPATCH_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=True,
            operation="alert_dispatch",
            context={"channel": channel, "details": details, **(context or {})},
            remediation="Retry the alert transport, switch to a backup notification channel, and verify delivery credentials and routing rules.",
        )


class IncidentClusteringError(ObservabilityError):
    def __init__(self, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Incident clustering failed: {details}",
            error_type=ObservabilityErrorType.INCIDENT_CLUSTERING_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=True,
            operation="incident_clustering",
            context={"details": details, **(context or {})},
            remediation="Fallback to coarse fingerprint clustering, validate incident features, and recompute the cluster graph.",
        )


class IncidentClassificationError(ObservabilityError):
    def __init__(self, incident_id: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Incident classification ambiguous for '{incident_id}': {details}",
            error_type=ObservabilityErrorType.INCIDENT_CLASSIFICATION_AMBIGUOUS,
            severity=ObservabilitySeverity.MEDIUM,
            retryable=True,
            incident_id=incident_id,
            operation="incident_classification",
            context={"incident_id": incident_id, "details": details, **(context or {})},
            remediation="Re-evaluate incident fingerprints, expand discriminating features, and request human confirmation when confidence remains low.",
        )


class IncidentBriefGenerationError(ObservabilityError):
    def __init__(self, incident_id: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Incident brief generation failed for '{incident_id}': {details}",
            error_type=ObservabilityErrorType.INCIDENT_BRIEF_GENERATION_FAILED,
            severity=ObservabilitySeverity.MEDIUM,
            retryable=True,
            incident_id=incident_id,
            operation="incident_brief_generation",
            context={"incident_id": incident_id, "details": details, **(context or {})},
            remediation="Fallback to template-only incident summaries and retry brief synthesis after reconstructing the correlated event timeline.",
        )


class RunbookLookupError(ObservabilityError):
    def __init__(self, playbook_id: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Runbook lookup failed for '{playbook_id}': {details}",
            error_type=ObservabilityErrorType.RUNBOOK_LOOKUP_FAILED,
            severity=ObservabilitySeverity.MEDIUM,
            retryable=True,
            operation="runbook_lookup",
            context={"playbook_id": playbook_id, "details": details, **(context or {})},
            remediation="Fallback to the default remediation set, verify runbook index integrity, and refresh the playbook cache.",
        )


class RCAGenerationError(ObservabilityError):
    def __init__(self, incident_id: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Root-cause ranking failed for '{incident_id}': {details}",
            error_type=ObservabilityErrorType.RCA_GENERATION_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=True,
            incident_id=incident_id,
            operation="root_cause_ranking",
            context={"incident_id": incident_id, "details": details, **(context or {})},
            remediation="Fallback to deterministic heuristics, rehydrate missing signals, and rerun causal ranking with validated incident inputs.",
        )


class LatencyRegressionError(ObservabilityError):
    def __init__(
        self,
        agent_name: str,
        percentile: str,
        current_ms: float,
        baseline_ms: float,
        window: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        delta_ratio = (current_ms / baseline_ms) if baseline_ms > 0 else None
        super().__init__(
            message=(
                f"Latency regression detected for '{agent_name}' at {percentile}: "
                f"current_ms={current_ms}, baseline_ms={baseline_ms}, window={window}"
            ),
            error_type=ObservabilityErrorType.LATENCY_REGRESSION_DETECTED,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            agent_name=agent_name,
            operation="latency_regression_detection",
            context={
                "agent_name": agent_name,
                "percentile": percentile,
                "current_ms": current_ms,
                "baseline_ms": baseline_ms,
                "delta_ratio": delta_ratio,
                "window": window,
                **(context or {}),
            },
            remediation="Inspect the critical path, compare recent span distributions to baseline, and apply the matching performance recovery playbook.",
        )


class ThroughputRegressionError(ObservabilityError):
    def __init__(
        self,
        service: str,
        current_rps: float,
        baseline_rps: float,
        window: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        delta_ratio = (current_rps / baseline_rps) if baseline_rps > 0 else None
        super().__init__(
            message=(
                f"Throughput regression detected for '{service}': current_rps={current_rps}, "
                f"baseline_rps={baseline_rps}, window={window}"
            ),
            error_type=ObservabilityErrorType.THROUGHPUT_REGRESSION_DETECTED,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            service=service,
            operation="throughput_regression_detection",
            context={
                "service": service,
                "current_rps": current_rps,
                "baseline_rps": baseline_rps,
                "delta_ratio": delta_ratio,
                "window": window,
                **(context or {}),
            },
            remediation="Check upstream saturation, scheduler backlog, and dependency error rates before scaling or traffic shaping.",
        )


class QueueBacklogError(ObservabilityError):
    def __init__(
        self,
        queue_name: str,
        backlog_depth: int,
        threshold: int,
        context: Optional[Dict[str, Any]] = None,
        *,
        growth_rate_per_min: Optional[float] = None,
    ):
        super().__init__(
            message=(
                f"Queue backlog growth detected for '{queue_name}': backlog_depth={backlog_depth}, "
                f"threshold={threshold}, growth_rate_per_min={growth_rate_per_min}"
            ),
            error_type=ObservabilityErrorType.QUEUE_BACKLOG_GROWTH,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            operation="queue_backlog_detection",
            context={
                "queue_name": queue_name,
                "backlog_depth": backlog_depth,
                "threshold": threshold,
                "growth_rate_per_min": growth_rate_per_min,
                **(context or {}),
            },
            remediation="Shed non-critical load, unblock stuck workers, and rebalance queue consumers before backlog reaches exhaustion thresholds.",
        )


class ResourceSaturationError(ObservabilityError):
    def __init__(
        self,
        resource_type: str,
        observed: float,
        limit: float,
        context: Optional[Dict[str, Any]] = None,
        *,
        host: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        super().__init__(
            message=(
                f"Resource saturation detected for '{resource_type}': observed={observed}, "
                f"limit={limit}, host={host}"
            ),
            error_type=ObservabilityErrorType.RESOURCE_SATURATION_DETECTED,
            severity=ObservabilitySeverity.CRITICAL if observed >= limit else ObservabilitySeverity.HIGH,
            retryable=False,
            operation="resource_saturation_detection",
            agent_name=agent_name,
            context={
                "resource_type": resource_type,
                "observed": observed,
                "limit": limit,
                "host": host,
                **(context or {}),
            },
            remediation="Throttle high-cost workflows, rebalance workloads, and scale or free constrained resources before cascading failures begin.",
        )


def normalize_observability_exception(
    exc: Exception,
    *,
    stage: str,
    context: Optional[Dict[str, Any]] = None,
    default_error_type: Optional[ObservabilityErrorType] = None,
    default_severity: Optional[ObservabilitySeverity] = None,
    retryable: Optional[bool] = None,
) -> ObservabilityError:
    """Normalize unexpected exceptions into structured observability errors.

    The helper preserves explicit ObservabilityError instances, while enriching
    them with stage/context data when missing. Non-observability exceptions are
    mapped using stage-aware defaults and a compact traceback excerpt.
    """

    if isinstance(exc, ObservabilityError):
        if context:
            exc.with_context(**context)
        if "stage" not in exc.context:
            exc.context["stage"] = stage
        if exc.operation is None:
            exc.operation = stage
            exc.tags = exc._build_tags(exc.tags)
        return exc

    resolved_type, resolved_severity, resolved_retryable = _resolve_stage_defaults(
        stage,
        default_error_type=default_error_type,
        default_severity=default_severity,
        retryable=retryable,
    )

    merged_context: Dict[str, Any] = {
        "stage": stage,
        "cause_type": exc.__class__.__name__,
        "cause_message": _truncate_text(str(exc), limit=2048),
        "traceback_excerpt": _extract_traceback_excerpt(exc),
        **(context or {}),
    }

    if resolved_type == ObservabilityErrorType.TELEMETRY_CONTRACT_VIOLATION and isinstance(exc, KeyError):
        missing_field = str(exc).strip("\"'")
        return TelemetryContractError(
            agent_name=str((context or {}).get("agent_name", "unknown_agent")),
            missing_fields=[missing_field],
            context=merged_context,
        )

    if resolved_type in {
        ObservabilityErrorType.OBSERVABILITY_MEMORY_UNAVAILABLE,
        ObservabilityErrorType.OBSERVABILITY_MEMORY_CORRUPTED,
    }:
        return ObservabilityMemoryError(
            operation=stage,
            details=str(exc),
            context=merged_context,
            retryable=resolved_retryable,
            backend=(context or {}).get("backend") if context else None,
            corrupted=(resolved_type == ObservabilityErrorType.OBSERVABILITY_MEMORY_CORRUPTED),
        )

    if resolved_type == ObservabilityErrorType.TRACE_COLLECTION_FAILED:
        return TraceCollectionError(
            source=str((context or {}).get("source", stage)),
            details=str(exc),
            context=merged_context,
        )

    if resolved_type == ObservabilityErrorType.METRIC_PIPELINE_FAILED:
        return MetricPipelineError(
            pipeline=str((context or {}).get("pipeline", stage)),
            details=str(exc),
            context=merged_context,
        )

    if resolved_type == ObservabilityErrorType.ALERT_DISPATCH_FAILED:
        return AlertDispatchError(
            channel=str((context or {}).get("channel", "default")),
            details=str(exc),
            context=merged_context,
        )

    return ObservabilityError(
        message=f"Unhandled exception during observability stage '{stage}': {exc}",
        error_type=resolved_type,
        severity=resolved_severity,
        retryable=resolved_retryable,
        operation=stage,
        context=merged_context,
        remediation=(
            "Retry the failing stage if safe, preserve partial telemetry, and notify the handler for degradation if the error repeats."
        ),
        cause=f"{exc.__class__.__name__}: {exc}",
    )


__all__ = [
    "OBSERVABILITY_ERROR_CODES",
    "AlertDispatchError",
    "IncidentBriefGenerationError",
    "IncidentClassificationError",
    "IncidentClusteringError",
    "LatencyRegressionError",
    "MetricBackpressureError",
    "MetricCardinalityError",
    "MetricPipelineError",
    "ObservabilityError",
    "ObservabilityErrorType",
    "ObservabilityMemoryError",
    "ObservabilitySeverity",
    "QueueBacklogError",
    "RCAGenerationError",
    "ResourceSaturationError",
    "RunbookLookupError",
    "SLABreachError",
    "SLOBreachError",
    "TelemetryContractError",
    "ThroughputRegressionError",
    "TraceCollectionError",
    "TraceContextMissingError",
    "clear_observability_sinks",
    "normalize_observability_exception",
    "set_observability_audit_sink",
    "set_observability_metrics_sink",
]