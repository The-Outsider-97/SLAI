from __future__ import annotations

import hashlib
import json
import socket
import time
import traceback
import uuid

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence


class QualitySeverity(str, Enum):
    """Severity level used for policy escalation and operational triage."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {
            QualitySeverity.LOW: 10,
            QualitySeverity.MEDIUM: 20,
            QualitySeverity.HIGH: 30,
            QualitySeverity.CRITICAL: 40,
        }[self]

    def at_least(self, other: "QualitySeverity") -> bool:
        return self.rank >= other.rank


class QualityDomain(str, Enum):
    """Owning subsystem for a quality failure."""

    STRUCTURAL = "structural"
    STATISTICAL = "statistical"
    SEMANTIC = "semantic"
    WORKFLOW = "workflow"
    MEMORY = "memory"
    SYSTEM = "system"


class QualityStage(str, Enum):
    """Execution stage where the quality issue was observed."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    BASELINE = "baseline"
    PROFILING = "profiling"
    SCORING = "scoring"
    QUARANTINE = "quarantine"
    REMEDIATION = "remediation"
    PERSISTENCE = "persistence"
    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    UNKNOWN = "unknown"


class QualityDisposition(str, Enum):
    """Suggested operator or pipeline action after a failure."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    RETRY = "retry"
    ESCALATE = "escalate"
    FALLBACK = "fallback"


class QualityErrorType(str, Enum):
    # Structural quality
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    SCHEMA_VERSION_MISMATCH = "schema_version_mismatch"
    REQUIRED_FIELD_MISSING = "required_field_missing"
    TYPE_COERCION_FAILED = "type_coercion_failed"
    FIELD_DOMAIN_VIOLATION = "field_domain_violation"
    FIELD_RANGE_VIOLATION = "field_range_violation"

    # Statistical quality
    DISTRIBUTION_DRIFT_DETECTED = "distribution_drift_detected"
    CONCEPT_DRIFT_DETECTED = "concept_drift_detected"
    MISSINGNESS_RATE_EXCEEDED = "missingness_rate_exceeded"
    OUTLIER_RATE_EXCEEDED = "outlier_rate_exceeded"
    DUPLICATE_RATE_EXCEEDED = "duplicate_rate_exceeded"
    BASELINE_NOT_FOUND = "baseline_not_found"

    # Semantic quality
    LEAKAGE_DETECTED = "leakage_detected"
    INCONSISTENT_LABELS = "inconsistent_labels"
    CROSS_FIELD_CONFLICT = "cross_field_conflict"
    PROVENANCE_UNTRUSTED = "provenance_untrusted"
    PROVENANCE_MISSING = "provenance_missing"
    PRIVACY_POLICY_CONFLICT = "privacy_policy_conflict"

    # Workflow / control plane
    QUARANTINE_OPERATION_FAILED = "quarantine_operation_failed"
    QUARANTINE_STORAGE_UNAVAILABLE = "quarantine_storage_unavailable"
    ROUTING_FAILED = "routing_failed"
    REMEDIATION_FAILED = "remediation_failed"
    REMEDIATION_POLICY_MISSING = "remediation_policy_missing"

    # Memory / persistence
    QUALITY_MEMORY_UNAVAILABLE = "quality_memory_unavailable"
    QUALITY_MEMORY_CONFLICT = "quality_memory_conflict"

    # Policy / scoring / system
    POLICY_THRESHOLD_INVALID = "policy_threshold_invalid"
    CONFIGURATION_INVALID = "configuration_invalid"
    SCORING_PIPELINE_FAILED = "scoring_pipeline_failed"
    AUDIT_EMISSION_FAILED = "audit_emission_failed"
    METRICS_EMISSION_FAILED = "metrics_emission_failed"
    INTERNAL_QUALITY_AGENT_FAILURE = "internal_quality_agent_failure"


QUALITY_ERROR_CODES: Dict[QualityErrorType, str] = {
    # Structural: 1000 series
    QualityErrorType.SCHEMA_VALIDATION_FAILED: "DQA-1001",
    QualityErrorType.SCHEMA_VERSION_MISMATCH: "DQA-1002",
    QualityErrorType.REQUIRED_FIELD_MISSING: "DQA-1003",
    QualityErrorType.TYPE_COERCION_FAILED: "DQA-1004",
    QualityErrorType.FIELD_DOMAIN_VIOLATION: "DQA-1005",
    QualityErrorType.FIELD_RANGE_VIOLATION: "DQA-1006",

    # Statistical: 1100 series
    QualityErrorType.DISTRIBUTION_DRIFT_DETECTED: "DQA-1101",
    QualityErrorType.CONCEPT_DRIFT_DETECTED: "DQA-1102",
    QualityErrorType.MISSINGNESS_RATE_EXCEEDED: "DQA-1103",
    QualityErrorType.OUTLIER_RATE_EXCEEDED: "DQA-1104",
    QualityErrorType.DUPLICATE_RATE_EXCEEDED: "DQA-1105",
    QualityErrorType.BASELINE_NOT_FOUND: "DQA-1106",

    # Semantic: 1200 series
    QualityErrorType.LEAKAGE_DETECTED: "DQA-1201",
    QualityErrorType.INCONSISTENT_LABELS: "DQA-1202",
    QualityErrorType.CROSS_FIELD_CONFLICT: "DQA-1203",
    QualityErrorType.PROVENANCE_UNTRUSTED: "DQA-1204",
    QualityErrorType.PROVENANCE_MISSING: "DQA-1205",
    QualityErrorType.PRIVACY_POLICY_CONFLICT: "DQA-1206",

    # Workflow: 1300 series
    QualityErrorType.QUARANTINE_OPERATION_FAILED: "DQA-1301",
    QualityErrorType.QUARANTINE_STORAGE_UNAVAILABLE: "DQA-1302",
    QualityErrorType.ROUTING_FAILED: "DQA-1303",
    QualityErrorType.REMEDIATION_FAILED: "DQA-1304",
    QualityErrorType.REMEDIATION_POLICY_MISSING: "DQA-1305",

    # Memory: 1400 series
    QualityErrorType.QUALITY_MEMORY_UNAVAILABLE: "DQA-1401",
    QualityErrorType.QUALITY_MEMORY_CONFLICT: "DQA-1402",

    # Policy / system: 1500 series
    QualityErrorType.POLICY_THRESHOLD_INVALID: "DQA-1501",
    QualityErrorType.CONFIGURATION_INVALID: "DQA-1502",
    QualityErrorType.SCORING_PIPELINE_FAILED: "DQA-1503",
    QualityErrorType.AUDIT_EMISSION_FAILED: "DQA-1504",
    QualityErrorType.METRICS_EMISSION_FAILED: "DQA-1505",
    QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE: "DQA-1599",
}


ERROR_TYPE_TO_DOMAIN: Dict[QualityErrorType, QualityDomain] = {
    QualityErrorType.SCHEMA_VALIDATION_FAILED: QualityDomain.STRUCTURAL,
    QualityErrorType.SCHEMA_VERSION_MISMATCH: QualityDomain.STRUCTURAL,
    QualityErrorType.REQUIRED_FIELD_MISSING: QualityDomain.STRUCTURAL,
    QualityErrorType.TYPE_COERCION_FAILED: QualityDomain.STRUCTURAL,
    QualityErrorType.FIELD_DOMAIN_VIOLATION: QualityDomain.STRUCTURAL,
    QualityErrorType.FIELD_RANGE_VIOLATION: QualityDomain.STRUCTURAL,
    QualityErrorType.DISTRIBUTION_DRIFT_DETECTED: QualityDomain.STATISTICAL,
    QualityErrorType.CONCEPT_DRIFT_DETECTED: QualityDomain.STATISTICAL,
    QualityErrorType.MISSINGNESS_RATE_EXCEEDED: QualityDomain.STATISTICAL,
    QualityErrorType.OUTLIER_RATE_EXCEEDED: QualityDomain.STATISTICAL,
    QualityErrorType.DUPLICATE_RATE_EXCEEDED: QualityDomain.STATISTICAL,
    QualityErrorType.BASELINE_NOT_FOUND: QualityDomain.STATISTICAL,
    QualityErrorType.LEAKAGE_DETECTED: QualityDomain.SEMANTIC,
    QualityErrorType.INCONSISTENT_LABELS: QualityDomain.SEMANTIC,
    QualityErrorType.CROSS_FIELD_CONFLICT: QualityDomain.SEMANTIC,
    QualityErrorType.PROVENANCE_UNTRUSTED: QualityDomain.SEMANTIC,
    QualityErrorType.PROVENANCE_MISSING: QualityDomain.SEMANTIC,
    QualityErrorType.PRIVACY_POLICY_CONFLICT: QualityDomain.SEMANTIC,
    QualityErrorType.QUARANTINE_OPERATION_FAILED: QualityDomain.WORKFLOW,
    QualityErrorType.QUARANTINE_STORAGE_UNAVAILABLE: QualityDomain.WORKFLOW,
    QualityErrorType.ROUTING_FAILED: QualityDomain.WORKFLOW,
    QualityErrorType.REMEDIATION_FAILED: QualityDomain.WORKFLOW,
    QualityErrorType.REMEDIATION_POLICY_MISSING: QualityDomain.WORKFLOW,
    QualityErrorType.QUALITY_MEMORY_UNAVAILABLE: QualityDomain.MEMORY,
    QualityErrorType.QUALITY_MEMORY_CONFLICT: QualityDomain.MEMORY,
    QualityErrorType.POLICY_THRESHOLD_INVALID: QualityDomain.SYSTEM,
    QualityErrorType.CONFIGURATION_INVALID: QualityDomain.SYSTEM,
    QualityErrorType.SCORING_PIPELINE_FAILED: QualityDomain.SYSTEM,
    QualityErrorType.AUDIT_EMISSION_FAILED: QualityDomain.SYSTEM,
    QualityErrorType.METRICS_EMISSION_FAILED: QualityDomain.SYSTEM,
    QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE: QualityDomain.SYSTEM,
}


DEFAULT_SEVERITY_BY_TYPE: Dict[QualityErrorType, QualitySeverity] = {
    QualityErrorType.SCHEMA_VALIDATION_FAILED: QualitySeverity.HIGH,
    QualityErrorType.SCHEMA_VERSION_MISMATCH: QualitySeverity.HIGH,
    QualityErrorType.REQUIRED_FIELD_MISSING: QualitySeverity.HIGH,
    QualityErrorType.TYPE_COERCION_FAILED: QualitySeverity.MEDIUM,
    QualityErrorType.FIELD_DOMAIN_VIOLATION: QualitySeverity.MEDIUM,
    QualityErrorType.FIELD_RANGE_VIOLATION: QualitySeverity.MEDIUM,
    QualityErrorType.DISTRIBUTION_DRIFT_DETECTED: QualitySeverity.HIGH,
    QualityErrorType.CONCEPT_DRIFT_DETECTED: QualitySeverity.HIGH,
    QualityErrorType.MISSINGNESS_RATE_EXCEEDED: QualitySeverity.MEDIUM,
    QualityErrorType.OUTLIER_RATE_EXCEEDED: QualitySeverity.MEDIUM,
    QualityErrorType.DUPLICATE_RATE_EXCEEDED: QualitySeverity.MEDIUM,
    QualityErrorType.BASELINE_NOT_FOUND: QualitySeverity.MEDIUM,
    QualityErrorType.LEAKAGE_DETECTED: QualitySeverity.CRITICAL,
    QualityErrorType.INCONSISTENT_LABELS: QualitySeverity.HIGH,
    QualityErrorType.CROSS_FIELD_CONFLICT: QualitySeverity.HIGH,
    QualityErrorType.PROVENANCE_UNTRUSTED: QualitySeverity.CRITICAL,
    QualityErrorType.PROVENANCE_MISSING: QualitySeverity.HIGH,
    QualityErrorType.PRIVACY_POLICY_CONFLICT: QualitySeverity.CRITICAL,
    QualityErrorType.QUARANTINE_OPERATION_FAILED: QualitySeverity.HIGH,
    QualityErrorType.QUARANTINE_STORAGE_UNAVAILABLE: QualitySeverity.CRITICAL,
    QualityErrorType.ROUTING_FAILED: QualitySeverity.HIGH,
    QualityErrorType.REMEDIATION_FAILED: QualitySeverity.HIGH,
    QualityErrorType.REMEDIATION_POLICY_MISSING: QualitySeverity.HIGH,
    QualityErrorType.QUALITY_MEMORY_UNAVAILABLE: QualitySeverity.CRITICAL,
    QualityErrorType.QUALITY_MEMORY_CONFLICT: QualitySeverity.HIGH,
    QualityErrorType.POLICY_THRESHOLD_INVALID: QualitySeverity.HIGH,
    QualityErrorType.CONFIGURATION_INVALID: QualitySeverity.HIGH,
    QualityErrorType.SCORING_PIPELINE_FAILED: QualitySeverity.HIGH,
    QualityErrorType.AUDIT_EMISSION_FAILED: QualitySeverity.MEDIUM,
    QualityErrorType.METRICS_EMISSION_FAILED: QualitySeverity.MEDIUM,
    QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE: QualitySeverity.CRITICAL,
}


DEFAULT_RETRYABLE_BY_TYPE: Dict[QualityErrorType, bool] = {
    QualityErrorType.SCHEMA_VALIDATION_FAILED: False,
    QualityErrorType.SCHEMA_VERSION_MISMATCH: False,
    QualityErrorType.REQUIRED_FIELD_MISSING: False,
    QualityErrorType.TYPE_COERCION_FAILED: False,
    QualityErrorType.FIELD_DOMAIN_VIOLATION: False,
    QualityErrorType.FIELD_RANGE_VIOLATION: False,
    QualityErrorType.DISTRIBUTION_DRIFT_DETECTED: False,
    QualityErrorType.CONCEPT_DRIFT_DETECTED: False,
    QualityErrorType.MISSINGNESS_RATE_EXCEEDED: False,
    QualityErrorType.OUTLIER_RATE_EXCEEDED: False,
    QualityErrorType.DUPLICATE_RATE_EXCEEDED: False,
    QualityErrorType.BASELINE_NOT_FOUND: True,
    QualityErrorType.LEAKAGE_DETECTED: False,
    QualityErrorType.INCONSISTENT_LABELS: False,
    QualityErrorType.CROSS_FIELD_CONFLICT: False,
    QualityErrorType.PROVENANCE_UNTRUSTED: False,
    QualityErrorType.PROVENANCE_MISSING: False,
    QualityErrorType.PRIVACY_POLICY_CONFLICT: False,
    QualityErrorType.QUARANTINE_OPERATION_FAILED: True,
    QualityErrorType.QUARANTINE_STORAGE_UNAVAILABLE: True,
    QualityErrorType.ROUTING_FAILED: True,
    QualityErrorType.REMEDIATION_FAILED: True,
    QualityErrorType.REMEDIATION_POLICY_MISSING: False,
    QualityErrorType.QUALITY_MEMORY_UNAVAILABLE: True,
    QualityErrorType.QUALITY_MEMORY_CONFLICT: True,
    QualityErrorType.POLICY_THRESHOLD_INVALID: False,
    QualityErrorType.CONFIGURATION_INVALID: False,
    QualityErrorType.SCORING_PIPELINE_FAILED: True,
    QualityErrorType.AUDIT_EMISSION_FAILED: True,
    QualityErrorType.METRICS_EMISSION_FAILED: True,
    QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE: True,
}


DEFAULT_DISPOSITION_BY_TYPE: Dict[QualityErrorType, QualityDisposition] = {
    QualityErrorType.SCHEMA_VALIDATION_FAILED: QualityDisposition.BLOCK,
    QualityErrorType.SCHEMA_VERSION_MISMATCH: QualityDisposition.BLOCK,
    QualityErrorType.REQUIRED_FIELD_MISSING: QualityDisposition.BLOCK,
    QualityErrorType.TYPE_COERCION_FAILED: QualityDisposition.WARN,
    QualityErrorType.FIELD_DOMAIN_VIOLATION: QualityDisposition.WARN,
    QualityErrorType.FIELD_RANGE_VIOLATION: QualityDisposition.WARN,
    QualityErrorType.DISTRIBUTION_DRIFT_DETECTED: QualityDisposition.QUARANTINE,
    QualityErrorType.CONCEPT_DRIFT_DETECTED: QualityDisposition.QUARANTINE,
    QualityErrorType.MISSINGNESS_RATE_EXCEEDED: QualityDisposition.WARN,
    QualityErrorType.OUTLIER_RATE_EXCEEDED: QualityDisposition.WARN,
    QualityErrorType.DUPLICATE_RATE_EXCEEDED: QualityDisposition.WARN,
    QualityErrorType.BASELINE_NOT_FOUND: QualityDisposition.FALLBACK,
    QualityErrorType.LEAKAGE_DETECTED: QualityDisposition.BLOCK,
    QualityErrorType.INCONSISTENT_LABELS: QualityDisposition.QUARANTINE,
    QualityErrorType.CROSS_FIELD_CONFLICT: QualityDisposition.QUARANTINE,
    QualityErrorType.PROVENANCE_UNTRUSTED: QualityDisposition.BLOCK,
    QualityErrorType.PROVENANCE_MISSING: QualityDisposition.BLOCK,
    QualityErrorType.PRIVACY_POLICY_CONFLICT: QualityDisposition.BLOCK,
    QualityErrorType.QUARANTINE_OPERATION_FAILED: QualityDisposition.ESCALATE,
    QualityErrorType.QUARANTINE_STORAGE_UNAVAILABLE: QualityDisposition.ESCALATE,
    QualityErrorType.ROUTING_FAILED: QualityDisposition.RETRY,
    QualityErrorType.REMEDIATION_FAILED: QualityDisposition.RETRY,
    QualityErrorType.REMEDIATION_POLICY_MISSING: QualityDisposition.ESCALATE,
    QualityErrorType.QUALITY_MEMORY_UNAVAILABLE: QualityDisposition.FALLBACK,
    QualityErrorType.QUALITY_MEMORY_CONFLICT: QualityDisposition.ESCALATE,
    QualityErrorType.POLICY_THRESHOLD_INVALID: QualityDisposition.ESCALATE,
    QualityErrorType.CONFIGURATION_INVALID: QualityDisposition.ESCALATE,
    QualityErrorType.SCORING_PIPELINE_FAILED: QualityDisposition.RETRY,
    QualityErrorType.AUDIT_EMISSION_FAILED: QualityDisposition.WARN,
    QualityErrorType.METRICS_EMISSION_FAILED: QualityDisposition.WARN,
    QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE: QualityDisposition.ESCALATE,
}


_AUDIT_SINKS: List[Callable[[Dict[str, Any]], None]] = []
_METRICS_SINKS: List[Callable[[str, str, int], None]] = []
_SINKS_LOCK = RLock()
_MAX_TEXT_LENGTH = 4096


def _coerce_enum(value: Any, enum_cls: type[Enum], default: Enum) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
    return default


def _truncate(value: str, limit: int = _MAX_TEXT_LENGTH) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _safe_serialize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
        }

    if isinstance(value, Mapping):
        return {str(k): _safe_serialize(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_serialize(item) for item in value]

    return _truncate(repr(value))


def _normalized_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    serialized = _safe_serialize(dict(context or {}))
    return serialized if isinstance(serialized, dict) else {"value": serialized}


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    stable_payload = {
        "error_type": payload.get("error_type"),
        "stage": payload.get("stage"),
        "domain": payload.get("domain"),
        "dataset_id": payload.get("dataset_id"),
        "source_id": payload.get("source_id"),
        "batch_id": payload.get("batch_id"),
        "record_id": payload.get("record_id"),
        "rule_id": payload.get("rule_id"),
        "message": payload.get("message"),
        "context": payload.get("context"),
    }
    encoded = json.dumps(stable_payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _default_domain(error_type: QualityErrorType) -> QualityDomain:
    return ERROR_TYPE_TO_DOMAIN.get(error_type, QualityDomain.SYSTEM)


def _default_severity(error_type: QualityErrorType) -> QualitySeverity:
    return DEFAULT_SEVERITY_BY_TYPE.get(error_type, QualitySeverity.MEDIUM)


def _default_retryable(error_type: QualityErrorType) -> bool:
    return DEFAULT_RETRYABLE_BY_TYPE.get(error_type, False)


def _default_disposition(error_type: QualityErrorType) -> QualityDisposition:
    return DEFAULT_DISPOSITION_BY_TYPE.get(error_type, QualityDisposition.WARN)


def set_quality_audit_sink(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Replace the registered audit sink with a single callback."""
    if not callable(callback):
        raise TypeError("callback must be callable")
    with _SINKS_LOCK:
        _AUDIT_SINKS.clear()
        _AUDIT_SINKS.append(callback)


def add_quality_audit_sink(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Register an additional structured audit sink callback."""
    if not callable(callback):
        raise TypeError("callback must be callable")
    with _SINKS_LOCK:
        _AUDIT_SINKS.append(callback)


def clear_quality_audit_sinks() -> None:
    with _SINKS_LOCK:
        _AUDIT_SINKS.clear()


def set_quality_metrics_sink(callback: Callable[[str, str, int], None]) -> None:
    """Replace the registered metrics sink with a single callback.

    Expected signature: (error_code, severity, count).
    """
    if not callable(callback):
        raise TypeError("callback must be callable")
    with _SINKS_LOCK:
        _METRICS_SINKS.clear()
        _METRICS_SINKS.append(callback)


def add_quality_metrics_sink(callback: Callable[[str, str, int], None]) -> None:
    """Register an additional metrics sink callback."""
    if not callable(callback):
        raise TypeError("callback must be callable")
    with _SINKS_LOCK:
        _METRICS_SINKS.append(callback)


def clear_quality_metrics_sinks() -> None:
    with _SINKS_LOCK:
        _METRICS_SINKS.clear()


@dataclass
class DataQualityError(Exception):
    """Canonical structured exception for the Data Quality Agent.

    The object is intentionally rich enough to support:
    - operational logging and metrics emission,
    - policy routing (warn / quarantine / block),
    - batch-level incident debugging,
    - safe JSON serialization for audit storage and shared memory.
    """

    message: str
    error_type: QualityErrorType
    severity: QualitySeverity = QualitySeverity.MEDIUM
    retryable: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    stage: QualityStage = QualityStage.UNKNOWN
    domain: Optional[QualityDomain] = None
    disposition: Optional[QualityDisposition] = None
    dataset_id: Optional[str] = None
    source_id: Optional[str] = None
    batch_id: Optional[str] = None
    record_id: Optional[str] = None
    rule_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cause: Optional[BaseException] = None
    host: str = field(default_factory=socket.gethostname)

    def __post_init__(self) -> None:
        self.message = _truncate(str(self.message).strip())
        if not self.message:
            raise ValueError("DataQualityError.message must not be empty")

        self.error_type = _coerce_enum(
            self.error_type,
            QualityErrorType,
            QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE,
        )
        self.severity = _coerce_enum(
            self.severity,
            QualitySeverity,
            _default_severity(self.error_type),
        )
        self.stage = _coerce_enum(self.stage, QualityStage, QualityStage.UNKNOWN)
        self.domain = _coerce_enum(
            self.domain,
            QualityDomain,
            _default_domain(self.error_type),
        )
        self.disposition = _coerce_enum(
            self.disposition,
            QualityDisposition,
            _default_disposition(self.error_type),
        )
        self.retryable = bool(self.retryable)
        self.context = _normalized_context(self.context)
        self.tags = {str(k): str(v) for k, v in (self.tags or {}).items()}

        identifiers = {
            "dataset_id": self.dataset_id or self.context.get("dataset_id"),
            "source_id": self.source_id or self.context.get("source_id"),
            "batch_id": self.batch_id or self.context.get("batch_id"),
            "record_id": self.record_id or self.context.get("record_id"),
            "rule_id": self.rule_id or self.context.get("rule_id"),
        }
        self.dataset_id = identifiers["dataset_id"]
        self.source_id = identifiers["source_id"]
        self.batch_id = identifiers["batch_id"]
        self.record_id = identifiers["record_id"]
        self.rule_id = identifiers["rule_id"]
        self.context.update({k: v for k, v in identifiers.items() if v is not None})

        if self.cause is not None and "cause" not in self.context:
            self.context["cause"] = _safe_serialize(self.cause)

        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return self.message

    @property
    def error_code(self) -> str:
        return QUALITY_ERROR_CODES.get(self.error_type, "DQA-9999")

    @property
    def is_critical(self) -> bool:
        return self.severity is QualitySeverity.CRITICAL

    @property
    def should_block(self) -> bool:
        return self.disposition is QualityDisposition.BLOCK or self.is_critical

    @property
    def should_quarantine(self) -> bool:
        return self.disposition is QualityDisposition.QUARANTINE

    @property
    def timestamp_iso(self) -> str:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()

    @property
    def fingerprint(self) -> str:
        return _fingerprint_payload(self.to_dict(include_traceback=False))

    def add_context(self, **context: Any) -> "DataQualityError":
        self.context.update(_normalized_context(context))
        if self.dataset_id is None:
            self.dataset_id = self.context.get("dataset_id")
        if self.source_id is None:
            self.source_id = self.context.get("source_id")
        if self.batch_id is None:
            self.batch_id = self.context.get("batch_id")
        if self.record_id is None:
            self.record_id = self.context.get("record_id")
        if self.rule_id is None:
            self.rule_id = self.context.get("rule_id")
        return self

    def with_updates(self, **changes: Any) -> "DataQualityError":
        updated = replace(self, **changes)
        return updated

    def to_dict(self, *, include_traceback: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error_code": self.error_code,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "retryable": self.retryable,
            "message": self.message,
            "remediation": self.remediation,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "stage": self.stage.value,
            "domain": self.domain.value if self.domain else None,
            "disposition": self.disposition.value if self.disposition else None,
            "dataset_id": self.dataset_id,
            "source_id": self.source_id,
            "batch_id": self.batch_id,
            "record_id": self.record_id,
            "rule_id": self.rule_id,
            "tags": self.tags,
            "context": self.context,
            "correlation_id": self.correlation_id,
            "host": self.host,
        }
        payload["fingerprint"] = _fingerprint_payload(payload)
        if include_traceback and self.cause is not None:
            payload["traceback"] = _truncate(
                "".join(
                    traceback.format_exception(
                        type(self.cause),
                        self.cause,
                        self.cause.__traceback__,
                    )
                )
            )
        return payload

    def to_json(self, *, include_traceback: bool = False) -> str:
        return json.dumps(self.to_dict(include_traceback=include_traceback), sort_keys=True)

    def report(self, *, raise_on_sink_failure: bool = False) -> Dict[str, bool]:
        """Emit the error to all registered sinks.

        Returns a small status map that callers may inspect when making fallback
        decisions.
        """
        payload = self.to_dict(include_traceback=False)
        audit_ok = True
        metrics_ok = True

        with _SINKS_LOCK:
            audit_sinks = list(_AUDIT_SINKS)
            metrics_sinks = list(_METRICS_SINKS)

        for sink in audit_sinks:
            try:
                sink(payload)
            except Exception as sink_exc:
                audit_ok = False
                if raise_on_sink_failure:
                    raise DataQualityError(
                        message=f"Audit sink emission failed: {sink_exc}",
                        error_type=QualityErrorType.AUDIT_EMISSION_FAILED,
                        severity=QualitySeverity.MEDIUM,
                        retryable=True,
                        stage=self.stage,
                        domain=QualityDomain.SYSTEM,
                        disposition=QualityDisposition.WARN,
                        context={
                            "original_error_code": self.error_code,
                            "correlation_id": self.correlation_id,
                        },
                        remediation="Fail open for audit emission, persist local fallback log, and retry asynchronously.",
                        cause=sink_exc,
                    ) from sink_exc

        for sink in metrics_sinks:
            try:
                sink(self.error_code, self.severity.value, 1)
            except Exception as sink_exc:
                metrics_ok = False
                if raise_on_sink_failure:
                    raise DataQualityError(
                        message=f"Metrics sink emission failed: {sink_exc}",
                        error_type=QualityErrorType.METRICS_EMISSION_FAILED,
                        severity=QualitySeverity.MEDIUM,
                        retryable=True,
                        stage=self.stage,
                        domain=QualityDomain.SYSTEM,
                        disposition=QualityDisposition.WARN,
                        context={
                            "original_error_code": self.error_code,
                            "correlation_id": self.correlation_id,
                        },
                        remediation="Fail open for metrics emission and retry through buffered exporter.",
                        cause=sink_exc,
                    ) from sink_exc

        return {"audit": audit_ok, "metrics": metrics_ok}

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        *,
        message: Optional[str] = None,
        error_type: QualityErrorType = QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE,
        severity: Optional[QualitySeverity] = None,
        retryable: Optional[bool] = None,
        stage: QualityStage = QualityStage.UNKNOWN,
        context: Optional[Mapping[str, Any]] = None,
        remediation: Optional[str] = None,
        disposition: Optional[QualityDisposition] = None,
    ) -> "DataQualityError":
        return cls(
            message=message or str(exc) or f"Unhandled exception during quality stage '{stage.value}'",
            error_type=error_type,
            severity=severity or _default_severity(error_type),
            retryable=_default_retryable(error_type) if retryable is None else retryable,
            stage=stage,
            context={**dict(context or {}), "exception_type": type(exc).__name__},
            remediation=remediation,
            disposition=disposition,
            cause=exc,
        )


@dataclass
class DataQualityErrorGroup(DataQualityError):
    """Aggregate error representing multiple quality failures for one batch or source."""

    errors: Sequence[DataQualityError] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.errors:
            highest = max((err.severity for err in self.errors), key=lambda sev: sev.rank)
            self.severity = highest
            self.retryable = any(err.retryable for err in self.errors)
            if self.disposition is None:
                self.disposition = max(
                    (err.disposition for err in self.errors if err.disposition is not None),
                    key=lambda disp: {
                        QualityDisposition.PASS: 0,
                        QualityDisposition.WARN: 1,
                        QualityDisposition.RETRY: 2,
                        QualityDisposition.FALLBACK: 3,
                        QualityDisposition.QUARANTINE: 4,
                        QualityDisposition.BLOCK: 5,
                        QualityDisposition.ESCALATE: 6,
                    }[disp],
                    default=QualityDisposition.WARN,
                )
            self.context = {
                **dict(self.context or {}),
                "error_count": len(self.errors),
                "children": [
                    {
                        "error_code": err.error_code,
                        "error_type": err.error_type.value,
                        "severity": err.severity.value,
                        "message": err.message,
                        "fingerprint": err.fingerprint,
                    }
                    for err in self.errors
                ],
            }
        super().__post_init__()


class SchemaValidationError(DataQualityError):
    def __init__(self, dataset_id: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Schema validation failed for dataset '{dataset_id}': {details}",
            error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.VALIDATION,
            disposition=QualityDisposition.BLOCK,
            dataset_id=dataset_id,
            context={"dataset_id": dataset_id, "details": details, **(context or {})},
            remediation="Review required fields, types, and schema version before allowing ingest.",
        )


class RequiredFieldError(DataQualityError):
    def __init__(
        self,
        *,
        dataset_id: str,
        field_name: str,
        record_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Required field '{field_name}' missing in dataset '{dataset_id}'",
            error_type=QualityErrorType.REQUIRED_FIELD_MISSING,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.VALIDATION,
            disposition=QualityDisposition.BLOCK,
            dataset_id=dataset_id,
            record_id=record_id,
            context={
                "dataset_id": dataset_id,
                "field_name": field_name,
                "record_id": record_id,
                **(context or {}),
            },
            remediation="Populate the missing field or drop the invalid record before downstream use.",
        )


class ThresholdConfigurationError(DataQualityError):
    def __init__(
        self,
        *,
        rule_id: str,
        details: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Invalid policy threshold for rule '{rule_id}': {details}",
            error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.SCORING,
            disposition=QualityDisposition.ESCALATE,
            rule_id=rule_id,
            context={"rule_id": rule_id, "details": details, **(context or {})},
            remediation="Correct the configured threshold values and rerun validation before scoring traffic.",
        )


class QualityMemoryError(DataQualityError):
    def __init__(self, operation: str, details: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"quality_memory operation '{operation}' failed: {details}",
            error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
            severity=QualitySeverity.CRITICAL,
            retryable=True,
            stage=QualityStage.PERSISTENCE,
            disposition=QualityDisposition.FALLBACK,
            context={"operation": operation, "details": details, **(context or {})},
            remediation="Fail over to cached snapshot, mark the batch as partially evaluated, and enqueue memory repair.",
        )


class DriftThresholdError(DataQualityError):
    def __init__(
        self,
        *,
        source_id: str,
        metric: str,
        observed: float,
        threshold: float,
        batch_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=(
                f"Drift threshold exceeded for source '{source_id}' on metric '{metric}': "
                f"observed={observed}, threshold={threshold}"
            ),
            error_type=QualityErrorType.DISTRIBUTION_DRIFT_DETECTED,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.PROFILING,
            disposition=QualityDisposition.QUARANTINE,
            source_id=source_id,
            batch_id=batch_id,
            context={
                "source_id": source_id,
                "metric": metric,
                "observed": observed,
                "threshold": threshold,
                "batch_id": batch_id,
                **(context or {}),
            },
            remediation="Quarantine the affected shard, compare against the last trusted baseline, and re-fetch if the drift is source-induced.",
        )


class ProvenanceTrustError(DataQualityError):
    def __init__(self, source_id: str, trust_score: float, minimum_trust: float, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=(
                f"Source '{source_id}' failed provenance trust check: "
                f"trust_score={trust_score}, minimum={minimum_trust}"
            ),
            error_type=QualityErrorType.PROVENANCE_UNTRUSTED,
            severity=QualitySeverity.CRITICAL,
            retryable=False,
            stage=QualityStage.VALIDATION,
            disposition=QualityDisposition.BLOCK,
            source_id=source_id,
            context={
                "source_id": source_id,
                "trust_score": trust_score,
                "minimum_trust": minimum_trust,
                **(context or {}),
            },
            remediation="Block ingestion from the source and require manual attestation or trust-policy override.",
        )


class QuarantineOperationError(DataQualityError):
    def __init__(
        self,
        *,
        batch_id: str,
        details: str,
        retryable: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Failed to quarantine batch '{batch_id}': {details}",
            error_type=QualityErrorType.QUARANTINE_OPERATION_FAILED,
            severity=QualitySeverity.HIGH,
            retryable=retryable,
            stage=QualityStage.QUARANTINE,
            disposition=QualityDisposition.ESCALATE,
            batch_id=batch_id,
            context={"batch_id": batch_id, "details": details, **(context or {})},
            remediation="Persist the failure, stop downstream fan-out, and retry quarantine in an idempotent workflow.",
        )


class RemediationExecutionError(DataQualityError):
    def __init__(
        self,
        *,
        rule_id: str,
        action: str,
        details: str,
        retryable: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"Remediation action '{action}' failed for rule '{rule_id}': {details}",
            error_type=QualityErrorType.REMEDIATION_FAILED,
            severity=QualitySeverity.HIGH,
            retryable=retryable,
            stage=QualityStage.REMEDIATION,
            disposition=QualityDisposition.RETRY if retryable else QualityDisposition.ESCALATE,
            rule_id=rule_id,
            context={"rule_id": rule_id, "action": action, "details": details, **(context or {})},
            remediation="Retry the remediation with backoff and fall back to operator review if the failure persists.",
        )


def _infer_error_type(exc: Exception, stage: QualityStage) -> QualityErrorType:
    if isinstance(exc, KeyError):
        return QualityErrorType.REQUIRED_FIELD_MISSING

    if isinstance(exc, (TypeError, ValueError)):
        if stage in {QualityStage.VALIDATION, QualityStage.INGESTION}:
            return QualityErrorType.SCHEMA_VALIDATION_FAILED
        if stage is QualityStage.SCORING:
            return QualityErrorType.POLICY_THRESHOLD_INVALID
        return QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE

    if isinstance(exc, FileNotFoundError):
        if stage is QualityStage.PERSISTENCE:
            return QualityErrorType.QUALITY_MEMORY_UNAVAILABLE
        if stage is QualityStage.QUARANTINE:
            return QualityErrorType.QUARANTINE_STORAGE_UNAVAILABLE

    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        if stage is QualityStage.PERSISTENCE:
            return QualityErrorType.QUALITY_MEMORY_UNAVAILABLE
        if stage is QualityStage.QUARANTINE:
            return QualityErrorType.QUARANTINE_STORAGE_UNAVAILABLE
        return QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE

    return QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE


@contextmanager
def quality_error_boundary(
    *,
    stage: QualityStage | str,
    context: Optional[Mapping[str, Any]] = None,
    error_type: Optional[QualityErrorType] = None,
    severity: Optional[QualitySeverity] = None,
    retryable: Optional[bool] = None,
    remediation: Optional[str] = None,
    disposition: Optional[QualityDisposition] = None,
) -> Iterator[None]:
    """Normalize arbitrary failures inside a quality pipeline stage.

    Example:
        with quality_error_boundary(stage=QualityStage.VALIDATION, context={"dataset_id": "users"}):
            validate_records(records)
    """
    try:
        yield
    except Exception as exc:  # pragma: no cover - intentionally broad boundary.
        normalized = normalize_quality_exception(
            exc,
            stage=stage,
            context=context,
            error_type=error_type,
            severity=severity,
            retryable=retryable,
            remediation=remediation,
            disposition=disposition,
        )
        raise normalized from exc


def normalize_quality_exception(
    exc: Exception,
    *,
    stage: QualityStage | str,
    context: Optional[Mapping[str, Any]] = None,
    error_type: Optional[QualityErrorType] = None,
    severity: Optional[QualitySeverity] = None,
    retryable: Optional[bool] = None,
    remediation: Optional[str] = None,
    disposition: Optional[QualityDisposition] = None,
) -> DataQualityError:
    """Normalize arbitrary exceptions into ``DataQualityError``.

    This function is intentionally conservative: existing ``DataQualityError``
    instances are preserved, while generic runtime exceptions are converted into
    consistent, serializable, policy-aware failures.
    """
    stage_enum = _coerce_enum(stage, QualityStage, QualityStage.UNKNOWN)
    context_dict = dict(context or {})

    if isinstance(exc, DataQualityError):
        merged_context = {**exc.context, **_normalized_context(context_dict)}
        updates: Dict[str, Any] = {"context": merged_context}
        if exc.stage is QualityStage.UNKNOWN and stage_enum is not QualityStage.UNKNOWN:
            updates["stage"] = stage_enum
        if severity is not None:
            updates["severity"] = severity
        if retryable is not None:
            updates["retryable"] = retryable
        if remediation is not None:
            updates["remediation"] = remediation
        if disposition is not None:
            updates["disposition"] = disposition
        if error_type is not None and exc.error_type is QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE:
            updates["error_type"] = error_type
        return exc.with_updates(**updates)

    resolved_error_type = error_type or _infer_error_type(exc, stage_enum)
    resolved_severity = severity or _default_severity(resolved_error_type)
    resolved_retryable = _default_retryable(resolved_error_type) if retryable is None else retryable
    resolved_disposition = disposition or _default_disposition(resolved_error_type)

    return DataQualityError(
        message=f"Unhandled exception during data quality stage '{stage_enum.value}': {exc}",
        error_type=resolved_error_type,
        severity=resolved_severity,
        retryable=resolved_retryable,
        stage=stage_enum,
        disposition=resolved_disposition,
        context={
            "stage": stage_enum.value,
            "exception_type": type(exc).__name__,
            **context_dict,
        },
        remediation=remediation or "Retry with backoff, preserve the failed payload, and escalate to Handler if the failure persists.",
        cause=exc,
    )
