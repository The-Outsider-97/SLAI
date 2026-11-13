import json
import time
import hashlib

from enum import Enum
from typing import Any, Callable, Dict, Optional, cast

from ...base.utils.base_errors import BaseError


# -----------------------------------------------------------------------------
# Standardized severity levels
# -----------------------------------------------------------------------------
class Severity(str, Enum):
    """Standard severity levels for knowledge agent errors.

    Inherits from ``str`` so that instances compare equal to, and hash
    identically to, their string values.  This lets the ``Severity`` enum be
    passed straight into :class:`BaseError` logging helpers (which key off
    plain strings) without any conversion shims.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "Severity":
        """Convert a string to a :class:`Severity`, defaulting to ``MEDIUM``."""
        try:
            return cls(str(value).strip().lower())
        except (ValueError, AttributeError):
            return cls.MEDIUM


# -----------------------------------------------------------------------------
# Deterministic machine-readable error codes
# -----------------------------------------------------------------------------
class KnowledgeErrorType(Enum):
    # -- Runtime / operational ------------------------------------------------
    RETRIEVAL_FAILURE = "Knowledge Retrieval Failure"
    ONTOLOGY_ERROR = "Ontology Management Error"
    BIAS_DETECTION_FAILURE = "Bias Detection Failure"
    GOVERNANCE_VIOLATION = "Governance Policy Violation"
    CACHE_ERROR = "Knowledge Cache Error"
    MEMORY_UPDATE_ERROR = "Memory Update Failure"
    INVALID_DOCUMENT = "Invalid Document Format"
    EMBEDDING_GENERATION_ERROR = "Embedding Generation Failure"
    ONTOLOGY_EXPANSION_FAILURE = "Ontology Query Expansion Failure"
    KNOWLEDGE_BROADCAST_FAILURE = "Knowledge Broadcast Failure"
    RUNTIME_HEALTH_FAILURE = "Runtime Health Check Failure"
    METRICS_COLLECTION_FAILURE = "Metrics Collection Failure"
    THRESHOLD_VIOLATION = "Operational Threshold Violation"
    COMPONENT_DEGRADATION = "Component Degradation Warning"
    RULE_TIMEOUT = "Rule Execution Timeout"

    # -- Canonical knowledge service (assurance layer) ------------------------
    AUTHORIZATION_DENIED = "Authorization Denied"
    VALIDATION_FAILURE = "Validation Failure"
    REVISION_CONFLICT = "Revision Conflict"
    NOT_FOUND = "Record Not Found"
    UNSUPPORTED_EVALUATION = "Unsupported Evaluation Method"
    MISSING_INPUT = "Missing Required Input"


# Format: KNW-1xxx = runtime, KNW-2xxx = assurance.
_ERROR_CODE_MAP: Dict[KnowledgeErrorType, str] = {
    # runtime
    KnowledgeErrorType.RETRIEVAL_FAILURE:            "KNW-1001",
    KnowledgeErrorType.ONTOLOGY_ERROR:               "KNW-1002",
    KnowledgeErrorType.BIAS_DETECTION_FAILURE:       "KNW-1003",
    KnowledgeErrorType.GOVERNANCE_VIOLATION:         "KNW-1004",
    KnowledgeErrorType.CACHE_ERROR:                  "KNW-1005",
    KnowledgeErrorType.MEMORY_UPDATE_ERROR:          "KNW-1006",
    KnowledgeErrorType.INVALID_DOCUMENT:             "KNW-1007",
    KnowledgeErrorType.EMBEDDING_GENERATION_ERROR:   "KNW-1008",
    KnowledgeErrorType.ONTOLOGY_EXPANSION_FAILURE:   "KNW-1009",
    KnowledgeErrorType.KNOWLEDGE_BROADCAST_FAILURE:  "KNW-1010",
    KnowledgeErrorType.RUNTIME_HEALTH_FAILURE:       "KNW-1011",
    KnowledgeErrorType.METRICS_COLLECTION_FAILURE:   "KNW-1012",
    KnowledgeErrorType.THRESHOLD_VIOLATION:          "KNW-1013",
    KnowledgeErrorType.COMPONENT_DEGRADATION:        "KNW-1014",
    KnowledgeErrorType.RULE_TIMEOUT:                 "KNW-1015",
    # assurance
    KnowledgeErrorType.AUTHORIZATION_DENIED:         "KNW-2001",
    KnowledgeErrorType.VALIDATION_FAILURE:           "KNW-2002",
    KnowledgeErrorType.REVISION_CONFLICT:            "KNW-2003",
    KnowledgeErrorType.NOT_FOUND:                    "KNW-2004",
    KnowledgeErrorType.UNSUPPORTED_EVALUATION:       "KNW-2005",
    KnowledgeErrorType.MISSING_INPUT:                "KNW-2006",
}


# -----------------------------------------------------------------------------
# Central audit / metrics pipeline integration
# -----------------------------------------------------------------------------
_audit_pipeline: Optional[Callable[[Dict[str, Any]], None]] = None
_metrics_counter: Optional[Callable[[str, Severity, int], None]] = None


def set_audit_pipeline(callback: Callable[[Dict[str, Any]], None]) -> None:
    """Set the global callback receiving every reported error's audit dict."""
    global _audit_pipeline
    _audit_pipeline = callback


def set_metrics_counter(callback: Callable[[str, Severity, int], None]) -> None:
    """Set the global callback for metric counter increments."""
    global _metrics_counter
    _metrics_counter = callback


# -----------------------------------------------------------------------------
# Base exception class (inherits BaseError, single inheritance)
# -----------------------------------------------------------------------------
class KnowledgeError(BaseError):
    """Base exception for knowledge agent errors.

    Extends :class:`BaseError` with:

    - ``error_type`` / ``error_code`` (deterministic KNW-xxxx code)
    - ``agent_state`` (forensic snapshot)
    - ``error_id`` / ``forensic_hash`` (tamper-evident identifiers)
    - :meth:`to_audit_dict` / :meth:`report` (pipeline integration)
    """

    def __init__(
        self,
        error_type: KnowledgeErrorType,
        message: str,
        severity: "Severity | str" = Severity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        agent_state: Optional[Dict[str, Any]] = None,
        remediation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Normalise severity, keep it as an enum for callers.
        if isinstance(severity, str):
            severity = Severity.from_string(severity)

        # Set knowledge-specific fields *before* delegating, so that any
        # BaseError hook that inspects ``self`` sees them.
        cast(Any, self).error_type = error_type
        self.agent_state = dict(agent_state or {})
        self.remediation = remediation

        # Merge user context + agent state so serializers see everything.
        merged_context: Dict[str, Any] = dict(context or {})
        if self.agent_state:
            merged_context["agent_state"] = self.agent_state

        # BaseError owns: code, severity, context, details, metadata,
        # timestamp (ISO-8601 UTC), resolution_hint, cause, retryable.
        super().__init__(
            message,
            code=_ERROR_CODE_MAP.get(error_type, "KNW-9999"),
            severity=severity.value,
            context=merged_context,
            resolution_hint=remediation,
            **kwargs,
        )

        # Re-attach the enum so callers can do ``e.severity.value`` / ``.name``.
        self.severity = severity

        # Forensic identifiers (deterministic, safe even if context is huge).
        self.error_id = self._generate_error_id()
        self.forensic_hash = self._generate_forensic_hash()

    # ------------------------------------------------------------------ #
    # Deterministic identifiers
    # ------------------------------------------------------------------ #
    def _generate_error_id(self) -> str:
        payload = {
            "ts": self.timestamp,
            "type": self.error_type.value,
            "ctx": self.context,
        }
        blob = json.dumps(payload, default=str, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def _generate_forensic_hash(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "context": self.context,
            "agent_state": self.agent_state,
        }
        blob = json.dumps(payload, default=str, sort_keys=True)
        return hashlib.sha3_256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    @property
    def error_code(self) -> str:
        """Deterministic machine-readable error code (KNW-xxxx)."""
        return self.code

    def to_audit_dict(self) -> Dict[str, Any]:
        """Structured, JSON-safe representation for audit pipelines."""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "forensic_hash": self.forensic_hash,
            "context": self.context,
            "agent_state_snapshot": self.agent_state,
            "remediation": self.remediation,
            "retryable": self.retryable,
            "component": self.component,
            "operation": self.operation,
        }

    def report(self) -> None:
        """Send the error to the audit pipeline and metrics counter.

        Safe against pipeline failures — reporting must never crash the
        caller.  Intended to be invoked explicitly from ``except`` blocks.
        """
        audit_dict = self.to_audit_dict()

        if _audit_pipeline is not None:
            try:
                _audit_pipeline(audit_dict)
            except Exception:  # pragma: no cover - defensive
                pass

        if _metrics_counter is not None:
            try:
                _metrics_counter(self.error_code, self.severity, 1)
            except Exception:  # pragma: no cover - defensive
                pass


# -----------------------------------------------------------------------------
# Action / rule execution
# -----------------------------------------------------------------------------
class ActionExecutionError(KnowledgeError):
    """Execution or routing failure for a downstream action."""

    def __init__(self, action_type: str, payload: str, details: str):
        super().__init__(
            error_type=KnowledgeErrorType.KNOWLEDGE_BROADCAST_FAILURE,
            message=f"Action execution failed for '{action_type}': {details}",
            severity=Severity.HIGH,
            context={
                "action_type": action_type,
                "payload": payload[:200],
                "details": details,
            },
            remediation=(
                "Validate the action payload, review safety policy, and "
                "inspect the downstream subsystem."
            ),
        )


class RuleTimeoutError(KnowledgeError):
    """Raised when a rule execution exceeds its allowed time."""

    def __init__(
        self,
        rule_name: str,
        timeout_seconds: float,
        rule_code_preview: str = "",
    ) -> None:
        super().__init__(
            error_type=KnowledgeErrorType.RULE_TIMEOUT,
            message=f"Rule '{rule_name}' exceeded timeout of {timeout_seconds}s",
            severity=Severity.HIGH,
            context={
                "rule_name": rule_name,
                "timeout_seconds": timeout_seconds,
                "rule_code_preview": (
                    rule_code_preview[:200] + "..."
                    if len(rule_code_preview) > 200
                    else rule_code_preview
                ),
            },
            remediation=(
                "Increase rule_timeout_seconds, optimize rule code, or split "
                "large rules."
            ),
            retryable=True,
        )


# -----------------------------------------------------------------------------
# Runtime health / observability
# -----------------------------------------------------------------------------
class RuntimeHealthError(KnowledgeError):
    """Indicates a failure in the runtime health check subsystem."""

    def __init__(self, component: str, check_name: str, details: str):
        super().__init__(
            error_type=KnowledgeErrorType.RUNTIME_HEALTH_FAILURE,
            message=(
                f"Health check '{check_name}' failed for component "
                f"'{component}': {details}"
            ),
            severity=Severity.CRITICAL,
            context={
                "component": component,
                "health_check": check_name,
                "details": details,
            },
            remediation=(
                "Check component logs, resource availability, and network "
                "connectivity."
            ),
        )


class MetricsCollectionError(KnowledgeError):
    """Failure when collecting or exporting runtime metrics."""

    def __init__(self, metric_name: str, collector: str, reason: str):
        super().__init__(
            error_type=KnowledgeErrorType.METRICS_COLLECTION_FAILURE,
            message=(
                f"Metric '{metric_name}' collection failed in "
                f"'{collector}': {reason}"
            ),
            severity=Severity.MEDIUM,
            context={
                "metric_name": metric_name,
                "collector": collector,
                "reason": reason,
            },
            remediation=(
                "Verify metric pipeline configuration, permissions, and "
                "backend availability."
            ),
            retryable=True,
        )


class ThresholdViolationError(KnowledgeError):
    """Raised when an operational metric exceeds a configured threshold."""

    def __init__(
        self,
        metric: str,
        value: float,
        threshold: float,
        direction: str = "above",
    ):
        super().__init__(
            error_type=KnowledgeErrorType.THRESHOLD_VIOLATION,
            message=(
                f"Metric '{metric}' exceeded threshold: "
                f"{value} {direction} {threshold}"
            ),
            severity=Severity.HIGH,
            context={
                "metric": metric,
                "current_value": value,
                "threshold": threshold,
                "direction": direction,
            },
            remediation=(
                "Scale resources, tune thresholds, or investigate anomaly."
            ),
        )


class ComponentDegradationWarning(KnowledgeError):
    """Warns about degraded performance or partial failure of a component."""

    def __init__(self, component: str, symptom: str, impact: str):
        super().__init__(
            error_type=KnowledgeErrorType.COMPONENT_DEGRADATION,
            message=f"Component '{component}' degraded: {symptom}",
            severity=Severity.MEDIUM,
            context={
                "component": component,
                "symptom": symptom,
                "impact": impact,
            },
            remediation=(
                "Restart component, increase timeouts, or allocate more "
                "resources."
            ),
        )


# -----------------------------------------------------------------------------
# Domain-specific errors
# -----------------------------------------------------------------------------
class RetrievalError(KnowledgeError):
    """Failure in knowledge retrieval process."""

    def __init__(self, query: str, reason: str, retrieval_mode: str):
        super().__init__(
            error_type=KnowledgeErrorType.RETRIEVAL_FAILURE,
            message=f"Retrieval failed for query: '{query}' ({reason})",
            severity=Severity.HIGH,
            context={
                "failed_query": query,
                "retrieval_mode": retrieval_mode,
                "reason": reason,
            },
            remediation=(
                "Check retrieval configuration, embedding model availability, "
                "and document corpus."
            ),
            retryable=True,
        )


class OntologyError(KnowledgeError):
    """Failure in ontology operations."""

    def __init__(self, operation: str, subject: str, details: str):
        super().__init__(
            error_type=KnowledgeErrorType.ONTOLOGY_ERROR,
            message=f"Ontology {operation} failed for '{subject}': {details}",
            severity=Severity.MEDIUM,
            context={
                "operation": operation,
                "subject": subject,
                "details": details,
            },
            remediation=(
                "Verify ontology relationships and ensure proper "
                "initialization."
            ),
        )


class BiasDetectionError(KnowledgeError):
    """Failure in bias detection subsystem."""

    def __init__(self, text_sample: str, error_details: str):
        super().__init__(
            error_type=KnowledgeErrorType.BIAS_DETECTION_FAILURE,
            message=f"Bias detection failed for text sample: {error_details}",
            severity=Severity.MEDIUM,
            context={
                "text_sample": (
                    text_sample[:200] + "..."
                    if len(text_sample) > 200
                    else text_sample
                ),
                "error": error_details,
            },
            remediation=(
                "Check bias detection model and configuration thresholds."
            ),
        )


class GovernanceViolation(KnowledgeError):
    """Governance policy violation during knowledge operation."""

    def __init__(
        self,
        policy_name: str,
        violation_details: Dict[str, Any],
        query: str,
    ):
        super().__init__(
            error_type=KnowledgeErrorType.GOVERNANCE_VIOLATION,
            message=f"Governance violation: {policy_name}",
            severity=Severity.CRITICAL,
            context={
                "policy": policy_name,
                "violation_details": violation_details,
                "offending_query": query,
            },
            remediation=("Review knowledge corpus and governance rules alignment."),
        )


class CacheError(KnowledgeError):
    """Failure in knowledge cache operations."""

    def __init__(self, operation: str, key: str, error_details: str):
        super().__init__(
            error_type=KnowledgeErrorType.CACHE_ERROR,
            message=f"Cache {operation} failed for key '{key}': {error_details}",
            severity=Severity.MEDIUM,
            context={
                "cache_operation": operation,
                "cache_key": key,
                "error": error_details,
            },
            remediation="Verify cache storage and serialization mechanisms.",
            retryable=True,
        )


class MemoryUpdateError(KnowledgeError):
    """Failure in memory update operations."""

    def __init__(self, key: str, value: Any, error_details: str):
        super().__init__(
            error_type=KnowledgeErrorType.MEMORY_UPDATE_ERROR,
            message=f"Memory update failed for key '{key}'",
            severity=Severity.HIGH,
            context={
                "memory_key": key,
                "attempted_value": (
                    str(value)[:100] + "..."
                    if len(str(value)) > 100
                    else str(value)
                ),
                "error": error_details,
            },
            remediation=("Check shared memory connection and serialization formats."),
            retryable=True,
        )


class InvalidDocumentError(KnowledgeError):
    """Invalid document format or content."""

    def __init__(self, document: Any, reason: str):
        preview = str(document)
        super().__init__(
            error_type=KnowledgeErrorType.INVALID_DOCUMENT,
            message=f"Invalid document: {reason}",
            severity=Severity.LOW,
            context={
                "document_type": type(document).__name__,
                "document_preview": (preview[:200] + "..." if len(preview) > 200 else preview),
                "rejection_reason": reason,
            },
            remediation=("Validate documents before ingestion: min_length=3, must be string."),
        )


class EmbeddingError(KnowledgeError):
    """Failure in embedding generation."""

    def __init__(self, doc_id: str, model_name: str, error_details: str):
        super().__init__(
            error_type=KnowledgeErrorType.EMBEDDING_GENERATION_ERROR,
            message=f"Embedding failed for doc {doc_id} using {model_name}",
            severity=Severity.HIGH,
            context={
                "doc_id": doc_id,
                "model": model_name,
                "error": error_details,
            },
            remediation=("Verify embedding model availability and input data format."),
            retryable=True,
        )


# -----------------------------------------------------------------------------
# Assurance Errors (canonical knowledge service)
# -----------------------------------------------------------------------------
class KnowledgeAssuranceError(KnowledgeError):
    """Base class for canonical knowledge-service errors.

    All concrete assurance errors accept an optional ``details`` mapping for
    forensic extras and default to :class:`Severity.MEDIUM`.  Callers can
    override ``severity`` and pass additional keyword arguments through to
    :class:`KnowledgeError` when needed.
    """


class AuthorizationDenied(KnowledgeAssuranceError):
    """Raised when a caller lacks permission for the requested action."""

    def __init__(
        self,
        principal_id: str,
        action: str,
        resource: Optional[str] = None,
        required_scope: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: "Severity | str" = Severity.HIGH,
    ) -> None:
        super().__init__(
            error_type=KnowledgeErrorType.AUTHORIZATION_DENIED,
            message=(
                f"Authorization denied for principal '{principal_id}' "
                f"attempting '{action}'"
                + (f" on '{resource}'" if resource else "")
            ),
            severity=severity,
            context={
                "principal_id": principal_id,
                "action": action,
                "resource": resource,
                "required_scope": required_scope,
                "details": details,
            },
            remediation=("Verify principal credentials and the required scope assignments for this action."),
        )


class ValidationFailure(KnowledgeAssuranceError):
    """Raised when a payload fails schema/semantic validation."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: "Severity | str" = Severity.MEDIUM,
    ) -> None:
        super().__init__(
            error_type=KnowledgeErrorType.VALIDATION_FAILURE,
            message=message,
            severity=severity,
            context={
                "field": field,
                "details": details,
            },
            remediation=("Review the payload against the contract schema; ensure all required fields are present and well-typed."),
        )


class RevisionConflict(KnowledgeAssuranceError):
    """Raised when an optimistic-concurrency revision check fails."""

    def __init__(
        self,
        record_id: str,
        expected_revision: int,
        actual_revision: int,
        details: Optional[Dict[str, Any]] = None,
        severity: "Severity | str" = Severity.HIGH,
    ) -> None:
        super().__init__(
            error_type=KnowledgeErrorType.REVISION_CONFLICT,
            message=(
                f"Revision conflict for record '{record_id}': "
                f"expected {expected_revision}, got {actual_revision}"
            ),
            severity=severity,
            context={
                "record_id": record_id,
                "expected_revision": expected_revision,
                "actual_revision": actual_revision,
                "details": details,
            },
            remediation=("Fetch the latest revision, rebase the change, and retry."),
            retryable=True,
        )


class NotFound(KnowledgeAssuranceError):
    """Raised when a referenced record cannot be located."""

    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        details: Optional[Dict[str, Any]] = None,
        severity: "Severity | str" = Severity.MEDIUM,
    ) -> None:
        super().__init__(
            error_type=KnowledgeErrorType.NOT_FOUND,
            message=f"{entity_type} '{entity_id}' not found",
            severity=severity,
            context={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "details": details,
            },
            remediation=("Verify the identifier and confirm the record exists in the current scope."),
        )


class UnsupportedEvaluation(KnowledgeAssuranceError):
    """Raised when a rule's evaluation method is not supported."""

    def __init__(
        self,
        evaluation_method: str,
        rule_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: "Severity | str" = Severity.MEDIUM,
    ) -> None:
        message = f"Unsupported evaluation method '{evaluation_method}'"
        if rule_id:
            message += f" for rule '{rule_id}'"
        super().__init__(
            error_type=KnowledgeErrorType.UNSUPPORTED_EVALUATION,
            message=message,
            severity=severity,
            context={
                "evaluation_method": evaluation_method,
                "rule_id": rule_id,
                "details": details,
            },
            remediation=("Register the evaluation method or switch the rule to a supported one."),
        )


class MissingInput(KnowledgeAssuranceError):
    """Raised when one or more required input fields are missing."""

    def __init__(
        self,
        fields: "set[str] | frozenset[str] | list[str] | tuple[str, ...]",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: "Severity | str" = Severity.MEDIUM,
    ) -> None:
        field_set = frozenset(fields)
        self.fields = field_set
        sorted_fields = sorted(field_set)
        super().__init__(
            error_type=KnowledgeErrorType.MISSING_INPUT,
            message=("Missing required input fields: " + ", ".join(sorted_fields)),
            severity=severity,
            context={
                "missing_fields": sorted_fields,
                "operation": operation,
                "details": details,
            },
            remediation=("Provide the missing fields and retry the operation."),
        )


class UnsafeLegacyRule(ValidationFailure):
    pass


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # Enums
    "Severity",
    "KnowledgeErrorType",

    # Base exception
    "KnowledgeError",

    # Runtime errors
    "ActionExecutionError",
    "RuleTimeoutError",
    "RuntimeHealthError",
    "MetricsCollectionError",
    "ThresholdViolationError",
    "ComponentDegradationWarning",
    "RetrievalError",
    "OntologyError",
    "BiasDetectionError",
    "GovernanceViolation",
    "CacheError",
    "MemoryUpdateError",
    "InvalidDocumentError",
    "EmbeddingError",

    # Assurance errors
    "KnowledgeAssuranceError",
    "AuthorizationDenied",
    "ValidationFailure",
    "RevisionConflict",
    "NotFound",
    "UnsupportedEvaluation",
    "MissingInput",
    "UnsafeLegacyRule",

    # Global hooks
    "set_audit_pipeline",
    "set_metrics_counter",
]


if __name__ == "__main__":  # pragma: no cover
    def _audit(audit_dict: Dict[str, Any]) -> None:
        print(f"[AUDIT] {json.dumps(audit_dict, indent=2, default=str)}")

    def _metrics(code: str, severity: Severity, count: int) -> None:
        print(f"[METRICS] {code}:{severity.value} += {count}")

    set_audit_pipeline(_audit)
    set_metrics_counter(_metrics)

    try:
        raise RetrievalError("What is RAG?", "embedding model timeout", "vector")
    except RetrievalError as exc:
        exc.report()
        print(f"Handled: {exc.error_code} - {exc}")