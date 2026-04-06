import os
import json
import time
import hashlib
from enum import Enum
from typing import Dict, Any, Optional, Callable

# -----------------------------------------------------------------------------
# Standardized severity levels
# -----------------------------------------------------------------------------
class Severity(Enum):
    """Standard severity levels for knowledge agent errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, value: str) -> "Severity":
        """Convert a string to a Severity enum, defaulting to MEDIUM."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.MEDIUM


# -----------------------------------------------------------------------------
# Deterministic machine‑readable error codes
# -----------------------------------------------------------------------------
class KnowledgeErrorType(Enum):
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
    RULE_TIMEOUT = ""

# Mapping from error type to unique, deterministic error code.
# Format: KNW-1xxx (Knowledge Agent)
_ERROR_CODE_MAP = {
    KnowledgeErrorType.RETRIEVAL_FAILURE: "KNW-1001",
    KnowledgeErrorType.ONTOLOGY_ERROR: "KNW-1002",
    KnowledgeErrorType.BIAS_DETECTION_FAILURE: "KNW-1003",
    KnowledgeErrorType.GOVERNANCE_VIOLATION: "KNW-1004",
    KnowledgeErrorType.CACHE_ERROR: "KNW-1005",
    KnowledgeErrorType.MEMORY_UPDATE_ERROR: "KNW-1006",
    KnowledgeErrorType.INVALID_DOCUMENT: "KNW-1007",
    KnowledgeErrorType.EMBEDDING_GENERATION_ERROR: "KNW-1008",
    KnowledgeErrorType.ONTOLOGY_EXPANSION_FAILURE: "KNW-1009",
    KnowledgeErrorType.KNOWLEDGE_BROADCAST_FAILURE: "KNW-1010",
    KnowledgeErrorType.RUNTIME_HEALTH_FAILURE: "KNW-1011",
    KnowledgeErrorType.METRICS_COLLECTION_FAILURE: "KNW-1012",
    KnowledgeErrorType.THRESHOLD_VIOLATION: "KNW-1013",
    KnowledgeErrorType.COMPONENT_DEGRADATION: "KNW-1014",
}


# -----------------------------------------------------------------------------
# Central audit / metrics pipeline integration
# -----------------------------------------------------------------------------
# Global hooks – can be set once at application startup.
_audit_pipeline: Optional[Callable[[Dict[str, Any]], None]] = None
_metrics_counter: Optional[Callable[[str, Severity, int], None]] = None  # (error_code, severity, count=1)


def set_audit_pipeline(callback: Callable[[Dict[str, Any]], None]) -> None:
    """
    Set a global callback that receives the full audit dict for every reported error.
    Typical use: send to Elasticsearch, CloudWatch, or a custom audit log.
    """
    global _audit_pipeline
    _audit_pipeline = callback


def set_metrics_counter(callback: Callable[[str, Severity, int], None]) -> None:
    """
    Set a global callback for incrementing metrics counters.
    Example: Prometheus counter labelled by error_code and severity.
    """
    global _metrics_counter
    _metrics_counter = callback

# -----------------------------------------------------------------------------
# Base exception class with forensic and reporting capabilities
# -----------------------------------------------------------------------------
class KnowledgeError(Exception):
    """Base exception for knowledge agent errors with forensic capabilities."""

    def __init__(
        self,
        error_type: KnowledgeErrorType,
        message: str,
        severity: Severity | str = Severity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        agent_state: Optional[Dict] = None,
        remediation: Optional[str] = None,
    ):
        # Normalize severity to enum
        if isinstance(severity, str):
            severity = Severity.from_string(severity)
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.agent_state = agent_state or {}
        self.remediation = remediation
        self.timestamp = time.time()
        self.error_id = self._generate_error_id()
        self.forensic_hash = self._generate_forensic_hash()
        super().__init__(message)

    def _generate_error_id(self) -> str:
        """Generate unique error ID using context and timestamp."""
        unique_str = f"{self.timestamp}{self.error_type.value}{json.dumps(self.context)}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:12]

    def _generate_forensic_hash(self) -> str:
        """Create verifiable hash of error context."""
        data = {
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "context": self.context,
            "agent_state": self.agent_state,
        }
        return hashlib.sha3_256(json.dumps(data).encode()).hexdigest()

    @property
    def error_code(self) -> str:
        """Deterministic machine‑readable error code."""
        return _ERROR_CODE_MAP.get(self.error_type, "KNW-9999")

    def to_audit_dict(self) -> Dict[str, Any]:
        """Structured representation for logging and auditing."""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "type": self.error_type.value,
            "severity": self.severity.value,
            "message": str(self),
            "timestamp": self.timestamp,
            "forensic_hash": self.forensic_hash,
            "context": self.context,
            "agent_state_snapshot": self.agent_state,
            "remediation": self.remediation,
        }

    def report(self) -> None:
        """
        Send this error to the configured audit pipeline and increment metrics counters.
        Should be called explicitly in an exception handler (e.g., except block).
        """
        audit_dict = self.to_audit_dict()
        if _audit_pipeline is not None:
            try:
                _audit_pipeline(audit_dict)
            except Exception:
                # Audit pipeline failures must not crash the application.
                pass

        if _metrics_counter is not None:
            try:
                _metrics_counter(self.error_code, self.severity, 1)
            except Exception:
                pass


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
            remediation="Validate the action payload, review safety policy, and inspect the downstream subsystem.",
        )


# -----------------------------------------------------------------------------
# Safe rule execution with timeout and sandbox
# -----------------------------------------------------------------------------
class RuleTimeoutError(KnowledgeError):
    """Raised when a rule execution exceeds its allowed time."""
    def __init__(self, rule_name: str, timeout_seconds: float, rule_code_preview: str = ""):
        super().__init__(
            error_type=KnowledgeErrorType.RULE_TIMEOUT,  # requires adding RULE_TIMEOUT to enum
            message=f"Rule '{rule_name}' exceeded timeout of {timeout_seconds}s",
            severity=Severity.HIGH,
            context={
                "rule_name": rule_name,
                "timeout_seconds": timeout_seconds,
                "rule_code_preview": rule_code_preview[:200] + "..." if len(rule_code_preview) > 200 else rule_code_preview,
            },
            remediation="Increase rule_timeout_seconds, optimize rule code, or split large rules."
        )

# -----------------------------------------------------------------------------
# New error classes for runtime health
# -----------------------------------------------------------------------------
class RuntimeHealthError(KnowledgeError):
    """Indicates a failure in the runtime health check subsystem."""
    def __init__(self, component: str, check_name: str, details: str):
        super().__init__(
            KnowledgeErrorType.RUNTIME_HEALTH_FAILURE,
            f"Health check '{check_name}' failed for component '{component}': {details}",
            severity=Severity.CRITICAL,
            context={
                "component": component,
                "health_check": check_name,
                "details": details,
            },
            remediation="Check component logs, resource availability, and network connectivity."
        )


class MetricsCollectionError(KnowledgeError):
    """Failure when collecting or exporting runtime metrics."""
    def __init__(self, metric_name: str, collector: str, reason: str):
        super().__init__(
            KnowledgeErrorType.METRICS_COLLECTION_FAILURE,
            f"Metric '{metric_name}' collection failed in '{collector}': {reason}",
            severity=Severity.MEDIUM,
            context={
                "metric_name": metric_name,
                "collector": collector,
                "reason": reason,
            },
            remediation="Verify metric pipeline configuration, permissions, and backend availability."
        )


class ThresholdViolationError(KnowledgeError):
    """Raised when an operational metric exceeds a configured threshold."""
    def __init__(self, metric: str, value: float, threshold: float, direction: str = "above"):
        super().__init__(
            KnowledgeErrorType.THRESHOLD_VIOLATION,
            f"Metric '{metric}' exceeded threshold: {value} {direction} {threshold}",
            severity=Severity.HIGH,
            context={
                "metric": metric,
                "current_value": value,
                "threshold": threshold,
                "direction": direction,
            },
            remediation="Scale resources, tune thresholds, or investigate anomaly."
        )


class ComponentDegradationWarning(KnowledgeError):
    """Warns about degraded performance or partial failure of a component."""
    def __init__(self, component: str, symptom: str, impact: str):
        super().__init__(
            KnowledgeErrorType.COMPONENT_DEGRADATION,
            f"Component '{component}' degraded: {symptom}",
            severity=Severity.MEDIUM,
            context={
                "component": component,
                "symptom": symptom,
                "impact": impact,
            },
            remediation="Restart component, increase timeouts, or allocate more resources."
        )

# -----------------------------------------------------------------------------
# Specific error classes – using Severity enum for clarity
# -----------------------------------------------------------------------------
class RetrievalError(KnowledgeError):
    """Failure in knowledge retrieval process."""

    def __init__(self, query: str, reason: str, retrieval_mode: str):
        super().__init__(
            KnowledgeErrorType.RETRIEVAL_FAILURE,
            f"Retrieval failed for query: '{query}' ({reason})",
            severity=Severity.HIGH,
            context={
                "failed_query": query,
                "retrieval_mode": retrieval_mode,
            },
            remediation="Check retrieval configuration, embedding model availability, and document corpus",
        )


class OntologyError(KnowledgeError):
    """Failure in ontology operations."""

    def __init__(self, operation: str, subject: str, details: str):
        super().__init__(
            KnowledgeErrorType.ONTOLOGY_ERROR,
            f"Ontology {operation} failed for '{subject}': {details}",
            context={
                "operation": operation,
                "subject": subject,
                "details": details,
            },
            remediation="Verify ontology relationships and ensure proper initialization",
        )


class BiasDetectionError(KnowledgeError):
    """Failure in bias detection subsystem."""

    def __init__(self, text_sample: str, error_details: str):
        super().__init__(
            KnowledgeErrorType.BIAS_DETECTION_FAILURE,
            f"Bias detection failed for text sample: {error_details}",
            severity=Severity.MEDIUM,
            context={
                "text_sample": text_sample[:200] + "..." if len(text_sample) > 200 else text_sample,
                "error": error_details,
            },
            remediation="Check bias detection model and configuration thresholds",
        )


class GovernanceViolation(KnowledgeError):
    """Governance policy violation during knowledge operation."""

    def __init__(self, policy_name: str, violation_details: Dict, query: str):
        super().__init__(
            KnowledgeErrorType.GOVERNANCE_VIOLATION,
            f"Governance violation: {policy_name}",
            severity=Severity.CRITICAL,
            context={
                "policy": policy_name,
                "violation_details": violation_details,
                "offending_query": query,
            },
            remediation="Review knowledge corpus and governance rules alignment",
        )


class CacheError(KnowledgeError):
    """Failure in knowledge cache operations."""

    def __init__(self, operation: str, key: str, error_details: str):
        super().__init__(
            KnowledgeErrorType.CACHE_ERROR,
            f"Cache {operation} failed for key '{key}': {error_details}",
            severity=Severity.MEDIUM,
            context={
                "cache_operation": operation,
                "cache_key": key,
                "error": error_details,
            },
            remediation="Verify cache storage and serialization mechanisms",
        )


class MemoryUpdateError(KnowledgeError):
    """Failure in memory update operations."""

    def __init__(self, key: str, value: Any, error_details: str):
        super().__init__(
            KnowledgeErrorType.MEMORY_UPDATE_ERROR,
            f"Memory update failed for key '{key}'",
            severity=Severity.HIGH,
            context={
                "memory_key": key,
                "attempted_value": str(value)[:100] + "..." if len(str(value)) > 100 else str(value),
                "error": error_details,
            },
            remediation="Check shared memory connection and serialization formats",
        )


class InvalidDocumentError(KnowledgeError):
    """Invalid document format or content."""

    def __init__(self, document: Any, reason: str):
        super().__init__(
            KnowledgeErrorType.INVALID_DOCUMENT,
            f"Invalid document: {reason}",
            context={
                "document_type": type(document).__name__,
                "document_preview": str(document)[:200] + "..." if len(str(document)) > 200 else str(document),
                "rejection_reason": reason,
            },
            remediation="Validate documents before ingestion: min_length=3, must be string",
        )


class EmbeddingError(KnowledgeError):
    """Failure in embedding generation."""

    def __init__(self, doc_id: str, model_name: str, error_details: str):
        super().__init__(
            KnowledgeErrorType.EMBEDDING_GENERATION_ERROR,
            f"Embedding failed for doc {doc_id} using {model_name}",
            severity=Severity.HIGH,
            context={
                "doc_id": doc_id,
                "model": model_name,
                "error": error_details,
            },
            remediation="Verify embedding model availability and input data format",
        )


if __name__ == "__main__":
    # Optional: Configure audit pipeline and metrics counter at startup.
    def my_audit_pipeline(audit_dict):
        print(f"[AUDIT] {json.dumps(audit_dict, indent=2)}")

    def my_metrics_counter(code, severity, count):
        print(f"[METRICS] {code}:{severity.value} += {count}")

    set_audit_pipeline(my_audit_pipeline)
    set_metrics_counter(my_metrics_counter)

    try:
        raise RetrievalError("What is RAG?", "embedding model timeout", "vector")
    except RetrievalError as e:
        e.report()   # Explicitly send to configured pipeline
        print(f"Handled: {e.error_code} - {e}")
