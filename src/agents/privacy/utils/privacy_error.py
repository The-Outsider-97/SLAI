from __future__ import annotations

"""Production-grade privacy error model for the Privacy Agent.

This module provides:
- structured privacy error typing across the personal-data lifecycle,
- safe serialization and context sanitization to avoid secondary leakage,
- auditable error payloads suitable for memory, tracing, and compliance sinks,
- normalization of generic exceptions into privacy-aware runtime decisions.
"""

import hashlib
import json
import time
import traceback
import uuid

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


class PrivacySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrivacyDecision(str, Enum):
    ALLOW = "allow"
    MODIFY = "modify"
    BLOCK = "block"
    ESCALATE = "escalate"


class PrivacyDomain(str, Enum):
    IDENTIFICATION = "identification"
    MINIMIZATION = "minimization"
    CONSENT = "consent"
    RETENTION = "retention"
    TRANSFER = "transfer"
    MEMORY = "memory"
    AUDITABILITY = "auditability"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    INTERNAL = "internal"


class PrivacyErrorType(str, Enum):
    PII_DETECTION_FAILED = "pii_detection_failed"
    SENSITIVE_ENTITY_CLASSIFIER_UNCERTAIN = "sensitive_entity_classifier_uncertain"
    DATA_CLASSIFICATION_POLICY_MISSING = "data_classification_policy_missing"

    REDACTION_FAILED = "redaction_failed"
    OVER_REDACTION_DETECTED = "over_redaction_detected"
    UNDER_REDACTION_DETECTED = "under_redaction_detected"
    DATA_MINIMIZATION_VIOLATION = "data_minimization_violation"
    TOOL_PAYLOAD_SANITIZATION_FAILED = "tool_payload_sanitization_failed"

    CONSENT_VALIDATION_FAILED = "consent_validation_failed"
    CONSENT_ARTIFACT_MISSING = "consent_artifact_missing"
    PURPOSE_LIMITATION_VIOLATION = "purpose_limitation_violation"
    CROSS_CONTEXT_SHARING_BLOCKED = "cross_context_sharing_blocked"

    RETENTION_POLICY_VIOLATION = "retention_policy_violation"
    RETENTION_OBLIGATION_MISSING = "retention_obligation_missing"
    DELETION_SLA_VIOLATION = "deletion_sla_violation"
    DELETION_WORKFLOW_FAILED = "deletion_workflow_failed"

    DATA_EXPORT_BLOCKED = "data_export_blocked"
    CROSS_BORDER_TRANSFER_BLOCKED = "cross_border_transfer_blocked"

    PRIVACY_MEMORY_UNAVAILABLE = "privacy_memory_unavailable"
    PRIVACY_MEMORY_WRITE_FAILED = "privacy_memory_write_failed"

    AUDIT_LOG_WRITE_FAILED = "audit_log_write_failed"
    AUDIT_EVIDENCE_GENERATION_FAILED = "audit_evidence_generation_failed"
    AUDIT_REPORT_GENERATION_FAILED = "audit_report_generation_failed"

    ENCRYPTION_POLICY_VIOLATION = "encryption_policy_violation"

    PRIVACY_CONFIGURATION_ERROR = "privacy_configuration_error"
    POLICY_EVALUATION_FAILED = "policy_evaluation_failed"
    PRIVACY_OPERATION_TIMEOUT = "privacy_operation_timeout"
    INTERNAL_PRIVACY_ERROR = "internal_privacy_error"


@dataclass(frozen=True)
class PrivacyErrorSpec:
    code: str
    severity: PrivacySeverity
    retryable: bool
    decision: PrivacyDecision
    domain: PrivacyDomain
    remediation: str


PRIVACY_ERROR_SPECS: Dict[PrivacyErrorType, PrivacyErrorSpec] = {
    PrivacyErrorType.PII_DETECTION_FAILED: PrivacyErrorSpec(
        code="PRV-3001",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.IDENTIFICATION,
        remediation="Defer storage and transfer, retry with a secondary detector, and require manual review for high-risk flows.",
    ),
    PrivacyErrorType.SENSITIVE_ENTITY_CLASSIFIER_UNCERTAIN: PrivacyErrorSpec(
        code="PRV-3002",
        severity=PrivacySeverity.MEDIUM,
        retryable=True,
        decision=PrivacyDecision.MODIFY,
        domain=PrivacyDomain.IDENTIFICATION,
        remediation="Apply conservative masking and escalate ambiguous entities for secondary classification.",
    ),
    PrivacyErrorType.DATA_CLASSIFICATION_POLICY_MISSING: PrivacyErrorSpec(
        code="PRV-3016",
        severity=PrivacySeverity.HIGH,
        retryable=False,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.CONFIGURATION,
        remediation="Load the correct policy pack or block sensitive processing until classification rules are available.",
    ),
    PrivacyErrorType.REDACTION_FAILED: PrivacyErrorSpec(
        code="PRV-3003",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.MINIMIZATION,
        remediation="Block downstream storage or tool calls until redaction succeeds or a safe fallback mask is applied.",
    ),
    PrivacyErrorType.OVER_REDACTION_DETECTED: PrivacyErrorSpec(
        code="PRV-3004",
        severity=PrivacySeverity.MEDIUM,
        retryable=True,
        decision=PrivacyDecision.MODIFY,
        domain=PrivacyDomain.MINIMIZATION,
        remediation="Re-run redaction with context-aware exceptions and preserve only the minimum fields required for the approved purpose.",
    ),
    PrivacyErrorType.UNDER_REDACTION_DETECTED: PrivacyErrorSpec(
        code="PRV-3005",
        severity=PrivacySeverity.CRITICAL,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.MINIMIZATION,
        remediation="Immediately block output, quarantine the payload, and trigger incident response with forensic capture.",
    ),
    PrivacyErrorType.DATA_MINIMIZATION_VIOLATION: PrivacyErrorSpec(
        code="PRV-3017",
        severity=PrivacySeverity.HIGH,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.MINIMIZATION,
        remediation="Remove non-essential fields, re-bind the payload to the approved purpose, and retry with the minimized schema only.",
    ),
    PrivacyErrorType.TOOL_PAYLOAD_SANITIZATION_FAILED: PrivacyErrorSpec(
        code="PRV-3018",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.MINIMIZATION,
        remediation="Abort the external tool call, sanitize the payload again, and allow only the least-data-required subset.",
    ),
    PrivacyErrorType.CONSENT_VALIDATION_FAILED: PrivacyErrorSpec(
        code="PRV-3006",
        severity=PrivacySeverity.HIGH,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.CONSENT,
        remediation="Block the action, request explicit consent, and record the denial in the privacy audit trail.",
    ),
    PrivacyErrorType.CONSENT_ARTIFACT_MISSING: PrivacyErrorSpec(
        code="PRV-3007",
        severity=PrivacySeverity.HIGH,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.CONSENT,
        remediation="Require a valid consent artifact before continuing processing or sharing.",
    ),
    PrivacyErrorType.PURPOSE_LIMITATION_VIOLATION: PrivacyErrorSpec(
        code="PRV-3008",
        severity=PrivacySeverity.CRITICAL,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.CONSENT,
        remediation="Block processing, bind the request to an approved purpose, and record a compliance event.",
    ),
    PrivacyErrorType.CROSS_CONTEXT_SHARING_BLOCKED: PrivacyErrorSpec(
        code="PRV-3019",
        severity=PrivacySeverity.HIGH,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.CONSENT,
        remediation="Prevent cross-context propagation unless a valid purpose binding and legal basis are present.",
    ),
    PrivacyErrorType.RETENTION_POLICY_VIOLATION: PrivacyErrorSpec(
        code="PRV-3009",
        severity=PrivacySeverity.CRITICAL,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.RETENTION,
        remediation="Immediately quarantine the record and enqueue deletion or legal-hold handling.",
    ),
    PrivacyErrorType.RETENTION_OBLIGATION_MISSING: PrivacyErrorSpec(
        code="PRV-3020",
        severity=PrivacySeverity.HIGH,
        retryable=False,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.RETENTION,
        remediation="Stop write access for the record until a retention policy is resolved and attached.",
    ),
    PrivacyErrorType.DELETION_SLA_VIOLATION: PrivacyErrorSpec(
        code="PRV-3010",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.RETENTION,
        remediation="Escalate to deletion operations, prioritize the request, and create a tombstone reference for verification.",
    ),
    PrivacyErrorType.DELETION_WORKFLOW_FAILED: PrivacyErrorSpec(
        code="PRV-3021",
        severity=PrivacySeverity.CRITICAL,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.RETENTION,
        remediation="Retry against the deletion backend, preserve the request record, and hold downstream access until completion is verified.",
    ),
    PrivacyErrorType.DATA_EXPORT_BLOCKED: PrivacyErrorSpec(
        code="PRV-3011",
        severity=PrivacySeverity.HIGH,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.TRANSFER,
        remediation="Block the export and require policy review before any transfer proceeds.",
    ),
    PrivacyErrorType.CROSS_BORDER_TRANSFER_BLOCKED: PrivacyErrorSpec(
        code="PRV-3012",
        severity=PrivacySeverity.CRITICAL,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.TRANSFER,
        remediation="Block the transfer and route through an approved regional processor or valid transfer mechanism.",
    ),
    PrivacyErrorType.PRIVACY_MEMORY_UNAVAILABLE: PrivacyErrorSpec(
        code="PRV-3013",
        severity=PrivacySeverity.CRITICAL,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.MEMORY,
        remediation="Fail closed for sensitive actions and retry against a replicated privacy memory store.",
    ),
    PrivacyErrorType.PRIVACY_MEMORY_WRITE_FAILED: PrivacyErrorSpec(
        code="PRV-3022",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.MEMORY,
        remediation="Retry the write, reconcile lineage gaps, and prevent stateful privacy decisions until the write is durable.",
    ),
    PrivacyErrorType.AUDIT_LOG_WRITE_FAILED: PrivacyErrorSpec(
        code="PRV-3014",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.AUDITABILITY,
        remediation="Use a fallback audit channel and raise a privacy observability incident if durable logging cannot be completed.",
    ),
    PrivacyErrorType.AUDIT_EVIDENCE_GENERATION_FAILED: PrivacyErrorSpec(
        code="PRV-3023",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.AUDITABILITY,
        remediation="Retry evidence generation and mark the compliance bundle as incomplete until regenerated.",
    ),
    PrivacyErrorType.ENCRYPTION_POLICY_VIOLATION: PrivacyErrorSpec(
        code="PRV-3015",
        severity=PrivacySeverity.CRITICAL,
        retryable=False,
        decision=PrivacyDecision.BLOCK,
        domain=PrivacyDomain.SECURITY,
        remediation="Block persistence or transfer until approved encryption controls are enforced.",
    ),
    PrivacyErrorType.PRIVACY_CONFIGURATION_ERROR: PrivacyErrorSpec(
        code="PRV-3024",
        severity=PrivacySeverity.HIGH,
        retryable=False,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.CONFIGURATION,
        remediation="Validate the privacy configuration and policy pack before enabling the affected workflow.",
    ),
    PrivacyErrorType.POLICY_EVALUATION_FAILED: PrivacyErrorSpec(
        code="PRV-3025",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.INTERNAL,
        remediation="Retry policy evaluation with a stable snapshot of inputs and escalate unresolved failures for manual review.",
    ),
    PrivacyErrorType.PRIVACY_OPERATION_TIMEOUT: PrivacyErrorSpec(
        code="PRV-3026",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.INTERNAL,
        remediation="Retry the privacy stage with bounded backoff and hold the request in a safe state until evaluation completes.",
    ),
    PrivacyErrorType.INTERNAL_PRIVACY_ERROR: PrivacyErrorSpec(
        code="PRV-3027",
        severity=PrivacySeverity.HIGH,
        retryable=True,
        decision=PrivacyDecision.ESCALATE,
        domain=PrivacyDomain.INTERNAL,
        remediation="Trigger privacy incident handling, capture diagnostics, and prevent unsafe continuation.",
    ),
}

PRIVACY_ERROR_CODES: Dict[PrivacyErrorType, str] = {
    error_type: spec.code for error_type, spec in PRIVACY_ERROR_SPECS.items()
}

SENSITIVE_CONTEXT_KEYWORDS = {
    "access_token",
    "address",
    "api_key",
    "attachment",
    "authorization",
    "body",
    "cookie",
    "content",
    "credit_card",
    "cvv",
    "date_of_birth",
    "diagnosis",
    "dob",
    "document_content",
    "email",
    "full_name",
    "medical",
    "message",
    "national_id",
    "passport",
    "password",
    "patient",
    "payload",
    "phone",
    "prompt",
    "raw_text",
    "refresh_token",
    "secret",
    "session",
    "social_security_number",
    "ssn",
    "text",
    "token",
}

_audit_sink: Optional[Callable[[Dict[str, Any]], None]] = None
_metrics_sink: Optional[Callable[[str, str, int], None]] = None
_sink_lock = RLock()


def _generate_incident_id() -> str:
    return f"prv-{uuid.uuid4().hex[:16]}"


def get_privacy_error_spec(error_type: PrivacyErrorType) -> PrivacyErrorSpec:
    return PRIVACY_ERROR_SPECS[error_type]


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"request_id", "subject_id", "record_id", "policy_id", "policy_version", "audit_trail_ref"}:
        return False
    return normalized in SENSITIVE_CONTEXT_KEYWORDS or any(
        token in normalized
        for token in (
            "password",
            "secret",
            "token",
            "authorization",
            "cookie",
            "ssn",
            "passport",
            "patient",
            "diagnosis",
            "medical",
            "payload",
            "prompt",
            "raw_text",
            "document_content",
            "content",
        )
    )


def _short_hash(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        raw = repr(value).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:12]


def _redacted_summary(value: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "__redacted__": True,
        "type": type(value).__name__,
        "fingerprint": _short_hash(value),
    }
    if isinstance(value, str):
        summary["length"] = len(value)
    elif isinstance(value, (bytes, bytearray)):
        summary["length"] = len(value)
    elif isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        try:
            summary["items"] = len(value)
        except Exception:
            pass
    return summary


_UNSERIALIZABLE = object()


def _safe_scalar(value: Any, *, max_string_length: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value
        return f"{value[: max_string_length - 3]}..."
    if isinstance(value, (bytes, bytearray)):
        return {
            "__type__": type(value).__name__,
            "length": len(value),
            "fingerprint": _short_hash(value[:64]),
        }
    return _UNSERIALIZABLE


def sanitize_privacy_context(
    context: Optional[Mapping[str, Any]],
    *,
    max_depth: int = 4,
    max_items: int = 20,
    max_string_length: int = 160,
) -> Dict[str, Any]:
    """Sanitize context so error reporting does not leak sensitive data.

    The function preserves stable identifiers and operational metadata while
    summarizing or redacting freeform payloads.
    """

    if not context:
        return {}

    def _sanitize(value: Any, *, depth: int, parent_key: Optional[str] = None) -> Any:
        if depth > max_depth:
            return {"__trimmed__": True, "reason": "max_depth_exceeded"}

        if parent_key and _is_sensitive_key(parent_key):
            return _redacted_summary(value)

        scalar = _safe_scalar(value, max_string_length=max_string_length)
        if scalar is not _UNSERIALIZABLE:
            return scalar

        if isinstance(value, Mapping):
            items = list(value.items())
            sanitized: Dict[str, Any] = {}
            for index, (key, nested_value) in enumerate(items):
                if index >= max_items:
                    sanitized["__truncated_items__"] = len(items) - max_items
                    break
                key_str = str(key)
                sanitized[key_str] = _sanitize(nested_value, depth=depth + 1, parent_key=key_str)
            return sanitized

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            result = []
            sequence = list(value)
            for index, item in enumerate(sequence):
                if index >= max_items:
                    result.append({"__truncated_items__": len(sequence) - max_items})
                    break
                result.append(_sanitize(item, depth=depth + 1, parent_key=parent_key))
            return result

        return {
            "__type__": type(value).__name__,
            "repr": repr(value)[:max_string_length],
        }

    return {
        str(key): _sanitize(value, depth=0, parent_key=str(key))
        for key, value in context.items()
    }


def _safe_exception_message(exc: BaseException, *, max_length: int = 180) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) <= max_length:
        return message
    return f"{message[: max_length - 3]}..."


def _serialize_cause(exc: BaseException, *, include_traceback: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": exc.__class__.__name__,
        "message": _safe_exception_message(exc),
    }
    if include_traceback:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        payload["traceback"] = tb[-1500:]
    return payload


def set_privacy_audit_sink(callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    global _audit_sink
    if callback is not None and not callable(callback):
        raise TypeError("Privacy audit sink must be callable or None.")
    with _sink_lock:
        _audit_sink = callback


def set_privacy_metrics_sink(callback: Optional[Callable[[str, str, int], None]]) -> None:
    global _metrics_sink
    if callback is not None and not callable(callback):
        raise TypeError("Privacy metrics sink must be callable or None.")
    with _sink_lock:
        _metrics_sink = callback


def clear_privacy_sinks() -> None:
    set_privacy_audit_sink(None)
    set_privacy_metrics_sink(None)


@dataclass
class PrivacyError(Exception):
    message: str
    error_type: PrivacyErrorType
    severity: Optional[PrivacySeverity] = None
    retryable: Optional[bool] = None
    decision: Optional[PrivacyDecision] = None
    domain: Optional[PrivacyDomain] = None
    context: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None
    request_id: Optional[str] = None
    subject_id: Optional[str] = None
    record_id: Optional[str] = None
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    audit_trail_ref: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    incident_id: str = field(default_factory=_generate_incident_id)
    cause: Optional[BaseException] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        spec = get_privacy_error_spec(self.error_type)
        if self.severity is None:
            self.severity = spec.severity
        if self.retryable is None:
            self.retryable = spec.retryable
        if self.decision is None:
            self.decision = spec.decision
        if self.domain is None:
            self.domain = spec.domain
        if self.remediation is None:
            self.remediation = spec.remediation
        self.context = sanitize_privacy_context(self.context)
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.error_code} {self.message}"

    @property
    def error_code(self) -> str:
        return PRIVACY_ERROR_CODES.get(self.error_type, "PRV-9999")

    def add_context(self, **extra_context: Any) -> "PrivacyError":
        self.context = {
            **self.context,
            **sanitize_privacy_context(extra_context),
        }
        return self

    def to_dict(
        self,
        *,
        include_cause: bool = False,
        include_traceback: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "incident_id": self.incident_id,
            "error_code": self.error_code,
            "error_type": self.error_type.value,
            "domain": self.domain.value,
            "severity": self.severity.value,
            "decision": self.decision.value,
            "retryable": self.retryable,
            "message": self.message,
            "remediation": self.remediation,
            "request_id": self.request_id,
            "subject_id": self.subject_id,
            "record_id": self.record_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "audit_trail_ref": self.audit_trail_ref,
            "context": self.context,
            "timestamp": self.timestamp,
        }
        if include_cause and self.cause is not None:
            payload["cause"] = _serialize_cause(self.cause, include_traceback=include_traceback)
        return payload

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "error_code": self.error_code,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "decision": self.decision.value,
            "message": self.message,
            "remediation": self.remediation,
            "timestamp": self.timestamp,
        }

    def report(
        self,
        *,
        raise_on_failure: bool = False,
        include_cause: bool = False,
        include_traceback: bool = False,
    ) -> Dict[str, bool]:
        payload = self.to_dict(include_cause=include_cause, include_traceback=include_traceback)
        result = {"audit": False, "metrics": False}

        with _sink_lock:
            audit_sink = _audit_sink
            metrics_sink = _metrics_sink

        if audit_sink is not None:
            try:
                audit_sink(payload)
                result["audit"] = True
            except Exception as sink_exc:
                if raise_on_failure:
                    raise AuditLogWriteError(
                        channel="audit_sink",
                        details=sink_exc,
                        context={"failed_incident_id": self.incident_id},
                    ) from sink_exc

        if metrics_sink is not None:
            try:
                metrics_sink(self.error_code, self.severity.value, 1)
                result["metrics"] = True
            except Exception as sink_exc:
                if raise_on_failure:
                    raise AuditLogWriteError(
                        channel="metrics_sink",
                        details=sink_exc,
                        context={"failed_incident_id": self.incident_id},
                    ) from sink_exc

        return result


class DataClassificationPolicyError(PrivacyError):
    def __init__(
        self,
        section: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        policy_version: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Data classification policy is unavailable or invalid for section '{section}'.",
            error_type=PrivacyErrorType.DATA_CLASSIFICATION_POLICY_MISSING,
            policy_version=policy_version,
            context={"section": section, "details": _safe_exception_message(Exception(str(details))) if details else None, **(context or {})},
        )


class RedactionError(PrivacyError):
    def __init__(
        self,
        operation: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Redaction or masking failed during operation '{operation}'.",
            error_type=PrivacyErrorType.REDACTION_FAILED,
            request_id=request_id,
            context={"operation": operation, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class ToolPayloadSanitizationError(PrivacyError):
    def __init__(
        self,
        tool_name: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Tool payload sanitization failed for tool '{tool_name}'.",
            error_type=PrivacyErrorType.TOOL_PAYLOAD_SANITIZATION_FAILED,
            request_id=request_id,
            context={"tool_name": tool_name, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class ConsentValidationError(PrivacyError):
    def __init__(
        self,
        subject_id: str,
        purpose: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Consent validation failed for subject '{subject_id}' and purpose '{purpose}'.",
            error_type=PrivacyErrorType.CONSENT_VALIDATION_FAILED,
            subject_id=subject_id,
            policy_id=policy_id,
            policy_version=policy_version,
            audit_trail_ref=audit_trail_ref,
            context={"purpose": purpose, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class ConsentArtifactMissingError(PrivacyError):
    def __init__(
        self,
        subject_id: str,
        purpose: str,
        artifact_type: str = "consent_record",
        *,
        context: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Consent artifact is missing for subject '{subject_id}' and purpose '{purpose}'.",
            error_type=PrivacyErrorType.CONSENT_ARTIFACT_MISSING,
            subject_id=subject_id,
            policy_id=policy_id,
            context={
                "purpose": purpose,
                "artifact_type": artifact_type,
                **(context or {}),
            },
        )


class PurposeLimitationError(PrivacyError):
    def __init__(
        self,
        purpose: str,
        *,
        subject_id: Optional[str] = None,
        allowed_purposes: Optional[Sequence[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Purpose limitation violation detected for purpose '{purpose}'.",
            error_type=PrivacyErrorType.PURPOSE_LIMITATION_VIOLATION,
            subject_id=subject_id,
            policy_id=policy_id,
            policy_version=policy_version,
            context={
                "purpose": purpose,
                "allowed_purposes": list(allowed_purposes or []),
                **(context or {}),
            },
        )


class CrossContextSharingError(PrivacyError):
    def __init__(
        self,
        source_context: str,
        destination_context: str,
        *,
        subject_id: Optional[str] = None,
        purpose: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=(
                f"Cross-context sharing blocked from '{source_context}' to '{destination_context}'."
            ),
            error_type=PrivacyErrorType.CROSS_CONTEXT_SHARING_BLOCKED,
            subject_id=subject_id,
            context={
                "source_context": source_context,
                "destination_context": destination_context,
                "purpose": purpose,
                **(context or {}),
            },
        )


class PrivacyMemoryError(PrivacyError):
    def __init__(
        self,
        operation: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(
            message=f"privacy_memory operation '{operation}' failed.",
            error_type=PrivacyErrorType.PRIVACY_MEMORY_UNAVAILABLE,
            request_id=request_id,
            retryable=retryable,
            context={"operation": operation, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class PrivacyMemoryWriteError(PrivacyError):
    def __init__(
        self,
        operation: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"privacy_memory write failed during operation '{operation}'.",
            error_type=PrivacyErrorType.PRIVACY_MEMORY_WRITE_FAILED,
            request_id=request_id,
            context={"operation": operation, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class RetentionViolationError(PrivacyError):
    def __init__(
        self,
        *,
        record_id: str,
        policy_id: str,
        retention_days: int,
        age_days: int,
        context: Optional[Dict[str, Any]] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=(
                f"Retention policy violation for record '{record_id}' "
                f"(policy={policy_id}, age_days={age_days}, retention_days={retention_days})."
            ),
            error_type=PrivacyErrorType.RETENTION_POLICY_VIOLATION,
            record_id=record_id,
            policy_id=policy_id,
            policy_version=policy_version,
            audit_trail_ref=audit_trail_ref,
            context={
                "retention_days": retention_days,
                "age_days": age_days,
                **(context or {}),
            },
        )


class RetentionObligationMissingError(PrivacyError):
    def __init__(
        self,
        *,
        record_id: str,
        details: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=f"Retention obligation is missing for record '{record_id}'.",
            error_type=PrivacyErrorType.RETENTION_OBLIGATION_MISSING,
            record_id=record_id,
            context={"details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class DeletionSlaViolationError(PrivacyError):
    def __init__(
        self,
        *,
        record_id: str,
        due_timestamp: float,
        status: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=f"Deletion SLA violation detected for record '{record_id}'.",
            error_type=PrivacyErrorType.DELETION_SLA_VIOLATION,
            record_id=record_id,
            context={
                "due_timestamp": due_timestamp,
                "status": status,
                **(context or {}),
            },
        )


class DeletionWorkflowError(PrivacyError):
    def __init__(
        self,
        *,
        record_id: str,
        workflow: str,
        details: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=f"Deletion workflow '{workflow}' failed for record '{record_id}'.",
            error_type=PrivacyErrorType.DELETION_WORKFLOW_FAILED,
            record_id=record_id,
            context={
                "workflow": workflow,
                "details": _safe_exception_message(Exception(str(details))),
                **(context or {}),
            },
        )


class DataExportBlockedError(PrivacyError):
    def __init__(
        self,
        destination: str,
        reason: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Data export blocked for destination '{destination}'.",
            error_type=PrivacyErrorType.DATA_EXPORT_BLOCKED,
            policy_id=policy_id,
            context={"destination": destination, "reason": reason, **(context or {})},
        )


class CrossBorderTransferError(PrivacyError):
    def __init__(
        self,
        from_region: str,
        to_region: str,
        legal_basis: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=(
                f"Cross-border transfer blocked: {from_region}->{to_region} "
                f"(legal_basis={legal_basis})."
            ),
            error_type=PrivacyErrorType.CROSS_BORDER_TRANSFER_BLOCKED,
            policy_id=policy_id,
            policy_version=policy_version,
            context={
                "from_region": from_region,
                "to_region": to_region,
                "legal_basis": legal_basis,
                **(context or {}),
            },
        )


class AuditLogWriteError(PrivacyError):
    def __init__(
        self,
        channel: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Privacy audit log write failed for channel '{channel}'.",
            error_type=PrivacyErrorType.AUDIT_LOG_WRITE_FAILED,
            request_id=request_id,
            context={"channel": channel, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class AuditReportGenerationError(PrivacyError):
    def __init__(
        self,
        report_name: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Privacy audit report generation failed for report '{report_name}'.",
            error_type=PrivacyErrorType.AUDIT_REPORT_GENERATION_FAILED,
            request_id=request_id,
            context={"report_name": report_name, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class AuditEvidenceGenerationError(PrivacyError):
    def __init__(
        self,
        bundle_name: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Privacy audit evidence generation failed for bundle '{bundle_name}'.",
            error_type=PrivacyErrorType.AUDIT_EVIDENCE_GENERATION_FAILED,
            request_id=request_id,
            context={"bundle_name": bundle_name, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class EncryptionPolicyViolationError(PrivacyError):
    def __init__(
        self,
        resource_id: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=f"Encryption policy violation detected for resource '{resource_id}'.",
            error_type=PrivacyErrorType.ENCRYPTION_POLICY_VIOLATION,
            record_id=resource_id,
            policy_id=policy_id,
            context={"details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class PrivacyConfigurationError(PrivacyError):
    def __init__(
        self,
        section: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=f"Privacy configuration error detected in section '{section}'.",
            error_type=PrivacyErrorType.PRIVACY_CONFIGURATION_ERROR,
            context={"section": section, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
        )


class PolicyEvaluationError(PrivacyError):
    def __init__(
        self,
        stage: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message=f"Privacy policy evaluation failed during stage '{stage}'.",
            error_type=PrivacyErrorType.POLICY_EVALUATION_FAILED,
            request_id=request_id,
            policy_id=policy_id,
            context={"stage": stage, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
            cause=cause,
        )


class PrivacyOperationTimeoutError(PrivacyError):
    def __init__(
        self,
        stage: str,
        timeout_seconds: Optional[float] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message=f"Privacy operation timed out during stage '{stage}'.",
            error_type=PrivacyErrorType.PRIVACY_OPERATION_TIMEOUT,
            request_id=request_id,
            context={"stage": stage, "timeout_seconds": timeout_seconds, **(context or {})},
            cause=cause,
        )


class InternalPrivacyError(PrivacyError):
    def __init__(
        self,
        stage: str,
        details: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            message=f"Unhandled internal privacy error during stage '{stage}'.",
            error_type=PrivacyErrorType.INTERNAL_PRIVACY_ERROR,
            request_id=request_id,
            context={"stage": stage, "details": _safe_exception_message(Exception(str(details))), **(context or {})},
            cause=cause,
        )


def _default_error_type_for_stage(stage: str) -> PrivacyErrorType:
    stage_key = stage.strip().lower()
    if any(token in stage_key for token in ("detect", "classif", "entity", "pii", "phi")):
        return PrivacyErrorType.PII_DETECTION_FAILED
    if any(token in stage_key for token in ("redact", "mask", "minimiz", "sanitize")):
        return PrivacyErrorType.REDACTION_FAILED
    if any(token in stage_key for token in ("consent", "purpose", "sharing")):
        return PrivacyErrorType.CONSENT_VALIDATION_FAILED
    if any(token in stage_key for token in ("retention", "delete", "deletion", "forget")):
        return PrivacyErrorType.POLICY_EVALUATION_FAILED
    if any(token in stage_key for token in ("transfer", "export", "egress", "share")):
        return PrivacyErrorType.DATA_EXPORT_BLOCKED
    if any(token in stage_key for token in ("memory", "lineage", "trace")):
        return PrivacyErrorType.PRIVACY_MEMORY_UNAVAILABLE
    if any(token in stage_key for token in ("audit", "evidence", "log")):
        return PrivacyErrorType.AUDIT_LOG_WRITE_FAILED
    if any(token in stage_key for token in ("encrypt", "key", "cipher")):
        return PrivacyErrorType.ENCRYPTION_POLICY_VIOLATION
    if any(token in stage_key for token in ("config", "policy_pack", "yaml")):
        return PrivacyErrorType.PRIVACY_CONFIGURATION_ERROR
    return PrivacyErrorType.INTERNAL_PRIVACY_ERROR


def normalize_privacy_exception(
    exc: Exception,
    *,
    stage: str,
    context: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    policy_id: Optional[str] = None,
    policy_version: Optional[str] = None,
) -> PrivacyError:
    """Normalize an arbitrary exception into a privacy-aware error.

    The mapper is intentionally conservative:
    - native PrivacyError instances are preserved,
    - timeout and configuration failures receive explicit typing,
    - generic failures are categorized by stage instead of being collapsed into audit errors.
    """

    if isinstance(exc, PrivacyError):
        if context:
            exc.add_context(stage=stage, **context)
        else:
            exc.add_context(stage=stage)
        if request_id and not exc.request_id:
            exc.request_id = request_id
        if policy_id and not exc.policy_id:
            exc.policy_id = policy_id
        if policy_version and not exc.policy_version:
            exc.policy_version = policy_version
        return exc

    merged_context = {
        "stage": stage,
        "source_exception_type": exc.__class__.__name__,
        "source_exception_message": _safe_exception_message(exc),
        **(context or {}),
    }

    if isinstance(exc, TimeoutError):
        return PrivacyOperationTimeoutError(
            stage=stage,
            context=merged_context,
            request_id=request_id,
            cause=exc,
        )

    if isinstance(exc, (FileNotFoundError, KeyError, ImportError)):
        return PrivacyError(
            message=f"Privacy configuration error detected in section '{stage}'.",
            error_type=PrivacyErrorType.PRIVACY_CONFIGURATION_ERROR,
            request_id=request_id,
            policy_id=policy_id,
            policy_version=policy_version,
            context=merged_context,
            cause=exc,
        )

    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return PolicyEvaluationError(
            stage=stage,
            details=exc,
            context=merged_context,
            request_id=request_id,
            policy_id=policy_id,
            cause=exc,
        )

    if isinstance(exc, PermissionError):
        return PrivacyError(
            message=f"Permission denied during privacy stage '{stage}'.",
            error_type=_default_error_type_for_stage(stage),
            request_id=request_id,
            policy_id=policy_id,
            policy_version=policy_version,
            context=merged_context,
            cause=exc,
        )

    return PrivacyError(
        message=f"Unhandled exception during privacy stage '{stage}'.",
        error_type=_default_error_type_for_stage(stage),
        request_id=request_id,
        policy_id=policy_id,
        policy_version=policy_version,
        context=merged_context,
        cause=exc,
    )
