from __future__ import annotations

import hashlib
import json
import time
import traceback

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EvaluationErrorType(str, Enum):
    """Canonical error categories used across the evaluator stack."""

    METRIC_CALCULATION = "Metric Calculation Failure"
    REPORT_GENERATION = "Report Generation Failure"
    CONFIG_LOAD = "Configuration Loading Error"
    VALIDATION_FAILURE = "Validation Rule Violation"
    DATA_INTEGRITY = "Evaluation Data Integrity Error"
    MEMORY_ACCESS = "Evaluation Memory Access Failure"
    VISUALIZATION = "Result Visualization Error"
    THRESHOLD_VIOLATION = "Quality Threshold Violation"
    COMPARISON_FAILURE = "Comparative Analysis Failure"
    TEMPLATE_ERROR = "Report Template Processing Error"

    CHECKPOINT_FAILURE = "Checkpoint Processing Failure"
    SERIALIZATION_FAILURE = "Serialization Failure"
    DEPENDENCY_FAILURE = "Dependency Resolution Failure"
    MODEL_RUNTIME = "Model Runtime Failure"
    TRANSFORMER_RUNTIME = "Transformer Runtime Failure"
    STATISTICAL_ANALYSIS = "Statistical Analysis Failure"
    STATIC_ANALYSIS = "Static Analysis Failure"
    SECURITY_ANALYSIS = "Security Analysis Failure"
    CERTIFICATION_FAILURE = "Certification Processing Failure"
    DOCUMENTATION_FAILURE = "Documentation Processing Failure"
    RESOURCE_EXHAUSTION = "Resource Exhaustion"
    CONCURRENCY_FAILURE = "Concurrency Failure"
    UNSUPPORTED_OPERATION = "Unsupported Operation"


class ErrorSeverity(str, Enum):
    """Normalized severity levels for operational triage."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CertificationLevel(str, Enum):
    """Stable certification stages used by the framework."""

    DEVELOPMENT = "DEVELOPMENT"
    PILOT = "PILOT"
    DEPLOYMENT = "DEPLOYMENT"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Structured certification- and documentation-specific exceptions
# ---------------------------------------------------------------------------


class _StructuredDomainErrorMixin:
    """
    Lightweight structured metadata support for certification and documentation
    exception families.

    This mixin intentionally stays narrower than ``EvaluationError`` while still
    making leaf exceptions more useful in logs, reports, audits, and tests.
    """

    domain_family: str = "domain"
    error_code: str = "DOMAIN_ERROR"

    def _initialize_domain_error(
        self,
        message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string.")

        Exception.__init__(self, message.strip())
        self.context = self._sanitize_mapping(context or {})
        self.cause = cause
        self.timestamp = time.time()
        self.timestamp_iso = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        self.error_code = str(getattr(self, "error_code", self.__class__.__name__)).strip() or self.__class__.__name__
        self.domain_family = str(getattr(self, "domain_family", "domain")).strip() or "domain"
        self.error_id = self._generate_error_id()

    @classmethod
    def _sanitize_mapping(cls, value: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {"_raw": cls._stringify(value)}
        return {str(key): cls._coerce_json_safe(item) for key, item in value.items()}

    @classmethod
    def _coerce_json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {str(key): cls._coerce_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._coerce_json_safe(item) for item in value]
        if isinstance(value, BaseException):
            return {
                "exception_type": value.__class__.__name__,
                "message": str(value),
            }
        return cls._stringify(value)

    @staticmethod
    def _stringify(value: Any, max_length: int = 500) -> str:
        text = str(value)
        return text if len(text) <= max_length else text[: max_length - 3] + "..."

    def _generate_error_id(self) -> str:
        payload = {
            "timestamp": round(self.timestamp, 6),
            "family": self.domain_family,
            "type": self.__class__.__name__,
            "message": str(self),
            "context": self.context,
        }
        return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "family": self.domain_family,
            "type": self.__class__.__name__,
            "message": str(self),
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "context": dict(self.context),
            "cause": {
                "type": self.cause.__class__.__name__,
                "message": str(self.cause),
            } if self.cause else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def with_context(self, **kwargs: Any) -> "_StructuredDomainErrorMixin":
        merged = dict(self.context)
        for key, value in kwargs.items():
            merged[str(key)] = self._coerce_json_safe(value)
        self.context = merged
        self.error_id = self._generate_error_id()
        return self


# ---------------------------------------------------------------------------
# Certification-specific exceptions
# ---------------------------------------------------------------------------


class CertificationFrameworkError(Exception):
    """Base exception for certification framework failures."""


class CertificationConfigurationError(CertificationFrameworkError, _StructuredDomainErrorMixin):
    """Raised when the certification configuration is invalid or incomplete."""

    domain_family = "certification"
    error_code = "CERTIFICATION_CONFIGURATION_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        section: Optional[str] = None,
        path: Optional[str] = None,
        field: Optional[str] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            scope = field or section or "certification configuration"
            message = f"Invalid or incomplete certification configuration for '{scope}'."
            if details:
                message = f"{message} {details}"
        self.section = section
        self.path = path
        self.field = field
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "section": section,
                "path": path,
                "field": field,
                "details": details,
            },
            cause=cause,
        )


class TemplateLoadError(CertificationFrameworkError, _StructuredDomainErrorMixin):
    """Raised when certification templates cannot be loaded or parsed."""

    domain_family = "certification"
    error_code = "CERTIFICATION_TEMPLATE_LOAD_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        template_path: Optional[str] = None,
        domain: Optional[str] = None,
        level: Optional[str] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            target = f"template '{template_path}'" if template_path else "certification template"
            message = f"Failed to load {target}."
            if domain:
                message = f"{message} Domain: {domain}."
            if level:
                message = f"{message} Level: {level}."
            if details:
                message = f"{message} {details}"
        self.template_path = template_path
        self.domain = domain
        self.level = level
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "template_path": template_path,
                "domain": domain,
                "level": level,
                "details": details,
            },
            cause=cause,
        )


class RequirementDefinitionError(CertificationFrameworkError, _StructuredDomainErrorMixin):
    """Raised when a requirement definition is malformed."""

    domain_family = "certification"
    error_code = "CERTIFICATION_REQUIREMENT_DEFINITION_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        requirement_id: Optional[str] = None,
        field_name: Optional[str] = None,
        level: Optional[str] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            target = requirement_id or "requirement definition"
            message = f"Malformed certification requirement '{target}'."
            if field_name:
                message = f"{message} Invalid field: {field_name}."
            if level:
                message = f"{message} Level: {level}."
            if details:
                message = f"{message} {details}"
        self.requirement_id = requirement_id
        self.field_name = field_name
        self.level = level
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "requirement_id": requirement_id,
                "field_name": field_name,
                "level": level,
                "details": details,
            },
            cause=cause,
        )


class EvidenceValidationError(CertificationFrameworkError, _StructuredDomainErrorMixin):
    """Raised when evidence submitted to the framework is invalid."""

    domain_family = "certification"
    error_code = "CERTIFICATION_EVIDENCE_VALIDATION_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        evidence_id: Optional[str] = None,
        evidence_type: Optional[str] = None,
        requirement_id: Optional[str] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            target = evidence_id or "submitted evidence"
            message = f"Invalid certification evidence '{target}'."
            if evidence_type:
                message = f"{message} Type: {evidence_type}."
            if requirement_id:
                message = f"{message} Requirement: {requirement_id}."
            if details:
                message = f"{message} {details}"
        self.evidence_id = evidence_id
        self.evidence_type = evidence_type
        self.requirement_id = requirement_id
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "evidence_id": evidence_id,
                "evidence_type": evidence_type,
                "requirement_id": requirement_id,
                "details": details,
            },
            cause=cause,
        )


class CertificationEvaluationError(CertificationFrameworkError, _StructuredDomainErrorMixin):
    """Raised when certification evaluation cannot be completed."""

    domain_family = "certification"
    error_code = "CERTIFICATION_EVALUATION_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        level: Optional[str] = None,
        domain: Optional[str] = None,
        unmet_requirements: Optional[Sequence[str]] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        unmet = [str(item) for item in (unmet_requirements or []) if str(item).strip()]
        if message is None:
            message = "Certification evaluation could not be completed."
            if domain:
                message = f"{message} Domain: {domain}."
            if level:
                message = f"{message} Level: {level}."
            if unmet:
                message = f"{message} Unmet requirements: {', '.join(unmet)}."
            if details:
                message = f"{message} {details}"
        self.level = level
        self.domain = domain
        self.unmet_requirements = unmet
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "level": level,
                "domain": domain,
                "unmet_requirements": unmet,
                "details": details,
            },
            cause=cause,
        )


class CertificationStateError(CertificationFrameworkError, _StructuredDomainErrorMixin):
    """Raised when certification workflow state is invalid for an operation."""

    domain_family = "certification"
    error_code = "CERTIFICATION_STATE_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        current_state: Optional[str] = None,
        operation: Optional[str] = None,
        expected_state: Optional[str | Sequence[str]] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if isinstance(expected_state, (list, tuple, set)):
            expected = [str(item) for item in expected_state if str(item).strip()]
        elif expected_state is None:
            expected = []
        else:
            expected = [str(expected_state)]
        if message is None:
            action = operation or "operation"
            state = current_state or "<unknown>"
            message = f"Certification workflow state '{state}' is invalid for '{action}'."
            if expected:
                message = f"{message} Expected state(s): {', '.join(expected)}."
            if details:
                message = f"{message} {details}"
        self.current_state = current_state
        self.operation = operation
        self.expected_state = expected
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "current_state": current_state,
                "operation": operation,
                "expected_state": expected,
                "details": details,
            },
            cause=cause,
        )


# ---------------------------------------------------------------------------
# Documentation- and audit-specific exceptions
# ---------------------------------------------------------------------------


class DocumentationError(Exception):
    """Base exception for documentation and audit-trail failures."""


class DocumentationConfigurationError(DocumentationError, _StructuredDomainErrorMixin):
    """Raised when documentation configuration is invalid or incomplete."""

    domain_family = "documentation"
    error_code = "DOCUMENTATION_CONFIGURATION_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        section: Optional[str] = None,
        path: Optional[str] = None,
        field: Optional[str] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            scope = field or section or "documentation configuration"
            message = f"Invalid or incomplete documentation configuration for '{scope}'."
            if details:
                message = f"{message} {details}"
        self.section = section
        self.path = path
        self.field = field
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "section": section,
                "path": path,
                "field": field,
                "details": details,
            },
            cause=cause,
        )


class SchemaLoadError(DocumentationError, _StructuredDomainErrorMixin):
    """Raised when a configured validation schema cannot be loaded."""

    domain_family = "documentation"
    error_code = "DOCUMENTATION_SCHEMA_LOAD_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        schema_path: Optional[str] = None,
        schema_name: Optional[str] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            target = schema_name or schema_path or "validation schema"
            message = f"Failed to load schema '{target}'."
            if details:
                message = f"{message} {details}"
        self.schema_path = schema_path
        self.schema_name = schema_name
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "schema_path": schema_path,
                "schema_name": schema_name,
                "details": details,
            },
            cause=cause,
        )


class InvalidDocumentError(DocumentationError, _StructuredDomainErrorMixin):
    """Raised when a document fails validation or normalization."""

    domain_family = "documentation"
    error_code = "DOCUMENTATION_INVALID_DOCUMENT_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        field_name: Optional[str] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            target = document_id or "document"
            message = f"Invalid or malformed document '{target}'."
            if document_type:
                message = f"{message} Type: {document_type}."
            if field_name:
                message = f"{message} Field: {field_name}."
            if details:
                message = f"{message} {details}"
        self.document_id = document_id
        self.document_type = document_type
        self.field_name = field_name
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "document_id": document_id,
                "document_type": document_type,
                "field_name": field_name,
                "details": details,
            },
            cause=cause,
        )


class AuditIntegrityError(DocumentationError, _StructuredDomainErrorMixin):
    """Raised when the audit chain fails integrity verification."""

    domain_family = "documentation"
    error_code = "DOCUMENTATION_AUDIT_INTEGRITY_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        block_index: Optional[int] = None,
        expected_hash: Optional[str] = None,
        observed_hash: Optional[str] = None,
        chain_length: Optional[int] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            target = f"block {block_index}" if block_index is not None else "audit chain"
            message = f"Audit integrity check failed for {target}."
            if expected_hash is not None or observed_hash is not None:
                message = f"{message} Expected hash: {expected_hash}; observed hash: {observed_hash}."
            if details:
                message = f"{message} {details}"
        self.block_index = block_index
        self.expected_hash = expected_hash
        self.observed_hash = observed_hash
        self.chain_length = chain_length
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "block_index": block_index,
                "expected_hash": expected_hash,
                "observed_hash": observed_hash,
                "chain_length": chain_length,
                "details": details,
            },
            cause=cause,
        )


class UnsupportedExportFormatError(DocumentationError, _StructuredDomainErrorMixin):
    """Raised when an unsupported export format is requested."""

    domain_family = "documentation"
    error_code = "DOCUMENTATION_UNSUPPORTED_EXPORT_FORMAT_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        requested_format: Optional[str] = None,
        supported_formats: Optional[Sequence[str]] = None,
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        supported = [str(item) for item in (supported_formats or []) if str(item).strip()]
        if message is None:
            message = f"Unsupported export format requested: {requested_format or '<unknown>'}."
            if supported:
                message = f"{message} Supported formats: {', '.join(supported)}."
            if details:
                message = f"{message} {details}"
        self.requested_format = requested_format
        self.supported_formats = supported
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "requested_format": requested_format,
                "supported_formats": supported,
                "details": details,
            },
            cause=cause,
        )


class CheckpointLoadError(DocumentationError, _StructuredDomainErrorMixin):
    """Raised when a checkpoint or persisted state cannot be restored."""

    domain_family = "documentation"
    error_code = "DOCUMENTATION_CHECKPOINT_LOAD_ERROR"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        checkpoint_path: Optional[str] = None,
        operation: str = "load",
        details: Optional[str] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        if message is None:
            target = checkpoint_path or "checkpoint"
            message = f"Failed to {operation} persisted state from '{target}'."
            if details:
                message = f"{message} {details}"
        self.checkpoint_path = checkpoint_path
        self.operation = operation
        self.details = details
        self._initialize_domain_error(
            message,
            context={
                "checkpoint_path": checkpoint_path,
                "operation": operation,
                "details": details,
            },
            cause=cause,
        )


# ---------------------------------------------------------------------------
# Core evaluation error type
# ---------------------------------------------------------------------------


class EvaluationError(Exception):
    """
    Base exception for evaluator stack failures with forensic metadata.

    The class is intentionally richer than a plain exception so the same object
    can be used for:
    - runtime control flow
    - structured logging
    - checkpoint/audit persistence
    - dashboard and report serialization
    """

    def __init__(
        self,
        error_type: EvaluationErrorType,
        message: str,
        severity: str | ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Mapping[str, Any]] = None,
        agent_state: Optional[Mapping[str, Any]] = None,
        remediation: Optional[str] = None,
        *,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        cause: Optional[BaseException] = None,
        retryable: bool = False,
    ) -> None:
        if not isinstance(error_type, EvaluationErrorType):
            raise TypeError("error_type must be an EvaluationErrorType.")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string.")

        super().__init__(message.strip())

        self.error_type = error_type
        self.severity = self._normalize_severity(severity)
        self.context = self._sanitize_mapping(context or {})
        self.agent_state = self._sanitize_mapping(agent_state or {})
        self.remediation = remediation.strip() if isinstance(remediation, str) and remediation.strip() else None
        self.component = component.strip() if isinstance(component, str) and component.strip() else None
        self.operation = operation.strip() if isinstance(operation, str) and operation.strip() else None
        self.tags = self._normalize_tags(tags or [])
        self.retryable = bool(retryable)
        self.cause = cause

        self.timestamp = time.time()
        self.timestamp_iso = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        self.error_id = self._generate_error_id()
        self.forensic_hash = self._generate_forensic_hash()

    @staticmethod
    def _normalize_severity(value: str | ErrorSeverity) -> ErrorSeverity:
        if isinstance(value, ErrorSeverity):
            return value
        normalized = str(value).strip().lower()
        for candidate in ErrorSeverity:
            if candidate.value == normalized:
                return candidate
        raise ValueError(f"Unsupported severity level: {value}")

    @staticmethod
    def _normalize_tags(tags: Iterable[Any]) -> List[str]:
        normalized: List[str] = []
        seen: set[str] = set()
        for item in tags:
            if item is None:
                continue
            value = str(item).strip()
            if not value:
                continue
            key = value.casefold()
            if key not in seen:
                normalized.append(value)
                seen.add(key)
        return normalized

    @classmethod
    def _sanitize_mapping(cls, value: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {"_raw": cls._stringify(value)}
        return {str(key): cls._coerce_json_safe(item) for key, item in value.items()}

    @classmethod
    def _coerce_json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {str(key): cls._coerce_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._coerce_json_safe(item) for item in value]
        if isinstance(value, BaseException):
            return {
                "exception_type": value.__class__.__name__,
                "message": str(value),
            }
        return cls._stringify(value)

    @staticmethod
    def _stringify(value: Any, max_length: int = 500) -> str:
        text = str(value)
        return text if len(text) <= max_length else text[: max_length - 3] + "..."

    def _generate_error_id(self) -> str:
        payload = {
            "timestamp": round(self.timestamp, 6),
            "type": self.error_type.value,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "context": self.context,
        }
        encoded = _stable_json_dumps(payload)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    def _generate_forensic_hash(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "error_id": self.error_id,
            "type": self.error_type.value,
            "severity": self.severity.value,
            "message": str(self),
            "component": self.component,
            "operation": self.operation,
            "context": self.context,
            "agent_state": self.agent_state,
            "remediation": self.remediation,
            "retryable": self.retryable,
            "tags": self.tags,
            "cause": {
                "type": self.cause.__class__.__name__,
                "message": str(self.cause),
            } if self.cause else None,
        }
        return hashlib.sha3_256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()

    def to_dict(self, include_traceback: bool = False) -> Dict[str, Any]:
        payload = {
            "error_id": self.error_id,
            "type": self.error_type.value,
            "severity": self.severity.value,
            "message": str(self),
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "forensic_hash": self.forensic_hash,
            "component": self.component,
            "operation": self.operation,
            "retryable": self.retryable,
            "tags": list(self.tags),
            "context": dict(self.context),
            "agent_state_snapshot": dict(self.agent_state),
            "remediation": self.remediation,
            "cause": {
                "type": self.cause.__class__.__name__,
                "message": str(self.cause),
            } if self.cause else None,
        }
        if include_traceback and self.__traceback__ is not None:
            payload["traceback"] = "".join(traceback.format_exception(type(self), self, self.__traceback__))
        return payload

    def to_audit_dict(self) -> Dict[str, Any]:
        """Structured representation for logging and auditing."""
        return self.to_dict(include_traceback=False)

    def to_json(self, include_traceback: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_traceback=include_traceback), indent=indent, sort_keys=False)

    def with_context(self, **kwargs: Any) -> "EvaluationError":
        merged_context = dict(self.context)
        for key, value in kwargs.items():
            merged_context[str(key)] = self._coerce_json_safe(value)
        self.context = merged_context
        self.forensic_hash = self._generate_forensic_hash()
        return self

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        error_type: EvaluationErrorType = EvaluationErrorType.DATA_INTEGRITY,
        severity: str | ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Mapping[str, Any]] = None,
        agent_state: Optional[Mapping[str, Any]] = None,
        remediation: Optional[str] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        retryable: bool = False,
    ) -> "EvaluationError":
        return cls(
            error_type=error_type,
            message=str(exc) or exc.__class__.__name__,
            severity=severity,
            context=context,
            agent_state=agent_state,
            remediation=remediation,
            component=component,
            operation=operation,
            tags=tags,
            cause=exc,
            retryable=retryable,
        )


# ---------------------------------------------------------------------------
# Evaluator stack specific error classes
# ---------------------------------------------------------------------------


class MetricCalculationError(EvaluationError):
    """Failure in metric calculation process."""

    def __init__(self, metric_name: str, inputs: Any, reason: str):
        super().__init__(
            EvaluationErrorType.METRIC_CALCULATION,
            f"Metric calculation failed for '{metric_name}': {reason}",
            severity=ErrorSeverity.HIGH,
            context={
                "metric": metric_name,
                "input_data": inputs,
                "failure_reason": reason,
            },
            remediation="Verify input data format, sample validity, and calculation parameters.",
            component="evaluators_calculations",
            operation="metric_calculation",
            tags=["metrics", metric_name],
        )


class StatisticalAnalysisError(EvaluationError):
    """Failure during statistical analysis or inference routines."""

    def __init__(self, analysis_name: str, inputs: Any, reason: str):
        super().__init__(
            EvaluationErrorType.STATISTICAL_ANALYSIS,
            f"Statistical analysis failed for '{analysis_name}': {reason}",
            severity=ErrorSeverity.HIGH,
            context={
                "analysis": analysis_name,
                "inputs": inputs,
                "failure_reason": reason,
            },
            remediation="Inspect dataset size, variance, and numeric validity before retrying the analysis.",
            component="evaluators_calculations",
            operation="statistical_analysis",
            tags=["statistics", analysis_name],
        )


class ReportGenerationError(EvaluationError):
    """Failure in report generation process."""

    def __init__(self, report_type: str, template: str, error_details: str):
        super().__init__(
            EvaluationErrorType.REPORT_GENERATION,
            f"Report generation failed for {report_type} using template {template}",
            severity=ErrorSeverity.MEDIUM,
            context={
                "report_type": report_type,
                "template": template,
                "error": error_details,
            },
            remediation="Check template structure, input data completeness, and report serialization logic.",
            component="report",
            operation="report_generation",
            tags=["reporting", report_type],
        )


class ConfigLoadError(EvaluationError):
    """Failure in configuration loading or normalization."""

    def __init__(self, config_path: str, section: str, error_details: str):
        super().__init__(
            EvaluationErrorType.CONFIG_LOAD,
            f"Config loading failed for {section} in {config_path}",
            severity=ErrorSeverity.CRITICAL,
            context={
                "config_path": config_path,
                "section": section,
                "error": error_details,
            },
            remediation="Validate configuration file structure, field names, defaults, and file permissions.",
            component="configuration",
            operation="config_load",
            tags=["config", section],
        )


class ValidationFailureError(EvaluationError):
    """Violation of evaluation validation rules."""

    def __init__(self, rule_name: str, data: Any, expected: Any):
        super().__init__(
            EvaluationErrorType.VALIDATION_FAILURE,
            f"Validation rule '{rule_name}' violated",
            severity=ErrorSeverity.HIGH,
            context={
                "rule": rule_name,
                "actual_value": data,
                "expected_value": expected,
            },
            remediation="Review data contracts, preprocessing rules, and caller assumptions.",
            component="validation",
            operation="validation_check",
            tags=["validation", rule_name],
        )


class ThresholdViolationError(EvaluationError):
    """Violation of configured quality or resource thresholds."""

    def __init__(self, metric: str, value: float, threshold: float, comparator: str = "threshold"):
        message = (
            f"Quality threshold violation: {metric} (observed={value:.6g}, threshold={threshold:.6g}, comparator={comparator})"
        )
        super().__init__(
            EvaluationErrorType.THRESHOLD_VIOLATION,
            message,
            severity=ErrorSeverity.HIGH,
            context={
                "metric": metric,
                "actual_value": value,
                "threshold": threshold,
                "comparator": comparator,
            },
            remediation="Investigate metric drift, threshold calibration, and operational degradation.",
            component="thresholds",
            operation="threshold_evaluation",
            tags=["threshold", metric],
        )


class DataIntegrityError(EvaluationError):
    """Failure caused by corrupted, inconsistent, or incomplete runtime data."""

    def __init__(self, resource: str, error_details: str, context: Optional[Mapping[str, Any]] = None):
        merged_context = dict(context or {})
        merged_context.update({"resource": resource, "error": error_details})
        super().__init__(
            EvaluationErrorType.DATA_INTEGRITY,
            f"Data integrity failure in {resource}: {error_details}",
            severity=ErrorSeverity.HIGH,
            context=merged_context,
            remediation="Inspect the upstream data source, serialization process, and integrity checks.",
            component="data_integrity",
            operation="integrity_validation",
            tags=["data", resource],
        )


class MemoryAccessError(EvaluationError):
    """Failure in evaluation memory access."""

    def __init__(self, operation: str, key: str, error_details: str):
        super().__init__(
            EvaluationErrorType.MEMORY_ACCESS,
            f"Memory {operation} failed for key '{key}'",
            severity=ErrorSeverity.MEDIUM,
            context={
                "operation": operation,
                "key": key,
                "error": error_details,
            },
            remediation="Check memory indexing, storage synchronization, and serialization formats.",
            component="evaluators_memory",
            operation=operation,
            tags=["memory", operation],
        )


class CheckpointError(EvaluationError):
    """Failure during checkpoint creation, loading, or maintenance."""

    def __init__(self, operation: str, path: str, error_details: str):
        super().__init__(
            EvaluationErrorType.CHECKPOINT_FAILURE,
            f"Checkpoint {operation} failed for {path}",
            severity=ErrorSeverity.HIGH,
            context={
                "operation": operation,
                "path": path,
                "error": error_details,
            },
            remediation="Validate the checkpoint path, schema, permissions, and persistent-state compatibility.",
            component="evaluators_memory",
            operation=f"checkpoint_{operation}",
            tags=["checkpoint", operation],
        )


class SerializationError(EvaluationError):
    """Failure during serialization or deserialization of runtime artifacts."""

    def __init__(self, target: str, error_details: str, payload: Any = None):
        super().__init__(
            EvaluationErrorType.SERIALIZATION_FAILURE,
            f"Serialization failed for {target}: {error_details}",
            severity=ErrorSeverity.MEDIUM,
            context={
                "target": target,
                "error": error_details,
                "payload_sample": payload,
            },
            remediation="Ensure the payload is deterministic, JSON-safe, and compatible with the persistence format.",
            component="serialization",
            operation="serialize",
            tags=["serialization", target],
        )


class VisualizationError(EvaluationError):
    """Failure in results visualization."""

    def __init__(self, chart_type: str, data: Any, error_details: str):
        super().__init__(
            EvaluationErrorType.VISUALIZATION,
            f"Visualization failed for {chart_type} chart",
            severity=ErrorSeverity.LOW,
            context={
                "chart_type": chart_type,
                "data_sample": data,
                "error": error_details,
            },
            remediation="Verify data dimensions, chart configuration, and visualization dependencies.",
            component="visualization",
            operation="render_chart",
            tags=["visualization", chart_type],
        )


class ComparisonError(EvaluationError):
    """Failure in comparative analysis."""

    def __init__(self, baseline: str, current: str, error_details: str):
        super().__init__(
            EvaluationErrorType.COMPARISON_FAILURE,
            f"Comparison failed between {baseline} and {current}",
            severity=ErrorSeverity.MEDIUM,
            context={
                "baseline": baseline,
                "current": current,
                "error": error_details,
            },
            remediation="Check baseline availability, alignment logic, and comparison algorithm assumptions.",
            component="comparison",
            operation="compare",
            tags=["comparison", baseline, current],
        )


class TemplateError(EvaluationError):
    """Failure in report or artifact template processing."""

    def __init__(self, template_path: str, error_details: str):
        super().__init__(
            EvaluationErrorType.TEMPLATE_ERROR,
            f"Template processing failed for {template_path}",
            severity=ErrorSeverity.HIGH,
            context={
                "template_path": template_path,
                "error": error_details,
            },
            remediation="Validate template syntax, placeholders, schema, and source compatibility.",
            component="templating",
            operation="template_processing",
            tags=["template", template_path],
        )


class DependencyError(EvaluationError):
    """Failure caused by an unavailable or incompatible dependency."""

    def __init__(self, dependency: str, required_by: str, error_details: str = ""):
        message = f"Dependency '{dependency}' failed for {required_by}"
        if error_details:
            message = f"{message}: {error_details}"
        super().__init__(
            EvaluationErrorType.DEPENDENCY_FAILURE,
            message,
            severity=ErrorSeverity.HIGH,
            context={
                "dependency": dependency,
                "required_by": required_by,
                "error": error_details,
            },
            remediation="Install, configure, or version-align the required dependency before retrying.",
            component=required_by,
            operation="dependency_resolution",
            tags=["dependency", dependency],
        )


class ModelRuntimeError(EvaluationError):
    """Failure during model execution, loading, or inference."""

    def __init__(self, operation: str, model_name: str, error_details: str, context: Optional[Mapping[str, Any]] = None):
        merged_context = dict(context or {})
        merged_context.update({
            "operation": operation,
            "model_name": model_name,
            "error": error_details,
        })
        super().__init__(
            EvaluationErrorType.MODEL_RUNTIME,
            f"Model runtime failure during {operation} for {model_name}: {error_details}",
            severity=ErrorSeverity.HIGH,
            context=merged_context,
            remediation="Inspect model dimensions, checkpoint compatibility, runtime device selection, and input validity.",
            component=model_name,
            operation=operation,
            tags=["model", model_name, operation],
        )


class TransformerError(EvaluationError):
    """Failure in transformer-specific configuration or runtime logic."""

    def __init__(self, operation: str, error_details: str, context: Optional[Mapping[str, Any]] = None):
        merged_context = dict(context or {})
        merged_context.update({"operation": operation, "error": error_details})
        super().__init__(
            EvaluationErrorType.TRANSFORMER_RUNTIME,
            f"Transformer operation '{operation}' failed: {error_details}",
            severity=ErrorSeverity.HIGH,
            context=merged_context,
            remediation="Verify transformer configuration, tensor shapes, masks, and checkpoint compatibility.",
            component="evaluation_transformer",
            operation=operation,
            tags=["transformer", operation],
        )


class StaticAnalysisError(EvaluationError):
    """Failure while performing static code analysis."""

    def __init__(self, target: str, error_details: str, context: Optional[Mapping[str, Any]] = None):
        merged_context = dict(context or {})
        merged_context.update({"target": target, "error": error_details})
        super().__init__(
            EvaluationErrorType.STATIC_ANALYSIS,
            f"Static analysis failed for {target}: {error_details}",
            severity=ErrorSeverity.MEDIUM,
            context=merged_context,
            remediation="Inspect parser inputs, AST integrity, and analysis configuration thresholds.",
            component="static_analyzer",
            operation="static_analysis",
            tags=["static_analysis", target],
        )


class SecurityAnalysisError(EvaluationError):
    """Failure or finding associated with security-oriented analysis."""

    def __init__(
        self,
        target: str,
        error_details: str,
        severity: str | ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Mapping[str, Any]] = None,
    ):
        merged_context = dict(context or {})
        merged_context.update({"target": target, "error": error_details})
        super().__init__(
            EvaluationErrorType.SECURITY_ANALYSIS,
            f"Security analysis issue for {target}: {error_details}",
            severity=severity,
            context=merged_context,
            remediation="Review unsafe sinks, taint propagation, dependency posture, and threat-model coverage.",
            component="static_analyzer",
            operation="security_analysis",
            tags=["security", target],
        )


class ResourceExhaustionError(EvaluationError):
    """Failure caused by depletion of compute, memory, disk, or other runtime resources."""

    def __init__(self, resource: str, observed: Any, limit: Any, context: Optional[Mapping[str, Any]] = None):
        merged_context = dict(context or {})
        merged_context.update({"resource": resource, "observed": observed, "limit": limit})
        super().__init__(
            EvaluationErrorType.RESOURCE_EXHAUSTION,
            f"Resource exhaustion detected for {resource}: observed={observed}, limit={limit}",
            severity=ErrorSeverity.CRITICAL,
            context=merged_context,
            remediation="Reduce resource demand, increase capacity, or tighten batching and checkpointing behavior.",
            component="runtime",
            operation="resource_guard",
            tags=["resource", resource],
        )


class ConcurrencyError(EvaluationError):
    """Failure caused by race conditions, lock misuse, or concurrent state corruption."""

    def __init__(self, operation: str, error_details: str, context: Optional[Mapping[str, Any]] = None):
        merged_context = dict(context or {})
        merged_context.update({"operation": operation, "error": error_details})
        super().__init__(
            EvaluationErrorType.CONCURRENCY_FAILURE,
            f"Concurrency failure during {operation}: {error_details}",
            severity=ErrorSeverity.HIGH,
            context=merged_context,
            remediation="Inspect lock boundaries, mutation ordering, shared indexes, and checkpoint synchronization.",
            component="runtime",
            operation=operation,
            tags=["concurrency", operation],
        )


class UnsupportedOperationError(EvaluationError):
    """Raised when a requested operation or mode is not supported."""

    def __init__(self, operation: str, details: str = ""):
        message = f"Unsupported operation: {operation}"
        if details:
            message = f"{message} ({details})"
        super().__init__(
            EvaluationErrorType.UNSUPPORTED_OPERATION,
            message,
            severity=ErrorSeverity.MEDIUM,
            context={"operation": operation, "details": details},
            remediation="Use a supported mode, provide compatible inputs, or extend the implementation.",
            component="runtime",
            operation=operation,
            tags=["unsupported", operation],
        )


class OperationalError(EvaluationError):
    """General-purpose operational failure for evaluator runtime issues."""

    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None):
        super().__init__(
            error_type=EvaluationErrorType.DATA_INTEGRITY,
            message=message,
            severity=ErrorSeverity.CRITICAL,
            context=context or {},
            remediation="Inspect system configuration, runtime dependencies, and calling workflow state.",
            component="runtime",
            operation="operational_failure",
            tags=["operations"],
            cause=cause,
        )


class CertificationError(EvaluationError):
    """Custom exception for certification-related evaluation failures."""

    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None):
        super().__init__(
            error_type=EvaluationErrorType.CERTIFICATION_FAILURE,
            message=message,
            severity=ErrorSeverity.HIGH,
            context=context or {},
            remediation="Review architectural safety guarantees, compliance mappings, and certification evidence quality.",
            component="certification_framework",
            operation="certification_processing",
            tags=["certification"],
            cause=cause,
        )


class DocumentationProcessingError(EvaluationError):
    """Custom exception for documentation, schema, and audit-trail processing failures."""

    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None):
        super().__init__(
            error_type=EvaluationErrorType.DOCUMENTATION_FAILURE,
            message=message,
            severity=ErrorSeverity.HIGH,
            context=context or {},
            remediation="Review document schema, export format, audit-chain state, and versioning metadata.",
            component="documentation",
            operation="documentation_processing",
            tags=["documentation"],
            cause=cause,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _stable_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


__all__ = [
    "AuditIntegrityError",
    "CertificationConfigurationError",
    "CertificationError",
    "CertificationEvaluationError",
    "CertificationFrameworkError",
    "CertificationLevel",
    "CertificationStateError",
    "CheckpointError",
    "CheckpointLoadError",
    "ComparisonError",
    "ConcurrencyError",
    "ConfigLoadError",
    "DataIntegrityError",
    "DependencyError",
    "DocumentationConfigurationError",
    "DocumentationError",
    "DocumentationProcessingError",
    "ErrorSeverity",
    "EvaluationError",
    "EvaluationErrorType",
    "EvidenceValidationError",
    "InvalidDocumentError",
    "MemoryAccessError",
    "MetricCalculationError",
    "ModelRuntimeError",
    "OperationalError",
    "ReportGenerationError",
    "RequirementDefinitionError",
    "ResourceExhaustionError",
    "SchemaLoadError",
    "SecurityAnalysisError",
    "SerializationError",
    "StaticAnalysisError",
    "StatisticalAnalysisError",
    "TemplateError",
    "TemplateLoadError",
    "ThresholdViolationError",
    "TransformerError",
    "UnsupportedExportFormatError",
    "UnsupportedOperationError",
    "ValidationFailureError",
    "VisualizationError",
]
