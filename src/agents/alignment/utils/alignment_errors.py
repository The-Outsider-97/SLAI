"""
Structured exception hierarchy and validation helpers for the alignment agent stack.

This module provides a stable domain-specific exception taxonomy for the
AlignmentAgent and its submodules, including AlignmentMonitor,
AlignmentMemory, BiasDetector, FairnessEvaluator, EthicalConstraints,
CounterfactualAuditor, ValueEmbeddingModel, and HumanOversightInterface.

Design goals:
- Stable, deterministic error codes suitable for logging and alerting.
- Rich, serialisable context for audit trails and human oversight.
- Reusable validation helpers for inputs, configuration, state, and schemas.
- Safe behaviour inside error paths so exception construction does not itself
  introduce secondary failures.
"""

from __future__ import annotations

import json

from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union

from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter  # type: ignore

logger = get_logger("Alignment Error")
printer = PrettyPrinter

T = TypeVar("T")

_VALID_SEVERITIES = {"low", "medium", "high", "critical"}


class AlignmentErrorType(Enum):
    """Canonical error domains across the alignment stack."""

    UNKNOWN = "unknown"
    CONFIGURATION = "configuration"
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    DATA_VALIDATION = "data_validation"
    MISSING_FIELD = "missing_field"
    TYPE_MISMATCH = "type_mismatch"
    SENSITIVE_ATTRIBUTE = "sensitive_attribute"
    STATE = "state"
    MEMORY = "memory"
    BIAS_DETECTION = "bias_detection"
    FAIRNESS_EVALUATION = "fairness_evaluation"
    ETHICAL_CONSTRAINT = "ethical_constraint"
    COUNTERFACTUAL_AUDIT = "counterfactual_audit"
    CAUSAL_MODEL = "causal_model"
    VALUE_EMBEDDING = "value_embedding"
    RISK_ASSESSMENT = "risk_assessment"
    CONCEPT_DRIFT = "concept_drift"
    POLICY_ADJUSTMENT = "policy_adjustment"
    HUMAN_OVERSIGHT = "human_oversight"
    AUTHORIZATION = "authorization"
    TIMEOUT = "timeout"
    INTERVENTION = "intervention"
    PERSISTENCE = "persistence"
    EXTERNAL_DEPENDENCY = "external_dependency"
    DIAGNOSTICS = "diagnostics"


# Mapping from error type to unique, deterministic error code.
# Format: ALG-1xxx (Alignment Agent)
_ERROR_CODE_MAP: Dict[AlignmentErrorType, str] = {
    AlignmentErrorType.UNKNOWN: "ALG-1000",
    AlignmentErrorType.CONFIGURATION: "ALG-1100",
    AlignmentErrorType.INITIALIZATION: "ALG-1101",
    AlignmentErrorType.VALIDATION: "ALG-1200",
    AlignmentErrorType.DATA_VALIDATION: "ALG-1201",
    AlignmentErrorType.MISSING_FIELD: "ALG-1202",
    AlignmentErrorType.TYPE_MISMATCH: "ALG-1203",
    AlignmentErrorType.SENSITIVE_ATTRIBUTE: "ALG-1204",
    AlignmentErrorType.STATE: "ALG-1300",
    AlignmentErrorType.MEMORY: "ALG-1400",
    AlignmentErrorType.BIAS_DETECTION: "ALG-1500",
    AlignmentErrorType.FAIRNESS_EVALUATION: "ALG-1510",
    AlignmentErrorType.ETHICAL_CONSTRAINT: "ALG-1520",
    AlignmentErrorType.COUNTERFACTUAL_AUDIT: "ALG-1530",
    AlignmentErrorType.CAUSAL_MODEL: "ALG-1531",
    AlignmentErrorType.VALUE_EMBEDDING: "ALG-1540",
    AlignmentErrorType.RISK_ASSESSMENT: "ALG-1600",
    AlignmentErrorType.CONCEPT_DRIFT: "ALG-1601",
    AlignmentErrorType.POLICY_ADJUSTMENT: "ALG-1602",
    AlignmentErrorType.HUMAN_OVERSIGHT: "ALG-1700",
    AlignmentErrorType.AUTHORIZATION: "ALG-1701",
    AlignmentErrorType.TIMEOUT: "ALG-1702",
    AlignmentErrorType.INTERVENTION: "ALG-1703",
    AlignmentErrorType.PERSISTENCE: "ALG-1800",
    AlignmentErrorType.EXTERNAL_DEPENDENCY: "ALG-1900",
    AlignmentErrorType.DIAGNOSTICS: "ALG-1950",
}


def _safe_load_config() -> Dict[str, Any]:
    """Safely load config for the error subsystem without masking root failures."""
    try:
        config = load_global_config()
        return config if isinstance(config, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to load global config inside alignment_errors: %s", exc)
        return {}


_ERROR_FALLBACK_CONFIG = _safe_load_config()


def _safe_get_error_config() -> Dict[str, Any]:
    try:
        section = get_config_section("alignment_error")
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to load 'alignment_error' config section: %s", exc)
        return {}


_ERROR_SECTION = _safe_get_error_config()


def _normalise_severity(severity: Optional[str], default: str = "medium") -> str:
    value = (severity or default or "medium").strip().lower()
    return value if value in _VALID_SEVERITIES else default


def _deduplicate_tags(tags: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if not tags:
        return tuple()
    seen = []
    for tag in tags:
        text = str(tag).strip()
        if text and text not in seen:
            seen.append(text)
    return tuple(seen)


def _json_safe(value: Any) -> Any:
    """Convert common runtime objects into JSON-serialisable representations."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
        }
    if hasattr(value, "__dict__"):
        try:
            return {k: _json_safe(v) for k, v in vars(value).items()}
        except Exception:
            pass
    return repr(value)


def _ensure_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_code(error_type: AlignmentErrorType, explicit_code: Optional[str] = None) -> str:
    if explicit_code:
        return explicit_code
    return _ERROR_CODE_MAP.get(error_type, "ALG-1999")


def _default_remediation(error_type: AlignmentErrorType) -> Optional[str]:
    remediations = _ERROR_SECTION.get("remediation", {})
    if isinstance(remediations, dict):
        value = remediations.get(error_type.value) or remediations.get(error_type.name)
        if value:
            return str(value)
    fallback_map = {
        AlignmentErrorType.CONFIGURATION: "Validate configuration sections and required keys before initialisation.",
        AlignmentErrorType.DATA_VALIDATION: "Validate schema, dtypes, cardinality, and required columns before processing.",
        AlignmentErrorType.SENSITIVE_ATTRIBUTE: "Review configured sensitive attributes and ensure they exist in the dataset.",
        AlignmentErrorType.HUMAN_OVERSIGHT: "Inspect intervention channels, persistence, and reviewer response workflow.",
        AlignmentErrorType.TIMEOUT: "Increase timeout only if justified and verify that the corresponding external dependency is reachable.",
        AlignmentErrorType.PERSISTENCE: "Verify storage connectivity, filesystem permissions, and transactional integrity.",
    }
    return fallback_map.get(error_type)


class AlignmentError(Exception):
    """
    Base exception for the alignment subsystem.

    The class is designed to be both human-readable and machine-actionable.
    It supports deterministic codes, structured context, serialisation, logging,
    and exception wrapping for lower-level failures.
    """

    error_type: AlignmentErrorType = AlignmentErrorType.UNKNOWN
    default_code = "ALG-1000"
    default_severity = "medium"
    default_retryable = False
    default_category: Union[str, Dict[str, Any]] = "alignment"

    def __init__(
        self,
        error_type: Optional[AlignmentErrorType] = None,
        message: Optional[str] = None,
        severity: Optional[str] = None,
        *,
        code: Optional[str] = None,
        category: Optional[Union[str, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        alignment_agent_state: Optional[Dict[str, Any]] = None,
        remediation_guidance: Optional[str] = None,
        retryable: Optional[bool] = None,
        tags: Optional[Iterable[Any]] = None,
        cause: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        resolved_error_type = error_type or getattr(self, "error_type", AlignmentErrorType.UNKNOWN)
        resolved_message = (message or "An alignment subsystem error occurred.").strip()
        resolved_severity = _normalise_severity(severity, getattr(self, "default_severity", "medium"))
        resolved_code = code or _resolve_code(resolved_error_type, getattr(self, "default_code", None))

        super().__init__(resolved_message)

        self.message = resolved_message
        self.error_type = resolved_error_type
        self.severity = resolved_severity
        self.code = resolved_code
        self.category = category if category is not None else getattr(self, "default_category", "alignment")
        self.context = _ensure_mapping(context)
        self.alignment_agent_state = _ensure_mapping(alignment_agent_state)
        self.retryable = bool(retryable) if retryable is not None else bool(getattr(self, "default_retryable", False))
        self.tags = _deduplicate_tags(tags)
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.cause = cause
        self.metadata = _ensure_mapping(metadata)

        self.config = _ERROR_FALLBACK_CONFIG
        self.error_config = _ERROR_SECTION
        self.remediation_guidance = remediation_guidance or _default_remediation(self.error_type)

        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        parts = [f"[{self.code}]", f"[{self.severity.upper()}]", self.message]
        if self.remediation_guidance:
            parts.append(f"Remediation: {self.remediation_guidance}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(code={self.code!r}, error_type={self.error_type.value!r}, "
            f"severity={self.severity!r}, message={self.message!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "type": self.error_type.value,
            "class_name": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity,
            "retryable": self.retryable,
            "category": _json_safe(self.category),
            "timestamp": self.timestamp,
            "context": _json_safe(self.context),
            "alignment_agent_state": _json_safe(self.alignment_agent_state),
            "remediation_guidance": self.remediation_guidance,
            "tags": list(self.tags),
            "metadata": _json_safe(self.metadata),
            "cause": _json_safe(self.cause),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_log_record(self) -> Dict[str, Any]:
        record = self.to_dict()
        record["event"] = "alignment_error"
        return record

    def log(self, log_instance: Any = None) -> Dict[str, Any]:
        record = self.to_log_record()
        active_logger = log_instance or logger

        if self.severity == "critical":
            active_logger.critical(self.message, extra={"alignment_error": record})
        elif self.severity == "high":
            active_logger.error(self.message, extra={"alignment_error": record})
        elif self.severity == "medium":
            active_logger.warning(self.message, extra={"alignment_error": record})
        else:
            active_logger.info(self.message, extra={"alignment_error": record})
        return record

    def with_context(self, **kwargs: Any) -> "AlignmentError":
        self.context.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    def with_state(self, **kwargs: Any) -> "AlignmentError":
        self.alignment_agent_state.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    @classmethod
    def wrap(
        cls,
        exc: BaseException,
        *,
        error_type: Optional[AlignmentErrorType] = None,
        message: Optional[str] = None,
        severity: Optional[str] = None,
        code: Optional[str] = None,
        category: Optional[Union[str, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        alignment_agent_state: Optional[Dict[str, Any]] = None,
        remediation_guidance: Optional[str] = None,
        retryable: Optional[bool] = None,
        tags: Optional[Iterable[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "AlignmentError":
        wrapped_message = message or f"{type(exc).__name__}: {exc}"
        return cls(
            error_type=error_type,
            message=wrapped_message,
            severity=severity,
            code=code,
            category=category,
            context=context,
            alignment_agent_state=alignment_agent_state,
            remediation_guidance=remediation_guidance,
            retryable=retryable,
            tags=tags,
            cause=exc,
            metadata=metadata,
        )


class _TypedAlignmentError(AlignmentError):
    """Convenience base class for typed alignment errors."""

    error_type = AlignmentErrorType.UNKNOWN
    default_severity = "medium"
    default_retryable = False
    default_category = "alignment"

    def __init__(
        self,
        message: str,
        severity: Optional[str] = None,
        *,
        error_type: Optional[AlignmentErrorType] = None,  # ← added (ignored)
        code: Optional[str] = None,
        category: Optional[Union[str, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        alignment_agent_state: Optional[Dict[str, Any]] = None,
        remediation_guidance: Optional[str] = None,
        retryable: Optional[bool] = None,
        tags: Optional[Iterable[Any]] = None,
        cause: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        # The error_type parameter is ignored; the class attribute is used.
        super().__init__(
            error_type=self.error_type,
            message=message,
            severity=severity,
            code=code,
            category=category,
            context=context,
            alignment_agent_state=alignment_agent_state,
            remediation_guidance=remediation_guidance,
            retryable=retryable,
            tags=tags,
            cause=cause,
            metadata=metadata,
        )


class ConfigurationError(_TypedAlignmentError):
    error_type = AlignmentErrorType.CONFIGURATION
    default_severity = "high"
    default_category = "configuration"


class InitializationError(_TypedAlignmentError):
    error_type = AlignmentErrorType.INITIALIZATION
    default_severity = "high"
    default_category = "initialization"


class ValidationError(_TypedAlignmentError, ValueError):
    error_type = AlignmentErrorType.VALIDATION
    default_severity = "medium"
    default_category = "validation"


class DataValidationError(ValidationError):
    error_type = AlignmentErrorType.DATA_VALIDATION
    default_category = "data_validation"


class MissingFieldError(DataValidationError):
    error_type = AlignmentErrorType.MISSING_FIELD
    default_category = "schema"


class TypeMismatchError(DataValidationError):
    error_type = AlignmentErrorType.TYPE_MISMATCH
    default_category = "schema"


class SensitiveAttributeError(DataValidationError):
    error_type = AlignmentErrorType.SENSITIVE_ATTRIBUTE
    default_category = "sensitive_attributes"
    default_severity = "high"


class AlignmentStateError(_TypedAlignmentError):
    error_type = AlignmentErrorType.STATE
    default_severity = "high"
    default_category = "state"


class AlignmentMemoryError(_TypedAlignmentError):
    error_type = AlignmentErrorType.MEMORY
    default_severity = "high"
    default_category = "memory"


class BiasDetectionError(_TypedAlignmentError):
    error_type = AlignmentErrorType.BIAS_DETECTION
    default_severity = "high"
    default_category = "bias_detection"


class FairnessEvaluationError(_TypedAlignmentError):
    error_type = AlignmentErrorType.FAIRNESS_EVALUATION
    default_severity = "high"
    default_category = "fairness_evaluation"


class EthicalConstraintError(_TypedAlignmentError):
    error_type = AlignmentErrorType.ETHICAL_CONSTRAINT
    default_severity = "high"
    default_category = "ethical_constraints"


class CounterfactualAuditError(_TypedAlignmentError):
    error_type = AlignmentErrorType.COUNTERFACTUAL_AUDIT
    default_severity = "high"
    default_category = "counterfactual"


class CausalModelError(_TypedAlignmentError):
    error_type = AlignmentErrorType.CAUSAL_MODEL
    default_severity = "high"
    default_category = "causal_model"


class ValueEmbeddingError(_TypedAlignmentError):
    error_type = AlignmentErrorType.VALUE_EMBEDDING
    default_severity = "high"
    default_category = "value_embedding"


class RiskAssessmentError(_TypedAlignmentError):
    error_type = AlignmentErrorType.RISK_ASSESSMENT
    default_severity = "high"
    default_category = "risk"


class ConceptDriftError(_TypedAlignmentError):
    error_type = AlignmentErrorType.CONCEPT_DRIFT
    default_severity = "medium"
    default_category = "drift"


class PolicyAdjustmentError(_TypedAlignmentError):
    error_type = AlignmentErrorType.POLICY_ADJUSTMENT
    default_severity = "high"
    default_category = "policy"


class HumanOversightError(_TypedAlignmentError):
    error_type = AlignmentErrorType.HUMAN_OVERSIGHT
    default_severity = "high"
    default_category = "human_oversight"


class AuthorizationError(_TypedAlignmentError, PermissionError):
    error_type = AlignmentErrorType.AUTHORIZATION
    default_severity = "high"
    default_category = "authorization"


class AlignmentTimeoutError(_TypedAlignmentError):
    error_type = AlignmentErrorType.TIMEOUT
    default_severity = "high"
    default_category = "timeout"
    default_retryable = True


class TimeoutError(AlignmentTimeoutError):
    """Backward-compatible timeout name kept for existing imports."""


class InterventionError(_TypedAlignmentError):
    error_type = AlignmentErrorType.INTERVENTION
    default_severity = "critical"
    default_category = "intervention"


class PersistenceError(_TypedAlignmentError):
    error_type = AlignmentErrorType.PERSISTENCE
    default_severity = "high"
    default_category = "persistence"


class ExternalDependencyError(_TypedAlignmentError):
    error_type = AlignmentErrorType.EXTERNAL_DEPENDENCY
    default_severity = "high"
    default_category = "dependency"
    default_retryable = True


class DiagnosticsError(_TypedAlignmentError):
    error_type = AlignmentErrorType.DIAGNOSTICS
    default_severity = "medium"
    default_category = "diagnostics"


class HumanOversightValidationError(ValidationError):
    """Raised when a human response is structurally invalid."""

    error_type = AlignmentErrorType.HUMAN_OVERSIGHT
    default_category = "human_oversight_validation"


class HumanOversightAuthError(AuthorizationError):
    """Raised when reviewer authentication/authorisation fails."""

    error_type = AlignmentErrorType.AUTHORIZATION
    default_category = "human_oversight_authorization"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def ensure_not_none(
    value: Optional[T],
    field_name: str,
    *,
    error_cls: Type[AlignmentError] = ValidationError,
    context: Optional[Dict[str, Any]] = None,
) -> T:
    if value is None:
        raise error_cls(
            message=f"'{field_name}' must not be None.",
            context={**(context or {}), "field": field_name},
        )
    return value


def ensure_instance(
    value: Any,
    expected_type: Union[Type[Any], Tuple[Type[Any], ...]],
    field_name: str,
    *,
    error_cls: Type[AlignmentError] = TypeMismatchError,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    if not isinstance(value, expected_type):
        expected = (
            [t.__name__ for t in expected_type]
            if isinstance(expected_type, tuple)
            else [expected_type.__name__]
        )
        raise error_cls(
            message=f"'{field_name}' must be an instance of {expected}, got {type(value).__name__}.",
            context={
                **(context or {}),
                "field": field_name,
                "expected_type": expected,
                "actual_type": type(value).__name__,
            },
        )
    return value


def ensure_non_empty_string(
    value: Any,
    field_name: str,
    *,
    error_cls: Type[AlignmentError] = ValidationError,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_cls(
            message=f"'{field_name}' must be a non-empty string.",
            context={**(context or {}), "field": field_name},
        )
    return value.strip()


def ensure_mapping(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
    error_cls: Type[AlignmentError] = ValidationError,
    context: Optional[Dict[str, Any]] = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_cls(
            message=f"'{field_name}' must be a mapping.",
            context={**(context or {}), "field": field_name, "actual_type": type(value).__name__},
        )
    if not allow_empty and not value:
        raise error_cls(
            message=f"'{field_name}' must not be empty.",
            context={**(context or {}), "field": field_name},
        )
    return value


def ensure_sequence(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
    allow_strings: bool = False,
    error_cls: Type[AlignmentError] = ValidationError,
    context: Optional[Dict[str, Any]] = None,
) -> Sequence[Any]:
    if isinstance(value, str) and not allow_strings:
        raise error_cls(
            message=f"'{field_name}' must be a sequence, not a raw string.",
            context={**(context or {}), "field": field_name},
        )
    if not isinstance(value, Sequence):
        raise error_cls(
            message=f"'{field_name}' must be a sequence.",
            context={**(context or {}), "field": field_name, "actual_type": type(value).__name__},
        )
    if not allow_empty and len(value) == 0:
        raise error_cls(
            message=f"'{field_name}' must not be empty.",
            context={**(context or {}), "field": field_name},
        )
    return value


def ensure_keys_present(
    mapping: Mapping[str, Any],
    required_keys: Iterable[str],
    *,
    field_name: str = "mapping",
    error_cls: Type[AlignmentError] = MissingFieldError,
    context: Optional[Dict[str, Any]] = None,
) -> Mapping[str, Any]:
    ensure_mapping(mapping, field_name, error_cls=error_cls, context=context)
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise error_cls(
            message=f"'{field_name}' is missing required keys: {missing}.",
            context={**(context or {}), "field": field_name, "missing_keys": missing},
        )
    return mapping


def ensure_columns_present(
    columns_source: Any,
    required_columns: Iterable[str],
    *,
    field_name: str = "data",
    error_cls: Type[AlignmentError] = DataValidationError,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    if hasattr(columns_source, "columns"):
        available = [str(col) for col in columns_source.columns]
    elif isinstance(columns_source, Mapping):
        available = [str(col) for col in columns_source.keys()]
    else:
        available = [str(col) for col in columns_source]
    missing = [col for col in required_columns if col not in available]
    if missing:
        raise error_cls(
            message=f"'{field_name}' is missing required columns: {missing}.",
            context={
                **(context or {}),
                "field": field_name,
                "missing_columns": missing,
                "available_columns": available,
            },
        )
    return columns_source


def ensure_numeric_range(
    value: Any,
    field_name: str,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    inclusive: bool = True,
    error_cls: Type[AlignmentError] = ValidationError,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    if not isinstance(value, (int, float)):
        raise error_cls(
            message=f"'{field_name}' must be numeric.",
            context={**(context or {}), "field": field_name, "actual_type": type(value).__name__},
        )

    if min_value is not None:
        if (inclusive and value < min_value) or (not inclusive and value <= min_value):
            raise error_cls(
                message=f"'{field_name}' must be {'>=' if inclusive else '>'} {min_value}.",
                context={**(context or {}), "field": field_name, "value": value, "min_value": min_value},
            )
    if max_value is not None:
        if (inclusive and value > max_value) or (not inclusive and value >= max_value):
            raise error_cls(
                message=f"'{field_name}' must be {'<=' if inclusive else '<'} {max_value}.",
                context={**(context or {}), "field": field_name, "value": value, "max_value": max_value},
            )
    return float(value)


def ensure_path_exists(
    path_value: Union[str, Path],
    field_name: str,
    *,
    error_cls: Type[AlignmentError] = ConfigurationError,
    context: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise error_cls(
            message=f"Path for '{field_name}' does not exist: {path}",
            context={**(context or {}), "field": field_name, "path": str(path)},
        )
    return path


def validate_sensitive_attributes(
    data: Any,
    sensitive_attributes: Sequence[str],
    *,
    field_name: str = "data",
    error_cls: Type[AlignmentError] = SensitiveAttributeError,
    context: Optional[Dict[str, Any]] = None,
) -> Sequence[str]:
    ensure_sequence(
        sensitive_attributes,
        "sensitive_attributes",
        allow_empty=False,
        error_cls=error_cls,
        context=context,
    )
    ensure_columns_present(
        data,
        sensitive_attributes,
        field_name=field_name,
        error_cls=error_cls,
        context=context,
    )
    return sensitive_attributes


def wrap_alignment_exception(
    exc: BaseException,
    *,
    target_cls: Type[AlignmentError] = AlignmentError,
    message: Optional[str] = None,
    severity: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    alignment_agent_state: Optional[Dict[str, Any]] = None,
    remediation_guidance: Optional[str] = None,
    retryable: Optional[bool] = None,
    tags: Optional[Iterable[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AlignmentError:
    """Convert arbitrary exceptions into the alignment error contract."""
    if isinstance(exc, AlignmentError):
        return exc
    return target_cls.wrap(
        exc,
        message=message,
        severity=severity,
        context=context,
        alignment_agent_state=alignment_agent_state,
        remediation_guidance=remediation_guidance,
        retryable=retryable,
        tags=tags,
        metadata=metadata,
    )
