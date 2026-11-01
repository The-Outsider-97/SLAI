"""Structured exception hierarchy and validation helpers for SLAI perception.

The module is deliberately dependency-light.  It defines stable perception
error categories, serializable error metadata, and small validation primitives
used by the perception contracts, modality pipelines, fusion, objectives, and
trainer layers.

Design constraints
------------------
- No imports from higher-level perception modules.
- No model construction or configuration loading.
- Validation failures use semantic perception exceptions rather than bare
  ``ValueError`` / ``TypeError`` at subsystem boundaries.
- Error payloads remain safe to log and serialize without dumping tensor data.
"""

from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union


T = TypeVar("T")


class ErrorSeverity(str, Enum):
    """Operational severity classification for perception exceptions."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def _serialize_for_error(value: Any) -> Any:
    """Best-effort structured serialization for error metadata."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize_for_error(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize_for_error(item) for item in value]

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None or dtype is not None or device is not None:
        return {
            "type": type(value).__name__,
            "shape": list(shape) if shape is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
            "device": str(device) if device is not None else None,
        }

    return repr(value)


def _normalize_details(details: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not details:
        return {}
    return {str(key): _serialize_for_error(value) for key, value in details.items()}


class PerceptionError(Exception):
    """Base exception for the perception subsystem."""

    default_code = "PERCEPTION_ERROR"
    default_component = "perception"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        component: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
        remediation: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        retryable: bool = False,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.message = str(message)
        self.code = code or self.default_code
        self.component = component or self.default_component
        self.details = _normalize_details(details)
        self.remediation = remediation
        self.severity = severity
        self.retryable = bool(retryable)
        self.cause = cause

        if cause is not None:
            self.__cause__ = cause

        super().__init__(self.message)

    def __str__(self) -> str:
        base = f"[{self.component}:{self.code}] {self.message}"
        extras = []
        if self.retryable:
            extras.append("retryable=True")
        if self.remediation:
            extras.append(f"remediation={self.remediation}")
        if self.details:
            extras.append(f"details={self.details}")
        if self.cause is not None:
            extras.append(f"cause={type(self.cause).__name__}: {self.cause}")
        return f"{base} | " + " | ".join(extras) if extras else base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "component": self.component,
            "severity": self.severity.value,
            "retryable": self.retryable,
            "remediation": self.remediation,
            "details": dict(self.details),
            "cause_type": type(self.cause).__name__ if self.cause else None,
            "cause_message": str(self.cause) if self.cause else None,
        }

    def with_details(self, **details: Any) -> "PerceptionError":
        self.details.update(_normalize_details(details))
        return self

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        message: str,
        *,
        component: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
        remediation: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        retryable: bool = False,
    ) -> "PerceptionError":
        merged = dict(_normalize_details(details))
        merged.setdefault("wrapped_exception_type", type(exc).__name__)
        return cls(
            message,
            component=component,
            code=code,
            details=merged,
            remediation=remediation,
            severity=severity,
            retryable=retryable,
            cause=exc,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PerceptionConfigurationError(PerceptionError):
    default_code = "CONFIGURATION_ERROR"
    default_component = "perception_config"


class MissingPerceptionConfigurationError(PerceptionConfigurationError):
    default_code = "MISSING_CONFIGURATION"


class InvalidPerceptionConfigurationError(PerceptionConfigurationError):
    default_code = "INVALID_CONFIGURATION"


class UnsupportedPerceptionOptionError(PerceptionConfigurationError):
    default_code = "UNSUPPORTED_OPTION"


# ---------------------------------------------------------------------------
# Validation / contracts
# ---------------------------------------------------------------------------


class PerceptionValidationError(PerceptionError):
    default_code = "VALIDATION_ERROR"
    default_component = "perception_validation"


class MissingPerceptionFieldError(PerceptionValidationError):
    default_code = "MISSING_FIELD"


class InvalidPerceptionTypeError(PerceptionValidationError, TypeError):
    default_code = "INVALID_TYPE"


class InvalidPerceptionValueError(PerceptionValidationError, ValueError):
    default_code = "INVALID_VALUE"


class PerceptionRangeError(PerceptionValidationError, ValueError):
    default_code = "OUT_OF_RANGE"


class PerceptionShapeError(PerceptionValidationError, ValueError):
    default_code = "SHAPE_ERROR"


class PerceptionDimensionError(PerceptionShapeError):
    default_code = "DIMENSION_MISMATCH"


class PerceptionContractError(PerceptionValidationError):
    default_code = "CONTRACT_ERROR"
    default_component = "perception_contracts"


# ---------------------------------------------------------------------------
# Modality execution
# ---------------------------------------------------------------------------


class PerceptionModalityError(PerceptionError):
    default_code = "MODALITY_ERROR"
    default_component = "perception_modality"


class UnsupportedModalityError(PerceptionModalityError):
    default_code = "UNSUPPORTED_MODALITY"


class ModalityInputError(PerceptionModalityError):
    default_code = "MODALITY_INPUT_ERROR"


class ModalityEncodingError(PerceptionModalityError):
    default_code = "MODALITY_ENCODING_ERROR"


class ModalityDecodingError(PerceptionModalityError):
    default_code = "MODALITY_DECODING_ERROR"


class UnsupportedModalityOperationError(PerceptionModalityError):
    default_code = "UNSUPPORTED_MODALITY_OPERATION"


# ---------------------------------------------------------------------------
# Fusion / objectives / training
# ---------------------------------------------------------------------------


class PerceptionFusionError(PerceptionError):
    default_code = "FUSION_ERROR"
    default_component = "perception_fusion"


class PerceptionObjectiveError(PerceptionError):
    default_code = "OBJECTIVE_ERROR"
    default_component = "perception_objectives"


class PerceptionTrainingError(PerceptionError):
    default_code = "TRAINING_ERROR"
    default_component = "perception_trainer"


class OptimizerConfigurationError(PerceptionTrainingError):
    default_code = "OPTIMIZER_CONFIGURATION_ERROR"


class NonFiniteLossError(PerceptionTrainingError):
    default_code = "NON_FINITE_LOSS"


class PerceptionStateError(PerceptionError):
    default_code = "STATE_ERROR"
    default_component = "perception_state"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def ensure(
    condition: bool,
    message: str,
    *,
    exc_type: Type[PerceptionError] = PerceptionValidationError,
    component: Optional[str] = None,
    code: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    remediation: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    retryable: bool = False,
) -> None:
    if not condition:
        raise exc_type(
            message,
            component=component,
            code=code,
            details=details,
            remediation=remediation,
            severity=severity,
            retryable=retryable,
        )


def ensure_not_none(
    value: Optional[T],
    name: str,
    *,
    component: Optional[str] = None,
    exc_type: Type[PerceptionError] = MissingPerceptionFieldError,
) -> T:
    if value is None:
        raise exc_type(
            f"Required value '{name}' is missing.",
            component=component,
            details={"field": name},
            remediation=f"Provide a non-null value for '{name}'.",
        )
    return value


def ensure_instance(
    value: Any,
    expected_types: Union[Type[Any], Tuple[Type[Any], ...]],
    name: str,
    *,
    component: Optional[str] = None,
    exc_type: Type[PerceptionError] = InvalidPerceptionTypeError,
) -> Any:
    if not isinstance(value, expected_types):
        expected = (
            expected_types.__name__
            if isinstance(expected_types, type)
            else [item.__name__ for item in expected_types]
        )
        raise exc_type(
            f"'{name}' must be an instance of {expected}, got {type(value).__name__}.",
            component=component,
            details={"field": name, "expected": expected, "actual": type(value).__name__},
        )
    return value


def ensure_mapping(
    value: Any,
    name: str,
    *,
    component: Optional[str] = None,
) -> Mapping[str, Any]:
    ensure_instance(value, Mapping, name, component=component)
    return value


def ensure_keys(
    mapping: Mapping[str, Any],
    required_keys: Iterable[str],
    *,
    name: str = "mapping",
    component: Optional[str] = None,
) -> Mapping[str, Any]:
    ensure_mapping(mapping, name, component=component)
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise MissingPerceptionFieldError(
            f"'{name}' is missing required keys: {missing}.",
            component=component,
            details={"name": name, "missing_keys": missing},
        )
    return mapping


def ensure_non_empty(
    value: Union[str, Sequence[Any], Mapping[str, Any]],
    name: str,
    *,
    component: Optional[str] = None,
) -> Union[str, Sequence[Any], Mapping[str, Any]]:
    if len(value) == 0:
        raise InvalidPerceptionValueError(
            f"'{name}' must not be empty.",
            component=component,
            details={"field": name},
        )
    return value


def ensure_in_range(
    value: Union[int, float],
    name: str,
    *,
    minimum: Optional[Union[int, float]] = None,
    maximum: Optional[Union[int, float]] = None,
    component: Optional[str] = None,
) -> Union[int, float]:
    failed = (minimum is not None and value < minimum) or (
        maximum is not None and value > maximum
    )
    if failed:
        raise PerceptionRangeError(
            f"'{name}'={value} is outside the allowed range.",
            component=component,
            details={"field": name, "value": value, "minimum": minimum, "maximum": maximum},
        )
    return value


def ensure_probability(
    value: Union[int, float],
    name: str,
    *,
    component: Optional[str] = None,
) -> Union[int, float]:
    return ensure_in_range(value, name, minimum=0.0, maximum=1.0, component=component)


def ensure_one_of(
    value: T,
    allowed: Iterable[T],
    name: str,
    *,
    component: Optional[str] = None,
    exc_type: Type[PerceptionError] = InvalidPerceptionValueError,
) -> T:
    allowed_values = tuple(allowed)
    if value not in allowed_values:
        raise exc_type(
            f"'{name}' must be one of {allowed_values}, got {value!r}.",
            component=component,
            details={"field": name, "value": value, "allowed": allowed_values},
        )
    return value


def ensure_rank(
    value: Any,
    expected_rank: int,
    name: str,
    *,
    component: Optional[str] = None,
) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise InvalidPerceptionTypeError(
            f"'{name}' must expose a tensor-like shape.",
            component=component,
            details={"field": name, "actual_type": type(value).__name__},
        )
    actual_rank = len(shape)
    if actual_rank != expected_rank:
        raise PerceptionShapeError(
            f"'{name}' must have rank {expected_rank}, got rank {actual_rank}.",
            component=component,
            details={"field": name, "shape": list(shape), "expected_rank": expected_rank},
        )
    return value


def ensure_same_batch(
    *values: Any,
    names: Optional[Sequence[str]] = None,
    component: Optional[str] = None,
) -> int:
    if not values:
        raise InvalidPerceptionValueError(
            "At least one value is required for batch validation.",
            component=component,
        )

    batch_sizes = []
    for index, value in enumerate(values):
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            label = names[index] if names and index < len(names) else f"value_{index}"
            raise PerceptionShapeError(
                f"'{label}' must have a batch dimension.",
                component=component,
                details={"shape": list(shape) if shape is not None else None},
            )
        batch_sizes.append(int(shape[0]))

    if len(set(batch_sizes)) != 1:
        raise PerceptionDimensionError(
            "Batch dimensions do not match.",
            component=component,
            details={"batch_sizes": batch_sizes, "names": list(names or [])},
        )
    return batch_sizes[0]


def wrap_exception(
    exc: BaseException,
    exc_type: Type[PerceptionError],
    message: str,
    *,
    component: Optional[str] = None,
    code: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    remediation: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    retryable: bool = False,
) -> PerceptionError:
    if isinstance(exc, PerceptionError):
        return exc
    return exc_type(
        message,
        component=component,
        code=code,
        details=details,
        remediation=remediation,
        severity=severity,
        retryable=retryable,
        cause=exc,
    )


__all__ = [
    "ErrorSeverity",
    "PerceptionError",
    "PerceptionConfigurationError",
    "MissingPerceptionConfigurationError",
    "InvalidPerceptionConfigurationError",
    "UnsupportedPerceptionOptionError",
    "PerceptionValidationError",
    "MissingPerceptionFieldError",
    "InvalidPerceptionTypeError",
    "InvalidPerceptionValueError",
    "PerceptionRangeError",
    "PerceptionShapeError",
    "PerceptionDimensionError",
    "PerceptionContractError",
    "PerceptionModalityError",
    "UnsupportedModalityError",
    "ModalityInputError",
    "ModalityEncodingError",
    "ModalityDecodingError",
    "UnsupportedModalityOperationError",
    "PerceptionFusionError",
    "PerceptionObjectiveError",
    "PerceptionTrainingError",
    "OptimizerConfigurationError",
    "NonFiniteLossError",
    "PerceptionStateError",
    "ensure",
    "ensure_not_none",
    "ensure_instance",
    "ensure_mapping",
    "ensure_keys",
    "ensure_non_empty",
    "ensure_in_range",
    "ensure_probability",
    "ensure_one_of",
    "ensure_rank",
    "ensure_same_batch",
    "wrap_exception",
]
