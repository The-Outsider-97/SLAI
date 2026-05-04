"""
Structured exception hierarchy and validation helpers for the adaptive agent stack.

This module provides:

1. A stable domain-specific exception taxonomy.
2. Structured, machine-readable error payloads.
3. Consistent validation helpers to reduce repetitive guard code.
4. Exception chaining utilities for preserving root causes.
5. Backward-friendly exceptions for persistence/file-path failure modes.

Design principles
-----------------
- Errors should be specific enough for callers to branch on.
- Error payloads should be safe to log and serialize.
- Validation helpers should raise semantic exceptions, not bare built-ins.
- Helper functions should stay dependency-light and standard-library only.
- The hierarchy should map cleanly to the adaptive stack's real components.
"""

from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union


T = TypeVar("T")


class ErrorSeverity(str, Enum):
    """Operational severity classification for adaptive-stack exceptions."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def _serialize_for_error(value: Any) -> Any:
    """
    Best-effort serialization helper for exception details.

    The goal is not strict JSON encoding at this layer, but stable and safe
    structured output for logs, telemetry, or API responses.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): _serialize_for_error(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize_for_error(v) for v in value]

    if hasattr(value, "shape"):
        shape = getattr(value, "shape", None)
        return {
            "type": type(value).__name__,
            "shape": tuple(shape) if shape is not None else None,
        }

    if hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "dtype": str(getattr(value, "dtype", None)),
        }

    return repr(value)


def _normalize_details(details: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not details:
        return {}
    return {str(k): _serialize_for_error(v) for k, v in details.items()}


class AdaptiveError(Exception):
    """
    Base exception for the adaptive subsystem.

    Every subclass supports:
    - code: stable machine-readable identifier
    - component: owning subsystem
    - details: structured payload for logs/telemetry
    - remediation: operator/developer hint
    - severity: operational severity
    - retryable: whether the caller may safely retry
    - cause: original underlying exception
    """

    default_code = "ADAPTIVE_ERROR"
    default_component = "adaptive"

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
            "details": self.details,
            "cause_type": type(self.cause).__name__ if self.cause else None,
            "cause_message": str(self.cause) if self.cause else None,
        }

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
        remediation: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        retryable: bool = False,
        component: Optional[str] = None,
        code: Optional[str] = None,
    ) -> "AdaptiveError":
        merged_details = dict(_normalize_details(details))
        merged_details.setdefault("wrapped_exception_type", type(exc).__name__)
        return cls(
            message,
            code=code,
            component=component,
            details=merged_details,
            remediation=remediation,
            severity=severity,
            retryable=retryable,
            cause=exc,
        )


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class AdaptiveConfigurationError(AdaptiveError):
    default_code = "CONFIGURATION_ERROR"
    default_component = "config"


class MissingConfigurationError(AdaptiveConfigurationError):
    default_code = "MISSING_CONFIGURATION"


class MissingConfigurationSectionError(AdaptiveConfigurationError):
    default_code = "MISSING_CONFIGURATION_SECTION"


class InvalidConfigurationError(AdaptiveConfigurationError):
    default_code = "INVALID_CONFIGURATION"


class InvalidConfigurationValueError(AdaptiveConfigurationError):
    default_code = "INVALID_CONFIGURATION_VALUE"


class UnsupportedOptionError(AdaptiveConfigurationError):
    default_code = "UNSUPPORTED_OPTION"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class AdaptiveValidationError(AdaptiveError):
    default_code = "VALIDATION_ERROR"
    default_component = "validation"


class MissingFieldError(AdaptiveValidationError):
    default_code = "MISSING_FIELD"


class InvalidTypeError(AdaptiveValidationError, TypeError):
    default_code = "INVALID_TYPE"


class InvalidValueError(AdaptiveValidationError, ValueError):
    default_code = "INVALID_VALUE"


class RangeValidationError(AdaptiveValidationError, ValueError):
    default_code = "OUT_OF_RANGE"


class DimensionMismatchError(AdaptiveValidationError, ValueError):
    default_code = "DIMENSION_MISMATCH"


class EmptyCollectionError(AdaptiveValidationError, ValueError):
    default_code = "EMPTY_COLLECTION"


# ---------------------------------------------------------------------------
# Lifecycle and state errors
# ---------------------------------------------------------------------------


class AdaptiveStateError(AdaptiveError):
    default_code = "STATE_ERROR"
    default_component = "state"


class InitializationError(AdaptiveStateError):
    default_code = "INITIALIZATION_ERROR"


class ComponentNotInitializedError(InitializationError):
    default_code = "COMPONENT_NOT_INITIALIZED"


class PolicyNotInitializedError(ComponentNotInitializedError):
    default_code = "POLICY_NOT_INITIALIZED"
    default_component = "policy_manager"


class ModelNotFittedError(AdaptiveStateError, RuntimeError):
    default_code = "MODEL_NOT_FITTED"


class InvalidLifecycleStateError(AdaptiveStateError):
    default_code = "INVALID_LIFECYCLE_STATE"


class RegistryStateError(AdaptiveStateError):
    default_code = "REGISTRY_STATE_ERROR"


class EmptyRegistryError(RegistryStateError):
    default_code = "EMPTY_REGISTRY"


# ---------------------------------------------------------------------------
# Memory and replay errors
# ---------------------------------------------------------------------------


class AdaptiveMemoryError(AdaptiveError):
    default_code = "MEMORY_ERROR"
    default_component = "adaptive_memory"


class ExperienceValidationError(AdaptiveMemoryError):
    default_code = "INVALID_EXPERIENCE"


class ReplayBufferError(AdaptiveMemoryError):
    default_code = "REPLAY_BUFFER_ERROR"


class MemoryRetrievalError(AdaptiveMemoryError):
    default_code = "MEMORY_RETRIEVAL_ERROR"


class MemorySerializationError(AdaptiveMemoryError):
    default_code = "MEMORY_SERIALIZATION_ERROR"


class DriftDetectionError(AdaptiveMemoryError):
    default_code = "DRIFT_DETECTION_ERROR"


# ---------------------------------------------------------------------------
# Policy-management errors
# ---------------------------------------------------------------------------


class AdaptivePolicyError(AdaptiveError):
    default_code = "POLICY_ERROR"
    default_component = "policy_manager"


class SkillInitializationError(AdaptivePolicyError):
    default_code = "SKILL_INITIALIZATION_ERROR"


class InvalidSkillSpecificationError(AdaptivePolicyError):
    default_code = "INVALID_SKILL_SPECIFICATION"


class SkillSelectionError(AdaptivePolicyError):
    default_code = "SKILL_SELECTION_ERROR"


class PolicyUpdateError(AdaptivePolicyError):
    default_code = "POLICY_UPDATE_ERROR"


# ---------------------------------------------------------------------------
# Learning / worker errors
# ---------------------------------------------------------------------------


class AdaptiveLearningError(AdaptiveError):
    default_code = "LEARNING_ERROR"
    default_component = "learning"


class ReinforcementLearningError(AdaptiveLearningError):
    default_code = "RL_ERROR"
    default_component = "reinforcement_learning"


class ReturnComputationError(ReinforcementLearningError):
    default_code = "RETURN_COMPUTATION_ERROR"


class RewardNormalizationError(ReinforcementLearningError):
    default_code = "REWARD_NORMALIZATION_ERROR"


class ImitationLearningError(AdaptiveLearningError):
    default_code = "IMITATION_LEARNING_ERROR"
    default_component = "imitation_learning"


class ExpertPolicyError(ImitationLearningError):
    default_code = "EXPERT_POLICY_ERROR"


class DemonstrationError(ImitationLearningError):
    default_code = "DEMONSTRATION_ERROR"


class DemonstrationFormatError(DemonstrationError):
    default_code = "DEMONSTRATION_FORMAT_ERROR"


class DemonstrationPersistenceError(DemonstrationError):
    default_code = "DEMONSTRATION_PERSISTENCE_ERROR"


class DemonstrationNotFoundError(DemonstrationPersistenceError, FileNotFoundError):
    default_code = "DEMONSTRATION_NOT_FOUND"


class MetaLearningError(AdaptiveLearningError):
    default_code = "META_LEARNING_ERROR"
    default_component = "meta_learning"


class HyperparameterOptimizationError(MetaLearningError):
    default_code = "HYPERPARAMETER_OPTIMIZATION_ERROR"


class WorkerMetricsError(MetaLearningError):
    default_code = "WORKER_METRICS_ERROR"


class WorkerRegistryError(MetaLearningError):
    default_code = "WORKER_REGISTRY_ERROR"


# ---------------------------------------------------------------------------
# Neural-network and optimization errors
# ---------------------------------------------------------------------------


class AdaptiveNetworkError(AdaptiveError):
    default_code = "NETWORK_ERROR"
    default_component = "neural_network"


class NetworkConfigurationError(AdaptiveNetworkError):
    default_code = "NETWORK_CONFIGURATION_ERROR"


class UnsupportedActivationError(AdaptiveNetworkError):
    default_code = "UNSUPPORTED_ACTIVATION"


class UnsupportedInitializationMethodError(AdaptiveNetworkError):
    default_code = "UNSUPPORTED_INITIALIZATION_METHOD"


class UnsupportedOptimizerError(AdaptiveNetworkError):
    default_code = "UNSUPPORTED_OPTIMIZER"


class UnsupportedLossError(AdaptiveNetworkError):
    default_code = "UNSUPPORTED_LOSS"


class ForwardPassError(AdaptiveNetworkError):
    default_code = "FORWARD_PASS_ERROR"


class TrainingStepError(AdaptiveNetworkError):
    default_code = "TRAINING_STEP_ERROR"


# ---------------------------------------------------------------------------
# Regressor errors
# ---------------------------------------------------------------------------


class AdaptiveRegressorError(AdaptiveError):
    default_code = "REGRESSOR_ERROR"
    default_component = "sgd_regressor"


class UnsupportedLearningRateScheduleError(AdaptiveRegressorError):
    default_code = "UNSUPPORTED_LEARNING_RATE_SCHEDULE"


class PredictionBeforeFitError(AdaptiveRegressorError, RuntimeError):
    default_code = "PREDICT_BEFORE_FIT"


# ---------------------------------------------------------------------------
# Persistence and checkpoint errors
# ---------------------------------------------------------------------------


class AdaptivePersistenceError(AdaptiveError):
    default_code = "PERSISTENCE_ERROR"
    default_component = "persistence"


class CheckpointError(AdaptivePersistenceError):
    default_code = "CHECKPOINT_ERROR"


class CheckpointSaveError(CheckpointError):
    default_code = "CHECKPOINT_SAVE_ERROR"


class CheckpointLoadError(CheckpointError):
    default_code = "CHECKPOINT_LOAD_ERROR"


class CheckpointNotFoundError(CheckpointLoadError, FileNotFoundError):
    default_code = "CHECKPOINT_NOT_FOUND"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def ensure(
    condition: bool,
    message: str,
    *,
    exc_type: Type[AdaptiveError] = AdaptiveValidationError,
    code: Optional[str] = None,
    component: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    remediation: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    retryable: bool = False,
) -> None:
    """Raise a structured exception if condition is false."""
    if not condition:
        raise exc_type(
            message,
            code=code,
            component=component,
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
    exc_type: Type[AdaptiveError] = MissingFieldError,
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
    exc_type: Type[AdaptiveError] = InvalidTypeError,
) -> Any:
    if not isinstance(value, expected_types):
        expected_repr = (
            expected_types.__name__
            if isinstance(expected_types, type)
            else [t.__name__ for t in expected_types]
        )
        raise exc_type(
            f"'{name}' must be an instance of {expected_repr}, got {type(value).__name__}.",
            component=component,
            details={
                "field": name,
                "expected": expected_repr,
                "actual": type(value).__name__,
            },
            remediation=f"Validate the type of '{name}' before calling this operation.",
        )
    return value


def ensure_keys(
    mapping: Mapping[str, Any],
    required_keys: Iterable[str],
    *,
    name: str = "mapping",
    component: Optional[str] = None,
) -> Mapping[str, Any]:
    ensure_instance(mapping, Mapping, name, component=component)
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise MissingFieldError(
            f"'{name}' is missing required keys: {missing}.",
            component=component,
            details={"name": name, "missing_keys": missing},
            remediation=f"Populate all required keys for '{name}' before use.",
        )
    return mapping


def ensure_non_empty(
    value: Union[str, Sequence[Any], Mapping[str, Any]],
    name: str,
    *,
    component: Optional[str] = None,
) -> Union[str, Sequence[Any], Mapping[str, Any]]:
    if len(value) == 0:  # noqa: PLR2004
        raise EmptyCollectionError(
            f"'{name}' must not be empty.",
            component=component,
            details={"name": name},
            remediation=f"Provide at least one item in '{name}'.",
        )
    return value


def ensure_positive(
    value: Union[int, float],
    name: str,
    *,
    allow_zero: bool = False,
    component: Optional[str] = None,
) -> Union[int, float]:
    lower_ok = value >= 0 if allow_zero else value > 0
    if not lower_ok:
        operator = ">=" if allow_zero else ">"
        raise RangeValidationError(
            f"'{name}' must be {operator} 0, got {value}.",
            component=component,
            details={"name": name, "value": value, "allow_zero": allow_zero},
            remediation=f"Ensure '{name}' is initialized to a valid positive numeric value.",
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
    failed = False
    if minimum is not None and value < minimum:
        failed = True
    if maximum is not None and value > maximum:
        failed = True

    if failed:
        raise RangeValidationError(
            f"'{name}'={value} is outside the allowed range.",
            component=component,
            details={"name": name, "value": value, "minimum": minimum, "maximum": maximum},
            remediation=f"Clamp or validate '{name}' before invoking the operation.",
        )
    return value


def ensure_probability(
    value: Union[int, float],
    name: str,
    *,
    component: Optional[str] = None,
) -> Union[int, float]:
    return ensure_in_range(value, name, minimum=0.0, maximum=1.0, component=component)


def ensure_dimension(
    actual: int,
    expected: int,
    *,
    name: str,
    component: Optional[str] = None,
) -> None:
    if actual != expected:
        raise DimensionMismatchError(
            f"'{name}' has dimension {actual}, expected {expected}.",
            component=component,
            details={"name": name, "actual": actual, "expected": expected},
            remediation=f"Align '{name}' to the configured dimension before continuing.",
        )


def wrap_exception(
    exc: BaseException,
    exc_type: Type[AdaptiveError],
    message: str,
    *,
    component: Optional[str] = None,
    code: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
    remediation: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    retryable: bool = False,
) -> AdaptiveError:
    """
    Convert an arbitrary exception into a structured adaptive exception.

    If the incoming exception is already an AdaptiveError, it is returned as-is.
    """
    if isinstance(exc, AdaptiveError):
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
    # Base types
    "AdaptiveError",
    "ErrorSeverity",
    # Configuration
    "AdaptiveConfigurationError",
    "MissingConfigurationError",
    "MissingConfigurationSectionError",
    "InvalidConfigurationError",
    "InvalidConfigurationValueError",
    "UnsupportedOptionError",
    # Validation
    "AdaptiveValidationError",
    "MissingFieldError",
    "InvalidTypeError",
    "InvalidValueError",
    "RangeValidationError",
    "DimensionMismatchError",
    "EmptyCollectionError",
    # State
    "AdaptiveStateError",
    "InitializationError",
    "ComponentNotInitializedError",
    "PolicyNotInitializedError",
    "ModelNotFittedError",
    "InvalidLifecycleStateError",
    "RegistryStateError",
    "EmptyRegistryError",
    # Memory
    "AdaptiveMemoryError",
    "ExperienceValidationError",
    "ReplayBufferError",
    "MemoryRetrievalError",
    "MemorySerializationError",
    "DriftDetectionError",
    # Policy
    "AdaptivePolicyError",
    "SkillInitializationError",
    "InvalidSkillSpecificationError",
    "SkillSelectionError",
    "PolicyUpdateError",
    # Learning
    "AdaptiveLearningError",
    "ReinforcementLearningError",
    "ReturnComputationError",
    "RewardNormalizationError",
    "ImitationLearningError",
    "ExpertPolicyError",
    "DemonstrationError",
    "DemonstrationFormatError",
    "DemonstrationPersistenceError",
    "DemonstrationNotFoundError",
    "MetaLearningError",
    "HyperparameterOptimizationError",
    "WorkerMetricsError",
    "WorkerRegistryError",
    # Network
    "AdaptiveNetworkError",
    "NetworkConfigurationError",
    "UnsupportedActivationError",
    "UnsupportedInitializationMethodError",
    "UnsupportedOptimizerError",
    "UnsupportedLossError",
    "ForwardPassError",
    "TrainingStepError",
    # Regressor
    "AdaptiveRegressorError",
    "UnsupportedLearningRateScheduleError",
    "PredictionBeforeFitError",
    # Persistence
    "AdaptivePersistenceError",
    "CheckpointError",
    "CheckpointSaveError",
    "CheckpointLoadError",
    "CheckpointNotFoundError",
    # Helpers
    "ensure",
    "ensure_not_none",
    "ensure_instance",
    "ensure_keys",
    "ensure_non_empty",
    "ensure_positive",
    "ensure_in_range",
    "ensure_probability",
    "ensure_dimension",
    "wrap_exception",
]