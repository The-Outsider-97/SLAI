from __future__ import annotations

import traceback

from typing import Any, Dict, Optional, Type, TypeVar

from ...base.utils.base_errors import BaseError, BaseErrorType


TReasoningError = TypeVar("TReasoningError", bound="ReasoningError")


class ReasoningError(BaseError):
    """
    Root of the Reasoning domain error hierarchy.

    Keeps Reasoning-specific stable codes while participating in the
    BaseAgent severity/category/retry/cause/context contract.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "reasoning_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "reasoning"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
        recoverable: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        retryable = (
            self.default_retryable
            if recoverable is None
            else bool(recoverable)
        )

        super().__init__(
            message,
            code=code or self.default_code,
            context=context,
            cause=cause,
            retryable=retryable,
            category="reasoning",
            component=kwargs.pop(
                "component",
                "reasoning",
            ),
            **kwargs,
        )

        # v2.2/v2.3 compatibility.
        self.recoverable = self.retryable

    def to_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload["recoverable"] = (self.recoverable)
        return payload

    def with_context(self: TReasoningError, **context: Any) -> TReasoningError:
        self.add_context(**context)
        return self

    @classmethod
    def chain(cls: Type[TReasoningError], message: str, cause: BaseException, **context: Any) -> TReasoningError:
        return cls(message, cause=cause, context=context)

    def format_trace(self) -> str:
        lines = [
            f"{type(self).__name__}: "
            f"{self.message}"
        ]

        if self.context:
            lines.append("Context:")
            for key, value in (self.context.items()):
                lines.append(f"  {key}: {value}")

        if self.cause:
            lines.append("Caused by:")
            lines.extend(traceback.format_exception_only(type(self.cause), self.cause))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ReasoningValidationError(ReasoningError):
    error_type = BaseErrorType.VALIDATION
    default_code = "reasoning_validation_error"
    default_retryable = True


class FactNormalizationError(ReasoningValidationError):
    default_code = "fact_normalization_error"


class RuleDefinitionError(ReasoningValidationError):
    default_code = "rule_definition_error"


class ReasoningTypeError(ReasoningValidationError):
    default_code = "reasoning_type_error"


class ConfidenceBoundsError(ReasoningValidationError):
    default_code = "confidence_bounds_error"


# ---------------------------------------------------------------------------
# Configuration / resources
# ---------------------------------------------------------------------------

class ReasoningConfigurationError(ReasoningError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = "reasoning_configuration_error"
    default_severity = "high"


class ConfigLoadError(ReasoningConfigurationError):
    default_code = "config_load_error"


class ResourceLoadError(ReasoningError):
    error_type = BaseErrorType.RESOURCE
    default_code = "resource_load_error"


class DependencyUnavailableError(ReasoningError):
    error_type = BaseErrorType.DEPENDENCY
    default_code = "dependency_unavailable_error"
    default_retryable = True


# ---------------------------------------------------------------------------
# Knowledge / state / memory
# ---------------------------------------------------------------------------

class KnowledgeBaseError(ReasoningError):
    error_type = BaseErrorType.STATE
    default_code = "knowledge_base_error"


class ContradictionError(KnowledgeBaseError):
    default_code = "contradiction_error"
    default_retryable = True


class RedundancyError(KnowledgeBaseError):
    default_code = "redundancy_error"
    default_retryable = True


class KnowledgePersistenceError(KnowledgeBaseError):
    error_type = BaseErrorType.IO
    default_code = "knowledge_persistence_error"


class MemoryOperationError(ReasoningError):
    error_type = BaseErrorType.STATE
    default_code = "memory_operation_error"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class InferenceExecutionError(ReasoningError):
    error_type = BaseErrorType.RUNTIME
    default_code = "inference_execution_error"


class RuleExecutionError(InferenceExecutionError):
    default_code = "rule_execution_error"
    default_retryable = True


class CircularReasoningError(InferenceExecutionError):
    default_code = "circular_reasoning_error"


class ReasoningTimeoutError(InferenceExecutionError):
    error_type = BaseErrorType.TIMEOUT
    default_code = "reasoning_timeout_error"


class ConvergenceError(InferenceExecutionError):
    default_code = "convergence_error"


# ---------------------------------------------------------------------------
# Probabilistic / hybrid
# ---------------------------------------------------------------------------

class ProbabilisticModelError(ReasoningError):
    error_type = BaseErrorType.RUNTIME
    default_code = "probabilistic_model_error"


class ModelInitializationError(ProbabilisticModelError):
    error_type = BaseErrorType.INITIALIZATION
    default_code = "model_initialization_error"


class ModelInferenceError(ProbabilisticModelError):
    default_code = "model_inference_error"


class CircuitConstraintError(ProbabilisticModelError):
    default_code = "circuit_constraint_error"

    def describe_overlap(
        self,
    ) -> Optional[str]:
        violations = self.context.get(
            "violations"
        )

        if not violations:
            return None

        return "; ".join(
            str(item)
            for item in violations
        )


class TrainingError(ProbabilisticModelError):
    default_code = "training_error"


# ---------------------------------------------------------------------------
# Validation engine / external interaction / lifecycle
# ---------------------------------------------------------------------------

class ValidationEngineError(ReasoningError):
    error_type = BaseErrorType.VALIDATION
    default_code = "validation_engine_error"


class ExternalServiceError(ReasoningError):
    error_type = BaseErrorType.EXTERNAL_SERVICE
    default_code = "external_service_error"
    default_retryable = True


class AgentLifecycleError(ReasoningError):
    error_type = BaseErrorType.STATE
    default_code = "agent_lifecycle_error"


__all__ = [
    "ReasoningError",
    "ReasoningValidationError",
    "FactNormalizationError",
    "RuleDefinitionError",
    "ReasoningTypeError",
    "ConfidenceBoundsError",
    "ReasoningConfigurationError",
    "ConfigLoadError",
    "ResourceLoadError",
    "DependencyUnavailableError",
    "KnowledgeBaseError",
    "ContradictionError",
    "RedundancyError",
    "KnowledgePersistenceError",
    "MemoryOperationError",
    "InferenceExecutionError",
    "RuleExecutionError",
    "CircularReasoningError",
    "ReasoningTimeoutError",
    "ConvergenceError",
    "ProbabilisticModelError",
    "ModelInitializationError",
    "ModelInferenceError",
    "CircuitConstraintError",
    "TrainingError",
    "ValidationEngineError",
    "ExternalServiceError",
    "AgentLifecycleError",
]