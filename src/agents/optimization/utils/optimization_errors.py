from __future__ import annotations

__version__ = "2.3.0"

import traceback

from typing import Any, Dict, Optional, Type, TypeVar

from ...base.utils.base_errors import BaseError, BaseErrorType


TOptimizationError = TypeVar("TOptimizationError", bound="OptimizationError")


class OptimizationError(BaseError):
    """
    Root of the Reasoning domain error hierarchy.

    Keeps Reasoning-specific stable codes while participating in the
    BaseAgent severity/category/retry/cause/context contract.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "optimization_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "optimization"

    def __init__(self, message: str, *, code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None) -> None:
        # was `pass`: without this the message/code/context never reach BaseError
        super().__init__(message, code=code, context=context)   # adjust to BaseError's real signature


class OptimizationTemplateError(OptimizationError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = "optimization_template_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "optimization.template" 


class OptimizationConfigurationError(OptimizationError):
    """Invalid pareto_efficiency settings."""
    error_type = BaseErrorType.CONFIGURATION
    default_code = "optimization_configuration_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "optimization.configuration"


class OptimizationValidationError(OptimizationError):
    """Bad objectives, candidates, limits, weights or reference points supplied by the caller."""
    error_type = BaseErrorType.RUNTIME
    default_code = "optimization_validation_error"
    default_severity = "low"
    default_retryable = False
    default_category = "optimization.validation"


class OptimizationEvaluationError(OptimizationError):
    """Evaluation failed during objective, metric, or constraint evaluation."""
    error_type = BaseErrorType.RUNTIME
    default_code = "optimization_evaluation_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "optimization.evaluation"


__all__ = [
    "OptimizationError",
    "OptimizationTemplateError",
    "OptimizationConfigurationError",
    "OptimizationValidationError",
    "OptimizationEvaluationError",
]