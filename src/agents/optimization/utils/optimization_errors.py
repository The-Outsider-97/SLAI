"""Structured exception hierarchy for SLAI mathematical optimization.

The hierarchy participates in SLAI's BaseError contract while keeping stable,
domain-specific categories for modeling, registry, solver, evaluation, and
multi-objective failures.  Errors intentionally carry structured context rather
than being collapsed to ``None`` or broad generic exceptions.
"""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Mapping

from ...base.utils.base_errors import BaseError, BaseErrorType


class OptimizationError(BaseError):
    error_type = BaseErrorType.RUNTIME
    default_code = "OPT-1000"
    default_severity = "medium"
    default_retryable = False
    default_category = "optimization"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
        retryable: bool | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "code": code or self.default_code,
            "context": dict(context or {}),
            "cause": cause,
        }
        if retryable is not None:
            kwargs["retryable"] = retryable
        super().__init__(message, **kwargs)


class OptimizationConfigurationError(OptimizationError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = "OPT-1100"
    default_category = "optimization.configuration"


class OptimizationTemplateError(OptimizationError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = "OPT-1101"
    default_category = "optimization.template"


class OptimizationValidationError(OptimizationError):
    error_type = BaseErrorType.VALIDATION
    default_code = "OPT-1200"
    default_severity = "low"
    default_category = "optimization.validation"


class MalformedProblemError(OptimizationValidationError):
    default_code = "OPT-1201"
    default_category = "optimization.problem.malformed"


class UnsupportedProblemError(OptimizationError):
    default_code = "OPT-1202"
    default_category = "optimization.problem.unsupported"


class IncompatibleSolverError(OptimizationError):
    default_code = "OPT-1300"
    default_category = "optimization.solver.incompatible"


class SolverUnavailableError(OptimizationError):
    error_type = BaseErrorType.DEPENDENCY
    default_code = "OPT-1301"
    default_category = "optimization.solver.unavailable"


class SolverExecutionError(OptimizationError):
    default_code = "OPT-1302"
    default_category = "optimization.solver.execution"


class InvalidSolverResultError(OptimizationError):
    default_code = "OPT-1303"
    default_category = "optimization.solver.result"


class NumericalFailureError(OptimizationError):
    default_code = "OPT-1304"
    default_category = "optimization.solver.numerical"


class InfeasibleProblemError(OptimizationError):
    default_code = "OPT-1400"
    default_category = "optimization.problem.infeasible"


class UnboundedProblemError(OptimizationError):
    default_code = "OPT-1401"
    default_category = "optimization.problem.unbounded"


class OptimizationTimeoutError(OptimizationError):
    error_type = BaseErrorType.TIMEOUT
    default_code = "OPT-1500"
    default_retryable = True
    default_category = "optimization.timeout"


class EvaluationLimitError(OptimizationError):
    default_code = "OPT-1501"
    default_category = "optimization.evaluation.limit"


class OptimizationEvaluationError(OptimizationError):
    default_code = "OPT-1600"
    default_category = "optimization.evaluation"


class RegistryError(OptimizationError):
    default_code = "OPT-1700"
    default_category = "optimization.registry"


class MultiObjectiveError(OptimizationError):
    default_code = "OPT-1800"
    default_category = "optimization.multiobjective"


class InterruptedOptimizationError(OptimizationError):
    default_code = "OPT-1900"
    default_category = "optimization.interrupted"


class StopForTime(RuntimeError):
    """Internal control signal raised when the configured wall-clock budget expires."""


class StopForEvaluations(RuntimeError):
    """Internal control signal raised when the function-evaluation budget expires."""


__all__ = [
    "EvaluationLimitError",
    "IncompatibleSolverError",
    "InfeasibleProblemError",
    "InterruptedOptimizationError",
    "InvalidSolverResultError",
    "MalformedProblemError",
    "MultiObjectiveError",
    "NumericalFailureError",
    "OptimizationConfigurationError",
    "OptimizationError",
    "OptimizationEvaluationError",
    "OptimizationTemplateError",
    "OptimizationTimeoutError",
    "OptimizationValidationError",
    "RegistryError",
    "SolverExecutionError",
    "SolverUnavailableError",
    "UnboundedProblemError",
    "UnsupportedProblemError",
    "StopForTime",
    "StopForEvaluations",
]
