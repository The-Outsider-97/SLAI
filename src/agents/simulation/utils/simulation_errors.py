"""Simulation-domain exception hierarchy for SLAI v2.3.

The hierarchy participates in SLAI's :mod:`src.agents.base.utils.base_errors`
contract while keeping stable Simulation-specific error codes.  It contains no
simulation algorithms and deliberately does not implement retry, recovery, or
agent lifecycle behavior.
"""

from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Mapping
from typing import Any, Optional

from ...base.utils.base_errors import BaseError, BaseErrorType


class SimulationError(BaseError):
    """Root exception for the Simulation subsystem."""

    error_type = BaseErrorType.RUNTIME
    default_code = "SIM-1000"
    default_severity = "medium"
    default_retryable = False
    default_category = "simulation"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        severity: Optional[str] = None,
        retryable: Optional[bool] = None,
        context: Optional[Mapping[str, Any]] = None,
        details: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        operation: Optional[str] = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "code": code or self.default_code,
            "context": dict(context or {}),
            "details": dict(details or {}),
            "cause": cause,
            "operation": operation,
            "component": "simulation",
        }
        if severity is not None:
            kwargs["severity"] = severity
        if retryable is not None:
            kwargs["retryable"] = retryable
        super().__init__(message, None, **kwargs)


class SimulationConfigurationError(SimulationError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = "SIM-1100"
    default_severity = "high"
    default_category = "simulation.configuration"


class SimulationValidationError(SimulationError):
    error_type = BaseErrorType.VALIDATION
    default_code = "SIM-1200"
    default_category = "simulation.validation"


class SimulationStateError(SimulationError):
    error_type = BaseErrorType.STATE
    default_code = "SIM-1201"
    default_category = "simulation.state"


class SimulationProbabilityError(SimulationValidationError):
    default_code = "SIM-1202"
    default_category = "simulation.probability"


class SimulationModelError(SimulationError):
    default_code = "SIM-1300"
    default_severity = "high"
    default_category = "simulation.model"


class SimulationTransitionError(SimulationError):
    default_code = "SIM-1301"
    default_severity = "high"
    default_category = "simulation.transition"


class SimulationNumericalError(SimulationError):
    default_code = "SIM-1302"
    default_severity = "high"
    default_category = "simulation.numerical"


class SimulationDivergenceError(SimulationNumericalError):
    default_code = "SIM-1303"
    default_category = "simulation.divergence"


class SimulationTimeoutError(SimulationError):
    error_type = BaseErrorType.TIMEOUT
    default_code = "SIM-1400"
    default_severity = "high"
    default_category = "simulation.timeout"


class SimulationCancelledError(SimulationError):
    default_code = "SIM-1401"
    default_category = "simulation.cancelled"


class SimulationBranchLimitError(SimulationError):
    error_type = BaseErrorType.RESOURCE
    default_code = "SIM-1500"
    default_severity = "high"
    default_category = "simulation.branch_limit"


class SimulationMemoryError(SimulationError):
    default_code = "SIM-1600"
    default_category = "simulation.memory"


class SimulationCallbackError(SimulationError):
    default_code = "SIM-1800"
    default_category = "simulation.callback"


class SimulationReproducibilityError(SimulationError):
    default_code = "SIM-1700"
    default_severity = "high"
    default_category = "simulation.reproducibility"


__all__ = [
    "SimulationError",
    "SimulationConfigurationError",
    "SimulationValidationError",
    "SimulationStateError",
    "SimulationProbabilityError",
    "SimulationModelError",
    "SimulationTransitionError",
    "SimulationNumericalError",
    "SimulationDivergenceError",
    "SimulationTimeoutError",
    "SimulationCancelledError",
    "SimulationCallbackError",
    "SimulationBranchLimitError",
    "SimulationMemoryError",
    "SimulationReproducibilityError",
]
