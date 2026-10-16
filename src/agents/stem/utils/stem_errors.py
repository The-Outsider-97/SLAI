"""
STEM exception hierarchy.

All STEM-domain exceptions live here. They extend the shared BaseAgent
``BaseError`` contract so STEM errors retain deterministic codes, severity,
retry semantics, structured context, cause chaining, and serialization.
"""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Dict, Mapping, Optional, TypeVar

from ...base.utils.base_errors import BaseError, BaseErrorType


TSTEMError = TypeVar("TSTEMError", bound="STEMError")


class STEMError(BaseError):
    """
    Root of the STEM domain error hierarchy.

    Keeps STEM-specific stable codes while participating in the BaseAgent
    severity/category/retry/cause/context contract.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "stem_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "stem"

    def __init__(
        self,
        message: str,
        config: Optional[Mapping[str, Any]] = None,
        *,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if code is not None:
            kwargs["code"] = code
        if context is not None:
            kwargs["context"] = context

        super().__init__(message, config=config, **kwargs)


class STEMValidationError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_validation_error"
    default_severity = "medium"
    default_category = "stem.validation"


class STEMConfigurationError(STEMError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = "stem_configuration_error"
    default_severity = "high"
    default_category = "stem.configuration"


class STEMUnitError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_unit_error"
    default_severity = "medium"
    default_category = "stem.unit"


class STEMUnitConversionError(STEMUnitError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_unit_conversion_error"
    default_severity = "medium"
    default_category = "stem.unit.conversion"


class STEMIncommensurableUnitError(STEMUnitError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_incommensurable_unit_error"
    default_severity = "medium"
    default_category = "stem.unit.incommensurable"


class STEMPrefixError(STEMUnitError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_prefix_error"
    default_severity = "medium"
    default_category = "stem.unit.prefix"


class STEMUnitSystemError(STEMUnitError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_unit_system_error"
    default_severity = "medium"
    default_category = "stem.unit.system"


class STEMDimensionError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_dimension_error"
    default_severity = "medium"
    default_category = "stem.dimension"


class STEMDimensionMismatchError(STEMDimensionError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_dimension_mismatch_error"
    default_severity = "medium"
    default_category = "stem.dimension.mismatch"


class STEMBuckinghamPiError(STEMDimensionError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_buckingham_pi_error"
    default_severity = "medium"
    default_category = "stem.dimension.buckingham_pi"


class STEMUncertaintyError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_uncertainty_error"
    default_severity = "medium"
    default_category = "stem.uncertainty"


class STEMSolverError(STEMError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_solver_error"
    default_severity = "high"
    default_retryable = False
    default_category = "stem.solver"


class STEMDomainError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_domain_error"
    default_severity = "medium"
    default_category = "stem.domain"


class STEMEquationError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_equation_error"
    default_severity = "medium"
    default_category = "stem.equation"


class STEMPhysicalConstantError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_physical_constant_error"
    default_severity = "medium"
    default_category = "stem.physical_constant"


class STEMCalculusError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_calculus_error"
    default_severity = "medium"
    default_category = "stem.calculus"


class STEMNumericalError(STEMError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_numerical_error"
    default_severity = "high"
    default_category = "stem.numerical"


class STEMFloatingPointError(STEMNumericalError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_floating_point_error"
    default_severity = "high"
    default_retryable = False
    default_category = "stem.numerical.floating_point"


class STEMOverflowError(STEMNumericalError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_overflow_error"
    default_severity = "high"
    default_retryable = False
    default_category = "stem.numerical.overflow"


class STEMUnderflowError(STEMNumericalError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_underflow_error"
    default_severity = "low"
    default_retryable = False
    default_category = "stem.numerical.underflow"


class STEMPrecisionLossError(STEMNumericalError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_precision_loss_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "stem.numerical.precision_loss"


class STEMIllConditionedError(STEMNumericalError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_ill_conditioned_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "stem.numerical.ill_conditioned"


class STEMStatisticsError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_statistics_error"
    default_severity = "medium"
    default_category = "stem.statistics"


class STEMConvergenceError(STEMError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_convergence_error"
    default_severity = "high"
    default_retryable = True
    default_category = "stem.convergence"


class STEMLinearAlgebraError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_linear_algebra_error"
    default_severity = "high"
    default_category = "stem.linear_algebra"


class STEMSingularSystemError(STEMLinearAlgebraError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_singular_system_error"
    default_severity = "high"
    default_retryable = False
    default_category = "stem.linear_algebra.singular"


class STEMInterpolationError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_interpolation_error"
    default_severity = "medium"
    default_category = "stem.interpolation"


class STEMIntegrationError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_integration_error"
    default_severity = "medium"
    default_category = "stem.integration"


class STEMDifferentiationError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_differentiation_error"
    default_severity = "medium"
    default_category = "stem.differentiation"


class STEMODEError(STEMError):
    error_type = BaseErrorType.RUNTIME
    default_code = "stem_ode_error"
    default_severity = "high"
    default_category = "stem.ode"


class STEMPDEInterfaceError(STEMError):
    error_type = BaseErrorType.VALIDATION
    default_code = "stem_pde_interface_error"
    default_severity = "medium"
    default_category = "stem.pde_interface"


class STEMTemplateError(STEMError):
    """
    Raised when a STEM template cannot be resolved, read, parsed, or is
    otherwise invalid.

    Covers: non-string/empty template names, absolute paths, path-escape
    attempts outside the template root, missing template files, unreadable
    files, and malformed JSON payloads.
    """
    error_type = BaseErrorType.CONFIGURATION
    default_code = "stem_template_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "stem.template"


__all__ = [
    "STEMError",
    "STEMValidationError",
    "STEMConfigurationError",
    "STEMUnitError",
    "STEMUnitConversionError",
    "STEMIncommensurableUnitError",
    "STEMPrefixError",
    "STEMUnitSystemError",
    "STEMDimensionError",
    "STEMDimensionMismatchError",
    "STEMBuckinghamPiError",
    "STEMUncertaintyError",
    "STEMSolverError",
    "STEMDomainError",
    "STEMEquationError",
    "STEMPhysicalConstantError",
    "STEMCalculusError",
    "STEMNumericalError",
    "STEMFloatingPointError",
    "STEMOverflowError",
    "STEMUnderflowError",
    "STEMPrecisionLossError",
    "STEMIllConditionedError",
    "STEMStatisticsError",
    "STEMConvergenceError",
    "STEMLinearAlgebraError",
    "STEMSingularSystemError",
    "STEMInterpolationError",
    "STEMIntegrationError",
    "STEMDifferentiationError",
    "STEMODEError",
    "STEMPDEInterfaceError",
    "STEMTemplateError",
]