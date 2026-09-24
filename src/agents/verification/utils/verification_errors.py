"""Verification-domain exceptions for SLAI v2.3.

All exceptions derive from SLAI's existing :class:`BaseError`.  The hierarchy
separates malformed formal input from execution/infrastructure faults while
keeping formal outcomes such as UNKNOWN, BOUNDED, SAT and UNSAT in result
objects rather than turning them into exceptions.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ...base.utils.base_errors import BaseError, BaseErrorType


class VerificationError(BaseError):
    """Root error for Verification-subsystem structural/infrastructure faults."""

    error_type = BaseErrorType.RUNTIME
    default_code = "verification_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "verification"
    default_component = "verification"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        retryable: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Create an SLAI-native Verification error.

        ``BaseError`` remains responsible for structured context, cause
        propagation, severity, retry metadata, serialisation, and logging.
        Verification only supplies domain-specific defaults.
        """

        super().__init__(
            message,
            code=code or self.default_code,
            context=dict(context or {}),
            cause=cause,
            retryable=self.default_retryable if retryable is None else bool(retryable),
            category=kwargs.pop("category", self.default_category),
            component=kwargs.pop("component", self.default_component),
            **kwargs,
        )


class MalformedSpecificationError(VerificationError):
    """A supplied formal artifact violates its declared structural contract."""

    error_type = BaseErrorType.VALIDATION
    default_code = "verification_malformed_specification"
    default_category = "verification.specification"
    default_component = "verification.formal"


class PredicateEvaluationError(VerificationError):
    """A caller-supplied state predicate failed while being evaluated."""

    error_type = BaseErrorType.RUNTIME
    default_code = "verification_predicate_evaluation"
    default_category = "verification.specification"
    default_component = "verification.formal"


class UnsupportedVerificationError(VerificationError):
    """The requested formal operation/evidence mode is intentionally unsupported."""

    error_type = BaseErrorType.VALIDATION
    default_code = "verification_unsupported_operation"
    default_category = "verification.unsupported"


class AbstractionError(VerificationError):
    """Failure while executing caller-supplied abstract-domain operations."""

    error_type = BaseErrorType.RUNTIME
    default_code = "verification_abstraction_failure"
    default_category = "verification.abstraction"
    default_component = "verification.formal.abstraction"


class AbstractionContractError(AbstractionError):
    """An abstract-domain operation violates the ordering/soundness contract."""

    error_type = BaseErrorType.VALIDATION
    default_code = "verification_abstraction_contract"
    default_retryable = False


class SolverUnavailableError(VerificationError):
    """A requested optional SAT/SMT backend is not installed or usable."""

    error_type = BaseErrorType.DEPENDENCY
    default_code = "verification_solver_unavailable"
    default_category = "verification.solver"
    default_component = "verification.solving"
    default_retryable = True


class SolverBackendError(VerificationError):
    """An available solver backend failed during a solver operation."""

    error_type = BaseErrorType.RUNTIME
    default_code = "verification_solver_backend_failure"
    default_category = "verification.solver"
    default_component = "verification.solving"
    default_retryable = True


class InvalidTransitionModelError(MalformedSpecificationError):
    """An explicit transition system violates its finite-model contract."""

    default_code = "verification_invalid_transition_model"
    default_category = "verification.model"
    default_component = "verification.model"


__all__ = [
    "AbstractionContractError",
    "AbstractionError",
    "InvalidTransitionModelError",
    "MalformedSpecificationError",
    "PredicateEvaluationError",
    "SolverBackendError",
    "SolverUnavailableError",
    "UnsupportedVerificationError",
    "VerificationError",
]
