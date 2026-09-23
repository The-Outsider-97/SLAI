"""Verification-domain exceptions for SLAI v2.3.

All exceptions in this module derive from SLAI's existing ``BaseError``.  The
hierarchy is deliberately small: formal uncertainty belongs in
``VerificationResult`` and must not be converted into an exception.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ...base.utils.base_errors import BaseError, BaseErrorType


class VerificationError(BaseError):
    """Root error for malformed requests and verification infrastructure faults."""

    error_type = BaseErrorType.RUNTIME
    default_code = "verification_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "verification"

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
        super().__init__(
            message,
            code=code or self.default_code,
            context=dict(context or {}),
            cause=cause,
            retryable=self.default_retryable if retryable is None else bool(retryable),
            category="verification",
            component=kwargs.pop("component", "verification"),
            **kwargs,
        )


class MalformedSpecificationError(VerificationError):
    """Raised when a supplied formal object is structurally invalid."""

    error_type = BaseErrorType.VALIDATION
    default_code = "verification_malformed_specification"


class UnsupportedVerificationError(VerificationError):
    """Raised for a requested operation/evidence mode that cannot be supported."""

    error_type = BaseErrorType.VALIDATION
    default_code = "verification_unsupported_operation"


class SolverUnavailableError(VerificationError):
    """Raised when a requested optional SAT/SMT backend is not installed/usable."""

    error_type = BaseErrorType.DEPENDENCY
    default_code = "verification_solver_unavailable"
    default_retryable = True


class SolverBackendError(VerificationError):
    """Raised when an available backend fails while performing a solver operation."""

    error_type = BaseErrorType.RUNTIME
    default_code = "verification_solver_backend_failure"
    default_retryable = True


class InvalidTransitionModelError(VerificationError):
    """Raised when an explicit transition system violates its declared contract."""

    error_type = BaseErrorType.VALIDATION
    default_code = "verification_invalid_transition_model"


__all__ = [
    "InvalidTransitionModelError",
    "MalformedSpecificationError",
    "SolverBackendError",
    "SolverUnavailableError",
    "UnsupportedVerificationError",
    "VerificationError",
]
