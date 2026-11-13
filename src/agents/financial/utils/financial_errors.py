"""SLAI Financial subsystem: stable, structured financial exceptions.

All public errors extend SLAI's canonical ``BaseError`` contract.  Errors carry
stable FIN-xxxx codes, subtype-specific severity/retry classification, context,
and an optional chained cause.  No exception is logged when merely constructed:
call ``error.log()`` explicitly at a handling boundary to avoid duplicate logs.

Security: ``log()`` deliberately emits only nonsensitive classification fields.
Callers must never put credentials, card numbers, account IDs, or personal data
in the error message or context they intend to expose externally.
"""

from __future__ import annotations

__version__ = "2.4.0"

import logging

from collections.abc import Mapping
from typing import Any, Optional, TypeVar

from ...base.utils.base_errors import BaseError, BaseErrorType
from logs.logger import get_logger # pyright: ignore[reportMissingImports]

TFinancialError = TypeVar("TFinancialError", bound="FinancialError")

logger = get_logger("Financial Errors")


class FinancialError(BaseError):
    """Root of the Financial domain error hierarchy.

    Every subtype participates in SLAI BaseError classification, serialization,
    and explicit logging. Avoid embedding credentials or private account data
    in messages or structured context.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "FIN-1000"
    default_severity = "medium"
    default_retryable = False
    default_category = "financial"
    default_resolution_hint: str | None = None

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
        retryable: bool | None = None,
        severity: str | None = None,
        resolution_hint: str | None = None,
        operation: str | None = None,
    ) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("FinancialError message must be non-empty text")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("FinancialError context must be a mapping")
        if cause is not None and not isinstance(cause, BaseException):
            raise TypeError("FinancialError cause must be an exception")
        kwargs: dict[str, Any] = {
            "code": code or self.default_code,
            "context": dict(context or {}),
            "cause": cause,
            "category": self.default_category,
            "component": "financial",
            "operation": operation,
            "severity": severity or self.default_severity,
            "retryable": self.default_retryable if retryable is None else retryable,
            "resolution_hint": resolution_hint or self.default_resolution_hint,
            "auto_log": False,
        }
        super().__init__(message.strip(), **kwargs)

    @property
    def error_code(self) -> str:
        """Backwards-compatible, stable alias for the BaseError ``code``."""
        return self.code

    def log(
        self,
        level: int | None = None,
        include_traceback: bool = False,
        extra: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        """Log classification without raw context, messages or PII.

        The logging caller decides whether an exception merits a log entry.
        Explicitly not logging the full error payload prevents accidental
        disclosure of untrusted financial content in shared operational logs.
        """
        default_level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(self.severity, logging.ERROR)
        include_traceback = include_traceback and self.cause is not None
        extra = dict(extra or {})
        logger.log(
            default_level if level is None else level,
            "Financial error code=%s type=%s category=%s retryable=%s",
            self.code,
            type(self).__name__,
            self.category,
            self.retryable,
            extra=extra,
            exc_info=self.cause if include_traceback else None,
        )


class FinancialConfigurationError(FinancialError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = "FIN-1100"
    default_severity = "high"
    default_category = "financial.configuration"


class FinancialValidationError(FinancialError):
    error_type = BaseErrorType.VALIDATION
    default_code = "FIN-1200"
    default_severity = "medium"
    default_category = "financial.validation"


class InvalidCurrencyError(FinancialValidationError):
    """Malformed, unrecognized or disallowed currency code."""
    default_code = "FIN-1201"
    default_category = "financial.currency.invalid"


class CurrencyMismatchError(FinancialValidationError):
    """Amounts cannot be combined without an explicit FX conversion."""
    default_code = "FIN-1202"
    default_category = "financial.currency.mismatch"


class UnsupportedConventionError(FinancialValidationError):
    """Unsupported day count, rate, date, or accounting convention."""
    default_code = "FIN-1203"
    default_category = "financial.convention.unsupported"


class InvalidFinancialPeriodError(FinancialValidationError):
    default_code = "FIN-1204"
    default_category = "financial.period.invalid"


class UnsupportedFinancialOperationError(FinancialValidationError):
    """Requested domain/operation is unknown or not enabled for this runtime."""
    default_code = "FIN-1205"
    default_category = "financial.operation.unsupported"


class FinancialSerializationError(FinancialError):
    error_type = BaseErrorType.SERIALIZATION
    default_code = "FIN-1300"
    default_category = "financial.serialization"


class FinancialDataError(FinancialError):
    error_type = BaseErrorType.EXTERNAL_SERVICE
    default_code = "FIN-1400"
    default_category = "financial.data"


class MissingMarketDataError(FinancialDataError):
    error_type = BaseErrorType.VALIDATION
    default_code = "FIN-1401"
    default_category = "financial.market.missing"


class StaleMarketDataError(FinancialDataError):
    error_type = BaseErrorType.STATE
    default_code = "FIN-1402"
    default_category = "financial.market.stale"


class InvalidMarketDataError(FinancialDataError):
    error_type = BaseErrorType.VALIDATION
    default_code = "FIN-1403"
    default_category = "financial.market.invalid"


class FinancialCalculationError(FinancialError):
    default_code = "FIN-1500"
    default_category = "financial.calculation"


class FinancialDomainError(FinancialCalculationError):
    """A calculation is undefined for the supplied financial parameters."""
    default_code = "FIN-1501"
    default_category = "financial.calculation.domain"


class FinancialPrecisionError(FinancialCalculationError):
    default_code = "FIN-1502"
    default_category = "financial.calculation.precision"


class FinancialConvergenceError(FinancialCalculationError):
    default_code = "FIN-1503"
    default_category = "financial.calculation.convergence"


class FinancialModelError(FinancialError):
    default_code = "FIN-1600"
    default_category = "financial.model"


class FinancialModelUnavailableError(FinancialModelError):
    error_type = BaseErrorType.DEPENDENCY
    default_code = "FIN-1601"
    default_category = "financial.model.unavailable"


class FinancialModelValidationError(FinancialModelError):
    error_type = BaseErrorType.VALIDATION
    default_code = "FIN-1602"
    default_category = "financial.model.validation"


class FinancialAuthorizationError(FinancialError):
    error_type = BaseErrorType.VALIDATION
    default_code = "FIN-1700"
    default_severity = "high"
    default_category = "financial.authorization"


class UnauthorizedFinancialOperationError(FinancialAuthorizationError):
    """An operation was not authorized under the caller's permissions."""
    default_code = "FIN-1701"
    default_category = "financial.authorization.denied"


class FinancialIntegrityError(FinancialError):
    default_code = "FIN-1800"
    default_severity = "high"
    default_category = "financial.integrity"


class FinancialProviderError(FinancialError):
    error_type = BaseErrorType.EXTERNAL_SERVICE
    default_code = "FIN-1900"
    default_severity = "high"
    default_retryable = True
    default_category = "financial.provider"


class FinancialProviderTimeoutError(FinancialProviderError):
    error_type = BaseErrorType.TIMEOUT
    default_code = "FIN-1901"
    default_category = "financial.provider.timeout"


class FinancialDelegationError(FinancialError):
    """A delegated SLAI component (optimizer, safety) was unavailable or answered unusably."""
    error_type = BaseErrorType.EXTERNAL_SERVICE
    default_code = "FIN-2000"
    default_severity = "medium"
    default_category = "financial.delegation"


__all__ = [
    "FinancialError",
    "FinancialConfigurationError",
    "FinancialValidationError",
    "InvalidCurrencyError",
    "CurrencyMismatchError",
    "UnsupportedConventionError",
    "InvalidFinancialPeriodError",
    "UnsupportedFinancialOperationError",
    "FinancialSerializationError",
    "FinancialDataError",
    "MissingMarketDataError",
    "StaleMarketDataError",
    "InvalidMarketDataError",
    "FinancialCalculationError",
    "FinancialDomainError",
    "FinancialPrecisionError",
    "FinancialConvergenceError",
    "FinancialModelError",
    "FinancialModelUnavailableError",
    "FinancialModelValidationError",
    "FinancialAuthorizationError",
    "UnauthorizedFinancialOperationError",
    "FinancialIntegrityError",
    "FinancialProviderError",
    "FinancialProviderTimeoutError",
    "FinancialDelegationError",
]
