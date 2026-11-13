"""Software-domain exception hierarchy for SLAI v2.4.

The hierarchy participates in SLAI's :mod:`src.agents.base.utils.base_errors`
contract while keeping stable Software-specific error codes.  It contains no
analysis logic and deliberately implements no retry, recovery, or lifecycle
behavior.

Analytical findings (unused imports, breaking API changes, high change risk)
are *results*, never exceptions.  Exceptions describe inputs the subsystem
cannot interpret, limits it refuses to exceed, and refactoring/patch
preconditions that do not hold.
"""

from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping
from typing import Any, Optional, TypeVar

from ...base.utils.base_errors import BaseError, BaseErrorType


TSoftwareError = TypeVar("TSoftwareError", bound="SoftwareError")


class SoftwareError(BaseError):
    """Root of the Software domain error hierarchy.

    Keeps Software-specific stable codes while participating in the BaseAgent
    severity/category/retry/cause/context contract.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "SFT-1000"
    default_severity = "medium"
    default_retryable = False
    default_category = "software"

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
            "component": "software",
        }
        if severity is not None:
            kwargs["severity"] = severity
        if retryable is not None:
            kwargs["retryable"] = retryable
        super().__init__(message, None, **kwargs)


class SoftwareConfigurationError(SoftwareError):
    error_type = BaseErrorType.CONFIGURATION
    default_code = "SFT-1100"
    default_severity = "high"
    default_category = "software.configuration"


class SoftwareValidationError(SoftwareError):
    error_type = BaseErrorType.VALIDATION
    default_code = "SFT-1200"
    default_category = "software.validation"


class SoftwareRequestError(SoftwareValidationError):
    """A request mapping violates the SoftwareAgent request contract."""

    default_code = "SFT-1201"
    default_category = "software.request"


class SoftwareSnapshotError(SoftwareValidationError):
    """A supplied repository snapshot is malformed (paths, duplicates, types)."""

    default_code = "SFT-1202"
    default_category = "software.snapshot"


class SoftwareStateError(SoftwareError):
    error_type = BaseErrorType.STATE
    default_code = "SFT-1210"
    default_category = "software.state"


class SoftwareParseError(SoftwareError):
    """Raised only in strict mode; lenient parsing records diagnostics instead."""

    error_type = BaseErrorType.VALIDATION
    default_code = "SFT-1300"
    default_category = "software.parse"


class UnsupportedLanguageError(SoftwareParseError):
    default_code = "SFT-1301"
    default_category = "software.parse.language"


class SoftwareModelError(SoftwareError):
    """A program model could not be built, found, or queried."""

    default_code = "SFT-1310"
    default_category = "software.model"


class SoftwareSymbolError(SoftwareError):
    """A symbol is unknown, ambiguous, or unsuitable for the requested query."""

    error_type = BaseErrorType.VALIDATION
    default_code = "SFT-1320"
    default_category = "software.symbol"


class SoftwareGraphError(SoftwareError):
    default_code = "SFT-1400"
    default_category = "software.graph"


class SoftwareLimitError(SoftwareError):
    """A configured size/complexity bound would be exceeded."""

    error_type = BaseErrorType.RESOURCE
    default_code = "SFT-1500"
    default_severity = "high"
    default_category = "software.limit"


class SoftwareAnalysisError(SoftwareError):
    default_code = "SFT-1600"
    default_category = "software.analysis"


class SoftwareRefactoringError(SoftwareError):
    """A refactoring precondition failed (ambiguity, collision, unsupported)."""

    error_type = BaseErrorType.VALIDATION
    default_code = "SFT-1700"
    default_category = "software.refactoring"


class SoftwareNameConflictError(SoftwareRefactoringError):
    default_code = "SFT-1701"
    default_category = "software.refactoring.conflict"


class SoftwarePatchError(SoftwareError):
    """An edit set or diff is invalid, stale, overlapping, or breaks syntax."""

    error_type = BaseErrorType.VALIDATION
    default_code = "SFT-1800"
    default_severity = "high"
    default_category = "software.patch"


class SoftwareMemoryError(SoftwareError):
    default_code = "SFT-1900"
    default_category = "software.memory"


class UnsupportedSoftwareOperationError(SoftwareError):
    error_type = BaseErrorType.VALIDATION
    default_code = "SFT-1950"
    default_category = "software.unsupported"


__all__ = [
    "SoftwareAnalysisError",
    "SoftwareConfigurationError",
    "SoftwareError",
    "SoftwareGraphError",
    "SoftwareLimitError",
    "SoftwareMemoryError",
    "SoftwareModelError",
    "SoftwareNameConflictError",
    "SoftwareParseError",
    "SoftwarePatchError",
    "SoftwareRefactoringError",
    "SoftwareRequestError",
    "SoftwareSnapshotError",
    "SoftwareStateError",
    "SoftwareSymbolError",
    "SoftwareValidationError",
    "TSoftwareError",
    "UnsupportedLanguageError",
    "UnsupportedSoftwareOperationError",
]
