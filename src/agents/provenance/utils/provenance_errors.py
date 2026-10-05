"""Structured exception hierarchy for SLAI provenance infrastructure."""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Dict, Mapping, Optional

from ...base.utils.base_errors import BaseError, BaseErrorType
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Provenance Errors")
printer = PrettyPrinter()


class ProvenanceError(BaseError):
    """Root error for provenance-specific failures.

    Provenance errors retain BaseError's machine-readable severity, retry,
    cause, and context contract while assigning stable subsystem-specific codes.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "PROV-1000"
    default_severity = "medium"
    default_retryable = False
    default_category = "provenance"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        severity: Optional[str] = None,
        retryable: Optional[bool] = None,
        operation: Optional[str] = None,
        resolution_hint: Optional[str] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {
            "code": code or self.default_code,
            "context": dict(context or {}),
            "cause": cause,
            "component": "provenance",
            "operation": operation,
            "resolution_hint": resolution_hint,
        }
        if severity is not None:
            kwargs["severity"] = severity
        if retryable is not None:
            kwargs["retryable"] = retryable
        super().__init__(message, None, **kwargs)


class ProvenanceConfigurationError(ProvenanceError):
    """Invalid or inconsistent provenance subsystem configuration."""

    error_type = BaseErrorType.CONFIGURATION
    default_code = "PROV-1100"
    default_severity = "high"
    default_category = "configuration"


class ProvenanceValidationError(ProvenanceError):
    """A provenance record violates its structural/data contract."""

    error_type = BaseErrorType.VALIDATION
    default_code = "PROV-1200"
    default_category = "validation"


class ProvenanceGraphError(ProvenanceValidationError):
    """The provenance graph is malformed, cyclic, or internally inconsistent."""

    default_code = "PROV-1201"
    default_category = "graph"


class ProvenanceConflictError(ProvenanceValidationError):
    """A stable identity was reused with conflicting immutable content."""

    default_code = "PROV-1202"
    default_severity = "high"
    default_category = "identity_conflict"


class ProvenanceNotFoundError(ProvenanceError):
    """A requested provenance artifact/source/record is unknown."""

    error_type = BaseErrorType.STATE
    default_code = "PROV-1203"
    default_severity = "low"
    default_category = "not_found"


class ProvenanceTrackingError(ProvenanceError):
    """A provenance capture or reconstruction operation failed."""

    default_code = "PROV-1300"
    default_retryable = True
    default_category = "tracking"


class ProvenanceLineageError(ProvenanceTrackingError):
    """Lineage capture or reconstruction failed."""

    default_code = "PROV-1301"
    default_category = "lineage"


class ProvenanceSourceError(ProvenanceTrackingError):
    """Source registration or source-identity lookup failed."""

    default_code = "PROV-1302"
    default_category = "source"


class ProvenanceReproducibilityError(ProvenanceTrackingError):
    """Reproducibility metadata could not be reconstructed or interpreted."""

    default_code = "PROV-1303"
    default_category = "reproducibility"


class ProvenanceStorageError(ProvenanceError):
    """Provenance persistence or retrieval failed."""

    error_type = BaseErrorType.IO
    default_code = "PROV-1400"
    default_severity = "high"
    default_retryable = True
    default_category = "storage"


class ProvenanceCustodyError(ProvenanceError):
    """Chain-of-custody capture or retrieval failed."""

    error_type = BaseErrorType.STATE
    default_code = "PROV-1500"
    default_category = "custody"


__all__ = [
    "ProvenanceConfigurationError",
    "ProvenanceConflictError",
    "ProvenanceCustodyError",
    "ProvenanceError",
    "ProvenanceGraphError",
    "ProvenanceLineageError",
    "ProvenanceNotFoundError",
    "ProvenanceReproducibilityError",
    "ProvenanceSourceError",
    "ProvenanceStorageError",
    "ProvenanceTrackingError",
    "ProvenanceValidationError",
]


if __name__ == "__main__":
    configure_logging()
    printer.status("PROVENANCE", "provenance error hierarchy loaded", "success")
