from __future__ import annotations

__version__ = "2.3.0"

import traceback

from typing import Any, Dict, Optional, Type, TypeVar

from ...base.utils.base_errors import BaseError, BaseErrorType


TProvenanceError = TypeVar("TProvenanceError", bound="ProvenanceError")


class ProvenanceError(BaseError):
    """
    Root of the Provenance domain error hierarchy.

    Keeps Provenance-specific stable codes while participating in the
    BaseAgent severity/category/retry/cause/context contract.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "provenance_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "provenance"

    def __init__(self, message: str,
            *,
            code: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None) -> None:
        pass

class ProvenanceConfigurationError(ProvenanceError):
    """
    Raised when there is a configuration error in the Provenance domain.
    """

    default_code = "provenance_configuration_error"
    default_severity = "high"
    default_retryable = False
    default_category = "configuration"


class ProvenanceValidationError(ProvenanceError):
    """
    Raised when there is a validation error in the Provenance domain.
    """

    default_code = "provenance_validation_error"
    default_severity = "medium"
    default_retryable = False
    default_category = "validation"


class ProvenanceTrackingError(ProvenanceError):
    """
    Raised when there is an error in tracking provenance information.
    """

    default_code = "provenance_tracking_error"
    default_severity = "medium"
    default_retryable = True
    default_category = "tracking"


class ProvenanceStorageError(ProvenanceError):
    """
    Raised when there is an error in storing or retrieving provenance information.
    """

    default_code = "provenance_storage_error"
    default_severity = "high"
    default_retryable = True
    default_category = "storage" 


class ProvenanceCustodyError(ProvenanceError):
    """
    Raised when there is an error in managing the custody of artifacts.
    """

    default_code = "provenance_custody_error"
    default_severity = "medium"
    default_retryable = True
    default_category = "custody"

__all__ = [
    "ProvenanceError",
    "ProvenanceConfigurationError",
    "ProvenanceValidationError",
    "ProvenanceTrackingError",
    "ProvenanceStorageError",
    "ProvenanceCustodyError"
]
