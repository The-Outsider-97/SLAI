"""Error taxonomy for the SLAI QNN numerical boundary.

This module defines a hierarchy of QNN-specific exceptions, each with a
deterministic error code (e.g., QNN-1001) that can be used for logging,
monitoring, and structured error handling. New error types should be added
here and assigned a unique code.
"""

from __future__ import annotations

import os
import json
import time
import hashlib

from enum import Enum
from typing import Dict, Any, Optional, Callable, Type, ClassVar


# -----------------------------------------------------------------------------
# Machine‑readable error codes
# -----------------------------------------------------------------------------
# Each exception class is assigned a unique code. The base code for the
# QNN capability is 1000; sub‑codes are assigned sequentially.
class QNNErrorCode:
    """Central registry of deterministic error codes for QNN exceptions."""

    BASE = 1000

    # Base error
    QNN_ERROR = BASE

    # Configuration and input errors (1001‑1099)
    CONFIGURATION_ERROR = BASE + 1
    INPUT_ERROR = BASE + 2
    RESOURCE_LIMIT_ERROR = BASE + 3
    CHECKPOINT_STATE_ERROR = BASE + 4
    DIMENSION_ERROR = BASE + 5
    STATE_ERROR = BASE + 6
    GRADIENT_ERROR = BASE + 7
    NORM_ERROR = BASE + 8

    # Additional numeric errors (1100‑1199)
    FINITE_VALUE_ERROR = BASE + 9
    CONVERSION_ERROR = BASE + 10

    @classmethod
    def get_code(cls, exception_type: Type[QNNError]) -> int:
        """Return the numeric error code for a given QNNError subclass."""
        mapping = {
            QNNError: cls.QNN_ERROR,
            QNNConfigurationError: cls.CONFIGURATION_ERROR,
            QNNInputError: cls.INPUT_ERROR,
            QNNResourceLimitError: cls.RESOURCE_LIMIT_ERROR,
            QNNCheckpointStateError: cls.CHECKPOINT_STATE_ERROR,
            QNNDimensionError: cls.DIMENSION_ERROR,
            QNNStateError: cls.STATE_ERROR,
            QNNGradientError: cls.GRADIENT_ERROR,
            QNNNormError: cls.NORM_ERROR,
            QNNFiniteValueError: cls.FINITE_VALUE_ERROR,
            QNNConversionError: cls.CONVERSION_ERROR,
        }
        return mapping.get(exception_type, cls.QNN_ERROR)


# -----------------------------------------------------------------------------
# Exception hierarchy
# -----------------------------------------------------------------------------
class QNNError(Exception):
    """Base error for the QNN numerical capability.

    All QNN-specific exceptions inherit from this class. Each subclass
    should be assigned a unique error code via QNNErrorCode.
    """

    # Override in subclasses to provide a default error code.
    code: ClassVar[int] = QNNErrorCode.QNN_ERROR

    def __init__(self, message: str = "", *args: Any) -> None:
        self.message = message
        super().__init__(message, *args)

    def __str__(self) -> str:
        return f"[QNN-{self.code:04d}] {self.message}"


class QNNConfigurationError(QNNError, ValueError):
    """Raised when QNN configuration violates a declared invariant."""
    code = QNNErrorCode.CONFIGURATION_ERROR


class QNNInputError(QNNError, ValueError):
    """Raised when a state vector, task, or model-state payload is malformed."""
    code = QNNErrorCode.INPUT_ERROR


class QNNResourceLimitError(QNNInputError):
    """Raised before a QNN operation exceeds its configured resource policy."""
    code = QNNErrorCode.RESOURCE_LIMIT_ERROR


class QNNCheckpointStateError(QNNInputError):
    """Raised when decoded QNN state violates the model-state schema."""
    code = QNNErrorCode.CHECKPOINT_STATE_ERROR


class QNNDimensionError(QNNInputError):
    """Raised when an array or vector has an unexpected shape or size."""
    code = QNNErrorCode.DIMENSION_ERROR


class QNNStateError(QNNInputError):
    """Raised when a quantum state vector is invalid (non‑finite, zero norm, etc.)."""
    code = QNNErrorCode.STATE_ERROR


class QNNGradientError(QNNInputError):
    """Raised when gradient computation fails (non‑finite, overflow, etc.)."""
    code = QNNErrorCode.GRADIENT_ERROR


class QNNNormError(QNNStateError):
    """Raised when the norm of a state vector is not within tolerance of unity."""
    code = QNNErrorCode.NORM_ERROR


class QNNFiniteValueError(QNNInputError):
    """Raised when a numeric value is not finite (inf or NaN)."""
    code = QNNErrorCode.FINITE_VALUE_ERROR


class QNNConversionError(QNNInputError):
    """Raised when conversion to a required numeric type fails."""
    code = QNNErrorCode.CONVERSION_ERROR


# -----------------------------------------------------------------------------
# Utility functions for working with error codes
# -----------------------------------------------------------------------------
def error_code(exception: QNNError) -> str:
    """Return the formatted error code (e.g., 'QNN-1001') for an exception."""
    return f"QNN-{exception.code:04d}"


def error_code_for_type(error_type: Type[QNNError]) -> str:
    """Return the formatted error code for a given exception class."""
    return f"QNN-{QNNErrorCode.get_code(error_type):04d}"


__all__ = [
    "QNNError",
    "QNNConfigurationError",
    "QNNInputError",
    "QNNResourceLimitError",
    "QNNCheckpointStateError",
    "QNNDimensionError",
    "QNNStateError",
    "QNNGradientError",
    "QNNNormError",
    "QNNFiniteValueError",
    "QNNConversionError",
    "QNNErrorCode",
    "error_code",
    "error_code_for_type",
]