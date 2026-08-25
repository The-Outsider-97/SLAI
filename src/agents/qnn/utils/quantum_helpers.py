"""Side-effect-free helpers shared by QNN numerical modules.

This module provides foundational validation, conversion, and serialisation
utilities. Functions raise appropriate QNNError subclasses (from quantum_errors)
when invariants are violated, ensuring a consistent error‑handling strategy
across all QNN components.
"""

from __future__ import annotations

import math
import numpy as np
import sys

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, TypeVar, Union

from .quantum_errors import *


# -----------------------------------------------------------------------------
# Basic numeric validators (already present, kept for compatibility)
# -----------------------------------------------------------------------------
def positive_int(value: Any, name: str) -> int:
    """Return a positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise QNNConfigurationError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise QNNConfigurationError(f"{name} must be positive")
    return result


def positive_float(value: Any, name: str) -> float:
    """Return a finite positive float without accepting booleans."""
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.number),
    ):
        raise QNNConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise QNNConfigurationError(f"{name} must be finite and positive")
    return result


# -----------------------------------------------------------------------------
# Extended validators and converters
# -----------------------------------------------------------------------------
def is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two greater than zero."""
    return n > 0 and (n & (n - 1)) == 0


def as_complex128(value: Any, *, name: str = "array") -> np.ndarray:
    """Convert value to a complex128 NumPy array, raising on failure.

    Returns a C‑contiguous array of complex128. Raises QNNConversionError
    if conversion fails.
    """
    try:
        arr = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise QNNConversionError(f"{name} cannot be converted to complex128") from exc
    return np.ascontiguousarray(arr, dtype=np.complex128)


def as_float64(value: Any, *, name: str = "array") -> np.ndarray:
    """Convert value to a float64 NumPy array, raising on failure.

    Returns a C‑contiguous array of float64. Raises QNNConversionError
    if conversion fails.
    """
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QNNConversionError(f"{name} cannot be converted to float64") from exc
    return np.ascontiguousarray(arr, dtype=np.float64)


def validate_finite(value: Any, *, name: str = "array") -> np.ndarray:
    """Ensure all elements are finite (no inf or NaN). Returns the array.

    Raises QNNFiniteValueError if any non‑finite value is encountered.
    """
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise QNNFiniteValueError(f"{name} contains non-finite values")
    return arr


def validate_nonzero(value: Any, *, name: str = "array") -> np.ndarray:
    """Ensure the array has at least one non‑zero element.

    Raises QNNStateError if the array is all zero or has zero norm.
    """
    arr = np.asarray(value)
    if np.all(arr == 0):
        raise QNNStateError(f"{name} must have at least one non-zero element")
    return arr


def validate_shape(value: Any, expected_shape: tuple[int, ...], *, name: str = "array") -> np.ndarray:
    """Ensure the array has the exact expected shape.

    Raises QNNDimensionError on mismatch.
    """
    arr = np.asarray(value)
    if arr.shape != expected_shape:
        raise QNNDimensionError(
            f"{name} expected shape {expected_shape}, got {arr.shape}"
        )
    return arr


def validate_ndim(value: Any, ndim: int, *, name: str = "array") -> np.ndarray:
    """Ensure the array has exactly ndim dimensions.

    Raises QNNDimensionError on mismatch.
    """
    arr = np.asarray(value)
    if arr.ndim != ndim:
        raise QNNDimensionError(
            f"{name} expected {ndim} dimensions, got {arr.ndim}"
        )
    return arr


def stable_vector_norm(value: Any) -> float:
    """Compute a stable Euclidean norm for a vector, handling overflow/underflow.

    Returns 0.0 for zero or invalid inputs; otherwise a finite positive float.
    """
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    scale = float(np.max(np.abs(arr)))
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    scaled_norm = float(np.linalg.norm(arr / scale))
    if not math.isfinite(scaled_norm):
        return 0.0
    return float(scale * scaled_norm)


def is_normalized(value: Any, tolerance: float = 1e-12) -> bool:
    """Return True if the Euclidean norm of the array is within tolerance of 1.

    Handles zero‑norm arrays gracefully.
    """
    norm = stable_vector_norm(value)
    return math.isfinite(norm) and math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=tolerance)


def validate_normalized(value: Any, tolerance: float = 1e-12, *, name: str = "state") -> np.ndarray:
    """Ensure the array is normalized within the given tolerance.

    Raises QNNNormError if the norm is not within tolerance of 1.
    """
    arr = np.asarray(value)
    norm = stable_vector_norm(arr)
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise QNNNormError(
            f"{name} must be normalized within tolerance {tolerance}; "
            f"observed norm={norm:.16g}"
        )
    return arr


# -----------------------------------------------------------------------------
# JSON serialisation helpers
# -----------------------------------------------------------------------------
def json_safe(
    value: Any,
    *,
    object_hook: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Recursively convert supported NumPy values to JSON‑native containers.

    This extends the original function to support a callable hook that can
    process custom objects. The hook receives each object that is not directly
    serialisable and should return a JSON‑safe representation or raise TypeError.

    The default behaviour:
      - Mapping → dict with string keys
      - list/tuple → list
      - numpy.ndarray → list
      - numpy.generic → Python scalar
      - built‑in types (str, int, float, bool, None) → unchanged
      - other objects → passed to object_hook if provided; otherwise TypeError.

    Args:
        value: The object to convert.
        object_hook: Optional callable to handle non‑serialisable objects.

    Returns:
        A JSON‑safe representation of the input.
    """
    if isinstance(value, Mapping):
        return {str(key): json_safe(item, object_hook=object_hook) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item, object_hook=object_hook) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # If we have a hook, try to process it.
    if object_hook is not None:
        try:
            return object_hook(value)
        except TypeError:
            pass  # fall through to error

    raise TypeError(
        f"value of type {type(value).__name__} is not JSON-compatible and no hook handled it"
    )


# -----------------------------------------------------------------------------
# Additional convenience wrappers
# -----------------------------------------------------------------------------
def ensure_complex_state(
    value: Any,
    expected_dim: Optional[int] = None,
    *,
    name: str = "state",
    normalize: bool = False,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Convert and optionally normalize a state vector.

    This is a higher‑level helper that combines several validations:
      1. Convert to complex128.
      2. Ensure it is 1‑dimensional.
      3. Optionally check shape.
      4. Ensure finite and non‑zero.
      5. Optionally normalize to unit norm.

    Args:
        value: The input state.
        expected_dim: If provided, enforce exact size.
        name: Name used in error messages.
        normalize: If True, normalise the state (after validation).
        tolerance: Tolerance for norm checking.

    Returns:
        A C‑contiguous complex128 array representing the validated state.

    Raises:
        QNNConversionError, QNNDimensionError, QNNStateError, QNNNormError.
    """
    arr = as_complex128(value, name=name)
    if arr.ndim != 1:
        raise QNNDimensionError(f"{name} must be a 1‑dimensional vector, got {arr.ndim}D")
    if expected_dim is not None and arr.size != expected_dim:
        raise QNNDimensionError(
            f"{name} expected size {expected_dim}, got {arr.size}"
        )
    validate_finite(arr, name=name)
    validate_nonzero(arr, name=name)

    if normalize:
        # Normalise using the stable norm.
        norm = stable_vector_norm(arr)
        if not math.isfinite(norm) or norm <= 0.0:
            raise QNNStateError(f"{name} has invalid norm for normalisation")
        arr = arr / norm
        # Check that normalisation succeeded.
        if not math.isclose(stable_vector_norm(arr), 1.0, rel_tol=0.0, abs_tol=tolerance):
            raise QNNNormError(f"{name} normalisation failed within tolerance {tolerance}")
    else:
        # Only check that norm is close to 1.
        validate_normalized(arr, tolerance=tolerance, name=name)

    return np.ascontiguousarray(arr, dtype=np.complex128)


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------
__all__ = [
    "positive_int",
    "positive_float",
    "is_power_of_two",
    "as_complex128",
    "as_float64",
    "validate_finite",
    "validate_nonzero",
    "validate_shape",
    "validate_ndim",
    "stable_vector_norm",
    "is_normalized",
    "validate_normalized",
    "json_safe",
    "ensure_complex_state",
]
