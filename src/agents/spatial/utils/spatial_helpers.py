from __future__ import annotations

__version__ = "2.3.0"

"""Small reusable helpers shared by the Spatial subsystem.

Numerical policy is centralized here so geometry code does not accumulate
incompatible epsilon constants. Domain algorithms remain in their owning
modules rather than in this helper layer.
"""

import numpy as np # type: ignore

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from .spatial_errors import SpatialValidationError
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Helpers")
printer = PrettyPrinter()

DEFAULT_ABS_TOL = 1.0e-9
DEFAULT_REL_TOL = 1.0e-9
DEFAULT_ANGLE_TOL = 1.0e-8


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_close(a: float, b: float, *, abs_tol: float = DEFAULT_ABS_TOL, rel_tol: float = DEFAULT_REL_TOL) -> bool:
    return bool(np.isclose(float(a), float(b), atol=abs_tol, rtol=rel_tol))


def all_close(a: Any, b: Any, *, abs_tol: float = DEFAULT_ABS_TOL, rel_tol: float = DEFAULT_REL_TOL) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=abs_tol, rtol=rel_tol))


def clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise SpatialValidationError("lower bound must not exceed upper bound", context={"lower": lower, "upper": upper})
    return float(min(max(float(value), float(lower)), float(upper)))


def finite_array(
    value: Any,
    *,
    name: str = "value",
    ndim: int | None = None,
    shape: Sequence[int] | None = None,
    copy: bool = True,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=copy)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError(
            f"{name} must be coercible to a finite float array",
            context={"name": name, "type": type(value).__name__},
            cause=exc,
        ) from exc

    if ndim is not None and array.ndim != ndim:
        raise SpatialValidationError(
            f"{name} must have ndim={ndim}",
            context={"name": name, "actual_ndim": array.ndim, "shape": array.shape},
        )
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise SpatialValidationError(
            f"{name} must have shape {tuple(shape)}",
            context={"name": name, "actual_shape": array.shape},
        )
    if not np.all(np.isfinite(array)):
        raise SpatialValidationError(f"{name} contains NaN or infinite values", context={"name": name})
    return array


def finite_vector(value: Any, *, name: str = "vector", dimension: int | None = None, allow_empty: bool = False) -> np.ndarray:
    vector = finite_array(value, name=name, ndim=1)
    if not allow_empty and vector.size == 0:
        raise SpatialValidationError(f"{name} must not be empty", context={"name": name})
    if dimension is not None and vector.size != dimension:
        raise SpatialValidationError(
            f"{name} must have dimension {dimension}",
            context={"name": name, "actual_dimension": int(vector.size)},
        )
    return vector


def stable_norm(value: Any) -> float:
    vector = finite_vector(value, allow_empty=True)
    if vector.size == 0:
        return 0.0
    return float(np.linalg.norm(vector))


def unit_vector(value: Any, *, name: str = "vector", tolerance: float = DEFAULT_ABS_TOL) -> np.ndarray:
    vector = finite_vector(value, name=name)
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise SpatialValidationError(
            f"{name} must have non-zero length",
            context={"name": name, "norm": norm, "tolerance": tolerance},
        )
    return vector / norm


def validate_positive(value: float, *, name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric) or numeric <= 0.0:
        raise SpatialValidationError(
            f"{name} must be a finite positive value",
            context={"name": name, "value": value},
        )
    return numeric


def validate_non_negative(value: float, *, name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise SpatialValidationError(
            f"{name} must be a finite non-negative value",
            context={"name": name, "value": value},
        )
    return numeric


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def to_json_safe(value: Any, *, _depth: int = 0, max_depth: int = 8) -> Any:
    if _depth >= max_depth:
        return repr(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else repr(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_json_safe(getattr(value, item.name), _depth=_depth + 1, max_depth=max_depth)
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): to_json_safe(item, _depth=_depth + 1, max_depth=max_depth)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_safe(item, _depth=_depth + 1, max_depth=max_depth) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_json_safe(to_dict(), _depth=_depth + 1, max_depth=max_depth)
    if hasattr(value, "__dict__"):
        return to_json_safe(vars(value), _depth=_depth + 1, max_depth=max_depth)
    return repr(value)


def validate_identifier(value: str, *, name: str = "identifier") -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpatialValidationError(f"{name} must be a non-empty string", context={"name": name})
    return value.strip()


__all__ = [
    "DEFAULT_ABS_TOL",
    "DEFAULT_REL_TOL",
    "DEFAULT_ANGLE_TOL",
    "utc_now_iso",
    "is_close",
    "all_close",
    "clamp",
    "finite_array",
    "finite_vector",
    "stable_norm",
    "unit_vector",
    "validate_positive",
    "validate_non_negative",
    "canonical_pair",
    "to_json_safe",
    "validate_identifier",
]
