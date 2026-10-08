"""
Reusable helper functions for STEM workflows.

This module owns generic validation, normalization, conversion, comparison,
uncertainty-combination, and JSON-safe serialization helpers. STEM domain types
should call these helpers rather than duplicating their behaviour.

sources:
- Wilson et al. (2014) for scientific-software design discipline.
- Higham (2002) for numerical helper semantics.
"""

from __future__ import annotations

__version__ = "2.3.0"

import json
import math
import numbers

from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .stem_errors import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Helpers")
printer = PrettyPrinter()



_DEFAULT_MAX_STRING_LENGTH = 500
_DEFAULT_MAX_SERIALIZATION_DEPTH = 5
_DEFAULT_MAX_COLLECTION_ITEMS = 25


def _safe_repr(value: Any, max_length: int = _DEFAULT_MAX_STRING_LENGTH) -> str:
    """Return a bounded repr that never raises."""
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _truncate_text(value: str, max_length: int = _DEFAULT_MAX_STRING_LENGTH) -> str:
    return value if len(value) <= max_length else value[: max_length - 3] + "..."


def ensure_number(
    value: Any,
    name: str,
    *,
    allow_bool: bool = False,
    error_cls: type = STEMValidationError,
) -> Union[int, float]:
    """Ensure a real numeric value, excluding bool unless explicitly allowed."""
    if isinstance(value, bool) and not allow_bool:
        raise error_cls(f"'{name}' must be a number, not bool")
    if not isinstance(value, numbers.Real):
        raise error_cls(f"'{name}' must be a real number")
    return int(value) if isinstance(value, int) else float(value)


def ensure_finite_number(
    value: Any,
    name: str,
    *,
    allow_nan: bool = False,
    allow_inf: bool = False,
    error_cls: type = STEMValidationError,
) -> float:
    """Ensure a finite real number, with optional NaN/Inf allowances."""
    numeric = ensure_number(value, name, error_cls=error_cls)
    numeric_f = float(numeric)
    if math.isnan(numeric_f) and not allow_nan:
        raise error_cls(f"'{name}' must not be NaN")
    if math.isinf(numeric_f) and not allow_inf:
        raise error_cls(f"'{name}' must be finite")
    return numeric_f


def ensure_non_negative(
    value: Any,
    name: str,
    *,
    allow_zero: bool = True,
    error_cls: type = STEMValidationError,
) -> float:
    """Ensure a numeric value is non-negative or strictly positive."""
    numeric_f = float(ensure_number(value, name, error_cls=error_cls))
    if allow_zero:
        if numeric_f < 0.0:
            raise error_cls(f"'{name}' must be non-negative")
    elif numeric_f <= 0.0:
        raise error_cls(f"'{name}' must be positive")
    return numeric_f


def ensure_positive(
    value: Any,
    name: str,
    *,
    allow_zero: bool = False,
    error_cls: type = STEMValidationError,
) -> float:
    """Ensure a numeric value is positive."""
    return ensure_non_negative(value, name, allow_zero=allow_zero, error_cls=error_cls)


def ensure_non_empty_string(
    value: Any,
    name: str,
    *,
    error_cls: type = STEMValidationError,
) -> str:
    """Ensure a stripped, non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise error_cls(
            f"'{name}' must be a non-empty string",
            context={"field": name, "received_type": type(value).__name__},
        )
    return value.strip()


def ensure_mapping(
    value: Any,
    name: str,
    *,
    allow_none: bool = False,
    error_cls: type = STEMValidationError,
) -> Mapping[Any, Any]:
    """Ensure a mapping type."""
    if value is None and allow_none:
        return {}
    if not isinstance(value, ABCMapping):
        raise error_cls(
            f"'{name}' must be a mapping",
            context={"field": name, "received_type": type(value).__name__},
        )
    return value


def ensure_sequence(
    value: Any,
    name: str,
    *,
    allow_str: bool = False,
    error_cls: type = STEMValidationError,
) -> Sequence[Any]:
    """Ensure a non-string sequence unless strings are explicitly allowed."""
    is_sequence = isinstance(value, ABCSequence)
    is_string = isinstance(value, (str, bytes, bytearray))
    if not is_sequence or (is_string and not allow_str):
        raise error_cls(
            f"'{name}' must be a non-string sequence",
            context={"field": name, "received_type": type(value).__name__},
        )
    return value


# ---------------------------------------------------------------------------
# Dimension algebra helpers
# ---------------------------------------------------------------------------


def normalize_exponents(
    exponents: Optional[Mapping[str, Any]],
    *,
    error_cls: type = STEMDimensionError,
) -> Dict[str, float]:
    """Normalize dimension exponents by validating and dropping zero exponents."""
    if exponents is None:
        return {}
    if not isinstance(exponents, ABCMapping):
        raise error_cls(
            "dimension exponents must be a mapping",
            context={"received_type": type(exponents).__name__},
        )

    normalized: Dict[str, float] = {}
    for symbol, exponent in exponents.items():
        if not isinstance(symbol, str) or not symbol.strip():
            raise error_cls(
                "dimension symbols must be non-empty strings",
                context={"symbol": _safe_repr(symbol)},
            )
        exponent_f = ensure_finite_number(
            exponent,
            f"dimension exponent for {symbol}",
            error_cls=error_cls,
        )
        if exponent_f != 0.0:
            normalized[symbol.strip()] = exponent_f
    return dict(sorted(normalized.items()))


def combine_exponents(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    sign: int = 1,
) -> Dict[str, float]:
    """Combine two dimension exponent mappings."""
    result = dict(normalize_exponents(left))
    for symbol, exponent in normalize_exponents(right).items():
        result[symbol] = result.get(symbol, 0.0) + sign * exponent
    return normalize_exponents(result)


def power_exponents(exponents: Mapping[str, Any], power: float) -> Dict[str, float]:
    """Raise a dimension exponent mapping to a scalar power."""
    power_f = ensure_finite_number(power, "power", error_cls=STEMDimensionError)
    base = normalize_exponents(exponents, error_cls=STEMDimensionError)
    return normalize_exponents(
        {symbol: exponent * power_f for symbol, exponent in base.items()},
        error_cls=STEMDimensionError,
    )


def format_dimension(exponents: Mapping[str, float]) -> str:
    """Format dimension exponents as a human-readable string."""
    if not exponents:
        return ""
    parts = []
    for symbol, exponent in sorted(exponents.items()):
        if exponent == 1.0:
            parts.append(symbol)
        elif exponent == -1.0:
            parts.append(f"{symbol}^-1")
        else:
            parts.append(f"{symbol}^{exponent:g}")
    return " ".join(parts)


def dimension_from_mapping(
    exponents: Mapping[str, Any],
    *,
    error_cls: type = STEMDimensionError,
) -> Dict[str, float]:
    """Return a normalized exponent mapping, dropping zero exponents."""
    return normalize_exponents(exponents, error_cls=error_cls)


def dimensions_compatible(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Compare two dimension exponent mappings for equality."""
    return normalize_exponents(a) == normalize_exponents(b)


def dimensionless_ratio(
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
    *,
    error_cls: type = STEMDimensionError,
) -> float:
    """Verify that a ratio of dimensions is dimensionless and return 1.0."""
    na = normalize_exponents(numerator)
    nb = normalize_exponents(denominator)
    if na != nb:
        raise error_cls(
            "ratio is not dimensionless",
            context={"numerator": na, "denominator": nb},
        )
    return 1.0


def combine_dimension_algebra(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    sign: int = 1,
) -> Dict[str, float]:
    """Combine two dimension exponent mappings with the given sign."""
    return combine_exponents(left, right, sign=sign)


def require_dimensions_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    context: str = "operation",
    *,
    error_cls: type = STEMDimensionMismatchError,
) -> None:
    """Raise when two dimension mappings are not equal."""
    if not dimensions_compatible(left, right):
        raise error_cls(
            f"dimension mismatch in {context}",
            context={"left": normalize_exponents(left), "right": normalize_exponents(right)},
        )


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------


def affine_to_base(value: float, scale: float, offset: float) -> float:
    """Convert a value in an affine unit to its base-unit value."""
    value_f = ensure_finite_number(value, "value", allow_nan=True, allow_inf=True, error_cls=STEMUnitError)
    scale_f = ensure_positive(scale, "scale", allow_zero=False, error_cls=STEMUnitError)
    offset_f = ensure_finite_number(offset, "offset", error_cls=STEMUnitError)
    return value_f * scale_f + offset_f


def affine_from_base(base: float, scale: float, offset: float) -> float:
    """Convert a base-unit value to an affine unit value."""
    base_f = ensure_finite_number(base, "base", allow_nan=True, allow_inf=True, error_cls=STEMUnitError)
    scale_f = ensure_positive(scale, "scale", allow_zero=False, error_cls=STEMUnitError)
    offset_f = ensure_finite_number(offset, "offset", error_cls=STEMUnitError)
    return (base_f - offset_f) / scale_f


def parse_si_prefix(
    symbol: str,
    prefixes: Mapping[str, Any],
) -> Tuple[Optional[str], str]:
    """Parse an SI prefix from a symbol; return ``(prefix, remainder)``."""
    if not isinstance(symbol, str) or not symbol:
        return None, symbol or ""
    for prefix in sorted(prefixes.keys(), key=len, reverse=True):
        if prefix and symbol.startswith(prefix) and len(symbol) > len(prefix):
            return prefix, symbol[len(prefix):]
    return None, symbol


def format_si_prefix(
    factor: float,
    prefixes: Optional[Mapping[int, str]] = None,
    *,
    error_cls: type = STEMPrefixError,
) -> str:
    """Return the SI prefix symbol closest to the given decimal scale factor."""
    factor_f = ensure_finite_number(factor, "factor", error_cls=error_cls)
    if factor_f <= 0.0:
        raise error_cls("factor must be strictly positive", context={"factor": factor_f})
    table = prefixes or {
        24: "Y", 21: "Z", 18: "E", 15: "P", 12: "T", 9: "G", 6: "M", 3: "k",
        2: "h", 1: "da", 0: "", -1: "d", -2: "c", -3: "m", -6: "µ",
        -9: "n", -12: "p", -15: "f", -18: "a", -21: "z", -24: "y",
        27: "R", 30: "Q", -27: "r", -30: "q",
    }
    exponent = round(math.log10(factor_f))
    return table.get(exponent, "")


def linear_unit_only(unit: Any, *, error_cls: type = STEMUnitError) -> Any:
    """Reject affine units for algebra that requires a linear scale."""
    if getattr(unit, "offset", 0.0) != 0.0:
        raise error_cls(
            "affine units are not permitted in linear algebra",
            context={"symbol": getattr(unit, "symbol", "?")},
        )
    return unit


def affine_safe_convert(
    value: float,
    source: Any,
    target: Any,
    *,
    error_cls: type = STEMUnitError,
) -> float:
    """Convert between two units, supporting affine endpoints."""
    if getattr(source, "dimension", None) != getattr(target, "dimension", None):
        raise error_cls(
            "incompatible dimensions for conversion",
            context={
                "source": getattr(source, "symbol", "?"),
                "target": getattr(target, "symbol", "?"),
            },
        )
    base = affine_to_base(value, getattr(source, "scale", 1.0), getattr(source, "offset", 0.0))
    return affine_from_base(base, getattr(target, "scale", 1.0), getattr(target, "offset", 0.0))


def coherence_check(unit: Any) -> bool:
    """Return True when a unit is coherent (scale == 1.0, offset == 0.0)."""
    return getattr(unit, "scale", 1.0) == 1.0 and getattr(unit, "offset", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------


def combine_standard_uncertainties(
    values: Sequence[float],
    correlations: Optional[Mapping[Tuple[int, int], float]] = None,
    dof: Optional[Sequence[Optional[float]]] = None,
) -> Tuple[float, Optional[float]]:
    """Combine standard uncertainties using RSS with optional correlations."""
    materialized = [
        ensure_non_negative(value, "uncertainty", allow_zero=True, error_cls=STEMUncertaintyError)
        for value in values
    ]
    n = len(materialized)
    if n == 0:
        return 0.0, None

    covariance = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        covariance[i][i] = materialized[i] ** 2

    if correlations:
        for (i, j), rho in correlations.items():
            if not (0 <= i < n and 0 <= j < n):
                raise STEMUncertaintyError(
                    "correlation index out of range",
                    context={"i": i, "j": j, "component_count": n},
                )
            rho_f = ensure_finite_number(rho, "correlation", error_cls=STEMUncertaintyError)
            if rho_f < -1.0 or rho_f > 1.0:
                raise STEMUncertaintyError(
                    "correlation must be between -1 and 1",
                    context={"i": i, "j": j, "correlation": rho_f},
                )
            covariance[i][j] = rho_f * materialized[i] * materialized[j]
            covariance[j][i] = covariance[i][j]

    variance = sum(covariance[i][j] for i in range(n) for j in range(n))
    combined = math.sqrt(max(0.0, variance))

    effective_dof: Optional[float] = None
    if dof is not None and all(item is not None for item in dof):
        numerator = combined ** 4
        denominator = 0.0
        for uncertainty_value, dof_value in zip(materialized, dof):
            dof_f = ensure_positive(
                dof_value, "degrees_freedom", allow_zero=False, error_cls=STEMUncertaintyError,
            )
            denominator += (uncertainty_value ** 4) / dof_f
        if denominator > 0.0:
            effective_dof = numerator / denominator
    return combined, effective_dof


def welch_satterthwaite(
    uncertainties: Sequence[float],
    dofs: Sequence[float],
    *,
    error_cls: type = STEMUncertaintyError,
) -> float:
    """Welch-Satterthwaite effective degrees of freedom."""
    u = [ensure_non_negative(v, "uncertainty", allow_zero=True, error_cls=error_cls) for v in uncertainties]
    v = [ensure_positive(v, "degrees_freedom", allow_zero=False, error_cls=error_cls) for v in dofs]
    if len(u) != len(v):
        raise error_cls("uncertainties and dofs must have the same length")
    numerator = sum(value ** 2 for value in u) ** 2
    denominator = sum((ui ** 4) / vi for ui, vi in zip(u, v))
    return numerator / denominator if denominator > 0.0 else float("inf")


# ---------------------------------------------------------------------------
# Comparison and serialization
# ---------------------------------------------------------------------------


def isclose(
    a: float,
    b: float,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
    nan_policy: str = "equal",
) -> bool:
    """IEEE-754-aware approximate comparison with explicit NaN policy."""
    a_f = float(a)
    b_f = float(b)
    if math.isnan(a_f) or math.isnan(b_f):
        if nan_policy == "always":
            return True
        if nan_policy == "never":
            return False
        return math.isnan(a_f) and math.isnan(b_f)
    if math.isinf(a_f) or math.isinf(b_f):
        return a_f == b_f
    return math.isclose(a_f, b_f, rel_tol=rel_tol, abs_tol=abs_tol)


def json_safe(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = _DEFAULT_MAX_SERIALIZATION_DEPTH,
    max_items: int = _DEFAULT_MAX_COLLECTION_ITEMS,
) -> Any:
    """Convert arbitrary Python objects into JSON-safe primitives defensively."""
    if depth >= max_depth:
        return _safe_repr(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        try:
            return _truncate_text(value.decode("utf-8", errors="replace"))
        except Exception:
            return _safe_repr(value)
    if isinstance(value, BaseException):
        return {"type": value.__class__.__name__, "message": _truncate_text(str(value))}
    if isinstance(value, ABCMapping):
        result: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                result["__truncated__"] = True
                result["__remaining_items__"] = (
                    max(0, len(value) - max_items) if hasattr(value, "__len__") else True
                )
                break
            result[str(key)] = json_safe(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        payload = [
            json_safe(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            for item in sequence[:max_items]
        ]
        if len(sequence) > max_items:
            payload.append({"__truncated__": True, "__remaining_items__": len(sequence) - max_items})
        return payload
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return json_safe(value.to_dict(), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        except Exception:
            pass
    return _safe_repr(value)


def to_json(value: Any, *, indent: int = 2, sort_keys: bool = True) -> str:
    """Serialize a value to JSON, falling back to a safe representation."""
    try:
        return json.dumps(json_safe(value), indent=indent, sort_keys=sort_keys)
    except Exception:
        return json.dumps(_safe_repr(value), indent=indent, sort_keys=sort_keys)


# ---------------------------------------------------------------------------
# Numeric convenience helpers
# ---------------------------------------------------------------------------


def safe_divide(
    numerator: Any,
    denominator: Any,
    *,
    default: float = 0.0,
    error_cls: type = STEMNumericalError,
) -> float:
    """Return ``numerator / denominator``, falling back to ``default`` on zero."""
    num = ensure_finite_number(numerator, "numerator", allow_nan=True, allow_inf=True, error_cls=error_cls)
    den = ensure_finite_number(denominator, "denominator", allow_nan=True, allow_inf=True, error_cls=error_cls)
    if den == 0.0 or math.isnan(den):
        return float(default)
    return num / den


def safe_log(x: Any, *, base: float = math.e, error_cls: type = STEMNumericalError) -> float:
    """Return log(x), raising a STEMError for non-positive arguments."""
    x_f = ensure_finite_number(x, "x", error_cls=error_cls)
    if x_f <= 0.0:
        raise error_cls("log requires a strictly positive argument", context={"x": x_f})
    base_f = ensure_positive(base, "base", allow_zero=False, error_cls=error_cls)
    if base_f == 1.0:
        raise error_cls("log base must not be 1", context={"base": base_f})
    return math.log(x_f) if base_f == math.e else math.log(x_f, base_f)


def forward_error(x_approx: Any, x_exact: Any, *, error_cls: type = STEMNumericalError) -> float:
    """Absolute forward error ``|x_approx - x_exact|``."""
    a = ensure_finite_number(x_approx, "x_approx", allow_nan=True, allow_inf=True, error_cls=error_cls)
    b = ensure_finite_number(x_exact, "x_exact", allow_nan=True, allow_inf=True, error_cls=error_cls)
    return abs(a - b)


def relative_error(x_approx: Any, x_exact: Any, *, error_cls: type = STEMNumericalError) -> float:
    """Relative error with safe handling of a zero exact value."""
    return safe_relative_error(x_approx, x_exact, error_cls=error_cls)


def safe_relative_error(
    x_approx: Any,
    x_exact: Any,
    *,
    error_cls: type = STEMNumericalError,
) -> float:
    """Relative error with explicit zero and non-finite handling."""
    try:
        a = float(x_approx)
        b = float(x_exact)
    except (TypeError, ValueError) as exc:
        raise error_cls(
            "safe_relative_error requires numeric inputs",
            context={"x_approx": _safe_repr(x_approx), "x_exact": _safe_repr(x_exact)},
            cause=exc,
        ) from exc

    if math.isnan(a) or math.isnan(b):
        return float("nan")
    if math.isinf(a) or math.isinf(b):
        return 0.0 if a == b else float("inf")
    if b == 0.0:
        return 0.0 if a == 0.0 else float("inf")
    return abs(a - b) / abs(b)


def is_near_zero(value: Any, *, atol: float = 1e-12, rtol: float = 0.0) -> bool:
    """Tolerance-aware zero test for scalars and finite arrays."""
    if isinstance(value, ABCMapping):
        return all(is_near_zero(v, atol=atol, rtol=rtol) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_near_zero(v, atol=atol, rtol=rtol) for v in value)
    v = float(value)
    if math.isnan(v):
        return False
    return abs(v) <= atol + rtol * abs(v)


def backward_error(
    A: Sequence[Sequence[float]],
    x_approx: Sequence[float],
    b: Sequence[float],
    *,
    error_cls: type = STEMNumericalError,
) -> float:
    """Backward error ``||A x - b|| / (||A|| * ||x|| + ||b||)`` (infinity norm)."""
    matrix = as_float_matrix(A, "A", error_cls=error_cls)
    x = as_float_array(x_approx, "x_approx", error_cls=error_cls)
    rhs = as_float_array(b, "b", error_cls=error_cls)

    if len(matrix) != len(rhs):
        raise error_cls("A and b have incompatible shapes")
    for row in matrix:
        if len(row) != len(x):
            raise error_cls("A and x have incompatible shapes")

    residual = [sum(row[j] * x[j] for j in range(len(x))) - rhs[i] for i, row in enumerate(matrix)]
    residual_norm = max(abs(value) for value in residual) if residual else 0.0

    A_norm = max(sum(abs(v) for v in row) for row in matrix) if matrix else 0.0
    x_norm = max(abs(v) for v in x) if x else 0.0
    b_norm = max(abs(v) for v in rhs) if rhs else 0.0

    denominator = A_norm * x_norm + b_norm
    return residual_norm / denominator if denominator > 0.0 else residual_norm


# ---------------------------------------------------------------------------
# Callable / bound / shape validation
# ---------------------------------------------------------------------------


def validate_callable(f: Any, name: str, *, error_cls: type = STEMValidationError) -> Callable[..., Any]:
    """Validate that a value is callable and return it unchanged."""
    if not callable(f):
        raise error_cls(
            f"'{name}' must be callable",
            context={"field": name, "received_type": type(f).__name__},
        )
    return f


def validate_bounds(
    a: Any,
    b: Any,
    name: str = "bounds",
    *,
    error_cls: type = STEMValidationError,
) -> Tuple[float, float]:
    """Validate an ordered numeric interval ``[a, b]`` with ``a <= b``."""
    a_f = ensure_finite_number(a, f"{name}.lower", error_cls=error_cls)
    b_f = ensure_finite_number(b, f"{name}.upper", error_cls=error_cls)
    if a_f > b_f:
        raise error_cls(
            f"'{name}' lower bound must not exceed upper bound",
            context={"lower": a_f, "upper": b_f},
        )
    return a_f, b_f


def validate_finite(value: Any, name: str = "value", *, error_cls: type = STEMValidationError) -> bool:
    """Validate finiteness of a scalar, sequence, or nested structure."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise error_cls(f"'{name}' must be finite", context={"value": value})
        return True
    if isinstance(value, ABCMapping):
        for k, v in value.items():
            validate_finite(v, f"{name}[{k}]", error_cls=error_cls)
        return True
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            validate_finite(v, f"{name}[{i}]", error_cls=error_cls)
        return True
    return True


def validate_shape(
    value: Any,
    expected_shape: Sequence[Optional[int]],
    name: str = "value",
    *,
    error_cls: type = STEMValidationError,
) -> bool:
    """Validate that a nested sequence has the expected shape."""
    if not isinstance(expected_shape, (tuple, list)):
        raise error_cls("expected_shape must be a tuple or list")
    seq: Any = value
    for depth, expected_len in enumerate(expected_shape):
        if not isinstance(seq, (list, tuple)):
            raise error_cls(
                f"'{name}' shape mismatch: expected sequence at depth {depth}",
                context={"depth": depth, "received_type": type(seq).__name__},
            )
        if expected_len is not None and len(seq) != int(expected_len):
            raise error_cls(
                f"'{name}' shape mismatch at depth {depth}",
                context={"expected_length": int(expected_len), "actual_length": len(seq)},
            )
        if depth + 1 < len(expected_shape) and len(seq) > 0:
            seq = seq[0]
    return True


# ---------------------------------------------------------------------------
# Array conversion
# ---------------------------------------------------------------------------


def as_float_array(xs: Any, name: str, *, error_cls: type = STEMNumericalError) -> List[float]:
    """Convert a sequence to a list of finite floats."""
    sequence = ensure_sequence(xs, name, error_cls=error_cls)
    result: List[float] = []
    for index, item in enumerate(sequence):
        result.append(
            ensure_finite_number(item, f"{name}[{index}]", allow_nan=True, allow_inf=True, error_cls=error_cls)
        )
    return result


def as_float_matrix(rows: Any, name: str, *, error_cls: type = STEMNumericalError) -> List[List[float]]:
    """Convert a nested sequence to a list of lists of finite floats."""
    sequence = ensure_sequence(rows, name, error_cls=error_cls)
    result: List[List[float]] = []
    for i, row in enumerate(sequence):
        row_seq = ensure_sequence(row, f"{name}[{i}]", error_cls=error_cls)
        result.append(
            [
                ensure_finite_number(
                    item, f"{name}[{i}][{j}]", allow_nan=True, allow_inf=True, error_cls=error_cls,
                )
                for j, item in enumerate(row_seq)
            ]
        )
    return result


def check_finite_array(xs: Any, name: str, *, error_cls: type = STEMNumericalError) -> List[float]:
    """Validate a sequence contains only finite numbers."""
    sequence = ensure_sequence(xs, name, error_cls=error_cls)
    result: List[float] = []
    for index, item in enumerate(sequence):
        result.append(ensure_finite_number(item, f"{name}[{index}]", error_cls=error_cls))
    return result


def check_square_matrix(A: Any, name: str = "A", *, error_cls: type = STEMLinearAlgebraError) -> List[List[float]]:
    """Validate that A is a square float matrix."""
    matrix = as_float_matrix(A, name, error_cls=error_cls)
    n = len(matrix)
    if n == 0:
        raise error_cls(f"'{name}' must not be empty")
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise error_cls(
                f"'{name}' must be square",
                context={"row": i, "row_length": len(row), "expected": n},
            )
    return matrix


def check_symmetric_matrix(
    A: Any,
    name: str = "A",
    tol: float = 1e-10,
    *,
    error_cls: type = STEMLinearAlgebraError,
) -> List[List[float]]:
    """Validate that A is square and symmetric within tol."""
    matrix = check_square_matrix(A, name, error_cls=error_cls)
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > tol:
                raise error_cls(
                    f"'{name}' must be symmetric",
                    context={"i": i, "j": j, "a_ij": matrix[i][j], "a_ji": matrix[j][i]},
                )
    return matrix


def check_positive_definite(A: Any, name: str = "A", *, error_cls: type = STEMLinearAlgebraError) -> List[List[float]]:
    """Validate positive-definiteness via Cholesky with no pivoting."""
    matrix = check_symmetric_matrix(A, name, error_cls=error_cls)
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = matrix[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise error_cls(
                        f"'{name}' is not positive definite",
                        context={"pivot_index": i, "pivot_value": s},
                    )
                L[i][j] = math.sqrt(s)
            else:
                if L[j][j] == 0.0:
                    raise error_cls(f"'{name}' is not positive definite")
                L[i][j] = s / L[j][j]
    return matrix


# ---------------------------------------------------------------------------
# Numeric canonicalization and diagnostics
# ---------------------------------------------------------------------------


def normalize_numeric_input(value: Any, name: str = "value", *, error_cls: type = STEMValidationError) -> Union[int, float, complex]:
    """Coerce a numeric input into a canonical int/float/complex value."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, complex)):
        return value
    if isinstance(value, str):
        try:
            lowered = value.strip().lower()
            if lowered in {"nan", "+nan", "-nan"}:
                return float("nan")
            if lowered in {"inf", "+inf", "infinity", "+infinity"}:
                return float("inf")
            if lowered in {"-inf", "-infinity"}:
                return float("-inf")
            if "j" in lowered:
                return complex(lowered)
            if "." in lowered or "e" in lowered:
                return float(lowered)
            return int(lowered)
        except ValueError as exc:
            raise error_cls(
                f"'{name}' cannot be parsed as a number",
                context={"value": value},
                cause=exc,
            ) from exc
    raise error_cls(
        f"'{name}' must be numeric",
        context={"received_type": type(value).__name__},
    )


def canonicalize_numeric_type(value: Any) -> Union[int, float, complex]:
    """Collapse a numeric value into a canonical type (int, float, complex)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, complex):
        if value.imag == 0.0:
            return float(value.real)
        return complex(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise STEMValidationError(
            "cannot canonicalize numeric type",
            context={"received_type": type(value).__name__},
            cause=exc,
        ) from exc


def build_diagnostics(**kwargs: Any) -> Dict[str, Any]:
    """Assemble a structured diagnostics mapping from keyword arguments."""
    diagnostics: Dict[str, Any] = {}
    for key, value in kwargs.items():
        diagnostics[key] = json_safe(value)
    return diagnostics


# ---------------------------------------------------------------------------
# Tolerance normalization and precision policy
# ---------------------------------------------------------------------------


def normalize_tolerance(tolerance: Any) -> Dict[str, Any]:
    """Canonicalize a tolerance specification into a normalized mapping."""
    if tolerance is None:
        return {
            "lower": None,
            "upper": None,
            "absolute": None,
            "relative": None,
            "inclusive_min": True,
            "inclusive_max": True,
        }
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
        return {
            "lower": None,
            "upper": None,
            "absolute": abs(float(tolerance)),
            "relative": None,
            "inclusive_min": True,
            "inclusive_max": True,
        }
    if isinstance(tolerance, ABCMapping):
        return {
            "lower": tolerance.get("lower"),
            "upper": tolerance.get("upper"),
            "absolute": tolerance.get("absolute"),
            "relative": tolerance.get("relative"),
            "inclusive_min": bool(tolerance.get("inclusive_min", True)),
            "inclusive_max": bool(tolerance.get("inclusive_max", True)),
        }
    raise STEMValidationError(
        "tolerance must be a number or mapping",
        context={"received_type": type(tolerance).__name__},
    )


def resolve_precision_policy(policy: Any) -> Any:
    """Return a usable PrecisionPolicy, falling back to the default."""
    from ..stem_types import PrecisionPolicy  # local import to avoid cycles

    if policy is None:
        return PrecisionPolicy.default()
    if not isinstance(policy, PrecisionPolicy):
        raise STEMValidationError(
            "policy must be a PrecisionPolicy",
            context={"received_type": type(policy).__name__},
        )
    return policy


# ---------------------------------------------------------------------------
# Statistical / numeric recurrence helpers
# ---------------------------------------------------------------------------


def stable_variance(xs: Sequence[float], ddof: int = 1, *, error_cls: type = STEMStatisticsError) -> float:
    """Welford/Chan stable sample variance."""
    values = as_float_array(xs, "xs", error_cls=error_cls)
    n = len(values)
    if n - ddof <= 0:
        raise error_cls(
            "sample size is too small for requested ddof",
            context={"n": n, "ddof": ddof},
        )
    mean = 0.0
    m2 = 0.0
    for index, value in enumerate(values):
        delta = value - mean
        mean += delta / (index + 1)
        delta2 = value - mean
        m2 += delta * delta2
    return m2 / (n - ddof)


def richardson_extrapolate(
    values: Sequence[float],
    factor: float = 2.0,
    *,
    error_cls: type = STEMNumericalError,
) -> float:
    """Richardson extrapolation over a sequence of estimates at successive grids."""
    sequence = [ensure_finite_number(v, "value", error_cls=error_cls) for v in values]
    if len(sequence) < 2:
        raise error_cls("Richardson extrapolation requires at least two values")
    factor_f = ensure_positive(factor, "factor", allow_zero=False, error_cls=error_cls)
    if factor_f == 1.0:
        raise error_cls("Richardson factor must not be 1")
    current = list(sequence)
    for level in range(1, len(sequence)):
        next_level = []
        divisor = factor_f ** level - 1.0
        for i in range(len(current) - 1):
            next_level.append(current[i + 1] + (current[i + 1] - current[i]) / divisor)
        current = next_level
    return current[0]


def condition_number_estimate(A: Sequence[Sequence[float]]) -> float:
    """Return a crude 1-norm condition-number estimate."""
    matrix = as_float_matrix(A, "A", error_cls=STEMLinearAlgebraError)
    n = len(matrix)
    if n == 0:
        return float("inf")
    for row in matrix:
        if len(row) != n:
            raise STEMLinearAlgebraError("condition_number_estimate requires a square matrix")
    norm_a = max(sum(abs(matrix[i][j]) for i in range(n)) for j in range(n))
    # Simple inverse via Gaussian elimination with partial pivoting
    aug = [
        [matrix[i][j] for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(aug[i][k]))
        if abs(aug[pivot][k]) < 1e-300:
            return float("inf")
        if pivot != k:
            aug[k], aug[pivot] = aug[pivot], aug[k]
        for i in range(n):
            if i != k:
                factor = aug[i][k] / aug[k][k]
                for j in range(k, 2 * n):
                    aug[i][j] -= factor * aug[k][j]
    inverse = [[aug[i][j + n] / aug[i][i] for j in range(n)] for i in range(n)]
    norm_inv = max(sum(abs(inverse[i][j]) for i in range(n)) for j in range(n))
    return norm_a * norm_inv


__all__ = [
    # Existing primitives
    "affine_from_base",
    "affine_to_base",
    "combine_exponents",
    "combine_standard_uncertainties",
    "ensure_finite_number",
    "ensure_mapping",
    "ensure_non_empty_string",
    "ensure_non_negative",
    "ensure_number",
    "ensure_positive",
    "ensure_sequence",
    "format_dimension",
    "isclose",
    "json_safe",
    "normalize_exponents",
    "power_exponents",
    "to_json",
    "normalize_tolerance",
    "validate_finite",
    "normalize_numeric_input",
    "safe_relative_error",
    "is_near_zero",
    "validate_shape",
    "canonicalize_numeric_type",
    "build_diagnostics",
    # Shared algebraic / conversion / numeric helpers
    "as_float_array",
    "as_float_matrix",
    "affine_safe_convert",
    "backward_error",
    "check_finite_array",
    "check_positive_definite",
    "check_square_matrix",
    "check_symmetric_matrix",
    "coherence_check",
    "combine_dimension_algebra",
    "condition_number_estimate",
    "dimension_from_mapping",
    "dimensionless_ratio",
    "dimensions_compatible",
    "format_si_prefix",
    "forward_error",
    "linear_unit_only",
    "parse_si_prefix",
    "relative_error",
    "require_dimensions_match",
    "resolve_precision_policy",
    "richardson_extrapolate",
    "safe_divide",
    "safe_log",
    "stable_variance",
    "validate_bounds",
    "validate_callable",
    "welch_satterthwaite",
]