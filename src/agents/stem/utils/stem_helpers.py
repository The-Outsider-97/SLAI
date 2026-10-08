"""Shared deterministic helper functions for the STEM subsystem.

Algorithms belong in their domain modules; this file contains reusable
validation, dimension algebra, numerical diagnostics, and serialization only.
"""
from __future__ import annotations

__version__ = "2.3.0"

import json
import math
import numbers

from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

from .stem_errors import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Helpers")
printer = PrettyPrinter()

_ZERO_EPS = 1e-15


def ensure_number(value: Any, name: str, *, allow_bool: bool = False, error_cls: type = STEMValidationError) -> Union[int, float]:
    if isinstance(value, bool) and not allow_bool:
        raise error_cls(f"{name} must be numeric, not bool", context={"value": value})
    if not isinstance(value, numbers.Real):
        raise error_cls(f"{name} must be a real number", context={"type": type(value).__name__})
    return cast(Union[int, float], value)


def ensure_finite_number(value: Any, name: str, *, allow_nan: bool = False, allow_inf: bool = False, error_cls: type = STEMValidationError) -> float:
    result = float(ensure_number(value, name, error_cls=error_cls))
    if math.isnan(result) and not allow_nan:
        raise error_cls(f"{name} must not be NaN")
    if math.isinf(result) and not allow_inf:
        raise error_cls(f"{name} must be finite")
    return result


def ensure_non_negative(value: Any, name: str, *, allow_zero: bool = True, error_cls: type = STEMValidationError) -> float:
    result = ensure_finite_number(value, name, error_cls=error_cls)
    if result < 0.0 or (not allow_zero and result == 0.0):
        raise error_cls(f"{name} must be {'positive' if not allow_zero else 'non-negative'}", context={"value": result})
    return result


def ensure_positive(value: Any, name: str, *, allow_zero: bool = False, error_cls: type = STEMValidationError) -> float:
    return ensure_non_negative(value, name, allow_zero=allow_zero, error_cls=error_cls)


def ensure_non_empty_string(value: Any, name: str, *, error_cls: type = STEMValidationError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_cls(f"{name} must be a non-empty string")
    return value.strip()


def ensure_mapping(value: Any, name: str, *, allow_none: bool = False, error_cls: type = STEMValidationError) -> Mapping[Any, Any]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, ABCMapping):
        raise error_cls(f"{name} must be a mapping", context={"type": type(value).__name__})
    return value


def ensure_sequence(value: Any, name: str, *, allow_str: bool = False, error_cls: type = STEMValidationError) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) and not allow_str:
        raise error_cls(f"{name} must be a non-string sequence")
    if not isinstance(value, ABCSequence):
        raise error_cls(f"{name} must be a sequence", context={"type": type(value).__name__})
    return value


def normalize_exponents(exponents: Optional[Mapping[str, Any]], *, error_cls: type = STEMDimensionError) -> Dict[str, float]:
    if exponents is None:
        return {}
    ensure_mapping(exponents, "exponents", error_cls=error_cls)
    out: Dict[str, float] = {}
    for key, value in exponents.items():
        symbol = ensure_non_empty_string(str(key), "dimension symbol", error_cls=error_cls)
        exponent = ensure_finite_number(value, f"exponent[{symbol}]", error_cls=error_cls)
        if abs(exponent) > _ZERO_EPS:
            out[symbol] = float(exponent)
    return dict(sorted(out.items()))


def combine_exponents(left: Mapping[str, Any], right: Mapping[str, Any], sign: int = 1) -> Dict[str, float]:
    if sign not in (-1, 1):
        raise STEMDimensionError("sign must be +1 or -1")
    result = normalize_exponents(left)
    for key, value in normalize_exponents(right).items():
        result[key] = result.get(key, 0.0) + sign * value
    return normalize_exponents(result)


def power_exponents(exponents: Mapping[str, Any], power: float) -> Dict[str, float]:
    p = ensure_finite_number(power, "power", error_cls=STEMDimensionError)
    return normalize_exponents({key: value * p for key, value in normalize_exponents(exponents).items()})


def format_dimension(exponents: Mapping[str, float]) -> str:
    normalized = normalize_exponents(exponents)
    if not normalized:
        return "1"
    return " ".join(key if exp == 1.0 else f"{key}^{exp:g}" for key, exp in normalized.items())


def dimension_from_mapping(exponents: Mapping[str, Any], *, error_cls: type = STEMDimensionError) -> Dict[str, float]:
    return normalize_exponents(exponents, error_cls=error_cls)


def dimensions_compatible(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return normalize_exponents(a) == normalize_exponents(b)


def require_dimensions_match(left: Mapping[str, Any], right: Mapping[str, Any], context: str = "operation", *, error_cls: type = STEMDimensionMismatchError) -> None:
    if not dimensions_compatible(left, right):
        raise error_cls("Dimension mismatch", context={"operation": context, "left": dict(left), "right": dict(right)})


def dimensionless_ratio(numerator: Mapping[str, Any], denominator: Mapping[str, Any], *, error_cls: type = STEMDimensionError) -> float:
    if not dimensions_compatible(numerator, denominator):
        raise error_cls("Ratio is not dimensionless")
    return 1.0


def combine_dimension_algebra(left: Mapping[str, Any], right: Mapping[str, Any], sign: int = 1) -> Dict[str, float]:
    return combine_exponents(left, right, sign)


def affine_to_base(value: float, scale: float, offset: float) -> float:
    return ensure_finite_number(value, "value", error_cls=STEMUnitError) * ensure_positive(scale, "scale", error_cls=STEMUnitError) + ensure_finite_number(offset, "offset", error_cls=STEMUnitError)


def affine_from_base(base: float, scale: float, offset: float) -> float:
    return (ensure_finite_number(base, "base", error_cls=STEMUnitError) - ensure_finite_number(offset, "offset", error_cls=STEMUnitError)) / ensure_positive(scale, "scale", error_cls=STEMUnitError)


def parse_si_prefix(symbol: str, prefixes: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    text = ensure_non_empty_string(symbol, "symbol", error_cls=STEMPrefixError)
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text.startswith(prefix) and len(text) > len(prefix):
            return prefix, text[len(prefix):]
    return None, text


def format_si_prefix(factor: float, prefixes: Optional[Mapping[int, str]] = None, *, error_cls: type = STEMPrefixError) -> str:
    value = ensure_positive(factor, "factor", error_cls=error_cls)
    if value == 1.0:
        return ""
    exp = int(round(math.log10(value)))
    mapping = dict(prefixes or {30:"Q",27:"R",24:"Y",21:"Z",18:"E",15:"P",12:"T",9:"G",6:"M",3:"k",-3:"m",-6:"µ",-9:"n",-12:"p",-15:"f",-18:"a",-21:"z",-24:"y",-27:"r",-30:"q"})
    if not math.isclose(value, 10.0 ** exp, rel_tol=1e-12) or exp not in mapping:
        raise error_cls("factor has no canonical SI prefix", context={"factor": value})
    return mapping[exp]


def linear_unit_only(unit: Any, *, error_cls: type = STEMUnitError) -> Any:
    if bool(getattr(unit, "is_affine", False)):
        raise error_cls("Affine units cannot participate in multiplicative unit algebra")
    return unit


def affine_safe_convert(value: float, source: Any, target: Any, *, error_cls: type = STEMUnitError) -> float:
    if getattr(source, "dimension", None) != getattr(target, "dimension", None):
        raise error_cls("Units are dimensionally incompatible")
    return target.from_base(source.to_base(value))


def coherence_check(unit: Any) -> bool:
    return math.isclose(float(getattr(unit, "scale", math.nan)), 1.0, rel_tol=0.0, abs_tol=0.0) and float(getattr(unit, "offset", math.nan)) == 0.0


def combine_standard_uncertainties(values: Sequence[float], correlations: Optional[Mapping[Tuple[int, int], float]] = None, dof: Optional[Sequence[Optional[float]]] = None) -> Tuple[float, Optional[float]]:
    vals = [ensure_non_negative(v, f"u[{i}]", error_cls=STEMUncertaintyError) for i, v in enumerate(values)]
    variance = math.fsum(v * v for v in vals)
    for (i, j), rho in (correlations or {}).items():
        if not (0 <= i < len(vals) and 0 <= j < len(vals) and i != j):
            raise STEMUncertaintyError("Invalid correlation index", context={"pair": (i, j)})
        r = ensure_finite_number(rho, "correlation", error_cls=STEMUncertaintyError)
        if abs(r) > 1.0:
            raise STEMUncertaintyError("Correlation must lie in [-1, 1]")
        variance += 2.0 * r * vals[i] * vals[j]
    if variance < -1e-15:
        raise STEMUncertaintyError("Correlation model produced negative variance", context={"variance": variance})
    combined = math.sqrt(max(0.0, variance))
    eff: Optional[float] = None
    if dof is not None and len(dof) == len(vals) and combined > 0.0:
        denom = 0.0
        for u, nu in zip(vals, dof):
            if nu is not None and math.isfinite(float(nu)) and float(nu) > 0.0:
                denom += (u ** 4) / float(nu)
        if denom > 0.0:
            eff = combined ** 4 / denom
    return combined, eff


def welch_satterthwaite(uncertainties: Sequence[float], dofs: Sequence[float], *, error_cls: type = STEMUncertaintyError) -> float:
    if len(uncertainties) != len(dofs) or not uncertainties:
        raise error_cls("uncertainties and dofs must have equal non-zero length")
    combined, eff = combine_standard_uncertainties(uncertainties, dof=list(dofs))
    if combined == 0.0:
        return math.inf
    if eff is None:
        raise error_cls("Effective degrees of freedom are undefined")
    return eff


def isclose(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12, nan_policy: str = "never") -> bool:
    af, bf = float(a), float(b)
    if math.isnan(af) or math.isnan(bf):
        if nan_policy == "always":
            return True
        if nan_policy == "equal":
            return math.isnan(af) and math.isnan(bf)
        return False
    return math.isclose(af, bf, rel_tol=rel_tol, abs_tol=abs_tol)


def json_safe(value: Any, *, depth: int = 0, max_depth: int = 8, max_items: int = 100) -> Any:
    if depth >= max_depth:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, ABCMapping):
        return {str(k): json_safe(v, depth=depth+1, max_depth=max_depth, max_items=max_items) for k, v in list(value.items())[:max_items]}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v, depth=depth+1, max_depth=max_depth, max_items=max_items) for v in list(value)[:max_items]]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict(), depth=depth+1, max_depth=max_depth, max_items=max_items)
    return repr(value)


def to_json(value: Any, *, indent: int = 2, sort_keys: bool = True) -> str:
    return json.dumps(json_safe(value), indent=indent, sort_keys=sort_keys, ensure_ascii=False)


def safe_divide(numerator: Any, denominator: Any, *, default: Optional[float] = None, error_cls: type = STEMNumericalError) -> float:
    n = ensure_finite_number(numerator, "numerator", error_cls=error_cls)
    d = ensure_finite_number(denominator, "denominator", error_cls=error_cls)
    if d == 0.0:
        if default is not None:
            return float(default)
        raise error_cls("Division by zero")
    result = n / d
    if not math.isfinite(result):
        raise error_cls("Division produced a non-finite result")
    return result


def safe_log(x: Any, *, base: float = math.e, error_cls: type = STEMNumericalError) -> float:
    xv = ensure_finite_number(x, "x", error_cls=error_cls)
    bv = ensure_positive(base, "base", error_cls=error_cls)
    if xv <= 0.0 or bv == 1.0:
        raise error_cls("Logarithm domain error", context={"x": xv, "base": bv})
    return math.log(xv, bv)


def forward_error(x_approx: Any, x_exact: Any, *, error_cls: type = STEMNumericalError) -> float:
    return abs(ensure_finite_number(x_approx, "x_approx", error_cls=error_cls) - ensure_finite_number(x_exact, "x_exact", error_cls=error_cls))


def relative_error(x_approx: Any, x_exact: Any, *, error_cls: type = STEMNumericalError) -> float:
    exact = ensure_finite_number(x_exact, "x_exact", error_cls=error_cls)
    if exact == 0.0:
        raise error_cls("Relative error is undefined for exact value zero")
    return forward_error(x_approx, exact, error_cls=error_cls) / abs(exact)


def safe_relative_error(x_approx: Any, x_exact: Any, *, error_cls: type = STEMNumericalError) -> float:
    exact = ensure_finite_number(x_exact, "x_exact", error_cls=error_cls)
    return forward_error(x_approx, exact, error_cls=error_cls) if exact == 0.0 else relative_error(x_approx, exact, error_cls=error_cls)


def is_near_zero(value: Any, *, atol: float = 1e-12, rtol: float = 0.0) -> bool:
    return math.isclose(float(value), 0.0, abs_tol=atol, rel_tol=rtol)


def validate_callable(f: Any, name: str, *, error_cls: type = STEMValidationError) -> Callable[..., Any]:
    if not callable(f):
        raise error_cls(f"{name} must be callable")
    return f


def validate_bounds(a: Any, b: Any, name: str = "bounds", *, error_cls: type = STEMValidationError) -> Tuple[float, float]:
    af, bf = ensure_finite_number(a, f"{name}.a", error_cls=error_cls), ensure_finite_number(b, f"{name}.b", error_cls=error_cls)
    if af == bf:
        raise error_cls(f"{name} endpoints must differ")
    return af, bf


def validate_finite(value: Any, name: str = "value", *, error_cls: type = STEMValidationError) -> bool:
    ensure_finite_number(value, name, error_cls=error_cls)
    return True


def ensure_sequence_finite(values: Sequence[Any], name: str, *, error_cls: type = STEMNumericalError) -> List[float]:
    ensure_sequence(values, name, error_cls=error_cls)
    return [ensure_finite_number(v, f"{name}[{i}]", error_cls=error_cls) for i, v in enumerate(values)]


def as_float_array(xs: Any, name: str, *, error_cls: type = STEMNumericalError) -> List[float]:
    return ensure_sequence_finite(xs, name, error_cls=error_cls)


def as_float_matrix(rows: Any, name: str, *, error_cls: type = STEMNumericalError) -> List[List[float]]:
    ensure_sequence(rows, name, error_cls=error_cls)
    matrix = [as_float_array(row, f"{name}[{i}]", error_cls=error_cls) for i, row in enumerate(rows)]
    if not matrix or not matrix[0]:
        raise error_cls(f"{name} must be non-empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise error_cls(f"{name} must be rectangular")
    return matrix


def check_finite_array(xs: Any, name: str, *, error_cls: type = STEMNumericalError) -> List[float]:
    return as_float_array(xs, name, error_cls=error_cls)


def check_square_matrix(A: Any, name: str = "A", *, error_cls: type = STEMLinearAlgebraError) -> List[List[float]]:
    matrix = as_float_matrix(A, name, error_cls=error_cls)
    if len(matrix) != len(matrix[0]):
        raise error_cls(f"{name} must be square", context={"shape": (len(matrix), len(matrix[0]))})
    return matrix


def check_symmetric_matrix(A: Any, name: str = "A", tol: float = 1e-10, *, error_cls: type = STEMLinearAlgebraError) -> List[List[float]]:
    matrix = check_square_matrix(A, name, error_cls=error_cls)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if not math.isclose(matrix[i][j], matrix[j][i], abs_tol=tol, rel_tol=tol):
                raise error_cls(f"{name} must be symmetric", context={"i": i, "j": j})
    return matrix


def check_positive_definite(A: Any, name: str = "A", *, error_cls: type = STEMLinearAlgebraError) -> List[List[float]]:
    matrix = check_symmetric_matrix(A, name, error_cls=error_cls)
    try:
        import numpy as np # type: ignore
        np.linalg.cholesky(np.asarray(matrix, dtype=float))
    except Exception as exc:
        raise error_cls(f"{name} must be positive definite", cause=exc) from exc
    return matrix


def validate_shape(value: Any, expected_shape: Sequence[Optional[int]], name: str = "value", *, error_cls: type = STEMValidationError) -> bool:
    current = value
    for depth, expected in enumerate(expected_shape):
        if not isinstance(current, ABCSequence) or isinstance(current, (str, bytes)):
            raise error_cls(f"{name} has insufficient dimensions", context={"depth": depth})
        if expected is not None and len(current) != expected:
            raise error_cls(f"{name} shape mismatch", context={"depth": depth, "expected": expected, "actual": len(current)})
        current = current[0] if current else []
    return True


def normalize_numeric_input(value: Any, name: str = "value", *, error_cls: type = STEMValidationError) -> Union[int, float, complex]:
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        raise error_cls(f"{name} must be numeric")
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise error_cls(f"{name} must be finite")
        return value
    return ensure_finite_number(value, name, error_cls=error_cls)


def canonicalize_numeric_type(value: Any) -> Union[int, float, complex]:
    normalized = normalize_numeric_input(value)
    if isinstance(normalized, float) and normalized.is_integer():
        return int(normalized)
    return normalized


def build_diagnostics(**kwargs: Any) -> Dict[str, Any]:
    return {key: json_safe(value) for key, value in kwargs.items() if value is not None}


def normalize_tolerance(tolerance: Any) -> Dict[str, float]:
    if tolerance is None:
        return {"absolute": 1e-12, "relative": 1e-9}
    if isinstance(tolerance, numbers.Real):
        value = ensure_non_negative(tolerance, "tolerance")
        return {"absolute": value, "relative": value}
    mapping = ensure_mapping(tolerance, "tolerance")
    return {
        "absolute": ensure_non_negative(mapping.get("absolute", 1e-12), "tolerance.absolute"),
        "relative": ensure_non_negative(mapping.get("relative", 1e-9), "tolerance.relative"),
    }


def resolve_precision_policy(policy: Any) -> Any:
    return policy


def stable_variance(xs: Sequence[float], ddof: int = 1, *, error_cls: type = STEMStatisticsError) -> float:
    values = as_float_array(xs, "xs", error_cls=error_cls)
    if ddof < 0 or len(values) <= ddof:
        raise error_cls("Invalid degrees-of-freedom correction", context={"n": len(values), "ddof": ddof})
    mean = math.fsum(values) / len(values)
    correction = math.fsum(v - mean for v in values)
    ss = math.fsum((v - mean) ** 2 for v in values) - correction ** 2 / len(values)
    return max(0.0, ss / (len(values) - ddof))


def richardson_extrapolate(values: Sequence[float], factor: float = 2.0, *, error_cls: type = STEMNumericalError) -> float:
    vals = as_float_array(values, "values", error_cls=error_cls)
    if len(vals) < 2:
        raise error_cls("At least two approximations are required")
    f = ensure_positive(factor, "factor", error_cls=error_cls)
    table = vals[:]
    for k in range(1, len(vals)):
        denom = f ** (2 * k) - 1.0
        table = [table[i + 1] + (table[i + 1] - table[i]) / denom for i in range(len(table) - 1)]
    return table[0]


def condition_number_estimate(A: Sequence[Sequence[float]]) -> float:
    matrix = check_square_matrix(A)
    try:
        import numpy as np # type: ignore
        return float(np.linalg.cond(np.asarray(matrix, dtype=float)))
    except Exception as exc:
        raise STEMLinearAlgebraError("Failed to estimate matrix condition number", cause=exc) from exc


def backward_error(A: Sequence[Sequence[float]], x_approx: Sequence[float], b: Sequence[float], *, error_cls: type = STEMNumericalError) -> float:
    matrix = as_float_matrix(A, "A", error_cls=error_cls)
    x = as_float_array(x_approx, "x_approx", error_cls=error_cls)
    rhs = as_float_array(b, "b", error_cls=error_cls)
    if len(matrix) != len(rhs) or len(matrix[0]) != len(x):
        raise error_cls("Incompatible matrix/vector dimensions")
    residual = [math.fsum(row[j] * x[j] for j in range(len(x))) - rhs[i] for i, row in enumerate(matrix)]
    norm_r = max((abs(v) for v in residual), default=0.0)
    norm_a = max((math.fsum(abs(v) for v in row) for row in matrix), default=0.0)
    norm_x = max((abs(v) for v in x), default=0.0)
    norm_b = max((abs(v) for v in rhs), default=0.0)
    denom = norm_a * norm_x + norm_b
    return 0.0 if denom == 0.0 and norm_r == 0.0 else (math.inf if denom == 0.0 else norm_r / denom)


__all__ = [
    "ensure_number", "ensure_finite_number", "ensure_non_negative", "ensure_positive",
    "ensure_non_empty_string", "ensure_mapping", "ensure_sequence", "normalize_exponents",
    "combine_exponents", "power_exponents", "format_dimension", "dimension_from_mapping",
    "dimensions_compatible", "require_dimensions_match", "dimensionless_ratio",
    "combine_dimension_algebra", "affine_to_base", "affine_from_base", "parse_si_prefix",
    "format_si_prefix", "linear_unit_only", "affine_safe_convert", "coherence_check",
    "combine_standard_uncertainties", "welch_satterthwaite", "isclose", "json_safe",
    "to_json", "safe_divide", "safe_log", "forward_error", "relative_error",
    "safe_relative_error", "is_near_zero", "validate_callable", "validate_bounds",
    "validate_finite", "as_float_array", "as_float_matrix", "check_finite_array",
    "check_square_matrix", "check_symmetric_matrix", "check_positive_definite",
    "validate_shape", "normalize_numeric_input", "canonicalize_numeric_type",
    "build_diagnostics", "normalize_tolerance", "resolve_precision_policy",
    "stable_variance", "richardson_extrapolate", "condition_number_estimate",
    "backward_error",
]
