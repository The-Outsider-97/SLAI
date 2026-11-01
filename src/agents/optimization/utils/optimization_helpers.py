"""Numerical and deterministic helper primitives for SLAI optimization.

This module deliberately contains *shared primitives*, not solver algorithms.
Solver algorithms stay in ``optimization/solvers`` and orchestration stays in
``optimization/registries`` / ``optimization_solver``.
"""
from __future__ import annotations

__version__ = "2.3.0"

import ast
import hashlib
import json
import math
import time

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Callable, FrozenSet, Mapping, Sequence, Type, cast

from .optimization_errors import OptimizationEvaluationError, OptimizationValidationError


EXPRESSION_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": cast(Callable[..., float], abs),
    "min": cast(Callable[..., float], min),
    "max": cast(Callable[..., float], max),
    "sqrt": cast(Callable[..., float], math.sqrt),
    "exp": cast(Callable[..., float], math.exp),
    "log": cast(Callable[..., float], math.log),
    "log10": cast(Callable[..., float], math.log10),
    "sin": cast(Callable[..., float], math.sin),
    "cos": cast(Callable[..., float], math.cos),
    "tan": cast(Callable[..., float], math.tan),
    "floor": cast(Callable[..., float], math.floor),
    "ceil": cast(Callable[..., float], math.ceil),
}
EXPRESSION_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}
_ALLOWED_EXPRESSION_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Not,
)


def to_finite_float(value: Any) -> float | None:
    """Convert a non-bool numeric value to finite ``float`` or return ``None``."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def require_finite(value: Any, field_name: str) -> float:
    result = to_finite_float(value)
    if result is None:
        raise OptimizationValidationError(
            f"{field_name} must be a finite real number",
            context={"field": field_name, "value": repr(value)},
        )
    return result


def is_number(value: Any, minimum: float | None = None) -> bool:
    numeric = to_finite_float(value)
    return numeric is not None and (minimum is None or numeric >= minimum)


def is_int(value: Any, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def stable_fingerprint(obj: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-like data."""
    def normalize(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                return repr(value)
            return {"__float__": value.hex()}
        if isinstance(value, Mapping):
            return {str(key): normalize(value[key]) for key in sorted(value, key=lambda k: str(k))}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [normalize(item) for item in value]
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=repr))
        if isinstance(value, Enum):
            return normalize(value.value)
        if hasattr(value, "__dict__"):
            return {"__type__": value.__class__.__qualname__, "state": normalize(vars(value))}
        return {"__repr__": repr(value), "__type__": value.__class__.__qualname__}

    payload = json.dumps(normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def absolute_relative_gap(
    primal_bound: float | None,
    dual_bound: float | None,
    *,
    denominator_floor: float = 1e-12,
) -> tuple[float | None, float | None]:
    """Compute absolute and scale-invariant relative optimality gaps.

    The convention is ``|primal-dual| / max(|primal|, |dual|, floor)``.  This
    avoids division-by-zero ambiguity and is symmetric with respect to
    minimization/maximization after the backend has normalized its bounds.
    """
    if primal_bound is None or dual_bound is None:
        return None, None
    p = to_finite_float(primal_bound)
    d = to_finite_float(dual_bound)
    if p is None or d is None:
        return None, None
    absolute = abs(p - d)
    relative = absolute / max(abs(p), abs(d), denominator_floor)
    return absolute, relative


def dominates_values(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance: float = 0.0,
    strict: bool = False,
) -> bool:
    """Return whether a minimization vector Pareto-dominates another vector."""
    if len(left) != len(right):
        raise OptimizationValidationError("dominance vectors must have equal dimension")
    if not left:
        return False
    tol = require_finite(tolerance, "tolerance")
    if tol < 0:
        raise OptimizationValidationError("dominance tolerance must be non-negative")
    if strict:
        return all(float(a) < float(b) - tol for a, b in zip(left, right))
    weakly_better = all(float(a) <= float(b) + tol for a, b in zip(left, right))
    if not weakly_better:
        return False
    return any(float(a) < float(b) - tol for a, b in zip(left, right))


def validate_settings(
    section_name: str,
    settings_obj: Any,
    checks: Sequence[tuple[str, bool, str]],
) -> None:
    errors = [message for _, valid, message in checks if not valid]
    if errors:
        raise OptimizationValidationError(
            f"Invalid {section_name} settings: {'; '.join(errors)}",
            context={"section": section_name, "settings": repr(settings_obj)},
        )


def settings_from_mapping(
    cls: Type[Any],
    config: Mapping[str, Any],
    *,
    section: str,
    ignore: Sequence[str] = (),
) -> Any:
    raw = config.get(section, {})
    if not isinstance(raw, Mapping):
        raw = {}
    names = {f.name for f in fields(cls)} - set(ignore)
    return cls(**{key: value for key, value in raw.items() if key in names})


@dataclass(frozen=True, slots=True)
class Deadline:
    """Monotonic deadline helper used by iterative solvers."""

    expires_at: float | None

    @classmethod
    def from_seconds(cls, seconds: float | None) -> "Deadline":
        if seconds is None:
            return cls(None)
        finite = require_finite(seconds, "time_limit_seconds")
        if finite <= 0:
            raise OptimizationValidationError("time_limit_seconds must be positive")
        return cls(time.monotonic() + finite)

    def expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() >= self.expires_at

    def remaining(self) -> float | None:
        if self.expires_at is None:
            return None
        return max(0.0, self.expires_at - time.monotonic())


class _ExpressionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)
        self.generic_visit(node)


def compile_expression(
    source: str,
    *,
    max_length: int = 1000,
) -> tuple[Callable[[Mapping[str, Any]], float], FrozenSet[str]]:
    """Compile a restricted arithmetic expression into a finite-valued callable."""
    if not isinstance(source, str) or not source.strip():
        raise OptimizationValidationError("Expression must be a non-empty string")
    if not is_int(max_length, 1):
        raise OptimizationValidationError("max_length must be a positive integer")
    if len(source) > max_length:
        raise OptimizationValidationError(
            "Expression exceeds maximum length",
            context={"length": len(source), "max_length": max_length},
        )
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise OptimizationValidationError(
            "Invalid expression syntax",
            context={"source": source, "line": exc.lineno, "offset": exc.offset},
            cause=exc,
        ) from exc

    visitor = _ExpressionVisitor()
    visitor.visit(tree)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_EXPRESSION_NODES):
            raise OptimizationValidationError(
                "Unsupported expression construct",
                context={"source": source, "node": type(node).__name__},
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in EXPRESSION_FUNCTIONS:
                raise OptimizationValidationError(
                    "Unsupported function in expression",
                    context={"source": source},
                )

    code = compile(tree, filename="<optimization-expression>", mode="eval")
    references = frozenset(
        name
        for name in visitor.names
        if name not in EXPRESSION_FUNCTIONS and name not in EXPRESSION_CONSTANTS
    )

    def evaluate(values: Mapping[str, Any]) -> float:
        namespace: dict[str, Any] = {**EXPRESSION_FUNCTIONS, **EXPRESSION_CONSTANTS}
        for name in references:
            if name not in values:
                raise OptimizationEvaluationError(
                    "Expression reference is missing",
                    context={"reference": name, "source": source},
                )
            namespace[name] = require_finite(values[name], name)
        try:
            raw = eval(code, {"__builtins__": {}}, namespace)
        except OptimizationEvaluationError:
            raise
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise OptimizationEvaluationError(
                "Expression evaluation failed",
                context={"source": source},
                cause=exc,
            ) from exc
        result = to_finite_float(raw)
        if result is None:
            raise OptimizationEvaluationError(
                "Expression produced a non-finite value",
                context={"source": source, "result": repr(raw)},
            )
        return result

    return evaluate, references


def solve_linear_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> list[float]:
    """Solve a small dense linear system with partial pivoting.

    This utility intentionally avoids importing a solver backend for tiny
    internal calculations.  Optimization algorithms themselves use SciPy.
    """
    n = len(matrix)
    if n == 0 or len(vector) != n or any(len(row) != n for row in matrix):
        raise OptimizationValidationError("linear system must be square and dimensionally consistent")
    tol = require_finite(tolerance, "tolerance")
    a = [[require_finite(v, "matrix entry") for v in row] for row in matrix]
    b = [require_finite(v, "vector entry") for v in vector]
    for pivot in range(n):
        row = max(range(pivot, n), key=lambda i: abs(a[i][pivot]))
        if abs(a[row][pivot]) <= tol:
            raise OptimizationValidationError("linear system is singular or ill-conditioned")
        if row != pivot:
            a[pivot], a[row] = a[row], a[pivot]
            b[pivot], b[row] = b[row], b[pivot]
        scale = a[pivot][pivot]
        for j in range(pivot, n):
            a[pivot][j] /= scale
        b[pivot] /= scale
        for i in range(n):
            if i == pivot:
                continue
            factor = a[i][pivot]
            if abs(factor) <= tol:
                continue
            for j in range(pivot, n):
                a[i][j] -= factor * a[pivot][j]
            b[i] -= factor * b[pivot]
    return b


__all__ = [
    "Deadline",
    "EXPRESSION_CONSTANTS",
    "EXPRESSION_FUNCTIONS",
    "absolute_relative_gap",
    "compile_expression",
    "dominates_values",
    "is_int",
    "is_number",
    "require_finite",
    "settings_from_mapping",
    "solve_linear_system",
    "stable_fingerprint",
    "to_finite_float",
    "validate_settings",
]
