"""
Centralized helper functions for Optimization workflows.

This module is intentionally broad so every layer in the Optimization subsystem can
reuse common operations with consistent semantics.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import ast
import hashlib
import json

from dataclasses import fields
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple, Type

from .optimization_errors import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Helpers")
printer = PrettyPrinter()

def to_finite_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or None for bools, non-numbers, NaN and +/-inf."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def solve_linear_system(matrix: Sequence[Sequence[float]], vector: Sequence[float],
                        *, tolerance: float = 1e-12) -> Optional[List[float]]:
    """Solve ``matrix @ x = vector`` (square) by Gaussian elimination with partial pivoting.

    Returns None when the system is singular (pivot magnitude <= ``tolerance``).
    """
    n = len(vector)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise OptimizationValidationError("solve_linear_system requires a square matrix matching the vector",
                                          context={"rows": len(matrix), "vector": n})
    a = [[float(x) for x in row] + [float(v)] for row, v in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= tolerance:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            if factor:
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


# ---------------------------------------------------------------------------
# Expression constants and functions
# ---------------------------------------------------------------------------

EXPRESSION_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

EXPRESSION_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
}


# ---------------------------------------------------------------------------
# Fingerprinting and validation primitives
# ---------------------------------------------------------------------------

def stable_fingerprint(obj: Any) -> str:
    """Return a stable SHA-256 hex digest fingerprint for any JSON-serializable object or string."""
    if isinstance(obj, str):
        data = obj.encode("utf-8")
    else:
        try:
            data = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        except Exception:
            data = str(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def is_number(value: Any, minimum: Optional[float] = None) -> bool:
    """Return True if ``value`` is a finite number >= ``minimum``."""
    num = to_finite_float(value)
    if num is None:
        return False
    if minimum is not None and num < minimum:
        return False
    return True


def is_int(value: Any, minimum: Optional[int] = None) -> bool:
    """Return True if ``value`` is an integer >= ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    if not num.is_integer() or not math.isfinite(num):
        return False
    ival = int(num)
    if minimum is not None and ival < minimum:
        return False
    return True


def validate_settings(section_name: str, settings_obj: Any, checks: Sequence[Tuple[str, bool, str]]) -> None:
    """Validate settings attributes against boolean conditions; raises OptimizationConfigurationError on failure."""
    for field_name, condition, expected in checks:
        if not condition:
            val = getattr(settings_obj, field_name, None)
            raise OptimizationConfigurationError(
                f"Invalid setting '{section_name}.{field_name}': expected {expected}, got {val!r}",
                context={"section": section_name, "setting": field_name, "value": val, "expected": expected}
            )


def settings_from_mapping(cls: Type[Any], config: Mapping[str, Any], *, section: str, ignore: Sequence[str] = ()) -> Any:
    """Instantiate a dataclass settings class from a configuration mapping."""
    sec_data = config.get(section, {})
    if not isinstance(sec_data, Mapping):
        sec_data = {}
    known_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in sec_data.items() if k in known_fields and k not in ignore}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Expression compilation
# ---------------------------------------------------------------------------

class _ExpressionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)
        self.generic_visit(node)


def compile_expression(source: str, *, max_length: int = 500) -> Tuple[Callable[[Mapping[str, Any]], float], FrozenSet[str]]:
    """Compile an analytic expression string into an evaluatable function and its variable references."""
    if not isinstance(source, str) or not source.strip():
        raise OptimizationValidationError("Expression must be a non-empty string", context={"source": source})
    if len(source) > max_length:
        raise OptimizationValidationError(f"Expression exceeds maximum length of {max_length}", context={"length": len(source)})

    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise OptimizationValidationError(f"Invalid expression syntax: {exc}", context={"source": source}) from exc

    visitor = _ExpressionVisitor()
    visitor.visit(tree)

    allowed_node_types = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
        ast.Load, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Compare,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.BoolOp,
        ast.And, ast.Or, ast.Not
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_node_types):
            raise OptimizationValidationError(f"Unsupported expression construct: {type(node).__name__}", context={"source": source})
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in EXPRESSION_FUNCTIONS:
                func_name = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
                raise OptimizationValidationError(f"Unsupported function call '{func_name}' in expression", context={"source": source})

    code = compile(tree, filename="<string>", mode="eval")
    references = frozenset(n for n in visitor.names if n not in EXPRESSION_CONSTANTS and n not in EXPRESSION_FUNCTIONS)

    def evaluate(values: Mapping[str, Any]) -> float:
        namespace = {**EXPRESSION_CONSTANTS, **EXPRESSION_FUNCTIONS}
        for ref in references:
            if ref not in values:
                raise OptimizationEvaluationError(f"Missing value for variable/output '{ref}' in expression", context={"reference": ref})
            val = to_finite_float(values[ref])
            if val is None:
                raise OptimizationEvaluationError(f"Value for '{ref}' is not a finite number", context={"reference": ref, "value": values[ref]})
            namespace[ref] = val
        try:
            result = eval(code, {"__builtins__": {}}, namespace)
            fin = to_finite_float(result)
            if fin is None:
                raise OptimizationEvaluationError("Expression produced a non-finite value", context={"source": source, "result": result})
            return fin
        except OptimizationEvaluationError:
            raise
        except Exception as exc:
            raise OptimizationEvaluationError(f"Error evaluating expression '{source}': {exc}", context={"source": source}) from exc

    return evaluate, references


__all__ = [
    "EXPRESSION_CONSTANTS",
    "EXPRESSION_FUNCTIONS",
    "compile_expression",
    "is_int",
    "is_number",
    "settings_from_mapping",
    "solve_linear_system",
    "stable_fingerprint",
    "to_finite_float",
    "validate_settings",
]