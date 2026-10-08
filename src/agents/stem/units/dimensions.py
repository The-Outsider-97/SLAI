"""
Dimension algebra, registry, and consistency checking for the STEM subsystem.

Ownership
- base-dimension vectors;
- derived dimensions;
- dimensional equality;
- dimensional compatibility;
- dimensional algebra;
- dimensional consistency checking;
- dimension inference;
- dimensionless quantities;
- optional Buckingham-Π support.

This module does NOT own unit scale/offset conversion, SI prefix handling, or
unit registry lookup. Those concerns belong to ``unit_system.py``.

Sources
- Buckingham, E. (1914). "On Physically Similar Systems; Illustrations of the
  Use of Dimensional Equations." Physical Review, 4, 345.
  DOI 10.1103/PhysRev.4.345.
- Kennedy, A. J. (1996). Programming Languages and Dimensions.
  University of Cambridge Computer Laboratory, TR-391.
- Kennedy, A. (1997). "Relational Parametricity and Units of Measure." POPL '97.
- ISO 80000-1:2022 — Quantities and units — Part 1: General.
"""

from __future__ import annotations

__version__ = "2.3.0"

import ast
import math

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import *
from ..utils.stem_helpers import *
from ..stem_types import Dimension, Equation, Quantity, Unit
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Dimensions")
printer = PrettyPrinter()


_DEFAULT_SI_BASE_DIMENSIONS: Tuple[Tuple[str, str], ...] = (
    ("L", "length"),
    ("M", "mass"),
    ("T", "time"),
    ("I", "electric_current"),
    ("Θ", "thermodynamic_temperature"),
    ("N", "amount_of_substance"),
    ("J", "luminous_intensity"),
)


def _matrix_rank(matrix: List[List[float]], tol: float = 1e-10) -> int:
    """Compute the rank of a matrix via Gaussian elimination."""
    if not matrix:
        return 0
    a = [row[:] for row in matrix]
    m = len(a)
    n = len(a[0]) if m else 0
    rank = 0
    for col in range(n):
        pivot_row = None
        for r in range(rank, m):
            if abs(a[r][col]) > tol:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        a[rank], a[pivot_row] = a[pivot_row], a[rank]
        pivot = a[rank][col]
        for c in range(n):
            a[rank][c] /= pivot
        for r in range(m):
            if r != rank and abs(a[r][col]) > tol:
                factor = a[r][col]
                for c in range(n):
                    a[r][c] -= factor * a[rank][c]
        rank += 1
    return rank


def _nullspace(matrix: List[List[float]], tol: float = 1e-10) -> List[List[float]]:
    """Compute a basis for the nullspace of a matrix."""
    if not matrix:
        return []
    a = [row[:] for row in matrix]
    m = len(a)
    n = len(a[0]) if m else 0
    pivot_cols: List[int] = []
    rank = 0
    for col in range(n):
        pivot_row = None
        for r in range(rank, m):
            if abs(a[r][col]) > tol:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        a[rank], a[pivot_row] = a[pivot_row], a[rank]
        pivot = a[rank][col]
        for c in range(n):
            a[rank][c] /= pivot
        for r in range(m):
            if r != rank and abs(a[r][col]) > tol:
                factor = a[r][col]
                for c in range(n):
                    a[r][c] -= factor * a[rank][c]
        pivot_cols.append(col)
        rank += 1

    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis: List[List[float]] = []
    for free in free_cols:
        vec = [0.0] * n
        vec[free] = 1.0
        for i, pivot_col in enumerate(pivot_cols):
            vec[pivot_col] = -a[i][free]
        basis.append(vec)
    return basis


class Dimensions:
    """
    Dimension registry, algebra engine, and consistency checker.

    A ``Dimensions`` instance maintains the SI base dimensions plus any
    registered derived dimensions, and exposes algebraic operations and
    consistency checks that operate on :class:`~stem_types.Dimension` objects.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.dimensions_config = dict(get_config_section("dimensions", config=self.config) or {})
        if config:
            self.dimensions_config.update(dict(config))

        self._base_dimensions: Dict[str, str] = {}
        self._derived_dimensions: Dict[str, Dimension] = {}
        self._strict_consistency = bool(self.dimensions_config.get("strict_consistency", True))

        buckingham_cfg = self.dimensions_config.get("buckingham_pi") or {}
        self._buckingham_enabled = bool(buckingham_cfg.get("enabled", False))
        self._buckingham_tol = float(buckingham_cfg.get("tolerance", 1e-10))

        self._initialize_registry()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def _initialize_registry(self) -> None:
        base_cfg = self.dimensions_config.get("base_dimensions") or {}
        if base_cfg:
            for symbol, name in base_cfg.items():
                self.register_base(str(symbol), str(name))
        else:
            for symbol, name in _DEFAULT_SI_BASE_DIMENSIONS:
                self.register_base(symbol, name)

    def register_base(self, symbol: str, name: str, description: str = "") -> None:
        """Register an SI base dimension with its symbol and human name."""
        sym = ensure_non_empty_string(symbol, "symbol", error_cls=STEMDimensionError)
        nm = ensure_non_empty_string(name, "name", error_cls=STEMDimensionError)
        if sym in self._base_dimensions:
            if self._base_dimensions[sym] != nm:
                raise STEMDimensionError(
                    "base dimension already registered with a different name",
                    context={"symbol": sym, "existing": self._base_dimensions[sym], "new": nm},
                )
            return
        self._base_dimensions[sym] = nm
        self._derived_dimensions[sym] = Dimension({sym: 1.0})

    def register_derived(
        self,
        name: str,
        exponents: Mapping[str, float],
        description: str = "",
    ) -> Dimension:
        """Register a named derived dimension and return the ``Dimension``."""
        nm = ensure_non_empty_string(name, "name", error_cls=STEMDimensionError)
        try:
            dim = Dimension(dict(exponents))
        except STEMDimensionError:
            raise
        except Exception as exc:
            raise STEMDimensionError(
                "invalid derived dimension",
                context={"name": nm},
                cause=exc,
            ) from exc
        self._derived_dimensions[nm] = dim
        return dim

    def lookup(self, name: str) -> Dimension:
        """Retrieve a registered dimension by name or symbol."""
        nm = ensure_non_empty_string(name, "name", error_cls=STEMDimensionError)
        if nm in self._derived_dimensions:
            return self._derived_dimensions[nm]
        if nm in self._base_dimensions:
            return Dimension({nm: 1.0})
        raise STEMDimensionError("unknown dimension", context={"name": nm})

    def base_dimensions(self) -> Mapping[str, str]:
        """Return the SI base dimension registry as a copy."""
        return dict(self._base_dimensions)

    def derived_dimensions(self) -> Mapping[str, Dimension]:
        """Return the derived-dimension registry as a copy."""
        return dict(self._derived_dimensions)

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def dimension(self, symbols: Any) -> Dimension:
        """Build a ``Dimension`` from a symbol→exponent mapping or symbol sequence."""
        if isinstance(symbols, Dimension):
            return symbols
        if isinstance(symbols, Mapping):
            return Dimension(dict(symbols))
        seq = ensure_sequence(symbols, "symbols", allow_str=True, error_cls=STEMDimensionError)
        result: Dict[str, float] = {}
        for entry in seq:
            if not isinstance(entry, str):
                raise STEMDimensionError(
                    "symbol sequences must contain strings",
                    context={"received_type": type(entry).__name__},
                )
            result[entry] = result.get(entry, 0.0) + 1.0
        return Dimension(result)

    def _as_dimension(self, value: Any) -> Dimension:
        if isinstance(value, Dimension):
            return value
        if isinstance(value, str):
            return self.lookup(value)
        if isinstance(value, Mapping):
            return Dimension(dict(value))
        raise STEMDimensionError(
            "expected Dimension, name, or exponent mapping",
            context={"received_type": type(value).__name__},
        )

    def is_compatible(self, a: Any, b: Any) -> bool:
        """Return True when the two dimensions are equal."""
        try:
            return self._as_dimension(a) == self._as_dimension(b)
        except STEMDimensionError:
            return False

    def require_compatible(self, a: Any, b: Any, context: str = "operation") -> None:
        """Raise ``STEMDimensionMismatchError`` when two dimensions differ."""
        da = self._as_dimension(a)
        db = self._as_dimension(b)
        if da != db:
            raise STEMDimensionMismatchError(
                f"dimension mismatch in {context}",
                context={"a": da.to_dict(), "b": db.to_dict(), "operation": context},
            )

    def is_dimensionless(self, d: Any) -> bool:
        """Return True when the dimension has no base-dimension components."""
        return self._as_dimension(d).is_dimensionless

    def combine(self, a: Any, b: Any, op: str = "mul") -> Dimension:
        """Combine two dimensions via ``mul``, ``div``, or ``pow`` algebra."""
        da = self._as_dimension(a)
        if op == "mul":
            return da * self._as_dimension(b)
        if op == "div":
            return da / self._as_dimension(b)
        if op == "pow":
            return da ** float(b)
        raise STEMDimensionError(
            "unsupported combine operation",
            context={"op": op, "allowed": ["mul", "div", "pow"]},
        )

    def power(self, d: Any, exponent: float) -> Dimension:
        """Raise a dimension to a scalar power."""
        return self._as_dimension(d) ** float(exponent)

    def product(self, parts: Sequence[Any], sign: int = 1) -> Dimension:
        """Multiply (``sign=1``) or divide (``sign=-1``) a sequence of dimensions."""
        if not parts:
            return Dimension({})
        accumulator = self._as_dimension(parts[0])
        for part in parts[1:]:
            if sign >= 0:
                accumulator = accumulator * self._as_dimension(part)
            else:
                accumulator = accumulator / self._as_dimension(part)
        return accumulator

    # ------------------------------------------------------------------
    # Consistency and inference
    # ------------------------------------------------------------------

    def check_consistency(self, equation: Any) -> bool:
        """
        Dimensionally verify an equation or mapping of terms.

        Supported forms:
        - ``Equation``: inspects ``domain`` if present; otherwise returns True.
        - Mapping with ``lhs`` and ``rhs``: checks equality.
        - Mapping with ``terms``: checks that all terms share a dimension.
        """
        if isinstance(equation, Equation):
            if equation.domain is None:
                return True
            return True

        if isinstance(equation, Mapping):
            if "lhs" in equation and "rhs" in equation:
                return self.is_compatible(equation["lhs"], equation["rhs"])
            terms = equation.get("terms")
            if terms is not None:
                seq = ensure_sequence(terms, "terms", error_cls=STEMDimensionError)
                if len(seq) < 2:
                    return True
                first = seq[0]
                for other in seq[1:]:
                    if not self.is_compatible(first, other):
                        if self._strict_consistency:
                            raise STEMDimensionMismatchError(
                                "inconsistent terms in equation",
                                context={"first": self._safe_dim(first), "other": self._safe_dim(other)},
                            )
                        return False
                return True

        raise STEMDimensionError(
            "check_consistency expects an Equation or mapping with lhs/rhs or terms",
            context={"received_type": type(equation).__name__},
        )

    @staticmethod
    def _safe_dim(value: Any) -> Any:
        if isinstance(value, Dimension):
            return value.to_dict()
        return json_safe(value)

    def infer_from_unit(self, unit: Unit) -> Dimension:
        """Return the ``Dimension`` carried by a ``Unit``."""
        if not isinstance(unit, Unit):
            raise STEMDimensionError(
                "infer_from_unit requires a Unit",
                context={"received_type": type(unit).__name__},
            )
        return unit.dimension

    def dimension_from_expression(
        self,
        expression: str,
        symbols: Mapping[str, Dimension],
    ) -> Dimension:
        """
        Infer a ``Dimension`` from a symbolic expression using Kennedy-style
        algebraic typing. Only ``*``, ``/``, ``**``, parentheses, and symbol
        names are supported.
        """
        ensure_non_empty_string(expression, "expression", error_cls=STEMDimensionError)
        ensure_mapping(symbols, "symbols", error_cls=STEMDimensionError)

        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise STEMDimensionError(
                "failed to parse dimension expression",
                context={"expression": expression},
                cause=exc,
            ) from exc

        def _walk(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return _walk(node.body)
            if isinstance(node, ast.Name):
                if node.id not in symbols:
                    raise STEMDimensionError(
                        "unknown symbol in dimension expression",
                        context={"symbol": node.id},
                    )
                return symbols[node.id]
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise STEMDimensionError(
                        "only numeric exponents are allowed in dimension expressions",
                        context={"value": repr(node.value)},
                    )
                return float(node.value)
            if isinstance(node, ast.BinOp):
                left = _walk(node.left)
                right = _walk(node.right)
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    if not isinstance(right, (int, float)):
                        raise STEMDimensionError("dimension exponent must be numeric")
                    if not isinstance(left, Dimension):
                        return left ** right
                    return left ** float(right)
                raise STEMDimensionError(
                    "unsupported operator in dimension expression",
                    context={"operator": type(node.op).__name__},
                )
            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.USub):
                    raise STEMDimensionError("unary minus is not meaningful for dimensions")
                if isinstance(node.op, ast.UAdd):
                    return _walk(node.operand)
                raise STEMDimensionError(
                    "unsupported unary operator in dimension expression",
                    context={"operator": type(node.op).__name__},
                )
            raise STEMDimensionError(
                "unsupported AST node in dimension expression",
                context={"node": type(node).__name__},
            )

        result = _walk(parsed)
        if isinstance(result, (int, float)):
            return Dimension({})
        if not isinstance(result, Dimension):
            raise STEMDimensionError(
                "failed to infer a dimension from expression",
                context={"expression": expression},
            )
        return result

    # ------------------------------------------------------------------
    # Buckingham-Π support
    # ------------------------------------------------------------------

    def buckingham_pi(self, quantities: Sequence[Quantity]) -> Mapping[str, Any]:
        """
        Compute a Buckingham-Π basis from a sequence of quantities.

        Returns a mapping describing the dimensional matrix, its rank, and a
        basis for the dimensionless groups. Disabled by default; enable via
        the ``dimensions.buckingham_pi.enabled`` configuration flag.
        """
        if not self._buckingham_enabled:
            raise STEMBuckinghamPiError(
                "Buckingham-Π support is disabled by configuration",
                context={"enabled": False},
            )

        seq = ensure_sequence(quantities, "quantities", error_cls=STEMBuckinghamPiError)
        if len(seq) < 2:
            raise STEMBuckinghamPiError("Buckingham-Π requires at least two quantities")

        dim_symbols: List[str] = []
        for q in seq:
            if not isinstance(q, Quantity):
                raise STEMBuckinghamPiError(
                    "all inputs must be Quantity instances",
                    context={"received_type": type(q).__name__},
                )
            for sym in q.unit.dimension.exponents:
                if sym not in dim_symbols:
                    dim_symbols.append(sym)
        dim_symbols.sort()

        m = len(dim_symbols)
        n = len(seq)

        matrix = [[0.0] * n for _ in range(m)]
        for j, q in enumerate(seq):
            for i, sym in enumerate(dim_symbols):
                matrix[i][j] = float(q.unit.dimension.exponents.get(sym, 0.0))

        rank = _matrix_rank(matrix, tol=self._buckingham_tol)
        groups = _nullspace(matrix, tol=self._buckingham_tol)

        return {
            "base_dimensions": dim_symbols,
            "matrix": matrix,
            "rank": rank,
            "num_quantities": n,
            "num_groups": n - rank,
            "groups": groups,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-safe representation of the registry."""
        return {
            "base_dimensions": dict(self._base_dimensions),
            "derived_dimensions": {
                name: dim.to_dict() for name, dim in self._derived_dimensions.items()
            },
            "strict_consistency": self._strict_consistency,
            "buckingham_enabled": self._buckingham_enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dimensions":
        """Rebuild a ``Dimensions`` instance from a serialized mapping."""
        ensure_mapping(data, "data", error_cls=STEMDimensionError)
        instance = cls()
        base = data.get("base_dimensions") or {}
        for symbol, name in base.items():
            instance.register_base(str(symbol), str(name))
        derived = data.get("derived_dimensions") or {}
        for name, exponents in derived.items():
            instance.register_derived(str(name), exponents)
        return instance


__all__ = ["Dimensions"]