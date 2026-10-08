"""Dimensional algebra and consistency checks for SLAI STEM.

Grounding: BIPM SI Brochure, ISO 80000-1, Kennedy (1996/1997), and
Buckingham (1914). Units themselves are handled by :mod:`unit_system`.
"""
from __future__ import annotations

__version__ = "2.3.0"

import ast
import math
import numpy as np # type: ignore

from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..stem_types import Dimension, Equation, Quantity, Unit
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import STEMBuckinghamPiError, STEMDimensionError, STEMDimensionMismatchError
from ..utils.stem_helpers import ensure_finite_number, ensure_mapping, ensure_non_empty_string, ensure_sequence, normalize_exponents
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Dimensions")
printer = PrettyPrinter()


def _matrix_rank(matrix: List[List[float]], tol: float = 1e-10) -> int:
    if not matrix:
        return 0
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=float), tol=tol))


def _nullspace(matrix: List[List[float]], tol: float = 1e-10) -> List[List[float]]:
    if not matrix:
        return []
    a = np.asarray(matrix, dtype=float)
    _, singular, vh = np.linalg.svd(a, full_matrices=True)
    rank = int(np.sum(singular > tol))
    return [row.tolist() for row in vh[rank:]]


class Dimensions:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.dimension_config = dict(get_config_section("dimensions", config=self.config) or {})
        if config:
            self.dimension_config.update(dict(config))
        self.allow_rational_exponents = bool(self.dimension_config.get("allow_rational_exponents", True))
        self.strict_consistency = bool(self.dimension_config.get("strict_consistency", True))
        self._base: Dict[str, str] = {}
        self._derived: Dict[str, Dimension] = {}
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        base = self.dimension_config.get("base_dimensions") or {
            "L": "length", "M": "mass", "T": "time", "I": "electric_current",
            "Θ": "thermodynamic_temperature", "N": "amount_of_substance", "J": "luminous_intensity",
        }
        ensure_mapping(base, "dimensions.base_dimensions", error_cls=STEMDimensionError)
        for symbol, name in base.items():
            self.register_base(str(symbol), str(name))

    def register_base(self, symbol: str, name: str, description: str = "") -> None:
        del description
        sym = ensure_non_empty_string(symbol, "symbol", error_cls=STEMDimensionError)
        label = ensure_non_empty_string(name, "name", error_cls=STEMDimensionError)
        if sym in self._base and self._base[sym] != label:
            raise STEMDimensionError("Base dimension symbol already registered", context={"symbol": sym})
        self._base[sym] = label

    def register_derived(self, name: str, exponents: Mapping[str, float], description: str = "") -> Dimension:
        del description
        label = ensure_non_empty_string(name, "name", error_cls=STEMDimensionError)
        normalized = normalize_exponents(exponents, error_cls=STEMDimensionError)
        unknown = sorted(set(normalized) - set(self._base))
        if unknown:
            raise STEMDimensionError("Derived dimension uses unknown base symbols", context={"unknown": unknown})
        dim = Dimension(normalized)
        self._derived[label] = dim
        return dim

    def lookup(self, name: str) -> Dimension:
        key = ensure_non_empty_string(name, "name", error_cls=STEMDimensionError)
        if key in self._derived:
            return self._derived[key]
        if key in self._base:
            return Dimension({key: 1.0})
        for symbol, label in self._base.items():
            if label == key:
                return Dimension({symbol: 1.0})
        raise STEMDimensionError("Unknown dimension", context={"name": key})

    def base_dimensions(self) -> Mapping[str, str]:
        return dict(self._base)

    def derived_dimensions(self) -> Mapping[str, Dimension]:
        return dict(self._derived)

    def dimension(self, symbols: Any) -> Dimension:
        return self._as_dimension(symbols)

    def _as_dimension(self, value: Any) -> Dimension:
        if isinstance(value, Dimension):
            return value
        if isinstance(value, Unit):
            return value.dimension
        if isinstance(value, Quantity):
            return value.unit.dimension
        if isinstance(value, str):
            return self.lookup(value)
        if isinstance(value, Mapping):
            return Dimension(value)
        raise STEMDimensionError("Cannot interpret value as a Dimension", context={"type": type(value).__name__})

    def is_compatible(self, a: Any, b: Any) -> bool:
        return self._as_dimension(a) == self._as_dimension(b)

    def require_compatible(self, a: Any, b: Any, context: str = "operation") -> None:
        da, db = self._as_dimension(a), self._as_dimension(b)
        if da != db:
            raise STEMDimensionMismatchError("Dimension mismatch", context={"operation": context, "left": da.to_dict(), "right": db.to_dict()})

    def is_dimensionless(self, d: Any) -> bool:
        return self._as_dimension(d).is_dimensionless

    def combine(self, a: Any, b: Any, op: str = "mul") -> Dimension:
        da, db = self._as_dimension(a), self._as_dimension(b)
        if op == "mul":
            return da * db
        if op == "div":
            return da / db
        raise STEMDimensionError("op must be 'mul' or 'div'")

    def power(self, d: Any, exponent: float) -> Dimension:
        value = ensure_finite_number(exponent, "exponent", error_cls=STEMDimensionError)
        if not self.allow_rational_exponents and not float(value).is_integer():
            raise STEMDimensionError("Fractional dimension exponents are disabled")
        return self._as_dimension(d) ** value

    def product(self, parts: Sequence[Any], sign: int = 1) -> Dimension:
        ensure_sequence(parts, "parts", error_cls=STEMDimensionError)
        if sign not in (-1, 1):
            raise STEMDimensionError("sign must be +/-1")
        result = Dimension()
        for part in parts:
            result = result * self._as_dimension(part) if sign == 1 else result / self._as_dimension(part)
        return result

    def infer_from_unit(self, unit: Unit) -> Dimension:
        if not isinstance(unit, Unit):
            raise STEMDimensionError("unit must be a Unit")
        return unit.dimension

    def dimension_from_expression(self, expression: str, symbols: Mapping[str, Dimension]) -> Dimension:
        ensure_non_empty_string(expression, "expression", error_cls=STEMDimensionError)
        ensure_mapping(symbols, "symbols", error_cls=STEMDimensionError)
        tree = ast.parse(expression, mode="eval")

        def walk(node: ast.AST) -> Dimension:
            if isinstance(node, ast.Constant):
                return Dimension()
            if isinstance(node, ast.Name):
                if node.id not in symbols:
                    raise STEMDimensionError("Unknown symbol in dimensional expression", context={"symbol": node.id})
                return self._as_dimension(symbols[node.id])
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return walk(node.operand)
            if isinstance(node, ast.BinOp):
                left = walk(node.left)
                if isinstance(node.op, (ast.Add, ast.Sub)):
                    right = walk(node.right)
                    self.require_compatible(left, right, "addition/subtraction")
                    return left
                if isinstance(node.op, ast.Mult):
                    return left * walk(node.right)
                if isinstance(node.op, ast.Div):
                    return left / walk(node.right)
                if isinstance(node.op, ast.Pow):
                    if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, (int, float)):
                        raise STEMDimensionError("Dimension exponent must be numeric literal")
                    return self.power(left, float(node.right.value))
            raise STEMDimensionError("Unsupported dimensional-expression syntax", context={"node": type(node).__name__})

        return walk(tree.body)

    def check_consistency(self, equation: Any) -> bool:
        if isinstance(equation, Mapping):
            if "left" not in equation or "right" not in equation:
                raise STEMDimensionError("Equation mapping requires left and right dimensions")
            consistent = self.is_compatible(equation["left"], equation["right"])
        elif isinstance(equation, Equation):
            metadata = equation.parameters.get("dimensions") if isinstance(equation.parameters, Mapping) else None
            if not isinstance(metadata, Mapping) or "left" not in metadata or "right" not in metadata:
                raise STEMDimensionError("Equation has no dimensional metadata to check")
            consistent = self.is_compatible(metadata["left"], metadata["right"])
        else:
            raise STEMDimensionError("Unsupported equation type for dimensional check")
        if self.strict_consistency and not consistent:
            raise STEMDimensionMismatchError("Equation is dimensionally inconsistent")
        return consistent

    def buckingham_pi(self, quantities: Sequence[Quantity]) -> Mapping[str, Any]:
        ensure_sequence(quantities, "quantities", error_cls=STEMBuckinghamPiError)
        if not quantities:
            raise STEMBuckinghamPiError("At least one quantity is required")
        if not all(isinstance(q, Quantity) for q in quantities):
            raise STEMBuckinghamPiError("All items must be Quantity instances")
        bases = sorted({key for q in quantities for key in q.unit.dimension.exponents})
        matrix = [[q.unit.dimension.exponents.get(base, 0.0) for q in quantities] for base in bases]
        tol = float((self.dimension_config.get("buckingham_pi") or {}).get("tolerance", 1e-10))
        vectors = _nullspace(matrix, tol)
        groups = []
        for vector in vectors:
            # Normalize by the largest coefficient for readable deterministic output.
            max_abs = max((abs(v) for v in vector), default=1.0)
            normalized = [0.0 if abs(v) < tol else v / max_abs for v in vector]
            groups.append({"exponents": normalized, "dimension": Dimension(), "expression": " * ".join(f"q{i}^{v:.6g}" for i, v in enumerate(normalized) if abs(v) >= tol) or "1"})
        return {"rank": _matrix_rank(matrix, tol), "base_dimensions": bases, "quantity_count": len(quantities), "pi_count": len(groups), "groups": groups}

    def to_dict(self) -> Mapping[str, Any]:
        return {"base_dimensions": dict(self._base), "derived_dimensions": {name: dim.to_dict() for name, dim in self._derived.items()}, "allow_rational_exponents": self.allow_rational_exponents, "strict_consistency": self.strict_consistency}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dimensions":
        instance = cls({"base_dimensions": data.get("base_dimensions", {})})
        for name, exponents in (data.get("derived_dimensions") or {}).items():
            instance.register_derived(str(name), exponents)
        return instance


__all__ = ["Dimensions"]
