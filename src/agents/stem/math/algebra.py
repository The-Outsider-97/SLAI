"""Exact and symbolic algebra primitives for the SLAI STEM subsystem.

Grounding: Geddes, Czapor & Labahn, *Algorithms for Computer Algebra*; Bronstein,
*Symbolic Integration I*. General inference and optimization remain outside
this module.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math

from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import STEMEquationError, STEMValidationError
from ..utils.stem_helpers import ensure_finite_number, ensure_sequence
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Algebra")
printer = PrettyPrinter()

Number = Union[int, float, Fraction]


def _trim(coefficients: Sequence[Fraction]) -> List[Fraction]:
    out = list(coefficients)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out or [Fraction(0)]


def _fraction(value: Number) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise STEMValidationError("Boolean is not an algebraic coefficient")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float) and math.isfinite(value):
        return Fraction(str(value))
    raise STEMValidationError("Unsupported/non-finite algebraic coefficient", context={"value": repr(value)})


class Algebra:
    """Deterministic exact polynomial/rational algebra service."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.algebra_config = dict(get_config_section("algebra", config=self.config) or {})
        if config:
            self.algebra_config.update(dict(config))

    @staticmethod
    def rational(value: Number, denominator: Optional[int] = None) -> Fraction:
        return Fraction(_fraction(value), denominator) if denominator is not None else _fraction(value)

    @staticmethod
    def normalize_polynomial(coefficients: Sequence[Number]) -> Tuple[Fraction, ...]:
        ensure_sequence(coefficients, "coefficients")
        if not coefficients:
            raise STEMEquationError("Polynomial requires at least one coefficient")
        return tuple(_trim([_fraction(value) for value in coefficients]))

    @staticmethod
    def polynomial_add(a: Sequence[Number], b: Sequence[Number]) -> Tuple[Fraction, ...]:
        aa, bb = list(Algebra.normalize_polynomial(a)), list(Algebra.normalize_polynomial(b))
        n = max(len(aa), len(bb))
        aa += [Fraction(0)] * (n - len(aa))
        bb += [Fraction(0)] * (n - len(bb))
        return tuple(_trim([x + y for x, y in zip(aa, bb)]))

    @staticmethod
    def polynomial_subtract(a: Sequence[Number], b: Sequence[Number]) -> Tuple[Fraction, ...]:
        return Algebra.polynomial_add(a, [-_fraction(v) for v in b])

    @staticmethod
    def polynomial_multiply(a: Sequence[Number], b: Sequence[Number]) -> Tuple[Fraction, ...]:
        aa, bb = Algebra.normalize_polynomial(a), Algebra.normalize_polynomial(b)
        out = [Fraction(0)] * (len(aa) + len(bb) - 1)
        for i, x in enumerate(aa):
            for j, y in enumerate(bb):
                out[i + j] += x * y
        return tuple(_trim(out))

    @staticmethod
    def polynomial_divmod(dividend: Sequence[Number], divisor: Sequence[Number]) -> Tuple[Tuple[Fraction, ...], Tuple[Fraction, ...]]:
        a, b = list(Algebra.normalize_polynomial(dividend)), list(Algebra.normalize_polynomial(divisor))
        if len(b) == 1 and b[0] == 0:
            raise STEMEquationError("Polynomial division by zero")
        if len(a) < len(b):
            return (Fraction(0),), tuple(a)
        q = [Fraction(0)] * (len(a) - len(b) + 1)
        r = a[:]
        while len(r) >= len(b) and not (len(r) == 1 and r[0] == 0):
            k = len(r) - len(b)
            factor = r[-1] / b[-1]
            q[k] = factor
            for j in range(len(b)):
                r[k + j] -= factor * b[j]
            r = _trim(r)
        return tuple(_trim(q)), tuple(_trim(r))

    @staticmethod
    def polynomial_gcd(a: Sequence[Number], b: Sequence[Number]) -> Tuple[Fraction, ...]:
        x, y = Algebra.normalize_polynomial(a), Algebra.normalize_polynomial(b)
        while not (len(y) == 1 and y[0] == 0):
            _, r = Algebra.polynomial_divmod(x, y)
            x, y = y, r
        lead = x[-1]
        return tuple(c / lead for c in x) if lead else (Fraction(0),)

    @staticmethod
    def polynomial_derivative(coefficients: Sequence[Number], order: int = 1) -> Tuple[Fraction, ...]:
        if order < 0:
            raise STEMValidationError("order must be non-negative")
        result = list(Algebra.normalize_polynomial(coefficients))
        for _ in range(order):
            result = [Fraction(i) * result[i] for i in range(1, len(result))] or [Fraction(0)]
        return tuple(result)

    @staticmethod
    def polynomial_evaluate(coefficients: Sequence[Number], x: Number) -> Fraction:
        coeffs = Algebra.normalize_polynomial(coefficients)
        value = _fraction(x)
        acc = Fraction(0)
        for coefficient in reversed(coeffs):
            acc = acc * value + coefficient
        return acc

    @staticmethod
    def quadratic_roots(a: Number, b: Number, c: Number) -> Tuple[complex, complex]:
        af, bf, cf = float(_fraction(a)), float(_fraction(b)), float(_fraction(c))
        if af == 0.0:
            if bf == 0.0:
                raise STEMEquationError("Degenerate equation has no unique linear/quadratic solution")
            root = -cf / bf
            return complex(root), complex(root)
        disc = bf * bf - 4.0 * af * cf
        sqrt_disc = complex(disc) ** 0.5
        # Stable variant avoids catastrophic cancellation for real discriminants.
        if disc >= 0.0:
            q = -0.5 * (bf + math.copysign(math.sqrt(disc), bf))
            if q != 0.0:
                return complex(q / af), complex(cf / q)
        return ((-bf + sqrt_disc) / (2.0 * af), (-bf - sqrt_disc) / (2.0 * af))

    @staticmethod
    def solve_linear(A: Sequence[Sequence[Number]], b: Sequence[Number]) -> Tuple[Fraction, ...]:
        rows = [list(map(_fraction, row)) for row in A]
        rhs = list(map(_fraction, b))
        n = len(rows)
        if n == 0 or len(rhs) != n or any(len(row) != n for row in rows):
            raise STEMEquationError("Exact linear solve requires a non-empty square system")
        aug = [rows[i] + [rhs[i]] for i in range(n)]
        for col in range(n):
            pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
            if pivot is None:
                raise STEMEquationError("Exact linear system is singular")
            aug[col], aug[pivot] = aug[pivot], aug[col]
            pv = aug[col][col]
            aug[col] = [v / pv for v in aug[col]]
            for r in range(n):
                if r == col:
                    continue
                factor = aug[r][col]
                if factor:
                    aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
        return tuple(aug[i][-1] for i in range(n))

    @staticmethod
    def substitute_polynomial(coefficients: Sequence[Number], x: Number) -> Fraction:
        return Algebra.polynomial_evaluate(coefficients, x)


__all__ = ["Algebra"]
