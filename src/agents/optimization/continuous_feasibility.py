"""Exact-rational continuous-time feasibility certificates for polynomial paths.

This is a general mathematical verification utility, NOT a collision engine.
It certifies inequalities over closed time intervals by enclosing a rational
polynomial in its Bernstein convex hull, with exact de Casteljau subdivision.
No floating-point approximation is used in the proof calculations.

Scope: rational-coefficient polynomials only. Unsupported geometry, uncertain
coefficients, or arbitrary black-box evaluations MUST NOT receive a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Any, Sequence

from .utils.optimization_errors import OptimizationValidationError

Rational = Fraction | int | str


def _rational(value: Rational, name: str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise OptimizationValidationError(
            f"{name} requires exact integer, Fraction or decimal/rational string; float is ambiguous"
        )
    if not isinstance(value, (int, Fraction, str)):
        raise OptimizationValidationError(f"{name} must be an exact rational")
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise OptimizationValidationError(f"{name} is not a valid rational") from exc
    if result.numerator.bit_length() > 8192 or result.denominator.bit_length() > 8192:
        raise OptimizationValidationError(f"{name} exceeds rational coefficient precision limit")
    return result


def _coefficients(values: Sequence[Rational], name: str) -> tuple[Fraction, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise OptimizationValidationError(f"{name} needs a coefficient sequence")
    if not 1 <= len(values) <= 33:
        raise OptimizationValidationError(f"{name} needs 1..33 rational coefficients")
    return tuple(_rational(item, f"{name}[{i}]") for i, item in enumerate(values))


def _eval(coefficients: Sequence[Fraction], time: Fraction) -> Fraction:
    value = Fraction(0)
    for coef in reversed(coefficients):
        value = value * time + coef
    return value


def _bernstein(
    coefficients: tuple[Fraction, ...], start: Fraction, end: Fraction,
) -> tuple[Fraction, ...]:
    """Convert ascending-power p(t) to Bernstein coefficients on [start,end]."""
    n = len(coefficients) - 1
    span = end - start
    power = [
        sum(
            (coefficients[k] * comb(k, j) * start ** (k - j) * span ** j
             for k in range(j, n + 1)), Fraction(0)
        )
        for j in range(n + 1)
    ]
    return tuple(
        sum(
            (power[j] * Fraction(comb(i, j), comb(n, j))
             for j in range(i + 1)), Fraction(0)
        )
        for i in range(n + 1)
    )


def _split(bernstein: tuple[Fraction, ...]) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Exact de Casteljau split at t=1/2, preserving convex hull bounds."""
    layers = [bernstein]
    while len(layers[-1]) > 1:
        row = layers[-1]
        layers.append(tuple((row[i] + row[i + 1]) / 2 for i in range(len(row) - 1)))
    return (
        tuple(row[0] for row in layers),
        tuple(row[-1] for row in reversed(layers)),
    )


@dataclass(frozen=True)
class ContinuousTimeCertificate:
    """Machine-auditable exact-rational proof evidence, not an AI assertion."""
    status: str
    inequality: str
    interval: tuple[str, str]
    polynomial: tuple[str, ...]
    certified_boxes: tuple[tuple[str, str, str, str], ...]
    unresolved_boxes: tuple[tuple[str, str], ...]
    visited_boxes: int
    witness_time: str | None = None
    witness_value: str | None = None
    guarantee: str = "exact_rational_polynomial_over_closed_interval"

    @property
    def feasible(self) -> bool:
        return self.status == "certified_feasible"


def certify_polynomial_nonnegative(
    coefficients: Sequence[Rational],
    *,
    start: Rational = 0,
    end: Rational = 1,
    max_depth: int = 20,
    max_boxes: int = 20000,
) -> ContinuousTimeCertificate:
    """Prove p(t)>=0 for ALL t in [start,end], disprove by rational witness,
    or return inconclusive. All proof arithmetic is exact fractions.

    A non-negative Bernstein lower hull bound certifies the entire interval;
    uncertain subintervals are subdivided. Reaching a depth/budget limit
    returns inconclusive rather than guessing feasibility.
    """
    coeffs = _coefficients(coefficients, "polynomial")
    a, b = _rational(start, "start"), _rational(end, "end")
    if not a < b:
        raise OptimizationValidationError("continuous certificate requires start < end")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or not 0 <= max_depth <= 40:
        raise OptimizationValidationError("max_depth must be an integer 0..40")
    if isinstance(max_boxes, bool) or not isinstance(max_boxes, int) or not 1 <= max_boxes <= 100000:
        raise OptimizationValidationError("max_boxes must be an integer 1..100000")
    items = [(a, b, _bernstein(coeffs, a, b), 0)]
    safe: list[tuple[str, str, str, str]] = []
    unknown: list[tuple[str, str]] = []
    checked = 0
    while items:
        left, right, controls, depth = items.pop()
        checked += 1
        lo, hi = min(controls), max(controls)
        if lo >= 0:
            safe.append((str(left), str(right), str(lo), str(hi)))
            continue
        mid = (left + right) / 2
        for test in (left, mid, right):
            value = _eval(coeffs, test)
            if value < 0:
                return ContinuousTimeCertificate(
                    "certified_violated", "p(t) >= 0", (str(a), str(b)),
                    tuple(str(c) for c in coeffs), tuple(safe), tuple(unknown),
                    checked, str(test), str(value),
                )
        if depth >= max_depth or checked + len(items) + 2 > max_boxes:
            unknown.append((str(left), str(right)))
            continue
        lcontrols, rcontrols = _split(controls)
        items.append((mid, right, rcontrols, depth + 1))
        items.append((left, mid, lcontrols, depth + 1))
    return ContinuousTimeCertificate(
        "inconclusive" if unknown else "certified_feasible",
        "p(t) >= 0", (str(a), str(b)),
        tuple(str(c) for c in coeffs), tuple(safe), tuple(unknown), checked,
    )


def _derivative(values: tuple[Fraction, ...], order: int) -> tuple[Fraction, ...]:
    result = values
    for _ in range(order):
        result = tuple(result[k] * k for k in range(1, len(result))) or (Fraction(0),)
    return result


def _multiply(left: Sequence[Fraction], right: Sequence[Fraction]) -> tuple[Fraction, ...]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return tuple(result)


def _squared_norm(paths: Sequence[Sequence[Fraction]]) -> tuple[Fraction, ...]:
    result: list[Fraction] = []
    for component in paths:
        squared = _multiply(component, component)
        if len(squared) > len(result):
            result.extend([Fraction(0)] * (len(squared) - len(result)))
        for i, coef in enumerate(squared):
            result[i] += coef
    return tuple(result)


def certify_polynomial_trajectory_limits(
    position_coefficients: Sequence[Sequence[Rational]],
    *,
    start: Rational = 0,
    end: Rational = 1,
    max_speed: Rational | None = None,
    max_acceleration: Rational | None = None,
    max_jerk: Rational | None = None,
    max_depth: int = 20,
) -> dict[str, ContinuousTimeCertificate]:
    """Certified continuous v/a/jerk magnitude limits for polynomial vectors.

    Constraints use L² norms squared to avoid irrational square roots.
    Coefficients must represent position in chosen consistent physical units.
    """
    if not position_coefficients:
        raise OptimizationValidationError("trajectory requires at least one position polynomial")
    positions = tuple(_coefficients(c, f"position[{i}]")
                      for i, c in enumerate(position_coefficients))
    limits = ((1, max_speed, "speed"), (2, max_acceleration, "acceleration"),
              (3, max_jerk, "jerk"))
    if all(value is None for _, value, _ in limits):
        raise OptimizationValidationError("at least one trajectory limit must be supplied")
    result: dict[str, ContinuousTimeCertificate] = {}
    for order, raw, label in limits:
        if raw is None:
            continue
        maximum = _rational(raw, label)
        if maximum < 0:
            raise OptimizationValidationError(f"{label} limit must be non-negative")
        sum_of_squares = _squared_norm([_derivative(c, order) for c in positions])
        residual = [-c for c in sum_of_squares]
        residual[0] += maximum ** 2
        result[label] = certify_polynomial_nonnegative(
            residual, start=start, end=end, max_depth=max_depth
        )
    return result


def certify_polynomial_sphere_separation(
    center_a: Sequence[Sequence[Rational]],
    center_b: Sequence[Sequence[Rational]],
    *,
    radius_a: Rational,
    radius_b: Rational,
    clearance: Rational = 0,
    start: Rational = 0,
    end: Rational = 1,
    max_depth: int = 20,
) -> ContinuousTimeCertificate:
    """Proof of continuous separation of two moving spheres in Euclidean space.

    Proves squared center distance >= (rA+rB+clearance)^2 at all times.
    This is exact for the modeled spheres and rational-polynomial centers ONLY.
    It is NOT a general polygon/mesh/rigid-body collision certificate.
    """
    if len(center_a) != len(center_b) or len(center_a) not in (2, 3):
        raise OptimizationValidationError("sphere centers must have equal 2D/3D dimension")
    ra, rb, space = (
        _rational(radius_a, "radius_a"),
        _rational(radius_b, "radius_b"),
        _rational(clearance, "clearance"),
    )
    if min(ra, rb, space) < 0:
        raise OptimizationValidationError("radii and clearance must be non-negative")
    components: list[tuple[Fraction, ...]] = []
    for i, (a, b) in enumerate(zip(center_a, center_b)):
        aa = _coefficients(a, f"center_a[{i}]")
        bb = _coefficients(b, f"center_b[{i}]")
        size = max(len(aa), len(bb))
        components.append(tuple(
            (aa[k] if k < len(aa) else Fraction(0))
            - (bb[k] if k < len(bb) else Fraction(0))
            for k in range(size)
        ))
    squared = list(_squared_norm(components))
    squared[0] -= (ra + rb + space) ** 2
    return certify_polynomial_nonnegative(
        squared, start=start, end=end, max_depth=max_depth
    )




def verify_continuous_certificate(certificate: ContinuousTimeCertificate) -> bool:
    """Independently replay certificate evidence using exact fractions.

    For a feasibility certificate, the claimed Bernstein hull intervals must
    cover the entire original domain with no gaps. For a violation, a rational
    time/value witness must evaluate below zero. Inconclusive is never proof.
    """
    if not isinstance(certificate, ContinuousTimeCertificate):
        return False
    if certificate.guarantee != "exact_rational_polynomial_over_closed_interval":
        return False
    if certificate.inequality != "p(t) >= 0":
        return False
    try:
        coefficients = _coefficients(certificate.polynomial, "certificate.polynomial")
        begin = _rational(certificate.interval[0], "begin")
        finish = _rational(certificate.interval[1], "finish")
        if begin >= finish:
            return False
        if certificate.status == "certified_violated":
            if certificate.witness_time is None or certificate.witness_value is None:
                return False
            t = _rational(certificate.witness_time, "witness_time")
            value = _rational(certificate.witness_value, "witness_value")
            return begin <= t <= finish and value < 0 and _eval(coefficients, t) == value
        if certificate.status != "certified_feasible" or certificate.unresolved_boxes:
            return False
        intervals: list[tuple[Fraction, Fraction]] = []
        for left_raw, right_raw, low_raw, high_raw in certificate.certified_boxes:
            left, right = _rational(left_raw, "segment_start"), _rational(right_raw, "segment_end")
            low, high = _rational(low_raw, "bernstein_min"), _rational(high_raw, "bernstein_max")
            if left >= right or low < 0:
                return False
            control = _bernstein(coefficients, left, right)
            if min(control) != low or max(control) != high:
                return False
            intervals.append((left, right))
        if not intervals:
            return False
        intervals.sort()
        if intervals[0][0] != begin or intervals[-1][1] != finish:
            return False
        return all(a[1] == b[0] for a, b in zip(intervals[:-1], intervals[1:]))
    except (OptimizationValidationError, TypeError, ValueError, IndexError, OverflowError):
        return False

__all__ = [
    "ContinuousTimeCertificate",
    "certify_polynomial_nonnegative",
    "certify_polynomial_trajectory_limits",
    "certify_polynomial_sphere_separation",
    "verify_continuous_certificate",
]
