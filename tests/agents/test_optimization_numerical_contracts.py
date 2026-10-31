"""Regression checks for general-purpose optimization numerical contracts.

These tests deliberately exercise the existing canonical types and helpers.
No CAD/animation-specific code, solver substitutes, or monkeypatches.
"""
from __future__ import annotations

import math

import pytest

from src.agents.optimization.optimization_math import (
    domain_violation,
    initial_point,
    is_feasible,
)
from src.agents.optimization.optimization_types import (
    Objective,
    OptimizationProblem,
    Variable,
    VariableDomain,
    DomainKind,
)
from src.agents.optimization.registries.scalarization_registry import (
    weighted_sum_transform,
)


def _problem(variable: Variable) -> OptimizationProblem:
    return OptimizationProblem(
        name="numerical_domain_regression",
        variables=(variable,),
        objectives=(Objective("minimize", linear={variable.name: 1.0}),),
    )


@pytest.mark.parametrize(
    ("domain", "invalid", "valid"),
    [
        (VariableDomain.real(0, 10, strict_lower=True), 0.0, 0.1),
        (VariableDomain.real(0, 10, strict_upper=True), 10.0, 9.9),
        (VariableDomain.real(0, None, strict_lower=True), 0.0, 1.0),
        (VariableDomain.real(None, 0, strict_upper=True), 0.0, -1.0),
    ],
)
def test_strict_bounds_exclude_endpoints_even_with_large_tolerance(domain, invalid, valid):
    problem = _problem(Variable("x", domain))
    assert not is_feasible(problem, {"x": invalid}, tolerance=1.0)
    assert is_feasible(problem, {"x": valid}, tolerance=1e-8)


def test_initial_point_of_open_bounds_is_inside_domain():
    for domain in (
        VariableDomain.real(0, None, strict_lower=True),
        VariableDomain.real(None, 0, strict_upper=True),
        VariableDomain.real(0, 10, strict_lower=True, strict_upper=True),
        VariableDomain.real(1e308, 1.7e308),
        VariableDomain(DomainKind.INTEGER, lower=1, strict_lower=True),
    ):
        problem = _problem(Variable("x", domain))
        point = initial_point(problem)
        assert math.isfinite(point["x"])
        assert is_feasible(problem, point, tolerance=1e-8)


def test_semi_continuous_distance_to_disjoint_domain():
    variable = Variable("x", VariableDomain.semi_continuous(10, 20))
    assert domain_violation(variable, 0.0) == 0
    assert domain_violation(variable, 2.0) == pytest.approx(2.0)
    assert domain_violation(variable, 8.0) == pytest.approx(2.0)
    assert domain_violation(variable, 21.0) == pytest.approx(1.0)
    problem = _problem(variable)
    assert is_feasible(problem, {"x": 0.0})
    assert not is_feasible(problem, {"x": 2.0}, tolerance=0.1)


def test_semi_integer_distance_respects_integer_lattice_and_zero_branch():
    variable = Variable("x", VariableDomain.semi_integer(10, 20))
    assert domain_violation(variable, 0.0) == 0
    assert domain_violation(variable, 2.0) == pytest.approx(2.0)
    assert domain_violation(variable, 9.4) == pytest.approx(0.6)
    assert domain_violation(variable, 10.4) == pytest.approx(0.4)
    assert domain_violation(variable, 10.0) == 0
    assert domain_violation(variable, 21.0) == pytest.approx(1.0)


def test_large_multiobjective_weights_are_finite_and_normalized():
    variable = Variable("x", VariableDomain.real(-5, 5))
    problem = OptimizationProblem(
        name="very_large_finite_weights",
        variables=(variable,),
        objectives=(
            Objective("first", linear={"x": 1.0}),
            Objective("second", linear={"x": -1.0}),
        ),
    )
    transformed = weighted_sum_transform(
        problem, weights={"first": 1e308, "second": 1e308}
    )
    weights = transformed.objectives[0].metadata["weights"]
    assert weights["first"] == pytest.approx(0.5)
    assert weights["second"] == pytest.approx(0.5)
    assert all(math.isfinite(v) for v in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
