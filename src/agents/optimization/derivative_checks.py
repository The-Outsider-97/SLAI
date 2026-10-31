"""Optional numerical checks of user-supplied Optimization objective gradients.

Finite-difference kernels belong to the STEM NumericalMethods subsystem.
This module only adapts the canonical Optimization mapping contract and
compares results; it does not implement or replace differentiation algorithms.

Valid only for deterministic, smooth-enough scalar *real* objectives.
Near boundaries, uses one-sided differences with correspondingly lower
accuracy. Numerical agreement is diagnostic, not a mathematical proof.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .optimization_math import evaluate_objective
from .optimization_types import DomainKind, Objective, OptimizationProblem
from .utils.optimization_errors import OptimizationValidationError
from .utils.optimization_helpers import require_finite


def verify_objective_gradient(
    problem: OptimizationProblem,
    point: Mapping[str, Any],
    *,
    step: float = 1e-5,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare an analytical gradient to STEM's numerical finite differences."""
    if problem.is_multi_objective or problem.objective is None:
        raise OptimizationValidationError("gradient verification requires a scalar objective")
    objective: Objective = problem.objective
    if objective.gradient is None:
        raise OptimizationValidationError("objective has no declared analytical gradient")
    if not problem.deterministic:
        raise OptimizationValidationError("stochastic objectives need independent derivative validation")
    if any(v.shape or v.domain.kind is not DomainKind.REAL for v in problem.variables):
        raise OptimizationValidationError("numerical gradient check requires scalar continuous variables")
    step = require_finite(step, "gradient_step")
    atol = require_finite(absolute_tolerance, "absolute_tolerance")
    rtol = require_finite(relative_tolerance, "relative_tolerance")
    if step <= 0 or atol < 0 or rtol < 0:
        raise OptimizationValidationError("gradient step must be positive and tolerances non-negative")
    from src.agents.stem.math.numerical_methods import NumericalMethods

    differentiation = NumericalMethods()
    variables = tuple(problem.variables)
    mapping = {v.name: require_finite(point[v.name], v.name) for v in variables}
    try:
        supplied = objective.gradient(mapping)
    except Exception as exc:
        raise OptimizationValidationError("objective analytical gradient callback failed") from exc
    if isinstance(supplied, Mapping):
        if set(supplied) != set(mapping):
            raise OptimizationValidationError("gradient keys must match variable names exactly")
        analytic = {n: require_finite(supplied[n], f"gradient[{n}]") for n in mapping}
    else:
        try:
            values = tuple(supplied)
        except TypeError as exc:
            raise OptimizationValidationError("gradient must return mapping or sequence") from exc
        if len(values) != len(variables):
            raise OptimizationValidationError("gradient sequence has incorrect length")
        analytic = {v.name: require_finite(values[i], v.name) for i, v in enumerate(variables)}

    details: dict[str, Any] = {}
    for variable in variables:
        name, x = variable.name, mapping[variable.name]
        local_step = step * max(1.0, abs(x))
        lower, upper = variable.domain.lower, variable.domain.upper
        left_room = math.inf if lower is None else max(0.0, x - lower)
        right_room = math.inf if upper is None else max(0.0, upper - x)
        def evaluate_at(z: float) -> float:
            modified = dict(mapping)
            modified[name] = z
            return evaluate_objective(objective, modified)
        if left_room > 0 and right_room > 0:
            h = min(local_step, left_room / 2, right_room / 2)
            numerical = differentiation.central_difference(evaluate_at, x, h=h)
            scheme = "central"
        elif right_room > 0:
            h = min(local_step, right_room / 2)
            numerical = differentiation.forward_difference(evaluate_at, x, h=h)
            scheme = "forward"
        elif left_room > 0:
            h = min(local_step, left_room / 2)
            numerical = -differentiation.forward_difference(lambda neg: evaluate_at(-neg), -x, h=h)
            scheme = "backward"
        else:
            raise OptimizationValidationError("cannot differentiate a fixed or empty variable interval")
        numeric = require_finite(numerical, f"numeric_gradient[{name}]")
        difference = abs(analytic[name] - numeric)
        threshold = atol + rtol * max(abs(numeric), abs(analytic[name]))
        details[name] = {
            "analytic": analytic[name], "numerical": numeric,
            "absolute_error": difference, "tolerance": threshold,
            "method": scheme, "step": h, "passes": difference <= threshold,
        }
    return {"passes": all(v["passes"] for v in details.values()),
            "variables": details, "method_owner": "STEM.NumericalMethods"}


__all__ = ["verify_objective_gradient"]
