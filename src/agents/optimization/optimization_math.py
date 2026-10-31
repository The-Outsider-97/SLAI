"""Mathematical semantics shared by SLAI optimization solvers.

The functions in this module define objective/constraint evaluation,
classification, feasibility, vectorization, algebraic matrix extraction,
quadratic convexity checks, and exact finite-domain bounding primitives.  They
are deliberately solver-neutral so every backend observes the same numerical
contract.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import numpy as np # type: ignore

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .optimization_types import *
from .utils.optimization_errors import OptimizationEvaluationError, OptimizationValidationError, UnsupportedProblemError
from .utils.optimization_helpers import absolute_relative_gap, require_finite, to_finite_float


def variable_index(problem: OptimizationProblem) -> dict[str, int]:
    return {variable.name: index for index, variable in enumerate(problem.variables)}


def evaluate_objective(objective: Objective, point: Mapping[str, Any]) -> float:
    """Evaluate an objective in its original mathematical sense."""
    total = objective.constant
    for name, coefficient in objective.linear.items():
        if name not in point:
            raise OptimizationEvaluationError("objective variable missing", context={"objective": objective.name, "variable": name})
        value = to_finite_float(point[name])
        if value is None:
            raise OptimizationEvaluationError("objective variable is non-numeric/non-finite", context={"objective": objective.name, "variable": name})
        total += coefficient * value
    for (left, right), coefficient in objective.quadratic.items():
        if left not in point or right not in point:
            raise OptimizationEvaluationError("quadratic objective variable missing", context={"objective": objective.name, "variables": [left, right]})
        a = to_finite_float(point[left])
        b = to_finite_float(point[right])
        if a is None or b is None:
            raise OptimizationEvaluationError("quadratic objective requires finite numeric values", context={"objective": objective.name})
        total += coefficient * a * b
    if objective.function is not None:
        try:
            raw = objective.function(point)
        except OptimizationEvaluationError:
            raise
        except Exception as exc:
            raise OptimizationEvaluationError(
                "objective callback failed",
                context={"objective": objective.name},
                cause=exc,
            ) from exc
        callback_value = to_finite_float(raw)
        if callback_value is None:
            raise OptimizationEvaluationError("objective callback returned non-finite value", context={"objective": objective.name, "value": repr(raw)})
        total += callback_value
    if not math.isfinite(total):
        raise OptimizationEvaluationError("objective evaluated to non-finite value", context={"objective": objective.name})
    return float(total)


def signed_objective_value(objective: Objective, point: Mapping[str, Any]) -> float:
    value = evaluate_objective(objective, point)
    return value if objective.sense is ObjectiveSense.MINIMIZE else -value


def objective_values(problem: OptimizationProblem, point: Mapping[str, Any]) -> dict[str, float]:
    return {objective.name: evaluate_objective(objective, point) for objective in problem.objectives}


def evaluate_constraint_value(constraint: Constraint, point: Mapping[str, Any]) -> float:
    total = 0.0
    for name, coefficient in constraint.linear.items():
        if name not in point:
            raise OptimizationEvaluationError("constraint variable missing", context={"constraint": constraint.name, "variable": name})
        value = to_finite_float(point[name])
        if value is None:
            raise OptimizationEvaluationError("constraint variable must be finite numeric", context={"constraint": constraint.name, "variable": name})
        total += coefficient * value
    if constraint.function is not None:
        try:
            raw = constraint.function(point)
        except OptimizationEvaluationError:
            raise
        except Exception as exc:
            raise OptimizationEvaluationError(
                "constraint callback failed",
                context={"constraint": constraint.name},
                cause=exc,
            ) from exc
        callback_value = to_finite_float(raw)
        if callback_value is None:
            raise OptimizationEvaluationError("constraint callback returned non-finite value", context={"constraint": constraint.name, "value": repr(raw)})
        total += callback_value
    if not math.isfinite(total):
        raise OptimizationEvaluationError("constraint evaluated to non-finite value", context={"constraint": constraint.name})
    return float(total)


def constraint_violation(constraint: Constraint, point: Mapping[str, Any]) -> float:
    """Return the raw mathematical violation magnitude for one constraint.

    Tolerances are intentionally *not* subtracted here.  Keeping diagnostics in
    physical/objective units avoids double-applying tolerances and lets callers
    decide whether a measured violation is acceptable.
    """
    lhs = evaluate_constraint_value(constraint, point)
    rhs = constraint.rhs
    if constraint.sense is ConstraintSense.LE:
        return max(0.0, lhs - rhs)
    if constraint.sense is ConstraintSense.GE:
        return max(0.0, rhs - lhs)
    return abs(lhs - rhs)


def domain_violation(variable: Variable, value: Any) -> float:
    """Return the raw distance from a declared scalar variable domain."""
    domain = variable.domain
    if domain.kind is DomainKind.FINITE:
        return 0.0 if value in domain.values else math.inf
    finite = to_finite_float(value)
    if finite is None:
        return math.inf

    violation = 0.0
    if domain.kind in {DomainKind.INTEGER, DomainKind.BINARY, DomainKind.SEMI_INTEGER}:
        violation = max(violation, abs(finite - round(finite)))
    if domain.kind is DomainKind.BINARY:
        violation = max(violation, min(abs(finite), abs(finite - 1.0)))

    # Semi domains are the union {0} U [lower, upper], not an ordinary
    # bounded interval. Report distance to the nearest *permitted* branch.
    # The previous max(bound_violation, integrality_violation) overreported
    # values near the isolated zero branch.
    if domain.kind is DomainKind.SEMI_CONTINUOUS:
        assert domain.lower is not None and domain.upper is not None
        interval_distance = max(0.0, domain.lower - finite, finite - domain.upper)
        return min(abs(finite), interval_distance)
    if domain.kind is DomainKind.SEMI_INTEGER:
        assert domain.lower is not None and domain.upper is not None
        first = int(domain.lower) + int(domain.strict_lower)
        last = int(domain.upper) - int(domain.strict_upper)
        if first > last:
            return abs(finite)
        nearest = min(max(round(finite), first), last)
        return min(abs(finite), abs(finite - nearest))

    if domain.lower is not None:
        if domain.strict_lower:
            if finite <= domain.lower:
                violation = max(violation, domain.lower - finite if finite < domain.lower else math.ulp(max(1.0, abs(finite))))
        else:
            violation = max(violation, domain.lower - finite)
    if domain.upper is not None:
        if domain.strict_upper:
            if finite >= domain.upper:
                violation = max(violation, finite - domain.upper if finite > domain.upper else math.ulp(max(1.0, abs(finite))))
        else:
            violation = max(violation, finite - domain.upper)
    return max(0.0, violation)


def constraint_violations(problem: OptimizationProblem, point: Mapping[str, Any]) -> dict[str, float]:
    """Return raw constraint and domain violations keyed by stable identifiers."""
    violations = {constraint.name: constraint_violation(constraint, point) for constraint in problem.constraints}
    for variable in problem.variables:
        if variable.name not in point:
            violations[f"domain:{variable.name}"] = math.inf
        else:
            violations[f"domain:{variable.name}"] = domain_violation(variable, point[variable.name])
    return violations


def max_constraint_violation(problem: OptimizationProblem, point: Mapping[str, Any] | None) -> float | None:
    """Return the maximum raw constraint/domain violation, without tolerance adjustment."""
    if point is None:
        return None
    violations = constraint_violations(problem, point)
    return max(violations.values(), default=0.0)


def feasibility_evidence(
    problem: OptimizationProblem, point: Mapping[str, Any], *,
    tolerance: float = 1e-8,
) -> tuple[float, bool]:
    """Evaluate each constraint once and return raw max violation + feasibility.

    Preserve each constraint's tolerance and strict/open domain semantics.
    Intended for expensive candidate evaluators needing *both* diagnostics.
    """
    tol = require_finite(tolerance, "tolerance")
    if tol < 0:
        raise OptimizationValidationError("feasibility tolerance must be non-negative")
    domain_values: list[float] = []
    feasible = True
    for variable in problem.variables:
        if variable.name not in point:
            return math.inf, False
        value = point[variable.name]
        violation = domain_violation(variable, value)
        domain_values.append(violation)
        if not math.isfinite(violation) or violation > tol:
            feasible = False
        numeric = to_finite_float(value)
        domain = variable.domain
        if numeric is not None and not (
            domain.kind in {DomainKind.SEMI_CONTINUOUS, DomainKind.SEMI_INTEGER}
            and numeric == 0.0
        ):
            if domain.strict_lower and domain.lower is not None and numeric <= domain.lower:
                feasible = False
            if domain.strict_upper and domain.upper is not None and numeric >= domain.upper:
                feasible = False
    residuals: list[float] = []
    for constraint in problem.constraints:
        violation = constraint_violation(constraint, point)
        residuals.append(violation)
        if not math.isfinite(violation) or violation > max(tol, constraint.tolerance):
            feasible = False
    return max((*domain_values, *residuals), default=0.0), feasible


def is_feasible(problem: OptimizationProblem, point: Mapping[str, Any], *, tolerance: float = 1e-8) -> bool:
    """Evaluate feasibility once, applying each tolerance exactly once.

    The caller tolerance governs variable-domain feasibility and provides a
    floor for constraint tolerances.  A constraint-specific tolerance may be
    larger, but is never added to the caller tolerance.
    """
    tol = require_finite(tolerance, "tolerance")
    if tol < 0:
        raise OptimizationValidationError("feasibility tolerance must be non-negative")
    for variable in problem.variables:
        if variable.name not in point:
            return False
        # A strict bound excludes its endpoint exactly. A positive numerical
        # tolerance may relax residuals, but must not legalize an excluded
        # endpoint or a point beyond an open bound.
        domain = variable.domain
        numeric = to_finite_float(point[variable.name])
        if numeric is not None and not (
            domain.kind in {DomainKind.SEMI_CONTINUOUS, DomainKind.SEMI_INTEGER}
            and numeric == 0.0
        ):
            if domain.strict_lower and domain.lower is not None and numeric <= domain.lower:
                return False
            if domain.strict_upper and domain.upper is not None and numeric >= domain.upper:
                return False
        violation = domain_violation(variable, point[variable.name])
        if not math.isfinite(violation) or violation > tol:
            return False
    for constraint in problem.constraints:
        violation = constraint_violation(constraint, point)
        allowed = max(tol, constraint.tolerance)
        if not math.isfinite(violation) or violation > allowed:
            return False
    return True


def initial_point(problem: OptimizationProblem) -> dict[str, Any]:
    point: dict[str, Any] = {}
    for variable in problem.variables:
        domain = variable.domain
        if variable.initial is not None:
            point[variable.name] = variable.validate_value(variable.initial)
        elif domain.kind is DomainKind.FINITE:
            point[variable.name] = domain.values[0]
        elif domain.kind is DomainKind.BINARY:
            point[variable.name] = 0
        elif domain.lower is not None and domain.upper is not None:
            # Halving before addition avoids overflow for large same-sign
            # finite bounds. For strict bounds, do not seed the excluded edge
            # if floating-point rounding collapses the midpoint onto it.
            midpoint = domain.lower / 2.0 + domain.upper / 2.0
            if domain.kind in {DomainKind.INTEGER, DomainKind.SEMI_INTEGER}:
                midpoint = int(round(midpoint))
                if domain.strict_lower and midpoint <= domain.lower:
                    midpoint = int(domain.lower) + 1
                if domain.strict_upper and midpoint >= domain.upper:
                    midpoint = int(domain.upper) - 1
            else:
                if domain.strict_lower and midpoint <= domain.lower:
                    midpoint = math.nextafter(domain.lower, math.inf)
                if domain.strict_upper and midpoint >= domain.upper:
                    midpoint = math.nextafter(domain.upper, -math.inf)
            point[variable.name] = variable.validate_value(midpoint)
        elif domain.lower is not None:
            candidate = max(domain.lower, 0.0)
            if domain.strict_lower and candidate <= domain.lower:
                candidate = (
                    int(domain.lower) + 1
                    if domain.kind in {DomainKind.INTEGER, DomainKind.SEMI_INTEGER}
                    else math.nextafter(domain.lower, math.inf)
                )
            point[variable.name] = variable.validate_value(candidate)
        elif domain.upper is not None:
            candidate = min(domain.upper, 0.0)
            if domain.strict_upper and candidate >= domain.upper:
                candidate = (
                    int(domain.upper) - 1
                    if domain.kind in {DomainKind.INTEGER, DomainKind.SEMI_INTEGER}
                    else math.nextafter(domain.upper, -math.inf)
                )
            point[variable.name] = variable.validate_value(candidate)
        else:
            point[variable.name] = 0.0
    return point


def vector_from_point(problem: OptimizationProblem, point: Mapping[str, Any]) -> np.ndarray:
    values: list[float] = []
    for variable in problem.variables:
        if variable.shape:
            raise UnsupportedProblemError("built-in solvers currently support scalar decision variables only", context={"variable": variable.name, "shape": variable.shape})
        if variable.name not in point:
            raise OptimizationValidationError("point is missing variable", context={"variable": variable.name})
        value = point[variable.name]
        if variable.domain.kind is DomainKind.FINITE:
            try:
                values.append(float(variable.domain.values.index(value)))
            except ValueError as exc:
                raise OptimizationValidationError("point value outside finite domain", context={"variable": variable.name}) from exc
        else:
            finite = to_finite_float(value)
            if finite is None:
                raise OptimizationValidationError("numeric variable must be finite", context={"variable": variable.name})
            values.append(finite)
    return np.asarray(values, dtype=float)


def initial_vector(problem: OptimizationProblem) -> np.ndarray:
    return vector_from_point(problem, initial_point(problem))


def point_from_vector(
    problem: OptimizationProblem,
    vector: Sequence[float],
    *,
    project: bool = True,
    integrality_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Decode a numeric backend/search vector to the problem's decision mapping.

    ``project=True`` is intended for candidate-generating search methods: values
    are clipped/rounded into their declared domains. ``project=False`` is for
    decoding a mathematical solver's *reported solution*. In that mode the
    function deliberately preserves material bound/integrality violations so
    common result validation can detect them instead of silently repairing a
    bad backend result. Tiny integer residuals within ``integrality_tolerance``
    are normalized to exact integers.
    """
    if len(vector) != len(problem.variables):
        raise OptimizationValidationError(
            "vector dimension does not match problem",
            context={"expected": len(problem.variables), "actual": len(vector)},
        )
    tolerance = require_finite(integrality_tolerance, "integrality_tolerance")
    if tolerance < 0:
        raise OptimizationValidationError("integrality_tolerance must be non-negative")

    point: dict[str, Any] = {}
    for variable, raw in zip(problem.variables, vector):
        value = require_finite(raw, variable.name)
        domain = variable.domain
        if domain.kind is DomainKind.FINITE:
            rounded = round(value)
            if project:
                index = min(max(int(rounded), 0), len(domain.values) - 1)
                point[variable.name] = domain.values[index]
            elif abs(value - rounded) <= tolerance and 0 <= int(rounded) < len(domain.values):
                point[variable.name] = domain.values[int(rounded)]
            else:
                # Built-in algebraic solvers do not advertise FINITE support;
                # retain the raw value so a misuse cannot masquerade as a valid
                # categorical solution.
                point[variable.name] = value
        elif domain.kind is DomainKind.BINARY:
            if project:
                point[variable.name] = int(value >= 0.5)
            else:
                rounded = round(value)
                point[variable.name] = int(rounded) if abs(value - rounded) <= tolerance else value
        elif domain.kind in {DomainKind.INTEGER, DomainKind.SEMI_INTEGER}:
            if project:
                candidate: float | int = int(round(value))
                if domain.kind is DomainKind.SEMI_INTEGER and abs(value) < 0.5:
                    candidate = 0
                if domain.lower is not None and candidate != 0:
                    candidate = max(candidate, int(domain.lower))
                if domain.upper is not None:
                    candidate = min(candidate, int(domain.upper))
                point[variable.name] = candidate
            else:
                rounded = round(value)
                point[variable.name] = int(rounded) if abs(value - rounded) <= tolerance else value
        elif domain.kind is DomainKind.SEMI_CONTINUOUS:
            if project:
                if abs(value) < (domain.lower or 0.0) / 2.0:
                    point[variable.name] = 0.0
                else:
                    candidate = value
                    if domain.lower is not None:
                        candidate = max(candidate, domain.lower)
                    if domain.upper is not None:
                        candidate = min(candidate, domain.upper)
                    point[variable.name] = float(candidate)
            else:
                point[variable.name] = float(value)
        else:
            if project:
                candidate = value
                if domain.lower is not None:
                    candidate = max(candidate, domain.lower)
                if domain.upper is not None:
                    candidate = min(candidate, domain.upper)
                point[variable.name] = float(candidate)
            else:
                point[variable.name] = float(value)
    return point


def scipy_bounds(problem: OptimizationProblem) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = []
    for variable in problem.variables:
        domain = variable.domain
        if domain.kind in {DomainKind.FINITE, DomainKind.CUSTOM}:
            raise UnsupportedProblemError("numeric SciPy bounds unavailable for finite/custom domain", context={"variable": variable.name})
        bounds.append((domain.lower, domain.upper))
    return bounds


def linear_objective_vector(problem: OptimizationProblem, objective: Objective | None = None) -> tuple[np.ndarray, float, float]:
    objective = objective or problem.objective
    if objective is None:
        raise UnsupportedProblemError("linear objective extraction requires exactly one objective")
    if objective.function is not None or objective.quadratic:
        raise UnsupportedProblemError("objective is not linear", context={"objective": objective.name})
    index = variable_index(problem)
    c = np.zeros(len(problem.variables), dtype=float)
    for name, coefficient in objective.linear.items():
        c[index[name]] = coefficient
    sign = 1.0 if objective.sense is ObjectiveSense.MINIMIZE else -1.0
    return sign * c, objective.constant, sign


def quadratic_objective_matrices(problem: OptimizationProblem, objective: Objective | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``Q, c, constant, sign`` for f(x)=0.5*x'Q*x+c'x+constant."""
    objective = objective or problem.objective
    if objective is None:
        raise UnsupportedProblemError("quadratic extraction requires exactly one objective")
    if objective.function is not None:
        raise UnsupportedProblemError("callable objectives cannot be extracted as explicit quadratic form")
    index = variable_index(problem)
    n = len(problem.variables)
    q = np.zeros((n, n), dtype=float)
    c = np.zeros(n, dtype=float)
    for name, coefficient in objective.linear.items():
        c[index[name]] += coefficient
    # Objective representation stores literal coefficient*x_i*x_j.  Build a
    # symmetric Hessian-form matrix satisfying 0.5*x'Q*x == the same sum.
    for (left, right), coefficient in objective.quadratic.items():
        i, j = index[left], index[right]
        if i == j:
            q[i, i] += 2.0 * coefficient
        else:
            q[i, j] += coefficient
            q[j, i] += coefficient
    sign = 1.0 if objective.sense is ObjectiveSense.MINIMIZE else -1.0
    return sign * q, sign * c, objective.constant, sign


def linear_constraint_matrices(problem: OptimizationProblem) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    index = variable_index(problem)
    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    a_eq: list[list[float]] = []
    b_eq: list[float] = []
    for constraint in problem.constraints:
        if constraint.function is not None:
            raise UnsupportedProblemError("nonlinear/callable constraint cannot be represented as linear matrix", context={"constraint": constraint.name})
        row = [0.0] * len(problem.variables)
        for name, coefficient in constraint.linear.items():
            row[index[name]] = coefficient
        if constraint.sense is ConstraintSense.LE:
            a_ub.append(row)
            b_ub.append(constraint.rhs)
        elif constraint.sense is ConstraintSense.GE:
            a_ub.append([-value for value in row])
            b_ub.append(-constraint.rhs)
        else:
            a_eq.append(row)
            b_eq.append(constraint.rhs)
    return (
        np.asarray(a_ub, dtype=float) if a_ub else None,
        np.asarray(b_ub, dtype=float) if b_ub else None,
        np.asarray(a_eq, dtype=float) if a_eq else None,
        np.asarray(b_eq, dtype=float) if b_eq else None,
    )


def is_positive_semidefinite(matrix: np.ndarray, *, tolerance: float = 1e-10) -> bool:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise OptimizationValidationError("PSD test requires square matrix")
    if not np.all(np.isfinite(matrix)):
        return False
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    return bool(np.min(eigenvalues) >= -abs(tolerance))


def classify_problem(problem: OptimizationProblem, *, validate_declared: bool = True) -> ProblemClass:
    if not problem.objectives:
        raise OptimizationValidationError("problem requires at least one objective")
    if len(problem.objectives) > 1:
        inferred = ProblemClass.MULTIOBJECTIVE
    else:
        objective = next(iter(problem.objectives), None)
        if objective is None:
            raise OptimizationValidationError("problem requires at least one objective")
        has_callable_obj = objective.function is not None
        has_quadratic = bool(objective.quadratic)
        has_callable_constraints = any(constraint.function is not None for constraint in problem.constraints)
        kinds = {variable.domain.kind for variable in problem.variables}
        any_finite = DomainKind.FINITE in kinds
        any_integer = bool(kinds & {DomainKind.INTEGER, DomainKind.BINARY, DomainKind.SEMI_INTEGER})
        any_continuous = bool(kinds & {DomainKind.REAL, DomainKind.SEMI_CONTINUOUS})
        any_mip_special = bool(kinds & {DomainKind.SEMI_CONTINUOUS, DomainKind.SEMI_INTEGER})

        if problem.problem_class is ProblemClass.BLACK_BOX or (has_callable_obj and problem.known_smooth is False):
            inferred = ProblemClass.BLACK_BOX
        elif any_finite:
            inferred = ProblemClass.CP if all(kind in {DomainKind.FINITE, DomainKind.BINARY, DomainKind.INTEGER} for kind in kinds) else ProblemClass.COMBINATORIAL
        elif has_callable_obj or has_callable_constraints:
            inferred = ProblemClass.NLP
        elif has_quadratic:
            inferred = ProblemClass.QP
        elif any_mip_special or (any_integer and any_continuous):
            inferred = ProblemClass.MILP
        elif any_integer:
            inferred = ProblemClass.IP
        else:
            inferred = ProblemClass.LP

    declared = problem.problem_class
    if validate_declared and declared not in {ProblemClass.UNKNOWN, ProblemClass.MULTIOBJECTIVE}:
        compatible_declared = declared == inferred
        # CP/combinatorial are valid deliberate formulations for any fully
        # discrete bounded decision space, even when the algebraic structure
        # could also be handed to an IP/MILP backend.
        if declared in {ProblemClass.CP, ProblemClass.COMBINATORIAL}:
            compatible_declared = all(
                variable.domain.kind in {DomainKind.INTEGER, DomainKind.BINARY, DomainKind.FINITE, DomainKind.SEMI_INTEGER}
                for variable in problem.variables
            )
        # A caller may deliberately force a bounded problem through a generic
        # black-box backend; backend compatibility will enforce finite bounds.
        if declared is ProblemClass.BLACK_BOX:
            compatible_declared = True
        if not compatible_declared:
            raise OptimizationValidationError(
                "declared problem class conflicts with mathematical structure",
                context={"declared": declared.value, "inferred": inferred.value},
            )
        return declared
    return inferred


def with_classification(problem: OptimizationProblem) -> OptimizationProblem:
    if problem.problem_class is ProblemClass.UNKNOWN:
        inferred = classify_problem(problem, validate_declared=False)
        return replace(problem, problem_class=inferred)
    classify_problem(problem, validate_declared=True)
    return problem


def problem_features(problem: OptimizationProblem) -> ProblemFeatures:
    classified = classify_problem(problem, validate_declared=True)
    counts = {kind: 0 for kind in DomainKind}
    has_unbounded = False
    warm_start = False
    for variable in problem.variables:
        counts[variable.domain.kind] += 1
        warm_start = warm_start or variable.initial is not None
        if variable.domain.kind not in {DomainKind.FINITE, DomainKind.BINARY} and (
            variable.domain.lower is None or variable.domain.upper is None
        ):
            has_unbounded = True
    return ProblemFeatures(
        problem_class=classified if not problem.is_multi_objective else ProblemClass.MULTIOBJECTIVE,
        dimensionality=len(problem.variables),
        objective_count=len(problem.objectives),
        continuous_variables=counts[DomainKind.REAL],
        integer_variables=counts[DomainKind.INTEGER],
        binary_variables=counts[DomainKind.BINARY],
        finite_domain_variables=counts[DomainKind.FINITE],
        semi_continuous_variables=counts[DomainKind.SEMI_CONTINUOUS],
        semi_integer_variables=counts[DomainKind.SEMI_INTEGER],
        custom_domain_variables=counts[DomainKind.CUSTOM],
        has_quadratic_objective=any(bool(objective.quadratic) for objective in problem.objectives),
        has_callable_objective=any(objective.function is not None for objective in problem.objectives),
        has_nonlinear_constraints=any(constraint.function is not None for constraint in problem.constraints),
        has_equality_constraints=any(constraint.sense is ConstraintSense.EQ for constraint in problem.constraints),
        has_unbounded_variables=has_unbounded,
        known_convex=problem.known_convex,
        known_smooth=problem.known_smooth,
        derivatives_available=problem.derivatives_available,
        deterministic=problem.deterministic,
        expected_evaluation_cost=problem.expected_evaluation_cost,
        warm_start_available=warm_start,
    )


def finite_difference_gradient(function: Any, x: np.ndarray, *, step: float = 1e-8) -> np.ndarray:
    """Central finite-difference gradient for smooth deterministic callbacks."""
    h = require_finite(step, "step")
    if h <= 0:
        raise OptimizationValidationError("finite-difference step must be positive")
    x = np.asarray(x, dtype=float)
    gradient = np.empty_like(x)
    for i in range(len(x)):
        delta = h * max(1.0, abs(x[i]))
        plus = x.copy(); plus[i] += delta
        minus = x.copy(); minus[i] -= delta
        fp = require_finite(function(plus), "function(x+h)")
        fm = require_finite(function(minus), "function(x-h)")
        gradient[i] = (fp - fm) / (2.0 * delta)
    return gradient


def enumerable_domain(variable: Variable, *, max_cardinality: int) -> tuple[Any, ...]:
    domain = variable.domain
    if domain.kind is DomainKind.FINITE:
        values = domain.values
    elif domain.kind is DomainKind.BINARY:
        values = (0, 1)
    elif domain.kind is DomainKind.INTEGER:
        if domain.lower is None or domain.upper is None:
            raise UnsupportedProblemError("exact finite search requires bounded integer variables", context={"variable": variable.name})
        start, stop = int(domain.lower), int(domain.upper)
        values = tuple(range(start, stop + 1))
    elif domain.kind is DomainKind.SEMI_INTEGER:
        if domain.lower is None or domain.upper is None:
            raise UnsupportedProblemError("semi-integer variable requires finite bounds", context={"variable": variable.name})
        values = (0,) + tuple(range(int(domain.lower), int(domain.upper) + 1))
    else:
        raise UnsupportedProblemError("variable domain is not finitely enumerable", context={"variable": variable.name, "kind": domain.kind.value})
    if len(values) > max_cardinality:
        raise UnsupportedProblemError(
            "finite-domain cardinality exceeds configured exact-search limit",
            context={"variable": variable.name, "cardinality": len(values), "max_cardinality": max_cardinality},
        )
    return tuple(values)


def partial_linear_constraint_possible(
    constraint: Constraint,
    assignment: Mapping[str, Any],
    remaining_domains: Mapping[str, Sequence[Any]],
) -> bool:
    """Interval-filter a linear constraint under a partial assignment."""
    if constraint.function is not None:
        return True
    minimum = 0.0
    maximum = 0.0
    for name, coefficient in constraint.linear.items():
        if name in assignment:
            value = to_finite_float(assignment[name])
            if value is None:
                return True
            minimum += coefficient * value
            maximum += coefficient * value
            continue
        domain_values = remaining_domains.get(name)
        if not domain_values:
            return True
        numeric = [to_finite_float(value) for value in domain_values]
        if any(value is None for value in numeric):
            return True
        lo = min(value for value in numeric if value is not None)
        hi = max(value for value in numeric if value is not None)
        if coefficient >= 0:
            minimum += coefficient * lo
            maximum += coefficient * hi
        else:
            minimum += coefficient * hi
            maximum += coefficient * lo
    tolerance = constraint.tolerance
    if constraint.sense is ConstraintSense.LE:
        return minimum <= constraint.rhs + tolerance
    if constraint.sense is ConstraintSense.GE:
        return maximum >= constraint.rhs - tolerance
    return minimum <= constraint.rhs + tolerance and maximum >= constraint.rhs - tolerance


def optimistic_linear_objective_bound(
    objective: Objective,
    assignment: Mapping[str, Any],
    remaining_domains: Mapping[str, Sequence[Any]],
) -> float | None:
    """Return a minimization-space optimistic bound for a linear objective."""
    if objective.function is not None or objective.quadratic:
        return None
    total = objective.constant
    for name, coefficient in objective.linear.items():
        if name in assignment:
            value = to_finite_float(assignment[name])
            if value is None:
                return None
            total += coefficient * value
            continue
        domain_values = remaining_domains.get(name)
        if not domain_values:
            return None
        numeric = [to_finite_float(value) for value in domain_values]
        if any(value is None for value in numeric):
            return None
        lo = min(value for value in numeric if value is not None)
        hi = max(value for value in numeric if value is not None)
        if objective.sense is ObjectiveSense.MINIMIZE:
            total += coefficient * (lo if coefficient >= 0 else hi)
        else:
            # We return minimization-space value (-f) for maximization.
            total += coefficient * (hi if coefficient >= 0 else lo)
    return total if objective.sense is ObjectiveSense.MINIMIZE else -total


def optimality_gaps(primal_bound: float | None, dual_bound: float | None) -> tuple[float | None, float | None]:
    return absolute_relative_gap(primal_bound, dual_bound)


__all__ = [
    "classify_problem",
    "constraint_violation",
    "constraint_violations",
    "domain_violation",
    "enumerable_domain",
    "evaluate_constraint_value",
    "evaluate_objective",
    "finite_difference_gradient",
    "initial_point",
    "initial_vector",
    "is_feasible",
    "is_positive_semidefinite",
    "linear_constraint_matrices",
    "linear_objective_vector",
    "max_constraint_violation",
    "objective_values",
    "optimistic_linear_objective_bound",
    "optimality_gaps",
    "partial_linear_constraint_possible",
    "point_from_vector",
    "problem_features",
    "quadratic_objective_matrices",
    "scipy_bounds",
    "signed_objective_value",
    "variable_index",
    "vector_from_point",
    "with_classification",
]
