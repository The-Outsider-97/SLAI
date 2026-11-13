"""Reusable domain problem formulations for the canonical OptimizationSolver.

These are *problem builders*, not alternate optimizers.  All generated problems
use the shared Variable/Objective/Constraint/OptimizationProblem contracts and
may be solved by the existing capability registry.  Callbacks must be trusted,
deterministic when declared, and side-effect free; they are not sandboxed.

Time-dependent constraints in trajectory problems are enforced on a discrete
knot grid, not over continuous time.  Animation curves use cubic Hermite basis
functions, with optional C2 and analytically checked monotonicity.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .optimization_types import (
    Constraint, ConstraintSense, Objective, OptimizationProblem, Variable,
    VariableDomain,
)
from .utils.optimization_errors import OptimizationValidationError

__all__ = [
    "geometry_parameter_problem", "trajectory_problem", "trajectory_samples",
    "camera_pose_problem", "camera_look_at_basis", "animation_curve_problem",
    "hermite_curve_value", "robust_scenario_problem", "layout_arrangement_problem",
    "expected_scenario_problem", "cvar_scenario_problem",
    "sampled_trajectory_derivatives", "orientation_geodesic_distance", "orientation_path_metrics",
]


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise OptimizationValidationError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizationValidationError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise OptimizationValidationError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str, *, zero: bool = False) -> float:
    result = _finite(value, label)
    if result < 0 if zero else result <= 0:
        raise OptimizationValidationError(f"{label} must be {'non-negative' if zero else 'positive'}")
    return result


def _vector(values: Sequence[float], label: str, *, size: int | None = None) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise OptimizationValidationError(f"{label} must be a numeric sequence")
    result = tuple(_finite(v, f"{label}[{i}]") for i, v in enumerate(values))
    if not result or (size is not None and len(result) != size):
        raise OptimizationValidationError(f"{label} has incorrect dimension")
    return result


def _squared(values: Sequence[float]) -> float:
    return sum(v * v for v in values)


def _bounds(raw: Sequence[tuple[float | None, float | None]] | None, size: int) -> tuple[VariableDomain, ...]:
    if raw is None:
        return tuple(VariableDomain.real() for _ in range(size))
    if len(raw) != size:
        raise OptimizationValidationError("bounds must match the spatial dimension")
    result = []
    for index, pair in enumerate(raw):
        if len(pair) != 2:
            raise OptimizationValidationError(f"bounds[{index}] must be (lower, upper)")
        low = None if pair[0] is None else _finite(pair[0], f"bounds[{index}].lower")
        high = None if pair[1] is None else _finite(pair[1], f"bounds[{index}].upper")
        result.append(VariableDomain.real(low, high))
    return tuple(result)


def geometry_parameter_problem(
    name: str,
    *,
    variables: Sequence[Variable],
    objective: Callable[[Mapping[str, Any]], float],
    constraints: Sequence[Constraint] = (),
    deterministic: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> OptimizationProblem:
    """Optimize caller-owned geometric parameters without duplicating geometry.

    The objective and constraints may delegate to Spatial or SGEO via *trusted*
    callbacks.  Applications must validate and apply solution proposals; this
    builder does not edit geometry or infer collision/clearance guarantees.
    """
    if not callable(objective):
        raise OptimizationValidationError("geometry objective must be callable")
    if not variables or not all(isinstance(v, Variable) for v in variables):
        raise OptimizationValidationError("variables must be a nonempty sequence of Variable")
    if any(not isinstance(c, Constraint) for c in constraints):
        raise OptimizationValidationError("constraints must be Constraint instances")
    return OptimizationProblem(
        name=name, variables=tuple(variables),
        objectives=(Objective(name="geometric_error", function=objective),),
        constraints=tuple(constraints), known_smooth=None,
        deterministic=bool(deterministic),
        metadata={"domain": "geometry_parameters", **dict(metadata or {})},
    )


def trajectory_samples(
    assignment: Mapping[str, Any], start: Sequence[float], end: Sequence[float],
    intervals: int,
) -> tuple[tuple[float, ...], ...]:
    """Decode interior waypoints. The fixed endpoints are not decision variables."""
    first = _vector(start, "start")
    last = _vector(end, "end", size=len(first))
    if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 1:
        raise OptimizationValidationError("intervals must be a positive integer")
    middle = tuple(
        tuple(_finite(assignment[f"p_{i}_{axis}"], f"p_{i}_{axis}") for axis in range(len(first)))
        for i in range(1, intervals)
    )
    return (first, *middle, last)


def sampled_trajectory_derivatives(
    positions: Sequence[Sequence[float]], duration: float, *, order: int = 1,
) -> tuple[tuple[float, ...], ...]:
    """Consistent forward differences for a *uniformly sampled* trajectory.

    Exact for polynomials of degree <= order at the corresponding stencil;
    no assertion about derivatives between samples. Repeated differencing
    amplifies observation noise, especially for jerk.
    """
    values = tuple(_vector(p, f"positions[{i}]") for i, p in enumerate(positions))
    if len(values) < 2 or any(len(v) != len(values[0]) for v in values):
        raise OptimizationValidationError("trajectory samples must share a dimension")
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2, 3):
        raise OptimizationValidationError("derivative order must be 1, 2, or 3")
    if len(values) <= order:
        raise OptimizationValidationError("too few trajectory samples for derivative order")
    step = _positive(duration, "duration") / (len(values) - 1)
    sequence = values
    for _ in range(order):
        sequence = tuple(
            tuple((sequence[i + 1][j] - sequence[i][j]) / step for j in range(len(sequence[0])))
            for i in range(len(sequence) - 1)
        )
    return sequence


def trajectory_problem(
    name: str,
    *,
    start: Sequence[float], end: Sequence[float], duration: float,
    intervals: int = 8,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    acceleration_weight: float = 1.0,
    jerk_weight: float = 1.0,
    max_velocity: float | None = None,
    max_acceleration: float | None = None,
    max_jerk: float | None = None,
    initial_velocity: Sequence[float] | None = None,
    final_velocity: Sequence[float] | None = None,
    initial_acceleration: Sequence[float] | None = None,
    final_acceleration: Sequence[float] | None = None,
    clearance_evaluator: Callable[[tuple[tuple[float, ...], ...]], float] | None = None,
    minimum_clearance: float | None = None,
    sampled_state_constraints: Sequence[
        tuple[str, Callable[[float, tuple[float, ...]], float], ConstraintSense, float]
    ] = (),
) -> OptimizationProblem:
    """Knot-discretized Cartesian trajectory optimization in SI units.

    Outputs only optimized intermediate waypoints. Velocity/acceleration/jerk
    bounds constrain finite differences at knots; they do *not* certify
    continuous-time feasibility or swept-volume collision freedom.
    """
    a = _vector(start, "start")
    b = _vector(end, "end", size=len(a))
    if len(a) not in (2, 3):
        raise OptimizationValidationError("trajectory dimension must be 2 or 3")
    if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 3:
        raise OptimizationValidationError("intervals must be an integer >= 3 for jerk evaluation")
    h = _positive(duration, "duration") / intervals
    aw = _positive(acceleration_weight, "acceleration_weight", zero=True)
    jw = _positive(jerk_weight, "jerk_weight", zero=True)
    if aw == 0 and jw == 0:
        raise OptimizationValidationError("at least one trajectory objective weight must be positive")
    domains = _bounds(bounds, len(a))
    variables = tuple(
        Variable(f"p_{i}_{j}", domain=domains[j], initial=a[j] + (b[j] - a[j]) * i / intervals)
        for i in range(1, intervals) for j in range(len(a))
    )
    def points(x: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
        return trajectory_samples(x, a, b, intervals)
    def derivative_norms(x: Mapping[str, Any], order: int) -> tuple[float, ...]:
        sequence = sampled_trajectory_derivatives(points(x), duration, order=order)
        return tuple(math.sqrt(_squared(v)) for v in sequence)
    def cost(x: Mapping[str, Any]) -> float:
        return h * (aw * sum(v * v for v in derivative_norms(x, 2))
                    + jw * sum(v * v for v in derivative_norms(x, 3)))
    constraints: list[Constraint] = []
    for order, limit, label in (
        (1, max_velocity, "max_velocity"),
        (2, max_acceleration, "max_acceleration"),
        (3, max_jerk, "max_jerk"),
    ):
        if limit is not None:
            bound = _positive(limit, label)
            constraints.append(Constraint(name=label, sense=ConstraintSense.LE, rhs=bound,
                                          function=lambda x, o=order: max(derivative_norms(x, o))))
    for order, initial, final, label in (
        (1, initial_velocity, final_velocity, "velocity"),
        (2, initial_acceleration, final_acceleration, "acceleration"),
    ):
        for which, desired in (("initial", initial), ("final", final)):
            if desired is None:
                continue
            target = _vector(desired, f"{which}_{label}", size=len(a))
            def residual(x: Mapping[str, Any], *, d=order, side=which,
                         expected=target) -> float:
                seq = points(x)
                for _ in range(d):
                    seq = tuple(tuple((seq[i+1][k] - seq[i][k]) / h
                                      for k in range(len(a))) for i in range(len(seq)-1))
                actual = seq[0] if side == "initial" else seq[-1]
                return math.sqrt(_squared(_subtract(actual, expected)))
            constraints.append(Constraint(name=f"{which}_{label}",
                                          sense=ConstraintSense.EQ, rhs=0.0,
                                          function=residual))
    # The caller supplies state-dependent restrictions at each sample time.
    # This supports time-varying limits and moving obstacles *at knots only*.
    # Every callable is trusted; no continuous-time guarantee is inferred.
    names = {rule.name for rule in constraints}
    for item in sampled_state_constraints:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            raise OptimizationValidationError("sampled state constraint must be (name, evaluator, sense, rhs)")
        label, evaluator, sense, rhs_raw = item
        if not isinstance(label, str) or not label.strip() or not callable(evaluator):
            raise OptimizationValidationError("sampled state constraint needs a name and callable")
        if not isinstance(sense, ConstraintSense):
            raise OptimizationValidationError("sampled state constraint sense must be ConstraintSense")
        rhs = _finite(rhs_raw, f"{label}.rhs")
        for i in range(intervals + 1):
            identifier = f"sample:{label}:{i}"
            if identifier in names:
                raise OptimizationValidationError("duplicate sampled-state constraint name")
            names.add(identifier)
            at = i * h
            constraints.append(Constraint(
                name=identifier, sense=sense, rhs=rhs,
                function=lambda x, idx=i, t=at, fn=evaluator: _finite(
                    fn(t, points(x)[idx]), "sampled_state_constraint"
                ),
                metadata={"time_seconds": at, "sample_index": i, "sampled_only": True},
            ))
    if minimum_clearance is not None:
        min_clearance = _positive(minimum_clearance, "minimum_clearance", zero=True)
        if not callable(clearance_evaluator):
            raise OptimizationValidationError("minimum_clearance requires a clearance_evaluator")
        constraints.append(Constraint(name="minimum_sample_clearance", sense=ConstraintSense.GE,
                                      rhs=min_clearance,
                                      function=lambda x: _finite(clearance_evaluator(points(x)), "clearance")))
    elif clearance_evaluator is not None:
        raise OptimizationValidationError("clearance_evaluator requires minimum_clearance")
    return OptimizationProblem(
        name=name, variables=variables,
        objectives=(Objective(name="sampled_motion_smoothness", function=cost),),
        constraints=tuple(constraints), deterministic=True,
        metadata={"domain": "trajectory", "dimension": len(a), "intervals": intervals,
                  "duration_seconds": h * intervals, "time_step_seconds": h,
                  "collision_certification": "sampled_only",
                  "sampled_state_constraint_count": len(sampled_state_constraints)},
    )


def _subtract(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(a[i] - b[i] for i in range(len(a)))


def _normalize(v: Sequence[float], label: str) -> tuple[float, ...]:
    norm = math.sqrt(_squared(v))
    if norm <= 1e-12:
        raise OptimizationValidationError(f"{label} is degenerate")
    return tuple(c / norm for c in v)


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def camera_look_at_basis(
    position: Sequence[float], target: Sequence[float],
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """(right, corrected_up, forward) in a right-handed world convention.

    Explicitly rejects coincident camera/target and forward/up parallelism.
    No Euler-angle interpolation or renderer is implied.
    """
    p = _vector(position, "position", size=3)
    t = _vector(target, "target", size=3)
    axis_up = _normalize(_vector(up, "up", size=3), "up")
    forward = _normalize(_subtract(t, p), "camera view direction")
    right = _normalize(_cross(forward, axis_up), "camera right axis")
    correct_up = _normalize(_cross(right, forward), "camera up axis")
    return right, correct_up, forward


def camera_pose_problem(
    name: str,
    *,
    target: Sequence[float], preferred_position: Sequence[float],
    desired_distance: float,
    position_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    distance_weight: float = 1.0,
    position_weight: float = 0.25,
    minimum_distance: float = 1e-5,
    visibility_evaluator: Callable[[tuple[float, float, float]], float] | None = None,
    minimum_visibility: float | None = None,
) -> OptimizationProblem:
    """Optimize a look-at camera position; visibility requires caller evaluator.

    Camera orientation is derived deterministically from the look-at target.
    This does not model lens distortion, occlusion, or FOV automatically.
    """
    t = _vector(target, "target", size=3)
    preferred = _vector(preferred_position, "preferred_position", size=3)
    desired = _positive(desired_distance, "desired_distance")
    clearance = _positive(minimum_distance, "minimum_distance")
    dw = _positive(distance_weight, "distance_weight", zero=True)
    pw = _positive(position_weight, "position_weight", zero=True)
    if dw == 0 and pw == 0:
        raise OptimizationValidationError("at least one camera objective weight must be positive")
    domains = _bounds(position_bounds, 3)
    variables = tuple(Variable(axis, domain=domain, initial=initial)
                      for axis, domain, initial in zip(("x", "y", "z"), domains, preferred))
    def position(x: Mapping[str, Any]) -> tuple[float, float, float]:
        return (_finite(x["x"], "x"), _finite(x["y"], "y"), _finite(x["z"], "z"))
    def radius(x: Mapping[str, Any]) -> float:
        return math.sqrt(_squared(_subtract(position(x), t)))
    def objective(x: Mapping[str, Any]) -> float:
        p = position(x)
        return dw * (radius(x) - desired) ** 2 + pw * _squared(_subtract(p, preferred))
    constraints: list[Constraint] = [
        Constraint(name="camera_target_separation", sense=ConstraintSense.GE,
                   rhs=clearance, function=radius),
    ]
    if minimum_visibility is not None:
        threshold = _finite(minimum_visibility, "minimum_visibility")
        if not 0 <= threshold <= 1 or not callable(visibility_evaluator):
            raise OptimizationValidationError("visibility threshold must be [0,1] with an evaluator")
        constraints.append(Constraint(
            name="minimum_visibility", sense=ConstraintSense.GE, rhs=threshold,
            function=lambda x: _finite(visibility_evaluator(position(x)), "visibility"),
        ))
    elif visibility_evaluator is not None:
        raise OptimizationValidationError("visibility_evaluator requires minimum_visibility")
    return OptimizationProblem(
        name=name, variables=variables,
        objectives=(Objective(name="camera_framing_proxy", function=objective),),
        constraints=tuple(constraints), deterministic=True,
        metadata={"domain": "camera_pose", "convention": "look_at_right_handed",
                  "visibility": "external_evaluator" if visibility_evaluator else "not_evaluated"},
    )


def _hermite_coefficients(y0: float, y1: float, m0: float, m1: float, h: float) -> tuple[float, float, float, float]:
    a3 = 2*(y0-y1) + h*(m0+m1)
    a2 = 3*(y1-y0) - h*(2*m0+m1)
    return y0, h*m0, a2, a3


def hermite_curve_value(times: Sequence[float], values: Sequence[float], tangents: Sequence[float], t: float) -> float:
    """Evaluate scalar cubic Hermite interpolation using time-unit derivatives."""
    ts = _vector(times, "times")
    vs = _vector(values, "values", size=len(ts))
    ms = _vector(tangents, "tangents", size=len(ts))
    if len(ts) < 2 or any(ts[i] >= ts[i+1] for i in range(len(ts)-1)):
        raise OptimizationValidationError("times must be strictly increasing")
    at = _finite(t, "t")
    if not ts[0] <= at <= ts[-1]:
        raise OptimizationValidationError("t is outside the curve's defined time interval")
    import bisect
    segment = min(bisect.bisect_right(ts, at) - 1, len(ts) - 2)
    h = ts[segment+1]-ts[segment]
    u = (at - ts[segment])/h
    a0, a1, a2, a3 = _hermite_coefficients(vs[segment], vs[segment+1], ms[segment], ms[segment+1], h)
    return ((a3*u + a2)*u + a1)*u + a0


def animation_curve_problem(
    name: str,
    *,
    times: Sequence[float], values: Sequence[float],
    fixed_tangents: Mapping[int, float] | None = None,
    max_abs_slope: float | None = None,
    require_c2: bool = False,
    monotonic: bool = False,
    tangent_regularization: float = 1e-6,
) -> OptimizationProblem:
    """Optimize cubic Hermite tangents with exact integrated squared jerk.

    Knot values and times are fixed. A single tangent per knot guarantees C1.
    Optional C2 equations are exact within the piecewise cubic representation.
    Exact analytic derivative minima constrain per-segment monotonicity, not
    sampled monotonicity. Distinct keyframe values are not optimized here.
    """
    ts = _vector(times, "times")
    vs = _vector(values, "values", size=len(ts))
    if len(ts) < 2 or any(ts[i] >= ts[i+1] for i in range(len(ts)-1)):
        raise OptimizationValidationError("times must contain >= 2 strictly increasing values")
    if not isinstance(require_c2, bool) or not isinstance(monotonic, bool):
        raise OptimizationValidationError("continuity and monotonic flags must be booleans")
    regularization = _positive(tangent_regularization, "tangent_regularization", zero=True)
    limit = None if max_abs_slope is None else _positive(max_abs_slope, "max_abs_slope")
    prescribed: dict[int, float] = {}
    for i, value in (fixed_tangents or {}).items():
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(ts):
            raise OptimizationValidationError("fixed_tangents indices must refer to existing knots")
        prescribed[i] = _finite(value, f"fixed_tangents[{i}]")
    slopes = tuple((vs[i+1]-vs[i])/(ts[i+1]-ts[i]) for i in range(len(ts)-1))
    estimates = tuple(slopes[0] if i==0 else slopes[-1] if i==len(ts)-1
                      else (slopes[i-1]+slopes[i])/2 for i in range(len(ts)))
    variables: list[Variable] = []
    for i, estimate in enumerate(estimates):
        if i in prescribed:
            if limit is not None and abs(prescribed[i]) > limit:
                raise OptimizationValidationError("fixed tangent violates max_abs_slope")
            d = VariableDomain.real(prescribed[i], prescribed[i])
            initial = prescribed[i]
        else:
            d = VariableDomain.real(-limit, limit) if limit is not None else VariableDomain.real()
            initial = max(-limit, min(limit, estimate)) if limit is not None else estimate
        variables.append(Variable(f"m_{i}", domain=d, initial=initial))
    def tangents(x: Mapping[str, Any]) -> tuple[float, ...]:
        return tuple(_finite(x[f"m_{i}"], f"m_{i}") for i in range(len(ts)))
    def coeffs(x: Mapping[str, Any], i: int) -> tuple[float, float, float, float]:
        ms = tangents(x)
        return _hermite_coefficients(vs[i], vs[i+1], ms[i], ms[i+1], ts[i+1]-ts[i])
    def cost(x: Mapping[str, Any]) -> float:
        ms = tangents(x)
        total = regularization * sum((m-r)**2 for m, r in zip(ms, estimates))
        for i in range(len(ts)-1):
            h = ts[i+1]-ts[i]
            a3 = coeffs(x, i)[3]
            total += 36.0*a3*a3/(h**5)  # Integral of jerk^2 on this interval.
        return total
    constraints: list[Constraint] = []
    if require_c2:
        for i in range(1, len(ts)-1):
            def acceleration_jump(x: Mapping[str, Any], knot: int=i) -> float:
                _, _, a2l, a3l = coeffs(x, knot-1)
                _, _, a2r, _ = coeffs(x, knot)
                hl = ts[knot]-ts[knot-1]
                hr = ts[knot+1]-ts[knot]
                return (6*a3l+2*a2l)/(hl*hl) - 2*a2r/(hr*hr)
            constraints.append(Constraint(name=f"C2_at_{i}", sense=ConstraintSense.EQ,
                                          function=acceleration_jump, rhs=0.0))
    if monotonic:
        changes = [vs[i+1]-vs[i] for i in range(len(ts)-1)]
        increasing = all(d>=0 for d in changes)
        decreasing = all(d<=0 for d in changes)
        if not (increasing or decreasing):
            raise OptimizationValidationError("nonmonotonic knot values cannot admit a monotonic interpolant")
        direction = 1.0 if increasing else -1.0
        for i in range(len(ts)-1):
            def minimum_signed_derivative(x: Mapping[str, Any], segment: int=i) -> float:
                _, a1, a2, a3 = coeffs(x, segment)
                h = ts[segment+1]-ts[segment]
                candidates = [direction*a1/h, direction*(a1+2*a2+3*a3)/h]
                # Derivative is quadratic, so its minimum is at an endpoint
                # or the sole stationary point inside [0,1].
                if abs(a3)>1e-15:
                    u = -a2/(3*a3)
                    if 0 < u < 1:
                        candidates.append(direction*(a1+2*a2*u+3*a3*u*u)/h)
                return min(candidates)
            constraints.append(Constraint(name=f"monotonic_segment_{i}",
                                          sense=ConstraintSense.GE, rhs=0.0,
                                          function=minimum_signed_derivative))
    return OptimizationProblem(
        name=name, variables=tuple(variables),
        objectives=(Objective(name="integrated_squared_jerk", function=cost),),
        constraints=tuple(constraints), deterministic=True,
        metadata={"domain": "animation_curve", "basis": "cubic_hermite",
                  "continuity": "C2" if require_c2 else "C1",
                  "monotonicity": "analytic" if monotonic else "unconstrained",
                  "times": ts, "values": vs},
    )


def robust_scenario_problem(
    name: str,
    *,
    variables: Sequence[Variable],
    scenario_losses: Sequence[Callable[[Mapping[str, Any]], float]],
    auxiliary_variable: str = "worst_case_loss",
    metadata: Mapping[str, Any] | None = None,
) -> OptimizationProblem:
    """Finite-scenario minimax formulation with a reusable epigraph variable.

    Minimize z subject to loss_i(x) <= z for every supplied scenario.
    This is *not* distributional robustness or a probabilistic chance
    constraint; its guarantee covers only the supplied finite scenarios.
    External callbacks must be trusted and evaluation-pure.
    """
    if not variables or any(not isinstance(v, Variable) for v in variables):
        raise OptimizationValidationError("variables must be a nonempty sequence of Variable")
    if not scenario_losses or any(not callable(cb) for cb in scenario_losses):
        raise OptimizationValidationError("scenario_losses must be a nonempty sequence of callables")
    if not auxiliary_variable or auxiliary_variable in {v.name for v in variables}:
        raise OptimizationValidationError("auxiliary_variable must be unique and nonempty")
    aux = Variable(auxiliary_variable, domain=VariableDomain.real(), initial=0.0)
    rules = tuple(Constraint(
        name=f"scenario_{i}_epigraph", sense=ConstraintSense.LE, rhs=0.0,
        function=lambda x, callback=callback: _finite(callback(x), "scenario_loss")
        - _finite(x[auxiliary_variable], "epigraph"),
    ) for i, callback in enumerate(scenario_losses))
    return OptimizationProblem(
        name=name, variables=(*variables, aux),
        objectives=(Objective(name="worst_case_scenario_loss",
                              function=lambda x: _finite(x[auxiliary_variable], "epigraph")),),
        constraints=rules, deterministic=True,
        metadata={"domain": "finite_scenario_robust", "scenario_count": len(rules),
                  "guarantee": "enumerated_scenarios_only", **dict(metadata or {})},
    )


def layout_arrangement_problem(
    name: str,
    *,
    preferred_positions: Sequence[Sequence[float]],
    minimum_separation: float,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    radii: Sequence[float] | None = None,
) -> OptimizationProblem:
    """Generic finite-point layout optimization using distance *proxies*.

    No polygon containment, shape collision, packing optimality, or physical
    interpenetration guarantees are implied. Consumers must independently
    validate native geometry before applying candidate positions.
    """
    if not 2 <= len(preferred_positions) <= 64:
        raise OptimizationValidationError("layout requires 2..64 points")
    centers = tuple(_vector(p, f"preferred_positions[{i}]") for i, p in enumerate(preferred_positions))
    dims = len(centers[0])
    if dims not in (2, 3) or any(len(p) != dims for p in centers):
        raise OptimizationValidationError("layout points must all be 2D or 3D")
    sep = _positive(minimum_separation, "minimum_separation", zero=True)
    domain = _bounds(bounds, dims)
    if radii is None:
        radius = (0.0,) * len(centers)
    else:
        if len(radii) != len(centers):
            raise OptimizationValidationError("radii length must match positions")
        radius = tuple(_positive(r, f"radius[{i}]", zero=True) for i, r in enumerate(radii))
    variables = tuple(
        Variable(f"position_{i}_{d}", domain=domain[d], initial=centers[i][d])
        for i in range(len(centers)) for d in range(dims)
    )
    def position(x: Mapping[str, Any], i: int) -> tuple[float, ...]:
        return tuple(_finite(x[f"position_{i}_{d}"], f"position_{i}_{d}")
                     for d in range(dims))
    def cost(x: Mapping[str, Any]) -> float:
        return sum(_squared(_subtract(position(x, i), center))
                   for i, center in enumerate(centers))
    rules = tuple(Constraint(
        name=f"pair_{i}_{j}_distance", sense=ConstraintSense.GE,
        rhs=sep + radius[i] + radius[j],
        function=lambda x, a=i, b=j: math.sqrt(
            _squared(_subtract(position(x, a), position(x, b)))),
    ) for i in range(len(centers)) for j in range(i+1, len(centers)))
    return OptimizationProblem(
        name=name, variables=variables,
        objectives=(Objective(name="squared_layout_displacement", function=cost),),
        constraints=rules, deterministic=True,
        metadata={"domain": "layout", "objects": len(centers), "dimension": dims,
                  "collision_guarantee": "distance_proxy_only"},
    )

def _scenario_distribution(
    losses: Sequence[Callable[[Mapping[str, Any]], float]],
    probabilities: Sequence[float] | None,
) -> tuple[tuple[Callable[[Mapping[str, Any]], float], ...], tuple[float, ...]]:
    """Normalize an explicit finite probability distribution, not a sampler."""
    if not losses or any(not callable(fn) for fn in losses):
        raise OptimizationValidationError("scenario losses require nonempty trusted callbacks")
    weights = (tuple(1.0 for _ in losses) if probabilities is None
               else tuple(_positive(v, "probability", zero=True) for v in probabilities))
    if len(weights) != len(losses) or not any(v > 0 for v in weights):
        raise OptimizationValidationError("scenario probabilities must match losses and have positive mass")
    scale = max(weights)
    from math import fsum
    scaled = tuple(v / scale for v in weights)
    denom = fsum(scaled)
    return tuple(losses), tuple(v / denom for v in scaled)


def expected_scenario_problem(
    name: str, *, variables: Sequence[Variable],
    scenario_losses: Sequence[Callable[[Mapping[str, Any]], float]],
    probabilities: Sequence[float] | None = None,
    constraints: Sequence[Constraint] = (),
) -> OptimizationProblem:
    """Finite-distribution expected loss (exact for listed scenarios only).

    Stochastic generators, empirical calibration and uncertainty distributions
    remain caller-owned. No Monte-Carlo confidence guarantee is asserted.
    """
    callbacks, weights = _scenario_distribution(scenario_losses, probabilities)
    if not variables or any(not isinstance(x, Variable) for x in variables):
        raise OptimizationValidationError("variables must be nonempty Variables")
    if any(not isinstance(x, Constraint) for x in constraints):
        raise OptimizationValidationError("constraints must be canonical Constraint objects")
    def expected(point: Mapping[str, Any]) -> float:
        return math.fsum(w * _finite(fn(point), "scenario loss")
                         for fn, w in zip(callbacks, weights))
    return OptimizationProblem(
        name=name, variables=tuple(variables),
        objectives=(Objective(name="expected_scenario_loss", function=expected),),
        constraints=tuple(constraints), deterministic=True,
        metadata={"domain": "finite_scenario_expectation",
                  "scenario_count": len(callbacks),
                  "scenario_probabilities": weights,
                  "guarantee": "enumerated_scenarios_only"},
    )


def cvar_scenario_problem(
    name: str, *, variables: Sequence[Variable],
    scenario_losses: Sequence[Callable[[Mapping[str, Any]], float]],
    confidence: float = 0.95,
    probabilities: Sequence[float] | None = None,
    constraints: Sequence[Constraint] = (),
) -> OptimizationProblem:
    """Finite-scenario CVaR minimization via Rockafellar-Uryasev epigraph.

    Minimize eta + sum(p_i * slack_i)/(1-alpha), subject to
    slack_i >= loss_i(x) - eta and slack_i >= 0.
    The result addresses the enumerated empirical distribution only.
    """
    alpha = _finite(confidence, "confidence")
    if not 0.0 < alpha < 1.0:
        raise OptimizationValidationError("confidence must be strictly between 0 and 1")
    callbacks, weights = _scenario_distribution(scenario_losses, probabilities)
    if not variables or any(not isinstance(x, Variable) for x in variables):
        raise OptimizationValidationError("variables must be nonempty Variables")
    if any(not isinstance(x, Constraint) for x in constraints):
        raise OptimizationValidationError("constraints must be canonical Constraint objects")
    used = {v.name for v in variables}
    aux_names = ["cvar_eta"] + [f"cvar_excess_{i}" for i in range(len(callbacks))]
    if any(n in used for n in aux_names):
        raise OptimizationValidationError("CVaR auxiliary variable collides with application variable")
    aux = [Variable("cvar_eta", VariableDomain.real(), initial=0.0)]
    aux.extend(Variable(name, VariableDomain.nonnegative(), initial=0.0)
               for name in aux_names[1:])
    def cost(point: Mapping[str, Any]) -> float:
        return _finite(point["cvar_eta"], "eta") + math.fsum(
            w * _finite(point[aux_names[i + 1]], "excess") / (1.0 - alpha)
            for i, w in enumerate(weights)
        )
    risk_constraints = [
        Constraint(
            name=f"cvar_scenario_{i}",
            sense=ConstraintSense.LE, rhs=0.0,
            function=lambda point, fn=fn, slack=aux_names[i + 1]:
                _finite(fn(point), "scenario loss")
                - _finite(point["cvar_eta"], "eta")
                - _finite(point[slack], "excess"),
        ) for i, fn in enumerate(callbacks)
    ]
    return OptimizationProblem(
        name=name, variables=(*variables, *aux),
        objectives=(Objective(name="scenario_cvar", function=cost),),
        constraints=(*constraints, *risk_constraints),
        deterministic=True,
        metadata={"domain": "finite_scenario_cvar",
                  "scenario_count": len(callbacks), "confidence": alpha,
                  "scenario_probabilities": weights,
                  "guarantee": "enumerated_scenarios_only"},
    )


def orientation_geodesic_distance(
    first: Any, second: Any,
) -> float:
    """Shortest relative rotation angle in radians, with Spatial SO(3) semantics.

    Uses the authoritative Spatial transform utilities; Optimization does not
    maintain another quaternion, Euler or rotation-matrix implementation.
    """
    import numpy as np
    from src.agents.spatial.modules.transform import validate_rotation_matrix, so3_log

    left = validate_rotation_matrix(first)
    right = validate_rotation_matrix(second)
    return float(np.linalg.norm(so3_log(left.T @ right)))


def orientation_path_metrics(
    rotations: Sequence[Any], duration: float,
) -> Mapping[str, Any]:
    """Sampled SO(3) geodesic angles and angular *speed magnitudes*.

    This is not angular acceleration or jerk. Subtracting instantaneous
    angular velocities requires transporting them to a common frame.
    It does not certify any between-sample angular-rate limit.
    """
    samples = tuple(rotations)
    if len(samples) < 2:
        raise OptimizationValidationError("orientation path needs at least two rotations")
    interval = _positive(duration, "duration") / (len(samples) - 1)
    angles = tuple(orientation_geodesic_distance(a, b)
                   for a, b in zip(samples[:-1], samples[1:]))
    return {
        "step_angles_rad": angles,
        "angular_speeds_rad_per_s": tuple(angle / interval for angle in angles),
        "maximum_sampled_angular_speed_rad_per_s": max(angles) / interval,
        "total_geodesic_angle_rad": math.fsum(angles),
        "continuous_time_certified": False,
    }
