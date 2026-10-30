"""Domain-independent optimization formulation regression tests."""
import math

import pytest

from src.agents.optimization.domain_problems import (
    animation_curve_problem,
    camera_look_at_basis,
    camera_pose_problem,
    geometry_parameter_problem,
    hermite_curve_value,
    trajectory_problem,
    trajectory_samples,
)
from src.agents.optimization.optimization_types import Constraint, ConstraintSense, Variable, VariableDomain
from src.agents.optimization.utils.optimization_errors import OptimizationValidationError


def test_generic_geometry_preserves_canonical_problem_and_user_constraints():
    variables = [Variable("width", domain=VariableDomain.real(1, 5), initial=2)]
    rule = Constraint("min_width", sense=ConstraintSense.GE, rhs=1, linear={"width": 1})
    p = geometry_parameter_problem("shape", variables=variables,
                                    objective=lambda x: (x["width"] - 3)**2,
                                    constraints=[rule])
    assert p.metadata["domain"] == "geometry_parameters"
    assert p.objectives[0].function({"width": 3}) == 0
    assert p.constraints == (rule,)
    with pytest.raises(OptimizationValidationError):
        geometry_parameter_problem("invalid", variables=(), objective=lambda x: 0)


def test_trajectory_linear_waypoints_and_sampled_constraints():
    p = trajectory_problem("move", start=(0,0), end=(2,0), duration=2,
                           intervals=4, max_velocity=2, max_acceleration=1, max_jerk=1)
    linear = {f"p_{i}_{j}": (0.5*i if j == 0 else 0.0)
              for i in range(1,4) for j in range(2)}
    assert trajectory_samples(linear, (0,0), (2,0), 4)[-1] == (2,0)
    assert p.objectives[0].function(linear) == pytest.approx(0, abs=1e-12)
    assert all(c.function(linear) <= c.rhs+1e-10 for c in p.constraints)
    bent = dict(linear)
    bent["p_2_1"] = 0.5
    assert p.objectives[0].function(bent) > 0
    assert p.metadata["collision_certification"] == "sampled_only"


def test_trajectory_clearance_is_external_and_explicit():
    with pytest.raises(OptimizationValidationError):
        trajectory_problem("invalid", start=(0,0), end=(1,1), duration=1,
                           minimum_clearance=0.1)
    p = trajectory_problem("obstacle", start=(0,0), end=(1,0), duration=1,
                           clearance_evaluator=lambda samples: min(abs(y-0.2) for x,y in samples),
                           minimum_clearance=0.1)
    x = {f"p_{i}_{j}": (i/8 if j == 0 else 0.0)
         for i in range(1,8) for j in range(2)}
    c = next(c for c in p.constraints if c.name == "minimum_sample_clearance")
    assert c.function(x) == pytest.approx(0.2)


def test_camera_orientation_and_degeneracy():
    right, up, forward = camera_look_at_basis((0,-5,2),(0,0,2))
    assert math.isclose(sum(a*a for a in forward), 1)
    assert math.isclose(sum(a*b for a,b in zip(right,forward)), 0, abs_tol=1e-12)
    assert math.isclose(sum(a*b for a,b in zip(up,forward)), 0, abs_tol=1e-12)
    with pytest.raises(OptimizationValidationError):
        camera_look_at_basis((0,0,0),(0,0,0))
    with pytest.raises(OptimizationValidationError):
        camera_look_at_basis((0,0,1),(0,0,0))  # forward parallel to up


def test_camera_objective_is_evaluable_and_visibility_explicit():
    p = camera_pose_problem("camera", target=(0,0,0), preferred_position=(0,-5,1),
                            desired_distance=5, minimum_visibility=0.8,
                            visibility_evaluator=lambda position: 0.9)
    x = {"x":0.0,"y":-5.0,"z":1.0}
    assert math.isfinite(p.objectives[0].function(x))
    assert p.constraints[-1].function(x) == pytest.approx(0.9)
    with pytest.raises(OptimizationValidationError):
        camera_pose_problem("bad", target=(0,0,0), preferred_position=(0,0,1),
                            desired_distance=1, minimum_visibility=0.9)


def test_animation_hermite_jerk_c2_and_monotonicity():
    p = animation_curve_problem("anim", times=(0,1,2), values=(0,1,2),
                                require_c2=True, monotonic=True)
    x = {"m_0":1.0, "m_1":1.0,"m_2":1.0}
    assert p.objectives[0].function(x) == pytest.approx(0, abs=1e-12)
    assert all(abs(c.function(x)) <= 1e-10 if c.sense is ConstraintSense.EQ
               else c.function(x) >= -1e-10 for c in p.constraints)
    assert hermite_curve_value((0,1,2),(0,1,2),(1,1,1),0.5) == pytest.approx(0.5)
    changed = dict(x)
    changed["m_1"] = -10
    assert any(c.function(changed) < -1e-6 for c in p.constraints
               if c.name.startswith("monotonic"))
    assert p.objectives[0].function(changed) > 0
    with pytest.raises(OptimizationValidationError):
        animation_curve_problem("impossible", times=(0,1,2), values=(0,2,1), monotonic=True)


def test_invalid_and_nonfinite_inputs_fail_closed():
    with pytest.raises(OptimizationValidationError):
        trajectory_problem("bad", start=(0, 0), end=(1,1), duration=float("nan"))
    with pytest.raises(OptimizationValidationError):
        trajectory_problem("bad", start=(0, 0), end=(1,1), duration=1, intervals=2)
    with pytest.raises(OptimizationValidationError):
        camera_pose_problem("bad", target=(0,0,0), preferred_position=(0,0,1),
                            desired_distance=-2)
    with pytest.raises(OptimizationValidationError):
        animation_curve_problem("bad", times=(0,0), values=(0,1))


def test_real_solver_geometry_smoke():
    """Run the existing canonical solver, rather than a new domain-specific one."""
    from src.agents.optimization.optimization_solver import OptimizationSolver
    from src.agents.optimization.optimization_math import with_classification
    p = geometry_parameter_problem(
        "fitted_width",
        variables=[Variable("width", domain=VariableDomain.real(0.1, 10), initial=1)],
        objective=lambda x: (x["width"] - 3) ** 2,
    )
    result = OptimizationSolver(config={"use_memory_warm_start": False}).solve_single_detailed(
        with_classification(p), solver="scipy_nonlinear", options={"max_iterations": 150}
    )
    assert result.solution is not None
    assert result.solution.primal_solution is not None
    assert result.solution.primal_solution["width"] == pytest.approx(3, abs=1e-3)


def test_real_solver_animation_smoke():
    from src.agents.optimization.optimization_solver import OptimizationSolver
    from src.agents.optimization.optimization_math import with_classification
    p = animation_curve_problem("minimal_jerk", times=(0, 1, 2), values=(0, 1, 2),
                                require_c2=True, monotonic=True,
                                fixed_tangents={0: 1.0, 2: 1.0})
    result = OptimizationSolver(config={"use_memory_warm_start": False}).solve_single_detailed(
        with_classification(p), solver="scipy_nonlinear", options={"max_iterations": 250}
    )
    assert result.solution is not None
    assert result.solution.primal_solution is not None
    assert result.solution.primal_solution["m_1"] == pytest.approx(1, abs=1e-3)