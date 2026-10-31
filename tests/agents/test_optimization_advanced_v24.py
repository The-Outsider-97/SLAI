"""Regression tests for advanced general-purpose Optimization contracts.

Run with: py -m pytest tests/agents/test_optimization_advanced_v24.py -v
"""
from __future__ import annotations

import json
import math
import threading

import numpy as np
import pytest

from src.agents.optimization.domain_problems import (
    cvar_scenario_problem,
    expected_scenario_problem,
    orientation_geodesic_distance,
    orientation_path_metrics,
    sampled_trajectory_derivatives,
    trajectory_problem,
)
from src.agents.optimization.evaluation_runtime import EvaluationJournal
from src.agents.optimization.derivative_checks import verify_objective_gradient
from src.agents.optimization.optimization_math import (
    evaluate_objective,
    is_feasible,
)
from src.agents.optimization.optimization_types import (
    ConstraintSense,
    Objective,
    OptimizationProblem,
    ProblemClass,
    SolverStatus,
    TerminationReason,
    Variable,
    VariableDomain,
)
from src.agents.optimization.solvers.black_box_solver import BlackBoxSolver
from src.agents.optimization.utils.optimization_errors import OptimizationValidationError
from src.agents.spatial.modules.transform import rotation_from_axis_angle


def test_cubic_trajectory_jerk_is_exact_at_uniform_stencil():
    # For x(t)=t^3, third-order forward finite differences equal 6.
    positions = [(float(t**3),) for t in range(5)]
    velocity = sampled_trajectory_derivatives(positions, 4.0, order=1)
    acceleration = sampled_trajectory_derivatives(positions, 4.0, order=2)
    jerk = sampled_trajectory_derivatives(positions, 4.0, order=3)
    assert velocity == ((1.,), (7.,), (19.,), (37.,))
    assert acceleration == ((6.,), (12.,), (18.,))
    assert jerk == ((6.,), (6.,))
    with pytest.raises(OptimizationValidationError):
        sampled_trajectory_derivatives(positions, 4.0, order=4)


def test_time_dependent_state_constraints_preserve_sample_times():
    problem = trajectory_problem(
        "moving_boundary", start=(0., 0.), end=(4., 0.), duration=4.,
        intervals=4,
        sampled_state_constraints=[
            ("moving_wall", lambda t, pos: pos[0] - t, ConstraintSense.LE, 0.0),
        ],
    )
    assignment = {
        f"p_{i}_{j}": (float(i) if j == 0 else 0.)
        for i in range(1, 4) for j in range(2)
    }
    assert len([c for c in problem.constraints if c.name.startswith("sample:moving_wall:")]) == 5
    assert all(math.isclose(c.function(assignment), 0., abs_tol=1e-12)
               for c in problem.constraints)
    assert problem.metadata["collision_certification"] == "sampled_only"
    moving = dict(assignment, p_2_0=2.5)
    assert not is_feasible(problem, moving)


def test_quaternion_orientation_uses_spatial_so3_math():
    identity = np.eye(3)
    quarter_turn = rotation_from_axis_angle(np.array([0., 0., 1.]), math.pi / 2)
    assert orientation_geodesic_distance(identity, quarter_turn) == pytest.approx(math.pi / 2)
    metrics = orientation_path_metrics((identity, quarter_turn), duration=2.0)
    assert metrics["angular_speeds_rad_per_s"] == pytest.approx((math.pi / 4,))
    assert metrics["continuous_time_certified"] is False
    with pytest.raises(OptimizationValidationError):
        orientation_path_metrics((identity,), 1.0)


def test_expected_scenario_loss_with_scaled_probabilities():
    variables = (Variable("x", VariableDomain.real(-3., 3.)),)
    problem = expected_scenario_problem(
        "expected", variables=variables,
        scenario_losses=(lambda p: (p["x"] - 1.)**2, lambda p: (p["x"] + 1.)**2),
        probabilities=(1e308, 1e308),
    )
    assert evaluate_objective(problem.objectives[0], {"x": 0.0}) == pytest.approx(1.0)
    assert problem.metadata["scenario_probabilities"] == pytest.approx((0.5, 0.5))
    with pytest.raises(OptimizationValidationError):
        expected_scenario_problem(
            "invalid", variables=variables,
            scenario_losses=(lambda p: p["x"],), probabilities=(0.,),
        )


def test_cvar_epigraph_matches_finite_empirical_distribution():
    variable = Variable("x", VariableDomain.real(-2., 2.))
    problem = cvar_scenario_problem(
        "risk", variables=(variable,),
        scenario_losses=(lambda p: (p["x"] - 1.)**2, lambda p: (p["x"] + 1.)**2),
        confidence=0.5, probabilities=(0.5, 0.5),
    )
    # Both scenario losses are one at x=0. CVaR_0.5 is one.
    candidate = {"x": 0., "cvar_eta": 1., "cvar_excess_0": 0., "cvar_excess_1": 0.}
    assert is_feasible(problem, candidate)
    assert evaluate_objective(problem.objectives[0], candidate) == pytest.approx(1.0)
    bad = dict(candidate, cvar_eta=0.)
    assert not is_feasible(problem, bad)
    with pytest.raises(OptimizationValidationError):
        cvar_scenario_problem(
            "invalid", variables=(variable,), confidence=1.0,
            scenario_losses=(lambda p: 0.,),
        )


def test_checkpoint_save_load_rejects_wrong_identity_and_tampering(tmp_path):
    path = tmp_path / "evaluation_checkpoint.json"
    journal = EvaluationJournal(path, identity="problem-v1")
    journal.record("key", point={"x": 0.5}, signed=0.25, violation=0., failed=False)
    journal.set_progress({"x": [0.5], "mesh": 0.5, "current": 0.25, "iterations": 3})
    restored = EvaluationJournal(path, identity="problem-v1", resume=True)
    assert restored.lookup("key")["signed"] == pytest.approx(0.25)
    assert restored.progress["iterations"] == 3
    with pytest.raises(OptimizationValidationError):
        EvaluationJournal(path, identity="problem-v2", resume=True)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["entries"]["key"]["signed"] = -999
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(OptimizationValidationError):
        EvaluationJournal(path, identity="problem-v1", resume=True)


def _black_box():
    return OptimizationProblem(
        name="cancel_resume_black_box",
        variables=(Variable("x", VariableDomain.real(-2., 2.), initial=-1.),),
        objectives=(Objective("minimum", function=lambda p: (p["x"] - 0.5)**2),),
        problem_class=ProblemClass.BLACK_BOX, deterministic=True,
    )


def test_pattern_search_cooperative_cancellation(tmp_path):
    cancelled = threading.Event()
    cancelled.set()
    solver = BlackBoxSolver({"method": "pattern_search", "max_evaluations": 40})
    result = solver.solve(
        _black_box(), method="pattern_search", max_evaluations=40,
        cancel_event=cancelled,
    )
    assert result.status is SolverStatus.INTERRUPTED
    assert result.termination_reason is TerminationReason.INTERRUPTED


def test_resume_pattern_search_with_same_identity_and_larger_total_budget(tmp_path):
    checkpoint = tmp_path / "progress.json"
    solver = BlackBoxSolver({"method": "pattern_search", "max_evaluations": 80})
    kwargs = dict(method="pattern_search", checkpoint_path=checkpoint,
                  evaluator_revision="test-r1", initial_mesh=0.5)
    partial = solver.solve(_black_box(), max_evaluations=8, **kwargs)
    assert checkpoint.exists()
    assert partial.function_evaluation_count <= 8
    resumed = solver.solve(_black_box(), max_evaluations=80, resume=True, **kwargs)
    assert resumed.function_evaluation_count <= 80
    assert resumed.success
    assert abs(resumed.primal_solution["x"] - 0.5) < 0.1
    with pytest.raises(OptimizationValidationError):
        solver.solve(_black_box(), max_evaluations=80, resume=True,
                     method="pattern_search", checkpoint_path=checkpoint,
                     evaluator_revision="changed-revision")


def test_resume_is_not_claimed_for_differential_evolution(tmp_path):
    solver = BlackBoxSolver({"method": "differential_evolution"})
    from src.agents.optimization.utils.optimization_errors import UnsupportedProblemError
    with pytest.raises(UnsupportedProblemError):
        solver.solve(_black_box(), method="differential_evolution",
                     resume=True, checkpoint_path=tmp_path / "other.json",
                     evaluator_revision="v1")


def test_gradient_check_uses_stem_differentiation_and_detects_bad_gradient():
    variable = Variable("x", VariableDomain.real(-5, 5), initial=1.5)
    objective = Objective(
        "quadratic", function=lambda p: (p["x"] - 3.)**2,
        gradient=lambda p: {"x": 2. * (p["x"] - 3.)},
    )
    problem = OptimizationProblem("gradient", (variable,), (objective,))
    evidence = verify_objective_gradient(problem, {"x": 1.5})
    assert evidence["passes"]
    assert evidence["variables"]["x"]["numerical"] == pytest.approx(-3., abs=1e-4)
    bad_problem = OptimizationProblem("bad_gradient", (variable,),
        (Objective("quadratic", function=objective.function,
                   gradient=lambda p: {"x": 123.}),))
    assert not verify_objective_gradient(bad_problem, {"x": 1.5})["passes"]


def test_mixed_variable_black_box_uses_existing_canonical_solver():
    problem = OptimizationProblem(
        "mixed_variable", variables=(
            Variable("continuous", VariableDomain.real(-2, 2), initial=0.),
            Variable("integer", VariableDomain.integer(0, 3), initial=0),
            Variable("flag", VariableDomain.binary(), initial=0),
        ),
        objectives=(Objective("loss", function=lambda p:
            (p["continuous"] - 0.5)**2
            + (p["integer"] - 2)**2
            + 4.0 * (p["flag"] - 1)**2),),
        problem_class=ProblemClass.BLACK_BOX,
    )
    result = BlackBoxSolver({"method": "pattern_search", "max_evaluations": 250}).solve(
        problem, method="pattern_search", max_evaluations=250,
    )
    assert result.primal_solution is not None
    assert result.feasibility.value == "feasible"
    assert result.primal_solution["integer"] == int(result.primal_solution["integer"])
    assert result.primal_solution["flag"] in (0, 1)


def test_real_scipy_solver_on_finite_scenario_expected_loss():
    from src.agents.optimization.optimization_solver import OptimizationSolver
    from src.agents.optimization.optimization_math import with_classification
    problem = expected_scenario_problem(
        "quadratic_finite_scenarios",
        variables=(Variable("x", VariableDomain.real(-3., 3.), initial=1.),),
        scenario_losses=(lambda p: (p["x"] - 1.)**2,
                         lambda p: (p["x"] + 1.)**2),
    )
    result = OptimizationSolver(config={"use_memory_warm_start": False}).solve_single_detailed(
        with_classification(problem), solver="scipy_nonlinear",
        options={"max_iterations": 150},
    )
    assert result.solution is not None
    assert result.solution.success
    assert result.solution.primal_solution is not None
    assert result.solution.primal_solution["x"] == pytest.approx(0.0, abs=1e-3)
    assert result.solution.feasibility.value == "feasible"
