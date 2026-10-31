"""Proof-grade polynomial bounds and SciPy DE full-state resume regression.

No claims about arbitrary mesh collision or general black-box continuous time.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
import threading

import numpy as np
import pytest
import scipy

from src.agents.optimization.continuous_feasibility import (
    certify_polynomial_nonnegative,
    certify_polynomial_sphere_separation,
    certify_polynomial_trajectory_limits,
)
from src.agents.optimization.de_checkpoint import SUPPORTED_SCIPY_DE_VERSIONS
from src.agents.optimization.evaluation_runtime import EvaluationJournal
from src.agents.optimization.optimization_types import (
    Objective, OptimizationProblem, ProblemClass, SolverStatus,
    TerminationReason, Variable, VariableDomain,
)
from src.agents.optimization.solvers.black_box_solver import BlackBoxSolver
from src.agents.optimization.utils.optimization_errors import (
    OptimizationValidationError, UnsupportedProblemError,
)


def test_exact_polynomial_nonnegativity_certified_on_entire_interval():
    # (t-1/2)^2 touches zero, remains nonnegative for all 0<=t<=1.
    proof = certify_polynomial_nonnegative(("1/4", "-1", "1"), max_depth=10)
    assert proof.status == "certified_feasible"
    assert proof.feasible
    assert proof.unresolved_boxes == ()
    assert proof.guarantee == "exact_rational_polynomial_over_closed_interval"
    assert proof.certified_boxes


def test_interior_violation_is_proven_even_when_endpoints_pass():
    # 4*(t-1/2)^2-1/4 is positive at the endpoints but negative at t=1/2.
    proof = certify_polynomial_nonnegative(("3/4", "-4", "4"))
    assert proof.status == "certified_violated"
    assert Fraction(proof.witness_value) < 0
    assert Fraction(proof.witness_time) == Fraction(1, 2)


def test_non_dyadic_double_root_is_inconclusive_not_certified():
    # (t-1/3)^2 is nonnegative but the Bernstein hull of the interval
    # containing its interior root may never become positive at finite depth.
    proof = certify_polynomial_nonnegative(("1/9", "-2/3", "1"), max_depth=5)
    assert proof.status == "inconclusive"
    assert proof.unresolved_boxes
    assert not proof.feasible


def test_exact_polynomial_derivative_limits():
    # x(t)=t^2, 0<=t<=1: |v|<=2, |a|=2, jerk=0.
    proofs = certify_polynomial_trajectory_limits(
        (("0", "0", "1"),),
        max_speed=2, max_acceleration=2, max_jerk=0,
    )
    assert {key: p.status for key, p in proofs.items()} == {
        "speed": "certified_feasible",
        "acceleration": "certified_feasible",
        "jerk": "certified_feasible",
    }
    violation = certify_polynomial_trajectory_limits(
        (("0", "0", "1"),), max_speed=1,
    )
    assert violation["speed"].status == "certified_violated"


def test_continuous_moving_sphere_separation():
    safe = certify_polynomial_sphere_separation(
        center_a=(("0",), ("0",)),
        center_b=(("2", "1"), ("0",)),
        radius_a="1/2", radius_b="1/2", clearance="1/4",
    )
    assert safe.feasible
    # Overlap exists at t=1/2, though endpoints are clear.
    crossed = certify_polynomial_sphere_separation(
        center_a=(("0",), ("0",)),
        center_b=(("2", "-4"), ("0",)),
        radius_a="1/2", radius_b="1/2",
    )
    assert crossed.status == "certified_violated"
    assert Fraction(crossed.witness_time) == Fraction(1, 2)


def test_unsupported_approximate_coefficients_fail_closed():
    with pytest.raises(OptimizationValidationError):
        certify_polynomial_nonnegative((0.1, 1))
    with pytest.raises(OptimizationValidationError):
        certify_polynomial_nonnegative(("1",), start="1", end="1")
    with pytest.raises(OptimizationValidationError):
        certify_polynomial_sphere_separation(
            center_a=(("0",), ("0",)), center_b=(("1",), ("0",)),
            radius_a="-1", radius_b="1",
        )


def _problem():
    return OptimizationProblem(
        name="de_full_state_replay",
        variables=(
            Variable("x", VariableDomain.real(-2, 2), initial=-0.5),
            Variable("choice", VariableDomain.integer(-2, 2), initial=1),
        ),
        objectives=(Objective("loss", function=lambda p:
            (p["x"] - 0.63)**2 + (p["choice"] + 1)**2),),
        problem_class=ProblemClass.BLACK_BOX,
        deterministic=True,
    )


def _run_de(checkpoint: Path, *, generations: int, resume: bool = False,
            event=None, evaluator_revision="frozen-v1"):
    solver = BlackBoxSolver({
        "method": "differential_evolution", "max_evaluations": 1000,
        "seed": 1729, "popsize": 6, "tolerance": 1e-12,
    })
    return solver.solve(
        _problem(), method="differential_evolution",
        checkpoint_path=checkpoint, evaluator_revision=evaluator_revision,
        max_evaluations=1000, max_generations=generations,
        resume=resume, cancel_event=event,
    )


@pytest.mark.skipif(scipy.__version__ not in SUPPORTED_SCIPY_DE_VERSIONS,
                    reason="verified SciPy DE checkpoint backend version unavailable")
def test_de_complete_generation_resume_bitwise_matches_uninterrupted(tmp_path):
    uninterrupted = tmp_path / "uninterrupted.json"
    resumed_path = tmp_path / "resumed.json"
    full = _run_de(uninterrupted, generations=8)
    partial = _run_de(resumed_path, generations=3)
    assert partial.termination_reason in (
        TerminationReason.ITERATION_LIMIT, TerminationReason.CONVERGED,
    )
    restored = _run_de(resumed_path, generations=8, resume=True)
    a = json.loads(uninterrupted.read_text(encoding="utf-8"))
    b = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert a["progress"]["generation"] == b["progress"]["generation"]
    assert a["progress"]["population"] == b["progress"]["population"]
    assert a["progress"]["energies"] == b["progress"]["energies"]
    assert a["progress"]["rng_state"] == b["progress"]["rng_state"]
    assert a["progress"]["random_index"] == b["progress"]["random_index"]
    assert a["progress"]["nfev"] == b["progress"]["nfev"]
    assert restored.primal_solution == full.primal_solution
    assert restored.objective_values == full.objective_values
    assert restored.diagnostics["restored_evaluations"] > 0


@pytest.mark.skipif(scipy.__version__ not in SUPPORTED_SCIPY_DE_VERSIONS,
                    reason="verified SciPy DE checkpoint backend version unavailable")
def test_de_checkpoint_rejects_stale_model_revision(tmp_path):
    path = tmp_path / "evolution.json"
    _run_de(path, generations=2)
    with pytest.raises(OptimizationValidationError):
        _run_de(path, generations=6, resume=True, evaluator_revision="changed-model")


@pytest.mark.skipif(scipy.__version__ not in SUPPORTED_SCIPY_DE_VERSIONS,
                    reason="verified SciPy DE checkpoint backend version unavailable")
def test_de_resume_after_pre_cancel_is_deterministic(tmp_path):
    path = tmp_path / "cancellable.json"
    cancelled = threading.Event()
    cancelled.set()
    first = _run_de(path, generations=5, event=cancelled)
    assert first.status is SolverStatus.INTERRUPTED
    assert path.exists()
    cancelled.clear()
    resumed = _run_de(path, generations=5, resume=True, event=cancelled)
    assert resumed.function_evaluation_count > 0
    assert resumed.primal_solution is not None


def test_de_backend_version_is_fail_closed(monkeypatch, tmp_path):
    import src.agents.optimization.de_checkpoint as implementation
    monkeypatch.setattr(implementation.scipy, "__version__", "999.0.0")
    with pytest.raises(UnsupportedProblemError):
        _run_de(tmp_path / "bad_backend.json", generations=2)


def test_checkpointed_de_disallows_unrestorable_external_backend_options(tmp_path):
    if scipy.__version__ not in SUPPORTED_SCIPY_DE_VERSIONS:
        pytest.skip("verified SciPy backend unavailable")
    solver = BlackBoxSolver({"method": "differential_evolution", "max_evaluations": 100})
    with pytest.raises(UnsupportedProblemError):
        solver.solve(
            _problem(), method="differential_evolution",
            checkpoint_path=tmp_path / "no_polish.json",
            evaluator_revision="v1",
            polish=True,
        )
