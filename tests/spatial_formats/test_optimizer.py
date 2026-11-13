"""The deterministic parametric optimizer: statuses, constraints, integers and choices, Pareto fronts, reproducibility, bad input."""
import math

import pytest

from src.utils.spatial_formats.common import optimizer as O
from src.utils.spatial_formats.common.errors import ValidationError


def test_unconstrained_quadratic_converges_near_the_known_minimum():
    r = O.optimize({"x": {"min": -5, "max": 5}, "y": {"min": -5, "max": 5}}, lambda p: {"objectives": [(p["x"] - 1.5) ** 2 + (p["y"] + 2.0) ** 2]}, budget=600, seed=3)
    assert r["status"] in ("converged", "budget_exhausted")
    assert abs(r["best"]["point"]["x"] - 1.5) < 0.05 and abs(r["best"]["point"]["y"] + 2.0) < 0.05
    assert r["best"]["objectives"][0] < 0.01 and r["proved"] is False


def test_a_constraint_is_respected_and_the_optimum_sits_on_it():
    # minimise x + y subject to x * y >= 4  (optimum x = y = 2, value 4)
    r = O.optimize({"x": {"min": 0.5, "max": 10}, "y": {"min": 0.5, "max": 10}},
                   lambda p: {"objectives": [p["x"] + p["y"]], "violations": [max(0.0, 4.0 - p["x"] * p["y"])]}, budget=2000, seed=1)
    assert r["status"] in ("converged", "budget_exhausted")
    b = r["best"]
    assert sum(b["violations"]) <= 1e-9 and b["point"]["x"] * b["point"]["y"] >= 4.0 - 1e-6
    assert b["objectives"][0] < 4.2


def test_infeasible_problems_are_reported_as_such_not_as_solutions():
    r = O.optimize({"x": {"min": 0, "max": 1}}, lambda p: {"objectives": [p["x"]], "violations": [2.0 - p["x"]]}, budget=100, seed=0)
    assert r["status"] == "infeasible" and r["feasible_found"] == 0
    assert r["best"] is not None and sum(r["best"]["violations"]) == pytest.approx(1.0, abs=0.05)   # the least-violating point, clearly flagged by the status
    assert r["proved"] is False


def test_unbounded_objective_is_flagged():
    r = O.optimize({"x": {"min": 0, "max": 1}}, lambda p: {"objectives": [-math.inf if p["x"] > 0.5 else 1.0]}, budget=60, seed=0)
    assert r["status"] == "unbounded" and r["best"] is None


def test_every_evaluation_failing_is_failed_with_the_reason():
    def boom(p):
        raise ValueError("cannot build")
    r = O.optimize({"x": {"min": 0, "max": 1}}, boom, budget=20)
    assert r["status"] == "failed" and r["failed_evaluations"] == r["evaluations"] and "cannot build" in r["history"][0]["message"]


def test_integers_and_choices():
    r = O.optimize({"n": {"kind": "integer", "min": 1, "max": 20}, "mat": {"kind": "choice", "choices": ["steel", "timber", "concrete"]}},
                   lambda p: {"objectives": [abs(p["n"] - 7) + {"steel": 3, "timber": 1, "concrete": 2}[p["mat"]]]}, budget=300, seed=5)
    assert r["best"]["point"] == {"n": 7, "mat": "timber"} and r["best"]["objectives"] == [1.0]
    assert isinstance(r["best"]["point"]["n"], (int, float)) and float(r["best"]["point"]["n"]).is_integer()


def test_pareto_front_is_non_dominated_and_complete_enough():
    # minimise x and (1-x)^2 over [0, 1]: every x is Pareto optimal
    r = O.optimize({"x": {"min": 0, "max": 1}}, lambda p: {"objectives": [p["x"], (1 - p["x"]) ** 2]}, senses=("min", "min"), budget=200, seed=2)
    front = r["pareto"]
    assert len(front) > 10
    for i, a in enumerate(front):
        for b in front[:i] + front[i + 1:]:
            assert not (all(x <= y for x, y in zip(b["objectives"], a["objectives"])) and any(x < y for x, y in zip(b["objectives"], a["objectives"])))


def test_same_seed_same_result_different_seed_may_differ_and_budget_is_a_hard_limit():
    f = lambda p: {"objectives": [math.sin(5 * p["x"]) + p["x"] ** 2]}      # noqa: E731
    a, b, c = (O.optimize({"x": {"min": -3, "max": 3}}, f, budget=150, seed=s) for s in (7, 7, 8))
    assert a["result_sha256"] == b["result_sha256"] and a["best"]["point"] == b["best"]["point"]
    assert a["evaluations"] <= 150 and c["evaluations"] <= 150
    assert a["best"]["objectives"][0] < -0.5                                  # finds a basin of this multimodal function


@pytest.mark.parametrize("bad", [dict(variables={}), dict(variables={"x": {"min": 2, "max": 1}}), dict(variables={"x": {"min": 0, "max": 1}}, budget=0),
                                 dict(variables={"x": {"min": 0, "max": 1}}, senses=("sideways",)), dict(variables={"x": {"kind": "choice", "choices": []}}),
                                 dict(variables={"x": {"min": 0, "max": 1}}, weights=[0.0])])
def test_bad_problems_are_refused(bad):
    with pytest.raises(ValidationError):
        O.optimize(bad.pop("variables"), lambda p: {"objectives": [0.0]}, **bad)


def test_time_limit_gives_a_timeout_status_and_reports_diagnostics():
    from src.utils.spatial_formats.common import optimizer as OPT
    res = OPT.optimize({"x": {"kind": "real", "min": 0, "max": 1}}, lambda p: {"objectives": [p["x"]], "violations": []}, budget=20000, seed=3, time_limit=1e-9)
    assert res["status"] == "timeout" and res["timed_out"] is True
    assert res["backend"]["name"] and res["termination"] and res["runtime_ms"] >= 0
    assert res["tolerances"]["feasibility"] == 1e-9 and res["proved"] is False


def test_no_time_limit_is_unchanged_and_the_hash_ignores_runtime():
    from src.utils.spatial_formats.common import optimizer as OPT
    f = lambda p: {"objectives": [(p["x"] - 0.3) ** 2], "violations": []}                  # noqa: E731
    a = OPT.optimize({"x": {"kind": "real", "min": 0, "max": 1}}, f, budget=200, seed=5)
    b = OPT.optimize({"x": {"kind": "real", "min": 0, "max": 1}}, f, budget=200, seed=5)
    assert a["status"] in ("converged", "budget_exhausted") and a["timed_out"] is False
    assert a["result_sha256"] == b["result_sha256"]


def test_invalid_time_limit_is_refused():
    import pytest
    from src.utils.spatial_formats.common import optimizer as OPT
    from src.utils.spatial_formats.common.errors import ValidationError
    for bad in (0, -1, True, "5", 4000):
        with pytest.raises(ValidationError):
            OPT.optimize({"x": {"kind": "real", "min": 0, "max": 1}}, lambda p: {"objectives": [0.0], "violations": []}, time_limit=bad)
