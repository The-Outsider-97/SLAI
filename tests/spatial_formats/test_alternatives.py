"""Deterministic design alternatives: feasible, ranked, different from each other, reproducible, and honest about being samples."""
import math

import pytest

from src.utils.spatial_formats.common import optimizer as OPT
from src.utils.spatial_formats.common.errors import ValidationError

VARS = {"w": {"min": 0.0, "max": 10.0}, "h": {"min": 0.0, "max": 10.0}}


def area(point):
    return {"objectives": [point["w"] * point["h"]], "violations": [max(0.0, (4.0 - point["w"]) / 4.0)]}          # minimise area, need w >= 4


def run(**kw):
    base = dict(senses=("min",), count=4, candidates=300, seed=3)
    base.update(kw)
    return OPT.alternatives(VARS, area, **base)


def test_every_alternative_satisfies_the_constraint_and_there_are_as_many_as_asked():
    r = run()
    assert r["status"] == "ok" and len(r["alternatives"]) == 4
    assert all(a["point"]["w"] >= 4.0 - 1e-9 and sum(a["violations"]) == 0 for a in r["alternatives"])
    assert r["proved"] is False and "not a proof" in r["note"]


def test_the_first_alternative_is_the_best_sampled_and_ranks_are_consistent():
    r = run()
    first = r["alternatives"][0]
    assert first["rank"] == 1 and first["distance"] is None
    assert all(a["objectives"][0] >= first["objectives"][0] for a in r["alternatives"])
    assert len({a["rank"] for a in r["alternatives"]}) == len(r["alternatives"])


def test_alternatives_are_spread_out_not_clustered():
    r = run(count=4, top_fraction=1.0)
    pts = [(a["point"]["w"] / 10, a["point"]["h"] / 10) for a in r["alternatives"]]
    nearest = min(math.dist(p, q) for i, p in enumerate(pts) for q in pts[i + 1:])
    assert nearest > 0.15                                                                       # four points picked by max-min distance from a 300-point sample of the unit square


def test_deterministic_for_a_seed_and_different_for_another():
    assert run()["result_sha256"] == run()["result_sha256"]
    assert run(seed=3)["alternatives"] == run(seed=3)["alternatives"]
    assert run(seed=3)["result_sha256"] != run(seed=4)["result_sha256"]


def test_choice_variables_count_as_different_and_infeasible_problems_say_so():
    r = OPT.alternatives({"kind": {"kind": "choice", "choices": ["a", "b", "c"]}, "x": {"min": 0, "max": 1}},
                         lambda p: {"objectives": [p["x"]], "violations": []}, count=3, candidates=60, seed=1, top_fraction=1.0)
    assert r["status"] == "ok" and len({a["point"]["kind"] for a in r["alternatives"]}) >= 2
    none = OPT.alternatives(VARS, lambda p: {"objectives": [1.0], "violations": [1.0]}, count=3, candidates=30)
    assert none["status"] == "infeasible" and none["alternatives"] == []


def test_fewer_feasible_designs_than_requested_is_reported_not_padded():
    r = OPT.alternatives({"n": {"kind": "integer", "min": 1, "max": 3}}, lambda p: {"objectives": [p["n"]], "violations": []}, count=6, candidates=50, seed=0)
    assert r["status"] == "fewer_than_requested" and len(r["alternatives"]) == 3                 # only three distinct integers exist


def test_a_point_the_model_cannot_build_is_dropped_and_counted():
    def flaky(p):
        if p["w"] > 5:
            raise ValueError("cannot build")
        return {"objectives": [p["w"]], "violations": []}
    r = OPT.alternatives(VARS, flaky, count=3, candidates=100, seed=2)
    assert r["failed_evaluations"] > 0 and all(a["point"]["w"] <= 5 for a in r["alternatives"])
    assert OPT.alternatives(VARS, lambda p: 1 / 0, count=2, candidates=10)["status"] == "failed"


@pytest.mark.parametrize("kw", [{"count": 0}, {"count": 21}, {"count": True}, {"candidates": 2, "count": 4}, {"top_fraction": 0}, {"top_fraction": 1.5}, {"senses": ("sideways",)}, {"weights": [-1]}])
def test_bad_requests_are_refused(kw):
    with pytest.raises(ValidationError):
        run(**kw)
