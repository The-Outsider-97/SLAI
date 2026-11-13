"""The Spatial Agent's analyze_constraints operation: solving and degrees-of-freedom reporting through the agent's own request/response contract."""
import pytest

pytest.importorskip("numpy")

from src.agents.spatial_agent import SpatialAgent
from src.agents.spatial.utils.spatial_errors import SpatialValidationError


def seg(x0, y0, x1, y1):
    return {"kind": "segment", "coords": [x0, y0, x1, y1]}


def quad():
    ents = {"a": seg(0, 0, 4.2, 0.3), "b": seg(4.0, 0.1, 4.1, 3.2), "c": seg(4.1, 3.0, 0.2, 3.1), "d": seg(0.1, 3.0, 0.0, 0.2)}
    cons = [{"kind": "coincident", "refs": [{"entity": p, "part": "end"}, {"entity": q, "part": "start"}]} for p, q in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))]
    return ents, cons


@pytest.fixture()
def agent():
    return SpatialAgent(None, None)


def ask(agent, **payload):
    return agent.perform_task({"operation": "analyze_constraints", **payload})


def test_the_operation_is_advertised(agent):
    assert "analyze_constraints" in agent.capabilities()["operations"]


def test_a_closed_chain_reports_its_remaining_freedom_and_does_not_change_the_input(agent):
    ents, cons = quad()
    before = repr(ents)
    out = ask(agent, entities=ents, constraints=cons)
    assert out["status"] == "ok" and out["operation"] == "analyze_constraints"
    result = out["result"]
    assert result["status"] == "under_constrained" and result["dof"] > 0
    assert repr(ents) == before


def test_a_conflict_is_named_not_hidden(agent):
    ents = {"p": {"kind": "point", "coords": [0.0, 0.0]}, "q": {"kind": "point", "coords": [3.0, 0.0]}}
    cons = [{"kind": "fixed", "refs": ["p"], "value": [0, 0]}, {"kind": "fixed", "refs": ["q"], "value": [3, 0]},
            {"kind": "distance", "refs": ["p", "q"], "value": 5.0}]
    result = ask(agent, entities=ents, constraints=cons)["result"]
    assert result["status"] == "conflicting" and result["conflicting"] == [0, 1, 2]       # the three constraints that cannot all hold, by index


def test_a_missing_reference_is_reported_as_unresolved(agent):
    ents = {"p": {"kind": "point", "coords": [0.0, 0.0]}}
    result = ask(agent, entities=ents, constraints=[{"kind": "coincident", "refs": ["p", "ghost"]}])["result"]
    assert result["unresolved"] and result["unresolved"][0]["state"] == "missing"


@pytest.mark.parametrize("bad", [{"entities": [], "constraints": []}, {"entities": {}, "constraints": "x"}, {"entities": {}}])
def test_malformed_requests_are_refused(agent, bad):
    with pytest.raises(SpatialValidationError):
        ask(agent, **bad)


def test_an_unknown_constraint_kind_is_refused_with_the_agents_own_error(agent):
    ents = {"p": {"kind": "point", "coords": [0.0, 0.0]}}
    with pytest.raises(SpatialValidationError):
        ask(agent, entities=ents, constraints=[{"kind": "no_such_kind", "refs": ["p"]}])
