"""The numpy-only constraint engine: solving, DOF, redundancy, conflict, stable references, 2D and 3D."""
import math

import pytest

np = pytest.importorskip("numpy")

from src.utils.spatial_formats.common import constraint_engine as CE
from src.utils.spatial_formats.common.errors import ValidationError


def seg(x0, y0, x1, y1, name=None):
    return {"kind": "segment", "coords": [x0, y0, x1, y1], "name": name}


def pt(*c):
    return {"kind": "point", "coords": list(c)}


def rect():
    """A slightly skewed quadrilateral a-b-c-d made of four segments sharing corner points through coincident constraints."""
    ents = {"a": seg(0, 0, 4.2, 0.3), "b": seg(4.0, 0.1, 4.1, 3.2), "c": seg(4.1, 3.0, 0.2, 3.1), "d": seg(0.1, 3.0, 0.0, 0.2)}
    cons = [{"kind": "coincident", "refs": [{"entity": "a", "part": "end"}, {"entity": "b", "part": "start"}]},
            {"kind": "coincident", "refs": [{"entity": "b", "part": "end"}, {"entity": "c", "part": "start"}]},
            {"kind": "coincident", "refs": [{"entity": "c", "part": "end"}, {"entity": "d", "part": "start"}]},
            {"kind": "coincident", "refs": [{"entity": "d", "part": "end"}, {"entity": "a", "part": "start"}]}]
    return ents, cons


def test_a_square_is_solved_and_its_dof_are_counted():
    ents, cons = rect()
    cons += [{"kind": "horizontal", "refs": ["a"]}, {"kind": "vertical", "refs": ["b"]}, {"kind": "horizontal", "refs": ["c"]}, {"kind": "vertical", "refs": ["d"]},
             {"kind": "distance", "refs": [{"entity": "a", "part": "start"}, {"entity": "a", "part": "end"}], "value": 4.0},
             {"kind": "distance", "refs": [{"entity": "b", "part": "start"}, {"entity": "b", "part": "end"}], "value": 3.0}]
    r = CE.analyze(ents, cons, fixed=[])
    # 16 coordinates, 8 coincidence equations, 4 orientations, 2 lengths = 14: a rectangle still has 2 translation degrees of freedom.
    assert (r["variables"], r["equations"], r["rank"], r["dof"]) == (16, 14, 14, 2)
    assert r["status"] == "under_constrained" and r["converged"] and r["max_residual"] < 1e-9
    a = r["solution"]["a"]
    assert math.isclose(a[2] - a[0], 4.0, abs_tol=1e-8) and math.isclose(a[3], a[1], abs_tol=1e-8)
    # pin one corner (a.start) with a fixed constraint: translation is gone
    cons.append({"kind": "fixed", "refs": [{"entity": "a", "part": "start"}], "value": [1.0, 2.0]})
    r = CE.analyze(ents, cons)
    assert (r["dof"], r["status"]) == (0, "well_constrained")
    assert np.allclose(r["solution"]["a"][:2], [1.0, 2.0], atol=1e-8)
    assert np.allclose(r["solution"]["c"][2:], [1.0, 5.0], atol=1e-8)


def test_a_redundant_but_consistent_constraint_is_named_and_a_conflicting_one_too():
    ents = {"p": pt(0, 0), "q": pt(3.5, 0.4)}
    base = [{"kind": "fixed", "refs": ["p"], "value": [0.0, 0.0]}, {"kind": "distance", "refs": ["p", "q"], "value": 4.0}]
    r = CE.analyze(ents, base + [{"kind": "distance", "refs": ["q", "p"], "value": 4.0}])          # the same fact twice
    assert r["status"] == "under_constrained" and r["dof"] == 1                                   # q can still swing round p...
    assert r["redundant"] == [2] and r["conflicting"] == []                                        # ...and the repeated distance is named as redundant
    pinned = base + [{"kind": "fixed", "refs": ["q"], "value": [4.0, 0.0]}]
    r = CE.analyze(ents, pinned)
    assert r["status"] == "over_constrained_consistent" and r["dof"] == 0 and r["redundant"] == [2]   # fully determined; the last constraint repeats part of what the distance already says (redundancy is read in order)
    r = CE.analyze(ents, base + [{"kind": "distance", "refs": ["q", "p"], "value": 5.0}])          # a different fact
    assert r["status"] == "conflicting" and r["conflicting"]
    assert {c["index"] for c in r["constraints"] if c["status"] == "conflicting"} <= {1, 2}
    assert not r["converged"] or r["max_residual"] > 1e-6


def test_under_constrained_says_which_entities_are_free():
    ents = {"p": pt(0, 0), "q": pt(3, 1), "r": pt(9, 9)}
    r = CE.analyze(ents, [{"kind": "distance", "refs": ["p", "q"], "value": 2.0}, {"kind": "fixed", "refs": ["p"], "value": [0, 0]}])
    assert r["status"] == "under_constrained" and r["dof"] == 1                                  # q can still swing on a circle; r is not in the system
    assert set(r["free_entities"]) == {"q"}


def test_the_input_is_never_changed_and_the_result_is_deterministic():
    ents, cons = rect()
    before = {k: list(v["coords"]) for k, v in ents.items()}
    a = CE.analyze(ents, cons)
    b = CE.analyze(ents, cons)
    assert before == {k: list(v["coords"]) for k, v in ents.items()}
    assert a["solution"] == b["solution"] and a["rank"] == b["rank"]


def test_references_report_missing_stale_ambiguous_and_unsupported_instead_of_being_dropped():
    ents = {"a": seg(0, 0, 1, 0, "wall"), "b": seg(0, 1, 1, 1, "wall"), "p": pt(0, 0)}
    fp = CE.fingerprint({"kind": "point", "coords": [0, 0, 0]})                                   # made when p was a 3D point
    cons = [{"kind": "distance", "refs": ["p", "ghost"], "value": 1},
            {"kind": "distance", "refs": [{"entity": "p", "fingerprint": fp}, {"entity": "a", "part": "end"}], "value": 1},
            {"kind": "parallel", "refs": [{"name": "wall"}, "a"]},
            {"kind": "distance", "refs": ["a", "p"], "value": 1},                                   # a whole segment where a point is needed
            {"kind": "parallel", "refs": ["a", "b"]}]
    r = CE.analyze(ents, cons, fixed=["a"])
    states = {u["index"]: u["state"] for u in r["unresolved"]}
    assert states == {0: "missing", 1: "stale", 2: "ambiguous", 3: "unsupported"}
    assert [c["index"] for c in r["constraints"]] == [4]                                           # the good one was still analysed


def test_three_dimensions():
    ents = {"a": {"kind": "segment", "coords": [0, 0, 0, 1, 0.1, 0.1]}, "b": {"kind": "segment", "coords": [0, 1, 0, 0.9, 1.1, 0.2]}}
    cons = [{"kind": "parallel", "refs": ["a", "b"]}, {"kind": "distance", "refs": [{"entity": "a", "part": "start"}, {"entity": "a", "part": "end"}], "value": 2.0},
            {"kind": "perpendicular", "refs": ["a", "b"]}]
    r = CE.analyze(ents, cons, fixed=[])
    assert r["status"] in ("conflicting",)                                                         # parallel and perpendicular cannot both hold
    cons[2] = {"kind": "angle", "refs": ["a", "b"], "value": 0.0}
    r = CE.analyze(ents, cons)
    assert r["status"] != "conflicting" and r["converged"]
    a, b = np.array(r["solution"]["a"]), np.array(r["solution"]["b"])
    u, v = a[3:] - a[:3], b[3:] - b[:3]
    assert math.isclose(np.linalg.norm(u), 2.0, abs_tol=1e-7)
    assert np.linalg.norm(np.cross(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)) < 1e-6


def test_bad_input_is_refused():
    with pytest.raises(ValidationError):
        CE.analyze({"p": {"kind": "circle", "coords": [0, 0, 1]}}, [])
    with pytest.raises(ValidationError):
        CE.analyze({"p": pt(0, 0)}, [{"kind": "teleport", "refs": ["p"]}])
    with pytest.raises(ValidationError):
        CE.analyze({"p": pt(0, 0), "q": pt(1, 1)}, [{"kind": "distance", "refs": ["p", "q"], "value": -1}])
    with pytest.raises(ValidationError):
        CE.analyze({"p": pt(float("nan"), 0)}, [])


def test_sdrw_documents_analyze_and_solve_without_scipy():
    pytest.importorskip("jsonschema")
    from src.utils.spatial_formats.sdrw.document import DrawingDocument
    try:
        import scipy  # noqa: F401
        have_scipy = True
    except ImportError:
        have_scipy = False
    doc = DrawingDocument.new("c")
    a = doc.add_line((0, 0), (4, 0.3))
    b = doc.add_line((0, 1), (4, 1.4))
    p = doc.add_point((0, 0))
    cons = [{"kind": "horizontal", "entities": [a]}, {"kind": "parallel", "entities": [a, b]}, {"kind": "distance", "entities": [p, p], "value": 0.0}]
    rep = doc.analyze_constraints(cons, fixed_ids=[p])
    assert rep["unresolved"] == [] and rep["status"] in ("under_constrained", "over_constrained_consistent", "well_constrained")
    rev = doc.revision
    sol = doc.solve_constraints(cons[:2], fixed_ids=[p])                       # preview: the document does not change
    assert doc.revision == rev
    assert abs(sol.geometries[a]["end"][1] - sol.geometries[a]["start"][1]) < 1e-8
    assert abs(sol.geometries[b]["end"][1] - sol.geometries[b]["start"][1]) < 1e-8
    doc.solve_constraints(cons[:2], fixed_ids=[p], commit=True, expected_revision=rev)
    assert doc.revision == rev + 1 and doc.get(a)["geometry"]["end"][1] == pytest.approx(doc.get(a)["geometry"]["start"][1], abs=1e-8)
    bad = [{"kind": "distance", "entities": [a, p], "value": 1.0}, {"kind": "horizontal", "entities": ["00000000-0000-4000-8000-000000000000"]}]
    rep = doc.analyze_constraints(bad)                                          # a missing reference is reported, not dropped and not fatal
    assert [(u["index"], u["state"]) for u in rep["unresolved"]] == [(1, "missing")]
    with pytest.raises(Exception):                                              # but solving with an unresolvable system is refused
        doc.solve_constraints(bad, fixed_ids=[p])
