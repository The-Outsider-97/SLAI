"""Phase 1 executable modifiers: independently checked geometry and rollback."""
from copy import deepcopy
from math import isclose

import pytest

from src.utils.sgeo import SGEODocument
from src.utils.sdrw import SDRWDocument
from src.utils.spatial_formats.common.errors import (ConflictError, GeometryValidationError,
                                                     UnsupportedFeatureError, ValidationError)
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.sgeo.geometry import validate_geometry
from src.utils.spatial_formats.sdrw.entities import validate_entity
from src.utils.spatial_formats.container.versions import FEATURES


def _add(doc, target, kind, parameters, *, operands=(), mode="exact"):
    mid = doc.add_modifier(target, kind=kind, operands=operands, parameters=parameters,
                           evaluation_policy=({"mode": mode, "absolute": 1e-6}
                                              if mode == "tolerance_controlled" else {"mode": mode}))
    assert "modifier-ops-v1" in doc.manifest()["requires"]
    return mid


def test_exact_box_boolean_and_transitive_caching_without_source_mutation():
    doc = SGEODocument.new("Boolean")
    a = doc.add_box(size=(2, 2, 2))
    b = doc.add_box(size=(2, 2, 2), center=(1, 0, 0))
    original_a, original_b = deepcopy(doc.get(a)), deepcopy(doc.get(b))
    _add(doc, a, "sgeo.boolean", {"operation": "union"}, operands=[b])
    result = doc.evaluate_modifiers()[0]
    assert result["diagnostics"]["backend"] == "axis-aligned-box"
    assert result["diagnostics"]["approximate"] is False
    assert validate_geometry(result["result"]["geometry"])
    assert not result["cache_hit"]
    assert doc.evaluate_modifiers()[0]["cache_hit"]
    assert doc.get(a) == original_a and doc.get(b) == original_b
    clone = SGEODocument.from_bytes(doc.dumps())
    assert clone.modifiers() == doc.modifiers() and clone.evaluate_modifiers()[0]["status"] == "ok"


def test_operand_changes_invalidate_only_dependent_modifier_results():
    doc = SGEODocument.new("Selective recomputation")
    first = doc.add_box(size=(2, 2, 2))
    operand = doc.add_box(size=(2, 2, 2), center=(1, 0, 0))
    unrelated = doc.add_box(size=(1, 1, 1), center=(8, 0, 0))
    boolean = _add(doc, first, "sgeo.boolean", {"operation": "intersect"}, operands=[operand])
    identity = doc.add_modifier(unrelated)
    assert doc.modifier_graph().affected(item_ids=[operand]) == (boolean,)
    assert doc.modifier_graph().affected(item_ids=[unrelated]) == (identity,)
    assert not any(item["cache_hit"] for item in doc.evaluate_modifiers())
    assert all(item["cache_hit"] for item in doc.evaluate_modifiers())
    doc.set_geometry(operand, {"kind": "box", "center": [0.5, 0, 0], "size": [2, 2, 2]})
    hits = {r["modifier_id"]: r["cache_hit"] for r in doc.evaluate_modifiers()}
    assert hits == {boolean: False, identity: True}


def test_parent_transform_changes_invalidate_referenced_world_geometry():
    doc = SGEODocument.new("Hierarchy")
    group = doc.add_group("operand parent")
    a = doc.add_box(size=(2, 2, 2))
    b = doc.add_box(size=(2, 2, 2), center=(1, 0, 0), parent=group)
    mid = _add(doc, a, "sgeo.boolean", {"operation": "intersect"}, operands=[b],
               mode="tolerance_controlled")
    assert doc.modifier_graph().affected(item_ids=[group]) == (mid,)
    initial = doc.evaluate_modifiers()[0]
    assert doc.evaluate_modifiers()[0]["cache_hit"]
    doc.move(group, 0.25, 0, 0)
    changed = doc.evaluate_modifiers()[0]
    assert not changed["cache_hit"]
    assert changed["fingerprint"] != initial["fingerprint"]
    assert changed["diagnostics"]["backend"] == "manifold3d"


def test_non_destructive_modifier_stacks_chain_derived_geometry():
    doc = SDRWDocument.new("2D modifier stack")
    a = doc.add_polyline([(0, 0), (4, 0), (4, 4), (0, 4)], closed=True)
    b = doc.add_polyline([(2, 1), (5, 1), (5, 3), (2, 3)], closed=True)
    one = _add(doc, a, "sdrw.boolean", {"operation": "subtract"},
               operands=[b], mode="tolerance_controlled")
    two = _add(doc, a, "sdrw.offset", {"distance": 0.1}, mode="tolerance_controlled")
    assert doc.modifier_graph().dependencies[two] == frozenset({one})
    results = doc.evaluate_modifiers()
    assert len(results) == 2
    assert results[0]["result"]["kind"] == "hatch"
    assert results[1]["result"]["kind"] == "hatch"
    assert results[1]["diagnostics"]["backend"] == "geos-region-buffer"
    assert all(record["cache_hit"] for record in doc.evaluate_modifiers())


@pytest.mark.parametrize("operation", ["intersect", "subtract"])
def test_sgeo_boolean_operations_have_correct_volume(operation):
    doc = SGEODocument.new("CSG")
    a = doc.add_box(size=(2, 2, 2))
    b = doc.add_box(size=(2, 2, 2), center=(1, 0, 0))
    _add(doc, a, "sgeo.boolean", {"operation": operation}, operands=[b])
    result = doc.evaluate_modifiers()[0]["result"]["geometry"]
    import trimesh
    solid = trimesh.Trimesh(vertices=result["vertices"], faces=result["faces"], process=False)
    assert solid.is_volume and isclose(solid.volume, 4, rel_tol=1e-8)


def test_general_boolean_world_transforms_requires_numerical_policy():
    doc = SGEODocument.new("Transforms")
    a = doc.add_box(size=(2, 2, 2))
    b = doc.add_box(size=(2, 2, 2), position=(1, 0, 0))
    mid = _add(doc, a, "sgeo.boolean", {"operation": "intersect"}, operands=[b])
    with pytest.raises(UnsupportedFeatureError):
        doc.evaluate_modifiers()
    doc.update_modifier(mid, evaluation_policy={"mode": "tolerance_controlled", "absolute": 1e-8})
    result = doc.evaluate_modifiers()[0]
    assert result["diagnostics"]["backend"] == "manifold3d"
    assert result["diagnostics"]["approximate"] is True


def test_sgeo_split_bend_chamfer_and_diagnostics():
    doc = SGEODocument.new("Modeling")
    a = doc.add_box(size=(4, 2, 2))
    mid = _add(doc, a, "sgeo.split", {"origin": [0, 0, 0], "normal": [1, 0, 0],
                                       "side": "positive"})
    split = doc.evaluate_modifiers()[0]
    assert split["result"]["geometry"]["kind"] == "box"
    assert split["result"]["geometry"]["size"] == [2.0, 2.0, 2.0]
    doc.remove_modifier(mid)

    _add(doc, a, "sgeo.chamfer", {"amount": 0.2, "depth": 0.3, "segments": 3},
         mode="tolerance_controlled")
    bevel = doc.evaluate_modifiers()[0]
    assert bevel["result"]["geometry"]["kind"] == "mesh"
    assert bevel["diagnostics"]["watertight"]
    import trimesh
    shape = bevel["result"]["geometry"]
    bevel_solid = trimesh.Trimesh(vertices=shape["vertices"], faces=shape["faces"], process=False)
    assert bevel_solid.is_volume and 0 < bevel_solid.volume < 16
    doc.remove_modifier(doc.modifiers()[0]["id"])

    _add(doc, a, "sgeo.bend", {"axis": "x", "start": -2, "end": 2,
                                      "angle": 0.6, "segments": 8}, mode="display_only")
    bent = doc.evaluate_modifiers()[0]
    assert bent["result"]["geometry"]["kind"] == "mesh"
    assert bent["diagnostics"]["segments"] == 8


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_bend_supports_all_three_local_axes(axis):
    doc = SGEODocument.new("Axis")
    obj = doc.add_box(size=(2, 2, 2))
    _add(doc, obj, "sgeo.bend", {"axis": axis, "start": -1, "end": 1,
                                 "angle": 0.35, "segments": 5}, mode="display_only")
    result = doc.evaluate_modifiers()[0]
    assert result["result"]["geometry"]["kind"] == "mesh"
    assert result["diagnostics"]["watertight"]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_2d_bend_supports_both_axes(axis):
    doc = SDRWDocument.new("Bend axis")
    obj = doc.add_line((-2, -2), (2, 2))
    _add(doc, obj, "sdrw.bend", {"axis": axis, "start": -2, "end": 2,
                                  "angle": 0.2, "segments": 6}, mode="display_only")
    result = doc.evaluate_modifiers()[0]
    assert result["result"]["kind"] == "polyline"
    assert result["diagnostics"]["approximate"]


def test_split_general_plane_caps_solid_and_reports_numerical_kernel():
    doc = SGEODocument.new("Oblique plane")
    a = doc.add_box(size=(2, 2, 2))
    _add(doc, a, "sgeo.split", {"origin": [0, 0, 0], "normal": [1, 1, 0],
                                "side": "positive"}, mode="tolerance_controlled")
    result = doc.evaluate_modifiers()[0]
    assert result["diagnostics"]["backend"] == "mesh-plane-split"
    import trimesh
    mesh = result["result"]["geometry"]
    solid = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False)
    assert solid.is_volume and isclose(solid.volume, 4, rel_tol=1e-7)


def test_empty_boolean_result_is_reported_without_inventing_geometry():
    doc = SDRWDocument.new("Empty result")
    a = doc.add_polyline([(0, 0), (2, 0), (2, 2), (0, 2)], closed=True)
    b = doc.add_polyline([(0, 0), (2, 0), (2, 2), (0, 2)], closed=True)
    _add(doc, a, "sdrw.boolean", {"operation": "subtract"}, operands=[b],
         mode="tolerance_controlled")
    result = doc.evaluate_modifiers()[0]
    assert result["status"] == "empty" and result["result"] is None
    assert doc.get(a)["kind"] == "polyline"


def test_curved_regions_respect_sampling_budget():
    doc = SDRWDocument.new("Curved overlay")
    a = doc.add_circle((0, 0), 2)
    b = doc.add_circle((1, 0), 2)
    _add(doc, a, "sdrw.boolean", {"operation": "intersect"}, operands=[b],
         mode="display_only")
    result = doc.evaluate_modifiers()[0]
    assert result["result"]["kind"] == "hatch"
    assert result["diagnostics"]["approximate"] is True
    doc.update_modifier(doc.modifiers()[0]["id"],
                        evaluation_policy={"mode": "tolerance_controlled", "absolute": 1e-12})
    from src.utils.spatial_formats.common.errors import ResourceLimitError
    with pytest.raises(ResourceLimitError):
        doc.evaluate_modifiers()


def test_sdrw_region_boolean_with_hole_and_roundtrip():
    doc = SDRWDocument.new("Regions")
    a = doc.add_polyline([(0, 0), (6, 0), (6, 6), (0, 6)], closed=True)
    b = doc.add_polyline([(2, 2), (4, 2), (4, 4), (2, 4)], closed=True)
    source = deepcopy(doc.get(a))
    _add(doc, a, "sdrw.boolean", {"operation": "subtract"}, operands=[b],
         mode="tolerance_controlled")
    result = doc.evaluate_modifiers()[0]
    assert result["result"]["kind"] == "hatch"
    assert len(result["result"]["geometry"]["boundary"]) == 2
    assert isclose(result["diagnostics"]["area"], 32)
    assert validate_entity(result["result"])
    assert doc.get(a) == source
    assert SDRWDocument.from_bytes(doc.dumps()).modifiers() == doc.modifiers()


def test_sdrw_trim_extend_offset_bend_chamfer():
    doc = SDRWDocument.new("Drafting")
    a = doc.add_line((0, 0), (10, 0))
    boundary = doc.add_line((5, -2), (5, 2))
    mid = _add(doc, a, "sdrw.trim", {"endpoint": "end"}, operands=[boundary])
    assert doc.evaluate_modifiers()[0]["result"]["geometry"]["end"] == [5.0, 0.0]
    doc.remove_modifier(mid)
    doc.update(a, geometry={"start": [0, 0], "end": [4, 0]})
    mid = _add(doc, a, "sdrw.extend", {"endpoint": "end"}, operands=[boundary])
    assert doc.evaluate_modifiers()[0]["result"]["geometry"]["end"] == [5.0, 0.0]
    doc.remove_modifier(mid)

    mid = _add(doc, a, "sdrw.offset", {"distance": 2}, mode="tolerance_controlled")
    assert doc.evaluate_modifiers()[0]["result"]["geometry"]["start"] == [0.0, 2.0]
    doc.remove_modifier(mid)
    mid = _add(doc, a, "sdrw.bend", {"axis": "x", "start": 0, "end": 4,
                                       "angle": 0.5, "segments": 8}, mode="display_only")
    assert len(doc.evaluate_modifiers()[0]["result"]["geometry"]["points"]) > 2
    doc.remove_modifier(mid)

    polygon = doc.add_polyline([(0, 0), (5, 0), (5, 5), (0, 5)], closed=True)
    _add(doc, polygon, "sdrw.chamfer", {"amount": 0.3, "depth": 0.4, "segments": 2})
    out = doc.evaluate_modifiers()[0]["result"]
    assert out["kind"] == "polyline" and len(out["geometry"]["points"]) == 12
    # Rounded, segmented corners must change geometry, not insert collinear
    # points along a single straight chamfer.
    first, midpoint, last = out["geometry"]["points"][:3]
    cross = (midpoint[0]-first[0])*(last[1]-first[1]) - (midpoint[1]-first[1])*(last[0]-first[0])
    assert abs(cross) > 1e-6


def test_errors_are_explicit_and_never_mutate_source():
    doc = SDRWDocument.new("Safety")
    a = doc.add_line((0, 0), (1, 0))
    b = doc.add_line((0, 1), (1, 1))
    old = doc.dumps()
    with pytest.raises(ValidationError):
        _add(doc, a, "sdrw.bend", {"axis": "z", "start": 0, "end": 1,
                                       "angle": 0.5, "segments": 4})
    assert doc.dumps() == old
    _add(doc, a, "sdrw.trim", {"endpoint": "end"}, operands=[b])
    with pytest.raises(GeometryValidationError):
        doc.evaluate_modifiers()
    assert doc.get(a)["geometry"]["end"] == [1, 0]


def test_numerical_policy_requires_real_tolerance():
    doc = SDRWDocument.new("Tolerance")
    entity = doc.add_circle((0, 0), 3)
    before = doc.dumps()
    with pytest.raises(ValidationError, match="positive tolerance"):
        doc.add_modifier(entity, kind="sdrw.offset", parameters={"distance": 1},
                         evaluation_policy={"mode": "tolerance_controlled", "absolute": 0})
    assert doc.dumps() == before


def test_malformed_modifier_payloads_raise_typed_errors():
    doc = SGEODocument.new("Malformed modifier")
    obj = doc.add_box(size=(1, 1, 1))
    modifier = doc.add_modifier(obj)
    before = doc.dumps()
    with pytest.raises(ValidationError):
        doc.update_modifier(modifier, evaluation_policy={"mode": {}})
    with pytest.raises(UnsupportedFeatureError):
        doc.apply([{"op": "add_modifier", "value": {
            **doc.modifiers()[0], "id": new_id(),
            "kind": ["bad"]}}])
    assert doc.dumps() == before


def test_constraint_solver_preview_commit_and_conflict_fencing():
    doc = SDRWDocument.new("Constraints")
    anchor = doc.add_point((0, 0))
    mobile = doc.add_point((6, 0))
    constraints = [{"kind": "distance", "entities": [anchor, mobile], "value": 4.0}]
    before, rev = doc.dumps(), doc.revision
    proposal = doc.solve_constraints(constraints, fixed_ids=[anchor])
    assert proposal.converged and all(c.satisfied for c in proposal.results)
    assert isclose(proposal.geometries[mobile]["position"][0], 4.0, abs_tol=1e-7)
    assert doc.dumps() == before
    with pytest.raises(ConflictError):
        doc.solve_constraints(constraints, fixed_ids=[anchor], commit=True,
                              expected_revision=rev - 1)
    doc.solve_constraints(constraints, fixed_ids=[anchor], commit=True, expected_revision=rev)
    assert doc.revision == rev+1
    assert all(c.satisfied for c in doc.check_constraints(constraints))


def test_constraint_solver_line_orientation_and_infeasibility_rollback():
    doc = SDRWDocument.new("Constraint diagnostics")
    line = doc.add_line((0, 0), (4, 1))
    target = [{"kind": "horizontal", "entities": [line]}]
    proposal = doc.solve_constraints(target)
    assert proposal.results[0].satisfied
    assert doc.get(line)["geometry"]["end"] == [4.0, 1.0]
    old = doc.dumps()
    with pytest.raises(ConflictError):
        doc.solve_constraints(target, fixed_ids=[line], commit=True)
    assert doc.dumps() == old


def test_persistent_constraints_roundtrip_reference_safety_and_solve():
    doc = SDRWDocument.new("Persistent constraints")
    line = doc.add_line((0, 0), (3, 1))
    definitions = [{"kind": "horizontal", "entities": [line]}]
    doc.set_constraints(definitions)
    assert doc.constraints() == definitions
    assert "sdrw-constraints-v1" in doc.manifest()["requires"]
    source = doc.dumps()
    reopened = SDRWDocument.from_bytes(source)
    assert reopened.constraints() == definitions
    assert not reopened.check_constraints()[0].satisfied
    with pytest.raises(ValidationError):
        reopened.remove(line)
    with pytest.raises((ValidationError, ConflictError)):
        reopened.set_constraints([{"kind": "horizontal", "entities": ["bad"]}])
    assert reopened.dumps() == source
    reopened.solve_constraints(commit=True)
    assert reopened.check_constraints()[0].satisfied
    assert reopened.constraints() == definitions


def test_interchange_reports_source_geometry_and_dropped_modifier_results():
    scene = SGEODocument.new("Loss reporting")
    first = scene.add_box(size=(2, 2, 2))
    second = scene.add_box(size=(2, 2, 2), center=(1, 0, 0))
    _add(scene, first, "sgeo.boolean", {"operation": "union"}, operands=[second])
    _, report = scene.to_glb(policy="best_effort")
    assert any("modifier" in item for item in report.dropped)

    drawing = SDRWDocument.new("Loss reporting")
    point = drawing.add_point((1, 2))
    drawing.set_constraints([{"kind": "coincident", "entities": [point, point]}])
    _add(drawing, point, "sdrw.offset", {"distance": 1}, mode="tolerance_controlled")
    _, report = drawing.to_svg(policy="best_effort")
    assert any("modifier" in item for item in report.dropped)
    assert any("constraint" in item for item in report.dropped)


def test_legacy_phase0_readers_must_refuse_unrecognized_phase1_features(monkeypatch):
    doc = SGEODocument.new("Feature compatibility")
    obj = doc.add_box(size=(2, 2, 2))
    _add(doc, obj, "sgeo.chamfer", {"amount": 0.2, "depth": 0.2, "segments": 1},
         mode="tolerance_controlled")
    blob = doc.dumps()
    with monkeypatch.context() as patch:
        patch.setitem(FEATURES, "0.3.0", FEATURES["0.3.0"] - {"modifier-ops-v1"})
        with pytest.raises(UnsupportedFeatureError):
            SGEODocument.from_bytes(blob)
    assert SGEODocument.from_bytes(blob).modifiers() == doc.modifiers()
