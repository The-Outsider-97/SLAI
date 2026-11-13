"""Regression contracts for strictly bounded box Booleans and native time evaluation."""
from __future__ import annotations

from copy import deepcopy
from random import Random

import pytest

from src.utils.spatial_formats.common.errors import UnsupportedFeatureError, ValidationError, ReferenceError
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.transforms import from_axis_angle
from src.utils.spatial_formats.sgeo.box_boolean import boolean_boxes
from src.utils.spatial_formats.sgeo.box_offset import offset_box_faces
from src.utils.spatial_formats.sgeo.analysis import analyze_mesh
from src.utils.spatial_formats.sgeo.document import SceneDocument
from src.agents.spatial.spatial_temporal import TemporalSpatialQueries
from src.agents.spatial.utils.spatial_errors import SpatialValidationError


def _volume(mesh):
    total = 0.0
    for i, j, k in mesh["faces"]:
        a, b, c = (mesh["vertices"][index] for index in (i, j, k))
        cross = (b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0])
        total += sum(a[p]*cross[p] for p in range(3))/6
    return total


BOX_A = {"kind": "box", "center": [0, 0, 0], "size": [2, 2, 2]}
BOX_B = {"kind": "box", "center": [1, 0, 0], "size": [2, 2, 2]}


@pytest.mark.parametrize("op,volume", [("union", 12), ("intersection", 4), ("difference", 4)])
def test_boolean_box_volumes_and_manifold_edges(op, volume):
    result = boolean_boxes(BOX_A, BOX_B, op)
    assert result["kind"] == "mesh"
    assert _volume(result) == pytest.approx(volume)
    counts = {}
    for a, b, c in result["faces"]:
        for i, j in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((i, j)))
            counts[edge] = counts.get(edge, 0) + 1
    assert all(count == 2 for count in counts.values())


def test_empty_and_unsupported_box_booleans():
    distant = {"kind": "box", "center": [10, 0, 0], "size": [1, 1, 1]}
    assert boolean_boxes(BOX_A, distant, "intersection") is None
    assert boolean_boxes(BOX_A, BOX_A, "difference") is None
    with pytest.raises(UnsupportedFeatureError):
        boolean_boxes({"kind": "sphere", "center": [0, 0, 0], "radius": 1}, BOX_A, "union")
    edge_touch = {"kind": "box", "center": [2, 2, 0], "size": [2, 2, 2]}
    with pytest.raises(UnsupportedFeatureError, match="non-manifold"):
        boolean_boxes(BOX_A, edge_touch, "union")


def test_document_box_boolean_atomic_and_transforms():
    doc = SceneDocument.new()
    a = doc.add_box(size=[2, 2, 2], position=(3, 0, 0))
    b = doc.add_box(size=[2, 2, 2], position=(4, 0, 0))
    rev = doc.revision
    created = doc.boolean_boxes(a, b, "difference", name="rectilinear cut")
    assert doc.revision == rev + 1
    assert _volume(doc.get(created)["geometry"]) == pytest.approx(4)
    original = doc.as_dict()
    doc.set_transform(b, from_axis_angle((0, 0, 1), 0.2))
    with pytest.raises(UnsupportedFeatureError):
        doc.boolean_boxes(a, b, "union")
    assert len(doc.as_dict()["objects"]) == len(original["objects"])


def test_mitered_box_offset_and_collapse_rollback():
    doc = SceneDocument.new()
    oid = doc.add_box(size=[2, 3, 4])
    new_id = doc.offset_box_faces(oid, 0.5)
    assert doc.get(new_id)["geometry"]["size"] == pytest.approx([3, 4, 5])
    assert doc.get(oid)["geometry"]["size"] == [2, 3, 4]
    previous = doc.as_dict()
    with pytest.raises(ValidationError, match="collapses"):
        doc.offset_box_faces(oid, -1)
    assert doc.as_dict() == previous
    with pytest.raises(UnsupportedFeatureError):
        offset_box_faces({"kind": "sphere", "center": [0, 0, 0], "radius": 1}, 0.2)


def test_rectilinear_boolean_randomized_volume_against_independent_analytic_formula():
    generator = Random(20261029)
    for _ in range(30):
        center = [generator.randint(-3, 3) for _ in range(3)]
        sizes = [generator.randint(1, 4) for _ in range(3)]
        second = {"kind": "box", "center": center, "size": sizes}
        common_axes = [max(0, min(BOX_A["center"][i] + BOX_A["size"][i]/2,
                                  center[i] + sizes[i]/2)
                              - max(BOX_A["center"][i] - BOX_A["size"][i]/2,
                                    center[i] - sizes[i]/2)) for i in range(3)]
        intersect_volume = common_axes[0] * common_axes[1] * common_axes[2]
        other_volume = sizes[0]*sizes[1]*sizes[2]
        for operation, expected in (("union", 8 + other_volume - intersect_volume),
                                    ("intersection", intersect_volume),
                                    ("difference", 8 - intersect_volume)):
            # Box contact can create non-manifold edge/corner contacts; these
            # are intentionally rejected rather than misrepresented as solids.
            try:
                mesh = boolean_boxes(BOX_A, second, operation)
            except UnsupportedFeatureError:
                assert operation == "union" and intersect_volume == 0
                continue
            if expected == 0:
                assert mesh is None
            else:
                diagnostics = analyze_mesh(mesh)
                assert diagnostics.is_closed_oriented_surface
                assert diagnostics.signed_volume == pytest.approx(expected)


def _animation(object_id, *, prop="translation", values=((0, [0, 0, 0]), (2, [2, 0, 0])), **kwargs):
    return {"duration": 2.0, "tracks": [{"object_id": object_id, "property": prop,
                                          "interpolation": "linear" if prop != "visibility" else "step",
                                          "keyframes": [{"time": t, "value": v} for t, v in values], **kwargs}]}


def test_animation_round_trip_detached_evaluation_and_reverse_order():
    doc = SceneDocument.new()
    parent = doc.add_group("parent")
    child = doc.add_box(size=[1, 1, 1], parent=parent)
    spec = _animation(parent)
    doc.set_animation(spec)
    before = doc.as_dict()
    queries = TemporalSpatialQueries(doc)
    assert queries.get_bounds_at(child, 1)[0] == pytest.approx((0.5, -0.5, -0.5))
    assert doc.evaluate(2).world_bounds(child) == doc.evaluate(2).world_bounds(child)
    assert doc.evaluate(0).world_bounds(child)[0][0] == pytest.approx(-0.5)
    assert doc.evaluate(2).world_bounds(child)[0][0] == pytest.approx(1.5)
    assert doc.as_dict() == before
    restored = SceneDocument.from_bytes(doc.dumps())
    assert restored.animation() == spec
    assert restored.evaluate(1).world_bounds(child) == doc.evaluate(1).world_bounds(child)


def test_animation_validation_rolls_back_and_refuses_dangling_targets():
    doc = SceneDocument.new()
    oid = doc.add_box(size=[1, 1, 1])
    good = _animation(oid)
    doc.set_animation(good)
    before = doc.as_dict()
    bad = deepcopy(good)
    bad["tracks"][0]["keyframes"][1]["time"] = 0
    with pytest.raises(ValidationError):
        doc.set_animation(bad)
    assert doc.as_dict() == before
    with pytest.raises(ReferenceError):
        doc.set_animation(_animation(new_id()))
    with pytest.raises(ValidationError):
        doc.remove(oid)


def test_parameter_rotation_visibility_and_temporal_classification():
    doc = SceneDocument.new()
    a = doc.add_box(size=[1, 1, 1])
    b = doc.add_box(size=[1, 1, 1], position=(4, 0, 0))
    rotation = _animation(a, prop="rotation_angle", values=((0, 0), (2, 1.5707963267948966)), axis=[0, 0, 1])["tracks"][0]
    movement = _animation(b)["tracks"][0]
    movement["keyframes"][-1]["value"] = [-4, 0, 0]
    doc.set_animation({"duration": 2, "tracks": [rotation, movement]})
    state = doc.evaluate(2)
    assert state.world_transform(a)[0] == pytest.approx(0, abs=1e-8)
    queries = TemporalSpatialQueries(doc)
    assert queries.get_distance_at(a, b, 0).classification == "verified_separation"
    assert queries.get_distance_at(a, b, 2).classification == "potential_overlap"
    assert queries.collision_at(a, b, 0).classification == "verified_separation"
    assert queries.collision_at(a, b, 2).classification == "broad_phase_potential"  # rotated box
    interval = queries.query_collision_interval(a, b, 0, 2, samples=5)
    assert interval.status == "potential_at_samples"
    assert interval.continuous_verified is False
    assert all(item.classification == "potential_overlap" for item in queries.find_intersections_at(2))
    doc.set_animation(_animation(b, prop="visibility", values=((0, True), (2, False))))
    assert b not in doc.evaluate(2).ids()


def test_temporal_narrow_phase_uses_existing_primitive_collision_engine():
    doc = SceneDocument.new()
    a = doc.add_box(size=[1, 1, 1])
    b = doc.add_box(size=[1, 1, 1], position=(3, 0, 0))
    doc.set_animation(_animation(b, values=((0, [0, 0, 0]), (2, [-3, 0, 0]))))
    queries = TemporalSpatialQueries(doc)
    assert queries.collision_at(a, b, 0).classification == "verified_separation"
    assert queries.collision_at(a, b, 2).classification == "verified_collision"
    sphere = doc.add_sphere(0.5, position=(0, 0, 0))
    assert queries.collision_at(a, sphere, 0).classification == "verified_collision"


def test_animated_family_parameter_regeneration():
    doc = SceneDocument.new()
    family = {"id": new_id(), "name": "Cabinet", "generator": {"kind": "builtin.box_v1"},
              "parameters": {"width": {"type": "length", "default": 1},
                             "depth": {"type": "length", "default": 1},
                             "height": {"type": "length", "default": 1}}}
    fid = doc.define_family(family["name"], family["parameters"], family["generator"]["kind"], family_id=family["id"])
    oid = doc.instantiate(fid)
    doc.set_animation(_animation(oid, prop="parameter.width", values=((0, 1), (2, 3))))
    assert doc.evaluate(1).get_object_state(oid)["object"]["geometry"]["size"][0] == pytest.approx(2)
    assert doc.get(oid)["geometry"]["size"][0] == 1


def test_bad_animation_data_cannot_be_persisted_or_evaluated():
    doc = SceneDocument.new()
    oid = doc.add_box(size=[1, 1, 1])
    before = doc.as_dict()
    for bad_spec in (
        _animation(oid, values=((0, [0, 0, 0]), (0, [2, 0, 0]))),
        _animation(oid, values=((0, [0, 0, 0]), (2, [float("nan"), 0, 0]))),
        _animation(oid, prop="visibility", values=((0, True), (2, 1))),
        {"duration": 2, "tracks": _animation(oid)["tracks"] * 2},
    ):
        with pytest.raises(ValidationError):
            doc.set_animation(bad_spec)
        assert doc.as_dict() == before
    doc.set_animation(_animation(oid))
    with pytest.raises(ValidationError):
        doc.evaluate(float("inf"))
    with pytest.raises(SpatialValidationError):
        TemporalSpatialQueries(doc).query_collision_interval(oid, oid, 2, 1)
