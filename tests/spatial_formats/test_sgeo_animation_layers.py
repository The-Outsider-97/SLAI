"""Focused native SGEO v2 layer composition regression checks."""
from math import cos, pi, sin

import pytest

from src.utils.spatial_formats.sgeo.scene import Scene
from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.common.transforms import transform_point
from src.utils.spatial_formats.common.ids import new_id


def track(oid, prop, a, b, interpolation="linear"):
    return {"object_id": oid, "property": prop, "interpolation": interpolation,
            "keyframes": [{"time": 0, "value": a}, {"time": 1, "value": b}]}


def layer(mode, tracks, reference=None, weight=1):
    result = {"id": new_id(), "mode": mode, "weight": weight, "tracks": tracks}
    if reference is not None:
        result["reference"] = reference
    return result


def test_additive_translation_and_scale_on_native_evaluated_scene():
    scene = Scene()
    oid = scene.add_object(name="layered")
    spec = {"duration": 1, "tracks": [
        track(oid, "translation", [1, 0, 0], [1, 0, 0]),
        track(oid, "scale", [2, 2, 2], [2, 2, 2]),
    ], "layers": [
        layer("additive", [track(oid, "translation", [0, 0, 0], [4, 0, 0])],
              {oid + ":translation": [0, 0, 0]}, weight=0.5),
        layer("additive", [track(oid, "scale", [1, 1, 1], [4, 4, 4])],
              {oid + ":scale": [1, 1, 1]}, weight=0.5),
    ]}
    scene.set_animation(spec, version=2)
    state = scene.evaluate(1)
    assert transform_point(state.world_transform(oid), (0, 0, 0))[0] == pytest.approx(3)
    assert transform_point(state.world_transform(oid), (1, 0, 0))[0] == pytest.approx(7)
    assert scene.metadata["slai.animation.v2"] == spec


def test_additive_quaternion_delta_and_override():
    scene = Scene()
    oid = scene.add_object(name="rotating")
    identity = [1, 0, 0, 0]
    quarter_turn = [cos(pi/4), 0, 0, sin(pi/4)]
    spec = {"duration": 1, "tracks": [
        track(oid, "rotation_quaternion", identity, identity, "slerp"),
    ], "layers": [
        layer("additive", [track(oid, "rotation_quaternion", identity, quarter_turn, "slerp")],
              {oid + ":rotation_quaternion": identity}, weight=1),
    ]}
    scene.set_animation(spec, version=2)
    point = transform_point(scene.evaluate(1).world_transform(oid), (1, 0, 0))
    assert point[:2] == pytest.approx((0, 1), abs=1e-8)


def test_layer_validation_preserves_authoritative_revision():
    scene = Scene()
    oid = scene.add_object(name="unchanged")
    initial_revision = scene.revision
    invalid = {"duration": 1, "tracks": [], "layers": [
        layer("additive", [track(oid, "scale", [1, 1, 1], [-1, 1, 1])],
              {oid + ":scale": [1, 1, 1]})
    ]}
    with pytest.raises(ValidationError):
        scene.set_animation(invalid, version=2)
    assert scene.revision == initial_revision
    assert "slai.animation.v2" not in scene.metadata
