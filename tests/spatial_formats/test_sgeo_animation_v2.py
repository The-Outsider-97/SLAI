"""Versioned SGEO animation regression tests (native scene and document facade)."""
from copy import deepcopy
from math import cos, pi, sin

import pytest

from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.sgeo.document import SceneDocument
from src.utils.spatial_formats.sgeo.scene import Scene
from src.agents.animations.animation_commands import update_animation
from src.agents.animations.timeline.interpolation import cubic_hermite, slerp


def _track(oid, prop, values, method="linear"):
    return {
        "object_id": oid, "property": prop, "interpolation": method,
        "keyframes": [{"time": t, "value": value} for t, value in values],
    }


def test_v2_quaternion_scale_cubic_roundtrip_and_non_mutating_evaluation():
    doc = SceneDocument.new()
    oid = doc.add_box(size=[1, 1, 1])
    translation = _track(oid, "translation", [(0, [0, 0, 0]), (2, [2, 0, 0])], "cubic")
    for keyframe in translation["keyframes"]:
        keyframe["in_tangent"] = [1, 0, 0]
        keyframe["out_tangent"] = [1, 0, 0]
    q0 = [1, 0, 0, 0]
    q1 = [cos(pi/4), 0, 0, sin(pi/4)]
    spec = {"duration": 2, "tracks": [
        translation,
        _track(oid, "rotation_quaternion", [(0, q0), (2, q1)], "slerp"),
        _track(oid, "scale", [(0, [1, 1, 1]), (2, [2, 2, 2])]),
    ]}
    doc.set_animation(spec, version=2)
    baseline = doc.as_dict()
    state = doc.evaluate(1)
    # Translation is applied before rotation and scaling in local space.
    transform = state.world_transform(oid)
    assert transform[3] == pytest.approx(1)
    assert transform[0] == pytest.approx(1.5*cos(pi/4))
    assert transform[4] == pytest.approx(1.5*sin(pi/4))
    assert doc.as_dict() == baseline
    restored = SceneDocument.from_bytes(doc.dumps())
    assert restored.animation() == spec
    assert restored.evaluate(1).world_transform(oid) == pytest.approx(transform)


def test_v2_agent_edits_preserve_schema_version():
    scene = Scene()
    oid = scene.add_object(name="animated")
    track = _track(oid, "scale", [(0, [1, 1, 1]), (1, [2, 2, 2])])
    scene.set_animation({"duration": 1, "tracks": [track]}, version=2)
    rev = scene.revision
    update_animation(scene, "set_interpolation", {
        "track_index": 0, "interpolation": "step",
    }, expected_revision=rev)
    assert "slai.animation.v2" in scene.metadata
    assert "slai.animation.v1" not in scene.metadata
    assert scene.metadata["slai.animation.v2"]["tracks"][0]["interpolation"] == "step"


def test_v2_rejects_conflicts_and_invalid_cubic_without_mutating_scene():
    scene = Scene()
    oid = scene.add_object(name="animated")
    spec = {"duration": 2, "tracks": [
        _track(oid, "scale", [(0, [1, 1, 1]), (2, [2, 2, 2])]),
    ]}
    scene.set_animation(spec, version=2)
    initial = deepcopy(scene.as_dict())
    invalid = deepcopy(spec)
    invalid["tracks"][0]["keyframes"][1]["value"] = [0, 2, 2]
    with pytest.raises(ValidationError):
        scene.set_animation(invalid)
    assert scene.as_dict() == initial
    invalid = deepcopy(spec)
    invalid["tracks"][0]["interpolation"] = "cubic"
    with pytest.raises(ValidationError):
        scene.set_animation(invalid)
    assert scene.as_dict() == initial


def test_v1_unchanged_and_v2_explicit_upgrade():
    scene = Scene()
    oid = scene.add_object(name="legacy")
    spec = {"duration": 2, "tracks": [
        _track(oid, "translation", [(0, [0, 0, 0]), (2, [2, 0, 0])])
    ]}
    scene.set_animation(spec)
    assert "slai.animation.v1" in scene.metadata
    assert scene.evaluate(1).world_transform(oid)[3] == pytest.approx(1)
    with pytest.raises(ValidationError):
        scene.set_animation({"duration": 2, "tracks": [
            _track(oid, "scale", [(0, [1, 1, 1]), (2, [2, 2, 2])])
        ]})
    scene.set_animation(spec, version=2)
    assert "slai.animation.v1" not in scene.metadata
    assert scene.evaluate(1).world_transform(oid)[3] == pytest.approx(1)


def test_shared_interpolation_primitives():
    assert cubic_hermite(0, 1, 2, 1, 0.5) == pytest.approx(1)
    q = slerp((1, 0, 0, 0), (0, 0, 0, 1), 0.5)
    assert sum(v*v for v in q) == pytest.approx(1)
