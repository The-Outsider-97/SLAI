"""P0-P2 native SGEO animation/kinematics regression contracts.

These tests exercise the existing public Scene and SceneDocument interfaces.
Numerical-optimizer tests require SLAI's configured SciPy optimization backend.
"""
from copy import deepcopy
from math import pi
from uuid import uuid4

import pytest

from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.common.transforms import from_translation
from src.utils.spatial_formats.sgeo.scene import Scene
from src.utils.spatial_formats.sgeo.document import SceneDocument


def uid():
    return str(uuid4())


def track(obj, prop, a, b, *, axis=None):
    result = {
        "object_id": obj, "property": prop, "interpolation": "linear",
        "keyframes": [{"time": 0, "value": a}, {"time": 2, "value": b}],
    }
    if axis is not None:
        result["axis"] = axis
    return result


def test_mechanical_driver_dag_updates_shaft_and_rack_without_duplicate_tracks():
    scene = Scene()
    motor = scene.add_object(name="motor")
    shaft = scene.add_object(name="shaft")
    rack = scene.add_object(name="rack")
    spec = {"duration": 2, "tracks": [
        track(motor, "rotation_angle", 0, 2, axis=[0, 0, 1]),
    ], "drivers": [
        # Intentionally out of order: should still topologically resolve.
        {"source_object_id": shaft, "source_property": "rotation_angle",
         "target_object_id": rack, "target_property": "translation",
         "factor": 2, "axis": [1, 0, 0]},
        {"source_object_id": motor, "source_property": "rotation_angle",
         "target_object_id": shaft, "target_property": "rotation_angle",
         "factor": -2, "axis": [0, 0, 1]},
    ]}
    scene.set_animation(spec, version=2)
    state = scene.evaluate(1)
    assert state.world_transform(shaft)[0] == pytest.approx(-0.4161468365)
    assert state.world_transform(rack)[3] == pytest.approx(-4)
    assert scene.metadata["slai.animation.v2"] == spec


def test_driver_cycle_refused_with_scene_rollback():
    scene = Scene()
    a, b = (scene.add_object(name=x) for x in ("a", "b"))
    spec = {"duration": 2, "tracks": [], "drivers": [
        {"source_object_id": b, "source_property": "rotation_angle",
         "target_object_id": a, "target_property": "rotation_angle",
         "factor": 1, "axis": [0, 0, 1]},
        {"source_object_id": a, "source_property": "rotation_angle",
         "target_object_id": b, "target_property": "rotation_angle",
         "factor": 1, "axis": [0, 0, 1]},
    ]}
    before = deepcopy(scene.as_dict())
    with pytest.raises(ValidationError, match="cycle"):
        scene.set_animation(spec, version=2)
    assert scene.as_dict() == before


def test_instanced_clip_bindings_reuse_one_definition_for_two_objects():
    scene = Scene()
    a, b = (scene.add_object(name=x) for x in ("a", "b"))
    clip = uid()
    spec = {
        "duration": 2, "tracks": [],
        "clips": [{
            "id": clip, "duration": 2,
            "tracks": [{
                "property": "translation", "interpolation": "linear",
                "keyframes": [
                    {"time": 0, "value": [0, 0, 0]},
                    {"time": 2, "value": [2, 0, 0]},
                ],
            }],
        }],
        "instances": [
            {"id": uid(), "clip_id": clip, "object_id": oid, "start": 0}
            for oid in (a, b)
        ],
    }
    scene.set_animation(spec, version=2)
    snap = scene.as_dict()
    result = scene.evaluate(1)
    assert result.world_transform(a)[3] == pytest.approx(1)
    assert result.world_transform(b)[3] == pytest.approx(1)
    assert scene.as_dict() == snap


def _arm():
    scene = Scene()
    root = scene.add_object(name="shoulder")
    second = scene.add_object(name="elbow", parent=root,
                              transform=from_translation(1, 0, 0))
    tip = scene.add_object(name="tool", parent=second,
                           transform=from_translation(1, 0, 0))
    joints = [
        {"id": uid(), "object_id": root, "kind": "revolute",
         "axis": [0, 0, 1], "lower": -pi, "upper": pi, "value": 0},
        {"id": uid(), "object_id": second, "kind": "revolute",
         "axis": [0, 0, 1], "lower": -pi, "upper": pi, "value": 0},
    ]
    scene.set_animation({"duration": 2, "tracks": [], "joints": joints}, version=2)
    return scene, root, second, tip, joints


def test_fk_matches_analytic_two_link_arm_and_joint_animation():
    scene, shoulder, elbow, tip, joints = _arm()
    root_pose = scene.forward_kinematics({shoulder: pi / 2})
    assert root_pose[tip][3] == pytest.approx(0, abs=1e-9)
    assert root_pose[tip][7] == pytest.approx(2, abs=1e-9)
    spec = deepcopy(scene.metadata["slai.animation.v2"])
    spec["tracks"] = [track(shoulder, "joint", 0, pi)]
    scene.set_animation(spec)
    at_one = scene.evaluate(1)
    assert at_one.world_transform(tip)[3] == pytest.approx(0, abs=1e-9)
    assert at_one.world_transform(tip)[7] == pytest.approx(2, abs=1e-9)


def test_joint_limit_violation_and_nonmutation():
    scene, shoulder, elbow, tip, joints = _arm()
    before = scene.as_dict()
    with pytest.raises(ValidationError, match="limits"):
        scene.forward_kinematics({shoulder: 10})
    assert scene.as_dict() == before


def test_position_constraints_solve_without_mutating_source():
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    scene = Scene()
    obj = scene.add_object(name="movable")
    spec = {"duration": 2, "tracks": [], "constraints": [
        {"id": uid(), "kind": "position", "object_id": obj,
         "target": [1, 2, 0], "movable": True, "tolerance": 1e-4},
    ]}
    scene.set_animation(spec, version=2)
    before = scene.as_dict()
    state = scene.evaluate(0)
    pose = state.world_transform(obj)
    assert pose[3] == pytest.approx(1, abs=1e-4)
    assert pose[7] == pytest.approx(2, abs=1e-4)
    assert state.constraint_report["status"] == "satisfied"
    assert scene.as_dict() == before


def test_fixed_position_constraint_is_provably_infeasible():
    scene = Scene()
    obj = scene.add_object(name="fixed")
    spec = {"duration": 2, "tracks": [], "constraints": [
        {"id": uid(), "kind": "position", "object_id": obj,
         "target": [5, 0, 0], "movable": False},
    ]}
    scene.set_animation(spec, version=2)
    with pytest.raises(ValidationError, match="constraints could not be satisfied"):
        scene.evaluate(0)


def test_inverse_kinematics_uses_existing_nonlinear_backend():
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    scene, shoulder, elbow, tip, joints = _arm()
    before = deepcopy(scene.as_dict())
    result = scene.solve_inverse_kinematics(tip, [1, 1, 0], tolerance=1e-3)
    assert result["status"] == "solved", result
    assert result["position_error"] <= 1e-3
    assert scene.as_dict() == before


def test_v2_dangling_references_prevent_deletion():
    scene, shoulder, elbow, tip, joints = _arm()
    before = scene.as_dict()
    with pytest.raises(ValidationError, match="joint"):
        scene.remove_object(elbow)
    assert scene.as_dict() == before


def test_version_upgrade_is_explicit_and_legacy_interpolation_unchanged():
    doc = SceneDocument.new()
    oid = doc.add_box(size=[1, 1, 1])
    legacy = {"duration": 2, "tracks": [track(oid, "translation", [0, 0, 0], [2, 0, 0])]}
    doc.set_animation(legacy)
    assert doc.evaluate(1).world_transform(oid)[3] == pytest.approx(1)
    doc.set_animation(legacy, version=2)
    restored = SceneDocument.from_bytes(doc.dumps())
    assert restored.animation() == legacy
    assert restored.evaluate(1).world_transform(oid)[3] == pytest.approx(1)
