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
    scene = Scene()
    obj = scene.add_object(name="standalone_joint")
    joints = [{"id": uid(), "object_id": obj, "kind": "revolute",
               "axis": [0, 0, 1], "lower": -pi, "upper": pi, "value": 0}]
    scene.set_animation({"duration": 1, "tracks": [], "joints": joints}, version=2)
    before = scene.as_dict()
    with pytest.raises(ValidationError, match="joint"):
        scene.remove_object(obj)
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


def test_adapter_v2_edit_and_validation_contract():
    from src.utils.spatial_formats.adapters.slai_animation import SGEOAnimationAdapter
    from src.agents.animations.animation_commands import update_animation
    scene = Scene()
    oid = scene.add_object(name="adapter_object")
    spec = {"duration": 2, "tracks": [
        track(oid, "scale", [1, 1, 1], [2, 2, 2]),
    ]}
    scene.set_animation(spec, version=2)
    adapter = SGEOAnimationAdapter(scene)
    assert adapter.validate_animation(spec) == spec
    result = update_animation(adapter, "set_interpolation",
                              {"track_index": 0, "interpolation": "step"},
                              expected_revision=scene.revision)
    assert result["revision"] == scene.revision
    assert scene.metadata["slai.animation.v2"]["tracks"][0]["interpolation"] == "step"


def test_unused_clip_malformed_data_rejected_before_scene_mutation():
    scene = Scene()
    old = scene.as_dict()
    spec = {"duration": 2, "tracks": [], "clips": [{
        "id": uid(), "duration": 1,
        "tracks": [{
            "property": "translation", "interpolation": "linear",
            "keyframes": [{"time": 0, "value": [1, float("nan"), 1]}],
        }],
    }]}
    with pytest.raises(ValidationError):
        scene.set_animation(spec, version=2)
    assert scene.as_dict() == old


def test_constraint_activation_interval():
    scene = Scene()
    oid = scene.add_object(name="moving")
    spec = {"duration": 2, "tracks": [], "constraints": [
        {"id": uid(), "kind": "position", "object_id": oid,
         "movable": True, "target": [2, 0, 0],
         "start": 1, "end": 2},
    ]}
    scene.set_animation(spec, version=2)
    assert scene.evaluate(0).world_transform(oid)[3] == pytest.approx(0)
    assert scene.evaluate(1.5).world_transform(oid)[3] == pytest.approx(2)


def test_conflicting_analytic_position_constraints_report_infeasibility():
    scene = Scene()
    oid = scene.add_object(name="overconstrained")
    spec = {"duration": 2, "tracks": [], "constraints": [
        {"id": uid(), "kind": "position", "object_id": oid,
         "movable": True, "target": [1, 0, 0], "tolerance": 1e-7},
        {"id": uid(), "kind": "position", "object_id": oid,
         "movable": True, "target": [2, 0, 0], "tolerance": 1e-7},
    ]}
    scene.set_animation(spec, version=2)
    with pytest.raises(ValidationError, match="constraints could not be satisfied") as exc:
        scene.evaluate(1)
    assert exc.value.context["constraint_report"]["status"] == "infeasible"


def test_ik_stretched_arm_recovery_requires_bounded_multistart():
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    scene, shoulder, elbow, tip, joints = _arm()
    outcome = scene.solve_inverse_kinematics(tip, [1, 0, 0],
                                             max_iterations=250, tolerance=2e-3)
    assert outcome["status"] == "solved", outcome
    assert outcome["position_error"] <= 2e-3


def test_scale_crossing_zero_is_rejected_before_animation_commit():
    scene = Scene()
    oid = scene.add_object(name="scale")
    initial = scene.as_dict()
    spec = {"duration": 2, "tracks": [
        track(oid, "scale", [-1, 1, 1], [1, 1, 1]),
    ]}
    with pytest.raises(ValidationError, match="singular"):
        scene.set_animation(spec, version=2)
    assert scene.as_dict() == initial


def test_cubic_joint_overshoot_rejected_analytically():
    scene, shoulder, elbow, tip, joints = _arm()
    candidate = deepcopy(scene.metadata["slai.animation.v2"])
    candidate["tracks"] = [
        {"object_id": shoulder, "property": "joint", "interpolation": "cubic",
         "keyframes": [
             {"time": 0, "value": 0, "in_tangent": 100, "out_tangent": 100},
             {"time": 2, "value": 0, "in_tangent": -100, "out_tangent": -100},
         ]}
    ]
    baseline = scene.as_dict()
    with pytest.raises(ValidationError, match="bound"):
        scene.set_animation(candidate)
    assert scene.as_dict() == baseline


def test_native_ik_pose_commit_changes_exactly_one_revision():
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    scene, shoulder, elbow, tip, joints = _arm()
    before = scene.revision
    committed = scene.apply_inverse_kinematics(
        tip, [1, 1, 0], expected_revision=before, tolerance=1e-3,
    )
    assert committed["status"] == "solved", committed
    assert committed["committed"] is True
    assert committed["revision"] == before + 1
    assert scene.revision == before + 1
    evaluated = scene.evaluate(0).world_transform(tip)
    assert evaluated[3] == pytest.approx(1, abs=1e-3)
    assert evaluated[7] == pytest.approx(1, abs=1e-3)


def test_native_ik_pose_commit_rejects_stale_revision_without_mutation():
    scene, shoulder, elbow, tip, joints = _arm()
    spec = deepcopy(scene.metadata["slai.animation.v2"])
    spec["tracks"] = [track(shoulder, "joint", 0, 1)]
    scene.set_animation(spec)
    prior = scene.as_dict()
    from src.utils.spatial_formats.common.errors import ConflictError
    with pytest.raises(ConflictError):
        scene.apply_inverse_kinematics(tip, [1, 1, 0],
                                       expected_revision=scene.revision - 1)
    assert scene.as_dict() == prior


def test_coupled_attachment_constraints_and_rigid_reference():
    pytest.importorskip("scipy")
    scene = Scene()
    anchor = scene.add_object(name="anchor", transform=from_translation(2, 0, 0))
    follower = scene.add_object(name="follower")
    cid = uid()
    spec = {"duration": 2, "tracks": [], "constraints": [
        {"id": cid, "kind": "attachment", "object_id": follower,
         "other_id": anchor, "offset": [0, 1, 0],
         "movable": True, "tolerance": 1e-3},
    ]}
    scene.set_animation(spec, version=2)
    original = scene.as_dict()
    sampled = scene.evaluate(0.5)
    xyz = sampled.world_transform(follower)
    assert xyz[3] == pytest.approx(2, abs=1e-3)
    assert xyz[7] == pytest.approx(1, abs=1e-3)
    assert sampled.constraint_report["constraints"][0]["id"] == cid
    assert scene.as_dict() == original


def test_bounded_world_position_constraints():
    pytest.importorskip("scipy")
    scene = Scene()
    obj = scene.add_object(name="bounded")
    scene.set_animation({
        "duration": 2, "tracks": [], "constraints": [
            {"id": uid(), "kind": "bounds", "object_id": obj, "movable": True,
             "minimum": [1, -1, -1], "maximum": [2, 1, 1],
             "tolerance": 1e-3},
        ],
    }, version=2)
    evaluated = scene.evaluate(0)
    x = evaluated.world_transform(obj)[3]
    assert 1 - 1e-3 <= x <= 2 + 1e-3
    assert evaluated.constraint_report["status"] == "satisfied"


def test_alignment_between_two_root_objects():
    pytest.importorskip("scipy")
    scene = Scene()
    anchor = scene.add_object(name="anchor", transform=from_translation(0, 0, 0))
    target = scene.add_object(name="align_target", transform=from_translation(1, 2, 0))
    scene.set_animation({
        "duration": 2, "tracks": [], "constraints": [
            {"id": uid(), "kind": "alignment", "object_id": target,
             "other_id": anchor, "axis": [1, 0, 0],
             "movable": True, "tolerance": 1e-3},
        ],
    }, version=2)
    evaluated = scene.evaluate(1)
    result = evaluated.world_transform(target)
    assert result[7] == pytest.approx(0, abs=1e-3)
    assert result[11] == pytest.approx(0, abs=1e-3)


def test_conflicting_immovable_distance_constraint_reports_infeasible():
    scene = Scene()
    a = scene.add_object(name="a")
    b = scene.add_object(name="b", transform=from_translation(2, 0, 0))
    scene.set_animation({"duration": 1, "tracks": [], "constraints": [
        {"id": uid(), "kind": "distance", "object_id": a,
         "other_id": b, "distance": 1, "movable": False,
         "tolerance": 1e-6},
    ]}, version=2)
    with pytest.raises(ValidationError) as exc:
        scene.evaluate(0.5)
    assert exc.value.context["constraint_report"]["status"] == "infeasible"


def test_unsupported_constraint_kind_fails_closed():
    scene = Scene()
    obj = scene.add_object()
    before = scene.as_dict()
    with pytest.raises(ValidationError, match="unsupported"):
        scene.set_animation({"duration": 1, "tracks": [], "constraints": [
            {"id": uid(), "kind": "collision", "object_id": obj,
             "movable": True},
        ]}, version=2)
    assert scene.as_dict() == before


def test_native_animation_feature_manifest_round_trip():
    doc = SceneDocument.new()
    oid = doc.add_box(size=[1, 1, 1])
    doc.set_animation({"duration": 2, "tracks": [
        track(oid, "translation", [0, 0, 0], [2, 0, 0]),
    ]}, version=2)
    manifest = doc.manifest()
    assert "animation-v2" in manifest["requires"]
    loaded = SceneDocument.from_bytes(doc.dumps())
    assert "slai.animation.v2" in loaded.snapshot().metadata
    assert loaded.evaluate(1).world_transform(oid)[3] == pytest.approx(1)


def test_ik_unreachable_reports_failure_without_scene_edit():
    pytest.importorskip("scipy")
    scene, shoulder, elbow, tip, joints = _arm()
    original = scene.as_dict()
    proposal = scene.solve_inverse_kinematics(tip, [100, 100, 0],
                                               max_iterations=30, tolerance=1e-4)
    assert proposal["status"] != "solved"
    assert proposal["position_error"] > 1e-4
    assert scene.as_dict() == original


def test_explicit_v1_migration_preserves_evaluation_and_revisions():
    scene = Scene()
    oid = scene.add_object(name="legacy")
    legacy = {"duration": 2, "tracks": [
        track(oid, "translation", [0, 0, 0], [2, 0, 0]),
    ]}
    scene.set_animation(legacy)
    baseline = scene.evaluate(0.75).world_transform(oid)
    before = scene.revision
    migrated = scene.upgrade_animation(expected_revision=before)
    assert migrated == before + 1
    assert "slai.animation.v1" not in scene.metadata
    assert scene.metadata["slai.animation.v2"] == legacy
    assert scene.evaluate(0.75).world_transform(oid) == pytest.approx(baseline)
    with pytest.raises(ValidationError, match="already v2"):
        scene.upgrade_animation(expected_revision=migrated)


def test_agent_adapter_explicit_v1_migration():
    from src.utils.spatial_formats.adapters.slai_animation import SGEOAnimationAdapter
    scene = Scene()
    oid = scene.add_object()
    scene.set_animation({"duration": 1, "tracks": [
        {"object_id": oid, "property": "visibility", "interpolation": "step",
         "keyframes": [{"time": 0, "value": True}]},
    ]})
    adapter = SGEOAnimationAdapter(scene)
    previous_revision = adapter.revision
    assert adapter.upgrade_animation(expected_revision=previous_revision) == previous_revision + 1
    assert "slai.animation.v2" in adapter.metadata
    assert adapter.evaluate(0).visible(oid)


def test_v2_rejects_noncanonical_id_aliases_before_commit():
    scene = Scene()
    obj = scene.add_object()
    spec = {"duration": 2, "tracks": [
        track(obj.upper(), "translation", [0, 0, 0], [2, 0, 0]),
    ]}
    original = scene.as_dict()
    with pytest.raises(ValidationError, match="canonical"):
        scene.set_animation(spec, version=2)
    assert scene.as_dict() == original


def test_repeated_arbitrary_time_evaluations_are_deterministic():
    scene = Scene()
    oid = scene.add_object(name="cyclic-safe")
    scene.set_animation({"duration": 2, "tracks": [
        track(oid, "translation", [0, 0, 0], [2, 0, 0]),
    ]}, version=2)
    initial = scene.as_dict()
    snapshots = [
        scene.evaluate(t).world_transform(oid)
        for t in (0, 0.25, 0.5, 1, 1.5, 2, 1, 0.25)
    ]
    assert snapshots[3] == snapshots[6]
    assert snapshots[1] == snapshots[7]
    assert scene.as_dict() == initial


def test_many_track_eval_budget_and_repeatability():
    from time import perf_counter
    scene = Scene()
    ids = [scene.add_object(name="node_%d" % i) for i in range(80)]
    frames = [{"time": float(i), "value": [float(i), 0, 0]}
              for i in range(100)]
    scene.set_animation({
        "duration": 99.0,
        "tracks": [{
            "object_id": oid, "property": "translation",
            "interpolation": "linear", "keyframes": frames,
        } for oid in ids],
    }, version=2)
    before = scene.as_dict()
    started = perf_counter()
    snapshots = [scene.evaluate(t) for t in (0, 33.25, 78.5, 99, 33.25)]
    elapsed = perf_counter() - started
    assert elapsed >= 0
    assert snapshots[1].world_transform(ids[-1]) == snapshots[4].world_transform(ids[-1])
    assert snapshots[1].world_transform(ids[-1])[3] == pytest.approx(33.25)
    assert scene.as_dict() == before


def test_coincident_distance_constraint_analytical_projection():
    scene = Scene()
    source = scene.add_object(name="movable")
    anchor = scene.add_object(name="fixed")
    scene.set_animation({"duration": 2, "tracks": [], "constraints": [
        {"id": uid(), "kind": "distance", "object_id": source,
         "other_id": anchor, "distance": 2.0, "movable": True,
         "tolerance": 1e-8},
    ]}, version=2)
    snapshot = scene.evaluate(0)
    assert snapshot.world_transform(source)[3] == pytest.approx(2)
    assert snapshot.constraint_report["solver_status"] == "analytic"
    assert scene.world_transform(source)[3] == pytest.approx(0)


def test_single_bounds_constraint_analytical_projection():
    scene = Scene()
    source = scene.add_object(name="movable")
    scene.set_animation({"duration": 1, "tracks": [], "constraints": [
        {"id": uid(), "kind": "bounds", "object_id": source,
         "minimum": [1, 2, 3], "maximum": [4, 5, 6], "movable": True},
    ]}, version=2)
    result = scene.evaluate(0)
    assert tuple(result.world_transform(source)[i] for i in (3, 7, 11)) == pytest.approx((1, 2, 3))
    assert result.constraint_report["solver_status"] == "analytic"


def test_sequential_reusable_clips_share_target_without_overlapping():
    scene = Scene()
    oid = scene.add_object(name="sequential")
    cid = uid()
    clip = {
        "id": cid, "duration": 1, "tracks": [{
            "property": "translation", "interpolation": "linear",
            "keyframes": [
                {"time": 0, "value": [0, 0, 0]},
                {"time": 1, "value": [1, 0, 0]},
            ],
        }],
    }
    first, second = (
        {"id": uid(), "clip_id": cid, "object_id": oid, "start": start}
        for start in (0, 1.1)
    )
    spec = {"duration": 2.2, "tracks": [], "clips": [clip],
            "instances": [second, first]}
    scene.set_animation(spec, version=2)
    assert scene.evaluate(.5).world_transform(oid)[3] == pytest.approx(.5)
    assert scene.evaluate(1.6).world_transform(oid)[3] == pytest.approx(.5)
    assert scene.evaluate(1.05).world_transform(oid)[3] == pytest.approx(0)
    bad = {**spec, "instances": [
        first,
        {**second, "start": 0.5},
    ]}
    baseline = scene.as_dict()
    with pytest.raises(ValidationError, match="overlapping"):
        scene.set_animation(bad)
    assert scene.as_dict() == baseline
