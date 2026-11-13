"""Bounded explicit SGEO joint-channel retarget regression tests."""
from copy import deepcopy
from math import pi
import pytest

from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.sgeo.scene import Scene


def rig(axis, default=0, lower=-pi, upper=pi):
    scene = Scene()
    oid = scene.add_object(name="joint")
    joint = {"id": new_id(), "object_id": oid, "kind": "revolute",
             "axis": axis, "lower": lower, "upper": upper, "value": default}
    scene.set_animation({"duration": 2, "tracks": [], "joints": [joint]}, version=2)
    return scene, oid


def test_retarget_single_joint_clip_with_reversed_axis_and_no_mutation():
    source, original_joint = rig([0, 0, 1])
    target, target_joint = rig([0, 0, -1], default=0.25)
    cid = new_id()
    authored = deepcopy(source.metadata["slai.animation.v2"])
    authored["clips"] = [{"id": cid, "duration": 2, "tracks": [{
        "property": "joint", "keyframes": [
            {"time": 0, "value": 0}, {"time": 2, "value": 1},
        ],
    }]}]
    authored["instances"] = [
        {"id": new_id(), "clip_id": cid, "object_id": original_joint, "start": 0}
    ]
    source.set_animation(authored)
    source_state = deepcopy(source.as_dict())
    target_state = deepcopy(target.as_dict())
    proposal = target.retarget_joint_clip(
        source, cid, {original_joint: target_joint}
    )
    assert proposal["status"] == "proposed"
    assert proposal["target_object_id"] == target_joint
    assert proposal["clip"]["tracks"][0]["keyframes"][1]["value"] == pytest.approx(-0.75)
    assert source.as_dict() == source_state
    assert target.as_dict() == target_state
    # The caller explicitly commits the proposed native clip/binding.
    spec = deepcopy(target.metadata["slai.animation.v2"])
    spec["clips"] = [proposal["clip"]]
    spec["instances"] = [{"id": new_id(), "clip_id": proposal["clip"]["id"],
                          "object_id": target_joint, "start": 0}]
    target.set_animation(spec)
    assert target.evaluate(2).world_transform(target_joint)[0] == pytest.approx(
        __import__("math").cos(-0.75), abs=1e-8
    )


def test_retarget_refuses_incompatible_local_axes():
    source, sid = rig([0, 0, 1])
    target, tid = rig([0, 1, 0])
    cid = new_id()
    spec = deepcopy(source.metadata["slai.animation.v2"])
    spec["clips"] = [{"id": cid, "duration": 2, "tracks": [
        {"property": "joint", "keyframes": [
            {"time": 0, "value": 0}, {"time": 2, "value": 1}
        ]}
    ]}]
    spec["instances"] = [{"id": new_id(), "clip_id": cid, "object_id": sid, "start": 0}]
    source.set_animation(spec)
    unchanged = deepcopy(target.as_dict())
    with pytest.raises(ValidationError, match="axes"):
        target.retarget_joint_clip(source, cid, {sid: tid})
    assert target.as_dict() == unchanged
