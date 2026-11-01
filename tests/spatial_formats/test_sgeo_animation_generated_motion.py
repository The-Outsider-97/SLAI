"""Kinematic native motion proposals use the established SGEO IK solver."""
from copy import deepcopy
from math import pi

import pytest

from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.transforms import from_translation
from src.utils.spatial_formats.sgeo.scene import Scene


def test_generate_joint_motion_is_nonmutating_native_and_endpoint_verified():
    pytest.importorskip("scipy")
    pytest.importorskip("numpy")
    scene = Scene()
    root = scene.add_object(name="shoulder")
    elbow = scene.add_object(name="elbow", parent=root,
                             transform=from_translation(1, 0, 0))
    tip = scene.add_object(name="tool", parent=elbow,
                           transform=from_translation(1, 0, 0))
    joints = [
        {"id": new_id(), "object_id": oid, "kind": "revolute",
         "axis": [0, 0, 1], "lower": -pi, "upper": pi, "value": 0}
        for oid in (root, elbow)
    ]
    scene.set_animation({"duration": 2, "tracks": [], "joints": joints}, version=2)
    original = deepcopy(scene.as_dict())
    proposal = scene.generate_joint_motion(tip, [1, 1, 0], 2, max_iterations=300, tolerance=1e-3)
    assert proposal["status"] == "proposed"
    assert proposal["endpoint_position_error"] <= 1e-3
    assert len(proposal["animation"]["tracks"]) == 2
    assert all(track["interpolation"] == "cubic" for track in proposal["animation"]["tracks"])
    assert scene.as_dict() == original
    previous = scene.revision
    scene.set_animation(proposal["animation"], version=2)
    assert scene.revision == previous + 1
    pose = scene.evaluate(2).world_transform(tip)
    assert [pose[3], pose[7], pose[11]] == pytest.approx([1, 1, 0], abs=1e-3)
