"""Two-joint retargeting generates synchronized native clips without editing scenes."""
from copy import deepcopy
from math import pi
import pytest

from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.transforms import from_translation
from src.utils.spatial_formats.sgeo.scene import Scene


def articulated(length):
    scene = Scene()
    root = scene.add_object(name="root")
    child = scene.add_object(name="child", parent=root,
                             transform=from_translation(length, 0, 0))
    joints = [
        {"id": new_id(), "object_id": oid, "kind": "revolute",
         "axis": [0, 0, 1], "lower": -pi, "upper": pi, "value": 0}
        for oid in (root, child)
    ]
    scene.set_animation({"duration": 2, "tracks": [], "joints": joints}, version=2)
    return scene, root, child


def test_rig_retarget_proposes_two_native_synchronized_clip_instances():
    source, sroot, schild = articulated(1)
    target, troot, tchild = articulated(2)
    spec = deepcopy(source.metadata["slai.animation.v2"])
    lookup = {}
    for oid, target_angle in ((sroot, pi/2), (schild, pi/4)):
        cid = new_id()
        lookup[oid] = cid
        spec.setdefault("clips", []).append({
            "id": cid, "duration": 2, "tracks": [
                {"property": "joint", "keyframes": [
                    {"time": 0, "value": 0}, {"time": 2, "value": target_angle},
                ]}
            ],
        })
        spec.setdefault("instances", []).append({
            "id": new_id(), "clip_id": cid, "object_id": oid, "start": 0
        })
    source.set_animation(spec)
    source_snapshot = deepcopy(source.as_dict())
    target_snapshot = deepcopy(target.as_dict())
    proposed = target.retarget_rig_motion(
        source, lookup, {sroot: troot, schild: tchild}
    )
    assert proposed["status"] == "proposed"
    assert len(proposed["clip_ids"]) == 2
    assert len(proposed["animation"]["choreographies"]) == 1
    assert source.as_dict() == source_snapshot
    assert target.as_dict() == target_snapshot
    target.set_animation(proposed["animation"])
    evaluated = target.evaluate(2)
    assert evaluated.world_transform(troot)[0] == pytest.approx(0, abs=1e-8)
    assert evaluated.world_transform(tchild)[3] == pytest.approx(0, abs=1e-8)
    assert evaluated.world_transform(tchild)[7] == pytest.approx(2, abs=1e-8)
