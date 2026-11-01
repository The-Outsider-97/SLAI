"""Native deterministic interruption and continuity regression."""
from copy import deepcopy
import pytest

from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.transforms import translation_of
from src.utils.spatial_formats.sgeo.scene import Scene


def constant_clip(value):
    return {"id": new_id(), "duration": 4, "tracks": [{
        "property": "translation", "keyframes": [
            {"time": 0, "value": [value, 0, 0]},
            {"time": 4, "value": [value, 0, 0]},
        ],
    }]}


def test_interrupted_transition_freezes_previous_blend_and_recovers_destination():
    scene = Scene()
    oid = scene.add_object(name="transition")
    clips = [constant_clip(x) for x in (0, 10, 20)]
    instances = [
        {"id": new_id(), "clip_id": c["id"], "object_id": oid, "start": 0}
        for c in clips
    ]
    first = {
        "id": new_id(), "source": instances[0]["id"],
        "destination": instances[1]["id"], "start": 0, "duration": 2,
    }
    interrupt = {
        "id": new_id(), "source": instances[0]["id"],
        "destination": instances[2]["id"], "start": 1, "duration": 2,
        "interruption": "from_blended_pose",
    }
    spec = {"duration": 4, "tracks": [], "clips": clips,
            "instances": instances, "transitions": [first, interrupt]}
    scene.set_animation(spec, version=2)
    authored = deepcopy(scene.as_dict())
    def x_at(t):
        return translation_of(scene.evaluate(t).world_transform(oid))[0]
    assert x_at(0) == pytest.approx(0)
    assert x_at(0.5) == pytest.approx(2.5)
    assert x_at(1) == pytest.approx(5)
    assert x_at(2) == pytest.approx(12.5)
    assert x_at(3) == pytest.approx(20)
    assert x_at(2) == pytest.approx(12.5)
    assert scene.as_dict() == authored
    invalid = deepcopy(spec)
    invalid["transitions"][1].pop("interruption")
    with pytest.raises(ValidationError, match="overlapping"):
        scene.set_animation(invalid)
    assert scene.as_dict() == authored
