"""Native animation events, synchronized clip choreography, and masked layers."""
from copy import deepcopy
import pytest

from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.transforms import translation_of
from src.utils.spatial_formats.sgeo.scene import Scene


def track(oid, x0, x1):
    return {"object_id": oid, "property": "translation",
            "keyframes": [{"time": 0, "value": [x0, 0, 0]},
                          {"time": 2, "value": [x1, 0, 0]}]}


def clip(x0, x1, *, duration=2):
    return {"id": new_id(), "duration": duration, "tracks": [
        {"property": "translation", "keyframes": [
            {"time": 0, "value": [x0, 0, 0]},
            {"time": duration, "value": [x1, 0, 0]}]},
    ]}


def test_events_are_bounded_pure_and_directional():
    scene = Scene()
    object_id = scene.add_object(name="marker")
    first, second = new_id(), new_id()
    events = [
        {"id": first, "name": "start", "time": 0, "object_id": object_id},
        {"id": second, "name": "contact", "time": 1},
    ]
    scene.set_animation({"duration": 2, "tracks": [], "events": events}, version=2)
    prior = deepcopy(scene.as_dict())
    assert [e["name"] for e in scene.animation_events(0, 1)] == ["contact"]
    assert [e["name"] for e in scene.animation_events(0, 1, include_start=True)] == ["start", "contact"]
    assert [e["name"] for e in scene.animation_events(1.5, 0)] == ["contact", "start"]
    assert scene.as_dict() == prior
    bad = deepcopy(prior["metadata"]["slai.animation.v2"])
    bad["events"].append({"id": first, "name": "repeated", "time": 1})
    with pytest.raises(ValidationError):
        scene.set_animation(bad)
    assert scene.as_dict() == prior


def test_choreography_follows_leader_normalized_phase():
    scene = Scene()
    leader = scene.add_object(name="leader")
    follower = scene.add_object(name="follower")
    leader_clip = clip(0, 2, duration=2)
    follower_clip = clip(0, 10, duration=5)
    lead = {"id": new_id(), "clip_id": leader_clip["id"], "object_id": leader, "start": 0}
    follow = {"id": new_id(), "clip_id": follower_clip["id"], "object_id": follower, "start": 0}
    spec = {"duration": 5, "tracks": [], "clips": [leader_clip, follower_clip],
            "instances": [lead, follow],
            "choreographies": [{"id": new_id(), "leader": lead["id"],
                                "followers": [{"instance_id": follow["id"]}]}]}
    scene.set_animation(spec, version=2)
    state = scene.evaluate(1)
    assert translation_of(state.world_transform(leader))[0] == pytest.approx(1)
    assert translation_of(state.world_transform(follower))[0] == pytest.approx(5)
    assert scene.metadata["slai.animation.v2"] == spec


def test_layer_mask_selects_only_declared_channels():
    scene = Scene()
    one = scene.add_object(name="one")
    two = scene.add_object(name="two")
    spec = {"duration": 2, "tracks": [
        track(one, 0, 0), track(two, 0, 0)
    ], "layers": [
        {"id": new_id(), "mode": "override", "weight": 1,
         "tracks": [track(one, 0, 2), track(two, 0, 20)],
         "mask": [one + ":translation"]}
    ]}
    scene.set_animation(spec, version=2)
    state = scene.evaluate(1)
    assert translation_of(state.world_transform(one))[0] == pytest.approx(1)
    assert translation_of(state.world_transform(two))[0] == pytest.approx(0)
    assert scene.metadata["slai.animation.v2"] == spec
