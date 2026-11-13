"""Native scene transaction journal for animation-only edits."""
from copy import deepcopy
import pytest

from src.utils.spatial_formats.common.errors import ConflictError, ValidationError
from src.utils.spatial_formats.sgeo.scene import Scene


def test_undo_redo_animation_is_revision_monotonic_and_non_destructive():
    scene = Scene()
    oid = scene.add_object(name="preserve")
    first = {"duration": 1, "tracks": []}
    second = {"duration": 2, "tracks": []}
    scene.set_animation(first)
    scene.set_animation(second)
    expected = scene.revision
    assert scene.undo_animation(expected_revision=expected) == expected + 1
    assert scene.metadata["slai.animation.v1"] == first
    assert scene.undo_animation(expected_revision=scene.revision) == expected + 2
    assert "slai.animation.v1" not in scene.metadata
    assert scene.redo_animation(expected_revision=scene.revision) == expected + 3
    assert scene.metadata["slai.animation.v1"] == first
    assert scene.redo_animation(expected_revision=scene.revision) == expected + 4
    assert scene.metadata["slai.animation.v1"] == second
    assert scene.objects[0]["id"] == oid


def test_animation_history_refuses_unrelated_interleaved_scene_changes():
    scene = Scene()
    scene.add_object(name="existing")
    scene.set_animation({"duration": 1, "tracks": []})
    scene.add_object(name="later")
    snapshot = deepcopy(scene.as_dict())
    with pytest.raises(ConflictError, match="intervening"):
        scene.undo_animation(expected_revision=scene.revision)
    assert scene.as_dict() == snapshot
    with pytest.raises(ValidationError):
        scene.redo_animation(expected_revision=scene.revision)


def test_clearing_animation_has_reversible_versioned_state():
    scene = Scene()
    oid = scene.add_object(name="box")
    track = {"object_id": oid, "property": "scale",
             "keyframes": [{"time": 0, "value": [1, 1, 1]},
                           {"time": 1, "value": [2, 2, 2]}]}
    scene.set_animation({"duration": 1, "tracks": [track]}, version=2)
    scene.set_animation(None)
    assert "slai.animation.v2" not in scene.metadata
    scene.undo_animation(expected_revision=scene.revision)
    assert "slai.animation.v2" in scene.metadata
