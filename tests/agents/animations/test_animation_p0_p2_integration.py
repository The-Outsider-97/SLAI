"""AnimationAgent ↔ SGEO native v2 P0/P2 operational dispatch regression tests."""
import pytest

from src.agents.animation_agent import AnimationAgent
from src.utils.spatial_formats.sgeo.scene import Scene
from src.utils.spatial_formats.adapters.slai_animation import SGEOAnimationAdapter
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.transforms import from_translation


def test_agent_dispatches_v1_to_v2_migration_and_versioned_queries():
    scene = Scene()
    object_id = scene.add_object()
    scene.set_animation({
        "duration": 2, "tracks": [{
            "object_id": object_id, "property": "translation",
            "interpolation": "linear",
            "keyframes": [
                {"time": 0, "value": [0, 0, 0]},
                {"time": 2, "value": [2, 0, 0]},
            ],
        }],
    })
    agent = AnimationAgent()
    scene_id = agent.bind_scene(SGEOAnimationAdapter(scene))
    old_revision = scene.revision
    response = agent.perform_task({
        "operation": "upgrade_animation", "scene_id": scene_id,
        "expected_revision": old_revision,
    })
    assert response["status"] == "ok"
    assert response["revision"] == old_revision + 1
    queried = agent.perform_task({"operation": "get_animation", "scene_id": scene_id})
    assert queried["animation"] == scene.metadata["slai.animation.v2"]
    sampled = agent.perform_task({
        "operation": "evaluate", "scene_id": scene_id, "payload": {"time": 1.0},
    })
    assert sampled["revision"] == scene.revision


def test_agent_dispatches_native_fk_and_constrained_ik_proposal():
    pytest.importorskip("scipy")
    scene = Scene()
    shoulder = scene.add_object(name="shoulder")
    elbow = scene.add_object(name="elbow", parent=shoulder,
                             transform=from_translation(1, 0, 0))
    tip = scene.add_object(name="tool", parent=elbow,
                           transform=from_translation(1, 0, 0))
    scene.set_animation({"duration": 2, "tracks": [], "joints": [
        {"id": new_id(), "object_id": shoulder,
         "kind": "revolute", "axis": [0, 0, 1],
         "lower": -3.14, "upper": 3.14, "value": 0},
        {"id": new_id(), "object_id": elbow,
         "kind": "revolute", "axis": [0, 0, 1],
         "lower": -3.14, "upper": 3.14, "value": 0},
    ]}, version=2)
    agent = AnimationAgent()
    scene_id = agent.bind_scene(SGEOAnimationAdapter(scene))
    before = scene.revision
    fk = agent.perform_task({"operation": "forward_kinematics", "scene_id": scene_id})
    assert fk["revision"] == before
    assert fk["object_count"] == 3
    ik = agent.perform_task({
        "operation": "solve_ik", "scene_id": scene_id,
        "expected_revision": before,
        "payload": {"end_effector": tip, "target": [1, 1, 0], "tolerance": 1e-3},
    })
    assert ik["status"] == "solved", ik
    assert scene.revision == before
    applied = agent.perform_task({
        "operation": "apply_ik", "scene_id": scene_id,
        "expected_revision": before,
        "payload": {"end_effector": tip, "target": [1, 1, 0], "tolerance": 1e-3},
    })
    assert applied["status"] == "solved", applied
    assert applied["committed"] is True
    assert scene.revision == before + 1
