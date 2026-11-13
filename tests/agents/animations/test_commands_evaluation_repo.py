"""Run in SLAI repository ROOT with the updated files installed.

python -m unittest discover -s tests/agents/animations -p test_commands_evaluation_repo.py -v

These tests exercise real SGEO Scene, real Animation protocols/helpers, and
real command/evaluation code. They have NOT been executed against the full
SLAI tree in the authoring environment; see VALIDATION.md.
"""
from __future__ import annotations

import unittest
from copy import deepcopy

from src.agents.animations.animation_commands import apply_animation_batch, update_animation
from src.agents.animations.animation_evaluation import (
    AnimationEvaluationBatch, evaluate_animation, evaluate_animation_many,
    evaluate_animation_frames,
)
from src.agents.animations.animation_protocols import is_animation_scene_provider
from src.agents.animations.timeline.timebase import Timebase
from src.agents.animations.utils.animation_errors import (
    AnimationIndexError, AnimationRevisionConflictError,
    AnimationValidationError,
)
from src.utils.spatial_formats.sgeo.scene import Scene


def make_scene():
    scene = Scene()
    obj = scene.add_object(name="Animated marker")
    track = {"object_id": obj, "property": "translation", "interpolation": "linear",
             "keyframes": [{"time": 0., "value": [0., 0., 0.]},
                           {"time": 2., "value": [2., 0., 0.]}]}
    scene.set_animation({"duration": 2., "tracks": [track]})
    return scene, obj


class SGEOIntegration(unittest.TestCase):
    def test_provider_protocol(self):
        scene, _ = make_scene()
        self.assertTrue(is_animation_scene_provider(scene))

    def test_historical_single_edit_wire_shapes(self):
        scene, _ = make_scene()
        before = scene.revision
        payload = {"track_index": 0, "keyframe": {"time": 1., "value": [1., 0., 0.]}}
        self.assertEqual(update_animation(scene, "add_keyframe", payload, expected_revision=before, dry_run=True),
                         {"revision": before, "would_change": True})
        self.assertEqual(scene.revision, before)
        self.assertEqual(update_animation(scene, "add_keyframe", payload, expected_revision=before),
                         {"revision": before + 1})
        self.assertEqual([f["time"] for f in scene.metadata["slai.animation.v1"]["tracks"][0]["keyframes"]],
                         [0., 1., 2.])

    def test_track_management_and_duration(self):
        scene, oid = make_scene()
        second = {"object_id": oid, "property": "visibility", "interpolation": "step",
                  "keyframes": [{"time": 0., "value": True}]}
        update_animation(scene, "add_track", {"track": second}, expected_revision=scene.revision)
        self.assertEqual(len(scene.metadata["slai.animation.v1"]["tracks"]), 2)
        update_animation(scene, "set_duration", {"duration": 3.}, expected_revision=scene.revision)
        self.assertEqual(scene.metadata["slai.animation.v1"]["duration"], 3.)
        update_animation(scene, "delete_track", {"track_index": 1}, expected_revision=scene.revision)
        self.assertEqual(len(scene.metadata["slai.animation.v1"]["tracks"]), 1)

    def test_batch_one_commit(self):
        scene, _ = make_scene()
        before = scene.revision
        result = apply_animation_batch(scene, [
            {"operation": "add_keyframe", "payload": {"track_index": 0, "keyframe": {"time": .5, "value": [.5, 0., 0.]}}},
            {"operation": "add_keyframe", "payload": {"track_index": 0, "keyframe": {"time": 1., "value": [1., 0., 0.]}}},
        ], expected_revision=before)
        self.assertEqual(result, {"revision": before + 1, "operations_applied": 2})
        self.assertEqual(len(scene.metadata["slai.animation.v1"]["tracks"][0]["keyframes"]), 4)

    def test_failed_batch_is_atomic(self):
        scene, _ = make_scene()
        before = scene.revision
        baseline = deepcopy(scene.metadata["slai.animation.v1"])
        with self.assertRaises(Exception):
            apply_animation_batch(scene, [
                {"operation": "add_keyframe", "payload": {"track_index": 0, "keyframe": {"time": .5, "value": [.5, 0., 0.]}}},
                {"operation": "add_keyframe", "payload": {"track_index": 0, "keyframe": {"time": .5, "value": [8., 0., 0.]}}},
            ], expected_revision=before)
        self.assertEqual(scene.revision, before)
        self.assertEqual(scene.metadata["slai.animation.v1"], baseline)

    def test_conflicts_and_invalid_indices(self):
        scene, _ = make_scene()
        with self.assertRaises(AnimationRevisionConflictError):
            update_animation(scene, "clear_animation", {}, expected_revision=scene.revision - 1)
        with self.assertRaises(AnimationIndexError):
            update_animation(scene, "delete_keyframe", {"track_index": 0, "keyframe_index": -1})

    def test_single_evaluation_not_mutating(self):
        scene, obj = make_scene()
        before = scene.revision
        state = evaluate_animation(scene, 1., expected_revision=before)
        self.assertIn(obj, state.ids())
        self.assertEqual(state.revision, before)
        self.assertEqual(scene.revision, before)

    def test_batch_evaluation_and_frame_sampling(self):
        scene, _ = make_scene()
        batch = evaluate_animation_many(scene, [0., 1., 2.], expected_revision=scene.revision, lock_scene=True)
        self.assertIsInstance(batch, AnimationEvaluationBatch)
        self.assertEqual(batch.sample_count, 3)
        self.assertEqual([x.time for x in batch.states], [0., 1., 2.])
        frames = evaluate_animation_frames(scene, 0, 2, timebase=Timebase(24000, 1001))
        self.assertAlmostEqual(frames.timestamps[1], 1001 / 24000)

    def test_bounded_sampling(self):
        scene, _ = make_scene()
        with self.assertRaises(AnimationValidationError):
            evaluate_animation_many(scene, (float(x) for x in range(10_000)))


if __name__ == "__main__":
    unittest.main()
