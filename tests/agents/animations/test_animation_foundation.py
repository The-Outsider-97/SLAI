"""Real SGEO/AnimationAgent smoke checks."""
import unittest
from src.agents.animations.timeline.timebase import Timebase
from src.agents.animations.timeline.interpolation import lerp, slerp
from src.agents.animations.animation_memory import AnimationMemory, AnimationEditRecord
from src.utils.spatial_formats.sgeo.scene import Scene
from src.agents.animations.animation_commands import update_animation
from src.agents.animations.animation_evaluation import evaluate_animation

class TestAnimationFoundation(unittest.TestCase):
    def test_fractional_time(self):
        t = Timebase(24000, 1001)
        self.assertAlmostEqual(float(t.frame_seconds(24000)), 1001.0)

    def test_interpolation(self):
        self.assertEqual(lerp(0., 10., 0.5), 5.)
        self.assertAlmostEqual(sum(v*v for v in slerp((1., 0., 0., 0.), (0., 0., 0., 1.), .5)), 1.)

    def test_scene_revisions(self):
        scene = Scene()
        original = scene.revision
        spec = {"duration": 1.0, "tracks": []}
        result = update_animation(scene, "set_animation", {"animation": spec}, expected_revision=original)
        self.assertEqual(result["revision"], original + 1)
        self.assertEqual(evaluate_animation(scene, .5).revision, original + 1)
        with self.assertRaises(Exception):
            update_animation(scene, "clear_animation", {}, expected_revision=original)
        self.assertIn("slai.animation.v1", scene.metadata)

    def test_memory(self):
        memory = AnimationMemory(1)
        memory.record(AnimationEditRecord("a", "s", 1, 2))
        memory.record(AnimationEditRecord("b", "s", 2, 3))
        self.assertEqual(len(memory.recent()), 1)

if __name__ == "__main__":
    unittest.main()
