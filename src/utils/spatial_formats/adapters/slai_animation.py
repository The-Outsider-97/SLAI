"""SGEO-to-SLAI Animation bridge; no second animation engine or scene store.

SGEO owns animation schema validation, scene revisions and deterministic scene
evaluation. This adapter exposes the small, application-neutral provider
surface consumed by SLAI's Animation Agent. It does not import any SLAI agent,
so dependencies flow only from agents to spatial formats, never the reverse.

Notes
-----
``animation_edit_lock`` deliberately exposes the original SGEO scene RLock:
``set_animation`` is also locked and therefore requires a reentrant lock.
The adapter does not grant permissions; the application/orchestrator must
control who can bind a scene and issue mutations.
"""
from __future__ import annotations

from ..sgeo.animation import (ANIMATION_V2_KEY, evaluate_scene,
                              validate_animation, validate_animation_v2)
from ..sgeo.scene import Scene
from logs.logger import get_logger # type: ignore

logger = get_logger("SGEO SLAI Animation Adapter")


class SGEOAnimationAdapter:
    """Live, zero-copy reference to one authoritative SGEO scene."""

    def __init__(self, scene: Scene) -> None:
        if not isinstance(scene, Scene):
            raise TypeError("SGEOAnimationAdapter requires a native SGEO Scene")
        self._scene = scene
        logger.debug("SGEO animation provider bound | document_id=%s", scene.document_id)

    @property
    def document_id(self) -> str:
        return self._scene.document_id

    @property
    def revision(self) -> int:
        return self._scene.revision

    @property
    def metadata(self) -> dict:
        """Live metadata view; callers MUST hold animation_edit_lock when editing."""
        return self._scene.metadata

    @property
    def animation_edit_lock(self):
        """Provider-owned reentrant lock used for read/modify/write operations."""
        return self._scene._mutation_lock

    def evaluate(self, time: float):
        """Reuse native SGEO detached, read-only scene evaluation."""
        return self._scene.evaluate(time)

    def evaluate_at_revision(self, time: float, expected_revision: int):
        """Check the revision under SGEO's snapshot lock, not before evaluation."""
        # ConflictError is the native SGEO signal; the agent facade translates
        # it at its boundary, preserving the spatial-format dependency direction.
        return evaluate_scene(self._scene, time, expected_revision=expected_revision)

    def validate_animation(self, spec: dict) -> dict:
        """Delegate active version's complete schema validation to native SGEO."""
        with self._scene._mutation_lock:
            if ANIMATION_V2_KEY in self._scene.metadata:
                return validate_animation_v2(spec, self._scene)
            return validate_animation(spec, self._scene)

    def set_animation(self, animation: dict | None, *, version: int | None = None) -> int:
        """Delegate the only authoritative mutation and revision increment."""
        return self._scene.set_animation(animation, version=version)

    def forward_kinematics(self, values: dict | None = None):
        """Use SGEO's native FK and stable scene revision."""
        return self._scene.forward_kinematics(values)

    def solve_inverse_kinematics(self, end_effector, target, *,
                                 expected_revision=None, max_iterations=200,
                                 tolerance=1e-5):
        """Return an SGEO-produced IK proposal without mutating the scene."""
        return self._scene.solve_inverse_kinematics(
            end_effector, target, expected_revision=expected_revision,
            max_iterations=max_iterations, tolerance=tolerance,
        )


__all__ = ["SGEOAnimationAdapter"]
