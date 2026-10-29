"""Thin evaluation adapter: SGEO remains the authoritative animation evaluator."""
from __future__ import annotations
from typing import Any
from ..animations.utils.animation_errors import AnimationRevisionConflictError, AnimationUnsupportedOperationError

def evaluate_animation(scene: Any, seconds: float, *, expected_revision: int | None = None):
    if not callable(getattr(scene, "evaluate", None)):
        raise AnimationUnsupportedOperationError("scene provider does not implement evaluate")
    if expected_revision is not None and scene.revision != expected_revision:
        raise AnimationRevisionConflictError("scene revision mismatch")
    return scene.evaluate(seconds)
