"""Atomic SGEO animation edits: validate and replace, never mutate live tracks."""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping
from .utils.animation_errors import AnimationRevisionConflictError, AnimationUnsupportedOperationError
from src.utils.spatial_formats.sgeo.animation import ANIMATION_KEY, validate_animation

def update_animation(scene: Any, operation: str, payload: Mapping[str, Any], *, expected_revision: int | None = None, dry_run: bool = False):
    if expected_revision is not None and scene.revision != expected_revision:
        raise AnimationRevisionConflictError("scene revision mismatch")
    if not callable(getattr(scene, "set_animation", None)):
        raise AnimationUnsupportedOperationError("scene provider does not support animation transactions")
    current = deepcopy(scene.metadata.get(ANIMATION_KEY))
    if operation == "set_animation":
        proposed = deepcopy(payload["animation"])
    elif operation == "clear_animation":
        proposed = None
    elif operation in {"add_keyframe", "delete_keyframe", "update_keyframe"}:
        if current is None:
            raise AnimationUnsupportedOperationError("no animation exists")
        index = payload["track_index"]
        if type(index) is not int or not 0 <= index < len(current["tracks"]):
            raise ValueError("invalid track index")
        frames = current["tracks"][index]["keyframes"]
        if operation == "add_keyframe":
            frames.append(deepcopy(payload["keyframe"]))
            frames.sort(key=lambda k: k["time"])
        elif operation == "delete_keyframe":
            frames.pop(payload["keyframe_index"])
        else:
            frames[payload["keyframe_index"]] = deepcopy(payload["keyframe"])
            frames.sort(key=lambda k: k["time"])
        proposed = current
    else:
        raise AnimationUnsupportedOperationError(f"unsupported animation edit: {operation}")
    if proposed is not None:
        validate_animation(proposed, scene)
    if dry_run:
        return {"revision": scene.revision, "would_change": proposed != scene.metadata.get(ANIMATION_KEY)}
    # SGEO owns the atomic mutation lock and revision update. No preflight is an atomic CAS;
    # callers must serialize mutations or use a provider with a proper expected-revision operation.
    if expected_revision is not None:
        with scene._mutation_lock:
            if scene.revision != expected_revision:
                raise AnimationRevisionConflictError("scene revision mismatch")
            revision = scene.set_animation(proposed)
    else:
        revision = scene.set_animation(proposed)
    return {"revision": revision}
