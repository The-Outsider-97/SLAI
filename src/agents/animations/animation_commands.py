"""Transactional, revision-aware editing of declarative animation documents.

SGEO (or a compliant scene provider) remains the *sole* authoritative owner
of animation data, schema validation and the commit/revision operation. This
module stages independent candidate dictionaries and delegates the final edit
via ``set_animation``. No scene state, keyframe cache, undo history, or
interpolation implementation lives here.

Supported historical commands: ``set_animation``, ``clear_animation``,
``add_keyframe``, ``update_keyframe``, ``delete_keyframe``. Additional direct
commands: ``add_track``, ``update_track``, ``delete_track``,
``set_interpolation`` and ``set_duration``. The latter commands require the
AnimationAgent's dispatch catalogue to be extended before that facade can
expose them; the base five are backward-compatible immediately.

Atomic multi-edit staging is exposed as ``apply_animation_batch``. Each batch
causes at most one provider commit and one scene revision increment. Provider
validation errors do not mutate SGEO, because its ``set_animation`` stages a
candidate and commits only after validation succeeds.

An edit requires the provider's *reentrant* edit lock (as promised by the
protocol). This lock must be the same lock used for authoritative scene writes;
structural protocol checks cannot prove that requirement. No lock means an
explicit refusal rather than unsafe optimistic-concurrency claims.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils.spatial_formats.sgeo.animation import (ANIMATION_KEY, ANIMATION_V2_KEY,
                                                       validate_animation, validate_animation_v2) # type: ignore
from src.utils.spatial_formats.sgeo.scene import Scene as SGEOScene # type: ignore
from .animation_protocols import get_animation_edit_lock, is_animation_scene_provider
from .animation_types import EDIT_OPERATIONS
from .utils.config_loader import get_config_section
from .utils.animation_errors import *
from .utils.animation_helpers import *
from logs.logger import get_logger # type: ignore

logger = get_logger("Animation Commands")
# _CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "animation_config.yaml" # config loader already has DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "animation_config.yaml"

# The historical five operations are already exposed by AnimationAgent and
# animation_types. Extended operations are opt-in until those dispatchers
# explicitly register them, rather than silently changing their API.
EXTENDED_EDITS = frozenset({
    "add_track", "update_track", "delete_track",
    "set_interpolation", "set_duration",
    "set_animation_v2", "add_clip", "update_clip", "delete_clip", "add_instance", "update_instance",
    "delete_instance", "add_layer", "update_layer", "delete_layer",
    "add_transition", "update_transition", "delete_transition",
    "add_choreography", "update_choreography", "delete_choreography",
    "add_event", "update_event", "delete_event",
})
SUPPORTED_EDITS = EDIT_OPERATIONS | EXTENDED_EDITS


def _settings() -> tuple[bool, int]:
    """Read edit policy from the existing Animation configuration facade."""
    config = get_config_section("animation")
    values = config.get("commands")
    if not isinstance(values, Mapping):
        raise AnimationConfigurationError(
            "animation.commands must be configured in animation_config.yaml"
        )
    strict = values.get("require_expected_revision")
    batch_limit = values.get("max_batch_operations")
    if type(strict) is not bool:
        raise AnimationConfigurationError(
            "animation.commands.require_expected_revision must be boolean"
        )
    if type(batch_limit) is not int or batch_limit < 1:
        raise AnimationConfigurationError(
            "animation.commands.max_batch_operations must be a positive integer"
        )
    return strict, batch_limit


def _required(payload: Mapping[str, Any], field: str) -> Any:
    if field not in payload:
        raise AnimationValidationError(
            f"missing required animation edit field {field!r}",
            context={"field": field},
        )
    return payload[field]


def _operation(value: Any) -> str:
    operation = ensure_non_empty_string(value, "operation", strip=True).strip()
    if operation not in SUPPORTED_EDITS:
        raise AnimationUnsupportedOperationError(
            "unsupported animation editing operation",
            context={"operation": operation},
        )
    return operation


def _mutable_animation(current: Any) -> dict[str, Any]:
    if current is None:
        raise AnimationStateError("scene has no animation document")
    if not isinstance(current, Mapping) or not isinstance(current.get("tracks"), list):
        raise AnimationStateError("scene contains malformed animation metadata")
    return deepcopy(dict(current))


def _track(candidate: dict[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    tracks = candidate["tracks"]
    index = ensure_track_index(_required(payload, "track_index"), len(tracks))
    track = tracks[index]
    if not isinstance(track, dict):
        raise AnimationStateError("scene contains malformed animation track")
    return track


def _stage(current: Any, operation: str, payload: Mapping[str, Any]) -> dict | None:
    """Return a detached candidate without validating the complete SGEO schema.

    SGEO or the scene provider validates the *complete* schema once after
    staging. Helpers here validate only operation-level indices and shape.
    """
    if not isinstance(payload, Mapping):
        raise AnimationValidationError("animation edit payload must be a mapping")
    if operation in {"set_animation", "set_animation_v2"}:
        document = _required(payload, "animation")
        if not isinstance(document, Mapping):
            raise AnimationValidationError("animation must be a mapping")
        return deepcopy(dict(document))
    if operation == "clear_animation":
        return None

    candidate = _mutable_animation(current)
    tracks = candidate["tracks"]
    for singular, collection in (
        ("clip", "clips"), ("instance", "instances"),
        ("layer", "layers"), ("transition", "transitions"),
        ("choreography", "choreographies"), ("event", "events")
    ):
        if operation not in {f"add_{singular}", f"update_{singular}", f"delete_{singular}"}:
            continue
        entries = candidate.setdefault(collection, [])
        if operation.startswith("add_"):
            value = _required(payload, singular)
            if not isinstance(value, Mapping):
                raise AnimationValidationError(f"{singular} must be a mapping")
            entries.append(deepcopy(dict(value)))
        else:
            idx = ensure_track_index(_required(payload, f"{singular}_index"), len(entries))
            if operation.startswith("delete_"):
                del entries[idx]
            else:
                value = _required(payload, singular)
                if not isinstance(value, Mapping):
                    raise AnimationValidationError(f"{singular} must be a mapping")
                entries[idx] = deepcopy(dict(value))
        return candidate
    if operation == "set_duration":
        # SGEO owns duration value/bound checks; do not clip keyframes.
        candidate["duration"] = _required(payload, "duration")
        return candidate
    if operation == "add_track":
        track = _required(payload, "track")
        if not isinstance(track, Mapping):
            raise AnimationValidationError("track must be a mapping")
        tracks.append(deepcopy(dict(track)))
        return candidate
    if operation in {"update_track", "delete_track"}:
        idx = ensure_track_index(_required(payload, "track_index"), len(tracks))
        if operation == "delete_track":
            del tracks[idx]
        else:
            track = _required(payload, "track")
            if not isinstance(track, Mapping):
                raise AnimationValidationError("track must be a mapping")
            tracks[idx] = deepcopy(dict(track))
        return candidate

    track = _track(candidate, payload)
    if operation == "set_interpolation":
        track["interpolation"] = _required(payload, "interpolation")
        return candidate
    frames = track.get("keyframes")
    if not isinstance(frames, list):
        raise AnimationStateError("scene contains malformed keyframe list")
    if operation == "add_keyframe":
        frame = ensure_keyframe(_required(payload, "keyframe"))
        frames.append(deepcopy(dict(frame)))
    elif operation in {"update_keyframe", "delete_keyframe"}:
        idx = ensure_keyframe_index(_required(payload, "keyframe_index"), len(frames))
        if operation == "delete_keyframe":
            del frames[idx]
        else:
            frame = ensure_keyframe(_required(payload, "keyframe"))
            frames[idx] = deepcopy(dict(frame))
    else:
        raise AnimationUnsupportedOperationError(
            "unsupported animation editing operation", context={"operation": operation}
        )
    # Stable sort; SGEO rejects duplicate timestamps and incompatible values.
    sort_keyframes_inplace(frames)
    return candidate


def _validate_candidate(scene: Any, candidate: dict | None, *, required: bool) -> None:
    """Prefer provider validation; use native SGEO validator for native Scenes.

    A third-party provider with no dry-run validator cannot promise a safe
    preview. Committed edits still rely on the authoritative setter, which
    performs the final validation and atomically commits its own document.
    """
    if candidate is None:
        return
    validate = getattr(scene, "validate_animation", None)
    if callable(validate):
        validate(candidate)
    elif isinstance(scene, SGEOScene):
        (validate_animation_v2 if ANIMATION_V2_KEY in scene.metadata else validate_animation)(candidate, scene)
    elif required:
        raise AnimationUnsupportedOperationError(
            "provider does not expose read-only animation validation for dry run"
        )


def _normalize_commands(commands: Sequence[Mapping[str, Any]], limit: int):
    if isinstance(commands, (str, bytes, bytearray)) or not isinstance(commands, Sequence):
        raise AnimationValidationError("commands must be a bounded sequence of mappings")
    if not 1 <= len(commands) <= limit:
        raise AnimationValidationError(
            "animation edit batch length exceeds configured limit or is empty",
            context={"count": len(commands), "max_batch_operations": limit},
        )
    result: list[tuple[str, Mapping[str, Any]]] = []
    for idx, command in enumerate(commands):
        if not isinstance(command, Mapping):
            raise AnimationValidationError(
                "batch item must be a mapping", context={"index": idx}
            )
        op = _operation(_required(command, "operation"))
        payload = command.get("payload", {})
        if not isinstance(payload, Mapping):
            raise AnimationValidationError(
                "batch command payload must be a mapping", context={"index": idx}
            )
        result.append((op, payload))
    return result


def apply_animation_batch(
    scene: Any,
    commands: Sequence[Mapping[str, Any]],
    *,
    expected_revision: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a bounded batch of edits atomically under the provider's lock.

    The entire batch stages in memory, then validates the resulting scene
    animation and calls the provider's setter exactly once. No partial stage
    changes can escape if an operation fails. Preserves historical revision
    semantics: even a no-op committed edit advances one revision when the
    provider's setter does so. A dry-run never calls that setter.

    Returns ``revision``, ``operations_applied`` and, for dry runs only,
    ``would_change``. Single edit callers should use ``update_animation``
    to retain its historical return dictionary.
    """
    if type(dry_run) is not bool:
        raise AnimationValidationError("dry_run must be boolean")
    if expected_revision is not None:
        ensure_positive_integer(expected_revision, "expected_revision")
    require_revision, max_batch = _settings()
    if require_revision and expected_revision is None:
        raise AnimationRevisionConflictError(
            "expected_revision is required by animation.commands policy"
        )
    normalized = _normalize_commands(commands, max_batch)
    if not is_animation_scene_provider(scene):
        raise AnimationUnsupportedOperationError(
            "scene does not implement mutable animation scene protocol"
        )
    lock = get_animation_edit_lock(scene)
    if lock is None:
        raise AnimationUnsupportedOperationError(
            "scene must expose an authoritative reentrant animation edit lock"
        )
    # This is a single critical section. It serializes the entire read,
    # revision check, edit plan, validation, and provider commit.
    with lock:
        revision = scene.revision
        if revision < 1:
            raise AnimationStateError("scene revision must be positive")
        if expected_revision is not None and revision != expected_revision:
            logger.info(
                "animation revision conflict | expected=%d actual=%d", expected_revision, revision
            )
            raise AnimationRevisionConflictError(
                "scene revision does not match expected_revision",
                context={"expected_revision": expected_revision, "actual_revision": revision},
            )
        key = ANIMATION_V2_KEY if ANIMATION_V2_KEY in scene.metadata else ANIMATION_KEY
        current = deepcopy(scene.metadata.get(key))
        candidate = current
        for operation, payload in normalized:
            candidate = _stage(candidate, operation, payload)
        # For native SGEO this ensures validation before the real commit;
        # for other providers their public validation hook is preferred.
        requested_v2 = any(op == "set_animation_v2" for op, _ in normalized)
        if requested_v2 and not isinstance(scene, SGEOScene):
            raise AnimationUnsupportedOperationError("v2 schema upgrade requires native SGEO scene")
        if requested_v2 and candidate is not None:
            validate_animation_v2(candidate, scene)
        else:
            _validate_candidate(scene, candidate, required=dry_run)
        if dry_run:
            changed = candidate != current
            logger.debug(
                "animation batch preview | scene=%s revision=%d ops=%d would_change=%s",
                scene.document_id, revision, len(normalized), changed,
            )
            return {"revision": revision, "operations_applied": len(normalized), "would_change": changed}
        # Provider owns atomicity and validation, including revision increments.
        new_revision = scene.set_animation(candidate, version=2) if requested_v2 else scene.set_animation(candidate)
        if type(new_revision) is not int or new_revision <= revision:
            logger.error("animation provider violated revision-advance contract")
            raise AnimationStateError(
                "scene provider returned an invalid committed revision; state must be inspected"
            )
        logger.info(
            "animation batch committed | scene=%s from=%d to=%d operations=%d",
            scene.document_id, revision, new_revision, len(normalized),
        )
        return {"revision": new_revision, "operations_applied": len(normalized)}


def update_animation(
    scene: Any,
    operation: str,
    payload: Mapping[str, Any],
    *,
    expected_revision: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backward-compatible single edit facade for AnimationAgent.

    Commits return exactly ``{'revision': int}``; previews return exactly
    ``{'revision': int, 'would_change': bool}``.
    """
    op = _operation(operation)
    result = apply_animation_batch(
        scene,
        ({"operation": op, "payload": payload},),
        expected_revision=expected_revision,
        dry_run=dry_run,
    )
    if dry_run:
        return {"revision": result["revision"], "would_change": result["would_change"]}
    return {"revision": result["revision"]}


__all__ = ["EXTENDED_EDITS", "SUPPORTED_EDITS", "update_animation", "apply_animation_batch"]
