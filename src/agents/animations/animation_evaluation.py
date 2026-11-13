"""Read-only animation scene evaluation and bounded, revision-consistent sampling.

This module deliberately does NOT sample tracks, interpolate curves, solve
constraints, construct world transforms, or create scene snapshots. Those are
provider responsibilities (native SGEO has its own evaluator). It coordinates
validated evaluation requests and verifies the returned native state envelope.

Three entry points:
- ``evaluate_animation``: historical single-timestamp API, unchanged.
- ``evaluate_animation_many``: bounded, consistently revision-stamped samples.
- ``evaluate_animation_frames``: rational frame-time sampling using existing
  SLAI ``Timebase``; no cumulative floating point frame stepping.

When a provider supports atomic ``evaluate_at_revision``, it is preferred.
Otherwise the revision of the actual returned snapshot is checked. For a
multi-sample request, all samples must have the same revision and the entire
call fails on a conflict (no partial results escape).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any

from src.utils.spatial_formats.common.errors import ConflictError # type: ignore
from .animation_protocols import (
    get_animation_edit_lock,
    is_evaluated_scene,
    is_read_only_animation_scene_provider,
    is_versioned_evaluator_provider,
)
from .timeline.timebase import Timebase
from .utils.config_loader import get_config_section
from .utils.animation_errors import *
from .utils.animation_helpers import *
from logs.logger import get_logger # type: ignore

logger = get_logger("Animation Evaluation")
# _CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "animation_config.yaml" # config loader already has DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "animation_config.yaml"


@dataclass(frozen=True, slots=True)
class AnimationEvaluationBatch:
    """Detached provider states sampled from one logically consistent revision.

    Native evaluated state objects are preserved without copying them. They
    must be independently detached by the scene provider as documented in
    ``EvaluatedAnimationScene``. This dataclass does not make those objects
    deeply immutable; it only fixes the collection membership.
    """

    timestamps: tuple[float, ...]
    states: tuple[Any, ...]
    revision: int
    elapsed_ms: float

    def __post_init__(self) -> None:
        if not self.timestamps or len(self.timestamps) != len(self.states):
            raise AnimationStateError("batch requires one state per timestamp")
        ensure_positive_integer(self.revision, "revision")
        ensure_finite_non_negative(self.elapsed_ms, "elapsed_ms")

    @property
    def sample_count(self) -> int:
        return len(self.timestamps)

    def summary(self) -> dict[str, Any]:
        """Bounded metadata for logging/UI without serializing scene geometry."""
        return {
            "revision": self.revision,
            "sample_count": self.sample_count,
            "first_time": self.timestamps[0],
            "last_time": self.timestamps[-1],
            "elapsed_ms": self.elapsed_ms,
        }


def _evaluation_config() -> tuple[int, int, int]:
    config = get_config_section("animation")
    values = config.get("evaluation")
    timeline = config.get("timeline")
    if not isinstance(values, dict) or not isinstance(timeline, dict):
        raise AnimationConfigurationError(
            "animation.evaluation and animation.timeline must be configured"
        )
    limit = values.get("max_batch_samples")
    num = timeline.get("default_fps_num")
    den = timeline.get("default_fps_den")
    if type(limit) is not int or limit < 1:
        raise AnimationConfigurationError(
            "animation.evaluation.max_batch_samples must be a positive integer"
        )
    if type(num) is not int or num < 1 or type(den) is not int or den < 1:
        raise AnimationConfigurationError("animation.timeline default FPS must be positive integers")
    return limit, num, den


def _check_state(state: Any, *, expected_revision: int | None = None) -> Any:
    if not is_evaluated_scene(state):
        logger.error("animation evaluator returned an invalid scene state envelope")
        raise AnimationStateError("provider returned a state missing time, revision or ids")
    if state.revision < 1:
        raise AnimationStateError("evaluated scene revision must be positive")
    ensure_finite_non_negative(state.time, "evaluated_scene.time")
    if expected_revision is not None and state.revision != expected_revision:
        logger.info(
            "animation evaluation conflict | expected=%d actual=%d",
            expected_revision, state.revision,
        )
        raise AnimationRevisionConflictError(
            "evaluated scene revision differs from expected_revision",
            context={"expected_revision": expected_revision, "actual_revision": state.revision},
        )
    return state


def evaluate_animation(
    scene: Any,
    seconds: float,
    *,
    expected_revision: int | None = None,
) -> Any:
    """Evaluate a scene at a time in seconds and validate its revision stamp.

    Returns the exact native provider state. A scene provider may clamp or
    snap time as specified in its own contract; this layer never changes its
    time semantics. Errors from provider evaluation propagate unchanged unless
    they are a native SGEO ``ConflictError``, translated to the Animation
    revision-conflict taxonomy with the original error as its cause.
    """
    timestamp = ensure_finite_non_negative(seconds, "seconds")
    if expected_revision is not None:
        ensure_positive_integer(expected_revision, "expected_revision")
    if not is_read_only_animation_scene_provider(scene) or scene.revision < 1:
        raise AnimationUnsupportedOperationError(
            "scene does not implement a valid read-only animation provider contract"
        )
    try:
        if expected_revision is not None and is_versioned_evaluator_provider(scene):
            state = scene.evaluate_at_revision(timestamp, expected_revision)
        else:
            state = scene.evaluate(timestamp)
    except ConflictError as exc:
        logger.info("animation evaluation rejected: native scene revision conflict")
        raise AnimationRevisionConflictError(
            "scene revision does not match expected_revision",
            context={"expected_revision": expected_revision}, cause=exc,
        ) from exc
    state = _check_state(state, expected_revision=expected_revision)
    logger.debug("animation scene evaluated | time=%s revision=%d", state.time, state.revision)
    return state


def evaluate_animation_many(
    scene: Any,
    times: Iterable[float],
    *,
    expected_revision: int | None = None,
    lock_scene: bool = False,
) -> AnimationEvaluationBatch:
    """Evaluate a bounded sequence of times, rejecting mixed revisions.

    ``lock_scene=True`` holds the provider's own reentrant edit lock across
    all samples (only when the provider exposes one), guaranteeing that
    cooperating scene writers cannot interleave with the batch. The default
    avoids long read locks; it validates each snapshot's revision and refuses
    to return a mixed-revision series. No result is emitted on failure.
    """
    if type(lock_scene) is not bool:
        raise AnimationValidationError("lock_scene must be boolean")
    if expected_revision is not None:
        ensure_positive_integer(expected_revision, "expected_revision")
    max_samples, _, _ = _evaluation_config()
    if isinstance(times, (str, bytes, bytearray)):
        raise AnimationValidationError("times must be an iterable of timestamps")
    try:
        iterator = iter(times)
    except TypeError as exc:
        raise AnimationValidationError("times must be iterable", cause=exc) from exc
    # Limit memory even when a caller supplies an infinite generator.
    samples = tuple(islice(iterator, max_samples + 1))
    if not 1 <= len(samples) <= max_samples:
        raise AnimationValidationError(
            "animation evaluation sample count exceeds configured budget or is empty",
            context={"max_batch_samples": max_samples, "count": len(samples)},
        )
    timestamps = tuple(ensure_finite_non_negative(x, f"times[{i}]") for i, x in enumerate(samples))
    lock = get_animation_edit_lock(scene) if lock_scene else None
    if lock_scene and lock is None:
        raise AnimationUnsupportedOperationError(
            "lock_scene requires a provider-owned reentrant edit lock"
        )
    started = perf_counter()
    states: list[Any] = []
    revision: int | None = expected_revision

    def evaluate_all() -> None:
        nonlocal revision
        for timestamp in timestamps:
            state = evaluate_animation(scene, timestamp, expected_revision=revision)
            if revision is None:
                revision = state.revision
            states.append(state)

    if lock is not None:
        with lock:
            evaluate_all()
    else:
        evaluate_all()
    assert revision is not None  # Every non-empty batch establishes a revision.
    batch = AnimationEvaluationBatch(timestamps, tuple(states), revision,
                                     (perf_counter() - started) * 1000.0)
    logger.debug("animation evaluated batch | %s", batch.summary())
    return batch


def evaluate_animation_frames(
    scene: Any,
    start_frame: int,
    end_frame: int,
    *,
    timebase: Timebase | None = None,
    expected_revision: int | None = None,
    lock_scene: bool = False,
) -> AnimationEvaluationBatch:
    """Evaluate inclusive integer frame range using SLAI's rational Timebase.

    Time is converted once per frame via the existing ``frame_seconds``
    operation. No repeated floating-point accumulation or resampling code.
    """
    start = ensure_non_negative_integer(start_frame, "start_frame")
    end = ensure_non_negative_integer(end_frame, "end_frame")
    if end < start:
        raise AnimationValidationError("end_frame must be >= start_frame")
    max_samples, num, den = _evaluation_config()
    if end - start + 1 > max_samples:
        raise AnimationValidationError(
            "frame range exceeds animation.evaluation.max_batch_samples",
            context={"sample_count": end - start + 1, "limit": max_samples},
        )
    if timebase is None:
        timebase = Timebase(num, den)
    elif not isinstance(timebase, Timebase):
        raise AnimationValidationError("timebase must be a Timebase instance")
    times = (float(timebase.frame_seconds(frame)) for frame in range(start, end + 1))
    return evaluate_animation_many(
        scene, times, expected_revision=expected_revision, lock_scene=lock_scene
    )


__all__ = ["AnimationEvaluationBatch", "evaluate_animation", "evaluate_animation_many", "evaluate_animation_frames"]
