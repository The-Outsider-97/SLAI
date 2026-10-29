"""Reusable, side-effect-free validation helpers for animation data.

This module houses the low-level, deterministic utility layer that sits
underneath :mod:`animation_commands`, :mod:`animation_evaluation`,
:mod:`interpolation`, :mod:`playback`, and :mod:`timebase`.  Nothing
here touches a scene, provider, filesystem, or global state — every
helper is a pure function that *either* returns a normalised value *or*
raises an :class:`~.animation_errors.AnimationError` subclass.  Callers
that need "the resulting value or an exception" get exactly that, and
callers that only need validation can safely ignore the return value.

Design invariants
-----------------
* **Purity.**  No I/O, no logging, no mutation of module state.
* **Determinism.**  Validation order and error codes are stable across
  runs, so tests and log-derived metrics are reproducible.
* **Domain errors, not bare built-ins.**  Every failure raises an
  ``AnimationError`` subclass.  Several of those also inherit from the
  corresponding built-in (``ValueError``/``IndexError``), so legacy
  ``except ValueError`` handlers continue to work during migration.
* **No schema re-implementation.**  Deep SGEO track validation remains
  the responsibility of
  :func:`src.utils.spatial_formats.sgeo.animation.validate_animation`.
  The helpers here cover the finer-grained, caller-level checks that
  surround it: scalar finiteness, index ranges, quaternion shape and
  magnitude, and keyframe structural minimums.
* **Conservative value coercion.**  ``bool`` is rejected everywhere an
  ``int``/``float`` is expected — ``True`` silently behaving as ``1``
  is virtually always a bug in animation code.

The helpers do *not* deep-copy values.  Copy semantics (e.g. the
``deepcopy`` used by ``animation_commands`` when staging edits) remain
the caller's responsibility so that helper composition stays cheap and
side-effect-free.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from math import isfinite, sqrt
from typing import Any

from .animation_errors import AnimationError, AnimationIndexError, AnimationValidationError


# Guard against dividing by a near-zero quaternion magnitude.  Matches
# the epsilon used by ``interpolation.slerp`` so validation, normalisation,
# and interpolation agree on what "zero quaternion" means.
_QUATERNION_EPSILON = 1e-15
_QUATERNION_AXES = ("w", "x", "y", "z")


# ---------------------------------------------------------------------------
# Scalar validators
# ---------------------------------------------------------------------------
def ensure_finite_scalar(value: Any, name: str) -> float:
    """Coerce ``value`` to :class:`float` and require it to be finite.

    ``bool`` is rejected even though it is an ``int`` subclass: in the
    animation domain, ``True`` silently behaving as ``1`` (or
    ``False`` as ``0``) is a near-certain caller error.

    Raises
    ------
    AnimationValidationError
        If ``value`` is not an :class:`int`/:class:`float` (excluding
        :class:`bool`), or if the coerced value is ``NaN`` or
        ``+/-inf``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnimationValidationError(
            f"{name} must be a real number, got {type(value).__name__}",
            context={"field": name, "received_type": type(value).__name__},
        )
    result = float(value)
    if not isfinite(result):
        raise AnimationValidationError(
            f"{name} must be finite, got {result!r}",
            context={"field": name, "received": result},
        )
    return result


def ensure_finite_non_negative(value: Any, name: str) -> float:
    """Validate a finite scalar that must be ``>= 0``."""
    result = ensure_finite_scalar(value, name)
    if result < 0.0:
        raise AnimationValidationError(
            f"{name} must be non-negative, got {result!r}",
            context={"field": name, "received": result, "minimum": 0.0},
        )
    return result


def ensure_finite_positive(value: Any, name: str) -> float:
    """Validate a finite scalar that must be strictly ``> 0``."""
    result = ensure_finite_scalar(value, name)
    if result <= 0.0:
        raise AnimationValidationError(
            f"{name} must be strictly positive, got {result!r}",
            context={"field": name, "received": result, "exclusive_minimum": 0.0},
        )
    return result


def ensure_unit_interval(value: Any, name: str) -> float:
    """Validate a finite scalar in the closed interval ``[0, 1]``.

    Used for interpolation weights, playback-rate weighting factors,
    and any other blending coefficient where an out-of-range value
    would silently extrapolate.
    """
    result = ensure_finite_scalar(value, name)
    if not 0.0 <= result <= 1.0:
        raise AnimationValidationError(
            f"{name} must lie in [0, 1], got {result!r}",
            context={"field": name, "received": result, "minimum": 0.0, "maximum": 1.0},
        )
    return result


# ---------------------------------------------------------------------------
# Integer validators
# ---------------------------------------------------------------------------
def _ensure_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnimationValidationError(
            f"{name} must be an integer, got {type(value).__name__}",
            context={"field": name, "received_type": type(value).__name__},
        )
    return value


def ensure_positive_integer(value: Any, name: str) -> int:
    """Validate an integer that must be ``>= 1``.

    Used for frame-rate numerators/denominators, ``expected_revision``,
    and ``AnimationMemory.capacity``-style configuration values.
    """
    result = _ensure_int(value, name)
    if result < 1:
        raise AnimationValidationError(
            f"{name} must be a positive integer, got {result!r}",
            context={"field": name, "received": result, "minimum": 1},
        )
    return result


def ensure_non_negative_integer(value: Any, name: str) -> int:
    """Validate an integer that must be ``>= 0``."""
    result = _ensure_int(value, name)
    if result < 0:
        raise AnimationValidationError(
            f"{name} must be a non-negative integer, got {result!r}",
            context={"field": name, "received": result, "minimum": 0},
        )
    return result


# ---------------------------------------------------------------------------
# String validators
# ---------------------------------------------------------------------------
def ensure_non_empty_string(value: Any, name: str, *, strip: bool = False) -> str:
    """Validate that ``value`` is a non-empty :class:`str`.

    Parameters
    ----------
    value
        Candidate value; must be an instance of :class:`str`.
    name
        Field name used in error messages and structured context.
    strip
        When ``True``, leading and trailing whitespace is ignored when
        deciding whether the string is empty (the original value,
        including whitespace, is still returned unchanged).  This
        mirrors ``AnimationRequest.__post_init__``'s treatment of
        ``operation``.

    Raises
    ------
    AnimationValidationError
        If ``value`` is not a string, or is empty (or, with
        ``strip=True``, contains only whitespace).
    """
    if not isinstance(value, str):
        raise AnimationValidationError(
            f"{name} must be a string, got {type(value).__name__}",
            context={"field": name, "received_type": type(value).__name__},
        )
    considered = value.strip() if strip else value
    if not considered:
        raise AnimationValidationError(
            f"{name} must be a non-empty string",
            context={"field": name, "strip": strip},
        )
    return value


# ---------------------------------------------------------------------------
# Index validators
# ---------------------------------------------------------------------------
def ensure_track_index(index: Any, track_count: int) -> int:
    """Validate a track index against the live track count.

    The check is *strict* about ``bool`` (which Python would otherwise
    treat as ``0``/``1``) to match the existing ``type(index) is not
    int`` guard in ``animation_commands``.

    Raises
    ------
    AnimationIndexError
        If ``index`` is not a plain :class:`int` or falls outside
        ``[0, track_count)``.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise AnimationIndexError(
            f"track index must be an integer, got {type(index).__name__}",
            context={"index": repr(index), "track_count": track_count},
        )
    if not 0 <= index < track_count:
        raise AnimationIndexError(
            f"track index {index} out of range for {track_count} track(s)",
            context={"index": index, "track_count": track_count},
        )
    return index


def ensure_keyframe_index(index: Any, keyframe_count: int) -> int:
    """Validate a keyframe index against the live keyframe count.

    Same ``bool``-rejection policy as :func:`ensure_track_index`, for
    the same reason.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise AnimationIndexError(
            f"keyframe index must be an integer, got {type(index).__name__}",
            context={"index": repr(index), "keyframe_count": keyframe_count},
        )
    if not 0 <= index < keyframe_count:
        raise AnimationIndexError(
            f"keyframe index {index} out of range for {keyframe_count} keyframe(s)",
            context={"index": index, "keyframe_count": keyframe_count},
        )
    return index


# ---------------------------------------------------------------------------
# Quaternion validators
# ---------------------------------------------------------------------------
def ensure_quaternion(value: Any, name: str = "quaternion") -> tuple[float, float, float, float]:
    """Validate a ``(w, x, y, z)`` quaternion of four finite reals.

    The component ordering matches :mod:`interpolation` (scalar-first,
    ``w, x, y, z``).  Magnitude is **not** checked here — use
    :func:`quaternion_norm` or :func:`normalize_quaternion` when the
    downstream operation requires a non-zero or unit quaternion.

    Raises
    ------
    AnimationValidationError
        If ``value`` is not a 4-element sequence of finite reals.
    """
    if not isinstance(value, (tuple, list)):
        raise AnimationValidationError(
            f"{name} must be a 4-element sequence, got {type(value).__name__}",
            context={"field": name, "received_type": type(value).__name__},
        )
    if len(value) != 4:
        raise AnimationValidationError(
            f"{name} must contain exactly 4 components, got {len(value)}",
            context={"field": name, "length": len(value)},
        )
    w = ensure_finite_scalar(value[0], f"{name}[w]")
    x = ensure_finite_scalar(value[1], f"{name}[x]")
    y = ensure_finite_scalar(value[2], f"{name}[y]")
    z = ensure_finite_scalar(value[3], f"{name}[z]")
    return (w, x, y, z)


def _validated_magnitude(q: tuple[float, float, float, float]) -> float:
    """Euclidean norm with the shared zero-quaternion guard."""
    magnitude = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if magnitude < _QUATERNION_EPSILON:
        raise AnimationValidationError(
            "quaternion has (near-)zero magnitude",
            context={"quaternion": q, "magnitude": magnitude},
        )
    return magnitude


def quaternion_norm(value: Any, name: str = "quaternion") -> float:
    """Return the Euclidean norm of a validated quaternion.

    Raises :class:`AnimationValidationError` for malformed components
    *or* for a zero-magnitude quaternion — a zero quaternion cannot be
    normalised and is therefore treated as invalid even for
    norm-only use, matching the ``interpolation.slerp`` contract.
    """
    q = ensure_quaternion(value, name)
    return _validated_magnitude(q)


def normalize_quaternion(value: Any, name: str = "quaternion") -> tuple[float, float, float, float]:
    """Return the unit quaternion equivalent to ``value``.

    Raises :class:`AnimationValidationError` for malformed components
    or a zero-magnitude quaternion.
    """
    q = ensure_quaternion(value, name)
    magnitude = _validated_magnitude(q)
    return (
        q[0] / magnitude,
        q[1] / magnitude,
        q[2] / magnitude,
        q[3] / magnitude,
    )


# ---------------------------------------------------------------------------
# Keyframe validators
# ---------------------------------------------------------------------------
def ensure_keyframe(keyframe: Any, *, name: str = "keyframe") -> Mapping[str, Any]:
    """Validate the *minimum* structural contract of a keyframe.

    Per the ``slai.animation.v1`` schema, a keyframe is a mapping whose
    ``time`` field is a finite real.  This helper enforces exactly that
    minimum and no more — full track/frame validation (values, tangents,
    interpolation modes, etc.) remains the responsibility of
    :func:`src.utils.spatial_formats.sgeo.animation.validate_animation`.

    Returns the keyframe unchanged on success so callers can write
    ``kf = ensure_keyframe(candidate)`` without an extra lookup.

    Raises
    ------
    AnimationValidationError
        If ``keyframe`` is not a mapping, lacks ``time``, or has a
        non-finite ``time``.
    """
    if not isinstance(keyframe, Mapping):
        raise AnimationValidationError(
            f"{name} must be a mapping, got {type(keyframe).__name__}",
            context={"field": name, "received_type": type(keyframe).__name__},
        )
    if "time" not in keyframe:
        raise AnimationValidationError(
            f"{name} is missing required 'time' field",
            context={"field": name, "keys": list(keyframe.keys())},
        )
    ensure_finite_scalar(keyframe["time"], f"{name}.time")
    return keyframe


def ensure_keyframes(keyframes: Any, *, name: str = "keyframes") -> Sequence[Mapping[str, Any]]:
    """Validate that ``keyframes`` is a sequence of valid keyframe mappings.

    Rejects :class:`str`, :class:`bytes`, and :class:`bytearray` even
    though they are technically ``Sequence`` instances — a string is
    never a valid keyframe sequence.  Individual elements are validated
    by :func:`ensure_keyframe`, so the first invalid element surfaces
    with a precise index in its context.
    """
    if isinstance(keyframes, (str, bytes, bytearray)) or not isinstance(keyframes, Sequence):
        raise AnimationValidationError(
            f"{name} must be a sequence of mappings, got {type(keyframes).__name__}",
            context={"field": name, "received_type": type(keyframes).__name__},
        )
    for position, keyframe in enumerate(keyframes):
        ensure_keyframe(keyframe, name=f"{name}[{position}]")
    return keyframes


def keyframe_time(keyframe: Any, *, name: str = "keyframe") -> float:
    """Return the validated ``time`` of ``keyframe`` as a :class:`float`.

    Convenience wrapper that runs :func:`ensure_keyframe` and then
    coerces ``time`` — useful when the caller needs the numeric value
    for comparisons or ordering and does not want to repeat the
    coercion inline.
    """
    ensure_keyframe(keyframe, name=name)
    return float(keyframe["time"])


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------
def sort_keyframes_inplace(
    keyframes: MutableSequence[Mapping[str, Any]],
    *,
    name: str = "keyframes",
) -> None:
    """Stable in-place sort of ``keyframes`` by ascending ``time``.

    The whole sequence is validated *before* the sort mutates it, so a
    malformed element cannot leave the caller with a partially
    reordered list.  Idempotent — already-sorted input produces no
    reordering.

    Python's ``list.sort`` is stable, which is the correct semantic for
    animation editing: keyframes with identical timestamps (rare but
    legal) preserve their insertion order across edits.
    """
    ensure_keyframes(keyframes, name=name)
    keyframes[:] = sorted(keyframes, key=lambda k: k["time"])


__all__ = [
    # Backward-compatible re-export
    "AnimationError",
    # Scalar validators
    "ensure_finite_scalar",
    "ensure_finite_non_negative",
    "ensure_finite_positive",
    "ensure_unit_interval",
    # Integer validators
    "ensure_positive_integer",
    "ensure_non_negative_integer",
    # String validators
    "ensure_non_empty_string",
    # Index validators
    "ensure_track_index",
    "ensure_keyframe_index",
    # Quaternion validators
    "ensure_quaternion",
    "quaternion_norm",
    "normalize_quaternion",
    # Keyframe validators
    "ensure_keyframe",
    "ensure_keyframes",
    "keyframe_time",
    "sort_keyframes_inplace",
]