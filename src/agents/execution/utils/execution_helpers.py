"""Shared helper primitives for the Execution agent subsystem.

This module centralizes deterministic, dependency-light mechanics reused by
execution selection, validation, recovery, persistence, and action modules.

It intentionally does not own configuration, logging, persistence, task or
action lifecycle policy, recovery strategy selection, hardware access, or
world-model semantics. The module is a dependency-direction leaf: it must not
import ExecutionMemory, ExecutionValidator, ExecutionRecovery, TaskCoordinator,
ActionSelector, action classes, or ``modules.robot_interface``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random

from collections.abc import Callable, Iterable, Mapping, MutableSequence, Sequence
from numbers import Real
from typing import Any, TypeVar

_T = TypeVar("_T")

_ACTION_TARGET_KEYS = {
    "move_to": "destination",
    "pick_object": "object_position",
    "place_object": "place_position",
}


def _finite_float(value: Any, *, parameter: str) -> float:
    """Convert a numeric input to a finite float or raise a precise error."""
    if isinstance(value, bool):
        raise ValueError(f"{parameter} must be a finite real number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{parameter} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{parameter} must be a finite real number")
    return converted


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Return ``value`` constrained to the inclusive numeric interval.

    ``ValueError`` is raised for an inverted interval so configuration errors
    are not silently converted into plausible execution values.
    """
    candidate = _finite_float(value, parameter="value")
    lower = _finite_float(minimum, parameter="minimum")
    upper = _finite_float(maximum, parameter="maximum")
    if lower > upper:
        raise ValueError("minimum cannot be greater than maximum")
    return max(lower, min(upper, candidate))


def clamp01(value: float) -> float:
    """Return ``value`` constrained to the closed unit interval."""
    return clamp(value, 0.0, 1.0)


def safe_divide(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Divide numeric values, returning ``default`` for a zero divisor."""
    numerator_value = _finite_float(numerator, parameter="numerator")
    denominator_value = _finite_float(denominator, parameter="denominator")
    if denominator_value == 0.0:
        return _finite_float(default, parameter="default")
    return numerator_value / denominator_value


def is_finite_number(value: Any) -> bool:
    """Return whether ``value`` is a finite real scalar, excluding booleans."""
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def is_position(value: Any, *, dimensions: int = 2) -> bool:
    """Return whether ``value`` provides the requested finite coordinates."""
    if dimensions < 1:
        raise ValueError("dimensions must be at least one")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return False
    return len(value) >= dimensions and all(
        is_finite_number(component) for component in value[:dimensions]
    )


def euclidean_distance(
    first: Sequence[Real],
    second: Sequence[Real],
    *,
    dimensions: int = 2,
) -> float:
    """Return finite Euclidean distance for two coordinate sequences.

    The strict contract keeps invalid or non-finite positions from propagating
    through safety, reachability, or energy calculations.
    """
    if not is_position(first, dimensions=dimensions):
        raise ValueError("first must contain finite coordinate values")
    if not is_position(second, dimensions=dimensions):
        raise ValueError("second must contain finite coordinate values")
    return math.dist(
        [float(component) for component in first[:dimensions]],
        [float(component) for component in second[:dimensions]],
    )


def normalize_angle(angle_radians: float) -> float:
    """Normalize a finite angle to the closed-open interval ``[-pi, pi)``."""
    angle = float(angle_radians)
    if not math.isfinite(angle):
        raise ValueError("angle_radians must be finite")
    normalized = math.atan2(math.sin(angle), math.cos(angle))
    return -math.pi if normalized == math.pi else normalized


def missing_required_keys(
    context: Mapping[str, Any],
    required_keys: Iterable[str],
) -> tuple[str, ...]:
    """Return required keys absent from ``context`` in first-seen order."""
    missing = []
    seen = set()
    for key in required_keys:
        normalized = str(key)
        if normalized not in seen and normalized not in context:
            missing.append(normalized)
        seen.add(normalized)
    return tuple(missing)


def preconditions_met(
    preconditions: Iterable[str],
    context: Mapping[str, Any],
) -> bool:
    """Return whether every named precondition is truthy in ``context``."""
    return all(bool(context.get(str(condition), False)) for condition in preconditions)


def find_named_item(
    items: Iterable[_T],
    name: str,
    *,
    key: str = "name",
) -> _T | None:
    """Return the first mapping or object whose named field matches ``name``."""
    for item in items:
        value = item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)
        if value == name:
            return item
    return None


def action_target_key(action_name: str) -> str:
    """Return the canonical context key for an action's spatial target."""
    return _ACTION_TARGET_KEYS.get(str(action_name), "target_position")


def action_target_position(
    context: Mapping[str, Any],
    action_name: str,
    *,
    default: Any = None,
) -> Any:
    """Return an action target using the subsystem's canonical key mapping."""
    return context.get(action_target_key(action_name), default)


def resolve_planar_position(
    robot: Any,
    fallback: Any = (0.0, 0.0),
) -> tuple[float, float]:
    """Resolve a robot's planar pose with a validated context fallback.

    A missing or explicitly unsupported ``get_pose`` capability uses the
    fallback. Hardware/runtime failures from an implemented method propagate
    to the caller instead of being misreported as a position fallback.
    """
    pose_getter = getattr(robot, "get_pose", None)
    pose = fallback
    if callable(pose_getter):
        try:
            pose = pose_getter()
        except (AttributeError, NotImplementedError):
            pose = fallback

    if not is_position(pose):
        raise ValueError("resolved robot pose must contain finite coordinates")
    pose_seq: Sequence[Any] = pose  # type: ignore[assignment]
    return float(pose_seq[0]), float(pose_seq[1])


def missing_callable_attributes(target: Any, names: Iterable[str]) -> tuple[str, ...]:
    """Return capability names that are not callable on ``target``."""
    missing = []
    seen = set()
    for name in names:
        normalized = str(name)
        if normalized not in seen and not callable(getattr(target, normalized, None)):
            missing.append(normalized)
        seen.add(normalized)
    return tuple(missing)


def missing_capabilities(target: Any, names: Iterable[str]) -> tuple[str, ...]:
    """Return unsupported callable capabilities for an adapter-like object.

    Structural adapters are checked by callability. Contract classes may expose
    ``supports_capability(name)`` to distinguish an implemented override from a
    callable placeholder for an optional feature.
    """
    normalized_names = tuple(dict.fromkeys(str(name) for name in names))
    missing = set(missing_callable_attributes(target, normalized_names))
    capability_checker = getattr(target, "supports_capability", None)
    if callable(capability_checker):
        for name in normalized_names:
            if name not in missing and not bool(capability_checker(name)):
                missing.add(name)
    return tuple(name for name in normalized_names if name in missing)


def stable_json_dumps(
    value: Any,
    *,
    sort_keys: bool = True,
    separators: tuple[str, str] | None = (",", ":"),
) -> str:
    """Serialize execution metadata deterministically using v2.2 fallbacks."""
    options: dict[str, Any] = {
        "default": str,
        "sort_keys": sort_keys,
    }
    if separators is not None:
        options["separators"] = separators
    return json.dumps(value, **options)


def stable_digest(
    value: Any,
    *,
    algorithm: str = "sha256",
    digest_size: int | None = None,
) -> str:
    """Return a deterministic hexadecimal digest for bytes, text, or metadata."""
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = stable_json_dumps(value).encode("utf-8")

    normalized_algorithm = str(algorithm).strip().lower()
    if normalized_algorithm == "blake2b":
        size = 64 if digest_size is None else int(digest_size)
        if not 1 <= size <= 64:
            raise ValueError("blake2b digest_size must be between 1 and 64")
        return hashlib.blake2b(payload, digest_size=size).hexdigest()
    if digest_size is not None:
        raise ValueError("digest_size is supported only for blake2b")

    try:
        return hashlib.new(normalized_algorithm, payload).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported hash algorithm: {algorithm}") from exc


def bounded_append(
    items: MutableSequence[_T],
    item: _T,
    max_length: int,
) -> None:
    """Append an item and retain only the newest ``max_length`` entries."""
    limit = int(max_length)
    if limit < 1:
        raise ValueError("max_length must be at least one")
    items.append(item)
    overflow = len(items) - limit
    if overflow > 0:
        del items[:overflow]


def redact_mapping(
    value: Mapping[str, Any],
    *,
    sensitive_keys: Iterable[str] = (),
    omitted_keys: Iterable[str] = (),
    redaction: Any = "***",
    omission: Any = "<omitted>",
) -> dict[str, Any]:
    """Return a shallow, order-preserving redacted mapping copy."""
    sensitive = {str(key) for key in sensitive_keys}
    omitted = {str(key) for key in omitted_keys}
    result: dict[str, Any] = {}
    for key, item in value.items():
        normalized = str(key)
        if normalized in sensitive:
            result[normalized] = redaction
        elif normalized in omitted:
            result[normalized] = omission
        else:
            result[normalized] = item
    return result


def exponential_backoff(
    attempt: int,
    *,
    base_delay: float,
    max_delay: float,
    jitter: bool = False,
    random_uniform: Callable[[float, float], float] = random.uniform,
) -> float:
    """Calculate capped exponential retry delay without sleeping."""
    base = max(0.0, _finite_float(base_delay, parameter="base_delay"))
    maximum = max(base, _finite_float(max_delay, parameter="max_delay"))
    exponent = max(0, int(attempt) - 1)
    delay = min(maximum, base * (2**exponent))
    if jitter and delay > base:
        return clamp(random_uniform(base, delay), base, delay)
    return float(delay)


__all__ = [
    "action_target_key",
    "action_target_position",
    "bounded_append",
    "clamp",
    "clamp01",
    "euclidean_distance",
    "exponential_backoff",
    "find_named_item",
    "is_finite_number",
    "is_position",
    "missing_capabilities",
    "missing_callable_attributes",
    "missing_required_keys",
    "normalize_angle",
    "preconditions_met",
    "redact_mapping",
    "resolve_planar_position",
    "safe_divide",
    "stable_digest",
    "stable_json_dumps",
]
