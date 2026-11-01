"""
Shared helper primitives for the SLAI evaluation subsystem.

This module owns small, stateless, cross-cutting operations that are shared by
evaluation components and do not belong to a specific evaluator, model,
persistence implementation, validation protocol, or calculation service.

Design constraints
------------------
- No evaluator lifecycle ownership.
- No configuration loading beyond constructing existing typed configuration
  errors for validation failures.
- No persistence or shared-memory access.
- No metric or statistical algorithms.
- No imports from evaluator modules.
- No mutable module-level state.

Keeping this module dependency-light allows evaluator implementations,
evaluation modules, and data-access components to depend on it without
introducing circular imports.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from statistics import mean
from typing import Any, Iterable, List, Sequence

from .evaluation_errors import *


# ----------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------

def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailureError("timestamp", value, "ISO-8601 string")

    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationFailureError("timestamp", value, "valid ISO-8601 string") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed

def _utcnow() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)

def _coerce_timestamp(value: str) -> datetime:
    """
    Parse an ISO-8601 timestamp and normalize it to timezone-aware UTC.

    Naive timestamps retain the existing evaluator contract and are interpreted
    as UTC.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailureError("timestamp", value, "non-empty ISO-8601 string")

    normalized = value.strip().replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationFailureError("timestamp", value, "valid ISO-8601 string") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _coerce_non_negative_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return number


def _normalize_non_empty_string(value: Any, field_name: str) -> str:
    """Validate and normalize a required non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailureError(field_name, value, "non-empty string")

    return value.strip()


def _normalize_string_list(value: Any, field_name: str) -> List[str]:
    """
    Validate and normalize a non-empty sequence of strings.

    Duplicate entries are removed case-insensitively while preserving the
    ordering and original casing of their first occurrence.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationFailureError(field_name, type(value).__name__, "sequence of strings")

    normalized: List[str] = []
    seen: set[str] = set()

    for item in value:
        text = _normalize_non_empty_string(item, field_name)
        key = text.casefold()

        if key not in seen:
            normalized.append(text)
            seen.add(key)

    if not normalized:
        raise ValidationFailureError(field_name, value, "non-empty sequence of strings")

    return normalized


def _require_positive_float(value: Any, field_name: str) -> float:
    """Coerce a configuration value to a strictly positive float."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError("<config>", field_name, f"Expected a positive float, got {value!r}") from exc

    if number <= 0:
        raise ConfigLoadError("<config>", field_name, f"Expected a positive float, got {number!r}")

    return number


def _require_non_negative_float(value: Any, field_name: str) -> float:
    """Coerce a configuration value to a non-negative float."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError("<config>", field_name, f"Expected a non-negative float, got {value!r}") from exc

    if number < 0:
        raise ConfigLoadError("<config>", field_name, f"Expected a non-negative float, got {number!r}")

    return number


def _require_positive_int(value: Any, field_name: str) -> int:
    """Coerce a configuration value to a strictly positive integer."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError("<config>", field_name, f"Expected a positive integer, got {value!r}") from exc

    if number <= 0:
        raise ConfigLoadError("<config>", field_name, f"Expected a positive integer, got {number!r}")

    return number


def _require_non_negative_int(value: Any, field_name: str) -> int:
    """Coerce a value to a non-negative integer."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailureError( field_name, value, "non-negative integer") from exc

    if number < 0:
        raise ValidationFailureError( field_name, value, "non-negative integer")

    return number


def _coerce_probability(value: Any, field_name: str, *, inclusive_zero: bool, inclusive_one: bool) -> float:
    """
    Coerce and validate a probability using explicit endpoint semantics.

    Supported domains are:
        [0, 1]
        (0, 1]
        [0, 1)
        (0, 1)
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError("<config>", field_name, f"Expected a probability, got {value!r}") from exc

    lower_ok = number > 0.0 or (inclusive_zero and number == 0.0)
    upper_ok = number < 1.0 or (inclusive_one and number == 1.0)

    if not (lower_ok and upper_ok):
        comparator = (
            "[0, 1]"
            if inclusive_zero and inclusive_one
            else "(0, 1]"
            if not inclusive_zero and inclusive_one
            else "[0, 1)"
            if inclusive_zero and not inclusive_one
            else "(0, 1)"
        )

        raise ConfigLoadError("<config>", field_name, f"Expected probability in {comparator}, got {number!r}")

    return number


def _canonical_json(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError) as exc:
        raise InvalidDocumentError(
            f"Document payload cannot be serialized deterministically: {exc}"
        ) from exc


def _get_hash_constructor(name: str):
    if not isinstance(name, str) or not name.strip():
        raise DocumentationConfigurationError("Hash algorithm name must be a non-empty string.")

    algorithm = name.strip().lower()
    try:
        return getattr(hashlib, algorithm)
    except AttributeError as exc:
        raise DocumentationConfigurationError(
            f"Unsupported hash algorithm configured for audit trail: {algorithm}"
        ) from exc


def _safe_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(mean(materialized)) if materialized else 0.0


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, include_attributes=False)


def _clamp_severity(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))


def _scaled_severity(observed: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0
    ratio = observed / threshold
    return max(0.4, min(1.0, 0.5 + (ratio - 1.0) * 0.5))



__all__ = [
    "_parse_timestamp",
    "_utcnow",
    "_coerce_timestamp",
    "_coerce_non_negative_float",
    "_normalize_non_empty_string",
    "_normalize_string_list",
    "_require_positive_float",
    "_require_non_negative_float",
    "_require_positive_int",
    "_require_non_negative_int",
    "_coerce_probability",
    "_canonical_json",
    "_get_hash_constructor",
    "_safe_mean",
    "_safe_unparse",
    "_clamp_severity",
    "_scaled_severity",
]