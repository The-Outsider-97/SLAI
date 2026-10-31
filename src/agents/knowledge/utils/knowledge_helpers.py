"""Shared helper primitives for the Knowledge agent subsystem.

This module centralises dependency-light mechanics reused by multiple
Knowledge components.

It intentionally does not own configuration loading, governance policy,
persistence, lifecycle management, action execution, ontology semantics,
or synchronization/conflict-resolution behavior.

The helpers defined here are dependency-direction leaves.  They must not
import KnowledgeCache, KnowledgeMemory, KnowledgeMonitor,
KnowledgeSynchronizer, KnowledgeOrchestrator, OntologyManager, Governor,
PerformAction, or components from the runtime and validation packages.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

from .knowledge_errors import KnowledgeError, UnsafeLegacyRule, ValidationFailure


# -----------------------------------------------------------------------------
# Identifier contract (shared with ``contracts.py``)
# -----------------------------------------------------------------------------
#: A safe identifier: 1..128 chars, begins alphanumeric, then alphanumeric
#: or one of ``_.:/-``.
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")

#: A lowercase snake_case kind identifier: 1..64 chars.
KIND = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# -----------------------------------------------------------------------------
# Action directive parsing
# -----------------------------------------------------------------------------
_ACTION_DIRECTIVE_PATTERN = re.compile(r"action:(\w+):(.+)", re.IGNORECASE)


def match_action_directive(value: Any) -> Optional[Tuple[str, str]]:
    """Match an action directive at the beginning of a string.

    Returns the normalised action type and stripped payload when ``value``
    starts with a valid action directive.  Non-string values and
    non-matching strings return ``None``.
    """
    if not isinstance(value, str):
        return None
    match = _ACTION_DIRECTIVE_PATTERN.match(value)
    if match is None:
        return None
    return match.group(1).lower(), match.group(2).strip()


def find_action_directives(value: Any) -> List[Tuple[str, str]]:
    """Return action directives found in a string in encounter order."""
    if not isinstance(value, str):
        return []
    return [
        (match.group(1).lower(), match.group(2).strip())
        for match in _ACTION_DIRECTIVE_PATTERN.finditer(value)
    ]


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------
def config_relative_candidates(path: Path, config_file: Path) -> Tuple[Path, Path, Path]:
    """Build the standard Knowledge config-relative path candidates."""
    config_dir = config_file.resolve().parent
    return (
        config_dir / path,
        config_dir.parent / path,
        config_dir.parent.parent / path,
    )


def first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    """Return the first existing candidate without changing its order."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# -----------------------------------------------------------------------------
# Date / time helpers
# -----------------------------------------------------------------------------
def parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse a date string using common academic formats.

    Supports ``'%Y-%m-%d'``, ``'%Y/%m/%d'``, and ``'%Y'``.  Returns a
    timezone-aware UTC ``datetime`` or ``None``.
    """
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 ``...Z`` string."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def normalize_time(value: Optional[str], field_name: str) -> Optional[str]:
    """Normalise an ISO-8601 timestamp to canonical UTC ``...Z`` form."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationFailure(
            f"{field_name} must be an ISO-8601 UTC timestamp",
            field=field_name,
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationFailure(
            f"Invalid {field_name}",
            field=field_name,
            details={"input": value},
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValidationFailure(f"{field_name} must specify UTC", field=field_name)
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


# -----------------------------------------------------------------------------
# Identifiers / DOI / URLs
# -----------------------------------------------------------------------------
def validate_id(value: str, field_name: str) -> str:
    """Validate a safe identifier.

    Raises :class:`ValidationFailure` when the value is not a string or
    does not match the canonical identifier grammar.
    """
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValidationFailure(
            f"Invalid {field_name}: required nonempty safe identifier (max 128 characters)",
            field=field_name,
        )
    return value


def is_valid_doi(doi: str) -> bool:
    """Basic DOI validation: starts with ``'10.'`` and contains a slash."""
    doi = (doi or "").strip()
    return bool(doi and doi.startswith("10.") and "/" in doi)


def extract_domain(url: str) -> str:
    """Return the lowercase netloc (domain) from a URL, or ``''``."""
    parsed = urlparse(url or "")
    return parsed.netloc.lower().strip()


# -----------------------------------------------------------------------------
# JSON / hashing helpers
# -----------------------------------------------------------------------------
def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialise to JSON with ``default=str``, sorting keys, error-safe."""
    options: Dict[str, Any] = {"default": str, "sort_keys": True}
    options.update(kwargs)
    return json.dumps(obj, **options)


def safe_hash(data: Dict[str, Any]) -> str:
    """Compute a SHA-256 hex digest of a dict via deterministic JSON."""
    return hashlib.sha256(safe_json_dumps(data).encode("utf-8")).hexdigest()


def json_object(value: Any, field_name: str, max_bytes: int = 524288) -> Dict[str, Any]:
    """Validate and re-normalise a JSON-compatible dict.

    Ensures:

    * value is a dict with string keys,
    * all values are finite, JSON-compatible primitives,
    * encoded form is under ``max_bytes``.
    """
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValidationFailure(f"{field_name} must be an object with string keys", field=field_name)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=lambda _: (_ for _ in ()).throw(TypeError()),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationFailure(f"{field_name} must contain only finite JSON-compatible values", field=field_name) from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ValidationFailure(
            f"{field_name} exceeds {max_bytes} bytes",
            field=field_name,
            details={"max_bytes": max_bytes},
        )
    return json.loads(encoded)


def canonical_json(value: Any) -> str:
    """Deterministic, compact JSON serialisation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest_json(value: Any) -> str:
    """SHA-256 hex digest over the canonical JSON representation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject(message: str) -> None:
    raise UnsafeLegacyRule(f"Unsupported legacy rule: {message}")


def _safe_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject("numeric expression expected")
    if not math.isfinite(float(value)) or abs(float(value)) > 1e12:
        _reject("non-finite or oversized numeric expression")
    return float(value)


def _nonblank(payload: Mapping[str, Any], field: str, limit: int = 256) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValidationFailure(f"{field} must be a nonempty string of at most {limit} characters")
    return value.strip()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # Regexes
    "IDENTIFIER",
    "KIND",

    # Action directives
    "match_action_directive",
    "find_action_directives",

    # Filesystem
    "config_relative_candidates",
    "first_existing_path",

    # Dates / time
    "parse_date",
    "utc_now",
    "normalize_time",

    # Identifiers / URLs
    "validate_id",
    "is_valid_doi",
    "extract_domain",

    # JSON / hashing
    "safe_json_dumps",
    "safe_hash",
    "json_object",
    "canonical_json",
    "digest_json",

    # Re-exported for consumer convenience
    "KnowledgeError",
    "ValidationFailure",
    "_reject",
    "_safe_number",
    "_nonblank",
]