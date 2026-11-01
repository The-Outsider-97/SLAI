"""
Deterministic helper primitives for the SLAI provenance subsystem.

The helpers deliberately keep provenance mechanics separate from provenance
policy.  They normalize identifiers/timestamps, provide lossless canonical JSON
for stable provenance identities, hash artifact content, perform atomic JSON
writes through SLAI's existing data helper, and provide graph-cycle checks.

W3C PROV-DM/PROV Constraints motivate stable identities, explicit temporal
values, and structurally valid derivation graphs.  SLAI-specific canonical JSON
is intentionally lossless: the generic BaseAgent serializers are bounded for
logging and therefore unsuitable for provenance identity or persistence.
"""

from __future__ import annotations

__version__ = "2.3.0"

import base64
import hashlib
import json
import math

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from data.utils.data_helpers import atomic_write_json as _atomic_write_json # type: ignore
from data.utils.data_helpers import compute_file_hash # type: ignore
from .provenance_errors import ProvenanceStorageError, ProvenanceValidationError
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Provenance Helpers")
printer = PrettyPrinter()


_JSON_SCALAR = (str, int, bool, type(None))


def get_current_timestamp(_owner: Any = None) -> str:
    """Return a canonical RFC-3339 UTC timestamp using ``Z`` notation.

    ``_owner`` is accepted for backward compatibility with the previous helper
    signature, which was sometimes called as ``get_current_timestamp(self)``.
    """

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def normalize_timestamp(value: Optional[str | datetime] = None) -> str:
    """Normalize a timestamp to UTC RFC-3339 form.

    Naive datetimes are rejected because provenance ordering should not silently
    depend on a machine-local timezone.
    """

    if value is None:
        return get_current_timestamp()

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ProvenanceValidationError("timestamp must not be empty")
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProvenanceValidationError(
                "timestamp must be valid ISO-8601",
                context={"timestamp": value},
                cause=exc,
            ) from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ProvenanceValidationError(
            "timestamp must be a string, datetime, or None",
            context={"type": type(value).__name__},
        )

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProvenanceValidationError(
            "timestamp must include timezone information",
            context={"timestamp": str(value)},
        )
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def require_identifier(value: Any, *, field_name: str = "identifier", max_length: int = 512) -> str:
    """Validate a stable provenance identifier without changing its semantics."""

    if value is None:
        raise ProvenanceValidationError(f"{field_name} must not be None")
    text = str(value).strip()
    if not text:
        raise ProvenanceValidationError(f"{field_name} must not be empty")
    if len(text) > max_length:
        raise ProvenanceValidationError(
            f"{field_name} exceeds the maximum length",
            context={"field": field_name, "max_length": max_length, "length": len(text)},
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ProvenanceValidationError(
            f"{field_name} contains control characters",
            context={"field": field_name},
        )
    return text


def normalize_id_sequence(value: Optional[str | Iterable[Any]], *, field_name: str = "identifiers") -> Tuple[str, ...]:
    """Normalize one identifier or an iterable into a deterministic unique tuple."""

    if value is None:
        return ()
    raw_values = [value] if isinstance(value, str) else list(value)
    normalized: List[str] = []
    seen: Set[str] = set()
    for item in raw_values:
        identifier = require_identifier(item, field_name=field_name)
        if identifier not in seen:
            normalized.append(identifier)
            seen.add(identifier)
    return tuple(normalized)


def _canonicalize(value: Any, *, seen: Set[int], depth: int) -> Any:
    if depth > 100:
        raise ProvenanceValidationError("provenance value exceeds maximum nesting depth")

    if isinstance(value, _JSON_SCALAR):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProvenanceValidationError(
                "non-finite floating-point values are not valid provenance metadata",
                context={"value": repr(value)},
            )
        return value
    if isinstance(value, Enum):
        return _canonicalize(value.value, seen=seen, depth=depth + 1)
    if isinstance(value, datetime):
        return normalize_timestamp(value)
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "encoding": "base64",
            "length": len(payload),
            "value": base64.b64encode(payload).decode("ascii"),
        }

    object_id = id(value)
    if object_id in seen:
        raise ProvenanceValidationError(
            "cyclic Python object cannot be represented as canonical provenance metadata"
        )
    seen.add(object_id)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: _canonicalize(
                    getattr(value, field.name), seen=seen, depth=depth + 1
                )
                for field in fields(value)
            }

        if isinstance(value, Mapping):
            normalized: Dict[str, Any] = {}
            for key in sorted(value.keys(), key=lambda item: str(item)):
                key_text = str(key)
                normalized[key_text] = _canonicalize(
                    value[key], seen=seen, depth=depth + 1
                )
            return normalized

        if isinstance(value, (set, frozenset)):
            canonical_items = [
                _canonicalize(item, seen=seen, depth=depth + 1) for item in value
            ]
            canonical_items.sort(
                key=lambda item: json.dumps(
                    item, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
                )
            )
            return canonical_items

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                _canonicalize(item, seen=seen, depth=depth + 1) for item in value
            ]

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if converted is value:
                raise ProvenanceValidationError(
                    "to_dict() returned the original object and cannot be canonicalized",
                    context={"type": type(value).__name__},
                )
            return _canonicalize(converted, seen=seen, depth=depth + 1)

        raise ProvenanceValidationError(
            "unsupported value in provenance metadata",
            context={"type": type(value).__name__},
        )
    finally:
        seen.discard(object_id)


def canonicalize_value(value: Any) -> Any:
    """Return a lossless, deterministic JSON-compatible representation."""

    return _canonicalize(value, seen=set(), depth=0)


def normalize_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return an independent canonical metadata mapping."""

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ProvenanceValidationError(
            "metadata must be a mapping",
            context={"type": type(metadata).__name__},
        )
    normalized = canonicalize_value(metadata)
    if not isinstance(normalized, dict):
        raise ProvenanceValidationError("metadata normalization did not produce a mapping")
    return normalized


def canonical_json_dumps(value: Any, *, pretty: bool = False) -> str:
    """Serialize provenance content deterministically without truncation."""

    canonical = canonicalize_value(value)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def stable_provenance_id(
    prefix: str,
    *components: Any,
    length: int = 24,
    algorithm: str = "sha256",
) -> str:
    """Create a deterministic identifier from complete canonical provenance data."""

    safe_prefix = require_identifier(prefix, field_name="prefix", max_length=64)
    if length < 8:
        raise ProvenanceValidationError("stable provenance ID length must be >= 8")
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ProvenanceValidationError(
            "unsupported digest algorithm",
            context={"algorithm": algorithm},
            cause=exc,
        ) from exc
    digest.update(canonical_json_dumps(components).encode("utf-8"))
    return f"{safe_prefix}:{digest.hexdigest()[:length]}"


def content_digest(value: Any, *, algorithm: str = "sha256") -> str:
    """Return a content digest for bytes, text, files, or canonical structured data."""

    if isinstance(value, Path):
        try:
            return compute_file_hash(value, algorithm=algorithm)
        except Exception as exc:
            raise ProvenanceStorageError(
                "failed to hash provenance file content",
                context={"path": str(value), "algorithm": algorithm},
                cause=exc,
            ) from exc

    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ProvenanceValidationError(
            "unsupported digest algorithm",
            context={"algorithm": algorithm},
            cause=exc,
        ) from exc

    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    else:
        payload = canonical_json_dumps(value).encode("utf-8")
    digest.update(payload)
    return digest.hexdigest()


def atomic_write_provenance_json(path: str | Path, payload: Any) -> Path:
    """Persist canonical provenance JSON atomically via SLAI's shared data helper."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = canonicalize_value(payload)
    try:
        return _atomic_write_json(canonical, target, indent=2)
    except Exception as exc:
        raise ProvenanceStorageError(
            "failed to atomically persist provenance JSON",
            context={"path": str(target)},
            cause=exc,
        ) from exc


def load_json_mapping(path: str | Path, *, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load a JSON object from disk, returning a copy of ``default`` when absent."""

    target = Path(path).expanduser().resolve()
    if not target.exists():
        return dict(default or {})
    if not target.is_file():
        raise ProvenanceStorageError(
            "provenance storage path is not a file",
            context={"path": str(target)},
        )
    try:
        with target.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceStorageError(
            "failed to load provenance JSON",
            context={"path": str(target)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise ProvenanceStorageError(
            "provenance JSON root must be a mapping",
            context={"path": str(target), "type": type(payload).__name__},
        )
    return payload


def lineage_would_create_cycle(
    existing_records: Iterable[Mapping[str, Any]],
    *,
    child_id: str,
    parent_ids: Iterable[str],
) -> bool:
    """Return whether adding ``parent -> child`` derivations would create a cycle."""

    child = require_identifier(child_id, field_name="child_id")
    parents = normalize_id_sequence(parent_ids, field_name="parent_ids")
    if child in parents:
        return True

    children_by_parent: Dict[str, Set[str]] = {}
    for raw_record in existing_records:
        record_child = raw_record.get("artifact_id") or raw_record.get("child_id")
        if not record_child:
            continue
        try:
            normalized_child = require_identifier(record_child, field_name="artifact_id")
            record_parents = raw_record.get("parent_artifact_ids")
            if record_parents is None and raw_record.get("parent_artifact_id") is not None:
                record_parents = [raw_record.get("parent_artifact_id")]
            normalized_parents = normalize_id_sequence(
                record_parents or (), field_name="parent_artifact_ids"
            )
        except ProvenanceValidationError:
            continue
        for parent in normalized_parents:
            children_by_parent.setdefault(parent, set()).add(normalized_child)

    # Adding parent -> child is cyclic exactly when parent is already reachable
    # from child in the existing parent->child graph.
    for parent in parents:
        frontier = [child]
        visited: Set[str] = set()
        while frontier:
            current = frontier.pop()
            if current == parent:
                return True
            if current in visited:
                continue
            visited.add(current)
            frontier.extend(children_by_parent.get(current, ()))
    return False


__all__ = [
    "atomic_write_provenance_json",
    "canonical_json_dumps",
    "canonicalize_value",
    "content_digest",
    "get_current_timestamp",
    "lineage_would_create_cycle",
    "load_json_mapping",
    "normalize_id_sequence",
    "normalize_metadata",
    "normalize_timestamp",
    "require_identifier",
    "stable_provenance_id",
]


if __name__ == "__main__":
    configure_logging()
    printer.status("PROVENANCE", "provenance helper smoke check complete", "success")
