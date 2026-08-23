"""Small, dependency-light helpers shared by the SLAI tuning subsystem.

The functions in this module are intentionally free of tuning policy.  They
provide deterministic identifiers and fingerprints, bounded JSON conversion,
secret redaction, lazy symbol loading, and atomic JSON writes.  Keeping these
mechanics here prevents search, evaluation, and orchestration code from
reimplementing subtly different versions of the same behavior.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import tempfile

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4


T = TypeVar("T")

DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_ITEMS = 200
DEFAULT_MAX_STRING_LENGTH = 8_192
REDACTED_VALUE = "***REDACTED***"

_SENSITIVE_CANONICAL_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "connection_string",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "access_token",
        "secret",
        "session_cookie",
        "signature",
        "token",
    }
)

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
        r"client[_-]?secret|password|passwd|authorization)\b\s*[:=]\s*)"
        r"([^\s,;]+)"
    ),
    re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9._~+/=-]+)"),
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp.

    A datetime is retained internally rather than a preformatted string so
    callers can perform sound duration comparisons before serialization.
    """

    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None = None, *, timespec: str = "milliseconds") -> str:
    """Return an ISO-8601 UTC timestamp using ``Z`` notation."""

    current = value or utc_now()
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("UTC serialization requires a timezone-aware datetime")
    return current.astimezone(timezone.utc).isoformat(timespec=timespec).replace(
        "+00:00", "Z"
    )


def elapsed_seconds(started_at: datetime, completed_at: datetime) -> float:
    """Return a non-negative duration between aware timestamps."""

    for name, value in (("started_at", started_at), ("completed_at", completed_at)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")
    elapsed = (completed_at - started_at).total_seconds()
    if elapsed < 0:
        raise ValueError("completed_at cannot precede started_at")
    return elapsed


def generate_run_id(prefix: str = "tuning") -> str:
    """Generate a sortable, collision-resistant run identifier."""

    normalized = sanitize_identifier(prefix, fallback="tuning")
    timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{normalized}-{timestamp}-{uuid4().hex[:12]}"


def sanitize_identifier(value: Any, *, fallback: str = "item", max_length: int = 80) -> str:
    """Convert a value into a conservative identifier or filename component."""

    if max_length < 1:
        raise ValueError("max_length must be positive")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = re.sub(r"[-_.]{2,}", "-", text).strip("-._")
    if not text:
        text = fallback
    return text[:max_length]


def qualified_name(value: Any) -> str:
    """Return a stable diagnostic name for a class, function, or instance."""

    target = value if isinstance(value, type) else type(value)
    if callable(value) and hasattr(value, "__qualname__"):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", repr(value))
    else:
        module = getattr(target, "__module__", "")
        name = getattr(target, "__qualname__", target.__name__)
    return f"{module}.{name}" if module else name


def load_symbol(path: str) -> Any:
    """Load ``module:attribute`` without importing optional modules eagerly."""

    if not isinstance(path, str) or ":" not in path:
        raise ValueError("Symbol path must use the form 'module:attribute'")
    module_name, attribute_path = (part.strip() for part in path.split(":", 1))
    if not module_name or not attribute_path:
        raise ValueError("Symbol path must contain both module and attribute")
    value: Any = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        if not attribute:
            raise ValueError(f"Invalid empty attribute in symbol path {path!r}")
        value = getattr(value, attribute)
    return value


def coerce_bool(value: Any, *, name: str = "value") -> bool:
    """Parse a boolean without Python's misleading ``bool('false')`` behavior."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean or an unambiguous boolean literal")


def is_sensitive_key(key: Any) -> bool:
    """Return whether a mapping key conventionally contains a secret.

    Token-based matching avoids false positives such as ``monkey`` while still
    recognizing variants such as ``service-api-key``.
    """

    raw = str(key).strip().casefold()
    canonical = "_".join(re.findall(r"[a-z0-9]+", raw))
    if canonical in _SENSITIVE_CANONICAL_NAMES:
        return True
    tokens = canonical.split("_") if canonical else []
    if any(
        token in {"password", "passwd", "secret", "credential", "credentials"}
        for token in tokens
    ):
        return True
    return bool(
        tokens
        and tokens[-1] in {"key", "token", "cookie", "signature"}
        and any(
            token in {"api", "access", "auth", "bearer", "client", "private", "refresh", "session"}
            for token in tokens[:-1]
        )
    )


def redact_text(value: str) -> str:
    """Best-effort redaction for common ``name=value`` secret forms."""

    redacted = value
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}{REDACTED_VALUE}", redacted)
    return redacted


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return f"{value[: max_length - 3]}..."


def _non_finite_float(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def to_json_safe(
    value: Any,
    *,
    redact_sensitive: bool = True,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
    _depth: int = 0,
    _field_name: str | None = None,
    _seen: set[int] | None = None,
) -> Any:
    """Convert arbitrary values into bounded, valid JSON data.

    Non-finite floating-point values are rendered as explicit strings because
    RFC 8259 JSON has no NaN or infinity literals.  Large array-like objects are
    summarized by shape and dtype instead of being materialized into logs.
    Cycles, serialization errors, and hostile ``repr`` implementations are
    handled without masking the original tuning failure.
    """

    if max_depth < 1 or max_items < 1 or max_string_length < 1:
        raise ValueError("Serialization limits must be positive")
    if redact_sensitive and _field_name and is_sensitive_key(_field_name):
        return REDACTED_VALUE
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _non_finite_float(value)
    if isinstance(value, str):
        text = redact_text(value) if redact_sensitive else value
        return _truncate(text, max_string_length)
    if isinstance(value, Enum):
        return to_json_safe(
            value.value,
            redact_sensitive=redact_sensitive,
            max_depth=max_depth,
            max_items=max_items,
            max_string_length=max_string_length,
            _depth=_depth,
            _field_name=_field_name,
            _seen=_seen,
        )
    if isinstance(value, (datetime, date, time)):
        if isinstance(value, datetime) and value.tzinfo is not None:
            return utc_iso(value)
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, BaseException):
        return {
            "name": value.__class__.__name__,
            "message": to_json_safe(
                str(value),
                redact_sensitive=redact_sensitive,
                max_string_length=max_string_length,
            ),
        }
    if _depth >= max_depth:
        return f"<{type(value).__name__}:depth-limit>"

    seen = _seen if _seen is not None else set()
    object_id = id(value)
    if object_id in seen:
        return f"<{type(value).__name__}:cycle>"
    seen.add(object_id)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            raw = {field.name: getattr(value, field.name) for field in fields(value)}
            return to_json_safe(
                raw,
                redact_sensitive=redact_sensitive,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
                _field_name=_field_name,
                _seen=seen,
            )

        shape = getattr(value, "shape", None)
        raw_size = getattr(value, "size", None)
        try:
            size = None if raw_size is None or isinstance(raw_size, bool) else int(raw_size)
        except (TypeError, ValueError, OverflowError):
            size = None
        if shape is not None and size is not None:
            if size > max_items:
                return {
                    "type": type(value).__name__,
                    "shape": to_json_safe(tuple(shape), redact_sensitive=False),
                    "size": size,
                    "dtype": str(getattr(value, "dtype", "unknown")),
                }
            to_list = getattr(value, "tolist", None)
            if callable(to_list):
                return to_json_safe(
                    to_list(),
                    redact_sensitive=redact_sensitive,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string_length=max_string_length,
                    _depth=_depth + 1,
                    _field_name=_field_name,
                    _seen=seen,
                )

        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            count = 0
            omitted = 0
            for key, item in value.items():
                if count >= max_items:
                    omitted += 1
                    continue
                key_text = _truncate(str(key), max_string_length)
                result[key_text] = to_json_safe(
                    item,
                    redact_sensitive=redact_sensitive,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string_length=max_string_length,
                    _depth=_depth + 1,
                    _field_name=key_text,
                    _seen=seen,
                )
                count += 1
            if omitted:
                result["__truncated__"] = f"{omitted} additional items omitted"
            return result

        if isinstance(value, (set, frozenset)):
            sequence = sorted(value, key=lambda item: repr(item))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sequence = value
        else:
            to_dict = getattr(value, "to_dict", None)
            if callable(to_dict):
                try:
                    converted = to_dict()
                except Exception:
                    converted = None
                if converted is not None and converted is not value:
                    return to_json_safe(
                        converted,
                        redact_sensitive=redact_sensitive,
                        max_depth=max_depth,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        _depth=_depth + 1,
                        _field_name=_field_name,
                        _seen=seen,
                    )
            item = getattr(value, "item", None)
            if callable(item):
                try:
                    scalar = item()
                except Exception:
                    scalar = value
                if scalar is not value:
                    return to_json_safe(
                        scalar,
                        redact_sensitive=redact_sensitive,
                        max_depth=max_depth,
                        max_items=max_items,
                        max_string_length=max_string_length,
                        _depth=_depth + 1,
                        _field_name=_field_name,
                        _seen=seen,
                    )
            try:
                rendered = repr(value)
            except Exception:
                rendered = f"<{type(value).__name__}>"
            return _truncate(rendered, max_string_length)

        serialized = [
            to_json_safe(
                item,
                redact_sensitive=redact_sensitive,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
                _field_name=_field_name,
                _seen=seen,
            )
            for item in sequence[:max_items]
        ]
        if len(sequence) > max_items:
            serialized.append(f"... {len(sequence) - max_items} additional items omitted")
        return serialized
    finally:
        seen.discard(object_id)


def canonical_json_bytes(value: Any, *, redact_sensitive: bool = False) -> bytes:
    """Return deterministic UTF-8 JSON bytes for hashing or persistence.

    Canonicalization uses deliberately high bounds rather than the conservative
    logging bounds used by :func:`to_json_safe`; otherwise large configuration
    values could be truncated into the same fingerprint.
    """

    normalized = to_json_safe(
        value,
        redact_sensitive=redact_sensitive,
        max_depth=100,
        max_items=10_000_000,
        max_string_length=10_000_000,
    )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_fingerprint(value: Any, *, algorithm: str = "sha256") -> str:
    """Hash a JSON-normalized value using a named hashlib algorithm."""

    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported digest algorithm {algorithm!r}") from exc
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    redact_sensitive: bool = True,
) -> Path:
    """Atomically persist valid JSON and return the resolved target path.

    The temporary file is created in the destination directory so
    ``os.replace`` remains atomic on a single filesystem.  Existing file mode
    bits are retained when possible.
    """

    if indent < 0:
        raise ValueError("indent must be non-negative")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(str(target))

    prior_mode = target.stat().st_mode & 0o777 if target.exists() else None
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                to_json_safe(payload, redact_sensitive=redact_sensitive),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=indent,
                sort_keys=sort_keys,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if prior_mode is not None:
            os.chmod(temporary, prior_mode)
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return target
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# Backward-compatible name retained for the v2.2 tuning callers.
safe_serialize = to_json_safe


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_MAX_STRING_LENGTH",
    "REDACTED_VALUE",
    "atomic_write_json",
    "canonical_json_bytes",
    "coerce_bool",
    "elapsed_seconds",
    "generate_run_id",
    "is_sensitive_key",
    "load_symbol",
    "qualified_name",
    "redact_text",
    "safe_serialize",
    "sanitize_identifier",
    "stable_fingerprint",
    "to_json_safe",
    "utc_iso",
    "utc_now",
]
