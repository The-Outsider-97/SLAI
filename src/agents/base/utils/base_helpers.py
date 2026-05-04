"""
Shared helpers for the base subsystem.

This module centralizes the reusable helper layer for the base subsystem.
It provides production-ready primitives for configuration access, structured
serialization, redaction, endpoint handling, identifier generation,
normalisation, collection utilities, and timing/backoff logic.

The goal is to keep higher-level subsystem modules focused on domain logic
instead of repeatedly re-implementing cross-cutting behaviors. The helpers are
therefore intentionally generic, defensive, and stable enough to be reused
across telemetry, orchestration, model-serving, and infrastructure-adjacent
code paths.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import socket
import unicodedata
import uuid

from pathlib import Path
from time import monotonic
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

from .base_errors import *
from logs.logger import get_logger, PrettyPrinter  # type: ignore

logger = get_logger("base_helpers")
printer = PrettyPrinter()

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

_DEFAULT_MAX_SERIALIZATION_DEPTH = 6
_DEFAULT_MAX_COLLECTION_ITEMS = 50
_DEFAULT_MAX_STRING_LENGTH = 2048
_DEFAULT_SECRET_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "private_key",
    "client_secret",
    "session",
    "session_id",
    "cookie",
    "set_cookie",
}
_DEFAULT_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[a-z0-9\-\._~\+/]+=*"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(token\s*[=:]\s*)([^\s,;]+)"),
)
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")
_PORT_RE = re.compile(r"^(?P<host>[^:]+)(?::(?P<port>\d+))?$")


@dataclass(frozen=True)
class BackoffPolicy:
    """Reusable exponential backoff configuration."""

    initial_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 60.0
    jitter_ratio: float = 0.0

    def compute_delay(self, attempt: int) -> float:
        """Return the bounded delay for a 1-based retry attempt."""
        if attempt < 1:
            raise BaseValidationError(
                "attempt must be >= 1",
                None,
                context={"attempt": attempt},
            )
        base_delay = self.initial_delay * (self.multiplier ** (attempt - 1))
        bounded = min(self.max_delay, max(0.0, base_delay))
        if self.jitter_ratio <= 0:
            return bounded
        spread = bounded * self.jitter_ratio
        return max(0.0, random.uniform(bounded - spread, bounded + spread))


class Stopwatch:
    """Minimal monotonic stopwatch for lightweight timing code paths."""
    def __init__(self, start_immediately: bool = True):
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        if start_immediately:
            self.start()

    def start(self) -> "Stopwatch":
        self._started_at = monotonic()
        self._stopped_at = None
        return self

    def stop(self) -> float:
        if self._started_at is None:
            raise BaseValidationError("Stopwatch has not been started", None)
        self._stopped_at = monotonic()
        return self.elapsed_seconds

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        end = self._stopped_at if self._stopped_at is not None else monotonic()
        return max(0.0, end - self._started_at)

    @property
    def elapsed_ms(self) -> int:
        return int(round(self.elapsed_seconds * 1000))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def monotonic_seconds() -> float:
    return monotonic()

def coerce_bool(value: Any, default: bool = False) -> bool:
    """Best-effort boolean coercion for config and environment values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def coerce_int(value: Any, default: int = 0, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    """Coerce a value into an integer with optional bounds."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_float(
    value: Any,
    default: float = 0.0,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Coerce a value into a float with optional bounds."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


# ---------------------------------------------------------------------------
# Dictionary and collection helpers
# ---------------------------------------------------------------------------
def deep_merge_dicts(*mappings: Optional[Mapping[str, Any]], inplace: bool = False) -> Dict[str, Any]:
    """Deep-merge one or more mappings into a single dictionary."""
    if not mappings:
        return {}

    if inplace:
        first = mappings[0]
        if not isinstance(first, MutableMapping):
            raise BaseValidationError(
                "inplace=True requires the first mapping to be mutable",
                None,
                context={"type": type(first).__name__ if first is not None else None},
            )
        target: Dict[str, Any] = dict(first)
    else:
        target = {}

    def _merge(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
        for key, value in src.items():
            if (
                key in dst
                and isinstance(dst[key], Mapping)
                and isinstance(value, Mapping)
            ):
                dst[key] = _merge(dict(dst[key]), value)
            else:
                dst[key] = value
        return dst

    for mapping in mappings:
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise BaseValidationError(
                "All values passed to deep_merge_dicts must be mappings",
                None,
                context={"offending_type": type(mapping).__name__},
            )
        _merge(target, mapping)
    return target


def drop_none_values(data: Mapping[str, Any], *, recursive: bool = True, drop_empty: bool = False) -> Dict[str, Any]:
    """Remove ``None`` values from a mapping, optionally recursively."""
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if recursive and isinstance(value, Mapping):
            nested = drop_none_values(value, recursive=True, drop_empty=drop_empty)
            if nested or not drop_empty:
                cleaned[key] = nested
            continue
        if recursive and isinstance(value, list):
            nested_list = [
                drop_none_values(item, recursive=True, drop_empty=drop_empty) if isinstance(item, Mapping) else item
                for item in value
                if item is not None
            ]
            if nested_list or not drop_empty:
                cleaned[key] = nested_list
            continue
        if drop_empty and value in ("", [], {}, ()):
            continue
        cleaned[key] = value
    return cleaned


def flatten_mapping(
    data: Mapping[str, Any],
    *,
    parent_key: str = "",
    separator: str = ".",
    max_depth: int = 10,
) -> Dict[str, Any]:
    """Flatten a nested mapping into dotted keys."""
    if max_depth < 0:
        raise BaseValidationError("max_depth must be >= 0", None, context={"max_depth": max_depth})

    flattened: Dict[str, Any] = {}

    def _walk(prefix: str, value: Any, depth: int) -> None:
        if depth > max_depth or not isinstance(value, Mapping):
            flattened[prefix] = value
            return
        for child_key, child_value in value.items():
            next_key = f"{prefix}{separator}{child_key}" if prefix else str(child_key)
            _walk(next_key, child_value, depth + 1)

    _walk(parent_key, data, 0)
    return flattened


def chunked(sequence: Sequence[T], size: int) -> List[List[T]]:
    """Split a sequence into equally sized chunks except for the tail."""
    if size <= 0:
        raise BaseValidationError("size must be > 0", None, context={"size": size})
    return [list(sequence[index : index + size]) for index in range(0, len(sequence), size)]


def ensure_list(value: Union[T, Iterable[T], None]) -> List[T]:
    """Normalize a scalar or iterable into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (str, bytes)):
        return [value]  # type: ignore[list-item]
    if isinstance(value, Iterable):
        return list(value)
    return [value]  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Text and parsing helpers
# ---------------------------------------------------------------------------
def ensure_text(value: Any, *, encoding: str = "utf-8", errors: str = "replace") -> str:
    """Return a text representation that is safe for logs and serialization."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(encoding, errors=errors)
    return str(value)


def normalize_text(
    text: Any,
    *,
    lowercase: bool = False,
    strip: bool = True,
    collapse_whitespace: bool = True,
    unicode_form: Literal['NFC', 'NFD', 'NFKC', 'NFKD'] = "NFKC",
) -> str:
    """Normalize text for stable comparison, hashing, and transport."""
    normalized = ensure_text(text)
    if unicode_form:
        normalized = unicodedata.normalize(unicode_form, normalized)
    if collapse_whitespace:
        normalized = _WHITESPACE_RE.sub(" ", normalized)
    if strip:
        normalized = normalized.strip()
    if lowercase:
        normalized = normalized.lower()
    return normalized


def slugify(text: Any, *, default: str = "item", max_length: int = 80) -> str:
    """Generate a filesystem- and URL-friendly slug."""
    normalized = normalize_text(text, lowercase=True)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = _NON_SLUG_RE.sub("-", ascii_text).strip("-")
    if not slug:
        slug = default
    if max_length > 0:
        slug = slug[:max_length].rstrip("-")
    return slug or default


def normalize_identifier(
    value: Any,
    *,
    lowercase: bool = True,
    separator: str = "_",
    max_length: int = 120,
) -> str:
    """Normalize free-form input into a conservative identifier."""
    text = normalize_text(value, lowercase=lowercase)
    text = re.sub(r"[^a-zA-Z0-9]+", separator, text)
    text = text.strip(separator)
    if max_length > 0:
        text = text[:max_length].strip(separator)
    if not text:
        raise BaseValidationError("Could not derive a valid identifier", None, context={"value": value})
    return text


def parse_delimited_text(
    value: Union[str, Sequence[Any], None],
    *,
    separator: str = ",",
    strip_items: bool = True,
    drop_empty: bool = True,
    unique: bool = False,
) -> List[str]:
    """Parse a delimited string or sequence into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(separator)
    elif isinstance(value, Sequence):
        items = [ensure_text(item) for item in value]
    else:
        raise BaseValidationError(
            "value must be a string, sequence, or None",
            None,
            context={"type": type(value).__name__},
        )

    parsed: List[str] = []
    seen: set[str] = set()
    for item in items:
        current = item.strip() if strip_items else item
        if drop_empty and not current:
            continue
        if unique:
            if current in seen:
                continue
            seen.add(current)
        parsed.append(current)
    return parsed


def parse_json_if_needed(value: Any, *, default: Optional[T] = None) -> Any:
    """Parse JSON strings while leaving already-materialized objects unchanged."""
    if value is None:
        return default
    if isinstance(value, (dict, list, tuple, int, float, bool)):
        return value
    if not isinstance(value, str):
        return default if default is not None else value
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default if default is not None else value


def safe_getattr_path(obj: Any, path: str, default: Any = None, *, separator: str = ".") -> Any:
    """Resolve a dotted attribute path safely."""
    if not path:
        return default
    current = obj
    for part in path.split(separator):
        if current is None or not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def is_json_serializable(value: Any) -> bool:
    """Return ``True`` if the value can be encoded by ``json.dumps`` as-is."""
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError, ValueError):
        return False


def to_json_safe(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = _DEFAULT_MAX_SERIALIZATION_DEPTH,
    max_items: int = _DEFAULT_MAX_COLLECTION_ITEMS,
    max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
) -> Any:
    """Recursively convert values into JSON-safe primitives."""
    if depth >= max_depth:
        return safe_repr(value, max_length=max_string_length)

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return value if len(value) <= max_string_length else value[: max_string_length - 3] + "..."

    if isinstance(value, bytes):
        return ensure_text(value)[:max_string_length]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        items = list(value.items())
        for index, (key, item) in enumerate(items[:max_items]):
            result[str(key)] = to_json_safe(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            )
        if len(items) > max_items:
            result["__truncated__"] = True
            result["__remaining_items__"] = len(items) - max_items
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        payload = [
            to_json_safe(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for item in sequence[:max_items]
        ]
        if len(sequence) > max_items:
            payload.append({"__truncated__": True, "__remaining_items__": len(sequence) - max_items})
        return payload

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return to_json_safe(
                value.to_dict(),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return to_json_safe(
                vars(value),
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
            )
        except Exception:
            pass

    return safe_repr(value, max_length=max_string_length)


def json_dumps(
    value: Any,
    *,
    pretty: bool = False,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    **kwargs: Any,
) -> str:
    """Serialize arbitrary values into JSON using safe coercion."""
    payload = to_json_safe(value)
    try:
        return json.dumps(
            payload,
            indent=2 if pretty else None,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            separators=(",", ": ") if pretty else (",", ":"),
            **kwargs,
        )
    except Exception as exc:
        raise BaseSerializationError(
            "Failed to serialize payload to JSON",
            None,
            context={"error": str(exc), "type": type(value).__name__},
        ) from exc


def json_loads(value: Union[str, bytes, bytearray], *, default: Any = None) -> Any:
    """Parse JSON with predictable fallback behavior."""
    if value is None:
        return default
    try:
        return json.loads(ensure_text(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def stable_fingerprint(value: Any, *, algorithm: str = "sha256", length: int = 16) -> str:
    """Compute a deterministic fingerprint for arbitrary structured data."""
    algo = algorithm.lower()
    if algo not in hashlib.algorithms_available:
        raise BaseValidationError(
            "Unsupported hashing algorithm",
            None,
            context={"algorithm": algorithm},
        )
    digest = hashlib.new(algo)
    digest.update(json_dumps(value, pretty=False, sort_keys=True).encode("utf-8"))
    hexdigest = digest.hexdigest()
    return hexdigest[:length] if length > 0 else hexdigest


def safe_repr(value: Any, *, max_length: int = _DEFAULT_MAX_STRING_LENGTH) -> str:
    """Return a bounded repr that will not raise secondary errors."""
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------
def redact_text(
    text: Any,
    *,
    replacement: str = "***REDACTED***",
    patterns: Optional[Sequence[re.Pattern[str]]] = None,
) -> str:
    """Redact common secret-bearing patterns from free-form text."""
    redacted = ensure_text(text)
    active_patterns = patterns or _DEFAULT_SECRET_PATTERNS
    for pattern in active_patterns:
        redacted = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + replacement, redacted)
    return redacted


def redact_mapping(
    mapping: Mapping[str, Any],
    *,
    secret_keys: Optional[Sequence[str]] = None,
    replacement: str = "***REDACTED***",
    recursive: bool = True,
) -> Dict[str, Any]:
    """Redact likely secret values in a mapping."""
    secrets = {key.lower() for key in (secret_keys or _DEFAULT_SECRET_KEYS)}
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        key_str = str(key)
        if key_str.lower() in secrets:
            result[key_str] = replacement
            continue
        if recursive and isinstance(value, Mapping):
            result[key_str] = redact_mapping(
                value,
                secret_keys=secret_keys,
                replacement=replacement,
                recursive=True,
            )
            continue
        if recursive and isinstance(value, list):
            result[key_str] = [
                redact_mapping(item, secret_keys=secret_keys, replacement=replacement, recursive=True)
                if isinstance(item, Mapping)
                else redact_text(item, replacement=replacement) if isinstance(item, str) else item
                for item in value
            ]
            continue
        if isinstance(value, str):
            result[key_str] = redact_text(value, replacement=replacement)
        else:
            result[key_str] = value
    return result


def redact_headers(headers: Mapping[str, Any], *, replacement: str = "***REDACTED***") -> Dict[str, Any]:
    """Redact security-sensitive HTTP header values."""
    return redact_mapping(
        headers,
        secret_keys=("authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"),
        replacement=replacement,
        recursive=False,
    )


# ---------------------------------------------------------------------------
# Endpoint and networking helpers
# ---------------------------------------------------------------------------
def normalize_endpoint(
    endpoint: Any,
    *,
    default_scheme: str = "http",
    strip_trailing_slash: bool = True,
    require_host: bool = True,
) -> str:
    """Normalize a host or URL-like endpoint into a canonical URL string."""
    if endpoint is None:
        raise BaseValidationError("endpoint must not be None", None)

    text = normalize_text(endpoint)
    if not text:
        raise BaseValidationError("endpoint must not be empty", None)

    if not _URL_SCHEME_RE.match(text):
        text = f"{default_scheme}://{text}"

    parsed = urlparse(text)
    if require_host and not parsed.netloc:
        raise BaseValidationError(
            "endpoint does not include a host",
            None,
            context={"endpoint": endpoint},
        )

    path = parsed.path or ""
    if strip_trailing_slash and path not in {"", "/"}:
        path = path.rstrip("/")

    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path,
        fragment="",
    )
    return urlunparse(normalized)


def join_endpoint(base: Any, *parts: Any, query: Optional[Mapping[str, Any]] = None) -> str:
    """Join a base endpoint with one or more path parts and query parameters."""
    normalized_base = normalize_endpoint(base)
    parsed = urlparse(normalized_base)

    base_segments = [segment for segment in parsed.path.split("/") if segment]
    extra_segments = [ensure_text(part).strip("/") for part in parts if ensure_text(part).strip("/")]
    joined_path = "/" + "/".join(base_segments + extra_segments) if (base_segments or extra_segments) else ""

    current_query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query:
        for key, value in query.items():
            if value is None:
                continue
            current_query[str(key)] = ensure_text(value)

    return urlunparse(parsed._replace(path=joined_path, query=urlencode(current_query, doseq=True)))


def split_host_port(value: Any, *, default_port: Optional[int] = None) -> Tuple[str, Optional[int]]:
    """Split ``host:port`` strings into a host and optional port."""
    text = normalize_text(value)
    match = _PORT_RE.match(text)
    if not match:
        raise BaseValidationError("Invalid host:port value", None, context={"value": value})
    host = match.group("host")
    port_text = match.group("port")
    port = int(port_text) if port_text is not None else default_port
    return host, port


def is_probably_url(value: Any) -> bool:
    """Heuristic URL detection used for lightweight branching logic."""
    if value is None:
        return False
    text = normalize_text(value)
    if not text:
        return False
    parsed = urlparse(text if _URL_SCHEME_RE.match(text) else f"http://{text}")
    return bool(parsed.netloc and ("." in parsed.netloc or parsed.netloc == "localhost"))


def is_port_open(host: str, port: int, *, timeout: float = 1.0) -> bool:
    """Check whether a TCP port is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Identifier and timestamp helpers
# ---------------------------------------------------------------------------
def utc_now_iso(*, timespec: str = "seconds") -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec=timespec)


def monotonic_ms() -> int:
    """Return the current monotonic clock in milliseconds."""
    return int(round(monotonic() * 1000))


def generate_request_id(prefix: str = "req", *, include_timestamp: bool = False) -> str:
    """Generate a short request identifier suitable for logs and tracing."""
    random_part = uuid.uuid4().hex[:12]
    if include_timestamp:
        return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{random_part}"
    return f"{prefix}_{random_part}"


def generate_deterministic_id(
    value: Any,
    *,
    prefix: str = "id",
    length: int = 16,
    algorithm: str = "sha256",
) -> str:
    """Generate a deterministic identifier from arbitrary structured input."""
    if length < 4:
        raise BaseValidationError("length must be >= 4", None, context={"length": length})
    return f"{prefix}_{stable_fingerprint(value, algorithm=algorithm, length=length)}"


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------
def compute_backoff_delay(attempt: int, policy: Optional[BackoffPolicy] = None) -> float:
    """Convenience wrapper around :class:`BackoffPolicy`."""
    return (policy or BackoffPolicy()).compute_delay(attempt)


def truncate_string(value: Any, max_length: int = 200, *, suffix: str = "...") -> str:
    """Truncate a value's textual form without raising."""
    text = ensure_text(value)
    if max_length <= 0 or len(text) <= max_length:
        return text
    if len(suffix) >= max_length:
        return suffix[:max_length]
    return text[: max_length - len(suffix)] + suffix


def pick(mapping: Mapping[K, V], *keys: K) -> Dict[K, V]:
    """Return a new mapping containing only the requested keys."""
    return {key: mapping[key] for key in keys if key in mapping}


def omit(mapping: Mapping[K, V], *keys: K) -> Dict[K, V]:
    """Return a new mapping excluding the requested keys."""
    excluded = set(keys)
    return {key: value for key, value in mapping.items() if key not in excluded}
