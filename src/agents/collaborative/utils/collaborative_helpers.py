"""
Production-grade shared helpers for the collaborative agent subsystem.

This module centralizes reusable helper primitives for SLAI's collaborative
multi-agent runtime. The helpers are intentionally scoped to cross-cutting
collaboration concerns: task envelopes, agent registration metadata, routing
telemetry, load snapshots, shared-memory safety wrappers, audit-event payloads,
result normalization, redaction, deterministic serialization, identifiers, and
retry/backoff utilities.

Scope
-----
The helpers here do not own orchestration, registry discovery, routing strategy,
policy evaluation, reliability state transitions, or shared-memory storage.
Those concerns remain in their dedicated modules. Instead, this module provides
stable, defensive contracts that those modules can share without reimplementing
normalization and telemetry logic.

Design principles
-----------------
1. Stable contracts: dataclasses and dictionaries have predictable shapes.
2. Safe diagnostics: all logging/audit helpers are JSON-safe and redact secrets.
3. Subsystem-aware: names, keys, envelopes, stats, and snapshots align with the
   existing collaborative manager, registry, router, shared memory, contracts,
   policy, and reliability modules.
4. Dependency-tolerant: config, logger, and collaboration error modules are
   optional at import time so tests and isolated linting can import this module.
5. Production-ready defaults: helper behavior is conservative when config is
   missing, malformed, or only partially populated.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
import time as time_module
import traceback
import uuid

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

from .config_loader import get_config_section, load_global_config
from .collaboration_error import CollaborationError as _collaboration_error_module
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Collaborative Helpers")
printer = PrettyPrinter()

T = TypeVar("T")

JsonDict = Dict[str, Any]
JsonValue = Union[None, bool, int, float, str, List[Any], Dict[str, Any]]
MemoryLike = Any
ExceptionPredicate = Callable[[BaseException], bool]

# ---------------------------------------------------------------------------
# Constants and config-backed defaults
# ---------------------------------------------------------------------------
DEFAULT_AGENT_STATS_KEY = "agent_stats"
DEFAULT_AGENT_KEY_PREFIX = "agent"
DEFAULT_AUDIT_KEY = "collaboration:audit_events"
DEFAULT_TASK_EVENT_KEY = "collaboration:task_events"
DEFAULT_RESULT_STATUS_SUCCESS = "success"
DEFAULT_RESULT_STATUS_ERROR = "error"
DEFAULT_RESULT_STATUS_REVIEW = "require_review"
DEFAULT_RESULT_STATUS_SKIPPED = "skipped"
DEFAULT_MAX_SERIALIZATION_DEPTH = 8
DEFAULT_MAX_COLLECTION_ITEMS = 100
DEFAULT_MAX_STRING_LENGTH = 4096
DEFAULT_MAX_AUDIT_EVENTS = 1000
DEFAULT_TASK_PRIORITY = 0
DEFAULT_MAX_TASK_RETRIES = 1
DEFAULT_NEW_AGENT_SUCCESS_BIAS = 1.0
DEFAULT_AGENT_TASK_MULTIPLIER = 5
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_BACKOFF_MAX_SECONDS = 30.0
DEFAULT_BACKOFF_JITTER_RATIO = 0.1

_WHITESPACE_RE = re.compile(r"\s+")
_NON_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_UNSAFE_CHANNEL_RE = re.compile(r"[^a-zA-Z0-9_.:/-]+")
_SECRET_VALUE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(bearer\s+)([a-z0-9\-._~+/]+=*)"),
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(token\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(secret\s*[=:]\s*)([^\s,;]+)"),
)
_DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "jwt",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "session",
        "session_id",
        "set_cookie",
        "token",
        "x-api-key",
    }
)

# Error symbols are resolved dynamically to support both the existing minimal
# error module and the production-ready replacement generated earlier.
CollaborationError = getattr(_collaboration_error_module, "CollaborationError", None)
CollaborationErrorType = getattr(_collaboration_error_module, "CollaborationErrorType", None)
OverloadError = getattr(_collaboration_error_module, "OverloadError", None)
NoCapableAgentError = getattr(_collaboration_error_module, "NoCapableAgentError", None)
RoutingFailureError = getattr(_collaboration_error_module, "RoutingFailureError", None)
DelegationFailureError = getattr(_collaboration_error_module, "DelegationFailureError", None)
RegistrationFailureError = getattr(_collaboration_error_module, "RegistrationFailureError", None)
SharedMemoryFailureError = getattr(_collaboration_error_module, "SharedMemoryFailureError", None)
SharedMemoryAccessError = getattr(_collaboration_error_module, "SharedMemoryAccessError", SharedMemoryFailureError)


class CollaborationStatus(str, Enum):
    """Common normalized statuses used by helper result payloads."""

    SUCCESS = DEFAULT_RESULT_STATUS_SUCCESS
    ERROR = DEFAULT_RESULT_STATUS_ERROR
    REQUIRE_REVIEW = DEFAULT_RESULT_STATUS_REVIEW
    SKIPPED = DEFAULT_RESULT_STATUS_SKIPPED


class AgentHealthStatus(str, Enum):
    """Normalized health/status labels for agent snapshots."""

    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Safe config access
# ---------------------------------------------------------------------------
def _safe_load_global_config() -> Dict[str, Any]:
    if load_global_config is None:
        return {}
    try:
        config = load_global_config()
        return dict(config or {}) if isinstance(config, Mapping) else {}
    except Exception as exc:  # pragma: no cover - defensive import/config path
        logger.debug("Unable to load collaborative global config: %s", exc)
        return {}


def _safe_get_config_section(section_name: str) -> Dict[str, Any]:
    if get_config_section is None:
        return {}
    try:
        section = get_config_section(section_name)
        return dict(section or {}) if isinstance(section, Mapping) else {}
    except Exception as exc:  # pragma: no cover - defensive import/config path
        logger.debug("Unable to load collaborative config section %s: %s", section_name, exc)
        return {}


GLOBAL_CONFIG: Dict[str, Any] = _safe_load_global_config()
COLLABORATIVE_HELPERS_CONFIG: Dict[str, Any] = _safe_get_config_section("collaborative_helpers")
COLLABORATION_CONFIG: Dict[str, Any] = _safe_get_config_section("collaboration")
TASK_ROUTING_CONFIG: Dict[str, Any] = _safe_get_config_section("task_routing")
AGENTS_CONFIG: Dict[str, Any] = _safe_get_config_section("agents")
SHARED_MEMORY_CONFIG: Dict[str, Any] = _safe_get_config_section("shared_memory")


def refresh_runtime_config() -> Dict[str, Dict[str, Any]]:
    """Refresh module-level config snapshots and return the updated sections.

    The project's config loader caches globally. This helper does not invalidate
    that cache; it simply re-reads the currently available runtime sections so
    callers can pick up changes made by tests or process initialization code.
    """

    global GLOBAL_CONFIG, COLLABORATIVE_HELPERS_CONFIG, COLLABORATION_CONFIG
    global TASK_ROUTING_CONFIG, AGENTS_CONFIG, SHARED_MEMORY_CONFIG

    GLOBAL_CONFIG = _safe_load_global_config()
    COLLABORATIVE_HELPERS_CONFIG = _safe_get_config_section("collaborative_helpers")
    COLLABORATION_CONFIG = _safe_get_config_section("collaboration")
    TASK_ROUTING_CONFIG = _safe_get_config_section("task_routing")
    AGENTS_CONFIG = _safe_get_config_section("agents")
    SHARED_MEMORY_CONFIG = _safe_get_config_section("shared_memory")
    return {
        "global": GLOBAL_CONFIG,
        "collaborative_helpers": COLLABORATIVE_HELPERS_CONFIG,
        "collaboration": COLLABORATION_CONFIG,
        "task_routing": TASK_ROUTING_CONFIG,
        "agents": AGENTS_CONFIG,
        "shared_memory": SHARED_MEMORY_CONFIG,
    }


# ---------------------------------------------------------------------------
# Time, identifiers, hashing, and low-level coercion
# ---------------------------------------------------------------------------
def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for audit payloads."""

    return utc_now().isoformat()


def epoch_seconds() -> float:
    """Return wall-clock epoch seconds."""

    return time_module.time()


def monotonic_ms() -> float:
    """Return monotonic time in milliseconds."""

    return time_module.monotonic() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    """Return elapsed monotonic milliseconds from a prior monotonic_ms value."""

    return round(max(0.0, monotonic_ms() - float(start_ms)), 3)


def normalize_whitespace(value: Any) -> str:
    """Normalize text while preserving word boundaries."""

    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(value).replace("\x00", " ")).strip()


def truncate_text(value: Any, max_length: Optional[int] = DEFAULT_MAX_STRING_LENGTH, *, suffix: str = "...") -> str:
    """Convert a value to text and truncate safely."""

    if value is None:
        return ""
    text = str(value)
    if max_length is None or max_length < 0 or len(text) <= max_length:
        return text
    suffix = suffix or ""
    return text[: max(0, int(max_length) - len(suffix))] + suffix


def normalize_identifier_component(
    value: Any,
    *,
    default: str = "item",
    lowercase: bool = False,
    max_length: int = 120,
    separator: str = "_",
) -> str:
    """Normalize free-form text into a conservative identifier component."""

    text = normalize_whitespace(value)
    if lowercase:
        text = text.lower()
    text = _NON_IDENTIFIER_RE.sub(separator, text).strip(separator)
    if max_length > 0:
        text = text[:max_length].strip(separator)
    return text or default


def generate_uuid(prefix: str = "id", *, length: int = 32, separator: str = "_") -> str:
    """Generate a collision-resistant identifier with a normalized prefix."""

    normalized_prefix = normalize_identifier_component(prefix, default="id", lowercase=True, separator="-")
    token = uuid.uuid4().hex[: max(8, min(32, int(length)))]
    return f"{normalized_prefix}{separator}{token}"


def generate_task_id(prefix: str = "task") -> str:
    """Generate a task identifier."""

    return generate_uuid(prefix, length=24)


def generate_correlation_id(prefix: str = "collab") -> str:
    """Generate a correlation identifier for distributed tracing."""

    return generate_uuid(prefix, length=24)


def generate_agent_session_id(prefix: str = "agent_session") -> str:
    """Generate a session identifier for agent runtime snapshots."""

    return generate_uuid(prefix, length=24)


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Best-effort boolean coercion for config and metadata values."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


def coerce_int(value: Any, *, default: int = 0, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    """Coerce a value into an int with optional bounds."""

    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not an integer value")
        result = int(float(value))
    except Exception:
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def coerce_float(
    value: Any,
    *,
    default: float = 0.0,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Coerce a value into a finite float with optional bounds."""

    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not a float value")
        result = float(value)
    except Exception:
        result = float(default)
    if math.isnan(result) or math.isinf(result):
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def clamp(value: Union[int, float], minimum: Union[int, float], maximum: Union[int, float]) -> Union[int, float]:
    """Clamp a numeric value between two bounds."""

    return max(minimum, min(maximum, value))


def ensure_list(value: Any, *, drop_none: bool = True) -> List[Any]:
    """Return value as a list without treating strings/bytes as iterables."""

    if value is None:
        return [] if drop_none else [None]
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, set):
        items = list(value)
    elif isinstance(value, frozenset):
        items = list(value)
    else:
        items = [value]
    return [item for item in items if item is not None] if drop_none else list(items)


def ensure_mapping(value: Any, *, field_name: str = "value", allow_none: bool = False) -> Dict[str, Any]:
    """Validate and normalize a mapping-like value into a dictionary."""

    if value is None:
        if allow_none:
            return {}
        raise ValueError(f"{field_name} must be a mapping, received None.")
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"{field_name} must be a mapping-like object, received {type(value).__name__}.")


def ensure_sequence(
    value: Any,
    *,
    field_name: str = "value",
    allow_none: bool = False,
    coerce_scalar: bool = False,
) -> Tuple[Any, ...]:
    """Validate sequence-like input while avoiding accidental string iteration."""

    if value is None:
        if allow_none:
            return ()
        raise ValueError(f"{field_name} must be a sequence, received None.")
    if isinstance(value, (str, bytes, bytearray)):
        if coerce_scalar:
            return (value,)
        raise ValueError(f"{field_name} must be a sequence, not a scalar string/bytes value.")
    if isinstance(value, Sequence):
        return tuple(value)
    if coerce_scalar:
        return (value,)
    raise ValueError(f"{field_name} must be a sequence-like object, received {type(value).__name__}.")


def require_non_empty_string(value: Any, field_name: str, *, max_length: Optional[int] = None) -> str:
    """Validate and normalize a required string field."""

    text = normalize_whitespace(value)
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if max_length is not None and max_length > 0 and len(text) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length {max_length}.")
    return text


# ---------------------------------------------------------------------------
# JSON-safe serialization, hashing, redaction, and logging safety
# ---------------------------------------------------------------------------
def safe_repr(value: Any, *, max_length: int = DEFAULT_MAX_STRING_LENGTH) -> str:
    """Return a bounded repr that will not raise secondary errors."""

    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    return truncate_text(text, max_length)


def json_safe(
    value: Any,
    *,
    max_depth: int = DEFAULT_MAX_SERIALIZATION_DEPTH,
    max_items: int = DEFAULT_MAX_COLLECTION_ITEMS,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
    _depth: int = 0,
) -> JsonValue:
    """Convert arbitrary values into JSON-safe primitives.

    This helper is deliberately conservative because results are used in
    telemetry, audit events, error contexts, and shared-memory persistence.
    """

    if _depth >= max_depth:
        return safe_repr(value, max_length=max_string_length)

    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            return truncate_text(value, max_string_length)
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value

    if isinstance(value, bytes):
        chunk = value[:max_string_length]
        try:
            return chunk.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "length": len(value),
                "truncated": len(value) > len(chunk),
                "data": base64.b64encode(chunk).decode("ascii"),
            }

    if isinstance(value, (datetime,)):
        return value.isoformat()

    if isinstance(value, timedelta):
        return value.total_seconds()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if is_dataclass(value) and not isinstance(value, type):
        try:
            return json_safe(
                asdict(value),
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
        except Exception:
            return safe_repr(value, max_length=max_string_length)

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            result[str(json_safe(key, max_depth=max_depth, max_items=max_items, max_string_length=max_string_length, _depth=_depth + 1))] = json_safe(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
        if len(items) > max_items:
            result["__truncated__"] = True
            result["__remaining_items__"] = len(items) - max_items
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        seq_payload: List[JsonValue] = [
            json_safe(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
            for item in sequence[:max_items]
        ]
        if len(sequence) > max_items:
            # Append a dict marker; we cast because the list element type is JsonValue,
            # and a dict is a valid JsonValue.
            seq_payload.append(
                cast(JsonValue, {"__truncated__": True, "__remaining_items__": len(sequence) - max_items})
            )
        return seq_payload

    if isinstance(value, BaseException):
        payload: Dict[str, Any] = {
            "type": type(value).__name__,
            "module": type(value).__module__,
            "message": truncate_text(str(value), max_string_length),
        }
        context = getattr(value, "context", None)
        if context is not None:
            payload["context"] = json_safe(
                context,
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
        return payload

    to_dict_method = getattr(value, "to_dict", None)
    if callable(to_dict_method):
        try:
            return json_safe(to_dict_method(),
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return json_safe(
                vars(value),
                max_depth=max_depth,
                max_items=max_items,
                max_string_length=max_string_length,
                _depth=_depth + 1,
            )
        except Exception:
            pass

    return safe_repr(value, max_length=max_string_length)


def stable_json_dumps(value: Any, *, sort_keys: bool = True, indent: Optional[int] = None) -> str:
    """Serialize a value deterministically for hashing, caching, and telemetry."""

    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=sort_keys, indent=indent, separators=(",", ":") if indent is None else None)


def json_loads(value: Union[str, bytes, bytearray], *, default: Any = None) -> Any:
    """Parse JSON safely and return default when decoding fails."""

    if value is None:
        return default
    try:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="replace")
        return json.loads(str(value))
    except Exception:
        return default


def stable_hash(value: Any, *, algorithm: str = "sha256", length: int = 16) -> str:
    """Create a deterministic short hash for any JSON-serializable-ish value."""

    try:
        digest = hashlib.new((algorithm or "sha256").lower())
    except Exception:
        digest = hashlib.sha256()
    digest.update(stable_json_dumps(value).encode("utf-8", errors="replace"))
    hexdigest = digest.hexdigest()
    return hexdigest[: max(1, int(length))] if length else hexdigest


def generate_idempotency_key(
    payload: Any,
    *,
    namespace: str = "collaboration",
    task_type: Optional[str] = None,
    source: Optional[str] = None,
    length: int = 64,
) -> str:
    """Generate a deterministic idempotency key from stable task properties."""

    basis = {
        "namespace": namespace,
        "task_type": normalize_task_type(task_type, allow_empty=True),
        "source": source,
        "payload": json_safe(payload),
    }
    return stable_hash(basis, length=length)


def is_sensitive_key(key: Any, *, sensitive_keys: Optional[Iterable[str]] = None) -> bool:
    """Return True when a mapping key is likely to contain sensitive data."""

    key_text = str(key or "").strip().lower().replace("-", "_")
    active = set(_DEFAULT_SENSITIVE_KEYS)
    active.update(str(item).strip().lower().replace("-", "_") for item in (sensitive_keys or ()) if str(item).strip())
    return key_text in active or any(token in key_text for token in ("password", "secret", "token", "credential", "private_key"))


def redact_text(text: Any, *, replacement: str = "***REDACTED***") -> str:
    """Redact common secret-bearing patterns from free-form text."""

    redacted = str(text or "")
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}{replacement}", redacted)
    return redacted


def redact_sensitive_value(value: Any, *, replacement: str = "***REDACTED***", preserve_length: bool = False) -> Any:
    """Redact a sensitive scalar/container value while preserving shape when useful."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): redact_sensitive_value(item, replacement=replacement, preserve_length=preserve_length) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_sensitive_value(item, replacement=replacement, preserve_length=preserve_length) for item in value]
    if isinstance(value, bytes):
        return f"{replacement}[bytes:{len(value)}]" if preserve_length else replacement
    if preserve_length:
        return f"{replacement}[len:{len(str(value))}]"
    return replacement


def redact_mapping(
    mapping: Optional[Mapping[str, Any]],
    *,
    replacement: str = "***REDACTED***",
    sensitive_keys: Optional[Iterable[str]] = None,
    preserve_length: bool = False,
) -> Dict[str, Any]:
    """Recursively redact likely secret values in a mapping."""

    if mapping is None:
        return {}
    source = ensure_mapping(mapping, field_name="mapping", allow_none=True)
    redacted: Dict[str, Any] = {}
    for key, value in source.items():
        key_text = str(key)
        if is_sensitive_key(key_text, sensitive_keys=sensitive_keys):
            redacted[key_text] = redact_sensitive_value(value, replacement=replacement, preserve_length=preserve_length)
        elif isinstance(value, Mapping):
            redacted[key_text] = redact_mapping(value, replacement=replacement, sensitive_keys=sensitive_keys, preserve_length=preserve_length)
        elif isinstance(value, (list, tuple)):
            redacted[key_text] = [
                redact_mapping(item, replacement=replacement, sensitive_keys=sensitive_keys, preserve_length=preserve_length)
                if isinstance(item, Mapping)
                else redact_text(item, replacement=replacement)
                if isinstance(item, str)
                else item
                for item in value
            ]
        elif isinstance(value, str):
            redacted[key_text] = redact_text(value, replacement=replacement)
        else:
            redacted[key_text] = json_safe(value)
    return redacted


def sanitize_for_logging(
    value: Any,
    *,
    replacement: str = "***REDACTED***",
    preserve_length: bool = False,
    max_depth: int = DEFAULT_MAX_SERIALIZATION_DEPTH,
    _depth: int = 0,
) -> Any:
    """Produce a JSON-safe and secret-redacted representation of arbitrary data."""

    if _depth >= max_depth:
        return safe_repr(value)
    if isinstance(value, Mapping):
        return redact_mapping(
            {
                str(key): sanitize_for_logging(
                    item,
                    replacement=replacement,
                    preserve_length=preserve_length,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
                for key, item in value.items()
            },
            replacement=replacement,
            preserve_length=preserve_length,
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            sanitize_for_logging(
                item,
                replacement=replacement,
                preserve_length=preserve_length,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]
    if isinstance(value, BaseException):
        return exception_to_error_payload(value, include_traceback=False)
    return json_safe(value, max_depth=max_depth)


# ---------------------------------------------------------------------------
# Generic mapping and metadata helpers
# ---------------------------------------------------------------------------
def merge_mappings(
    *mappings: Optional[Mapping[str, Any]],
    deep: bool = True,
    drop_none: bool = False,
) -> Dict[str, Any]:
    """Merge mapping-like values into a new dictionary. Later mappings win."""

    merged: Dict[str, Any] = {}
    for candidate in mappings:
        if candidate is None:
            continue
        current = ensure_mapping(candidate, field_name="mapping")
        for key, value in current.items():
            if drop_none and value is None:
                continue
            if deep and key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
                merged[key] = merge_mappings(merged[key], value, deep=True, drop_none=drop_none)
            else:
                merged[key] = value
    return merged


def prune_none(value: Any, *, drop_empty: bool = False) -> Any:
    """Remove None values recursively from mappings and sequences."""

    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if item is None:
                continue
            cleaned = prune_none(item, drop_empty=drop_empty)
            if drop_empty and cleaned in ({}, [], (), ""):
                continue
            result[key] = cleaned
        return result
    if isinstance(value, list):
        return [prune_none(item, drop_empty=drop_empty) for item in value if item is not None]
    if isinstance(value, tuple):
        return tuple(prune_none(item, drop_empty=drop_empty) for item in value if item is not None)
    return value


def flatten_mapping(data: Mapping[str, Any], *, parent_key: str = "", separator: str = ".", max_depth: int = 10) -> Dict[str, Any]:
    """Flatten a nested mapping into dotted keys."""

    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
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


def normalize_metadata(metadata: Optional[Mapping[str, Any]], *, drop_none: bool = True, redact: bool = False) -> Dict[str, Any]:
    """Normalize metadata for envelopes, telemetry, and error contexts."""

    if metadata is None:
        return {}
    source = ensure_mapping(metadata, field_name="metadata")
    normalized: Dict[str, Any] = {}
    for key, value in source.items():
        if drop_none and value is None:
            continue
        normalized[str(key)] = json_safe(value)
    return redact_mapping(normalized) if redact else normalized


def normalize_tags(tags: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    """Normalize tags while preserving order and removing duplicates."""

    if tags is None:
        return ()
    if isinstance(tags, (str, bytes)):
        values = [tags]
    else:
        values = list(tags)
    deduplicated: Dict[str, None] = {}
    for item in values:
        text = normalize_whitespace(item)
        if text:
            deduplicated[text] = None
    return tuple(deduplicated.keys())


# ---------------------------------------------------------------------------
# Dataclasses for collaborative contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackoffPolicy:
    """Reusable exponential backoff configuration for collaborative operations."""

    max_attempts: int = DEFAULT_MAX_TASK_RETRIES
    base_delay_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS
    multiplier: float = DEFAULT_BACKOFF_MULTIPLIER
    max_delay_seconds: float = DEFAULT_BACKOFF_MAX_SECONDS
    jitter_ratio: float = DEFAULT_BACKOFF_JITTER_RATIO

    def delay_for_attempt(self, attempt: int) -> float:
        """Return bounded delay for a 1-based retry attempt."""

        return calculate_backoff_delay(
            attempt,
            base_delay_seconds=self.base_delay_seconds,
            multiplier=self.multiplier,
            max_delay_seconds=self.max_delay_seconds,
            jitter_ratio=self.jitter_ratio,
        )


class Stopwatch:
    """Minimal monotonic stopwatch for lightweight timing code paths."""

    def __init__(self, start_immediately: bool = True):
        self._started_ms: Optional[float] = None
        self._stopped_ms: Optional[float] = None
        if start_immediately:
            self.start()

    def start(self) -> "Stopwatch":
        self._started_ms = monotonic_ms()
        self._stopped_ms = None
        return self

    def stop(self) -> float:
        if self._started_ms is None:
            raise ValueError("Stopwatch has not been started.")
        self._stopped_ms = monotonic_ms()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        if self._started_ms is None:
            return 0.0
        end = self._stopped_ms if self._stopped_ms is not None else monotonic_ms()
        return round(max(0.0, end - self._started_ms), 3)

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ms / 1000.0


@dataclass(frozen=True)
class ValidationResult:
    """Stable validation result shape used by helper validators."""

    valid: bool
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskEnvelope:
    """Canonical task envelope for collaboration runtime boundaries."""

    task_id: str
    task_type: str
    payload: Dict[str, Any]
    required_capabilities: Tuple[str, ...] = ()
    priority: int = DEFAULT_TASK_PRIORITY
    retry_limit: int = DEFAULT_MAX_TASK_RETRIES
    submitted_at: float = field(default_factory=epoch_seconds)
    deadline_at: Optional[float] = None
    correlation_id: str = field(default_factory=generate_correlation_id)
    idempotency_key: str = ""
    source: str = "unknown"
    tags: Tuple[str, ...] = ()
    fallback_plan: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return self.deadline_at is not None and epoch_seconds() > self.deadline_at

    @property
    def time_remaining_seconds(self) -> Optional[float]:
        if self.deadline_at is None:
            return None
        return max(0.0, self.deadline_at - epoch_seconds())

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = asdict(self)
        if redact:
            payload = redact_mapping(payload)
        return payload


@dataclass(frozen=True)
class RoutingAttempt:
    """Normalized record for an individual routing/delegation attempt."""

    agent_name: str
    task_type: str
    status: str
    score: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_ms: Optional[float] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class AgentSnapshot:
    """Serializable agent status and telemetry snapshot."""

    name: str
    capabilities: Tuple[str, ...] = ()
    status: str = AgentHealthStatus.UNKNOWN.value
    version: Optional[Union[int, float, str]] = None
    successes: int = 0
    failures: int = 0
    active_tasks: int = 0
    last_seen: Optional[float] = None
    heartbeat_age_seconds: Optional[float] = None
    success_rate: float = DEFAULT_NEW_AGENT_SUCCESS_BIAS
    failure_rate: float = 0.0
    score: Optional[float] = None
    circuit_state: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class LoadSnapshot:
    """Point-in-time collaborative load snapshot."""

    current_load: int
    max_load: int
    agent_count: int
    active_agent_count: int
    overloaded: bool
    utilization: float
    captured_at: float = field(default_factory=epoch_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperationResult:
    """Stable result shape for helper-assisted collaborative operations."""

    status: str
    message: str = ""
    action: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    correlation_id: str = field(default_factory=generate_correlation_id)
    timestamp_utc: str = field(default_factory=utc_timestamp)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


# ---------------------------------------------------------------------------
# Task, capability, and envelope normalization
# ---------------------------------------------------------------------------
def normalize_task_type(task_type: Any, *, allow_empty: bool = False) -> str:
    """Normalize a task type/capability identifier while preserving case."""

    text = normalize_whitespace(task_type)
    if not text:
        if allow_empty:
            return ""
        raise ValueError("task_type must be a non-empty string.")
    return text


def normalize_capability(capability: Any) -> str:
    """Normalize a single capability string."""

    return require_non_empty_string(capability, "capability", max_length=160)


def normalize_capabilities(capabilities: Optional[Iterable[Any]], *, include_task_type: Optional[str] = None) -> Tuple[str, ...]:
    """Normalize capabilities while preserving order and removing duplicates."""

    values: List[Any] = []
    if include_task_type:
        values.append(include_task_type)
    if capabilities is not None:
        if isinstance(capabilities, (str, bytes)):
            values.append(capabilities)
        else:
            values.extend(list(capabilities))
    deduplicated: Dict[str, None] = {}
    for item in values:
        text = normalize_whitespace(item)
        if text:
            deduplicated[text] = None
    return tuple(deduplicated.keys())


def normalize_task_payload(payload: Optional[Mapping[str, Any]], *, allow_none: bool = True, redact: bool = False) -> Dict[str, Any]:
    """Normalize a task payload into a concrete JSON-safe mapping."""

    source = ensure_mapping(payload, field_name="payload", allow_none=allow_none)
    normalized = {str(key): json_safe(value) for key, value in source.items()}
    return redact_mapping(normalized) if redact else normalized


def normalize_priority(priority: Any = None) -> int:
    """Normalize a task priority with config-backed default."""

    default_priority = coerce_int(AGENTS_CONFIG.get("default_priority"), default=DEFAULT_TASK_PRIORITY)
    return coerce_int(priority if priority is not None else default_priority, default=default_priority)


def normalize_retry_limit(retries: Any = None) -> int:
    """Normalize task retry limits using task_routing.retry_policy when absent."""

    retry_config = TASK_ROUTING_CONFIG.get("retry_policy") if isinstance(TASK_ROUTING_CONFIG.get("retry_policy"), Mapping) else {}
    configured = retry_config.get("max_attempts", DEFAULT_MAX_TASK_RETRIES) if isinstance(retry_config, Mapping) else DEFAULT_MAX_TASK_RETRIES
    return coerce_int(retries if retries is not None else configured, default=DEFAULT_MAX_TASK_RETRIES, minimum=0, maximum=100)


def coerce_deadline_at(value: Any = None, *, now: Optional[float] = None) -> Optional[float]:
    """Normalize a deadline value into epoch seconds.

    Accepted values: None, int/float epoch seconds, datetime, timedelta, or
    numeric strings. A timedelta is treated as a relative deadline from `now`.
    """

    if value is None:
        return None
    current = epoch_seconds() if now is None else float(now)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, timedelta):
        return current + value.total_seconds()
    try:
        timestamp = float(value)
    except Exception as exc:
        raise ValueError(f"deadline_at must be a datetime, timedelta, epoch seconds, or None; got {type(value).__name__}.") from exc
    return timestamp


def normalize_fallback_plan(plan: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    """Normalize a fallback plan into task/subtask identifiers."""

    if plan is None:
        return ()
    if isinstance(plan, (str, bytes)):
        values = [plan]
    else:
        values = list(plan)
    return tuple(normalize_task_type(item) for item in values if normalize_whitespace(item))


def get_configured_fallback_plan(task_type: str) -> Tuple[str, ...]:
    """Return configured fallback steps for a task type, if present."""

    fallback_plans = TASK_ROUTING_CONFIG.get("fallback_plans", {})
    if not isinstance(fallback_plans, Mapping):
        return ()
    plan = fallback_plans.get(task_type) or fallback_plans.get(normalize_task_type(task_type, allow_empty=True))
    return normalize_fallback_plan(plan) if plan else ()


def build_task_envelope(
    envelope: Optional[Union[TaskEnvelope, Mapping[str, Any]]] = None,
    *,
    task_type: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
    required_capabilities: Optional[Iterable[Any]] = None,
    priority: Optional[Any] = None,
    retry_limit: Optional[Any] = None,
    deadline_at: Any = None,
    timeout_seconds: Optional[Union[int, float]] = None,
    correlation_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    task_id: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[Iterable[Any]] = None,
    fallback_plan: Optional[Iterable[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> TaskEnvelope:
    """Build and normalize a canonical task envelope.

    The function accepts a partial existing mapping or TaskEnvelope and lets
    explicit keyword arguments override that base. It does not execute policy,
    routing, or contract validation; it only gives those systems a stable input.
    """

    base = envelope.to_dict(redact=False) if isinstance(envelope, TaskEnvelope) else ensure_mapping(envelope, field_name="envelope", allow_none=True)
    resolved_task_type = normalize_task_type(task_type or base.get("task_type"))
    resolved_payload = normalize_task_payload(payload if payload is not None else base.get("payload", {}), allow_none=True)
    submitted_at = coerce_float(base.get("submitted_at"), default=epoch_seconds(), minimum=0.0)

    if deadline_at is None and timeout_seconds is not None:
        deadline_at = timedelta(seconds=coerce_float(timeout_seconds, default=0.0, minimum=0.0))
    elif deadline_at is None:
        deadline_at = base.get("deadline_at")

    resolved_fallback = normalize_fallback_plan(fallback_plan if fallback_plan is not None else base.get("fallback_plan"))
    if not resolved_fallback:
        resolved_fallback = get_configured_fallback_plan(resolved_task_type)

    resolved_metadata = normalize_metadata(merge_mappings(base.get("metadata"), metadata), drop_none=True)
    resolved_source = normalize_whitespace(source or base.get("source") or "unknown") or "unknown"
    resolved_correlation_id = normalize_whitespace(correlation_id or base.get("correlation_id") or generate_correlation_id())
    resolved_task_id = normalize_whitespace(task_id or base.get("task_id") or generate_task_id())
    resolved_idempotency_key = normalize_whitespace(
        idempotency_key
        or base.get("idempotency_key")
        or generate_idempotency_key(resolved_payload, task_type=resolved_task_type, source=resolved_source)
    )

    return TaskEnvelope(
        task_id=resolved_task_id,
        task_type=resolved_task_type,
        payload=resolved_payload,
        required_capabilities=normalize_capabilities(
            required_capabilities if required_capabilities is not None else base.get("required_capabilities"),
        ),
        priority=normalize_priority(priority if priority is not None else base.get("priority")),
        retry_limit=normalize_retry_limit(retry_limit if retry_limit is not None else base.get("retry_limit")),
        submitted_at=submitted_at,
        deadline_at=coerce_deadline_at(deadline_at, now=submitted_at),
        correlation_id=resolved_correlation_id,
        idempotency_key=resolved_idempotency_key,
        source=resolved_source,
        tags=normalize_tags(tags if tags is not None else base.get("tags")),
        fallback_plan=resolved_fallback,
        metadata=resolved_metadata,
    )


def validate_task_envelope(envelope: Union[TaskEnvelope, Mapping[str, Any]]) -> ValidationResult:
    """Validate a task envelope shape without executing policy/contract checks."""

    errors: List[str] = []
    warnings: List[str] = []
    try:
        env = envelope if isinstance(envelope, TaskEnvelope) else build_task_envelope(envelope)
    except Exception as exc:
        return ValidationResult(valid=False, errors=(str(exc),), warnings=())

    if not env.task_id:
        errors.append("task_id is required")
    if not env.task_type:
        errors.append("task_type is required")
    if not isinstance(env.payload, Mapping):
        errors.append("payload must be a mapping")
    if env.retry_limit < 0:
        errors.append("retry_limit must be >= 0")
    if env.deadline_at is not None and env.deadline_at <= env.submitted_at:
        warnings.append("deadline_at is not later than submitted_at")
    if env.expired:
        warnings.append("task envelope is already expired")
    return ValidationResult(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# Agent metadata, registration, and snapshot helpers
# ---------------------------------------------------------------------------
def normalize_agent_name(agent_name: Any) -> str:
    """Normalize an agent name for runtime metadata and memory keys."""

    return require_non_empty_string(agent_name, "agent_name", max_length=200)


def agent_memory_key(agent_name: Any, *, prefix: str = DEFAULT_AGENT_KEY_PREFIX) -> str:
    """Return the shared-memory key used for an agent heartbeat/status record."""

    return f"{prefix}:{normalize_agent_name(agent_name)}"


def normalize_channel_name(channel: Any, *, default: str = "collaboration") -> str:
    """Normalize pub/sub channel names without destroying hierarchical separators."""

    text = normalize_whitespace(channel) or default
    text = _UNSAFE_CHANNEL_RE.sub("_", text).strip("_")
    return text or default


def task_channel(task_type: Any, *, prefix: str = "task") -> str:
    """Return a normalized task pub/sub channel name."""

    return normalize_channel_name(f"{prefix}/{normalize_task_type(task_type)}")


def agent_channel(agent_name: Any, *, prefix: str = "agent") -> str:
    """Return a normalized agent pub/sub channel name."""

    return normalize_channel_name(f"{prefix}/{normalize_agent_name(agent_name)}")


def is_agent_like(value: Any) -> bool:
    """Return True when an object looks executable as a collaborative agent."""

    return hasattr(value, "execute") and callable(getattr(value, "execute"))


def extract_agent_capabilities(agent_or_meta: Any, *, default: Optional[Iterable[Any]] = None) -> Tuple[str, ...]:
    """Extract capabilities from an agent instance, class, or metadata mapping."""

    capabilities: Any = None
    if isinstance(agent_or_meta, Mapping):
        capabilities = agent_or_meta.get("capabilities")
        if capabilities is None and isinstance(agent_or_meta.get("meta"), Mapping):
            capabilities = agent_or_meta["meta"].get("capabilities")
    if capabilities is None and hasattr(agent_or_meta, "capabilities"):
        capabilities = getattr(agent_or_meta, "capabilities")
        if callable(capabilities):
            try:
                capabilities = capabilities()
            except TypeError:
                capabilities = None
    if capabilities is None:
        capabilities = default
    return normalize_capabilities(capabilities)


def normalize_agent_meta(
    agent_name: Any,
    agent_or_meta: Any = None,
    *,
    capabilities: Optional[Iterable[Any]] = None,
    version: Optional[Union[int, float, str]] = None,
    instance: Any = None,
    agent_class: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize agent metadata into the registry-compatible shape."""

    name = normalize_agent_name(agent_name)
    base = dict(agent_or_meta) if isinstance(agent_or_meta, Mapping) else {}
    resolved_instance = instance if instance is not None else base.get("instance")
    if resolved_instance is None and agent_or_meta is not None and not isinstance(agent_or_meta, Mapping):
        resolved_instance = agent_or_meta
    resolved_class = agent_class if agent_class is not None else base.get("class")
    if resolved_class is None and resolved_instance is not None:
        resolved_class = type(resolved_instance)
    if resolved_class is None and agent_or_meta is not None and isinstance(agent_or_meta, type):
        resolved_class = agent_or_meta
    resolved_capabilities = normalize_capabilities(capabilities if capabilities is not None else extract_agent_capabilities(base or resolved_instance or resolved_class))
    if not resolved_capabilities:
        raise ValueError(f"Agent '{name}' must declare at least one capability.")
    resolved_version = version if version is not None else base.get("version", 1.0)
    normalized = merge_mappings(
        base,
        {
            "class": resolved_class,
            "instance": resolved_instance,
            "capabilities": list(resolved_capabilities),
            "version": resolved_version,
            "metadata": normalize_metadata(merge_mappings(base.get("metadata"), metadata), drop_none=True),
        },
        deep=True,
        drop_none=False,
    )
    return normalized


def build_agent_registration(
    agent_name: Any,
    agent_instance: Any = None,
    capabilities: Optional[Iterable[Any]] = None,
    *,
    meta: Optional[Mapping[str, Any]] = None,
    version: Optional[Union[int, float, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a registry.batch_register-compatible registration dictionary."""

    name = normalize_agent_name(agent_name)
    normalized_meta = normalize_agent_meta(
        name,
        meta or agent_instance,
        capabilities=capabilities,
        version=version,
        metadata=metadata,
    )
    return {"name": name, "meta": normalized_meta}


def build_agent_batch_registrations(agents: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a collection of agent registration definitions."""

    registrations: List[Dict[str, Any]] = []
    for item in agents:
        data = ensure_mapping(item, field_name="agent_registration")
        name = data.get("name") or data.get("agent_name")
        meta = data.get("meta")
        instance = data.get("instance") or data.get("agent_instance")
        registrations.append(
            build_agent_registration(
                name,
                agent_instance=instance,
                capabilities=data.get("capabilities"),
                meta=meta,
                version=data.get("version"),
                metadata=data.get("metadata"),
            )
        )
    return registrations


def normalize_agent_stats(stats: Optional[Mapping[str, Any]], *, new_agent_success_bias: float = DEFAULT_NEW_AGENT_SUCCESS_BIAS) -> Dict[str, Any]:
    """Normalize a single agent stats row."""

    source = ensure_mapping(stats, field_name="agent_stats", allow_none=True)
    successes = coerce_int(source.get("successes"), default=0, minimum=0)
    failures = coerce_int(source.get("failures"), default=0, minimum=0)
    active_tasks = coerce_int(source.get("active_tasks"), default=0, minimum=0)
    last_seen = source.get("last_seen")
    last_seen_float = coerce_float(last_seen, default=0.0, minimum=0.0) if last_seen is not None else 0.0
    total = successes + failures
    success_rate = successes / total if total else float(new_agent_success_bias)
    failure_rate = failures / total if total else 0.0
    row = merge_mappings(
        source,
        {
            "successes": successes,
            "failures": failures,
            "active_tasks": active_tasks,
            "last_seen": last_seen_float,
            "success_rate": round(success_rate, 6),
            "failure_rate": round(failure_rate, 6),
        },
        deep=True,
    )
    return row


def normalize_agent_stats_map(stats: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize the entire shared-memory agent_stats mapping."""

    source = ensure_mapping(stats, field_name="agent_stats", allow_none=True)
    normalized: Dict[str, Dict[str, Any]] = {}
    for agent_name, row in source.items():
        if isinstance(row, Mapping):
            normalized[str(agent_name)] = normalize_agent_stats(row)
    return normalized


def calculate_success_rate(stats: Optional[Mapping[str, Any]], *, default: float = DEFAULT_NEW_AGENT_SUCCESS_BIAS) -> float:
    """Calculate success rate from a single agent stats row."""

    row = normalize_agent_stats(stats, new_agent_success_bias=default)
    return float(row.get("success_rate", default))


def calculate_agent_score(
    stats: Optional[Mapping[str, Any]],
    *,
    success_rate_weight: Optional[float] = None,
    load_penalty_weight: Optional[float] = None,
    new_agent_success_bias: float = DEFAULT_NEW_AGENT_SUCCESS_BIAS,
) -> float:
    """Calculate a weighted routing score consistent with WeightedRouterStrategy."""

    row = normalize_agent_stats(stats, new_agent_success_bias=new_agent_success_bias)
    success_weight = coerce_float(
        success_rate_weight if success_rate_weight is not None else TASK_ROUTING_CONFIG.get("weight_success_rate", 1.0),
        default=1.0,
    )
    load_penalty = coerce_float(
        load_penalty_weight if load_penalty_weight is not None else TASK_ROUTING_CONFIG.get("weight_load_penalty", 0.25),
        default=0.25,
    )
    return round((success_weight * float(row.get("success_rate", new_agent_success_bias))) - (load_penalty * int(row.get("active_tasks", 0))), 6)


def build_agent_snapshot(
    agent_name: Any,
    *,
    meta: Optional[Mapping[str, Any]] = None,
    stats: Optional[Mapping[str, Any]] = None,
    heartbeat: Optional[Mapping[str, Any]] = None,
    circuit_status: Optional[Mapping[str, Any]] = None,
    now: Optional[float] = None,
) -> AgentSnapshot:
    """Build a serializable agent snapshot from registry/stats/heartbeat data."""

    name = normalize_agent_name(agent_name)
    current = epoch_seconds() if now is None else float(now)
    meta_dict = ensure_mapping(meta, field_name="meta", allow_none=True)
    stats_row = normalize_agent_stats(stats)
    heartbeat_row = ensure_mapping(heartbeat, field_name="heartbeat", allow_none=True)
    last_seen = coerce_float(
        heartbeat_row.get("last_seen", stats_row.get("last_seen", 0.0)),
        default=0.0,
        minimum=0.0,
    )
    heartbeat_age = round(max(0.0, current - last_seen), 3) if last_seen else None
    status = normalize_whitespace(heartbeat_row.get("status") or meta_dict.get("status") or AgentHealthStatus.UNKNOWN.value)
    if heartbeat_age is not None:
        stale_after = coerce_float(COLLABORATION_CONFIG.get("health_check_interval", 60), default=60.0, minimum=1.0)
        if heartbeat_age > stale_after:
            status = AgentHealthStatus.STALE.value
    circuit_state = None
    if isinstance(circuit_status, Mapping):
        circuit_state = normalize_whitespace(circuit_status.get("state")) or None
        if circuit_state == "open":
            status = AgentHealthStatus.UNAVAILABLE.value
    active_tasks = coerce_int(stats_row.get("active_tasks"), default=0, minimum=0)
    if status == AgentHealthStatus.ACTIVE.value and active_tasks > 0:
        status = AgentHealthStatus.BUSY.value
    return AgentSnapshot(
        name=name,
        capabilities=normalize_capabilities(meta_dict.get("capabilities") or heartbeat_row.get("capabilities")),
        status=status or AgentHealthStatus.UNKNOWN.value,
        version=meta_dict.get("version") or heartbeat_row.get("version"),
        successes=coerce_int(stats_row.get("successes"), default=0, minimum=0),
        failures=coerce_int(stats_row.get("failures"), default=0, minimum=0),
        active_tasks=active_tasks,
        last_seen=last_seen or None,
        heartbeat_age_seconds=heartbeat_age,
        success_rate=calculate_success_rate(stats_row),
        failure_rate=float(stats_row.get("failure_rate", 0.0)),
        score=calculate_agent_score(stats_row),
        circuit_state=circuit_state,
        metadata=normalize_metadata(merge_mappings(meta_dict.get("metadata"), heartbeat_row.get("metadata")), drop_none=True),
    )


def build_agent_snapshots(
    agents: Optional[Mapping[str, Any]],
    *,
    stats: Optional[Mapping[str, Any]] = None,
    heartbeats: Optional[Mapping[str, Any]] = None,
    reliability_status: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build snapshots for all known agents."""

    agents_map = ensure_mapping(agents, field_name="agents", allow_none=True)
    stats_map = normalize_agent_stats_map(stats)
    heartbeat_map = ensure_mapping(heartbeats, field_name="heartbeats", allow_none=True)
    circuit_map = ensure_mapping(reliability_status, field_name="reliability_status", allow_none=True)
    names = set(agents_map.keys()) | set(stats_map.keys())
    for key in heartbeat_map:
        if str(key).startswith(f"{DEFAULT_AGENT_KEY_PREFIX}:"):
            names.add(str(key).split(":", 1)[1])
        else:
            names.add(str(key))
    snapshots: Dict[str, Dict[str, Any]] = {}
    for name in sorted(str(item) for item in names):
        meta = agents_map.get(name)
        if isinstance(meta, list):
            meta = {"capabilities": meta}
        heartbeat = heartbeat_map.get(agent_memory_key(name)) or heartbeat_map.get(name)
        snapshots[name] = build_agent_snapshot(
            name,
            meta=meta if isinstance(meta, Mapping) else {},
            stats=stats_map.get(name),
            heartbeat=heartbeat if isinstance(heartbeat, Mapping) else {},
            circuit_status=circuit_map.get(name) if isinstance(circuit_map.get(name), Mapping) else {},
        ).to_dict()
    return snapshots


# ---------------------------------------------------------------------------
# Load and routing telemetry helpers
# ---------------------------------------------------------------------------
def get_system_load_from_stats(stats: Optional[Mapping[str, Any]]) -> int:
    """Return total active tasks from an agent_stats mapping."""

    stats_map = normalize_agent_stats_map(stats)
    return int(sum(coerce_int(row.get("active_tasks"), default=0, minimum=0) for row in stats_map.values()))


def count_active_agents(agents: Optional[Mapping[str, Any]], *, heartbeats: Optional[Mapping[str, Any]] = None) -> int:
    """Count known active agents from registry and heartbeat maps."""

    agents_map = ensure_mapping(agents, field_name="agents", allow_none=True)
    heartbeat_map = ensure_mapping(heartbeats, field_name="heartbeats", allow_none=True)
    if not heartbeat_map:
        return len(agents_map)
    count = 0
    for name in agents_map.keys():
        record = heartbeat_map.get(agent_memory_key(name)) or heartbeat_map.get(name) or {}
        if not isinstance(record, Mapping):
            continue
        status = normalize_whitespace(record.get("status") or AgentHealthStatus.UNKNOWN.value)
        if status in {AgentHealthStatus.ACTIVE.value, AgentHealthStatus.BUSY.value, AgentHealthStatus.IDLE.value}:
            count += 1
    return count


def calculate_max_load(
    agent_count: int,
    *,
    max_concurrent_tasks: Optional[Any] = None,
    load_factor: Optional[Any] = None,
    tasks_per_agent: Optional[Any] = None,
) -> int:
    """Calculate max load consistently with CollaborationManager.max_load."""

    agent_total = max(1, coerce_int(agent_count, default=1, minimum=1))
    max_tasks = coerce_int(
        max_concurrent_tasks if max_concurrent_tasks is not None else COLLABORATION_CONFIG.get("max_concurrent_tasks", 100),
        default=100,
        minimum=1,
    )
    factor = coerce_float(load_factor if load_factor is not None else COLLABORATION_CONFIG.get("load_factor", 0.75), default=0.75, minimum=0.0)
    per_agent = coerce_int(tasks_per_agent if tasks_per_agent is not None else AGENTS_CONFIG.get("max_tasks_per_agent", DEFAULT_AGENT_TASK_MULTIPLIER), default=DEFAULT_AGENT_TASK_MULTIPLIER, minimum=1)
    return max(1, min(max_tasks, int(agent_total * per_agent * factor)))


def build_load_snapshot(
    *,
    agents: Optional[Mapping[str, Any]] = None,
    stats: Optional[Mapping[str, Any]] = None,
    heartbeats: Optional[Mapping[str, Any]] = None,
    max_concurrent_tasks: Optional[Any] = None,
    load_factor: Optional[Any] = None,
) -> LoadSnapshot:
    """Build a point-in-time load snapshot."""

    agents_map = ensure_mapping(agents, field_name="agents", allow_none=True)
    stats_map = normalize_agent_stats_map(stats)
    agent_count = max(1, len(agents_map) or len(stats_map) or 1)
    active_count = count_active_agents(agents_map or {name: {} for name in stats_map.keys()}, heartbeats=heartbeats)
    current_load = get_system_load_from_stats(stats_map)
    max_load = calculate_max_load(agent_count, max_concurrent_tasks=max_concurrent_tasks, load_factor=load_factor)
    utilization = round(current_load / max_load, 6) if max_load else 1.0
    return LoadSnapshot(
        current_load=current_load,
        max_load=max_load,
        agent_count=agent_count,
        active_agent_count=active_count,
        overloaded=current_load >= max_load,
        utilization=utilization,
    )


def is_overloaded(*, stats: Optional[Mapping[str, Any]] = None, agents: Optional[Mapping[str, Any]] = None, max_load: Optional[int] = None) -> bool:
    """Return True if current load meets or exceeds max load."""

    current = get_system_load_from_stats(stats)
    resolved_max = max_load if max_load is not None else calculate_max_load(len(ensure_mapping(agents, field_name="agents", allow_none=True)) or 1)
    return current >= resolved_max


def normalize_routing_attempt(attempt: Any, *, task_type: Optional[str] = None) -> Dict[str, Any]:
    """Normalize a routing attempt object/mapping/dataclass into a dictionary."""

    if isinstance(attempt, RoutingAttempt):
        return attempt.to_dict()
    if is_dataclass(attempt) and not isinstance(attempt, type):
        data = asdict(attempt)
    elif isinstance(attempt, Mapping):
        data = dict(attempt)
    else:
        data = {"value": json_safe(attempt)}
    if task_type is not None:
        data.setdefault("task_type", normalize_task_type(task_type))
    return redact_mapping(prune_none(data))


def summarize_routing_attempts(attempts: Optional[Iterable[Any]]) -> Dict[str, Any]:
    """Summarize routing attempts for audit events and failure context."""

    normalized = [normalize_routing_attempt(item) for item in (attempts or [])]
    successes = sum(1 for item in normalized if item.get("status") == DEFAULT_RESULT_STATUS_SUCCESS)
    failures = sum(1 for item in normalized if item.get("status") == DEFAULT_RESULT_STATUS_ERROR)
    skipped = sum(1 for item in normalized if item.get("status") == DEFAULT_RESULT_STATUS_SKIPPED)
    return {
        "attempt_count": len(normalized),
        "success_count": successes,
        "failure_count": failures,
        "skipped_count": skipped,
        "attempted_agents": [item.get("agent_name") for item in normalized if item.get("agent_name")],
        "attempts": normalized,
    }


def build_routing_context(
    *,
    task: Optional[Union[TaskEnvelope, Mapping[str, Any]]] = None,
    ranked_agents: Optional[Iterable[Any]] = None,
    attempts: Optional[Iterable[Any]] = None,
    stats: Optional[Mapping[str, Any]] = None,
    policy_decision: Optional[Any] = None,
    contract_result: Optional[Any] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a routing context payload for errors, audit, and debugging."""

    task_payload: Dict[str, Any] = {}
    if task is not None:
        try:
            env = task if isinstance(task, TaskEnvelope) else build_task_envelope(task)
            task_payload = env.to_dict()
        except Exception:
            safe_task = json_safe(task)
            if isinstance(safe_task, dict):
                task_payload = safe_task
            else:
                task_payload = {"task": safe_task}

    ranked_payload = []
    for item in ranked_agents or []:
        if isinstance(item, tuple) and len(item) >= 3:
            ranked_payload.append({"agent_name": item[0], "score": item[2], "meta": json_safe(item[1])})
        else:
            ranked_payload.append(json_safe(item))

    return redact_mapping(
        merge_mappings(
            {
                "task": task_payload,
                "ranked_agents": ranked_payload,
                "routing_attempts": summarize_routing_attempts(attempts),
                "agent_stats": normalize_agent_stats_map(stats),
                "policy_decision": policy_evaluation_to_dict(policy_decision),
                "contract_validation": contract_validation_to_dict(contract_result),
            },
            extra,
            deep=True,
            drop_none=True,
        )
    )


def build_delegation_record(
    *,
    agent_name: Any,
    task_type: Any,
    status: str,
    score: Optional[float] = None,
    started_at: Optional[float] = None,
    finished_at: Optional[float] = None,
    error: Optional[Union[BaseException, Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> RoutingAttempt:
    """Build a normalized delegation attempt record."""

    start = started_at if started_at is not None else epoch_seconds()
    finish = finished_at
    duration = round(max(0.0, (finish - start) * 1000.0), 3) if finish is not None else None
    error_payload = None
    if isinstance(error, BaseException):
        error_payload = exception_to_error_payload(error).get("error")
    elif isinstance(error, Mapping):
        error_payload = dict(error)
    return RoutingAttempt(
        agent_name=normalize_agent_name(agent_name),
        task_type=normalize_task_type(task_type),
        status=normalize_whitespace(status) or DEFAULT_RESULT_STATUS_ERROR,
        score=score,
        started_at=start,
        finished_at=finish,
        duration_ms=duration,
        error=error_payload,
        metadata=normalize_metadata(metadata, drop_none=True),
    )


# ---------------------------------------------------------------------------
# Shared-memory safety wrappers
# ---------------------------------------------------------------------------
def memory_get(memory: MemoryLike, key: str, default: Any = None) -> Any:
    """Safely read a key from a SharedMemory-like object."""

    if memory is None:
        return default
    try:
        try:
            return memory.get(key, default)
        except TypeError:
            result = memory.get(key)
            return default if result is None else result
    except Exception as exc:
        logger.warning("Shared memory get failed for key %s: %s", key, exc)
        return default


def memory_set(memory: MemoryLike, key: str, value: Any, *, ttl: Optional[Any] = None, **kwargs: Any) -> bool:
    """Safely set a key on a SharedMemory-like object."""

    if memory is None:
        return False
    try:
        if ttl is None and not kwargs:
            memory.set(key, value)
        else:
            memory.set(key, value, ttl=ttl, **kwargs)
        return True
    except TypeError:
        try:
            memory.set(key, value)
            return True
        except Exception as exc:
            logger.warning("Shared memory set failed for key %s: %s", key, exc)
            return False
    except Exception as exc:
        logger.warning("Shared memory set failed for key %s: %s", key, exc)
        return False


def memory_delete(memory: MemoryLike, key: str) -> bool:
    """Safely delete a key from a SharedMemory-like object."""

    if memory is None:
        return False
    try:
        result = memory.delete(key)
        return bool(result) if result is not None else True
    except Exception as exc:
        logger.warning("Shared memory delete failed for key %s: %s", key, exc)
        return False


def memory_append(memory: MemoryLike, key: str, value: Any, *, ttl: Optional[Any] = None, priority: Optional[int] = None) -> bool:
    """Safely append a value using SharedMemory.append when available."""

    if memory is None:
        return False
    try:
        if hasattr(memory, "append"):
            memory.append(key, value, ttl=ttl, priority=priority)
            return True
        current = memory_get(memory, key, default=[])
        if not isinstance(current, list):
            current = [current]
        current.append(value)
        return memory_set(memory, key, current, ttl=ttl)
    except Exception as exc:
        logger.warning("Shared memory append failed for key %s: %s", key, exc)
        return False


def memory_increment(memory: MemoryLike, key: str, delta: int = 1, *, default: int = 0) -> int:
    """Safely increment a numeric memory value and return the new value."""

    if memory is None:
        return default
    try:
        if hasattr(memory, "increment"):
            return int(memory.increment(key, delta))
    except Exception:
        pass
    current = coerce_int(memory_get(memory, key, default=default), default=default)
    updated = current + int(delta)
    memory_set(memory, key, updated)
    return updated


def memory_compare_and_swap(memory: MemoryLike, key: str, expected_value: Any, new_value: Any) -> bool:
    """Safely run compare-and-swap when available, otherwise emulate best-effort."""

    if memory is None:
        return False
    try:
        if hasattr(memory, "compare_and_swap"):
            return bool(memory.compare_and_swap(key, expected_value, new_value))
    except Exception as exc:
        logger.warning("Shared memory CAS failed for key %s: %s", key, exc)
        return False
    current = memory_get(memory, key, default=None)
    if current == expected_value:
        return memory_set(memory, key, new_value)
    return False


def memory_publish(memory: MemoryLike, channel: str, message: Any) -> bool:
    """Safely publish a message on a SharedMemory-like pub/sub channel."""

    if memory is None or not hasattr(memory, "publish"):
        return False
    try:
        memory.publish(normalize_channel_name(channel), json_safe(message))
        return True
    except Exception as exc:
        logger.warning("Shared memory publish failed for channel %s: %s", channel, exc)
        return False


def memory_snapshot(memory: MemoryLike, *, include_metrics: bool = True, include_keys: bool = False) -> Dict[str, Any]:
    """Return a safe snapshot of shared-memory diagnostics."""

    if memory is None:
        return {"available": False}
    snapshot: Dict[str, Any] = {"available": True, "type": type(memory).__name__}
    try:
        snapshot["length"] = len(memory)
    except Exception:
        pass
    if include_metrics and hasattr(memory, "metrics"):
        try:
            snapshot["metrics"] = sanitize_for_logging(memory.metrics())
        except Exception as exc:
            snapshot["metrics_error"] = str(exc)
    if include_metrics and hasattr(memory, "get_usage_stats"):
        try:
            snapshot["usage_stats"] = sanitize_for_logging(memory.get_usage_stats())
        except Exception as exc:
            snapshot["usage_stats_error"] = str(exc)
    if include_keys and hasattr(memory, "get_all_keys"):
        try:
            snapshot["keys"] = list(memory.get_all_keys())
        except Exception as exc:
            snapshot["keys_error"] = str(exc)
    return snapshot


def get_agent_stats(memory: MemoryLike, *, stats_key: str = DEFAULT_AGENT_STATS_KEY) -> Dict[str, Dict[str, Any]]:
    """Read and normalize agent stats from shared memory."""

    return normalize_agent_stats_map(memory_get(memory, stats_key, default={}))


def set_agent_stats(memory: MemoryLike, stats: Mapping[str, Any], *, stats_key: str = DEFAULT_AGENT_STATS_KEY) -> bool:
    """Write normalized agent stats to shared memory."""

    return memory_set(memory, stats_key, normalize_agent_stats_map(stats))


def update_agent_stats(
    stats: Optional[MutableMapping[str, Any]],
    agent_name: Any,
    *,
    successes_delta: int = 0,
    failures_delta: int = 0,
    active_tasks_delta: int = 0,
    timestamp: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Update an in-memory stats mapping and return a normalized copy."""

    stats_map: Dict[str, Any] = dict(stats or {})
    name = normalize_agent_name(agent_name)
    row = normalize_agent_stats(stats_map.get(name))
    row["successes"] = max(0, int(row.get("successes", 0)) + int(successes_delta))
    row["failures"] = max(0, int(row.get("failures", 0)) + int(failures_delta))
    row["active_tasks"] = max(0, int(row.get("active_tasks", 0)) + int(active_tasks_delta))
    row["last_seen"] = float(timestamp if timestamp is not None else epoch_seconds())
    stats_map[name] = normalize_agent_stats(row)
    return normalize_agent_stats_map(stats_map)


def record_agent_success(memory: MemoryLike, agent_name: Any, *, stats_key: str = DEFAULT_AGENT_STATS_KEY) -> Dict[str, Dict[str, Any]]:
    """Increment an agent success count in shared memory."""

    stats = update_agent_stats(get_agent_stats(memory, stats_key=stats_key), agent_name, successes_delta=1)
    set_agent_stats(memory, stats, stats_key=stats_key)
    touch_agent_heartbeat(memory, agent_name)
    return stats


def record_agent_failure(memory: MemoryLike, agent_name: Any, *, stats_key: str = DEFAULT_AGENT_STATS_KEY) -> Dict[str, Dict[str, Any]]:
    """Increment an agent failure count in shared memory."""

    stats = update_agent_stats(get_agent_stats(memory, stats_key=stats_key), agent_name, failures_delta=1)
    set_agent_stats(memory, stats, stats_key=stats_key)
    touch_agent_heartbeat(memory, agent_name)
    return stats


def set_agent_active_delta(memory: MemoryLike, agent_name: Any, delta: int, *, stats_key: str = DEFAULT_AGENT_STATS_KEY) -> Dict[str, Dict[str, Any]]:
    """Adjust active task count for an agent in shared memory."""

    stats = update_agent_stats(get_agent_stats(memory, stats_key=stats_key), agent_name, active_tasks_delta=int(delta))
    set_agent_stats(memory, stats, stats_key=stats_key)
    touch_agent_heartbeat(memory, agent_name)
    return stats


def touch_agent_heartbeat(
    memory: MemoryLike,
    agent_name: Any,
    *,
    status: str = AgentHealthStatus.ACTIVE.value,
    capabilities: Optional[Iterable[Any]] = None,
    version: Optional[Union[int, float, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Create/update an agent heartbeat record in shared memory."""

    name = normalize_agent_name(agent_name)
    key = agent_memory_key(name)
    current = memory_get(memory, key, default={})
    if not isinstance(current, Mapping):
        current = {}
    heartbeat = merge_mappings(
        current,
        {
            "status": normalize_whitespace(status) or AgentHealthStatus.ACTIVE.value,
            "last_seen": float(timestamp if timestamp is not None else epoch_seconds()),
            "capabilities": list(normalize_capabilities(capabilities or current.get("capabilities"))),
            "version": version if version is not None else current.get("version"),
            "metadata": normalize_metadata(merge_mappings(current.get("metadata"), metadata), drop_none=True),
        },
        deep=True,
        drop_none=True,
    )
    memory_set(memory, key, heartbeat)
    return heartbeat


def read_agent_heartbeats(memory: MemoryLike, agent_names: Optional[Iterable[Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Read heartbeat records for known agent names."""

    heartbeats: Dict[str, Dict[str, Any]] = {}
    if agent_names is not None:
        for name in agent_names:
            normalized_name = normalize_agent_name(name)
            record = memory_get(memory, agent_memory_key(normalized_name), default={})
            if isinstance(record, Mapping):
                heartbeats[normalized_name] = dict(record)
        return heartbeats

    snapshot = memory_snapshot(memory, include_metrics=False, include_keys=True)
    for key in snapshot.get("keys", []) or []:
        key_text = str(key)
        if key_text.startswith(f"{DEFAULT_AGENT_KEY_PREFIX}:"):
            name = key_text.split(":", 1)[1]
            record = memory_get(memory, key_text, default={})
            if isinstance(record, Mapping):
                heartbeats[name] = dict(record)
    return heartbeats


# ---------------------------------------------------------------------------
# Error construction and result helpers
# ---------------------------------------------------------------------------
def make_collaboration_exception(
    class_name: str,
    message: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
    cause: Optional[BaseException] = None,
    **kwargs: Any,
) -> Exception:
    """Construct a collaboration exception by class name when available.

    This keeps helpers compatible with both the minimal existing error module
    and a richer production taxonomy. If the requested class is unavailable or
    has an incompatible constructor, the function falls back to RuntimeError.
    """

    cls = getattr(_collaboration_error_module, class_name, None) if _collaboration_error_module is not None else globals().get(class_name)
    if isinstance(cls, type) and issubclass(cls, Exception):
        attempts = (
            lambda: cls(message=message, context=context, cause=cause, **kwargs),  # type: ignore
            lambda: cls(message, context=context, cause=cause, **kwargs),  # type: ignore
            lambda: cls(message, context=context, **kwargs),  # type: ignore
            lambda: cls(message),
        )
        for build in attempts:
            try:
                return build()
            except TypeError:
                continue
    base = CollaborationError
    if isinstance(base, type) and issubclass(base, Exception):
        error_type = getattr(CollaborationErrorType, "ROUTING_FAILURE", None) if CollaborationErrorType is not None else "Routing Failure"
        attempts = (
            lambda: base(error_type=error_type, message=message, context=context, cause=cause, **kwargs),  # type: ignore
            lambda: base(error_type, message, context=context, **kwargs),  # type: ignore
            lambda: base(message),
        )
        for build in attempts:
            try:
                return build()
            except TypeError:
                continue
    return RuntimeError(message)


def exception_to_error_payload(
    exc: BaseException,
    *,
    action: Optional[str] = None,
    include_traceback: bool = False,
    redact: bool = True,
) -> Dict[str, Any]:
    """Normalize any exception into a collaboration result error shape."""

    if hasattr(exc, "to_audit_format"):
        try:
            audit = exc.to_audit_format(include_traceback=include_traceback)  # type: ignore[attr-defined]
            if isinstance(audit, Mapping):
                payload = {
                    "status": DEFAULT_RESULT_STATUS_ERROR,
                    "action": action,
                    "message": str(exc),
                    "error": dict(audit),
                }
                return redact_mapping(prune_none(payload)) if redact else prune_none(payload)
        except Exception:
            pass
    payload: Dict[str, Any] = {
        "status": DEFAULT_RESULT_STATUS_ERROR,
        "action": action,
        "message": str(exc),
        "error": {
            "type": type(exc).__name__,
            "module": type(exc).__module__,
            "message": str(exc),
            "retryable": getattr(exc, "retryable", None),
            "severity": getattr(exc, "severity", None),
            "context": getattr(exc, "context", None),
        },
    }
    if include_traceback:
        payload["traceback"] = traceback.format_exc()
    return redact_mapping(prune_none(payload)) if redact else prune_none(payload)


def success_result(
    *,
    action: Optional[str] = None,
    message: str = "Success",
    data: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    duration_ms: Optional[float] = None,
    correlation_id: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a stable success result payload."""

    payload = OperationResult(
        status=CollaborationStatus.SUCCESS.value,
        action=action,
        message=message,
        data=normalize_metadata(data, drop_none=True),
        metadata=normalize_metadata(metadata, drop_none=True),
        duration_ms=duration_ms,
        correlation_id=correlation_id or generate_correlation_id(action or "collab"),
    ).to_dict()
    safe_extra = json_safe(extra)
    if isinstance(safe_extra, dict):
        payload.update(safe_extra)
    return redact_mapping(prune_none(payload))


def error_result(
    *,
    action: Optional[str] = None,
    message: str = "Collaborative operation failed",
    error: Optional[Union[BaseException, Mapping[str, Any]]] = None,
    data: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    duration_ms: Optional[float] = None,
    correlation_id: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a stable error result payload."""

    if isinstance(error, BaseException):
        error_payload = exception_to_error_payload(error, action=action).get("error", {"message": str(error)})
        message = message or str(error)
    elif isinstance(error, Mapping):
        error_payload = dict(error)
    else:
        error_payload = None
    payload = OperationResult(
        status=CollaborationStatus.ERROR.value,
        action=action,
        message=message,
        data=normalize_metadata(data, drop_none=True),
        error=error_payload,
        metadata=normalize_metadata(metadata, drop_none=True),
        duration_ms=duration_ms,
        correlation_id=correlation_id or generate_correlation_id(action or "collab_err"),
    ).to_dict()
    safe_extra = json_safe(extra)
    if isinstance(safe_extra, dict):
        payload.update(safe_extra)
    return redact_mapping(prune_none(payload))


def review_result(
    *,
    action: Optional[str] = None,
    message: str = "Review required",
    reasons: Optional[Iterable[Any]] = None,
    data: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a stable require-review result payload."""

    return OperationResult(
        status=CollaborationStatus.REQUIRE_REVIEW.value,
        action=action,
        message=message,
        data=normalize_metadata(merge_mappings(data, {"reasons": list(reasons or [])}), drop_none=True),
        metadata=normalize_metadata(metadata, drop_none=True),
        correlation_id=correlation_id or generate_correlation_id(action or "review"),
    ).to_dict()


def normalize_result(result: Any, *, action: Optional[str] = None, default_message: str = "Completed") -> Dict[str, Any]:
    """Normalize arbitrary module output into a collaboration result dictionary."""

    if isinstance(result, OperationResult):
        payload = result.to_dict()
        if action is not None:
            payload.setdefault("action", action)
        return payload
    if isinstance(result, Mapping):
        payload = dict(result)
        payload.setdefault("status", DEFAULT_RESULT_STATUS_SUCCESS if not payload.get("error") else DEFAULT_RESULT_STATUS_ERROR)
        if action is not None:
            payload.setdefault("action", action)
        payload.setdefault("message", default_message if payload["status"] == DEFAULT_RESULT_STATUS_SUCCESS else "Collaborative operation failed")
        return redact_mapping(prune_none(payload))
    if isinstance(result, BaseException):
        return exception_to_error_payload(result, action=action)
    return success_result(action=action, message=default_message, data={"value": json_safe(result)})


# ---------------------------------------------------------------------------
# Contract and policy normalization helpers
# ---------------------------------------------------------------------------
def contract_validation_to_dict(result: Any) -> Optional[Dict[str, Any]]:
    """Normalize task contract validation output into a dictionary."""

    if result is None:
        return None
    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            output = result.to_dict()
            return dict(output) if isinstance(output, Mapping) else {"value": json_safe(output)}
        except Exception:
            pass
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    if isinstance(result, Mapping):
        return dict(result)
    return {"value": json_safe(result), "valid": bool(getattr(result, "valid", False))}


def policy_evaluation_to_dict(evaluation: Any) -> Optional[Dict[str, Any]]:
    """Normalize policy evaluation output into a dictionary."""

    if evaluation is None:
        return None
    if hasattr(evaluation, "to_dict") and callable(evaluation.to_dict):
        try:
            output = evaluation.to_dict()
            return dict(output) if isinstance(output, Mapping) else {"value": json_safe(output)}
        except Exception:
            pass
    if is_dataclass(evaluation) and not isinstance(evaluation, type):
        return asdict(evaluation)
    if isinstance(evaluation, Mapping):
        return dict(evaluation)
    decision = getattr(evaluation, "decision", None)
    if isinstance(decision, Enum):
        decision = decision.value
    return {
        "decision": decision or json_safe(evaluation),
        "reasons": list(getattr(evaluation, "reasons", []) or []),
        "matched_rules": list(getattr(evaluation, "matched_rules", []) or []),
    }


def policy_allows(evaluation: Any) -> bool:
    """Return True when a policy evaluation permits execution."""

    data = policy_evaluation_to_dict(evaluation) or {}
    decision = str(data.get("decision", "allow")).lower()
    return decision == "allow"


def policy_requires_review(evaluation: Any) -> bool:
    """Return True when a policy evaluation requires manual review."""

    data = policy_evaluation_to_dict(evaluation) or {}
    decision = str(data.get("decision", "")).lower()
    return decision in {"require_review", "review", "manual_review"}


def contract_is_valid(result: Any) -> bool:
    """Return True when a contract validation result is valid or absent."""

    if result is None:
        return True
    data = contract_validation_to_dict(result) or {}
    return bool(data.get("valid", False))


# ---------------------------------------------------------------------------
# Audit event and health report helpers
# ---------------------------------------------------------------------------
def build_audit_event(event_type: str, message: str, *,
    severity: str = "info",
    component: str = "collaboration",
    correlation_id: Optional[str] = None,
    task: Optional[Union[TaskEnvelope, Mapping[str, Any]]] = None,
    agent_name: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    state: Optional[Mapping[str, Any]] = None,
    error: Optional[Union[BaseException, Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a redacted, JSON-safe audit event for the collaborative runtime."""

    task_payload = None
    if task is not None:
        try:
            task_payload = task.to_dict() if isinstance(task, TaskEnvelope) else build_task_envelope(task).to_dict()
        except Exception:
            task_payload = json_safe(task)
    error_payload = None
    if isinstance(error, BaseException):
        error_payload = exception_to_error_payload(error).get("error")
    elif isinstance(error, Mapping):
        error_payload = dict(error)
    event = {
        "event_id": generate_uuid("audit", length=24),
        "event_type": normalize_identifier_component(event_type, default="event", lowercase=True),
        "message": truncate_text(message, DEFAULT_MAX_STRING_LENGTH),
        "severity": normalize_whitespace(severity).lower() or "info",
        "component": component or "collaboration",
        "correlation_id": correlation_id or generate_correlation_id("audit"),
        "task": task_payload,
        "agent_name": agent_name,
        "context": normalize_metadata(context, drop_none=True),
        "state": json_safe(state or {}),
        "error": error_payload,
        "metadata": normalize_metadata(metadata, drop_none=True),
        "timestamp": epoch_seconds(),
        "timestamp_utc": utc_timestamp(),
    }
    return redact_mapping(prune_none(event, drop_empty=True))


def append_audit_event(memory: MemoryLike, event: Mapping[str, Any], *, key: str = DEFAULT_AUDIT_KEY,
                       max_events: int = DEFAULT_MAX_AUDIT_EVENTS, ttl: Optional[Any] = None) -> bool:
    """Append an audit event list in shared memory with bounded retention."""

    current = memory_get(memory, key, default=[])
    if not isinstance(current, list):
        current = [current]
    current.append(redact_mapping(event))
    max_count = coerce_int(max_events, default=DEFAULT_MAX_AUDIT_EVENTS, minimum=1)
    bounded = current[-max_count:]
    return memory_set(memory, key, bounded, ttl=ttl)


def record_audit_event(memory: MemoryLike, event_type: str, message: str, **kwargs: Any) -> Dict[str, Any]:
    """Build and append an audit event, returning the event payload."""

    event = build_audit_event(event_type, message, **kwargs)
    append_audit_event(memory, event)
    return event


def build_health_report(*, agents: Optional[Mapping[str, Any]] = None, stats: Optional[Mapping[str, Any]] = None,
                        heartbeats: Optional[Mapping[str, Any]] = None, reliability_status: Optional[Mapping[str, Any]] = None,
                        shared_memory: MemoryLike = None) -> Dict[str, Any]:
    """Build a subsystem health report from available runtime components."""

    snapshots = build_agent_snapshots(agents, stats=stats, heartbeats=heartbeats, reliability_status=reliability_status)
    load = build_load_snapshot(agents=agents, stats=stats, heartbeats=heartbeats)
    status = "healthy"
    if load.overloaded:
        status = "overloaded"
    elif any(item.get("status") in {AgentHealthStatus.UNAVAILABLE.value, AgentHealthStatus.STALE.value} for item in snapshots.values()):
        status = "degraded"
    return redact_mapping(
        {
            "status": status,
            "captured_at": epoch_seconds(),
            "captured_at_utc": utc_timestamp(),
            "load": load.to_dict(),
            "agents": snapshots,
            "shared_memory": memory_snapshot(shared_memory, include_metrics=True, include_keys=False),
            "reliability_status": json_safe(reliability_status or {}),
        }
    )


def export_json_file(path: Union[str, Path], payload: Any, *, pretty: bool = True) -> Path:
    """Write JSON-safe payload to a file and return the resolved path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(stable_json_dumps(payload, indent=2 if pretty else None), encoding="utf-8")
    return destination.resolve()


# ---------------------------------------------------------------------------
# Retry/backoff helpers
# ---------------------------------------------------------------------------
def calculate_backoff_delay(attempt: int, *,
    base_delay_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    max_delay_seconds: float = DEFAULT_BACKOFF_MAX_SECONDS,
    jitter_ratio: float = DEFAULT_BACKOFF_JITTER_RATIO,
) -> float:
    """Calculate exponential backoff delay with optional jitter.

    `attempt` is 1-based: attempt=1 returns approximately base_delay_seconds.
    """

    attempt_index = coerce_int(attempt, default=1, minimum=1)
    base = coerce_float(base_delay_seconds, default=DEFAULT_BACKOFF_BASE_SECONDS, minimum=0.0)
    factor = coerce_float(multiplier, default=DEFAULT_BACKOFF_MULTIPLIER, minimum=0.0)
    max_delay = coerce_float(max_delay_seconds, default=DEFAULT_BACKOFF_MAX_SECONDS, minimum=0.0)
    delay = min(max_delay, base * (factor ** (attempt_index - 1)))
    jitter = coerce_float(jitter_ratio, default=0.0, minimum=0.0, maximum=1.0)
    if jitter <= 0 or delay <= 0:
        return round(delay, 6)
    spread = delay * jitter
    return round(max(0.0, random.uniform(delay - spread, delay + spread)), 6)


def build_backoff_policy_from_config(config: Optional[Mapping[str, Any]] = None) -> BackoffPolicy:
    """Build BackoffPolicy from task_routing/reliability-style config."""

    source = merge_mappings(TASK_ROUTING_CONFIG.get("retry_policy") if isinstance(TASK_ROUTING_CONFIG.get("retry_policy"), Mapping) else {}, config)
    return BackoffPolicy(
        max_attempts=coerce_int(source.get("max_attempts"), default=DEFAULT_MAX_TASK_RETRIES, minimum=1),
        base_delay_seconds=coerce_float(source.get("base_delay_seconds", source.get("backoff_factor", DEFAULT_BACKOFF_BASE_SECONDS)), default=DEFAULT_BACKOFF_BASE_SECONDS, minimum=0.0),
        multiplier=coerce_float(source.get("multiplier", DEFAULT_BACKOFF_MULTIPLIER), default=DEFAULT_BACKOFF_MULTIPLIER, minimum=0.0),
        max_delay_seconds=coerce_float(source.get("max_delay_seconds", source.get("max_backoff_seconds", DEFAULT_BACKOFF_MAX_SECONDS)), default=DEFAULT_BACKOFF_MAX_SECONDS, minimum=0.0),
        jitter_ratio=coerce_float(source.get("jitter_ratio", source.get("jitter_seconds", DEFAULT_BACKOFF_JITTER_RATIO)), default=DEFAULT_BACKOFF_JITTER_RATIO, minimum=0.0, maximum=1.0),
    )


def is_retryable_exception(exc: BaseException, *, default: bool = True) -> bool:
    """Return whether an exception should be treated as retryable."""

    if hasattr(exc, "retryable"):
        return bool(getattr(exc, "retryable"))
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return False
    return default


def retry_call(operation: Callable[[], T], *, policy: Optional[BackoffPolicy] = None,
               retryable: Optional[ExceptionPredicate] = None,
               on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
               sleep: bool = True) -> T:
    """Execute an operation with bounded retry/backoff behavior."""

    resolved_policy = policy or build_backoff_policy_from_config()
    predicate = retryable or is_retryable_exception
    attempts = max(1, int(resolved_policy.max_attempts))
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except BaseException as exc:  # noqa: BLE001 - helper boundary normalization
            last_error = exc
            if attempt >= attempts or not predicate(exc):
                raise
            delay = resolved_policy.delay_for_attempt(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            if sleep and delay > 0:
                time_module.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry_call operation did not execute")


# ---------------------------------------------------------------------------
# Convenience adapters for manager/router/registry style objects
# ---------------------------------------------------------------------------
def snapshot_manager(manager: Any) -> Dict[str, Any]:
    """Build a safe snapshot of a CollaborationManager-like object."""

    if manager is None:
        return {"available": False}
    try:
        agents = manager.list_agents() if hasattr(manager, "list_agents") else {}
    except Exception as exc:
        agents = {"error": str(exc)}
    try:
        stats = manager.get_agent_stats() if hasattr(manager, "get_agent_stats") else {}
    except Exception as exc:
        stats = {"error": str(exc)}
    try:
        reliability = manager.get_reliability_status() if hasattr(manager, "get_reliability_status") else {}
    except Exception as exc:
        reliability = {"error": str(exc)}
    shared_memory = getattr(manager, "shared_memory", None)
    heartbeats = read_agent_heartbeats(shared_memory, agents.keys() if isinstance(agents, Mapping) else None)
    return build_health_report(agents=agents if isinstance(agents, Mapping) else {}, stats=stats if isinstance(stats, Mapping) else {}, heartbeats=heartbeats, reliability_status=reliability if isinstance(reliability, Mapping) else {}, shared_memory=shared_memory)


def snapshot_registry(registry: Any, *, shared_memory: MemoryLike = None) -> Dict[str, Any]:
    """Build a safe snapshot of an AgentRegistry-like object."""

    if registry is None:
        return {"available": False}
    try:
        agents = registry.list_agents() if hasattr(registry, "list_agents") else {}
    except Exception as exc:
        agents = {"error": str(exc)}
    memory = shared_memory if shared_memory is not None else getattr(registry, "shared_memory", None)
    heartbeats = read_agent_heartbeats(memory, agents.keys() if isinstance(agents, Mapping) else None)
    return {
        "available": True,
        "agents": build_agent_snapshots(agents if isinstance(agents, Mapping) else {}, heartbeats=heartbeats),
        "shared_memory": memory_snapshot(memory, include_metrics=False, include_keys=False),
    }


def can_agent_handle_task(agent_meta: Mapping[str, Any], task_type: str, *, required_capabilities: Optional[Iterable[Any]] = None) -> bool:
    """Return True when agent metadata supports a task/capability set."""

    task = normalize_task_type(task_type)
    caps = set(normalize_capabilities(agent_meta.get("capabilities")))
    required = set(normalize_capabilities(required_capabilities))
    if task not in caps:
        return False
    return required.issubset(caps) if required else True


def filter_agents_for_task(agents: Mapping[str, Mapping[str, Any]], task_type: str, *, required_capabilities: Optional[Iterable[Any]] = None) -> Dict[str, Mapping[str, Any]]:
    """Filter registry-style agent metadata for task compatibility."""

    return {
        str(name): meta
        for name, meta in agents.items()
        if isinstance(meta, Mapping) and can_agent_handle_task(meta, task_type, required_capabilities=required_capabilities)
    }


if __name__ == "__main__":
    print("\n=== Running Collaborative Helpers Smoke Tests ===\n")

    envelope = build_task_envelope(
        task_type="translate",
        payload={"text": "hello", "token": "secret-value"},
        required_capabilities=["language"],
        timeout_seconds=30,
        source="smoke-test",
    )
    assert envelope.task_type == "translate"
    assert validate_task_envelope(envelope).valid
    assert envelope.deadline_at is not None

    class _Agent:
        capabilities = ["translate", "language"]

        def execute(self, data: Mapping[str, Any]) -> Dict[str, Any]:
            return {"ok": True, "data": dict(data)}

    registration = build_agent_registration("Translator", _Agent(), capabilities=["translate", "language"])
    assert registration["name"] == "Translator"
    assert registration["meta"]["capabilities"] == ["translate", "language"]

    class _Memory:
        def __init__(self):
            self.store: Dict[str, Any] = {}

        def get(self, key: str, default: Any = None) -> Any:
            return self.store.get(key, default)

        def set(self, key: str, value: Any, **kwargs: Any) -> None:
            self.store[key] = value

        def delete(self, key: str) -> bool:
            return self.store.pop(key, None) is not None

        def get_all_keys(self) -> List[str]:
            return list(self.store.keys())

    memory = _Memory()
    touch_agent_heartbeat(memory, "Translator", capabilities=["translate"], version=1.0)
    set_agent_active_delta(memory, "Translator", +1)
    record_agent_success(memory, "Translator")
    stats = get_agent_stats(memory)
    assert stats["Translator"]["successes"] == 1
    assert stats["Translator"]["active_tasks"] == 1

    report = build_health_report(
        agents={"Translator": registration["meta"]},
        stats=stats,
        heartbeats=read_agent_heartbeats(memory, ["Translator"]),
        shared_memory=memory,
    )
    assert report["status"] in {"healthy", "degraded", "overloaded"}

    audit = build_audit_event("task_submitted", "Task submitted", task=envelope, agent_name="Translator")
    assert audit["event_type"] == "task_submitted"
    assert audit["task"]["payload"]["token"] == "***REDACTED***"

    result = success_result(action="route", data={"agent": "Translator"})
    assert result["status"] == "success"

    print("All collaborative_helpers.py smoke tests passed.\n")
