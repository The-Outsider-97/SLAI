from __future__ import annotations

__version__ = "2.0.0"

import copy
import hashlib
import json
import math
import time as time_module
import uuid

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, MutableSequence, Optional, Sequence, Tuple, Union

from .handler_error import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Handler Helpers")
printer = PrettyPrinter()

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, Dict[str, Any], List[Any]]
PathLike = Union[str, Path]

DEFAULT_FAILURE_TYPE = "UnknownError"
DEFAULT_FAILURE_MESSAGE = "No error message provided"
DEFAULT_FAILURE_SCHEMA = "handler.failure.v2"
DEFAULT_TELEMETRY_SCHEMA = "handler.telemetry.v2"
DEFAULT_CHECKPOINT_SCHEMA = "handler.checkpoint.v2"

_SEVERITY_CRITICAL_TAGS: Tuple[str, ...] = (
    "critical",
    "fatal",
    "panic",
    "security",
    "breach",
    "unsafe",
    "data loss",
    "corruption",
    "permission denied",
    "unauthorized",
)
_SEVERITY_HIGH_TAGS: Tuple[str, ...] = (
    "oom",
    "outofmemory",
    "out of memory",
    "memoryerror",
    "dependency",
    "importerror",
    "modulenotfounderror",
    "runtimeerror",
    "unhandled",
)
_SEVERITY_MEDIUM_TAGS: Tuple[str, ...] = (
    "timeout",
    "timed out",
    "network",
    "connection",
    "socket",
    "http",
    "rate limit",
    "temporarily unavailable",
)
_RETRYABLE_TAGS: Tuple[str, ...] = (
    "timeout",
    "timed out",
    "network",
    "connection",
    "socket",
    "dns",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "rate limit",
    "resource busy",
    "temporary",
    "temporarily unavailable",
    "try again",
)
_NON_RETRYABLE_TAGS: Tuple[str, ...] = (
    "invalid",
    "validation",
    "permission denied",
    "unauthorized",
    "forbidden",
    "not found",
    "missing required",
    "malformed",
    "schema",
    "security",
    "policy",
)
_FAILURE_CATEGORY_TAGS: Mapping[str, Tuple[str, ...]] = {
    "security": ("security", "unauthorized", "forbidden", "permission denied", "token", "credential"),
    "timeout": ("timeout", "timed out"),
    "network": ("network", "connection", "socket", "dns", "http", "ssl", "tls"),
    "memory": ("memory", "oom", "outofmemory", "out of memory", "cuda"),
    "dependency": ("dependency", "import", "module", "dll", "package", "no module named"),
    "resource": ("resource", "busy", "quota", "rate limit", "cpu", "gpu", "disk"),
    "unicode": ("unicode", "codec", "encode", "decode"),
    "sla": ("sla", "deadline", "latency budget", "budget exhausted"),
    "validation": ("validation", "invalid", "schema", "missing required", "malformed"),
}


def utc_timestamp() -> float:
    """Return the current UNIX timestamp in UTC-compatible epoch seconds."""
    return time_module.time()


def utc_epoch_ms() -> int:
    """Return the current UNIX timestamp in milliseconds."""
    return int(utc_timestamp() * 1000)


def utc_iso_timestamp(*, milliseconds: bool = True) -> str:
    """Return a timezone-aware UTC ISO-8601 timestamp."""
    now = datetime.now(timezone.utc)
    if milliseconds:
        return now.isoformat(timespec="milliseconds")
    return now.isoformat()


def monotonic_timestamp() -> float:
    """Return a monotonic timestamp suitable for elapsed-time measurements."""
    return time_module.monotonic()


def elapsed_seconds(start: Optional[float]) -> float:
    """Return elapsed seconds from a monotonic start value."""
    if start is None:
        return 0.0
    try:
        return max(0.0, monotonic_timestamp() - float(start))
    except (TypeError, ValueError):
        return 0.0


def generate_correlation_id(prefix: str = "handler", *, separator: str = ":") -> str:
    """Create a compact correlation identifier for telemetry, checkpoints, and handoffs."""
    safe_prefix = normalize_identifier(prefix, default="handler")
    return f"{safe_prefix}{separator}{utc_epoch_ms()}{separator}{uuid.uuid4().hex[:12]}"


def generate_checkpoint_id(label: str = "checkpoint", *, prefix: str = "handler:checkpoint") -> str:
    """Create a stable checkpoint identifier using a label, epoch milliseconds, and random suffix."""
    safe_label = normalize_identifier(label, default="checkpoint")
    return f"{prefix}:{safe_label}:{utc_epoch_ms()}:{uuid.uuid4().hex[:8]}"


def normalize_identifier(value: Any, *, default: str = "unknown", max_chars: int = 96) -> str:
    """Normalize user/runtime labels into log-safe identifiers without changing meaning."""
    raw = str(value or default).strip().lower()
    if not raw:
        raw = default
    cleaned = []
    for char in raw:
        if char.isalnum() or char in {"_", "-", ".", ":"}:
            cleaned.append(char)
        elif char.isspace() or char in {"/", "\\"}:
            cleaned.append("_")
    normalized = "".join(cleaned).strip("_.:-") or default
    return normalized[: max(1, int(max_chars))]


def clamp(value: Union[int, float], minimum: Optional[Union[int, float]] = None, maximum: Optional[Union[int, float]] = None) -> Union[int, float]:
    """Clamp a numeric value to optional minimum and maximum bounds."""
    result = value
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_int(value: Any, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    """Convert a value to int with optional bounds and a safe fallback."""
    try:
        if isinstance(value, bool):
            parsed = int(value)
        elif isinstance(value, float) and not math.isfinite(value):
            parsed = default
        else:
            parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return int(clamp(parsed, minimum, maximum))


def coerce_float(value: Any, default: float, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    """Convert a value to float with optional bounds and a safe fallback."""
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            parsed = default
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return float(clamp(parsed, minimum, maximum))


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Convert common truthy/falsy config and payload values into bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "disabled", "disable"}:
        return False
    return default


def coerce_str(value: Any, default: str = "", *, max_chars: Optional[int] = None, strip: bool = True) -> str:
    """Convert a value to string with optional stripping and truncation."""
    if value is None:
        text = default
    else:
        text = str(value)
    if strip:
        text = text.strip()
    if not text and default:
        text = default
    if max_chars is not None:
        text = truncate_text(text, max_chars)
    return text


def coerce_mapping(value: Any, *, default: Optional[Mapping[str, Any]] = None, copy_value: bool = True) -> Dict[str, Any]:
    """Return a dictionary from mapping-like input without mutating the original."""
    if isinstance(value, Mapping):
        return dict(value) if copy_value else value  # type: ignore[return-value]
    return dict(default or {})


def coerce_list(value: Any, *, default: Optional[Sequence[Any]] = None, split_strings: bool = False) -> List[Any]:
    """Normalize scalar/sequence input into a list."""
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if split_strings and isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def truncate_text(value: Any, limit: int = 280, *, suffix: str = "...") -> str:
    """Return a string capped to a bounded display/log size."""
    text = str(value or "")
    safe_limit = max(1, int(limit))
    if len(text) <= safe_limit:
        return text
    if safe_limit <= len(suffix):
        return text[:safe_limit]
    return f"{text[: safe_limit - len(suffix)]}{suffix}"


def safe_ratio(numerator: Any, denominator: Any, *, default: float = 0.0, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    """Safely divide two values and optionally clamp the result."""
    num = coerce_float(numerator, default=0.0)
    den = coerce_float(denominator, default=0.0)
    if den == 0.0:
        return default
    return coerce_float(num / den, default=default, minimum=minimum, maximum=maximum)


def bounded_append(buffer: MutableSequence[Any], item: Any, max_items: int) -> None:
    """Append one item and trim the mutable sequence in-place."""
    limit = coerce_int(max_items, 1, minimum=1)
    buffer.append(item)
    if len(buffer) > limit:
        del buffer[:-limit]


def bounded_extend(buffer: MutableSequence[Any], items: Iterable[Any], max_items: int) -> None:
    """Extend a mutable sequence and trim it in-place."""
    limit = coerce_int(max_items, 1, minimum=1)
    buffer.extend(items)
    if len(buffer) > limit:
        del buffer[:-limit]


def bounded_list(items: Iterable[Any], max_items: int) -> List[Any]:
    """Return the newest max_items from an iterable as a list."""
    limit = coerce_int(max_items, 1, minimum=1)
    values = list(items)
    return values[-limit:]


def deep_merge(*mappings: Optional[Mapping[str, Any]], overwrite_none: bool = False) -> Dict[str, Any]:
    """Deep-merge dictionaries from left to right without mutating inputs."""
    merged: Dict[str, Any] = {}
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        for key, value in mapping.items():
            if value is None and not overwrite_none:
                continue
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = deep_merge(merged[key], value, overwrite_none=overwrite_none)  # type: ignore[arg-type]
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def deep_get(mapping: Mapping[str, Any], path: Union[str, Sequence[str]], default: Any = None, *, separator: str = ".") -> Any:
    """Read a nested mapping path such as 'sla.deadline_ts'."""
    if not isinstance(mapping, Mapping):
        return default
    parts = path.split(separator) if isinstance(path, str) else list(path)
    cursor: Any = mapping
    for part in parts:
        if isinstance(cursor, Mapping) and part in cursor:
            cursor = cursor[part]
        else:
            return default
    return cursor


def compact_dict(mapping: Mapping[str, Any], *, drop_none: bool = True, drop_empty: bool = False) -> Dict[str, Any]:
    """Return a copy with optional removal of None or empty values."""
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        if drop_none and value is None:
            continue
        if drop_empty and value in ({}, [], (), ""):
            continue
        result[str(key)] = value
    return result


def select_keys(mapping: Optional[Mapping[str, Any]], keys: Iterable[str], *, include_missing: bool = False, default: Any = None) -> Dict[str, Any]:
    """Extract a fixed subset of keys from a mapping."""
    source = mapping or {}
    selected: Dict[str, Any] = {}
    for key in keys:
        if key in source:
            selected[key] = source[key]
        elif include_missing:
            selected[key] = default
    return selected


def require_keys(mapping: Mapping[str, Any], required_keys: Iterable[str], *, source: str = "payload") -> None:
    """Validate required keys and raise a typed handler validation error when missing."""
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise ValidationError(
            f"Missing required {source} key(s): {', '.join(missing)}",
            context={"source": source, "missing_keys": missing},
            code="HANDLER_REQUIRED_KEYS_MISSING",
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, bytes):
        return f"[bytes:{len(value)}]"
    if isinstance(value, BaseException):
        return {"exception_type": type(value).__name__, "message": str(value)}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return repr(value)


def make_json_safe(
    value: Any,
    *,
    max_depth: int = 6,
    max_items: int = 100,
    max_string_chars: int = 1000,
    _depth: int = 0,
    _seen: Optional[set[int]] = None,
) -> JsonValue:
    """Convert arbitrary values to bounded JSON-compatible data structures."""
    if _seen is None:
        _seen = set()

    if _depth > max_depth:
        return "[MAX_DEPTH]"

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, str):
        return truncate_text(value, max_string_chars)
    if isinstance(value, Enum):
        return make_json_safe(value.value, max_depth=max_depth, max_items=max_items, max_string_chars=max_string_chars, _depth=_depth + 1, _seen=_seen)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"[bytes:{len(value)}]"
    if isinstance(value, BaseException):
        return {"exception_type": type(value).__name__, "message": truncate_text(str(value), max_string_chars)}

    object_id = id(value)
    if object_id in _seen:
        return "[CIRCULAR]"
    _seen.add(object_id)

    if isinstance(value, Mapping):
        result_dict: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                result_dict["[TRUNCATED]"] = f"{len(value) - max_items} additional item(s) omitted"
                break
            result_dict[str(key)] = make_json_safe(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string_chars=max_string_chars,
                _depth=_depth + 1,
                _seen=_seen,
            )
        _seen.discard(object_id)
        return result_dict

    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        result = [
            make_json_safe(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string_chars=max_string_chars,
                _depth=_depth + 1,
                _seen=_seen,
            )
            for item in sequence[:max_items]
        ]
        if len(sequence) > max_items:
            result.append(f"[TRUNCATED:{len(sequence) - max_items}]")
        _seen.discard(object_id)
        return result

    _seen.discard(object_id)
    return truncate_text(repr(value), max_string_chars)


def stable_json_dumps(value: Any, *, sort_keys: bool = True, ensure_ascii: bool = False) -> str:
    """Serialize arbitrary values deterministically for hashing and telemetry."""
    try:
        return json.dumps(make_json_safe(value), sort_keys=sort_keys, ensure_ascii=ensure_ascii, separators=(",", ":"), default=_json_default)
    except (TypeError, ValueError) as exc:
        raise SerializationError(
            "Unable to serialize handler payload",
            cause=exc,
            context={"value_type": type(value).__name__},
            code="HANDLER_JSON_SERIALIZATION_FAILED",
        ) from exc


def stable_hash(value: Any, *, algorithm: str = "sha256", length: Optional[int] = None, policy: Optional[HandlerErrorPolicy] = None) -> str:
    """Return a deterministic hash for arbitrary values."""
    payload = policy.sanitize_context(value) if policy else value
    serialized = stable_json_dumps(payload)
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValidationError(
            f"Unsupported hash algorithm: {algorithm}",
            cause=exc,
            context={"algorithm": algorithm},
            code="HANDLER_UNSUPPORTED_HASH_ALGORITHM",
        ) from exc
    digest.update(serialized.encode("utf-8"))
    value_hash = digest.hexdigest()
    if length is not None:
        return value_hash[: max(1, int(length))]
    return value_hash


def normalize_severity(value: Any, default: Union[str, FailureSeverity] = FailureSeverity.MEDIUM) -> str:
    """Normalize a severity value to the canonical Handler severity string."""
    return FailureSeverity.normalize(value or default).value


def severity_rank(value: Any) -> int:
    """Return the numeric rank for a severity value."""
    return FailureSeverity.normalize(value).rank


def normalize_recovery_action(value: Any, default: Union[str, HandlerRecoveryAction] = HandlerRecoveryAction.NONE) -> str:
    """Normalize a recovery action to a canonical policy action string."""
    if value is None:
        value = default
    return HandlerRecoveryAction.normalize(value).value


def normalize_error_type(value: Any, default: Union[str, HandlerErrorType] = HandlerErrorType.GENERIC) -> str:
    """Normalize Handler error type values without forcing unknown external types into a fixed enum."""
    if value is None:
        value = default
    return HandlerErrorType.normalize(value)


def infer_severity(error_type: Any = None, message: Any = None, *, default: str = FailureSeverity.LOW.value) -> str:
    """Infer a severity from error type and message when no policy decision is available."""
    lowered = f"{error_type or ''} {message or ''}".lower()
    if any(tag in lowered for tag in _SEVERITY_CRITICAL_TAGS):
        return FailureSeverity.CRITICAL.value
    if any(tag in lowered for tag in _SEVERITY_HIGH_TAGS):
        return FailureSeverity.HIGH.value
    if any(tag in lowered for tag in _SEVERITY_MEDIUM_TAGS):
        return FailureSeverity.MEDIUM.value
    return normalize_severity(default)


def infer_retryable(error_type: Any = None, message: Any = None, *, severity: Any = None, default: bool = False) -> bool:
    """Infer retryability from error tokens and severity."""
    normalized_severity = FailureSeverity.normalize(severity or infer_severity(error_type, message))
    if normalized_severity == FailureSeverity.CRITICAL:
        return False
    lowered = f"{error_type or ''} {message or ''}".lower()
    if any(tag in lowered for tag in _NON_RETRYABLE_TAGS):
        return False
    if any(tag in lowered for tag in _RETRYABLE_TAGS):
        return True
    return bool(default)


def classify_failure_category(error_type: Any = None, message: Any = None, *, default: str = "runtime") -> str:
    """Classify a failure into the Handler recovery strategy taxonomy."""
    lowered = f"{error_type or ''} {message or ''}".lower()
    for category, tags in _FAILURE_CATEGORY_TAGS.items():
        if any(tag in lowered for tag in tags):
            return category
    return default


def extract_exception_info(error: Optional[BaseException] = None, error_info: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Extract error_type/message/cause fields from raw exceptions and payloads."""
    payload = coerce_mapping(error_info)
    error_type = payload.get("error_type") or payload.get("type")
    error_message = payload.get("error_message") or payload.get("message")

    if error is not None:
        error_type = error_type or type(error).__name__
        error_message = error_message or str(error)

    return {
        "error_type": normalize_error_type(error_type or DEFAULT_FAILURE_TYPE),
        "error_message": coerce_str(error_message or DEFAULT_FAILURE_MESSAGE, max_chars=4000),
        "cause": type(error).__name__ if error is not None else payload.get("cause"),
    }


def make_context_hash(
    *,
    error_type: Any,
    message: Any,
    context: Optional[Mapping[str, Any]] = None,
    policy: Optional[HandlerErrorPolicy] = None,
    length: Optional[int] = None,
) -> str:
    """Create a stable context hash for de-duplication and routing."""
    return stable_hash(
        {
            "error_type": normalize_error_type(error_type or DEFAULT_FAILURE_TYPE),
            "message": coerce_str(message or DEFAULT_FAILURE_MESSAGE, max_chars=4000),
            "context": context or {},
        },
        length=length,
        policy=policy,
    )


def build_normalized_failure(
    *,
    error: Optional[BaseException] = None,
    error_info: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    policy: Optional[HandlerErrorPolicy] = None,
    source: Optional[str] = None,
    correlation_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the canonical failure payload consumed by recovery, telemetry, SLA, and escalation modules."""
    context = coerce_mapping(context)
    info = extract_exception_info(error=error, error_info=error_info)
    raw_payload = coerce_mapping(error_info)

    severity = normalize_severity(raw_payload.get("severity") or infer_severity(info["error_type"], info["error_message"]))
    retryable = raw_payload.get("retryable")
    if retryable is None:
        retryable = infer_retryable(info["error_type"], info["error_message"], severity=severity)
    else:
        retryable = coerce_bool(retryable)

    active_policy = policy or HandlerErrorPolicy()
    message = active_policy.sanitize_message(info["error_message"])
    context_hash = raw_payload.get("context_hash") or make_context_hash(
        error_type=info["error_type"],
        message=message,
        context=context,
        policy=active_policy,
    )
    policy_action = raw_payload.get("policy_action") or active_policy.resolve_action(
        error_type=info["error_type"],
        severity=FailureSeverity.normalize(severity),
        retryable=bool(retryable),
    ).value

    return compact_dict(
        {
            "schema": DEFAULT_FAILURE_SCHEMA,
            "type": info["error_type"],
            "message": message,
            "severity": severity,
            "retryable": bool(retryable),
            "category": classify_failure_category(info["error_type"], message),
            "context_hash": str(context_hash),
            "timestamp": coerce_float(timestamp, utc_timestamp()),
            "policy_action": normalize_recovery_action(policy_action),
            "source": source or raw_payload.get("source"),
            "correlation_id": correlation_id or raw_payload.get("correlation_id") or context.get("correlation_id"),
            "cause": info.get("cause"),
        },
        drop_none=True,
    )


def normalize_failure_payload(normalized_failure: Mapping[str, Any], *, policy: Optional[HandlerErrorPolicy] = None) -> Dict[str, Any]:
    """Normalize any failure-shaped mapping or HandlerError into the compact Handler failure schema."""
    if isinstance(normalized_failure, HandlerError):
        return normalized_failure.to_failure_payload()

    active_policy = policy or HandlerErrorPolicy()
    failure = coerce_mapping(normalized_failure)
    error_type = normalize_error_type(failure.get("type") or failure.get("error_type") or DEFAULT_FAILURE_TYPE)
    message = active_policy.sanitize_message(failure.get("message") or failure.get("error_message") or DEFAULT_FAILURE_MESSAGE)
    severity = normalize_severity(failure.get("severity") or infer_severity(error_type, message), default=FailureSeverity.LOW)
    retryable = failure.get("retryable")
    if retryable is None:
        retryable = infer_retryable(error_type, message, severity=severity)
    else:
        retryable = coerce_bool(retryable)

    context_hash = failure.get("context_hash") or failure.get("fingerprint") or make_context_hash(
        error_type=error_type,
        message=message,
        context=coerce_mapping(failure.get("context")),
        policy=active_policy,
    )

    return compact_dict(
        {
            "schema": str(failure.get("schema") or DEFAULT_FAILURE_SCHEMA),
            "type": error_type,
            "message": message,
            "severity": severity,
            "retryable": bool(retryable),
            "category": failure.get("category") or classify_failure_category(error_type, message),
            "context_hash": str(context_hash),
            "timestamp": coerce_float(failure.get("timestamp"), utc_timestamp()),
            "policy_action": normalize_recovery_action(failure.get("policy_action") or failure.get("action")),
            "code": failure.get("code"),
            "source": failure.get("source"),
            "correlation_id": failure.get("correlation_id"),
        },
        drop_none=True,
    )


def build_error_info(normalized_failure: Mapping[str, Any]) -> Dict[str, str]:
    """Create the IssueHandler-compatible error_info payload from normalized failure data."""
    failure = normalize_failure_payload(normalized_failure)
    return {
        "error_type": failure.get("type", DEFAULT_FAILURE_TYPE),
        "error_message": failure.get("message", DEFAULT_FAILURE_MESSAGE),
    }


def normalize_recovery_result(recovery_result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize recovery results emitted by strategies, fallback mode, or escalation paths."""
    result = coerce_mapping(recovery_result)
    status = str(result.get("status") or "unknown").lower()
    if status not in {"recovered", "failed", "degraded", "skipped", "unknown"}:
        status = "recovered" if result.get("ok") is True else status

    attempts = coerce_int(result.get("attempts", result.get("retry_count", 0)), 0, minimum=0)
    strategy = coerce_str(result.get("strategy") or result.get("selected_strategy") or "unknown", default="unknown")

    return compact_dict(
        {
            "status": status,
            "strategy": strategy,
            "attempts": attempts,
            "recommendation": result.get("recommendation"),
            "result": result.get("result"),
            "last_result": result.get("last_result"),
            "checkpoint_id": result.get("checkpoint_id"),
            "checkpoint_restored": coerce_bool(result.get("checkpoint_restored"), default=False),
            "max_retries": result.get("max_retries"),
            "sla": coerce_mapping(result.get("sla")),
            "strategy_distribution": coerce_mapping(result.get("strategy_distribution")),
            "escalation": result.get("escalation"),
        },
        drop_none=True,
    )


def is_recovered_result(result: Any) -> bool:
    """Return True when a strategy result should be treated as recovered."""
    if isinstance(result, Mapping):
        status = str(result.get("status", "")).lower()
        if status == "failed":
            return False
        if status in {"recovered", "ok", "success", "degraded"}:
            return True
        if result.get("ok") is False:
            return False
    return result is not None


def build_telemetry_event(
    *,
    event_type: str,
    failure: Mapping[str, Any],
    recovery: Mapping[str, Any],
    context: Optional[Mapping[str, Any]] = None,
    insight: Optional[Mapping[str, Any]] = None,
    sla: Optional[Mapping[str, Any]] = None,
    strategy_distribution: Optional[Mapping[str, Any]] = None,
    correlation_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a bounded telemetry event shared by HandlerAgent, memory, and evaluators."""
    context = coerce_mapping(context)
    failure_payload = normalize_failure_payload(failure)
    recovery_payload = normalize_recovery_result(recovery)
    telemetry_correlation_id = correlation_id or failure_payload.get("correlation_id") or context.get("correlation_id")

    return compact_dict(
        {
            "schema": DEFAULT_TELEMETRY_SCHEMA,
            "event_type": normalize_identifier(event_type, default="handler_event"),
            "timestamp": coerce_float(timestamp, utc_timestamp()),
            "correlation_id": telemetry_correlation_id,
            "failure": failure_payload,
            "recovery": recovery_payload,
            "insight": coerce_mapping(insight),
            "context": select_keys(context, ("route", "agent", "task_id", "priority", "correlation_id")),
            "sla": coerce_mapping(sla if sla is not None else recovery_payload.get("sla")),
            "strategy_distribution": coerce_mapping(strategy_distribution if strategy_distribution is not None else recovery_payload.get("strategy_distribution")),
        },
        drop_none=True,
    )


def build_postmortem(
    *,
    normalized_failure: Mapping[str, Any],
    recovery_result: Mapping[str, Any],
    telemetry: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the compact postmortem artifact used by learning/adaptive feedback loops."""
    failure = normalize_failure_payload(normalized_failure)
    recovery = normalize_recovery_result(recovery_result)
    telemetry = coerce_mapping(telemetry)
    context = coerce_mapping(context)
    insight = coerce_mapping(telemetry.get("insight"))

    return compact_dict(
        {
            "timestamp": coerce_float(timestamp, utc_timestamp()),
            "failure_type": failure.get("type"),
            "severity": failure.get("severity"),
            "retryable": failure.get("retryable"),
            "category": failure.get("category"),
            "context_hash": failure.get("context_hash"),
            "recovery_status": recovery.get("status"),
            "strategy": recovery.get("strategy"),
            "task_id": context.get("task_id"),
            "telemetry_ref": telemetry.get("timestamp"),
            "failure_signature": insight.get("signature"),
            "recommendation": insight.get("recommendation") or recovery.get("recommendation") or "collect_more_signals",
            "correlation_id": failure.get("correlation_id") or telemetry.get("correlation_id") or context.get("correlation_id"),
        },
        drop_none=True,
    )


def get_agent_name(agent: Any, *, context: Optional[Mapping[str, Any]] = None, default: str = "unknown_agent") -> str:
    """Resolve a stable agent name from object attributes or context."""
    context = context or {}
    for attr in ("name", "agent_name", "id"):
        value = getattr(agent, attr, None)
        if value:
            return str(value)
    return str(context.get("agent") or default)


def extract_task_context(task_data: Optional[Mapping[str, Any]], *, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Extract and normalize context from HandlerAgent task payloads."""
    task = coerce_mapping(task_data)
    context = coerce_mapping(task.get("context"), default=default)
    if "task_id" not in context and task.get("task_id"):
        context["task_id"] = task.get("task_id")
    if "agent" not in context and task.get("target_agent") is not None:
        context["agent"] = get_agent_name(task.get("target_agent"))
    if "correlation_id" not in context:
        context["correlation_id"] = generate_correlation_id("handler")
    return context


def build_checkpoint_payload(
    *,
    label: str,
    state: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
    checkpoint_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    policy: Optional[HandlerErrorPolicy] = None,
) -> Dict[str, Any]:
    """Build a checkpoint payload with bounded, redacted JSON-safe state and metadata."""
    safe_label = normalize_identifier(label, default="checkpoint")
    active_policy = policy or HandlerErrorPolicy()
    return {
        "schema": DEFAULT_CHECKPOINT_SCHEMA,
        "id": checkpoint_id or generate_checkpoint_id(safe_label),
        "label": safe_label,
        "created": coerce_float(timestamp, utc_timestamp()),
        "state": make_json_safe(active_policy.sanitize_context(state)),
        "metadata": make_json_safe(active_policy.sanitize_context(metadata or {})),
    }


def shared_memory_get(shared_memory: Any, key: str, default: Any = None) -> Any:
    """Safely read from SharedMemory-like objects without assuming implementation details."""
    if shared_memory is None or not hasattr(shared_memory, "get"):
        return default
    value = shared_memory.get(key)
    return default if value is None else value


def shared_memory_set(shared_memory: Any, key: str, value: Any, *, ttl: Optional[int] = None) -> bool:
    """Safely write to SharedMemory-like objects, supporting implementations with or without ttl."""
    if shared_memory is None or not hasattr(shared_memory, "set"):
        return False
    if ttl is not None:
        try:
            shared_memory.set(key, value, ttl=ttl)
            return True
        except TypeError:
            shared_memory.set(key, value)
            return True
    shared_memory.set(key, value)
    return True


def append_shared_memory_list(
    shared_memory: Any,
    key: str,
    item: Any,
    *,
    max_items: int = 1000,
    ttl: Optional[int] = None,
    default_factory: Callable[[], MutableSequence[Any]] = list,
) -> List[Any]:
    """Append an item to a list stored in SharedMemory and persist the bounded list."""
    current = shared_memory_get(shared_memory, key)
    if not isinstance(current, list):
        current = list(default_factory())
    bounded_append(current, item, max_items=max_items)
    shared_memory_set(shared_memory, key, current, ttl=ttl)
    return current


def save_shared_checkpoint(
    shared_memory: Any,
    *,
    label: str,
    state: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
    ttl: Optional[int] = None,
    key_prefix: str = "handler:checkpoint",
    policy: Optional[HandlerErrorPolicy] = None,
) -> str:
    """Persist a redacted checkpoint payload into SharedMemory and return the key."""
    checkpoint = build_checkpoint_payload(label=label, state=state, metadata=metadata, policy=policy)
    checkpoint_id = str(checkpoint["id"])
    key = checkpoint_id if checkpoint_id.startswith(key_prefix) else f"{key_prefix}:{checkpoint_id}"
    shared_memory_set(shared_memory, key, checkpoint, ttl=ttl)
    return key


def read_shared_checkpoint(shared_memory: Any, checkpoint_id: str) -> Optional[Dict[str, Any]]:
    """Read a checkpoint payload from SharedMemory-like storage."""
    value = shared_memory_get(shared_memory, checkpoint_id)
    if isinstance(value, Mapping):
        return dict(value)
    return None


def compute_remaining_budget(
    *,
    context: Optional[Mapping[str, Any]] = None,
    default_seconds: float = 30.0,
    now: Optional[float] = None,
) -> float:
    """Compute remaining recovery budget from SLA context without owning SLA policy decisions."""
    context = context or {}
    sla = deep_get(context, "sla", default={})
    if not isinstance(sla, Mapping):
        return max(0.0, float(default_seconds))

    current_time = utc_timestamp() if now is None else float(now)
    deadline_ts = sla.get("deadline_ts")
    explicit_budget = sla.get("max_recovery_seconds")
    latency_budget_ms = sla.get("latency_budget_ms")

    if isinstance(deadline_ts, (int, float)):
        return max(0.0, float(deadline_ts) - current_time)
    if isinstance(explicit_budget, (int, float)):
        return max(0.0, float(explicit_budget))
    if isinstance(latency_budget_ms, (int, float)):
        return max(0.0, float(latency_budget_ms) / 1000.0)
    return max(0.0, float(default_seconds))


def build_escalation_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract the stable context fields used by escalation payloads."""
    return select_keys(context or {}, ("task_id", "route", "agent", "priority", "correlation_id"), include_missing=False)


def strategy_base_name(strategy: Any) -> str:
    """Return the primary recovery strategy name before '+fallback' suffixes."""
    return str(strategy or "unknown").split("+", 1)[0]


def summarize_strategy_distribution(distribution: Optional[Mapping[str, Any]], *, precision: int = 4) -> Dict[str, float]:
    """Normalize and round a recovery strategy probability distribution."""
    values = {str(key): max(0.0, coerce_float(value, 0.0)) for key, value in coerce_mapping(distribution).items()}
    total = sum(values.values())
    if total <= 0:
        return {}
    return {key: round(value / total, precision) for key, value in values.items()}


def stable_sort_events(events: Iterable[Mapping[str, Any]], *, timestamp_key: str = "timestamp", reverse: bool = False) -> List[Dict[str, Any]]:
    """Sort telemetry/postmortem events by timestamp using safe fallbacks."""
    normalized = [dict(event) for event in events if isinstance(event, Mapping)]
    return sorted(normalized, key=lambda event: coerce_float(event.get(timestamp_key), 0.0), reverse=reverse)


def recent_events(events: Iterable[Mapping[str, Any]], *, limit: int = 100, timestamp_key: str = "timestamp") -> List[Dict[str, Any]]:
    """Return the most recent events in chronological order."""
    sorted_events = stable_sort_events(events, timestamp_key=timestamp_key)
    return sorted_events[-coerce_int(limit, 100, minimum=1):]


def event_matches_failure(event: Mapping[str, Any], *, context_hash: Optional[str] = None, signature: Optional[str] = None) -> bool:
    """Check whether a telemetry event matches a failure fingerprint or insight signature."""
    failure = event.get("failure", {}) if isinstance(event, Mapping) else {}
    insight = event.get("insight", {}) if isinstance(event, Mapping) else {}
    if context_hash and isinstance(failure, Mapping) and failure.get("context_hash") == context_hash:
        return True
    if signature and isinstance(insight, Mapping) and insight.get("signature") == signature:
        return True
    return False


def success_rate_for_events(events: Iterable[Mapping[str, Any]]) -> Dict[str, Union[int, float]]:
    """Calculate recovery success statistics from telemetry-like events."""
    total = 0
    recovered = 0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        total += 1
        recovery = event.get("recovery", {})
        if isinstance(recovery, Mapping) and recovery.get("status") == "recovered":
            recovered += 1
    return {
        "total": total,
        "recovered": recovered,
        "failed": max(0, total - recovered),
        "success_rate": safe_ratio(recovered, total, default=0.0, minimum=0.0, maximum=1.0),
    }


if __name__ == "__main__":
    print("\n=== Running Handler Helpers ===\n")
    printer.status("TEST", "Handler Helpers initialized", "info")

    policy = HandlerErrorPolicy(
        name="handler_helpers.strict_test",
        expose_internal_messages=False,
        include_context_in_public=False,
        include_context_in_telemetry=True,
        max_message_chars=240,
        max_string_chars=160,
    )

    secret_context = {
        "route": "handler.recovery",
        "agent": "demo_agent",
        "task_id": "handler-helper-smoke-001",
        "password": "SuperSecret123",
        "nested": {"api_key": "sk-test-123", "safe": "visible"},
    }

    failure = build_normalized_failure(
        error=TimeoutError("Upstream timed out with Authorization: Bearer token-123"),
        context=secret_context,
        policy=policy,
        source="handler.helpers.__main__",
        correlation_id="corr-handler-helper-test",
    )
    assert failure["type"] == "TimeoutError"
    assert failure["severity"] == FailureSeverity.MEDIUM.value
    assert failure["retryable"] is True
    assert failure["category"] == "timeout"
    assert failure["policy_action"] == HandlerRecoveryAction.RETRY.value

    failure_payload = normalize_failure_payload(failure, policy=policy)
    error_info = build_error_info(failure_payload)
    assert error_info["error_type"] == "TimeoutError"
    assert "token-123" not in stable_json_dumps(failure_payload)

    telemetry = build_telemetry_event(
        event_type="handler_recovery",
        failure=failure_payload,
        recovery={"status": "recovered", "strategy": "timeout", "attempts": 1, "sla": {"remaining_seconds": 12.5}},
        context=secret_context,
        insight={"signature": "timeout:abc", "recommendation": "retry_with_backoff"},
    )
    postmortem = build_postmortem(
        normalized_failure=failure_payload,
        recovery_result=telemetry["recovery"],
        telemetry=telemetry,
        context=secret_context,
    )

    class DemoSharedMemory:
        def __init__(self):
            self.store: Dict[str, Any] = {}

        def get(self, key: str) -> Any:
            return self.store.get(key)

        def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
            self.store[key] = value
            if ttl is not None:
                self.store[f"{key}:ttl"] = ttl

    shared_memory = DemoSharedMemory()
    append_shared_memory_list(shared_memory, "handler:telemetry", telemetry, max_items=2, ttl=60)
    append_shared_memory_list(shared_memory, "handler:telemetry", {"timestamp": utc_timestamp(), "recovery": {"status": "failed"}}, max_items=2)
    stats = success_rate_for_events(shared_memory_get(shared_memory, "handler:telemetry", []))

    checkpoint_key = save_shared_checkpoint(
        shared_memory,
        label="pre_recovery",
        state={"task_data": {"operation": "demo"}, "context": secret_context},
        metadata={"strategy": "timeout"},
        ttl=120,
        policy=policy,
    )
    checkpoint = read_shared_checkpoint(shared_memory, checkpoint_key)

    serialized_all = stable_json_dumps(
        {
            "failure": failure_payload,
            "telemetry": telemetry,
            "postmortem": postmortem,
            "checkpoint": checkpoint,
            "stats": stats,
        }
    )
    assert "SuperSecret123" not in serialized_all
    assert "sk-test-123" not in serialized_all
    assert "token-123" not in serialized_all
    assert checkpoint is not None
    assert stats["total"] == 2
    assert 0.0 <= stats["success_rate"] <= 1.0

    printer.pretty("Normalized failure", failure_payload, "success")
    printer.pretty("Telemetry", telemetry, "success")
    printer.pretty("Postmortem", postmortem, "success")
    printer.pretty("Recovery stats", stats, "success")
    print("\n=== Test ran successfully ===\n")
