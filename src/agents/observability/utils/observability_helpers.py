"""
Shared helpers for the observability subsystem.

This module centralizes the private helper logic.
"""

from __future__ import annotations

import hashlib
import math
import time

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, TypeVar

from .observability_error import normalize_observability_exception

T = TypeVar("T")

LEVEL_ORDER: Dict[str, int] = {
    "info": 10,
    "warning": 20,
    "critical": 30,
}

STATUS_ALIASES: Dict[str, str] = {
    "ok": "ok",
    "success": "ok",
    "succeeded": "ok",
    "completed": "ok",
    "complete": "ok",
    "done": "ok",
    "running": "running",
    "in_progress": "running",
    "queued": "queued",
    "pending": "queued",
    "retry": "retry",
    "retried": "retry",
    "retrying": "retry",
    "error": "error",
    "failed": "error",
    "failure": "error",
    "exception": "error",
    "timeout": "timeout",
    "timed_out": "timeout",
    "deadline_exceeded": "timeout",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "skipped": "skipped",
}

TERMINAL_STATUSES = {"ok", "error", "timeout", "cancelled", "skipped"}
DEFAULT_REQUIRED_CONTEXT_FIELDS = ("trace_id", "agent_name", "operation_name")


# ---------------------------------------------------------------------------
# Core time / collection helpers
# ---------------------------------------------------------------------------
def now_ms() -> float:
    return time.time() * 1000.0


def coerce_mapping(value: Any, *, strict: bool = False) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if strict:
        raise TypeError(f"expected mapping, received {type(value).__name__}")
    return {}


def coerce_sequence(value: Optional[Iterable[Any]]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    return list(value)


def iter_record_mappings(value: Any, *, key_field: Optional[str] = None) -> List[Dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, Mapping):
        records: List[Dict[str, Any]] = []
        if all(isinstance(item, Mapping) for item in value.values()):
            for key, item in value.items():
                record = dict(item)
                if key_field and key_field not in record:
                    record[key_field] = str(key)
                records.append(record)
            return records
        return [dict(value)]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                records.append(dict(item))
        return records

    return []


def evict_ordered_dict(dictionary: MutableMapping[str, Any], limit: int) -> None:
    while len(dictionary) > max(0, int(limit)):
        dictionary.pop(next(iter(dictionary)), None)


# ---------------------------------------------------------------------------
# String / scalar helpers
# ---------------------------------------------------------------------------
def optional_str(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def require_non_empty_str(value: Any, *, field_name: str, operation: str,
                          error_factory: Callable[[str, str], Exception]) -> str:
    text = str(value).strip() if value is not None else ""
    if text:
        return text
    raise error_factory(field_name, operation)


def truncate_text(value: Any, limit: int = 280) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def dedupe_preserve_order(values: Iterable[Any], *, limit: Optional[int] = None) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if limit is not None and len(output) >= limit:
            break
    return output


def merge_limited_strings(existing: Sequence[str], incoming: Optional[Iterable[Any]], *, limit: int) -> List[str]:
    merged = list(existing)
    for value in incoming or []:
        text = str(value).strip()
        if not text or text in merged:
            continue
        merged.append(text)
        if len(merged) >= limit:
            break
    return merged


def safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): safe_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_scalar(item) for item in value]
    return str(value)


def safe_mapping(value: Optional[Mapping[str, Any]], *, scalarize: bool = False) -> Dict[str, Any]:
    if value is None:
        return {}
    mapping = dict(value)
    if scalarize:
        return {str(key): safe_scalar(item) for key, item in mapping.items()}
    return {str(key): item for key, item in mapping.items()}


def safe_numeric_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if value is None:
        return {}
    return {str(key): float(item) for key, item in dict(value).items()}


# ---------------------------------------------------------------------------
# Numeric / math helpers
# ---------------------------------------------------------------------------
def coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    return coerce_float(value, default=default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def coerce_non_negative_float(
    value: Any,
    *,
    field_name: str,
    operation: str,
    error_factory: Callable[[str, str, Any], Exception],
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise error_factory(field_name, operation, value) from exc

    if number < 0:
        raise error_factory(field_name, operation, number)
    return number


def coerce_utilization_pct(
    value: Any,
    *,
    field_name: str,
    operation: str,
    error_factory: Callable[[str, str, Any], Exception],
) -> float:
    pct = coerce_non_negative_float(
        value,
        field_name=field_name,
        operation=operation,
        error_factory=error_factory,
    )
    if pct > 100.0:
        raise error_factory(field_name, operation, pct)
    return pct


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(float(value) for value in values)
    pct = min(100.0, max(0.0, float(pct)))
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]

    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0 if numerator <= 0 else float("inf")
    return numerator / denominator


def bucketize(value: float, buckets: Sequence[float], *, suffix: str = "ms") -> str:
    for bucket in buckets:
        if value <= bucket:
            return f"<= {bucket:g}{suffix}"
    if not buckets:
        return "> unbounded"
    return f"> {float(buckets[-1]):g}{suffix}"


# ---------------------------------------------------------------------------
# Incident / status normalization helpers
# ---------------------------------------------------------------------------
def level_rank(level: str, *, level_order: Optional[Mapping[str, int]] = None) -> int:
    mapping = dict(level_order or LEVEL_ORDER)
    return int(mapping.get(str(level or "info").lower(), mapping.get("info", 10)))


def normalize_level(level: Any) -> str:
    text = str(level or "info").strip().lower()
    if text in {"critical", "crit", "sev1", "p1", "fatal"}:
        return "critical"
    if text in {"warning", "warn", "high", "medium", "sev2", "p2"}:
        return "warning"
    return "info"


def normalize_status(value: Any, *, aliases: Optional[Mapping[str, str]] = None) -> str:
    text = str(value or "running").strip().lower()
    mapping = dict(aliases or STATUS_ALIASES)
    return mapping.get(text, text or "running")


def normalize_severity(value: Any) -> str:
    text = str(value or "info").strip().lower()
    if text not in {"info", "warning", "error", "critical"}:
        return "info"
    return text


def severity_for_status(status: str) -> str:
    normalized = normalize_status(status)
    if normalized in {"error", "timeout"}:
        return "error"
    if normalized == "retry":
        return "warning"
    return "info"


def extract_trace_id(*sources: Optional[Mapping[str, Any]]) -> Optional[str]:
    for source in sources:
        if not source:
            continue
        for key in ("trace_id", "observability.trace_id"):
            value = source.get(key)
            if value:
                return str(value)
    return None


def infer_incident_start(timeline: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if not timeline:
        return None
    return min(safe_float(item.get("timestamp_ms"), now_ms()) for item in timeline)


# ---------------------------------------------------------------------------
# Signature / similarity helpers
# ---------------------------------------------------------------------------
def normalize_signature(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    return " ".join(raw.split())


def signature_hash(signature: str) -> str:
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def signature_tokens(signature: str) -> set[str]:
    normalized = normalize_signature(signature).replace("|", " ").replace(":", " ")
    return {token for token in normalized.split(" ") if token}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left.intersection(right))
    union = len(left.union(right))
    return intersection / union if union else 0.0


# ---------------------------------------------------------------------------
# Sanitization helpers for tracing payloads / metadata
# ---------------------------------------------------------------------------
def sanitize_value(
    value: Any,
    *,
    max_string_length: int = 1024,
    max_items: int = 64,
    depth: int = 0,
    max_depth: int = 3,
) -> Any:
    if depth >= max_depth:
        return "<max_depth_exceeded>"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return truncate_text(value, max_string_length)

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_value(
                item,
                max_string_length=max_string_length,
                max_items=max_items,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for key, item in list(value.items())[:max_items]
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized = [
            sanitize_value(
                item,
                max_string_length=max_string_length,
                max_items=max_items,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            sanitized.append("<truncated>")
        return sanitized

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return sanitize_value(
                value.to_dict(),
                max_string_length=max_string_length,
                max_items=max_items,
                depth=depth + 1,
                max_depth=max_depth,
            )
        except Exception:
            return truncate_text(repr(value), max_string_length)

    return truncate_text(repr(value), max_string_length)


def sanitize_mapping(
    value: Any,
    *,
    max_items: int,
    max_string_length: int = 1024,
    nested_max_items: Optional[int] = None,
    max_depth: int = 3,
) -> Dict[str, Any]:
    mapping = coerce_mapping(value)
    sanitized: Dict[str, Any] = {}
    child_limit = nested_max_items if nested_max_items is not None else max_items

    for index, (key, item) in enumerate(mapping.items()):
        if index >= max_items:
            sanitized["__truncated__"] = True
            break
        sanitized[str(key)] = sanitize_value(
            item,
            max_string_length=max_string_length,
            max_items=child_limit,
            depth=0,
            max_depth=max_depth,
        )
    return sanitized


# ---------------------------------------------------------------------------
# Filesystem / duration helpers
# ---------------------------------------------------------------------------
def resolve_storage_path(raw_path: str, *, config_path: Optional[str] = None) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate

    if config_path:
        resolved_config = Path(str(config_path)).resolve()
        return (resolved_config.parent / candidate).resolve()

    return candidate.resolve()


def parse_window_seconds(
    window: Any,
    *,
    default_seconds: float,
    error_factory: Callable[[str, Any], Exception],
    operation: str = "latency_trend",
) -> float:
    if window is None:
        return float(default_seconds)

    if isinstance(window, (int, float)):
        if float(window) <= 0:
            raise error_factory(operation, window)
        return float(window)

    if not isinstance(window, str):
        raise error_factory(operation, window)

    raw = window.strip().lower()
    if raw.isdigit():
        return float(raw)

    unit_map = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = raw[-1]
    magnitude = raw[:-1]
    if unit not in unit_map or not magnitude:
        raise error_factory(operation, window)

    seconds = float(magnitude) * unit_map[unit]
    if seconds <= 0:
        raise error_factory(operation, window)
    return seconds


# ---------------------------------------------------------------------------
# Component state / exception helpers
# ---------------------------------------------------------------------------
def ensure_component_enabled(
    enabled: bool,
    *,
    operation: str,
    disabled_error_factory: Callable[[str], Exception],
) -> None:
    if enabled:
        return
    raise disabled_error_factory(operation)


def handle_component_exception(
    exc: Exception,
    *,
    stage: str,
    context: Optional[Mapping[str, Any]] = None,
    component: Optional[str] = None,
    logger: Any = None,
    report: bool = False,
    passthrough_error_types: Sequence[type[BaseException]] = (),
    trace_id_from_context: bool = False,
):
    if passthrough_error_types and isinstance(exc, tuple(passthrough_error_types)):
        return exc

    normalized = normalize_observability_exception(
        exc,
        stage=stage,
        context={
            **({"component": component} if component else {}),
            **safe_mapping(context),
        },
    )

    if trace_id_from_context and context and getattr(normalized, "trace_id", None) is None and context.get("trace_id"):
        normalized.trace_id = str(context["trace_id"])
        if hasattr(normalized, "_build_tags") and hasattr(normalized, "tags"):
            normalized.tags = normalized._build_tags(normalized.tags)

    if report and hasattr(normalized, "report"):
        normalized.report()

    if logger is not None:
        try:
            logger.error(
                "%s%s",
                f"{component} " if component else "",
                normalized,
            )
        except Exception:
            pass

    return normalized


__all__ = [
    "DEFAULT_REQUIRED_CONTEXT_FIELDS",
    "LEVEL_ORDER",
    "STATUS_ALIASES",
    "TERMINAL_STATUSES",
    "bucketize",
    "coerce_float",
    "coerce_mapping",
    "coerce_non_negative_float",
    "coerce_sequence",
    "coerce_utilization_pct",
    "dedupe_preserve_order",
    "ensure_component_enabled",
    "evict_ordered_dict",
    "extract_trace_id",
    "handle_component_exception",
    "infer_incident_start",
    "iter_record_mappings",
    "jaccard_similarity",
    "level_rank",
    "merge_limited_strings",
    "normalize_level",
    "normalize_severity",
    "normalize_signature",
    "normalize_status",
    "now_ms",
    "optional_str",
    "parse_window_seconds",
    "percentile",
    "ratio",
    "require_non_empty_str",
    "resolve_storage_path",
    "safe_float",
    "safe_int",
    "safe_mapping",
    "safe_numeric_mapping",
    "safe_scalar",
    "sanitize_mapping",
    "sanitize_value",
    "severity_for_status",
    "signature_hash",
    "signature_tokens",
    "truncate_text",
]
