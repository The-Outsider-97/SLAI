"""
Shared helpers for the alignment subsystem.

This module centralizes the alignment helper logic. It exists to keep the
subsystem and telemetry code focused on their core concerns instead of
repeatedly re-implementing parsing, normalization, serialization, redaction,
endpoint handling, identifier generation, and config-backed utility behaviors.

The helpers here are intentionally scoped to reusable alignment-domain
utilities. They do not own fairness evaluation, bias detection, intervention
policy, memory persistence strategy, causal reasoning, or human-oversight
workflow state transitions. Instead, they provide the stable primitives those
modules depend on.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .config_loader import load_global_config, get_config_section
from .alignment_errors import *

GLOBAL_CONFIG = load_global_config()
ALIGNMENT_HELPERS_CONFIG = get_config_section("alignment_helpers") or {}

_DEFAULT_ENCODING = str(ALIGNMENT_HELPERS_CONFIG.get("default_encoding", "utf-8") or "utf-8")
_MAX_JSON_DEPTH = int(ALIGNMENT_HELPERS_CONFIG.get("max_json_depth", 6) or 6)
_DEFAULT_SCHEME = str(ALIGNMENT_HELPERS_CONFIG.get("default_scheme", "https") or "https").strip().lower()
_DEFAULT_TIMEZONE = str(ALIGNMENT_HELPERS_CONFIG.get("default_timezone", "UTC") or "UTC")
_DEFAULT_RISK_LEVEL = str(ALIGNMENT_HELPERS_CONFIG.get("default_risk_level", "medium") or "medium").strip().lower()
_DEFAULT_SEVERITY = str(ALIGNMENT_HELPERS_CONFIG.get("default_severity", "medium") or "medium").strip().lower()
_DEFAULT_NAMESPACE = str(ALIGNMENT_HELPERS_CONFIG.get("default_namespace", "alignment") or "alignment").strip()
_MAX_TAGS = int(ALIGNMENT_HELPERS_CONFIG.get("max_tags", 32) or 32)
_DEFAULT_REDACTION = str(ALIGNMENT_HELPERS_CONFIG.get("redaction_replacement", "***REDACTED***") or "***REDACTED***")
_DEFAULT_HASH_ALGORITHM = str(ALIGNMENT_HELPERS_CONFIG.get("hash_algorithm", "sha256") or "sha256").strip().lower()

_ALLOWED_RISK_LEVELS = tuple(
    str(level).strip().lower()
    for level in ALIGNMENT_HELPERS_CONFIG.get("risk_levels", ["low", "medium", "high", "critical"])
    if str(level).strip()
)
_ALLOWED_SEVERITIES = tuple(
    str(level).strip().lower()
    for level in ALIGNMENT_HELPERS_CONFIG.get("severity_levels", ["low", "medium", "high", "critical"])
    if str(level).strip()
)
_TRUE_VALUES = {
    str(value).strip().lower()
    for value in ALIGNMENT_HELPERS_CONFIG.get(
        "true_values",
        ["true", "1", "yes", "y", "on", "enabled", "approve", "approved"],
    )
}
_FALSE_VALUES = {
    str(value).strip().lower()
    for value in ALIGNMENT_HELPERS_CONFIG.get(
        "false_values",
        ["false", "0", "no", "n", "off", "disabled", "reject", "rejected"],
    )
}
_SENSITIVE_KEYS = {
    str(key).strip().lower()
    for key in ALIGNMENT_HELPERS_CONFIG.get(
        "sensitive_keys",
        [
            "password",
            "passwd",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "api_key",
            "api-key",
            "authorization",
            "cookie",
            "set-cookie",
            "private_key",
            "certificate",
            "webhook_url",
            "smtp_password",
            "username",
            "email",
            "reviewer_id",
            "operator_id",
            "human_input",
            "pii",
            "sensitive_attributes",
        ],
    )
    if str(key).strip()
}
_IDENTIFIER_PREFIXES = {
    "event": "align_evt",
    "audit": "align_audit",
    "trace": "align_trace",
    "context": "align_ctx",
    "snapshot": "align_snap",
    "intervention": "align_int",
    "memory": "align_mem",
    "drift": "align_drift",
}
_IDENTIFIER_PREFIXES.update(
    {
        str(key).strip().lower(): str(value).strip()
        for key, value in dict(ALIGNMENT_HELPERS_CONFIG.get("identifier_prefixes", {})).items()
        if str(key).strip() and str(value).strip()
    }
)
_SECURE_SCHEMES = {
    str(value).strip().lower()
    for value in ALIGNMENT_HELPERS_CONFIG.get(
        "secure_schemes",
        ["https", "wss", "grpcs", "smtps", "ssl", "tls"],
    )
    if str(value).strip()
}
_DEFAULT_PORTS = {
    str(key).strip().lower(): int(value)
    for key, value in dict(
        ALIGNMENT_HELPERS_CONFIG.get(
            "default_ports",
            {
                "http": 80,
                "https": 443,
                "ws": 80,
                "wss": 443,
                "smtp": 25,
                "smtps": 465,
                "grpc": 50051,
                "grpcs": 443,
            },
        )
    ).items()
}


@dataclass(frozen=True, slots=True)
class ParsedAlignmentEndpoint:
    """
    Normalized endpoint representation for alignment-domain integrations.

    The helper intentionally keeps endpoint parsing lightweight and generic.
    Human oversight, audit export, telemetry forwarding, and artifact emission
    can reuse a common representation without pulling routing or transport
    policy into the alignment subsystem.
    """

    raw: str
    scheme: str
    host: str
    port: Optional[int]
    path: str = ""
    query: str = ""
    fragment: str = ""
    username: Optional[str] = None
    password: Optional[str] = None

    @property
    def secure(self) -> bool:
        return is_secure_scheme(self.scheme)

    @property
    def authority(self) -> str:
        credentials = ""
        if self.username:
            credentials = self.username
            if self.password:
                credentials += ":***"
            credentials += "@"
        if self.port is not None:
            return f"{credentials}{self.host}:{self.port}"
        return f"{credentials}{self.host}"

    @property
    def netloc(self) -> str:
        credentials = ""
        if self.username:
            credentials = self.username
            if self.password:
                credentials += f":{self.password}"
            credentials += "@"
        if self.port is not None:
            return f"{credentials}{self.host}:{self.port}"
        return f"{credentials}{self.host}"

    @property
    def normalized(self) -> str:
        return urlunparse(
            (
                self.scheme,
                self.netloc,
                self.path or "",
                "",
                self.query or "",
                self.fragment or "",
            )
        )

    def to_dict(self, *, redact_credentials: bool = True) -> Dict[str, Any]:
        netloc = self.authority if redact_credentials else self.netloc
        normalized = urlunparse(
            (
                self.scheme,
                netloc,
                self.path or "",
                "",
                self.query or "",
                self.fragment or "",
            )
        )
        return {
            "raw": self.raw,
            "scheme": self.scheme,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "query": self.query,
            "fragment": self.fragment,
            "username": self.username,
            "secure": self.secure,
            "normalized": normalized,
        }

def utcnow() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return utcnow().isoformat()


def ensure_timezone_aware(value: datetime, *, field_name: str = "datetime") -> datetime:
    """Ensure a datetime is timezone-aware, assuming UTC when tzinfo is absent."""
    if not isinstance(value, datetime):
        raise ValidationError(
            message=f"'{field_name}' must be a datetime instance.",
            context={"field": field_name, "actual_type": type(value).__name__},
        )
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_datetime(value: Any, *, field_name: str = "datetime", allow_none: bool = False) -> Optional[datetime]:
    """
    Parse common datetime-like representations into a timezone-aware UTC datetime.

    Supported input forms:
    - datetime
    - date
    - ISO-8601 datetime strings
    - POSIX timestamps (int/float)
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(
            message=f"'{field_name}' must not be None.",
            context={"field": field_name},
        )

    try:
        if isinstance(value, datetime):
            return ensure_timezone_aware(value, field_name=field_name)
        if isinstance(value, date):
            return datetime.combine(value, time.min, tzinfo=timezone.utc)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValidationError(
                    message=f"'{field_name}' must not be empty.",
                    context={"field": field_name},
                )
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            return ensure_timezone_aware(parsed, field_name=field_name)
    except AlignmentError:
        raise
    except Exception as exc:
        raise ValidationError(
            message=f"Failed to parse '{field_name}' as a datetime.",
            context={"field": field_name, "value": repr(value)},
            cause=exc,
        ) from exc

    raise ValidationError(
        message=f"Unsupported datetime representation for '{field_name}'.",
        context={"field": field_name, "actual_type": type(value).__name__},
    )


def normalize_identifier_prefix(prefix: Optional[str], *, fallback: str = "align") -> str:
    """Normalize an identifier prefix into a conservative alphanumeric token."""
    if prefix is None:
        prefix = fallback
    text = ensure_non_empty_string(prefix, "prefix", error_cls=ValidationError).lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def generate_identifier(prefix: str = "align") -> str:
    """Generate a collision-resistant identifier with a stable normalized prefix."""
    normalized_prefix = normalize_identifier_prefix(prefix)
    return f"{normalized_prefix}_{uuid.uuid4().hex}"


def generate_event_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or _IDENTIFIER_PREFIXES.get("event", "align_evt"))


def generate_audit_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or _IDENTIFIER_PREFIXES.get("audit", "align_audit"))


def generate_trace_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or _IDENTIFIER_PREFIXES.get("trace", "align_trace"))


def generate_context_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or _IDENTIFIER_PREFIXES.get("context", "align_ctx"))


def generate_snapshot_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or _IDENTIFIER_PREFIXES.get("snapshot", "align_snap"))


def generate_intervention_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or _IDENTIFIER_PREFIXES.get("intervention", "align_int"))


def generate_memory_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or _IDENTIFIER_PREFIXES.get("memory", "align_mem"))


def generate_drift_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or _IDENTIFIER_PREFIXES.get("drift", "align_drift"))


def json_safe(value: Any, *, max_depth: int = _MAX_JSON_DEPTH, _depth: int = 0) -> Any:
    """
    Convert arbitrary values into a JSON-safe representation.

    The result is suitable for telemetry, audit logs, shared-memory storage,
    redaction pipelines, intervention payloads, and error contexts.
    """
    if _depth >= max_depth:
        return repr(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, timedelta):
        return value.total_seconds()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode(_DEFAULT_ENCODING)
        except UnicodeDecodeError:
            return {
                "encoding": "base64",
                "length": len(value),
                "data": base64.b64encode(value).decode("ascii"),
            }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item, max_depth=max_depth, _depth=_depth + 1)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item, max_depth=max_depth, _depth=_depth + 1) for item in value]

    if isinstance(value, Exception):
        if isinstance(value, AlignmentError) and hasattr(value, "to_dict"):
            return value.to_dict()
        return {"type": type(value).__name__, "message": str(value)}

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): json_safe(item, max_depth=max_depth, _depth=_depth + 1)
                for key, item in vars(value).items()
            }
        except Exception:
            pass

    return repr(value)


def stable_json_dumps(
    value: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    separators: Tuple[str, str] = (",", ":"),
) -> str:
    """Serialize a value deterministically for hashing, caching, and telemetry."""
    try:
        return json.dumps(
            json_safe(value),
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            separators=separators,
        )
    except Exception as exc:
        raise wrap_alignment_exception(
            exc,
            target_cls=ValidationError,
            message="Failed to serialize value to stable JSON.",
            context={"operation": "stable_json_dumps"},
            metadata={"value_type": type(value).__name__},
        ) from exc


def stable_json_loads(value: Union[str, bytes, bytearray]) -> Any:
    """Load JSON using the subsystem's default encoding and normalized error handling."""
    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode(_DEFAULT_ENCODING)
        return json.loads(value)
    except Exception as exc:
        raise wrap_alignment_exception(
            exc,
            target_cls=ValidationError,
            message="Failed to deserialize stable JSON payload.",
            context={"operation": "stable_json_loads"},
            metadata={"value_type": type(value).__name__},
        ) from exc


def merge_mappings(
    *mappings: Optional[Mapping[str, Any]],
    deep: bool = True,
    drop_none: bool = False,
) -> Dict[str, Any]:
    """
    Merge multiple mappings into a new dictionary.

    Later mappings win. Nested mappings are merged recursively when deep=True.
    """
    merged: Dict[str, Any] = {}
    for candidate in mappings:
        if candidate is None:
            continue
        current = ensure_mapping(
            candidate,
            field_name="mapping",
            allow_empty=True,
            error_cls=ValidationError,
        )
        for key, value in current.items():
            if drop_none and value is None:
                continue
            if (
                deep
                and key in merged
                and isinstance(merged[key], Mapping)
                and isinstance(value, Mapping)
            ):
                merged[key] = merge_mappings(merged[key], value, deep=True, drop_none=drop_none)
            else:
                merged[key] = value
    return merged


def normalize_metadata(
    metadata: Optional[Mapping[str, Any]],
    *,
    drop_none: bool = True,
) -> Dict[str, Any]:
    """Normalize metadata for telemetry, memory storage, and reports."""
    if metadata is None:
        return {}
    source = ensure_mapping(metadata, field_name="metadata", allow_empty=True, error_cls=ValidationError)
    normalized: Dict[str, Any] = {}
    for key, value in source.items():
        if drop_none and value is None:
            continue
        normalized[str(key)] = json_safe(value)
    return normalized


def normalize_tags(tags: Optional[Sequence[Any]], *, max_items: int = _MAX_TAGS) -> Tuple[str, ...]:
    """Normalize tags while preserving order and removing duplicates."""
    if tags is None:
        return ()
    values = ensure_sequence(
        tags,
        "tags",
        allow_empty=True,
        error_cls=ValidationError,
        allow_strings=False,
    )
    deduplicated: List[str] = []
    seen = set()
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        if text not in seen:
            deduplicated.append(text)
            seen.add(text)
        if len(deduplicated) >= max_items:
            break
    return tuple(deduplicated)


def normalize_metric_name(metric_name: Optional[str], *, namespace: Optional[str] = None) -> str:
    """
    Normalize metric names into a stable snake_case identifier.

    Examples:
    - "Statistical Parity" -> "statistical_parity"
    - "group/fairness.parity" -> "group_fairness_parity"
    """
    text = ensure_non_empty_string(metric_name, "metric_name", error_cls=ValidationError).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if namespace:
        ns = normalize_identifier_prefix(namespace, fallback=_DEFAULT_NAMESPACE)
        if not text.startswith(f"{ns}_"):
            text = f"{ns}_{text}"
    return text


def normalize_risk_level(level: Optional[str], *, field_name: str = "risk_level") -> str:
    """Normalize risk labels into the configured risk vocabulary."""
    if level is None:
        return _DEFAULT_RISK_LEVEL
    normalized = ensure_non_empty_string(level, field_name, error_cls=ValidationError).strip().lower()
    if normalized not in _ALLOWED_RISK_LEVELS:
        raise ValidationError(
            message=f"'{field_name}' must be one of {_ALLOWED_RISK_LEVELS}.",
            context={"field": field_name, "value": normalized},
        )
    return normalized


def normalize_severity(level: Optional[str], *, field_name: str = "severity") -> str:
    """Normalize severity labels into the configured severity vocabulary."""
    if level is None:
        return _DEFAULT_SEVERITY
    normalized = ensure_non_empty_string(level, field_name, error_cls=ValidationError).strip().lower()
    if normalized not in _ALLOWED_SEVERITIES:
        raise ValidationError(
            message=f"'{field_name}' must be one of {_ALLOWED_SEVERITIES}.",
            context={"field": field_name, "value": normalized},
        )
    return normalized


def normalize_sensitive_attributes(
    attributes: Optional[Sequence[Any]],
    *,
    lowercase: bool = False,
    allow_empty: bool = True,
) -> Tuple[str, ...]:
    """Normalize sensitive attribute names while preserving order."""
    if attributes is None:
        return tuple()
    values = ensure_sequence(
        attributes,
        "sensitive_attributes",
        allow_empty=allow_empty,
        error_cls=DataValidationError,
        allow_strings=False,
    )
    normalized: List[str] = []
    seen = set()
    for item in values:
        text = ensure_non_empty_string(str(item), "sensitive_attribute", error_cls=DataValidationError)
        if lowercase:
            text = text.lower()
        if text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


def normalize_weight_mapping(
    weights: Optional[Mapping[str, Any]],
    *,
    drop_none: bool = True,
    normalize_sum: bool = False,
    allow_negative: bool = False,
) -> Dict[str, float]:
    """Normalize weight mappings used by risk assemblers and evaluators."""
    if weights is None:
        return {}
    source = ensure_mapping(weights, field_name="weights", allow_empty=True, error_cls=ValidationError)
    normalized: Dict[str, float] = {}
    for key, value in source.items():
        if value is None and drop_none:
            continue
        numeric = coerce_float(value, field_name=f"weight:{key}")
        if not allow_negative and numeric < 0:
            raise ValidationError(
                message="Weight values must not be negative.",
                context={"field": f"weight:{key}", "value": numeric},
            )
        normalized[str(key)] = numeric

    if normalize_sum and normalized:
        total = sum(normalized.values())
        if total <= 0:
            raise ValidationError(
                message="Normalized weight mapping requires a positive total.",
                context={"field": "weights", "total": total},
            )
        normalized = {key: value / total for key, value in normalized.items()}
    return normalized


def coerce_bool(value: Any, *, field_name: str = "value") -> bool:
    """Coerce common scalar forms into a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 0.0):
            return False
        if value in (1, 1.0):
            return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValidationError(
        message=f"'{field_name}' could not be coerced to a boolean.",
        context={"field": field_name, "value": repr(value)},
    )


def coerce_int(
    value: Any,
    *,
    field_name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Coerce scalar values into an integer with optional bounds."""
    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not a valid integer value")
        numeric = int(float(value))
    except Exception as exc:
        raise ValidationError(
            message=f"'{field_name}' must be coercible to an integer.",
            context={"field": field_name, "value": repr(value)},
            cause=exc,
        ) from exc

    if minimum is not None:
        ensure_numeric_range(numeric, field_name, min_value=minimum, error_cls=ValidationError)
    if maximum is not None:
        ensure_numeric_range(numeric, field_name, max_value=maximum, error_cls=ValidationError)
    return numeric


def coerce_float(
    value: Any,
    *,
    field_name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Coerce scalar values into a float with optional bounds."""
    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not a valid float value")
        numeric = float(value)
    except Exception as exc:
        raise ValidationError(
            message=f"'{field_name}' must be coercible to a float.",
            context={"field": field_name, "value": repr(value)},
            cause=exc,
        ) from exc

    if minimum is not None:
        ensure_numeric_range(numeric, field_name, min_value=minimum, error_cls=ValidationError)
    if maximum is not None:
        ensure_numeric_range(numeric, field_name, max_value=maximum, error_cls=ValidationError)
    return numeric


def coerce_probability(value: Any, *, field_name: str = "probability") -> float:
    """Coerce a scalar to a probability in the inclusive range [0, 1]."""
    return coerce_float(value, field_name=field_name, minimum=0.0, maximum=1.0)


def coerce_score(
    value: Any,
    *,
    field_name: str = "score",
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Coerce a bounded alignment score or metric value."""
    return coerce_float(value, field_name=field_name, minimum=minimum, maximum=maximum)


def coerce_positive_int(value: Any, *, field_name: str = "value") -> int:
    """Coerce a scalar into a strictly positive integer."""
    return coerce_int(value, field_name=field_name, minimum=1)


def coerce_window_size(value: Any, *, field_name: str = "window_size") -> int:
    """Coerce a scalar into a valid rolling-window size."""
    return coerce_positive_int(value, field_name=field_name)


def stable_hash(
    value: Any,
    *,
    algorithm: Optional[str] = None,
    namespace: Optional[str] = None,
    encoding: str = _DEFAULT_ENCODING,
) -> str:
    """Create a deterministic digest from JSON-stable data."""
    resolved_algorithm = (algorithm or _DEFAULT_HASH_ALGORITHM or "sha256").strip().lower()
    try:
        hasher = hashlib.new(resolved_algorithm)
    except Exception as exc:
        raise ConfigurationError(
            message="Unsupported hash algorithm configured for alignment helpers.",
            context={"operation": "stable_hash", "algorithm": resolved_algorithm},
            cause=exc,
        ) from exc

    payload = {"namespace": namespace or _DEFAULT_NAMESPACE, "value": json_safe(value)}
    hasher.update(stable_json_dumps(payload).encode(encoding))
    return hasher.hexdigest()


def stable_context_hash(context: Optional[Mapping[str, Any]], *, namespace: str = "context") -> str:
    """Create a deterministic hash for alignment contexts."""
    normalized = normalize_context(context, drop_none=False)
    return stable_hash(normalized, namespace=namespace)


def stable_record_fingerprint(record: Any, *, namespace: str = "record") -> str:
    """Create a stable fingerprint for event, report, or checkpoint records."""
    return stable_hash(record, namespace=namespace)


def normalize_context(
    context: Optional[Mapping[str, Any]],
    *,
    drop_none: bool = False,
    drop_empty_mappings: bool = False,
) -> Dict[str, Any]:
    """
    Normalize contextual data used by audit, telemetry, and memory layers.

    The helper recursively converts the structure into a JSON-safe, key-stable,
    log-friendly mapping without introducing domain-policy decisions.
    """
    if context is None:
        return {}
    source = ensure_mapping(context, field_name="context", allow_empty=True, error_cls=ValidationError)
    normalized: Dict[str, Any] = {}
    for key, value in source.items():
        if value is None and drop_none:
            continue
        safe_value = json_safe(value)
        if drop_empty_mappings and isinstance(safe_value, dict) and not safe_value:
            continue
        normalized[str(key)] = safe_value
    return normalized


def build_alignment_event(
    event_type: str,
    *,
    event_id: Optional[str] = None,
    timestamp: Optional[Union[str, datetime, date, int, float]] = None,
    severity: Optional[str] = None,
    risk_level: Optional[str] = None,
    source: Optional[str] = None,
    trace_id: Optional[str] = None,
    tags: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    payload: Any = None,
) -> Dict[str, Any]:
    """
    Build a normalized alignment event envelope.

    This is intentionally generic so the same event shape can be reused by
    memory, monitoring, intervention reporting, and oversight modules.
    """
    normalized_context = normalize_context(context, drop_none=False)
    normalized_metadata = normalize_metadata(metadata, drop_none=True)
    occurred_at = parse_datetime(timestamp, field_name="timestamp", allow_none=True)
    return {
        "event_id": event_id or generate_event_id(),
        "event_type": normalize_metric_name(event_type, namespace=None),
        "timestamp": (occurred_at or utcnow()).isoformat(),
        "severity": normalize_severity(severity),
        "risk_level": normalize_risk_level(risk_level),
        "source": source.strip() if isinstance(source, str) and source.strip() else None,
        "trace_id": trace_id or generate_trace_id(),
        "tags": list(normalize_tags(tags)),
        "metadata": normalized_metadata,
        "context": normalized_context,
        "context_hash": stable_context_hash(normalized_context),
        "payload": json_safe(payload),
        "fingerprint": stable_record_fingerprint(
            {
                "event_type": event_type,
                "severity": severity,
                "risk_level": risk_level,
                "source": source,
                "tags": normalize_tags(tags),
                "metadata": normalized_metadata,
                "context": normalized_context,
                "payload": json_safe(payload),
            },
            namespace="alignment_event",
        ),
    }


def redact_sensitive_value(
    value: Any,
    *,
    replacement: str = _DEFAULT_REDACTION,
    preserve_length: bool = False,
) -> Any:
    """Redact sensitive scalar values for logs, telemetry, and reports."""
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return sanitize_for_logging(value, replacement=replacement, preserve_length=preserve_length)
    if isinstance(value, bytes):
        if preserve_length:
            return f"{replacement}[bytes:{len(value)}]"
        return replacement
    if preserve_length:
        return f"{replacement}[len:{len(str(value))}]"
    return replacement


def redact_sensitive_mapping(
    mapping: Optional[Mapping[str, Any]],
    *,
    replacement: str = _DEFAULT_REDACTION,
    preserve_length: bool = False,
) -> Dict[str, Any]:
    """Redact configured secret-bearing and privacy-bearing keys in a mapping."""
    if mapping is None:
        return {}
    source = ensure_mapping(mapping, field_name="mapping", allow_empty=True, error_cls=ValidationError)
    redacted: Dict[str, Any] = {}
    for key, value in source.items():
        key_text = str(key)
        if key_text.strip().lower() in _SENSITIVE_KEYS:
            redacted[key_text] = redact_sensitive_value(value, replacement=replacement, preserve_length=preserve_length)
        elif isinstance(value, Mapping):
            redacted[key_text] = redact_sensitive_mapping(
                value,
                replacement=replacement,
                preserve_length=preserve_length,
            )
        elif isinstance(value, (list, tuple)):
            normalized_items = []
            for item in value:
                if isinstance(item, Mapping):
                    normalized_items.append(
                        redact_sensitive_mapping(
                            item,
                            replacement=replacement,
                            preserve_length=preserve_length,
                        )
                    )
                else:
                    normalized_items.append(item)
            redacted[key_text] = normalized_items
        else:
            redacted[key_text] = value
    return redacted


def sanitize_for_logging(
    value: Any,
    *,
    replacement: str = _DEFAULT_REDACTION,
    preserve_length: bool = False,
    max_depth: int = _MAX_JSON_DEPTH,
    _depth: int = 0,
) -> Any:
    """
    Produce a log-safe representation of arbitrary values.

    This combines JSON-safe normalization with recursive secret redaction.
    """
    if _depth >= max_depth:
        return repr(value)

    if isinstance(value, Mapping):
        return redact_sensitive_mapping(
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

    if isinstance(value, bytes):
        return redact_sensitive_value(value, replacement=replacement, preserve_length=preserve_length)

    if isinstance(value, AlignmentError):
        return sanitize_for_logging(
            value.to_dict(),
            replacement=replacement,
            preserve_length=preserve_length,
            max_depth=max_depth,
            _depth=_depth + 1,
        )

    return json_safe(value, max_depth=max_depth, _depth=_depth)


def is_url_like(value: Optional[str]) -> bool:
    """Return True when a string looks like a URL or URI with a scheme."""
    if not value or not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)



def is_secure_scheme(scheme: Optional[str]) -> bool:
    """Return True when an endpoint scheme is configured as secure."""
    if not scheme:
        return False
    normalized = str(scheme).strip().lower()
    return normalized in _SECURE_SCHEMES or normalized.endswith("+tls") or normalized.endswith("+ssl")


def default_port_for_scheme(scheme: Optional[str]) -> Optional[int]:
    """Resolve a configured default port for a scheme."""
    if not scheme:
        return _DEFAULT_PORTS.get(_DEFAULT_SCHEME)
    return _DEFAULT_PORTS.get(str(scheme).strip().lower())


def parse_endpoint(
    endpoint: str,
    *,
    default_scheme: Optional[str] = None,
    require_host: bool = True,
    default_port: Optional[int] = None,
) -> ParsedAlignmentEndpoint:
    """
    Parse and normalize an endpoint string into a structured representation.

    Supported forms include:
    - example.org
    - example.org:8443
    - https://example.org/api/intervention
    - smtp://mail.example.org:587
    """
    try:
        endpoint_text = ensure_non_empty_string(endpoint, "endpoint", error_cls=ValidationError)
        endpoint_to_parse = endpoint_text
        if "://" not in endpoint_text:
            scheme = default_scheme or _DEFAULT_SCHEME
            endpoint_to_parse = f"{scheme}://{endpoint_text}"

        parsed = urlparse(endpoint_to_parse)
        scheme = (parsed.scheme or default_scheme or _DEFAULT_SCHEME).strip().lower()
        host = parsed.hostname or ""
        if require_host and not host:
            raise ValidationError(
                message="Endpoint must include a host.",
                context={"operation": "parse_endpoint", "endpoint": endpoint_text},
            )

        resolved_port = parsed.port or default_port or default_port_for_scheme(scheme)
        return ParsedAlignmentEndpoint(
            raw=endpoint_text,
            scheme=scheme,
            host=host,
            port=resolved_port,
            path=parsed.path or "",
            query=parsed.query or "",
            fragment=parsed.fragment or "",
            username=parsed.username,
            password=parsed.password,
        )
    except AlignmentError:
        raise
    except Exception as exc:
        raise wrap_alignment_exception(
            exc,
            target_cls=ValidationError,
            message="Failed to parse endpoint.",
            context={"operation": "parse_endpoint", "endpoint": endpoint},
        ) from exc


def normalize_endpoint(
    endpoint: Union[str, ParsedAlignmentEndpoint],
    *,
    default_scheme: Optional[str] = None,
    require_host: bool = True,
    default_port: Optional[int] = None,
    normalize_query: bool = True,
) -> str:
    """Return a normalized endpoint string with explicit scheme and stable query ordering."""
    if isinstance(endpoint, ParsedAlignmentEndpoint):
        parsed = endpoint
    else:
        parsed = parse_endpoint(
            endpoint,
            default_scheme=default_scheme,
            require_host=require_host,
            default_port=default_port,
        )

    query = parsed.query or ""
    if normalize_query and query:
        query = urlencode(sorted(parse_qsl(query, keep_blank_values=True)), doseq=True)

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "",
            "",
            query,
            parsed.fragment or "",
        )
    )


def build_endpoint(
    *,
    scheme: str,
    host: str,
    port: Optional[int] = None,
    path: str = "",
    query: Optional[Mapping[str, Any] | str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    fragment: str = "",
) -> str:
    """Construct a normalized endpoint string from explicit components."""
    normalized_scheme = ensure_non_empty_string(scheme, "scheme", error_cls=ValidationError).lower()
    normalized_host = ensure_non_empty_string(host, "host", error_cls=ValidationError)
    resolved_port = int(port) if port is not None else default_port_for_scheme(normalized_scheme)

    normalized_path = path or ""
    if normalized_path and not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"

    if isinstance(query, Mapping):
        items: List[Tuple[str, str]] = []
        for key, value in query.items():
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    items.append((str(key), "" if item is None else str(item)))
            else:
                items.append((str(key), "" if value is None else str(value)))
        query_string = urlencode(items, doseq=True)
    elif query is None:
        query_string = ""
    else:
        query_string = str(query)

    credentials = ""
    if username:
        credentials = username
        if password:
            credentials += f":{password}"
        credentials += "@"

    netloc = f"{credentials}{normalized_host}"
    if resolved_port is not None:
        netloc += f":{resolved_port}"

    return urlunparse(
        (
            normalized_scheme,
            netloc,
            normalized_path,
            "",
            query_string,
            fragment or "",
        )
    )


def normalize_resource_reference(
    reference: Union[str, Path],
    *,
    resolve_path: bool = False,
) -> str:
    """
    Normalize a resource reference for templates, checkpoints, and artifact paths.

    - URL-like inputs are normalized as endpoints.
    - Filesystem-like inputs are normalized as POSIX-like paths.
    """
    if isinstance(reference, Path):
        path = reference.expanduser()
        if resolve_path:
            path = path.resolve()
        return str(path)

    text = ensure_non_empty_string(reference, "reference", error_cls=ValidationError)
    if is_url_like(text):
        return normalize_endpoint(text)
    path = Path(text).expanduser()
    if resolve_path:
        path = path.resolve()
    return str(path)


def normalize_threshold_mapping(
    thresholds: Optional[Mapping[str, Any]],
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Dict[str, float]:
    """Normalize numeric threshold mappings into a clean float dictionary."""
    if thresholds is None:
        return {}
    source = ensure_mapping(thresholds, field_name="thresholds", allow_empty=True, error_cls=ValidationError)
    normalized: Dict[str, float] = {}
    for key, value in source.items():
        normalized[str(key)] = coerce_float(
            value,
            field_name=f"threshold:{key}",
            minimum=minimum,
            maximum=maximum,
        )
    return normalized


def flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    parent_key: str = "",
    separator: str = ".",
) -> Dict[str, Any]:
    """Flatten a nested mapping into a single-level dot-separated dictionary."""
    source = ensure_mapping(mapping, field_name="mapping", allow_empty=True, error_cls=ValidationError)
    flattened: Dict[str, Any] = {}
    for key, value in source.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, parent_key=new_key, separator=separator))
        else:
            flattened[new_key] = value
    return flattened


def unflatten_mapping(flat_mapping: Mapping[str, Any], *, separator: str = ".") -> Dict[str, Any]:
    """Expand a dot-separated flat mapping into a nested dictionary."""
    source = ensure_mapping(flat_mapping, field_name="flat_mapping", allow_empty=True, error_cls=ValidationError)
    expanded: Dict[str, Any] = {}
    for flat_key, value in source.items():
        parts = [part for part in str(flat_key).split(separator) if part]
        if not parts:
            continue
        cursor: MutableMapping[str, Any] = expanded
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], MutableMapping):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return expanded


if __name__ == "__main__":
    print("\n=== Running Alignment Helpers ===\n")

    sample_context = {
        "domain": "finance",
        "operator_id": "reviewer_123",
        "decision": {"approved": True, "score": 0.81},
        "auth": {"token": "secret-token-value"},
        "sensitive_attributes": ["gender", "age_group"],
    }
    sample_endpoint = "oversight.example.org:8443/interventions?priority=high&channel=dashboard"

    print("Normalized metric:", normalize_metric_name("Statistical Parity", namespace="alignment"))
    print("Audit ID:", generate_audit_id())
    print("Context hash:", stable_context_hash(sample_context))
    print("Normalized endpoint:", normalize_endpoint(sample_endpoint, default_scheme="https"))
    print("Secure endpoint:", parse_endpoint(sample_endpoint, default_scheme="https").secure)
    print("Normalized sensitive attrs:", normalize_sensitive_attributes(["gender", "age_group", "gender"]))
    print("Normalized weights:", normalize_weight_mapping({"fairness": 0.4, "ethics": 0.3, "safety": 0.3}, normalize_sum=True))
    print("Sanitized context:", sanitize_for_logging(sample_context, preserve_length=True))
    print(
        "Alignment event:\n",
        stable_json_dumps(
            build_alignment_event(
                "human_intervention_requested",
                severity="high",
                risk_level="critical",
                source="alignment_agent",
                tags=["oversight", "critical", "oversight"],
                metadata={"delivery": {"channels": ["dashboard", "email"]}},
                context=sample_context,
                payload={"report_id": generate_snapshot_id(), "endpoint": normalize_endpoint(sample_endpoint, default_scheme="https")},
            )
        ),
    )
    print("Flattened mapping:", flatten_mapping({"a": {"b": {"c": 1}}, "x": 2}))
    print("Unflattened mapping:", unflatten_mapping({"a.b.c": 1, "x": 2}))
    print("\n=== Alignment Helpers Test Completed ===\n")
