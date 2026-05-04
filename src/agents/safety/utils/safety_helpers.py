"""
Shared helpers for the safety subsystem.

This module centralizes the safety helper logic. It exists to keep the
subsystem and telemetry code focused on their core concerns instead of
repeatedly re-implementing parsing, normalization, serialization, redaction,
endpoint handling, identifier generation, and config-backed utility behaviors.

The helpers here are intentionally scoped to reusable safety-domain utilities.
They do not own fairness evaluation, bias detection, intervention policy,
memory persistence strategy, causal reasoning, or human-oversight workflow
state transitions. Instead, they provide the stable primitives those modules
depend on.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import ipaddress
import json
import math
import re
import secrets
import unicodedata

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar, Union
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlparse, urlunparse

from .config_loader import load_global_config, get_config_section
from .security_error import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Safety Helpers")
printer = PrettyPrinter()

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

MODULE_VERSION = "2.1.0"
HELPER_SCHEMA_VERSION = "safety_helpers.v3"

DEFAULT_MAX_TEXT_LENGTH = 4096
DEFAULT_MAX_COLLECTION_ITEMS = 250
DEFAULT_MAX_RECURSION_DEPTH = 8
DEFAULT_HASH_PREFIX_LENGTH = 16
DEFAULT_IDENTIFIER_PREFIX = "safe"

DEFAULT_SAFETY_HELPER_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "environment": "production",
    "schema_version": HELPER_SCHEMA_VERSION,
    "timezone": "UTC",
    "hash_algorithm": "sha256",
    "hash_salt": "",
    "fingerprint_length": DEFAULT_HASH_PREFIX_LENGTH,
    "identifier_prefix": DEFAULT_IDENTIFIER_PREFIX,
    "correlation_id_prefix": "corr",
    "trace_id_prefix": "trace",
    "request_id_prefix": "req",
    "max_text_length": DEFAULT_MAX_TEXT_LENGTH,
    "max_collection_items": DEFAULT_MAX_COLLECTION_ITEMS,
    "max_recursion_depth": DEFAULT_MAX_RECURSION_DEPTH,
    "redaction_marker": "[REDACTED]",
    "sensitive_key_patterns": [
        r"(?i)(?:^|[_\-.])(api[_\-.]?key|token|secret|password|passwd|pwd|credential|authorization|auth|cookie|session|private[_\-.]?key|access[_\-.]?key|refresh[_\-.]?token|client[_\-.]?secret)(?:$|[_\-.])",
        r"(?i)(?:bearer|oauth|jwt|csrf|xsrf)",
    ],
    "identifier_key_patterns": [
        r"(?i)(?:^|[_\-.])(user[_\-.]?id|subject[_\-.]?id|account[_\-.]?id|actor[_\-.]?id|tenant[_\-.]?id|session[_\-.]?id|request[_\-.]?id|trace[_\-.]?id|correlation[_\-.]?id|email|phone|ip|source[_\-.]?ip)(?:$|[_\-.])",
    ],
    "redaction_patterns": [
        {"name": "authorization_header", "pattern": r"(?i)authorization\s*:\s*(?:basic|bearer)\s+[A-Za-z0-9._~+/=-]+"},
        {"name": "bearer_token", "pattern": r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"},
        {"name": "api_key_assignment", "pattern": r"(?i)\b(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"]?[^\s,'\"]+"},
        {"name": "private_key", "pattern": r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----", "flags": ["DOTALL"]},
        {"name": "email", "pattern": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "flags": ["IGNORECASE"]},
        {"name": "phone", "pattern": r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}\b"},
        {"name": "payment_card", "pattern": r"\b(?:\d[ -]*?){13,19}\b"},
        {"name": "ipv4", "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"},
        {"name": "long_hex_token", "pattern": r"\b[0-9a-f]{32,128}\b", "flags": ["IGNORECASE"]},
    ],
    "redaction_patterns":[
        {"name": "ssn", "pattern": r"\b\d{3}-\d{2}-\d{4}\b"},
    ],
    "url": {
        "allowed_schemes": ["http", "https"],
        "redact_query_keys": ["token", "access_token", "refresh_token", "api_key", "apikey", "secret", "password", "passwd", "session", "code", "auth", "signature", "sig"],
        "max_url_length": 2048,
        "strip_fragment": True,
        "lowercase_scheme_host": True,
    },
    "risk": {
        "low": 0.25,
        "medium": 0.50,
        "high": 0.75,
        "critical": 0.90,
        "block_threshold": 0.75,
        "review_threshold": 0.50,
    },
    "logging": {
        "redact_by_default": True,
        "include_fingerprints": True,
        "max_exception_message_length": 1024,
    },
}

_COMPILED_REDACTION_PATTERNS: Optional[Tuple[Tuple[str, re.Pattern[str]], ...]] = None
_COMPILED_SENSITIVE_KEY_RE: Optional[re.Pattern[str]] = None
_COMPILED_IDENTIFIER_KEY_RE: Optional[re.Pattern[str]] = None


@dataclass(frozen=True)
class HelperOperationResult:
    """Small structured result object for helper self-tests and utility pipelines."""

    ok: bool
    operation: str
    details: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if not data["timestamp"]:
            data["timestamp"] = utc_iso()
        return data


@dataclass(frozen=True)
class SanitizedURL:
    """Parsed URL bundle that keeps raw URL handling out of logs and telemetry."""

    original_fingerprint: str
    sanitized_url: str
    scheme: str
    hostname: str
    port: Optional[int]
    path: str
    query_keys: Tuple[str, ...]
    is_private_host: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _copy_default_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_SAFETY_HELPER_CONFIG)


def deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""

    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    if not override:
        return merged

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_helper_config(refresh: bool = False) -> Dict[str, Any]:
    """Return the merged safety helper configuration from secure_config.yaml."""

    global _COMPILED_REDACTION_PATTERNS, _COMPILED_SENSITIVE_KEY_RE, _COMPILED_IDENTIFIER_KEY_RE
    if refresh:
        _COMPILED_REDACTION_PATTERNS = None
        _COMPILED_SENSITIVE_KEY_RE = None
        _COMPILED_IDENTIFIER_KEY_RE = None

    configured = get_config_section("safety_helpers") or {}
    return deep_merge(_copy_default_config(), configured)


def get_helper_setting(path: Union[str, Sequence[str]], default: Any = None) -> Any:
    """Read a nested helper setting using dotted path or sequence path syntax."""

    config = get_helper_config()
    keys = path.split(".") if isinstance(path, str) else list(path)
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "disabled"}:
            return False
    return default


def coerce_int(value: Any, default: int = 0, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_float(value: Any, default: float = 0.0, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: Optional[Union[datetime, float, int]] = None) -> str:
    """Return a UTC ISO-8601 timestamp with a Z suffix."""

    if value is None:
        dt = utc_now()
    elif isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    else:
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


# Backwards-friendly aliases used by earlier helper drafts.
_utc_now = utc_now


def _isoformat(ts: Optional[float] = None) -> str:
    return utc_iso(ts)


def parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def seconds_until(value: Union[datetime, float, int]) -> float:
    if isinstance(value, datetime):
        target = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return max(0.0, (target.astimezone(timezone.utc) - utc_now()).total_seconds())
    return max(0.0, float(value) - utc_now().timestamp())


def safe_hash_algorithm(name: Optional[str] = None) -> str:
    algorithm = (name or str(get_helper_setting("hash_algorithm", "sha256"))).lower().strip()
    if algorithm not in hashlib.algorithms_available:
        raise SecurityError(
            SecurityErrorType.CONFIGURATION_TAMPERING,
            f"Unsupported hash algorithm configured for safety helpers: {algorithm}",
            severity=SecuritySeverity.HIGH,
            context={"configured_algorithm": algorithm, "component": "safety_helpers"},
            remediation_guidance=(
                "Use a hashlib-supported algorithm such as sha256, sha3_256, or blake2s.",
                "Review secure_config.yaml for accidental or unauthorized changes.",
            ),
            component="safety_helpers",
        )
    return algorithm


def hash_bytes(data: bytes, algorithm: Optional[str] = None, salt: Union[str, bytes, None] = None) -> str:
    algorithm = safe_hash_algorithm(algorithm)
    configured_salt = get_helper_setting("hash_salt", "")
    effective_salt = salt if salt is not None else configured_salt
    if effective_salt:
        salt_bytes = effective_salt.encode("utf-8") if isinstance(effective_salt, str) else effective_salt
        return hmac.new(salt_bytes, data, algorithm).hexdigest()
    return hashlib.new(algorithm, data).hexdigest()


_hash_bytes = hash_bytes


def to_jsonable(value: Any, *, max_depth: Optional[int] = None, _depth: int = 0) -> Any:
    """Convert Python objects into deterministic JSON-compatible structures."""

    max_depth = coerce_int(max_depth, coerce_int(get_helper_setting("max_recursion_depth"), DEFAULT_MAX_RECURSION_DEPTH), minimum=1)
    if _depth > max_depth:
        return "[MAX_DEPTH_EXCEEDED]"

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value), max_depth=max_depth, _depth=_depth + 1)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v, max_depth=max_depth, _depth=_depth + 1) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item, max_depth=max_depth, _depth=_depth + 1) for item in list(value)]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"encoding": "base64", "sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return safe_repr(value)


def stable_json(data: Any, *, ensure_ascii: bool = False) -> str:
    return json.dumps(to_jsonable(data), sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii, default=str)


_stable_json = stable_json


def hash_text(text: str, algorithm: Optional[str] = None, salt: Union[str, bytes, None] = None) -> str:
    return hash_bytes(str(text).encode("utf-8", errors="replace"), algorithm=algorithm, salt=salt)


def fingerprint(value: Any, algorithm: Optional[str] = None, salt: Union[str, bytes, None] = None, length: Optional[int] = None) -> str:
    length = coerce_int(length, coerce_int(get_helper_setting("fingerprint_length"), DEFAULT_HASH_PREFIX_LENGTH), minimum=8, maximum=64)
    return hash_text(stable_json(value), algorithm=algorithm, salt=salt)[:length]


_fingerprint = fingerprint


def constant_time_equals(left: Union[str, bytes], right: Union[str, bytes]) -> bool:
    left_bytes = left.encode("utf-8") if isinstance(left, str) else left
    right_bytes = right.encode("utf-8") if isinstance(right, str) else right
    return hmac.compare_digest(left_bytes, right_bytes)


def generate_identifier(prefix: Optional[str] = None, *, entropy_bytes: int = 16) -> str:
    prefix = normalize_identifier(prefix or str(get_helper_setting("identifier_prefix", DEFAULT_IDENTIFIER_PREFIX)))
    token = secrets.token_urlsafe(max(8, entropy_bytes)).rstrip("=")
    return f"{prefix}_{token}"


def generate_correlation_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or str(get_helper_setting("correlation_id_prefix", "corr")))


def generate_trace_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or str(get_helper_setting("trace_id_prefix", "trace")), entropy_bytes=20)


def generate_request_id(prefix: Optional[str] = None) -> str:
    return generate_identifier(prefix or str(get_helper_setting("request_id_prefix", "req")))


def normalize_unicode(text: Any, *, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC") -> str:
    return unicodedata.normalize(form, "" if text is None else str(text))


def strip_control_chars(text: str, *, allow_newlines: bool = True) -> str:
    if allow_newlines:
        return "".join(ch for ch in text if ch in "\n\r\t" or not unicodedata.category(ch).startswith("C"))
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("C"))


def normalize_whitespace(text: str, *, preserve_newlines: bool = False) -> str:
    if preserve_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: Any, limit: Optional[int] = None, *, marker: str = "...[truncated:{remaining}]") -> str:
    text = "" if text is None else str(text)
    limit = coerce_int(limit, coerce_int(get_helper_setting("max_text_length"), DEFAULT_MAX_TEXT_LENGTH), minimum=0)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    suffix = marker.format(remaining=len(text) - limit)
    keep = max(0, limit - len(suffix))
    return f"{text[:keep]}{suffix}"


_truncate = truncate_text


def normalize_text(text: Any, *, max_length: Optional[int] = None, preserve_newlines: bool = False, lowercase: bool = False) -> str:
    normalized = normalize_unicode(text)
    normalized = strip_control_chars(normalized, allow_newlines=preserve_newlines)
    normalized = normalize_whitespace(normalized, preserve_newlines=preserve_newlines)
    if lowercase:
        normalized = normalized.lower()
    return truncate_text(normalized, max_length)


def safe_repr(value: Any, *, max_length: Optional[int] = None) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:
        rendered = f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"
    return truncate_text(rendered, max_length)


def normalize_identifier(value: Any, *, max_length: int = 64, default: str = "id") -> str:
    text = normalize_text(value, max_length=max_length, lowercase=True)
    text = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("._-")
    return text[:max_length] or default


def _compile_regex_flags(flag_names: Optional[Sequence[str]]) -> int:
    flags = 0
    for raw_name in flag_names or []:
        name = str(raw_name).upper()
        flags |= getattr(re, name, 0)
    return flags


def get_redaction_patterns(refresh: bool = False) -> Tuple[Tuple[str, re.Pattern[str]], ...]:
    global _COMPILED_REDACTION_PATTERNS
    if _COMPILED_REDACTION_PATTERNS is not None and not refresh:
        return _COMPILED_REDACTION_PATTERNS

    config = get_helper_config(refresh=refresh)
    compiled: List[Tuple[str, re.Pattern[str]]] = []
    for item in config.get("redaction_patterns", []):
        if isinstance(item, Mapping):
            name = str(item.get("name", "pattern"))
            pattern = str(item.get("pattern", ""))
            flags = _compile_regex_flags(item.get("flags", []))
        else:
            name = "pattern"
            pattern = str(item)
            flags = 0
        if not pattern:
            continue
        compiled.append((name, re.compile(pattern, flags)))
    _COMPILED_REDACTION_PATTERNS = tuple(compiled)
    return _COMPILED_REDACTION_PATTERNS


def _strip_inline_global_flags(pattern: str) -> str:
    """Remove leading inline global flags before combining configured regexes."""

    return re.sub(r"^\(\?[aiLmsux-]+\)", "", pattern)


def get_sensitive_key_regex(refresh: bool = False) -> re.Pattern[str]:
    global _COMPILED_SENSITIVE_KEY_RE
    if _COMPILED_SENSITIVE_KEY_RE is not None and not refresh:
        return _COMPILED_SENSITIVE_KEY_RE
    patterns = [_strip_inline_global_flags(str(p)) for p in get_helper_setting("sensitive_key_patterns", [])]
    _COMPILED_SENSITIVE_KEY_RE = re.compile("|".join(f"(?:{p})" for p in patterns) or r"a^", re.IGNORECASE)
    return _COMPILED_SENSITIVE_KEY_RE


def get_identifier_key_regex(refresh: bool = False) -> re.Pattern[str]:
    global _COMPILED_IDENTIFIER_KEY_RE
    if _COMPILED_IDENTIFIER_KEY_RE is not None and not refresh:
        return _COMPILED_IDENTIFIER_KEY_RE
    patterns = [_strip_inline_global_flags(str(p)) for p in get_helper_setting("identifier_key_patterns", [])]
    _COMPILED_IDENTIFIER_KEY_RE = re.compile("|".join(f"(?:{p})" for p in patterns) or r"a^", re.IGNORECASE)
    return _COMPILED_IDENTIFIER_KEY_RE


SENSITIVE_KEY_RE = get_sensitive_key_regex()
IDENTIFIER_KEY_RE = get_identifier_key_regex()
REDACTION_PATTERNS = get_redaction_patterns()


def redact_text(text: Any, *, max_length: Optional[int] = None, include_fingerprint: Optional[bool] = None) -> str:
    """Redact common secrets and PII while preserving safe correlation fingerprints."""

    config = get_helper_config()
    max_length = coerce_int(max_length, coerce_int(config.get("max_text_length"), DEFAULT_MAX_TEXT_LENGTH), minimum=0)
    include_fingerprint = coerce_bool(include_fingerprint, coerce_bool(get_nested(config, "logging.include_fingerprints", True)))
    marker = str(config.get("redaction_marker", "[REDACTED]"))
    hash_salt = str(config.get("hash_salt", ""))
    redacted = normalize_unicode(text)

    for label, pattern in get_redaction_patterns():
        def replace(match: re.Match[str]) -> str:
            if include_fingerprint:
                return f"{marker[:-1] if marker.endswith(']') else marker}:{label}:{fingerprint(match.group(0), salt=hash_salt)}]"
            return marker

        redacted = pattern.sub(replace, redacted)

    return truncate_text(redacted, max_length)


_redact_string = redact_text


def redact_value(
    value: Any,
    *,
    key_hint: str = "",
    depth: int = 0,
    max_depth: Optional[int] = None,
    max_text_length: Optional[int] = None,
    max_items: Optional[int] = None,
    hash_salt: Optional[str] = None,
    mask_identifiers: bool = True,
) -> Any:
    """Recursively redact sensitive fields while preserving useful structure."""

    config = get_helper_config()
    max_depth = coerce_int(max_depth, coerce_int(config.get("max_recursion_depth"), DEFAULT_MAX_RECURSION_DEPTH), minimum=1)
    max_text_length = coerce_int(max_text_length, coerce_int(config.get("max_text_length"), DEFAULT_MAX_TEXT_LENGTH), minimum=0)
    max_items = coerce_int(max_items, coerce_int(config.get("max_collection_items"), DEFAULT_MAX_COLLECTION_ITEMS), minimum=1)
    hash_salt = str(config.get("hash_salt", "")) if hash_salt is None else hash_salt

    if depth > max_depth:
        return "[REDACTED:max_depth]"

    if key_hint and get_sensitive_key_regex().search(key_hint):
        return f"[REDACTED:{normalize_identifier(key_hint)}:{fingerprint(value, salt=hash_salt)}]"

    if mask_identifiers and key_hint and get_identifier_key_regex().search(key_hint) and value not in (None, ""):
        return f"[IDENTIFIER:{normalize_identifier(key_hint)}:{fingerprint(value, salt=hash_salt)}]"

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return redact_text(value, max_length=max_text_length)
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value), "sha256": hashlib.sha256(value).hexdigest()[:16]}
    if is_dataclass(value) and not isinstance(value, type):
        return redact_value(asdict(value), key_hint=key_hint, depth=depth + 1, max_depth=max_depth, max_text_length=max_text_length, max_items=max_items, hash_salt=hash_salt, mask_identifiers=mask_identifiers)
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        items = list(value.items())
        for idx, (k, v) in enumerate(items[:max_items]):
            key = str(k)
            result[key] = redact_value(v, key_hint=key, depth=depth + 1, max_depth=max_depth, max_text_length=max_text_length, max_items=max_items, hash_salt=hash_salt, mask_identifiers=mask_identifiers)
        if len(items) > max_items:
            result["[truncated_items]"] = len(items) - max_items
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        sequence = list(value)
        redacted_list: List[Any] = [
            redact_value(item, key_hint=key_hint, depth=depth + 1, max_depth=max_depth, max_text_length=max_text_length, max_items=max_items, hash_salt=hash_salt, mask_identifiers=mask_identifiers)
            for item in sequence[:max_items]
        ]
        if len(sequence) > max_items:
            redacted_list.append({"[truncated_items]": len(sequence) - max_items})
        return redacted_list
    return redact_text(safe_repr(value), max_length=max_text_length)


_redact_value = redact_value


def sanitize_for_logging(payload: Any, *, max_text_length: Optional[int] = None) -> Any:
    return redact_value(payload, max_text_length=max_text_length, mask_identifiers=True)


def safe_log_payload(event: str, payload: Optional[Mapping[str, Any]] = None, **fields: Any) -> Dict[str, Any]:
    combined = dict(payload or {})
    combined.update(fields)
    return {
        "schema_version": HELPER_SCHEMA_VERSION,
        "event": normalize_identifier(event, max_length=128, default="event"),
        "timestamp": utc_iso(),
        "payload": sanitize_for_logging(combined),
    }


def parse_json_object(raw: Union[str, bytes, Mapping[str, Any]], *, context: str = "json_payload") -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        decoded = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise SecurityError(
            SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
            f"Invalid JSON supplied to safety helper parser for {context}.",
            severity=SecuritySeverity.MEDIUM,
            context={"context": context, "error": str(exc), "sample": str(raw)[:256]},
            remediation_guidance=(
                "Reject malformed JSON before downstream safety analysis.",
                "Validate producer serialization and schema expectations.",
            ),
            component="safety_helpers",
            cause=exc,
        ) from exc
    if not isinstance(parsed, Mapping):
        raise SecurityError(
            SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
            f"Expected a JSON object for {context}, got {type(parsed).__name__}.",
            severity=SecuritySeverity.MEDIUM,
            context={"context": context, "parsed_type": type(parsed).__name__},
            component="safety_helpers",
        )
    return dict(parsed)


def get_nested(mapping: Mapping[str, Any], path: Union[str, Sequence[str]], default: Any = None, *, separator: str = ".") -> Any:
    keys = path.split(separator) if isinstance(path, str) else list(path)
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def set_nested(mapping: MutableMapping[str, Any], path: Union[str, Sequence[str]], value: Any, *, separator: str = ".") -> MutableMapping[str, Any]:
    keys = path.split(separator) if isinstance(path, str) else list(path)
    if not keys:
        raise ValueError("path must contain at least one key")
    current: MutableMapping[str, Any] = mapping
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value
    return mapping


def flatten_mapping(mapping: Mapping[str, Any], *, parent_key: str = "", separator: str = ".") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in mapping.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, parent_key=new_key, separator=separator))
        else:
            flattened[new_key] = value
    return flattened


def require_keys(mapping: Mapping[str, Any], required_keys: Sequence[str], *, context: str = "mapping") -> None:
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise SecurityError(
            SecurityErrorType.CONFIGURATION_TAMPERING,
            f"Required keys missing from {context}.",
            severity=SecuritySeverity.HIGH,
            context={"context": context, "missing_keys": missing, "available_keys": list(mapping.keys())},
            remediation_guidance=(
                "Restore the expected configuration or schema fields.",
                "Block dependent safety checks until the required fields are available.",
            ),
            component="safety_helpers",
        )


def dedupe_preserve_order(values: Iterable[T]) -> List[T]:
    seen: set = set()
    result: List[T] = []
    for value in values:
        marker = stable_json(value) if isinstance(value, (Mapping, list, tuple, set, frozenset)) else value
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def chunked(values: Sequence[T], size: int) -> Iterator[List[T]]:
    size = max(1, int(size))
    for index in range(0, len(values), size):
        yield list(values[index:index + size])


def safe_b64encode(data: Union[str, bytes]) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def safe_b64decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def normalize_url(raw_url: str, *, default_scheme: str = "https") -> str:
    text = normalize_text(raw_url, max_length=coerce_int(get_helper_setting("url.max_url_length"), 2048))
    if not text:
        return ""
    parsed = urlparse(text if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text) else f"{default_scheme}://{text}")
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{host}{port}"
    path = quote(parsed.path or "/", safe="/%:@")
    return urlunparse((scheme, netloc, path, "", parsed.query, parsed.fragment))


def _is_private_hostname(hostname: str) -> bool:
    if not hostname:
        return False
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
    except ValueError:
        lowered = hostname.lower().strip(".")
        return lowered in {"localhost"} or lowered.endswith(".local") or lowered.endswith(".internal")


def sanitize_url(raw_url: str, *, allowed_schemes: Optional[Sequence[str]] = None) -> SanitizedURL:
    config = get_helper_config()
    url_config = config.get("url", {})
    max_url_length = coerce_int(url_config.get("max_url_length"), 2048, minimum=64)
    allowed = {scheme.lower() for scheme in (allowed_schemes or url_config.get("allowed_schemes", ["http", "https"]))}
    normalized = normalize_url(raw_url)
    parsed = urlparse(normalized)

    if parsed.scheme.lower() not in allowed:
        raise SecurityError(
            SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
            f"URL scheme is not allowed for safety helper endpoint handling: {parsed.scheme}",
            severity=SecuritySeverity.HIGH,
            context={"scheme": parsed.scheme, "allowed_schemes": sorted(allowed), "url_fingerprint": fingerprint(raw_url)},
            remediation_guidance=(
                "Reject or sanitize URLs before use in network, retrieval, or tool-execution contexts.",
                "Review allowed_schemes in secure_config.yaml if this is an expected endpoint type.",
            ),
            component="safety_helpers",
        )

    redact_keys = {str(key).lower() for key in url_config.get("redact_query_keys", [])}
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    sanitized_pairs = [
        (key, "[REDACTED]" if key.lower() in redact_keys or get_sensitive_key_regex().search(key) else value)
        for key, value in query_pairs
    ]
    sanitized_query = urlencode(sanitized_pairs, doseq=True)
    fragment = "" if coerce_bool(url_config.get("strip_fragment"), True) else parsed.fragment
    sanitized = urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", "", sanitized_query, fragment))
    sanitized = truncate_text(sanitized, max_url_length)
    hostname = parsed.hostname or ""
    return SanitizedURL(
        original_fingerprint=fingerprint(raw_url),
        sanitized_url=sanitized,
        scheme=parsed.scheme.lower(),
        hostname=hostname.lower(),
        port=parsed.port,
        path=parsed.path or "/",
        query_keys=tuple(key for key, _ in query_pairs),
        is_private_host=_is_private_hostname(hostname),
    )


def normalize_endpoint(base_url: str, *path_parts: str, query: Optional[Mapping[str, Any]] = None) -> str:
    base = sanitize_url(base_url).sanitized_url
    parsed = urlparse(base)
    clean_parts = [quote(str(part).strip("/"), safe="") for part in path_parts if str(part).strip("/")]
    path = "/".join([parsed.path.strip("/"), *clean_parts]).strip("/")
    query_string = urlencode(query or {}, doseq=True)
    return sanitize_url(urlunparse((parsed.scheme, parsed.netloc, f"/{path}" if path else "/", "", query_string, ""))).sanitized_url


def extract_domain(raw_url: str) -> str:
    return sanitize_url(raw_url).hostname


def clamp_score(value: Any, *, default: float = 0.0) -> float:
    return coerce_float(value, default, minimum=0.0, maximum=1.0)


def weighted_average(scores: Mapping[str, Any], weights: Optional[Mapping[str, Any]] = None, *, default: float = 0.0) -> float:
    if not scores:
        return clamp_score(default)
    weights = weights or {key: 1.0 for key in scores}
    total_weight = 0.0
    weighted_sum = 0.0
    for key, raw_score in scores.items():
        weight = max(0.0, coerce_float(weights.get(key, 0.0), 0.0))
        weighted_sum += clamp_score(raw_score) * weight
        total_weight += weight
    return clamp_score(weighted_sum / total_weight if total_weight else default)


def combine_risk_scores(*scores: Any, method: str = "noisy_or") -> float:
    normalized = [clamp_score(score) for score in scores if score is not None]
    if not normalized:
        return 0.0
    if method == "max":
        return max(normalized)
    if method == "average":
        return sum(normalized) / len(normalized)
    if method == "weighted_high":
        return clamp_score((max(normalized) * 0.7) + ((sum(normalized) / len(normalized)) * 0.3))
    product = 1.0
    for score in normalized:
        product *= (1.0 - score)
    return clamp_score(1.0 - product)


def categorize_risk(score: Any) -> str:
    thresholds = get_helper_setting("risk", {})
    score_value = clamp_score(score)
    if score_value >= clamp_score(thresholds.get("critical", 0.90)):
        return "critical"
    if score_value >= clamp_score(thresholds.get("high", 0.75)):
        return "high"
    if score_value >= clamp_score(thresholds.get("medium", 0.50)):
        return "medium"
    if score_value >= clamp_score(thresholds.get("low", 0.25)):
        return "low"
    return "minimal"


def threshold_decision(score: Any, *, block_threshold: Optional[float] = None, review_threshold: Optional[float] = None) -> str:
    risk_config = get_helper_setting("risk", {})
    score_value = clamp_score(score)
    block_threshold = clamp_score(block_threshold if block_threshold is not None else risk_config.get("block_threshold", 0.75))
    review_threshold = clamp_score(review_threshold if review_threshold is not None else risk_config.get("review_threshold", 0.50))
    if score_value >= block_threshold:
        return "block"
    if score_value >= review_threshold:
        return "review"
    return "allow"


def make_incident_context(
    *,
    operation: str,
    component: str = "safety_helpers",
    payload: Optional[Mapping[str, Any]] = None,
    **fields: Any,
) -> Dict[str, Any]:
    context = {
        "schema_version": HELPER_SCHEMA_VERSION,
        "operation": normalize_identifier(operation, max_length=128, default="operation"),
        "component": component,
        "timestamp": utc_iso(),
        "correlation_id": fields.pop("correlation_id", generate_correlation_id()),
    }
    if payload:
        context["payload"] = sanitize_for_logging(payload)
    if fields:
        context["fields"] = sanitize_for_logging(fields)
    return context


def wrap_security_exception(
    exc: BaseException,
    *,
    operation: str,
    component: str = "safety_helpers",
    context: Optional[Mapping[str, Any]] = None,
    error_type: Any = None,
    severity: Any = None,
) -> SecurityError:
    if isinstance(exc, SecurityError):
        return exc
    selected_error_type = error_type or SecurityErrorType.UNKNOWN_SECURITY_ERROR
    selected_severity = severity or SecuritySeverity.HIGH
    if hasattr(SecurityError, "from_exception"):
        return SecurityError.from_exception(
            exc,
            error_type=selected_error_type,
            message=f"Unhandled exception during helper operation: {operation}",
            context=make_incident_context(operation=operation, component=component, payload=context or {}),
            component=component,
            severity=selected_severity,
        )
    return SecurityError(
        selected_error_type,
        f"Unhandled exception during helper operation: {operation}",
        severity=selected_severity,
        context=make_incident_context(operation=operation, component=component, payload=context or {}, wrapped_exception=type(exc).__name__),
        component=component,
    )


def exception_to_safe_dict(exc: BaseException, *, operation: str = "unknown", component: str = "safety_helpers") -> Dict[str, Any]:
    if isinstance(exc, SecurityError):
        return exc.to_public_response() if hasattr(exc, "to_public_response") else {"message": str(exc), "type": type(exc).__name__}
    wrapped = wrap_security_exception(exc, operation=operation, component=component)
    return wrapped.to_public_response() if hasattr(wrapped, "to_public_response") else {"message": str(wrapped), "type": type(wrapped).__name__}


def assert_safe_condition(condition: bool, message: str, *, context: Optional[Mapping[str, Any]] = None, error_type: Any = None, severity: Any = None) -> None:
    if condition:
        return
    raise SecurityError(
        error_type or SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
        message,
        severity=severity or SecuritySeverity.HIGH,
        context=make_incident_context(operation="assert_safe_condition", payload=context or {}),
        component="safety_helpers",
        remediation_guidance=(
            "Fail closed and block the unsafe operation.",
            "Review the failing precondition and add a regression test.",
        ),
    )


def load_text_file(path: Union[str, Path], *, max_bytes: Optional[int] = None, encoding: str = "utf-8") -> str:
    file_path = Path(path)
    max_bytes = coerce_int(max_bytes, coerce_int(get_helper_setting("max_file_bytes", 1_048_576), minimum=1024))
    if not file_path.exists() or not file_path.is_file():
        raise SecurityError(
            SecurityErrorType.ACCESS_VIOLATION,
            "Requested helper file does not exist or is not a file.",
            severity=SecuritySeverity.MEDIUM,
            context={"path": str(file_path)},
            component="safety_helpers",
        )
    size = file_path.stat().st_size
    if size > max_bytes:
        raise ResourceExhaustionError(
            resource_type="file_read_bytes",
            current_usage=float(size),
            limit=float(max_bytes),
            source_identifier=str(file_path),
            component="safety_helpers",
        )
    return file_path.read_text(encoding=encoding)


def validate_config_shape(config: Optional[Mapping[str, Any]] = None) -> HelperOperationResult:
    config = dict(config or get_helper_config(refresh=True))
    required = ["hash_algorithm", "fingerprint_length", "max_text_length", "redaction_patterns", "url", "risk"]
    require_keys(config, required, context="safety_helpers")
    safe_hash_algorithm(str(config.get("hash_algorithm")))
    sanitize_url("https://example.com/path?token=secret")
    return HelperOperationResult(True, "validate_config_shape", {"required_keys": required, "config_fingerprint": fingerprint(config)}, timestamp=utc_iso())


__all__ = [
    "DEFAULT_MAX_TEXT_LENGTH",
    "DEFAULT_MAX_COLLECTION_ITEMS",
    "DEFAULT_MAX_RECURSION_DEPTH",
    "DEFAULT_HASH_PREFIX_LENGTH",
    "DEFAULT_SAFETY_HELPER_CONFIG",
    "HELPER_SCHEMA_VERSION",
    "MODULE_VERSION",
    "SENSITIVE_KEY_RE",
    "IDENTIFIER_KEY_RE",
    "REDACTION_PATTERNS",
    "HelperOperationResult",
    "SanitizedURL",
    "get_helper_config",
    "get_helper_setting",
    "coerce_bool",
    "coerce_int",
    "coerce_float",
    "utc_now",
    "utc_iso",
    "parse_iso_datetime",
    "seconds_until",
    "safe_hash_algorithm",
    "hash_bytes",
    "hash_text",
    "fingerprint",
    "constant_time_equals",
    "generate_identifier",
    "generate_correlation_id",
    "generate_trace_id",
    "generate_request_id",
    "to_jsonable",
    "stable_json",
    "normalize_unicode",
    "strip_control_chars",
    "normalize_whitespace",
    "truncate_text",
    "normalize_text",
    "safe_repr",
    "normalize_identifier",
    "get_redaction_patterns",
    "get_sensitive_key_regex",
    "get_identifier_key_regex",
    "redact_text",
    "redact_value",
    "sanitize_for_logging",
    "safe_log_payload",
    "parse_json_object",
    "deep_merge",
    "get_nested",
    "set_nested",
    "flatten_mapping",
    "require_keys",
    "dedupe_preserve_order",
    "chunked",
    "safe_b64encode",
    "safe_b64decode",
    "normalize_url",
    "sanitize_url",
    "normalize_endpoint",
    "extract_domain",
    "clamp_score",
    "weighted_average",
    "combine_risk_scores",
    "categorize_risk",
    "threshold_decision",
    "make_incident_context",
    "wrap_security_exception",
    "exception_to_safe_dict",
    "assert_safe_condition",
    "load_text_file",
    "validate_config_shape",
]


if __name__ == "__main__":
    print("\n=== Running Safety Helpers ===\n")
    printer.status("TEST", "Safety Helpers initialized", "info")

    config = get_helper_config(refresh=True)
    printer.status("TEST", f"Loaded helper config schema: {config.get('schema_version')}", "info")

    validation = validate_config_shape()
    printer.status("TEST", f"Config validation fingerprint: {validation.details['config_fingerprint']}", "info")

    raw_text = "  User email alice@example.com used token=secret-token-123 from 192.168.1.10  "
    normalized_text = normalize_text(raw_text)
    redacted_text = redact_text(normalized_text)
    assert "alice@example.com" not in redacted_text
    assert "secret-token-123" not in redacted_text
    printer.status("TEST", f"Redaction sample: {redacted_text}", "info")

    nested_payload = {
        "user_id": "user-123",
        "profile": {"email": "alice@example.com", "api_key": "sk-live-secret", "risk": 0.82},
        "items": list(range(5)),
    }
    sanitized_payload = sanitize_for_logging(nested_payload)
    assert "sk-live-secret" not in stable_json(sanitized_payload)
    assert "alice@example.com" not in stable_json(sanitized_payload)
    printer.status("TEST", f"Sanitized payload: {stable_json(sanitized_payload)}", "info")

    merged = deep_merge({"a": {"b": 1}, "x": 1}, {"a": {"c": 2}})
    assert get_nested(merged, "a.b") == 1
    assert get_nested(merged, "a.c") == 2
    set_nested(merged, "a.d", 3)
    assert get_nested(merged, "a.d") == 3

    encoded = safe_b64encode("safety-test")
    assert safe_b64decode(encoded).decode("utf-8") == "safety-test"

    safe_url = sanitize_url("https://Example.com/login?token=abc123&next=/home#secret")
    assert "abc123" not in safe_url.sanitized_url
    assert safe_url.scheme == "https"
    printer.status("TEST", f"Sanitized URL: {safe_url.sanitized_url}", "info")

    endpoint = normalize_endpoint("https://api.example.com", "v1", "safety checks", query={"request_id": "req-1"})
    assert endpoint.startswith("https://api.example.com/")
    printer.status("TEST", f"Endpoint: {endpoint}", "info")

    combined_risk = combine_risk_scores(0.2, 0.7, 0.4)
    assert categorize_risk(combined_risk) in {"medium", "high", "critical"}
    assert threshold_decision(0.8) == "block"
    printer.status("TEST", f"Risk decision: {categorize_risk(combined_risk)} / {threshold_decision(combined_risk)}", "info")

    try:
        parse_json_object("{bad json", context="self_test")
    except SecurityError as exc:
        public = exception_to_safe_dict(exc, operation="parse_json_object_self_test")
        assert public.get("message")
        printer.status("TEST", f"SecurityError handled safely: {public.get('error_id', 'no-id')}", "info")

    try:
        assert_safe_condition(False, "Self-test unsafe condition", context={"password": "not-for-logs"})
    except SecurityError as exc:
        audit = exc.to_audit_format() if hasattr(exc, "to_audit_format") else {"message": str(exc)}
        assert "not-for-logs" not in stable_json(audit)
        printer.status("TEST", "Audit-safe SecurityError generated", "info")

    result = HelperOperationResult(
        ok=True,
        operation="safety_helpers_self_test",
        details={
            "module_version": MODULE_VERSION,
            "schema_version": HELPER_SCHEMA_VERSION,
            "correlation_id": generate_correlation_id(),
            "trace_id": generate_trace_id(),
            "request_id": generate_request_id(),
        },
        timestamp=utc_iso(),
    )
    printer.status("TEST", f"Result: {stable_json(result.to_dict())}", "info")

    print("\n=== Test ran successfully ===\n")
