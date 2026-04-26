"""
Production-grade shared helpers for the browser subsystem.

This module centralizes the stable, reusable primitives that browser-facing
agents and browser function modules need when navigating, searching, clicking,
typing, scrolling, extracting content, recording telemetry, and normalizing
workflow execution results.

Scope
-----
The helpers here are intentionally browser-domain utilities. They do not own
browser lifecycle, planning, concrete Selenium actions, content extraction, or
workflow orchestration. Instead, they provide consistent primitives for those
modules so the codebase can grow without duplicating:

- URL parsing, normalization, validation, and origin comparison.
- Selector normalization and safe selector construction.
- Safe Selenium driver and element inspection.
- Page, element, and search-result snapshots for logs and downstream agents.
- Result payload construction for success/error/action outcomes.
- JSON-safe serialization, truncation, fingerprinting, and redaction.
- Retry/backoff timing utilities that can be shared by agent modules.
- Workflow normalization helpers aligned to supported browser actions.
- Query/link scoring primitives for browsing relevance decisions.
- Lightweight content helpers for text/html/url classification.

Design principles
-----------------
1. Deterministic contracts: helpers return stable dictionaries/dataclasses and
   avoid raising where a safe fallback is better for telemetry.
2. Safe error paths: serializer/redactor/snapshot helpers must not crash just
   because a WebDriver object is stale, detached, or partially unavailable.
3. Browser-focused: every helper should serve browsing automation, extraction,
   logging, or orchestration directly.
4. Expandable: helper names are intentionally granular so future modules can use
   a stable primitive instead of modifying large monolithic helpers.
5. Dependency-tolerant: Selenium and the project logger/config/error modules are
   optional at import time so this module remains importable in test, lint, and
   offline contexts.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
import random
import re
import time as time_module
import uuid

from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    JavascriptException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar, Union, Type
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlparse, urlunparse

from .config_loader import load_global_config, get_config_section
from .browser_errors import * # type: ignore
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Browser Helpers")
printer = PrettyPrinter()

_R = TypeVar('_R')
T = TypeVar("T", bound="BrowserURL")
JsonMapping = Mapping[str, Any]
MutableJsonMapping = MutableMapping[str, Any]
Predicate = Callable[[Any], bool]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HTTP_SCHEMES = {"http", "https"}
SAFE_RESOURCE_SCHEMES = {"http", "https", "file", "data", "about", "chrome", "edge", "brave"}
TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "utm_name",
    "utm_reader",
    "utm_viz_id",
    "utm_pubreferrer",
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref",
    "ref_src",
    "spm",
    "yclid",
}

SENSITIVE_KEY_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in (
        r"password", r"passwd", r"pwd", r"secret", r"token", r"api[_-]?key",
        r"access[_-]?key", r"auth", r"authorization", r"credential", r"session",
        r"cookie", r"csrf", r"xsrf", r"private[_-]?key", r"client[_-]?secret",
        r"refresh[_-]?token", r"clipboard"
    )
)

SENSITIVE_KEY_PATTERNS = tuple(SENSITIVE_KEY_PATTERNS)

# Common selectors that are useful for BrowserAgent.search and consent handling.
DEFAULT_SEARCH_BOX_SELECTORS: Tuple[str, ...] = (
    "input[name='q']",
    "textarea[name='q']",
    "input[type='search']",
    "input[type='text']",
    "[role='searchbox']",
    "[aria-label='Search']",
    "[aria-label='Cerca']",
    "[aria-label='Pesquisar']",
    "[aria-label='Buscar']",
    "[aria-label='Recherche']",
    "[aria-label='Suche']",
    ".gLFyf",
)

SEARCH_RESULT_LINK_SELECTORS: Tuple[str, ...] = (
    "h3 a",
    ".yuRUbf a",
    ".g a",
    "a[href^='http']",
)

CONSENT_BUTTON_SELECTORS: Tuple[str, ...] = (
    "button[aria-label='Accept all']",
    "button[aria-label='Accept']",
    "button[aria-label='I agree']",
    "button[aria-label='Consent']",
    "button[id='L2AGLb']",
    "#L2AGLb",
    "form button[jsname]",
    "button[type='submit']",
)

CAPTCHA_INDICATORS: Tuple[str, ...] = (
    "captcha",
    "robot check",
    "verify you are human",
    "are you a robot",
    "unusual traffic",
    "automated queries",
    "security check",
    "human verification",
    "cf-challenge",
    "hcaptcha",
    "recaptcha",
)

BROWSER_ACTIONS: Tuple[str, ...] = (
    "navigate",
    "search",
    "click",
    "type",
    "scroll",
    "copy",
    "cut",
    "paste",
    "extract",
)

TEXT_INPUT_TAGS = {"input", "textarea"}
INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "option", "summary", "details"}
MAX_CONTEXT_TEXT = 2_000
MAX_HTML_CONTEXT = 4_000
MAX_URL_LENGTH = 8_192

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetryPolicy:
    """Reusable retry policy for browser operations.

    The policy is intentionally simple and deterministic enough for telemetry,
    while still supporting jitter to avoid synchronized retries when multiple
    browser agents run at once.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: Tuple[type, ...] = field(default_factory=lambda: (TimeoutException, WebDriverException))

    def delay_for_attempt(self, attempt_index: int) -> float:
        return calculate_backoff_delay(
            attempt_index=attempt_index,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            multiplier=self.multiplier,
            jitter=self.jitter,
        )


@dataclass(frozen=True)
class BrowserURL:
    """Normalized URL representation for browser operations."""

    original: str
    normalized: str
    scheme: str
    host: str
    port: Optional[int]
    path: str
    query: str
    fragment: str
    is_http: bool
    origin: str
    resource_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ElementSnapshot:
    """Safe, serializable element metadata for logs and agent context."""

    tag: Optional[str] = None
    text: str = ""
    accessible_name: Optional[str] = None
    role: Optional[str] = None
    element_id: Optional[str] = None
    name: Optional[str] = None
    href: Optional[str] = None
    value: Optional[str] = None
    placeholder: Optional[str] = None
    aria_label: Optional[str] = None
    classes: Tuple[str, ...] = ()
    outer_html: str = ""
    location: Dict[str, Any] = field(default_factory=dict)
    size: Dict[str, Any] = field(default_factory=dict)
    displayed: Optional[bool] = None
    enabled: Optional[bool] = None
    selected: Optional[bool] = None
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PageSnapshot:
    """Safe, serializable page metadata for reasoning, audit trails, and debugging."""

    url: str = ""
    title: str = ""
    ready_state: Optional[str] = None
    text: str = ""
    html: str = ""
    screenshot_b64: Optional[str] = None
    viewport: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    detected_captcha: bool = False
    content_fingerprint: str = ""
    captured_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResultSnapshot:
    """Normalized search result/link candidate snapshot."""

    url: str
    text: str = ""
    title: str = ""
    snippet: str = ""
    rank: Optional[int] = None
    score: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionOutcome:
    """Stable browser action result shape."""

    status: str
    action: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    correlation_id: str = field(default_factory=lambda: new_correlation_id("act"))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return prune_none(payload)

# ---------------------------------------------------------------------------
# Time, identifiers, hashing, truncation
# ---------------------------------------------------------------------------
def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return an ISO-8601 timestamp suitable for logs and result payloads."""

    return utc_now().isoformat()


def monotonic_ms() -> float:
    """Return monotonic time in milliseconds."""

    return time_module.monotonic() * 1000.0


def elapsed_ms(start_ms: float) -> float:
    """Return elapsed monotonic milliseconds from a prior monotonic_ms value."""

    return round(max(0.0, monotonic_ms() - start_ms), 3)


def new_correlation_id(prefix: str = "brw") -> str:
    """Create a compact correlation id for a browser operation."""

    normalized_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(prefix or "brw")).strip("-") or "brw"
    return f"{normalized_prefix}-{uuid.uuid4().hex[:16]}"


def stable_hash(value: Any, *, length: int = 16, algorithm: str = "sha256") -> str:
    """Create a deterministic short hash for any JSON-serializable-ish value."""

    try:
        serialized = safe_json_dumps(value, sort_keys=True)
    except Exception:
        serialized = repr(value)
    digest = hashlib.new(algorithm)
    digest.update(serialized.encode("utf-8", errors="replace"))
    return digest.hexdigest()[: max(1, int(length))]


def fingerprint_text(text: str, *, length: int = 16) -> str:
    """Fingerprint text after whitespace normalization."""

    return stable_hash(normalize_whitespace(text), length=length)


def truncate_text(value: Any, max_length: Optional[int] = MAX_CONTEXT_TEXT, *, suffix: str = "...") -> str:
    """Convert a value to text and truncate safely."""

    if value is None:
        return ""
    text = str(value)
    if max_length is None or max_length < 0 or len(text) <= max_length:
        return text
    suffix = suffix or ""
    return text[: max(0, max_length - len(suffix))] + suffix


def normalize_whitespace(value: Any) -> str:
    """Normalize whitespace in text while preserving word boundaries."""

    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def normalize_newlines(value: Any, *, max_blank_lines: int = 2) -> str:
    """Normalize newline-heavy extracted browser text."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    blank_pattern = r"\n{" + str(max(2, max_blank_lines + 1)) + r",}"
    return re.sub(blank_pattern, "\n" * max_blank_lines, text).strip()


def compact_text(value: Any, *, max_length: int = MAX_CONTEXT_TEXT) -> str:
    """Normalize whitespace and truncate browser-visible text."""

    return truncate_text(normalize_whitespace(value), max_length=max_length)

# ---------------------------------------------------------------------------
# Safe serialization and redaction
# ---------------------------------------------------------------------------
def is_sensitive_key(key: Any) -> bool:
    """Return True when a mapping key is likely to contain sensitive data."""

    key_text = str(key or "")
    return any(pattern.search(key_text) for pattern in SENSITIVE_KEY_PATTERNS)


def redact_scalar(value: Any, *, replacement: str = "[REDACTED]") -> Any:
    """Redact a scalar value while preserving empty/null semantics."""

    if value is None or value == "":
        return value
    return replacement


def redact_mapping(value: Mapping[str, Any], *, replacement: str = "[REDACTED]", max_depth: int = 8) -> Dict[str, Any]:
    """Recursively redact sensitive values in a mapping."""

    return redact_data(value, replacement=replacement, max_depth=max_depth)  # type: ignore[return-value]


def redact_data(value: Any, *, replacement: str = "[REDACTED]", max_depth: int = 8, _depth: int = 0, _key: Any = None) -> Any:
    """Recursively redact sensitive browser telemetry data.

    Redaction is key-aware. Values under keys such as cookies, tokens,
    credentials, clipboard, and authorization headers are replaced while the
    overall payload shape remains intact for debugging.
    """

    if _depth > max_depth:
        return "[MAX_DEPTH]"
    if is_sensitive_key(_key):
        return redact_scalar(value, replacement=replacement)
    if isinstance(value, Mapping):
        return {
            str(k): redact_data(v, replacement=replacement, max_depth=max_depth, _depth=_depth + 1, _key=k)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_data(v, replacement=replacement, max_depth=max_depth, _depth=_depth + 1, _key=_key) for v in value]
    if isinstance(value, str) and looks_like_secret(value):
        return replacement
    return value


def looks_like_secret(value: str) -> bool:
    """Best-effort detection for accidentally included secrets/tokens."""

    if not value or len(value) < 20:
        return False
    tokenish = re.fullmatch(r"[A-Za-z0-9_\-\.~+/=]{20,}", value.strip()) is not None
    jwtish = re.fullmatch(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+", value.strip()) is not None
    return tokenish or jwtish


def safe_serialize(value: Any, *, max_string: int = MAX_CONTEXT_TEXT, max_depth: int = 10, _depth: int = 0) -> Any:
    """Convert arbitrary browser objects into JSON-safe data.

    This is intentionally defensive. WebDriver elements can become stale while
    being logged, exceptions may contain non-serializable attributes, and config
    payloads may include Paths, datetimes, enums, bytes, dataclasses, or sets.
    """

    if _depth > max_depth:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    if isinstance(value, str):
        return truncate_text(value, max_string)
    if isinstance(value, bytes):
        encoded = base64.b64encode(value[: max_string]).decode("ascii", errors="replace")
        return {"encoding": "base64", "truncated": len(value) > max_string, "data": encoded}
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return safe_serialize(asdict(value), max_string=max_string, max_depth=max_depth, _depth=_depth + 1)
    if isinstance(value, Mapping):
        return {
            str(safe_serialize(k, max_string=max_string, max_depth=max_depth, _depth=_depth + 1)): safe_serialize(
                v, max_string=max_string, max_depth=max_depth, _depth=_depth + 1
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [safe_serialize(v, max_string=max_string, max_depth=max_depth, _depth=_depth + 1) for v in value]
    if isinstance(value, BaseException):
        return {
            "type": value.__class__.__name__,
            "message": truncate_text(str(value), max_string),
            "context": safe_serialize(getattr(value, "context", None), max_string=max_string, max_depth=max_depth, _depth=_depth + 1),
            "code": getattr(value, "code", None),
        }
    if is_web_element(value):
        return element_snapshot(value).to_dict()
    return truncate_text(repr(value), max_string)


def safe_json_dumps(value: Any, *, sort_keys: bool = False, indent: Optional[int] = None, redact: bool = False) -> str:
    """Serialize a value to JSON using safe serialization first."""
    payload = safe_serialize(value)
    if redact:
        payload = redact_data(payload)
    return json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys, indent=indent)


def safe_json_loads(value: str, default: Optional[_R] = None) -> Union[Any, _R, None]:
    """Load JSON safely and return default instead of raising."""
    try:
        return json.loads(value)
    except Exception:
        return default


def prune_none(value: Any) -> Any:
    """Remove None values recursively from dictionaries and lists."""
    if isinstance(value, Mapping):
        return {k: prune_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [prune_none(v) for v in value if v is not None]
    if isinstance(value, tuple):
        return tuple(prune_none(v) for v in value if v is not None)
    return value


def merge_dicts(*values: Optional[Mapping[str, Any]], deep: bool = True) -> Dict[str, Any]:
    """Merge dictionaries while preserving nested browser metadata."""
    merged: Dict[str, Any] = {}
    for value in values:
        if not value:
            continue
        for key, item in value.items():
            if deep and isinstance(merged.get(key), dict) and isinstance(item, Mapping):
                merged[key] = merge_dicts(merged[key], item, deep=True)
            else:
                merged[key] = item
    return merged

# ---------------------------------------------------------------------------
# Type coercion and validation helpers
# ---------------------------------------------------------------------------
def ensure_list(value: Any, *, drop_none: bool = True) -> List[Any]:
    """Return value as a list without treating strings as iterables."""
    if value is None:
        return [] if drop_none else [None]
    if isinstance(value, list):
        return [v for v in value if v is not None] if drop_none else value
    if isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, set):
        items = list(value)
    else:
        items = [value]
    return [v for v in items if v is not None] if drop_none else items


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Coerce config/user values into a bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
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
        result = int(value)
    except Exception:
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def coerce_float(value: Any, *, default: float = 0.0, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    """Coerce a value into a float with optional bounds."""
    try:
        result = float(value)
    except Exception:
        result = default
    if math.isnan(result) or math.isinf(result):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def clamp(value: Union[int, float], minimum: Union[int, float], maximum: Union[int, float]) -> Union[int, float]:
    """Clamp a numeric value between bounds."""
    return max(minimum, min(maximum, value))


def require_non_empty_string(value: Any, field_name: str) -> str:
    """Validate and normalize a required string field."""
    text = str(value or "").strip()
    if not text:
        raise make_browser_exception("ValidationError", f"{field_name} must be a non-empty string", context={"field": field_name})
    return text

# ---------------------------------------------------------------------------
# Error construction and result helpers
# ---------------------------------------------------------------------------
def make_browser_exception(class_name: str, message: str, *, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None, **kwargs: Any) -> Exception:
    """Construct a browser exception by class name when available.

    This keeps helpers compatible with both the minimal existing error module and
    a richer production taxonomy. If the requested class does not exist, it falls
    back to BrowserError or RuntimeError.
    """
    cls = globals().get(class_name)
    if isinstance(cls, type) and issubclass(cls, Exception):
        attempts = (
            lambda: cls(message=message, context=context, cause=cause, **kwargs),  # type: ignore[call-arg]
            lambda: cls(message, context=context, cause=cause, **kwargs),          # type: ignore[call-arg]
            lambda: cls(message),                                                   # type: ignore[call-arg]
        )
        for build in attempts:
            try:
                return build()
            except TypeError:
                continue
    base = globals().get("BrowserError")
    if isinstance(base, type) and issubclass(base, Exception):
        for build in (
            lambda: base(message=message, context=context, cause=cause, **kwargs),  # type: ignore[call-arg]
            lambda: base(message, context=context, cause=cause, **kwargs),          # type: ignore[call-arg]
            lambda: base(message),
        ):
            try:
                return build()
            except TypeError:
                continue
    return RuntimeError(message)


def exception_to_error_payload(exc: BaseException, *, action: Optional[str] = None, include_traceback: bool = False) -> Dict[str, Any]:
    """Normalize any exception into the project's dict-result error shape."""
    if hasattr(exc, "to_result"):
        try:
            result = exc.to_result(action=action)  # type: ignore[attr-defined]
            if isinstance(result, dict):
                return redact_mapping(result)
        except Exception:
            pass
    payload = {
        "status": "error",
        "action": action,
        "message": str(exc),
        "error": {
            "type": exc.__class__.__name__,
            "code": getattr(exc, "code", None),
            "retryable": getattr(exc, "retryable", None),
            "severity": getattr(exc, "severity", None),
            "context": getattr(exc, "context", None),
        },
    }
    if include_traceback:
        import traceback

        payload["traceback"] = traceback.format_exc()
    return redact_mapping(prune_none(payload))


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

    payload = ActionOutcome(
        status="success",
        action=action,
        message=message,
        data=dict(data or {}),
        metadata=dict(metadata or {}),
        duration_ms=duration_ms,
        correlation_id=correlation_id or new_correlation_id(action or "act"),
    ).to_dict()
    payload.update(extra)
    return redact_mapping(prune_none(payload))


def error_result(
    *,
    action: Optional[str] = None,
    message: str = "Browser operation failed",
    error: Optional[Union[BaseException, Mapping[str, Any]]] = None,
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
    payload = ActionOutcome(
        status="error",
        action=action,
        message=message,
        error=error_payload,
        metadata=dict(metadata or {}),
        duration_ms=duration_ms,
        correlation_id=correlation_id or new_correlation_id(action or "err"),
    ).to_dict()
    payload.update(extra)
    return redact_mapping(prune_none(payload))


def normalize_result(result: Any, *, action: Optional[str] = None, default_message: str = "Completed") -> Dict[str, Any]:
    """Normalize arbitrary module output into a browser result dictionary."""

    if isinstance(result, Mapping):
        payload = dict(result)
        payload.setdefault("status", "success" if not payload.get("error") else "error")
        if action is not None:
            payload.setdefault("action", action)
        payload.setdefault("message", default_message if payload["status"] == "success" else "Browser operation failed")
        return redact_mapping(prune_none(payload))
    if isinstance(result, BaseException):
        return exception_to_error_payload(result, action=action)
    return success_result(action=action, message=default_message, data={"value": safe_serialize(result)})

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def ensure_url_scheme(url: str, *, default_scheme: str = "https") -> str:
    """Add a scheme to a URL-like string when missing."""

    text = str(url or "").strip()
    if not text:
        return text
    parsed = urlparse(text)
    if parsed.scheme:
        return text
    if text.startswith("//"):
        return f"{default_scheme}:{text}"
    return f"{default_scheme}://{text}"


def normalize_url(
    url: str,
    *,
    default_scheme: str = "https",
    remove_tracking: bool = True,
    strip_fragment_value: bool = False,
    lowercase_host: bool = True,
    quote_path: bool = True,
) -> str:
    """Normalize a URL for navigation, comparison, and telemetry."""

    text = ensure_url_scheme(str(url or "").strip(), default_scheme=default_scheme)
    if not text:
        return ""
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower() if lowercase_host else parsed.netloc
    path = parsed.path or "/"
    if quote_path:
        path = quote(unquote(path), safe="/%:@")
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if remove_tracking:
        query_pairs = [(k, v) for k, v in query_pairs if k.lower() not in TRACKING_QUERY_PARAMS]
    query = urlencode(query_pairs, doseq=True)
    fragment = "" if strip_fragment_value else parsed.fragment
    return urlunparse((scheme, netloc, path, parsed.params, query, fragment))


def parse_browser_url(url: str, *, default_scheme: str = "https") -> BrowserURL:
    """Parse and normalize a browser URL into a stable dataclass."""

    original = str(url or "").strip()
    normalized = normalize_url(original, default_scheme=default_scheme)
    parsed = urlparse(normalized)
    host = parsed.hostname or ""
    port = parsed.port
    origin = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    return BrowserURL(
        original=original,
        normalized=normalized,
        scheme=parsed.scheme,
        host=host,
        port=port,
        path=parsed.path or "/",
        query=parsed.query,
        fragment=parsed.fragment,
        is_http=parsed.scheme in HTTP_SCHEMES,
        origin=origin,
        resource_type=classify_resource_url(normalized),
    )


def is_valid_url(url: str, *, allowed_schemes: Optional[Iterable[str]] = None, require_netloc: bool = True) -> bool:
    """Validate whether a URL can be used for browser navigation."""

    try:
        if not url or len(str(url)) > MAX_URL_LENGTH:
            return False
        parsed = urlparse(ensure_url_scheme(str(url).strip()))
        schemes = set(allowed_schemes or HTTP_SCHEMES)
        if schemes and parsed.scheme.lower() not in schemes:
            return False
        if require_netloc and not parsed.netloc:
            return False
        return True
    except Exception:
        return False


def validate_url(url: str, *, field_name: str = "url", allowed_schemes: Optional[Iterable[str]] = None) -> str:
    """Validate and return a normalized URL, raising BrowserError on failure."""

    normalized = normalize_url(url)
    if not is_valid_url(normalized, allowed_schemes=allowed_schemes):
        raise make_browser_exception("InvalidURLError", f"Invalid {field_name}: {url}", context={"field": field_name, "url": url})
    return normalized


def same_origin(url_a: str, url_b: str) -> bool:
    """Return True if two URLs share scheme, host, and port."""

    try:
        return parse_browser_url(url_a).origin == parse_browser_url(url_b).origin
    except Exception:
        return False


def domain_matches(url: str, domain: str, *, include_subdomains: bool = True) -> bool:
    """Check whether a URL host matches a domain."""

    try:
        host = parse_browser_url(url).host.lower().strip(".")
        needle = str(domain or "").lower().strip(".")
        if not host or not needle:
            return False
        return host == needle or (include_subdomains and host.endswith(f".{needle}"))
    except Exception:
        return False


def strip_url_fragment(url: str) -> str:
    """Return URL without its fragment."""

    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def redact_url(url: str, *, sensitive_params: Optional[Iterable[str]] = None) -> str:
    """Redact sensitive query parameters while preserving URL debuggability."""

    if not url:
        return ""
    params = {p.lower() for p in (sensitive_params or [])}
    params.update({"token", "access_token", "refresh_token", "api_key", "apikey", "key", "password", "session", "auth"})
    parsed = urlparse(str(url))
    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        query_pairs.append((key, "[REDACTED]" if key.lower() in params or is_sensitive_key(key) else value))
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query_pairs), parsed.fragment))


def join_browser_url(base_url: str, relative_or_absolute: str) -> str:
    """Join a browser-visible base URL with a relative URL and normalize it."""

    return normalize_url(urljoin(base_url or "", relative_or_absolute or ""))


def classify_resource_url(url: str) -> str:
    """Classify the likely resource type of a URL."""

    path = urlparse(str(url or "")).path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if "arxiv.org/abs/" in str(url).lower() or "arxiv.org/pdf/" in str(url).lower():
        return "arxiv"
    if path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp")):
        return "image"
    if path.endswith((".csv", ".tsv", ".xlsx", ".xls", ".json", ".xml", ".yaml", ".yml")):
        return "data"
    if path.endswith((".zip", ".tar", ".gz", ".tgz", ".rar", ".7z")):
        return "archive"
    if path.endswith((".mp4", ".mov", ".webm", ".mp3", ".wav", ".m4a")):
        return "media"
    return "html"


def is_probably_pdf_url(url: str) -> bool:
    """Return True if a URL likely points to a PDF."""

    return classify_resource_url(url) == "pdf" or "application/pdf" in str(url).lower()


def is_arxiv_url(url: str) -> bool:
    """Return True if a URL is likely an arXiv URL."""

    return "arxiv.org" in str(url or "").lower()

# ---------------------------------------------------------------------------
# CSS selector and locator helpers
# ---------------------------------------------------------------------------

def normalize_selector(selector: Any) -> str:
    """Normalize a CSS selector string."""

    return str(selector or "").strip()


def validate_selector(selector: str, *, field_name: str = "selector") -> str:
    """Validate a CSS selector string enough for browser-module input checks."""

    normalized = normalize_selector(selector)
    if not normalized:
        raise make_browser_exception("InvalidSelectorError", f"{field_name} must be a non-empty CSS selector", context={"field": field_name})
    if "\x00" in normalized:
        raise make_browser_exception("InvalidSelectorError", f"{field_name} contains a null byte", context={"field": field_name})
    return normalized


def css_escape_identifier(value: Any) -> str:
    """Escape a string for use as a simple CSS identifier."""

    text = str(value or "")
    if not text:
        return ""
    escaped = []
    for index, char in enumerate(text):
        code = ord(char)
        if char == "\x00":
            escaped.append("\uFFFD")
        elif ("a" <= char <= "z") or ("A" <= char <= "Z") or char == "_" or char == "-" or (char.isdigit() and index > 0):
            escaped.append(char)
        else:
            escaped.append(f"\\{code:x} ")
    return "".join(escaped)


def css_string(value: Any) -> str:
    """Return a double-quoted CSS string literal."""

    text = str(value or "")
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\a ")
    return f'"{text}"'


def attr_selector(attribute: str, value: Any, *, tag: Optional[str] = None, operator: str = "=") -> str:
    """Build a safe attribute selector such as input[name="q"]."""

    attr = re.sub(r"[^a-zA-Z0-9_:\-]", "", str(attribute or ""))
    if not attr:
        raise make_browser_exception("InvalidSelectorError", "Attribute name is required", context={"attribute": attribute})
    prefix = str(tag or "").strip()
    if prefix:
        prefix = css_escape_identifier(prefix)
    return f"{prefix}[{attr}{operator}{css_string(value)}]"


def id_selector(element_id: Any) -> str:
    """Return a safe CSS id selector."""

    return f"#{css_escape_identifier(element_id)}"


def class_selector(class_name: Any) -> str:
    """Return a safe CSS class selector."""

    return f".{css_escape_identifier(class_name)}"


def selector_candidates_from_metadata(metadata: Mapping[str, Any]) -> List[str]:
    """Build likely CSS selectors from element metadata."""

    candidates: List[str] = []
    tag = str(metadata.get("tag") or "").lower().strip() or None
    element_id = metadata.get("id") or metadata.get("element_id")
    name = metadata.get("name")
    aria = metadata.get("aria_label") or metadata.get("aria-label")
    placeholder = metadata.get("placeholder")
    role = metadata.get("role")
    href = metadata.get("href")
    if element_id:
        candidates.append(id_selector(element_id))
    if name:
        candidates.append(attr_selector("name", name, tag=tag))
    if aria:
        candidates.append(attr_selector("aria-label", aria, tag=tag))
    if placeholder:
        candidates.append(attr_selector("placeholder", placeholder, tag=tag))
    if role:
        candidates.append(attr_selector("role", role, tag=tag))
    if href and tag == "a":
        candidates.append(attr_selector("href", href, tag="a"))
    return dedupe_preserve_order(candidates)


def combine_selectors(*selectors: Any) -> str:
    """Combine non-empty selectors into a comma-separated CSS selector."""

    return ", ".join(dedupe_preserve_order(normalize_selector(s) for s in selectors if normalize_selector(s)))

# ---------------------------------------------------------------------------
# Driver and element safety helpers
# ---------------------------------------------------------------------------
def is_web_element(value: Any) -> bool:
    """Best-effort check for Selenium WebElement-like objects."""

    return hasattr(value, "get_attribute") and hasattr(value, "tag_name")


def safe_call(
    func: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    ignored: Tuple[Type[BaseException], ...] = (Exception,),
    **kwargs: Any
) -> Optional[T]:
    """Call a function and return default on ignored exceptions."""
    try:
        return func(*args, **kwargs)
    except ignored:   # now type‑safe
        return default


def safe_get_attribute(element: Any, name: str, *, default: Any = None, max_length: Optional[int] = None) -> Any:
    """Safely retrieve a Selenium element attribute/property."""

    try:
        value = element.get_attribute(name)
    except Exception:
        return default
    if isinstance(value, str) and max_length is not None:
        return truncate_text(value, max_length)
    return default if value is None else value


def element_text(element: Any, *, include_value: bool = True, max_length: int = MAX_CONTEXT_TEXT) -> str:
    """Extract visible/value/placeholder text from an element."""

    parts: List[str] = []
    for getter in (
        lambda: getattr(element, "text", None),
        lambda: safe_get_attribute(element, "innerText"),
        lambda: safe_get_attribute(element, "textContent"),
    ):
        text = normalize_whitespace(safe_call(getter, default=""))
        if text:
            parts.append(text)
            break
    if include_value:
        for attr in ("value", "placeholder", "aria-label", "title", "alt"):
            text = normalize_whitespace(safe_get_attribute(element, attr, default=""))
            if text and text not in parts:
                parts.append(text)
    return truncate_text(" ".join(parts), max_length)


def element_classes(element: Any) -> Tuple[str, ...]:
    """Return normalized element class names."""

    classes = safe_get_attribute(element, "class", default="") or ""
    return tuple(part for part in str(classes).split() if part)


def element_snapshot(element: Any, *, include_html: bool = True, max_text: int = 500, max_html: int = 1_000) -> ElementSnapshot:
    """Capture safe metadata from a Selenium WebElement-like object."""

    if element is None:
        return ElementSnapshot()
    tag = safe_call(lambda: getattr(element, "tag_name", None), default=None)
    text = element_text(element, max_length=max_text)
    role = safe_get_attribute(element, "role")
    element_id = safe_get_attribute(element, "id")
    name = safe_get_attribute(element, "name")
    href = safe_get_attribute(element, "href")
    value = safe_get_attribute(element, "value", max_length=max_text)
    placeholder = safe_get_attribute(element, "placeholder", max_length=max_text)
    aria_label = safe_get_attribute(element, "aria-label", max_length=max_text)
    accessible_name = first_truthy(aria_label, safe_get_attribute(element, "title"), safe_get_attribute(element, "alt"), text)
    outer_html = safe_get_attribute(element, "outerHTML", default="", max_length=max_html) if include_html else ""
    location = safe_call(lambda: dict(getattr(element, "location", {}) or {}), default={}) or {}
    size = safe_call(lambda: dict(getattr(element, "size", {}) or {}), default={}) or {}
    displayed = safe_call(element.is_displayed, default=None) if hasattr(element, "is_displayed") else None
    enabled = safe_call(element.is_enabled, default=None) if hasattr(element, "is_enabled") else None
    selected = safe_call(element.is_selected, default=None) if hasattr(element, "is_selected") else None
    fingerprint = stable_hash(
        {
            "tag": tag,
            "text": text,
            "role": role,
            "id": element_id,
            "name": name,
            "href": href,
            "outer_html": outer_html[:300],
        }
    )
    return ElementSnapshot(
        tag=tag,
        text=text,
        accessible_name=accessible_name,
        role=role,
        element_id=element_id,
        name=name,
        href=href,
        value=value,
        placeholder=placeholder,
        aria_label=aria_label,
        classes=element_classes(element),
        outer_html=outer_html,
        location=location,
        size=size,
        displayed=displayed,
        enabled=enabled,
        selected=selected,
        fingerprint=fingerprint,
    )


def element_metadata(element: Any, *, max_text: int = 500, max_html: int = 1_000) -> Dict[str, Any]:
    """Return element snapshot as a redacted dictionary."""

    return redact_mapping(element_snapshot(element, max_text=max_text, max_html=max_html).to_dict())


def is_interactive_element(element: Any) -> bool:
    """Return True if an element is likely interactable."""

    tag = str(safe_call(lambda: getattr(element, "tag_name", ""), default="") or "").lower()
    role = str(safe_get_attribute(element, "role", default="") or "").lower()
    if tag in INTERACTIVE_TAGS:
        return True
    return role in {"button", "link", "checkbox", "radio", "menuitem", "tab", "textbox", "searchbox", "combobox"}


def is_text_input_element(element: Any) -> bool:
    """Return True when an element likely accepts text input."""

    tag = str(safe_call(lambda: getattr(element, "tag_name", ""), default="") or "").lower()
    input_type = str(safe_get_attribute(element, "type", default="") or "").lower()
    role = str(safe_get_attribute(element, "role", default="") or "").lower()
    contenteditable = str(safe_get_attribute(element, "contenteditable", default="") or "").lower()
    if tag == "textarea":
        return True
    if tag == "input" and input_type not in {"button", "submit", "reset", "checkbox", "radio", "file", "image", "hidden"}:
        return True
    return role in {"textbox", "searchbox"} or contenteditable == "true"


def find_first_element(driver: Any, selectors: Sequence[str], *, timeout: float = 0.0, visible: bool = False) -> Optional[Any]:
    """Find the first matching element from a list of selectors."""

    if not driver:
        return None
    for selector in selectors:
        normalized = normalize_selector(selector)
        if not normalized:
            continue
        try:
            if timeout and WebDriverWait is not None and EC is not None and By is not None:
                condition = EC.visibility_of_element_located((By.CSS_SELECTOR, normalized)) if visible else EC.presence_of_element_located((By.CSS_SELECTOR, normalized))
                return WebDriverWait(driver, timeout).until(condition)
            if By is not None:
                element = driver.find_element(By.CSS_SELECTOR, normalized)
            else:
                element = driver.find_element("css selector", normalized)
            if visible and hasattr(element, "is_displayed") and not element.is_displayed():
                continue
            return element
        except Exception:
            continue
    return None


def find_elements(driver: Any, selector: str, *, max_results: Optional[int] = None) -> List[Any]:
    """Safely find elements with a CSS selector."""

    try:
        if By is not None:
            elements = list(driver.find_elements(By.CSS_SELECTOR, selector))
        else:
            elements = list(driver.find_elements("css selector", selector))
    except Exception:
        return []
    if max_results is not None:
        return elements[: max(0, max_results)]
    return elements


def wait_for_page_load(driver: Any, *, timeout: float = 10.0, poll_interval: float = 0.1, acceptable_states: Sequence[str] = ("complete",)) -> bool:
    """Wait until document.readyState reaches an acceptable state."""

    deadline = time_module.monotonic() + max(0.0, timeout)
    acceptable = set(acceptable_states)
    while time_module.monotonic() <= deadline:
        state = get_document_ready_state(driver)
        if state in acceptable:
            return True
        time_module.sleep(max(0.01, poll_interval))
    return False


def get_document_ready_state(driver: Any) -> Optional[str]:
    """Return document.readyState safely."""

    return safe_call(lambda: driver.execute_script("return document.readyState"), default=None)


def get_current_url(driver: Any) -> str:
    """Return current driver URL safely."""

    return str(safe_call(lambda: driver.current_url, default="") or "")


def get_page_title(driver: Any) -> str:
    """Return current page title safely."""

    return str(safe_call(lambda: driver.title, default="") or "")


def get_body_text(driver: Any, *, max_length: Optional[int] = None) -> str:
    """Return visible body text safely."""

    text = ""
    try:
        if By is not None:
            body = driver.find_element(By.TAG_NAME, "body")
        else:
            body = driver.find_element("tag name", "body")
        text = getattr(body, "text", "") or safe_get_attribute(body, "innerText", default="") or ""
    except Exception:
        text = safe_call(lambda: driver.execute_script("return document.body ? document.body.innerText : ''"), default="") or ""
    return truncate_text(text, max_length) if max_length else str(text)


def get_page_html(driver: Any, *, max_length: Optional[int] = None) -> str:
    """Return current page HTML safely."""

    html_text = safe_call(lambda: driver.page_source, default="") or ""
    return truncate_text(html_text, max_length) if max_length else str(html_text)


def get_viewport(driver: Any) -> Dict[str, Any]:
    """Capture viewport and scroll metrics."""

    script = """
    return {
      width: window.innerWidth || 0,
      height: window.innerHeight || 0,
      scrollX: window.scrollX || 0,
      scrollY: window.scrollY || 0,
      pageWidth: Math.max(document.documentElement.scrollWidth || 0, document.body ? document.body.scrollWidth || 0 : 0),
      pageHeight: Math.max(document.documentElement.scrollHeight || 0, document.body ? document.body.scrollHeight || 0 : 0),
      devicePixelRatio: window.devicePixelRatio || 1
    };
    """
    result = safe_call(lambda: driver.execute_script(script), default={}) or {}
    return safe_serialize(result) if isinstance(result, Mapping) else {}


def get_performance_timing(driver: Any) -> Dict[str, Any]:
    """Capture basic Navigation Timing metrics when available."""

    script = """
    const nav = performance.getEntriesByType && performance.getEntriesByType('navigation')[0];
    if (nav) {
      return {
        type: nav.type,
        startTime: nav.startTime,
        domContentLoadedEventEnd: nav.domContentLoadedEventEnd,
        loadEventEnd: nav.loadEventEnd,
        duration: nav.duration,
        transferSize: nav.transferSize || 0,
        encodedBodySize: nav.encodedBodySize || 0,
        decodedBodySize: nav.decodedBodySize || 0
      };
    }
    return {};
    """
    result = safe_call(lambda: driver.execute_script(script), default={}) or {}
    return safe_serialize(result) if isinstance(result, Mapping) else {}


def detect_captcha_text(text: str) -> bool:
    """Detect CAPTCHA/security-check wording in page text/html."""

    lowered = str(text or "").lower()
    return any(indicator in lowered for indicator in CAPTCHA_INDICATORS)


def detect_captcha(driver: Any) -> bool:
    """Detect common CAPTCHA/security-check states from page source/text/url."""

    try:
        source = get_page_html(driver, max_length=250_000)
        body = get_body_text(driver, max_length=50_000)
        url = get_current_url(driver)
        return detect_captcha_text("\n".join([source, body, url]))
    except Exception:
        return False


def capture_screenshot_b64(driver: Any) -> Optional[str]:
    """Capture a PNG screenshot as base64, returning None on failure."""

    try:
        raw = driver.get_screenshot_as_png()
        return base64.b64encode(raw).decode("ascii")
    except Exception:
        try:
            return driver.get_screenshot_as_base64()
        except Exception:
            return None


def page_snapshot(
    driver: Any,
    *,
    include_html: bool = False,
    include_screenshot: bool = False,
    max_text: int = 2_000,
    max_html: int = 4_000,
) -> PageSnapshot:
    """Capture a safe page snapshot for logs and downstream reasoning."""

    url = get_current_url(driver)
    title = get_page_title(driver)
    ready_state = get_document_ready_state(driver)
    text = get_body_text(driver, max_length=max_text)
    html_text = get_page_html(driver, max_length=max_html) if include_html else ""
    screenshot_b64 = capture_screenshot_b64(driver) if include_screenshot else None
    snapshot = PageSnapshot(
        url=redact_url(url),
        title=truncate_text(title, 500),
        ready_state=ready_state,
        text=text,
        html=html_text,
        screenshot_b64=screenshot_b64,
        viewport=get_viewport(driver),
        timing=get_performance_timing(driver),
        detected_captcha=detect_captcha_text("\n".join([url, title, text, html_text])),
        content_fingerprint=stable_hash({"url": url, "title": title, "text": text}),
        captured_at=utc_now_iso(),
    )
    return snapshot


def page_snapshot_dict(driver: Any, **kwargs: Any) -> Dict[str, Any]:
    """Return a redacted page snapshot dictionary."""

    return redact_mapping(page_snapshot(driver, **kwargs).to_dict())

# ---------------------------------------------------------------------------
# Search and relevance helpers
# ---------------------------------------------------------------------------

def tokenize_query(value: Any) -> List[str]:
    """Tokenize a query or link text for simple browser relevance scoring."""

    text = normalize_whitespace(value).lower()
    tokens = re.findall(r"[\w\-]{2,}", text, flags=re.UNICODE)
    stopwords = {"the", "and", "for", "with", "from", "into", "this", "that", "http", "https", "www"}
    return [token for token in tokens if token not in stopwords]


def score_text_relevance(query: str, candidate_text: str, *, url: str = "") -> float:
    """Score candidate text/url against a query using transparent heuristics."""

    query_tokens = tokenize_query(query)
    if not query_tokens:
        return 0.0
    candidate_tokens = tokenize_query(" ".join([candidate_text or "", url or ""]))
    if not candidate_tokens:
        return 0.0
    candidate_set = set(candidate_tokens)
    query_set = set(query_tokens)
    overlap = len(query_set.intersection(candidate_set))
    coverage = overlap / max(1, len(query_set))
    density = overlap / max(1, len(candidate_set))
    phrase_bonus = 0.15 if normalize_whitespace(query).lower() in normalize_whitespace(candidate_text).lower() else 0.0
    domain_bonus = 0.05 if any(token in url.lower() for token in query_set) else 0.0
    return round(min(1.0, coverage * 0.75 + density * 0.25 + phrase_bonus + domain_bonus), 4)


def search_result_from_element(element: Any, *, query: Optional[str] = None, rank: Optional[int] = None, base_url: Optional[str] = None) -> Optional[SearchResultSnapshot]:
    """Build a normalized search-result snapshot from a link element."""

    href = safe_get_attribute(element, "href", default="") or ""
    if not href:
        return None
    url = join_browser_url(base_url or "", href) if base_url else normalize_url(href)
    if not is_valid_url(url):
        return None
    text = element_text(element, max_length=500)
    title = text or safe_get_attribute(element, "title", default="") or ""
    score = score_text_relevance(query, text, url=url) if query else None
    metadata = element_metadata(element, max_text=300, max_html=600)
    return SearchResultSnapshot(url=url, text=text, title=title, rank=rank, score=score, source="dom", metadata=metadata)


def extract_link_snapshots(
    driver: Any,
    *,
    query: Optional[str] = None,
    selectors: Sequence[str] = SEARCH_RESULT_LINK_SELECTORS,
    max_results: int = 10,
) -> List[SearchResultSnapshot]:
    """Extract normalized link/search-result snapshots from the current page."""

    base_url = get_current_url(driver)
    snapshots: List[SearchResultSnapshot] = []
    seen: set = set()
    for selector in selectors:
        for element in find_elements(driver, selector, max_results=max_results * 3):
            snapshot = search_result_from_element(element, query=query, rank=len(snapshots) + 1, base_url=base_url)
            if not snapshot or snapshot.url in seen:
                continue
            seen.add(snapshot.url)
            snapshots.append(snapshot)
            if len(snapshots) >= max_results:
                return snapshots
    return snapshots


def select_best_link(query: str, candidates: Sequence[Any], *, base_url: Optional[str] = None) -> Optional[Any]:
    """Select the best link-like candidate for a query."""

    if not candidates:
        return None
    scored: List[Tuple[float, int, Any]] = []
    for index, candidate in enumerate(candidates):
        if is_web_element(candidate):
            text = element_text(candidate, max_length=500)
            href = safe_get_attribute(candidate, "href", default="") or ""
            url = join_browser_url(base_url or "", href) if base_url and href else href
        elif isinstance(candidate, Mapping):
            text = str(candidate.get("text") or candidate.get("title") or candidate.get("snippet") or "")
            url = str(candidate.get("url") or candidate.get("link") or "")
        else:
            text = str(candidate)
            url = ""
        score = score_text_relevance(query, text, url=url)
        scored.append((score, -index, candidate))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def search_result_dicts(results: Sequence[SearchResultSnapshot]) -> List[Dict[str, Any]]:
    """Convert search-result snapshots to legacy result dicts."""

    payload = []
    for result in results:
        item = result.to_dict()
        item["link"] = item.pop("url")
        payload.append(redact_mapping(item))
    return payload

# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------

def html_to_text(markup: str, *, max_length: Optional[int] = None) -> str:
    """Lightweight HTML-to-text fallback without external dependencies."""

    if not markup:
        return ""
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", str(markup))
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = normalize_newlines(text)
    return truncate_text(text, max_length) if max_length else text


def limit_content(value: Any, *, max_chars: int = 2_000, preserve_newlines: bool = True) -> str:
    """Normalize and limit extracted page/PDF/text content."""

    text = normalize_newlines(value) if preserve_newlines else normalize_whitespace(value)
    return truncate_text(text, max_chars)


def content_fingerprint(*parts: Any, length: int = 16) -> str:
    """Fingerprint content parts for deduplication and cache keys."""

    return stable_hash([normalize_whitespace(part) for part in parts if part is not None], length=length)


def extract_title_from_html(markup: str) -> str:
    """Extract title text from HTML markup."""

    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", markup or "")
    return normalize_whitespace(html.unescape(match.group(1))) if match else ""


def classify_content_type(content_type: str, url: str = "") -> str:
    """Classify HTTP content type and URL into a browser resource category."""

    content_type_lower = str(content_type or "").lower().split(";", 1)[0].strip()
    if content_type_lower == "application/pdf" or is_probably_pdf_url(url):
        return "pdf"
    if content_type_lower.startswith("text/html"):
        return "html"
    if content_type_lower.startswith("text/"):
        return "text"
    if "json" in content_type_lower:
        return "json"
    if "xml" in content_type_lower:
        return "xml"
    if content_type_lower.startswith("image/"):
        return "image"
    if content_type_lower.startswith("audio/") or content_type_lower.startswith("video/"):
        return "media"
    return classify_resource_url(url) if url else "unknown"

# ---------------------------------------------------------------------------
# Retry and timing helpers
# ---------------------------------------------------------------------------
def calculate_backoff_delay(
    attempt_index: int,
    *,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    multiplier: float = 2.0,
    jitter: float = 0.1,
) -> float:
    """Calculate exponential backoff delay with optional jitter."""

    attempt = max(0, int(attempt_index))
    delay = min(max_delay, base_delay * (multiplier ** attempt))
    if jitter > 0:
        jitter_amount = delay * jitter
        delay += random.uniform(-jitter_amount, jitter_amount)
    return round(max(0.0, delay), 4)


def sleep_backoff(attempt_index: int, **kwargs: Any) -> float:
    """Sleep for a calculated backoff delay and return the delay."""

    delay = calculate_backoff_delay(attempt_index, **kwargs)
    time_module.sleep(delay)
    return delay


def retry_operation(
    operation: Callable[..., T],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    action: Optional[str] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> T:
    """Run an operation with a browser RetryPolicy.

    This helper raises the final exception rather than returning a dict so it can
    be used from lower-level modules. Agent facades can convert the exception via
    exception_to_error_payload or error_result.
    """

    policy = policy or RetryPolicy()
    last_exc: Optional[BaseException] = None
    for attempt in range(max(1, policy.max_attempts)):
        try:
            return operation(*args, **kwargs)
        except policy.retryable_exceptions as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt >= policy.max_attempts - 1:
                break
            delay = policy.delay_for_attempt(attempt)
            if on_retry:
                safe_call(on_retry, attempt + 1, exc, delay, default=None)
            time_module.sleep(delay)
    raise make_browser_exception(
        "RetryExhaustedError",
        f"Browser operation exhausted retries: {action or getattr(operation, '__name__', 'operation')}",
        context={"action": action, "max_attempts": policy.max_attempts},
        cause=last_exc if isinstance(last_exc, Exception) else None,
    )

# ---------------------------------------------------------------------------
# Workflow and task normalization helpers
# ---------------------------------------------------------------------------
def normalize_action_name(action: Any) -> str:
    """Normalize a browser workflow action name."""

    aliases = {
        "open": "navigate",
        "open_url": "navigate",
        "go_to_url": "navigate",
        "google_search": "search",
        "entertext": "type",
        "enter_text": "type",
        "input": "type",
        "press": "type",
        "get_dom": "extract",
        "get_url": "extract",
        "extract_page": "extract",
        "pdf_extractor": "extract",
    }
    text = str(action or "").lower().strip().replace("-", "_").replace(" ", "_")
    return aliases.get(text, text)


def normalize_workflow_step(step: Mapping[str, Any], *, index: int = 0, supported_actions: Sequence[str] = BROWSER_ACTIONS) -> Dict[str, Any]:
    """Normalize one workflow step into {action, params}."""

    if not isinstance(step, Mapping):
        raise make_browser_exception("WorkflowValidationError", f"Workflow step {index} must be a mapping", context={"index": index, "step": safe_serialize(step)})
    action = normalize_action_name(step.get("action") or step.get("task") or step.get("tool"))
    if not action:
        raise make_browser_exception("WorkflowValidationError", f"Workflow step {index} is missing an action", context={"index": index, "step": safe_serialize(step)})
    if action not in set(supported_actions):
        raise make_browser_exception(
            "UnsupportedWorkflowActionError",
            f"Unsupported workflow action at index {index}: {action}",
            context={"index": index, "action": action, "supported_actions": list(supported_actions)},
        )
    params = step.get("params")
    if params is None:
        params = {k: v for k, v in step.items() if k not in {"action", "task", "tool"}}
    if not isinstance(params, Mapping):
        raise make_browser_exception("WorkflowValidationError", f"Workflow params at index {index} must be a mapping", context={"index": index, "params": safe_serialize(params)})
    return {"action": action, "params": dict(params)}


def normalize_workflow(workflow_script: Sequence[Mapping[str, Any]], *, supported_actions: Sequence[str] = BROWSER_ACTIONS) -> List[Dict[str, Any]]:
    """Normalize a browser workflow script."""

    if workflow_script is None:
        return []
    if not isinstance(workflow_script, Sequence) or isinstance(workflow_script, (str, bytes)):
        raise make_browser_exception("WorkflowValidationError", "Workflow script must be a sequence of steps", context={"workflow": safe_serialize(workflow_script)})
    return [normalize_workflow_step(step, index=index, supported_actions=supported_actions) for index, step in enumerate(workflow_script)]


def normalize_task_payload(task_data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize BrowserAgent.perform_task-style payloads."""

    if not isinstance(task_data, Mapping):
        raise make_browser_exception("InvalidTaskPayloadError", "Browser task payload must be a mapping", context={"task_data": safe_serialize(task_data)})
    payload = dict(task_data)
    if "workflow" in payload:
        payload["workflow"] = normalize_workflow(payload["workflow"])
        return payload
    task = normalize_action_name(payload.get("task") or ("search" if payload.get("query") else ""))
    if task:
        payload["task"] = task
    return payload

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def get_config_value(section_name: str, key: str, default: Any = None) -> Any:
    """Read a config value from the browser config section safely."""

    try:
        section = get_config_section(section_name) or {}
        return section.get(key, default) if isinstance(section, Mapping) else default
    except Exception:
        return default


def get_browser_helper_config() -> Dict[str, Any]:
    """Return browser_helpers config section if present."""

    section = get_config_section("browser_helpers") or {}
    return dict(section) if isinstance(section, Mapping) else {}


def config_bool(section_name: str, key: str, default: bool = False) -> bool:
    return coerce_bool(get_config_value(section_name, key, default), default=default)


def config_int(section_name: str, key: str, default: int = 0, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    return coerce_int(get_config_value(section_name, key, default), default=default, minimum=minimum, maximum=maximum)


def config_float(section_name: str, key: str, default: float = 0.0, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    return coerce_float(get_config_value(section_name, key, default), default=default, minimum=minimum, maximum=maximum)

# ---------------------------------------------------------------------------
# Miscellaneous browser-domain utilities
# ---------------------------------------------------------------------------
def first_truthy(*values: Any, default: Any = None) -> Any:
    """Return the first truthy value, or default."""

    for value in values:
        if value:
            return value
    return default


def dedupe_preserve_order(values: Iterable[T]) -> List[T]:
    """Deduplicate values while preserving first-seen order."""

    seen: set = set()
    result: List[T] = []
    for value in values:
        marker = safe_json_dumps(value, sort_keys=True) if isinstance(value, (Mapping, list, tuple, set)) else value
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def chunk_sequence(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield chunks from a sequence."""

    chunk_size = max(1, int(size))
    for index in range(0, len(values), chunk_size):
        yield values[index : index + chunk_size]


def safe_filename(value: Any, *, default: str = "browser-artifact", max_length: int = 120) -> str:
    """Build a safe filename from a page title, URL, or action name."""

    text = normalize_whitespace(value)
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip(".-_")
    text = text[:max_length].strip(".-_")
    return text or default


def infer_file_extension_from_url(url: str, *, default: str = ".html") -> str:
    """Infer file extension from URL path."""

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix and len(suffix) <= 12 and re.fullmatch(r"\.[a-z0-9]+", suffix):
        return suffix
    resource = classify_resource_url(url)
    mapping = {"pdf": ".pdf", "image": ".img", "data": ".data", "archive": ".archive", "media": ".media", "html": ".html"}
    return mapping.get(resource, default)


def build_artifact_name(url: str = "", title: str = "", *, prefix: str = "page", extension: Optional[str] = None) -> str:
    """Build a deterministic artifact filename for captured browser data."""

    base = safe_filename(title or parse_browser_url(url).host or prefix, default=prefix)
    suffix = extension or infer_file_extension_from_url(url)
    digest = stable_hash({"url": url, "title": title}, length=8)
    return f"{base}-{digest}{suffix}"


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""

    return coerce_bool(os.getenv(name), default=default)


def log_result(result: Mapping[str, Any], *, level: int = logger.info) -> None:
    """Log a redacted browser result payload."""

    try:
        logger.log(level, safe_json_dumps(result, redact=True))
    except Exception:
        logger.log(level, "%s", redact_data(result))

