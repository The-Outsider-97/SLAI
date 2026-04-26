"""
Production-grade structured exception hierarchy and validation helpers for the
browser agent stack.

This module gives the BrowserAgent and its lower-level browser function modules
one stable error contract:

- deterministic error codes for logs, metrics, alerts, and tests;
- domain-specific exception classes for browser operations;
- retryability metadata for orchestration and backoff logic;
- safe, JSON-serialisable context for audit trails;
- redaction of sensitive values before logging or returning errors;
- Selenium/request-style exception wrapping without hard dependencies;
- input, configuration, workflow, selector, and URL validation helpers.

The module intentionally avoids importing the browser agent or concrete action
modules. Error handling must stay dependency-light and safe to use from failure
paths.
"""

from __future__ import annotations

import hashlib
import json
import traceback

from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Browser Error")
printer = PrettyPrinter

T = TypeVar("T", bound="BrowserError")
ErrorFactory = Union[Type["BrowserError"], Callable[..., "BrowserError"]]


class BrowserErrorType(str, Enum):
    """Canonical error domains across the browser subsystem."""

    UNKNOWN = "unknown"
    CONFIGURATION = "configuration"
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    DRIVER = "driver"
    NAVIGATION = "navigation"
    SEARCH = "search"
    ELEMENT = "element"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    CLIPBOARD = "clipboard"
    CONTENT = "content"
    SECURITY = "security"
    WORKFLOW = "workflow"
    RETRY = "retry"
    NETWORK = "network"
    TASK = "task"
    STATE = "state"
    SCRIPT = "script"


class BrowserErrorSeverity(str, Enum):
    """Operational severity levels used by logs, alerting, and result payloads."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


VALID_SEVERITIES = {severity.value for severity in BrowserErrorSeverity}

# Broad domain fallback codes. Concrete subclasses override these with more
# specific codes, while custom errors can still rely on the domain defaults.
_ERROR_CODE_MAP: Dict[BrowserErrorType, str] = {
    BrowserErrorType.UNKNOWN: "BRW-1000",
    BrowserErrorType.CONFIGURATION: "BRW-1100",
    BrowserErrorType.INITIALIZATION: "BRW-1200",
    BrowserErrorType.VALIDATION: "BRW-1300",
    BrowserErrorType.DRIVER: "BRW-1400",
    BrowserErrorType.NAVIGATION: "BRW-1500",
    BrowserErrorType.SEARCH: "BRW-1600",
    BrowserErrorType.ELEMENT: "BRW-1700",
    BrowserErrorType.CLICK: "BRW-1800",
    BrowserErrorType.TYPE: "BRW-1900",
    BrowserErrorType.SCROLL: "BRW-2000",
    BrowserErrorType.CLIPBOARD: "BRW-2100",
    BrowserErrorType.CONTENT: "BRW-2200",
    BrowserErrorType.SECURITY: "BRW-2300",
    BrowserErrorType.WORKFLOW: "BRW-2400",
    BrowserErrorType.RETRY: "BRW-2500",
    BrowserErrorType.NETWORK: "BRW-2600",
    BrowserErrorType.TASK: "BRW-2700",
    BrowserErrorType.STATE: "BRW-2800",
    BrowserErrorType.SCRIPT: "BRW-2900",
}

SENSITIVE_KEY_PATTERNS = (
    "authorization",
    "access_token",
    "refresh_token",
    "id_token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "credential",
    "cookie",
    "session",
    "csrf",
    "xsrf",
    "bearer",
    "private_key",
    "client_secret",
    "clipboard",
)

DEFAULT_MAX_STRING_LENGTH = 2_000
DEFAULT_MAX_SEQUENCE_LENGTH = 50
DEFAULT_MAX_MAPPING_LENGTH = 80
REDACTION_PLACEHOLDER = "[REDACTED]"


@dataclass(frozen=True)
class BrowserErrorPayload:
    """Portable representation of a browser error for logs and API results."""

    code: str
    type: str
    message: str
    severity: str
    retryable: bool
    category: Union[str, Dict[str, Any]]
    context: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[Dict[str, Any]] = None
    retry_after_seconds: Optional[float] = None
    fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "code": self.code,
            "type": self.type,
            "message": self.message,
            "severity": self.severity,
            "retryable": self.retryable,
            "category": self.category,
            "context": self.context,
        }
        if self.cause is not None:
            payload["cause"] = self.cause
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        if self.fingerprint is not None:
            payload["fingerprint"] = self.fingerprint
        return payload


# ---------------------------------------------------------------------------
# Safe serialisation and redaction
# ---------------------------------------------------------------------------
def _safe_load_config() -> Dict[str, Any]:
    """Best-effort global config load used only for optional error behaviour."""

    if load_global_config is None:
        return {}
    try:
        loaded = load_global_config()
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:  # pragma: no cover - depends on deployment files
        try:
            logger.debug("Unable to load browser config while building error: %s", exc)
        except Exception:
            pass
        return {}


def _safe_get_error_config() -> Dict[str, Any]:
    if get_config_section is None:
        return {}
    try:
        section = get_config_section("browser_errors")
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # pragma: no cover - depends on deployment files
        try:
            logger.debug("Unable to load browser_errors config section: %s", exc)
        except Exception:
            pass
        return {}


def _coerce_error_type(value: Optional[Union[str, BrowserErrorType]]) -> BrowserErrorType:
    if value is None:
        return BrowserErrorType.UNKNOWN
    if isinstance(value, BrowserErrorType):
        return value
    try:
        return BrowserErrorType(str(value).lower())
    except ValueError:
        return BrowserErrorType.UNKNOWN


def _coerce_severity(value: Optional[Union[str, BrowserErrorSeverity]], default: str = "medium") -> str:
    if isinstance(value, BrowserErrorSeverity):
        return value.value
    candidate = (str(value).lower().strip() if value is not None else default).strip()
    if candidate not in VALID_SEVERITIES:
        raise ValueError(f"Invalid severity level: {candidate}. Must be one of {sorted(VALID_SEVERITIES)}.")
    return candidate


def _is_sensitive_key(key: Any) -> bool:
    key_text = str(key).lower()
    return any(pattern in key_text for pattern in SENSITIVE_KEY_PATTERNS)


def _truncate(text: str, max_length: int = DEFAULT_MAX_STRING_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}…[truncated {len(text) - max_length} chars]"


def _redact_url(value: str) -> str:
    """Redact sensitive query parameters from a URL-like string."""

    try:
        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            return _truncate(value)
        query_pairs = []
        redacted = False
        for key, item_value in parse_qsl(parsed.query, keep_blank_values=True):
            if _is_sensitive_key(key):
                query_pairs.append((key, REDACTION_PLACEHOLDER))
                redacted = True
            else:
                query_pairs.append((key, item_value))
        if not redacted:
            return _truncate(value)
        return _truncate(urlunparse(parsed._replace(query=urlencode(query_pairs, doseq=True))))
    except Exception:
        return _truncate(value)


def _serialise_web_element(value: Any) -> Optional[Dict[str, Any]]:
    """Return a safe Selenium WebElement snapshot when possible."""

    if not hasattr(value, "tag_name") or not hasattr(value, "get_attribute"):
        return None
    try:
        text = getattr(value, "text", "") or ""
    except Exception:
        text = ""
    snapshot: Dict[str, Any] = {
        "tag": safe_serialize(getattr(value, "tag_name", None)),
        "text": _truncate(str(text), 200),
    }
    for attr in ("id", "name", "role", "aria-label", "placeholder", "class", "href", "value"):
        try:
            attr_value = value.get_attribute(attr)
        except Exception:
            attr_value = None
        if attr_value:
            snapshot[attr.replace("-", "_")] = safe_serialize(attr_value, max_string_length=500)
    try:
        outer_html = value.get_attribute("outerHTML")
        if outer_html:
            snapshot["outer_html"] = _truncate(str(outer_html), 500)
    except Exception:
        pass
    return snapshot


def safe_serialize(
    value: Any,
    *,
    redact: bool = True,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    max_mapping_length: int = DEFAULT_MAX_MAPPING_LENGTH,
) -> Any:
    """Convert arbitrary Python values into JSON-safe primitives.

    This function is deliberately defensive because it is commonly called while
    handling exceptions. It must not raise for unusual runtime objects.
    """

    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return f"<bytes length={len(value)}>"
        if isinstance(value, str):
            return _redact_url(value) if redact else _truncate(value, max_string_length)
        if isinstance(value, BaseException):
            return {
                "type": value.__class__.__name__,
                "message": _truncate(str(value), max_string_length),
            }

        element_snapshot = _serialise_web_element(value)
        if element_snapshot is not None:
            return element_snapshot

        if isinstance(value, Mapping):
            output: Dict[str, Any] = {}
            for index, (key, item_value) in enumerate(value.items()):
                if index >= max_mapping_length:
                    output["__truncated__"] = f"{len(value) - max_mapping_length} additional keys omitted"
                    break
                key_text = str(key)
                if redact and _is_sensitive_key(key_text):
                    output[key_text] = REDACTION_PLACEHOLDER
                else:
                    output[key_text] = safe_serialize(
                        item_value,
                        redact=redact,
                        max_string_length=max_string_length,
                        max_sequence_length=max_sequence_length,
                        max_mapping_length=max_mapping_length,
                    )
            return output

        if isinstance(value, (Sequence, set, frozenset)) and not isinstance(value, (str, bytes, bytearray)):
            items = list(value)
            output_items = [
                safe_serialize(
                    item,
                    redact=redact,
                    max_string_length=max_string_length,
                    max_sequence_length=max_sequence_length,
                    max_mapping_length=max_mapping_length,
                )
                for item in items[:max_sequence_length]
            ]
            if len(items) > max_sequence_length:
                output_items.append(f"…[{len(items) - max_sequence_length} additional items omitted]")
            return output_items

        return _truncate(repr(value), max_string_length)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"<unserialisable {type(value).__name__}: {type(exc).__name__}>"


def sanitize_context(context: Optional[Mapping[str, Any]], *, redact: bool = True) -> Dict[str, Any]:
    """Return a JSON-safe, optionally redacted context dictionary."""

    if not context:
        return {}
    serialised = safe_serialize(dict(context), redact=redact)
    return serialised if isinstance(serialised, dict) else {"value": serialised}


def _build_fingerprint(code: str, error_type: str, message: str, category: Union[str, Dict[str, Any]]) -> str:
    source = json.dumps(
        {"code": code, "type": error_type, "message": message, "category": category},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _exception_cause_payload(cause: Optional[BaseException], *, include_traceback: bool = False, redact: bool = True) -> Optional[Dict[str, Any]]:
    if cause is None:
        return None
    payload = {
        "type": cause.__class__.__name__,
        "message": safe_serialize(str(cause), redact=redact),
    }
    if include_traceback:
        payload["traceback"] = _truncate("".join(traceback.format_exception(type(cause), cause, cause.__traceback__)), 10_000)
    return payload


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------
class BrowserError(Exception):
    """Base exception for the browser subsystem.

    Every subclass should provide a stable ``default_code`` and a specific
    ``error_type``. Instances can be converted into JSON-safe dictionaries or
    backwards-compatible ``{"status": "error"}`` results.
    """

    error_type: BrowserErrorType = BrowserErrorType.UNKNOWN
    default_code = "BRW-1000"
    default_message = "Browser error"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category: Union[str, Dict[str, Any]] = "browser"
    default_retry_after_seconds: Optional[float] = None

    def __init__(self, message: Optional[str] = None, *,
        error_type: Optional[Union[str, BrowserErrorType]] = None,
        severity: Optional[Union[str, BrowserErrorSeverity]] = None,
        code: Optional[str] = None, retryable: Optional[bool] = None,
        category: Optional[Union[str, Dict[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
        retry_after_seconds: Optional[float] = None,
    ):
        resolved_type = _coerce_error_type(error_type or self.error_type)
        resolved_code = code or self.default_code or _ERROR_CODE_MAP.get(resolved_type, "BRW-1000")
        resolved_message = message or self.default_message or f"{resolved_type.value.replace('_', ' ').title()} error"

        self.error_type = resolved_type
        self.code = str(resolved_code)
        self.message = str(resolved_message)
        self.severity = _coerce_severity(severity, default=self.default_severity)
        self.retryable = bool(self.default_retryable if retryable is None else retryable)
        self.category = category if category is not None else self.default_category
        self.context = sanitize_context(context, redact=False)
        self.cause = cause
        self.retry_after_seconds = self.default_retry_after_seconds if retry_after_seconds is None else retry_after_seconds

        self.global_config = _safe_load_config()
        self.error_config = _safe_get_error_config()

        super().__init__(f"[{self.code}] {self.message}")

    def with_context(self: T, **context: Any) -> T:
        """Mutate context in-place and return self for concise raise paths."""
        self.context.update(sanitize_context(context, redact=False))
        return self

    def to_payload(self, *, redact: bool = True, include_cause: bool = True,
                   include_traceback: bool = False) -> BrowserErrorPayload:
        context = sanitize_context(self.context, redact=redact)
        cause_payload = _exception_cause_payload(self.cause, include_traceback=include_traceback, redact=redact) if include_cause else None
        category = safe_serialize(self.category, redact=redact)
        return BrowserErrorPayload(
            code=self.code,
            type=self.error_type.value,
            message=self.message,
            severity=self.severity,
            retryable=self.retryable,
            category=category,
            context=context,
            cause=cause_payload,
            retry_after_seconds=self.retry_after_seconds,
            fingerprint=_build_fingerprint(self.code, self.error_type.value, self.message, category),
        )

    def to_dict(self, *, redact: bool = True, include_cause: bool = True,
                include_traceback: bool = False) -> Dict[str, Any]:
        return self.to_payload(redact=redact, include_cause=include_cause, include_traceback=include_traceback).to_dict()

    def to_json(self, *, redact: bool = True, include_cause: bool = True, include_traceback: bool = False) -> str:
        return json.dumps(self.to_dict(redact=redact, include_cause=include_cause, include_traceback=include_traceback), sort_keys=True)

    def to_result(self, *, action: Optional[str] = None, redact: bool = True, include_cause: bool = True,
                  include_traceback: bool = False, extra: Optional[Mapping[str, Any]] = None ) -> Dict[str, Any]:
        """Return a backwards-compatible action result dictionary."""

        result: Dict[str, Any] = {
            "status": "error",
            "message": self.message,
            "code": self.code,
            "error_type": self.error_type.value,
            "severity": self.severity,
            "retryable": self.retryable,
            "error": self.to_dict(redact=redact, include_cause=include_cause, include_traceback=include_traceback),
        }
        if action:
            result["action"] = action
        if self.retry_after_seconds is not None:
            result["retry_after_seconds"] = self.retry_after_seconds
        if extra:
            result.update(sanitize_context(extra, redact=redact))
        return result

    def to_log_record(self, *, include_traceback: bool = False) -> Dict[str, Any]:
        return self.to_dict(redact=True, include_cause=True, include_traceback=include_traceback)

    def log(self, *, level: Optional[int] = None, include_traceback: bool = False) -> None:
        """Best-effort structured logging for the error."""

        if level is None:
            level = {
                BrowserErrorSeverity.LOW.value: logger.INFO,
                BrowserErrorSeverity.MEDIUM.value: logger.WARNING,
                BrowserErrorSeverity.HIGH.value: logger.ERROR,
                BrowserErrorSeverity.CRITICAL.value: logger.CRITICAL,
            }.get(self.severity, logger.ERROR)
        try:
            logger.log(level, "%s", self.to_json(redact=True, include_cause=True, include_traceback=include_traceback))
        except Exception:
            pass

    @classmethod
    def from_exception(cls, exc: BaseException, *, message: Optional[str] = None, action: Optional[str] = None,
                       context: Optional[Mapping[str, Any]] = None, default_error_cls: Type["BrowserError"] = None,  # type: ignore[assignment]
                       ) -> "BrowserError":
        """Convert any exception into the nearest browser-domain error."""

        if isinstance(exc, BrowserError):
            if context:
                exc.context.update(sanitize_context(context, redact=False))
            if action and "action" not in exc.context:
                exc.context["action"] = action
            return exc

        error_cls = _map_exception_to_browser_error(exc, action=action) or default_error_cls or cls
        merged_context: Dict[str, Any] = {}
        if action:
            merged_context["action"] = action
        if context:
            merged_context.update(dict(context))
        return error_cls(message or str(exc) or error_cls.default_message, context=merged_context, cause=exc)


# ---------------------------------------------------------------------------
# General/system errors
# ---------------------------------------------------------------------------
class UnknownBrowserError(BrowserError):
    error_type = BrowserErrorType.UNKNOWN
    default_code = "BRW-1000"
    default_message = "Unknown browser error"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "browser.unknown"


class BrowserConfigurationError(BrowserError):
    error_type = BrowserErrorType.CONFIGURATION
    default_code = "BRW-1100"
    default_message = "Browser configuration error"
    default_severity = BrowserErrorSeverity.HIGH.value
    default_retryable = False
    default_category = "browser.configuration"


class MissingConfigurationError(BrowserConfigurationError):
    default_code = "BRW-1101"
    default_message = "Required browser configuration is missing"


class InvalidConfigurationError(BrowserConfigurationError):
    default_code = "BRW-1102"
    default_message = "Browser configuration value is invalid"


class BrowserInitializationError(BrowserError):
    error_type = BrowserErrorType.INITIALIZATION
    default_code = "BRW-1200"
    default_message = "Browser initialization failed"
    default_severity = BrowserErrorSeverity.CRITICAL.value
    default_retryable = True
    default_category = "browser.initialization"


class BrowserDriverStartupError(BrowserInitializationError):
    default_code = "BRW-1201"
    default_message = "Browser driver failed to start"


class BrowserDependencyError(BrowserInitializationError):
    default_code = "BRW-1202"
    default_message = "Required browser dependency is unavailable"
    default_retryable = False


class BrowserValidationError(BrowserError):
    error_type = BrowserErrorType.VALIDATION
    default_code = "BRW-1300"
    default_message = "Browser input validation failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "browser.validation"


class InvalidURLError(BrowserValidationError):
    default_code = "BRW-1301"
    default_message = "Invalid URL"


class InvalidSelectorError(BrowserValidationError):
    default_code = "BRW-1302"
    default_message = "Invalid CSS selector"


class InvalidTaskPayloadError(BrowserValidationError):
    default_code = "BRW-1303"
    default_message = "Invalid browser task payload"


class MissingRequiredFieldError(BrowserValidationError):
    default_code = "BRW-1304"
    default_message = "Missing required field"


class InvalidTimeoutError(BrowserValidationError):
    default_code = "BRW-1305"
    default_message = "Invalid timeout value"


class BrowserDriverError(BrowserError):
    error_type = BrowserErrorType.DRIVER
    default_code = "BRW-1400"
    default_message = "Browser driver error"
    default_severity = BrowserErrorSeverity.HIGH.value
    default_retryable = True
    default_category = "browser.driver"


class BrowserTimeoutError(BrowserDriverError):
    default_code = "BRW-1401"
    default_message = "Browser operation timed out"


class BrowserSessionError(BrowserDriverError):
    default_code = "BRW-1402"
    default_message = "Browser session is unavailable or invalid"


class BrowserWindowError(BrowserDriverError):
    default_code = "BRW-1403"
    default_message = "Browser window or tab is unavailable"


# ---------------------------------------------------------------------------
# Browsing operation errors
# ---------------------------------------------------------------------------
class NavigationError(BrowserError):
    error_type = BrowserErrorType.NAVIGATION
    default_code = "BRW-1500"
    default_message = "Navigation failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "browser.navigation"


class PageLoadTimeoutError(NavigationError):
    default_code = "BRW-1501"
    default_message = "Page load timed out"


class NavigationHistoryError(NavigationError):
    default_code = "BRW-1502"
    default_message = "Navigation history operation failed"


class RedirectError(NavigationError):
    default_code = "BRW-1503"
    default_message = "Unexpected redirect during navigation"


class SearchError(BrowserError):
    error_type = BrowserErrorType.SEARCH
    default_code = "BRW-1600"
    default_message = "Search operation failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "browser.search"


class SearchBoxNotFoundError(SearchError):
    default_code = "BRW-1601"
    default_message = "Could not locate search box"


class SearchResultsNotFoundError(SearchError):
    default_code = "BRW-1602"
    default_message = "Could not locate search results"


class CookieConsentError(SearchError):
    default_code = "BRW-1603"
    default_message = "Cookie consent handling failed"
    default_retryable = False


class ElementError(BrowserError):
    error_type = BrowserErrorType.ELEMENT
    default_code = "BRW-1700"
    default_message = "Element operation failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "browser.element"


class ElementNotFoundError(ElementError):
    default_code = "BRW-1701"
    default_message = "Element not found"


class ElementNotVisibleError(ElementError):
    default_code = "BRW-1702"
    default_message = "Element is not visible"


class ElementNotInteractableError(ElementError):
    default_code = "BRW-1703"
    default_message = "Element is not interactable"


class StaleElementError(ElementError):
    default_code = "BRW-1704"
    default_message = "Element reference became stale"


class ShadowDomError(ElementError):
    default_code = "BRW-1705"
    default_message = "Shadow DOM interaction failed"


class ClickError(BrowserError):
    error_type = BrowserErrorType.CLICK
    default_code = "BRW-1800"
    default_message = "Click operation failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "browser.click"


class ClickInterceptedError(ClickError):
    default_code = "BRW-1801"
    default_message = "Click was intercepted by another element"


class JavaScriptClickError(ClickError):
    default_code = "BRW-1802"
    default_message = "JavaScript click fallback failed"


class SpecialElementHandlingError(ClickError):
    default_code = "BRW-1803"
    default_message = "Special element handling failed"


class BrowserTypingError(BrowserError):
    error_type = BrowserErrorType.TYPE
    default_code = "BRW-1900"
    default_message = "Typing operation failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "browser.type"


class InputClearError(BrowserTypingError):
    default_code = "BRW-1901"
    default_message = "Could not clear input before typing"


class InputSendKeysError(BrowserTypingError):
    default_code = "BRW-1902"
    default_message = "Could not send keys to input"


class ScrollError(BrowserError):
    error_type = BrowserErrorType.SCROLL
    default_code = "BRW-2000"
    default_message = "Scroll operation failed"
    default_severity = BrowserErrorSeverity.LOW.value
    default_retryable = True
    default_category = "browser.scroll"


class InvalidScrollTargetError(ScrollError):
    default_code = "BRW-2001"
    default_message = "Invalid scroll target"
    default_retryable = False


class ClipboardError(BrowserError):
    error_type = BrowserErrorType.CLIPBOARD
    default_code = "BRW-2100"
    default_message = "Clipboard operation failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "browser.clipboard"


class CopyError(ClipboardError):
    default_code = "BRW-2101"
    default_message = "Copy operation failed"


class CutError(ClipboardError):
    default_code = "BRW-2102"
    default_message = "Cut operation failed"


class PasteError(ClipboardError):
    default_code = "BRW-2103"
    default_message = "Paste operation failed"


class ContentExtractionError(BrowserError):
    error_type = BrowserErrorType.CONTENT
    default_code = "BRW-2200"
    default_message = "Content extraction failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "browser.content"


class PDFExtractionError(ContentExtractionError):
    default_code = "BRW-2201"
    default_message = "PDF extraction failed"


class ArxivExtractionError(ContentExtractionError):
    default_code = "BRW-2202"
    default_message = "arXiv content extraction failed"


class PageSnapshotError(ContentExtractionError):
    default_code = "BRW-2203"
    default_message = "Page snapshot extraction failed"


class UnsupportedContentTypeError(ContentExtractionError):
    default_code = "BRW-2204"
    default_message = "Unsupported content type"
    default_retryable = False


# ---------------------------------------------------------------------------
# Security, workflow, retry, task, and state errors
# ---------------------------------------------------------------------------
class BrowserSecurityError(BrowserError):
    error_type = BrowserErrorType.SECURITY
    default_code = "BRW-2300"
    default_message = "Browser security check failed"
    default_severity = BrowserErrorSeverity.HIGH.value
    default_retryable = False
    default_category = "browser.security"


class CaptchaDetectedError(BrowserSecurityError):
    default_code = "BRW-2301"
    default_message = "CAPTCHA challenge detected"


class BotDetectionError(BrowserSecurityError):
    default_code = "BRW-2302"
    default_message = "Automated browsing appears to be blocked"


class RateLimitError(BrowserSecurityError):
    default_code = "BRW-2303"
    default_message = "Rate limit encountered"
    default_retryable = True
    default_retry_after_seconds = 60.0


class PermissionDeniedError(BrowserSecurityError):
    default_code = "BRW-2304"
    default_message = "Permission denied during browser operation"


class WorkflowError(BrowserError):
    error_type = BrowserErrorType.WORKFLOW
    default_code = "BRW-2400"
    default_message = "Workflow execution failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "browser.workflow"


class WorkflowValidationError(WorkflowError):
    default_code = "BRW-2401"
    default_message = "Workflow validation failed"


class UnsupportedWorkflowActionError(WorkflowValidationError):
    default_code = "BRW-2402"
    default_message = "Unsupported workflow action"


class WorkflowStepFailedError(WorkflowError):
    default_code = "BRW-2403"
    default_message = "Workflow step failed"
    default_retryable = True


class RetryError(BrowserError):
    error_type = BrowserErrorType.RETRY
    default_code = "BRW-2500"
    default_message = "Retry orchestration failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "browser.retry"


class RetryExhaustedError(RetryError):
    default_code = "BRW-2501"
    default_message = "Retry attempts exhausted"


class BackoffError(RetryError):
    default_code = "BRW-2502"
    default_message = "Backoff execution failed"


class NetworkError(BrowserError):
    error_type = BrowserErrorType.NETWORK
    default_code = "BRW-2600"
    default_message = "Network operation failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "browser.network"


class NetworkTimeoutError(NetworkError):
    default_code = "BRW-2601"
    default_message = "Network request timed out"


class HTTPRequestError(NetworkError):
    default_code = "BRW-2602"
    default_message = "HTTP request failed"


class BrowserTaskError(BrowserError):
    error_type = BrowserErrorType.TASK
    default_code = "BRW-2700"
    default_message = "Browser task failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = False
    default_category = "browser.task"


class UnsupportedBrowserTaskError(BrowserTaskError):
    default_code = "BRW-2701"
    default_message = "Unsupported browser task payload"


class BrowserStateError(BrowserError):
    error_type = BrowserErrorType.STATE
    default_code = "BRW-2800"
    default_message = "Browser state error"
    default_severity = BrowserErrorSeverity.HIGH.value
    default_retryable = True
    default_category = "browser.state"


class ClosedDriverError(BrowserStateError):
    default_code = "BRW-2801"
    default_message = "Browser driver is closed"


class MissingDriverError(BrowserStateError):
    default_code = "BRW-2802"
    default_message = "Browser driver is missing"


class JavaScriptExecutionError(BrowserError):
    error_type = BrowserErrorType.SCRIPT
    default_code = "BRW-2900"
    default_message = "JavaScript execution failed"
    default_severity = BrowserErrorSeverity.MEDIUM.value
    default_retryable = True
    default_category = "browser.script"


# ---------------------------------------------------------------------------
# Exception wrapping and result helpers
# ---------------------------------------------------------------------------
def _exception_name(exc: BaseException) -> str:
    return exc.__class__.__name__


def _map_exception_to_browser_error(exc: BaseException, *, action: Optional[str] = None) -> Optional[Type[BrowserError]]:
    name = _exception_name(exc)
    action_name = (action or "").lower().strip()

    selenium_map: Dict[str, Type[BrowserError]] = {
        "NoSuchElementException": ElementNotFoundError,
        "InvalidSelectorException": InvalidSelectorError,
        "ElementNotVisibleException": ElementNotVisibleError,
        "ElementNotInteractableException": ElementNotInteractableError,
        "StaleElementReferenceException": StaleElementError,
        "ElementClickInterceptedException": ClickInterceptedError,
        "JavascriptException": JavaScriptExecutionError,
        "NoSuchWindowException": BrowserWindowError,
        "NoSuchFrameException": BrowserWindowError,
        "NoSuchDriverException": MissingDriverError,
        "InvalidSessionIdException": BrowserSessionError,
        "SessionNotCreatedException": BrowserDriverStartupError,
        "WebDriverException": BrowserDriverError,
    }
    if name == "TimeoutException":
        if action_name in {"navigate", "open_url", "go_to_url", "refresh", "back", "forward"}:
            return PageLoadTimeoutError
        return BrowserTimeoutError
    if name in selenium_map:
        return selenium_map[name]

    request_timeout_names = {"Timeout", "ReadTimeout", "ConnectTimeout", "TimeoutError"}
    request_error_names = {"HTTPError", "ConnectionError", "TooManyRedirects", "RequestException"}
    if name in request_timeout_names:
        return NetworkTimeoutError
    if name in request_error_names:
        return HTTPRequestError

    if isinstance(exc, PermissionError):
        return PermissionDeniedError
    if isinstance(exc, FileNotFoundError):
        return MissingConfigurationError
    if isinstance(exc, ValueError):
        if "url" in str(exc).lower():
            return InvalidURLError
        if "selector" in str(exc).lower():
            return InvalidSelectorError
        if "workflow" in str(exc).lower() or "unsupported" in str(exc).lower():
            return WorkflowValidationError
        return BrowserValidationError

    return None


def wrap_browser_exception(exc: BaseException, *, action: Optional[str] = None,
                           message: Optional[str] = None,
                           context: Optional[Mapping[str, Any]] = None,
                           default_error_cls: Type[BrowserError] = UnknownBrowserError) -> BrowserError:
    """Public wrapper for arbitrary exceptions."""
    return BrowserError.from_exception(
        exc,
        action=action,
        message=message,
        context=context,
        default_error_cls=default_error_cls,
    )


def error_result(error: Union[BrowserError, BaseException, str], *, action: Optional[str] = None,
                 context: Optional[Mapping[str, Any]] = None,
                 error_cls: Type[BrowserError] = UnknownBrowserError,
                 redact: bool = True) -> Dict[str, Any]:
    """Create a standard ``status=error`` result from an error-like object."""

    if isinstance(error, BrowserError):
        browser_error = error
        if context:
            browser_error.context.update(sanitize_context(context, redact=False))
    elif isinstance(error, BaseException):
        browser_error = wrap_browser_exception(error, action=action, context=context, default_error_cls=error_cls)
    else:
        browser_error = error_cls(str(error), context=context)
    return browser_error.to_result(action=action, redact=redact)


def raise_for_error_result(result: Mapping[str, Any], *, action: Optional[str] = None) -> None:
    """Raise a BrowserError when a module returns ``{"status": "error"}``."""

    if (result or {}).get("status") != "error":
        return
    error_payload = result.get("error") if isinstance(result, Mapping) else None
    message = str(result.get("message") or "Browser action returned an error")
    context = dict(result)
    if isinstance(error_payload, Mapping):
        code = str(error_payload.get("code") or "BRW-1000")
        error_type = _coerce_error_type(error_payload.get("type"))
        severity = error_payload.get("severity") or BrowserErrorSeverity.MEDIUM.value
        retryable = bool(error_payload.get("retryable", False))
        raise BrowserError(
            message,
            code=code,
            error_type=error_type,
            severity=severity,
            retryable=retryable,
            context=context,
        )
    raise BrowserTaskError(message, context={"action": action, "result": context})


def is_retryable(error: Union[BrowserError, BaseException, Mapping[str, Any]]) -> bool:
    """Return whether an error/result should be considered retryable."""

    if isinstance(error, BrowserError):
        return error.retryable
    if isinstance(error, Mapping):
        if "retryable" in error:
            return bool(error["retryable"])
        nested = error.get("error")
        if isinstance(nested, Mapping) and "retryable" in nested:
            return bool(nested["retryable"])
        return False
    return wrap_browser_exception(error).retryable


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def require(condition: bool, message: str, *, error_cls: Type[BrowserError] = BrowserValidationError, context: Optional[Mapping[str, Any]] = None) -> None:
    if not condition:
        raise error_cls(message, context=context)


def require_mapping(value: Any, field_name: str, *, allow_empty: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BrowserValidationError(f"{field_name} must be a mapping", context={"field": field_name, "value_type": type(value).__name__})
    if not allow_empty and not value:
        raise MissingRequiredFieldError(f"{field_name} cannot be empty", context={"field": field_name})
    return value


def require_sequence(value: Any, field_name: str, *, allow_empty: bool = True) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BrowserValidationError(f"{field_name} must be a sequence", context={"field": field_name, "value_type": type(value).__name__})
    if not allow_empty and not value:
        raise MissingRequiredFieldError(f"{field_name} cannot be empty", context={"field": field_name})
    return value


def require_non_empty_str(value: Any, field_name: str, *, error_cls: Type[BrowserError] = MissingRequiredFieldError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_cls(f"{field_name} must be a non-empty string", context={"field": field_name, "value": value})
    return value.strip()


def validate_url(url: Any, *, field_name: str = "url",
                 allowed_schemes: Iterable[str] = ("http", "https"),
                 require_netloc: bool = True) -> str:
    url_text = require_non_empty_str(url, field_name, error_cls=InvalidURLError)
    parsed = urlparse(url_text)
    allowed = {scheme.lower() for scheme in allowed_schemes}
    if parsed.scheme.lower() not in allowed:
        raise InvalidURLError(
            f"{field_name} must use one of: {sorted(allowed)}",
            context={"field": field_name, "url": url_text, "scheme": parsed.scheme},
        )
    if require_netloc and not parsed.netloc:
        raise InvalidURLError(f"{field_name} must include a host", context={"field": field_name, "url": url_text})
    return url_text


def validate_css_selector(selector: Any, *, field_name: str = "selector") -> str:
    selector_text = require_non_empty_str(selector, field_name, error_cls=InvalidSelectorError)
    # This is not a full CSS grammar parser. It catches the common malformed
    # cases early while leaving Selenium/the browser to perform final parsing.
    if "\x00" in selector_text:
        raise InvalidSelectorError("CSS selector contains a null byte", context={"field": field_name})
    if selector_text.count("[") != selector_text.count("]"):
        raise InvalidSelectorError("CSS selector has unbalanced attribute brackets", context={"field": field_name, "selector": selector_text})
    if selector_text.count("(") != selector_text.count(")"):
        raise InvalidSelectorError("CSS selector has unbalanced parentheses", context={"field": field_name, "selector": selector_text})
    return selector_text


def validate_timeout(value: Any, *, field_name: str = "timeout", minimum: float = 0.0, maximum: float = 300.0) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidTimeoutError(f"{field_name} must be numeric", context={"field": field_name, "value": value}, cause=exc) from exc
    if timeout < minimum or timeout > maximum:
        raise InvalidTimeoutError(
            f"{field_name} must be between {minimum} and {maximum} seconds",
            context={"field": field_name, "value": timeout, "minimum": minimum, "maximum": maximum},
        )
    return timeout


def validate_choice(value: Any, *, field_name: str, choices: Iterable[str], error_cls: Type[BrowserError] = BrowserValidationError) -> str:
    text = require_non_empty_str(value, field_name, error_cls=error_cls)
    valid = {choice.lower() for choice in choices}
    if text.lower() not in valid:
        raise error_cls(f"{field_name} must be one of: {sorted(valid)}", context={"field": field_name, "value": text, "choices": sorted(valid)})
    return text


def validate_config_section(section: Mapping[str, Any], *, section_name: str,
                            required_keys: Iterable[str] = ()) -> Mapping[str, Any]:
    require_mapping(section, section_name, allow_empty=True)
    missing = [key for key in required_keys if key not in section or section.get(key) is None]
    if missing:
        raise MissingConfigurationError(
            f"Missing required browser config keys for section '{section_name}'",
            context={"section": section_name, "missing_keys": missing},
        )
    return section


def validate_workflow_step(step: Any, *, index: int, supported_actions: Iterable[str]) -> Dict[str, Any]:
    require_mapping(step, f"workflow[{index}]", allow_empty=False)
    action = str(step.get("action", "")).lower().strip()
    supported = {item.lower().strip() for item in supported_actions}
    if not action:
        raise MissingRequiredFieldError("Workflow step is missing action", context={"index": index, "step": step})
    if action not in supported:
        raise UnsupportedWorkflowActionError(
            f"Unsupported workflow action at index {index}: '{action}'",
            context={"index": index, "action": action, "supported_actions": sorted(supported)},
        )
    params = step.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise WorkflowValidationError("Workflow step params must be a mapping", context={"index": index, "action": action, "params_type": type(params).__name__})
    return {"action": action, "params": dict(params)}


def validate_workflow_script(workflow_script: Any, *, supported_actions: Iterable[str]) -> Sequence[Dict[str, Any]]:
    steps = require_sequence(workflow_script, "workflow_script", allow_empty=True)
    return [validate_workflow_step(step, index=index, supported_actions=supported_actions) for index, step in enumerate(steps)]


def validate_browser_task_payload(task_data: Any) -> Mapping[str, Any]:
    payload = require_mapping(task_data, "task_data", allow_empty=False)
    if not any(key in payload for key in ("task", "workflow", "query", "url")):
        raise InvalidTaskPayloadError(
            "Browser task payload must include task, workflow, query, or url",
            context={"payload_keys": sorted(str(key) for key in payload.keys())},
        )
    return payload


# ---------------------------------------------------------------------------
# Convenience constructors for action modules
# ---------------------------------------------------------------------------
def element_not_found(selector: str, *, action: Optional[str] = None, timeout: Optional[float] = None) -> ElementNotFoundError:
    return ElementNotFoundError(
        f"Element not found: {selector}",
        context={"selector": selector, "action": action, "timeout": timeout},
    )


def captcha_detected(*, url: Optional[str] = None, action: Optional[str] = None) -> CaptchaDetectedError:
    return CaptchaDetectedError(context={"url": url, "action": action})


def retry_exhausted(operation_name: str, *, attempts: int,
    last_error: Optional[BaseException] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> RetryExhaustedError:
    merged_context = {"operation": operation_name, "attempts": attempts}
    if context:
        merged_context.update(dict(context))
    return RetryExhaustedError(
        f"Retry attempts exhausted for {operation_name}",
        context=merged_context,
        cause=last_error,
    )


# ---------------------------------------------------------------------------
# Drag-and-drop errors
# ---------------------------------------------------------------------------
_DRAG_ERROR_TYPE = getattr(BrowserErrorType, "DRAG_AND_DROP", getattr(BrowserErrorType, "ELEMENT", BrowserErrorType.UNKNOWN))


class DragAndDropError(BrowserError):
    """Base drag-and-drop error built on the shared browser error contract."""

    error_type = _DRAG_ERROR_TYPE
    default_code = "BRW-1850"
    default_message = "Drag-and-drop operation failed"
    default_severity = "medium"
    default_retryable = True
    default_category = "browser.drag_and_drop"


class DragAndDropValidationError(DragAndDropError):
    default_code = "BRW-1851"
    default_message = "Invalid drag-and-drop request"
    default_retryable = False


class DragSourceNotFoundError(DragAndDropError):
    default_code = "BRW-1852"
    default_message = "Drag source element was not found"


class DragTargetNotFoundError(DragAndDropError):
    default_code = "BRW-1853"
    default_message = "Drag target element was not found"


class DragSourceNotReadyError(DragAndDropError):
    default_code = "BRW-1854"
    default_message = "Drag source element is not ready"


class DragTargetNotReadyError(DragAndDropError):
    default_code = "BRW-1855"
    default_message = "Drag target element is not ready"


class DragStrategyError(DragAndDropError):
    default_code = "BRW-1856"
    default_message = "Drag-and-drop strategy failed"


class Html5DragAndDropError(DragStrategyError):
    default_code = "BRW-1857"
    default_message = "HTML5 drag-and-drop simulation failed"


class DragVerificationError(DragAndDropError):
    default_code = "BRW-1858"
    default_message = "Drag-and-drop verification failed"
    default_retryable = False


# ---------------------------------------------------------------------------
# Clipboard errors
# ---------------------------------------------------------------------------
class ClipboardValidationError(ClipboardError):
    """Raised when a copy/cut/paste request is structurally invalid."""

    default_code = "BRW-2104"
    default_message = "Invalid clipboard operation request"
    default_retryable = False


class ClipboardReadError(ClipboardError):
    """Raised when clipboard contents cannot be read."""

    default_code = "BRW-2105"
    default_message = "Clipboard read failed"
    default_retryable = False


class ClipboardWriteError(ClipboardError):
    """Raised when clipboard contents cannot be written."""

    default_code = "BRW-2106"
    default_message = "Clipboard write failed"
    default_retryable = False


class ClipboardStrategyError(ClipboardError):
    """Raised when a concrete clipboard strategy fails."""

    default_code = "BRW-2107"
    default_message = "Clipboard strategy failed"
    default_retryable = True


class ClipboardVerificationError(ClipboardError):
    """Raised when post-operation verification fails."""

    default_code = "BRW-2108"
    default_message = "Clipboard verification failed"
    default_retryable = False
