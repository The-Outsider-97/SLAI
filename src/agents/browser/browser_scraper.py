from __future__ import annotations

"""Bounded MediaWiki API scraper for the SLAI browser subsystem.

``BrowserScraper`` is intentionally narrower than ``ContentHandling`` and
``BrowserFunctions``.  It owns structured MediaWiki ``action=query`` requests,
request pacing, retry handling for transient HTTP/MediaWiki failures, bounded
JSON response reads, and MediaWiki continuation handling.  It does not own
browser-driver lifecycle, HTML/PDF extraction, workflow compilation, browser
security policy, persistent state, or knowledge ingestion.

The module keeps the public ``BrowserScraper(...).query(**params)`` contract
that existed in SLAI v2.3 while removing workflow-specific state that did not
belong to a scraper.
"""

import json
import sys
import threading
import time

from dataclasses import asdict, dataclass, field as dataclass_field
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .utils.browser_errors import *
from .utils.Browser_helpers import *
from .utils.config_loader import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from .browser_memory import BrowserMemory

logger = get_logger("Scraper")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API = "https://en.wikipedia.org/w/api.php"
SCRAPER_SCHEMA_VERSION = "1.0"
SCRAPER_ACTION = "scraper"

DEFAULT_USER_AGENT = "SLAI-BrowserScraper/2.3 (https://github.com/The-Outsider-97/SLAI)"
DEFAULT_RETRYABLE_HTTP_STATUSES: Tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504)
DEFAULT_RETRYABLE_API_CODES: Tuple[str, ...] = ("maxlag", "ratelimited", "readonly")
DEFAULT_MAX_RESPONSE_BYTES = 8_000_000
DEFAULT_MAX_CONTINUATION_PAGES = 25


# ---------------------------------------------------------------------------
# Public data contracts
# ---------------------------------------------------------------------------
class ScraperIssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ScraperIssueCode(str, Enum):
    """Stable scraper issue codes.

    The historical ``WScraperIssueCode`` name is retained for compatibility
    with the v2.3 public export even though the old values were workflow-copy
    artifacts.
    """

    INVALID_CONFIG = "invalid_config"
    INVALID_PARAMS = "invalid_params"
    INVALID_ENDPOINT = "invalid_endpoint"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    INVALID_RESPONSE = "invalid_response"
    RESPONSE_TOO_LARGE = "response_too_large"
    RETRY_EXHAUSTED = "retry_exhausted"
    CONTINUATION_LIMIT = "continuation_limit"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ScraperOptions:
    """Resolved runtime policy for MediaWiki API acquisition."""

    enabled: bool = True
    schema_version: str = SCRAPER_SCHEMA_VERSION
    api_url: str = API
    user_agent: str = DEFAULT_USER_AGENT
    delay_seconds: float = 1.0
    max_attempts: int = 6
    timeout_seconds: float = 60.0
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_continuation_pages: int = DEFAULT_MAX_CONTINUATION_PAGES
    maxlag: int = 5
    allow_http: bool = False
    allow_cross_host_redirects: bool = False
    retryable_http_statuses: Tuple[int, ...] = DEFAULT_RETRYABLE_HTTP_STATUSES
    retryable_api_codes: Tuple[str, ...] = DEFAULT_RETRYABLE_API_CODES
    backoff_base_delay: float = 1.0
    backoff_max_delay: float = 120.0
    backoff_multiplier: float = 2.0
    backoff_jitter: float = 0.25
    log_api_warnings: bool = True

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any]],
        *,
        delay_override: Optional[float] = None,
        retries_override: Optional[int] = None,
    ) -> "ScraperOptions":
        cfg = dict(config or {})
        request_cfg = dict(cfg.get("request") or {})
        limits_cfg = dict(cfg.get("limits") or {})
        retry_cfg = dict(cfg.get("retry") or {})
        diagnostics_cfg = dict(cfg.get("diagnostics") or {})

        retry_statuses = _coerce_int_tuple(
            retry_cfg.get("http_statuses", cfg.get("retryable_http_statuses")),
            default=DEFAULT_RETRYABLE_HTTP_STATUSES,
            minimum=100,
            maximum=599,
        )
        retry_api_codes = _coerce_str_tuple(
            retry_cfg.get("api_codes", cfg.get("retryable_api_codes")),
            default=DEFAULT_RETRYABLE_API_CODES,
        )

        delay_value = (
            delay_override
            if delay_override is not None
            else request_cfg.get("delay_seconds", cfg.get("delay_seconds", 1.0))
        )
        attempts_value = (
            retries_override
            if retries_override is not None
            else retry_cfg.get("max_attempts", cfg.get("max_attempts", 6))
        )

        return cls(
            enabled=coerce_bool(cfg.get("enabled", True), default=True),
            schema_version=str(cfg.get("schema_version") or SCRAPER_SCHEMA_VERSION),
            api_url=str(cfg.get("api_url") or API).strip(),
            user_agent=str(
                request_cfg.get(
                    "user_agent",
                    cfg.get("user_agent", DEFAULT_USER_AGENT),
                )
                or DEFAULT_USER_AGENT
            ).strip(),
            delay_seconds=coerce_float(
                delay_value,
                default=1.0,
                minimum=0.0,
                maximum=3600.0,
            ),
            max_attempts=coerce_int(
                attempts_value,
                default=6,
                minimum=1,
                maximum=50,
            ),
            timeout_seconds=coerce_float(
                request_cfg.get(
                    "timeout_seconds",
                    cfg.get("timeout_seconds", 60.0),
                ),
                default=60.0,
                minimum=0.1,
                maximum=600.0,
            ),
            max_response_bytes=coerce_int(
                limits_cfg.get(
                    "max_response_bytes",
                    cfg.get(
                        "max_response_bytes",
                        DEFAULT_MAX_RESPONSE_BYTES,
                    ),
                ),
                default=DEFAULT_MAX_RESPONSE_BYTES,
                minimum=1_024,
                maximum=250_000_000,
            ),
            max_continuation_pages=coerce_int(
                limits_cfg.get(
                    "max_continuation_pages",
                    cfg.get(
                        "max_continuation_pages",
                        DEFAULT_MAX_CONTINUATION_PAGES,
                    ),
                ),
                default=DEFAULT_MAX_CONTINUATION_PAGES,
                minimum=1,
                maximum=10_000,
            ),
            maxlag=coerce_int(
                cfg.get("maxlag", 5),
                default=5,
                minimum=0,
                maximum=60,
            ),
            allow_http=coerce_bool(
                request_cfg.get(
                    "allow_http",
                    cfg.get("allow_http", False),
                ),
                default=False,
            ),
            allow_cross_host_redirects=coerce_bool(
                request_cfg.get(
                    "allow_cross_host_redirects",
                    cfg.get("allow_cross_host_redirects", False),
                ),
                default=False,
            ),
            retryable_http_statuses=retry_statuses,
            retryable_api_codes=tuple(
                code.lower() for code in retry_api_codes
            ),
            backoff_base_delay=coerce_float(
                retry_cfg.get(
                    "base_delay",
                    cfg.get("backoff_base_delay", 1.0),
                ),
                default=1.0,
                minimum=0.0,
                maximum=3600.0,
            ),
            backoff_max_delay=coerce_float(
                retry_cfg.get(
                    "max_delay",
                    cfg.get("backoff_max_delay", 120.0),
                ),
                default=120.0,
                minimum=0.0,
                maximum=3600.0,
            ),
            backoff_multiplier=coerce_float(
                retry_cfg.get(
                    "multiplier",
                    cfg.get("backoff_multiplier", 2.0),
                ),
                default=2.0,
                minimum=1.0,
                maximum=10.0,
            ),
            backoff_jitter=coerce_float(
                retry_cfg.get(
                    "jitter",
                    cfg.get("backoff_jitter", 0.25),
                ),
                default=0.25,
                minimum=0.0,
                maximum=60.0,
            ),
            log_api_warnings=coerce_bool(
                diagnostics_cfg.get(
                    "log_api_warnings",
                    cfg.get("log_api_warnings", True),
                ),
                default=True,
            ),
        )


@dataclass(frozen=True)
class ScraperValidationIssue:
    """Structured scraper configuration or request validation issue."""

    severity: str
    code: str
    message: str
    field: Optional[str] = None
    context: Dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.severity == ScraperIssueSeverity.ERROR.value

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
class _RetryableMediaWikiError(Exception):
    def __init__(
        self,
        code: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.code = str(code or "unknown").lower()
        self.payload = dict(payload or {})
        super().__init__(
            f"Retryable MediaWiki API error: {self.code}"
        )


class _EndpointRedirectHandler(HTTPRedirectHandler):
    """Reject unsafe redirects before urllib connects to the redirect target."""

    def __init__(
        self,
        *,
        endpoint_host: str,
        allow_http: bool,
        allow_cross_host_redirects: bool,
    ) -> None:
        super().__init__()
        self._endpoint_host = endpoint_host.lower()
        self._allowed_schemes = (
            {"https"} | ({"http"} if allow_http else set())
        )
        self._allow_cross_host_redirects = (
            allow_cross_host_redirects
        )

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Optional[Request]:
        parsed = urlparse(newurl)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()

        if scheme not in self._allowed_schemes:
            raise HTTPRequestError(
                "Browser scraper rejected a redirect to a disallowed scheme",
                retryable=False,
                context={
                    "status_code": code,
                    "scheme": scheme,
                },
            )

        if (
            not self._allow_cross_host_redirects
            and host != self._endpoint_host
        ):
            raise HTTPRequestError(
                "Browser scraper rejected a cross-host redirect",
                retryable=False,
                context={
                    "status_code": code,
                    "expected_host": self._endpoint_host,
                    "redirect_host": host,
                },
            )

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


def _coerce_int_tuple(
    value: Any,
    *,
    default: Tuple[int, ...],
    minimum: int,
    maximum: int,
) -> Tuple[int, ...]:
    if value is None:
        return default

    if isinstance(value, (str, bytes)):
        raw_values = [
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        ]
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]

    normalized: List[int] = []

    for item in raw_values:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue

        if (
            minimum <= number <= maximum
            and number not in normalized
        ):
            normalized.append(number)

    return tuple(normalized) or default


def _coerce_str_tuple(
    value: Any,
    *,
    default: Tuple[str, ...],
) -> Tuple[str, ...]:
    if value is None:
        return default

    if isinstance(value, (str, bytes)):
        raw_values = [
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        ]
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]

    normalized: List[str] = []

    for item in raw_values:
        text = str(item).strip()

        if text and text not in normalized:
            normalized.append(text)

    return tuple(normalized) or default


def _parse_retry_after(
    value: Optional[str],
    *,
    max_seconds: float = 3600.0,
) -> float:
    """Parse Retry-After seconds or HTTP-date into a bounded delay."""

    if not value:
        return 0.0

    try:
        return min(
            max_seconds,
            max(0.0, float(value)),
        )
    except (TypeError, ValueError):
        pass

    try:
        parsed = parsedate_to_datetime(value)

        if parsed.tzinfo is None:
            return 0.0

        return min(
            max_seconds,
            max(
                0.0,
                parsed.timestamp() - time.time(),
            ),
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return 0.0


def _mapping_error_code(
    data: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    raw_error = data.get("error")

    if not isinstance(raw_error, Mapping):
        return "", {}

    payload = dict(raw_error)

    return (
        str(payload.get("code") or "unknown").lower(),
        payload,
    )


def _normalize_query_param(value: Any) -> Any:
    """Normalize common Python values to MediaWiki Action API parameter form."""

    if isinstance(value, set):
        return "|".join(
            str(item)
            for item in sorted(
                value,
                key=lambda item: str(item),
            )
        )

    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)

    return value


# ---------------------------------------------------------------------------
# BrowserScraper
# ---------------------------------------------------------------------------
class BrowserScraper:
    """Thread-safe, rate-paced MediaWiki ``action=query`` client.

    The class deliberately does not perform HTML scraping. Browser pages and
    arbitrary document extraction belong to ``ContentHandling``; Selenium
    lifecycle and navigation belong to ``BrowserDriver``/``BrowserFunctions``.

    ``memory`` is accepted for backward compatibility and composition, but the
    scraper does not create, persist, or mutate BrowserMemory implicitly. A
    caller that wants persistence remains responsible for explicitly delegating
    returned data to BrowserMemory or KnowledgeAgent.
    """

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        memory: Optional["BrowserMemory"] = None,
        delay: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> None:
        self.config, self.scraper_config = self._resolve_config(
            config
        )

        self.options = ScraperOptions.from_config(
            self.scraper_config,
            delay_override=delay,
            retries_override=retries,
        )

        allowed_schemes = (
            ("http", "https")
            if self.options.allow_http
            else ("https",)
        )

        try:
            self.api_url = validate_url(
                self.options.api_url,
                field_name="browser_scraper.api_url",
                allowed_schemes=allowed_schemes,
            )
        except Exception as exc:
            raise BrowserConfigurationError(
                "Invalid browser scraper API endpoint",
                context={
                    "api_url": self.options.api_url,
                    "allowed_schemes": allowed_schemes,
                },
                cause=exc,
            ) from exc

        parsed_endpoint = urlparse(self.api_url)
        self._endpoint_host = (
            parsed_endpoint.hostname or ""
        ).lower()

        if not self._endpoint_host:
            raise BrowserConfigurationError(
                "Browser scraper API endpoint must contain a hostname",
                context={
                    "api_url": self.api_url,
                },
            )

        self._opener = build_opener(
            _EndpointRedirectHandler(
                endpoint_host=self._endpoint_host,
                allow_http=self.options.allow_http,
                allow_cross_host_redirects=(
                    self.options.allow_cross_host_redirects
                ),
            )
        )

        self.memory = memory

        # Backward-compatible attribute name from the unfinished v2.3 module.
        self.reasoning_memory = memory

        self.delay = self.options.delay_seconds
        self.retries = self.options.max_attempts
        self.last = 0.0

        self._lock = threading.RLock()

        self._request_count = 0
        self._success_count = 0
        self._retry_count = 0

        self._last_error: Optional[str] = None

        logger.info(
            "Browser Scraper initialized | "
            "endpoint_host=%s | "
            "attempts=%s | "
            "delay=%.3fs | "
            "max_response_bytes=%s",
            self._endpoint_host,
            self.options.max_attempts,
            self.options.delay_seconds,
            self.options.max_response_bytes,
        )

    @staticmethod
    def _resolve_config(
        config: Optional[Mapping[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resolve both whole-config and section-only constructor inputs.

        Sibling browser modules accept section-specific overrides, while the
        unfinished scraper treated ``config`` as a whole browser config. Both
        shapes are accepted here so callers are not forced into a compatibility
        break.
        """

        if config is None:
            global_config = dict(load_global_config())

            return (
                global_config,
                dict(
                    get_config_section(
                        "browser_scraper",
                        global_config,
                        default={},
                    )
                    or {}
                ),
            )

        supplied = dict(config)

        if "browser_scraper" in supplied:
            return (
                supplied,
                dict(
                    get_config_section(
                        "browser_scraper",
                        supplied,
                        default={},
                    )
                    or {}
                ),
            )

        return supplied, supplied

    def _ensure_enabled(self) -> None:
        if not self.options.enabled:
            raise BrowserConfigurationError(
                "Browser scraper is disabled",
                context={
                    "section": "browser_scraper",
                },
            )

    def _wait_for_request_slot(self) -> None:
        """Enforce a minimum interval between request starts across threads."""

        with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self.options.delay_seconds - (now - self.last))

            if wait_seconds > 0:
                time.sleep(wait_seconds)

            self.last = time.monotonic()
            self._request_count += 1

    def _build_query_url(self, params: Mapping[str, Any]) -> str:
        query_params: Dict[str, Any] = dict(params or {})

        # BrowserScraper intentionally exposes only MediaWiki action=query.
        query_params["action"] = "query"
        query_params["format"] = "json"
        query_params.setdefault("formatversion", 2)
        query_params.setdefault("maxlag", self.options.maxlag)

        # MediaWiki multi-value parameters use pipe-separated values. Sorting
        # mapping keys and set values keeps equivalent inputs deterministic.
        normalized_params = {
            key: _normalize_query_param(value)
            for key, value in query_params.items()
        }

        encoded = urlencode(sorted(normalized_params.items(), key=lambda item: str(item[0])))

        return f"{self.api_url}?{encoded}"

    def _validate_final_url(self, final_url: str) -> None:
        allowed_schemes = (
            ("http", "https")
            if self.options.allow_http
            else ("https",)
        )

        normalized = validate_url(
            final_url,
            field_name="response_url",
            allowed_schemes=allowed_schemes,
        )

        final_host = (urlparse(normalized).hostname or "").lower()

        if (
            not self.options.allow_cross_host_redirects
            and final_host != self._endpoint_host
        ):
            raise HTTPRequestError(
                "Browser scraper rejected a cross-host redirect",
                retryable=False,
                context={
                    "expected_host": self._endpoint_host,
                    "response_host": final_host,
                },
            )

    def _read_json_response(self, response: Any) -> Dict[str, Any]:
        self._validate_final_url(str(response.geturl() or self.api_url))
        content_length = response.headers.get("Content-Length")

        if content_length:
            try:
                announced_size = int(content_length)
            except (TypeError, ValueError):
                announced_size = 0

            if (
                announced_size
                > self.options.max_response_bytes
            ):
                raise HTTPRequestError(
                    "Browser scraper response exceeds configured size limit",
                    retryable=False,
                    context={
                        "content_length": announced_size,
                        "max_response_bytes": (
                            self.options.max_response_bytes
                        ),
                    },
                )

        body = response.read(
            self.options.max_response_bytes + 1
        )

        if len(body) > self.options.max_response_bytes:
            raise HTTPRequestError(
                "Browser scraper response exceeded configured size limit "
                "while reading",
                retryable=False,
                context={
                    "bytes_read": len(body),
                    "max_response_bytes": (
                        self.options.max_response_bytes
                    ),
                },
            )

        charset = (
            response.headers.get_content_charset()
            or "utf-8"
        )

        try:
            text = body.decode(
                charset,
                errors="strict",
            )
        except (
            LookupError,
            UnicodeDecodeError,
        ) as exc:
            raise HTTPRequestError(
                "Browser scraper could not decode API response",
                retryable=False,
                context={
                    "charset": charset,
                    "response_bytes": len(body),
                },
                cause=exc,
            ) from exc

        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPRequestError(
                "Browser scraper received invalid JSON",
                retryable=False,
                context={
                    "response_bytes": len(body),
                },
                cause=exc,
            ) from exc

        if not isinstance(decoded, dict):
            raise HTTPRequestError(
                "Browser scraper expected a JSON object response",
                retryable=False,
                context={
                    "response_type": type(decoded).__name__,
                },
            )

        return decoded

    def _request_once(self, url: str) -> Dict[str, Any]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.options.user_agent,
            },
            method="GET",
        )

        with self._opener.open(request, timeout=self.options.timeout_seconds) as response:
            return self._read_json_response(response)

    def _retry_delay(self, attempt_index: int, *, retry_after: float = 0.0) -> float:
        calculated = calculate_backoff_delay(
            attempt_index=attempt_index,
            base_delay=self.options.backoff_base_delay,
            max_delay=self.options.backoff_max_delay,
            multiplier=self.options.backoff_multiplier,
            jitter=self.options.backoff_jitter,
        )

        return max(0.0, retry_after, calculated)

    def _inspect_api_payload(self, data: Mapping[str, Any]) -> None:
        error_code, error_payload = _mapping_error_code(data)

        if error_code:
            if (
                error_code
                in self.options.retryable_api_codes
            ):
                raise _RetryableMediaWikiError(error_code, error_payload)

            raise HTTPRequestError(
                "MediaWiki API returned non-retryable "
                f"error '{error_code}'",
                retryable=False,
                context={
                    "api_error": error_payload,
                },
            )

        warnings = data.get("warnings")

        if warnings and self.options.log_api_warnings:
            logger.warning(
                "MediaWiki API warning: %s",
                redact_mapping(
                    {
                        "warnings": warnings,
                    }
                ),
            )

        if "query" not in data:
            raise HTTPRequestError(
                "MediaWiki API response is missing query data",
                retryable=False,
                context={
                    "response_keys": sorted(
                        str(key)
                        for key in data.keys()
                    ),
                },
            )

    def query(self, **params: Any) -> Dict[str, Any]:
        """
        Execute one bounded MediaWiki ``action=query`` request.

        Transient HTTP statuses, transport failures, and MediaWiki ``maxlag``,
        ``ratelimited``, and ``readonly`` responses are retried. Permanent API
        errors fail immediately. The returned dictionary is the raw MediaWiki
        JSON object, preserving the v2.3 behavior expected by callers.
        """

        self._ensure_enabled()

        url = self._build_query_url(params)

        last_error: Optional[BaseException] = None

        for attempt in range(
            self.options.max_attempts
        ):
            self._wait_for_request_slot()

            retry_after = 0.0

            try:
                data = self._request_once(url)

                self._inspect_api_payload(data)

                with self._lock:
                    self._success_count += 1
                    self._last_error = None

                logger.debug(
                    "Browser scraper query succeeded | "
                    "attempt=%s/%s | keys=%s",
                    attempt + 1,
                    self.options.max_attempts,
                    sorted(
                        str(key)
                        for key in data.keys()
                    ),
                )

                return data

            except HTTPError as exc:
                retry_after = _parse_retry_after(
                    exc.headers.get("Retry-After")
                    if exc.headers
                    else None
                )

                exc.close()

                if (
                    exc.code
                    not in self.options.retryable_http_statuses
                ):
                    raise HTTPRequestError(
                        "Browser scraper HTTP request "
                        f"failed with status {exc.code}",
                        retryable=False,
                        context={
                            "status_code": exc.code,
                            "host": self._endpoint_host,
                        },
                        cause=exc,
                    ) from exc

                last_error = HTTPRequestError(
                    f"Retryable HTTP status {exc.code}",
                    retryable=True,
                    retry_after_seconds=(
                        retry_after or None
                    ),
                    context={
                        "status_code": exc.code,
                        "host": self._endpoint_host,
                    },
                    cause=exc,
                )

            except _RetryableMediaWikiError as exc:
                last_error = HTTPRequestError(
                    "Retryable MediaWiki API "
                    f"error '{exc.code}'",
                    retryable=True,
                    context={"api_error": exc.payload},
                    cause=exc,
                )

            except TimeoutError as exc:
                last_error = NetworkTimeoutError(
                    "Browser scraper request timed out",
                    retryable=True,
                    context={
                        "host": self._endpoint_host,
                        "timeout_seconds": self.options.timeout_seconds,
                    },
                    cause=exc,
                )

            except URLError as exc:
                reason = getattr(exc, "reason", None)

                if isinstance(reason, TimeoutError):
                    last_error = NetworkTimeoutError(
                        "Browser scraper request timed out",
                        retryable=True,
                        context={
                            "host": self._endpoint_host,
                            "timeout_seconds": (
                                self.options.timeout_seconds
                            ),
                        },
                        cause=exc,
                    )
                else:
                    last_error = NetworkError(
                        "Browser scraper network request failed",
                        retryable=True,
                        context={"host": self._endpoint_host},
                        cause=exc,
                    )

            except OSError as exc:
                last_error = NetworkError(
                    "Browser scraper network request failed",
                    retryable=True,
                    context={"host": self._endpoint_host},
                    cause=exc,
                )

            except (
                HTTPRequestError,
                BrowserValidationError,
            ):
                # Structured permanent errors raised by response validation
                # must not be converted into retries.
                raise

            with self._lock:
                self._last_error = (
                    str(last_error)
                    if last_error
                    else "unknown request failure"
                )

            if (
                attempt + 1
                >= self.options.max_attempts
            ):
                break

            pause = self._retry_delay(attempt, retry_after=retry_after)

            with self._lock:
                self._retry_count += 1

            logger.warning(
                "Browser scraper request failed; retrying | attempt=%s/%s | delay=%.2fs | error=%s",
                attempt + 1,
                self.options.max_attempts,
                pause,
                (
                    type(last_error).__name__
                    if last_error
                    else "UnknownError"
                ),
            )

            if pause > 0:
                time.sleep(pause)

        raise RetryExhaustedError(
            "Browser scraper request failed after "
            f"{self.options.max_attempts} attempts",
            context={
                "host": self._endpoint_host,
                "attempts": self.options.max_attempts,
            },
            cause=last_error,
        ) from last_error

    def iter_query(self, *, max_pages: Optional[int] = None, **params: Any) -> Iterator[Dict[str, Any]]:
        """
        Yield MediaWiki query responses while following continuation tokens.

        Continuation is explicitly bounded. The method does not merge arbitrary
        MediaWiki query result shapes because different query modules have
        different merge semantics; callers receive each canonical raw response.
        """

        page_limit = coerce_int(
            (
                max_pages
                if max_pages is not None
                else self.options.max_continuation_pages
            ),
            default=self.options.max_continuation_pages, minimum=1, maximum=10_000)

        next_params: Dict[str, Any] = dict(params)

        for page_index in range(page_limit):
            data = self.query(**next_params)

            yield data

            continuation = data.get("continue")

            if (
                not isinstance(continuation, Mapping)
                or not continuation
            ):
                return

            next_params = dict(params)
            next_params.update(dict(continuation))

            if page_index + 1 == page_limit:
                logger.warning("Browser scraper continuation limit reached | pages=%s | host=%s", page_limit, self._endpoint_host)

    def query_all(self, *, max_pages: Optional[int] = None, **params: Any) -> List[Dict[str, Any]]:
        """Return all bounded continuation pages as a list of raw responses."""
        return list(self.iter_query(max_pages=max_pages, **params))

    def status(self) -> Dict[str, Any]:
        """Return compact non-sensitive runtime diagnostics."""

        with self._lock:
            return {
                "enabled": self.options.enabled,
                "schema_version": self.options.schema_version,
                "endpoint_host": self._endpoint_host,
                "request_count": self._request_count,
                "success_count": self._success_count,
                "retry_count": self._retry_count,
                "last_error": self._last_error,
                "delay_seconds": self.options.delay_seconds,
                "max_attempts": self.options.max_attempts,
                "max_response_bytes": self.options.max_response_bytes,
                "max_continuation_pages": self.options.max_continuation_pages,
            }


def main() -> int:
    """Import/lifecycle smoke entrypoint; it intentionally performs no network I/O."""
    scraper = BrowserScraper()
    printer.status("SCRAPER", scraper.status(), "success")

    return 0


__all__ = [
    "API",
    "SCRAPER_SCHEMA_VERSION",
    "SCRAPER_ACTION",
    "ScraperIssueSeverity",
    "ScraperIssueCode",
    "ScraperOptions",
    "ScraperValidationIssue",
    "BrowserScraper",
    "main",
]


if __name__ == "__main__":
    printer.status("TEST", main, "info")
    # sys.exit(main())