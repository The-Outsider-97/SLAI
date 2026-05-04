from __future__ import annotations

"""
Production-grade navigation executor for the browser subsystem.

This module owns concrete navigation actions for Selenium-backed browser
sessions. It intentionally stays focused on navigation and leaves browser
lifecycle, planning, content extraction, and high-level orchestration to the
BrowserAgent and surrounding modules.

Design goals
------------
- Preserve the existing DoNavigate public API used by BrowserAgent:
  go_to_url(), go_back(), go_forward(), refresh_page(), get_current_url(), and
  get_navigation_history().
- Use the shared browser error hierarchy and helper utilities instead of
  duplicating validation, serialization, result construction, URL handling,
  page snapshots, redaction, timing, and retry/backoff behavior.
- Keep navigation behavior config-driven through browser_config.yaml.
- Provide structured diagnostics suitable for logs, agent memory, workflows,
  tests, and future telemetry/learning integrations.
- Remain conservative about side effects: navigation methods should only
  navigate/history/refresh the browser and record metadata about that action.
"""

import time as time_module

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from selenium.common.exceptions import TimeoutException, WebDriverException

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.browser_errors import *
from ..utils.Browser_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Navigate")
printer = PrettyPrinter()


@dataclass(frozen=True)
class NavigateOptions:
    """Runtime options for one navigation operation.

    Values are resolved from ``browser_config.yaml`` and may be overridden per
    method call. The class deliberately mirrors config names so deployments can
    tune behavior without editing this module.
    """

    timeout: float = 10.0
    poll_frequency: float = 0.1
    wait_before_execution: float = 0.0
    wait_after_navigation: float = 0.0
    retries: int = 1
    retry_backoff_base: float = 0.25
    retry_backoff_multiplier: float = 1.8
    retry_backoff_max: float = 3.0
    retry_jitter: float = 0.05
    default_scheme: str = "https"
    allowed_schemes: Tuple[str, ...] = ("http", "https")
    normalize_urls: bool = True
    remove_tracking_params: bool = True
    strip_fragment: bool = False
    require_netloc: bool = True
    wait_for_page_load: bool = True
    acceptable_ready_states: Tuple[str, ...] = ("complete",)
    verify_after_navigation: bool = True
    verify_url_available: bool = True
    verify_not_blank: bool = False
    allow_same_url_navigation: bool = True
    record_history: bool = True
    max_history_entries: int = 250
    include_page_snapshot: bool = False
    include_performance_timing: bool = True
    include_screenshot_on_error: bool = False
    include_url_parts: bool = True
    redact_urls_in_logs: bool = True
    history_capture_state: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **overrides: Any) -> "NavigateOptions":
        merged = dict(config or {})
        merged.update({key: value for key, value in overrides.items() if value is not None})

        allowed = ensure_list(merged.get("allowed_schemes", ("http", "https")))
        states = ensure_list(merged.get("acceptable_ready_states", ("complete",)))
        return cls(
            timeout=coerce_float(merged.get("timeout"), default=10.0, minimum=0.0, maximum=300.0),
            poll_frequency=coerce_float(merged.get("poll_frequency"), default=0.1, minimum=0.01, maximum=10.0),
            wait_before_execution=coerce_float(merged.get("wait_before_execution"), default=0.0, minimum=0.0, maximum=300.0),
            wait_after_navigation=coerce_float(merged.get("wait_after_navigation"), default=0.0, minimum=0.0, maximum=300.0),
            retries=coerce_int(merged.get("retries"), default=1, minimum=0, maximum=20),
            retry_backoff_base=coerce_float(merged.get("retry_backoff_base"), default=0.25, minimum=0.0, maximum=60.0),
            retry_backoff_multiplier=coerce_float(merged.get("retry_backoff_multiplier"), default=1.8, minimum=1.0, maximum=10.0),
            retry_backoff_max=coerce_float(merged.get("retry_backoff_max"), default=3.0, minimum=0.0, maximum=300.0),
            retry_jitter=coerce_float(merged.get("retry_jitter"), default=0.05, minimum=0.0, maximum=10.0),
            default_scheme=str(merged.get("default_scheme") or "https").strip() or "https",
            allowed_schemes=tuple(str(item).lower().strip() for item in allowed if str(item).strip()) or ("http", "https"),
            normalize_urls=coerce_bool(merged.get("normalize_urls"), default=True),
            remove_tracking_params=coerce_bool(merged.get("remove_tracking_params"), default=True),
            strip_fragment=coerce_bool(merged.get("strip_fragment"), default=False),
            require_netloc=coerce_bool(merged.get("require_netloc"), default=True),
            wait_for_page_load=coerce_bool(merged.get("wait_for_page_load"), default=True),
            acceptable_ready_states=tuple(str(item).strip() for item in states if str(item).strip()) or ("complete",),
            verify_after_navigation=coerce_bool(merged.get("verify_after_navigation"), default=True),
            verify_url_available=coerce_bool(merged.get("verify_url_available"), default=True),
            verify_not_blank=coerce_bool(merged.get("verify_not_blank"), default=False),
            allow_same_url_navigation=coerce_bool(merged.get("allow_same_url_navigation"), default=True),
            record_history=coerce_bool(merged.get("record_history"), default=True),
            max_history_entries=coerce_int(merged.get("max_history_entries"), default=250, minimum=1, maximum=100_000),
            include_page_snapshot=coerce_bool(merged.get("include_page_snapshot"), default=False),
            include_performance_timing=coerce_bool(merged.get("include_performance_timing"), default=True),
            include_screenshot_on_error=coerce_bool(merged.get("include_screenshot_on_error"), default=False),
            include_url_parts=coerce_bool(merged.get("include_url_parts"), default=True),
            redact_urls_in_logs=coerce_bool(merged.get("redact_urls_in_logs"), default=True),
            history_capture_state=coerce_bool(merged.get("history_capture_state"), default=True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationRequest:
    """Normalized request passed through the navigation execution pipeline."""

    action: str
    url: Optional[str] = None
    timeout: Optional[float] = None
    correlation_id: str = field(default_factory=lambda: new_correlation_id("nav"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationState:
    """Snapshot of lightweight browser state before or after navigation."""

    url: str = ""
    title: str = ""
    ready_state: Optional[str] = None
    captured_at: str = field(default_factory=utc_now_iso)
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationHistoryEntry:
    """Structured navigation history entry maintained by DoNavigate."""

    action: str
    requested_url: Optional[str]
    final_url: str
    title: str
    ready_state: Optional[str]
    success: bool
    correlation_id: str
    duration_ms: float
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DoNavigate:
    """Concrete browser navigation helper.

    The class wraps Selenium navigation primitives with validation, retry,
    page-load waiting, structured diagnostics, bounded history, and stable
    BrowserAgent-compatible result dictionaries.
    """

    def __init__(self, driver):
        self.config = load_global_config()
        self.navigate_config = get_config_section("do_navigate") or {}
        self.driver = driver
        self.history: List[NavigationHistoryEntry] = []
        logger.info("Web navigation functionality initiated.")

    # ------------------------------------------------------------------
    # Public API retained for BrowserAgent and existing callers
    # ------------------------------------------------------------------
    def go_to_url(self, url: str, **overrides: Any) -> dict:
        try:
            options = self._build_options(**overrides)
            prepared_url = self._prepare_url(url, options)
            request = NavigationRequest(action="navigate", url=prepared_url, timeout=options.timeout)
            return self._execute_navigation(request, options, lambda: self.driver.get(request.url))
        except InvalidURLError as exc:
            return self._error_response(
                exc,
                action="navigate",
                context={"url": url, "original_error": str(exc)},
            )

    def navigate(self, url: str, **overrides: Any) -> dict:
        """Alias for go_to_url for action-module consistency."""

        return self.go_to_url(url, **overrides)

    def open_url(self, url: str, **overrides: Any) -> dict:
        """Alias for go_to_url used by some browser tool naming schemes."""

        return self.go_to_url(url, **overrides)

    def go_back(self, **overrides: Any) -> dict:
        options = self._build_options(**overrides)
        request = NavigationRequest(action="back", timeout=options.timeout)
        return self._execute_navigation(request, options, self.driver.back)

    def back(self, **overrides: Any) -> dict:
        return self.go_back(**overrides)

    def go_forward(self, **overrides: Any) -> dict:
        options = self._build_options(**overrides)
        request = NavigationRequest(action="forward", timeout=options.timeout)
        return self._execute_navigation(request, options, self.driver.forward)

    def forward(self, **overrides: Any) -> dict:
        return self.go_forward(**overrides)

    def refresh_page(self, **overrides: Any) -> dict:
        options = self._build_options(**overrides)
        request = NavigationRequest(action="refresh", timeout=options.timeout)
        return self._execute_navigation(request, options, self.driver.refresh)

    def refresh(self, **overrides: Any) -> dict:
        return self.refresh_page(**overrides)

    def get_current_url(self) -> dict:
        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("nav-current-url")
        try:
            url = get_current_url(self.driver)
            data: Dict[str, Any] = {"url": url, "redacted_url": redact_url(url)}
            parsed = self._safe_url_parts(url)
            if parsed:
                data["url_parts"] = parsed
            return success_result(
                action="get_current_url",
                message="Current URL retrieved",
                data=data,
                duration_ms=elapsed_ms(start_ms),
                correlation_id=correlation_id,
            )
        except Exception as exc:
            return self._error_response(
                exc,
                action="get_current_url",
                context={"correlation_id": correlation_id},
                duration_ms=elapsed_ms(start_ms),
            )

    def get_current_title(self) -> dict:
        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("nav-current-title")
        try:
            return success_result(
                action="get_current_title",
                message="Current title retrieved",
                data={"title": get_page_title(self.driver)},
                duration_ms=elapsed_ms(start_ms),
                correlation_id=correlation_id,
            )
        except Exception as exc:
            return self._error_response(
                exc,
                action="get_current_title",
                context={"correlation_id": correlation_id},
                duration_ms=elapsed_ms(start_ms),
            )

    def get_navigation_history(self, *, as_dict: bool = True, limit: Optional[int] = None) -> list:
        """Return bounded in-memory navigation history.

        Existing callers expected a list, so the return type remains a list.
        By default it returns dictionaries, which are easier to serialize and log.
        """

        entries = self.history[-limit:] if limit else list(self.history)
        return [entry.to_dict() for entry in entries] if as_dict else entries

    def clear_navigation_history(self) -> dict:
        count = len(self.history)
        self.history.clear()
        return success_result(
            action="clear_navigation_history",
            message="Navigation history cleared",
            data={"cleared_entries": count},
        )

    def wait_until_loaded(self, timeout: Optional[float] = None, **overrides: Any) -> dict:
        """Public page-load wait helper for workflows and tests."""

        options = self._build_options(timeout=timeout, **overrides)
        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("nav-wait")
        try:
            loaded = self._wait_for_page_load(timeout=options.timeout, options=options)
            state = self._capture_state()
            if not loaded:
                raise PageLoadTimeoutError(
                    "Page load timed out",
                    context={
                        "timeout": options.timeout,
                        "ready_state": state.ready_state,
                        "acceptable_ready_states": options.acceptable_ready_states,
                    },
                )
            return success_result(
                action="wait_until_loaded",
                message="Page reached an acceptable ready state",
                data={"loaded": True, "state": state.to_dict()},
                duration_ms=elapsed_ms(start_ms),
                correlation_id=correlation_id,
            )
        except Exception as exc:
            return self._error_response(
                exc,
                action="wait_until_loaded",
                context={"correlation_id": correlation_id},
                duration_ms=elapsed_ms(start_ms),
                options=options,
            )

    # ------------------------------------------------------------------
    # Backward-compatible internal helpers
    # ------------------------------------------------------------------
    def _is_valid_url(self, url: str) -> bool:
        return is_valid_url(url, allowed_schemes=self._build_options().allowed_schemes, require_netloc=True)

    def _wait_for_page_load(self, timeout: float = 5, options: Optional[NavigateOptions] = None):
        resolved = options or self._build_options(timeout=timeout)
        return wait_for_page_load(
            self.driver,
            timeout=float(timeout),
            poll_interval=resolved.poll_frequency,
            acceptable_states=resolved.acceptable_ready_states,
        )

    # ------------------------------------------------------------------
    # Execution pipeline
    # ------------------------------------------------------------------
    def _build_options(self, **overrides: Any) -> NavigateOptions:
        return NavigateOptions.from_config(self.navigate_config, **overrides)

    def _prepare_url(self, url: str, options: NavigateOptions) -> str:
        if not isinstance(url, str) or not url.strip():
            raise InvalidURLError("URL must be a non-empty string", context={"url": url})

        prepared = url.strip()
        if options.normalize_urls:
            prepared = normalize_url(
                prepared,
                default_scheme=options.default_scheme,
                remove_tracking=options.remove_tracking_params,
                strip_fragment_value=options.strip_fragment,
            )
        if not is_valid_url(prepared, allowed_schemes=options.allowed_schemes, require_netloc=options.require_netloc):
            raise InvalidURLError(
                f"Invalid URL: {url}",
                context={
                    "url": url,
                    "prepared_url": prepared,
                    "allowed_schemes": options.allowed_schemes,
                    "require_netloc": options.require_netloc,
                },
            )
        return prepared

    def _execute_navigation(
        self,
        request: NavigationRequest,
        options: NavigateOptions,
        operation: Callable[[], Any],
    ) -> dict:
        start_ms = monotonic_ms()
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[BaseException] = None
        before_state = self._capture_state()

        for attempt_index in range(options.retries + 1):
            attempt_started = monotonic_ms()
            try:
                if options.wait_before_execution > 0:
                    time_module.sleep(options.wait_before_execution)

                operation()

                loaded = True
                if options.wait_for_page_load:
                    loaded = self._wait_for_page_load(timeout=options.timeout, options=options)
                    if not loaded:
                        raise PageLoadTimeoutError(
                            "Page load timed out after navigation",
                            context={
                                "action": request.action,
                                "url": request.url,
                                "timeout": options.timeout,
                                "attempt": attempt_index + 1,
                                "ready_state": get_document_ready_state(self.driver),
                            },
                        )

                if options.wait_after_navigation > 0:
                    time_module.sleep(options.wait_after_navigation)

                after_state = self._capture_state()
                self._verify_navigation(request, options, before_state, after_state)

                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "success",
                        "duration_ms": elapsed_ms(attempt_started),
                        "ready_state": after_state.ready_state,
                        "url": self._maybe_redact_url(after_state.url, options),
                    }
                )

                duration = elapsed_ms(start_ms)
                if options.record_history:
                    self._record_history(request, before_state, after_state, True, request.correlation_id, duration, options)

                data = self._build_success_data(request, options, before_state, after_state, attempts)
                return success_result(
                    action=request.action,
                    message=self._success_message(request.action, after_state.url),
                    data=data,
                    metadata={"navigation": {"attempts": attempts, "options": self._public_options(options)}},
                    duration_ms=duration,
                    correlation_id=request.correlation_id,
                )
            except Exception as exc:
                last_error = exc
                browser_error = wrap_browser_exception(
                    exc,
                    action=request.action,
                    context={
                        "action": request.action,
                        "url": request.url,
                        "attempt": attempt_index + 1,
                        "correlation_id": request.correlation_id,
                    },
                    default_error_cls=NavigationError,
                )
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "error",
                        "duration_ms": elapsed_ms(attempt_started),
                        "message": browser_error.message,
                        "code": browser_error.code,
                        "retryable": browser_error.retryable,
                    }
                )
                logger.warning(
                    "Navigation action %s failed on attempt %s: %s",
                    request.action,
                    attempt_index + 1,
                    browser_error.message,
                )
                if attempt_index >= options.retries or not browser_error.retryable:
                    break
                delay = calculate_backoff_delay(
                    attempt_index=attempt_index,
                    base_delay=options.retry_backoff_base,
                    max_delay=options.retry_backoff_max,
                    multiplier=options.retry_backoff_multiplier,
                    jitter=options.retry_jitter,
                )
                attempts[-1]["retry_delay_seconds"] = delay
                if delay > 0:
                    time_module.sleep(delay)

        duration = elapsed_ms(start_ms)
        after_state = self._capture_state()
        if options.record_history:
            self._record_history(request, before_state, after_state, False, request.correlation_id, duration, options)

        retry_error = RetryExhaustedError(
            f"Navigation action '{request.action}' failed after {len(attempts)} attempt(s)",
            context={
                "action": request.action,
                "url": request.url,
                "attempts": attempts,
                "correlation_id": request.correlation_id,
            },
            cause=last_error,
        )
        return self._error_response(
            retry_error,
            action=request.action,
            context={
                "request": request.to_dict(),
                "attempts": attempts,
                "before_state": before_state.to_dict(),
                "after_state": after_state.to_dict(),
            },
            duration_ms=duration,
            options=options,
        )

    def _verify_navigation(
        self,
        request: NavigationRequest,
        options: NavigateOptions,
        before_state: NavigationState,
        after_state: NavigationState,
    ) -> None:
        if not options.verify_after_navigation:
            return

        if options.verify_url_available and not after_state.url:
            raise NavigationError(
                "Navigation completed but current URL is unavailable",
                context={"action": request.action, "requested_url": request.url},
            )

        if options.verify_not_blank and after_state.url.startswith("about:blank"):
            raise NavigationError(
                "Navigation ended on a blank page",
                context={"action": request.action, "requested_url": request.url, "final_url": after_state.url},
            )

        if request.action == "navigate" and request.url and not options.allow_same_url_navigation:
            if before_state.url == after_state.url:
                raise NavigationError(
                    "Navigation did not change the current URL",
                    context={"requested_url": request.url, "before_url": before_state.url, "after_url": after_state.url},
                )

        if options.wait_for_page_load and after_state.ready_state not in options.acceptable_ready_states:
            raise PageLoadTimeoutError(
                "Page did not reach an acceptable ready state",
                context={
                    "action": request.action,
                    "ready_state": after_state.ready_state,
                    "acceptable_ready_states": options.acceptable_ready_states,
                },
            )

    # ------------------------------------------------------------------
    # State, history, and result construction
    # ------------------------------------------------------------------
    def _capture_state(self) -> NavigationState:
        url = get_current_url(self.driver)
        title = get_page_title(self.driver)
        ready_state = get_document_ready_state(self.driver)
        fingerprint = stable_hash({"url": url, "title": title, "ready_state": ready_state}, length=16)
        return NavigationState(url=url, title=title, ready_state=ready_state, fingerprint=fingerprint)

    def _record_history(
        self,
        request: NavigationRequest,
        before_state: NavigationState,
        after_state: NavigationState,
        success: bool,
        correlation_id: str,
        duration_ms: float,
        options: NavigateOptions,
    ) -> None:
        metadata: Dict[str, Any] = {
            "before_url": before_state.url,
            "before_title": before_state.title,
        }
        if options.history_capture_state:
            metadata["before_state"] = before_state.to_dict()
            metadata["after_state"] = after_state.to_dict()
        entry = NavigationHistoryEntry(
            action=request.action,
            requested_url=request.url,
            final_url=after_state.url,
            title=after_state.title,
            ready_state=after_state.ready_state,
            success=success,
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        self.history.append(entry)
        if len(self.history) > options.max_history_entries:
            del self.history[: len(self.history) - options.max_history_entries]

    def _build_success_data(
        self,
        request: NavigationRequest,
        options: NavigateOptions,
        before_state: NavigationState,
        after_state: NavigationState,
        attempts: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "url": after_state.url,
            "redacted_url": self._maybe_redact_url(after_state.url, options),
            "title": after_state.title,
            "ready_state": after_state.ready_state,
            "requested_url": request.url,
            "before": before_state.to_dict(),
            "after": after_state.to_dict(),
            "attempt_count": len(attempts),
        }
        if options.include_url_parts and after_state.url:
            data["url_parts"] = self._safe_url_parts(after_state.url)
        if options.include_page_snapshot:
            data["page_snapshot"] = page_snapshot_dict(
                self.driver,
                include_html=False,
                include_screenshot=False,
                max_text=2_000,
            )
        return data

    def _error_response(
        self,
        exc: BaseException,
        *,
        action: str,
        context: Optional[Mapping[str, Any]] = None,
        duration_ms: Optional[float] = None,
        options: Optional[NavigateOptions] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"context": safe_serialize(context or {})}
        if options is not None:
            metadata["options"] = self._public_options(options)
        if options is not None and options.include_screenshot_on_error:
            metadata["screenshot_b64"] = capture_screenshot_b64(self.driver)
        browser_error = wrap_browser_exception(exc, action=action, context=context, default_error_cls=NavigationError)
        return error_result(
            action=action,
            message=browser_error.message,
            error=browser_error,
            metadata=metadata,
            duration_ms=duration_ms,
        )

    def _safe_url_parts(self, url: str) -> Dict[str, Any]:
        try:
            parsed = parse_browser_url(url)
            return parsed.to_dict()
        except Exception:
            parsed = urlparse(url or "")
            return {
                "scheme": parsed.scheme,
                "host": parsed.hostname or "",
                "path": parsed.path,
                "query_present": bool(parsed.query),
                "fragment_present": bool(parsed.fragment),
            }

    def _maybe_redact_url(self, url: str, options: NavigateOptions) -> str:
        return redact_url(url) if options.redact_urls_in_logs else url

    def _public_options(self, options: NavigateOptions) -> Dict[str, Any]:
        data = options.to_dict()
        # These are operational diagnostics, not secrets, but keeping this path
        # centralized makes future sensitive option additions safe.
        return safe_serialize(data)

    @staticmethod
    def _success_message(action: str, final_url: str) -> str:
        if action == "navigate":
            return f"Navigated to {final_url}"
        if action == "back":
            return "Went back"
        if action == "forward":
            return "Went forward"
        if action == "refresh":
            return "Page refreshed"
        return f"Navigation action '{action}' completed"


if __name__ == "__main__":
    print("\n=== Running Do Navigate ===\n")
    printer.status("TEST", "Do Navigate initialized", "info")

    class FakeDriver:
        def __init__(self):
            self.urls = ["about:blank"]
            self.index = 0
            self.title = "Blank"
            self.ready_state = "complete"
            self.calls: List[str] = []

        @property
        def current_url(self):
            return self.urls[self.index]

        def get(self, url):
            self.calls.append(f"get:{url}")
            if self.index < len(self.urls) - 1:
                self.urls = self.urls[: self.index + 1]
            self.urls.append(url)
            self.index += 1
            self.title = f"Title for {url}"
            self.ready_state = "complete"

        def back(self):
            self.calls.append("back")
            if self.index > 0:
                self.index -= 1
            self.title = f"Title for {self.current_url}"

        def forward(self):
            self.calls.append("forward")
            if self.index < len(self.urls) - 1:
                self.index += 1
            self.title = f"Title for {self.current_url}"

        def refresh(self):
            self.calls.append("refresh")
            self.title = f"Refreshed {self.current_url}"

        def execute_script(self, script, *args):
            if "document.readyState" in script:
                return self.ready_state
            if "window.innerWidth" in script:
                return {"width": 1280, "height": 720, "devicePixelRatio": 1}
            if "performance.timing" in script or "performance.getEntriesByType" in script:
                return {}
            if "document.body" in script:
                return ""
            return None

        def find_element(self, by=None, value=None):
            class Body:
                text = "Fake body"

                def get_attribute(self, name):
                    if name == "outerHTML":
                        return "<body>Fake body</body>"
                    return None

            return Body()

        def get_screenshot_as_base64(self):
            return ""

    driver = FakeDriver()
    navigator = DoNavigate(driver)

    result = navigator.go_to_url("example.com/path?utm_source=test&x=1")
    assert result["status"] == "success", result
    assert result["data"]["url"].startswith("https://example.com/path"), result

    current = navigator.get_current_url()
    assert current["status"] == "success", current

    refresh = navigator.refresh_page()
    assert refresh["status"] == "success", refresh

    back = navigator.go_back()
    assert back["status"] == "success", back

    forward = navigator.go_forward()
    assert forward["status"] == "success", forward

    history = navigator.get_navigation_history()
    assert len(history) >= 4, history

    invalid = navigator.go_to_url("ftp://example.com/file")
    assert invalid["status"] == "error", invalid

    cleared = navigator.clear_navigation_history()
    assert cleared["status"] == "success", cleared

    print("\n=== Test ran successfully ===\n")
