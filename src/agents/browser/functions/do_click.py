from __future__ import annotations

"""
Production-grade click functionality for the browser subsystem.

This module owns concrete click execution only. It intentionally does not own
browser lifecycle, workflow orchestration, high-level task routing, content
extraction, or memory. Those responsibilities belong to the BrowserAgent and
adjacent browser modules. The click module focuses on one stable contract:
turn a selector into a safe, observable, configurable browser click result.

Design goals
------------
- Use shared browser errors and helpers instead of redefining result, redaction,
  serialization, validation, retry, or element-snapshot behavior.
- Keep backwards-compatible entry points used by BrowserAgent:
  ``do_click()``, ``_perform_click()``, and ``click_element()``.
- Make execution configurable from ``browser_config.yaml`` instead of burying
  timing, retry, fallback, scrolling, and diagnostics values in code.
- Support normal elements and special browser controls such as ``<option>``.
- Record enough metadata for debugging, memory, workflow replay, and telemetry,
  without leaking sensitive values.
- Remain easy to extend with additional click strategies, policy checks, and
  post-click verification signals.
"""

import asyncio
import time

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    JavascriptException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.browser_errors import *
from ..utils.Browser_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Click")
printer = PrettyPrinter


DEFAULT_CLICK_STRATEGIES: Tuple[str, ...] = (
    "native",
    "action_chains",
    "javascript",
    "dispatch_event",
    "keyboard",
)

SUPPORTED_CLICK_STRATEGIES = set(DEFAULT_CLICK_STRATEGIES)
SPECIAL_TAGS = {"option"}
CHECKABLE_INPUT_TYPES = {"checkbox", "radio"}
KEYBOARD_ACTIVATION_TAGS = {"button", "a", "summary", "input", "textarea", "select"}


@dataclass(frozen=True)
class ClickOptions:
    """Resolved execution policy for one click operation."""

    timeout: float = 10.0
    poll_frequency: float = 0.1
    wait_before_execution: float = 0.0
    wait_after_click: float = 0.0
    retries: int = 2
    retry_backoff_base: float = 0.25
    retry_backoff_multiplier: float = 1.8
    retry_backoff_max: float = 3.0
    retry_jitter: float = 0.05
    require_visible: bool = True
    require_enabled: bool = True
    require_clickable: bool = True
    scroll_into_view: bool = True
    scroll_position: str = "center"
    scroll_behavior: str = "auto"
    hover_before_click: bool = False
    focus_before_click: bool = False
    verify_after_click: bool = True
    wait_for_page_load_after_click: bool = False
    page_load_timeout: float = 5.0
    include_page_snapshot: bool = False
    include_element_html: bool = True
    max_element_text_chars: int = 500
    max_element_html_chars: int = 1_500
    strategies: Tuple[str, ...] = DEFAULT_CLICK_STRATEGIES
    screenshot_on_error: bool = False
    allow_javascript_fallback: bool = True
    allow_keyboard_fallback: bool = True
    allow_dispatch_event_fallback: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **overrides: Any) -> "ClickOptions":
        """Build options from ``browser_config.yaml`` plus per-call overrides."""

        strategies = overrides.pop("strategies", None) or config.get("strategies") or DEFAULT_CLICK_STRATEGIES
        resolved_strategies = normalize_click_strategies(strategies)

        values = {
            "timeout": coerce_float(overrides.pop("timeout", config.get("timeout", 10.0)), default=10.0, minimum=0.1),
            "poll_frequency": coerce_float(
                overrides.pop("poll_frequency", config.get("poll_frequency", 0.1)), default=0.1, minimum=0.05
            ),
            "wait_before_execution": coerce_float(
                overrides.pop("wait_before_execution", config.get("wait_before_execution", 0.0)), default=0.0, minimum=0.0
            ),
            "wait_after_click": coerce_float(
                overrides.pop("wait_after_click", config.get("wait_after_click", 0.0)), default=0.0, minimum=0.0
            ),
            "retries": coerce_int(overrides.pop("retries", config.get("retries", 2)), default=2, minimum=0),
            "retry_backoff_base": coerce_float(
                overrides.pop("retry_backoff_base", config.get("retry_backoff_base", 0.25)), default=0.25, minimum=0.0
            ),
            "retry_backoff_multiplier": coerce_float(
                overrides.pop("retry_backoff_multiplier", config.get("retry_backoff_multiplier", 1.8)), default=1.8, minimum=1.0
            ),
            "retry_backoff_max": coerce_float(
                overrides.pop("retry_backoff_max", config.get("retry_backoff_max", 3.0)), default=3.0, minimum=0.0
            ),
            "retry_jitter": coerce_float(overrides.pop("retry_jitter", config.get("retry_jitter", 0.05)), default=0.05, minimum=0.0),
            "require_visible": coerce_bool(overrides.pop("require_visible", config.get("require_visible", True)), default=True),
            "require_enabled": coerce_bool(overrides.pop("require_enabled", config.get("require_enabled", True)), default=True),
            "require_clickable": coerce_bool(overrides.pop("require_clickable", config.get("require_clickable", True)), default=True),
            "scroll_into_view": coerce_bool(overrides.pop("scroll_into_view", config.get("scroll_into_view", True)), default=True),
            "scroll_position": str(overrides.pop("scroll_position", config.get("scroll_position", "center")) or "center"),
            "scroll_behavior": str(overrides.pop("scroll_behavior", config.get("scroll_behavior", "auto")) or "auto"),
            "hover_before_click": coerce_bool(overrides.pop("hover_before_click", config.get("hover_before_click", False)), default=False),
            "focus_before_click": coerce_bool(overrides.pop("focus_before_click", config.get("focus_before_click", False)), default=False),
            "verify_after_click": coerce_bool(overrides.pop("verify_after_click", config.get("verify_after_click", True)), default=True),
            "wait_for_page_load_after_click": coerce_bool(
                overrides.pop("wait_for_page_load_after_click", config.get("wait_for_page_load_after_click", False)), default=False
            ),
            "page_load_timeout": coerce_float(
                overrides.pop("page_load_timeout", config.get("page_load_timeout", 5.0)), default=5.0, minimum=0.1
            ),
            "include_page_snapshot": coerce_bool(
                overrides.pop("include_page_snapshot", config.get("include_page_snapshot", False)), default=False
            ),
            "include_element_html": coerce_bool(
                overrides.pop("include_element_html", config.get("include_element_html", True)), default=True
            ),
            "max_element_text_chars": coerce_int(
                overrides.pop("max_element_text_chars", config.get("max_element_text_chars", 500)), default=500, minimum=0
            ),
            "max_element_html_chars": coerce_int(
                overrides.pop("max_element_html_chars", config.get("max_element_html_chars", 1500)), default=1500, minimum=0
            ),
            "screenshot_on_error": coerce_bool(
                overrides.pop("screenshot_on_error", config.get("screenshot_on_error", False)), default=False
            ),
            "allow_javascript_fallback": coerce_bool(
                overrides.pop("allow_javascript_fallback", config.get("allow_javascript_fallback", True)), default=True
            ),
            "allow_keyboard_fallback": coerce_bool(
                overrides.pop("allow_keyboard_fallback", config.get("allow_keyboard_fallback", True)), default=True
            ),
            "allow_dispatch_event_fallback": coerce_bool(
                overrides.pop("allow_dispatch_event_fallback", config.get("allow_dispatch_event_fallback", True)), default=True
            ),
            "strategies": resolved_strategies,
        }
        return cls(**values)


@dataclass(frozen=True)
class ClickRequest:
    """Normalized click request for validation, telemetry, and replay."""

    selector: str
    options: ClickOptions
    correlation_id: str = field(default_factory=lambda: new_correlation_id("click"))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["options"] = asdict(self.options)
        return payload


@dataclass
class ClickExecutionContext:
    """Mutable per-click execution context."""

    request: ClickRequest
    start_ms: float
    attempt: int = 0
    strategy: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    element: Optional[WebElement] = None

    def record_attempt(self, strategy: str, status: str, message: str, **metadata: Any) -> None:
        self.attempts.append(
            prune_none(
                {
                    "strategy": strategy,
                    "status": status,
                    "message": message,
                    "attempt": self.attempt,
                    "metadata": safe_serialize(metadata),
                }
            )
        )


def normalize_click_strategies(strategies: Iterable[Any]) -> Tuple[str, ...]:
    """Normalize click strategies and remove disabled/unknown values."""

    normalized: List[str] = []
    for strategy in ensure_list(strategies):
        name = str(strategy or "").strip().lower().replace("-", "_")
        if name in SUPPORTED_CLICK_STRATEGIES and name not in normalized:
            normalized.append(name)
    return tuple(normalized or DEFAULT_CLICK_STRATEGIES)


class DoClick:
    """Concrete Selenium click executor used by ``BrowserAgent``.

    The public contract returns dict payloads so callers can continue composing
    click results with workflow results and agent task results. Internally,
    failures are represented by browser-domain errors and converted through the
    shared helper result functions.
    """

    def __init__(self, driver: Any):
        self.driver = driver
        self.config = load_global_config()
        self.click_config = get_config_section("do_click") or {}
        self.options = ClickOptions.from_config(self.click_config)
        logger.info("Browser click functionality initiated.")

    async def do_click(self, selector: str, wait_before_execution: float = 0.0, **kwargs: Any) -> dict:
        """Asynchronously click an element by CSS selector."""

        return await asyncio.to_thread(self._perform_click, selector, wait_before_execution, **kwargs)

    def click(self, selector: str, wait_before_execution: Optional[float] = None, **kwargs: Any) -> dict:
        """Public synchronous click API with optional per-call overrides."""

        wait = self.options.wait_before_execution if wait_before_execution is None else wait_before_execution
        return self._perform_click(selector, wait, **kwargs)

    def _perform_click(self, selector: str, wait_before_execution: float = 0.0, **kwargs: Any) -> dict:
        """Backwards-compatible click entrypoint used by BrowserAgent."""

        start_ms = monotonic_ms()
        options = ClickOptions.from_config(self.click_config, wait_before_execution=wait_before_execution, **kwargs)
        try:
            request = self._build_request(selector, options)
            context = ClickExecutionContext(request=request, start_ms=start_ms)
            return self._execute_click(context)
        except Exception as exc:
            browser_error = self._coerce_click_error(
                exc,
                selector=selector,
                phase="perform_click",
                elapsed_ms=elapsed_ms(start_ms),
            )
            return self._error_result(browser_error, start_ms=start_ms, selector=selector, options=options)

    def _build_request(self, selector: str, options: ClickOptions) -> ClickRequest:
        normalized_selector = validate_selector(selector, field_name="selector")
        self._validate_options(options)
        return ClickRequest(selector=normalized_selector, options=options)

    def _validate_options(self, options: ClickOptions) -> None:
        if options.timeout <= 0:
            raise InvalidTimeoutError("Click timeout must be greater than zero", context={"timeout": options.timeout})
        if options.poll_frequency <= 0:
            raise InvalidTimeoutError("Click poll frequency must be greater than zero", context={"poll_frequency": options.poll_frequency})
        if not options.strategies:
            raise ClickError("At least one click strategy must be enabled", context={"strategies": options.strategies})
        invalid = [strategy for strategy in options.strategies if strategy not in SUPPORTED_CLICK_STRATEGIES]
        if invalid:
            raise ClickError("Unsupported click strategy configured", context={"invalid_strategies": invalid})

    def _execute_click(self, context: ClickExecutionContext) -> dict:
        request = context.request
        options = request.options

        if options.wait_before_execution > 0:
            time.sleep(options.wait_before_execution)

        for attempt in range(options.retries + 1):
            context.attempt = attempt + 1
            try:
                element = self._resolve_element(request.selector, options)
                context.element = element
                context.before = self._capture_before_state(element, options)
                self._prepare_element(element, request.selector, options)

                special_result = self._handle_special_elements(element, context)
                if special_result is not None:
                    return special_result

                result = self._attempt_click_strategies(element, context)
                return result
            except (StaleElementReferenceException, TimeoutException, ElementClickInterceptedException, ElementNotInteractableException, WebDriverException) as exc:
                context.record_attempt("retry", "error", str(exc), exception_type=exc.__class__.__name__)
                if attempt >= options.retries:
                    raise self._coerce_click_error(exc, selector=request.selector, phase="retry_exhausted", attempts=context.attempts) from exc
                self._sleep_before_retry(attempt, options)
            except BrowserError:
                raise
            except Exception as exc:
                raise self._coerce_click_error(exc, selector=request.selector, phase="execute_click", attempts=context.attempts) from exc

        raise RetryExhaustedError("Click retries exhausted", context={"selector": request.selector, "attempts": context.attempts})

    def _resolve_element(self, selector: str, options: ClickOptions) -> WebElement:
        """Wait for and return the target element according to click policy."""

        condition = EC.element_to_be_clickable((By.CSS_SELECTOR, selector)) if options.require_clickable else EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        try:
            element = WebDriverWait(self.driver, options.timeout, poll_frequency=options.poll_frequency).until(condition)
        except TimeoutException as exc:
            raise ElementNotFoundError(
                f"Element not found or not ready for click: {selector}",
                context={"selector": selector, "timeout": options.timeout, "require_clickable": options.require_clickable},
                cause=exc,
            ) from exc

        if element is None:
            raise ElementNotFoundError(f"Element not found: {selector}", context={"selector": selector})
        return element

    def _wait_for_element(self, selector: str, timeout: Optional[float] = None) -> Optional[WebElement]:
        """Backwards-compatible element wait helper."""

        resolved_timeout = self.options.timeout if timeout is None else timeout
        try:
            return WebDriverWait(self.driver, resolved_timeout, poll_frequency=self.options.poll_frequency).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, validate_selector(selector)))
            )
        except TimeoutException:
            return None

    def _prepare_element(self, element: WebElement, selector: str, options: Optional[ClickOptions] = None) -> None:
        """Prepare an element for click without changing click semantics."""

        resolved = options or self.options
        if resolved.scroll_into_view:
            self._scroll_into_view(element, resolved)
        if resolved.require_visible:
            self._ensure_visible(element, selector, resolved)
        if resolved.require_enabled:
            self._ensure_enabled(element, selector)
        if resolved.focus_before_click:
            self._focus_element(element)
        if resolved.hover_before_click:
            self._hover_element(element)

    def _scroll_into_view(self, element: WebElement, options: ClickOptions) -> None:
        position = options.scroll_position if options.scroll_position in {"start", "center", "end", "nearest"} else "center"
        behavior = options.scroll_behavior if options.scroll_behavior in {"auto", "smooth"} else "auto"
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block: arguments[1], inline: 'nearest', behavior: arguments[2]});",
            element,
            position,
            behavior,
        )

    def _ensure_visible(self, element: WebElement, selector: str, options: ClickOptions) -> None:
        try:
            WebDriverWait(self.driver, min(options.timeout, 2.0), poll_frequency=options.poll_frequency).until(EC.visibility_of(element))
        except TimeoutException as exc:
            raise ElementNotVisibleError("Element is not visible before click", context={"selector": selector}, cause=exc) from exc

    def _ensure_enabled(self, element: WebElement, selector: str) -> None:
        try:
            enabled = bool(element.is_enabled())
        except Exception as exc:
            raise ElementNotInteractableError("Could not determine whether element is enabled", context={"selector": selector}, cause=exc) from exc
        if not enabled or self._is_disabled_by_attribute(element):
            raise ElementNotInteractableError("Element is disabled and cannot be clicked", context={"selector": selector, "element": element_metadata(element)})

    def _is_disabled_by_attribute(self, element: WebElement) -> bool:
        disabled = safe_get_attribute(element, "disabled")
        aria_disabled = str(safe_get_attribute(element, "aria-disabled", default="")).lower()
        return disabled is not None or aria_disabled == "true"

    def _focus_element(self, element: WebElement) -> None:
        try:
            self.driver.execute_script("arguments[0].focus({preventScroll: true});", element)
        except JavascriptException as exc:
            logger.debug("Focus before click failed: %s", exc)

    def _hover_element(self, element: WebElement) -> None:
        ActionChains(self.driver).move_to_element(element).pause(0.05).perform()

    def _handle_special_elements(self, element: WebElement, context: ClickExecutionContext) -> Optional[dict]:
        tag = str(safe_call(lambda: element.tag_name, default="") or "").lower()
        if tag not in SPECIAL_TAGS:
            return None
        if tag == "option":
            return self._click_option_element(element, context)
        return None

    def _click_option_element(self, element: WebElement, context: ClickExecutionContext) -> dict:
        request = context.request
        try:
            value = safe_get_attribute(element, "value")
            parent = element.find_element(By.XPATH, "./..")
            if value is not None:
                Select(parent).select_by_value(value)
            else:
                Select(parent).select_by_visible_text(element_text(element, max_length=500))
            context.strategy = "select_option"
            context.record_attempt("select_option", "success", "Option selected")
            return self._success_result("Special option element selected", context)
        except Exception as exc:
            raise SpecialElementHandlingError(
                "Failed to handle special option element",
                context={"selector": request.selector, "element": element_metadata(element)},
                cause=exc,
            ) from exc

    def _attempt_click_strategies(self, element: WebElement, context: ClickExecutionContext) -> dict:
        request = context.request
        options = request.options
        last_error: Optional[BaseException] = None

        for strategy in options.strategies:
            if strategy == "javascript" and not options.allow_javascript_fallback:
                continue
            if strategy == "keyboard" and not options.allow_keyboard_fallback:
                continue
            if strategy == "dispatch_event" and not options.allow_dispatch_event_fallback:
                continue

            try:
                self._attempt_strategy(strategy, element)
                context.strategy = strategy
                context.record_attempt(strategy, "success", f"{strategy} click succeeded")
                if options.wait_after_click > 0:
                    time.sleep(options.wait_after_click)
                self._post_click_verification(element, context)
                return self._success_result(f"{self._strategy_label(strategy)} click succeeded", context)
            except Exception as exc:
                last_error = exc
                context.record_attempt(strategy, "error", str(exc), exception_type=exc.__class__.__name__)
                logger.debug("Click strategy %s failed for %s: %s", strategy, request.selector, exc)

        raise ClickError(
            f"All click strategies failed for selector: {request.selector}",
            context={"selector": request.selector, "attempts": context.attempts, "element": element_metadata(element)},
            cause=last_error,
        )

    def _attempt_strategy(self, strategy: str, element: WebElement) -> None:
        if strategy == "native":
            element.click()
            return
        if strategy == "action_chains":
            ActionChains(self.driver).move_to_element(element).pause(0.05).click(element).perform()
            return
        if strategy == "javascript":
            self.driver.execute_script("arguments[0].click();", element)
            return
        if strategy == "dispatch_event":
            self._dispatch_mouse_event(element)
            return
        if strategy == "keyboard":
            self._keyboard_activate(element)
            return
        raise ClickError("Unsupported click strategy", context={"strategy": strategy})

    def _dispatch_mouse_event(self, element: WebElement) -> None:
        self.driver.execute_script(
            """
            const element = arguments[0];
            const events = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'];
            for (const type of events) {
                const event = new MouseEvent(type, {
                    bubbles: true,
                    cancelable: true,
                    composed: true,
                    view: window,
                    button: 0,
                    buttons: type.endsWith('down') ? 1 : 0
                });
                element.dispatchEvent(event);
            }
            """,
            element,
        )

    def _keyboard_activate(self, element: WebElement) -> None:
        tag = str(safe_call(lambda: element.tag_name, default="") or "").lower()
        role = str(safe_get_attribute(element, "role", default="") or "").lower()
        input_type = str(safe_get_attribute(element, "type", default="") or "").lower()
        if tag not in KEYBOARD_ACTIVATION_TAGS and role not in {"button", "link", "menuitem", "tab", "checkbox", "radio"}:
            raise ElementNotInteractableError("Element is not keyboard-activatable", context={"tag": tag, "role": role})
        if tag == "input" and input_type not in CHECKABLE_INPUT_TYPES | {"button", "submit", "reset"}:
            raise ElementNotInteractableError("Input type is not keyboard-activatable", context={"input_type": input_type})
        element.send_keys(Keys.ENTER)

    def _post_click_verification(self, element: WebElement, context: ClickExecutionContext) -> None:
        options = context.request.options
        if options.wait_for_page_load_after_click:
            wait_for_page_load(self.driver, timeout=options.page_load_timeout)
        context.after = self._capture_after_state(element, options)
        if not options.verify_after_click:
            return
        verification = self._verify_click_effect(context.before, context.after, element)
        context.after["verification"] = verification

    def _verify_click_effect(self, before: Mapping[str, Any], after: Mapping[str, Any], element: WebElement) -> Dict[str, Any]:
        """Collect non-invasive signals that a click had an effect.

        Verification intentionally records signals instead of failing the click:
        many valid clicks open menus, trigger async UI, or start downloads without
        immediately changing URL, title, or element state.
        """

        signals = {
            "url_changed": before.get("url") != after.get("url"),
            "title_changed": before.get("title") != after.get("title"),
            "ready_state": after.get("ready_state"),
            "element_selected_changed": before.get("element_selected") != after.get("element_selected"),
            "element_value_changed": before.get("element_value") != after.get("element_value"),
            "active_element_changed": before.get("active_element_fingerprint") != after.get("active_element_fingerprint"),
        }
        signals["has_observable_effect"] = any(bool(value) for key, value in signals.items() if key.endswith("_changed")) or bool(
            signals.get("url_changed")
        )
        return signals

    def _capture_before_state(self, element: WebElement, options: ClickOptions) -> Dict[str, Any]:
        return self._capture_click_state(element, options, phase="before")

    def _capture_after_state(self, element: WebElement, options: ClickOptions) -> Dict[str, Any]:
        return self._capture_click_state(element, options, phase="after")

    def _capture_click_state(self, element: WebElement, options: ClickOptions, *, phase: str) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "phase": phase,
            "url": get_current_url(self.driver),
            "title": get_page_title(self.driver),
            "ready_state": get_document_ready_state(self.driver),
            "element_selected": safe_call(element.is_selected, default=None),
            "element_value": safe_get_attribute(element, "value", default=None, max_length=500),
            "element_checked": safe_get_attribute(element, "checked", default=None),
            "element_expanded": safe_get_attribute(element, "aria-expanded", default=None),
        }
        active = safe_call(lambda: self.driver.switch_to.active_element, default=None)
        if active is not None:
            state["active_element_fingerprint"] = element_snapshot(active, include_html=False).fingerprint
        if options.include_page_snapshot:
            state["page"] = page_snapshot_dict(self.driver, include_html=False, include_screenshot=False, max_text=1_000)
        return prune_none(redact_mapping(state))

    def _sleep_before_retry(self, attempt: int, options: ClickOptions) -> None:
        delay = calculate_backoff_delay(
            attempt_index=attempt,
            base_delay=options.retry_backoff_base,
            max_delay=options.retry_backoff_max,
            multiplier=options.retry_backoff_multiplier,
            jitter=options.retry_jitter,
        )
        if delay > 0:
            time.sleep(delay)

    def _success_result(self, message: str, context: ClickExecutionContext) -> dict:
        request = context.request
        options = request.options
        element = context.element
        data: Dict[str, Any] = {
            "selector": request.selector,
            "strategy": context.strategy,
            "attempt_count": len(context.attempts),
            "attempts": context.attempts,
            "before": context.before,
            "after": context.after,
        }
        if element is not None:
            data["element"] = element_metadata(
                element,
                max_text=options.max_element_text_chars,
                max_html=options.max_element_html_chars if options.include_element_html else 0,
            )
        return success_result(
            action="click",
            message=message,
            data=data,
            duration_ms=elapsed_ms(context.start_ms),
            correlation_id=request.correlation_id,
        )

    def _error_result(self, error: BaseException, *, start_ms: float, selector: str, options: Optional[ClickOptions] = None) -> dict:
        metadata: Dict[str, Any] = {
            "selector": selector,
            "options": asdict(options) if options else None,
        }
        if options and options.screenshot_on_error:
            screenshot = capture_screenshot_b64(self.driver)
            if screenshot:
                metadata["screenshot_b64"] = screenshot
        return error_result(
            action="click",
            message=str(error),
            error=error,
            metadata=metadata,
            duration_ms=elapsed_ms(start_ms),
            correlation_id=new_correlation_id("click-error"),
        )

    def _coerce_click_error(self, exc: BaseException, **context: Any) -> BrowserError:
        if isinstance(exc, BrowserError):
            exc.context.update(sanitize_context(context, redact=False) if "sanitize_context" in globals() else safe_serialize(context))
            return exc
        mapped = self._map_selenium_exception(exc)
        return mapped(str(exc) or mapped.default_message, context=context, cause=exc)

    def _map_selenium_exception(self, exc: BaseException):
        if isinstance(exc, TimeoutException):
            return BrowserTimeoutError
        if isinstance(exc, NoSuchElementException):
            return ElementNotFoundError
        if isinstance(exc, StaleElementReferenceException):
            return StaleElementError
        if isinstance(exc, ElementClickInterceptedException):
            return ClickInterceptedError
        if isinstance(exc, ElementNotInteractableException):
            return ElementNotInteractableError
        if isinstance(exc, JavascriptException):
            return JavaScriptClickError
        if isinstance(exc, WebDriverException):
            return ClickError
        return ClickError

    def _strategy_label(self, strategy: str) -> str:
        return {
            "native": "Standard",
            "action_chains": "ActionChains",
            "javascript": "JavaScript",
            "dispatch_event": "DOM event",
            "keyboard": "Keyboard",
            "select_option": "Select option",
        }.get(strategy, strategy)

    def _attempt_click(self, element: WebElement, selector: str) -> dict:
        """Backwards-compatible direct element click helper."""

        start_ms = monotonic_ms()
        try:
            request = self._build_request(selector, self.options)
            context = ClickExecutionContext(request=request, start_ms=start_ms, element=element)
            context.before = self._capture_before_state(element, self.options)
            return self._attempt_click_strategies(element, context)
        except Exception as exc:
            error = self._coerce_click_error(exc, selector=selector, phase="attempt_click")
            return self._error_result(error, start_ms=start_ms, selector=selector, options=self.options)

    def _perform_javascript_click(self, element: WebElement, selector: str) -> dict:
        """Backwards-compatible JavaScript fallback helper."""

        start_ms = monotonic_ms()
        try:
            request = self._build_request(selector, self.options)
            context = ClickExecutionContext(request=request, start_ms=start_ms, element=element)
            context.before = self._capture_before_state(element, self.options)
            self._attempt_strategy("javascript", element)
            context.strategy = "javascript"
            context.record_attempt("javascript", "success", "JavaScript click succeeded")
            self._post_click_verification(element, context)
            return self._success_result("JavaScript click succeeded", context)
        except Exception as exc:
            error = self._coerce_click_error(exc, selector=selector, phase="perform_javascript_click")
            return self._error_result(error, start_ms=start_ms, selector=selector, options=self.options)

    def _build_success_message(self, message: str, element: WebElement) -> dict:
        """Compatibility wrapper for legacy callers."""

        return success_result(
            action="click",
            message=message,
            data={"element": element_metadata(element)},
            correlation_id=new_correlation_id("click-legacy"),
        )

    def _build_error_message(self, error_msg: str) -> dict:
        """Compatibility wrapper for legacy callers."""

        error = ClickError(error_msg)
        return error_result(action="click", message=error_msg, error=error, correlation_id=new_correlation_id("click-legacy-error"))

    def click_element(self, selector: str, wait_time: float = 0.0) -> str:
        """Legacy string-returning wrapper."""

        result = self._perform_click(selector, wait_time)
        return str(result.get("message", ""))


if __name__ == "__main__":
    print("\n=== Running Do Click ===\n")
    printer.status("TEST", "Do Click initialized", "info")

    class _FakeSwitchTo:
        def __init__(self, driver: "_FakeDriver") -> None:
            self._driver = driver

        @property
        def active_element(self):
            return self._driver.active_element

    class _FakeElement:
        def __init__(self, tag: str = "button", text: str = "Click me", *, fail_native: bool = False, displayed: bool = True, enabled: bool = True):
            self.tag_name = tag
            self.text = text
            self.fail_native = fail_native
            self.clicked = False
            self._displayed = displayed
            self._enabled = enabled
            self.location = {"x": 10, "y": 20}
            self.size = {"width": 120, "height": 35}
            self.attrs = {
                "id": "fake-button",
                "class": "btn primary",
                "role": "button",
                "outerHTML": f"<{tag} id='fake-button' class='btn primary'>{text}</{tag}>",
            }

        def click(self):
            if self.fail_native:
                raise WebDriverException("native click blocked for test")
            self.clicked = True

        def send_keys(self, value):
            self.clicked = True
            self.attrs["last_keys"] = str(value)

        def is_displayed(self):
            return self._displayed

        def is_enabled(self):
            return self._enabled

        def is_selected(self):
            return bool(self.attrs.get("checked"))

        def get_attribute(self, name):
            return self.attrs.get(name)

        def find_element(self, by=By.CSS_SELECTOR, value=None):
            raise NoSuchElementException(f"No nested element for {by}={value}")

    class _FakeDriver:
        def __init__(self) -> None:
            self.current_url = "https://example.test/start"
            self.title = "Example Test Page"
            self.page_source = "<html><body><button id='fake-button'>Click me</button></body></html>"
            self.elements = {
                "#fake-button": _FakeElement(),
                "#js-button": _FakeElement(text="JS fallback", fail_native=True),
            }
            self.active_element = self.elements["#fake-button"]
            self.switch_to = _FakeSwitchTo(self)
            self.scripts: List[str] = []

        def find_element(self, by=By.CSS_SELECTOR, value=None):
            if by == By.TAG_NAME and value == "body":
                return _FakeElement(tag="body", text="Click me JS fallback")
            if value in self.elements:
                return self.elements[value]
            raise NoSuchElementException(f"Missing fake selector: {value}")

        def find_elements(self, by=By.CSS_SELECTOR, value=None):
            try:
                return [self.find_element(by, value)]
            except NoSuchElementException:
                return []

        def execute_script(self, script, *args):
            self.scripts.append(str(script))
            if "document.readyState" in str(script):
                return "complete"
            if "arguments[0].click" in str(script) and args:
                args[0].clicked = True
            return None

        def get_screenshot_as_base64(self):
            return ""

    clicker = DoClick(_FakeDriver())

    native_result = clicker.click("#fake-button", strategies=["native"], verify_after_click=True)
    assert native_result["status"] == "success", native_result
    assert native_result["data"]["strategy"] == "native", native_result

    js_result = clicker.click("#js-button", strategies=["javascript"], verify_after_click=True)
    assert js_result["status"] == "success", js_result
    assert js_result["data"]["strategy"] == "javascript", js_result

    missing_result = clicker.click("#missing", retries=0, timeout=0.2, strategies=["native"])
    assert missing_result["status"] == "error", missing_result

    legacy_message = clicker.click_element("#fake-button")
    assert legacy_message, legacy_message

    print("\n=== Test ran successfully ===\n")
