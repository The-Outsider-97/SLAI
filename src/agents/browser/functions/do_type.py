from __future__ import annotations

"""
Production-grade typing functionality for the browser subsystem.

This module owns concrete text-entry behavior for Selenium-backed browser
agents. It intentionally does not own browser lifecycle, workflow routing,
content extraction, memory, or high-level task planning. Those concerns belong
in BrowserAgent and adjacent browser modules. This module focuses on one stable
contract: turn a selector and text payload into a safe, observable,
configurable browser typing result.

Design goals
------------
- Preserve the existing BrowserAgent integration method: ``type_text``.
- Use shared browser errors and helpers instead of redefining validation,
  serialization, redaction, result payloads, timing, screenshots, or element
  snapshot behavior.
- Keep runtime behavior configurable from ``browser_config.yaml`` via the
  ``do_type`` section.
- Support realistic input fields, textareas, contenteditable nodes, ARIA
  textboxes, custom widgets, and fallback JavaScript value assignment.
- Provide precise diagnostics for debugging, workflow replay, memory, and
  telemetry without leaking typed secrets by default.
- Stay extensible for future typing strategies, keyboard shortcuts, IME-aware
  entry, form submission, rich text editing, and accessibility-driven controls.
"""

import asyncio
import time as time_module

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from selenium.common.exceptions import (
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
from selenium.webdriver.support.ui import WebDriverWait

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.browser_errors import *
from ..utils.Browser_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Type")
printer = PrettyPrinter


DEFAULT_TYPE_STRATEGIES: Tuple[str, ...] = (
    "native_send_keys",
    "human_send_keys",
    "action_chains",
    "javascript_value",
    "contenteditable_javascript",
)

DEFAULT_CLEAR_STRATEGIES: Tuple[str, ...] = (
    "native_clear",
    "keyboard_select_delete",
    "javascript_clear",
)

SUPPORTED_TYPE_STRATEGIES = set(DEFAULT_TYPE_STRATEGIES)
SUPPORTED_CLEAR_STRATEGIES = set(DEFAULT_CLEAR_STRATEGIES)
VALID_NEWLINE_MODES = {"literal", "enter", "shift_enter", "normalize_space"}
VALID_VERIFY_MODES = {"none", "contains", "equals", "prefix", "non_empty", "changed"}
TEXT_INPUT_TYPES = {
    "",
    "text",
    "search",
    "email",
    "password",
    "url",
    "tel",
    "number",
    "date",
    "datetime-local",
    "month",
    "week",
    "time",
    "color",
}
WRITABLE_TAGS = {"input", "textarea"}
CONTENTEDITABLE_VALUES = {"", "true", "plaintext-only"}


@dataclass(frozen=True)
class TypeOptions:
    """Resolved execution policy for one typing operation."""

    timeout: float = 10.0
    poll_frequency: float = 0.1
    wait_before_execution: float = 0.0
    wait_after_type: float = 0.0
    retries: int = 2
    retry_backoff_base: float = 0.2
    retry_backoff_multiplier: float = 1.8
    retry_backoff_max: float = 2.5
    retry_jitter: float = 0.05
    clear_before: bool = True
    append_mode: bool = False
    require_visible: bool = True
    require_enabled: bool = True
    require_writable: bool = True
    scroll_into_view: bool = True
    scroll_position: str = "center"
    scroll_behavior: str = "auto"
    focus_before_type: bool = True
    click_before_type: bool = False
    blur_after_type: bool = False
    submit_after_type: bool = False
    press_enter_after_type: bool = False
    dispatch_input_events: bool = True
    verify_after_type: bool = True
    verify_mode: str = "contains"
    allow_sensitive_text_in_result: bool = False
    include_text_preview: bool = False
    text_preview_chars: int = 120
    include_page_snapshot: bool = False
    include_element_html: bool = True
    max_element_text_chars: int = 500
    max_element_html_chars: int = 1_500
    screenshot_on_error: bool = False
    page_load_timeout: float = 5.0
    human_typing_min_delay: float = 0.03
    human_typing_max_delay: float = 0.12
    max_text_length: int = 100_000
    newline_mode: str = "literal"
    type_strategies: Tuple[str, ...] = DEFAULT_TYPE_STRATEGIES
    clear_strategies: Tuple[str, ...] = DEFAULT_CLEAR_STRATEGIES
    allow_javascript_fallback: bool = True
    allow_contenteditable_fallback: bool = True

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]], **overrides: Any) -> "TypeOptions":
        """Build options from ``browser_config.yaml`` and per-call overrides."""

        cfg = dict(config or {})
        retry_cfg = dict(cfg.get("retry") or {})
        diagnostics = dict(cfg.get("diagnostics") or {})
        verification = dict(cfg.get("verification") or {})
        preparation = dict(cfg.get("preparation") or {})
        text_cfg = dict(cfg.get("text") or {})
        typing_cfg = dict(cfg.get("typing") or {})
        clear_cfg = dict(cfg.get("clear") or {})
        fallbacks = dict(cfg.get("fallbacks") or {})

        type_strategies = overrides.pop("type_strategies", None) or overrides.pop("strategies", None) or typing_cfg.get("strategies") or cfg.get("strategies") or DEFAULT_TYPE_STRATEGIES
        clear_strategies = overrides.pop("clear_strategies", None) or clear_cfg.get("strategies") or DEFAULT_CLEAR_STRATEGIES

        newline_mode = str(overrides.pop("newline_mode", text_cfg.get("newline_mode", cfg.get("newline_mode", "literal")))).lower().strip()
        if newline_mode not in VALID_NEWLINE_MODES:
            newline_mode = "literal"

        verify_mode = str(overrides.pop("verify_mode", verification.get("mode", cfg.get("verify_mode", "contains")))).lower().strip()
        if verify_mode not in VALID_VERIFY_MODES:
            verify_mode = "contains"

        return cls(
            timeout=coerce_float(overrides.pop("timeout", cfg.get("timeout", 10.0)), default=10.0, minimum=0.1),
            poll_frequency=coerce_float(overrides.pop("poll_frequency", cfg.get("poll_frequency", 0.1)), default=0.1, minimum=0.05),
            wait_before_execution=coerce_float(
                overrides.pop("wait_before_execution", cfg.get("wait_before_execution", 0.0)), default=0.0, minimum=0.0
            ),
            wait_after_type=coerce_float(overrides.pop("wait_after_type", cfg.get("wait_after_type", 0.0)), default=0.0, minimum=0.0),
            retries=coerce_int(overrides.pop("retries", retry_cfg.get("max_attempts", cfg.get("retries", 2))), default=2, minimum=0),
            retry_backoff_base=coerce_float(
                overrides.pop("retry_backoff_base", retry_cfg.get("base_delay", cfg.get("retry_backoff_base", 0.2))), default=0.2, minimum=0.0
            ),
            retry_backoff_multiplier=coerce_float(
                overrides.pop("retry_backoff_multiplier", retry_cfg.get("multiplier", cfg.get("retry_backoff_multiplier", 1.8))), default=1.8, minimum=1.0
            ),
            retry_backoff_max=coerce_float(
                overrides.pop("retry_backoff_max", retry_cfg.get("max_delay", cfg.get("retry_backoff_max", 2.5))), default=2.5, minimum=0.0
            ),
            retry_jitter=coerce_float(overrides.pop("retry_jitter", retry_cfg.get("jitter", cfg.get("retry_jitter", 0.05))), default=0.05, minimum=0.0),
            clear_before=coerce_bool(overrides.pop("clear_before", cfg.get("clear_before", True)), default=True),
            append_mode=coerce_bool(overrides.pop("append_mode", cfg.get("append_mode", False)), default=False),
            require_visible=coerce_bool(overrides.pop("require_visible", preparation.get("require_visible", cfg.get("require_visible", True))), default=True),
            require_enabled=coerce_bool(overrides.pop("require_enabled", preparation.get("require_enabled", cfg.get("require_enabled", True))), default=True),
            require_writable=coerce_bool(overrides.pop("require_writable", preparation.get("require_writable", cfg.get("require_writable", True))), default=True),
            scroll_into_view=coerce_bool(overrides.pop("scroll_into_view", preparation.get("scroll_into_view", cfg.get("scroll_into_view", True))), default=True),
            scroll_position=str(overrides.pop("scroll_position", preparation.get("scroll_position", cfg.get("scroll_position", "center")))).lower().strip(),
            scroll_behavior=str(overrides.pop("scroll_behavior", preparation.get("scroll_behavior", cfg.get("scroll_behavior", "auto")))).lower().strip(),
            focus_before_type=coerce_bool(overrides.pop("focus_before_type", preparation.get("focus_before_type", cfg.get("focus_before_type", True))), default=True),
            click_before_type=coerce_bool(overrides.pop("click_before_type", preparation.get("click_before_type", cfg.get("click_before_type", False))), default=False),
            blur_after_type=coerce_bool(overrides.pop("blur_after_type", cfg.get("blur_after_type", False)), default=False),
            submit_after_type=coerce_bool(overrides.pop("submit_after_type", cfg.get("submit_after_type", False)), default=False),
            press_enter_after_type=coerce_bool(overrides.pop("press_enter_after_type", cfg.get("press_enter_after_type", False)), default=False),
            dispatch_input_events=coerce_bool(
                overrides.pop("dispatch_input_events", typing_cfg.get("dispatch_input_events", cfg.get("dispatch_input_events", True))), default=True
            ),
            verify_after_type=coerce_bool(
                overrides.pop("verify_after_type", verification.get("enabled", cfg.get("verify_after_type", True))), default=True
            ),
            verify_mode=verify_mode,
            allow_sensitive_text_in_result=coerce_bool(
                overrides.pop("allow_sensitive_text_in_result", text_cfg.get("allow_sensitive_text_in_result", cfg.get("allow_sensitive_text_in_result", False))),
                default=False,
            ),
            include_text_preview=coerce_bool(
                overrides.pop("include_text_preview", text_cfg.get("include_text_preview", cfg.get("include_text_preview", False))), default=False
            ),
            text_preview_chars=coerce_int(
                overrides.pop("text_preview_chars", text_cfg.get("preview_chars", cfg.get("text_preview_chars", 120))), default=120, minimum=0
            ),
            include_page_snapshot=coerce_bool(
                overrides.pop("include_page_snapshot", diagnostics.get("include_page_snapshot", cfg.get("include_page_snapshot", False))), default=False
            ),
            include_element_html=coerce_bool(
                overrides.pop("include_element_html", diagnostics.get("include_element_html", cfg.get("include_element_html", True))), default=True
            ),
            max_element_text_chars=coerce_int(
                overrides.pop("max_element_text_chars", diagnostics.get("max_element_text_chars", cfg.get("max_element_text_chars", 500))), default=500, minimum=0
            ),
            max_element_html_chars=coerce_int(
                overrides.pop("max_element_html_chars", diagnostics.get("max_element_html_chars", cfg.get("max_element_html_chars", 1500))), default=1500, minimum=0
            ),
            screenshot_on_error=coerce_bool(
                overrides.pop("screenshot_on_error", diagnostics.get("screenshot_on_error", cfg.get("screenshot_on_error", False))), default=False
            ),
            page_load_timeout=coerce_float(overrides.pop("page_load_timeout", cfg.get("page_load_timeout", 5.0)), default=5.0, minimum=0.0),
            human_typing_min_delay=coerce_float(
                overrides.pop("human_typing_min_delay", typing_cfg.get("human_min_delay", cfg.get("human_typing_min_delay", 0.03))), default=0.03, minimum=0.0
            ),
            human_typing_max_delay=coerce_float(
                overrides.pop("human_typing_max_delay", typing_cfg.get("human_max_delay", cfg.get("human_typing_max_delay", 0.12))), default=0.12, minimum=0.0
            ),
            max_text_length=coerce_int(overrides.pop("max_text_length", text_cfg.get("max_length", cfg.get("max_text_length", 100000))), default=100000, minimum=0),
            newline_mode=newline_mode,
            type_strategies=normalize_type_strategies(type_strategies),
            clear_strategies=normalize_clear_strategies(clear_strategies),
            allow_javascript_fallback=coerce_bool(
                overrides.pop("allow_javascript_fallback", fallbacks.get("javascript", cfg.get("allow_javascript_fallback", True))), default=True
            ),
            allow_contenteditable_fallback=coerce_bool(
                overrides.pop("allow_contenteditable_fallback", fallbacks.get("contenteditable", cfg.get("allow_contenteditable_fallback", True))), default=True
            ),
        )


@dataclass(frozen=True)
class TypeRequest:
    """Normalized input for one type operation."""

    selector: str
    text: str
    options: TypeOptions
    correlation_id: str = field(default_factory=lambda: new_correlation_id("type"))

    @property
    def text_length(self) -> int:
        return len(self.text)

    @property
    def text_fingerprint(self) -> str:
        return stable_hash(self.text, length=16)


@dataclass
class TypeExecutionContext:
    """Mutable execution state for diagnostics and telemetry."""

    request: TypeRequest
    start_ms: float
    attempt: int = 0
    element: Optional[WebElement] = None
    strategy: Optional[str] = None
    clear_strategy: Optional[str] = None
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    attempts: List[Dict[str, Any]] = field(default_factory=list)

    def record_attempt(self, strategy: str, status: str, message: str = "", **metadata: Any) -> None:
        self.attempts.append(
            prune_none(
                redact_mapping(
                    {
                        "attempt": self.attempt,
                        "strategy": strategy,
                        "status": status,
                        "message": message,
                        "metadata": safe_serialize(metadata),
                    }
                )
            )
        )


def normalize_type_strategies(strategies: Any) -> Tuple[str, ...]:
    """Normalize configured type strategies and remove unsupported values."""

    normalized: List[str] = []
    for item in ensure_list(strategies):
        strategy = str(item or "").strip().lower().replace("-", "_")
        if strategy in SUPPORTED_TYPE_STRATEGIES and strategy not in normalized:
            normalized.append(strategy)
    return tuple(normalized or DEFAULT_TYPE_STRATEGIES)


def normalize_clear_strategies(strategies: Any) -> Tuple[str, ...]:
    """Normalize configured clear strategies and remove unsupported values."""

    normalized: List[str] = []
    for item in ensure_list(strategies):
        strategy = str(item or "").strip().lower().replace("-", "_")
        if strategy in SUPPORTED_CLEAR_STRATEGIES and strategy not in normalized:
            normalized.append(strategy)
    return tuple(normalized or DEFAULT_CLEAR_STRATEGIES)


class DoType:
    """Concrete text-entry executor for browser automation."""

    def __init__(self, driver: Any):
        self.config = load_global_config()
        self.type_config = get_config_section("do_type") or {}
        self.driver = driver
        self.options = TypeOptions.from_config(self.type_config)
        logger.info("Browser typing functionality initiated.")

    async def do_type(self, selector: str, text: str, clear_before: Optional[bool] = None, **kwargs: Any) -> dict:
        """Asynchronously type text into an element."""

        return await asyncio.to_thread(self.type_text, selector, text, self.options.clear_before if clear_before is None else clear_before, **kwargs)

    async def do_clear(self, selector: str, **kwargs: Any) -> dict:
        """Asynchronously clear an element."""

        return await asyncio.to_thread(self.clear_text, selector, **kwargs)

    async def do_press_key(self, selector: str, key: str, **kwargs: Any) -> dict:
        """Asynchronously press a keyboard key on an element."""

        return await asyncio.to_thread(self.press_key, selector, key, **kwargs)

    def perform(self, action: str, selector: str, text: Optional[str] = None, **kwargs: Any) -> dict:
        """Normalized dispatcher for future workflow/task integrations."""

        action_name = str(action or "type").lower().strip()
        if action_name in {"type", "entertext", "enter_text", "input"}:
            return self.type_text(selector, "" if text is None else str(text), kwargs.pop("clear_before", self.options.clear_before), **kwargs)
        if action_name in {"append", "append_text"}:
            return self.append_text(selector, "" if text is None else str(text), **kwargs)
        if action_name in {"clear", "clear_text"}:
            return self.clear_text(selector, **kwargs)
        if action_name in {"press_key", "key"}:
            key = kwargs.pop("key", text)
            return self.press_key(selector, str(key or ""), **kwargs)
        if action_name in {"submit"}:
            return self.submit(selector, **kwargs)
        error = BrowserTypingError("Unsupported type action", context={"action": action_name, "selector": selector})
        return self._error_result(error, start_ms=monotonic_ms(), selector=selector, options=self.options)

    def type_text(self, selector: str, raw_input: str, clear_before: bool = True, **kwargs: Any) -> dict:
        """Type text into a matching element.

        This method preserves the original BrowserAgent-facing API while adding
        config-backed validation, preparation, retries, strategies, verification,
        and diagnostics.
        """

        start_ms = monotonic_ms()
        options = TypeOptions.from_config(self.type_config, clear_before=clear_before, **kwargs)
        try:
            request = self._build_request(selector, raw_input, options)
            context = TypeExecutionContext(request=request, start_ms=start_ms)
            return self._execute_type(context)
        except Exception as exc:
            browser_error = self._coerce_type_error(
                exc,
                selector=selector,
                phase="type_text",
                elapsed_ms=elapsed_ms(start_ms),
            )
            return self._error_result(browser_error, start_ms=start_ms, selector=selector, options=options)

    def append_text(self, selector: str, raw_input: str, **kwargs: Any) -> dict:
        """Append text without clearing the element first."""

        return self.type_text(selector, raw_input, clear_before=False, append_mode=True, **kwargs)

    def clear_text(self, selector: str, **kwargs: Any) -> dict:
        """Clear an input-like element and return a standard result."""

        start_ms = monotonic_ms()
        options = TypeOptions.from_config(self.type_config, clear_before=True, **kwargs)
        try:
            selector_text = validate_selector(selector, field_name="selector")
            self._validate_options(options)
            element = self._resolve_element(selector_text, options)
            context = TypeExecutionContext(
                request=TypeRequest(selector=selector_text, text="", options=options),
                start_ms=start_ms,
                element=element,
            )
            context.before = self._capture_type_state(element, options, phase="before_clear")
            self._prepare_element(element, selector_text, options)
            self._clear_element(element, context)
            context.after = self._capture_type_state(element, options, phase="after_clear")
            return self._success_result("Element cleared", context)
        except Exception as exc:
            error = self._coerce_type_error(exc, selector=selector, phase="clear_text")
            return self._error_result(error, start_ms=start_ms, selector=selector, options=options)

    def press_key(self, selector: str, key: str, **kwargs: Any) -> dict:
        """Send one Selenium key or key string to an element."""

        start_ms = monotonic_ms()
        options = TypeOptions.from_config(self.type_config, clear_before=False, **kwargs)
        try:
            selector_text = validate_selector(selector, field_name="selector")
            key_text = self._resolve_key(key)
            element = self._resolve_element(selector_text, options)
            context = TypeExecutionContext(
                request=TypeRequest(selector=selector_text, text=str(key), options=options),
                start_ms=start_ms,
                element=element,
            )
            context.before = self._capture_type_state(element, options, phase="before_key")
            self._prepare_element(element, selector_text, options)
            element.send_keys(key_text)
            context.strategy = "press_key"
            context.record_attempt("press_key", "success", "Key sent")
            context.after = self._capture_type_state(element, options, phase="after_key")
            return self._success_result("Key sent", context)
        except Exception as exc:
            error = self._coerce_type_error(exc, selector=selector, key=key, phase="press_key")
            return self._error_result(error, start_ms=start_ms, selector=selector, options=options)

    def submit(self, selector: str, **kwargs: Any) -> dict:
        """Submit an element's nearest form when Selenium supports it."""

        start_ms = monotonic_ms()
        options = TypeOptions.from_config(self.type_config, clear_before=False, **kwargs)
        try:
            selector_text = validate_selector(selector, field_name="selector")
            element = self._resolve_element(selector_text, options)
            context = TypeExecutionContext(
                request=TypeRequest(selector=selector_text, text="", options=options),
                start_ms=start_ms,
                element=element,
            )
            context.before = self._capture_type_state(element, options, phase="before_submit")
            self._prepare_element(element, selector_text, options)
            element.submit()
            self._wait_for_page_load_if_requested(options)
            context.strategy = "submit"
            context.record_attempt("submit", "success", "Form submitted")
            context.after = self._capture_type_state(element, options, phase="after_submit")
            return self._success_result("Form submitted", context)
        except Exception as exc:
            error = self._coerce_type_error(exc, selector=selector, phase="submit")
            return self._error_result(error, start_ms=start_ms, selector=selector, options=options)

    def _build_request(self, selector: str, raw_input: Any, options: TypeOptions) -> TypeRequest:
        selector_text = validate_selector(selector, field_name="selector")
        self._validate_options(options)
        text = self._normalize_input_text(raw_input, options)
        return TypeRequest(selector=selector_text, text=text, options=options)

    def _validate_options(self, options: TypeOptions) -> None:
        if options.timeout <= 0:
            raise InvalidTimeoutError("Type timeout must be greater than zero", context={"timeout": options.timeout})
        if options.poll_frequency <= 0:
            raise InvalidTimeoutError("Type poll frequency must be greater than zero", context={"poll_frequency": options.poll_frequency})
        if not options.type_strategies:
            raise BrowserTypingError("At least one type strategy must be configured")
        invalid_type = [strategy for strategy in options.type_strategies if strategy not in SUPPORTED_TYPE_STRATEGIES]
        invalid_clear = [strategy for strategy in options.clear_strategies if strategy not in SUPPORTED_CLEAR_STRATEGIES]
        if invalid_type:
            raise BrowserTypingError("Unsupported type strategies configured", context={"invalid_strategies": invalid_type})
        if invalid_clear:
            raise BrowserTypingError("Unsupported clear strategies configured", context={"invalid_strategies": invalid_clear})
        if options.human_typing_max_delay < options.human_typing_min_delay:
            raise BrowserTypingError(
                "Human typing max delay cannot be lower than min delay",
                context={"min_delay": options.human_typing_min_delay, "max_delay": options.human_typing_max_delay},
            )

    def _normalize_input_text(self, raw_input: Any, options: TypeOptions) -> str:
        text = "" if raw_input is None else str(raw_input)
        if options.newline_mode == "normalize_space":
            text = normalize_whitespace(text)
        if len(text) > options.max_text_length:
            raise BrowserTypingError(
                "Input text exceeds configured maximum length",
                context={"length": len(text), "max_text_length": options.max_text_length},
            )
        return text

    def _execute_type(self, context: TypeExecutionContext) -> dict:
        request = context.request
        options = request.options

        if options.wait_before_execution > 0:
            time_module.sleep(options.wait_before_execution)

        for attempt in range(options.retries + 1):
            context.attempt = attempt + 1
            try:
                element = self._resolve_element(request.selector, options)
                context.element = element
                context.before = self._capture_type_state(element, options, phase="before")
                self._prepare_element(element, request.selector, options)

                if options.clear_before and not options.append_mode:
                    self._clear_element(element, context)

                self._type_with_strategies(element, context)
                self._apply_post_type_actions(element, options)

                if options.wait_after_type > 0:
                    time_module.sleep(options.wait_after_type)

                context.after = self._capture_type_state(element, options, phase="after")
                self._verify_type_result(element, context)
                return self._success_result("Text typed", context)
            except (StaleElementReferenceException, TimeoutException, ElementNotInteractableException, WebDriverException) as exc:
                context.record_attempt("retry", "error", str(exc), exception_type=exc.__class__.__name__)
                if attempt >= options.retries:
                    raise self._coerce_type_error(exc, selector=request.selector, phase="retry_exhausted", attempts=context.attempts) from exc
                self._sleep_before_retry(attempt, options)
            except BrowserError:
                raise
            except Exception as exc:
                raise self._coerce_type_error(exc, selector=request.selector, phase="execute_type", attempts=context.attempts) from exc

        raise RetryExhaustedError("Typing retries exhausted", context={"selector": request.selector, "attempts": context.attempts})

    def _resolve_element(self, selector: str, options: TypeOptions) -> WebElement:
        condition = EC.visibility_of_element_located((By.CSS_SELECTOR, selector)) if options.require_visible else EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        try:
            element = WebDriverWait(self.driver, options.timeout, poll_frequency=options.poll_frequency).until(condition)
        except TimeoutException as exc:
            raise ElementNotFoundError(
                f"Element not found or not visible for typing: {selector}",
                context={"selector": selector, "timeout": options.timeout, "require_visible": options.require_visible},
                cause=exc,
            ) from exc
        if element is None:
            raise ElementNotFoundError(f"Element not found: {selector}", context={"selector": selector})
        return element

    def _prepare_element(self, element: WebElement, selector: str, options: TypeOptions) -> None:
        if options.scroll_into_view:
            self._scroll_into_view(element, options)
        if options.require_visible:
            self._ensure_visible(element, selector, options)
        if options.require_enabled:
            self._ensure_enabled(element, selector)
        if options.require_writable:
            self._ensure_writable(element, selector)
        if options.focus_before_type:
            self._focus_element(element)
        if options.click_before_type:
            self._click_element_for_focus(element)

    def _scroll_into_view(self, element: WebElement, options: TypeOptions) -> None:
        position = options.scroll_position if options.scroll_position in {"start", "center", "end", "nearest"} else "center"
        behavior = options.scroll_behavior if options.scroll_behavior in {"auto", "smooth", "instant"} else "auto"
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block: arguments[1], inline: 'nearest', behavior: arguments[2]});",
            element,
            position,
            behavior,
        )

    def _ensure_visible(self, element: WebElement, selector: str, options: TypeOptions) -> None:
        try:
            WebDriverWait(self.driver, min(options.timeout, 2.0), poll_frequency=options.poll_frequency).until(EC.visibility_of(element))
        except TimeoutException as exc:
            raise ElementNotInteractableError("Element is not visible for typing", context={"selector": selector}, cause=exc) from exc

    def _ensure_enabled(self, element: WebElement, selector: str) -> None:
        enabled = safe_call(element.is_enabled, default=True)
        disabled_attr = safe_get_attribute(element, "disabled", default=None)
        aria_disabled = str(safe_get_attribute(element, "aria-disabled", default="")).lower()
        if enabled is False or disabled_attr is not None or aria_disabled == "true":
            raise ElementNotInteractableError(
                "Element is disabled and cannot be typed into",
                context={"selector": selector, "enabled": enabled, "disabled": disabled_attr, "aria_disabled": aria_disabled},
            )

    def _ensure_writable(self, element: WebElement, selector: str) -> None:
        tag = str(safe_call(lambda: element.tag_name, default="") or "").lower()
        input_type = str(safe_get_attribute(element, "type", default="") or "").lower()
        readonly = safe_get_attribute(element, "readonly", default=None)
        aria_readonly = str(safe_get_attribute(element, "aria-readonly", default="")).lower()
        contenteditable_raw = safe_get_attribute(element, "contenteditable", default=None)
        contenteditable = str(contenteditable_raw or "").lower() if contenteditable_raw is not None else None
        role = str(safe_get_attribute(element, "role", default="") or "").lower()

        if readonly is not None or aria_readonly == "true":
            raise ElementNotInteractableError("Element is read-only", context={"selector": selector, "readonly": readonly, "aria_readonly": aria_readonly})
        if tag == "input" and input_type not in TEXT_INPUT_TYPES:
            raise ElementNotInteractableError("Input type is not text-writable", context={"selector": selector, "input_type": input_type})
        if tag in WRITABLE_TAGS:
            return
        if contenteditable_raw is not None and contenteditable in CONTENTEDITABLE_VALUES:
            return
        if role in {"textbox", "searchbox", "combobox"}:
            return
        raise ElementNotInteractableError(
            "Element is not recognized as writable",
            context={"selector": selector, "tag": tag, "role": role, "contenteditable": contenteditable},
        )

    def _focus_element(self, element: WebElement) -> None:
        try:
            self.driver.execute_script("arguments[0].focus({preventScroll: true});", element)
        except Exception:
            try:
                element.click()
            except Exception:
                logger.debug("Unable to focus element before typing", exc_info=True)

    def _click_element_for_focus(self, element: WebElement) -> None:
        try:
            element.click()
        except Exception:
            try:
                ActionChains(self.driver).move_to_element(element).click().perform()
            except Exception:
                logger.debug("Unable to click element for focus before typing", exc_info=True)

    def _clear_element(self, element: WebElement, context: TypeExecutionContext) -> None:
        last_error: Optional[BaseException] = None
        for strategy in context.request.options.clear_strategies:
            try:
                self._apply_clear_strategy(strategy, element, context)
                context.clear_strategy = strategy
                context.record_attempt(strategy, "success", "Element cleared")
                return
            except Exception as exc:
                last_error = exc
                context.record_attempt(strategy, "error", str(exc), exception_type=exc.__class__.__name__)
        raise InputClearError("All configured clear strategies failed", context={"selector": context.request.selector, "attempts": context.attempts}, cause=last_error)

    def _apply_clear_strategy(self, strategy: str, element: WebElement, context: TypeExecutionContext) -> None:
        if strategy == "native_clear":
            element.clear()
            return
        if strategy == "keyboard_select_delete":
            modifier = Keys.COMMAND if self._platform_is_macos() else Keys.CONTROL
            element.send_keys(modifier, "a")
            element.send_keys(Keys.BACKSPACE)
            return
        if strategy == "javascript_clear":
            self._set_element_value_js(element, "", context.request.options)
            return
        raise InputClearError("Unsupported clear strategy", context={"strategy": strategy, "selector": context.request.selector})

    def _type_with_strategies(self, element: WebElement, context: TypeExecutionContext) -> None:
        last_error: Optional[BaseException] = None
        options = context.request.options
        for strategy in options.type_strategies:
            if strategy == "javascript_value" and not options.allow_javascript_fallback:
                context.record_attempt(strategy, "skipped", "JavaScript fallback disabled")
                continue
            if strategy == "contenteditable_javascript" and not options.allow_contenteditable_fallback:
                context.record_attempt(strategy, "skipped", "Contenteditable fallback disabled")
                continue
            try:
                self._apply_type_strategy(strategy, element, context)
                context.strategy = strategy
                context.record_attempt(strategy, "success", "Text input applied")
                return
            except Exception as exc:
                last_error = exc
                context.record_attempt(strategy, "error", str(exc), exception_type=exc.__class__.__name__)
        raise InputSendKeysError("All configured type strategies failed", context={"selector": context.request.selector, "attempts": context.attempts}, cause=last_error)

    def _apply_type_strategy(self, strategy: str, element: WebElement, context: TypeExecutionContext) -> None:
        text = self._text_for_send_keys(context.request.text, context.request.options)
        if strategy == "native_send_keys":
            element.send_keys(text)
            return
        if strategy == "human_send_keys":
            self._human_send_keys(element, text, context.request.options)
            return
        if strategy == "action_chains":
            ActionChains(self.driver).move_to_element(element).click().send_keys(text).perform()
            return
        if strategy == "javascript_value":
            self._set_element_value_js(element, context.request.text, context.request.options)
            return
        if strategy == "contenteditable_javascript":
            self._set_contenteditable_text_js(element, context.request.text, context.request.options)
            return
        raise InputSendKeysError("Unsupported type strategy", context={"strategy": strategy, "selector": context.request.selector})

    def _text_for_send_keys(self, text: str, options: TypeOptions) -> str:
        if options.newline_mode == "enter":
            return text.replace("\n", Keys.ENTER)
        if options.newline_mode == "shift_enter":
            # Selenium supports key chord-like sequences poorly across drivers;
            # literal newlines remain the safest default for multi-line fields.
            return text.replace("\n", Keys.SHIFT + Keys.ENTER)
        return text

    def _human_send_keys(self, element: WebElement, text: str, options: TypeOptions) -> None:
        for character in text:
            element.send_keys(character)
            if options.human_typing_max_delay > 0:
                # Deterministic enough for production behavior; jitter is not
                # security logic, just pacing for dynamic UIs.
                delay = calculate_backoff_delay(
                    attempt_index=0,
                    base_delay=options.human_typing_min_delay,
                    max_delay=options.human_typing_max_delay,
                    multiplier=1.0,
                    jitter=max(0.0, options.human_typing_max_delay - options.human_typing_min_delay),
                )
                if delay > 0:
                    time_module.sleep(delay)

    def _set_element_value_js(self, element: WebElement, text: str, options: TypeOptions) -> None:
        script = """
        const element = arguments[0];
        const value = arguments[1];
        const dispatchEvents = arguments[2];
        const previous = element.value;
        const prototype = Object.getPrototypeOf(element);
        const descriptor = prototype && Object.getOwnPropertyDescriptor(prototype, 'value');
        if (descriptor && descriptor.set) {
            descriptor.set.call(element, value);
        } else {
            element.value = value;
        }
        if (dispatchEvents) {
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
        }
        return {previous: previous, current: element.value};
        """
        self.driver.execute_script(script, element, text, options.dispatch_input_events)

    def _set_contenteditable_text_js(self, element: WebElement, text: str, options: TypeOptions) -> None:
        script = """
        const element = arguments[0];
        const value = arguments[1];
        const dispatchEvents = arguments[2];
        const previous = element.innerText || element.textContent || '';
        if (element.isContentEditable || element.getAttribute('contenteditable') !== null || element.getAttribute('role') === 'textbox') {
            element.textContent = value;
            if (dispatchEvents) {
                element.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
                element.dispatchEvent(new Event('change', { bubbles: true }));
            }
            return {previous: previous, current: element.innerText || element.textContent || ''};
        }
        throw new Error('Element is not contenteditable');
        """
        self.driver.execute_script(script, element, text, options.dispatch_input_events)

    def _apply_post_type_actions(self, element: WebElement, options: TypeOptions) -> None:
        if options.press_enter_after_type:
            element.send_keys(Keys.ENTER)
        if options.submit_after_type:
            element.submit()
            self._wait_for_page_load_if_requested(options)
        if options.blur_after_type:
            try:
                self.driver.execute_script("arguments[0].blur();", element)
            except Exception:
                logger.debug("Unable to blur element after typing", exc_info=True)

    def _wait_for_page_load_if_requested(self, options: TypeOptions) -> None:
        if options.page_load_timeout <= 0:
            return
        try:
            end = time_module.time() + options.page_load_timeout
            while time_module.time() < end:
                if get_document_ready_state(self.driver) == "complete":
                    return
                time_module.sleep(0.05)
        except Exception:
            logger.debug("Unable to wait for page load after typing", exc_info=True)

    def _verify_type_result(self, element: WebElement, context: TypeExecutionContext) -> None:
        options = context.request.options
        if not options.verify_after_type or options.verify_mode == "none":
            return
        expected = context.request.text
        actual = self._read_element_value(element)
        passed = True
        if options.verify_mode == "equals":
            passed = actual == expected
        elif options.verify_mode == "contains":
            passed = expected in actual if expected else True
        elif options.verify_mode == "prefix":
            passed = actual.startswith(expected)
        elif options.verify_mode == "non_empty":
            passed = bool(actual)
        elif options.verify_mode == "changed":
            before_value = str(context.before.get("value") or context.before.get("text") or "")
            passed = actual != before_value
        if not passed:
            raise BrowserTypingError(
                "Post-type verification failed",
                context={
                    "selector": context.request.selector,
                    "verify_mode": options.verify_mode,
                    "expected_fingerprint": context.request.text_fingerprint,
                    "actual_fingerprint": stable_hash(actual, length=16),
                    "actual_length": len(actual),
                    "expected_length": len(expected),
                },
            )

    def _read_element_value(self, element: WebElement) -> str:
        for attr in ("value", "innerText", "textContent"):
            value = safe_get_attribute(element, attr, default=None)
            if value is not None:
                return str(value)
        try:
            return str(element.text or "")
        except Exception:
            return ""

    def _capture_type_state(self, element: WebElement, options: TypeOptions, *, phase: str) -> Dict[str, Any]:
        value = self._read_element_value(element)
        state: Dict[str, Any] = {
            "phase": phase,
            "url": get_current_url(self.driver),
            "title": get_page_title(self.driver),
            "ready_state": get_document_ready_state(self.driver),
            "text_length": len(value),
            "text_fingerprint": stable_hash(value, length=16),
            "value": value if options.allow_sensitive_text_in_result else None,
            "text_preview": truncate_text(value, options.text_preview_chars) if options.include_text_preview else None,
            "tag": safe_call(lambda: element.tag_name, default=None),
            "type": safe_get_attribute(element, "type", default=None),
            "role": safe_get_attribute(element, "role", default=None),
            "contenteditable": safe_get_attribute(element, "contenteditable", default=None),
            "disabled": safe_get_attribute(element, "disabled", default=None),
            "readonly": safe_get_attribute(element, "readonly", default=None),
            "aria_readonly": safe_get_attribute(element, "aria-readonly", default=None),
            "active": self._is_active_element(element),
        }
        if options.include_page_snapshot:
            state["page"] = page_snapshot_dict(self.driver, include_html=False, include_screenshot=False, max_text=1_000)
        return prune_none(redact_mapping(state))

    def _is_active_element(self, element: WebElement) -> Optional[bool]:
        try:
            active = self.driver.switch_to.active_element
            return active == element
        except Exception:
            return None

    def _sleep_before_retry(self, attempt: int, options: TypeOptions) -> None:
        delay = calculate_backoff_delay(
            attempt_index=attempt,
            base_delay=options.retry_backoff_base,
            max_delay=options.retry_backoff_max,
            multiplier=options.retry_backoff_multiplier,
            jitter=options.retry_jitter,
        )
        if delay > 0:
            time_module.sleep(delay)

    def _success_result(self, message: str, context: TypeExecutionContext) -> dict:
        request = context.request
        options = request.options
        element = context.element
        data: Dict[str, Any] = {
            "selector": request.selector,
            "strategy": context.strategy,
            "clear_strategy": context.clear_strategy,
            "attempt_count": len(context.attempts),
            "attempts": context.attempts,
            "before": context.before,
            "after": context.after,
            "input": {
                "length": request.text_length,
                "fingerprint": request.text_fingerprint,
                "text": request.text if options.allow_sensitive_text_in_result else None,
                "preview": truncate_text(request.text, options.text_preview_chars) if options.include_text_preview else None,
            },
        }
        if element is not None:
            data["element"] = element_metadata(
                element,
                max_text=options.max_element_text_chars,
                max_html=options.max_element_html_chars if options.include_element_html else 0,
            )
        return success_result(
            action="type",
            message=message,
            data=prune_none(data),
            duration_ms=elapsed_ms(context.start_ms),
            correlation_id=request.correlation_id,
        )

    def _error_result(self, error: BaseException, *, start_ms: float, selector: str, options: Optional[TypeOptions] = None) -> dict:
        metadata: Dict[str, Any] = {
            "selector": selector,
            "options": asdict(options) if options else None,
        }
        if options and options.screenshot_on_error:
            screenshot = capture_screenshot_b64(self.driver)
            if screenshot:
                metadata["screenshot_b64"] = screenshot
        return error_result(
            action="type",
            message=str(error),
            error=error,
            metadata=metadata,
            duration_ms=elapsed_ms(start_ms),
            correlation_id=new_correlation_id("type-error"),
        )

    def _coerce_type_error(self, exc: BaseException, **context: Any) -> BrowserError:
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
        if isinstance(exc, ElementNotInteractableException):
            return ElementNotInteractableError
        if isinstance(exc, JavascriptException):
            return JavaScriptExecutionError
        if isinstance(exc, WebDriverException):
            return BrowserTypingError
        if isinstance(exc, ValueError):
            return BrowserValidationError
        return BrowserTypingError

    def _resolve_key(self, key: str) -> str:
        normalized = str(key or "").strip()
        key_map = {
            "enter": Keys.ENTER,
            "return": Keys.RETURN,
            "tab": Keys.TAB,
            "escape": Keys.ESCAPE,
            "esc": Keys.ESCAPE,
            "backspace": Keys.BACKSPACE,
            "delete": Keys.DELETE,
            "space": Keys.SPACE,
            "arrow_up": Keys.ARROW_UP,
            "arrow_down": Keys.ARROW_DOWN,
            "arrow_left": Keys.ARROW_LEFT,
            "arrow_right": Keys.ARROW_RIGHT,
            "home": Keys.HOME,
            "end": Keys.END,
            "page_up": Keys.PAGE_UP,
            "page_down": Keys.PAGE_DOWN,
        }
        return key_map.get(normalized.lower().replace("-", "_"), normalized)

    def _platform_is_macos(self) -> bool:
        try:
            platform = self.driver.execute_script("return navigator.platform || '';")
            return "mac" in str(platform).lower()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Backwards-compatible/legacy utility methods
    # ------------------------------------------------------------------
    def type_element(self, selector: str, text: str, clear_before: bool = True) -> str:
        """Legacy string-returning wrapper."""

        result = self.type_text(selector, text, clear_before=clear_before)
        return str(result.get("message", result))

    def enter_text(self, selector: str, text: str, clear_before: bool = True, **kwargs: Any) -> dict:
        """Alias used by some browser tool registries."""

        return self.type_text(selector, text, clear_before=clear_before, **kwargs)

    def input_text(self, selector: str, text: str, clear_before: bool = True, **kwargs: Any) -> dict:
        """Alias for frameworks that call this operation input_text."""

        return self.type_text(selector, text, clear_before=clear_before, **kwargs)


if __name__ == "__main__":
    print("\n=== Running Do Type ===\n")
    printer.status("TEST", "Do Type initialized", "info")

    class FakeElement:
        def __init__(self, tag: str = "input", value: str = ""):
            self.tag_name = tag
            self._value = value
            self.text = value
            self.location = {"x": 10, "y": 20}
            self.size = {"width": 200, "height": 40}
            self.clicked = False
            self.cleared = False
            self.submitted = False
            self.attributes = {
                "type": "text",
                "role": "textbox",
                "id": "fake-input",
                "name": "fake",
                "outerHTML": '<input id="fake-input" name="fake" type="text">',
            }

        def get_attribute(self, name):
            if name == "value":
                return self._value
            if name == "innerText" or name == "textContent":
                return self.text
            return self.attributes.get(name)

        def clear(self):
            self.cleared = True
            self._value = ""
            self.text = ""

        def send_keys(self, *values):
            joined = "".join(str(value) for value in values)
            self._value += joined
            self.text = self._value

        def click(self):
            self.clicked = True

        def submit(self):
            self.submitted = True

        def is_enabled(self):
            return True

        def is_displayed(self):
            return True

        def is_selected(self):
            return False

    class FakeSwitchTo:
        def __init__(self, element):
            self.active_element = element

    class FakeDriver:
        def __init__(self):
            self.element = FakeElement()
            self.current_url = "https://example.test/form"
            self.title = "Fake Form"
            self.switch_to = FakeSwitchTo(self.element)
            self.scripts = []

        def find_element(self, by=None, value=None):
            if value == ".missing":
                raise NoSuchElementException("missing")
            return self.element

        def execute_script(self, script, *args):
            self.scripts.append((script, args))
            if "document.readyState" in script:
                return "complete"
            if "navigator.platform" in script:
                return "Linux x86_64"
            if "scrollIntoView" in script or "focus" in script or "blur" in script:
                return None
            if "descriptor.set.call" in script or "element.value = value" in script:
                element = args[0]
                value = args[1]
                element._value = value
                element.text = value
                return {"current": value}
            if "element.textContent = value" in script:
                element = args[0]
                value = args[1]
                element.text = value
                element._value = value
                return {"current": value}
            return None

        def get_screenshot_as_base64(self):
            return "ZmFrZS1zY3JlZW5zaG90"

    driver = FakeDriver()
    typer = DoType(driver)

    result = typer.type_text("#fake-input", "hello world")
    assert result["status"] == "success", result
    assert driver.element.get_attribute("value") == "hello world", result

    append_result = typer.append_text("#fake-input", "!")
    assert append_result["status"] == "success", append_result
    assert driver.element.get_attribute("value").endswith("!"), append_result # type: ignore

    clear_result = typer.clear_text("#fake-input")
    assert clear_result["status"] == "success", clear_result
    assert driver.element.get_attribute("value") == "", clear_result

    key_result = typer.press_key("#fake-input", "enter")
    assert key_result["status"] == "success", key_result

    legacy_message = typer.type_element("#fake-input", "legacy")
    assert "Text typed" in legacy_message, legacy_message

    print("\n=== Test ran successfully ===\n")
