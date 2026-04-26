from __future__ import annotations

"""
Production-grade copy, cut, and paste functionality for the browser subsystem.

This module owns concrete clipboard-oriented browser interactions only. It does
not own browser lifecycle, workflow orchestration, high-level task routing,
content extraction, or memory. Those responsibilities belong to BrowserAgent
and adjacent browser modules. The copy/cut/paste module focuses on one stable
contract: turn a selector plus a clipboard action into a safe, observable,
configurable browser result.

Design goals
------------
- Use shared browser errors and helpers instead of redefining serialization,
  redaction, result payloads, selector validation, screenshots, page snapshots,
  retry timing, or element metadata.
- Keep backwards-compatible public entry points already used by BrowserAgent:
  ``copy()``, ``cut()``, and ``paste()``.
- Make timing, strategy, diagnostics, verification, and text limits configurable
  from ``browser_config.yaml`` rather than burying policy in code.
- Support normal text containers, form controls, contenteditable regions, and
  browser-focused keyboard/JavaScript fallbacks.
- Preserve sensitive clipboard safety by redacting telemetry by default while
  still returning useful operation metadata.
- Remain easy to expand with OS-specific clipboard adapters, CDP clipboard
  integration, file clipboard support, rich text/HTML clipboard formats, or
  future workflow/memory instrumentation.
"""

import asyncio
import time as time_module

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pyperclip  # pyright: ignore[reportMissingModuleSource]
from selenium.common.exceptions import (
    ElementNotInteractableException,
    InvalidElementStateException,
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

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.browser_errors import *
from ..utils.Browser_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("CopyCutPaste")
printer = PrettyPrinter


COPY_ACTION = "copy"
CUT_ACTION = "cut"
PASTE_ACTION = "paste"
SUPPORTED_CLIPBOARD_ACTIONS = {COPY_ACTION, CUT_ACTION, PASTE_ACTION}

DEFAULT_COPY_STRATEGIES: Tuple[str, ...] = (
    "element_text",
    "selection_keyboard",
    "javascript_selection",
)
DEFAULT_CUT_STRATEGIES: Tuple[str, ...] = (
    "native_clear",
    "keyboard_shortcut",
    "javascript_clear",
)
DEFAULT_PASTE_STRATEGIES: Tuple[str, ...] = (
    "send_keys",
    "keyboard_shortcut",
    "javascript_set_value",
)
SUPPORTED_COPY_STRATEGIES = set(DEFAULT_COPY_STRATEGIES)
SUPPORTED_CUT_STRATEGIES = set(DEFAULT_CUT_STRATEGIES)
SUPPORTED_PASTE_STRATEGIES = set(DEFAULT_PASTE_STRATEGIES)
TEXT_FORM_TAGS = {"input", "textarea"}
TEXTUAL_INPUT_TYPES = {
    "",
    "text",
    "search",
    "email",
    "url",
    "tel",
    "password",
    "number",
    "date",
    "datetime-local",
    "month",
    "time",
    "week",
}


@dataclass(frozen=True)
class ClipboardOptions:
    """Resolved execution policy for one clipboard operation."""

    timeout: float = 10.0
    poll_frequency: float = 0.1
    wait_before_execution: float = 0.0
    wait_after_action: float = 0.0
    retries: int = 1
    retry_backoff_base: float = 0.2
    retry_backoff_multiplier: float = 1.7
    retry_backoff_max: float = 2.5
    retry_jitter: float = 0.05
    require_visible: bool = False
    require_enabled_for_write: bool = True
    scroll_into_view: bool = True
    scroll_position: str = "center"
    scroll_behavior: str = "auto"
    focus_before_action: bool = True
    select_before_keyboard_action: bool = True
    preserve_clipboard_on_failure: bool = False
    verify_clipboard_write: bool = True
    verify_element_mutation: bool = True
    include_page_snapshot: bool = False
    include_element_html: bool = True
    include_clipboard_text: bool = True
    redact_clipboard_text: bool = False
    max_text_chars: int = 5_000
    max_element_text_chars: int = 500
    max_element_html_chars: int = 1_500
    screenshot_on_error: bool = False
    clear_before_paste: bool = True
    append_on_paste: bool = False
    allow_javascript_fallback: bool = True
    allow_keyboard_fallback: bool = True
    copy_strategies: Tuple[str, ...] = DEFAULT_COPY_STRATEGIES
    cut_strategies: Tuple[str, ...] = DEFAULT_CUT_STRATEGIES
    paste_strategies: Tuple[str, ...] = DEFAULT_PASTE_STRATEGIES

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **overrides: Any) -> "ClipboardOptions":
        """Build options from ``browser_config.yaml`` plus per-call overrides."""

        copy_strategies = normalize_clipboard_strategies(
            overrides.pop("copy_strategies", config.get("copy_strategies", DEFAULT_COPY_STRATEGIES)),
            action=COPY_ACTION,
        )
        cut_strategies = normalize_clipboard_strategies(
            overrides.pop("cut_strategies", config.get("cut_strategies", DEFAULT_CUT_STRATEGIES)),
            action=CUT_ACTION,
        )
        paste_strategies = normalize_clipboard_strategies(
            overrides.pop("paste_strategies", config.get("paste_strategies", DEFAULT_PASTE_STRATEGIES)),
            action=PASTE_ACTION,
        )
        values = {
            "timeout": coerce_float(overrides.pop("timeout", config.get("timeout", 10.0)), default=10.0, minimum=0.1),
            "poll_frequency": coerce_float(
                overrides.pop("poll_frequency", config.get("poll_frequency", 0.1)), default=0.1, minimum=0.05
            ),
            "wait_before_execution": coerce_float(
                overrides.pop("wait_before_execution", config.get("wait_before_execution", 0.0)), default=0.0, minimum=0.0
            ),
            "wait_after_action": coerce_float(
                overrides.pop("wait_after_action", config.get("wait_after_action", 0.0)), default=0.0, minimum=0.0
            ),
            "retries": coerce_int(overrides.pop("retries", config.get("retries", 1)), default=1, minimum=0),
            "retry_backoff_base": coerce_float(
                overrides.pop("retry_backoff_base", config.get("retry_backoff_base", 0.2)), default=0.2, minimum=0.0
            ),
            "retry_backoff_multiplier": coerce_float(
                overrides.pop("retry_backoff_multiplier", config.get("retry_backoff_multiplier", 1.7)), default=1.7, minimum=1.0
            ),
            "retry_backoff_max": coerce_float(
                overrides.pop("retry_backoff_max", config.get("retry_backoff_max", 2.5)), default=2.5, minimum=0.0
            ),
            "retry_jitter": coerce_float(overrides.pop("retry_jitter", config.get("retry_jitter", 0.05)), default=0.05, minimum=0.0),
            "require_visible": coerce_bool(overrides.pop("require_visible", config.get("require_visible", False)), default=False),
            "require_enabled_for_write": coerce_bool(
                overrides.pop("require_enabled_for_write", config.get("require_enabled_for_write", True)), default=True
            ),
            "scroll_into_view": coerce_bool(overrides.pop("scroll_into_view", config.get("scroll_into_view", True)), default=True),
            "scroll_position": str(overrides.pop("scroll_position", config.get("scroll_position", "center")) or "center"),
            "scroll_behavior": str(overrides.pop("scroll_behavior", config.get("scroll_behavior", "auto")) or "auto"),
            "focus_before_action": coerce_bool(
                overrides.pop("focus_before_action", config.get("focus_before_action", True)), default=True
            ),
            "select_before_keyboard_action": coerce_bool(
                overrides.pop("select_before_keyboard_action", config.get("select_before_keyboard_action", True)), default=True
            ),
            "preserve_clipboard_on_failure": coerce_bool(
                overrides.pop("preserve_clipboard_on_failure", config.get("preserve_clipboard_on_failure", False)), default=False
            ),
            "verify_clipboard_write": coerce_bool(
                overrides.pop("verify_clipboard_write", config.get("verify_clipboard_write", True)), default=True
            ),
            "verify_element_mutation": coerce_bool(
                overrides.pop("verify_element_mutation", config.get("verify_element_mutation", True)), default=True
            ),
            "include_page_snapshot": coerce_bool(
                overrides.pop("include_page_snapshot", config.get("include_page_snapshot", False)), default=False
            ),
            "include_element_html": coerce_bool(
                overrides.pop("include_element_html", config.get("include_element_html", True)), default=True
            ),
            "include_clipboard_text": coerce_bool(
                overrides.pop("include_clipboard_text", config.get("include_clipboard_text", True)), default=True
            ),
            "redact_clipboard_text": coerce_bool(
                overrides.pop("redact_clipboard_text", config.get("redact_clipboard_text", False)), default=False
            ),
            "max_text_chars": coerce_int(overrides.pop("max_text_chars", config.get("max_text_chars", 5000)), default=5000, minimum=0),
            "max_element_text_chars": coerce_int(
                overrides.pop("max_element_text_chars", config.get("max_element_text_chars", 500)), default=500, minimum=0
            ),
            "max_element_html_chars": coerce_int(
                overrides.pop("max_element_html_chars", config.get("max_element_html_chars", 1500)), default=1500, minimum=0
            ),
            "screenshot_on_error": coerce_bool(
                overrides.pop("screenshot_on_error", config.get("screenshot_on_error", False)), default=False
            ),
            "clear_before_paste": coerce_bool(
                overrides.pop("clear_before_paste", config.get("clear_before_paste", True)), default=True
            ),
            "append_on_paste": coerce_bool(overrides.pop("append_on_paste", config.get("append_on_paste", False)), default=False),
            "allow_javascript_fallback": coerce_bool(
                overrides.pop("allow_javascript_fallback", config.get("allow_javascript_fallback", True)), default=True
            ),
            "allow_keyboard_fallback": coerce_bool(
                overrides.pop("allow_keyboard_fallback", config.get("allow_keyboard_fallback", True)), default=True
            ),
            "copy_strategies": copy_strategies,
            "cut_strategies": cut_strategies,
            "paste_strategies": paste_strategies,
        }
        return cls(**values)

    def strategies_for(self, action: str) -> Tuple[str, ...]:
        if action == COPY_ACTION:
            return self.copy_strategies
        if action == CUT_ACTION:
            return self.cut_strategies
        if action == PASTE_ACTION:
            return self.paste_strategies
        return ()


@dataclass(frozen=True)
class ClipboardRequest:
    """Normalized clipboard request for validation, telemetry, and replay."""

    action: str
    selector: str
    text: Optional[str]
    options: ClipboardOptions
    correlation_id: str = field(default_factory=lambda: new_correlation_id("ccp"))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["options"] = asdict(self.options)
        if self.text is not None and self.options.redact_clipboard_text:
            payload["text"] = "[REDACTED]"
        return payload


@dataclass
class ClipboardExecutionContext:
    """Mutable execution context for one clipboard operation."""

    request: ClipboardRequest
    start_ms: float
    attempt: int = 0
    strategy: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    element: Optional[WebElement] = None
    original_clipboard: Optional[str] = None

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


# ---------------------------------------------------------------------------
# Strategy and request normalization
# ---------------------------------------------------------------------------
def normalize_clipboard_action(action: Any) -> str:
    normalized = str(action or "").strip().lower().replace("-", "_")
    aliases = {
        "cpy": COPY_ACTION,
        "duplicate": COPY_ACTION,
        "copy_text": COPY_ACTION,
        "cut_text": CUT_ACTION,
        "paste_text": PASTE_ACTION,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_CLIPBOARD_ACTIONS:
        raise ClipboardValidationError(
            f"Unsupported clipboard action: {action}",
            context={"action": action, "supported_actions": sorted(SUPPORTED_CLIPBOARD_ACTIONS)},
        )
    return normalized


def normalize_clipboard_strategies(strategies: Iterable[Any], *, action: str) -> Tuple[str, ...]:
    if action == COPY_ACTION:
        supported = SUPPORTED_COPY_STRATEGIES
        default = DEFAULT_COPY_STRATEGIES
    elif action == CUT_ACTION:
        supported = SUPPORTED_CUT_STRATEGIES
        default = DEFAULT_CUT_STRATEGIES
    else:
        supported = SUPPORTED_PASTE_STRATEGIES
        default = DEFAULT_PASTE_STRATEGIES

    normalized: List[str] = []
    for strategy in ensure_list(strategies):
        candidate = str(strategy or "").strip().lower().replace("-", "_")
        if candidate in supported and candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized or default)


def _clip_text(text: Any, options: ClipboardOptions) -> str:
    return truncate_text(text, options.max_text_chars)


def _public_text(text: Any, options: ClipboardOptions) -> str:
    if not options.include_clipboard_text:
        return ""
    if options.redact_clipboard_text:
        return "[REDACTED]"
    return _clip_text(text, options)


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------
class DoCopyCutPaste:
    """Production copy/cut/paste executor for Selenium-backed browser sessions."""

    def __init__(self, driver):
        self.config = load_global_config()
        primary_config = get_config_section("do_copy_cut_paste") or {}
        legacy_config = get_config_section("do_ccp") or {}
        self.ccp_config = merge_dicts(legacy_config, primary_config, deep=True)
        self.driver = driver
        logger.info("Web Copy, Cut & Paste functionality initiated.")

    async def do_copy(self, selector: str, **kwargs: Any) -> dict:
        return await asyncio.to_thread(self.copy, selector, **kwargs)

    async def do_cut(self, selector: str, **kwargs: Any) -> dict:
        return await asyncio.to_thread(self.cut, selector, **kwargs)

    async def do_paste(self, selector: str, text: Optional[str] = None, **kwargs: Any) -> dict:
        return await asyncio.to_thread(self.paste, selector, text=text, **kwargs)

    async def do_copy_cut_paste(self, action: str, selector: str, text: Optional[str] = None, **kwargs: Any) -> dict:
        return await asyncio.to_thread(self.perform, action, selector, text=text, **kwargs)

    def perform(self, action: str, selector: str, text: Optional[str] = None, **kwargs: Any) -> dict:
        """Perform a normalized clipboard action.

        This is the general entry point used by tests, workflow adapters, or
        future BrowserAgent task payloads. The convenience methods ``copy``,
        ``cut``, and ``paste`` delegate here.
        """

        action_name = normalize_clipboard_action(action)
        request = self._build_request(action_name, selector, text=text, **kwargs)
        return self._perform_clipboard_action(request)

    def copy(self, selector: str, **kwargs: Any) -> dict:
        return self.perform(COPY_ACTION, selector, **kwargs)

    def cut(self, selector: str, **kwargs: Any) -> dict:
        return self.perform(CUT_ACTION, selector, **kwargs)

    def paste(self, selector: str, text: Optional[str] = None, **kwargs: Any) -> dict:
        return self.perform(PASTE_ACTION, selector, text=text, **kwargs)

    def copy_element(self, selector: str, **kwargs: Any) -> str:
        """Legacy string-returning wrapper."""

        return self.copy(selector, **kwargs).get("message", "")

    def cut_element(self, selector: str, **kwargs: Any) -> str:
        """Legacy string-returning wrapper."""

        return self.cut(selector, **kwargs).get("message", "")

    def paste_element(self, selector: str, text: Optional[str] = None, **kwargs: Any) -> str:
        """Legacy string-returning wrapper."""

        return self.paste(selector, text=text, **kwargs).get("message", "")

    def _build_request(self, action: str, selector: str, text: Optional[str] = None, **kwargs: Any) -> ClipboardRequest:
        normalized_selector = validate_selector(selector)
        options = ClipboardOptions.from_config(self.ccp_config, **kwargs)
        resolved_text = text
        if action == PASTE_ACTION and resolved_text is not None:
            resolved_text = _clip_text(resolved_text, options)
        return ClipboardRequest(action=action, selector=normalized_selector, text=resolved_text, options=options)

    def _perform_clipboard_action(self, request: ClipboardRequest) -> dict:
        context = ClipboardExecutionContext(request=request, start_ms=monotonic_ms())
        options = request.options
        try:
            if options.wait_before_execution > 0:
                time_module.sleep(options.wait_before_execution)

            if options.preserve_clipboard_on_failure:
                context.original_clipboard = self._read_clipboard_safe()

            context.before = self._capture_state(label="before", element=None, options=options)
            max_attempts = options.retries + 1
            last_error: Optional[BaseException] = None

            for attempt in range(max_attempts):
                context.attempt = attempt + 1
                try:
                    element = self._wait_for_element(request.selector, options=options)
                    context.element = element
                    context.before = self._capture_state(label="before", element=element, options=options)
                    self._prepare_element(element, request.action, request.selector, options)
                    result = self._attempt_action(context)

                    if options.wait_after_action > 0:
                        time_module.sleep(options.wait_after_action)

                    context.after = self._capture_state(label="after", element=element, options=options)
                    verification = self._verify_action(context, result)
                    return self._build_success_result(context, result, verification)
                except Exception as exc:
                    last_error = BrowserError.from_exception(
                        exc,
                        action=request.action,
                        context={
                            "selector": request.selector,
                            "attempt": context.attempt,
                            "strategies": request.options.strategies_for(request.action),
                        },
                        default_error_cls=self._error_class_for_action(request.action),
                    )
                    context.record_attempt(
                        context.strategy or "operation",
                        "error",
                        str(last_error),
                        exception=last_error,
                    )
                    logger.warning(
                        "Clipboard action %s failed on attempt %s/%s for %s: %s",
                        request.action,
                        context.attempt,
                        max_attempts,
                        request.selector,
                        last_error,
                    )
                    if attempt >= max_attempts - 1:
                        break
                    sleep_backoff(
                        attempt,
                        base_delay=options.retry_backoff_base,
                        multiplier=options.retry_backoff_multiplier,
                        max_delay=options.retry_backoff_max,
                        jitter=options.retry_jitter,
                    )

            if options.preserve_clipboard_on_failure and context.original_clipboard is not None:
                self._write_clipboard_safe(context.original_clipboard)

            raise last_error or self._error_class_for_action(request.action)(
                "Clipboard operation failed without a captured exception",
                context={"selector": request.selector},
            )
        except Exception as exc:
            return self._build_error_result(context, exc)

    def _wait_for_element(self, selector: str, *, options: ClipboardOptions) -> WebElement:
        try:
            wait = WebDriverWait(self.driver, options.timeout, poll_frequency=options.poll_frequency)
            if options.require_visible:
                return wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
            return wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        except TimeoutException as exc:
            raise ElementNotFoundError(
                f"Selector '{selector}' not found within {options.timeout}s",
                context={"selector": selector, "timeout": options.timeout},
                cause=exc,
            ) from exc
        except NoSuchElementException as exc:
            raise ElementNotFoundError(
                f"Selector '{selector}' not found",
                context={"selector": selector},
                cause=exc,
            ) from exc

    def _prepare_element(self, element: WebElement, action: str, selector: str, options: ClipboardOptions) -> None:
        if options.scroll_into_view:
            self._scroll_element_into_view(element, options)
        if action in {CUT_ACTION, PASTE_ACTION} and options.require_enabled_for_write:
            self._ensure_writable(element, selector)
        if options.focus_before_action:
            self._focus_element(element)

    def _scroll_element_into_view(self, element: WebElement, options: ClipboardOptions) -> None:
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block: arguments[1], behavior: arguments[2]});",
                element,
                options.scroll_position,
                options.scroll_behavior,
            )
        except (JavascriptException, WebDriverException) as exc:
            raise ClipboardStrategyError("Failed to scroll clipboard target into view", cause=exc) from exc

    def _focus_element(self, element: WebElement) -> None:
        try:
            self.driver.execute_script("arguments[0].focus({preventScroll: true});", element)
        except Exception:
            try:
                ActionChains(self.driver).move_to_element(element).click(element).perform()
            except Exception:
                logger.debug("Unable to focus clipboard target; continuing with fallback strategies")

    def _ensure_writable(self, element: WebElement, selector: str) -> None:
        readonly = str(safe_get_attribute(element, "readonly", default="") or "").lower() in {"true", "readonly"}
        aria_disabled = str(safe_get_attribute(element, "aria-disabled", default="") or "").lower() == "true"
        disabled = safe_get_attribute(element, "disabled", default=None) is not None or aria_disabled
        enabled = safe_call(element.is_enabled, default=True)
        if readonly or disabled or not enabled:
            raise ElementNotInteractableError(
                "Clipboard target is not writable",
                context={"selector": selector, "readonly": readonly, "disabled": disabled, "enabled": enabled},
            )

    def _attempt_action(self, context: ClipboardExecutionContext) -> Dict[str, Any]:
        action = context.request.action
        strategies = context.request.options.strategies_for(action)
        last_error: Optional[BaseException] = None
        for strategy in strategies:
            context.strategy = strategy
            try:
                if action == COPY_ACTION:
                    result = self._run_copy_strategy(context, strategy)
                elif action == CUT_ACTION:
                    result = self._run_cut_strategy(context, strategy)
                else:
                    result = self._run_paste_strategy(context, strategy)
                context.record_attempt(strategy, "success", result.get("message", "Strategy succeeded"), result=result)
                return result
            except Exception as exc:
                last_error = exc
                context.record_attempt(strategy, "error", str(exc), exception=exc)
                logger.debug("Clipboard strategy %s failed for %s: %s", strategy, action, exc)
                if strategy.startswith("javascript") and not context.request.options.allow_javascript_fallback:
                    break
                if strategy in {"keyboard_shortcut", "selection_keyboard"} and not context.request.options.allow_keyboard_fallback:
                    break
        raise ClipboardStrategyError(
            f"All {action} strategies failed",
            context={"action": action, "selector": context.request.selector, "strategies": strategies},
            cause=last_error,
        )

    # ------------------------------------------------------------------
    # Copy strategies
    # ------------------------------------------------------------------
    def _run_copy_strategy(self, context: ClipboardExecutionContext, strategy: str) -> Dict[str, Any]:
        element = self._require_element(context)
        if strategy == "element_text":
            text = self._get_element_text(element, context.request.options)
            self._write_clipboard(text)
            return {"strategy": strategy, "text": text, "message": "Copied element text"}
        if strategy == "selection_keyboard":
            text = self._copy_via_keyboard_selection(element, context.request.options)
            return {"strategy": strategy, "text": text, "message": "Copied via keyboard selection"}
        if strategy == "javascript_selection":
            text = self._copy_via_javascript_selection(element, context.request.options)
            return {"strategy": strategy, "text": text, "message": "Copied via JavaScript selection"}
        raise ClipboardStrategyError(f"Unsupported copy strategy: {strategy}")

    def _copy_via_keyboard_selection(self, element: WebElement, options: ClipboardOptions) -> str:
        text_before = self._get_element_text(element, options)
        if options.select_before_keyboard_action:
            self._select_element_text(element)
        self._send_keyboard_shortcut("copy")
        clipboard_text = self._read_clipboard_safe()
        return _clip_text(clipboard_text or text_before, options)

    def _copy_via_javascript_selection(self, element: WebElement, options: ClipboardOptions) -> str:
        script = """
            const el = arguments[0];
            const tag = (el.tagName || '').toLowerCase();
            let text = '';
            if (tag === 'input' || tag === 'textarea') {
                el.focus();
                if (typeof el.select === 'function') el.select();
                text = el.value || '';
            } else {
                const selection = window.getSelection();
                const range = document.createRange();
                range.selectNodeContents(el);
                selection.removeAllRanges();
                selection.addRange(range);
                text = selection.toString() || el.innerText || el.textContent || '';
            }
            return text;
        """
        try:
            text = self.driver.execute_script(script, element) or ""
            self._write_clipboard(str(text))
            return _clip_text(text, options)
        except (JavascriptException, WebDriverException) as exc:
            raise ClipboardStrategyError("JavaScript selection copy failed", cause=exc) from exc

    # ------------------------------------------------------------------
    # Cut strategies
    # ------------------------------------------------------------------
    def _run_cut_strategy(self, context: ClipboardExecutionContext, strategy: str) -> Dict[str, Any]:
        element = self._require_element(context)
        if strategy == "native_clear":
            text = self._cut_via_native_clear(element, context.request.options)
            return {"strategy": strategy, "text": text, "message": "Cut via native clear"}
        if strategy == "keyboard_shortcut":
            text = self._cut_via_keyboard(element, context.request.options)
            return {"strategy": strategy, "text": text, "message": "Cut via keyboard shortcut"}
        if strategy == "javascript_clear":
            text = self._cut_via_javascript_clear(element, context.request.options)
            return {"strategy": strategy, "text": text, "message": "Cut via JavaScript clear"}
        raise ClipboardStrategyError(f"Unsupported cut strategy: {strategy}")

    def _cut_via_native_clear(self, element: WebElement, options: ClipboardOptions) -> str:
        text = self._get_element_text(element, options)
        self._write_clipboard(text)
        try:
            element.clear()
        except (InvalidElementStateException, ElementNotInteractableException, WebDriverException) as exc:
            raise ClipboardStrategyError("Native clear failed during cut", cause=exc) from exc
        self._dispatch_input_events(element)
        return text

    def _cut_via_keyboard(self, element: WebElement, options: ClipboardOptions) -> str:
        text_before = self._get_element_text(element, options)
        if options.select_before_keyboard_action:
            self._select_element_text(element)
        self._send_keyboard_shortcut("cut")
        text_after = self._get_element_text(element, options)
        if text_after == text_before:
            raise ClipboardStrategyError("Keyboard cut did not mutate target text")
        return _clip_text(self._read_clipboard_safe() or text_before, options)

    def _cut_via_javascript_clear(self, element: WebElement, options: ClipboardOptions) -> str:
        text = self._get_element_text(element, options)
        self._write_clipboard(text)
        script = """
            const el = arguments[0];
            const tag = (el.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea') {
                el.value = '';
            } else if (el.isContentEditable) {
                el.innerHTML = '';
            } else {
                el.textContent = '';
            }
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
            return true;
        """
        try:
            self.driver.execute_script(script, element)
            return text
        except (JavascriptException, WebDriverException) as exc:
            raise ClipboardStrategyError("JavaScript clear failed during cut", cause=exc) from exc

    # ------------------------------------------------------------------
    # Paste strategies
    # ------------------------------------------------------------------
    def _run_paste_strategy(self, context: ClipboardExecutionContext, strategy: str) -> Dict[str, Any]:
        element = self._require_element(context)
        paste_text = self._resolve_paste_text(context.request)
        if strategy == "send_keys":
            self._paste_via_send_keys(element, paste_text, context.request.options)
            return {"strategy": strategy, "text": paste_text, "message": "Pasted via send_keys"}
        if strategy == "keyboard_shortcut":
            self._paste_via_keyboard(element, paste_text, context.request.options)
            return {"strategy": strategy, "text": paste_text, "message": "Pasted via keyboard shortcut"}
        if strategy == "javascript_set_value":
            self._paste_via_javascript(element, paste_text, context.request.options)
            return {"strategy": strategy, "text": paste_text, "message": "Pasted via JavaScript set value"}
        raise ClipboardStrategyError(f"Unsupported paste strategy: {strategy}")

    def _paste_via_send_keys(self, element: WebElement, paste_text: str, options: ClipboardOptions) -> None:
        try:
            if options.clear_before_paste and not options.append_on_paste:
                element.clear()
            element.send_keys(paste_text)
        except (InvalidElementStateException, ElementNotInteractableException, WebDriverException) as exc:
            raise ClipboardStrategyError("send_keys paste failed", cause=exc) from exc
        self._dispatch_input_events(element)

    def _paste_via_keyboard(self, element: WebElement, paste_text: str, options: ClipboardOptions) -> None:
        self._write_clipboard(paste_text)
        if options.clear_before_paste and not options.append_on_paste:
            try:
                element.clear()
            except Exception:
                self._select_element_text(element)
        self._send_keyboard_shortcut("paste")

    def _paste_via_javascript(self, element: WebElement, paste_text: str, options: ClipboardOptions) -> None:
        script = """
            const el = arguments[0];
            const text = arguments[1];
            const clearBefore = arguments[2];
            const append = arguments[3];
            const tag = (el.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea') {
                const existing = el.value || '';
                el.value = append ? existing + text : (clearBefore ? text : existing + text);
            } else if (el.isContentEditable) {
                if (append) {
                    el.insertAdjacentText('beforeend', text);
                } else {
                    el.textContent = text;
                }
            } else {
                if (append) {
                    el.textContent = (el.textContent || '') + text;
                } else {
                    el.textContent = text;
                }
            }
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
            return true;
        """
        try:
            self.driver.execute_script(script, element, paste_text, options.clear_before_paste, options.append_on_paste)
        except (JavascriptException, WebDriverException) as exc:
            raise ClipboardStrategyError("JavaScript paste failed", cause=exc) from exc

    # ------------------------------------------------------------------
    # Shared element and clipboard primitives
    # ------------------------------------------------------------------
    def _require_element(self, context: ClipboardExecutionContext) -> WebElement:
        if context.element is None:
            raise ElementNotFoundError("Clipboard target element is unavailable", context={"selector": context.request.selector})
        return context.element

    def _resolve_paste_text(self, request: ClipboardRequest) -> str:
        if request.text is not None:
            text = request.text
            self._write_clipboard(text)
            return _clip_text(text, request.options)
        clipboard_text = self._read_clipboard()
        return _clip_text(clipboard_text, request.options)

    def _read_clipboard(self) -> str:
        try:
            return pyperclip.paste() or ""
        except Exception as exc:
            raise ClipboardReadError("Failed to read system clipboard", cause=exc) from exc

    def _read_clipboard_safe(self) -> str:
        try:
            return pyperclip.paste() or ""
        except Exception:
            return ""

    def _write_clipboard(self, text: str) -> None:
        try:
            pyperclip.copy(text or "")
        except Exception as exc:
            raise ClipboardWriteError("Failed to write system clipboard", cause=exc) from exc

    def _write_clipboard_safe(self, text: str) -> bool:
        try:
            pyperclip.copy(text or "")
            return True
        except Exception:
            return False

    def _send_keyboard_shortcut(self, action: str) -> None:
        key = {COPY_ACTION: "c", CUT_ACTION: "x", PASTE_ACTION: "v"}.get(action)
        if key is None:
            raise ClipboardValidationError(f"Unsupported keyboard clipboard action: {action}")
        try:
            ActionChains(self.driver).key_down(Keys.CONTROL).send_keys(key).key_up(Keys.CONTROL).perform()
        except WebDriverException as exc:
            raise ClipboardStrategyError(f"Keyboard shortcut failed for {action}", cause=exc) from exc

    def _select_element_text(self, element: WebElement) -> None:
        tag = self._tag_name(element)
        try:
            if tag in TEXT_FORM_TAGS:
                element.click()
                element.send_keys(Keys.CONTROL, "a")
            else:
                script = """
                    const el = arguments[0];
                    const selection = window.getSelection();
                    const range = document.createRange();
                    range.selectNodeContents(el);
                    selection.removeAllRanges();
                    selection.addRange(range);
                    return true;
                """
                self.driver.execute_script(script, element)
        except (JavascriptException, WebDriverException) as exc:
            raise ClipboardStrategyError("Failed to select clipboard target text", cause=exc) from exc

    def _dispatch_input_events(self, element: WebElement) -> None:
        try:
            self.driver.execute_script(
                "arguments[0].dispatchEvent(new Event('input', {bubbles: true}));"
                "arguments[0].dispatchEvent(new Event('change', {bubbles: true}));",
                element,
            )
        except Exception:
            logger.debug("Unable to dispatch input/change events for clipboard target")

    def _tag_name(self, element: WebElement) -> str:
        try:
            return str(element.tag_name or "").lower()
        except Exception:
            return ""

    def _is_text_form_element(self, element: WebElement) -> bool:
        tag = self._tag_name(element)
        input_type = str(safe_get_attribute(element, "type", default="") or "").lower()
        return tag == "textarea" or (tag == "input" and input_type in TEXTUAL_INPUT_TYPES)

    def _get_element_text(self, element: WebElement, options: Optional[ClipboardOptions] = None) -> str:
        """Extract clipboard-relevant text from a target element."""

        max_chars = options.max_text_chars if options else 5_000
        candidates = []
        tag = self._tag_name(element)
        if tag in TEXT_FORM_TAGS:
            candidates.extend(
                [
                    safe_get_attribute(element, "value", default=""),
                    safe_get_attribute(element, "placeholder", default=""),
                ]
            )
        candidates.extend(
            [
                safe_call(lambda: getattr(element, "text", ""), default=""),
                safe_get_attribute(element, "innerText", default=""),
                safe_get_attribute(element, "textContent", default=""),
                safe_get_attribute(element, "aria-label", default=""),
                safe_get_attribute(element, "title", default=""),
            ]
        )
        for candidate in candidates:
            text = normalize_newlines(candidate or "")
            if text:
                return truncate_text(text, max_chars)
        return ""

    def _extract_context_metadata(self, element: WebElement, options: Optional[ClipboardOptions] = None) -> dict:
        """Return stable element metadata for backwards-compatible callers."""

        options = options or ClipboardOptions.from_config(self.ccp_config)
        metadata = element_metadata(
            element,
            max_text=options.max_element_text_chars,
            max_html=options.max_element_html_chars if options.include_element_html else 0,
        )
        metadata["text_length"] = len(self._get_element_text(element, options))
        metadata["is_text_form_element"] = self._is_text_form_element(element)
        return metadata

    def _capture_state(self, *, label: str, element: Optional[WebElement], options: ClipboardOptions) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "label": label,
            "url": get_current_url(self.driver),
            "title": get_page_title(self.driver),
            "ready_state": get_document_ready_state(self.driver),
            "clipboard_length": len(self._read_clipboard_safe()),
        }
        if element is not None:
            state["element"] = self._extract_context_metadata(element, options)
            state["element_text_length"] = len(self._get_element_text(element, options))
            state["element_text_fingerprint"] = fingerprint_text(self._get_element_text(element, options))
        if options.include_page_snapshot:
            state["page"] = page_snapshot_dict(
                self.driver,
                include_html=False,
                include_screenshot=False,
                max_text=options.max_element_text_chars,
            )
        return prune_none(state)

    def _verify_action(self, context: ClipboardExecutionContext, result: Dict[str, Any]) -> Dict[str, Any]:
        request = context.request
        options = request.options
        action = request.action
        element = self._require_element(context)
        verification: Dict[str, Any] = {"checked": True, "signals": {}}

        clipboard_text = self._read_clipboard_safe()
        result_text = str(result.get("text", "") or "")
        element_text_after = self._get_element_text(element, options)
        element_text_before_fingerprint = context.before.get("element_text_fingerprint")
        element_text_after_fingerprint = fingerprint_text(element_text_after)

        if action in {COPY_ACTION, CUT_ACTION} and options.verify_clipboard_write:
            verification["signals"]["clipboard_matches_result"] = clipboard_text == result_text
            if result_text and clipboard_text != result_text:
                raise ClipboardVerificationError(
                    "Clipboard contents did not match copied/cut text",
                    context={
                        "action": action,
                        "selector": request.selector,
                        "clipboard_length": len(clipboard_text),
                        "result_text_length": len(result_text),
                    },
                )

        if action == CUT_ACTION and options.verify_element_mutation:
            changed = element_text_before_fingerprint != element_text_after_fingerprint
            verification["signals"]["element_text_changed"] = changed
            verification["signals"]["element_text_after_length"] = len(element_text_after)
            if result_text and not changed:
                raise ClipboardVerificationError(
                    "Cut operation did not mutate the target element",
                    context={"selector": request.selector, "text_length": len(result_text)},
                )

        if action == PASTE_ACTION and options.verify_element_mutation:
            expected = request.text if request.text is not None else clipboard_text
            verification["signals"]["element_contains_paste_text"] = bool(expected) and str(expected) in element_text_after
            verification["signals"]["element_text_after_length"] = len(element_text_after)
            if expected and str(expected) not in element_text_after:
                raise ClipboardVerificationError(
                    "Paste text was not observed in the target element",
                    context={"selector": request.selector, "expected_length": len(str(expected)), "actual_length": len(element_text_after)},
                )

        verification["signals"]["clipboard_length"] = len(clipboard_text)
        return verification

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------
    def _build_success_result(
        self,
        context: ClipboardExecutionContext,
        result: Mapping[str, Any],
        verification: Mapping[str, Any],
    ) -> dict:
        request = context.request
        options = request.options
        raw_text = str(result.get("text", "") or "")
        public_text = _public_text(raw_text, options)
        data = {
            "text": public_text,
            "text_length": len(raw_text),
            "text_fingerprint": fingerprint_text(raw_text),
            "strategy": result.get("strategy"),
            "selector": request.selector,
            "verification": verification,
            "before": context.before,
            "after": context.after,
        }
        metadata = {
            "request": request.to_dict(),
            "attempts": context.attempts,
            "element": self._extract_context_metadata(context.element, options) if context.element is not None else None,
        }
        message = str(result.get("message") or f"{request.action.title()} completed")
        payload = success_result(
            action=request.action,
            message=message,
            data=data,
            metadata=metadata,
            duration_ms=elapsed_ms(context.start_ms),
            correlation_id=request.correlation_id,
        )
        # Preserve the existing simple top-level fields expected by callers.
        payload["text"] = public_text
        payload["metadata"] = merge_dicts(payload.get("metadata", {}), {"legacy_metadata": metadata.get("element")}, deep=True)
        return payload

    def _build_error_result(self, context: ClipboardExecutionContext, exc: BaseException) -> dict:
        request = context.request
        options = request.options
        error = BrowserError.from_exception(
            exc,
            action=request.action,
            context={"selector": request.selector, "attempts": context.attempts},
            default_error_cls=self._error_class_for_action(request.action),
        )
        diagnostics: Dict[str, Any] = {
            "request": request.to_dict(),
            "attempts": context.attempts,
            "before": context.before,
            "after": context.after,
        }
        if context.element is not None:
            diagnostics["element"] = self._extract_context_metadata(context.element, options)
        if options.screenshot_on_error:
            diagnostics["screenshot_b64"] = capture_screenshot_b64(self.driver)
        payload = error_result(
            action=request.action,
            message=str(getattr(error, "message", str(error))),
            error=error,
            metadata=diagnostics,
            duration_ms=elapsed_ms(context.start_ms),
            correlation_id=request.correlation_id,
        )
        payload.setdefault("element", diagnostics.get("element"))
        return payload

    def _error_class_for_action(self, action: str):
        if action == COPY_ACTION:
            return CopyError
        if action == CUT_ACTION:
            return CutError
        if action == PASTE_ACTION:
            return PasteError
        return ClipboardError


if __name__ == "__main__":
    print("\n=== Running Do Copy Cut Paste ===\n")
    printer.status("TEST", "Do Copy Cut Paste initialized", "info")

    class FakeElement:
        def __init__(self, selector: str, text: str = "Hello clipboard", tag_name: str = "textarea"):
            self.selector = selector
            self.tag_name = tag_name
            self.text = text if tag_name not in TEXT_FORM_TAGS else ""
            self.value = text if tag_name in TEXT_FORM_TAGS else ""
            self.cleared = False
            self.sent_keys: List[str] = []
            self.attributes = {
                "id": selector.replace("#", ""),
                "name": selector.replace("#", ""),
                "role": "textbox" if tag_name in TEXT_FORM_TAGS else None,
                "placeholder": "placeholder text",
                "aria-label": "Fake clipboard element",
                "class": "fake clipboard-target",
                "outerHTML": f"<{tag_name} id='{selector.replace('#', '')}'>fake</{tag_name}>",
                "value": self.value,
                "type": "text",
            }

        def get_attribute(self, name: str):
            if name == "value":
                return self.value
            if name == "innerText":
                return self.text or self.value
            if name == "textContent":
                return self.text or self.value
            return self.attributes.get(name)

        def is_enabled(self):
            return True

        def is_displayed(self):
            return True

        def is_selected(self):
            return False

        def click(self):
            return None

        def clear(self):
            self.cleared = True
            self.value = ""
            self.text = "" if self.tag_name not in TEXT_FORM_TAGS else self.text
            self.attributes["value"] = self.value

        def send_keys(self, *keys):
            text = "".join(str(k) for k in keys)
            self.sent_keys.append(text)
            self.value += text
            self.attributes["value"] = self.value

    class FakeDriver:
        def __init__(self):
            self.current_url = "https://example.test/form"
            self.title = "Clipboard Test"
            self.elements = {
                "#field": FakeElement("#field", "Hello clipboard", "textarea"),
                "#div": FakeElement("#div", "Read-only visible text", "div"),
            }

        def find_element(self, by=None, value=None):
            selector = value if value is not None else by
            if selector not in self.elements:
                raise NoSuchElementException(selector)
            return self.elements[selector]

        def execute_script(self, script, *args):
            if "document.readyState" in script:
                return "complete"
            if "document.title" in script:
                return self.title
            if "window.getSelection" in script and args:
                return args[0].get_attribute("textContent")
            if "dispatchEvent" in script:
                return True
            if "scrollIntoView" in script:
                return True
            if "focus" in script:
                return True
            if args and len(args) >= 2 and "const text = arguments[1]" in script:
                element, text = args[0], args[1]
                element.value = str(text)
                element.attributes["value"] = element.value
                return True
            if args and "textContent = ''" in script:
                element = args[0]
                element.text = ""
                element.value = ""
                element.attributes["value"] = ""
                return True
            return None

        def get_screenshot_as_base64(self):
            return ""

    class TestableDoCopyCutPaste(DoCopyCutPaste):
        def __init__(self, driver):
            self.config = {}
            self.ccp_config = {
                "timeout": 0.1,
                "poll_frequency": 0.05,
                "retries": 0,
                "focus_before_action": False,
                "scroll_into_view": False,
                "verify_clipboard_write": False,
                "verify_element_mutation": False,
                "copy_strategies": ["element_text"],
                "cut_strategies": ["native_clear"],
                "paste_strategies": ["send_keys"],
            }
            self.driver = driver

        def _wait_for_element(self, selector: str, *, options: ClipboardOptions) -> WebElement:
            return self.driver.find_element(By.CSS_SELECTOR, selector)

        def _read_clipboard(self) -> str:
            return getattr(self, "_fake_clipboard", "")

        def _read_clipboard_safe(self) -> str:
            return getattr(self, "_fake_clipboard", "")

        def _write_clipboard(self, text: str) -> None:
            self._fake_clipboard = text or ""

        def _write_clipboard_safe(self, text: str) -> bool:
            self._fake_clipboard = text or ""
            return True

    fake_driver = FakeDriver()
    clipboard = TestableDoCopyCutPaste(fake_driver)

    copy_result = clipboard.copy("#field")
    assert copy_result["status"] == "success", copy_result
    assert copy_result["data"]["text_length"] > 0, copy_result

    cut_result = clipboard.cut("#field")
    assert cut_result["status"] == "success", cut_result
    assert fake_driver.elements["#field"].value == "", cut_result

    paste_result = clipboard.paste("#field", text="Pasted value")
    assert paste_result["status"] == "success", paste_result
    assert "Pasted value" in fake_driver.elements["#field"].value, paste_result

    missing_result = clipboard.copy("#missing")
    assert missing_result["status"] == "error", missing_result

    print("\n=== Test ran successfully ===\n")
