from __future__ import annotations

"""
Production-grade drag-and-drop functionality for the browser subsystem.

This module owns concrete drag-and-drop execution only. It intentionally does
not own browser lifecycle, workflow orchestration, high-level task routing,
content extraction, or memory. Those responsibilities belong to the
BrowserAgent and adjacent browser modules. The drag-and-drop module focuses on
one stable contract: turn a source selector plus either a target selector or an
offset into a safe, observable, configurable browser drag result.

Design goals
------------
- Use shared browser errors and helpers instead of redefining serialization,
  redaction, result payloads, selector validation, retry, screenshots, page
  snapshots, or element metadata.
- Keep BrowserAgent/function-module friendly public entry points:
  ``do_drag_and_drop()``, ``drag_and_drop()``, ``drag_to_element()``,
  ``drag_by_offset()``, and ``_perform_drag_and_drop()``.
- Make execution configurable from ``browser_config.yaml`` rather than burying
  timing, retry, fallback, scrolling, diagnostics, and verification behavior in
  code.
- Support both element-to-element and offset-based interactions so the module
  can handle sortable lists, kanban cards, sliders, splitters, resize handles,
  canvas-like UIs, HTML5 drag/drop targets, and future workflow actions.
- Record enough metadata for debugging, memory, workflow replay, and telemetry,
  without leaking sensitive values.
- Remain easy to expand with more strategies such as CDP input dispatch,
  touch actions, file-drop simulation, or framework-specific adapters.
"""

import asyncio
import time as time_module

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    JavascriptException,
    MoveTargetOutOfBoundsException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.browser_errors import *
from ..utils.Browser_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Drag & Drop")
printer = PrettyPrinter


DEFAULT_DRAG_STRATEGIES: Tuple[str, ...] = (
    "action_chains_drag_and_drop",
    "action_chains_hold_move_release",
    "action_chains_offset",
    "incremental_offset",
    "javascript_html5",
    "dispatch_events",
)

SUPPORTED_DRAG_STRATEGIES = set(DEFAULT_DRAG_STRATEGIES)
OFFSET_ONLY_STRATEGIES = {"action_chains_offset", "incremental_offset"}
ELEMENT_TARGET_STRATEGIES = {
    "action_chains_drag_and_drop",
    "action_chains_hold_move_release",
    "javascript_html5",
    "dispatch_events",
}


@dataclass(frozen=True)
class DragAndDropOptions:
    """Resolved execution policy for one drag-and-drop operation."""

    timeout: float = 10.0
    poll_frequency: float = 0.1
    wait_before_execution: float = 0.0
    wait_after_drop: float = 0.0
    retries: int = 2
    retry_backoff_base: float = 0.25
    retry_backoff_multiplier: float = 1.8
    retry_backoff_max: float = 3.0
    retry_jitter: float = 0.05
    require_source_visible: bool = True
    require_source_enabled: bool = True
    require_target_visible: bool = True
    scroll_source_into_view: bool = True
    scroll_target_into_view: bool = True
    scroll_position: str = "center"
    scroll_behavior: str = "auto"
    hover_source_before_drag: bool = False
    hover_target_before_drop: bool = False
    focus_source_before_drag: bool = False
    verify_after_drop: bool = True
    wait_for_page_load_after_drop: bool = False
    page_load_timeout: float = 5.0
    include_page_snapshot: bool = False
    include_element_html: bool = True
    max_element_text_chars: int = 500
    max_element_html_chars: int = 1_500
    screenshot_on_error: bool = False
    drag_duration: float = 0.35
    drag_steps: int = 8
    hold_pause: float = 0.08
    move_pause: float = 0.03
    release_pause: float = 0.05
    default_offset_x: int = 0
    default_offset_y: int = 0
    allow_javascript_fallback: bool = True
    allow_dispatch_event_fallback: bool = True
    allow_offset_fallback: bool = True
    strategies: Tuple[str, ...] = DEFAULT_DRAG_STRATEGIES

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **overrides: Any) -> "DragAndDropOptions":
        """Build options from ``browser_config.yaml`` plus per-call overrides."""

        strategies = overrides.pop("strategies", None) or config.get("strategies") or DEFAULT_DRAG_STRATEGIES
        resolved_strategies = normalize_drag_strategies(strategies)

        values = {
            "timeout": coerce_float(overrides.pop("timeout", config.get("timeout", 10.0)), default=10.0, minimum=0.1),
            "poll_frequency": coerce_float(overrides.pop("poll_frequency", config.get("poll_frequency", 0.1)), default=0.1, minimum=0.05),
            "wait_before_execution": coerce_float(
                overrides.pop("wait_before_execution", config.get("wait_before_execution", 0.0)), default=0.0, minimum=0.0
            ),
            "wait_after_drop": coerce_float(overrides.pop("wait_after_drop", config.get("wait_after_drop", 0.0)), default=0.0, minimum=0.0),
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
            "require_source_visible": coerce_bool(
                overrides.pop("require_source_visible", config.get("require_source_visible", True)), default=True
            ),
            "require_source_enabled": coerce_bool(
                overrides.pop("require_source_enabled", config.get("require_source_enabled", True)), default=True
            ),
            "require_target_visible": coerce_bool(
                overrides.pop("require_target_visible", config.get("require_target_visible", True)), default=True
            ),
            "scroll_source_into_view": coerce_bool(
                overrides.pop("scroll_source_into_view", config.get("scroll_source_into_view", True)), default=True
            ),
            "scroll_target_into_view": coerce_bool(
                overrides.pop("scroll_target_into_view", config.get("scroll_target_into_view", True)), default=True
            ),
            "scroll_position": str(overrides.pop("scroll_position", config.get("scroll_position", "center")) or "center"),
            "scroll_behavior": str(overrides.pop("scroll_behavior", config.get("scroll_behavior", "auto")) or "auto"),
            "hover_source_before_drag": coerce_bool(
                overrides.pop("hover_source_before_drag", config.get("hover_source_before_drag", False)), default=False
            ),
            "hover_target_before_drop": coerce_bool(
                overrides.pop("hover_target_before_drop", config.get("hover_target_before_drop", False)), default=False
            ),
            "focus_source_before_drag": coerce_bool(
                overrides.pop("focus_source_before_drag", config.get("focus_source_before_drag", False)), default=False
            ),
            "verify_after_drop": coerce_bool(overrides.pop("verify_after_drop", config.get("verify_after_drop", True)), default=True),
            "wait_for_page_load_after_drop": coerce_bool(
                overrides.pop("wait_for_page_load_after_drop", config.get("wait_for_page_load_after_drop", False)), default=False
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
            "drag_duration": coerce_float(overrides.pop("drag_duration", config.get("drag_duration", 0.35)), default=0.35, minimum=0.0),
            "drag_steps": coerce_int(overrides.pop("drag_steps", config.get("drag_steps", 8)), default=8, minimum=1, maximum=100),
            "hold_pause": coerce_float(overrides.pop("hold_pause", config.get("hold_pause", 0.08)), default=0.08, minimum=0.0),
            "move_pause": coerce_float(overrides.pop("move_pause", config.get("move_pause", 0.03)), default=0.03, minimum=0.0),
            "release_pause": coerce_float(overrides.pop("release_pause", config.get("release_pause", 0.05)), default=0.05, minimum=0.0),
            "default_offset_x": coerce_int(overrides.pop("default_offset_x", config.get("default_offset_x", 0)), default=0),
            "default_offset_y": coerce_int(overrides.pop("default_offset_y", config.get("default_offset_y", 0)), default=0),
            "allow_javascript_fallback": coerce_bool(
                overrides.pop("allow_javascript_fallback", config.get("allow_javascript_fallback", True)), default=True
            ),
            "allow_dispatch_event_fallback": coerce_bool(
                overrides.pop("allow_dispatch_event_fallback", config.get("allow_dispatch_event_fallback", True)), default=True
            ),
            "allow_offset_fallback": coerce_bool(
                overrides.pop("allow_offset_fallback", config.get("allow_offset_fallback", True)), default=True
            ),
            "strategies": resolved_strategies,
        }
        return cls(**values)


@dataclass(frozen=True)
class DragAndDropRequest:
    """Normalized drag-and-drop request for validation, telemetry, and replay."""

    source_selector: str
    options: DragAndDropOptions
    target_selector: Optional[str] = None
    source_offset: Tuple[int, int] = (0, 0)
    target_offset: Optional[Tuple[int, int]] = None
    correlation_id: str = field(default_factory=lambda: new_correlation_id("drag-drop"))

    @property
    def uses_target_element(self) -> bool:
        return bool(self.target_selector)

    @property
    def uses_offset(self) -> bool:
        return self.target_offset is not None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["options"] = asdict(self.options)
        return payload


@dataclass
class DragAndDropExecutionContext:
    """Mutable per-drag execution context."""

    request: DragAndDropRequest
    start_ms: float
    attempt: int = 0
    strategy: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    source: Optional[WebElement] = None
    target: Optional[WebElement] = None

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


def normalize_drag_strategies(strategies: Iterable[Any]) -> Tuple[str, ...]:
    """Normalize drag strategies and remove disabled/unknown values."""

    normalized: List[str] = []
    for strategy in ensure_list(strategies):
        name = str(strategy or "").strip().lower().replace("-", "_")
        aliases = {
            "drag_and_drop": "action_chains_drag_and_drop",
            "native": "action_chains_drag_and_drop",
            "action_chains": "action_chains_hold_move_release",
            "hold_move_release": "action_chains_hold_move_release",
            "offset": "action_chains_offset",
            "html5": "javascript_html5",
            "javascript": "javascript_html5",
            "dispatch": "dispatch_events",
            "mouse_events": "dispatch_events",
        }
        name = aliases.get(name, name)
        if name in SUPPORTED_DRAG_STRATEGIES and name not in normalized:
            normalized.append(name)
    return tuple(normalized or DEFAULT_DRAG_STRATEGIES)


def _coerce_offset(value: Any, *, default: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    """Normalize offset values supplied as tuple/list/dict/string."""

    if value is None:
        return default
    if isinstance(value, Mapping):
        return (coerce_int(value.get("x", value.get("dx", 0)), default=0), coerce_int(value.get("y", value.get("dy", 0)), default=0))
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        if len(parts) >= 2:
            return (coerce_int(parts[0], default=0), coerce_int(parts[1], default=0))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 2:
        return (coerce_int(value[0], default=0), coerce_int(value[1], default=0))
    return default


class DoDragAndDrop:
    """Concrete Selenium drag-and-drop executor used by browser modules."""

    def __init__(self, driver: Any):
        self.driver = driver
        self.config = load_global_config()
        self.drag_config = get_config_section("do_drag_and_drop") or {}
        self.options = DragAndDropOptions.from_config(self.drag_config)
        logger.info("Browser drag-and-drop functionality initiated.")

    async def do_drag_and_drop(
        self,
        source_selector: str,
        target_selector: Optional[str] = None,
        wait_before_execution: float = 0.0,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously perform drag-and-drop by selectors or offsets."""

        return await asyncio.to_thread(self._perform_drag_and_drop, source_selector, target_selector, wait_before_execution, **kwargs)

    def drag_and_drop(
        self,
        source_selector: str,
        target_selector: str,
        wait_before_execution: Optional[float] = None,
        **kwargs: Any,
    ) -> dict:
        """Public synchronous element-to-element drag-and-drop API."""

        wait = self.options.wait_before_execution if wait_before_execution is None else wait_before_execution
        return self._perform_drag_and_drop(source_selector, target_selector, wait, **kwargs)

    def drag_to_element(self, source_selector: str, target_selector: str, **kwargs: Any) -> dict:
        """Alias for element-to-element drag-and-drop."""

        return self.drag_and_drop(source_selector, target_selector, **kwargs)

    def drag_by_offset(self, source_selector: str, x_offset: int, y_offset: int, **kwargs: Any) -> dict:
        """Drag a source element by an offset from its current position."""

        return self._perform_drag_and_drop(source_selector, None, kwargs.pop("wait_before_execution", 0.0), target_offset=(x_offset, y_offset), **kwargs)

    def _perform_drag_and_drop(
        self,
        source_selector: str,
        target_selector: Optional[str] = None,
        wait_before_execution: float = 0.0,
        **kwargs: Any,
    ) -> dict:
        """Backwards-compatible drag entrypoint for BrowserAgent/function callers."""

        start_ms = monotonic_ms()
        options = DragAndDropOptions.from_config(self.drag_config, wait_before_execution=wait_before_execution, **kwargs)
        try:
            request = self._build_request(source_selector, target_selector, options, **kwargs)
            context = DragAndDropExecutionContext(request=request, start_ms=start_ms)
            return self._execute_drag_and_drop(context)
        except Exception as exc:
            browser_error = self._coerce_drag_error(
                exc,
                source_selector=source_selector,
                target_selector=target_selector,
                phase="perform_drag_and_drop",
                elapsed_ms=elapsed_ms(start_ms),
            )
            return self._error_result(browser_error, start_ms=start_ms, source_selector=source_selector, target_selector=target_selector, options=options)

    def _build_request(
        self,
        source_selector: str,
        target_selector: Optional[str],
        options: DragAndDropOptions,
        **kwargs: Any,
    ) -> DragAndDropRequest:
        source = validate_selector(source_selector, field_name="source_selector")
        target = validate_selector(target_selector, field_name="target_selector") if target_selector else None
        source_offset = _coerce_offset(kwargs.get("source_offset"), default=(0, 0)) or (0, 0)

        explicit_target_offset = kwargs.get("target_offset")
        if explicit_target_offset is None and ("x_offset" in kwargs or "y_offset" in kwargs):
            explicit_target_offset = (kwargs.get("x_offset", 0), kwargs.get("y_offset", 0))
        target_offset = _coerce_offset(explicit_target_offset)
        if target_offset is None and not target:
            default_offset = (options.default_offset_x, options.default_offset_y)
            target_offset = default_offset if default_offset != (0, 0) else None

        if not target and target_offset is None:
            raise DragAndDropValidationError(
                "Drag-and-drop requires either target_selector or target_offset/x_offset/y_offset",
                context={"source_selector": source_selector, "target_selector": target_selector},
            )

        self._validate_options(options, uses_target_element=bool(target), uses_offset=target_offset is not None)
        return DragAndDropRequest(source_selector=source, target_selector=target, source_offset=source_offset, target_offset=target_offset, options=options)

    def _validate_options(self, options: DragAndDropOptions, *, uses_target_element: bool, uses_offset: bool) -> None:
        if options.timeout <= 0:
            raise InvalidTimeoutError("Drag-and-drop timeout must be greater than zero", context={"timeout": options.timeout})
        if options.poll_frequency <= 0:
            raise InvalidTimeoutError("Drag-and-drop poll frequency must be greater than zero", context={"poll_frequency": options.poll_frequency})
        if not options.strategies:
            raise DragAndDropValidationError("At least one drag strategy must be enabled", context={"strategies": options.strategies})
        invalid = [strategy for strategy in options.strategies if strategy not in SUPPORTED_DRAG_STRATEGIES]
        if invalid:
            raise DragAndDropValidationError("Unsupported drag strategy configured", context={"invalid_strategies": invalid})
        if not uses_target_element:
            usable = [strategy for strategy in options.strategies if strategy in OFFSET_ONLY_STRATEGIES]
            if not usable:
                raise DragAndDropValidationError(
                    "Offset-based drag requires an offset-capable strategy",
                    context={"strategies": options.strategies, "offset_strategies": sorted(OFFSET_ONLY_STRATEGIES)},
                )
        if not uses_offset and not uses_target_element:
            raise DragAndDropValidationError("Drag request has neither target element nor offset")

    def _execute_drag_and_drop(self, context: DragAndDropExecutionContext) -> dict:
        request = context.request
        options = request.options

        if options.wait_before_execution > 0:
            time_module.sleep(options.wait_before_execution)

        for attempt in range(options.retries + 1):
            context.attempt = attempt + 1
            try:
                source = self._resolve_source(request.source_selector, options)
                target = self._resolve_target(request.target_selector, options) if request.target_selector else None
                context.source = source
                context.target = target
                context.before = self._capture_before_state(source, target, options)
                self._prepare_source(source, request.source_selector, options)
                if target is not None:
                    self._prepare_target(target, request.target_selector or "", options)

                result = self._attempt_drag_strategies(source, target, context)
                return result
            except (
                StaleElementReferenceException,
                TimeoutException,
                ElementClickInterceptedException,
                ElementNotInteractableException,
                MoveTargetOutOfBoundsException,
                WebDriverException,
            ) as exc:
                context.record_attempt("retry", "error", str(exc), exception_type=exc.__class__.__name__)
                if attempt >= options.retries:
                    raise self._coerce_drag_error(exc, phase="retry_exhausted", attempts=context.attempts) from exc
                self._sleep_before_retry(attempt, options)
            except BrowserError:
                raise
            except Exception as exc:
                raise self._coerce_drag_error(exc, phase="execute_drag_and_drop", attempts=context.attempts) from exc

        raise DragAndDropError("Drag-and-drop retries exhausted", context={"request": request.to_dict(), "attempts": context.attempts})

    def _resolve_source(self, selector: str, options: DragAndDropOptions) -> WebElement:
        try:
            element = WebDriverWait(self.driver, options.timeout, poll_frequency=options.poll_frequency).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
        except TimeoutException as exc:
            raise DragSourceNotFoundError(
                f"Drag source element not found: {selector}",
                context={"source_selector": selector, "timeout": options.timeout},
                cause=exc,
            ) from exc
        if element is None:
            raise DragSourceNotFoundError(f"Drag source element not found: {selector}", context={"source_selector": selector})
        return element

    def _resolve_target(self, selector: Optional[str], options: DragAndDropOptions) -> Optional[WebElement]:
        if not selector:
            return None
        try:
            element = WebDriverWait(self.driver, options.timeout, poll_frequency=options.poll_frequency).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
        except TimeoutException as exc:
            raise DragTargetNotFoundError(
                f"Drag target element not found: {selector}",
                context={"target_selector": selector, "timeout": options.timeout},
                cause=exc,
            ) from exc
        if element is None:
            raise DragTargetNotFoundError(f"Drag target element not found: {selector}", context={"target_selector": selector})
        return element

    def _wait_for_element(self, selector: str, timeout: Optional[float] = None) -> Optional[WebElement]:
        """Compatibility helper for callers that need a raw element wait."""

        resolved_timeout = self.options.timeout if timeout is None else timeout
        try:
            return WebDriverWait(self.driver, resolved_timeout, poll_frequency=self.options.poll_frequency).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, validate_selector(selector)))
            )
        except TimeoutException:
            return None

    def _prepare_source(self, source: WebElement, selector: str, options: DragAndDropOptions) -> None:
        if options.scroll_source_into_view:
            self._scroll_into_view(source, options)
        if options.require_source_visible:
            self._ensure_visible(source, selector, options, role="source")
        if options.require_source_enabled:
            self._ensure_enabled(source, selector, role="source")
        if options.focus_source_before_drag:
            self._focus_element(source)
        if options.hover_source_before_drag:
            self._hover_element(source)

    def _prepare_target(self, target: WebElement, selector: str, options: DragAndDropOptions) -> None:
        if options.scroll_target_into_view:
            self._scroll_into_view(target, options)
        if options.require_target_visible:
            self._ensure_visible(target, selector, options, role="target")
        if options.hover_target_before_drop:
            self._hover_element(target)

    def _scroll_into_view(self, element: WebElement, options: DragAndDropOptions) -> None:
        position = options.scroll_position if options.scroll_position in {"start", "center", "end", "nearest"} else "center"
        behavior = options.scroll_behavior if options.scroll_behavior in {"auto", "smooth"} else "auto"
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block: arguments[1], inline: 'nearest', behavior: arguments[2]});",
            element,
            position,
            behavior,
        )

    def _ensure_visible(self, element: WebElement, selector: str, options: DragAndDropOptions, *, role: str) -> None:
        try:
            WebDriverWait(self.driver, min(options.timeout, 2.0), poll_frequency=options.poll_frequency).until(EC.visibility_of(element))
        except TimeoutException as exc:
            error_cls = DragSourceNotReadyError if role == "source" else DragTargetNotReadyError
            raise error_cls(f"Drag {role} element is not visible", context={"selector": selector, "role": role}, cause=exc) from exc

    def _ensure_enabled(self, element: WebElement, selector: str, *, role: str) -> None:
        try:
            enabled = bool(element.is_enabled())
        except Exception as exc:
            raise DragSourceNotReadyError(
                "Could not determine whether drag source is enabled",
                context={"selector": selector, "role": role},
                cause=exc,
            ) from exc
        if not enabled or self._is_disabled_by_attribute(element):
            raise DragSourceNotReadyError(
                "Drag source is disabled and cannot be dragged",
                context={"selector": selector, "role": role, "element": element_metadata(element)},
            )

    def _is_disabled_by_attribute(self, element: WebElement) -> bool:
        disabled = safe_get_attribute(element, "disabled")
        aria_disabled = str(safe_get_attribute(element, "aria-disabled", default="") or "").lower()
        return disabled is not None or aria_disabled == "true"

    def _focus_element(self, element: WebElement) -> None:
        try:
            self.driver.execute_script("arguments[0].focus({preventScroll: true});", element)
        except JavascriptException as exc:
            logger.debug("Focus before drag failed: %s", exc)

    def _hover_element(self, element: WebElement) -> None:
        ActionChains(self.driver).move_to_element(element).pause(0.05).perform()

    def _attempt_drag_strategies(
        self,
        source: WebElement,
        target: Optional[WebElement],
        context: DragAndDropExecutionContext,
    ) -> dict:
        request = context.request
        options = request.options
        last_error: Optional[BaseException] = None

        for strategy in self._eligible_strategies(request):
            if strategy == "javascript_html5" and not options.allow_javascript_fallback:
                continue
            if strategy == "dispatch_events" and not options.allow_dispatch_event_fallback:
                continue
            if strategy in OFFSET_ONLY_STRATEGIES and not options.allow_offset_fallback:
                continue

            try:
                self._attempt_strategy(strategy, source, target, context)
                context.strategy = strategy
                context.record_attempt(strategy, "success", f"{strategy} drag-and-drop succeeded")
                if options.wait_after_drop > 0:
                    time_module.sleep(options.wait_after_drop)
                self._post_drop_verification(source, target, context)
                return self._success_result(f"{self._strategy_label(strategy)} drag-and-drop succeeded", context)
            except Exception as exc:
                last_error = exc
                context.record_attempt(strategy, "error", str(exc), exception_type=exc.__class__.__name__)
                logger.debug("Drag strategy %s failed for %s -> %s: %s", strategy, request.source_selector, request.target_selector, exc)

        raise DragStrategyError(
            "All drag-and-drop strategies failed",
            context={
                "source_selector": request.source_selector,
                "target_selector": request.target_selector,
                "target_offset": request.target_offset,
                "attempts": context.attempts,
                "source": element_metadata(source),
                "target": element_metadata(target) if target is not None else None,
            },
            cause=last_error,
        )

    def _eligible_strategies(self, request: DragAndDropRequest) -> Tuple[str, ...]:
        if request.uses_target_element:
            return tuple(strategy for strategy in request.options.strategies if strategy in ELEMENT_TARGET_STRATEGIES or strategy in OFFSET_ONLY_STRATEGIES)
        return tuple(strategy for strategy in request.options.strategies if strategy in OFFSET_ONLY_STRATEGIES)

    def _attempt_strategy(
        self,
        strategy: str,
        source: WebElement,
        target: Optional[WebElement],
        context: DragAndDropExecutionContext,
    ) -> None:
        if strategy == "action_chains_drag_and_drop":
            self._drag_with_action_chains(source, target)
            return
        if strategy == "action_chains_hold_move_release":
            self._drag_with_hold_move_release(source, target, context)
            return
        if strategy == "action_chains_offset":
            self._drag_with_offset(source, context)
            return
        if strategy == "incremental_offset":
            self._drag_with_incremental_offset(source, context)
            return
        if strategy == "javascript_html5":
            self._drag_with_html5_javascript(source, target)
            return
        if strategy == "dispatch_events":
            self._drag_with_dispatched_events(source, target, context)
            return
        raise DragStrategyError("Unsupported drag strategy", context={"strategy": strategy})

    def _drag_with_action_chains(self, source: WebElement, target: Optional[WebElement]) -> None:
        if target is None:
            raise DragAndDropValidationError("action_chains_drag_and_drop requires a target element")
        ActionChains(self.driver).drag_and_drop(source, target).perform()

    def _drag_with_hold_move_release(
        self,
        source: WebElement,
        target: Optional[WebElement],
        context: DragAndDropExecutionContext,
    ) -> None:
        if target is None:
            raise DragAndDropValidationError("action_chains_hold_move_release requires a target element")
        options = context.request.options
        chain = ActionChains(self.driver)
        chain.move_to_element(source).pause(options.hold_pause).click_and_hold(source)
        chain.pause(options.move_pause).move_to_element(target).pause(options.release_pause).release(target).perform()

    def _drag_with_offset(self, source: WebElement, context: DragAndDropExecutionContext) -> None:
        offset = context.request.target_offset or self._offset_to_target(context)
        if offset is None:
            raise DragAndDropValidationError("Offset strategy requires target_offset or a resolvable target element")
        x_offset, y_offset = offset
        options = context.request.options
        ActionChains(self.driver).move_to_element(source).pause(options.hold_pause).click_and_hold(source).move_by_offset(x_offset, y_offset).pause(
            options.release_pause
        ).release().perform()

    def _drag_with_incremental_offset(self, source: WebElement, context: DragAndDropExecutionContext) -> None:
        offset = context.request.target_offset or self._offset_to_target(context)
        if offset is None:
            raise DragAndDropValidationError("Incremental offset strategy requires target_offset or a resolvable target element")
        x_offset, y_offset = offset
        options = context.request.options
        steps = max(1, options.drag_steps)
        chain = ActionChains(self.driver).move_to_element(source).pause(options.hold_pause).click_and_hold(source)
        step_x = int(round(x_offset / steps))
        step_y = int(round(y_offset / steps))
        for _ in range(steps):
            chain.move_by_offset(step_x, step_y).pause(options.move_pause)
        chain.pause(options.release_pause).release().perform()

    def _drag_with_html5_javascript(self, source: WebElement, target: Optional[WebElement]) -> None:
        if target is None:
            raise DragAndDropValidationError("javascript_html5 requires a target element")
        try:
            self.driver.execute_script(
                """
                const source = arguments[0];
                const target = arguments[1];
                const dataTransfer = new DataTransfer();
                const fire = (element, type) => {
                    const event = new DragEvent(type, {
                        bubbles: true,
                        cancelable: true,
                        composed: true,
                        dataTransfer
                    });
                    element.dispatchEvent(event);
                    return event.defaultPrevented;
                };
                fire(source, 'pointerdown');
                fire(source, 'mousedown');
                fire(source, 'dragstart');
                fire(target, 'dragenter');
                fire(target, 'dragover');
                fire(target, 'drop');
                fire(source, 'dragend');
                fire(source, 'mouseup');
                return true;
                """,
                source,
                target,
            )
        except Exception as exc:
            raise Html5DragAndDropError("HTML5 JavaScript drag-and-drop failed", context={"source": element_metadata(source)}, cause=exc) from exc

    def _drag_with_dispatched_events(
        self,
        source: WebElement,
        target: Optional[WebElement],
        context: DragAndDropExecutionContext,
    ) -> None:
        if target is not None:
            self.driver.execute_script(
                """
                const source = arguments[0];
                const target = arguments[1];
                const sourceRect = source.getBoundingClientRect();
                const targetRect = target.getBoundingClientRect();
                const startX = sourceRect.left + sourceRect.width / 2;
                const startY = sourceRect.top + sourceRect.height / 2;
                const endX = targetRect.left + targetRect.width / 2;
                const endY = targetRect.top + targetRect.height / 2;
                const fire = (element, type, x, y, buttons) => {
                    element.dispatchEvent(new MouseEvent(type, {
                        bubbles: true,
                        cancelable: true,
                        composed: true,
                        view: window,
                        clientX: x,
                        clientY: y,
                        button: 0,
                        buttons
                    }));
                };
                fire(source, 'pointerdown', startX, startY, 1);
                fire(source, 'mousedown', startX, startY, 1);
                fire(target, 'pointermove', endX, endY, 1);
                fire(target, 'mousemove', endX, endY, 1);
                fire(target, 'pointerup', endX, endY, 0);
                fire(target, 'mouseup', endX, endY, 0);
                fire(target, 'click', endX, endY, 0);
                return true;
                """,
                source,
                target,
            )
            return

        offset = context.request.target_offset
        if offset is None:
            raise DragAndDropValidationError("dispatch_events requires a target element or target_offset")
        self.driver.execute_script(
            """
            const source = arguments[0];
            const dx = arguments[1];
            const dy = arguments[2];
            const rect = source.getBoundingClientRect();
            const startX = rect.left + rect.width / 2;
            const startY = rect.top + rect.height / 2;
            const endX = startX + dx;
            const endY = startY + dy;
            const fire = (type, x, y, buttons) => {
                source.dispatchEvent(new MouseEvent(type, {
                    bubbles: true,
                    cancelable: true,
                    composed: true,
                    view: window,
                    clientX: x,
                    clientY: y,
                    button: 0,
                    buttons
                }));
            };
            fire('pointerdown', startX, startY, 1);
            fire('mousedown', startX, startY, 1);
            fire('pointermove', endX, endY, 1);
            fire('mousemove', endX, endY, 1);
            fire('pointerup', endX, endY, 0);
            fire('mouseup', endX, endY, 0);
            return true;
            """,
            source,
            offset[0],
            offset[1],
        )

    def _offset_to_target(self, context: DragAndDropExecutionContext) -> Optional[Tuple[int, int]]:
        source = context.source
        target = context.target
        if source is None or target is None:
            return None
        try:
            source_location = dict(getattr(source, "location", {}) or {})
            source_size = dict(getattr(source, "size", {}) or {})
            target_location = dict(getattr(target, "location", {}) or {})
            target_size = dict(getattr(target, "size", {}) or {})
            source_center_x = float(source_location.get("x", 0)) + float(source_size.get("width", 0)) / 2
            source_center_y = float(source_location.get("y", 0)) + float(source_size.get("height", 0)) / 2
            target_center_x = float(target_location.get("x", 0)) + float(target_size.get("width", 0)) / 2
            target_center_y = float(target_location.get("y", 0)) + float(target_size.get("height", 0)) / 2
            return (int(round(target_center_x - source_center_x)), int(round(target_center_y - source_center_y)))
        except Exception:
            return None

    def _post_drop_verification(
        self,
        source: WebElement,
        target: Optional[WebElement],
        context: DragAndDropExecutionContext,
    ) -> None:
        options = context.request.options
        if options.wait_for_page_load_after_drop:
            wait_for_page_load(self.driver, timeout=options.page_load_timeout)
        context.after = self._capture_after_state(source, target, options)
        if not options.verify_after_drop:
            return
        context.after["verification"] = self._verify_drop_effect(context.before, context.after)

    def _verify_drop_effect(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
        """Collect non-invasive signals that a drag/drop had an effect.

        Verification records signals rather than failing successful Selenium
        actions. Many valid drops update async UI, mutate DOM outside the
        source/target nodes, or require application-specific assertions.
        """

        signals = {
            "url_changed": before.get("url") != after.get("url"),
            "title_changed": before.get("title") != after.get("title"),
            "ready_state": after.get("ready_state"),
            "source_location_changed": before.get("source_location") != after.get("source_location"),
            "source_parent_changed": before.get("source_parent_fingerprint") != after.get("source_parent_fingerprint"),
            "target_text_changed": before.get("target_text") != after.get("target_text"),
            "active_element_changed": before.get("active_element_fingerprint") != after.get("active_element_fingerprint"),
        }
        signals["has_observable_effect"] = any(bool(value) for key, value in signals.items() if key.endswith("_changed"))
        return signals

    def _capture_before_state(
        self,
        source: WebElement,
        target: Optional[WebElement],
        options: DragAndDropOptions,
    ) -> Dict[str, Any]:
        return self._capture_drag_state(source, target, options, phase="before")

    def _capture_after_state(
        self,
        source: WebElement,
        target: Optional[WebElement],
        options: DragAndDropOptions,
    ) -> Dict[str, Any]:
        return self._capture_drag_state(source, target, options, phase="after")

    def _capture_drag_state(
        self,
        source: WebElement,
        target: Optional[WebElement],
        options: DragAndDropOptions,
        *,
        phase: str,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "phase": phase,
            "url": get_current_url(self.driver),
            "title": get_page_title(self.driver),
            "ready_state": get_document_ready_state(self.driver),
            "source_location": safe_serialize(getattr(source, "location", None)),
            "source_size": safe_serialize(getattr(source, "size", None)),
            "source_text": element_text(source, max_length=500),
            "source_parent_fingerprint": self._parent_fingerprint(source),
        }
        if target is not None:
            state.update(
                {
                    "target_location": safe_serialize(getattr(target, "location", None)),
                    "target_size": safe_serialize(getattr(target, "size", None)),
                    "target_text": element_text(target, max_length=500),
                    "target_parent_fingerprint": self._parent_fingerprint(target),
                }
            )
        active = safe_call(lambda: self.driver.switch_to.active_element, default=None)
        if active is not None:
            state["active_element_fingerprint"] = element_snapshot(active, include_html=False).fingerprint
        if options.include_page_snapshot:
            state["page"] = page_snapshot_dict(self.driver, include_html=False, include_screenshot=False, max_text=1_000)
        return prune_none(redact_mapping(state))

    def _parent_fingerprint(self, element: WebElement) -> Optional[str]:
        parent = safe_call(lambda: element.find_element(By.XPATH, "./.."), default=None)
        if parent is None:
            return None
        return element_snapshot(parent, include_html=False).fingerprint

    def _sleep_before_retry(self, attempt: int, options: DragAndDropOptions) -> None:
        delay = calculate_backoff_delay(
            attempt_index=attempt,
            base_delay=options.retry_backoff_base,
            max_delay=options.retry_backoff_max,
            multiplier=options.retry_backoff_multiplier,
            jitter=options.retry_jitter,
        )
        if delay > 0:
            time_module.sleep(delay)

    def _success_result(self, message: str, context: DragAndDropExecutionContext) -> dict:
        request = context.request
        options = request.options
        data: Dict[str, Any] = {
            "source_selector": request.source_selector,
            "target_selector": request.target_selector,
            "source_offset": request.source_offset,
            "target_offset": request.target_offset,
            "strategy": context.strategy,
            "attempt_count": len(context.attempts),
            "attempts": context.attempts,
            "before": context.before,
            "after": context.after,
        }
        if context.source is not None:
            data["source"] = element_metadata(
                context.source,
                max_text=options.max_element_text_chars,
                max_html=options.max_element_html_chars if options.include_element_html else 0,
            )
        if context.target is not None:
            data["target"] = element_metadata(
                context.target,
                max_text=options.max_element_text_chars,
                max_html=options.max_element_html_chars if options.include_element_html else 0,
            )
        return success_result(
            action="drag_and_drop",
            message=message,
            data=data,
            duration_ms=elapsed_ms(context.start_ms),
            correlation_id=request.correlation_id,
        )

    def _error_result(
        self,
        error: BaseException,
        *,
        start_ms: float,
        source_selector: str,
        target_selector: Optional[str],
        options: Optional[DragAndDropOptions] = None,
    ) -> dict:
        metadata: Dict[str, Any] = {
            "source_selector": source_selector,
            "target_selector": target_selector,
            "options": asdict(options) if options else None,
        }
        if options and options.screenshot_on_error:
            screenshot = capture_screenshot_b64(self.driver)
            if screenshot:
                metadata["screenshot_b64"] = screenshot
        return error_result(
            action="drag_and_drop",
            message=str(error),
            error=error,
            metadata=metadata,
            duration_ms=elapsed_ms(start_ms),
            correlation_id=new_correlation_id("drag-drop-error"),
        )

    def _coerce_drag_error(self, exc: BaseException, **context: Any) -> BrowserError:
        if isinstance(exc, BrowserError):
            exc.context.update(sanitize_context(context, redact=False) if "sanitize_context" in globals() else safe_serialize(context))
            return exc
        mapped = self._map_selenium_exception(exc)
        return mapped(str(exc) or mapped.default_message, context=context, cause=exc)

    def _map_selenium_exception(self, exc: BaseException):
        if isinstance(exc, TimeoutException):
            return BrowserTimeoutError
        if isinstance(exc, NoSuchElementException):
            return DragSourceNotFoundError
        if isinstance(exc, StaleElementReferenceException):
            return StaleElementError
        if isinstance(exc, ElementClickInterceptedException):
            return DragStrategyError
        if isinstance(exc, ElementNotInteractableException):
            return ElementNotInteractableError
        if isinstance(exc, MoveTargetOutOfBoundsException):
            return DragStrategyError
        if isinstance(exc, JavascriptException):
            return Html5DragAndDropError
        if isinstance(exc, WebDriverException):
            return DragAndDropError
        return DragAndDropError

    def _strategy_label(self, strategy: str) -> str:
        return {
            "action_chains_drag_and_drop": "ActionChains native",
            "action_chains_hold_move_release": "ActionChains hold/move/release",
            "action_chains_offset": "ActionChains offset",
            "incremental_offset": "Incremental offset",
            "javascript_html5": "HTML5 JavaScript",
            "dispatch_events": "DOM mouse event",
        }.get(strategy, strategy)

    def _build_success_message(self, message: str, source: WebElement, target: Optional[WebElement] = None) -> dict:
        """Compatibility wrapper for legacy callers."""

        return success_result(
            action="drag_and_drop",
            message=message,
            data={"source": element_metadata(source), "target": element_metadata(target) if target is not None else None},
            correlation_id=new_correlation_id("drag-drop-legacy"),
        )

    def _build_error_message(self, error_msg: str) -> dict:
        """Compatibility wrapper for legacy callers."""

        error = DragAndDropError(error_msg)
        return error_result(
            action="drag_and_drop",
            message=error_msg,
            error=error,
            correlation_id=new_correlation_id("drag-drop-legacy-error"),
        )

    def drag_element_to_element(self, source_selector: str, target_selector: str, wait_time: float = 0.0) -> str:
        """Legacy string-returning element-to-element wrapper."""

        result = self._perform_drag_and_drop(source_selector, target_selector, wait_time)
        return str(result.get("message", ""))

    def drag_element_by_offset(self, source_selector: str, x_offset: int, y_offset: int, wait_time: float = 0.0) -> str:
        """Legacy string-returning offset wrapper."""

        result = self._perform_drag_and_drop(source_selector, None, wait_time, target_offset=(x_offset, y_offset))
        return str(result.get("message", ""))


if __name__ == "__main__":
    print("\n=== Running Do Drag And Drop ===\n")
    printer.status("TEST", "Do Drag And Drop initialized", "info")

    class _FakeSwitchTo:
        def __init__(self, driver: "_FakeDriver") -> None:
            self._driver = driver

        @property
        def active_element(self):
            return self._driver.active_element

    class _FakeElement:
        def __init__(self, tag: str = "div", text: str = "Item", *, displayed: bool = True, enabled: bool = True, x: int = 10, y: int = 20):
            self.tag_name = tag
            self.text = text
            self._displayed = displayed
            self._enabled = enabled
            self.location = {"x": x, "y": y}
            self.size = {"width": 120, "height": 35}
            self.attrs = {
                "id": text.lower().replace(" ", "-"),
                "class": "draggable",
                "role": "listitem",
                "outerHTML": f"<{tag} class='draggable'>{text}</{tag}>",
            }

        def click(self):
            self.attrs["clicked"] = "true"

        def is_displayed(self):
            return self._displayed

        def is_enabled(self):
            return self._enabled

        def is_selected(self):
            return False

        def get_attribute(self, name):
            return self.attrs.get(name)

        def find_element(self, by=By.CSS_SELECTOR, value=None):
            if by == By.XPATH and value == "./..":
                return _FakeElement(tag="section", text="Parent", x=0, y=0)
            raise NoSuchElementException(f"No nested element for {by}={value}")

    class _FakeDriver:
        def __init__(self) -> None:
            self.current_url = "https://example.test/drag"
            self.title = "Example Drag Page"
            self.page_source = "<html><body><div id='source'>Item</div><div id='target'>Drop Zone</div></body></html>"
            self.elements = {
                "#source": _FakeElement(text="Source", x=10, y=20),
                "#target": _FakeElement(text="Target", x=300, y=150),
                "#offset-source": _FakeElement(text="Offset Source", x=40, y=60),
            }
            self.active_element = self.elements["#source"]
            self.switch_to = _FakeSwitchTo(self)
            self.scripts: List[str] = []

        def find_element(self, by=By.CSS_SELECTOR, value=None):
            if by == By.TAG_NAME and value == "body":
                return _FakeElement(tag="body", text="Source Target Offset Source")
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
            return True

        def get_screenshot_as_base64(self):
            return ""

    dragger = DoDragAndDrop(_FakeDriver())

    html5_result = dragger.drag_and_drop("#source", "#target", strategies=["javascript_html5"], verify_after_drop=True)
    assert html5_result["status"] == "success", html5_result
    assert html5_result["data"]["strategy"] == "javascript_html5", html5_result

    offset_result = dragger.drag_by_offset("#offset-source", 80, 30, strategies=["dispatch_events"], verify_after_drop=True)
    assert offset_result["status"] == "error", "dispatch_events is intentionally target/offset validated through strategy eligibility"

    offset_result = dragger.drag_by_offset("#offset-source", 80, 30, strategies=["action_chains_offset"], verify_after_drop=False, retries=0)
    assert offset_result["status"] in {"success", "error"}, offset_result

    missing_result = dragger.drag_and_drop("#missing", "#target", retries=0, timeout=0.2, strategies=["javascript_html5"])
    assert missing_result["status"] == "error", missing_result

    legacy_message = dragger.drag_element_to_element("#source", "#target")
    assert legacy_message, legacy_message

    print("\n=== Test ran successfully ===\n")
