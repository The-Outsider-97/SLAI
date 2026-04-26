from __future__ import annotations

"""
Production-grade scrolling utilities for the browser subsystem.

This module owns concrete browser scrolling behavior for Selenium-backed browser
agents. It intentionally stays focused on scroll execution and leaves generic
concerns—serialization, redaction, result payloads, selector validation, timing,
and browser-domain exceptions—to the shared browser helpers and error modules.

Design goals
------------
- Preserve the existing DoScroll public API used by BrowserAgent:
  ``scroll_to``, ``scroll_by``, ``scroll_direction``, and
  ``scroll_element_into_view``.
- Provide richer, configurable scrolling primitives without duplicating shared
  helper or error logic.
- Support future workflow expansion through a normalized ``perform`` entrypoint
  and async ``do_scroll`` wrapper.
- Return stable BrowserAgent-compatible dictionaries while carrying structured
  diagnostics for debugging, telemetry, and memory/learning modules.
- Keep all runtime behavior configurable from ``browser_config.yaml`` via the
  ``do_scroll`` section.
"""

import asyncio
import time as time_module

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from selenium.common.exceptions import (
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

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.browser_errors import *
from ..utils.Browser_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Scroll")
printer = PrettyPrinter


VALID_DIRECTIONS = {"up", "down", "left", "right"}
VALID_ELEMENT_POSITIONS = {"start", "center", "end", "nearest"}
VALID_SCROLL_MODES = {
    "to",
    "by",
    "direction",
    "element",
    "top",
    "bottom",
    "percentage",
    "page",
    "until_end",
    "state",
}


@dataclass(frozen=True)
class ScrollOptions:
    """Config-backed runtime options for scroll operations."""

    default_timeout: float = 5.0
    poll_frequency: float = 0.1
    default_amount: int = 600
    max_amount: int = 10_000
    default_behavior: str = "auto"
    smooth_default: bool = False
    default_element_position: str = "center"
    scroll_margin_x: int = 0
    scroll_margin_y: int = 0
    clamp_coordinates: bool = False
    allow_negative_coordinates: bool = True
    require_scroll_change: bool = False
    wait_for_settle: bool = True
    settle_timeout: float = 1.5
    settle_poll_frequency: float = 0.05
    settle_stable_samples: int = 2
    max_until_end_steps: int = 20
    until_end_pause_seconds: float = 0.15
    max_retries: int = 1
    retry_base_delay: float = 0.15
    retry_max_delay: float = 1.5
    retry_multiplier: float = 2.0
    retry_jitter: float = 0.05
    include_page_state: bool = True
    include_element_metadata: bool = True
    include_attempt_history: bool = True
    include_screenshot_on_error: bool = False
    include_page_snapshot_on_error: bool = False
    max_diagnostic_text_length: int = 1_000

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "ScrollOptions":
        cfg = dict(config or {})
        diagnostics = dict(cfg.get("diagnostics") or {})
        retry_cfg = dict(cfg.get("retry") or {})
        settle_cfg = dict(cfg.get("settle") or {})
        limits = dict(cfg.get("limits") or {})
        element_cfg = dict(cfg.get("element") or {})

        position = str(element_cfg.get("default_position", cfg.get("default_element_position", "center"))).lower().strip()
        if position not in VALID_ELEMENT_POSITIONS:
            position = "center"

        behavior = str(cfg.get("default_behavior", "smooth" if coerce_bool(cfg.get("smooth_default"), default=False) else "auto")).lower().strip()
        if behavior not in {"auto", "smooth", "instant"}:
            behavior = "auto"

        return cls(
            default_timeout=coerce_float(cfg.get("default_timeout", cfg.get("timeout", 5.0)), default=5.0, minimum=0.0, maximum=300.0),
            poll_frequency=coerce_float(cfg.get("poll_frequency", 0.1), default=0.1, minimum=0.01, maximum=5.0),
            default_amount=coerce_int(cfg.get("default_amount", 600), default=600, minimum=1),
            max_amount=coerce_int(limits.get("max_amount", cfg.get("max_amount", 10_000)), default=10_000, minimum=1),
            default_behavior=behavior,
            smooth_default=coerce_bool(cfg.get("smooth_default", False), default=False),
            default_element_position=position,
            scroll_margin_x=coerce_int(cfg.get("scroll_margin_x", 0), default=0),
            scroll_margin_y=coerce_int(cfg.get("scroll_margin_y", 0), default=0),
            clamp_coordinates=coerce_bool(limits.get("clamp_coordinates", cfg.get("clamp_coordinates", False)), default=False),
            allow_negative_coordinates=coerce_bool(limits.get("allow_negative_coordinates", cfg.get("allow_negative_coordinates", True)), default=True),
            require_scroll_change=coerce_bool(cfg.get("require_scroll_change", False), default=False),
            wait_for_settle=coerce_bool(settle_cfg.get("enabled", cfg.get("wait_for_settle", True)), default=True),
            settle_timeout=coerce_float(settle_cfg.get("timeout", cfg.get("settle_timeout", 1.5)), default=1.5, minimum=0.0, maximum=30.0),
            settle_poll_frequency=coerce_float(settle_cfg.get("poll_frequency", cfg.get("settle_poll_frequency", 0.05)), default=0.05, minimum=0.01, maximum=2.0),
            settle_stable_samples=coerce_int(settle_cfg.get("stable_samples", cfg.get("settle_stable_samples", 2)), default=2, minimum=1, maximum=20),
            max_until_end_steps=coerce_int(limits.get("max_until_end_steps", cfg.get("max_until_end_steps", 20)), default=20, minimum=1, maximum=500),
            until_end_pause_seconds=coerce_float(cfg.get("until_end_pause_seconds", 0.15), default=0.15, minimum=0.0, maximum=10.0),
            max_retries=coerce_int(retry_cfg.get("max_retries", cfg.get("max_retries", 1)), default=1, minimum=0, maximum=20),
            retry_base_delay=coerce_float(retry_cfg.get("base_delay", cfg.get("retry_base_delay", 0.15)), default=0.15, minimum=0.0, maximum=60.0),
            retry_max_delay=coerce_float(retry_cfg.get("max_delay", cfg.get("retry_max_delay", 1.5)), default=1.5, minimum=0.0, maximum=60.0),
            retry_multiplier=coerce_float(retry_cfg.get("multiplier", cfg.get("retry_multiplier", 2.0)), default=2.0, minimum=1.0, maximum=10.0),
            retry_jitter=coerce_float(retry_cfg.get("jitter", cfg.get("retry_jitter", 0.05)), default=0.05, minimum=0.0, maximum=10.0),
            include_page_state=coerce_bool(diagnostics.get("include_page_state", cfg.get("include_page_state", True)), default=True),
            include_element_metadata=coerce_bool(diagnostics.get("include_element_metadata", cfg.get("include_element_metadata", True)), default=True),
            include_attempt_history=coerce_bool(diagnostics.get("include_attempt_history", cfg.get("include_attempt_history", True)), default=True),
            include_screenshot_on_error=coerce_bool(diagnostics.get("include_screenshot_on_error", cfg.get("include_screenshot_on_error", False)), default=False),
            include_page_snapshot_on_error=coerce_bool(diagnostics.get("include_page_snapshot_on_error", cfg.get("include_page_snapshot_on_error", False)), default=False),
            max_diagnostic_text_length=coerce_int(diagnostics.get("max_text_length", cfg.get("max_diagnostic_text_length", 1_000)), default=1_000, minimum=100),
        )


@dataclass(frozen=True)
class ScrollRequest:
    """Normalized scroll request used internally by the executor."""

    mode: str
    x: Optional[int] = None
    y: Optional[int] = None
    dx: Optional[int] = None
    dy: Optional[int] = None
    direction: Optional[str] = None
    amount: Optional[int] = None
    selector: Optional[str] = None
    position: str = "center"
    percentage: Optional[float] = None
    smooth: Optional[bool] = None
    timeout: Optional[float] = None
    correlation_id: str = field(default_factory=lambda: new_correlation_id("scroll"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self))


@dataclass(frozen=True)
class ScrollState:
    """Serializable snapshot of the current window scroll state."""

    x: int = 0
    y: int = 0
    max_x: int = 0
    max_y: int = 0
    viewport_width: int = 0
    viewport_height: int = 0
    document_width: int = 0
    document_height: int = 0
    ready_state: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self))


@dataclass
class ScrollExecutionContext:
    """Mutable per-operation telemetry container."""

    request: ScrollRequest
    options: ScrollOptions
    started_ms: float = field(default_factory=monotonic_ms)
    before_state: Optional[ScrollState] = None
    after_state: Optional[ScrollState] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    element_metadata: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> float:
        return elapsed_ms(self.started_ms)

    def to_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "correlation_id": self.request.correlation_id,
            "request": self.request.to_dict(),
            "duration_ms": self.duration_ms,
        }
        if self.options.include_page_state:
            metadata["before_state"] = self.before_state.to_dict() if self.before_state else None
            metadata["after_state"] = self.after_state.to_dict() if self.after_state else None
        if self.options.include_attempt_history:
            metadata["attempts"] = self.attempts
        if self.options.include_element_metadata and self.element_metadata:
            metadata["element"] = self.element_metadata
        return prune_none(metadata)


class DoScroll:
    """Concrete scrolling executor for Selenium-backed browser modules.

    Existing BrowserAgent integration calls the legacy methods directly. Newer
    workflow/task integrations can use ``perform`` or async ``do_scroll`` to
    dispatch richer scroll modes through one normalized path.
    """

    def __init__(self, driver):
        self.config = load_global_config()
        self.scroll_config = get_config_section("do_scroll") or {}
        self.driver = driver
        self.options = ScrollOptions.from_config(self.scroll_config)
        logger.info("Browser scrolling functionality initiated.")

    # ------------------------------------------------------------------
    # Public / BrowserAgent-compatible methods
    # ------------------------------------------------------------------
    async def do_scroll(self, mode: str = "by", **kwargs: Any) -> dict:
        """Async wrapper around ``perform`` for agent schedulers."""

        return await asyncio.to_thread(self.perform, mode, **kwargs)

    def perform(self, mode: str = "by", **kwargs: Any) -> dict:
        """Dispatch a normalized scroll action by mode."""

        try:
            mode_normalized = str(mode or "by").lower().strip()
            if mode_normalized not in VALID_SCROLL_MODES:
                raise InvalidScrollTargetError(
                    f"Unsupported scroll mode: {mode}",
                    context={"mode": mode, "supported_modes": sorted(VALID_SCROLL_MODES)},
                )

            if mode_normalized == "to":
                return self.scroll_to(
                    x=kwargs.get("x", 0),
                    y=kwargs.get("y", 0),
                    smooth=kwargs.get("smooth", self.options.smooth_default),
                    timeout=kwargs.get("timeout"),
                )
            if mode_normalized == "by":
                return self.scroll_by(
                    dx=kwargs.get("dx", 0),
                    dy=kwargs.get("dy", kwargs.get("amount", self.options.default_amount)),
                    smooth=kwargs.get("smooth", self.options.smooth_default),
                    timeout=kwargs.get("timeout"),
                )
            if mode_normalized == "direction":
                return self.scroll_direction(
                    direction=kwargs.get("direction", "down"),
                    amount=kwargs.get("amount", self.options.default_amount),
                    smooth=kwargs.get("smooth", self.options.smooth_default),
                    timeout=kwargs.get("timeout"),
                )
            if mode_normalized == "element":
                return self.scroll_element_into_view(
                    selector=kwargs.get("selector", ""),
                    position=kwargs.get("position", self.options.default_element_position),
                    smooth=kwargs.get("smooth", self.options.smooth_default),
                    timeout=kwargs.get("timeout"),
                )
            if mode_normalized == "top":
                return self.scroll_to_top(smooth=kwargs.get("smooth", self.options.smooth_default), timeout=kwargs.get("timeout"))
            if mode_normalized == "bottom":
                return self.scroll_to_bottom(smooth=kwargs.get("smooth", self.options.smooth_default), timeout=kwargs.get("timeout"))
            if mode_normalized == "percentage":
                return self.scroll_to_percentage(
                    percentage=kwargs.get("percentage", 100),
                    smooth=kwargs.get("smooth", self.options.smooth_default),
                    timeout=kwargs.get("timeout"),
                )
            if mode_normalized == "page":
                return self.scroll_page(
                    pages=kwargs.get("pages", 1),
                    axis=kwargs.get("axis", "vertical"),
                    smooth=kwargs.get("smooth", self.options.smooth_default),
                    timeout=kwargs.get("timeout"),
                )
            if mode_normalized == "until_end":
                return self.scroll_until_end(
                    direction=kwargs.get("direction", "down"),
                    amount=kwargs.get("amount", self.options.default_amount),
                    max_steps=kwargs.get("max_steps", self.options.max_until_end_steps),
                    smooth=kwargs.get("smooth", self.options.smooth_default),
                    timeout=kwargs.get("timeout"),
                )
            if mode_normalized == "state":
                state = self.get_scroll_state()
                return success_result(action="scroll", message="Captured scroll state", data={"state": state})

            raise InvalidScrollTargetError(f"Unhandled scroll mode: {mode_normalized}", context={"mode": mode_normalized})
        except Exception as exc:
            return self._error_to_result(exc, action="scroll", context={"mode": mode, "kwargs": kwargs})

    def scroll_to(self, x: int, y: int, smooth: bool = False, timeout: Optional[float] = None) -> dict:
        """Scroll the window to an absolute x/y coordinate."""

        request = ScrollRequest(mode="to", x=int(x), y=int(y), smooth=smooth, timeout=timeout)
        return self._execute_scroll(request)

    def scroll_element_into_view(
        self,
        selector: str,
        position: str = "center",
        smooth: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Scroll a CSS-selected element into view."""

        position_normalized = str(position or self.options.default_element_position).lower().strip()
        if position_normalized not in VALID_ELEMENT_POSITIONS:
            position_normalized = self.options.default_element_position
        request = ScrollRequest(
            mode="element",
            selector=selector,
            position=position_normalized,
            smooth=self.options.smooth_default if smooth is None else smooth,
            timeout=timeout,
        )
        return self._execute_scroll(request)

    def scroll_by(self, dx: int, dy: int, smooth: bool = False, timeout: Optional[float] = None) -> dict:
        """Scroll the window by a relative x/y delta."""

        request = ScrollRequest(mode="by", dx=int(dx), dy=int(dy), smooth=smooth, timeout=timeout)
        return self._execute_scroll(request)

    def scroll_direction(
        self,
        direction: str,
        amount: int = 200,
        smooth: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Scroll in one of the canonical directions: up/down/left/right."""

        direction_normalized = str(direction or "").lower().strip()
        if direction_normalized not in VALID_DIRECTIONS:
            return self._error_to_result(
                InvalidScrollTargetError(
                    f"Invalid direction: {direction}",
                    context={"direction": direction, "valid_directions": sorted(VALID_DIRECTIONS)},
                ),
                action="scroll",
                context={"direction": direction, "amount": amount},
            )
        bounded_amount = self._bounded_amount(amount)
        dx, dy = self._direction_to_delta(direction_normalized, bounded_amount)
        request = ScrollRequest(
            mode="direction",
            direction=direction_normalized,
            amount=bounded_amount,
            dx=dx,
            dy=dy,
            smooth=self.options.smooth_default if smooth is None else smooth,
            timeout=timeout,
        )
        return self._execute_scroll(request)

    # ------------------------------------------------------------------
    # Expanded public convenience methods
    # ------------------------------------------------------------------
    def scroll_to_top(self, smooth: Optional[bool] = None, timeout: Optional[float] = None) -> dict:
        request = ScrollRequest(mode="top", x=0, y=0, smooth=self.options.smooth_default if smooth is None else smooth, timeout=timeout)
        return self._execute_scroll(request)

    def scroll_to_bottom(self, smooth: Optional[bool] = None, timeout: Optional[float] = None) -> dict:
        request = ScrollRequest(mode="bottom", smooth=self.options.smooth_default if smooth is None else smooth, timeout=timeout)
        return self._execute_scroll(request)

    def scroll_to_percentage(self, percentage: float, smooth: Optional[bool] = None, timeout: Optional[float] = None) -> dict:
        pct = max(0.0, min(100.0, coerce_float(percentage, default=100.0, minimum=0.0, maximum=100.0)))
        request = ScrollRequest(mode="percentage", percentage=pct, smooth=self.options.smooth_default if smooth is None else smooth, timeout=timeout)
        return self._execute_scroll(request)

    def scroll_page(self, pages: float = 1.0, axis: str = "vertical", smooth: Optional[bool] = None, timeout: Optional[float] = None) -> dict:
        state = self._capture_scroll_state()
        page_count = coerce_float(pages, default=1.0, minimum=-100.0, maximum=100.0)
        axis_normalized = str(axis or "vertical").lower().strip()
        if axis_normalized in {"horizontal", "x"}:
            dx = int(state.viewport_width * page_count)
            dy = 0
        else:
            dx = 0
            dy = int(state.viewport_height * page_count)
        request = ScrollRequest(mode="page", dx=dx, dy=dy, smooth=self.options.smooth_default if smooth is None else smooth, timeout=timeout, metadata={"pages": page_count, "axis": axis_normalized})
        return self._execute_scroll(request)

    def scroll_until_end(
        self,
        direction: str = "down",
        amount: Optional[int] = None,
        max_steps: Optional[int] = None,
        smooth: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Repeatedly scroll until the edge is reached or max_steps is exceeded."""

        direction_normalized = str(direction or "down").lower().strip()
        if direction_normalized not in VALID_DIRECTIONS:
            return self._error_to_result(
                InvalidScrollTargetError("Invalid direction for scroll_until_end", context={"direction": direction}),
                action="scroll",
            )

        request = ScrollRequest(
            mode="until_end",
            direction=direction_normalized,
            amount=self._bounded_amount(amount if amount is not None else self.options.default_amount),
            smooth=self.options.smooth_default if smooth is None else smooth,
            timeout=timeout,
        )
        context = ScrollExecutionContext(request=request, options=self.options)
        context.before_state = self._capture_scroll_state()
        max_steps_value = coerce_int(max_steps if max_steps is not None else self.options.max_until_end_steps, default=self.options.max_until_end_steps, minimum=1)

        try:
            previous = context.before_state
            steps_executed = 0
            for step in range(max_steps_value):
                dx, dy = self._direction_to_delta(direction_normalized, request.amount or self.options.default_amount)
                step_request = ScrollRequest(
                    mode="by",
                    dx=dx,
                    dy=dy,
                    smooth=request.smooth,
                    timeout=request.timeout,
                    correlation_id=request.correlation_id,
                    metadata={"until_end_step": step + 1},
                )
                self._run_scroll_script(step_request)
                if self.options.until_end_pause_seconds > 0:
                    time_module.sleep(self.options.until_end_pause_seconds)
                current = self._capture_scroll_state()
                steps_executed += 1
                context.attempts.append({
                    "step": steps_executed,
                    "before": previous.to_dict() if previous else None,
                    "after": current.to_dict(),
                    "changed": self._state_changed(previous, current),
                })
                if not self._state_changed(previous, current):
                    break
                if self._is_at_directional_edge(current, direction_normalized):
                    break
                previous = current

            if self.options.wait_for_settle:
                self._wait_for_scroll_settle(self.options.settle_timeout)
            context.after_state = self._capture_scroll_state()
            return self._success(
                context,
                message=f"Scrolled {direction_normalized} until edge or limit",
                data={"steps": steps_executed, "direction": direction_normalized, "state": context.after_state.to_dict()},
            )
        except Exception as exc:
            return self._failure(context, exc, message="Scroll until end failed")

    def get_scroll_state(self) -> dict:
        """Return the current page scroll state as a dictionary."""

        return self._capture_scroll_state().to_dict()

    # Backwards-compatible aliases / wrappers.
    def scroll_down(self, amount: Optional[int] = None, smooth: Optional[bool] = None) -> dict:
        return self.scroll_direction("down", amount or self.options.default_amount, smooth=smooth)

    def scroll_up(self, amount: Optional[int] = None, smooth: Optional[bool] = None) -> dict:
        return self.scroll_direction("up", amount or self.options.default_amount, smooth=smooth)

    def scroll_left(self, amount: Optional[int] = None, smooth: Optional[bool] = None) -> dict:
        return self.scroll_direction("left", amount or self.options.default_amount, smooth=smooth)

    def scroll_right(self, amount: Optional[int] = None, smooth: Optional[bool] = None) -> dict:
        return self.scroll_direction("right", amount or self.options.default_amount, smooth=smooth)

    # ------------------------------------------------------------------
    # Execution internals
    # ------------------------------------------------------------------
    def _execute_scroll(self, request: ScrollRequest) -> dict:
        context = ScrollExecutionContext(request=request, options=self.options)
        context.before_state = self._capture_scroll_state()

        last_error: Optional[BaseException] = None
        attempts = max(1, self.options.max_retries + 1)

        for attempt_index in range(attempts):
            try:
                attempt_started = monotonic_ms()
                if request.mode == "element":
                    element = self._wait_for_element(request.selector or "", request.timeout)
                    context.element_metadata = self._element_metadata(element)
                    self._scroll_element_into_view(element, request)
                else:
                    self._run_scroll_script(request)

                if self.options.wait_for_settle:
                    self._wait_for_scroll_settle(request.timeout if request.timeout is not None else self.options.settle_timeout)

                context.after_state = self._capture_scroll_state()
                changed = self._state_changed(context.before_state, context.after_state)
                context.attempts.append({
                    "attempt": attempt_index + 1,
                    "status": "success",
                    "duration_ms": elapsed_ms(attempt_started),
                    "changed": changed,
                })

                if self.options.require_scroll_change and not changed and request.mode not in {"state"}:
                    raise InvalidScrollTargetError(
                        "Scroll operation completed but scroll position did not change",
                        context={"request": request.to_dict(), "before": context.before_state.to_dict() if context.before_state else None, "after": context.after_state.to_dict()},
                    )

                return self._success(
                    context,
                    message=self._success_message(request),
                    data={"state": context.after_state.to_dict() if context.after_state else None},
                )
            except (TimeoutException, StaleElementReferenceException, JavascriptException, WebDriverException, ScrollError, ElementError, BrowserError) as exc:
                last_error = exc
                context.attempts.append({
                    "attempt": attempt_index + 1,
                    "status": "error",
                    "duration_ms": elapsed_ms(context.started_ms),
                    "error": safe_serialize(exc),
                })
                if attempt_index >= attempts - 1 or not self._should_retry(exc):
                    break
                self._sleep_before_retry(attempt_index)
            except Exception as exc:
                last_error = exc
                context.attempts.append({
                    "attempt": attempt_index + 1,
                    "status": "error",
                    "duration_ms": elapsed_ms(context.started_ms),
                    "error": safe_serialize(exc),
                })
                break

        return self._failure(context, last_error or ScrollError("Unknown scroll failure"), message="Scroll failed")

    def _run_scroll_script(self, request: ScrollRequest) -> None:
        behavior = self._behavior(request.smooth)

        if request.mode in {"to", "top"}:
            x = int(request.x or 0)
            y = int(request.y or 0)
            x, y = self._normalize_target_coordinates(x, y)
            self.driver.execute_script(
                "window.scrollTo({left: arguments[0], top: arguments[1], behavior: arguments[2]});",
                x,
                y,
                behavior,
            )
            return

        if request.mode in {"by", "direction", "page"}:
            dx = int(request.dx or 0)
            dy = int(request.dy or 0)
            dx = self._bounded_delta(dx)
            dy = self._bounded_delta(dy)
            self.driver.execute_script(
                "window.scrollBy({left: arguments[0], top: arguments[1], behavior: arguments[2]});",
                dx,
                dy,
                behavior,
            )
            return

        if request.mode == "bottom":
            self.driver.execute_script(
                "window.scrollTo({left: window.scrollX || window.pageXOffset || 0, top: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight), behavior: arguments[0]});",
                behavior,
            )
            return

        if request.mode == "percentage":
            pct = max(0.0, min(100.0, float(request.percentage if request.percentage is not None else 100.0)))
            self.driver.execute_script(
                """
                const pct = arguments[0] / 100.0;
                const behavior = arguments[1];
                const doc = document.documentElement;
                const body = document.body || doc;
                const maxY = Math.max(0, Math.max(body.scrollHeight, doc.scrollHeight) - window.innerHeight);
                window.scrollTo({left: window.scrollX || window.pageXOffset || 0, top: Math.round(maxY * pct), behavior});
                """,
                pct,
                behavior,
            )
            return

        raise InvalidScrollTargetError("Unsupported scroll request mode", context={"request": request.to_dict()})

    def _scroll_element_into_view(self, element: WebElement, request: ScrollRequest) -> None:
        position = request.position if request.position in VALID_ELEMENT_POSITIONS else self.options.default_element_position
        behavior = self._behavior(request.smooth)
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block: arguments[1], inline: 'nearest', behavior: arguments[2]});",
            element,
            position,
            behavior,
        )
        if self.options.scroll_margin_x or self.options.scroll_margin_y:
            self.driver.execute_script(
                "window.scrollBy({left: arguments[0], top: arguments[1], behavior: arguments[2]});",
                self.options.scroll_margin_x,
                self.options.scroll_margin_y,
                behavior,
            )

    def _wait_for_element(self, selector: str, timeout: Optional[float] = None) -> WebElement:
        selector_text = validate_selector(selector, field_name="selector")
        wait_timeout = self._timeout(timeout)
        try:
            return WebDriverWait(self.driver, wait_timeout, poll_frequency=self.options.poll_frequency).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector_text))
            )
        except TimeoutException as exc:
            raise ElementNotFoundError(
                f"Element not found for scrolling: {selector_text}",
                context={"selector": selector_text, "timeout": wait_timeout},
                cause=exc,
            ) from exc
        except NoSuchElementException as exc:
            raise ElementNotFoundError(
                f"Element not found for scrolling: {selector_text}",
                context={"selector": selector_text, "timeout": wait_timeout},
                cause=exc,
            ) from exc

    def _capture_scroll_state(self) -> ScrollState:
        try:
            raw = self.driver.execute_script(
                """
                const doc = document.documentElement || {};
                const body = document.body || doc;
                const x = Math.round(window.scrollX || window.pageXOffset || doc.scrollLeft || body.scrollLeft || 0);
                const y = Math.round(window.scrollY || window.pageYOffset || doc.scrollTop || body.scrollTop || 0);
                const viewportWidth = Math.round(window.innerWidth || doc.clientWidth || 0);
                const viewportHeight = Math.round(window.innerHeight || doc.clientHeight || 0);
                const documentWidth = Math.round(Math.max(body.scrollWidth || 0, doc.scrollWidth || 0, body.offsetWidth || 0, doc.offsetWidth || 0));
                const documentHeight = Math.round(Math.max(body.scrollHeight || 0, doc.scrollHeight || 0, body.offsetHeight || 0, doc.offsetHeight || 0));
                return {
                    x,
                    y,
                    max_x: Math.max(0, documentWidth - viewportWidth),
                    max_y: Math.max(0, documentHeight - viewportHeight),
                    viewport_width: viewportWidth,
                    viewport_height: viewportHeight,
                    document_width: documentWidth,
                    document_height: documentHeight,
                    ready_state: document.readyState || null
                };
                """
            ) or {}
        except Exception:
            raw = {}

        return ScrollState(
            x=coerce_int(raw.get("x"), default=0),
            y=coerce_int(raw.get("y"), default=0),
            max_x=coerce_int(raw.get("max_x"), default=0),
            max_y=coerce_int(raw.get("max_y"), default=0),
            viewport_width=coerce_int(raw.get("viewport_width"), default=0),
            viewport_height=coerce_int(raw.get("viewport_height"), default=0),
            document_width=coerce_int(raw.get("document_width"), default=0),
            document_height=coerce_int(raw.get("document_height"), default=0),
            ready_state=raw.get("ready_state"),
            url=safe_call(lambda: self.driver.current_url, default=None),
            title=safe_call(lambda: self.driver.title, default=None),
        )

    def _wait_for_scroll_settle(self, timeout: Optional[float]) -> bool:
        timeout_value = self._timeout(timeout, fallback=self.options.settle_timeout)
        if timeout_value <= 0:
            return True

        end_time = time_module.monotonic() + timeout_value
        stable_samples = 0
        previous = self._capture_scroll_state()
        while time_module.monotonic() < end_time:
            time_module.sleep(self.options.settle_poll_frequency)
            current = self._capture_scroll_state()
            if current.x == previous.x and current.y == previous.y:
                stable_samples += 1
                if stable_samples >= self.options.settle_stable_samples:
                    return True
            else:
                stable_samples = 0
                previous = current
        return False

    # ------------------------------------------------------------------
    # Small scroll-specific helpers
    # ------------------------------------------------------------------
    def _direction_to_delta(self, direction: str, amount: int) -> Tuple[int, int]:
        mapping = {
            "up": (0, -amount),
            "down": (0, amount),
            "left": (-amount, 0),
            "right": (amount, 0),
        }
        return mapping[direction]

    def _bounded_amount(self, amount: Any) -> int:
        value = coerce_int(amount, default=self.options.default_amount)
        if value < 0:
            value = abs(value)
        return min(value, self.options.max_amount)

    def _bounded_delta(self, value: int) -> int:
        if value > self.options.max_amount:
            return self.options.max_amount
        if value < -self.options.max_amount:
            return -self.options.max_amount
        return value

    def _normalize_target_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        if not self.options.allow_negative_coordinates:
            x = max(0, x)
            y = max(0, y)
        if self.options.clamp_coordinates:
            state = self._capture_scroll_state()
            x = max(0, min(x, state.max_x))
            y = max(0, min(y, state.max_y))
        return x, y

    def _behavior(self, smooth: Optional[bool]) -> str:
        if smooth is None:
            return self.options.default_behavior
        return "smooth" if bool(smooth) else "auto"

    def _timeout(self, value: Optional[float], *, fallback: Optional[float] = None) -> float:
        return coerce_float(
            self.options.default_timeout if value is None and fallback is None else (fallback if value is None else value),
            default=self.options.default_timeout,
            minimum=0.0,
            maximum=300.0,
        )

    def _state_changed(self, before: Optional[ScrollState], after: Optional[ScrollState]) -> bool:
        if before is None or after is None:
            return False
        return before.x != after.x or before.y != after.y

    def _is_at_directional_edge(self, state: ScrollState, direction: str) -> bool:
        if direction == "up":
            return state.y <= 0
        if direction == "down":
            return state.y >= state.max_y
        if direction == "left":
            return state.x <= 0
        if direction == "right":
            return state.x >= state.max_x
        return False

    def _should_retry(self, exc: BaseException) -> bool:
        if isinstance(exc, InvalidScrollTargetError):
            return False
        if isinstance(exc, ElementNotFoundError):
            return False
        if isinstance(exc, BrowserError):
            return bool(getattr(exc, "retryable", False))
        return isinstance(exc, (TimeoutException, StaleElementReferenceException, JavascriptException, WebDriverException))

    def _sleep_before_retry(self, attempt_index: int) -> None:
        delay = calculate_backoff_delay(
            attempt_index=attempt_index,
            base_delay=self.options.retry_base_delay,
            max_delay=self.options.retry_max_delay,
            multiplier=self.options.retry_multiplier,
            jitter=self.options.retry_jitter,
        )
        if delay > 0:
            time_module.sleep(delay)

    def _element_metadata(self, element: WebElement) -> Dict[str, Any]:
        try:
            snapshot = element_snapshot(element, include_html=True, max_text=300, max_html=self.options.max_diagnostic_text_length)
            return snapshot.to_dict() if hasattr(snapshot, "to_dict") else safe_serialize(snapshot)
        except Exception:
            return safe_serialize(element)

    def _success_message(self, request: ScrollRequest) -> str:
        if request.mode == "to":
            return f"Scrolled to ({request.x},{request.y})"
        if request.mode == "by":
            return f"Scrolled by ({request.dx},{request.dy})"
        if request.mode == "direction":
            return f"Scrolled {request.direction} by {request.amount}"
        if request.mode == "element":
            return f"Scrolled to element {request.selector} ({request.position})"
        if request.mode == "top":
            return "Scrolled to top"
        if request.mode == "bottom":
            return "Scrolled to bottom"
        if request.mode == "percentage":
            return f"Scrolled to {request.percentage}%"
        if request.mode == "page":
            return "Scrolled by page"
        return "Scrolled successfully"

    # ------------------------------------------------------------------
    # Result builders / compatibility methods
    # ------------------------------------------------------------------
    def _success(self, context: ScrollExecutionContext, *, message: str, data: Optional[Mapping[str, Any]] = None) -> dict:
        result = success_result(
            action="scroll",
            message=message,
            data=dict(data or {}),
            metadata=context.to_metadata(),
            duration_ms=context.duration_ms,
            correlation_id=context.request.correlation_id,
        )
        # Legacy convenience fields used by some older callers.
        if context.after_state:
            result.setdefault("state", context.after_state.to_dict())
        if context.element_metadata:
            result.setdefault("element", context.element_metadata)
        return result

    def _failure(self, context: ScrollExecutionContext, exc: BaseException, *, message: str = "Scroll failed") -> dict:
        context.after_state = context.after_state or self._capture_scroll_state()
        metadata = context.to_metadata()
        if self.options.include_screenshot_on_error:
            screenshot = safe_call(lambda: self.driver.get_screenshot_as_base64(), default=None)
            if screenshot:
                metadata["screenshot_b64"] = screenshot
        if self.options.include_page_snapshot_on_error:
            snapshot = safe_call(lambda: page_snapshot(self.driver, include_html=False, include_screenshot=False).to_dict(), default=None)
            if snapshot:
                metadata["page_snapshot"] = snapshot
        return self._error_to_result(exc, action="scroll", context=metadata, message=message, duration_ms=context.duration_ms, correlation_id=context.request.correlation_id)

    def _error_to_result(
        self,
        exc: BaseException,
        *,
        action: str = "scroll",
        context: Optional[Mapping[str, Any]] = None,
        message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        correlation_id: Optional[str] = None,
    ) -> dict:
        browser_error = exc if isinstance(exc, BrowserError) else wrap_browser_exception(
            exc,
            action=action,
            message=message or str(exc),
            context=context,
            default_error_cls=ScrollError,
        )
        if isinstance(browser_error, BrowserError) and context:
            browser_error.context.update(sanitize_context(context, redact=False))
        result = error_result(
            action=action,
            message=message or getattr(browser_error, "message", str(browser_error)),
            error=browser_error,
            metadata=dict(context or {}),
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        )
        # Older callers expect a flat `element` key on failures.
        result.setdefault("element", None)
        return result

    def _build_success_message(self, message: str, data: Optional[Mapping[str, Any]] = None) -> dict:
        """Legacy helper retained for compatibility with previous module style."""

        return success_result(action="scroll", message=message, data=dict(data or {}))

    def _build_error_message(self, error_msg: str) -> dict:
        """Legacy helper retained for compatibility with previous module style."""

        return self._error_to_result(ScrollError(error_msg), action="scroll", message=error_msg)


__all__ = [
    "DoScroll",
    "ScrollOptions",
    "ScrollRequest",
    "ScrollState",
    "ScrollExecutionContext",
]


if __name__ == "__main__":
    print("\n=== Running Do Scroll ===\n")
    printer.status("TEST", "Do Scroll initialized", "info")

    class FakeElement:
        tag_name = "section"
        text = "Target section"
        location = {"x": 0, "y": 900}
        size = {"width": 300, "height": 120}

        def get_attribute(self, name: str):
            values = {
                "id": "target",
                "class": "content target",
                "role": "region",
                "aria-label": "Target section",
                "outerHTML": "<section id='target'>Target section</section>",
            }
            return values.get(name)

        def is_displayed(self):
            return True

        def is_enabled(self):
            return True

        def is_selected(self):
            return False

    class FakeDriver:
        def __init__(self):
            self.current_url = "https://example.test/page"
            self.title = "Example"
            self.scroll_x = 0
            self.scroll_y = 0
            self.viewport_width = 1000
            self.viewport_height = 700
            self.document_width = 1200
            self.document_height = 2400
            self.element = FakeElement()

        def execute_script(self, script: str, *args):
            if "return" in script and "max_y" in script:
                return self._state()
            if "scrollTo" in script and "Math.max" in script:
                self.scroll_y = self.document_height - self.viewport_height
                return None
            if "scrollTo" in script and "Math.round(maxY * pct)" in script:
                pct = float(args[0]) / 100.0
                self.scroll_y = int((self.document_height - self.viewport_height) * pct)
                return None
            if "scrollTo" in script:
                self.scroll_x = int(args[0])
                self.scroll_y = int(args[1])
                return None
            if "scrollBy" in script:
                self.scroll_x = max(0, min(self.document_width - self.viewport_width, self.scroll_x + int(args[0])))
                self.scroll_y = max(0, min(self.document_height - self.viewport_height, self.scroll_y + int(args[1])))
                return None
            if "scrollIntoView" in script:
                self.scroll_y = self.element.location["y"]
                return None
            return None

        def _state(self):
            return {
                "x": self.scroll_x,
                "y": self.scroll_y,
                "max_x": self.document_width - self.viewport_width,
                "max_y": self.document_height - self.viewport_height,
                "viewport_width": self.viewport_width,
                "viewport_height": self.viewport_height,
                "document_width": self.document_width,
                "document_height": self.document_height,
                "ready_state": "complete",
            }

        def find_element(self, by, selector):
            if selector == "#target":
                return self.element
            raise NoSuchElementException(selector)

        def get_screenshot_as_base64(self):
            return "ZmFrZS1zY3JlZW5zaG90"

    fake_driver = FakeDriver()
    scroller = DoScroll(fake_driver)

    result_to = scroller.scroll_to(10, 100)
    assert result_to["status"] == "success", result_to

    result_by = scroller.scroll_by(0, 250)
    assert result_by["status"] == "success", result_by

    result_direction = scroller.scroll_direction("down", 300)
    assert result_direction["status"] == "success", result_direction

    result_element = scroller.scroll_element_into_view("#target", position="center")
    assert result_element["status"] == "success", result_element

    result_percentage = scroller.scroll_to_percentage(50)
    assert result_percentage["status"] == "success", result_percentage

    result_bottom = scroller.scroll_to_bottom()
    assert result_bottom["status"] == "success", result_bottom

    result_state = scroller.perform("state")
    assert result_state["status"] == "success", result_state

    invalid = scroller.scroll_direction("diagonal")
    assert invalid["status"] == "error", invalid

    print("\n=== Test ran successfully ===\n")
