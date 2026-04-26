from __future__ import annotations

"""
Production-grade browser utility primitives.

This module is intentionally narrow. It provides browser-specific utility
behaviour that does not belong in concrete action modules such as DoClick,
DoType, DoScroll, DoNavigate, DoCopyCutPaste, or DoDragAndDrop.

Scope
-----
Utilities owns:
- human-paced typing against an already resolved element;
- human-like click movement against an already resolved element;
- compatibility link selection for callers that still use Utilities.select_link;
- browser-agent signal handling for pause/resume/termination flows;
- tiny module-specific wrappers that apply browser_utilities config and record
  optional browser memory events.

Utilities does not duplicate shared helper behaviour. Generic concerns such as
serialization, redaction, result construction, URL parsing, selector validation,
retry/backoff, snapshots, text normalization, timing, link scoring, and error
conversion stay in Browser_helpers.py and browser_errors.py.

Local imports are direct by design. They are not wrapped in try/except so import
or packaging issues fail clearly during development and deployment.
"""

import asyncio
import os
import platform
import random
import signal
import sys
import time as time_module

from sys import stderr
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from selenium.webdriver.common.action_chains import ActionChains
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, ParamSpec, Sequence, Tuple, TypeVar

from .utils.config_loader import load_global_config, get_config_section
from .utils.browser_errors import *
from .utils.Browser_helpers import *
from .browser_memory import BrowserMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Browser Utilities")
printer = PrettyPrinter()

R = TypeVar("R")
T = TypeVar("T")
P = ParamSpec("P")

_EXITING = False
DEFAULT_INTERRUPTIBLE_TASK_PATTERNS: Tuple[str, ...] = ("step", "multi_act", "get_next_action")


@dataclass(frozen=True)
class UtilityOptions:
    """Config-backed options for module-specific utility behaviours."""

    human_type_min_delay: float = 0.05
    human_type_max_delay: float = 0.15
    human_type_newline_pause: float = 0.05
    human_type_max_chars: int = 100_000
    human_click_min_pause: float = 0.10
    human_click_max_pause: float = 0.30
    select_link_use_helper: bool = True
    memory_enabled: bool = True
    memory_namespace: str = "actions"
    memory_source: str = "browser_utilities"
    include_event_payloads: bool = True

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "UtilityOptions":
        cfg = dict(config or {})
        human_type = dict(cfg.get("human_type") or {})
        human_click = dict(cfg.get("human_click") or {})
        link_selection = dict(cfg.get("link_selection") or {})
        memory = dict(cfg.get("memory") or {})

        min_delay = coerce_float(human_type.get("min_delay", cfg.get("human_type_min_delay", 0.05)), default=0.05, minimum=0.0, maximum=30.0)
        max_delay = coerce_float(human_type.get("max_delay", cfg.get("human_type_max_delay", 0.15)), default=0.15, minimum=0.0, maximum=30.0)
        if max_delay < min_delay:
            max_delay = min_delay

        click_min = coerce_float(human_click.get("min_pause", cfg.get("human_click_min_pause", 0.10)), default=0.10, minimum=0.0, maximum=30.0)
        click_max = coerce_float(human_click.get("max_pause", cfg.get("human_click_max_pause", 0.30)), default=0.30, minimum=0.0, maximum=30.0)
        if click_max < click_min:
            click_max = click_min

        return cls(
            human_type_min_delay=min_delay,
            human_type_max_delay=max_delay,
            human_type_newline_pause=coerce_float(human_type.get("newline_pause", 0.05), default=0.05, minimum=0.0, maximum=30.0),
            human_type_max_chars=coerce_int(human_type.get("max_chars", cfg.get("human_type_max_chars", 100_000)), default=100_000, minimum=1),
            human_click_min_pause=click_min,
            human_click_max_pause=click_max,
            select_link_use_helper=coerce_bool(link_selection.get("use_helper", True), default=True),
            memory_enabled=coerce_bool(memory.get("enabled", cfg.get("memory_enabled", True)), default=True),
            memory_namespace=str(memory.get("namespace", cfg.get("memory_namespace", "actions")) or "actions"),
            memory_source=str(memory.get("source", cfg.get("memory_source", "browser_utilities")) or "browser_utilities"),
            include_event_payloads=coerce_bool(memory.get("include_event_payloads", True), default=True),
        )


@dataclass(frozen=True)
class SignalHandlerOptions:
    """Runtime policy for browser-agent signal handling."""

    enabled: bool = True
    exit_on_second_int: bool = True
    windows_immediate_exit: bool = True
    restore_handlers_on_unregister: bool = True
    print_resume_prompt: bool = True
    reset_terminal_on_exit: bool = True
    interruptible_task_patterns: Tuple[str, ...] = DEFAULT_INTERRUPTIBLE_TASK_PATTERNS

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "SignalHandlerOptions":
        cfg = dict(config or {})
        signal_cfg = dict(cfg.get("signal_handler") or {})
        patterns = ensure_list(signal_cfg.get("interruptible_task_patterns", DEFAULT_INTERRUPTIBLE_TASK_PATTERNS))
        return cls(
            enabled=coerce_bool(signal_cfg.get("enabled", True), default=True),
            exit_on_second_int=coerce_bool(signal_cfg.get("exit_on_second_int", True), default=True),
            windows_immediate_exit=coerce_bool(signal_cfg.get("windows_immediate_exit", True), default=True),
            restore_handlers_on_unregister=coerce_bool(signal_cfg.get("restore_handlers_on_unregister", True), default=True),
            print_resume_prompt=coerce_bool(signal_cfg.get("print_resume_prompt", True), default=True),
            reset_terminal_on_exit=coerce_bool(signal_cfg.get("reset_terminal_on_exit", True), default=True),
            interruptible_task_patterns=tuple(str(item) for item in patterns if str(item).strip()) or DEFAULT_INTERRUPTIBLE_TASK_PATTERNS,
        )


@dataclass(frozen=True)
class UtilityEvent:
    """Small event shape used only for optional BrowserMemory recording."""

    action: str
    status: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: new_correlation_id("util"))
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Utilities:
    """Browser-specific utility helpers.

    Keep this class intentionally small. Concrete browser actions belong in the
    Do* modules, and generic primitives belong in Browser_helpers.py.
    """

    def __init__(self, memory: Optional[BrowserMemory] = None):
        self.config = load_global_config()
        self.utilities_config = get_config_section("browser_utilities") or {}
        self.options = UtilityOptions.from_config(self.utilities_config)
        self.memory = memory if memory is not None else BrowserMemory()
        logger.info("Browser Utilities initialized.")

    # ------------------------------------------------------------------
    # Backwards-compatible static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def human_type(element: Any, text: Any, min_delay: float = 0.05, max_delay: float = 0.15) -> None:
        """Type text into an already-resolved element with human-paced delays.

        This method intentionally does not locate elements, validate selectors,
        build result payloads, or retry. Those concerns belong to DoType or the
        shared helper/error layers.
        """

        text_value = "" if text is None else str(text)
        min_delay = max(0.0, float(min_delay))
        max_delay = max(min_delay, float(max_delay))
        for character in text_value:
            element.send_keys(character)
            if max_delay > 0:
                time_module.sleep(random.uniform(min_delay, max_delay))

    @staticmethod
    def human_click(driver: Any, element: Any, min_pause: float = 0.10, max_pause: float = 0.30) -> None:
        """Move to an already-resolved element and click with a small pause."""

        min_pause = max(0.0, float(min_pause))
        max_pause = max(min_pause, float(max_pause))
        pause = random.uniform(min_pause, max_pause) if max_pause > 0 else 0.0
        ActionChains(driver).move_to_element(element).pause(pause).click().perform()

    @staticmethod
    def select_link(query: str, elements: Sequence[Any]) -> Any:
        """Select the best link-like element for a query.

        Delegates scoring/normalization to Browser_helpers.select_best_link when
        available in the shared helper module. This compatibility method returns
        the selected original element, matching the original contract.
        """

        if not elements:
            raise BrowserValidationError("No elements available for selection", context={"query": query})
        selected = select_best_link(query, elements)
        if selected is None:
            raise BrowserValidationError("Unable to select a matching link", context={"query": query, "candidate_count": len(elements)})
        return selected

    # ------------------------------------------------------------------
    # Config-backed wrappers
    # ------------------------------------------------------------------
    def type_like_human(self, element: Any, text: Any, *, min_delay: Optional[float] = None, max_delay: Optional[float] = None) -> Dict[str, Any]:
        """Config-backed wrapper around ``human_type`` returning a result dict."""

        start = monotonic_ms()
        correlation_id = new_correlation_id("human-type")
        text_value = "" if text is None else str(text)
        try:
            if len(text_value) > self.options.human_type_max_chars:
                raise BrowserValidationError(
                    "Text exceeds configured browser utility human_type limit",
                    context={"length": len(text_value), "max_chars": self.options.human_type_max_chars},
                )
            self.human_type(
                element,
                text_value,
                self.options.human_type_min_delay if min_delay is None else min_delay,
                self.options.human_type_max_delay if max_delay is None else max_delay,
            )
            result = success_result(
                action="human_type",
                message="Human-paced typing completed",
                data={"chars_typed": len(text_value), "text_fingerprint": fingerprint_text(text_value)},
                metadata={"element": safe_serialize(element)},
                duration_ms=elapsed_ms(start),
                correlation_id=correlation_id,
            )
            self._record_event("human_type", result)
            return result
        except Exception as exc:
            result = error_result(
                action="human_type",
                message="Human-paced typing failed",
                error=wrap_browser_exception(exc, action="human_type", context={"chars": len(text_value)}),
                duration_ms=elapsed_ms(start),
                correlation_id=correlation_id,
            )
            self._record_event("human_type", result)
            return result

    def click_like_human(self, driver: Any, element: Any, *, min_pause: Optional[float] = None, max_pause: Optional[float] = None) -> Dict[str, Any]:
        """Config-backed wrapper around ``human_click`` returning a result dict."""

        start = monotonic_ms()
        correlation_id = new_correlation_id("human-click")
        try:
            self.human_click(
                driver,
                element,
                self.options.human_click_min_pause if min_pause is None else min_pause,
                self.options.human_click_max_pause if max_pause is None else max_pause,
            )
            result = success_result(
                action="human_click",
                message="Human-like click completed",
                data={"clicked": True},
                metadata={"element": safe_serialize(element)},
                duration_ms=elapsed_ms(start),
                correlation_id=correlation_id,
            )
            self._record_event("human_click", result)
            return result
        except Exception as exc:
            result = error_result(
                action="human_click",
                message="Human-like click failed",
                error=wrap_browser_exception(exc, action="human_click"),
                duration_ms=elapsed_ms(start),
                correlation_id=correlation_id,
            )
            self._record_event("human_click", result)
            return result

    def select_link_result(self, query: str, elements: Sequence[Any]) -> Dict[str, Any]:
        """Return a stable result payload for link selection diagnostics."""

        start = monotonic_ms()
        correlation_id = new_correlation_id("select-link")
        try:
            selected = self.select_link(query, elements)
            result = success_result(
                action="select_link",
                message="Selected best matching link",
                data={
                    "query": query,
                    "candidate_count": len(elements),
                    "selected": safe_serialize(selected),
                },
                duration_ms=elapsed_ms(start),
                correlation_id=correlation_id,
            )
            self._record_event("select_link", result)
            return result
        except Exception as exc:
            result = error_result(
                action="select_link",
                message="Link selection failed",
                error=wrap_browser_exception(exc, action="select_link", context={"query": query, "candidate_count": len(elements or [])}),
                duration_ms=elapsed_ms(start),
                correlation_id=correlation_id,
            )
            self._record_event("select_link", result)
            return result

    def _record_event(self, action: str, result: Mapping[str, Any]) -> None:
        """Record lightweight utility events when BrowserMemory supports it."""

        if not self.options.memory_enabled or self.memory is None:
            return
        try:
            event = UtilityEvent(
                action=action,
                status=str(result.get("status", "unknown")),
                message=str(result.get("message", "")),
                metadata={"result": safe_serialize(result)} if self.options.include_event_payloads else {},
            )
            if hasattr(self.memory, "remember_action"):
                self.memory.remember_action(
                    action=action,
                    result=event.to_dict(),
                    namespace=self.options.memory_namespace,
                    tags=("utility", action),
                )
            elif hasattr(self.memory, "put"):
                self.memory.put(
                    key=event.correlation_id,
                    value=event.to_dict(),
                    namespace=self.options.memory_namespace,
                    kind="utility_event",
                    action=action,
                    source=self.options.memory_source,
                    tags=("utility", action),
                )
        except Exception as exc:
            logger.debug("Unable to record browser utility event: %s", exc)


class SignalHandler:
    """Signal management for interactive browser-agent runs.

    First SIGINT pauses/cancels interruptible tasks. A second SIGINT exits when
    configured. SIGTERM always exits. The class can be disabled for notebooks,
    hosted workers, tests, or applications that own process signals.
    """

    def __init__(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        pause_callback: Optional[Callable[[], None]] = None,
        resume_callback: Optional[Callable[[], None]] = None,
        custom_exit_callback: Optional[Callable[[], None]] = None,
        exit_on_second_int: bool = True,
        interruptible_task_patterns: Optional[Sequence[str]] = None,
        disabled: bool = False,
        options: Optional[SignalHandlerOptions] = None,
    ):
        self.options = options or SignalHandlerOptions(
            enabled=not disabled,
            exit_on_second_int=exit_on_second_int,
            interruptible_task_patterns=tuple(interruptible_task_patterns or DEFAULT_INTERRUPTIBLE_TASK_PATTERNS),
        )
        self.loop = loop or asyncio.get_event_loop()
        self.pause_callback = pause_callback
        self.resume_callback = resume_callback
        self.custom_exit_callback = custom_exit_callback
        self.exit_on_second_int = self.options.exit_on_second_int
        self.interruptible_task_patterns = tuple(interruptible_task_patterns or self.options.interruptible_task_patterns)
        self.is_windows = platform.system() == "Windows"
        self.disabled = disabled or not self.options.enabled
        self.original_sigint_handler: Any = None
        self.original_sigterm_handler: Any = None
        self.registered = False
        self._initialize_loop_state()

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any]] = None,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        pause_callback: Optional[Callable[[], None]] = None,
        resume_callback: Optional[Callable[[], None]] = None,
        custom_exit_callback: Optional[Callable[[], None]] = None,
    ) -> "SignalHandler":
        options = SignalHandlerOptions.from_config(config or {})
        return cls(
            loop=loop,
            pause_callback=pause_callback,
            resume_callback=resume_callback,
            custom_exit_callback=custom_exit_callback,
            options=options,
            disabled=not options.enabled,
        )

    def __enter__(self) -> "SignalHandler":
        self.register()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.unregister()

    def _initialize_loop_state(self) -> None:
        setattr(self.loop, "ctrl_c_pressed", False)
        setattr(self.loop, "waiting_for_input", False)

    def register(self) -> None:
        """Register SIGINT/SIGTERM handlers unless disabled."""

        if self.disabled or self.registered:
            return
        try:
            if self.is_windows:
                self.original_sigint_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._windows_sigint_handler)
            else:
                self.original_sigint_handler = signal.getsignal(signal.SIGINT)
                self.original_sigterm_handler = signal.getsignal(signal.SIGTERM)
                self.loop.add_signal_handler(signal.SIGINT, self.sigint_handler)
                self.loop.add_signal_handler(signal.SIGTERM, self.sigterm_handler)
            self.registered = True
        except Exception as exc:
            logger.debug("Signal handlers were not registered: %s", exc)

    def unregister(self) -> None:
        """Unregister signal handlers and restore original handlers when possible."""

        if self.disabled or not self.registered:
            return
        try:
            if self.is_windows:
                if self.options.restore_handlers_on_unregister and self.original_sigint_handler is not None:
                    signal.signal(signal.SIGINT, self.original_sigint_handler)
            else:
                with suppress(Exception):
                    self.loop.remove_signal_handler(signal.SIGINT)
                with suppress(Exception):
                    self.loop.remove_signal_handler(signal.SIGTERM)
                if self.options.restore_handlers_on_unregister:
                    if self.original_sigint_handler is not None:
                        with suppress(Exception):
                            signal.signal(signal.SIGINT, self.original_sigint_handler)
                    if self.original_sigterm_handler is not None:
                        with suppress(Exception):
                            signal.signal(signal.SIGTERM, self.original_sigterm_handler)
        except Exception as exc:
            logger.warning("Error while unregistering signal handlers: %s", exc)
        finally:
            self.registered = False

    def _windows_sigint_handler(self, sig: int, frame: Any) -> None:
        if self.options.windows_immediate_exit:
            print("\n\n🛑 Got Ctrl+C. Exiting immediately on Windows...\n", file=stderr)
            self._run_exit_callback()
            os._exit(0)
        self.sigint_handler()

    def _run_exit_callback(self) -> None:
        if self.custom_exit_callback:
            try:
                self.custom_exit_callback()
            except Exception as exc:
                logger.error("Error in exit callback: %s", exc)

    def _reset_terminal(self) -> None:
        if not self.options.reset_terminal_on_exit:
            return
        for stream in (stderr, sys.stdout):
            with suppress(Exception):
                print("\033[?25h\033[0m\033[?1l\033[?2004l\r", end="", flush=True, file=stream)

    def _handle_second_ctrl_c(self) -> None:
        global _EXITING
        if not _EXITING:
            _EXITING = True
            self._run_exit_callback()
        print("\n\n🛑 Got second Ctrl+C. Exiting immediately...\n", file=stderr)
        self._reset_terminal()
        print("(tip: press [Enter] once to fix escape codes appearing after chrome exit)", file=stderr)
        os._exit(0)

    def sigint_handler(self) -> None:
        """Handle SIGINT: first press pauses, second press exits when enabled."""

        global _EXITING
        if _EXITING:
            os._exit(0)

        if getattr(self.loop, "ctrl_c_pressed", False):
            if getattr(self.loop, "waiting_for_input", False):
                return
            if self.exit_on_second_int:
                self._handle_second_ctrl_c()
            return

        setattr(self.loop, "ctrl_c_pressed", True)
        self._cancel_interruptible_tasks()
        if self.pause_callback:
            try:
                self.pause_callback()
            except Exception as exc:
                logger.error("Error in pause callback: %s", exc)
        print("----------------------------------------------------------------------", file=stderr)

    def sigterm_handler(self) -> None:
        """Handle SIGTERM as a clean immediate shutdown signal."""

        global _EXITING
        if not _EXITING:
            _EXITING = True
            print("\n\n🛑 SIGTERM received. Exiting immediately...\n\n", file=stderr)
            self._run_exit_callback()
        self._reset_terminal()
        os._exit(0)

    def _cancel_interruptible_tasks(self) -> List[str]:
        """Cancel matching asyncio tasks and return their names."""

        cancelled: List[str] = []
        try:
            current_task = asyncio.current_task(self.loop)
        except RuntimeError:
            current_task = None

        for task in asyncio.all_tasks(self.loop):
            if task.done() or task is current_task:
                continue
            task_name = task.get_name() if hasattr(task, "get_name") else str(task)
            if any(pattern in task_name for pattern in self.interruptible_task_patterns):
                logger.debug("Cancelling task: %s", task_name)
                task.cancel()
                task.add_done_callback(self._consume_cancelled_task_exception)
                cancelled.append(task_name)

        if current_task and not current_task.done():
            task_name = current_task.get_name() if hasattr(current_task, "get_name") else str(current_task)
            if any(pattern in task_name for pattern in self.interruptible_task_patterns):
                logger.debug("Cancelling current task: %s", task_name)
                current_task.cancel()
                cancelled.append(task_name)
        return cancelled

    @staticmethod
    def _consume_cancelled_task_exception(task: asyncio.Task[Any]) -> None:
        with suppress(asyncio.CancelledError, Exception):
            if task.cancelled():
                return
            task.exception()

    def wait_for_resume(self) -> None:
        """Wait for Enter to resume or Ctrl+C to exit."""

        setattr(self.loop, "waiting_for_input", True)
        original_handler = signal.getsignal(signal.SIGINT)
        with suppress(ValueError):
            signal.signal(signal.SIGINT, signal.default_int_handler)

        try:
            if self.options.print_resume_prompt:
                green = "\x1b[32;1m"
                red = "\x1b[31m"
                blink = "\033[33;5m"
                unblink = "\033[0m"
                reset = "\x1b[0m"
                print(
                    f"➡️  Press {green}[Enter]{reset} to resume or {red}[Ctrl+C]{reset} again to exit{blink}...{unblink} ",
                    end="",
                    flush=True,
                    file=stderr,
                )
            input()
            if self.resume_callback:
                self.resume_callback()
        except KeyboardInterrupt:
            self._handle_second_ctrl_c()
        finally:
            with suppress(Exception):
                signal.signal(signal.SIGINT, original_handler)
            setattr(self.loop, "waiting_for_input", False)

    def reset(self) -> None:
        """Reset pause/wait flags after resuming."""

        setattr(self.loop, "ctrl_c_pressed", False)
        setattr(self.loop, "waiting_for_input", False)

    def state(self) -> Dict[str, Any]:
        """Return a small state snapshot for diagnostics/tests."""

        return {
            "registered": self.registered,
            "disabled": self.disabled,
            "is_windows": self.is_windows,
            "ctrl_c_pressed": bool(getattr(self.loop, "ctrl_c_pressed", False)),
            "waiting_for_input": bool(getattr(self.loop, "waiting_for_input", False)),
            "interruptible_task_patterns": list(self.interruptible_task_patterns),
        }


if __name__ == "__main__":
    print("\n=== Running Utilities ===\n")
    printer.status("TEST", "Utilities initialized", "info")

    class _FakeElement:
        def __init__(self, text: str = ""):
            self.text = text
            self.typed = ""
            self.clicked = False
            self.tag_name = "a"

        def send_keys(self, value: str) -> None:
            self.typed += str(value)

        def get_attribute(self, name: str) -> str:
            values = {
                "href": "https://example.com/docs",
                "outerHTML": f"<a href='https://example.com/docs'>{self.text}</a>",
                "aria-label": self.text,
                "role": "link",
            }
            return values.get(name, "")

    class _FakeActionChains:
        def __init__(self, driver: Any):
            self.driver = driver
            self.element = None

        def move_to_element(self, element: Any) -> "_FakeActionChains":
            self.element = element
            return self

        def pause(self, seconds: float) -> "_FakeActionChains":
            return self

        def click(self) -> "_FakeActionChains":
            if self.element is not None:
                self.element.clicked = True
            return self

        def perform(self) -> None:
            return None

    ActionChains = _FakeActionChains  # type: ignore[assignment]

    element = _FakeElement("OpenAI documentation")
    Utilities.human_type(element, "abc", min_delay=0.0, max_delay=0.0)
    assert element.typed == "abc"

    Utilities.human_click(object(), element, min_pause=0.0, max_pause=0.0)
    assert element.clicked is True

    links = [_FakeElement("Random page"), _FakeElement("OpenAI API documentation")]
    selected = Utilities.select_link("OpenAI documentation", links)
    assert selected is links[1]

    utility = Utilities(memory=BrowserMemory())
    typed_result = utility.type_like_human(_FakeElement("Input"), "hello", min_delay=0.0, max_delay=0.0)
    assert typed_result["status"] == "success"

    click_result = utility.click_like_human(object(), _FakeElement("Button"), min_pause=0.0, max_pause=0.0)
    assert click_result["status"] == "success"

    select_result = utility.select_link_result("OpenAI docs", links)
    assert select_result["status"] == "success"

    handler = SignalHandler(disabled=True)
    handler.register()
    handler.reset()
    assert handler.state()["disabled"] is True
    handler.unregister()

    print("\n=== Test ran successfully ===\n")
