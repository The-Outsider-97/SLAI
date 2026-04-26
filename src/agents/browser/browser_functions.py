from __future__ import annotations

"""
Production-grade browser function orchestrator.

This module provides the stable function-facing façade for the browser agent
subsystem. It does not own browser lifecycle, concrete Selenium action logic,
content-specific extraction, or planning. Those responsibilities belong to the
BrowserAgent, the individual function modules, content handlers, and planning
layers.

Instead, BrowserFunctions coordinates the concrete browser function modules and
exposes one consistent contract for higher-level agents, workflows, tests, and
future tool adapters:

- attach or detach a driver without rebuilding unrelated state;
- lazily compose navigation, click, type, scroll, clipboard, and drag/drop
  executors around that driver;
- register built-in and custom browser functions with stable metadata;
- normalize aliases such as click_element, do_type, open_url, scroll_element,
  drag_element, and drop_element into canonical actions;
- execute one function, one task payload, or a multi-step workflow;
- build shared success/error payloads through browser helpers;
- convert arbitrary exceptions through the browser error hierarchy;
- optionally record action outcomes in BrowserMemory;
- provide utility functions such as screenshot and page extraction that are
  orchestration-level conveniences rather than concrete action modules.

Design principles
-----------------
1. No duplicated concrete action logic. Click/type/scroll/navigation/clipboard
   and drag/drop are delegated to their modules.
2. No duplicated helper logic. Result normalization, redaction, serialization,
   timing, validation, hashes, and browser-domain exceptions come from the
   shared browser utilities.
3. Direct local imports. Packaging errors should fail clearly; local imports are
   intentionally not wrapped in try/except.
4. Config driven. Runtime behavior is controlled by the browser_functions block
   in browser_config.yaml, while concrete modules retain their own config
   sections.
5. Expandable. New browser functions can be registered without rewriting the
   dispatcher or changing the public execution contract.
"""

import asyncio
import inspect
import time as time_module

from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.browser_errors import *
from .utils.Browser_helpers import *
from .functions import *
from .browser_memory import BrowserMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Browser Functions")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BROWSER_FUNCTIONS_SCHEMA_VERSION = "1.0"
DEFAULT_FUNCTION_HISTORY_LIMIT = 250
DEFAULT_PAGE_TEXT_LIMIT = 25_000
DEFAULT_HTML_LIMIT = 100_000
DEFAULT_SCREENSHOT_FORMAT = "base64"

CANONICAL_FUNCTIONS: Tuple[str, ...] = (
    "navigate",
    "back",
    "forward",
    "refresh",
    "current_url",
    "history",
    "click",
    "type",
    "clear",
    "press_key",
    "scroll",
    "copy",
    "cut",
    "paste",
    "drag_and_drop",
    "screenshot",
    "extract_page",
    "page_state",
    "execute_workflow",
)

DEFAULT_ALIASES: Dict[str, str] = {
    # Navigation
    "go_to_url": "navigate",
    "open_url": "navigate",
    "url": "navigate",
    "navigate_to": "navigate",
    "go_back": "back",
    "browser_back": "back",
    "go_forward": "forward",
    "browser_forward": "forward",
    "refresh_page": "refresh",
    "reload": "refresh",
    "get_current_url": "current_url",
    "current": "current_url",
    "get_navigation_history": "history",
    "navigation_history": "history",
    # Click
    "click_element": "click",
    "do_click": "click",
    "press": "click",
    # Type
    "do_type": "type",
    "type_text": "type",
    "enter_text": "type",
    "input_text": "type",
    "type_element": "type",
    "clear_text": "clear",
    "press": "click",
    "press_key": "press_key",
    # Scroll
    "scroll_element": "scroll",
    "scroll_to_element": "scroll",
    "scroll_element_into_view": "scroll",
    "scroll_direction": "scroll",
    # Clipboard
    "copy_element": "copy",
    "cut_element": "cut",
    "paste_element": "paste",
    "clipboard_copy": "copy",
    "clipboard_cut": "cut",
    "clipboard_paste": "paste",
    # Drag and drop
    "drag_element": "drag_and_drop",
    "drop_element": "drag_and_drop",
    "drag_to_element": "drag_and_drop",
    "drag_by_offset": "drag_and_drop",
    "do_drag_and_drop": "drag_and_drop",
    "drag_drop": "drag_and_drop",
    # Browser state/extraction
    "screenshot_page": "screenshot",
    "take_screenshot": "screenshot",
    "get_dom": "extract_page",
    "extract": "extract_page",
    "extract_page_content": "extract_page",
    "page_content": "extract_page",
    "get_page_state": "page_state",
}

MEMORY_ACTION_NAMESPACE = "actions"
MEMORY_WORKFLOW_NAMESPACE = "workflows"
MEMORY_PAGE_NAMESPACE = "pages"

BrowserCallable = Callable[..., Any]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BrowserFunctionsOptions:
    """Runtime policy for the browser function orchestrator."""

    enabled: bool = True
    require_driver_for_browser_actions: bool = True
    normalize_function_names: bool = True
    record_history: bool = True
    max_history_entries: int = DEFAULT_FUNCTION_HISTORY_LIMIT
    memory_enabled: bool = True
    memory_namespace: str = MEMORY_ACTION_NAMESPACE
    record_successes_to_memory: bool = True
    record_errors_to_memory: bool = True
    record_page_extracts_to_memory: bool = True
    include_call_params_in_result: bool = False
    include_function_metadata: bool = True
    include_page_state_after_action: bool = False
    include_traceback_on_error: bool = False
    screenshot_format: str = DEFAULT_SCREENSHOT_FORMAT
    screenshot_on_error: bool = False
    max_page_text_chars: int = DEFAULT_PAGE_TEXT_LIMIT
    max_html_chars: int = DEFAULT_HTML_LIMIT
    workflow_stop_on_error: bool = True
    workflow_record_each_step: bool = True
    default_scroll_mode: str = "by"
    default_extract_preview_only: bool = False
    aliases: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "BrowserFunctionsOptions":
        cfg = dict(config or {})
        memory_cfg = dict(cfg.get("memory") or {})
        diagnostics_cfg = dict(cfg.get("diagnostics") or {})
        workflow_cfg = dict(cfg.get("workflow") or {})
        extraction_cfg = dict(cfg.get("extraction") or {})
        aliases = dict(DEFAULT_ALIASES)
        aliases.update({str(k).strip().lower(): str(v).strip().lower() for k, v in dict(cfg.get("aliases") or {}).items()})

        screenshot_format = str(cfg.get("screenshot_format", diagnostics_cfg.get("screenshot_format", DEFAULT_SCREENSHOT_FORMAT))).lower().strip()
        if screenshot_format not in {"base64", "png", "bytes"}:
            screenshot_format = DEFAULT_SCREENSHOT_FORMAT

        return cls(
            enabled=coerce_bool(cfg.get("enabled", True), default=True),
            require_driver_for_browser_actions=coerce_bool(cfg.get("require_driver_for_browser_actions", True), default=True),
            normalize_function_names=coerce_bool(cfg.get("normalize_function_names", True), default=True),
            record_history=coerce_bool(cfg.get("record_history", True), default=True),
            max_history_entries=coerce_int(cfg.get("max_history_entries", DEFAULT_FUNCTION_HISTORY_LIMIT), default=DEFAULT_FUNCTION_HISTORY_LIMIT, minimum=1),
            memory_enabled=coerce_bool(memory_cfg.get("enabled", cfg.get("memory_enabled", True)), default=True),
            memory_namespace=str(memory_cfg.get("namespace", cfg.get("memory_namespace", MEMORY_ACTION_NAMESPACE)) or MEMORY_ACTION_NAMESPACE),
            record_successes_to_memory=coerce_bool(memory_cfg.get("record_successes", True), default=True),
            record_errors_to_memory=coerce_bool(memory_cfg.get("record_errors", True), default=True),
            record_page_extracts_to_memory=coerce_bool(memory_cfg.get("record_page_extracts", True), default=True),
            include_call_params_in_result=coerce_bool(diagnostics_cfg.get("include_call_params_in_result", cfg.get("include_call_params_in_result", False)), default=False),
            include_function_metadata=coerce_bool(diagnostics_cfg.get("include_function_metadata", cfg.get("include_function_metadata", True)), default=True),
            include_page_state_after_action=coerce_bool(diagnostics_cfg.get("include_page_state_after_action", cfg.get("include_page_state_after_action", False)), default=False),
            include_traceback_on_error=coerce_bool(diagnostics_cfg.get("include_traceback_on_error", cfg.get("include_traceback_on_error", False)), default=False),
            screenshot_format=screenshot_format,
            screenshot_on_error=coerce_bool(diagnostics_cfg.get("screenshot_on_error", cfg.get("screenshot_on_error", False)), default=False),
            max_page_text_chars=coerce_int(extraction_cfg.get("max_page_text_chars", cfg.get("max_page_text_chars", DEFAULT_PAGE_TEXT_LIMIT)), default=DEFAULT_PAGE_TEXT_LIMIT, minimum=0),
            max_html_chars=coerce_int(extraction_cfg.get("max_html_chars", cfg.get("max_html_chars", DEFAULT_HTML_LIMIT)), default=DEFAULT_HTML_LIMIT, minimum=0),
            workflow_stop_on_error=coerce_bool(workflow_cfg.get("stop_on_error", cfg.get("workflow_stop_on_error", True)), default=True),
            workflow_record_each_step=coerce_bool(workflow_cfg.get("record_each_step", cfg.get("workflow_record_each_step", True)), default=True),
            default_scroll_mode=str(cfg.get("default_scroll_mode", "by") or "by").lower().strip(),
            default_extract_preview_only=coerce_bool(extraction_cfg.get("preview_only", cfg.get("default_extract_preview_only", False)), default=False),
            aliases=aliases,
        )


@dataclass(frozen=True)
class BrowserFunctionSpec:
    """Metadata for a function registered in BrowserFunctions."""

    name: str
    handler: BrowserCallable
    description: str = ""
    aliases: Tuple[str, ...] = ()
    requires_driver: bool = True
    category: str = "browser"
    parameters: Tuple[str, ...] = ()
    memory_kind: str = "action_result"
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("handler", None)
        payload["handler"] = getattr(self.handler, "__name__", repr(self.handler))
        return payload


@dataclass(frozen=True)
class BrowserFunctionCall:
    """Normalized executable function call."""

    name: str
    canonical_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: new_correlation_id("fn"))
    source: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrowserFunctionExecution:
    """History record for one function execution."""

    call: BrowserFunctionCall
    status: str
    message: str
    duration_ms: float
    result_fingerprint: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Standalone helpers scoped to BrowserFunctions
# ---------------------------------------------------------------------------
def normalize_function_name(name: Any) -> str:
    """Normalize an external function/action name."""

    text = normalize_whitespace(name).lower().replace("-", "_").replace(" ", "_")
    return text.strip("_")


def normalize_call_payload(function_name: Optional[str] = None, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Normalize mixed function payload styles into a name/params mapping."""

    payload: Dict[str, Any] = dict(params or {})
    payload.update(kwargs)
    name = function_name or payload.pop("function", None) or payload.pop("name", None) or payload.pop("action", None) or payload.pop("task", None)
    if not name:
        raise MissingRequiredFieldError("Browser function call requires a function name", context={"payload_keys": sorted(payload.keys())})
    return {"name": normalize_function_name(name), "params": payload}


def function_result_status(result: Mapping[str, Any]) -> str:
    """Extract normalized status from a result payload."""

    return str((result or {}).get("status") or "success").lower().strip()


def is_success_result(result: Mapping[str, Any]) -> bool:
    return function_result_status(result) == "success"


def _callable_accepts_kwargs(handler: BrowserCallable) -> bool:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def filter_kwargs_for_callable(handler: BrowserCallable, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter params to a callable signature unless it accepts **kwargs."""

    if _callable_accepts_kwargs(handler):
        return dict(params)
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return dict(params)
    accepted = {
        name
        for name, param in signature.parameters.items()
        if param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {key: value for key, value in dict(params).items() if key in accepted}


def maybe_await(value: Any) -> Awaitable[Any]:
    """Wrap a value as an awaitable if it is not already awaitable."""

    async def _wrap() -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    return _wrap()


# ---------------------------------------------------------------------------
# BrowserFunctions
# ---------------------------------------------------------------------------
class BrowserFunctions:
    """Config-backed function orchestrator for the browser agent subsystem."""

    def __init__(self, driver: Any = None, memory: Optional[Union[BrowserMemory, bool]] = None, *, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = load_global_config()
        self.functions_config = dict(config or get_config_section("browser_functions") or {})
        self.options = BrowserFunctionsOptions.from_config(self.functions_config)
        self.driver = None
        if memory is False:
            self.memory = None
        elif isinstance(memory, BrowserMemory):
            self.memory = memory
        else:
            self.memory = None
        self.components: Dict[str, Any] = {}
        self.function_specs: "OrderedDict[str, BrowserFunctionSpec]" = OrderedDict()
        self.function_map: Dict[str, BrowserCallable] = {}
        self.alias_map: Dict[str, str] = dict(self.options.aliases)
        self.history: Deque[BrowserFunctionExecution] = deque(maxlen=self.options.max_history_entries)

        self._register_default_functions()
        if driver is not None:
            self.attach_driver(driver)

        logger.info("BrowserFunctions initialized with %s registered functions.", len(self.function_specs))

    # ------------------------------------------------------------------
    # Driver and component lifecycle
    # ------------------------------------------------------------------
    def attach_driver(self, driver: Any, *, rebuild_components: bool = True) -> Dict[str, Any]:
        """Attach a Selenium-like driver and optionally rebuild function modules."""

        start_ms = monotonic_ms()
        self.driver = driver
        if rebuild_components:
            self._build_components()
        return success_result(
            action="attach_driver",
            message="Browser driver attached",
            data={"components": sorted(self.components.keys())},
            duration_ms=elapsed_ms(start_ms),
        )

    def set_driver(self, driver: Any, *, rebuild_components: bool = True) -> Dict[str, Any]:
        """Alias for attach_driver, retained for caller readability."""

        return self.attach_driver(driver, rebuild_components=rebuild_components)

    def detach_driver(self, *, clear_components: bool = True) -> Dict[str, Any]:
        """Detach the active driver from the function orchestrator."""

        start_ms = monotonic_ms()
        self.driver = None
        if clear_components:
            self.components.clear()
        return success_result(
            action="detach_driver",
            message="Browser driver detached",
            data={"components_cleared": clear_components},
            duration_ms=elapsed_ms(start_ms),
        )

    def has_driver(self) -> bool:
        return self.driver is not None

    def require_driver(self, *, action: Optional[str] = None) -> Any:
        if self.driver is None:
            raise MissingDriverError("Browser function requires an attached driver", context={"action": action})
        return self.driver

    def _build_components(self) -> None:
        """Instantiate concrete browser function modules for the active driver."""

        driver = self.require_driver(action="build_components")
        self.components = {
            "navigate": DoNavigate(driver),
            "click": DoClick(driver),
            "scroll": DoScroll(driver),
            "type": DoType(driver),
            "clipboard": DoCopyCutPaste(driver),
            "drag_and_drop": DoDragAndDrop(driver),
        }
        # Convenience attributes retained for code that expects direct access.
        self.navigator = self.components["navigate"]
        self.clicker = self.components["click"]
        self.scroller = self.components["scroll"]
        self.typer = self.components["type"]
        self.clipboard = self.components["clipboard"]
        self.dragger = self.components["drag_and_drop"]

    def replace_component(self, name: str, component: Any) -> Dict[str, Any]:
        """Replace one component, primarily for tests or custom integrations."""

        canonical = self.resolve_name(name)
        self.components[canonical] = component
        if canonical == "navigate":
            self.navigator = component
        elif canonical == "click":
            self.clicker = component
        elif canonical == "scroll":
            self.scroller = component
        elif canonical == "type":
            self.typer = component
        elif canonical == "clipboard":
            self.clipboard = component
        elif canonical == "drag_and_drop":
            self.dragger = component
        return success_result(action="replace_component", message="Browser component replaced", data={"component": canonical})

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def _register_default_functions(self) -> None:
        """Register built-in browser functions and aliases."""

        self.register_function("navigate", self.navigate, description="Open a URL in the active browser tab.", aliases=("go_to_url", "open_url", "navigate_to"), category="navigation")
        self.register_function("back", self.back, description="Move one browser history entry backward.", aliases=("go_back", "browser_back"), category="navigation")
        self.register_function("forward", self.forward, description="Move one browser history entry forward.", aliases=("go_forward", "browser_forward"), category="navigation")
        self.register_function("refresh", self.refresh, description="Reload the active browser tab.", aliases=("refresh_page", "reload"), category="navigation")
        self.register_function("current_url", self.current_url, description="Return the current browser URL.", aliases=("get_current_url",), category="navigation")
        self.register_function("history", self.get_navigation_history, description="Return function/navigation history.", aliases=("get_navigation_history", "navigation_history"), requires_driver=False, category="navigation")

        self.register_function("click", self.click_element, description="Click an element by CSS selector.", aliases=("click_element", "do_click"), category="interaction")
        self.register_function("type", self.do_type, description="Type text into an element.", aliases=("do_type", "type_text", "enter_text", "input_text", "type_element"), category="interaction")
        self.register_function("clear", self.clear_text, description="Clear a writable element.", aliases=("clear_text",), category="interaction")
        self.register_function("press_key", self.press_key, description="Send a keyboard key to an element or active page.", aliases=("send_key",), category="interaction")
        self.register_function("scroll", self.scroll_element, description="Scroll the window or an element.", aliases=("scroll_element", "scroll_to_element", "scroll_direction"), category="interaction")

        self.register_function("copy", self.copy, description="Copy text from an element.", aliases=("copy_element", "clipboard_copy"), category="clipboard")
        self.register_function("cut", self.cut, description="Cut text from an element.", aliases=("cut_element", "clipboard_cut"), category="clipboard")
        self.register_function("paste", self.paste, description="Paste text into an element.", aliases=("paste_element", "clipboard_paste"), category="clipboard")

        self.register_function("drag_and_drop", self.drag_and_drop, description="Drag an element to a target or offset.", aliases=("drag_element", "drop_element", "drag_to_element", "drag_by_offset", "do_drag_and_drop"), category="interaction")
        self.register_function("screenshot", self.screenshot, description="Capture a browser screenshot.", aliases=("take_screenshot", "screenshot_page"), category="diagnostics")
        self.register_function("extract_page", self.extract_page, description="Extract current page URL, title, text, and optional HTML.", aliases=("extract", "get_dom", "page_content", "extract_page_content"), category="content")
        self.register_function("page_state", self.page_state, description="Return lightweight page state.", aliases=("get_page_state",), category="diagnostics")
        self.register_function("execute_workflow", self.execute_workflow, description="Execute a sequence of browser function steps.", aliases=("workflow", "run_workflow"), requires_driver=False, category="workflow")

    def register_function(
        self,
        name: str,
        handler: BrowserCallable,
        *,
        description: str = "",
        aliases: Iterable[str] = (),
        requires_driver: bool = True,
        category: str = "browser",
        parameters: Iterable[str] = (),
        memory_kind: str = "action_result",
        enabled: bool = True,
    ) -> Dict[str, Any]:
        canonical = normalize_function_name(name)
        if not canonical:
            raise MissingRequiredFieldError("Function name cannot be empty", context={"name": name})
        spec = BrowserFunctionSpec(
            name=canonical,
            handler=handler,
            description=description,
            aliases=tuple(normalize_function_name(alias) for alias in aliases if normalize_function_name(alias)),
            requires_driver=requires_driver,
            category=category,
            parameters=tuple(str(param) for param in parameters),
            memory_kind=memory_kind,
            enabled=enabled,
        )
        self.function_specs[canonical] = spec
        self.function_map[canonical] = handler
        for alias in spec.aliases:
            self.alias_map[alias] = canonical
            self.function_map[alias] = handler
        return success_result(action="register_function", message="Browser function registered", data=spec.to_dict())

    def unregister_function(self, name: str) -> Dict[str, Any]:
        canonical = self.resolve_name(name)
        spec = self.function_specs.pop(canonical, None)
        self.function_map.pop(canonical, None)
        if spec:
            for alias in spec.aliases:
                self.alias_map.pop(alias, None)
                self.function_map.pop(alias, None)
        return success_result(action="unregister_function", message="Browser function unregistered", data={"function": canonical, "removed": bool(spec)})

    def list_functions(self, *, include_aliases: bool = True) -> Dict[str, Any]:
        specs = [spec.to_dict() for spec in self.function_specs.values()]
        data: Dict[str, Any] = {"functions": specs}
        if include_aliases:
            data["aliases"] = dict(sorted(self.alias_map.items()))
        return success_result(action="list_functions", message="Browser functions listed", data=data)

    def describe_function(self, name: str) -> Dict[str, Any]:
        canonical = self.resolve_name(name)
        spec = self.function_specs.get(canonical)
        if not spec:
            return self._error(UnsupportedBrowserTaskError("Unknown browser function", context={"function": name}), action="describe_function")
        return success_result(action="describe_function", message="Browser function described", data=spec.to_dict())

    def resolve_name(self, name: Any) -> str:
        normalized = normalize_function_name(name)
        return self.alias_map.get(normalized, normalized)

    def _get_spec(self, function_name: Any) -> BrowserFunctionSpec:
        canonical = self.resolve_name(function_name)
        spec = self.function_specs.get(canonical)
        if not spec:
            raise UnsupportedBrowserTaskError("Unsupported browser function", context={"function": function_name, "canonical": canonical})
        if not spec.enabled:
            raise BrowserTaskError("Browser function is disabled", context={"function": canonical})
        return spec

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, function_name: Optional[str] = None, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute one registered browser function."""

        start_ms = monotonic_ms()
        call: Optional[BrowserFunctionCall] = None
        try:
            if not self.options.enabled:
                raise BrowserTaskError("BrowserFunctions is disabled by configuration")
            normalized = normalize_call_payload(function_name, params, **kwargs)
            spec = self._get_spec(normalized["name"])
            if spec.requires_driver and self.options.require_driver_for_browser_actions:
                self.require_driver(action=spec.name)

            call = BrowserFunctionCall(name=normalized["name"], canonical_name=spec.name, params=dict(normalized["params"]))
            raw_result = self._invoke_handler(spec.handler, call.params)
            result = self._normalize_execution_result(raw_result, call=call, spec=spec, start_ms=start_ms)
            self._record_execution(call, result, spec=spec, duration_ms=elapsed_ms(start_ms))
            return result
        except Exception as exc:
            return self._handle_execution_error(exc, call=call, function_name=function_name, start_ms=start_ms)

    async def async_execute(self, function_name: Optional[str] = None, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Async variant of execute for async callers."""

        start_ms = monotonic_ms()
        call: Optional[BrowserFunctionCall] = None
        try:
            normalized = normalize_call_payload(function_name, params, **kwargs)
            spec = self._get_spec(normalized["name"])
            if spec.requires_driver and self.options.require_driver_for_browser_actions:
                self.require_driver(action=spec.name)
            call = BrowserFunctionCall(name=normalized["name"], canonical_name=spec.name, params=dict(normalized["params"]))
            raw_result = spec.handler(**filter_kwargs_for_callable(spec.handler, call.params))
            resolved = await maybe_await(raw_result)
            result = self._normalize_execution_result(resolved, call=call, spec=spec, start_ms=start_ms)
            self._record_execution(call, result, spec=spec, duration_ms=elapsed_ms(start_ms))
            return result
        except Exception as exc:
            return self._handle_execution_error(exc, call=call, function_name=function_name, start_ms=start_ms)

    def perform(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute a BrowserAgent-style task payload."""

        payload = dict(validate_browser_task_payload(task_data))
        if "workflow" in payload:
            return self.execute_workflow(payload.get("workflow") or [])
        function_name = payload.pop("function", None) or payload.pop("action", None) or payload.pop("task", None)
        if not function_name:
            if "url" in payload:
                function_name = "navigate"
            elif "query" in payload:
                function_name = "search"
        return self.execute(function_name, payload)

    def perform_task(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        """BrowserAgent-compatible task entrypoint."""

        return self.perform(task_data)

    def __call__(self, function_name: str, **kwargs: Any) -> Dict[str, Any]:
        return self.execute(function_name, kwargs)

    def _invoke_handler(self, handler: BrowserCallable, params: Mapping[str, Any]) -> Any:
        filtered = filter_kwargs_for_callable(handler, params)
        return handler(**filtered)

    def _normalize_execution_result(self, raw_result: Any, *, call: BrowserFunctionCall, spec: BrowserFunctionSpec, start_ms: float) -> Dict[str, Any]:
        if isinstance(raw_result, Mapping):
            result = normalize_result(dict(raw_result), action=spec.name, default_message=f"{spec.name} completed")
        elif isinstance(raw_result, BaseException):
            result = exception_to_error_payload(raw_result, action=spec.name, include_traceback=self.options.include_traceback_on_error)
        else:
            result = success_result(action=spec.name, message=f"{spec.name} completed", data={"value": safe_serialize(raw_result)})

        result.setdefault("action", spec.name)
        result.setdefault("correlation_id", call.correlation_id)
        result["duration_ms"] = result.get("duration_ms", elapsed_ms(start_ms))
        if self.options.include_call_params_in_result:
            result["params"] = safe_serialize(call.params)
        if self.options.include_function_metadata:
            result.setdefault("metadata", {})
            if isinstance(result["metadata"], MutableMapping):
                result["metadata"].setdefault("function", spec.to_dict())
        if self.options.include_page_state_after_action and spec.requires_driver and self.driver is not None:
            result.setdefault("page_state", self._safe_page_state())
        return redact_mapping(prune_none(result))

    def _handle_execution_error(self, exc: BaseException, *, call: Optional[BrowserFunctionCall], function_name: Optional[str], start_ms: float) -> Dict[str, Any]:
        action = call.canonical_name if call else normalize_function_name(function_name or "browser_function")
        context = {"function": function_name, "call": call.to_dict() if call else None}
        browser_error = wrap_browser_exception(exc, action=action, context=context, default_error_cls=BrowserTaskError)
        extra: Dict[str, Any] = {"duration_ms": elapsed_ms(start_ms)}
        if self.options.screenshot_on_error and self.driver is not None:
            extra["screenshot"] = self._safe_screenshot_base64()
        result = browser_error.to_result(action=action, include_traceback=self.options.include_traceback_on_error, extra=extra)
        if call:
            spec = self.function_specs.get(call.canonical_name)
            self._record_execution(call, result, spec=spec, duration_ms=elapsed_ms(start_ms))
        return redact_mapping(result)

    def _error(self, exc: BaseException, *, action: Optional[str] = None, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        browser_error = wrap_browser_exception(exc, action=action, context=context, default_error_cls=BrowserTaskError)
        return browser_error.to_result(action=action)

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------
    def execute_workflow(self, workflow_script: Sequence[Mapping[str, Any]], *, stop_on_error: Optional[bool] = None) -> Dict[str, Any]:
        """Execute a list of function steps with normalized result capture."""

        start_ms = monotonic_ms()
        stop = self.options.workflow_stop_on_error if stop_on_error is None else bool(stop_on_error)
        executed: List[Dict[str, Any]] = []
        try:
            steps = validate_workflow_script(workflow_script, supported_actions=[*self.function_specs.keys(), *self.alias_map.keys()])
            for index, step in enumerate(steps):
                action = step.get("action")
                params = dict(step.get("params") or {})
                result = self.execute(action, params)
                executed.append({"index": index, "step": safe_serialize(step), "result": result})
                if function_result_status(result) == "error" and stop:
                    workflow_result = error_result(
                        action="execute_workflow",
                        message="Workflow step failed",
                        error=WorkflowStepFailedError("Workflow step failed", context={"index": index, "action": action, "result": result}),
                        data={"executed": executed},
                        duration_ms=elapsed_ms(start_ms),
                    )
                    self._remember_workflow(workflow_result)
                    return workflow_result
            workflow_result = success_result(
                action="execute_workflow",
                message="Workflow executed successfully",
                data={"executed": executed, "step_count": len(executed)},
                duration_ms=elapsed_ms(start_ms),
            )
            self._remember_workflow(workflow_result)
            return workflow_result
        except Exception as exc:
            result = self._error(exc, action="execute_workflow", context={"workflow_script": workflow_script, "executed": executed})
            result["duration_ms"] = elapsed_ms(start_ms)
            self._remember_workflow(result)
            return result

    async def async_execute_workflow(self, workflow_script: Sequence[Mapping[str, Any]], *, stop_on_error: Optional[bool] = None) -> Dict[str, Any]:
        """Async workflow execution variant."""

        start_ms = monotonic_ms()
        stop = self.options.workflow_stop_on_error if stop_on_error is None else bool(stop_on_error)
        executed: List[Dict[str, Any]] = []
        try:
            steps = validate_workflow_script(workflow_script, supported_actions=[*self.function_specs.keys(), *self.alias_map.keys()])
            for index, step in enumerate(steps):
                result = await self.async_execute(step.get("action"), dict(step.get("params") or {}))
                executed.append({"index": index, "step": safe_serialize(step), "result": result})
                if function_result_status(result) == "error" and stop:
                    workflow_result = error_result(
                        action="execute_workflow",
                        message="Workflow step failed",
                        error=WorkflowStepFailedError("Workflow step failed", context={"index": index, "result": result}),
                        data={"executed": executed},
                        duration_ms=elapsed_ms(start_ms),
                    )
                    self._remember_workflow(workflow_result)
                    return workflow_result
            workflow_result = success_result(action="execute_workflow", message="Workflow executed successfully", data={"executed": executed}, duration_ms=elapsed_ms(start_ms))
            self._remember_workflow(workflow_result)
            return workflow_result
        except Exception as exc:
            result = self._error(exc, action="execute_workflow", context={"workflow_script": workflow_script, "executed": executed})
            result["duration_ms"] = elapsed_ms(start_ms)
            self._remember_workflow(result)
            return result

    # ------------------------------------------------------------------
    # Concrete function wrappers
    # ------------------------------------------------------------------
    def navigate(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        navigator = self._component("navigate")
        if hasattr(navigator, "go_to_url"):
            return self._call_component(navigator.go_to_url, {"url": url, **kwargs}, action="navigate")
        if hasattr(navigator, "navigate"):
            return self._call_component(navigator.navigate, {"url": url, **kwargs}, action="navigate")
        raise BrowserTaskError("Navigation component does not expose go_to_url/navigate")

    def back(self, **kwargs: Any) -> Dict[str, Any]:
        navigator = self._component("navigate")
        return self._call_component(getattr(navigator, "go_back", getattr(navigator, "back", None)), kwargs, action="back")

    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        navigator = self._component("navigate")
        return self._call_component(getattr(navigator, "go_forward", getattr(navigator, "forward", None)), kwargs, action="forward")

    def refresh(self, **kwargs: Any) -> Dict[str, Any]:
        navigator = self._component("navigate")
        return self._call_component(getattr(navigator, "refresh_page", getattr(navigator, "refresh", None)), kwargs, action="refresh")

    def current_url(self, **kwargs: Any) -> Dict[str, Any]:
        navigator = self._component("navigate")
        if hasattr(navigator, "get_current_url"):
            return self._call_component(navigator.get_current_url, kwargs, action="current_url")
        return success_result(action="current_url", message="Current URL returned", data={"url": getattr(self.driver, "current_url", "")})

    def get_navigation_history(self, **kwargs: Any) -> Dict[str, Any]:
        navigation_history: Any = []
        if "navigate" in self.components and hasattr(self.components["navigate"], "get_navigation_history"):
            navigation_history = self.components["navigate"].get_navigation_history()
        return success_result(
            action="history",
            message="Browser function history returned",
            data={
                "function_history": [entry.to_dict() for entry in self.history],
                "navigation_history": safe_serialize(navigation_history),
            },
        )

    def click_element(self, selector: str, wait_before_execution: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
        clicker = self._component("click")
        params = {"selector": selector, **kwargs}
        if wait_before_execution is not None:
            params["wait_before_execution"] = wait_before_execution
        if hasattr(clicker, "click"):
            return self._call_component(clicker.click, params, action="click")
        if hasattr(clicker, "_perform_click"):
            return self._call_component(clicker._perform_click, params, action="click")
        if hasattr(clicker, "click_element"):
            result = clicker.click_element(selector, wait_before_execution or 0.0)
            return normalize_result(result, action="click")
        raise BrowserTaskError("Click component does not expose a click method")

    def do_type(self, selector: str, text: Optional[str] = None, raw_input: Optional[str] = None, clear_before: Optional[bool] = None, **kwargs: Any) -> Dict[str, Any]:
        typer = self._component("type")
        value = raw_input if raw_input is not None else text
        params = {"selector": selector, "raw_input": value, "text": value, **kwargs}
        if clear_before is not None:
            params["clear_before"] = clear_before
        if hasattr(typer, "type_text"):
            return self._call_component(typer.type_text, params, action="type")
        if hasattr(typer, "perform"):
            return self._call_component(typer.perform, {"action": "type", **params}, action="type")
        raise BrowserTaskError("Type component does not expose type_text/perform")

    def clear_text(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
        typer = self._component("type")
        if hasattr(typer, "clear_text"):
            return self._call_component(typer.clear_text, {"selector": selector, **kwargs}, action="clear")
        if hasattr(typer, "perform"):
            return self._call_component(typer.perform, {"action": "clear", "selector": selector, **kwargs}, action="clear")
        return self.do_type(selector=selector, text="", clear_before=True, **kwargs)

    def press_key(self, selector: Optional[str] = None, key: str = "ENTER", **kwargs: Any) -> Dict[str, Any]:
        typer = self._component("type")
        if hasattr(typer, "press_key"):
            return self._call_component(typer.press_key, {"selector": selector, "key": key, **kwargs}, action="press_key")
        if hasattr(typer, "perform"):
            return self._call_component(typer.perform, {"action": "press_key", "selector": selector, "key": key, **kwargs}, action="press_key")
        raise BrowserTaskError("Type component does not expose press_key/perform")

    def scroll_element(self, mode: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        scroller = self._component("scroll")
        resolved_mode = (mode or kwargs.pop("mode", None) or self.options.default_scroll_mode or "by").lower()
        params = {"mode": resolved_mode, **kwargs}
        if hasattr(scroller, "perform"):
            return self._call_component(scroller.perform, params, action="scroll")
        if resolved_mode == "to":
            return self._call_component(scroller.scroll_to, {"x": kwargs.get("x", 0), "y": kwargs.get("y", 0), "smooth": kwargs.get("smooth", False)}, action="scroll")
        if resolved_mode == "element":
            return self._call_component(scroller.scroll_element_into_view, {"selector": kwargs.get("selector", ""), "position": kwargs.get("position", "center")}, action="scroll")
        if resolved_mode == "direction":
            return self._call_component(scroller.scroll_direction, {"direction": kwargs.get("direction", "down"), "amount": kwargs.get("amount", 200)}, action="scroll")
        return self._call_component(scroller.scroll_by, {"dx": kwargs.get("dx", 0), "dy": kwargs.get("dy", 200), "smooth": kwargs.get("smooth", False)}, action="scroll")

    def copy(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
        clipboard = self._component("clipboard")
        if hasattr(clipboard, "copy"):
            return self._call_component(clipboard.copy, {"selector": selector, **kwargs}, action="copy")
        if hasattr(clipboard, "perform"):
            return self._call_component(clipboard.perform, {"action": "copy", "selector": selector, **kwargs}, action="copy")
        raise BrowserTaskError("Clipboard component does not expose copy/perform")

    def cut(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
        clipboard = self._component("clipboard")
        if hasattr(clipboard, "cut"):
            return self._call_component(clipboard.cut, {"selector": selector, **kwargs}, action="cut")
        if hasattr(clipboard, "perform"):
            return self._call_component(clipboard.perform, {"action": "cut", "selector": selector, **kwargs}, action="cut")
        raise BrowserTaskError("Clipboard component does not expose cut/perform")

    def paste(self, selector: str, text: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        clipboard = self._component("clipboard")
        params = {"selector": selector, **kwargs}
        if text is not None:
            params["text"] = text
        if hasattr(clipboard, "paste"):
            return self._call_component(clipboard.paste, params, action="paste")
        if hasattr(clipboard, "perform"):
            return self._call_component(clipboard.perform, {"action": "paste", **params}, action="paste")
        raise BrowserTaskError("Clipboard component does not expose paste/perform")

    def drag_and_drop(
        self,
        source_selector: str,
        target_selector: Optional[str] = None,
        target_offset: Optional[Tuple[int, int]] = None,
        offset_x: Optional[int] = None,
        offset_y: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        dragger = self._component("drag_and_drop")
        params: Dict[str, Any] = {"source_selector": source_selector, **kwargs}
        if target_selector is not None:
            params["target_selector"] = target_selector
        if target_offset is not None:
            params["target_offset"] = target_offset
        if offset_x is not None:
            params["offset_x"] = offset_x
        if offset_y is not None:
            params["offset_y"] = offset_y
        for method_name in ("drag_and_drop", "drag_to_element", "_perform_drag_and_drop"):
            if hasattr(dragger, method_name):
                return self._call_component(getattr(dragger, method_name), params, action="drag_and_drop")
        raise BrowserTaskError("Drag-and-drop component does not expose a drag method")

    def screenshot(self, *, format: Optional[str] = None, path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Capture a screenshot from the active driver."""

        start_ms = monotonic_ms()
        driver = self.require_driver(action="screenshot")
        screenshot_format = (format or self.options.screenshot_format or DEFAULT_SCREENSHOT_FORMAT).lower()
        try:
            data: Dict[str, Any]
            if path:
                if not hasattr(driver, "save_screenshot"):
                    raise BrowserTaskError("Driver does not support save_screenshot", context={"path": path})
                ok = driver.save_screenshot(path)
                data = {"path": path, "saved": bool(ok), "format": "file"}
            elif screenshot_format == "png" and hasattr(driver, "get_screenshot_as_png"):
                data = {"bytes": driver.get_screenshot_as_png(), "format": "png"}
            else:
                if not hasattr(driver, "get_screenshot_as_base64"):
                    raise BrowserTaskError("Driver does not support get_screenshot_as_base64")
                data = {"screenshot_b64": driver.get_screenshot_as_base64(), "format": "base64"}
            return success_result(action="screenshot", message="Screenshot captured", data=data, duration_ms=elapsed_ms(start_ms))
        except Exception as exc:
            return self._error(exc, action="screenshot")

    def extract_page(
        self,
        *,
        preview_only: Optional[bool] = None,
        include_html: bool = False,
        include_screenshot: bool = False,
        max_text_chars: Optional[int] = None,
        max_html_chars: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Extract URL, title, body text, and optional HTML from the current page."""

        start_ms = monotonic_ms()
        driver = self.require_driver(action="extract_page")
        try:
            text_limit = self.options.max_page_text_chars if max_text_chars is None else max_text_chars
            html_limit = self.options.max_html_chars if max_html_chars is None else max_html_chars
            preview = self.options.default_extract_preview_only if preview_only is None else preview_only
            current_url = safe_call(lambda: driver.current_url, default="") or ""
            title = safe_call(lambda: driver.title, default="") or ""
            ready_state = safe_call(lambda: driver.execute_script("return document.readyState"), default=None)
            body_text = ""
            html = ""
            try:
                body = driver.find_element("tag name", "body")
                body_text = element_text(body, max_length=text_limit or DEFAULT_PAGE_TEXT_LIMIT)
            except Exception:
                body_text = safe_call(lambda: driver.execute_script("return document.body ? document.body.innerText : ''"), default="") or ""
            if preview:
                body_text = truncate_text(body_text, min(text_limit or 1_000, 1_000))
            else:
                body_text = truncate_text(body_text, text_limit)
            if include_html:
                html = safe_call(lambda: driver.page_source, default="") or ""
                html = truncate_text(html, html_limit)
            data: Dict[str, Any] = {
                "url": current_url,
                "title": title,
                "ready_state": ready_state,
                "text": body_text,
                "text_length": len(body_text),
                "captured_at": utc_now_iso(),
                "fingerprint": fingerprint_text(f"{current_url}\n{title}\n{body_text}"),
            }
            if include_html:
                data["html"] = html
                data["html_length"] = len(html)
            if include_screenshot:
                data["screenshot"] = self._safe_screenshot_base64()
            result = success_result(action="extract_page", message="Page content extracted", data=data, duration_ms=elapsed_ms(start_ms))
            if self.options.record_page_extracts_to_memory:
                self._remember_page_extract(result)
            return result
        except Exception as exc:
            return self._error(exc, action="extract_page")

    def page_state(self, **kwargs: Any) -> Dict[str, Any]:
        return success_result(action="page_state", message="Page state returned", data=self._safe_page_state())

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _component(self, name: str) -> Any:
        canonical = self.resolve_name(name)
        if canonical not in self.components:
            if self.driver is None:
                self.require_driver(action=canonical)
            self._build_components()
        if canonical not in self.components:
            raise BrowserTaskError("Browser component is unavailable", context={"component": canonical})
        return self.components[canonical]

    def _call_component(self, method: Optional[BrowserCallable], params: Mapping[str, Any], *, action: str) -> Dict[str, Any]:
        if method is None:
            raise BrowserTaskError("Browser component method is unavailable", context={"action": action})
        try:
            result = method(**filter_kwargs_for_callable(method, params))
            return normalize_result(result, action=action, default_message=f"{action} completed")
        except Exception as exc:
            return self._error(exc, action=action, context={"params": params})

    def _safe_page_state(self) -> Dict[str, Any]:
        if self.driver is None:
            return {"driver_attached": False}
        driver = self.driver
        return prune_none(
            {
                "driver_attached": True,
                "url": safe_call(lambda: driver.current_url, default=""),
                "title": safe_call(lambda: driver.title, default=""),
                "ready_state": safe_call(lambda: driver.execute_script("return document.readyState"), default=None),
                "captured_at": utc_now_iso(),
            }
        )

    def _safe_screenshot_base64(self) -> Optional[str]:
        if self.driver is None or not hasattr(self.driver, "get_screenshot_as_base64"):
            return None
        return safe_call(lambda: self.driver.get_screenshot_as_base64(), default=None) # pyright: ignore[reportOptionalMemberAccess]

    def _record_execution(self, call: BrowserFunctionCall, result: Mapping[str, Any], *, spec: Optional[BrowserFunctionSpec], duration_ms: float) -> None:
        status = function_result_status(result)
        message = str(result.get("message", "")) if isinstance(result, Mapping) else ""
        entry = BrowserFunctionExecution(
            call=call,
            status=status,
            message=message,
            duration_ms=duration_ms,
            result_fingerprint=stable_hash(result, length=20),
        )
        if self.options.record_history:
            self.history.append(entry)
        should_record = self.memory is not None and ((status == "success" and self.options.record_successes_to_memory) or (status == "error" and self.options.record_errors_to_memory))
        if should_record:
            self._remember_action(call, result, spec=spec, execution=entry)

    def _remember_action(self, call: BrowserFunctionCall, result: Mapping[str, Any], *, spec: Optional[BrowserFunctionSpec], execution: BrowserFunctionExecution) -> None:
        if self.memory is None:
            return
        try:
            if hasattr(self.memory, "remember_action"):
                self.memory.remember_action(
                    action=call.canonical_name,
                    result=dict(result),
                    request={
                        "call": call.to_dict(),
                        "selector": call.params.get("selector") or call.params.get("source_selector"),
                        "execution": execution.to_dict(),
                        "function": spec.to_dict() if spec else None,
                    },
                )
            else:
                key = f"{call.canonical_name}:{call.correlation_id}"
                self.memory.put(
                    key,
                    dict(result),
                    namespace=self.options.memory_namespace,
                    kind=spec.memory_kind if spec else "action_result",
                    action=call.canonical_name,
                    metadata={"call": call.to_dict(), "execution": execution.to_dict()},
                )
        except Exception as exc:
            logger.warning("Failed to record browser function result in memory: %s", exc)

    def _remember_workflow(self, result: Mapping[str, Any]) -> None:
        if self.memory is None:
            return
        try:
            if hasattr(self.memory, "remember_workflow_execution"):
                self.memory.remember_workflow_execution(workflow_result=dict(result))
            else:
                self.memory.put(new_correlation_id("workflow"), dict(result), namespace=MEMORY_WORKFLOW_NAMESPACE, kind="workflow_result")
        except Exception as exc:
            logger.warning("Failed to record workflow result in memory: %s", exc)

    def _remember_page_extract(self, result: Mapping[str, Any]) -> None:
        if self.memory is None:
            return
        try:
            data = dict(result.get("data") or {}) if isinstance(result, Mapping) else {}
            if hasattr(self.memory, "remember_page"):
                self.memory.remember_page(page=data, metadata=data)
            else:
                self.memory.put(data.get("url") or new_correlation_id("page"), data, namespace=MEMORY_PAGE_NAMESPACE, kind="page_content", url=data.get("url"))
        except Exception as exc:
            logger.warning("Failed to record page extract in memory: %s", exc)

    def close(self) -> Dict[str, Any]:
        """Close the underlying driver when it exposes quit()."""

        start_ms = monotonic_ms()
        if self.driver is not None and hasattr(self.driver, "quit"):
            self.driver.quit()
        self.driver = None
        self.components.clear()
        return success_result(action="close", message="BrowserFunctions closed", duration_ms=elapsed_ms(start_ms))


# ---------------------------------------------------------------------------
# Self-test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Browser Functions ===\n")
    printer.status("TEST", "Browser Functions initialized", "info")

    class _FakeElement:
        def __init__(self, text: str = "Fake page body") -> None:
            self.text = text
            self.tag_name = "body"

        def get_attribute(self, name: str) -> str:
            if name == "outerHTML":
                return f"<body>{self.text}</body>"
            return ""

        def is_displayed(self) -> bool:
            return True

        def is_enabled(self) -> bool:
            return True

    class _FakeDriver:
        def __init__(self) -> None:
            self.current_url = "about:blank"
            self.title = "Blank"
            self.page_source = "<html><body>Fake page body</body></html>"
            self.saved_screenshots: List[str] = []

        def get(self, url: str) -> None:
            self.current_url = url
            self.title = f"Page for {url}"
            self.page_source = f"<html><body>Loaded {url}</body></html>"

        def back(self) -> None:
            self.current_url = "https://example.test/back"

        def forward(self) -> None:
            self.current_url = "https://example.test/forward"

        def refresh(self) -> None:
            self.title = f"{self.title} refreshed"

        def execute_script(self, script: str, *args: Any) -> Any:
            if "document.readyState" in script:
                return "complete"
            if "document.body" in script and "innerText" in script:
                return "Fake page body"
            return None

        def find_element(self, by: str, value: str) -> _FakeElement:
            return _FakeElement("Fake page body")

        def get_screenshot_as_base64(self) -> str:
            return "ZmFrZS1zY3JlZW5zaG90"

        def save_screenshot(self, path: str) -> bool:
            self.saved_screenshots.append(path)
            return True

        def quit(self) -> None:
            self.current_url = "closed"

    class _FakeClick:
        def click(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": f"Clicked {selector}", "selector": selector}

    class _FakeType:
        def type_text(self, selector: str, raw_input: str, clear_before: bool = True, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": "Typed text", "selector": selector, "text": raw_input, "clear_before": clear_before}

        def clear_text(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": "Cleared text", "selector": selector}

        def press_key(self, selector: Optional[str] = None, key: str = "ENTER", **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": "Pressed key", "selector": selector, "key": key}

    class _FakeScroll:
        def perform(self, mode: str = "by", **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": "Scrolled", "mode": mode, "params": kwargs}

    class _FakeClipboard:
        def copy(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": "Copied", "selector": selector, "text": "copied"}

        def cut(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": "Cut", "selector": selector, "text": "cut"}

        def paste(self, selector: str, text: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": "Pasted", "selector": selector, "text": text}

    class _FakeDrag:
        def drag_and_drop(self, source_selector: str, target_selector: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "message": "Dragged", "source_selector": source_selector, "target_selector": target_selector}

    browser_functions = BrowserFunctions(driver=_FakeDriver(), memory=False)
    browser_functions.replace_component("click", _FakeClick())
    browser_functions.replace_component("type", _FakeType())
    browser_functions.replace_component("scroll", _FakeScroll())
    browser_functions.replace_component("clipboard", _FakeClipboard())
    browser_functions.replace_component("drag_and_drop", _FakeDrag())

    list_result = browser_functions.list_functions()
    assert list_result["status"] == "success"
    assert "navigate" in {item["name"] for item in list_result["data"]["functions"]}

    nav_result = browser_functions.execute("navigate", {"url": "https://example.test"})
    assert nav_result["status"] == "success", nav_result

    click_result = browser_functions.execute("click_element", {"selector": "#submit"})
    assert click_result["status"] == "success", click_result

    type_result = browser_functions.execute("do_type", {"selector": "#search", "text": "browser functions"})
    assert type_result["status"] == "success", type_result

    scroll_result = browser_functions.execute("scroll", {"mode": "direction", "direction": "down", "amount": 250})
    assert scroll_result["status"] == "success", scroll_result

    copy_result = browser_functions.execute("copy", {"selector": "#copy"})
    assert copy_result["status"] == "success", copy_result

    paste_result = browser_functions.execute("paste", {"selector": "#paste", "text": "hello"})
    assert paste_result["status"] == "success", paste_result

    drag_result = browser_functions.execute("drag_element", {"source_selector": "#card", "target_selector": "#lane"})
    assert drag_result["status"] == "success", drag_result

    screenshot_result = browser_functions.execute("screenshot")
    assert screenshot_result["status"] == "success", screenshot_result

    extract_result = browser_functions.execute("extract_page", {"include_html": True})
    assert extract_result["status"] == "success", extract_result

    workflow_result = browser_functions.execute_workflow(
        [
            {"action": "navigate", "params": {"url": "https://example.test/workflow"}},
            {"action": "click", "params": {"selector": "#ok"}},
            {"action": "type", "params": {"selector": "#field", "text": "ok"}},
        ]
    )
    assert workflow_result["status"] == "success", workflow_result

    history_result = browser_functions.execute("history")
    assert history_result["status"] == "success", history_result
    assert len(history_result["data"]["function_history"]) >= 1

    browser_functions.close()
    print("\n=== Test ran successfully ===\n")
