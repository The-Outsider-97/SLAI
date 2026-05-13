from __future__ import annotations

__version__ = "2.2.0"

"""
Production-grade Browser Agent facade.

The BrowserAgent owns the browser session and agent-facing orchestration
contract. It does not own concrete click/type/scroll/navigation/clipboard logic;
those actions are delegated through BrowserFunctions so the agent stays a clean
runtime facade instead of duplicating lower-level modules.

Responsibilities
----------------
- Own Selenium browser lifecycle and driver attachment/restart/close behavior.
- Load browser_agent runtime policy from agents_config.yaml via main_config_loader.
- Compose BrowserFunctions, SecurityFeatures, ContentHandling, Utilities, and
  WorkFlow around one browser session.
- Expose stable agent methods such as navigate, search, click, type, scroll,
  copy, cut, paste, extract_page_content, execute_workflow, and perform_task.
- Use browser error/result helpers for validation, redaction, result envelopes,
  security decisions, and failure reporting.
- Keep workflow compilation separate from workflow execution: WorkFlow compiles,
  BrowserFunctions executes.
- Publish compact browser-agent telemetry to shared memory when configured.

Design boundaries
-----------------
BrowserAgent does not call DoClick, DoType, DoScroll, DoNavigate,
DoCopyCutPaste, or DoDragAndDrop directly. BrowserFunctions owns that dispatch.
BrowserAgent may inspect the driver only for lifecycle, security scans, and
helper-driven page/search-result inspection.

Local imports are intentionally direct. They are not wrapped in try/except so
packaging or path problems fail clearly during development and deployment.
"""

import asyncio
import time as time_module

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section, load_global_config
from .base.utils.config_contract import assert_valid_config_contract
from .browser.browser_functions import BrowserFunctions
from .browser.content import ContentHandling
from .browser.security import SecurityFeatures, exponential_backoff
from .browser.utilities import Utilities
from .browser.workflow import WorkFlow
from .browser.utils.browser_errors import *
from .browser.utils.Browser_helpers import *
from .browser.utils.browser_driver import BrowserDriver
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Browser Agent")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BROWSER_AGENT_SCHEMA_VERSION = "3.0"
DEFAULT_SEARCH_ENGINE = "https://www.google.com"
DEFAULT_SEARCH_BOX_SELECTOR = "input[name='q']"
DEFAULT_RESULT_SELECTORS: Tuple[str, ...] = (
    "h3 a",
    ".yuRUbf a",
    ".g a",
    "a[href^='http']",
)
AGENT_SHARED_NAMESPACE = "browser_agent"

BROWSER_AGENT_TASK_ALIASES: Dict[str, str] = {
    "url": "navigate",
    "open_url": "navigate",
    "go_to_url": "navigate",
    "navigate_to": "navigate",
    "google": "search",
    "query": "search",
    "click_element": "click",
    "type_text": "type",
    "enter_text": "type",
    "input_text": "type",
    "scroll_element": "scroll",
    "extract": "extract_page",
    "extract_page": "extract_page",
    "extract_page_content": "extract_page",
    "get_dom": "extract_page",
    "workflow": "workflow",
    "execute_workflow": "workflow",
    "run_workflow": "workflow",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BrowserAgentOptions:
    """Config-backed BrowserAgent orchestration policy.

    Driver creation/configuration values live in browser_config.yaml under
    browser_driver and are handled by BrowserDriver. This options object keeps
    only agent-level routing, retry, security, workflow, content, diagnostics,
    and shared-memory policy.
    """

    enabled: bool = True
    auto_start_browser: bool = True

    max_retries: int = 3
    default_wait: float = 0.0
    retry_sleep_enabled: bool = True
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_multiplier: float = 2.0
    retry_jitter: float = 0.0

    default_search_engine: str = DEFAULT_SEARCH_ENGINE
    search_box_selector: str = DEFAULT_SEARCH_BOX_SELECTOR
    search_submit_strategy: str = "type_enter"
    search_results_wait_seconds: float = 10.0
    max_search_results: int = 10
    search_result_selectors: Tuple[str, ...] = DEFAULT_RESULT_SELECTORS
    postprocess_search_results: bool = True

    guard_navigation: bool = True
    guard_actions: bool = True
    scan_after_actions: bool = True
    block_on_security_decision: bool = True
    attach_security_reports: bool = True

    compile_workflows: bool = True
    dry_run_workflows: bool = False
    workflow_stop_on_error: bool = True

    include_page_preview_after_navigation: bool = True
    navigation_preview_only: bool = True
    extract_preview_only: bool = False
    include_html_on_extract: bool = False
    include_screenshot_on_extract: bool = False

    publish_to_shared_memory: bool = True
    shared_result_key: str = "browser_agent:last_result"
    shared_history_key: str = "browser_agent:history"
    shared_health_key: str = "browser_agent:health"
    shared_event_topic: str = "agent.browser.events"
    max_history: int = 250

    include_traceback_on_error: bool = False
    include_function_metadata: bool = True
    include_page_state_after_action: bool = False
    screenshot_on_error: bool = False

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "BrowserAgentOptions":
        cfg = dict(config or {})
        driver_cfg = dict(cfg.get("driver") or {})  # retained only for backward-compatible auto_start override
        execution_cfg = dict(cfg.get("execution") or {})
        retry_cfg = dict(cfg.get("retry") or {})
        search_cfg = dict(cfg.get("search") or {})
        security_cfg = dict(cfg.get("security") or {})
        workflow_cfg = dict(cfg.get("workflow") or {})
        content_cfg = dict(cfg.get("content") or {})
        shared_cfg = dict(cfg.get("shared_memory") or {})
        diagnostics_cfg = dict(cfg.get("diagnostics") or {})

        result_selectors = ensure_list(search_cfg.get("result_selectors", cfg.get("search_result_selectors", DEFAULT_RESULT_SELECTORS)))
        if not result_selectors:
            result_selectors = list(DEFAULT_RESULT_SELECTORS)

        return cls(
            enabled=coerce_bool(cfg.get("enabled", True), default=True),
            auto_start_browser=coerce_bool(driver_cfg.get("auto_start", cfg.get("auto_start_browser", True)), default=True),
            max_retries=coerce_int(execution_cfg.get("max_retries", cfg.get("max_retries", 3)), default=3, minimum=0, maximum=100),
            default_wait=coerce_float(execution_cfg.get("default_wait", cfg.get("default_wait", 0.0)), default=0.0, minimum=0.0),
            retry_sleep_enabled=coerce_bool(retry_cfg.get("sleep_enabled", cfg.get("retry_sleep_enabled", True)), default=True),
            retry_base_delay=coerce_float(retry_cfg.get("base_delay", cfg.get("retry_base_delay", 1.0)), default=1.0, minimum=0.0),
            retry_max_delay=coerce_float(retry_cfg.get("max_delay", cfg.get("retry_max_delay", 60.0)), default=60.0, minimum=0.0),
            retry_multiplier=coerce_float(retry_cfg.get("multiplier", cfg.get("retry_multiplier", 2.0)), default=2.0, minimum=1.0),
            retry_jitter=coerce_float(retry_cfg.get("jitter", cfg.get("retry_jitter", 0.0)), default=0.0, minimum=0.0),
            default_search_engine=str(search_cfg.get("default_engine", cfg.get("default_search_engine", DEFAULT_SEARCH_ENGINE)) or DEFAULT_SEARCH_ENGINE),
            search_box_selector=str(search_cfg.get("box_selector", cfg.get("search_box_selector", DEFAULT_SEARCH_BOX_SELECTOR)) or DEFAULT_SEARCH_BOX_SELECTOR),
            search_submit_strategy=str(search_cfg.get("submit_strategy", cfg.get("search_submit_strategy", "type_enter")) or "type_enter"),
            search_results_wait_seconds=coerce_float(search_cfg.get("results_wait_seconds", cfg.get("search_results_wait_seconds", 10.0)), default=10.0, minimum=0.0),
            max_search_results=coerce_int(search_cfg.get("max_results", cfg.get("max_search_results", 10)), default=10, minimum=1, maximum=100),
            search_result_selectors=tuple(str(item) for item in result_selectors if str(item).strip()),
            postprocess_search_results=coerce_bool(search_cfg.get("postprocess_results", cfg.get("postprocess_search_results", True)), default=True),
            guard_navigation=coerce_bool(security_cfg.get("guard_navigation", cfg.get("guard_navigation", True)), default=True),
            guard_actions=coerce_bool(security_cfg.get("guard_actions", cfg.get("guard_actions", True)), default=True),
            scan_after_actions=coerce_bool(security_cfg.get("scan_after_actions", cfg.get("scan_after_actions", True)), default=True),
            block_on_security_decision=coerce_bool(security_cfg.get("block_on_security_decision", cfg.get("block_on_security_decision", True)), default=True),
            attach_security_reports=coerce_bool(security_cfg.get("attach_reports", cfg.get("attach_security_reports", True)), default=True),
            compile_workflows=coerce_bool(workflow_cfg.get("compile_before_execute", cfg.get("compile_workflows", True)), default=True),
            dry_run_workflows=coerce_bool(workflow_cfg.get("dry_run", cfg.get("dry_run_workflows", False)), default=False),
            workflow_stop_on_error=coerce_bool(workflow_cfg.get("stop_on_error", cfg.get("workflow_stop_on_error", True)), default=True),
            include_page_preview_after_navigation=coerce_bool(content_cfg.get("include_page_preview_after_navigation", cfg.get("include_page_preview_after_navigation", True)), default=True),
            navigation_preview_only=coerce_bool(content_cfg.get("navigation_preview_only", cfg.get("navigation_preview_only", True)), default=True),
            extract_preview_only=coerce_bool(content_cfg.get("extract_preview_only", cfg.get("extract_preview_only", False)), default=False),
            include_html_on_extract=coerce_bool(content_cfg.get("include_html_on_extract", cfg.get("include_html_on_extract", False)), default=False),
            include_screenshot_on_extract=coerce_bool(content_cfg.get("include_screenshot_on_extract", cfg.get("include_screenshot_on_extract", False)), default=False),
            publish_to_shared_memory=coerce_bool(shared_cfg.get("enabled", cfg.get("publish_to_shared_memory", True)), default=True),
            shared_result_key=str(shared_cfg.get("last_result_key", cfg.get("shared_result_key", "browser_agent:last_result")) or "browser_agent:last_result"),
            shared_history_key=str(shared_cfg.get("history_key", cfg.get("shared_history_key", "browser_agent:history")) or "browser_agent:history"),
            shared_health_key=str(shared_cfg.get("health_key", cfg.get("shared_health_key", "browser_agent:health")) or "browser_agent:health"),
            shared_event_topic=str(shared_cfg.get("event_topic", cfg.get("shared_event_topic", "agent.browser.events")) or "agent.browser.events"),
            max_history=coerce_int(shared_cfg.get("max_history", cfg.get("max_history", 250)), default=250, minimum=1),
            include_traceback_on_error=coerce_bool(diagnostics_cfg.get("include_traceback_on_error", cfg.get("include_traceback_on_error", False)), default=False),
            include_function_metadata=coerce_bool(diagnostics_cfg.get("include_function_metadata", cfg.get("include_function_metadata", True)), default=True),
            include_page_state_after_action=coerce_bool(diagnostics_cfg.get("include_page_state_after_action", cfg.get("include_page_state_after_action", False)), default=False),
            screenshot_on_error=coerce_bool(diagnostics_cfg.get("screenshot_on_error", cfg.get("screenshot_on_error", False)), default=False),
        )


@dataclass(frozen=True)
class BrowserAgentExecution:
    """Compact audit record for one BrowserAgent operation."""

    operation: str
    status: str
    message: str
    duration_ms: float
    attempts: int
    correlation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# BrowserAgent
# ---------------------------------------------------------------------------
class BrowserAgent(BaseAgent):
    """High-level browser orchestration agent.

    BrowserAgent owns browser lifecycle and agent-facing policy. Concrete
    browser operations are routed through ``BrowserFunctions``.
    """

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Optional[Mapping[str, Any]] = None, driver: Any = None, *,
        browser_driver: Optional[BrowserDriver] = None,
        browser_functions: Optional[BrowserFunctions] = None,
        content_handler: Optional[ContentHandling] = None,
        security: Optional[SecurityFeatures] = None,
        workflow: Optional[WorkFlow] = None,
        utilities: Optional[Utilities] = None,
        auto_start: Optional[bool] = None,
    ) -> None:
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.browser_agent_config: Dict[str, Any] = dict(get_config_section("browser_agent") or {})
        if config:
            self.browser_agent_config.update(dict(config))
        assert_valid_config_contract(
            global_config=self.config,
            agent_key="browser_agent",
            agent_config=self.browser_agent_config,
            logger=logger,
            require_global_keys=False,
            require_agent_section=False,
            warn_unknown_global_keys=False,
        )
        if auto_start is not None:
            self.browser_agent_config.setdefault("driver", {})
            if isinstance(self.browser_agent_config["driver"], dict):
                self.browser_agent_config["driver"]["auto_start"] = auto_start
            else:
                self.browser_agent_config["auto_start_browser"] = auto_start

        self.options = BrowserAgentOptions.from_config(self.browser_agent_config)
        self.enabled = self.options.enabled
        self.name = "BrowserAgent"
        self.execution_history: Deque[Dict[str, Any]] = deque(maxlen=self.options.max_history)

        # BrowserDriver is the single owner of Selenium lifecycle concerns:
        # startup, attachment, timeout/window configuration, health checks,
        # restart, detach, and close. BrowserAgent only wires the active driver
        # into orchestration components and never creates Selenium directly.
        if browser_driver is not None:
            self.browser_driver = browser_driver
            if driver is not None and driver is not browser_driver.driver:
                attach_result = self.browser_driver.attach(driver, owns_driver=False)
                if attach_result.get("status") != "success":
                    raise BrowserDriverStartupError(
                        "Failed to attach provided Selenium driver to BrowserDriver",
                        context={"attach_result": attach_result},
                    )
        elif isinstance(driver, BrowserDriver):
            # Backward-compatible guard: if someone passes BrowserDriver as `driver`,
            # treat it as the lifecycle manager, not as the Selenium WebDriver.
            self.browser_driver = driver
        else:
            self.browser_driver = BrowserDriver(
                driver=driver,
                config=self.browser_agent_config.get("browser_driver_overrides"),
                auto_start=(driver is None and self.options.auto_start_browser),
                owns_driver=(driver is None),
            )
        
        if not self.browser_driver.has_driver and self.options.auto_start_browser:
            start_result = self.browser_driver.start()
            if start_result.get("status") != "success":
                raise BrowserDriverStartupError(
                    "Failed to start browser driver",
                    context={"start_result": start_result},
                )
            time_module.sleep(0.5)  # brief pause to allow driver to initialize before attaching
        
        self.driver = self.browser_driver.driver
        self._owns_driver = self.browser_driver.owns_driver
        self._closed = not self.browser_driver.has_driver

        self.browser_functions = browser_functions or self._build_browser_functions(self.driver)
        self.content = content_handler or ContentHandling()
        self.security = security or SecurityFeatures(driver=self.driver)
        self.workflow = workflow or WorkFlow(config=self.browser_agent_config.get("workflow_config"))
        self.utilities = utilities or Utilities()

        if self.driver is not None:
            self._attach_driver_to_components(self.driver)

        self._publish_health("initialized")
        logger.info("BrowserAgent initialized with BrowserDriver lifecycle and BrowserFunctions orchestration.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _build_browser_functions(self, driver: Any = None) -> BrowserFunctions:
        return BrowserFunctions(driver=driver, config=self.browser_agent_config.get("browser_functions_overrides"))

    def _attach_driver_to_components(self, driver: Any) -> None:
        self.driver = driver
        if hasattr(self.browser_functions, "attach_driver"):
            self.browser_functions.attach_driver(driver)
        elif hasattr(self.browser_functions, "set_driver"):
            self.browser_functions.set_driver(driver)
        else:
            setattr(self.browser_functions, "driver", driver)
        if hasattr(self.security, "driver"):
            self.security.driver = driver

    def attach_driver(self, driver: Any, *, owns_driver: bool = False) -> Dict[str, Any]:
        result = self.browser_driver.attach(driver, owns_driver=owns_driver)
        if result.get("status") == "success":
            self.driver = self.browser_driver.driver
            self._owns_driver = self.browser_driver.owns_driver
            self._closed = False
            self._attach_driver_to_components(self.driver)
            self._publish_health("driver_attached")
        self._record_agent_result("attach_driver", result)
        return result

    def detach_driver(self) -> Dict[str, Any]:
        result = self.browser_driver.detach_result()
        self.driver = None
        self._owns_driver = False
        self._closed = True
        if hasattr(self.browser_functions, "detach_driver"):
            try:
                self.browser_functions.detach_driver()
            except Exception as exc:
                logger.debug("BrowserFunctions detach skipped: %s", exc)
        if hasattr(self.security, "driver"):
            self.security.driver = None
        self._publish_health("driver_detached")
        self._record_agent_result("detach_driver", result)
        return result

    def start(self) -> Dict[str, Any]:
        result = self.browser_driver.start()
        if result.get("status") == "success":
            self.driver = self.browser_driver.driver
            self._owns_driver = self.browser_driver.owns_driver
            self._closed = False
            self._attach_driver_to_components(self.driver)
            self._publish_health("started")
        self._record_agent_result("start_browser", result)
        return result

    def restart(self) -> Dict[str, Any]:
        result = self.browser_driver.restart()
        if result.get("status") == "success":
            self.driver = self.browser_driver.driver
            self._owns_driver = self.browser_driver.owns_driver
            self._closed = False
            self._attach_driver_to_components(self.driver)
            self._publish_health("restarted")
        self._record_agent_result("restart_browser", result)
        return result

    def close(self) -> Dict[str, Any]:
        try:
            if hasattr(self.browser_functions, "close"):
                self.browser_functions.close()
            result = self.browser_driver.close()
            self.driver = None
            self._owns_driver = False
            self._closed = True
            if hasattr(self.security, "driver"):
                self.security.driver = None
            self._publish_health("closed")
            self._record_agent_result("close_browser", result)
            return result
        except Exception as exc:
            return self._error(exc, action="close_browser")

    def __enter__(self) -> "BrowserAgent":
        if self.driver is None:
            self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        driver_manager = getattr(self, "browser_driver", None)
        should_close = bool(getattr(getattr(driver_manager, "options", None), "close_driver_on_del", False))
        if should_close:
            try:
                self.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Function execution through BrowserFunctions
    # ------------------------------------------------------------------
    def _execute_function(self, function_name: str, params: Optional[Mapping[str, Any]] = None, *, retries: Optional[int] = None) -> Dict[str, Any]:
        params = dict(params or {})
        attempts = self.options.max_retries if retries is None else max(0, int(retries))
        start_ms = monotonic_ms()
        correlation_id = params.get("correlation_id") or new_correlation_id(f"ba-{function_name}")
        params.setdefault("correlation_id", correlation_id)
        last_error: Optional[BaseException] = None
        restart_needed = False
    
        for attempt in range(attempts + 1):
            # Restart the driver before the attempt if flagged
            if restart_needed:
                logger.info("Restarting browser driver before retry attempt %s", attempt)
                try:
                    self.restart()
                except Exception as restart_exc:
                    logger.error("Failed to restart browser driver: %s", restart_exc)
                    # Continue anyway – the next attempt may still fail
                finally:
                    restart_needed = False
    
            try:
                if self.options.guard_actions:
                    guard = self._guard_action(function_name, params)
                    if guard.get("status") == "error":
                        return guard
    
                result = self.browser_functions.execute(function_name, params=params)
                result = normalize_result(result, action=function_name, default_message=f"{function_name} completed")
                result.setdefault("correlation_id", correlation_id)
                result.setdefault("attempts", attempt + 1)
                if self.options.attach_security_reports and self.options.scan_after_actions:
                    report = self._scan_current_page(action=function_name)
                    if report:
                        result.setdefault("security", report)
                        if self.options.block_on_security_decision and not self._security_allows(report):
                            return self._security_block_result(function_name, report, correlation_id=correlation_id, duration_ms=elapsed_ms(start_ms))
                self._record_agent_result(function_name, result, attempts=attempt + 1, duration_ms=elapsed_ms(start_ms))
                return result
    
            except Exception as exc:
                last_error = exc
                # Detect pickling errors – they indicate driver state is corrupted
                if "cannot pickle '_thread.lock'" in str(exc):
                    logger.warning("Pickling error detected in browser function %s, will restart driver before next attempt", function_name)
                    restart_needed = True
                else:
                    logger.warning("Browser function %s failed on attempt %s/%s: %s", function_name, attempt + 1, attempts + 1, exc)
    
                if attempt >= attempts:
                    # Exhausted all retries – build final error result
                    # For pickling errors, remove the cause to avoid serialising the lock
                    if restart_needed:
                        # Remove the cause from the exception context to prevent pickling lock
                        # We create a new exception with the same message but empty cause
                        clean_exc = BrowserTaskError(
                            f"Browser function '{function_name}' failed due to driver pickling error after {attempts + 1} attempts",
                            context={"original_error": str(exc)},
                        )
                        result = self._error(clean_exc, action=function_name, context={"attempts": attempt + 1, "params": params}, duration_ms=elapsed_ms(start_ms))
                    else:
                        result = self._error(exc, action=function_name, context={"attempts": attempt + 1, "params": params}, duration_ms=elapsed_ms(start_ms))
                    self._record_agent_result(function_name, result, attempts=attempt + 1, duration_ms=elapsed_ms(start_ms))
                    return result
    
                self._sleep_before_retry(attempt)
    
        return self._error(last_error or RuntimeError("Unknown browser function failure"), action=function_name, duration_ms=elapsed_ms(start_ms))

    async def _async_execute_function(self, function_name: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        if hasattr(self.browser_functions, "async_execute"):
            try:
                result = await self.browser_functions.async_execute(function_name, params=params)
                return normalize_result(result, action=function_name)
            except Exception as exc:
                return self._error(exc, action=function_name)
        return await asyncio.to_thread(self._execute_function, function_name, params)

    # ------------------------------------------------------------------
    # Public browser operations
    # ------------------------------------------------------------------
    def navigate(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        guard = self._guard_navigation(url)
        if guard.get("status") == "error":
            return guard
        result = self._execute_function("navigate", {"url": url, **kwargs})
        if result.get("status") == "success" and self.options.include_page_preview_after_navigation:
            preview = self.extract_page_content(preview_only=self.options.navigation_preview_only)
            result["page"] = preview.get("data", preview)
        return result

    def back(self, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("back", kwargs)

    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("forward", kwargs)

    def refresh(self, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("refresh", kwargs)

    def current_url(self) -> Dict[str, Any]:
        return self._execute_function("current_url", {})

    def history(self) -> Dict[str, Any]:
        return self._execute_function("history", {})

    def click(self, selector: str, wait_before_execution: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
        wait = self.options.default_wait if wait_before_execution is None else wait_before_execution
        return self._execute_function("click", {"selector": selector, "wait_before_execution": wait, **kwargs})

    def type(self, selector: str, text: str, clear_before: bool = True, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("type", {"selector": selector, "text": text, "clear_before": clear_before, **kwargs})

    def clear_text(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("clear", {"selector": selector, **kwargs})

    def press_key(self, selector: Optional[str] = None, key: str = "ENTER", **kwargs: Any) -> Dict[str, Any]:
        params = {"key": key, **kwargs}
        if selector is not None:
            params["selector"] = selector
        return self._execute_function("press_key", params)

    def scroll(self, mode: str = "by", **kwargs: Any) -> Dict[str, Any]:
        params = dict(kwargs)
        params.setdefault("mode", mode)
        return self._execute_function("scroll", params)

    def copy(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("copy", {"selector": selector, **kwargs})

    def cut(self, selector: str, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("cut", {"selector": selector, **kwargs})

    def paste(self, selector: str, text: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        params = {"selector": selector, **kwargs}
        if text is not None:
            params["text"] = text
        return self._execute_function("paste", params)

    def drag_and_drop(
        self,
        source_selector: str,
        target_selector: Optional[str] = None,
        *,
        offset_x: Optional[int] = None,
        offset_y: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params = {"source_selector": source_selector, "target_selector": target_selector, "offset_x": offset_x, "offset_y": offset_y, **kwargs}
        return self._execute_function("drag_and_drop", params)

    def screenshot(self, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("screenshot", kwargs)

    def page_state(self, **kwargs: Any) -> Dict[str, Any]:
        return self._execute_function("page_state", kwargs)

    def extract_page_content(self, preview_only: Optional[bool] = None, **kwargs: Any) -> Dict[str, Any]:
        preview = self.options.extract_preview_only if preview_only is None else preview_only
        params = {
            "preview_only": preview,
            "include_html": kwargs.pop("include_html", self.options.include_html_on_extract),
            "include_screenshot": kwargs.pop("include_screenshot", self.options.include_screenshot_on_extract),
            **kwargs,
        }
        result = self._execute_function("extract_page", params)
        if result.get("status") == "success" and self.content is not None and self.driver is not None:
            # ContentHandling can apply special post-processing and bounded extraction policies.
            try:
                content_result = self.content.extract_page(self.driver)
                if isinstance(content_result, Mapping) and content_result.get("status") == "success":
                    result.setdefault("content_handling", content_result)
            except Exception as exc:
                logger.debug("ContentHandling.extract_page skipped: %s", exc)
        return result

    def search(
        self,
        query: str,
        engine_url: Optional[str] = None,
        search_box_selector: Optional[str] = None,
        *,
        max_results: Optional[int] = None,
        postprocess: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Search using BrowserFunctions for all browser interactions."""

        start_ms = monotonic_ms()
        query_text = require_non_empty_string(query, "query")
        engine = engine_url or self.options.default_search_engine
        selector = search_box_selector or self.options.search_box_selector
        correlation_id = new_correlation_id("ba-search")

        nav = self.navigate(engine)
        if nav.get("status") != "success":
            nav.setdefault("action", "search")
            return nav

        type_result = self._execute_function(
            "type",
            {
                "selector": selector,
                "text": query_text,
                "clear_before": True,
                "press_enter_after_type": True,
                "correlation_id": correlation_id,
            },
        )
        if type_result.get("status") != "success":
            return type_result

        if self.options.search_results_wait_seconds:
            time_module.sleep(min(self.options.search_results_wait_seconds, 2.0))

        if self.options.scan_after_actions:
            report = self._scan_current_page(action="search")
            if report and self.options.block_on_security_decision and not self._security_allows(report):
                return self._security_block_result("search", report, correlation_id=correlation_id, duration_ms=elapsed_ms(start_ms))

        limit = max_results or self.options.max_search_results
        results = self._extract_search_results(query_text, max_results=limit)
        should_postprocess = self.options.postprocess_search_results if postprocess is None else postprocess
        if should_postprocess:
            results = self._postprocess_results(results)

        payload = success_result(
            action="search",
            message="Search completed",
            data={"query": query_text, "engine_url": engine, "results": results, "count": len(results)},
            duration_ms=elapsed_ms(start_ms),
            correlation_id=correlation_id,
        )
        self._record_agent_result("search", payload, duration_ms=elapsed_ms(start_ms))
        return payload

    async def async_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(self.search, query, **kwargs)

    # ------------------------------------------------------------------
    # Workflow and task entrypoints
    # ------------------------------------------------------------------
    def compile_workflow(self, workflow_definition: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, Any]:
        try:
            compiled = self.workflow.compile(workflow_definition)
            data = compiled.to_dict() if hasattr(compiled, "to_dict") else safe_serialize(compiled)
            return success_result(action="compile_workflow", message="Workflow compiled", data=data)
        except Exception as exc:
            return self._error(exc, action="compile_workflow")

    def dry_run_workflow(self, workflow_definition: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, Any]:
        try:
            dry = self.workflow.dry_run(workflow_definition)
            # dry is already a dictionary; serialize it safely
            data = safe_serialize(dry)
            return success_result(action="dry_run_workflow", message="Workflow dry run completed", data=data)
        except Exception as exc:
            return self._error(exc, action="dry_run_workflow")

    def execute_workflow(self, workflow_script: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]], *,
                         variables: Optional[Mapping[str, Any]] = None, stop_on_error: Optional[bool] = None,
                         ) -> Dict[str, Any]:
        start_ms = monotonic_ms()
        try:
            executable_steps: Sequence[Mapping[str, Any]]
            compiled_payload: Optional[Dict[str, Any]] = None
    
            if self.options.compile_workflows:
                # Merge variables if workflow_script is a dict (full workflow definition)
                workflow_input = workflow_script
                if variables and isinstance(workflow_script, Mapping):
                    workflow_input = merge_dicts(
                        dict(workflow_script),
                        {"variables": merge_dicts(workflow_script.get("variables", {}), variables)},
                        deep=True
                    )
    
                # Compile the workflow (returns CompiledWorkflow)
                compiled = self.workflow.compile(workflow_input)
    
                # compiled.steps is a tuple of dicts (executable steps)
                executable_steps = compiled.steps
    
                # Use safe_serialize for compiled (not .to_dict() unless exists)
                compiled_payload = safe_serialize(compiled)
    
                # Dry run if configured
                if self.options.dry_run_workflows:
                    dry = self.workflow.dry_run(workflow_input)
                    # dry is a dict (see WorkFlow.dry_run return type)
                    dry_payload = safe_serialize(dry)
                    if dry_payload.get("status") == "error":
                        return error_result(
                            action="execute_workflow",
                            message="Workflow dry run failed",
                            error=dry_payload
                        )
    
            else:
                # No compilation – just normalize the script into executable steps
                # Convert Sequence to list if needed to satisfy type checker
                if isinstance(workflow_script, Sequence) and not isinstance(workflow_script, (str, dict)):
                    # It's a list/tuple of steps – pass as list
                    normalized_input = list(workflow_script)
                else:
                    # It's a dict (full workflow definition) – pass as is
                    normalized_input = workflow_script
    
                executable_steps = self.workflow.normalize(normalized_input) # type: ignore
    
            # Execute through BrowserFunctions
            result = self.browser_functions.execute_workflow(
                executable_steps,
                stop_on_error=self.options.workflow_stop_on_error if stop_on_error is None else stop_on_error,
            )
            result = normalize_result(result, action="execute_workflow", default_message="Workflow executed")
    
            if compiled_payload and self.options.include_function_metadata:
                result.setdefault("workflow", {})
                result["workflow"].setdefault("compiled", compiled_payload)
    
            self._record_agent_result("execute_workflow", result, duration_ms=elapsed_ms(start_ms))
            return result
    
        except Exception as exc:
            result = self._error(exc, action="execute_workflow", duration_ms=elapsed_ms(start_ms))
            self._record_agent_result("execute_workflow", result, duration_ms=elapsed_ms(start_ms))
            return result

    async def async_execute_workflow(self, workflow_script: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(self.execute_workflow, workflow_script, **kwargs)

    def perform_task(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        """Agent-standard task entrypoint."""

        start_ms = monotonic_ms()
        try:
            payload = require_mapping(task_data, "task_data", allow_empty=False)
            if not self.enabled:
                return error_result(action="perform_task", message="BrowserAgent is disabled", error=BrowserTaskError("BrowserAgent is disabled"))

            if "workflow" in payload:
                return self.execute_workflow(payload["workflow"], variables=payload.get("variables"), stop_on_error=payload.get("stop_on_error"))

            task_name = self._resolve_task_name(payload)
            params = self._task_params(payload, task_name)

            if task_name == "navigate":
                return self.navigate(params.get("url", params.get("link", "")), **_without(params, "url", "link"))
            if task_name == "search":
                return self.search(params.get("query", ""), engine_url=params.get("engine_url"), search_box_selector=params.get("search_box_selector"), max_results=params.get("max_results"))
            if task_name == "click":
                return self.click(params.get("selector", ""), wait_before_execution=params.get("wait_before_execution"), **_without(params, "selector", "wait_before_execution"))
            if task_name == "type":
                return self.type(params.get("selector", ""), params.get("text", params.get("raw_input", "")), clear_before=params.get("clear_before", True), **_without(params, "selector", "text", "raw_input", "clear_before"))
            if task_name == "scroll":
                return self.scroll(mode=params.get("mode", "by"), **_without(params, "mode"))
            if task_name == "copy":
                return self.copy(params.get("selector", ""), **_without(params, "selector"))
            if task_name == "cut":
                return self.cut(params.get("selector", ""), **_without(params, "selector"))
            if task_name == "paste":
                return self.paste(params.get("selector", ""), text=params.get("text"), **_without(params, "selector", "text"))
            if task_name == "drag_and_drop":
                return self.drag_and_drop(params.get("source_selector", params.get("source", "")), target_selector=params.get("target_selector", params.get("target")), offset_x=params.get("offset_x"), offset_y=params.get("offset_y"), **_without(params, "source_selector", "source", "target_selector", "target", "offset_x", "offset_y"))
            if task_name in {"extract", "extract_page"}:
                return self.extract_page_content(preview_only=params.get("preview_only"), **_without(params, "preview_only"))
            if task_name == "screenshot":
                return self.screenshot(**params)
            if task_name == "page_state":
                return self.page_state(**params)
            if task_name in {"back", "forward", "refresh", "current_url", "history", "clear", "press_key"}:
                return self._execute_function(task_name, params)

            # Last-mile extension point: registered BrowserFunctions functions.
            return self._execute_function(task_name, params)
        except Exception as exc:
            return self._error(exc, action="perform_task", context={"task_data": safe_serialize(task_data)}, duration_ms=elapsed_ms(start_ms))

    async def async_perform_task(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.perform_task, task_data)

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    def _guard_navigation(self, url: str) -> Dict[str, Any]:
        if not self.options.guard_navigation or not self.security:
            return success_result(action="guard_navigation", message="Navigation guard skipped")
        try:
            report = self.security.assess_navigation(url)
            if self.options.block_on_security_decision and not self._security_allows(report):
                return self._security_block_result("navigate", report)
            return success_result(action="guard_navigation", message="Navigation allowed", data={"security": report if self.options.attach_security_reports else None})
        except Exception as exc:
            return self._error(exc, action="guard_navigation", context={"url": url})

    def _guard_action(self, action: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.options.guard_actions or not self.security:
            return success_result(action="guard_action", message="Action guard skipped")
        try:
            report = self.security.assess_action(action, payload)
            if self.options.block_on_security_decision and not self._security_allows(report):
                return self._security_block_result(action, report)
            return success_result(action="guard_action", message="Action allowed", data={"security": report if self.options.attach_security_reports else None})
        except Exception as exc:
            return self._error(exc, action="guard_action", context={"guarded_action": action})

    def _scan_current_page(self, *, action: str) -> Optional[Dict[str, Any]]:
        if not self.security or self.driver is None:
            return None
        try:
            if hasattr(self.security, "scan_current_page"):
                report = self.security.scan_current_page(action=action)
            else:
                report = {"decision": "block", "message": "CAPTCHA detected"} if SecurityFeatures.detect_captcha(self.driver) else {"decision": "allow"}
            return report if isinstance(report, dict) else safe_serialize(report)
        except Exception as exc:
            logger.debug("Security page scan skipped: %s", exc)
            return None

    def _security_allows(self, report: Mapping[str, Any]) -> bool:
        if not report:
            return True
        if hasattr(self.security, "should_continue"):
            try:
                return bool(self.security.should_continue(report))
            except Exception:
                pass
        decision = str(report.get("decision") or report.get("status") or "allow").lower()
        return decision not in {"block", "blocked", "error"}

    def _security_block_result(self, action: str, report: Mapping[str, Any], *, correlation_id: Optional[str] = None, duration_ms: Optional[float] = None) -> Dict[str, Any]:
        error = BrowserSecurityError("Browser security policy blocked the operation", context={"action": action, "security_report": report})
        return error_result(
            action=action,
            message="Browser security policy blocked the operation",
            error=error,
            metadata={"security": safe_serialize(report)},
            correlation_id=correlation_id,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _extract_search_results(self, query: str, *, max_results: int) -> List[Dict[str, Any]]:
        if self.driver is None:
            return []
        try:
            snapshots = extract_link_snapshots(
                self.driver,
                query=query,
                selectors=self.options.search_result_selectors,
                max_results=max_results,
            )
            return search_result_dicts(snapshots)
        except Exception as exc:
            logger.debug("Search-result helper extraction failed: %s", exc)
            return []

    def _postprocess_results(self, results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for result in results:
            item = dict(result)
            try:
                # Use instance method with return_result=False to get a dict
                if hasattr(self.content, "postprocess_result"):
                    processed_item = self.content.postprocess_result(item, driver=self.driver, return_result=True)
                    item = safe_serialize(processed_item) if not isinstance(processed_item, dict) else processed_item
                else:
                    item = ContentHandling.postprocess_if_special(item, self.driver)
            except Exception as exc:
                logger.debug("Search result post-processing skipped for %s: %s",
                             item.get("link") or item.get("url"), exc)
            processed.append(item)
        return processed

    # ------------------------------------------------------------------
    # Task normalization and memory publishing
    # ------------------------------------------------------------------
    def _resolve_task_name(self, payload: Mapping[str, Any]) -> str:
        raw = payload.get("task") or payload.get("action") or payload.get("function") or payload.get("name")
        if not raw:
            if "url" in payload:
                raw = "navigate"
            elif "query" in payload:
                raw = "search"
            else:
                raise UnsupportedBrowserTaskError("Unsupported browser task payload", context={"payload_keys": sorted(str(k) for k in payload.keys())})
        name = normalize_whitespace(raw).lower().replace("-", "_").replace(" ", "_")
        return BROWSER_AGENT_TASK_ALIASES.get(name, name)

    def _task_params(self, payload: Mapping[str, Any], task_name: str) -> Dict[str, Any]:
        params = dict(payload.get("params") or {}) if isinstance(payload.get("params"), Mapping) else {}
        for key, value in payload.items():
            if key not in {"task", "action", "function", "name", "params", "workflow", "variables", "stop_on_error"}:
                params.setdefault(str(key), value)
        return params

    def _record_agent_result(self, operation: str, result: Mapping[str, Any], *, attempts: int = 1, duration_ms: Optional[float] = None) -> None:
        status = str(result.get("status") or "unknown")
        message = str(result.get("message") or "")
        record = BrowserAgentExecution(
            operation=operation,
            status=status,
            message=message,
            duration_ms=coerce_float(duration_ms, default=0.0, minimum=0.0),
            attempts=max(1, int(attempts)),
            correlation_id=str(result.get("correlation_id") or new_correlation_id("ba")),
            metadata={"fingerprint": stable_hash(result), "action": result.get("action")},
        ).to_dict()
        self.execution_history.append(record)
        self._publish_shared_memory(result=result, record=record)

    def _publish_shared_memory(self, *, result: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        if not self.options.publish_to_shared_memory or not self.shared_memory:
            return
        payload = redact_mapping(safe_serialize({"result": result, "record": record}))
        try:
            if hasattr(self.shared_memory, "set"):
                self.shared_memory.set(self.options.shared_result_key, payload)
                self.shared_memory.set(self.options.shared_history_key, list(self.execution_history))
            elif hasattr(self.shared_memory, "put"):
                self.shared_memory.put(self.options.shared_result_key, payload)
            if hasattr(self.shared_memory, "publish"):
                self.shared_memory.publish(self.options.shared_event_topic, payload)
        except Exception as exc:
            logger.debug("Unable to publish BrowserAgent shared-memory event: %s", exc)

    def _publish_health(self, status: str) -> None:
        if not self.options.publish_to_shared_memory or not self.shared_memory:
            return
        health = {
            "name": self.name,
            "status": status,
            "enabled": self.enabled,
            "has_driver": self.driver is not None,
            "history_size": len(self.execution_history),
            "updated_at": utc_now_iso(),
        }
        try:
            if hasattr(self.shared_memory, "set"):
                self.shared_memory.set(self.options.shared_health_key, health)
        except Exception as exc:
            logger.debug("Unable to publish BrowserAgent health: %s", exc)

    # ------------------------------------------------------------------
    # Error and retry helpers
    # ------------------------------------------------------------------
    def _sleep_before_retry(self, attempt: int) -> None:
        if not self.options.retry_sleep_enabled:
            return
        try:
            delay = calculate_backoff_delay(
                attempt_index=attempt,
                base_delay=self.options.retry_base_delay,
                max_delay=self.options.retry_max_delay,
                multiplier=self.options.retry_multiplier,
                jitter=self.options.retry_jitter,
            )
        except Exception:
            delay = exponential_backoff(attempt)
        if delay > 0:
            time_module.sleep(delay)

    def _error(
        self,
        exc: BaseException,
        *,
        action: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        browser_error = wrap_browser_exception(
            exc,
            action=action,
            context=context,
            default_error_cls=BrowserTaskError,
        )
        return browser_error.to_result(
            action=action,
            include_traceback=self.options.include_traceback_on_error,
            extra={"duration_ms": duration_ms} if duration_ms is not None else None,
        )

    def status(self) -> Dict[str, Any]:
        driver_state = None
        driver_health = None
        if hasattr(self, "browser_driver"):
            try:
                driver_state = self.browser_driver.state()
            except Exception as exc:
                driver_state = {"status": "error", "message": str(exc)}
            try:
                driver_health = self.browser_driver.health()
            except Exception as exc:
                driver_health = {"status": "error", "message": str(exc)}
        data = {
            "enabled": self.enabled,
            "has_driver": self.driver is not None,
            "closed": self._closed,
            "owns_driver": self._owns_driver,
            "history_size": len(self.execution_history),
            "driver": driver_state,
            "driver_health": driver_health,
            "functions": self.browser_functions.list_functions() if hasattr(self.browser_functions, "list_functions") else None,
            "latest_security_report": self.security.latest_report() if hasattr(self.security, "latest_report") else None,
        }
        return success_result(action="browser_agent_status", message="BrowserAgent status returned", data=data)


# ---------------------------------------------------------------------------
# Small local helpers scoped to BrowserAgent task routing
# ---------------------------------------------------------------------------
def _without(mapping: Mapping[str, Any], *keys: str) -> Dict[str, Any]:
    excluded = set(keys)
    return {key: value for key, value in dict(mapping).items() if key not in excluded}


if __name__ == "__main__":
    print("\n=== Running  Browser agent ===\n")
    printer.status("TEST", " Data quality agent initialized", "info")

    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    agent = BrowserAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        config={
            "enabled": True,
            "driver": {
                "auto_start": True,
            },
            "shared_memory": {
                "enabled": True,
            },
            "content": {
                "include_page_preview_after_navigation": False,
                "navigation_preview_only": True,
                "extract_preview_only": True,
                "include_html_on_extract": False,
                "include_screenshot_on_extract": False,
            },
            "retry": {
                "sleep_enabled": False,
            },
            "security": {
                "guard_navigation": True,
                "guard_actions": True,
                "scan_after_actions": True,
                "block_on_security_decision": True,
                "attach_reports": True,
            },
            "workflow": {
                "compile_before_execute": True,
                "stop_on_error": True,
                "dry_run": False,
            },
        },
    )

    try:
        test_url = "https://www.selenium.dev/selenium/web/web-form.html"

        nav = agent.perform_task(
            {
                "task": "navigate",
                "url": test_url,
            }
        )
        assert nav["status"] == "success", nav

        page_state = agent.perform_task({"task": "page_state"})
        assert page_state["status"] == "success", page_state

        typed = agent.perform_task(
            {
                "task": "type",
                "selector": "input[name='my-text']",
                "text": "browser agent production test",
                "clear_before": True,
            }
        )
        assert typed["status"] == "success", typed

        click = agent.perform_task(
            {
                "task": "click",
                "selector": "button",
            }
        )
        assert click["status"] == "success", click

        # Navigate back to the form page after submit/click side effects.
        nav_again = agent.perform_task(
            {
                "task": "navigate",
                "url": test_url,
            }
        )
        assert nav_again["status"] == "success", nav_again

        scroll = agent.perform_task(
            {
                "task": "scroll",
                "mode": "direction",
                "direction": "down",
                "amount": 300,
            }
        )
        assert scroll["status"] == "success", scroll

        screenshot = agent.perform_task({"task": "screenshot"})
        assert screenshot["status"] == "success", screenshot

        extracted = agent.perform_task(
            {
                "task": "extract_page",
                "preview_only": True,
                "include_html": False,
                "include_screenshot": False,
            }
        )
        assert extracted["status"] == "success", extracted

        workflow = agent.perform_task(
            {
                "workflow": {
                    "name": "browser_agent_real_module_smoke_test",
                    "steps": [
                        {
                            "action": "navigate",
                            "params": {
                                "url": test_url,
                            },
                        },
                        {
                            "action": "type",
                            "params": {
                                "selector": "input[name='my-text']",
                                "text": "workflow test",
                                "clear_before": True,
                            },
                        },
                        {
                            "action": "scroll",
                            "params": {
                                "mode": "direction",
                                "direction": "down",
                                "amount": 150,
                            },
                        },
                        {
                            "action": "page_state",
                            "params": {},
                        },
                    ],
                }
            }
        )
        assert workflow["status"] == "success", workflow

        status_result = agent.status()
        assert status_result["status"] == "success", status_result

    finally:
        agent.close()

    print("\n=== Test ran successfully ===\n")
