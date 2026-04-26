from __future__ import annotations

__version__ = "1.0.0"

"""
Production-grade Selenium driver lifecycle manager for the browser subsystem.

This module owns browser-driver lifecycle only. It does not own concrete browser
actions, browser task routing, workflow execution, security policy, content
extraction, or agent orchestration. Those responsibilities belong to
BrowserFunctions, WorkFlow, SecurityFeatures, ContentHandling, and BrowserAgent.

Responsibilities
----------------
- Load driver runtime policy from browser_config.yaml through the existing
  browser config loader.
- Create Selenium drivers using a deterministic, testable policy object.
- Attach externally managed drivers without taking ownership unless requested.
- Configure browser timeouts, page-load strategy, window size, Chrome options,
  Chrome preferences, and remote WebDriver support.
- Provide health/state snapshots suitable for BrowserAgent, diagnostics,
  shared-memory telemetry, and tests.
- Restart or close owned drivers safely.
- Return stable browser-style result dictionaries using Browser_helpers and
  browser_errors rather than duplicating result or error handling.

Design boundaries
-----------------
BrowserDriver does not call DoNavigate, DoClick, DoType, DoScroll,
DoCopyCutPaste, DoDragAndDrop, BrowserFunctions.execute(), or Selenium action
methods such as click/type/scroll. It may inspect lightweight driver state for
health and diagnostics.

Local imports are intentionally direct. They are not wrapped in try/except so
packaging or path problems fail clearly during development and deployment.
"""

import time as time_module

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Tuple

from selenium import webdriver

from .config_loader import load_global_config, get_config_section
from .browser_errors import *
from .Browser_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Browser Driver")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BROWSER_DRIVER_SCHEMA_VERSION = "1.0"
DEFAULT_BROWSER_BACKEND = "chrome"
SUPPORTED_BROWSER_BACKENDS: Tuple[str, ...] = ("chrome", "chromium")
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
DEFAULT_WINDOW_SIZE: Tuple[int, int] = (1366, 920)
DEFAULT_CHROME_ARGS: Tuple[str, ...] = (
    "--no-sandbox",
    "--disable-dev-shm-usage",
)
DEFAULT_EVENT_HISTORY_LIMIT = 100
VALID_PAGE_LOAD_STRATEGIES = {"normal", "eager", "none"}

DriverFactory = Callable[[Any], Any]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BrowserDriverOptions:
    """Config-backed policy for creating and managing a browser driver."""

    enabled: bool = True
    auto_start: bool = True
    browser: str = DEFAULT_BROWSER_BACKEND
    remote_url: Optional[str] = None

    headless: bool = True
    user_agent: str = DEFAULT_USER_AGENT
    window_size: Tuple[int, int] = DEFAULT_WINDOW_SIZE
    page_load_strategy: str = "normal"
    implicit_wait_seconds: float = 0.0
    page_load_timeout_seconds: float = 45.0
    script_timeout_seconds: float = 30.0

    chrome_args: Tuple[str, ...] = DEFAULT_CHROME_ARGS
    chrome_experimental_options: Dict[str, Any] = field(default_factory=dict)
    chrome_prefs: Dict[str, Any] = field(default_factory=dict)

    configure_attached_driver: bool = True
    close_owned_driver: bool = True
    detach_driver_on_close: bool = False
    close_driver_on_del: bool = True
    restart_on_unhealthy: bool = False

    startup_retries: int = 0
    startup_retry_sleep_enabled: bool = True
    startup_retry_base_delay: float = 0.5
    startup_retry_max_delay: float = 5.0
    startup_retry_multiplier: float = 2.0
    startup_retry_jitter: float = 0.05

    health_check_script: str = "return document.readyState"
    health_check_timeout_seconds: float = 2.0
    include_capabilities_in_state: bool = False
    include_driver_repr_in_state: bool = False
    include_state_on_success: bool = True
    event_history_limit: int = DEFAULT_EVENT_HISTORY_LIMIT

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]], **overrides: Any) -> "BrowserDriverOptions":
        """Resolve options from browser_config.yaml plus optional runtime overrides."""

        cfg = merge_dicts(dict(config or {}), {k: v for k, v in overrides.items() if v is not None}, deep=True)
        chrome_cfg = dict(cfg.get("chrome") or {})
        timeout_cfg = dict(cfg.get("timeouts") or {})
        startup_cfg = dict(cfg.get("startup") or {})
        diagnostics_cfg = dict(cfg.get("diagnostics") or {})
        lifecycle_cfg = dict(cfg.get("lifecycle") or {})

        window_values = ensure_list(cfg.get("window_size", DEFAULT_WINDOW_SIZE))
        if len(window_values) < 2:
            window_values = list(DEFAULT_WINDOW_SIZE)

        chrome_args = ensure_list(chrome_cfg.get("args", cfg.get("chrome_args", DEFAULT_CHROME_ARGS)))
        if not chrome_args:
            chrome_args = list(DEFAULT_CHROME_ARGS)

        browser = str(cfg.get("browser", DEFAULT_BROWSER_BACKEND) or DEFAULT_BROWSER_BACKEND).lower().strip()
        page_load_strategy = str(cfg.get("page_load_strategy", "normal") or "normal").lower().strip()
        if page_load_strategy not in VALID_PAGE_LOAD_STRATEGIES:
            page_load_strategy = "normal"

        return cls(
            enabled=coerce_bool(cfg.get("enabled", True), default=True),
            auto_start=coerce_bool(cfg.get("auto_start", cfg.get("auto_start_browser", True)), default=True),
            browser=browser,
            remote_url=cfg.get("remote_url"),
            headless=coerce_bool(cfg.get("headless", True), default=True),
            user_agent=str(cfg.get("user_agent") or DEFAULT_USER_AGENT),
            window_size=(
                coerce_int(window_values[0], default=DEFAULT_WINDOW_SIZE[0], minimum=1),
                coerce_int(window_values[1], default=DEFAULT_WINDOW_SIZE[1], minimum=1),
            ),
            page_load_strategy=page_load_strategy,
            implicit_wait_seconds=coerce_float(
                timeout_cfg.get("implicit_wait_seconds", cfg.get("implicit_wait_seconds", 0.0)),
                default=0.0,
                minimum=0.0,
                maximum=300.0,
            ),
            page_load_timeout_seconds=coerce_float(
                timeout_cfg.get("page_load_timeout_seconds", cfg.get("page_load_timeout_seconds", 45.0)),
                default=45.0,
                minimum=0.0,
                maximum=600.0,
            ),
            script_timeout_seconds=coerce_float(
                timeout_cfg.get("script_timeout_seconds", cfg.get("script_timeout_seconds", 30.0)),
                default=30.0,
                minimum=0.0,
                maximum=600.0,
            ),
            chrome_args=tuple(str(item) for item in chrome_args if str(item).strip()),
            chrome_experimental_options=dict(chrome_cfg.get("experimental_options", cfg.get("chrome_experimental_options", {})) or {}),
            chrome_prefs=dict(chrome_cfg.get("prefs", cfg.get("chrome_prefs", {})) or {}),
            configure_attached_driver=coerce_bool(
                lifecycle_cfg.get("configure_attached_driver", cfg.get("configure_attached_driver", True)), default=True
            ),
            close_owned_driver=coerce_bool(lifecycle_cfg.get("close_owned_driver", cfg.get("close_owned_driver", True)), default=True),
            detach_driver_on_close=coerce_bool(lifecycle_cfg.get("detach_on_close", cfg.get("detach_driver_on_close", False)), default=False),
            close_driver_on_del=coerce_bool(lifecycle_cfg.get("close_on_del", cfg.get("close_driver_on_del", True)), default=True),
            restart_on_unhealthy=coerce_bool(lifecycle_cfg.get("restart_on_unhealthy", cfg.get("restart_on_unhealthy", False)), default=False),
            startup_retries=coerce_int(startup_cfg.get("retries", cfg.get("startup_retries", 0)), default=0, minimum=0, maximum=20),
            startup_retry_sleep_enabled=coerce_bool(
                startup_cfg.get("sleep_enabled", cfg.get("startup_retry_sleep_enabled", True)), default=True
            ),
            startup_retry_base_delay=coerce_float(
                startup_cfg.get("base_delay", cfg.get("startup_retry_base_delay", 0.5)), default=0.5, minimum=0.0, maximum=60.0
            ),
            startup_retry_max_delay=coerce_float(
                startup_cfg.get("max_delay", cfg.get("startup_retry_max_delay", 5.0)), default=5.0, minimum=0.0, maximum=300.0
            ),
            startup_retry_multiplier=coerce_float(
                startup_cfg.get("multiplier", cfg.get("startup_retry_multiplier", 2.0)), default=2.0, minimum=1.0, maximum=10.0
            ),
            startup_retry_jitter=coerce_float(
                startup_cfg.get("jitter", cfg.get("startup_retry_jitter", 0.05)), default=0.05, minimum=0.0, maximum=10.0
            ),
            health_check_script=str(cfg.get("health_check_script", "return document.readyState") or "return document.readyState"),
            health_check_timeout_seconds=coerce_float(
                cfg.get("health_check_timeout_seconds", 2.0), default=2.0, minimum=0.0, maximum=60.0
            ),
            include_capabilities_in_state=coerce_bool(
                diagnostics_cfg.get("include_capabilities", cfg.get("include_capabilities_in_state", False)), default=False
            ),
            include_driver_repr_in_state=coerce_bool(
                diagnostics_cfg.get("include_driver_repr", cfg.get("include_driver_repr_in_state", False)), default=False
            ),
            include_state_on_success=coerce_bool(
                diagnostics_cfg.get("include_state_on_success", cfg.get("include_state_on_success", True)), default=True
            ),
            event_history_limit=coerce_int(
                diagnostics_cfg.get("event_history_limit", cfg.get("event_history_limit", DEFAULT_EVENT_HISTORY_LIMIT)),
                default=DEFAULT_EVENT_HISTORY_LIMIT,
                minimum=1,
                maximum=10_000,
            ),
        )


@dataclass(frozen=True)
class BrowserDriverState:
    """Stable diagnostic state for a managed browser driver."""

    status: str
    has_driver: bool
    owns_driver: bool
    closed: bool
    browser: str
    remote_url: Optional[str]
    session_id: Optional[str] = None
    current_url: Optional[str] = None
    title: Optional[str] = None
    ready_state: Optional[str] = None
    window_size: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, Any]] = None
    driver_repr: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: str = field(default_factory=utc_now_iso)
    correlation_id: str = field(default_factory=lambda: new_correlation_id("drv-state"))

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self))


@dataclass(frozen=True)
class BrowserDriverEvent:
    """Compact audit event for driver lifecycle transitions."""

    action: str
    status: str
    message: str
    duration_ms: float
    correlation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self))


# ---------------------------------------------------------------------------
# BrowserDriver manager
# ---------------------------------------------------------------------------
class BrowserDriver:
    """Owns Selenium driver lifecycle for BrowserAgent.

    ``BrowserDriver`` is intentionally small at the action layer. It creates,
    configures, attaches, detaches, restarts, and closes drivers. Browser actions
    still flow through BrowserFunctions.
    """

    def __init__(
        self,
        driver: Any = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
        auto_start: Optional[bool] = None,
        owns_driver: Optional[bool] = None,
        driver_factory: Optional[DriverFactory] = None,
    ) -> None:
        self.config = load_global_config()
        self.driver_config: Dict[str, Any] = dict(get_config_section("browser_driver") or {})
        if config:
            self.driver_config = merge_dicts(self.driver_config, dict(config), deep=True)
        if auto_start is not None:
            self.driver_config["auto_start"] = auto_start

        self.options = BrowserDriverOptions.from_config(self.driver_config)
        self.driver_factory = driver_factory
        self.driver = driver
        self._owns_driver = bool(driver is None if owns_driver is None else owns_driver)
        self._closed = driver is None
        self._created_at: Optional[str] = None
        self.events: Deque[Dict[str, Any]] = deque(maxlen=self.options.event_history_limit)

        if self.driver is not None:
            self._closed = False
            self._created_at = utc_now_iso()
            if self.options.configure_attached_driver:
                self.configure_driver(self.driver)
            self._record_event("attach", "success", "Browser driver attached during initialization")
        elif self.options.enabled and self.options.auto_start:
            self.start()

        logger.info("Browser Driver initialized.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def has_driver(self) -> bool:
        return self.driver is not None and not self._closed

    @property
    def owns_driver(self) -> bool:
        return self._owns_driver

    def start(self, *, force: bool = False) -> Dict[str, Any]:
        """Create a driver if needed and return a browser-style result."""

        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("drv-start")
        if not self.options.enabled:
            return error_result(
                action="start_driver",
                message="Browser driver manager is disabled",
                error=BrowserConfigurationError("Browser driver manager is disabled"),
                correlation_id=correlation_id,
                duration_ms=elapsed_ms(start_ms),
            )
        if self.has_driver and not force:
            result = success_result(
                action="start_driver",
                message="Browser driver already started",
                data=self._success_state_data(),
                correlation_id=correlation_id,
                duration_ms=elapsed_ms(start_ms),
            )
            self._record_event("start", "success", "Browser driver already started", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            return result

        if self.has_driver and force:
            self.close()

        last_error: Optional[BaseException] = None
        attempts = self.options.startup_retries + 1
        for attempt in range(attempts):
            try:
                driver = self._create_driver()
                self.driver = driver
                self._owns_driver = True
                self._closed = False
                self._created_at = utc_now_iso()
                self.configure_driver(driver)
                result = success_result(
                    action="start_driver",
                    message="Browser driver started",
                    data=self._success_state_data(),
                    metadata={"attempts": attempt + 1},
                    correlation_id=correlation_id,
                    duration_ms=elapsed_ms(start_ms),
                )
                self._record_event("start", "success", "Browser driver started", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
                return result
            except Exception as exc:
                last_error = exc
                logger.warning("Browser driver startup failed on attempt %s/%s: %s", attempt + 1, attempts, exc)
                if attempt < attempts - 1:
                    self._sleep_before_startup_retry(attempt)

        error = BrowserDriverStartupError(
            "Failed to start browser driver",
            context={"browser": self.options.browser, "attempts": attempts},
            cause=last_error,
        )
        result = error.to_result(action="start_driver", extra={"duration_ms": elapsed_ms(start_ms), "correlation_id": correlation_id})
        self._record_event("start", "error", "Browser driver startup failed", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
        return result

    def attach(self, driver: Any, *, owns_driver: bool = False, configure: Optional[bool] = None) -> Dict[str, Any]:
        """Attach an externally created driver."""

        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("drv-attach")
        try:
            require(driver is not None, "driver cannot be None", error_cls=MissingDriverError)
            if self.has_driver and self.driver is not driver:
                self.close()
            self.driver = driver
            self._owns_driver = bool(owns_driver)
            self._closed = False
            self._created_at = self._created_at or utc_now_iso()
            should_configure = self.options.configure_attached_driver if configure is None else bool(configure)
            if should_configure:
                self.configure_driver(driver)
            result = success_result(
                action="attach_driver",
                message="Browser driver attached",
                data=self._success_state_data(),
                correlation_id=correlation_id,
                duration_ms=elapsed_ms(start_ms),
            )
            self._record_event("attach", "success", "Browser driver attached", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            return result
        except Exception as exc:
            result = self._error(exc, action="attach_driver", duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            self._record_event("attach", "error", "Browser driver attach failed", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            return result

    def detach(self) -> Any:
        """Detach and return the managed driver without closing it."""

        driver = self.driver
        self.driver = None
        self._owns_driver = False
        self._closed = True
        self._record_event("detach", "success", "Browser driver detached")
        return driver

    def detach_result(self) -> Dict[str, Any]:
        """Detach driver and return a result payload without exposing the object."""

        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("drv-detach")
        had_driver = self.driver is not None
        self.detach()
        return success_result(
            action="detach_driver",
            message="Browser driver detached",
            data={"had_driver": had_driver, "has_driver": False},
            duration_ms=elapsed_ms(start_ms),
            correlation_id=correlation_id,
        )

    def close(self) -> Dict[str, Any]:
        """Close an owned driver or detach according to lifecycle policy."""

        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("drv-close")
        if self.driver is None or self._closed:
            result = success_result(
                action="close_driver",
                message="Browser driver already closed",
                data={"closed": True, "has_driver": False},
                duration_ms=elapsed_ms(start_ms),
                correlation_id=correlation_id,
            )
            self._record_event("close", "success", "Browser driver already closed", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            return result

        try:
            should_close = self._owns_driver and self.options.close_owned_driver and not self.options.detach_driver_on_close
            if should_close and hasattr(self.driver, "quit"):
                self.driver.quit()
            self.driver = None
            self._closed = True
            result = success_result(
                action="close_driver",
                message="Browser driver closed" if should_close else "Browser driver detached on close",
                data={"closed": True, "closed_owned_driver": should_close, "has_driver": False},
                duration_ms=elapsed_ms(start_ms),
                correlation_id=correlation_id,
            )
            self._record_event("close", "success", result.get("message", "Browser driver closed"), result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            return result
        except Exception as exc:
            result = self._error(exc, action="close_driver", duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            self._record_event("close", "error", "Browser driver close failed", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            return result

    def restart(self) -> Dict[str, Any]:
        """Close and recreate the driver."""

        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("drv-restart")
        close_result = self.close()
        if close_result.get("status") == "error":
            return close_result
        start_result = self.start(force=True)
        start_result["action"] = "restart_driver"
        start_result["message"] = "Browser driver restarted" if start_result.get("status") == "success" else start_result.get("message")
        start_result["duration_ms"] = elapsed_ms(start_ms)
        start_result["correlation_id"] = correlation_id
        self._record_event("restart", start_result.get("status", "unknown"), str(start_result.get("message") or "Browser driver restart completed"), result=start_result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
        return start_result

    def ensure_started(self) -> Any:
        """Return an active driver, starting it when configured to do so."""

        if self.has_driver:
            return self.driver
        result = self.start()
        if result.get("status") != "success":
            raise BrowserDriverStartupError("Browser driver is not available", context={"result": result})
        return self.driver

    def require_driver(self) -> Any:
        """Return the driver or raise a browser state error."""

        if not self.has_driver:
            raise MissingDriverError("Browser driver is missing", context={"closed": self._closed})
        return self.driver

    def configure_driver(self, driver: Any = None) -> Dict[str, Any]:
        """Apply timeouts/window settings to a driver."""

        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("drv-config")
        target = driver or self.driver
        try:
            require(target is not None, "driver cannot be None", error_cls=MissingDriverError)
            if self.options.implicit_wait_seconds and target is not None and hasattr(target, "implicitly_wait"):
                target.implicitly_wait(self.options.implicit_wait_seconds)
            if self.options.page_load_timeout_seconds and target is not None and hasattr(target, "set_page_load_timeout"):
                target.set_page_load_timeout(self.options.page_load_timeout_seconds)
            if self.options.script_timeout_seconds and target is not None and hasattr(target, "set_script_timeout"):
                target.set_script_timeout(self.options.script_timeout_seconds)
            if self.options.window_size and target is not None and hasattr(target, "set_window_size"):
                try:
                    target.set_window_size(*self.options.window_size)
                except Exception as exc:
                    logger.debug("Unable to set driver window size: %s", exc)
            result = success_result(
                action="configure_driver",
                message="Browser driver configured",
                data={"configured": True},
                correlation_id=correlation_id,
                duration_ms=elapsed_ms(start_ms),
            )
            self._record_event("configure", "success", "Browser driver configured", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            return result
        except Exception as exc:
            result = self._error(exc, action="configure_driver", duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            self._record_event("configure", "error", "Browser driver configuration failed", result=result, duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)
            return result

    # ------------------------------------------------------------------
    # State and diagnostics
    # ------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        return self._capture_state().to_dict()

    def health(self) -> Dict[str, Any]:
        """Return a health result for the current driver."""

        start_ms = monotonic_ms()
        correlation_id = new_correlation_id("drv-health")
        if not self.has_driver:
            result = error_result(
                action="driver_health",
                message="Browser driver is missing",
                error=MissingDriverError("Browser driver is missing"),
                data={"healthy": False},
                duration_ms=elapsed_ms(start_ms),
                correlation_id=correlation_id,
            )
            return result
        try:
            ready_state = self._safe_ready_state(self.driver)
            healthy = ready_state in {None, "interactive", "complete", "loading"}
            state = self._capture_state(status="healthy" if healthy else "unhealthy")
            if not healthy and self.options.restart_on_unhealthy:
                restart_result = self.restart()
                return normalize_result(restart_result, action="driver_health")
            return success_result(
                action="driver_health",
                message="Browser driver health checked",
                data={"healthy": healthy, "state": state.to_dict()},
                duration_ms=elapsed_ms(start_ms),
                correlation_id=correlation_id,
            )
        except Exception as exc:
            if self.options.restart_on_unhealthy:
                return self.restart()
            return self._error(exc, action="driver_health", duration_ms=elapsed_ms(start_ms), correlation_id=correlation_id)

    def events_snapshot(self, *, limit: Optional[int] = None) -> list[Dict[str, Any]]:
        items = list(self.events)
        if limit is not None:
            items = items[-max(0, int(limit)):]
        return safe_serialize(items)

    # ------------------------------------------------------------------
    # Internal driver creation
    # ------------------------------------------------------------------
    def _create_driver(self) -> Any:
        if self.options.browser not in SUPPORTED_BROWSER_BACKENDS:
            raise BrowserConfigurationError(
                f"Unsupported browser backend: {self.options.browser}",
                context={"browser": self.options.browser, "supported": list(SUPPORTED_BROWSER_BACKENDS)},
            )
        options = self.build_chrome_options()
        if self.driver_factory is not None:
            return self.driver_factory(options)
        if self.options.remote_url:
            return webdriver.Remote(command_executor=self.options.remote_url, options=options)
        return webdriver.Chrome(options=options)

    def build_chrome_options(self) -> Any:
        """Create Selenium ChromeOptions from resolved config."""

        options = webdriver.ChromeOptions()
        options.page_load_strategy = self.options.page_load_strategy
        if self.options.user_agent:
            options.add_argument(f"user-agent={self.options.user_agent}")
        if self.options.window_size:
            options.add_argument(f"window-size={self.options.window_size[0]},{self.options.window_size[1]}")
        if self.options.headless:
            options.add_argument("--headless=new")
        for arg in self.options.chrome_args:
            if arg:
                options.add_argument(str(arg))
        if self.options.chrome_prefs:
            options.add_experimental_option("prefs", self.options.chrome_prefs)
        for key, value in self.options.chrome_experimental_options.items():
            options.add_experimental_option(str(key), value)
        return options

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _success_state_data(self) -> Dict[str, Any]:
        if not self.options.include_state_on_success:
            return {"has_driver": self.has_driver, "owns_driver": self._owns_driver}
        return {"state": self.state()}

    def _capture_state(self, *, status: Optional[str] = None) -> BrowserDriverState:
        driver = self.driver
        has_driver = driver is not None and not self._closed
        resolved_status = status or ("running" if has_driver else "closed")
        session_id = safe_serialize(getattr(driver, "session_id", None)) if driver is not None else None
        current_url = self._safe_attr(driver, "current_url") if driver is not None else None
        title = self._safe_attr(driver, "title") if driver is not None else None
        ready_state = self._safe_ready_state(driver) if driver is not None else None
        window_size = self._safe_window_size(driver) if driver is not None else None
        capabilities = None
        if self.options.include_capabilities_in_state and driver is not None:
            capabilities = safe_serialize(getattr(driver, "capabilities", None))
        driver_repr = repr(driver) if self.options.include_driver_repr_in_state and driver is not None else None
        return BrowserDriverState(
            status=resolved_status,
            has_driver=has_driver,
            owns_driver=self._owns_driver,
            closed=self._closed,
            browser=self.options.browser,
            remote_url=self.options.remote_url,
            session_id=session_id,
            current_url=current_url,
            title=title,
            ready_state=ready_state,
            window_size=window_size,
            capabilities=capabilities,
            driver_repr=driver_repr,
            created_at=self._created_at,
        )

    def _safe_attr(self, obj: Any, attr: str) -> Optional[str]:
        try:
            value = getattr(obj, attr, None)
            return str(value) if value is not None else None
        except Exception:
            return None

    def _safe_ready_state(self, driver: Any) -> Optional[str]:
        try:
            if driver is None or not hasattr(driver, "execute_script"):
                return None
            value = driver.execute_script(self.options.health_check_script)
            return str(value) if value is not None else None
        except Exception:
            return None

    def _safe_window_size(self, driver: Any) -> Optional[Dict[str, Any]]:
        if driver is None:
            return None
        try:
            if hasattr(driver, "get_window_size"):
                value = driver.get_window_size()
                return safe_serialize(value) if isinstance(value, Mapping) else {"value": safe_serialize(value)}
        except Exception:
            pass
        try:
            if hasattr(driver, "execute_script"):
                value = driver.execute_script(
                    "return {width: window.innerWidth, height: window.innerHeight, "
                    "scrollX: window.scrollX, scrollY: window.scrollY, "
                    "pageWidth: document.documentElement.scrollWidth, "
                    "pageHeight: document.documentElement.scrollHeight};"
                )
                return safe_serialize(value) if isinstance(value, Mapping) else {"value": safe_serialize(value)}
        except Exception:
            return None
        return None

    def _sleep_before_startup_retry(self, attempt: int) -> None:
        if not self.options.startup_retry_sleep_enabled:
            return
        delay = calculate_backoff_delay(
            attempt_index=attempt,
            base_delay=self.options.startup_retry_base_delay,
            max_delay=self.options.startup_retry_max_delay,
            multiplier=self.options.startup_retry_multiplier,
            jitter=self.options.startup_retry_jitter,
        )
        if delay > 0:
            time_module.sleep(delay)

    def _record_event(
        self,
        action: str,
        status: str,
        message: str,
        *,
        result: Optional[Mapping[str, Any]] = None,
        duration_ms: float = 0.0,
        correlation_id: Optional[str] = None,
    ) -> None:
        event = BrowserDriverEvent(
            action=action,
            status=status,
            message=message,
            duration_ms=coerce_float(duration_ms, default=0.0, minimum=0.0),
            correlation_id=correlation_id or new_correlation_id("drv"),
            metadata={"result_fingerprint": stable_hash(result) if result else None},
        ).to_dict()
        self.events.append(redact_mapping(safe_serialize(event)))

    def _error(self, exc: BaseException, *, action: str, duration_ms: Optional[float] = None, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        browser_error = wrap_browser_exception(exc, action=action, default_error_cls=BrowserDriverError)
        return browser_error.to_result(
            action=action,
            extra=prune_none({"duration_ms": duration_ms, "correlation_id": correlation_id}),
        )

    def __enter__(self) -> "BrowserDriver":
        self.ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "options", None) and self.options.close_driver_on_del:
            try:
                self.close()
            except Exception:
                pass


BrowserDriverManager = BrowserDriver


__all__ = [
    "BrowserDriver",
    "BrowserDriverManager",
    "BrowserDriverOptions",
    "BrowserDriverState",
    "BrowserDriverEvent",
]


if __name__ == "__main__":
    print("\n=== Running Browser Driver ===\n")
    printer.status("TEST", "Browser Driver initialized", "info")

    class _FakeDriver:
        def __init__(self, options: Any = None) -> None:
            self.options = options
            self.current_url = "about:blank"
            self.title = "Fake Browser"
            self.page_source = "<html><body>Fake Browser</body></html>"
            self.session_id = new_correlation_id("session")
            self.capabilities = {"browserName": "fake-chrome", "browserVersion": "test"}
            self.closed = False
            self.implicit_wait = 0.0
            self.page_load_timeout = 0.0
            self.script_timeout = 0.0
            self.window_size = {"width": 1366, "height": 920}

        def get(self, url: str) -> None:
            self.current_url = url
            self.title = f"Fake page for {url}"
            self.page_source = f"<html><body>Loaded {url}</body></html>"

        def execute_script(self, script: str, *args: Any) -> Any:
            if "document.readyState" in str(script):
                return "complete"
            if "window.innerWidth" in str(script):
                return {
                    "width": self.window_size["width"],
                    "height": self.window_size["height"],
                    "scrollX": 0,
                    "scrollY": 0,
                    "pageWidth": self.window_size["width"],
                    "pageHeight": 1600,
                }
            return None

        def implicitly_wait(self, seconds: float) -> None:
            self.implicit_wait = seconds

        def set_page_load_timeout(self, seconds: float) -> None:
            self.page_load_timeout = seconds

        def set_script_timeout(self, seconds: float) -> None:
            self.script_timeout = seconds

        def set_window_size(self, width: int, height: int) -> None:
            self.window_size = {"width": width, "height": height}

        def get_window_size(self) -> Dict[str, int]:
            return dict(self.window_size)

        def quit(self) -> None:
            self.closed = True
            self.current_url = "closed"

    def _fake_factory(options: Any) -> _FakeDriver:
        return _FakeDriver(options)

    driver_manager = BrowserDriver(
        config={
            "enabled": True,
            "auto_start": True,
            "headless": True,
            "window_size": [1440, 900],
            "timeouts": {
                "implicit_wait_seconds": 0.2,
                "page_load_timeout_seconds": 5,
                "script_timeout_seconds": 5,
            },
            "startup": {"retries": 0},
            "diagnostics": {"include_capabilities": True, "event_history_limit": 20},
            "lifecycle": {"close_owned_driver": True},
        },
        driver_factory=_fake_factory,
    )

    assert driver_manager.has_driver is True
    assert driver_manager.driver is not None
    assert driver_manager.driver.window_size == {"width": 1440, "height": 900}

    health = driver_manager.health()
    assert health["status"] == "success", health
    assert health["data"]["healthy"] is True, health

    state = driver_manager.state()
    assert state["has_driver"] is True, state
    assert state["ready_state"] == "complete", state

    external = _FakeDriver()
    attach = driver_manager.attach(external, owns_driver=False)
    assert attach["status"] == "success", attach
    assert driver_manager.driver is external
    assert driver_manager.owns_driver is False

    detached = driver_manager.detach()
    assert detached is external
    assert driver_manager.has_driver is False

    restart = driver_manager.restart()
    assert restart["status"] == "success", restart
    assert driver_manager.has_driver is True

    close = driver_manager.close()
    assert close["status"] == "success", close
    assert driver_manager.has_driver is False

    print("\n=== Test ran successfully ===\n")
