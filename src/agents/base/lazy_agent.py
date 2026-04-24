"""
Lazy agent wrapper for the Base Agent stack.

This module provides a production-ready deferred-initialization wrapper used by
the base subsystem when agent construction is expensive, optional, or should be
delayed until first real use. The wrapper centralizes lazy initialization,
bounded retry, lifecycle state tracking, optional shared-memory integration, and
safe delegation to the wrapped instance.

Key design goals:
- defer heavyweight initialization until first access
- preserve explicit lifecycle and auditability for init attempts
- integrate with the shared base error taxonomy and helper layer
- support bounded retry with config-driven backoff
- remain thread-safe for concurrent access to the same lazy wrapper
- expose practical observability via stats/history/export helpers
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.base_errors import *
from .utils.base_helpers import *
from .base_memory import BaseMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Lazy Agent")
printer = PrettyPrinter()


@dataclass(frozen=True)
class LazyInitRecord:
    """Immutable record of one initialization attempt."""

    attempt: int
    started_at: str
    finished_at: str
    duration_ms: int
    success: bool
    instance_type: Optional[str] = None
    timed_out: bool = False
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "instance_type": self.instance_type,
            "timed_out": self.timed_out,
            "error": to_json_safe(self.error),
        }


@dataclass(frozen=True)
class LazyAgentStats:
    """Summary view of the lazy wrapper lifecycle."""

    name: str
    request_id: str
    state: str
    initialized: bool
    has_instance: bool
    init_attempts: int
    successful_inits: int
    failed_inits: int
    reset_count: int
    last_duration_ms: Optional[int]
    history_length: int
    delegate_cache_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "request_id": self.request_id,
            "state": self.state,
            "initialized": self.initialized,
            "has_instance": self.has_instance,
            "init_attempts": self.init_attempts,
            "successful_inits": self.successful_inits,
            "failed_inits": self.failed_inits,
            "reset_count": self.reset_count,
            "last_duration_ms": self.last_duration_ms,
            "history_length": self.history_length,
            "delegate_cache_size": self.delegate_cache_size,
        }


class LazyAgent:
    """
    Thread-safe deferred initialization wrapper for a wrapped object or agent.

    The initialization function is invoked only when the wrapped instance is
    first needed, or when explicit initialization is requested.
    """

    VALID_STATES: Tuple[str, ...] = (
        "uninitialized",
        "initializing",
        "ready",
        "failed",
        "reset",
    )

    def __init__(
        self,
        init_fn: Callable[..., Any],
        *,
        init_args: Optional[Iterable[Any]] = None,
        init_kwargs: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        shared_memory: Optional[BaseMemory] = None,
        auto_initialize: Optional[bool] = None,
    ) -> None:
        self.config = load_global_config()
        self.lazy_config = get_config_section("lazy_agent") or {}

        ensure_callable(
            init_fn,
            "init_fn",
            config=self.lazy_config,
            error_cls=BaseValidationError,
            component="LazyAgent",
            operation="__init__",
        )

        if init_kwargs is not None:
            ensure_mapping(
                init_kwargs,
                "init_kwargs",
                config=self.lazy_config,
                error_cls=BaseValidationError,
                component="LazyAgent",
                operation="__init__",
            )

        if shared_memory is not None and not isinstance(shared_memory, BaseMemory):
            raise BaseValidationError(
                "'shared_memory' must be a BaseMemory instance when provided.",
                self.lazy_config,
                component="LazyAgent",
                operation="__init__",
                context={"received_type": type(shared_memory).__name__},
            )

        self._lock = RLock()
        self._init_fn = init_fn
        self._init_args = tuple(init_args or ())
        self._init_kwargs = dict(init_kwargs or {})
        self._instance: Optional[Any] = None
        self._state = "uninitialized"
        self._initialization_error: Optional[BaseError] = None
        self._history: List[LazyInitRecord] = []
        self._delegate_cache: Dict[str, Any] = {}

        self._request_id = generate_request_id("lazy", include_timestamp=True)
        self._created_at = utc_now_iso()
        self._initialized_at: Optional[str] = None
        self._last_duration_ms: Optional[int] = None
        self._init_attempts = 0
        self._successful_inits = 0
        self._failed_inits = 0
        self._reset_count = 0

        raw_name = name or getattr(init_fn, "__name__", type(init_fn).__name__)
        self.name = ensure_non_empty_string(
            raw_name,
            "name",
            config=self.lazy_config,
            error_cls=BaseValidationError,
            component="LazyAgent",
            operation="__init__",
        )

        self.logger = get_logger(f"LazyAgent[{self.name}]")

        self.max_init_attempts = coerce_int(
            self.lazy_config.get("max_init_attempts", 1),
            default=1,
            minimum=1,
        )
        self.enable_init_logging = coerce_bool(
            self.lazy_config.get("enable_init_logging", True),
            default=True,
        )
        self.cache_delegate_attributes = coerce_bool(
            self.lazy_config.get("cache_delegate_attributes", False),
            default=False,
        )
        self.allow_reset = coerce_bool(
            self.lazy_config.get("allow_reset", True),
            default=True,
        )
        self.enable_memory_integration = coerce_bool(
            self.lazy_config.get("enable_memory_integration", False),
            default=False,
        )
        self.memory_namespace = normalize_identifier(
            self.lazy_config.get("memory_namespace", "lazy_agent"),
            lowercase=True,
            separator="_",
            max_length=120,
        )
        self.auto_initialize = coerce_bool(
            auto_initialize if auto_initialize is not None else self.lazy_config.get("auto_initialize", False),
            default=False,
        )

        timeout_raw = self.lazy_config.get("timeout_seconds", None)
        self.timeout_seconds: Optional[float]
        if timeout_raw in (None, "", "none", "None"):
            self.timeout_seconds = None
        else:
            self.timeout_seconds = coerce_float(timeout_raw, default=0.0, minimum=0.0)

        self.backoff_policy = BackoffPolicy(
            initial_delay=coerce_float(
                self.lazy_config.get("retry_initial_delay", 0.25),
                default=0.25,
                minimum=0.0,
            ),
            multiplier=coerce_float(
                self.lazy_config.get("retry_multiplier", 2.0),
                default=2.0,
                minimum=1.0,
            ),
            max_delay=coerce_float(
                self.lazy_config.get("retry_max_delay", 5.0),
                default=5.0,
                minimum=0.0,
            ),
            jitter_ratio=coerce_float(
                self.lazy_config.get("retry_jitter_ratio", 0.0),
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            ),
        )

        self.shared_memory: Optional[BaseMemory] = shared_memory
        if self.enable_memory_integration and self.shared_memory is None:
            try:
                self.shared_memory = BaseMemory()
            except Exception as exc:
                wrapped = BaseRuntimeError.wrap(
                    exc,
                    message="LazyAgent failed to initialize optional shared memory integration.",
                    config=self.lazy_config,
                    component="LazyAgent",
                    operation="__init__",
                    context={"name": self.name},
                )
                if self.enable_init_logging:
                    wrapped.log()

        self._sync_memory_state()

        if self.enable_init_logging:
            self.logger.info("Lazy Agent successfully initialized")

        if self.auto_initialize:
            self.initialize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_state(self, state: str) -> None:
        ensure_one_of(
            state,
            self.VALID_STATES,
            "state",
            config=self.lazy_config,
            error_cls=BaseValidationError,
            component="LazyAgent",
            operation="_set_state",
        )
        self._state = state

    def _record_history(
        self,
        *,
        attempt: int,
        started_at: str,
        finished_at: str,
        duration_ms: int,
        success: bool,
        instance_type: Optional[str] = None,
        timed_out: bool = False,
        error: Optional[BaseError] = None,
    ) -> None:
        record = LazyInitRecord(
            attempt=attempt,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            success=success,
            instance_type=instance_type,
            timed_out=timed_out,
            error=error.to_dict(include_traceback=False) if error is not None else None,
        )
        self._history.append(record)

    def _sync_memory_state(self) -> None:
        if self.shared_memory is None:
            return

        payload = drop_none_values(
            {
                "name": self.name,
                "request_id": self._request_id,
                "state": self._state,
                "created_at": self._created_at,
                "initialized_at": self._initialized_at,
                "initialized": self.is_initialized,
                "wrapped_type": type(self._instance).__name__ if self._instance is not None else None,
                "init_attempts": self._init_attempts,
                "successful_inits": self._successful_inits,
                "failed_inits": self._failed_inits,
                "reset_count": self._reset_count,
                "last_duration_ms": self._last_duration_ms,
                "last_error": self._initialization_error.to_dict() if self._initialization_error else None,
            },
            recursive=True,
            drop_empty=False,
        )

        try:
            self.shared_memory.put(
                key=f"lazy_agent:{self.name}",
                value=payload,
                namespace=self.memory_namespace,
                metadata={
                    "component": "LazyAgent",
                    "request_id": self._request_id,
                },
                tags=["lazy_agent", self._state, normalize_identifier(self.name)],
                persistent=False,
            )
        except Exception as exc:
            wrapped = BaseRuntimeError.wrap(
                exc,
                message="LazyAgent failed to synchronize state to shared memory.",
                config=self.lazy_config,
                component="LazyAgent",
                operation="_sync_memory_state",
                context={"name": self.name, "memory_namespace": self.memory_namespace},
            )
            if self.enable_init_logging:
                wrapped.log()

    def _cleanup_wrapped_instance(self, instance: Any) -> None:
        for method_name in ("close", "shutdown", "cleanup"):
            method = getattr(instance, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:
                    wrapped = BaseRuntimeError.wrap(
                        exc,
                        message=f"Wrapped instance cleanup failed via '{method_name}'.",
                        config=self.lazy_config,
                        component="LazyAgent",
                        operation="reset",
                        context={"name": self.name, "cleanup_method": method_name},
                    )
                    if self.enable_init_logging:
                        wrapped.log()
                break

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def initialize(self, *, force: bool = False) -> Any:
        with self._lock:
            if self._instance is not None and not force:
                return self._instance

            if force and self._instance is not None:
                ensure_state(
                    self.allow_reset,
                    "Forced reinitialization is disabled for this LazyAgent.",
                    config=self.lazy_config,
                    error_cls=BaseStateError,
                    component="LazyAgent",
                    operation="initialize",
                )
                self.reset(reinitialize=False)

            if self._state == "initializing":
                raise BaseStateError(
                    "LazyAgent is already initializing.",
                    self.lazy_config,
                    component="LazyAgent",
                    operation="initialize",
                    context={"name": self.name},
                )

            self._set_state("initializing")
            self._delegate_cache.clear()
            self._sync_memory_state()

        last_error: Optional[BaseError] = None

        for attempt in range(1, self.max_init_attempts + 1):
            stopwatch = Stopwatch(start_immediately=True)
            started_at = utc_now_iso()

            try:
                instance = self._init_fn(*self._init_args, **self._init_kwargs)

                if instance is None:
                    raise BaseInitializationError(
                        "Initialization function returned None.",
                        self.lazy_config,
                        component="LazyAgent",
                        operation="initialize",
                        context={"name": self.name, "attempt": attempt},
                    )

                duration_s = stopwatch.stop()
                duration_ms = stopwatch.elapsed_ms
                timed_out = (
                    self.timeout_seconds is not None
                    and duration_s > self.timeout_seconds
                )

                if timed_out:
                    raise BaseTimeoutError(
                        "LazyAgent initialization exceeded the configured timeout.",
                        self.lazy_config,
                        component="LazyAgent",
                        operation="initialize",
                        context={
                            "name": self.name,
                            "attempt": attempt,
                            "duration_seconds": round(duration_s, 6),
                            "timeout_seconds": self.timeout_seconds,
                        },
                    )

                with self._lock:
                    self._instance = instance
                    self._initialized_at = utc_now_iso()
                    self._last_duration_ms = duration_ms
                    self._init_attempts += 1
                    self._successful_inits += 1
                    self._initialization_error = None
                    self._set_state("ready")
                    self._record_history(
                        attempt=attempt,
                        started_at=started_at,
                        finished_at=utc_now_iso(),
                        duration_ms=duration_ms,
                        success=True,
                        instance_type=type(instance).__name__,
                        timed_out=False,
                    )
                    self._sync_memory_state()

                if self.enable_init_logging:
                    self.logger.info(
                        "Wrapped object initialized successfully "
                        f"(attempt={attempt}, duration_ms={duration_ms}, type={type(instance).__name__})"
                    )
                return instance

            except Exception as exc:
                duration_ms = stopwatch.elapsed_ms
                wrapped = (
                    exc
                    if isinstance(exc, BaseError)
                    else BaseInitializationError.wrap(
                        exc,
                        message="LazyAgent failed to initialize the wrapped object.",
                        config=self.lazy_config,
                        component="LazyAgent",
                        operation="initialize",
                        context={
                            "name": self.name,
                            "attempt": attempt,
                            "init_fn": getattr(self._init_fn, "__name__", type(self._init_fn).__name__),
                        },
                    )
                )
                last_error = wrapped

                with self._lock:
                    self._init_attempts += 1
                    self._failed_inits += 1
                    self._last_duration_ms = duration_ms
                    self._initialization_error = wrapped
                    self._record_history(
                        attempt=attempt,
                        started_at=started_at,
                        finished_at=utc_now_iso(),
                        duration_ms=duration_ms,
                        success=False,
                        timed_out=isinstance(wrapped, BaseTimeoutError),
                        error=wrapped,
                    )
                    self._set_state("failed" if attempt >= self.max_init_attempts else "uninitialized")
                    self._sync_memory_state()

                if self.enable_init_logging:
                    wrapped.log()

                if attempt >= self.max_init_attempts:
                    raise wrapped from exc

                delay = self.backoff_policy.compute_delay(attempt)
                if delay > 0:
                    time.sleep(delay)

        raise BaseInitializationError(
            "LazyAgent failed to initialize after exhausting retry attempts.",
            self.lazy_config,
            component="LazyAgent",
            operation="initialize",
            context={"name": self.name},
            cause=last_error,
        )

    def reset(self, *, reinitialize: bool = False) -> Optional[Any]:
        ensure_state(
            self.allow_reset,
            "Reset is disabled for this LazyAgent.",
            config=self.lazy_config,
            error_cls=BaseStateError,
            component="LazyAgent",
            operation="reset",
        )

        with self._lock:
            previous_instance = self._instance
            self._instance = None
            self._delegate_cache.clear()
            self._initialization_error = None
            self._initialized_at = None
            self._last_duration_ms = None
            self._reset_count += 1
            self._set_state("reset")
            self._sync_memory_state()

        if previous_instance is not None:
            self._cleanup_wrapped_instance(previous_instance)

        with self._lock:
            self._set_state("uninitialized")
            self._sync_memory_state()

        if reinitialize:
            return self.initialize()
        return None

    def peek(self) -> Optional[Any]:
        with self._lock:
            return self._instance

    # ------------------------------------------------------------------
    # Accessors / observability
    # ------------------------------------------------------------------
    @property
    def is_initialized(self) -> bool:
        with self._lock:
            return self._instance is not None and self._state == "ready"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def initialization_error(self) -> Optional[BaseError]:
        with self._lock:
            return self._initialization_error

    @property
    def wrapped_instance(self) -> Any:
        return self.initialize()

    def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._history)
        if limit is not None:
            limit = coerce_int(limit, default=len(records), minimum=1)
            records = records[-limit:]
        return [record.to_dict() for record in records]

    def stats(self) -> LazyAgentStats:
        with self._lock:
            return LazyAgentStats(
                name=self.name,
                request_id=self._request_id,
                state=self._state,
                initialized=self.is_initialized,
                has_instance=self._instance is not None,
                init_attempts=self._init_attempts,
                successful_inits=self._successful_inits,
                failed_inits=self._failed_inits,
                reset_count=self._reset_count,
                last_duration_ms=self._last_duration_ms,
                history_length=len(self._history),
                delegate_cache_size=len(self._delegate_cache),
            )

    def to_dict(self, *, include_history: bool = True, include_instance_repr: bool = True) -> Dict[str, Any]:
        with self._lock:
            payload = {
                "name": self.name,
                "request_id": self._request_id,
                "state": self._state,
                "created_at": self._created_at,
                "initialized_at": self._initialized_at,
                "init_fn": getattr(self._init_fn, "__name__", type(self._init_fn).__name__),
                "initialized": self.is_initialized,
                "stats": self.stats().to_dict(),
                "last_error": self._initialization_error.to_dict() if self._initialization_error else None,
            }
            if include_instance_repr and self._instance is not None:
                payload["instance_repr"] = safe_repr(self._instance)
                payload["instance_type"] = type(self._instance).__name__
            if include_history:
                payload["history"] = self.history()
            return to_json_safe(payload)

    def to_json(self, *, pretty: bool = True) -> str:
        return json_dumps(self.to_dict(), pretty=pretty)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        with self._lock:
            if self.cache_delegate_attributes and name in self._delegate_cache:
                return self._delegate_cache[name]

        instance = self.initialize()

        try:
            value = getattr(instance, name)
        except AttributeError as exc:
            raise AttributeError(
                f"Attribute '{name}' not found on lazily initialized object "
                f"of type '{type(instance).__name__}'."
            ) from exc

        with self._lock:
            if self.cache_delegate_attributes:
                self._delegate_cache[name] = value

        return value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        instance = self.initialize()
        if not callable(instance):
            raise BaseStateError(
                "Wrapped instance is not callable.",
                self.lazy_config,
                component="LazyAgent",
                operation="__call__",
                context={"wrapped_type": type(instance).__name__},
            )
        return instance(*args, **kwargs)

    def __dir__(self) -> List[str]:
        base_dir = set(super().__dir__())
        with self._lock:
            if self._instance is not None:
                base_dir.update(dir(self._instance))
        return sorted(base_dir)

    def __repr__(self) -> str:
        with self._lock:
            if self._instance is not None:
                return (
                    f"<LazyAgent name='{self.name}' "
                    f"state='{self._state}' "
                    f"wrapped='{type(self._instance).__name__}'>"
                )
            return (
                f"<LazyAgent name='{self.name}' "
                f"state='{self._state}' "
                f"init_fn='{getattr(self._init_fn, '__name__', type(self._init_fn).__name__)}'>"
            )


# ====================== Usage Example / Self Test ======================
if __name__ == "__main__":
    print("\n=== Running Lazy Agent ===\n")
    printer.status("TEST", "Lazy Agent initialized", "info")

    class DemoAgent:
        def __init__(self) -> None:
            self.name = "demo_agent"
            self.ready = True

        def perform_task(self, task: str) -> str:
            return f"processed::{task}"

        def close(self) -> None:
            self.ready = False

    def build_demo_agent() -> DemoAgent:
        return DemoAgent()

    agent = LazyAgent(init_fn=build_demo_agent)
    print(agent)

    printer.pretty("STATS_BEFORE_INIT", agent.stats().to_dict(), "info")
    printer.status("ACCESS", "Triggering lazy initialization via attribute access", "info")

    result = agent.perform_task("hello world")
    printer.pretty("TASK_RESULT", {"result": result}, "success")
    printer.pretty("STATS_AFTER_INIT", agent.stats().to_dict(), "success")
    printer.pretty("HISTORY", agent.history(), "success")

    agent.reset(reinitialize=True)
    printer.pretty("STATS_AFTER_RESET", agent.stats().to_dict(), "success")
    printer.pretty("LAZY_AGENT_JSON", agent.to_dict(include_history=True), "success")

    print("\n=== Test ran successfully ===\n")