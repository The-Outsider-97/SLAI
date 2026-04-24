"""
Issue recovery and remediation orchestration for the Base Agent subsystem.

This module provides the production-grade issue handling layer used to classify,
route, record, and attempt recovery for operational failures raised while a
base agent performs work. It standardizes issue metadata, coordinates targeted
recovery strategies, preserves bounded history, integrates with shared memory
when available, and exposes backward-compatible standalone recovery functions
for modules that still call handlers directly.

The implementation is intentionally pragmatic: it can act as a lightweight
recovery utility for simple agents, while also supporting richer agents that
expose optional hooks such as memory cleanup, CPU fallback, lightweight mode,
input reshaping, or alternative implementations.

Design goals:
- deterministic issue classification and bounded operational history
- structured, explicit recovery semantics with audit-friendly outputs
- resilient integration with the existing base config/error/helper imports
- backward-compatible standalone handlers and registry access
- configuration-driven behavior through base_config.yaml
- practical interoperability with shared memory and agent helper hooks
"""

from __future__ import annotations

import time
import difflib
import traceback
import inspect
import importlib

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, Deque, Dict, List, Optional, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.base_errors import *
from .utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Issue Handler")
printer = PrettyPrinter()

IssueHandlerError = BaseError
IssueHandlerConfigurationError = BaseConfigurationError
IssueHandlerInitializationError = BaseInitializationError
IssueHandlerValidationError = BaseValidationError
IssueHandlerStateError = BaseStateError
IssueHandlerRuntimeError = BaseRuntimeError
IssueHandlerIOError = BaseIOError

IssueHandlerCallable = Callable[[Any, Any, Mapping[str, Any], "IssueHandler"], "IssueOutcome"]


@dataclass
class RecoveryAttempt:
    strategy: str
    success: bool
    message: str
    timestamp: str = field(default_factory=utc_now_iso)
    details: Dict[str, Any] = field(default_factory=dict)
    result_preview: Any = None
    exception_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return drop_none_values(
            {
                "strategy": self.strategy,
                "success": self.success,
                "message": self.message,
                "timestamp": self.timestamp,
                "details": to_json_safe(self.details),
                "result_preview": to_json_safe(self.result_preview),
                "exception_type": self.exception_type,
            },
            recursive=True,
            drop_empty=False,
        )


@dataclass
class IssueOutcome:
    handled: bool
    recovered: bool
    strategy: str
    status: str
    reason: str
    result: Any = None
    handler_name: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False
    issue_key: Optional[str] = None
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self, *, redact: bool = False) -> Dict[str, Any]:
        metadata = redact_mapping(self.metadata) if redact else self.metadata
        return drop_none_values(
            {
                "handled": self.handled,
                "recovered": self.recovered,
                "strategy": self.strategy,
                "status": self.status,
                "reason": self.reason,
                "result": to_json_safe(self.result),
                "handler_name": self.handler_name,
                "error_type": self.error_type,
                "error_message": self.error_message,
                "retryable": self.retryable,
                "issue_key": self.issue_key,
                "attempts": [attempt.to_dict() for attempt in self.attempts],
                "metadata": to_json_safe(metadata),
                "timestamp": self.timestamp,
            },
            recursive=True,
            drop_empty=False,
        )


@dataclass(frozen=True)
class IssueStats:
    total_events: int
    handled_events: int
    recovered_events: int
    failed_events: int
    total_attempts: int
    registry_size: int
    history_length: int
    shared_memory_records: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "handled_events": self.handled_events,
            "recovered_events": self.recovered_events,
            "failed_events": self.failed_events,
            "total_attempts": self.total_attempts,
            "registry_size": self.registry_size,
            "history_length": self.history_length,
            "shared_memory_records": self.shared_memory_records,
        }


class IssueHandler:
    """Configuration-driven issue recovery orchestrator for base agents."""

    def __init__(self) -> None:
        self.config = load_global_config()
        self.issue_config = get_config_section("issue_handler") or {}
        self._lock = RLock()

        self.enable_history = self._get_config_bool("enable_history", True)
        self.history_limit = self._get_config_int("history_limit", 250, minimum=1)
        self.enable_shared_memory_learning = self._get_config_bool("enable_shared_memory_learning", True)
        self.shared_memory_error_key_prefix = self._get_config_str("shared_memory_error_key_prefix", "errors")
        self.error_similarity_threshold = self._get_config_float("error_similarity_threshold", 0.72, minimum=0.0)
        self.max_retry_attempts = self._get_config_int("max_retry_attempts", 3, minimum=1)
        self.max_solution_attempts = self._get_config_int("max_solution_attempts", 3, minimum=1)
        self.runtime_retry_delay_seconds = self._get_config_float("runtime_retry_delay_seconds", 1.0, minimum=0.0)
        self.memory_trim_ratio = self._get_config_float("memory_trim_ratio", 0.5, minimum=0.05)
        self.timeout_simplified_dict_items = self._get_config_int("timeout_simplified_dict_items", 3, minimum=1)
        self.unicode_strip_non_ascii = self._get_config_bool("unicode_strip_non_ascii", True)
        self.enable_traceback_capture = self._get_config_bool("enable_traceback_capture", True)
        self.export_redact_secrets = self._get_config_bool("export_redact_secrets", False)
        self.max_error_message_length = self._get_config_int("max_error_message_length", 500, minimum=64)
        self.resource_wait_schedule = self._get_config_float_sequence("resource_wait_schedule", [2.0, 5.0, 10.0])
        self.network_backoff_policy = BackoffPolicy(
            initial_delay=self._get_config_float("network_backoff_initial_delay", 1.0, minimum=0.0),
            multiplier=self._get_config_float("network_backoff_multiplier", 2.0, minimum=1.0),
            max_delay=self._get_config_float("network_backoff_max_delay", 30.0, minimum=0.0),
            jitter_ratio=self._get_config_float("network_backoff_jitter_ratio", 0.0, minimum=0.0),
        )

        self._registry: Dict[str, IssueHandlerCallable] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self._stats = defaultdict(int)
        self._register_default_handlers()
        logger.info("Issue Handler successfully initialized")

    def _get_config_bool(self, key: str, default: bool) -> bool:
        return coerce_bool(self.issue_config.get(key, default), default)

    def _get_config_int(self, key: str, default: int, *, minimum: Optional[int] = None) -> int:
        value = coerce_int(self.issue_config.get(key, default), default, minimum=minimum)
        if minimum is not None and value < minimum:
            raise IssueHandlerConfigurationError(
                f"Configuration value '{key}' must be >= {minimum}.",
                self.issue_config,
                operation="configuration",
                context={"key": key, "value": value, "minimum": minimum},
            )
        return value

    def _get_config_float(self, key: str, default: float, *, minimum: Optional[float] = None) -> float:
        value = coerce_float(self.issue_config.get(key, default), default, minimum=minimum)
        if minimum is not None and value < minimum:
            raise IssueHandlerConfigurationError(
                f"Configuration value '{key}' must be >= {minimum}.",
                self.issue_config,
                operation="configuration",
                context={"key": key, "value": value, "minimum": minimum},
            )
        return value

    def _get_config_str(self, key: str, default: str) -> str:
        return ensure_non_empty_string(self.issue_config.get(key, default), key)

    def _get_config_float_sequence(self, key: str, default: Sequence[float]) -> List[float]:
        raw = self.issue_config.get(key, list(default))
        values = (
            parse_delimited_text(raw)
            if isinstance(raw, str)
            else list(raw)
            if isinstance(raw, Sequence)
            else list(default)
        )
        return [coerce_float(item, 0.0, minimum=0.0) for item in values] or list(default)

    def _register_default_handlers(self) -> None:
        self.register_handler("name 'inspect' is not defined", handle_missing_inspect_error)
        self.register_handler("NameError", handle_missing_inspect_error)
        self.register_handler("RuntimeError", handle_runtime_error)
        self.register_handler("unicode", handle_unicode_emoji_error)
        self.register_handler("emoji", handle_unicode_emoji_error)
        self.register_handler("UnicodeEncodeError", handle_unicode_emoji_error)
        self.register_handler("UnicodeDecodeError", handle_unicode_emoji_error)
        self.register_handler("network", handle_network_error)
        self.register_handler("ConnectionError", handle_network_error)
        self.register_handler("Timeout", handle_timeout_error)
        self.register_handler("memory", handle_memory_error)
        self.register_handler("MemoryError", handle_memory_error)
        self.register_handler("dependency", handle_common_dependency_error)
        self.register_handler("ImportError", handle_common_dependency_error)
        self.register_handler("resource", handle_resource_constraint)
        self.register_handler("past_error", handle_similar_past_error)
        self.register_handler("Exception", handle_similar_past_error)

    def register_handler(self, pattern: str, handler: IssueHandlerCallable) -> None:
        self._registry[ensure_non_empty_string(pattern, "pattern")] = ensure_callable(handler, "handler")

    def unregister_handler(self, pattern: str) -> bool:
        return self._registry.pop(pattern, None) is not None

    @property
    def registry(self) -> Dict[str, IssueHandlerCallable]:
        return dict(self._registry)

    def _safe_agent_name(self, agent: Any) -> str:
        return ensure_text(getattr(agent, "name", agent.__class__.__name__))

    def _truncate_error_message(self, message: Any) -> str:
        text = ensure_text(message)
        return truncate_string(text, self.max_error_message_length)

    def build_error_info(
        self,
        error: Optional[BaseException] = None,
        *,
        error_info: Optional[Mapping[str, Any]] = None,
        task_data: Any = None,
    ) -> Dict[str, Any]:
        payload = dict(error_info or {})
        if error is not None:
            payload.setdefault("error_type", error.__class__.__name__)
            payload.setdefault("error_message", str(error))
            payload.setdefault("timestamp", utc_now_iso())
            if self.enable_traceback_capture:
                payload.setdefault(
                    "traceback",
                    "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                )
        else:
            payload.setdefault("error_type", payload.get("error_type", "Exception"))
            payload.setdefault("error_message", payload.get("error_message", "Unknown error"))
            payload.setdefault("timestamp", payload.get("timestamp", utc_now_iso()))

        payload.setdefault("task_preview", safe_repr(task_data, max_length=160))
        payload["error_message"] = self._truncate_error_message(payload.get("error_message", ""))
        return payload

    def _issue_key(self, agent: Any) -> str:
        return f"{self.shared_memory_error_key_prefix}:{self._safe_agent_name(agent)}"

    def _select_handlers(self, error_info: Mapping[str, Any]) -> List[Tuple[str, IssueHandlerCallable]]:
        error_type = ensure_text(error_info.get("error_type", ""))
        error_message = ensure_text(error_info.get("error_message", ""))
        haystack = f"{error_type} {error_message}".lower()
        matched: List[Tuple[str, IssueHandlerCallable]] = []
        seen: set[int] = set()
        for pattern, handler in self._registry.items():
            if pattern.lower() in haystack or pattern == error_type:
                if id(handler) in seen:
                    continue
                seen.add(id(handler))
                matched.append((pattern, handler))
        if not matched and "Exception" in self._registry:
            matched.append(("Exception", self._registry["Exception"]))
        return matched

    def _is_retryable_error(self, error_info: Mapping[str, Any]) -> bool:
        haystack = f"{ensure_text(error_info.get('error_type', ''))} {ensure_text(error_info.get('error_message', ''))}".lower()
        return any(token in haystack for token in ("timeout", "connection", "network", "busy", "temporary", "resource"))

    def _invoke_task(self, agent: Any, task_data: Any) -> Any:
        perform_task = getattr(agent, "perform_task", None)
        if not callable(perform_task):
            raise IssueHandlerValidationError(
                "Agent does not expose a callable 'perform_task' method.",
                self.issue_config,
                operation="perform_task",
                context={"agent_type": type(agent).__name__},
            )
        return perform_task(task_data)

    def _success_outcome(
        self,
        *,
        strategy: str,
        reason: str,
        result: Any,
        error_info: Mapping[str, Any],
        attempts: Optional[List[RecoveryAttempt]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        handler_name: Optional[str] = None,
    ) -> IssueOutcome:
        return IssueOutcome(
            handled=True,
            recovered=True,
            strategy=strategy,
            status="recovered",
            reason=reason,
            result=result,
            handler_name=handler_name,
            error_type=ensure_text(error_info.get("error_type", "Exception")),
            error_message=ensure_text(error_info.get("error_message", "")),
            retryable=self._is_retryable_error(error_info),
            issue_key=stable_fingerprint({"type": error_info.get("error_type"), "message": error_info.get("error_message")}, length=24),
            attempts=list(attempts or []),
            metadata=dict(metadata or {}),
        )

    def _failure_outcome(
        self,
        *,
        strategy: str,
        reason: str,
        error_info: Mapping[str, Any],
        handled: bool = True,
        attempts: Optional[List[RecoveryAttempt]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        handler_name: Optional[str] = None,
    ) -> IssueOutcome:
        return IssueOutcome(
            handled=handled,
            recovered=False,
            strategy=strategy,
            status="failed" if handled else "ignored",
            reason=reason,
            handler_name=handler_name,
            error_type=ensure_text(error_info.get("error_type", "Exception")),
            error_message=ensure_text(error_info.get("error_message", "")),
            retryable=self._is_retryable_error(error_info),
            issue_key=stable_fingerprint({"type": error_info.get("error_type"), "message": error_info.get("error_message")}, length=24),
            attempts=list(attempts or []),
            metadata=dict(metadata or {}),
        )

    def _record_outcome(self, outcome: IssueOutcome) -> None:
        if self.enable_history:
            self._history.append(outcome.to_dict(redact=self.export_redact_secrets))
        self._stats["total_events"] += 1
        self._stats["total_attempts"] += len(outcome.attempts)
        if outcome.handled:
            self._stats["handled_events"] += 1
        if outcome.recovered:
            self._stats["recovered_events"] += 1
        else:
            self._stats["failed_events"] += 1

    def _store_in_shared_memory(self, agent: Any, outcome: IssueOutcome) -> None:
        if not self.enable_shared_memory_learning or not hasattr(agent, "shared_memory"):
            return
        shared_memory = getattr(agent, "shared_memory")
        issue_key = self._issue_key(agent)
        record = outcome.to_dict(redact=self.export_redact_secrets)
        try:
            if hasattr(shared_memory, "get") and hasattr(shared_memory, "put"):
                existing = shared_memory.get(issue_key, []) or []
                if not isinstance(existing, list):
                    existing = [existing]
                existing.append(record)
                if len(existing) > self.history_limit:
                    existing = existing[-self.history_limit :]
                shared_memory.put(issue_key, existing)
                self._stats["shared_memory_records"] += 1
            elif isinstance(shared_memory, MutableMapping):
                existing = shared_memory.get(issue_key, []) or []
                if not isinstance(existing, list):
                    existing = [existing]
                existing.append(record)
                if len(existing) > self.history_limit:
                    existing = existing[-self.history_limit :]
                shared_memory[issue_key] = existing
                self._stats["shared_memory_records"] += 1
        except Exception as exc:
            logger.warning(f"Failed to store issue outcome in shared memory: {exc}")

    def handle_issue(
        self,
        agent: Any,
        task_data: Any,
        *,
        error: Optional[BaseException] = None,
        error_info: Optional[Mapping[str, Any]] = None,
        allow_fallback_to_past_error: bool = True,
    ) -> Dict[str, Any]:
        normalized_error = self.build_error_info(error, error_info=error_info, task_data=task_data)
        handlers = self._select_handlers(normalized_error)
        if allow_fallback_to_past_error and not any(handler is handle_similar_past_error for _, handler in handlers):
            handlers.append(("past_error", handle_similar_past_error))

        last_failure: Optional[IssueOutcome] = None
        for pattern, strategy_handler in handlers:
            try:
                outcome = strategy_handler(agent, task_data, normalized_error, self)
                outcome.handler_name = strategy_handler.__name__
                if not outcome.strategy:
                    outcome.strategy = pattern
                self._record_outcome(outcome)
                self._store_in_shared_memory(agent, outcome)
                if outcome.recovered:
                    return outcome.to_dict(redact=self.export_redact_secrets)
                last_failure = outcome
            except Exception as exc:
                wrapped = IssueHandlerRuntimeError.wrap(
                    exc,
                    message="Issue handler strategy execution failed unexpectedly.",
                    config=self.issue_config,
                    operation=strategy_handler.__name__,
                    context={"pattern": pattern},
                )
                logger.error(wrapped.message)
                last_failure = self._failure_outcome(
                    strategy=pattern,
                    reason=wrapped.message,
                    error_info=normalized_error,
                    handled=True,
                    attempts=[
                        RecoveryAttempt(
                            strategy=strategy_handler.__name__,
                            success=False,
                            message="Handler raised an unexpected exception.",
                            details=wrapped.to_dict(),
                            exception_type=type(exc).__name__,
                        )
                    ],
                    handler_name=strategy_handler.__name__,
                )
                self._record_outcome(last_failure)

        if last_failure is None:
            last_failure = self._failure_outcome(
                strategy="unmatched",
                reason="No registered issue handler matched the error.",
                error_info=normalized_error,
                handled=False,
            )
            self._record_outcome(last_failure)

        self._store_in_shared_memory(agent, last_failure)
        return last_failure.to_dict(redact=self.export_redact_secrets)

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        count = coerce_int(limit, 20, minimum=1)
        return list(self._history)[-count:]

    def export_history_json(self, *, pretty: bool = True) -> str:
        return json_dumps(self.recent_history(self.history_limit), pretty=pretty)

    def save_history(self, path: str) -> str:
        target = Path(ensure_non_empty_string(path, "path"))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.export_history_json(pretty=True), encoding="utf-8")
            return str(target)
        except Exception as exc:
            raise IssueHandlerIOError.wrap(
                exc,
                message="Failed to save issue handler history.",
                config=self.issue_config,
                operation="save_history",
                context={"path": str(target)},
            ) from exc

    def clear_history(self) -> None:
        self._history.clear()

    def stats(self) -> Dict[str, Any]:
        summary = IssueStats(
            total_events=int(self._stats.get("total_events", 0)),
            handled_events=int(self._stats.get("handled_events", 0)),
            recovered_events=int(self._stats.get("recovered_events", 0)),
            failed_events=int(self._stats.get("failed_events", 0)),
            total_attempts=int(self._stats.get("total_attempts", 0)),
            registry_size=len(self._registry),
            history_length=len(self._history),
            shared_memory_records=int(self._stats.get("shared_memory_records", 0)),
        )
        return summary.to_dict()


# ---------------------------------------------------------------------------
# Standalone strategy helpers
# ---------------------------------------------------------------------------
def _coerce_error_mapping(error_info: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(ensure_mapping(error_info, "error_info"))


def _iter_string_leaves(value: Any) -> List[Tuple[Tuple[Any, ...], str]]:
    leaves: List[Tuple[Tuple[Any, ...], str]] = []

    def _walk(current: Any, path: Tuple[Any, ...]) -> None:
        if isinstance(current, str):
            leaves.append((path, current))
        elif isinstance(current, Mapping):
            for key, nested in current.items():
                _walk(nested, path + (key,))
        elif isinstance(current, list):
            for index, nested in enumerate(current):
                _walk(nested, path + (index,))

    _walk(value, ())
    return leaves


def _replace_at_path(root: Any, path: Tuple[Any, ...], value: Any) -> Any:
    if not path:
        return value
    head, *tail = path
    if isinstance(root, Mapping):
        updated = dict(root)
        updated[head] = _replace_at_path(updated[head], tuple(tail), value)
        return updated
    if isinstance(root, list):
        updated_list = list(root)
        updated_list[int(head)] = _replace_at_path(updated_list[int(head)], tuple(tail), value)
        return updated_list
    return root


def _result_not_applicable(reason: str, error_info: Mapping[str, Any], handler: IssueHandler, strategy: str) -> IssueOutcome:
    return handler._failure_outcome(strategy=strategy, reason=reason, error_info=error_info, handled=False)


def handle_unicode_emoji_error(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    error_type = ensure_text(info.get("error_type", ""))
    error_message = ensure_text(info.get("error_message", "")).lower()
    strategy = "unicode_emoji_recovery"
    is_unicode_error = error_type in {"UnicodeEncodeError", "UnicodeDecodeError"}
    mentions_emoji = any(token in error_message for token in ("emoji", "unicode", "codec"))
    if not (is_unicode_error or mentions_emoji):
        return _result_not_applicable("Error not identified as unicode/emoji issue.", info, handler, strategy)

    def _clean_text(text: str) -> str:
        if not handler.unicode_strip_non_ascii:
            return normalize_text(text)
        return text.encode("ascii", "ignore").decode("ascii")

    cleaned_input = task_data
    changed = False
    for path, original in _iter_string_leaves(task_data):
        cleaned = _clean_text(original)
        if cleaned != original:
            cleaned_input = _replace_at_path(cleaned_input, path, cleaned)
            changed = True

    if not changed:
        return handler._failure_outcome(
            strategy=strategy,
            reason="No changes were made after unicode sanitization.",
            error_info=info,
            attempts=[RecoveryAttempt(strategy=strategy, success=False, message="No sanitizable content changed.")],
            handler_name="handle_unicode_emoji_error",
        )

    try:
        result = handler._invoke_task(agent, cleaned_input)
        return handler._success_outcome(
            strategy=strategy,
            reason="Retried successfully with sanitized unicode input.",
            result=result,
            error_info=info,
            attempts=[RecoveryAttempt(strategy=strategy, success=True, message="Task succeeded after unicode sanitization.")],
            metadata={"changed": True},
            handler_name="handle_unicode_emoji_error",
        )
    except Exception as exc:
        return handler._failure_outcome(
            strategy=strategy,
            reason=f"Retry after cleaning failed: {handler._truncate_error_message(str(exc))}",
            error_info=info,
            attempts=[RecoveryAttempt(strategy=strategy, success=False, message="Retry after sanitization failed.", exception_type=type(exc).__name__)],
            handler_name="handle_unicode_emoji_error",
        )


def handle_network_error(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    haystack = f"{ensure_text(info.get('error_type', ''))} {ensure_text(info.get('error_message', ''))}".lower()
    strategy = "network_backoff_retry"
    if not any(token in haystack for token in ("connection", "timeout", "socket", "http", "https", "network")):
        return _result_not_applicable("Not a network-related error.", info, handler, strategy)

    attempts: List[RecoveryAttempt] = []
    for attempt_number in range(1, handler.max_retry_attempts + 1):
        delay = handler.network_backoff_policy.compute_delay(attempt_number)
        if delay > 0:
            time.sleep(delay)
        try:
            result = handler._invoke_task(agent, task_data)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message=f"Recovered after retry attempt {attempt_number}.", details={"attempt": attempt_number, "delay_seconds": delay}))
            return handler._success_outcome(strategy=strategy, reason=f"Recovered after {attempt_number} network retry attempt(s).", result=result, error_info=info, attempts=attempts, metadata={"attempts": attempt_number}, handler_name="handle_network_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message=f"Network retry attempt {attempt_number} failed.", details={"attempt": attempt_number, "delay_seconds": delay}, exception_type=type(exc).__name__))

    return handler._failure_outcome(strategy=strategy, reason="All network retries failed.", error_info=info, attempts=attempts, handler_name="handle_network_error")


def handle_memory_error(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    haystack = f"{ensure_text(info.get('error_type', ''))} {ensure_text(info.get('error_message', ''))}".lower()
    strategy = "memory_pressure_recovery"
    if not any(token in haystack for token in ("memoryerror", "outofmemory", "cuda out of memory", "resource exhausted", "oom")):
        return _result_not_applicable("Not a memory-related error.", info, handler, strategy)

    attempts: List[RecoveryAttempt] = []
    if isinstance(task_data, str) and len(task_data) > 1:
        reduced_input = task_data[: max(1, int(len(task_data) * handler.memory_trim_ratio))]
        try:
            result = handler._invoke_task(agent, reduced_input)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after trimming string input.", details={"original_length": len(task_data), "reduced_length": len(reduced_input)}))
            return handler._success_outcome(strategy=strategy, reason="Recovered after reducing string input size.", result=result, error_info=info, attempts=attempts, handler_name="handle_memory_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Reduced string input retry failed.", details={"original_length": len(task_data), "reduced_length": len(reduced_input)}, exception_type=type(exc).__name__))

    if isinstance(task_data, Mapping) and len(task_data) > 1:
        trim_count = max(1, int(len(task_data) * handler.memory_trim_ratio))
        reduced_mapping = {key: value for index, (key, value) in enumerate(task_data.items()) if index < trim_count}
        try:
            result = handler._invoke_task(agent, reduced_mapping)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after trimming mapping input.", details={"original_items": len(task_data), "reduced_items": len(reduced_mapping)}))
            return handler._success_outcome(strategy=strategy, reason="Recovered after reducing mapping input size.", result=result, error_info=info, attempts=attempts, handler_name="handle_memory_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Reduced mapping input retry failed.", details={"original_items": len(task_data), "reduced_items": len(reduced_mapping)}, exception_type=type(exc).__name__))

    if callable(getattr(agent, "free_up_memory", None)):
        try:
            agent.free_up_memory()
            result = handler._invoke_task(agent, task_data)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after agent memory cleanup.", details={"hook": "free_up_memory"}))
            return handler._success_outcome(strategy=strategy, reason="Recovered after calling free_up_memory().", result=result, error_info=info, attempts=attempts, handler_name="handle_memory_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Memory cleanup hook retry failed.", details={"hook": "free_up_memory"}, exception_type=type(exc).__name__))

    return handler._failure_outcome(strategy=strategy, reason="Memory error recovery failed.", error_info=info, attempts=attempts, handler_name="handle_memory_error")


def handle_timeout_error(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    haystack = f"{ensure_text(info.get('error_type', ''))} {ensure_text(info.get('error_message', ''))}".lower()
    strategy = "timeout_recovery"
    if not any(token in haystack for token in ("timeout", "timed out", "took too long")):
        return _result_not_applicable("Not a timeout-related error.", info, handler, strategy)

    attempts: List[RecoveryAttempt] = []
    if isinstance(task_data, Mapping) and len(task_data) > handler.timeout_simplified_dict_items:
        simplified = {key: value for index, (key, value) in enumerate(task_data.items()) if index < handler.timeout_simplified_dict_items}
        try:
            result = handler._invoke_task(agent, simplified)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after simplifying mapping input.", details={"kept_items": len(simplified)}))
            return handler._success_outcome(strategy=strategy, reason="Recovered after simplifying input for timeout recovery.", result=result, error_info=info, attempts=attempts, handler_name="handle_timeout_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Simplified input retry failed.", details={"kept_items": len(simplified)}, exception_type=type(exc).__name__))

    if callable(getattr(agent, "use_lightweight_mode", None)):
        enabled = False
        try:
            agent.use_lightweight_mode(True)
            enabled = True
            result = handler._invoke_task(agent, task_data)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered in lightweight mode.", details={"hook": "use_lightweight_mode"}))
            return handler._success_outcome(strategy=strategy, reason="Recovered by switching to lightweight mode.", result=result, error_info=info, attempts=attempts, handler_name="handle_timeout_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Lightweight mode retry failed.", details={"hook": "use_lightweight_mode"}, exception_type=type(exc).__name__))
        finally:
            if enabled:
                try:
                    agent.use_lightweight_mode(False)
                except Exception:
                    pass

    return handler._failure_outcome(strategy=strategy, reason="Timeout recovery failed.", error_info=info, attempts=attempts, handler_name="handle_timeout_error")


def handle_runtime_error(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    haystack = f"{ensure_text(info.get('error_type', ''))} {ensure_text(info.get('error_message', ''))}".lower()
    strategy = "runtime_recovery"
    if "runtimeerror" not in haystack and "runtime" not in haystack:
        return _result_not_applicable("Not a runtime-related error.", info, handler, strategy)

    attempts: List[RecoveryAttempt] = []
    if "shapes cannot be multiplied" in haystack and "mat1" in haystack and "mat2" in haystack and callable(getattr(agent, "reshape_input_for_model", None)):
        try:
            reshaped_task = agent.reshape_input_for_model(task_data)
            if isinstance(reshaped_task, Mapping) and reshaped_task.get("is_reshaped"):
                result = handler._invoke_task(agent, reshaped_task)
                attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after reshape_input_for_model().", details={"hook": "reshape_input_for_model"}))
                return handler._success_outcome(strategy=strategy, reason="Recovered after adjusting input dimensions.", result=result, error_info=info, attempts=attempts, handler_name="handle_runtime_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Reshape-based recovery failed.", details={"hook": "reshape_input_for_model"}, exception_type=type(exc).__name__))

    if handler.runtime_retry_delay_seconds > 0:
        time.sleep(handler.runtime_retry_delay_seconds)

    try:
        result = handler._invoke_task(agent, task_data)
        attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after generic runtime retry.", details={"delay_seconds": handler.runtime_retry_delay_seconds}))
        return handler._success_outcome(strategy=strategy, reason="Recovered after generic runtime retry.", result=result, error_info=info, attempts=attempts, handler_name="handle_runtime_error")
    except Exception as exc:
        attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Generic runtime retry failed.", details={"delay_seconds": handler.runtime_retry_delay_seconds}, exception_type=type(exc).__name__))

    if isinstance(task_data, Mapping) and "data" in task_data:
        simplified = {"data": ensure_text(task_data["data"])[:100]}
        try:
            result = handler._invoke_task(agent, simplified)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after simplified 'data' fallback."))
            return handler._success_outcome(strategy=strategy, reason="Recovered after simplified runtime fallback.", result=result, error_info=info, attempts=attempts, handler_name="handle_runtime_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Simplified runtime fallback failed.", exception_type=type(exc).__name__))

    return handler._failure_outcome(strategy=strategy, reason="Runtime recovery attempts failed.", error_info=info, attempts=attempts, handler_name="handle_runtime_error")


def handle_common_dependency_error(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    error_message = ensure_text(info.get("error_message", "")).lower()
    strategy = "dependency_recovery"
    issues = {
        "no module named": "missing_python_module",
        "cannot import name": "import_error",
        "dll load failed": "missing_binary_dependency",
        "version conflict": "version_conflict",
    }
    detected_issue = None
    for pattern, issue_type in issues.items():
        if pattern in error_message:
            detected_issue = issue_type
            break
    if not detected_issue:
        return _result_not_applicable("No recognized dependency issue.", info, handler, strategy)

    attempts: List[RecoveryAttempt] = []
    if callable(getattr(agent, "use_alternative_implementation", None)):
        try:
            agent.use_alternative_implementation(detected_issue)
            result = handler._invoke_task(agent, task_data)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered using alternative implementation.", details={"detected_issue": detected_issue}))
            return handler._success_outcome(strategy=strategy, reason=f"Recovered from dependency issue '{detected_issue}' via alternative implementation.", result=result, error_info=info, attempts=attempts, handler_name="handle_common_dependency_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Alternative implementation failed.", details={"detected_issue": detected_issue}, exception_type=type(exc).__name__))
        finally:
            try:
                agent.use_alternative_implementation(None)
            except Exception:
                pass

    return handler._failure_outcome(strategy=strategy, reason=f"Dependency issue recovery failed for '{detected_issue}'.", error_info=info, attempts=attempts, handler_name="handle_common_dependency_error")


def handle_resource_constraint(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    error_message = ensure_text(info.get("error_message", "")).lower()
    strategy = "resource_constraint_recovery"
    if not any(token in error_message for token in ("resource unavailable", "resource busy", "gpu", "cuda", "cpu", "memory")):
        return _result_not_applicable("Not a resource-constraint error.", info, handler, strategy)

    attempts: List[RecoveryAttempt] = []
    for wait_time in handler.resource_wait_schedule[: handler.max_retry_attempts]:
        if wait_time > 0:
            time.sleep(wait_time)
        try:
            result = handler._invoke_task(agent, task_data)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after waiting for resource availability.", details={"wait_seconds": wait_time}))
            return handler._success_outcome(strategy=strategy, reason="Recovered after waiting for resource availability.", result=result, error_info=info, attempts=attempts, handler_name="handle_resource_constraint")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Retry after wait failed.", details={"wait_seconds": wait_time}, exception_type=type(exc).__name__))

    if ("gpu" in error_message or "cuda" in error_message) and callable(getattr(agent, "switch_to_cpu", None)):
        switched = False
        try:
            agent.switch_to_cpu()
            switched = True
            result = handler._invoke_task(agent, task_data)
            attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered after switching to CPU execution.", details={"hook": "switch_to_cpu"}))
            return handler._success_outcome(strategy=strategy, reason="Recovered by switching execution to CPU.", result=result, error_info=info, attempts=attempts, handler_name="handle_resource_constraint")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="CPU fallback failed.", details={"hook": "switch_to_cpu"}, exception_type=type(exc).__name__))
        finally:
            if switched and callable(getattr(agent, "switch_to_gpu", None)):
                try:
                    agent.switch_to_gpu()
                except Exception:
                    pass

    return handler._failure_outcome(strategy=strategy, reason="Resource constraint recovery failed.", error_info=info, attempts=attempts, handler_name="handle_resource_constraint")


def handle_similar_past_error(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    strategy = "similar_past_error_recovery"
    current_error = ensure_text(info.get("error_message", ""))
    attempts: List[RecoveryAttempt] = []

    if not handler.enable_shared_memory_learning or not hasattr(agent, "shared_memory"):
        return _result_not_applicable("Shared-memory learning is unavailable for past-error recovery.", info, handler, strategy)

    shared_memory = getattr(agent, "shared_memory")
    issue_key = handler._issue_key(agent)
    try:
        if hasattr(shared_memory, "get"):
            past_errors = shared_memory.get(issue_key, []) or []
        elif isinstance(shared_memory, Mapping):
            past_errors = shared_memory.get(issue_key, []) or []
        else:
            past_errors = []
    except Exception:
        past_errors = []

    if not isinstance(past_errors, list) or not past_errors:
        return _result_not_applicable("No past errors available.", info, handler, strategy)

    similar_solutions: List[Tuple[float, Any]] = []
    for error_record in reversed(past_errors):
        if not isinstance(error_record, Mapping):
            continue
        if error_record.get("timestamp") == info.get("timestamp"):
            continue
        similarity = difflib.SequenceMatcher(None, current_error, ensure_text(error_record.get("error_message", ""))).ratio()
        if similarity >= handler.error_similarity_threshold:
            metadata = error_record.get("metadata") if isinstance(error_record.get("metadata"), Mapping) else {}
            solution = metadata.get("solution") if isinstance(metadata, Mapping) else None
            if solution is None:
                solution = error_record.get("solution")
            if solution is not None:
                similar_solutions.append((similarity, solution))

    if not similar_solutions:
        return _result_not_applicable("No similar past errors with reusable solutions were found.", info, handler, strategy)

    similar_solutions.sort(key=lambda item: item[0], reverse=True)
    for similarity, solution in similar_solutions[: handler.max_solution_attempts]:
        try:
            if callable(solution):
                result = solution(agent, task_data)
                attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered using callable solution from similar past error.", details={"similarity": round(similarity, 4), "solution_type": "callable"}))
                return handler._success_outcome(strategy=strategy, reason="Recovered using callable solution from similar past error.", result=result, error_info=info, attempts=attempts, handler_name="handle_similar_past_error")
            if isinstance(solution, Mapping) and isinstance(getattr(agent, "config", None), MutableMapping):
                original_config = dict(agent.config)
                try:
                    agent.config.update(solution)
                    result = handler._invoke_task(agent, task_data)
                    attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered using temporary config update from similar past error.", details={"similarity": round(similarity, 4), "solution_type": "mapping"}))
                    return handler._success_outcome(strategy=strategy, reason="Recovered using temporary configuration from similar past error.", result=result, error_info=info, attempts=attempts, handler_name="handle_similar_past_error")
                finally:
                    agent.config.clear()
                    agent.config.update(original_config)
            if isinstance(solution, str) and callable(getattr(agent, solution, None)):
                result = getattr(agent, solution)(task_data)
                attempts.append(RecoveryAttempt(strategy=strategy, success=True, message="Recovered using named agent method from similar past error.", details={"similarity": round(similarity, 4), "solution_type": "method", "method": solution}))
                return handler._success_outcome(strategy=strategy, reason="Recovered using method solution from similar past error.", result=result, error_info=info, attempts=attempts, handler_name="handle_similar_past_error")
        except Exception as exc:
            attempts.append(RecoveryAttempt(strategy=strategy, success=False, message="Applying similar past-error solution failed.", details={"similarity": round(similarity, 4)}, exception_type=type(exc).__name__))

    return handler._failure_outcome(strategy=strategy, reason="All similar past-error solutions failed.", error_info=info, attempts=attempts, handler_name="handle_similar_past_error")


def handle_missing_inspect_error(agent: Any, task_data: Any, error_info: Mapping[str, Any], handler: IssueHandler) -> IssueOutcome:
    info = _coerce_error_mapping(error_info)
    error_type = ensure_text(info.get("error_type", ""))
    error_message = ensure_text(info.get("error_message", ""))
    strategy = "missing_inspect_recovery"
    if error_type != "NameError" or "name 'inspect' is not defined" not in error_message:
        return _result_not_applicable("Not a missing-inspect NameError.", info, handler, strategy)

    try:
        module_name = agent.__class__.__module__
        module = importlib.import_module(module_name)
        if not hasattr(module, "inspect"):
            setattr(module, "inspect", inspect)
        result = handler._invoke_task(agent, task_data)
        return handler._success_outcome(strategy=strategy, reason="Recovered after injecting missing inspect binding into the agent module.", result=result, error_info=info, attempts=[RecoveryAttempt(strategy=strategy, success=True, message="Injected inspect into agent module and retried.", details={"module_name": module_name})], handler_name="handle_missing_inspect_error")
    except Exception as exc:
        return handler._failure_outcome(strategy=strategy, reason=f"Inspect recovery failed: {handler._truncate_error_message(str(exc))}", error_info=info, attempts=[RecoveryAttempt(strategy=strategy, success=False, message="Missing-inspect recovery failed.", exception_type=type(exc).__name__)], handler_name="handle_missing_inspect_error")


DEFAULT_ISSUE_HANDLERS: Dict[str, Callable[..., Any]] = {
    "name 'inspect' is not defined": handle_missing_inspect_error,
    "NameError": handle_missing_inspect_error,
    "RuntimeError": handle_runtime_error,
    "unicode": handle_unicode_emoji_error,
    "emoji": handle_unicode_emoji_error,
    "UnicodeEncodeError": handle_unicode_emoji_error,
    "UnicodeDecodeError": handle_unicode_emoji_error,
    "network": handle_network_error,
    "ConnectionError": handle_network_error,
    "Timeout": handle_timeout_error,
    "memory": handle_memory_error,
    "MemoryError": handle_memory_error,
    "dependency": handle_common_dependency_error,
    "ImportError": handle_common_dependency_error,
    "resource": handle_resource_constraint,
    "past_error": handle_similar_past_error,
    "Exception": handle_similar_past_error,
}


_default_issue_handler: Optional[IssueHandler] = None


def get_issue_handler() -> IssueHandler:
    global _default_issue_handler
    if _default_issue_handler is None:
        _default_issue_handler = IssueHandler()
    return _default_issue_handler


def build_error_info(error: Optional[BaseException] = None, *, error_info: Optional[Mapping[str, Any]] = None, task_data: Any = None) -> Dict[str, Any]:
    return get_issue_handler().build_error_info(error, error_info=error_info, task_data=task_data)


def handle_issue(agent: Any, task_data: Any, *, error: Optional[BaseException] = None, error_info: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return get_issue_handler().handle_issue(agent, task_data, error=error, error_info=error_info)


if __name__ == "__main__":
    print("\n=== Running Issue Handler ===\n")
    printer.status("TEST", "Issue Handler initialized", "info")
    from ..collaborative.shared_memory import SharedMemory

    class DummyAgent:
        def __init__(self) -> None:
            self.name = "DummyAgent"
            self.config = {"mode": "default"}
            self.shared_memory = SharedMemory()
            self.error_similarity_threshold = 0.7
            self.lightweight_mode = False
            self.on_cpu = False
            self.fail_network_once = True
            self.fail_timeout_once = True
            self.freed_memory = False
            self.used_alternative = None

        def perform_task(self, task_data: Any) -> Any:
            if isinstance(task_data, str) and "🙂" in task_data:
                raise UnicodeEncodeError("ascii", "🙂".encode("utf-8"), 0, 1, "ordinal not in range")
            if isinstance(task_data, Mapping) and task_data.get("mode") == "network" and self.fail_network_once:
                self.fail_network_once = False
                raise ConnectionError("temporary network failure")
            if isinstance(task_data, Mapping) and task_data.get("mode") == "timeout" and self.fail_timeout_once:
                self.fail_timeout_once = False
                raise TimeoutError("operation timed out")
            if isinstance(task_data, Mapping) and task_data.get("mode") == "memory" and not self.freed_memory:
                raise MemoryError("resource exhausted")
            if isinstance(task_data, Mapping) and task_data.get("mode") == "dependency" and self.used_alternative is None:
                raise ImportError("No module named 'optional_backend'")
            if isinstance(task_data, Mapping) and task_data.get("mode") == "runtime" and not task_data.get("is_reshaped"):
                raise RuntimeError("shapes cannot be multiplied: mat1 and mat2")
            if isinstance(task_data, Mapping) and task_data.get("mode") == "gpu" and not self.on_cpu:
                raise RuntimeError("GPU resource busy")
            return {
                "status": "ok",
                "task": to_json_safe(task_data),
                "lightweight_mode": self.lightweight_mode,
                "on_cpu": self.on_cpu,
                "used_alternative": self.used_alternative,
            }

        def free_up_memory(self) -> None:
            self.freed_memory = True

        def use_lightweight_mode(self, enabled: bool) -> None:
            self.lightweight_mode = bool(enabled)

        def reshape_input_for_model(self, task_data: Any) -> Dict[str, Any]:
            updated = dict(task_data)
            updated["is_reshaped"] = True
            return updated

        def use_alternative_implementation(self, issue_name: Optional[str]) -> None:
            self.used_alternative = issue_name

        def switch_to_cpu(self) -> None:
            self.on_cpu = True

        def switch_to_gpu(self) -> None:
            self.on_cpu = False

    issue_handler = IssueHandler()
    agent = DummyAgent()

    unicode_result = issue_handler.handle_issue(agent, "hello 🙂 world", error_info={"error_type": "UnicodeEncodeError", "error_message": "emoji encode failure"})
    network_result = issue_handler.handle_issue(agent, {"mode": "network", "payload": "ping"}, error_info={"error_type": "ConnectionError", "error_message": "transient network failure"})
    timeout_result = issue_handler.handle_issue(agent, {"mode": "timeout", "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, error_info={"error_type": "TimeoutError", "error_message": "operation timed out"})
    memory_result = issue_handler.handle_issue(agent, {"mode": "memory", "payload": "x" * 200}, error_info={"error_type": "MemoryError", "error_message": "resource exhausted"})
    dependency_result = issue_handler.handle_issue(agent, {"mode": "dependency", "payload": "run"}, error_info={"error_type": "ImportError", "error_message": "No module named 'optional_backend'"})
    runtime_result = issue_handler.handle_issue(agent, {"mode": "runtime", "data": "payload"}, error_info={"error_type": "RuntimeError", "error_message": "shapes cannot be multiplied: mat1 and mat2"})
    gpu_result = issue_handler.handle_issue(agent, {"mode": "gpu", "payload": "tensor-job"}, error_info={"error_type": "RuntimeError", "error_message": "GPU resource busy"})

    printer.pretty("UNICODE_RESULT", unicode_result, "success")
    printer.pretty("NETWORK_RESULT", network_result, "success")
    printer.pretty("TIMEOUT_RESULT", timeout_result, "success")
    printer.pretty("MEMORY_RESULT", memory_result, "success")
    printer.pretty("DEPENDENCY_RESULT", dependency_result, "success")
    printer.pretty("RUNTIME_RESULT", runtime_result, "success")
    printer.pretty("GPU_RESULT", gpu_result, "success")
    printer.pretty("ISSUE_HANDLER_STATS", issue_handler.stats(), "success")
    printer.pretty("RECENT_HISTORY", issue_handler.recent_history(), "success")

    print("\n=== Test ran successfully ===\n")
