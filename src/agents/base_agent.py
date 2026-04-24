from __future__ import annotations

__version__ = "2.2.0"

"""
SLAI Base Agent

Production-ready common runtime for all SLAI agents.

Responsibilities
----------------
- provide one stable execution envelope for every role-specific agent
- centralize config loading from agents_config.yaml through main_config_loader
- standardize shared-memory, metrics, lifecycle, retry, fallback, and recovery behavior
- keep domain logic in subclasses while offering reusable primitives
- integrate with the Base Agent issue handling and error taxonomy

This module intentionally keeps local project imports direct. Optional heavyweight
third-party imports such as torch are lazy-loaded only when their optional helper
methods are used.
"""

import abc
import contextlib
import difflib
import inspect
import json
import os
import re
import time
import traceback
import uuid

from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

from .base.utils.main_config_loader import load_global_config, get_config_section
from .base.utils.base_errors import *
from .base.issue_handler import *
from .base.lazy_agent import LazyAgent
from .base.light_metric_store import LightMetricStore
from .collaborative.shared_memory import SharedMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Base Agent")
printer = PrettyPrinter()

torch = None
nn = None
TORCH_AVAILABLE: Optional[bool] = None
TORCH_IMPORT_ERROR: Optional[BaseException] = None


def _ensure_torch_imported() -> bool:
    """Lazy-load torch for optional lightweight-network helpers only."""
    global torch, nn, TORCH_AVAILABLE, TORCH_IMPORT_ERROR
    if TORCH_AVAILABLE is True:
        return True
    if TORCH_AVAILABLE is False:
        return False
    try:
        import torch as torch_module
        import torch.nn as torch_nn
    except Exception as exc:  # third-party optional dependency, not a local import
        TORCH_AVAILABLE = False
        TORCH_IMPORT_ERROR = exc
        return False
    torch = torch_module
    nn = torch_nn
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
    return True


@dataclass(frozen=True)
class ExecutionRecord:
    """Compact audit record for one BaseAgent execution."""

    execution_id: str
    agent_name: str
    started_at: float
    finished_at: float
    duration_ms: int
    status: str
    attempts: int
    recovered_by: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    result_preview: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attempts": self.attempts,
            "recovered_by": self.recovered_by,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "result_preview": self.result_preview,
            "metadata": self.metadata,
        }


class BaseAgent(abc.ABC):
    """Common production runtime inherited by all SLAI agents.

    Subclasses may override ``perform_task`` for domain-specific execution. If
    they do not, the default dispatcher calls compatible ``predict``,
    ``get_action``, or ``act`` methods. This preserves compatibility with the
    existing agents while avoiding redundant execution wrappers in every role.
    """

    DEFAULT_CAPABILITY_ORDER: Tuple[str, ...] = ("predict", "get_action", "act")
    DEFAULT_CONTENT_KEYS: Tuple[str, ...] = ("text", "query", "input", "message", "data", "payload")

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Optional[Mapping[str, Any]] = None) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.name = self.__class__.__name__
        self.agent_id = f"{self.name}:{uuid.uuid4().hex[:12]}"
        self._lock = RLock()

        self.shared_memory = shared_memory if shared_memory is not None else SharedMemory()
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.global_config = self.config
        self.base_config: Dict[str, Any] = dict(get_config_section("base_agent") or {})
        if config:
            ensure_mapping(config, "config", component=self.name)
            self.base_config.update(dict(config))

        self._load_base_config()
        self._validate_base_config()

        self.current_plan: List[Any] = []
        self.current_goal: Any = None
        self.operational_state = "initialized"
        self.last_execution: Optional[Dict[str, Any]] = None
        self.execution_history: Deque[Dict[str, Any]] = deque(maxlen=self.execution_history_limit)
        self._known_issue_handlers: Dict[str, Callable[..., Any]] = {}
        self._component_initializers: Dict[str, Callable[[], Any]] = {
            "performance_metrics": lambda: defaultdict(lambda: deque(maxlen=self._get_metric_buffer_size()))
        }
        self._lazy_components: OrderedDict[str, Any] = OrderedDict()
        self._performance_metrics: Optional[MutableMapping[str, Deque[Any]]] = None
        self.retraining_thresholds: Dict[str, Any] = {}

        self.evaluation_log_dir = str(self.base_config.get("evaluation_log_dir", "evaluation_logs"))
        Path(self.evaluation_log_dir).mkdir(parents=True, exist_ok=True)

        self.metric_store = LightMetricStore()
        self.issue_handler = IssueHandler()
        self.register_default_known_issue_handlers()
        self._init_core_components()
        self._publish_lifecycle_event("initialized", {"agent_id": self.agent_id})

    def _load_base_config(self) -> None:
        self.defer_initialization = bool(self.base_config.get("defer_initialization", True))
        self.memory_profile = str(self.base_config.get("memory_profile", "medium")).lower()
        self.network_compression = bool(self.base_config.get("network_compression", True))
        self.max_error_log_size = int(self.base_config.get("max_error_log_size", 50))
        self.error_similarity_threshold = float(self.base_config.get("error_similarity_threshold", 0.75))
        self.max_task_retries = int(self.base_config.get("max_task_retries", 0))
        self.retry_backoff_seconds = float(self.base_config.get("retry_backoff_seconds", 0.5))
        self.retry_backoff_cap_seconds = float(self.base_config.get("retry_backoff_cap_seconds", 5.0))
        self.task_timeout_seconds = self._none_or_float(self.base_config.get("task_timeout_seconds"))
        self.enable_known_issue_recovery = bool(self.base_config.get("enable_known_issue_recovery", True))
        self.enable_alternative_execute = bool(self.base_config.get("enable_alternative_execute", True))
        self.enable_shared_memory_audit = bool(self.base_config.get("enable_shared_memory_audit", True))
        self.execution_history_limit = int(self.base_config.get("execution_history_limit", 200))
        self.metric_buffer_size_low = int(self.base_config.get("metric_buffer_size_low", 100))
        self.metric_buffer_size_medium = int(self.base_config.get("metric_buffer_size_medium", 500))
        self.metric_buffer_size_high = int(self.base_config.get("metric_buffer_size_high", 1000))
        self.task_similarity_str_threshold = float(self.base_config.get("task_similarity_str_threshold", 0.90))
        self.jaccard_threshold = float(self.base_config.get("jaccard_threshold", 0.50))
        self.jaccard_min_for_no_shared = float(self.base_config.get("jaccard_min_for_no_shared", 0.70))
        self.final_key_threshold = float(self.base_config.get("final_key_threshold", 0.70))
        self.final_value_threshold = float(self.base_config.get("final_value_threshold", 0.70))
        self.task_similarity_seq_elem_threshold = float(self.base_config.get("task_similarity_seq_elem_threshold", 0.80))
        self.plan_resource_warning_threshold = float(self.base_config.get("plan_resource_warning_threshold", 0.70))
        self.plan_resource_critical_threshold = float(self.base_config.get("plan_resource_critical_threshold", 0.90))
        self.shared_memory_error_key_prefix = str(self.base_config.get("shared_memory_error_key_prefix", "errors"))
        self.shared_memory_stats_key_prefix = str(self.base_config.get("shared_memory_stats_key_prefix", "agent_stats"))
        self.shared_memory_event_key_prefix = str(self.base_config.get("shared_memory_event_key_prefix", "agent_events"))
        self.warm_start_key_prefix = str(self.base_config.get("warm_start_key_prefix", "warm_state"))
        self.retraining_flag_key_prefix = str(self.base_config.get("retraining_flag_key_prefix", "retraining_flag"))

    def _validate_base_config(self) -> None:
        ensure_numeric_range(self.error_similarity_threshold, "error_similarity_threshold", minimum=0.0, maximum=1.0, component=self.name)
        ensure_numeric_range(self.task_similarity_str_threshold, "task_similarity_str_threshold", minimum=0.0, maximum=1.0, component=self.name)
        ensure_numeric_range(self.jaccard_threshold, "jaccard_threshold", minimum=0.0, maximum=1.0, component=self.name)
        ensure_numeric_range(self.final_key_threshold, "final_key_threshold", minimum=0.0, maximum=1.0, component=self.name)
        ensure_numeric_range(self.final_value_threshold, "final_value_threshold", minimum=0.0, maximum=1.0, component=self.name)
        ensure_numeric_range(self.task_similarity_seq_elem_threshold, "task_similarity_seq_elem_threshold", minimum=0.0, maximum=1.0, component=self.name)
        if self.max_error_log_size < 1:
            raise BaseConfigurationError("max_error_log_size must be positive.", component=self.name, details={"value": self.max_error_log_size})
        if self.max_task_retries < 0:
            raise BaseConfigurationError("max_task_retries cannot be negative.", component=self.name, details={"value": self.max_task_retries})
        if self.memory_profile not in {"low", "medium", "high"}:
            raise BaseConfigurationError("memory_profile must be low, medium, or high.", component=self.name, details={"value": self.memory_profile})

    @staticmethod
    def _none_or_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            return None
        return float(value)

    def _get_metric_buffer_size(self) -> int:
        if self.memory_profile == "low":
            return self.metric_buffer_size_low
        if self.memory_profile == "high":
            return self.metric_buffer_size_high
        return self.metric_buffer_size_medium

    def _init_core_components(self) -> None:
        self.lazy_agent = LazyAgent(lambda: self._create_expensive_components(self.network_compression))

    def _create_expensive_components(self, compression_enabled: bool = False) -> Dict[str, Any]:
        return {
            "performance_metrics": defaultdict(lambda: deque(maxlen=self._get_metric_buffer_size())),
            "compression_enabled": bool(compression_enabled),
        }

    @property
    def performance_metrics(self) -> MutableMapping[str, Deque[Any]]:
        if self._performance_metrics is None:
            self._performance_metrics = defaultdict(lambda: deque(maxlen=self._get_metric_buffer_size()))
        return self._performance_metrics

    @performance_metrics.setter
    def performance_metrics(self, value: MutableMapping[str, Deque[Any]]) -> None:
        self._performance_metrics = value

    def lazy_property(self, name: str) -> Any:
        ensure_non_empty_string(name, "name", component=self.name)
        with self._lock:
            if name not in self._lazy_components:
                initializer = self._component_initializers.get(name)
                if initializer is None:
                    raise BaseStateError(f"No initializer registered for lazy component: {name}", component=self.name)
                self._lazy_components[name] = initializer()
            return self._lazy_components[name]

    def register_lazy_component(self, name: str, initializer: Callable[[], Any], *, replace: bool = False) -> None:
        ensure_non_empty_string(name, "name", component=self.name)
        ensure_callable(initializer, "initializer", component=self.name)
        with self._lock:
            if name in self._component_initializers and not replace:
                raise BaseStateError(f"Lazy component already registered: {name}", component=self.name)
            self._component_initializers[name] = initializer
            if replace:
                self._lazy_components.pop(name, None)

    def register_default_known_issue_handlers(self) -> None:
        for pattern, handler in DEFAULT_ISSUE_HANDLERS.items():
            self.register_known_issue_handler(pattern, handler, replace=True)

    def register_known_issue_handler(self, issue_pattern_or_id: str, handler_func: Callable[..., Any], *, replace: bool = False) -> None:
        ensure_non_empty_string(issue_pattern_or_id, "issue_pattern_or_id", component=self.name)
        ensure_callable(handler_func, "handler_func", component=self.name)
        with self._lock:
            if issue_pattern_or_id in self._known_issue_handlers and not replace:
                raise BaseStateError(f"Known issue handler already registered: {issue_pattern_or_id}", component=self.name)
            self._known_issue_handlers[issue_pattern_or_id] = handler_func

    def execute(self, input_data: Any) -> Any:
        """Execute task data through the common production envelope."""
        execution_id = uuid.uuid4().hex
        started_at = time.time()
        attempts = 0
        last_exception: Optional[BaseException] = None
        self.operational_state = "running"
        self.metric_store.start_tracking("execute", "performance", metadata={"agent": self.name, "execution_id": execution_id})

        try:
            for attempt_index in range(self.max_task_retries + 1):
                attempts = attempt_index + 1
                try:
                    result = self._execute_once(input_data)
                    self._after_successful_execution(result, attempts, execution_id)
                    self._record_execution(started_at, "success", attempts, result=result, execution_id=execution_id)
                    return result
                except Exception as exc:
                    last_exception = exc
                    if attempt_index < self.max_task_retries:
                        self._publish_lifecycle_event("retry", {"attempt": attempts, "error": str(exc)})
                        time.sleep(min(self.retry_backoff_seconds * (attempt_index + 1), self.retry_backoff_cap_seconds))
                        continue
                    break

            error_info = self._build_error_info(last_exception, input_data, execution_id, attempts)
            self._log_error_to_shared_memory(error_info)
            self._check_and_log_similar_errors(error_info)

            if self.enable_known_issue_recovery:
                recovered = self.handle_known_issue(input_data, error_info, error=last_exception)
                if not self._is_failure_result(recovered):
                    self._after_successful_execution(recovered, attempts, execution_id, recovered_by="handle_known_issue")
                    self._record_execution(started_at, "recovered", attempts, result=recovered, recovered_by="handle_known_issue", execution_id=execution_id)
                    return recovered

            if self.enable_alternative_execute:
                alternative_result = self.alternative_execute(input_data, original_error=last_exception)
                if not self._is_failure_result(alternative_result):
                    self._after_successful_execution(alternative_result, attempts, execution_id, recovered_by="alternative_execute")
                    self._record_execution(started_at, "recovered", attempts, result=alternative_result, recovered_by="alternative_execute", execution_id=execution_id)
                    return alternative_result

            failure = {
                "status": "failed",
                "error": error_info["error_message"],
                "error_type": error_info["error_type"],
                "reason": "All execution and recovery attempts failed.",
                "execution_id": execution_id,
                "attempts": attempts,
            }
            self._record_execution(started_at, "failed", attempts, error_info=error_info, execution_id=execution_id)
            self._publish_stats(success=False, attempts=attempts, error=error_info)
            return failure
        finally:
            self.metric_store.stop_tracking("execute", "performance")
            self.operational_state = "idle"

    def _execute_once(self, input_data: Any) -> Any:
        return self.perform_task(input_data)

    def _after_successful_execution(self, result: Any, attempts: int, execution_id: str, *, recovered_by: Optional[str] = None) -> None:
        metrics = self.extract_performance_metrics(result)
        if metrics:
            self.evaluate_performance(metrics)
        self._publish_stats(success=True, attempts=attempts, result=result, recovered_by=recovered_by, execution_id=execution_id)

    def _record_execution(
        self,
        started_at: float,
        status: str,
        attempts: int,
        *,
        result: Any = None,
        error_info: Optional[Mapping[str, Any]] = None,
        recovered_by: Optional[str] = None,
        execution_id: str,
    ) -> None:
        finished_at = time.time()
        record = ExecutionRecord(
            execution_id=execution_id,
            agent_name=self.name,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=int((finished_at - started_at) * 1000),
            status=status,
            attempts=attempts,
            recovered_by=recovered_by,
            error_type=str(error_info.get("error_type")) if error_info else None,
            error_message=str(error_info.get("error_message")) if error_info else None,
            result_preview=self._preview(result) if result is not None else None,
        ).to_dict()
        self.last_execution = record
        self.execution_history.append(record)
        self._publish_lifecycle_event("execution_recorded", record)

    def _build_error_info(self, error: Optional[BaseException], task_data: Any, execution_id: str, attempts: int) -> Dict[str, Any]:
        if error is None:
            error = BaseRuntimeError("Task failed without an exception object.", component=self.name)
        return {
            "timestamp": time.time(),
            "agent_name": self.name,
            "execution_id": execution_id,
            "attempts": attempts,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "task_preview": self._preview(task_data),
        }

    def _log_error_to_shared_memory(self, error_entry: Mapping[str, Any]) -> None:
        if not self.enable_shared_memory_audit:
            return
        error_key = f"{self.shared_memory_error_key_prefix}:{self.name}"
        try:
            errors = self.shared_memory.get(error_key) or []
            errors.append(dict(error_entry))
            self.shared_memory.set(error_key, errors[-self.max_error_log_size:])
        except Exception as exc:
            self.logger.warning("[%s] Failed to write error audit to shared memory: %s", self.name, exc)

    def _check_and_log_similar_errors(self, new_error_info: Mapping[str, Any]) -> bool:
        if not self.enable_shared_memory_audit:
            return False
        error_key = f"{self.shared_memory_error_key_prefix}:{self.name}"
        try:
            history = self.shared_memory.get(error_key) or []
        except Exception:
            return False
        new_type = str(new_error_info.get("error_type", ""))
        new_message = str(new_error_info.get("error_message", ""))
        new_timestamp = float(new_error_info.get("timestamp", 0.0) or 0.0)
        for previous in reversed(history):
            if float(previous.get("timestamp", 0.0) or 0.0) >= new_timestamp:
                continue
            if str(previous.get("error_type", "")) != new_type:
                continue
            similarity = difflib.SequenceMatcher(None, new_message, str(previous.get("error_message", ""))).ratio()
            if similarity >= self.error_similarity_threshold:
                self.logger.warning("[%s] Similar %s detected with %.2f similarity.", self.name, new_type, similarity)
                return True
        return False

    def handle_known_issue(self, task_data: Any, error_info: Mapping[str, Any], *, error: Optional[BaseException] = None) -> Any:
        """Recover through the production IssueHandler first, then local handlers."""
        try:
            outcome = self.issue_handler.handle_issue(self, task_data, error=error, error_info=error_info)
            if isinstance(outcome, Mapping) and outcome.get("recovered"):
                return outcome.get("result", outcome)
        except Exception as exc:
            self.logger.warning("[%s] IssueHandler failed; checking local handlers: %s", self.name, exc)

        error_message_lower = str(error_info.get("error_message", "")).lower()
        error_type_lower = str(error_info.get("error_type", "")).lower()
        for pattern, handler_func in self._known_issue_handlers.items():
            pattern_lower = pattern.lower()
            if pattern_lower in error_message_lower or pattern_lower == error_type_lower:
                try:
                    return handler_func(self, task_data, dict(error_info), self.issue_handler)
                except TypeError:
                    return handler_func(self, task_data, dict(error_info))
                except Exception as exc:
                    return {"status": "failed", "reason": f"Handler for {pattern} failed: {exc}", "error": str(exc)}
        return {"status": "failed", "reason": "No applicable known issue handler recovered the task."}

    @staticmethod
    def _is_failure_result(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, Mapping):
            return str(value.get("status", "")).lower() in {"failed", "failure", "error"} and not value.get("recovered")
        if isinstance(value, str):
            return "[fallback failure]" in value.lower()
        return False

    def alternative_execute(self, task_data: Any, original_error: Optional[BaseException] = None) -> Any:
        if hasattr(self, "replan") and callable(getattr(self, "replan")) and self.current_goal:
            with contextlib.suppress(Exception):
                new_plan = self.replan(self.current_goal)
                if new_plan:
                    return self.execute_plan(new_plan, goal=self.current_goal)
        sanitized_input = self.sanitize_input(task_data)
        rule_based = self.apply_rule_based_processing(sanitized_input)
        if rule_based is not None:
            return rule_based
        return self.echo_strategy(sanitized_input, original_error=original_error)

    def sanitize_input(self, task_data: Any) -> str:
        if isinstance(task_data, str):
            return task_data.strip().replace("\n", " ").replace("\r", "")[:500]
        if isinstance(task_data, Mapping):
            for key in self.DEFAULT_CONTENT_KEYS:
                value = task_data.get(key)
                if isinstance(value, str):
                    return value.strip()[:500]
            simple_payload = {str(k): v for k, v in task_data.items() if isinstance(v, (str, int, float, bool))}
            return json.dumps(simple_payload, default=str)[:500]
        return str(task_data)[:500]

    def apply_rule_based_processing(self, sanitized_input: str) -> Optional[str]:
        patterns = {
            r"connection (timeout|refused)|network": "Network unavailable or unstable. Retry later or switch to an offline-safe path.",
            r"invalid (format|input)|validation": "Input validation failed. Check required fields and value types.",
            r"out of memory|resource exhausted|memory": "Resource constraints detected. Reduce batch size, payload size, or model footprint.",
            r"authentication failed|unauthorized|forbidden": "Authentication or authorization failed. Refresh credentials and permissions.",
        }
        for pattern, response in patterns.items():
            if re.search(pattern, sanitized_input, re.IGNORECASE):
                return f"[Fallback] {response}"
        grammar = getattr(self, "grammar", None)
        if callable(getattr(grammar, "compose_sentence", None)):
            with contextlib.suppress(Exception):
                return f"[Grammar Response] {grammar.compose_sentence({'input': sanitized_input[:100], 'error': 'unknown'})}"
        return None

    def echo_strategy(self, sanitized_input: str, *, original_error: Optional[BaseException] = None) -> str:
        if sanitized_input:
            return f"[Fallback] Processed input: {sanitized_input}"
        reason = str(original_error) if original_error else "no processable input"
        return f"[Fallback failure] Unable to process request through alternative methods: {reason}"

    def _invoke_capability(self, fn: Callable[..., Any], task_data: Any, context: Any = None) -> Any:
        signature = inspect.signature(fn)
        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return fn(task_data, context)

        positional: List[Any] = []
        kwargs: Dict[str, Any] = {}
        accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        payload_names = {"task_data", "data", "input_data", "state", "task", "query", "payload"}
        context_names = {"context", "ctx"}

        for param in params:
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                if param.name in payload_names:
                    positional.append(task_data)
                elif param.name in context_names:
                    positional.append(context)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                if param.name in payload_names:
                    kwargs[param.name] = task_data
                elif param.name in context_names:
                    kwargs[param.name] = context

        if not positional and not kwargs:
            if accepts_varkw:
                return fn(task_data=task_data, context=context)
            return fn(task_data)
        return fn(*positional, **kwargs)

    def perform_task(self, task_data: Any) -> Any:
        context = None
        operation = None
        payload = task_data
        if isinstance(task_data, Mapping):
            context = task_data.get("context")
            operation = task_data.get("operation")
            payload = task_data.get("task_data", task_data.get("input_data", task_data.get("payload", task_data)))

        dispatch_order = list(self.DEFAULT_CAPABILITY_ORDER)
        if operation:
            preferred = {
                "predict": "predict",
                "inference": "predict",
                "infer": "predict",
                "get_action": "get_action",
                "action": "get_action",
                "act": "act",
            }.get(str(operation).strip().lower())
            if preferred:
                dispatch_order = [preferred] + [name for name in dispatch_order if name != preferred]

        errors: List[str] = []
        for method_name in dispatch_order:
            capability = getattr(self, method_name, None)
            if not callable(capability):
                continue
            try:
                return self._invoke_capability(capability, payload, context)
            except TypeError as exc:
                errors.append(f"{method_name}: {exc}")
                continue
        details = "; ".join(errors) if errors else "No compatible capability methods found."
        raise NotImplementedError(f"{self.name} cannot execute task. {details}")

    def execute_plan(self, plan: Iterable[Any], goal: Any = None) -> Dict[str, Any]:
        if isinstance(plan, (str, bytes)) or not hasattr(plan, "__iter__"):
            return {"status": "error", "reason": "Plan must be an iterable sequence of steps."}
        steps = list(plan)
        if not steps:
            return {"status": "error", "reason": "Plan is empty."}
        if not self.validate_plan(steps):
            return {"status": "error", "reason": "Plan validation failed."}

        results: List[Any] = []
        context: Dict[str, Any] = {}
        for index, step in enumerate(steps):
            step_name = step.get("name", f"step_{index + 1}") if isinstance(step, Mapping) else f"step_{index + 1}"
            try:
                result = self.execute_step(step, context)
                results.append(result)
                if isinstance(result, Mapping):
                    context.update(dict(result.get("context", {})))
                if goal is not None and self.check_goal_achieved(context, goal):
                    return self.compile_results(results, True, reason="goal_achieved")
            except Exception as exc:
                results.append({"step": step_name, "status": "error", "error": str(exc)})
                if not self.recover_step(step, context):
                    return self.compile_results(results, False, reason=f"step_failed:{step_name}")
        return self.compile_results(results, True)

    def validate_plan(self, plan: Sequence[Any]) -> bool:
        for step in plan:
            if isinstance(step, Mapping) and "handler" in step and not str(step.get("handler", "")).strip():
                return False
        return True

    def execute_step(self, step: Any, context: MutableMapping[str, Any]) -> Any:
        if callable(step):
            return step(context)
        if isinstance(step, Mapping):
            handler_name = step.get("handler")
            if handler_name:
                handler = getattr(self, str(handler_name), None)
                if not callable(handler):
                    raise AttributeError(f"Missing step handler: {handler_name}")
                return handler(step.get("params", {}), context)
            if "module" in step and "action" in step:
                return self.call_external(str(step["module"]), str(step["action"]), dict(step.get("params", {})))
        if isinstance(step, str):
            return self.process_command(step, context)
        raise BaseValidationError(f"Unsupported step type: {type(step).__name__}", component=self.name)

    def recover_step(self, step: Any, step_context: MutableMapping[str, Any]) -> bool:
        if isinstance(step, Mapping) and step.get("recovery_handler"):
            handler = getattr(self, str(step["recovery_handler"]), None)
            if callable(handler):
                with contextlib.suppress(Exception):
                    return bool(handler(step, step_context))
        with contextlib.suppress(Exception):
            self.execute_step(step, step_context)
            return True
        return False

    def check_goal_achieved(self, step_context: Mapping[str, Any], goal: Any) -> bool:
        if isinstance(goal, Mapping):
            return all(step_context.get(key) == value for key, value in goal.items())
        return step_context.get("result") == goal

    @staticmethod
    def compile_results(results: Sequence[Any], success: bool, *, reason: Optional[str] = None) -> Dict[str, Any]:
        successful_steps = sum(1 for item in results if not (isinstance(item, Mapping) and item.get("status") == "error"))
        return {
            "status": "success" if success else "partial_success",
            "reason": reason,
            "steps_executed": len(results),
            "success_rate": successful_steps / len(results) if results else 0.0,
            "results": list(results),
            "timestamp": time.time(),
        }

    def call_external(self, module: str, action: str, params: Mapping[str, Any]) -> Any:
        factory = self.agent_factory
        if factory is None:
            raise BaseStateError("agent_factory is required for external calls.", component=self.name)
        if hasattr(factory, "create") and callable(factory.create):
            module_agent = factory.create(module, self.shared_memory)
        elif callable(factory):
            module_agent = factory(module, self.shared_memory)
        else:
            raise BaseStateError("agent_factory must be callable or expose create().", component=self.name)
        action_method = getattr(module_agent, action, None)
        if not callable(action_method):
            raise AttributeError(f"Module '{module}' has no callable action '{action}'.")
        return action_method(**dict(params))

    def process_command(self, command: str, context: MutableMapping[str, Any]) -> Dict[str, Any]:
        parts = command.split(" ", 1)
        cmd_type = parts[0].lower()
        payload = parts[1] if len(parts) > 1 else ""
        if cmd_type == "log":
            self.logger.info("[Command] %s", payload)
            return {"status": "logged", "message": payload}
        if cmd_type == "set":
            key, value = payload.split(" ", 1)
            context[key] = value
            return {"status": "set", "key": key, "value": value, "context": dict(context)}
        if cmd_type == "get":
            return {"status": "retrieved", "key": payload, "value": context.get(payload, "NOT_FOUND")}
        if cmd_type == "incr":
            tokens = payload.split(" ")
            key = tokens[0]
            increment = float(tokens[1]) if len(tokens) > 1 else 1.0
            context[key] = float(context.get(key, 0.0)) + increment
            return {"status": "incremented", "key": key, "value": context[key], "context": dict(context)}
        raise BaseValidationError(f"Unknown command: {cmd_type}", component="process_command")

    def extract_performance_metrics(self, result: Any) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if isinstance(result, Mapping):
            for key in ("accuracy", "precision", "recall", "f1_score", "latency_ms", "throughput", "loss", "risk_score"):
                value = result.get(key)
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        return metrics

    def evaluate_performance(self, metrics: Mapping[str, Any]) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            self.performance_metrics[str(key)].append(float(value))
            with contextlib.suppress(Exception):
                self.metric_store.record_value(str(key), float(value), category="performance")
        self.log_evaluation_result(dict(metrics))
        for metric_key, current_value in metrics.items():
            threshold_info = self.retraining_thresholds.get(metric_key)
            if threshold_info is None or not isinstance(current_value, (int, float)):
                continue
            threshold_value = threshold_info.get("value") if isinstance(threshold_info, Mapping) else threshold_info
            condition = threshold_info.get("condition", "less_than") if isinstance(threshold_info, Mapping) else "less_than"
            if isinstance(threshold_value, (int, float)):
                if (condition == "less_than" and current_value < threshold_value) or (condition == "greater_than" and current_value > threshold_value):
                    self.flag_for_retraining(reason=f"{metric_key}:{current_value}:{condition}:{threshold_value}")

    def flag_for_retraining(self, *, reason: Optional[str] = None) -> None:
        key = f"{self.retraining_flag_key_prefix}:{self.name}"
        self.shared_memory.set(key, {"flagged": True, "reason": reason, "timestamp": time.time()})

    def log_evaluation_result(self, metrics: Mapping[str, Any]) -> None:
        log_entry = {"timestamp": time.time(), "agent_name": self.name, "metrics": dict(metrics)}
        path = Path(self.evaluation_log_dir) / f"{self.name}_eval.jsonl"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_entry, default=str) + "\n")
        except Exception as exc:
            self.logger.warning("[%s] Failed to write evaluation log: %s", self.name, exc)

    def _publish_stats(self, *, success: bool, attempts: int, result: Any = None, error: Any = None, recovered_by: Optional[str] = None, execution_id: Optional[str] = None) -> None:
        if not self.enable_shared_memory_audit:
            return
        payload = {
            "last_run": time.time(),
            "success": success,
            "attempts": attempts,
            "recovered_by": recovered_by,
            "execution_id": execution_id,
            "result_summary": self._preview(result) if result is not None else None,
            "error": error,
        }
        with contextlib.suppress(Exception):
            self.shared_memory.set(f"{self.shared_memory_stats_key_prefix}:{self.name}", payload)

    def _publish_lifecycle_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.enable_shared_memory_audit:
            return
        key = f"{self.shared_memory_event_key_prefix}:{self.name}"
        event = {"event_type": event_type, "timestamp": time.time(), "payload": dict(payload)}
        with contextlib.suppress(Exception):
            events = self.shared_memory.get(key) or []
            events.append(event)
            self.shared_memory.set(key, events[-self.execution_history_limit:])

    def _warm_start_if_available(self) -> bool:
        warm_start_key = f"{self.warm_start_key_prefix}:{self.name}"
        cached_state = self.shared_memory.get(warm_start_key)
        if not isinstance(cached_state, Mapping):
            return False
        expected = set(getattr(self, "warm_start_attributes", []))
        loaded = 0
        for attr_name, attr_value in cached_state.items():
            if attr_name in expected and hasattr(self, attr_name) and not callable(getattr(self, attr_name)):
                setattr(self, attr_name, attr_value)
                loaded += 1
        return loaded > 0

    def broadcast(self, key: str, value: Any) -> None:
        self.shared_memory.set(key, value)

    def is_similar(self, task_data_1: Any, task_data_2: Any) -> bool:
        if type(task_data_1) is not type(task_data_2):
            return False
        if isinstance(task_data_1, str):
            s1, s2 = task_data_1.strip().lower(), task_data_2.strip().lower()
            return difflib.SequenceMatcher(None, s1, s2).ratio() >= self.task_similarity_str_threshold
        if isinstance(task_data_1, Mapping):
            keys1, keys2 = set(task_data_1.keys()), set(task_data_2.keys())
            if not keys1 and not keys2:
                return True
            key_union = len(keys1 | keys2)
            key_similarity = len(keys1 & keys2) / key_union if key_union else 0.0
            if key_similarity < self.jaccard_threshold:
                return False
            shared_keys = keys1 & keys2
            if not shared_keys:
                return key_similarity >= self.jaccard_min_for_no_shared
            value_similarity = sum(1.0 if self.is_similar(task_data_1[key], task_data_2[key]) else 0.0 for key in shared_keys) / len(shared_keys)
            return key_similarity >= self.final_key_threshold and value_similarity >= self.final_value_threshold
        if isinstance(task_data_1, (list, tuple)):
            if len(task_data_1) != len(task_data_2):
                return False
            if not task_data_1:
                return True
            score = sum(1.0 if self.is_similar(a, b) else 0.0 for a, b in zip(task_data_1, task_data_2)) / len(task_data_1)
            return score >= self.task_similarity_seq_elem_threshold
        return task_data_1 == task_data_2

    def create_lightweight_network(self, input_dim: int = 10, output_dim: int = 2) -> Any:
        if not _ensure_torch_imported():
            raise BaseInitializationError("torch is unavailable for create_lightweight_network.", component=self.name, cause=TORCH_IMPORT_ERROR)

        class PolicyNet(nn.Module):  # type: ignore[union-attr]
            def __init__(self, input_dim: int, output_dim: int) -> None:
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * torch.sqrt(torch.tensor(1.0 / input_dim)))
                self.bias = nn.Parameter(torch.zeros(output_dim))

            def predict(self, state: Any) -> int:
                state_vec = torch.tensor(state, dtype=torch.float32).flatten()
                if state_vec.shape[0] != self.input_dim:
                    raise ValueError(f"Input dimension {state_vec.shape[0]} does not match {self.input_dim}.")
                logits = torch.matmul(state_vec, self.weights) + self.bias
                return int(torch.argmax(logits).item())

        return PolicyNet(input_dim, output_dim)

    def update_projection(self, reward_scores: Sequence[float], lr: float) -> Optional[Dict[str, Any]]:
        if not _ensure_torch_imported():
            return {"status": "skipped", "reason": "torch_unavailable", "error": str(TORCH_IMPORT_ERROR)}
        projection = getattr(self, "projection", None)
        if projection is None or not isinstance(projection, torch.Tensor):
            return {"status": "skipped", "reason": "projection_tensor_missing"}
        rewards = torch.tensor(list(reward_scores), dtype=torch.float32, device=projection.device)
        mean_reward = rewards.mean() if rewards.numel() else torch.tensor(0.0, device=projection.device)
        with torch.no_grad():
            projection.data += float(lr) * torch.sign(projection.data) * mean_reward
        return {"status": "updated", "mean_reward": float(mean_reward.item())}

    @staticmethod
    def _preview(value: Any, limit: int = 300) -> str:
        try:
            text = json.dumps(value, default=str)
        except Exception:
            text = repr(value)
        return text[:limit]


class RetrainingManager:
    """Checks shared-memory retraining flags and invokes an agent retrain hook."""

    def __init__(self, agent: BaseAgent, shared_memory: Any) -> None:
        self.agent = agent
        self.shared_memory = shared_memory
        self.logger = get_logger(f"RetrainingManager[{agent.name}]")

    def check_and_trigger_retraining(self) -> bool:
        key = f"{self.agent.retraining_flag_key_prefix}:{self.agent.name}"
        flag = self.shared_memory.get(key)
        is_flagged = bool(flag.get("flagged")) if isinstance(flag, Mapping) else bool(flag)
        if not is_flagged:
            return False
        retrain = getattr(self.agent, "retrain", None)
        if callable(retrain):
            retrain()
            self.shared_memory.set(key, {"flagged": False, "timestamp": time.time(), "reason": "completed"})
            return True
        self.shared_memory.set(key, {"flagged": False, "timestamp": time.time(), "reason": "missing_retrain_hook"})
        return False


class ResourceMonitor:
    """Small resource monitor used by generic plan execution."""

    def __init__(self, warning_threshold: float = 0.70, critical_threshold: float = 0.90) -> None:
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def is_critical(self) -> bool:
        return False


if __name__ == "__main__":
    print("\n=== Running Base agent ===\n")
    printer.status("TEST", " Base agent initialized", "info")
    from .agent_factory import AgentFactory

    class SmokeAgent(BaseAgent):
        def predict(self, state: Any, context: Any = None) -> Dict[str, Any]:
            return {"status": "success", "prediction": state, "accuracy": 1.0, "context": {"seen": True}}

        def custom_step(self, params: Mapping[str, Any], context: MutableMapping[str, Any]) -> Dict[str, Any]:
            context["custom"] = params.get("value", "done")
            return {"status": "success", "context": dict(context)}

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = SmokeAgent(shared_memory=shared_memory, agent_factory=agent_factory, config={"max_task_retries": 0})

    result = agent.execute({"operation": "predict", "input_data": {"hello": "world"}, "context": {"source": "smoke"}})
    assert result["status"] == "success", result

    plan_result = agent.execute_plan([
        "set counter 1",
        "incr counter 2",
        {"name": "custom", "handler": "custom_step", "params": {"value": "ok"}},
    ])
    assert plan_result["status"] == "success", plan_result

    similarity = agent.is_similar({"a": "hello world"}, {"a": "hello world!"})
    assert similarity is True

    stats = shared_memory.get("agent_stats:SmokeAgent")
    assert stats and stats["success"] is True

    printer.status("TEST", " Base agent execute/plan/similarity/shared-memory checks passed", "success")
    print("\n=== Test ran successfully ===\n")
