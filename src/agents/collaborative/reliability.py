from __future__ import annotations

"""
Reliability primitives for SLAI's collaborative runtime.

This module provides production-ready retry and circuit-breaker controls for
collaborative agent execution. It intentionally does not own routing decisions,
agent discovery, task-contract validation, or policy evaluation. Those concerns
remain in their dedicated modules. The reliability layer focuses on:

- bounded retry/backoff around agent operations;
- per-agent circuit-breaker state transitions;
- half-open recovery probes;
- operational metrics, state snapshots, and audit records;
- shared-memory publication for observability;
- collaboration error integration at reliability boundaries.

The public API from the original implementation is retained: ``CircuitState``,
``RetryPolicy``, ``CircuitBreakerConfig``, ``AgentCircuitBreaker``,
``ReliabilityManager``, ``is_available()``, ``execute()``, ``status()``, and
``_sleep_backoff()`` remain available. Additional methods are additive and are
safe for task-router and collaboration-manager integration.
"""

import random
import threading
import time

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TypeVar

from .utils.config_loader import load_global_config, get_config_section
from .utils.collaboration_error import *
from .utils.collaborative_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reliability Manager")
printer = PrettyPrinter()

T = TypeVar("T")
RetryPredicate = Callable[[BaseException], bool]
AttemptCallback = Callable[[Dict[str, Any]], None]


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class RetryPolicy:
    """Retry/backoff policy used by ``ReliabilityManager.execute``.

    ``backoff_factor`` preserves the original semantics: it is the base delay
    in seconds for attempt 1 and grows exponentially by powers of two.
    ``jitter_seconds`` is additive jitter, also preserving the original field.
    """

    max_attempts: int = 1
    backoff_factor: float = 0.0
    max_backoff_seconds: float = 2.0
    jitter_seconds: float = 0.0
    retry_on_circuit_open: bool = False
    retryable_exception_types: Tuple[str, ...] = ()
    non_retryable_exception_types: Tuple[str, ...] = ("ValueError", "TypeError", "KeyError", "AssertionError")
    sleep_enabled: bool = True

    @classmethod
    def from_config(cls, reliability_config: Optional[Mapping[str, Any]] = None, task_routing_config: Optional[Mapping[str, Any]] = None) -> "RetryPolicy":
        reliability_config = dict(reliability_config or {})
        task_routing_config = dict(task_routing_config or {})
        routing_retry = task_routing_config.get("retry_policy") if isinstance(task_routing_config.get("retry_policy"), Mapping) else {}
        reliability_retry = reliability_config.get("retry_policy") if isinstance(reliability_config.get("retry_policy"), Mapping) else {}
        source = merge_mappings(routing_retry, reliability_retry, deep=True, drop_none=True)

        return cls(
            max_attempts=coerce_int(source.get("max_attempts", reliability_config.get("max_attempts", 1)), default=1, minimum=1),
            backoff_factor=coerce_float(source.get("backoff_factor", reliability_config.get("backoff_factor", 0.0)), default=0.0, minimum=0.0),
            max_backoff_seconds=coerce_float(source.get("max_backoff_seconds", reliability_config.get("max_backoff_seconds", 2.0)), default=2.0, minimum=0.0),
            jitter_seconds=coerce_float(source.get("jitter_seconds", reliability_config.get("jitter_seconds", 0.0)), default=0.0, minimum=0.0),
            retry_on_circuit_open=coerce_bool(source.get("retry_on_circuit_open", reliability_config.get("retry_on_circuit_open", False)), default=False),
            retryable_exception_types=tuple(str(item).strip() for item in ensure_list(source.get("retryable_exception_types", reliability_config.get("retryable_exception_types", []))) if str(item).strip()),
            non_retryable_exception_types=tuple(str(item).strip() for item in ensure_list(source.get("non_retryable_exception_types", reliability_config.get("non_retryable_exception_types", ["ValueError", "TypeError", "KeyError", "AssertionError"]))) if str(item).strip()),
            sleep_enabled=coerce_bool(source.get("sleep_enabled", reliability_config.get("sleep_enabled", True)), default=True),
        )

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the bounded delay for a 1-based retry attempt."""

        attempt_index = coerce_int(attempt, default=1, minimum=1)
        if self.backoff_factor <= 0:
            return 0.0
        base_delay = self.backoff_factor * (2 ** (attempt_index - 1))
        delay = min(self.max_backoff_seconds, max(0.0, base_delay))
        if self.jitter_seconds > 0:
            delay += random.uniform(0.0, self.jitter_seconds)
        return round(max(0.0, delay), 6)

    def should_retry_exception(self, exc: BaseException) -> bool:
        """Classify whether an exception is retryable under this policy."""

        if hasattr(exc, "retryable"):
            return bool(getattr(exc, "retryable"))
        exc_name = type(exc).__name__
        if exc_name in self.non_retryable_exception_types:
            return False
        if self.retryable_exception_types:
            return exc_name in self.retryable_exception_types
        return is_retryable_exception(exc, default=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker configuration for one agent/runtime key."""

    failure_threshold: int = 3
    recovery_timeout_seconds: float = 5.0
    half_open_success_threshold: int = 1
    half_open_max_requests: int = 1
    failure_window_seconds: Optional[float] = None
    minimum_request_count: int = 1
    state_transition_history_limit: int = 100

    @classmethod
    def from_config(cls, reliability_config: Optional[Mapping[str, Any]] = None) -> "CircuitBreakerConfig":
        source = dict(reliability_config or {})
        breaker_source = source.get("circuit_breaker") if isinstance(source.get("circuit_breaker"), Mapping) else {}
        merged = merge_mappings(source, breaker_source, deep=True, drop_none=True)
        failure_window = merged.get("failure_window_seconds")
        return cls(
            failure_threshold=coerce_int(merged.get("failure_threshold"), default=3, minimum=1),
            recovery_timeout_seconds=coerce_float(merged.get("recovery_timeout_seconds"), default=5.0, minimum=0.0),
            half_open_success_threshold=coerce_int(merged.get("half_open_success_threshold"), default=1, minimum=1),
            half_open_max_requests=coerce_int(merged.get("half_open_max_requests"), default=1, minimum=1),
            failure_window_seconds=None if failure_window is None else coerce_float(failure_window, default=0.0, minimum=0.0),
            minimum_request_count=coerce_int(merged.get("minimum_request_count"), default=1, minimum=1),
            state_transition_history_limit=coerce_int(merged.get("state_transition_history_limit"), default=100, minimum=1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReliabilityAttemptRecord:
    """Serializable record for one execution attempt."""

    attempt: int
    status: str
    started_at: float
    finished_at: float
    duration_ms: float
    circuit_state_before: str
    circuit_state_after: str
    delay_before_next_attempt_seconds: float = 0.0
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self), drop_empty=True)


@dataclass(frozen=True)
class ReliabilityExecutionRecord:
    """Bounded telemetry record for one reliability-managed operation."""

    execution_id: str
    key: str
    status: str
    started_at: float
    finished_at: float
    duration_ms: float
    attempts: Tuple[Dict[str, Any], ...]
    result_fingerprint: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("rel"))

    def to_dict(self) -> Dict[str, Any]:
        return redact_mapping(prune_none(asdict(self), drop_empty=True))


class AgentCircuitBreaker:
    """Per-agent circuit breaker with metrics and transition history."""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.RLock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = 0.0
        self._state_since = epoch_seconds()
        self._last_success_at: Optional[float] = None
        self._last_failure_at: Optional[float] = None
        self._last_failure: Optional[Dict[str, Any]] = None
        self._total_requests = 0
        self._total_successes = 0
        self._total_failures = 0
        self._rejected_requests = 0
        self._half_open_in_flight = 0
        self._failure_timestamps: Deque[float] = deque()
        self._transition_history: Deque[Dict[str, Any]] = deque(maxlen=self.config.state_transition_history_limit)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_from_open()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            self._prune_failure_window()
            return self._failure_count

    @property
    def success_count(self) -> int:
        with self._lock:
            return self._success_count

    def allow_request(self) -> bool:
        with self._lock:
            self._maybe_transition_from_open()
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                return self._half_open_in_flight < self.config.half_open_max_requests
            self._rejected_requests += 1
            return False

    def before_request(self) -> bool:
        """Reserve capacity for one request when allowed."""

        with self._lock:
            if not self.allow_request():
                return False
            self._total_requests += 1
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight += 1
            return True

    def record_success(self) -> None:
        with self._lock:
            current = self.state
            self._total_successes += 1
            self._last_success_at = epoch_seconds()
            if current == CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._success_count += 1
                if self._success_count >= self.config.half_open_success_threshold:
                    self._failure_count = 0
                    self._success_count = 0
                    self._failure_timestamps.clear()
                    self._transition_to(CircuitState.CLOSED, reason="half_open_success_threshold_met")
                return

            self._failure_count = 0
            self._success_count = 0
            self._failure_timestamps.clear()
            if current != CircuitState.CLOSED:
                self._transition_to(CircuitState.CLOSED, reason="success")

    def record_failure(self, exc: Optional[BaseException] = None) -> None:
        with self._lock:
            current = self.state
            now = epoch_seconds()
            self._total_failures += 1
            self._last_failure_at = now
            self._last_failure = exception_to_error_payload(exc, action="reliability") if exc is not None else None
            if current == CircuitState.HALF_OPEN:
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._success_count = 0
                self._failure_count = max(1, self._failure_count + 1)
                self._failure_timestamps.append(now)
                self._transition_to(CircuitState.OPEN, reason="half_open_failure")
                return

            self._success_count = 0
            self._failure_count += 1
            self._failure_timestamps.append(now)
            self._prune_failure_window(now)
            if self._should_open():
                self._transition_to(CircuitState.OPEN, reason="failure_threshold_reached")

    def reset(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._success_count = 0
            self._opened_at = 0.0
            self._last_failure = None
            self._half_open_in_flight = 0
            self._failure_timestamps.clear()
            self._transition_to(CircuitState.CLOSED, reason="manual_reset")

    def force_open(self, reason: str = "manual_force_open") -> None:
        with self._lock:
            self._transition_to(CircuitState.OPEN, reason=reason)

    def force_close(self, reason: str = "manual_force_close") -> None:
        with self._lock:
            self._failure_count = 0
            self._success_count = 0
            self._failure_timestamps.clear()
            self._half_open_in_flight = 0
            self._transition_to(CircuitState.CLOSED, reason=reason)

    def snapshot(self, *, include_history: bool = True) -> Dict[str, Any]:
        with self._lock:
            state = self.state
            total_completed = self._total_successes + self._total_failures
            failure_rate = (self._total_failures / total_completed) if total_completed else 0.0
            now = epoch_seconds()
            payload = {
                "state": state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_requests": self._total_requests,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "rejected_requests": self._rejected_requests,
                "failure_rate": round(failure_rate, 6),
                "opened_at": self._opened_at or None,
                "state_since": self._state_since,
                "state_age_seconds": round(now - self._state_since, 6),
                "last_success_at": self._last_success_at,
                "last_failure_at": self._last_failure_at,
                "last_failure": self._last_failure,
                "half_open_in_flight": self._half_open_in_flight,
                "config": self.config.to_dict(),
            }
            if include_history:
                payload["transition_history"] = list(self._transition_history)
            return redact_mapping(prune_none(payload, drop_empty=True))

    def _maybe_transition_from_open(self) -> None:
        if self._state == CircuitState.OPEN and (epoch_seconds() - self._opened_at) >= self.config.recovery_timeout_seconds:
            self._success_count = 0
            self._half_open_in_flight = 0
            self._transition_to(CircuitState.HALF_OPEN, reason="recovery_timeout_elapsed")

    def _transition_to(self, new_state: CircuitState, *, reason: str) -> None:
        old_state = self._state
        now = epoch_seconds()
        if old_state == new_state:
            if new_state == CircuitState.OPEN:
                self._opened_at = now
            return
        self._state = new_state
        self._state_since = now
        if new_state == CircuitState.OPEN:
            self._opened_at = now
        if new_state == CircuitState.CLOSED:
            self._opened_at = 0.0
        self._transition_history.append(
            {
                "from": old_state.value,
                "to": new_state.value,
                "reason": reason,
                "timestamp": now,
                "timestamp_utc": utc_timestamp(),
            }
        )

    def _should_open(self) -> bool:
        self._prune_failure_window()
        if self.config.failure_window_seconds is not None:
            observed = len(self._failure_timestamps)
            return observed >= self.config.minimum_request_count and observed >= self.config.failure_threshold
        return self._failure_count >= self.config.failure_threshold

    def _prune_failure_window(self, now: Optional[float] = None) -> None:
        if self.config.failure_window_seconds is None:
            return
        current = epoch_seconds() if now is None else now
        cutoff = current - self.config.failure_window_seconds
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()
        self._failure_count = len(self._failure_timestamps)


class ReliabilityManager:
    """Retry and circuit-breaker coordinator for collaborative agent execution."""

    def __init__(self, retry_policy: RetryPolicy | None = None, breaker_config: CircuitBreakerConfig | None = None,
                 *, shared_memory: Optional[Any] = None):
        self.config = load_global_config()
        self.reliability_config = get_config_section("reliability") or {}
        self.task_routing_config = get_config_section("task_routing") or {}
        self.retry_policy = retry_policy or RetryPolicy.from_config(self.reliability_config, self.task_routing_config)
        self.breaker_config = breaker_config or CircuitBreakerConfig.from_config(self.reliability_config)
        self.shared_memory = shared_memory
        self._breakers: Dict[str, AgentCircuitBreaker] = {}
        self._lock = threading.RLock()
        self._history: Deque[Dict[str, Any]] = deque(maxlen=coerce_int(self.reliability_config.get("execution_history_limit"), default=1000, minimum=1))
        self._last_execution_by_key: Dict[str, Dict[str, Any]] = {}
        self._audit_enabled = coerce_bool(self.reliability_config.get("audit_enabled"), default=True)
        self._audit_key = str(self.reliability_config.get("audit_key", "collaboration:reliability_events"))
        self._audit_max_events = coerce_int(self.reliability_config.get("audit_max_events"), default=1000, minimum=1)
        self._status_key = str(self.reliability_config.get("status_key", "collaboration:reliability_status"))
        self._publish_status_enabled = coerce_bool(self.reliability_config.get("publish_status"), default=True)
        self._include_success_events = coerce_bool(self.reliability_config.get("include_success_events"), default=False)
        self._last_status_publish_at = 0.0

        logger.info("Reliability Manager initialized")

    def _get_breaker(self, key: str) -> AgentCircuitBreaker:
        normalized_key = self._normalize_key(key)
        with self._lock:
            if normalized_key not in self._breakers:
                self._breakers[normalized_key] = AgentCircuitBreaker(config=self.breaker_config)
                self._audit(
                    "breaker_created",
                    f"Created circuit breaker for '{normalized_key}'.",
                    key=normalized_key,
                    severity="debug",
                )
            return self._breakers[normalized_key]

    def get_breaker(self, key: str) -> AgentCircuitBreaker:
        """Public accessor for advanced diagnostics/tests."""

        return self._get_breaker(key)

    def is_available(self, key: str) -> bool:
        return self._get_breaker(key).allow_request()

    def execute(
        self,
        key: str,
        operation: Callable[[], T],
        *,
        retry_policy: Optional[RetryPolicy] = None,
        retryable: Optional[RetryPredicate] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        on_attempt: Optional[AttemptCallback] = None,
        sleep: Optional[bool] = None,
    ) -> T:
        """Execute an operation with retry and circuit-breaker protection.

        Args:
            key: Agent/runtime key that owns the circuit breaker.
            operation: Zero-argument callable to execute.
            retry_policy: Optional per-call policy override.
            retryable: Optional exception predicate override.
            metadata: Optional JSON-safe telemetry metadata.
            on_attempt: Optional callback receiving each attempt record dict.
            sleep: Override policy sleep behavior, useful for tests.
        """

        normalized_key = self._normalize_key(key)
        if not callable(operation):
            raise make_collaboration_exception(
                "RoutingFailureError",
                "Reliability operation must be callable.",
                context={"key": normalized_key, "operation_type": type(operation).__name__},
            )

        policy = retry_policy or self.retry_policy
        predicate = retryable or policy.should_retry_exception
        should_sleep = policy.sleep_enabled if sleep is None else bool(sleep)
        attempts_allowed = max(1, int(policy.max_attempts))
        execution_id = generate_uuid("rel_exec", length=24)
        correlation_id = generate_correlation_id("reliability")
        started_at = epoch_seconds()
        start_ms = monotonic_ms()
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[BaseException] = None
        result_fingerprint: Optional[str] = None

        for attempt in range(1, attempts_allowed + 1):
            breaker = self._get_breaker(normalized_key)
            state_before = breaker.state.value
            if not breaker.before_request():
                circuit_error = self._circuit_open_error(normalized_key, breaker)
                last_error = circuit_error
                attempt_record = self._build_attempt_record(
                    attempt=attempt,
                    status="circuit_open",
                    started_at=epoch_seconds(),
                    finished_at=epoch_seconds(),
                    circuit_state_before=state_before,
                    circuit_state_after=breaker.state.value,
                    error=circuit_error,
                )
                attempts.append(attempt_record.to_dict())
                self._emit_attempt(on_attempt, attempt_record)
                if attempt >= attempts_allowed or not policy.retry_on_circuit_open:
                    self._finalize_execution(
                        key=normalized_key,
                        status="circuit_open",
                        started_at=started_at,
                        duration_ms=elapsed_ms(start_ms),
                        attempts=attempts,
                        error=circuit_error,
                        execution_id=execution_id,
                        correlation_id=correlation_id,
                        metadata=metadata,
                    )
                    raise circuit_error
                delay = self._sleep_backoff(attempt, policy=policy, sleep=should_sleep)
                attempts[-1]["delay_before_next_attempt_seconds"] = delay
                continue

            attempt_started_at = epoch_seconds()
            attempt_start_ms = monotonic_ms()
            try:
                result = operation()
                breaker.record_success()
                result_fingerprint = stable_hash(json_safe(result), length=16)
                attempt_record = self._build_attempt_record(
                    attempt=attempt,
                    status="success",
                    started_at=attempt_started_at,
                    finished_at=epoch_seconds(),
                    circuit_state_before=state_before,
                    circuit_state_after=breaker.state.value,
                    duration_ms=elapsed_ms(attempt_start_ms),
                )
                attempts.append(attempt_record.to_dict())
                self._emit_attempt(on_attempt, attempt_record)
                self._finalize_execution(
                    key=normalized_key,
                    status="success",
                    started_at=started_at,
                    duration_ms=elapsed_ms(start_ms),
                    attempts=attempts,
                    result_fingerprint=result_fingerprint,
                    execution_id=execution_id,
                    correlation_id=correlation_id,
                    metadata=metadata,
                )
                if self._include_success_events:
                    self._audit("operation_success", f"Reliability operation succeeded for '{normalized_key}'.", key=normalized_key, severity="info")
                return result
            except BaseException as exc:  # noqa: BLE001 - reliability boundary intentionally catches all operation failures.
                last_error = exc
                breaker.record_failure(exc)
                retry_allowed = attempt < attempts_allowed and predicate(exc) and (breaker.allow_request() or policy.retry_on_circuit_open)
                delay = self._sleep_backoff(attempt, policy=policy, sleep=should_sleep) if retry_allowed else 0.0
                attempt_record = self._build_attempt_record(
                    attempt=attempt,
                    status="retrying" if retry_allowed else "failed",
                    started_at=attempt_started_at,
                    finished_at=epoch_seconds(),
                    circuit_state_before=state_before,
                    circuit_state_after=breaker.state.value,
                    duration_ms=elapsed_ms(attempt_start_ms),
                    delay_before_next_attempt_seconds=delay,
                    error=exc,
                )
                attempts.append(attempt_record.to_dict())
                self._emit_attempt(on_attempt, attempt_record)
                self._audit(
                    "operation_attempt_failed",
                    f"Reliability attempt {attempt} failed for '{normalized_key}'.",
                    key=normalized_key,
                    severity="warning" if retry_allowed else "error",
                    error=exc,
                    metadata={"attempt": attempt, "retry_allowed": retry_allowed, "state": breaker.state.value},
                )
                if retry_allowed:
                    continue
                break

        if last_error is not None:
            self._finalize_execution(
                key=normalized_key,
                status="failed",
                started_at=started_at,
                duration_ms=elapsed_ms(start_ms),
                attempts=attempts,
                error=last_error,
                execution_id=execution_id,
                correlation_id=correlation_id,
                metadata=metadata,
            )
            raise last_error

        terminal_error = make_collaboration_exception(
            "RoutingFailureError",
            f"Reliability operation failed for '{normalized_key}' without an exception object.",
            context={"key": normalized_key, "attempts": attempts_allowed},
        )
        self._finalize_execution(
            key=normalized_key,
            status="failed",
            started_at=started_at,
            duration_ms=elapsed_ms(start_ms),
            attempts=attempts,
            error=terminal_error,
            execution_id=execution_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )
        raise terminal_error

    def status(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                key: {
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                    "success_count": breaker.success_count,
                    "total_requests": breaker.snapshot(include_history=False).get("total_requests", 0),
                    "total_successes": breaker.snapshot(include_history=False).get("total_successes", 0),
                    "total_failures": breaker.snapshot(include_history=False).get("total_failures", 0),
                    "rejected_requests": breaker.snapshot(include_history=False).get("rejected_requests", 0),
                }
                for key, breaker in sorted(self._breakers.items())
            }

    def detailed_status(self, *, include_history: bool = True) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {key: breaker.snapshot(include_history=include_history) for key, breaker in sorted(self._breakers.items())}

    def snapshot(self, *, include_history: bool = True) -> Dict[str, Any]:
        with self._lock:
            status_payload = self.detailed_status(include_history=include_history)
            open_count = sum(1 for payload in status_payload.values() if payload.get("state") == CircuitState.OPEN.value)
            half_open_count = sum(1 for payload in status_payload.values() if payload.get("state") == CircuitState.HALF_OPEN.value)
            return redact_mapping(
                {
                    "component": "reliability_manager",
                    "captured_at": epoch_seconds(),
                    "captured_at_utc": utc_timestamp(),
                    "breaker_count": len(status_payload),
                    "open_circuit_count": open_count,
                    "half_open_circuit_count": half_open_count,
                    "status": "degraded" if open_count or half_open_count else "healthy",
                    "retry_policy": self.retry_policy.to_dict(),
                    "breaker_config": self.breaker_config.to_dict(),
                    "breakers": status_payload,
                    "history_size": len(self._history),
                    "last_execution_by_key": self._last_execution_by_key,
                }
            )

    def health_report(self) -> Dict[str, Any]:
        snapshot = self.snapshot(include_history=False)
        status = snapshot.get("status", "unknown")
        return {
            "status": status,
            "captured_at": snapshot.get("captured_at"),
            "captured_at_utc": snapshot.get("captured_at_utc"),
            "summary": {
                "breaker_count": snapshot.get("breaker_count", 0),
                "open_circuit_count": snapshot.get("open_circuit_count", 0),
                "half_open_circuit_count": snapshot.get("half_open_circuit_count", 0),
                "history_size": snapshot.get("history_size", 0),
            },
        }

    def execution_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._history)
        if limit is not None:
            return items[-max(0, int(limit)):]
        return items

    def last_execution(self, key: str) -> Optional[Dict[str, Any]]:
        return self._last_execution_by_key.get(self._normalize_key(key))

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                for breaker in self._breakers.values():
                    breaker.reset()
                self._audit("reset_all", "Reset all circuit breakers.", severity="info")
            else:
                normalized_key = self._normalize_key(key)
                self._get_breaker(normalized_key).reset()
                self._audit("reset_breaker", f"Reset circuit breaker for '{normalized_key}'.", key=normalized_key, severity="info")
            self._publish_status()

    def force_open(self, key: str, *, reason: str = "manual_force_open") -> None:
        normalized_key = self._normalize_key(key)
        self._get_breaker(normalized_key).force_open(reason=reason)
        self._audit("force_open", f"Forced circuit open for '{normalized_key}'.", key=normalized_key, severity="warning", metadata={"reason": reason})
        self._publish_status()

    def force_close(self, key: str, *, reason: str = "manual_force_close") -> None:
        normalized_key = self._normalize_key(key)
        self._get_breaker(normalized_key).force_close(reason=reason)
        self._audit("force_close", f"Forced circuit closed for '{normalized_key}'.", key=normalized_key, severity="info", metadata={"reason": reason})
        self._publish_status()

    def remove_breaker(self, key: str) -> bool:
        normalized_key = self._normalize_key(key)
        with self._lock:
            removed = self._breakers.pop(normalized_key, None) is not None
        if removed:
            self._audit("breaker_removed", f"Removed circuit breaker for '{normalized_key}'.", key=normalized_key, severity="info")
            self._publish_status()
        return removed

    def list_keys(self) -> List[str]:
        with self._lock:
            return sorted(self._breakers.keys())

    def record_success(self, key: str) -> None:
        normalized_key = self._normalize_key(key)
        self._get_breaker(normalized_key).record_success()
        self._publish_status()

    def record_failure(self, key: str, exc: Optional[BaseException] = None) -> None:
        normalized_key = self._normalize_key(key)
        self._get_breaker(normalized_key).record_failure(exc)
        self._publish_status()

    def _sleep_backoff(self, attempt: int, *, policy: Optional[RetryPolicy] = None, sleep: bool = True) -> float:
        active_policy = policy or self.retry_policy
        delay = active_policy.delay_for_attempt(attempt)
        if sleep and delay > 0:
            time.sleep(delay)
        return delay

    def _normalize_key(self, key: Any) -> str:
        return normalize_agent_name(key)

    def _circuit_open_error(self, key: str, breaker: AgentCircuitBreaker) -> Exception:
        snapshot = breaker.snapshot(include_history=False)
        return make_collaboration_exception(
            "RoutingFailureError",
            f"Circuit open for '{key}'.",
            context={"key": key, "breaker": snapshot},
            collaborative_agent_state={"reliability_status": self.status()},
        )

    def _build_attempt_record(
        self,
        *,
        attempt: int,
        status: str,
        started_at: float,
        finished_at: float,
        circuit_state_before: str,
        circuit_state_after: str,
        duration_ms: Optional[float] = None,
        delay_before_next_attempt_seconds: float = 0.0,
        error: Optional[BaseException] = None,
    ) -> ReliabilityAttemptRecord:
        if duration_ms is None:
            duration_ms = round(max(0.0, (finished_at - started_at) * 1000.0), 3)
        return ReliabilityAttemptRecord(
            attempt=attempt,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            circuit_state_before=circuit_state_before,
            circuit_state_after=circuit_state_after,
            delay_before_next_attempt_seconds=delay_before_next_attempt_seconds,
            error=exception_to_error_payload(error, action="reliability_attempt").get("error") if error is not None else None,
        )

    def _emit_attempt(self, callback: Optional[AttemptCallback], record: ReliabilityAttemptRecord) -> None:
        if callback is None:
            return
        callback(record.to_dict())

    def _finalize_execution(
        self,
        *,
        key: str,
        status: str,
        started_at: float,
        duration_ms: float,
        attempts: Sequence[Mapping[str, Any]],
        execution_id: str,
        correlation_id: str,
        result_fingerprint: Optional[str] = None,
        error: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        finished_at = epoch_seconds()
        record = ReliabilityExecutionRecord(
            execution_id=execution_id,
            key=key,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            attempts=tuple(dict(item) for item in attempts),
            result_fingerprint=result_fingerprint,
            error=exception_to_error_payload(error, action="reliability_execute").get("error") if error is not None else None,
            metadata=normalize_metadata(metadata, drop_none=True, redact=True),
            correlation_id=correlation_id,
        ).to_dict()
        with self._lock:
            self._history.append(record)
            self._last_execution_by_key[key] = record
        self._publish_status()

    def _publish_status(self) -> None:
        if self.shared_memory is None or not self._publish_status_enabled:
            return
        payload = self.snapshot(include_history=False)
        memory_set(self.shared_memory, self._status_key, payload)
        self._last_status_publish_at = epoch_seconds()

    def _audit(self, event_type: str, message: str, *, key: Optional[str] = None, severity: str = "info",
               error: Optional[BaseException] = None, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if self.shared_memory is None or not self._audit_enabled:
            return
        event = build_audit_event(
            event_type=event_type,
            message=message,
            severity=severity,
            component="reliability_manager",
            agent_name=key,
            error=error,
            metadata=metadata,
        )
        append_audit_event(self.shared_memory, event, key=self._audit_key, max_events=self._audit_max_events)


if __name__ == "__main__":
    print("\n=== Running Reliability Manager ===\n")
    printer.status("TEST", "Reliability Manager initialized", "info")
    from .shared_memory import SharedMemory

    memory = SharedMemory()
    retry_policy = RetryPolicy(max_attempts=3, backoff_factor=0.0, max_backoff_seconds=0.0, jitter_seconds=0.0)
    breaker_config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=0.05, half_open_success_threshold=1)
    manager = ReliabilityManager(retry_policy=retry_policy, breaker_config=breaker_config, shared_memory=memory)

    attempts = {"count": 0}

    def flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient failure")
        return "ok"

    assert manager.execute("AgentA", flaky_operation, sleep=False) == "ok"
    assert attempts["count"] == 2
    assert manager.is_available("AgentA") is True
    assert manager.status()["AgentA"]["state"] == CircuitState.CLOSED.value

    def failing_operation() -> None:
        raise RuntimeError("persistent failure")

    for _ in range(2):
        try:
            manager.execute("AgentB", failing_operation, sleep=False)
        except Exception:
            pass
    assert manager.status()["AgentB"]["state"] == CircuitState.OPEN.value
    assert manager.is_available("AgentB") is False

    try:
        manager.execute("AgentB", lambda: "blocked", sleep=False)
    except Exception as exc:
        assert "Circuit open" in str(exc)

    time.sleep(0.06)
    assert manager.is_available("AgentB") is True
    assert manager.status()["AgentB"]["state"] == CircuitState.HALF_OPEN.value
    assert manager.execute("AgentB", lambda: "recovered", sleep=False) == "recovered"
    assert manager.status()["AgentB"]["state"] == CircuitState.CLOSED.value

    manager.force_open("AgentC", reason="smoke_test")
    assert manager.status()["AgentC"]["state"] == CircuitState.OPEN.value
    manager.force_close("AgentC", reason="smoke_test_complete")
    assert manager.status()["AgentC"]["state"] == CircuitState.CLOSED.value
    manager.reset("AgentC")
    assert manager.remove_breaker("AgentC") is True

    snapshot = manager.snapshot()
    assert snapshot["breaker_count"] >= 2
    assert manager.health_report()["status"] in {"healthy", "degraded"}
    assert manager.execution_history(limit=2)
    assert memory.get("collaboration:reliability_status") is not None

    printer.status("TEST", "Reliability Manager execute/retry/circuit/status checks passed", "success")
    print("\n=== Test ran successfully ===\n")
