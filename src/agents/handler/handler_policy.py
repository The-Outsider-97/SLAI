from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.handler_error import *
from .utils.handler_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Handler Policy")
printer = PrettyPrinter()


@dataclass(frozen=True)
class PolicyDecision:
    """
    Explainable guardrail decision emitted by HandlerPolicy.

    HandlerAgent can continue to call can_attempt(...) and retries_allowed(...), while
    advanced orchestration can call evaluate_attempt(...) for full policy context.
    """

    agent_name: str
    allowed: bool
    reason: str
    breaker_state: str
    breaker_open: bool
    retry_allowed: bool
    retry_limit: int
    attempted_retries: int = 0
    failures_in_window: int = 0
    weighted_failures_in_window: float = 0.0
    events_in_window: int = 0
    failure_rate: float = 0.0
    cooldown_seconds_remaining: float = 0.0
    open_until: float = 0.0
    half_open_probe_available: bool = False
    severity: str = FailureSeverity.LOW.value
    category: str = "runtime"
    retryable: bool = True
    action: str = HandlerRecoveryAction.NONE.value
    evaluator_hooks_enabled: bool = True
    budget_exhausted: bool = False
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=utc_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "schema": "handler.policy_decision.v2",
                "agent_name": self.agent_name,
                "allowed": self.allowed,
                "reason": self.reason,
                "breaker_state": self.breaker_state,
                "breaker_open": self.breaker_open,
                "retry_allowed": self.retry_allowed,
                "retry_limit": self.retry_limit,
                "attempted_retries": self.attempted_retries,
                "failures_in_window": self.failures_in_window,
                "weighted_failures_in_window": self.weighted_failures_in_window,
                "events_in_window": self.events_in_window,
                "failure_rate": self.failure_rate,
                "cooldown_seconds_remaining": self.cooldown_seconds_remaining,
                "open_until": self.open_until,
                "half_open_probe_available": self.half_open_probe_available,
                "severity": self.severity,
                "category": self.category,
                "retryable": self.retryable,
                "action": self.action,
                "evaluator_hooks_enabled": self.evaluator_hooks_enabled,
                "budget_exhausted": self.budget_exhausted,
                "correlation_id": self.correlation_id,
                "metadata": self.metadata,
                "timestamp": self.timestamp,
            },
            drop_none=True,
            drop_empty=True,
        )


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    """Public circuit-breaker state for one agent."""

    agent_name: str
    state: str
    is_open: bool
    open_until: float
    seconds_remaining: float
    failure_count: int
    success_count: int
    failures_in_window: int
    weighted_failures_in_window: float
    events_in_window: int
    failure_rate: float
    half_open_probes_used: int
    half_open_probe_limit: int
    opened_count: int
    last_opened_reason: Optional[str] = None
    last_failure_ts: Optional[float] = None
    last_success_ts: Optional[float] = None
    timestamp: float = field(default_factory=utc_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "schema": "handler.circuit_breaker.v2",
                "agent_name": self.agent_name,
                "state": self.state,
                "is_open": self.is_open,
                "open_until": self.open_until,
                "seconds_remaining": self.seconds_remaining,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "failures_in_window": self.failures_in_window,
                "weighted_failures_in_window": self.weighted_failures_in_window,
                "events_in_window": self.events_in_window,
                "failure_rate": self.failure_rate,
                "half_open_probes_used": self.half_open_probes_used,
                "half_open_probe_limit": self.half_open_probe_limit,
                "opened_count": self.opened_count,
                "last_opened_reason": self.last_opened_reason,
                "last_failure_ts": self.last_failure_ts,
                "last_success_ts": self.last_success_ts,
                "timestamp": self.timestamp,
            },
            drop_none=True,
        )


class HandlerPolicy:
    """
    Production Handler policy guardrails for retries, circuit-breakers, and evaluator hooks.

    Scope:
    - gates whether a target agent may be attempted
    - enforces retry-count limits supplied by AdaptiveRetryPolicy or defaults
    - owns circuit-breaker state, rolling failure budgets, cooldowns, and half-open probes
    - emits bounded policy telemetry to memory/shared-memory-like stores when configured
    - exposes explainable policy decisions for HandlerAgent and orchestration layers

    This module intentionally does not compute adaptive retry budgets, select recovery
    strategies, perform escalation routing, or store long-term memory. Those concerns remain
    in AdaptiveRetryPolicy, strategy selector, EscalationManager, and HandlerMemory.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    POLICY_EVENT_KEY = "handler:policy_events"

    DEFAULT_FAILURE_WEIGHT_BY_SEVERITY: Mapping[str, float] = {
        FailureSeverity.LOW.value: 1.0,
        FailureSeverity.MEDIUM.value: 1.5,
        FailureSeverity.HIGH.value: 2.5,
        FailureSeverity.CRITICAL.value: 4.0,
    }
    DEFAULT_FAILURE_WEIGHT_BY_CATEGORY: Mapping[str, float] = {
        "security": 4.0,
        "memory": 2.5,
        "dependency": 2.0,
        "sla": 2.0,
        "resource": 1.5,
        "runtime": 1.25,
        "timeout": 1.0,
        "network": 1.0,
        "validation": 1.0,
        "unicode": 0.75,
    }
    DEFAULT_SEVERITY_COOLDOWN_MULTIPLIER: Mapping[str, float] = {
        FailureSeverity.LOW.value: 0.75,
        FailureSeverity.MEDIUM.value: 1.0,
        FailureSeverity.HIGH.value: 1.5,
        FailureSeverity.CRITICAL.value: 2.0,
    }
    DEFAULT_CATEGORY_COOLDOWN_MULTIPLIER: Mapping[str, float] = {
        "security": 2.0,
        "memory": 1.5,
        "dependency": 1.25,
        "sla": 1.25,
        "runtime": 1.0,
        "resource": 1.0,
        "timeout": 0.8,
        "network": 0.8,
        "validation": 1.0,
        "unicode": 0.75,
    }
    NON_ATTEMPT_ACTIONS: Tuple[str, ...] = (
        HandlerRecoveryAction.FAIL_FAST.value,
        HandlerRecoveryAction.QUARANTINE.value,
    )

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *,
                 memory: Any = None, error_policy: Optional[HandlerErrorPolicy] = None):
        self.config = load_global_config()
        self.policy_config = get_config_section("policy")

        merged = deep_merge(self.policy_config, config or {})
        policy_config = merged.get("error_policy") if isinstance(merged.get("error_policy"), Mapping) else None
        self.error_policy = error_policy or HandlerErrorPolicy.from_mapping(policy_config)
        self.memory = memory

        self.enabled = coerce_bool(merged.get("enabled"), default=True)
        self.max_retries = coerce_int(merged.get("max_retries"), 2, minimum=0, maximum=100)
        self.circuit_breaker_enabled = coerce_bool(merged.get("circuit_breaker_enabled"), default=True)
        self.circuit_breaker_threshold = coerce_int(merged.get("circuit_breaker_threshold"), 5, minimum=1, maximum=100_000)
        self.cooldown_seconds = coerce_float(merged.get("cooldown_seconds"), 30.0, minimum=0.0)
        self.max_cooldown_seconds = coerce_float(merged.get("max_cooldown_seconds"), max(30.0, self.cooldown_seconds * 8), minimum=self.cooldown_seconds)
        self.cooldown_backoff_multiplier = coerce_float(merged.get("cooldown_backoff_multiplier"), 1.0, minimum=1.0)
        self.failure_budget_window_seconds = coerce_float(merged.get("failure_budget_window_seconds"), 300.0, minimum=1.0)
        self.failure_budget_threshold = coerce_float(merged.get("failure_budget_threshold"), float(self.circuit_breaker_threshold), minimum=1.0)
        self.failure_rate_threshold = coerce_float(merged.get("failure_rate_threshold"), 0.80, minimum=0.0, maximum=1.0)
        self.min_events_for_failure_rate = coerce_int(merged.get("min_events_for_failure_rate"), 4, minimum=1)
        self.weighted_budget_enabled = coerce_bool(merged.get("weighted_budget_enabled"), default=True)
        self.half_open_enabled = coerce_bool(merged.get("half_open_enabled"), default=True)
        self.half_open_max_probes = coerce_int(merged.get("half_open_max_probes"), 1, minimum=1, maximum=10_000)
        self.half_open_success_threshold = coerce_int(merged.get("half_open_success_threshold"), 1, minimum=1, maximum=10_000)
        self.success_reset_threshold = coerce_int(merged.get("success_reset_threshold"), 1, minimum=1, maximum=10_000)
        self.open_on_critical_non_retryable = coerce_bool(merged.get("open_on_critical_non_retryable"), default=True)
        self.open_on_quarantine = coerce_bool(merged.get("open_on_quarantine"), default=True)
        self.block_non_retryable_attempts = coerce_bool(merged.get("block_non_retryable_attempts"), default=False)
        self.evaluator_hooks_enabled = coerce_bool(merged.get("evaluator_hooks_enabled"), default=True)
        self.emit_to_memory = coerce_bool(merged.get("emit_to_memory"), default=False)
        self.memory_event_type = normalize_identifier(merged.get("memory_event_type"), default="handler_policy")
        self.max_policy_events = coerce_int(merged.get("max_policy_events"), 1000, minimum=1, maximum=1_000_000)
        self.policy_event_ttl_seconds = self._optional_int(merged.get("policy_event_ttl_seconds"), default=None)
        self.max_agent_name_chars = coerce_int(merged.get("max_agent_name_chars"), 96, minimum=8, maximum=512)
        self.include_failure_in_events = coerce_bool(merged.get("include_failure_in_events"), default=True)
        self.include_context_in_events = coerce_bool(merged.get("include_context_in_events"), default=True)

        self.failure_weight_by_severity = self._coerce_float_mapping(
            merged.get("failure_weight_by_severity"),
            default=self.DEFAULT_FAILURE_WEIGHT_BY_SEVERITY,
            minimum=0.0,
            maximum=100.0,
        )
        self.failure_weight_by_category = self._coerce_float_mapping(
            merged.get("failure_weight_by_category"),
            default=self.DEFAULT_FAILURE_WEIGHT_BY_CATEGORY,
            minimum=0.0,
            maximum=100.0,
        )
        self.severity_cooldown_multiplier = self._coerce_float_mapping(
            merged.get("severity_cooldown_multiplier"),
            default=self.DEFAULT_SEVERITY_COOLDOWN_MULTIPLIER,
            minimum=0.0,
            maximum=100.0,
        )
        self.category_cooldown_multiplier = self._coerce_float_mapping(
            merged.get("category_cooldown_multiplier"),
            default=self.DEFAULT_CATEGORY_COOLDOWN_MULTIPLIER,
            minimum=0.0,
            maximum=100.0,
        )
        self.agent_overrides = coerce_mapping(merged.get("agent_overrides"))
        self.category_overrides = coerce_mapping(merged.get("category_overrides"))
        self.action_overrides = coerce_mapping(merged.get("action_overrides"))

        self._lock = RLock()
        self._failure_counters: Dict[str, int] = defaultdict(int)
        self._success_counters: Dict[str, int] = defaultdict(int)
        self._breaker_state: Dict[str, str] = defaultdict(lambda: self.CLOSED)
        self._breaker_open_until: Dict[str, float] = defaultdict(float)
        self._half_open_probes: Dict[str, int] = defaultdict(int)
        self._opened_counts: Dict[str, int] = defaultdict(int)
        self._last_opened_reason: Dict[str, str] = {}
        self._last_failure_ts: Dict[str, float] = {}
        self._last_success_ts: Dict[str, float] = {}
        self._last_failures: Dict[str, List[float]] = defaultdict(list)
        self._failure_events: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.max_policy_events))
        self._policy_events: Deque[Dict[str, Any]] = deque(maxlen=self.max_policy_events)

        self._validate_configuration()
        logger.info(
            "Handler Policy successfully initialized | max_retries=%s threshold=%s cooldown=%s enabled=%s",
            self.max_retries,
            self.circuit_breaker_threshold,
            self.cooldown_seconds,
            self.enabled,
        )

    def attach_memory(self, memory: Any) -> None:
        """Attach a HandlerMemory-like object after construction."""
        self.memory = memory

    def can_attempt(self, agent_name: str, normalized_failure: Optional[Mapping[str, Any]] = None,
                    context: Optional[Mapping[str, Any]] = None) -> bool:
        """Legacy API: return True when policy allows an attempt for agent_name."""
        return self.evaluate_attempt(agent_name, normalized_failure=normalized_failure, context=context).allowed

    def retries_allowed(self, attempted_retries: int, max_retries: Optional[int] = None,
                        *, normalized_failure: Optional[Mapping[str, Any]] = None) -> bool:
        """Legacy API: return True when attempted_retries is below the active retry limit."""
        attempted = coerce_int(attempted_retries, 0, minimum=0)
        limit = self._retry_limit(max_retries=max_retries, normalized_failure=normalized_failure)
        if normalized_failure is not None:
            failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
            action = normalize_recovery_action(failure.get("policy_action") or failure.get("action"))
            if action in self.NON_ATTEMPT_ACTIONS:
                return False
            if self.block_non_retryable_attempts and not coerce_bool(failure.get("retryable"), default=True):
                return False
        return attempted < limit

    def evaluate_attempt(self, agent_name: str, *, normalized_failure: Optional[Mapping[str, Any]] = None,
                         context: Optional[Mapping[str, Any]] = None, attempted_retries: int = 0,
                         max_retries: Optional[int] = None) -> PolicyDecision:
        """Return a full policy decision for an agent attempt."""
        agent = self._normalize_agent_name(agent_name)
        context_map = coerce_mapping(context)
        failure = self._resolve_failure(normalized_failure)
        severity = normalize_severity(failure.get("severity"))
        retryable = coerce_bool(failure.get("retryable"), default=True)
        category = normalize_identifier(failure.get("category") or classify_failure_category(failure.get("type"), failure.get("message")), default="runtime")
        action = normalize_recovery_action(failure.get("policy_action") or failure.get("action"))
        attempted = coerce_int(attempted_retries, 0, minimum=0)
        retry_limit = self._retry_limit(max_retries=max_retries, normalized_failure=failure)
        retry_allowed = self.retries_allowed(attempted, max_retries=retry_limit, normalized_failure=failure)
        correlation_id = failure.get("correlation_id") or context_map.get("correlation_id")

        with self._lock:
            self._refresh_breaker_locked(agent)
            snapshot = self._snapshot_locked(agent)

            if not self.enabled:
                return self._build_decision(
                    agent=agent,
                    allowed=True,
                    reason="handler_policy_disabled",
                    snapshot=snapshot,
                    retry_allowed=retry_allowed,
                    retry_limit=retry_limit,
                    attempted=attempted,
                    severity=severity,
                    category=category,
                    retryable=retryable,
                    action=action,
                    correlation_id=correlation_id,
                )

            if self.block_non_retryable_attempts and not retryable:
                return self._build_decision(
                    agent=agent,
                    allowed=False,
                    reason="failure_marked_non_retryable",
                    snapshot=snapshot,
                    retry_allowed=False,
                    retry_limit=retry_limit,
                    attempted=attempted,
                    severity=severity,
                    category=category,
                    retryable=retryable,
                    action=action,
                    correlation_id=correlation_id,
                )

            if action in self.NON_ATTEMPT_ACTIONS:
                return self._build_decision(
                    agent=agent,
                    allowed=False,
                    reason=f"policy_action_{action}_blocks_attempt",
                    snapshot=snapshot,
                    retry_allowed=False,
                    retry_limit=retry_limit,
                    attempted=attempted,
                    severity=severity,
                    category=category,
                    retryable=False,
                    action=action,
                    correlation_id=correlation_id,
                )

            if not retry_allowed:
                return self._build_decision(
                    agent=agent,
                    allowed=False,
                    reason="retry_limit_exhausted",
                    snapshot=snapshot,
                    retry_allowed=False,
                    retry_limit=retry_limit,
                    attempted=attempted,
                    severity=severity,
                    category=category,
                    retryable=retryable,
                    action=action,
                    correlation_id=correlation_id,
                )

            if snapshot.is_open:
                return self._build_decision(
                    agent=agent,
                    allowed=False,
                    reason="circuit_breaker_open",
                    snapshot=snapshot,
                    retry_allowed=retry_allowed,
                    retry_limit=retry_limit,
                    attempted=attempted,
                    severity=severity,
                    category=category,
                    retryable=retryable,
                    action=action,
                    correlation_id=correlation_id,
                )

            if snapshot.state == self.HALF_OPEN:
                if self._half_open_probes[agent] >= self.half_open_max_probes:
                    return self._build_decision(
                        agent=agent,
                        allowed=False,
                        reason="half_open_probe_limit_reached",
                        snapshot=snapshot,
                        retry_allowed=retry_allowed,
                        retry_limit=retry_limit,
                        attempted=attempted,
                        severity=severity,
                        category=category,
                        retryable=retryable,
                        action=action,
                        correlation_id=correlation_id,
                    )
                self._half_open_probes[agent] += 1
                snapshot = self._snapshot_locked(agent)
                return self._build_decision(
                    agent=agent,
                    allowed=True,
                    reason="half_open_probe_allowed",
                    snapshot=snapshot,
                    retry_allowed=retry_allowed,
                    retry_limit=retry_limit,
                    attempted=attempted,
                    severity=severity,
                    category=category,
                    retryable=retryable,
                    action=action,
                    correlation_id=correlation_id,
                    metadata={"probe_number": self._half_open_probes[agent]},
                )

            budget_exhausted = self._budget_exhausted_locked(agent)
            if budget_exhausted:
                self._open_breaker_locked(
                    agent,
                    reason="rolling_failure_budget_exhausted",
                    failure=failure,
                    context=context_map,
                )
                snapshot = self._snapshot_locked(agent)
                return self._build_decision(
                    agent=agent,
                    allowed=False,
                    reason="rolling_failure_budget_exhausted",
                    snapshot=snapshot,
                    retry_allowed=retry_allowed,
                    retry_limit=retry_limit,
                    attempted=attempted,
                    severity=severity,
                    category=category,
                    retryable=retryable,
                    action=action,
                    correlation_id=correlation_id,
                    budget_exhausted=True,
                )

            return self._build_decision(
                agent=agent,
                allowed=True,
                reason="allowed",
                snapshot=snapshot,
                retry_allowed=retry_allowed,
                retry_limit=retry_limit,
                attempted=attempted,
                severity=severity,
                category=category,
                retryable=retryable,
                action=action,
                correlation_id=correlation_id,
            )

    def record_failure(
        self,
        agent_name: str,
        normalized_failure: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        *,
        reason: Optional[str] = None,
        weight: Optional[float] = None,
    ) -> None:
        """Record an agent failure and open/extend the circuit breaker when policy requires it."""
        agent = self._normalize_agent_name(agent_name)
        context_map = coerce_mapping(context)
        failure = self._resolve_failure(normalized_failure)
        severity = normalize_severity(failure.get("severity"))
        category = normalize_identifier(failure.get("category") or classify_failure_category(failure.get("type"), failure.get("message")), default="runtime")
        retryable = coerce_bool(failure.get("retryable"), default=False)
        action = normalize_recovery_action(failure.get("policy_action") or failure.get("action"))
        now = utc_timestamp()
        failure_weight = self._failure_weight(severity=severity, category=category, explicit_weight=weight)

        with self._lock:
            self._refresh_breaker_locked(agent)
            self._failure_counters[agent] += 1
            self._success_counters[agent] = 0
            self._last_failure_ts[agent] = now
            self._last_failures[agent].append(now)
            self._failure_events[agent].append(
                compact_dict(
                    {
                        "timestamp": now,
                        "status": "failed",
                        "severity": severity,
                        "category": category,
                        "retryable": retryable,
                        "action": action,
                        "weight": failure_weight,
                        "context_hash": failure.get("context_hash"),
                        "correlation_id": failure.get("correlation_id") or context_map.get("correlation_id"),
                    },
                    drop_none=True,
                )
            )
            self._trim_failure_window_locked(agent, now=now)

            open_reason = self._open_reason_for_failure_locked(
                agent=agent,
                severity=severity,
                category=category,
                retryable=retryable,
                action=action,
                explicit_reason=reason,
            )
            if open_reason:
                self._open_breaker_locked(agent, reason=open_reason, failure=failure, context=context_map)

            self._emit_policy_event_locked(
                event_type="handler_policy_failure",
                agent=agent,
                decision=None,
                failure=failure,
                context=context_map,
                extra={"reason": open_reason or reason or "failure_recorded", "weight": failure_weight},
            )

    def record_success(
        self,
        agent_name: str,
        context: Optional[Mapping[str, Any]] = None,
        *,
        reset_failure_window: bool = False,
    ) -> None:
        """Record success and close/reset circuit state when policy allows it."""
        agent = self._normalize_agent_name(agent_name)
        context_map = coerce_mapping(context)
        now = utc_timestamp()

        with self._lock:
            self._refresh_breaker_locked(agent)
            self._success_counters[agent] += 1
            self._last_success_ts[agent] = now
            self._failure_events[agent].append(
                {
                    "timestamp": now,
                    "status": "recovered",
                    "weight": 0.0,
                    "correlation_id": context_map.get("correlation_id"),
                }
            )
            self._trim_failure_window_locked(agent, now=now)

            if reset_failure_window or self._success_counters[agent] >= self.success_reset_threshold:
                self._failure_counters[agent] = 0
                if reset_failure_window:
                    self._failure_events[agent].clear()
                    self._last_failures[agent] = []

            if self._breaker_state[agent] == self.HALF_OPEN and self._success_counters[agent] >= self.half_open_success_threshold:
                self._close_breaker_locked(agent, reason="half_open_success")
            elif self._breaker_state[agent] == self.OPEN and self._breaker_open_until[agent] <= now:
                self._close_breaker_locked(agent, reason="success_after_cooldown")
            elif self._success_counters[agent] >= self.success_reset_threshold:
                self._breaker_state[agent] = self.CLOSED
                self._breaker_open_until[agent] = 0.0
                self._half_open_probes[agent] = 0

            self._emit_policy_event_locked(
                event_type="handler_policy_success",
                agent=agent,
                decision=None,
                failure=None,
                context=context_map,
                extra={"reason": "success_recorded"},
            )

    def breaker_status(self, agent_name: str) -> Dict[str, Any]:
        """Return circuit-breaker status. Keeps legacy keys and adds production diagnostics."""
        agent = self._normalize_agent_name(agent_name)
        with self._lock:
            self._refresh_breaker_locked(agent)
            return self._snapshot_locked(agent).to_dict()

    def open_breaker(
        self,
        agent_name: str,
        *,
        reason: str = "manual_open",
        normalized_failure: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        cooldown_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Manually open an agent circuit breaker."""
        agent = self._normalize_agent_name(agent_name)
        failure = self._resolve_failure(normalized_failure)
        context_map = coerce_mapping(context)
        with self._lock:
            self._open_breaker_locked(agent, reason=reason, failure=failure, context=context_map, cooldown_seconds=cooldown_seconds)
            return self._snapshot_locked(agent).to_dict()

    def close_breaker(self, agent_name: str, *, reason: str = "manual_close") -> Dict[str, Any]:
        """Manually close an agent circuit breaker."""
        agent = self._normalize_agent_name(agent_name)
        with self._lock:
            self._close_breaker_locked(agent, reason=reason)
            return self._snapshot_locked(agent).to_dict()

    def reset_agent(self, agent_name: str) -> Dict[str, Any]:
        """Clear all policy state for one agent."""
        agent = self._normalize_agent_name(agent_name)
        with self._lock:
            self._failure_counters.pop(agent, None)
            self._success_counters.pop(agent, None)
            self._breaker_state.pop(agent, None)
            self._breaker_open_until.pop(agent, None)
            self._half_open_probes.pop(agent, None)
            self._opened_counts.pop(agent, None)
            self._last_opened_reason.pop(agent, None)
            self._last_failure_ts.pop(agent, None)
            self._last_success_ts.pop(agent, None)
            self._last_failures.pop(agent, None)
            self._failure_events.pop(agent, None)
            return self._snapshot_locked(agent).to_dict()

    def should_call_evaluator(
        self,
        normalized_failure: Optional[Mapping[str, Any]] = None,
        *,
        decision: Optional[PolicyDecision] = None,
    ) -> bool:
        """Return whether evaluator/evaluation hooks should be invoked for this policy state."""
        if not self.evaluator_hooks_enabled:
            return False
        if decision is not None:
            return (not decision.allowed) or decision.breaker_open or decision.severity in {FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value}
        failure = self._resolve_failure(normalized_failure)
        severity = normalize_severity(failure.get("severity"))
        action = normalize_recovery_action(failure.get("policy_action") or failure.get("action"))
        return severity in {FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value} or action in self.NON_ATTEMPT_ACTIONS

    def recent_policy_events(self, limit: int = 100, *, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return recent policy events emitted by this policy object."""
        safe_limit = coerce_int(limit, 100, minimum=1, maximum=self.max_policy_events)
        with self._lock:
            events = list(self._policy_events)
            if agent_name:
                agent = self._normalize_agent_name(agent_name)
                events = [event for event in events if event.get("agent_name") == agent]
            return [make_json_safe(event) for event in events[-safe_limit:]]  # type: ignore[list-item]

    def policy_snapshot(self, *, include_events: bool = False) -> Dict[str, Any]:
        """Return operational policy state for diagnostics/tests."""
        with self._lock:
            agents = sorted(
                set(self._failure_counters)
                | set(self._success_counters)
                | set(self._breaker_state)
                | set(self._failure_events)
            )
            snapshot: Dict[str, Any] = {
                "schema": "handler.policy.snapshot.v2",
                "timestamp": utc_timestamp(),
                "enabled": self.enabled,
                "config": {
                    "max_retries": self.max_retries,
                    "circuit_breaker_enabled": self.circuit_breaker_enabled,
                    "circuit_breaker_threshold": self.circuit_breaker_threshold,
                    "cooldown_seconds": self.cooldown_seconds,
                    "failure_budget_window_seconds": self.failure_budget_window_seconds,
                    "failure_budget_threshold": self.failure_budget_threshold,
                    "failure_rate_threshold": self.failure_rate_threshold,
                    "half_open_enabled": self.half_open_enabled,
                    "half_open_max_probes": self.half_open_max_probes,
                    "evaluator_hooks_enabled": self.evaluator_hooks_enabled,
                },
                "agents": {agent: self._snapshot_locked(agent).to_dict() for agent in agents},
                "policy_events": len(self._policy_events),
            }
            if include_events:
                snapshot["events"] = [make_json_safe(event) for event in self._policy_events]
            return snapshot

    def health(self) -> Dict[str, Any]:
        """Return compact policy health for dashboards and smoke tests."""
        with self._lock:
            states = Counter(self._breaker_state.values())
            agents = set(self._failure_counters) | set(self._breaker_state) | set(self._failure_events)
            return {
                "status": "ok",
                "timestamp": utc_timestamp(),
                "enabled": self.enabled,
                "agents_tracked": len(agents),
                "breaker_states": dict(states),
                "open_breakers": sum(1 for agent in agents if self._snapshot_locked(agent).is_open),
                "policy_events": len(self._policy_events),
                "evaluator_hooks_enabled": self.evaluator_hooks_enabled,
            }

    def export_state(self) -> Dict[str, Any]:
        """Export JSON-safe policy runtime state."""
        with self._lock:
            payload = {
                "schema": "handler.policy.export.v2",
                "timestamp": utc_timestamp(),
                "failure_counters": dict(self._failure_counters),
                "success_counters": dict(self._success_counters),
                "breaker_state": dict(self._breaker_state),
                "breaker_open_until": dict(self._breaker_open_until),
                "half_open_probes": dict(self._half_open_probes),
                "opened_counts": dict(self._opened_counts),
                "last_opened_reason": dict(self._last_opened_reason),
                "last_failure_ts": dict(self._last_failure_ts),
                "last_success_ts": dict(self._last_success_ts),
                "last_failures": {agent: list(values) for agent, values in self._last_failures.items()},
                "failure_events": {agent: list(values) for agent, values in self._failure_events.items()},
                "policy_events": list(self._policy_events),
            }
            return make_json_safe(payload)  # type: ignore[return-value]

    def import_state(self, payload: Mapping[str, Any], *, replace: bool = False) -> Dict[str, int]:
        """Import state previously exported by export_state(...)."""
        if not isinstance(payload, Mapping):
            raise ValidationError(
                "HandlerPolicy import payload must be a mapping",
                context={"actual_type": type(payload).__name__},
                code="HANDLER_POLICY_IMPORT_MAPPING_REQUIRED",
                policy=self.error_policy,
            )
        with self._lock:
            if replace:
                self.clear()
            imported_agents = 0
            for agent, value in coerce_mapping(payload.get("failure_counters")).items():
                safe_agent = self._normalize_agent_name(agent)
                self._failure_counters[safe_agent] = coerce_int(value, 0, minimum=0)
                imported_agents += 1
            for agent, value in coerce_mapping(payload.get("success_counters")).items():
                self._success_counters[self._normalize_agent_name(agent)] = coerce_int(value, 0, minimum=0)
            for agent, value in coerce_mapping(payload.get("breaker_state")).items():
                state = str(value or self.CLOSED)
                self._breaker_state[self._normalize_agent_name(agent)] = state if state in {self.CLOSED, self.OPEN, self.HALF_OPEN} else self.CLOSED
            for agent, value in coerce_mapping(payload.get("breaker_open_until")).items():
                self._breaker_open_until[self._normalize_agent_name(agent)] = coerce_float(value, 0.0, minimum=0.0)
            for agent, value in coerce_mapping(payload.get("half_open_probes")).items():
                self._half_open_probes[self._normalize_agent_name(agent)] = coerce_int(value, 0, minimum=0)
            for agent, value in coerce_mapping(payload.get("opened_counts")).items():
                self._opened_counts[self._normalize_agent_name(agent)] = coerce_int(value, 0, minimum=0)
            for agent, value in coerce_mapping(payload.get("last_opened_reason")).items():
                self._last_opened_reason[self._normalize_agent_name(agent)] = str(value)
            for agent, values in coerce_mapping(payload.get("last_failures")).items():
                self._last_failures[self._normalize_agent_name(agent)] = [coerce_float(item, 0.0, minimum=0.0) for item in coerce_list(values)]
            for agent, values in coerce_mapping(payload.get("failure_events")).items():
                self._failure_events[self._normalize_agent_name(agent)] = deque(
                    [dict(item) for item in coerce_list(values) if isinstance(item, Mapping)],
                    maxlen=self.max_policy_events,
                )
            for event in coerce_list(payload.get("policy_events")):
                if isinstance(event, Mapping):
                    self._policy_events.append(dict(event))
            return {"agents": imported_agents, "policy_events": len(coerce_list(payload.get("policy_events")))}

    def clear(self) -> None:
        """Clear all policy runtime state."""
        with self._lock:
            self._failure_counters.clear()
            self._success_counters.clear()
            self._breaker_state.clear()
            self._breaker_open_until.clear()
            self._half_open_probes.clear()
            self._opened_counts.clear()
            self._last_opened_reason.clear()
            self._last_failure_ts.clear()
            self._last_success_ts.clear()
            self._last_failures.clear()
            self._failure_events.clear()
            self._policy_events.clear()

    @staticmethod
    def _optional_int(value: Any, *, default: Optional[int]) -> Optional[int]:
        if value is None:
            return default
        return coerce_int(value, default if default is not None else 0, minimum=0)

    @staticmethod
    def _coerce_float_mapping(
        value: Any,
        *,
        default: Mapping[str, float],
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> Dict[str, float]:
        merged = deep_merge(default, coerce_mapping(value))
        return {str(key): coerce_float(item, float(default.get(str(key), 0.0)), minimum=minimum, maximum=maximum) for key, item in merged.items()}

    def _validate_configuration(self) -> None:
        if self.cooldown_seconds > self.max_cooldown_seconds:
            raise ConfigurationError(
                "HandlerPolicy cooldown_seconds cannot exceed max_cooldown_seconds",
                context={"cooldown_seconds": self.cooldown_seconds, "max_cooldown_seconds": self.max_cooldown_seconds},
                code="HANDLER_POLICY_COOLDOWN_BOUNDS_INVALID",
                policy=self.error_policy,
            )
        if self.failure_rate_threshold <= 0 and self.failure_budget_threshold <= 0:
            raise ConfigurationError(
                "HandlerPolicy must have a positive failure threshold or failure-rate threshold",
                context={"failure_budget_threshold": self.failure_budget_threshold, "failure_rate_threshold": self.failure_rate_threshold},
                code="HANDLER_POLICY_THRESHOLDS_INVALID",
                policy=self.error_policy,
            )

    def _normalize_agent_name(self, agent_name: Any) -> str:
        return normalize_identifier(agent_name, default="unknown_agent", max_chars=self.max_agent_name_chars)

    def _resolve_failure(self, normalized_failure: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if isinstance(normalized_failure, Mapping):
            return normalize_failure_payload(normalized_failure, policy=self.error_policy)
        return {
            "type": DEFAULT_FAILURE_TYPE,
            "message": DEFAULT_FAILURE_MESSAGE,
            "severity": FailureSeverity.LOW.value,
            "retryable": True,
            "category": "runtime",
            "timestamp": utc_timestamp(),
        }

    def _retry_limit(self, *, max_retries: Optional[int], normalized_failure: Optional[Mapping[str, Any]]) -> int:
        limit = self.max_retries if max_retries is None else coerce_int(max_retries, self.max_retries, minimum=0, maximum=100)
        if normalized_failure is None:
            return limit
        failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
        category = normalize_identifier(failure.get("category") or classify_failure_category(failure.get("type"), failure.get("message")), default="runtime")
        action = normalize_recovery_action(failure.get("policy_action") or failure.get("action"))
        category_override = coerce_mapping(self.category_overrides.get(category))
        action_override = coerce_mapping(self.action_overrides.get(action))
        if "max_retries" in category_override:
            limit = min(limit, coerce_int(category_override.get("max_retries"), limit, minimum=0, maximum=100))
        if "max_retries" in action_override:
            limit = min(limit, coerce_int(action_override.get("max_retries"), limit, minimum=0, maximum=100))
        return limit

    def _refresh_breaker_locked(self, agent: str) -> None:
        now = utc_timestamp()
        if self._breaker_state[agent] == self.OPEN and self._breaker_open_until[agent] <= now:
            if self.half_open_enabled:
                self._breaker_state[agent] = self.HALF_OPEN
                self._half_open_probes[agent] = 0
            else:
                self._close_breaker_locked(agent, reason="cooldown_elapsed")
        self._trim_failure_window_locked(agent, now=now)

    def _trim_failure_window_locked(self, agent: str, *, now: Optional[float] = None) -> None:
        current_time = utc_timestamp() if now is None else now
        cutoff = current_time - self.failure_budget_window_seconds
        self._last_failures[agent] = [ts for ts in self._last_failures[agent] if ts >= cutoff]
        events = self._failure_events[agent]
        while events and coerce_float(events[0].get("timestamp"), 0.0) < cutoff:
            events.popleft()

    def _snapshot_locked(self, agent: str) -> CircuitBreakerSnapshot:
        now = utc_timestamp()
        self._trim_failure_window_locked(agent, now=now)
        state = self._breaker_state[agent]
        open_until = self._breaker_open_until[agent]
        is_open = state == self.OPEN and open_until > now
        events = list(self._failure_events[agent])
        failures = [event for event in events if event.get("status") == "failed"]
        weighted_failures = round(sum(coerce_float(event.get("weight"), 0.0, minimum=0.0) for event in failures), 4)
        failure_rate = round(safe_ratio(len(failures), len(events), default=0.0, minimum=0.0, maximum=1.0), 4)
        return CircuitBreakerSnapshot(
            agent_name=agent,
            state=state,
            is_open=is_open,
            open_until=open_until,
            seconds_remaining=max(0.0, open_until - now) if is_open else 0.0,
            failure_count=self._failure_counters[agent],
            success_count=self._success_counters[agent],
            failures_in_window=len(failures),
            weighted_failures_in_window=weighted_failures,
            events_in_window=len(events),
            failure_rate=failure_rate,
            half_open_probes_used=self._half_open_probes[agent],
            half_open_probe_limit=self.half_open_max_probes,
            opened_count=self._opened_counts[agent],
            last_opened_reason=self._last_opened_reason.get(agent),
            last_failure_ts=self._last_failure_ts.get(agent),
            last_success_ts=self._last_success_ts.get(agent),
        )

    def _budget_exhausted_locked(self, agent: str) -> bool:
        if not self.circuit_breaker_enabled:
            return False
        snapshot = self._snapshot_locked(agent)
        if self.weighted_budget_enabled and snapshot.weighted_failures_in_window >= self.failure_budget_threshold:
            return True
        if snapshot.failures_in_window >= self.circuit_breaker_threshold:
            return True
        if snapshot.events_in_window >= self.min_events_for_failure_rate and snapshot.failure_rate >= self.failure_rate_threshold:
            return True
        return False

    def _failure_weight(self, *, severity: str, category: str, explicit_weight: Optional[float]) -> float:
        if explicit_weight is not None:
            return coerce_float(explicit_weight, 1.0, minimum=0.0, maximum=100.0)
        severity_weight = coerce_float(self.failure_weight_by_severity.get(severity), 1.0, minimum=0.0, maximum=100.0)
        category_weight = coerce_float(self.failure_weight_by_category.get(category), 1.0, minimum=0.0, maximum=100.0)
        return round(max(severity_weight, category_weight), 4)

    def _open_reason_for_failure_locked(
        self,
        *,
        agent: str,
        severity: str,
        category: str,
        retryable: bool,
        action: str,
        explicit_reason: Optional[str],
    ) -> Optional[str]:
        if not self.circuit_breaker_enabled:
            return None
        if explicit_reason == "force_open":
            return "force_open"
        if self.open_on_quarantine and action == HandlerRecoveryAction.QUARANTINE.value:
            return "quarantine_action"
        if self.open_on_critical_non_retryable and severity == FailureSeverity.CRITICAL.value and not retryable:
            return "critical_non_retryable_failure"
        if self._failure_counters[agent] >= self.circuit_breaker_threshold:
            return "consecutive_failure_threshold_reached"
        if self._budget_exhausted_locked(agent):
            return "rolling_failure_budget_exhausted"
        category_override = coerce_mapping(self.category_overrides.get(category))
        if coerce_bool(category_override.get("open_breaker"), default=False):
            return f"category_{category}_override"
        action_override = coerce_mapping(self.action_overrides.get(action))
        if coerce_bool(action_override.get("open_breaker"), default=False):
            return f"action_{action}_override"
        return None

    def _cooldown_for(self, *, agent: str, failure: Mapping[str, Any]) -> float:
        severity = normalize_severity(failure.get("severity"))
        category = normalize_identifier(failure.get("category") or classify_failure_category(failure.get("type"), failure.get("message")), default="runtime")
        base = self.cooldown_seconds
        severity_multiplier = coerce_float(self.severity_cooldown_multiplier.get(severity), 1.0, minimum=0.0)
        category_multiplier = coerce_float(self.category_cooldown_multiplier.get(category), 1.0, minimum=0.0)
        opened_multiplier = self.cooldown_backoff_multiplier ** max(0, self._opened_counts[agent] - 1)
        agent_override = coerce_mapping(self.agent_overrides.get(agent))
        if "cooldown_seconds" in agent_override:
            base = coerce_float(agent_override.get("cooldown_seconds"), base, minimum=0.0)
        cooldown = base * max(severity_multiplier, category_multiplier) * opened_multiplier
        return round(min(self.max_cooldown_seconds, max(0.0, cooldown)), 4)

    def _open_breaker_locked(
        self,
        agent: str,
        *,
        reason: str,
        failure: Mapping[str, Any],
        context: Mapping[str, Any],
        cooldown_seconds: Optional[float] = None,
    ) -> None:
        now = utc_timestamp()
        self._opened_counts[agent] += 1
        cooldown = coerce_float(cooldown_seconds, self._cooldown_for(agent=agent, failure=failure), minimum=0.0) if cooldown_seconds is not None else self._cooldown_for(agent=agent, failure=failure)
        self._breaker_state[agent] = self.OPEN
        self._breaker_open_until[agent] = now + cooldown
        self._half_open_probes[agent] = 0
        self._last_opened_reason[agent] = truncate_text(reason, 240)
        self._emit_policy_event_locked(
            event_type="handler_policy_breaker_opened",
            agent=agent,
            decision=None,
            failure=failure,
            context=context,
            extra={"reason": reason, "cooldown_seconds": cooldown, "open_until": self._breaker_open_until[agent]},
        )

    def _close_breaker_locked(self, agent: str, *, reason: str) -> None:
        self._breaker_state[agent] = self.CLOSED
        self._breaker_open_until[agent] = 0.0
        self._half_open_probes[agent] = 0
        self._failure_counters[agent] = 0
        self._success_counters[agent] = 0
        self._last_opened_reason.pop(agent, None)
        self._emit_policy_event_locked(
            event_type="handler_policy_breaker_closed",
            agent=agent,
            decision=None,
            failure=None,
            context={},
            extra={"reason": reason},
        )

    def _build_decision(
        self,
        *,
        agent: str,
        allowed: bool,
        reason: str,
        snapshot: CircuitBreakerSnapshot,
        retry_allowed: bool,
        retry_limit: int,
        attempted: int,
        severity: str,
        category: str,
        retryable: bool,
        action: str,
        correlation_id: Optional[str],
        budget_exhausted: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            agent_name=agent,
            allowed=allowed,
            reason=reason,
            breaker_state=snapshot.state,
            breaker_open=snapshot.is_open,
            retry_allowed=retry_allowed,
            retry_limit=retry_limit,
            attempted_retries=attempted,
            failures_in_window=snapshot.failures_in_window,
            weighted_failures_in_window=snapshot.weighted_failures_in_window,
            events_in_window=snapshot.events_in_window,
            failure_rate=snapshot.failure_rate,
            cooldown_seconds_remaining=snapshot.seconds_remaining,
            open_until=snapshot.open_until,
            half_open_probe_available=snapshot.state == self.HALF_OPEN and snapshot.half_open_probes_used < snapshot.half_open_probe_limit,
            severity=severity,
            category=category,
            retryable=retryable,
            action=action,
            evaluator_hooks_enabled=self.evaluator_hooks_enabled,
            budget_exhausted=budget_exhausted,
            correlation_id=correlation_id,
            metadata=dict(metadata or {}),
        )

    def _emit_policy_event_locked(
        self,
        *,
        event_type: str,
        agent: str,
        decision: Optional[PolicyDecision],
        failure: Optional[Mapping[str, Any]],
        context: Mapping[str, Any],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        event = compact_dict(
            {
                "schema": "handler.policy_event.v2",
                "event_type": normalize_identifier(event_type, default="handler_policy"),
                "timestamp": utc_timestamp(),
                "agent_name": agent,
                "decision": decision.to_dict() if decision is not None else None,
                "failure": normalize_failure_payload(failure, policy=self.error_policy) if failure is not None and self.include_failure_in_events else None,
                "context": build_escalation_context(context) if self.include_context_in_events else None,
                "extra": make_json_safe(self.error_policy.sanitize_context(extra or {})),
            },
            drop_none=True,
            drop_empty=True,
        )
        self._policy_events.append(event)
        if self.emit_to_memory and self.memory is not None:
            if hasattr(self.memory, "append_telemetry") and callable(self.memory.append_telemetry):
                self.memory.append_telemetry(event)
            else:
                append_shared_memory_list(
                    self.memory,
                    self.POLICY_EVENT_KEY,
                    event,
                    max_items=self.max_policy_events,
                    ttl=self.policy_event_ttl_seconds,
                )


if __name__ == "__main__":
    print("\n=== Running Handler Policy ===\n")
    printer.status("TEST", "Handler Policy initialized", "info")

    from .handler_memory import HandlerMemory

    strict_policy = HandlerErrorPolicy(
        name="handler_policy.strict_test",
        expose_internal_messages=False,
        include_context_in_public=False,
        include_context_in_telemetry=True,
        max_message_chars=240,
        max_string_chars=160,
    )

    memory = HandlerMemory(
        config={
            "max_checkpoints": 3,
            "max_telemetry_events": 10,
            "max_postmortems": 3,
            "copy_on_read": True,
            "sanitize_payloads": True,
        },
        error_policy=strict_policy,
    )
    policy = HandlerPolicy(
        config={
            "max_retries": 2,
            "circuit_breaker_threshold": 2,
            "cooldown_seconds": 0.25,
            "max_cooldown_seconds": 1.0,
            "failure_budget_window_seconds": 5,
            "failure_budget_threshold": 3.0,
            "failure_rate_threshold": 0.75,
            "min_events_for_failure_rate": 2,
            "half_open_enabled": True,
            "half_open_max_probes": 1,
            "half_open_success_threshold": 1,
            "emit_to_memory": True,
        },
        memory=memory,
        error_policy=strict_policy,
    )

    agent_name = "demo_agent"
    context = {
        "task_id": "handler-policy-smoke-001",
        "route": "handler.recovery",
        "agent": agent_name,
        "correlation_id": "corr-handler-policy-test",
        "password": "SuperSecret123",
    }
    failure = build_normalized_failure(
        error=TimeoutError("Upstream timed out with Authorization: Bearer token-123"),
        context=context,
        policy=strict_policy,
        source="handler.policy.__main__",
        correlation_id="corr-handler-policy-test",
    )

    initial_decision = policy.evaluate_attempt(agent_name, normalized_failure=failure, context=context, attempted_retries=0, max_retries=2)
    assert initial_decision.allowed is True
    assert policy.can_attempt(agent_name, normalized_failure=failure, context=context) is True
    assert policy.retries_allowed(0, max_retries=2, normalized_failure=failure) is True
    assert policy.retries_allowed(2, max_retries=2, normalized_failure=failure) is False

    policy.record_failure(agent_name, normalized_failure=failure, context=context)
    status_after_first = policy.breaker_status(agent_name)
    assert status_after_first["is_open"] is False

    policy.record_failure(agent_name, normalized_failure=failure, context=context)
    status_after_second = policy.breaker_status(agent_name)
    assert status_after_second["is_open"] is True
    assert policy.can_attempt(agent_name, normalized_failure=failure, context=context) is False

    import time as _time

    _time.sleep(0.30)
    half_open_decision = policy.evaluate_attempt(agent_name, normalized_failure=failure, context=context, attempted_retries=0, max_retries=2)
    assert half_open_decision.allowed is True
    assert half_open_decision.breaker_state == HandlerPolicy.HALF_OPEN

    policy.record_success(agent_name, context=context, reset_failure_window=True)
    status_after_success = policy.breaker_status(agent_name)
    assert status_after_success["is_open"] is False
    assert status_after_success["state"] == HandlerPolicy.CLOSED

    security_failure = build_normalized_failure(
        error=PermissionError("Security policy violation with password=SuperSecret123"),
        error_info={"severity": "critical", "retryable": False, "policy_action": "quarantine"},
        context=context,
        policy=strict_policy,
        source="handler.policy.__main__",
    )
    policy.record_failure("safety_sensitive_agent", normalized_failure=security_failure, context=context)
    security_status = policy.breaker_status("safety_sensitive_agent")
    assert security_status["is_open"] is True

    snapshot = policy.policy_snapshot(include_events=True)
    exported = policy.export_state()
    imported_policy = HandlerPolicy(config={"cooldown_seconds": 0.25}, error_policy=strict_policy)
    imported = imported_policy.import_state(exported, replace=True)
    health = policy.health()
    recent_events_payload = policy.recent_policy_events(limit=10)
    memory_events = memory.recent_telemetry(limit=10)

    serialized = stable_json_dumps(
        {
            "initial_decision": initial_decision.to_dict(),
            "status_after_first": status_after_first,
            "status_after_second": status_after_second,
            "half_open_decision": half_open_decision.to_dict(),
            "status_after_success": status_after_success,
            "security_status": security_status,
            "snapshot": snapshot,
            "exported": exported,
            "imported": imported,
            "health": health,
            "recent_events": recent_events_payload,
            "memory_events": memory_events,
        }
    )

    assert health["status"] == "ok"
    assert imported["agents"] >= 1
    assert len(recent_events_payload) >= 1
    assert len(memory_events) >= 1
    assert "SuperSecret123" not in serialized
    assert "token-123" not in serialized

    printer.pretty("Initial decision", initial_decision.to_dict(), "success")
    printer.pretty("Breaker after threshold", status_after_second, "success")
    printer.pretty("Policy health", health, "success")
    print("\n=== Test ran successfully ===\n")
