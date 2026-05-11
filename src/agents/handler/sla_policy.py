from __future__ import annotations

import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.handler_error import *
from .utils.handler_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("SLA Recovery Policy")
printer = PrettyPrinter()


class SLABreachStatus(str, Enum):
    """Canonical SLA status emitted by SLARecoveryPolicy."""

    OK = "ok"
    WATCH = "watch"
    NEAR_BREACH = "near_breach"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class SLARecoveryMode(str, Enum):
    """Recovery mode hints consumed by HandlerAgent and recovery strategies."""

    STANDARD = "standard"
    CONSERVATIVE = "conservative"
    FAST_FAILOVER = "fast_failover"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"
    QUARANTINE = "quarantine"
    DISABLED = "disabled"


@dataclass(frozen=True)
class SLABudget:
    """Resolved recovery budget from context.sla and module defaults."""

    remaining_seconds: float
    budget_source: str
    deadline_ts: Optional[float] = None
    total_budget_seconds: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    raw_sla: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=utc_timestamp)

    @property
    def exhausted(self) -> bool:
        return self.remaining_seconds <= 0.0

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "remaining_seconds": round(self.remaining_seconds, 6),
                "budget_source": self.budget_source,
                "deadline_ts": self.deadline_ts,
                "total_budget_seconds": self.total_budget_seconds,
                "elapsed_seconds": self.elapsed_seconds,
                "sla_keys": sorted(str(key) for key in self.raw_sla.keys())[:25],
                "timestamp": self.timestamp,
            },
            drop_none=True,
            drop_empty=True,
        )


@dataclass(frozen=True)
class SLAEvaluation:
    """
    Production SLA decision emitted by SLARecoveryPolicy.

    The compatibility fields intentionally preserve the legacy shape used by HandlerAgent:
    remaining_seconds, recommended_attempts, mode, can_retry, priority.
    """

    remaining_seconds: float
    recommended_attempts: int
    mode: str
    can_retry: bool
    priority: str
    severity: str
    retryable: bool
    category: str
    action: str
    breach_status: str
    budget_source: str
    deadline_ts: Optional[float] = None
    total_budget_seconds: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    pressure: float = 0.0
    estimated_attempt_seconds: float = 0.0
    recommended_timeout_seconds: float = 0.0
    max_delay_seconds: float = 0.0
    degrade_allowed: bool = True
    fail_fast: bool = False
    reason: str = "sla_evaluated"
    recommendation: str = "continue_with_budget_guardrails"
    correlation_id: Optional[str] = None
    context_hash: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=utc_timestamp)
    schema: str = "handler.sla_recovery.v2"

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "schema": self.schema,
                "remaining_seconds": round(self.remaining_seconds, 6),
                "recommended_attempts": self.recommended_attempts,
                "mode": self.mode,
                "can_retry": self.can_retry,
                "priority": self.priority,
                "severity": self.severity,
                "retryable": self.retryable,
                "category": self.category,
                "action": self.action,
                "breach_status": self.breach_status,
                "budget_source": self.budget_source,
                "deadline_ts": self.deadline_ts,
                "total_budget_seconds": self.total_budget_seconds,
                "elapsed_seconds": self.elapsed_seconds,
                "pressure": self.pressure,
                "estimated_attempt_seconds": self.estimated_attempt_seconds,
                "recommended_timeout_seconds": self.recommended_timeout_seconds,
                "max_delay_seconds": self.max_delay_seconds,
                "degrade_allowed": self.degrade_allowed,
                "fail_fast": self.fail_fast,
                "reason": self.reason,
                "recommendation": self.recommendation,
                "correlation_id": self.correlation_id,
                "context_hash": self.context_hash,
                "metadata": dict(self.metadata),
                "timestamp": self.timestamp,
            },
            drop_none=True,
            drop_empty=True,
        )


class SLARecoveryPolicy:
    """
    Production SLA recovery policy for HandlerAgent recovery orchestration.

    Scope:
    - preserves evaluate(context, normalized_failure) -> dict for existing HandlerAgent use
    - reads SLA settings from handler_config.yaml via the dedicated sla_policy section
    - falls back to legacy policy.default_recovery_budget_seconds/default_sla_max_attempts
    - resolves deadline/budget forms from context.sla without owning retry execution
    - constrains recovery attempts and timeboxing using severity/category/action/SLA pressure
    - emits structured SLA telemetry to HandlerMemory-like objects when configured

    This class does not perform retries, open circuit breakers, select strategies, or route
    escalations. AdaptiveRetryPolicy, HandlerPolicy, StrategySelector, and EscalationManager
    keep those responsibilities.
    """

    DEFAULT_PRIORITY_BY_SEVERITY: Mapping[str, str] = {
        FailureSeverity.CRITICAL.value: "p0",
        FailureSeverity.HIGH.value: "p1",
        FailureSeverity.MEDIUM.value: "p2",
        FailureSeverity.LOW.value: "p3",
    }
    DEFAULT_MODE_BY_ACTION: Mapping[str, str] = {
        HandlerRecoveryAction.QUARANTINE.value: SLARecoveryMode.QUARANTINE.value,
        HandlerRecoveryAction.FAIL_FAST.value: SLARecoveryMode.FAIL_FAST.value,
        HandlerRecoveryAction.ESCALATE.value: SLARecoveryMode.CONSERVATIVE.value,
        HandlerRecoveryAction.DEGRADE.value: SLARecoveryMode.DEGRADE.value,
        HandlerRecoveryAction.RETRY.value: SLARecoveryMode.STANDARD.value,
        HandlerRecoveryAction.NONE.value: SLARecoveryMode.STANDARD.value,
    }
    DEFAULT_CATEGORY_ATTEMPT_MODIFIERS: Mapping[str, int] = {
        "timeout": 0,
        "network": 0,
        "unicode": 0,
        "resource": -1,
        "memory": -1,
        "dependency": -1,
        "validation": -1,
        "security": -10,
        "sla": -1,
        "runtime": 0,
    }
    DEFAULT_SEVERITY_ATTEMPT_MODIFIERS: Mapping[str, int] = {
        FailureSeverity.LOW.value: 1,
        FailureSeverity.MEDIUM.value: 0,
        FailureSeverity.HIGH.value: -1,
        FailureSeverity.CRITICAL.value: -1,
    }
    DEFAULT_CATEGORY_ATTEMPT_SECONDS: Mapping[str, float] = {
        "timeout": 1.5,
        "network": 1.5,
        "unicode": 0.75,
        "resource": 2.0,
        "memory": 2.5,
        "dependency": 3.0,
        "validation": 0.75,
        "security": 0.0,
        "sla": 0.5,
        "runtime": 2.0,
    }

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        memory: Any = None,
        error_policy: Optional[HandlerErrorPolicy] = None,
    ):
        self.config = load_global_config()
        policy_cfg = get_config_section("policy")
        sla_cfg = get_config_section("sla_policy")

        merged = deep_merge(policy_cfg, sla_cfg)
        if isinstance(config, Mapping):
            merged = deep_merge(merged, config)

        policy_config = merged.get("error_policy") if isinstance(merged.get("error_policy"), Mapping) else None
        self.error_policy = error_policy or HandlerErrorPolicy.from_mapping(policy_config)
        self.memory = memory

        self.enabled = coerce_bool(merged.get("enabled"), default=True)
        self.default_recovery_budget_seconds = coerce_float(
            merged.get("default_recovery_budget_seconds"),
            30.0,
            minimum=0.0,
        )
        self.default_max_attempts = coerce_int(
            merged.get("default_max_attempts", merged.get("default_sla_max_attempts")),
            2,
            minimum=0,
            maximum=100,
        )
        self.min_attempts = coerce_int(merged.get("min_attempts"), 0, minimum=0, maximum=100)
        self.max_attempts = coerce_int(
            merged.get("max_attempts"),
            max(self.default_max_attempts, 4),
            minimum=self.min_attempts,
            maximum=100,
        )
        self.allow_retry_without_sla = coerce_bool(merged.get("allow_retry_without_sla"), default=True)
        self.retry_non_retryable = coerce_bool(merged.get("retry_non_retryable"), default=False)
        self.allow_critical_retry = coerce_bool(merged.get("allow_critical_retry"), default=False)
        self.default_priority = coerce_str(merged.get("default_priority"), default="normal")

        self.min_remaining_seconds_for_retry = coerce_float(
            merged.get("min_remaining_seconds_for_retry"),
            0.5,
            minimum=0.0,
        )
        self.fast_failover_threshold_seconds = coerce_float(
            merged.get("fast_failover_threshold_seconds"),
            3.0,
            minimum=0.0,
        )
        self.conservative_threshold_seconds = coerce_float(
            merged.get("conservative_threshold_seconds"),
            8.0,
            minimum=0.0,
        )
        self.low_budget_threshold_seconds = coerce_float(
            merged.get("low_budget_threshold_seconds"),
            3.0,
            minimum=0.0,
        )
        self.sla_safety_margin_seconds = coerce_float(
            merged.get("sla_safety_margin_seconds"),
            0.25,
            minimum=0.0,
        )
        self.default_estimated_attempt_seconds = coerce_float(
            merged.get("default_estimated_attempt_seconds"),
            1.5,
            minimum=0.0,
        )
        self.timeout_safety_fraction = coerce_float(
            merged.get("timeout_safety_fraction"),
            0.80,
            minimum=0.05,
            maximum=1.0,
        )
        self.max_timeout_seconds = coerce_float(merged.get("max_timeout_seconds"), 30.0, minimum=0.0)
        self.max_delay_seconds = coerce_float(merged.get("max_delay_seconds"), 8.0, minimum=0.0)
        self.high_pressure_threshold = coerce_float(merged.get("high_pressure_threshold"), 0.75, minimum=0.0, maximum=1.0)
        self.breach_grace_seconds = coerce_float(merged.get("breach_grace_seconds"), 0.0, minimum=0.0)
        self.emit_to_memory = coerce_bool(merged.get("emit_to_memory"), default=False)
        self.memory_event_type = normalize_identifier(merged.get("memory_event_type"), default="handler_sla_evaluation")

        self.priority_by_severity = deep_merge(self.DEFAULT_PRIORITY_BY_SEVERITY, coerce_mapping(merged.get("priority_by_severity")))
        self.priority_floor_by_pressure = coerce_mapping(merged.get("priority_floor_by_pressure"))
        self.mode_by_action = deep_merge(self.DEFAULT_MODE_BY_ACTION, coerce_mapping(merged.get("mode_by_action")))
        self.category_attempt_modifiers = self._coerce_modifier_mapping(
            merged.get("category_attempt_modifiers"),
            default=self.DEFAULT_CATEGORY_ATTEMPT_MODIFIERS,
        )
        self.severity_attempt_modifiers = self._coerce_modifier_mapping(
            merged.get("severity_attempt_modifiers"),
            default=self.DEFAULT_SEVERITY_ATTEMPT_MODIFIERS,
        )
        self.category_attempt_seconds = self._coerce_float_mapping(
            merged.get("category_attempt_seconds"),
            default=self.DEFAULT_CATEGORY_ATTEMPT_SECONDS,
        )
        self.degrade_allowed_categories = tuple(
            normalize_identifier(item, default="runtime")
            for item in coerce_list(
                merged.get("degrade_allowed_categories"),
                default=("timeout", "network", "memory", "resource", "sla", "runtime", "unicode"),
                split_strings=True,
            )
        )
        self.fail_fast_actions = tuple(
            normalize_recovery_action(item)
            for item in coerce_list(
                merged.get("fail_fast_actions"),
                default=(HandlerRecoveryAction.FAIL_FAST.value, HandlerRecoveryAction.QUARANTINE.value),
                split_strings=True,
            )
        )

        self._validate_configuration()
        logger.info(
            "SLA Recovery Policy initialized | enabled=%s default_budget=%s default_attempts=%s",
            self.enabled,
            self.default_recovery_budget_seconds,
            self.default_max_attempts,
        )

    def attach_memory(self, memory: Any) -> None:
        """Attach a HandlerMemory-like object for SLA evaluation telemetry."""
        self.memory = memory

    def evaluate(self, context: Optional[Mapping[str, Any]], normalized_failure: Mapping[str, Any]) -> Dict[str, Any]:
        """Evaluate SLA budget and return the legacy-compatible recovery constraint payload."""
        return self.decide(context=context, normalized_failure=normalized_failure).to_dict()

    def decide(
        self,
        *,
        context: Optional[Mapping[str, Any]],
        normalized_failure: Mapping[str, Any],
        emit: Optional[bool] = None,
    ) -> SLAEvaluation:
        """Build a rich SLA evaluation for recovery, retry, and escalation consumers."""
        try:
            if not isinstance(normalized_failure, Mapping):
                raise ValidationError(
                    "SLARecoveryPolicy expected normalized_failure to be a mapping",
                    context={"actual_type": type(normalized_failure).__name__},
                    code="HANDLER_SLA_FAILURE_MAPPING_REQUIRED",
                    policy=self.error_policy,
                )

            context_map = coerce_mapping(context)
            failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
            budget = self.resolve_budget(context_map)
            severity = normalize_severity(failure.get("severity"))
            retryable = coerce_bool(failure.get("retryable"), default=False)
            category = normalize_identifier(failure.get("category") or classify_failure_category(failure.get("type"), failure.get("message")), default="runtime")
            action = normalize_recovery_action(failure.get("policy_action") or failure.get("action"))
            priority = self._resolve_priority(context=context_map, severity=severity)
            estimated_attempt_seconds = self._estimated_attempt_seconds(category=category, severity=severity)
            pressure = self._pressure(budget=budget)
            breach_status = self._breach_status(budget=budget, pressure=pressure)

            if not self.enabled:
                evaluation = self._evaluation(
                    failure=failure,
                    context=context_map,
                    budget=budget,
                    severity=severity,
                    retryable=retryable,
                    category=category,
                    action=action,
                    priority=priority,
                    recommended_attempts=self.default_max_attempts if retryable else 0,
                    mode=SLARecoveryMode.DISABLED.value,
                    breach_status=SLABreachStatus.UNKNOWN.value,
                    pressure=pressure,
                    estimated_attempt_seconds=estimated_attempt_seconds,
                    reason="sla_policy_disabled",
                    recommendation="use_default_recovery_constraints",
                )
                self._maybe_emit(evaluation=evaluation, failure=failure, context=context_map, emit=emit)
                return evaluation

            mode, reason = self._mode_for(
                budget=budget,
                severity=severity,
                retryable=retryable,
                category=category,
                action=action,
                pressure=pressure,
            )
            attempts, attempts_reason = self._recommended_attempts(
                budget=budget,
                severity=severity,
                retryable=retryable,
                category=category,
                action=action,
                mode=mode,
                estimated_attempt_seconds=estimated_attempt_seconds,
            )
            if attempts_reason != "base_attempt_budget":
                reason = attempts_reason

            priority = self._raise_priority_for_pressure(priority=priority, pressure=pressure, breach_status=breach_status)
            can_retry = self._can_retry(
                attempts=attempts,
                retryable=retryable,
                mode=mode,
                action=action,
                budget=budget,
            )
            recommendation = self._recommendation(
                mode=mode,
                breach_status=breach_status,
                can_retry=can_retry,
                category=category,
                severity=severity,
            )
            recommended_timeout = self._recommended_timeout(
                attempts=attempts,
                budget=budget,
                estimated_attempt_seconds=estimated_attempt_seconds,
            )

            evaluation = self._evaluation(
                failure=failure,
                context=context_map,
                budget=budget,
                severity=severity,
                retryable=retryable,
                category=category,
                action=action,
                priority=priority,
                recommended_attempts=attempts,
                mode=mode,
                breach_status=breach_status,
                pressure=pressure,
                estimated_attempt_seconds=estimated_attempt_seconds,
                recommended_timeout_seconds=recommended_timeout,
                reason=reason,
                recommendation=recommendation,
            )
            self._maybe_emit(evaluation=evaluation, failure=failure, context=context_map, emit=emit)
            return evaluation
        except HandlerError:
            raise
        except Exception as exc:
            raise SLAError(
                "SLA recovery policy failed to evaluate recovery budget",
                cause=exc,
                context={
                    "context_type": type(context).__name__,
                    "failure_type": type(normalized_failure).__name__,
                },
                code="HANDLER_SLA_EVALUATION_FAILED",
                policy=self.error_policy,
            ) from exc

    def resolve_budget(self, context: Optional[Mapping[str, Any]] = None, *, now: Optional[float] = None) -> SLABudget:
        """Resolve the active SLA budget from context.sla and defaults."""
        context_map = coerce_mapping(context)
        sla = coerce_mapping(context_map.get("sla"))
        current_time = utc_timestamp() if now is None else coerce_float(now, utc_timestamp())

        deadline_ts = self._numeric_or_none(
            sla.get("deadline_ts", sla.get("deadline_epoch", sla.get("deadline")))
        )
        if deadline_ts is not None:
            return SLABudget(
                remaining_seconds=max(0.0, deadline_ts - current_time),
                budget_source="deadline_ts",
                deadline_ts=deadline_ts,
                raw_sla=sla,
                timestamp=current_time,
            )

        deadline_ms = self._numeric_or_none(sla.get("deadline_ms"))
        if deadline_ms is not None:
            deadline_ts = deadline_ms / 1000.0
            return SLABudget(
                remaining_seconds=max(0.0, deadline_ts - current_time),
                budget_source="deadline_ms",
                deadline_ts=deadline_ts,
                raw_sla=sla,
                timestamp=current_time,
            )

        explicit_remaining = self._numeric_or_none(
            sla.get("remaining_seconds", sla.get("remaining_recovery_seconds"))
        )
        if explicit_remaining is not None:
            return SLABudget(
                remaining_seconds=max(0.0, explicit_remaining),
                budget_source="remaining_seconds",
                raw_sla=sla,
                timestamp=current_time,
            )

        total_budget = self._numeric_or_none(
            sla.get("max_recovery_seconds", sla.get("recovery_budget_seconds", sla.get("budget_seconds")))
        )
        started_ts = self._numeric_or_none(
            sla.get("recovery_started_ts", sla.get("started_ts", sla.get("start_ts")))
        )
        if total_budget is not None:
            elapsed = max(0.0, current_time - started_ts) if started_ts is not None else None
            remaining = max(0.0, total_budget - elapsed) if elapsed is not None else max(0.0, total_budget)
            return SLABudget(
                remaining_seconds=remaining,
                budget_source="max_recovery_seconds",
                total_budget_seconds=total_budget,
                elapsed_seconds=elapsed,
                raw_sla=sla,
                timestamp=current_time,
            )

        latency_budget_ms = self._numeric_or_none(sla.get("latency_budget_ms"))
        if latency_budget_ms is not None:
            total_seconds = max(0.0, latency_budget_ms / 1000.0)
            return SLABudget(
                remaining_seconds=total_seconds,
                budget_source="latency_budget_ms",
                total_budget_seconds=total_seconds,
                raw_sla=sla,
                timestamp=current_time,
            )

        default_remaining = self.default_recovery_budget_seconds if self.allow_retry_without_sla else 0.0
        return SLABudget(
            remaining_seconds=max(0.0, default_remaining),
            budget_source="default_recovery_budget_seconds",
            total_budget_seconds=max(0.0, default_remaining),
            raw_sla=sla,
            timestamp=current_time,
        )

    def should_continue(
        self,
        *,
        attempted_attempts: int,
        context: Optional[Mapping[str, Any]],
        normalized_failure: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return a compact gate payload for recovery loops."""
        evaluation = self.decide(context=context, normalized_failure=normalized_failure)
        attempted = coerce_int(attempted_attempts, 0, minimum=0)
        allowed = evaluation.can_retry and attempted < evaluation.recommended_attempts
        return {
            "allowed": allowed,
            "attempted_attempts": attempted,
            "remaining_attempts": max(0, evaluation.recommended_attempts - attempted),
            "recommended_timeout_seconds": evaluation.recommended_timeout_seconds if allowed else 0.0,
            "max_delay_seconds": evaluation.max_delay_seconds if allowed else 0.0,
            "evaluation": evaluation.to_dict(),
        }

    def timebox_for_attempt(
        self,
        attempt_index: int,
        *,
        context: Optional[Mapping[str, Any]],
        normalized_failure: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return timeout and delay caps for one recovery attempt."""
        evaluation = self.decide(context=context, normalized_failure=normalized_failure)
        attempt = coerce_int(attempt_index, 0, minimum=0)
        if not evaluation.can_retry or attempt >= evaluation.recommended_attempts:
            return {
                "allowed": False,
                "attempt_index": attempt,
                "timeout_seconds": 0.0,
                "max_delay_seconds": 0.0,
                "remaining_seconds": evaluation.remaining_seconds,
            }
        return {
            "allowed": True,
            "attempt_index": attempt,
            "timeout_seconds": evaluation.recommended_timeout_seconds,
            "max_delay_seconds": evaluation.max_delay_seconds,
            "remaining_seconds": evaluation.remaining_seconds,
            "mode": evaluation.mode,
            "priority": evaluation.priority,
        }

    def deadline_from_budget(
        self,
        budget_seconds: Any,
        *,
        now: Optional[float] = None,
    ) -> float:
        """Return an epoch deadline from a relative recovery budget."""
        current_time = utc_timestamp() if now is None else coerce_float(now, utc_timestamp())
        return current_time + max(0.0, coerce_float(budget_seconds, self.default_recovery_budget_seconds, minimum=0.0))

    def evaluate_many(
        self,
        failures: Iterable[Mapping[str, Any]],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple normalized failures against one context snapshot."""
        return [self.evaluate(context=context, normalized_failure=failure) for failure in failures if isinstance(failure, Mapping)]

    def summarize(self, evaluations: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
        """Summarize SLA policy config and optional evaluation payloads."""
        events = [coerce_mapping(item) for item in coerce_list(evaluations) if isinstance(item, Mapping)]
        mode_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        for event in events:
            mode = str(event.get("mode", "unknown"))
            status = str(event.get("breach_status", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
        return {
            "schema": "handler.sla_recovery.summary.v2",
            "timestamp": utc_timestamp(),
            "enabled": self.enabled,
            "default_recovery_budget_seconds": self.default_recovery_budget_seconds,
            "default_max_attempts": self.default_max_attempts,
            "min_attempts": self.min_attempts,
            "max_attempts": self.max_attempts,
            "fast_failover_threshold_seconds": self.fast_failover_threshold_seconds,
            "conservative_threshold_seconds": self.conservative_threshold_seconds,
            "evaluation_count": len(events),
            "mode_counts": mode_counts,
            "breach_status_counts": status_counts,
        }

    def emit_sla_event(
        self,
        *,
        evaluation: SLAEvaluation,
        failure: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Emit an SLA evaluation telemetry event into HandlerMemory-like storage."""
        if self.memory is None or not hasattr(self.memory, "append_telemetry"):
            return None
        event = {
            "schema": "handler.sla_recovery.event.v2",
            "event_type": self.memory_event_type,
            "timestamp": utc_timestamp(),
            "correlation_id": evaluation.correlation_id,
            "failure": normalize_failure_payload(failure, policy=self.error_policy),
            "sla": evaluation.to_dict(),
            "context": select_keys(coerce_mapping(context), ("task_id", "route", "agent", "priority", "correlation_id")),
        }
        self.memory.append_telemetry(event)
        return event

    def _evaluation(
        self,
        *,
        failure: Mapping[str, Any],
        context: Mapping[str, Any],
        budget: SLABudget,
        severity: str,
        retryable: bool,
        category: str,
        action: str,
        priority: str,
        recommended_attempts: int,
        mode: str,
        breach_status: str,
        pressure: float,
        estimated_attempt_seconds: float,
        reason: str,
        recommendation: str,
        recommended_timeout_seconds: Optional[float] = None,
    ) -> SLAEvaluation:
        attempts = coerce_int(recommended_attempts, 0, minimum=0, maximum=self.max_attempts)
        timeout_seconds = recommended_timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = self._recommended_timeout(
                attempts=attempts,
                budget=budget,
                estimated_attempt_seconds=estimated_attempt_seconds,
            )
        safe_mode = self._normalize_mode(mode)
        fail_fast = safe_mode in {SLARecoveryMode.FAIL_FAST.value, SLARecoveryMode.QUARANTINE.value}
        max_delay = self._max_delay_for_budget(budget=budget, attempts=attempts)
        correlation_id = failure.get("correlation_id") or context.get("correlation_id")
        return SLAEvaluation(
            remaining_seconds=max(0.0, budget.remaining_seconds),
            recommended_attempts=attempts,
            mode=safe_mode,
            can_retry=self._can_retry(
                attempts=attempts,
                retryable=retryable,
                mode=safe_mode,
                action=action,
                budget=budget,
            ),
            priority=str(priority),
            severity=severity,
            retryable=retryable,
            category=category,
            action=action,
            breach_status=breach_status,
            budget_source=budget.budget_source,
            deadline_ts=budget.deadline_ts,
            total_budget_seconds=budget.total_budget_seconds,
            elapsed_seconds=budget.elapsed_seconds,
            pressure=pressure,
            estimated_attempt_seconds=round(estimated_attempt_seconds, 6),
            recommended_timeout_seconds=round(timeout_seconds, 6),
            max_delay_seconds=round(max_delay, 6),
            degrade_allowed=category in self.degrade_allowed_categories,
            fail_fast=fail_fast,
            reason=reason,
            recommendation=recommendation,
            correlation_id=str(correlation_id) if correlation_id else None,
            context_hash=str(failure.get("context_hash")) if failure.get("context_hash") else None,
            metadata=compact_dict(
                {
                    "failure_type": failure.get("type"),
                    "task_id": context.get("task_id"),
                    "route": context.get("route"),
                    "agent": context.get("agent"),
                    "budget": budget.to_dict(),
                },
                drop_none=True,
                drop_empty=True,
            ),
        )

    def _mode_for(
        self,
        *,
        budget: SLABudget,
        severity: str,
        retryable: bool,
        category: str,
        action: str,
        pressure: float,
    ) -> Tuple[str, str]:
        if action == HandlerRecoveryAction.QUARANTINE.value or category == "security":
            return SLARecoveryMode.QUARANTINE.value, "security_or_quarantine_action"
        if action == HandlerRecoveryAction.FAIL_FAST.value:
            return SLARecoveryMode.FAIL_FAST.value, "policy_action_fail_fast"
        if not retryable and not self.retry_non_retryable:
            if severity in {FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value}:
                return SLARecoveryMode.FAIL_FAST.value, "non_retryable_high_severity"
            return SLARecoveryMode.DEGRADE.value, "failure_marked_non_retryable"
        if budget.remaining_seconds <= self.breach_grace_seconds:
            return SLARecoveryMode.DEGRADE.value, "sla_budget_exhausted"
        if severity == FailureSeverity.CRITICAL.value and not self.allow_critical_retry:
            return SLARecoveryMode.CONSERVATIVE.value, "critical_failure_conservative_recovery"
        if budget.remaining_seconds <= self.fast_failover_threshold_seconds:
            return SLARecoveryMode.FAST_FAILOVER.value, "low_sla_budget_fast_failover"
        if pressure >= self.high_pressure_threshold or budget.remaining_seconds <= self.conservative_threshold_seconds:
            return SLARecoveryMode.CONSERVATIVE.value, "high_sla_pressure_conservative_recovery"
        mode = self._normalize_mode(self.mode_by_action.get(action, SLARecoveryMode.STANDARD.value))
        return mode, "action_based_sla_mode"

    def _recommended_attempts(
        self,
        *,
        budget: SLABudget,
        severity: str,
        retryable: bool,
        category: str,
        action: str,
        mode: str,
        estimated_attempt_seconds: float,
    ) -> Tuple[int, str]:
        if action in self.fail_fast_actions or mode in {SLARecoveryMode.FAIL_FAST.value, SLARecoveryMode.QUARANTINE.value}:
            return 0, "fail_fast_or_quarantine_blocks_retry"
        if not retryable and not self.retry_non_retryable:
            return 0, "failure_marked_non_retryable"
        if budget.remaining_seconds <= self.breach_grace_seconds:
            return 0, "sla_budget_exhausted"

        attempts = self.default_max_attempts
        attempts += self.severity_attempt_modifiers.get(severity, 0)
        attempts += self.category_attempt_modifiers.get(category, 0)

        if mode == SLARecoveryMode.FAST_FAILOVER.value:
            attempts = min(attempts, 1)
        elif mode == SLARecoveryMode.CONSERVATIVE.value:
            attempts = min(attempts, max(1, self.default_max_attempts - 1))
        elif mode == SLARecoveryMode.DEGRADE.value:
            attempts = min(attempts, 1)

        usable_budget = max(0.0, budget.remaining_seconds - self.sla_safety_margin_seconds)
        if estimated_attempt_seconds > 0:
            budget_cap = math.floor(usable_budget / max(estimated_attempt_seconds, 0.001))
            attempts = min(attempts, budget_cap)

        if budget.remaining_seconds < self.min_remaining_seconds_for_retry:
            return 0, "remaining_budget_below_retry_floor"

        return coerce_int(attempts, 0, minimum=self.min_attempts, maximum=self.max_attempts), "base_attempt_budget"

    def _can_retry(
        self,
        *,
        attempts: int,
        retryable: bool,
        mode: str,
        action: str,
        budget: SLABudget,
    ) -> bool:
        if attempts <= 0:
            return False
        if mode in {SLARecoveryMode.FAIL_FAST.value, SLARecoveryMode.QUARANTINE.value}:
            return False
        if action in self.fail_fast_actions:
            return False
        if budget.remaining_seconds < self.min_remaining_seconds_for_retry:
            return False
        return bool(retryable or self.retry_non_retryable)

    def _recommendation(self, *, mode: str, breach_status: str, can_retry: bool, category: str, severity: str) -> str:
        if mode == SLARecoveryMode.QUARANTINE.value:
            return "quarantine_and_escalate_without_retry"
        if mode == SLARecoveryMode.FAIL_FAST.value:
            return "fail_fast_and_escalate"
        if breach_status == SLABreachStatus.BREACHED.value:
            return "degrade_or_fast_failover_due_to_sla_breach"
        if mode == SLARecoveryMode.FAST_FAILOVER.value:
            return "perform_single_fast_failover_attempt"
        if mode == SLARecoveryMode.CONSERVATIVE.value:
            return "limit_recovery_attempts_and_preserve_sla"
        if not can_retry:
            return "degrade_without_retry"
        if category in {"timeout", "network"}:
            return "retry_with_sla_bounded_backoff"
        if severity in {FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value}:
            return "recover_conservatively_and_prepare_escalation"
        return "continue_with_standard_sla_guardrails"

    def _recommended_timeout(self, *, attempts: int, budget: SLABudget, estimated_attempt_seconds: float) -> float:
        if attempts <= 0 or budget.remaining_seconds <= 0:
            return 0.0
        usable_budget = max(0.0, budget.remaining_seconds - self.sla_safety_margin_seconds)
        per_attempt = safe_ratio(usable_budget, attempts, default=estimated_attempt_seconds)
        timeout = min(per_attempt * self.timeout_safety_fraction, self.max_timeout_seconds)
        return max(0.0, timeout)

    def _max_delay_for_budget(self, *, budget: SLABudget, attempts: int) -> float:
        if attempts <= 0 or budget.remaining_seconds <= 0:
            return 0.0
        usable_budget = max(0.0, budget.remaining_seconds - self.sla_safety_margin_seconds)
        per_attempt = safe_ratio(usable_budget, attempts, default=0.0)
        return max(0.0, min(self.max_delay_seconds, per_attempt * 0.25))

    def _estimated_attempt_seconds(self, *, category: str, severity: str) -> float:
        value = coerce_float(self.category_attempt_seconds.get(category), self.default_estimated_attempt_seconds, minimum=0.0)
        if severity == FailureSeverity.CRITICAL.value:
            value *= 1.25
        elif severity == FailureSeverity.HIGH.value:
            value *= 1.10
        return max(0.0, value)

    def _pressure(self, *, budget: SLABudget) -> float:
        total = budget.total_budget_seconds
        if total is None or total <= 0:
            if budget.remaining_seconds <= self.fast_failover_threshold_seconds:
                return 1.0
            if budget.remaining_seconds <= self.conservative_threshold_seconds:
                return 0.75
            return 0.0
        consumed = max(0.0, total - budget.remaining_seconds)
        return round(coerce_float(safe_ratio(consumed, total, default=0.0), 0.0, minimum=0.0, maximum=1.0), 4)

    def _breach_status(self, *, budget: SLABudget, pressure: float) -> str:
        if budget.remaining_seconds <= self.breach_grace_seconds:
            return SLABreachStatus.BREACHED.value
        if budget.remaining_seconds <= self.low_budget_threshold_seconds:
            return SLABreachStatus.NEAR_BREACH.value
        if pressure >= self.high_pressure_threshold:
            return SLABreachStatus.WATCH.value
        return SLABreachStatus.OK.value

    def _resolve_priority(self, *, context: Mapping[str, Any], severity: str) -> str:
        return str(context.get("priority") or self.priority_by_severity.get(severity) or self.default_priority)

    def _raise_priority_for_pressure(self, *, priority: str, pressure: float, breach_status: str) -> str:
        if breach_status == SLABreachStatus.BREACHED.value:
            return self._higher_priority(priority, "p0")
        if breach_status == SLABreachStatus.NEAR_BREACH.value:
            return self._higher_priority(priority, "p1")
        configured_floor = None
        for threshold, floor in self.priority_floor_by_pressure.items():
            if pressure >= coerce_float(threshold, 1.0):
                configured_floor = str(floor)
        if configured_floor:
            return self._higher_priority(priority, configured_floor)
        return priority

    @staticmethod
    def _higher_priority(current: str, floor: str) -> str:
        order = {"p0": 0, "p1": 1, "p2": 2, "p3": 3, "critical": 0, "high": 1, "normal": 2, "low": 3}
        current_value = order.get(str(current).lower(), 2)
        floor_value = order.get(str(floor).lower(), 2)
        return floor if floor_value < current_value else current

    @staticmethod
    def _normalize_mode(value: Any) -> str:
        normalized = str(value or SLARecoveryMode.STANDARD.value).strip().lower()
        for mode in SLARecoveryMode:
            if mode.value == normalized:
                return mode.value
        return SLARecoveryMode.STANDARD.value

    @staticmethod
    def _numeric_or_none(value: Any) -> Optional[float]:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
            if not math.isfinite(parsed):
                return None
            return parsed
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _coerce_modifier_mapping(value: Any, *, default: Mapping[str, int]) -> Dict[str, int]:
        source = deep_merge(default, coerce_mapping(value))
        return {str(key): coerce_int(item, 0, minimum=-100, maximum=100) for key, item in source.items()}

    @staticmethod
    def _coerce_float_mapping(value: Any, *, default: Mapping[str, float]) -> Dict[str, float]:
        source = deep_merge(default, coerce_mapping(value))
        return {str(key): coerce_float(item, 0.0, minimum=0.0) for key, item in source.items()}

    def _maybe_emit(
        self,
        *,
        evaluation: SLAEvaluation,
        failure: Mapping[str, Any],
        context: Mapping[str, Any],
        emit: Optional[bool],
    ) -> None:
        should_emit = self.emit_to_memory if emit is None else bool(emit)
        if not should_emit:
            return
        self.emit_sla_event(evaluation=evaluation, failure=failure, context=context)

    def _validate_configuration(self) -> None:
        if self.min_attempts > self.max_attempts:
            raise ConfigurationError(
                "SLARecoveryPolicy min_attempts cannot exceed max_attempts",
                context={"min_attempts": self.min_attempts, "max_attempts": self.max_attempts},
                code="HANDLER_SLA_ATTEMPT_BOUNDS_INVALID",
                policy=self.error_policy,
            )
        if self.default_max_attempts > self.max_attempts:
            raise ConfigurationError(
                "SLARecoveryPolicy default_max_attempts cannot exceed max_attempts",
                context={"default_max_attempts": self.default_max_attempts, "max_attempts": self.max_attempts},
                code="HANDLER_SLA_DEFAULT_ATTEMPTS_INVALID",
                policy=self.error_policy,
            )
        if self.fast_failover_threshold_seconds > self.conservative_threshold_seconds:
            logger.warning(
                "SLA fast_failover_threshold_seconds is greater than conservative_threshold_seconds; low-budget mode may dominate."
            )


if __name__ == "__main__":
    print("\n=== Running SLA Recovery Policy ===\n")
    printer.status("TEST", "SLA Recovery Policy initialized", "info")
    from .handler_memory import HandlerMemory

    strict_policy = HandlerErrorPolicy(
        name="sla_policy.strict_test",
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
            "max_postmortems": 5,
            "sanitize_payloads": True,
        },
        error_policy=strict_policy,
    )
    policy = SLARecoveryPolicy(
        config={
            "default_recovery_budget_seconds": 30,
            "default_max_attempts": 2,
            "max_attempts": 4,
            "emit_to_memory": True,
            "fast_failover_threshold_seconds": 3,
            "conservative_threshold_seconds": 8,
        },
        memory=memory,
        error_policy=strict_policy,
    )

    context = {
        "task_id": "handler-sla-smoke-001",
        "route": "handler.recovery",
        "agent": "demo_agent",
        "priority": "normal",
        "correlation_id": "corr-handler-sla-test",
        "password": "SuperSecret123",
        "sla": {
            "max_recovery_seconds": 12,
            "recovery_started_ts": utc_timestamp() - 2,
            "api_key": "sk-test-123",
        },
    }
    failure = build_normalized_failure(
        error=TimeoutError("Upstream timed out with Authorization: Bearer token-123"),
        context=context,
        policy=strict_policy,
        source="handler.sla_policy.__main__",
        correlation_id="corr-handler-sla-test",
    )

    standard = policy.evaluate(context=context, normalized_failure=failure)
    gate = policy.should_continue(attempted_attempts=0, context=context, normalized_failure=failure)
    timebox = policy.timebox_for_attempt(0, context=context, normalized_failure=failure)

    low_budget = policy.evaluate(
        context={**context, "sla": {"max_recovery_seconds": 2}},
        normalized_failure=failure,
    )
    critical_failure = build_normalized_failure(
        error=RuntimeError("critical runtime corruption detected"),
        context=context,
        policy=strict_policy,
        source="handler.sla_policy.__main__",
        correlation_id="corr-handler-sla-critical",
    )
    critical = policy.evaluate(context=context, normalized_failure=critical_failure)

    security_failure = SecurityError(
        "Security breach with password=SuperSecret123 and token=abc123",
        context=context,
        policy=strict_policy,
    ).to_failure_payload()
    security = policy.evaluate(context=context, normalized_failure=security_failure)

    many = policy.evaluate_many([failure, critical_failure, security_failure], context=context)
    summary = policy.summarize(many)
    telemetry = memory.recent_telemetry(limit=10)
    deadline = policy.deadline_from_budget(5)

    serialized = stable_json_dumps(
        {
            "standard": standard,
            "gate": gate,
            "timebox": timebox,
            "low_budget": low_budget,
            "critical": critical,
            "security": security,
            "many": many,
            "summary": summary,
            "telemetry": telemetry,
            "deadline": deadline,
        }
    )

    assert standard["remaining_seconds"] > 0
    assert standard["recommended_attempts"] >= 1
    assert standard["can_retry"] is True
    assert gate["allowed"] is True
    assert timebox["allowed"] is True
    assert low_budget["mode"] in {SLARecoveryMode.FAST_FAILOVER.value, SLARecoveryMode.CONSERVATIVE.value, SLARecoveryMode.DEGRADE.value}
    assert critical["mode"] in {SLARecoveryMode.CONSERVATIVE.value, SLARecoveryMode.FAIL_FAST.value}
    assert security["mode"] == SLARecoveryMode.QUARANTINE.value
    assert security["can_retry"] is False
    assert len(many) == 3
    assert summary["evaluation_count"] == 3
    assert len(telemetry) >= 1
    assert "SuperSecret123" not in serialized
    assert "sk-test-123" not in serialized
    assert "token-123" not in serialized
    assert "abc123" not in serialized

    printer.pretty("Standard SLA evaluation", standard, "success")
    printer.pretty("Low-budget SLA evaluation", low_budget, "success")
    printer.pretty("SLA summary", summary, "success")
    print("\n=== Test ran successfully ===\n")
