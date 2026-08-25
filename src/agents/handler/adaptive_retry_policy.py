from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.handler_error import *
from .utils.handler_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Adaptive Retry Policy")
printer = PrettyPrinter()


@dataclass(frozen=True)
class RetryDecision:
    """
    Rich retry decision emitted by AdaptiveRetryPolicy.

    HandlerAgent can keep using retries_for_fingerprint(...), which returns only the
    integer retry budget. More advanced orchestration can call decide(...) to inspect
    why a budget was selected, what backoff schedule should be used, and whether SLA
    or historical failure pressure constrained the decision.
    """

    fingerprint: str
    retries: int
    retryable: bool
    severity: str
    category: str
    action: str
    reason: str
    base_retries: int
    min_retries: int
    max_retries: int
    historical_total: int = 0
    historical_recovered: int = 0
    historical_failed: int = 0
    historical_success_rate: float = 0.0
    recent_total: int = 0
    recent_recovered: int = 0
    recent_failed: int = 0
    recent_success_rate: float = 0.0
    consecutive_failures: int = 0
    confidence: float = 0.0
    budget_seconds: Optional[float] = None
    budget_limited: bool = False
    delay_schedule: List[float] = field(default_factory=list)
    modifiers: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=utc_timestamp)

    @property
    def allowed(self) -> bool:
        return self.retryable and self.retries > 0

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "schema": "handler.retry_decision.v2",
                "fingerprint": self.fingerprint,
                "allowed": self.allowed,
                "retries": self.retries,
                "retryable": self.retryable,
                "severity": self.severity,
                "category": self.category,
                "action": self.action,
                "reason": self.reason,
                "base_retries": self.base_retries,
                "min_retries": self.min_retries,
                "max_retries": self.max_retries,
                "historical_total": self.historical_total,
                "historical_recovered": self.historical_recovered,
                "historical_failed": self.historical_failed,
                "historical_success_rate": self.historical_success_rate,
                "recent_total": self.recent_total,
                "recent_recovered": self.recent_recovered,
                "recent_failed": self.recent_failed,
                "recent_success_rate": self.recent_success_rate,
                "consecutive_failures": self.consecutive_failures,
                "confidence": self.confidence,
                "budget_seconds": self.budget_seconds,
                "budget_limited": self.budget_limited,
                "delay_schedule": self.delay_schedule,
                "modifiers": self.modifiers,
                "metadata": self.metadata,
                "timestamp": self.timestamp,
            },
            drop_none=True,
        )


@dataclass(frozen=True)
class RetryHistoryStats:
    """Fingerprint-level recovery statistics derived from telemetry."""

    fingerprint: str
    total: int
    recovered: int
    failed: int
    degraded: int
    success_rate: float
    failure_rate: float
    recent_total: int
    recent_recovered: int
    recent_failed: int
    recent_success_rate: float
    consecutive_failures: int
    last_failure_ts: Optional[float] = None
    last_success_ts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "fingerprint": self.fingerprint,
                "total": self.total,
                "recovered": self.recovered,
                "failed": self.failed,
                "degraded": self.degraded,
                "success_rate": self.success_rate,
                "failure_rate": self.failure_rate,
                "recent_total": self.recent_total,
                "recent_recovered": self.recent_recovered,
                "recent_failed": self.recent_failed,
                "recent_success_rate": self.recent_success_rate,
                "consecutive_failures": self.consecutive_failures,
                "last_failure_ts": self.last_failure_ts,
                "last_success_ts": self.last_success_ts,
            },
            drop_none=True,
        )


class AdaptiveRetryPolicy:
    """
    Production adaptive retry policy for HandlerAgent recovery orchestration.

    Scope:
    - keeps the legacy retries_for_fingerprint(...) integer API
    - produces explainable RetryDecision payloads for richer orchestration
    - adapts retry budgets from severity, category, policy action, historical outcomes,
      recent failure pressure, and SLA budget constraints
    - computes bounded exponential backoff schedules with deterministic jitter
    - consumes telemetry_history or HandlerMemory-like objects without importing memory

    This class does not execute recovery, record policy failures, or own circuit-breakers.
    HandlerPolicy remains responsible for circuit-breaker gating. SLARecoveryPolicy remains
    responsible for high-level SLA mode decisions. This policy only computes adaptive retry
    budgets and retry timing hints.
    """

    DEFAULT_NON_RETRYABLE_ACTIONS: Tuple[str, ...] = (
        HandlerRecoveryAction.FAIL_FAST.value,
        HandlerRecoveryAction.QUARANTINE.value,
    )
    DEFAULT_FAIL_FAST_CATEGORIES: Tuple[str, ...] = ("security",)

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        memory: Any = None,
        error_policy: Optional[HandlerErrorPolicy] = None,
    ):
        self.config = load_global_config()
        policy_cfg = get_config_section("policy")
        retry_cfg = get_config_section("adaptive_retry_policy")

        merged = deep_merge(policy_cfg, retry_cfg)
        if isinstance(config, Mapping):
            merged = deep_merge(merged, config)

        policy_config = merged.get("error_policy") if isinstance(merged.get("error_policy"), Mapping) else None
        self.error_policy = error_policy or HandlerErrorPolicy.from_mapping(policy_config)
        self.memory = memory

        self.enabled = coerce_bool(merged.get("enabled"), default=True)
        self.base_max_retries = coerce_int(merged.get("base_max_retries", merged.get("max_retries")), 2, minimum=0, maximum=100)
        self.min_retries = coerce_int(merged.get("min_retries", merged.get("adaptive_retry_min")), 0, minimum=0, maximum=100)
        self.max_retries = coerce_int(
            merged.get("max_retries_cap", merged.get("adaptive_retry_max", max(self.base_max_retries, 4))),
            max(self.base_max_retries, 4),
            minimum=self.min_retries,
            maximum=100,
        )
        self.min_samples = coerce_int(merged.get("min_samples", merged.get("adaptive_retry_min_samples")), 3, minimum=0, maximum=100_000)
        self.default_history_limit = coerce_int(merged.get("default_history_limit"), 500, minimum=1, maximum=1_000_000)
        self.recent_window_seconds = coerce_float(merged.get("recent_window_seconds"), 300.0, minimum=0.0)

        self.high_success_rate_threshold = coerce_float(merged.get("high_success_rate_threshold"), 0.70, minimum=0.0, maximum=1.0)
        self.low_success_rate_threshold = coerce_float(merged.get("low_success_rate_threshold"), 0.20, minimum=0.0, maximum=1.0)
        self.recent_failure_rate_threshold = coerce_float(merged.get("recent_failure_rate_threshold"), 0.85, minimum=0.0, maximum=1.0)
        self.consecutive_failure_threshold = coerce_int(merged.get("consecutive_failure_threshold"), 3, minimum=1, maximum=100_000)
        self.recent_failure_suppression_enabled = coerce_bool(merged.get("recent_failure_suppression_enabled"), default=True)
        self.low_confidence_penalty_enabled = coerce_bool(merged.get("low_confidence_penalty_enabled"), default=True)

        self.severity_modifiers = self._coerce_modifier_mapping(
            merged.get("severity_modifiers"),
            default={
                FailureSeverity.LOW.value: 1,
                FailureSeverity.MEDIUM.value: 0,
                FailureSeverity.HIGH.value: -1,
                FailureSeverity.CRITICAL.value: -2,
            },
        )
        self.category_modifiers = self._coerce_modifier_mapping(
            merged.get("category_modifiers"),
            default={
                "timeout": 1,
                "network": 1,
                "resource": 0,
                "unicode": 0,
                "memory": -1,
                "dependency": -1,
                "validation": -1,
                "security": -self.max_retries,
                "runtime": 0,
            },
        )
        self.action_modifiers = self._coerce_modifier_mapping(
            merged.get("action_modifiers"),
            default={
                HandlerRecoveryAction.RETRY.value: 1,
                HandlerRecoveryAction.DEGRADE.value: -1,
                HandlerRecoveryAction.ESCALATE.value: -1,
                HandlerRecoveryAction.FAIL_FAST.value: -self.max_retries,
                HandlerRecoveryAction.QUARANTINE.value: -self.max_retries,
                HandlerRecoveryAction.NONE.value: 0,
            },
        )
        self.non_retryable_actions = tuple(
            normalize_recovery_action(action)
            for action in coerce_list(merged.get("non_retryable_actions"), default=self.DEFAULT_NON_RETRYABLE_ACTIONS)
        )
        self.fail_fast_categories = tuple(
            normalize_identifier(category, default="runtime")
            for category in coerce_list(merged.get("fail_fast_categories"), default=self.DEFAULT_FAIL_FAST_CATEGORIES)
        )

        self.respect_sla_budget = coerce_bool(merged.get("respect_sla_budget"), default=True)
        self.default_remaining_budget_seconds = coerce_float(merged.get("default_remaining_budget_seconds"), 30.0, minimum=0.0)
        self.min_seconds_per_attempt = coerce_float(merged.get("min_seconds_per_attempt"), 1.5, minimum=0.0)
        self.sla_safety_margin_seconds = coerce_float(merged.get("sla_safety_margin_seconds"), 0.25, minimum=0.0)

        self.backoff_initial_seconds = coerce_float(merged.get("backoff_initial_seconds"), 0.5, minimum=0.0)
        self.backoff_multiplier = coerce_float(merged.get("backoff_multiplier"), 2.0, minimum=1.0)
        self.backoff_max_seconds = coerce_float(merged.get("backoff_max_seconds"), 8.0, minimum=0.0)
        self.jitter_fraction = coerce_float(merged.get("jitter_fraction"), 0.15, minimum=0.0, maximum=1.0)
        self.round_delay_digits = coerce_int(merged.get("round_delay_digits"), 3, minimum=0, maximum=6)

        self.confidence_floor = coerce_float(merged.get("confidence_floor"), 0.35, minimum=0.0, maximum=1.0)
        self.confidence_ceiling = coerce_float(merged.get("confidence_ceiling"), 0.95, minimum=0.0, maximum=1.0)
        self.min_confidence_for_retry_bonus = coerce_float(merged.get("min_confidence_for_retry_bonus"), 0.55, minimum=0.0, maximum=1.0)

        self._validate_configuration()
        logger.info(
            "Adaptive Retry Policy initialized | base=%s min=%s max=%s enabled=%s",
            self.base_max_retries,
            self.min_retries,
            self.max_retries,
            self.enabled,
        )

    def attach_memory(self, memory: Any) -> None:
        """Attach a HandlerMemory-like object for future decisions."""
        self.memory = memory

    def retries_for_fingerprint(
        self,
        fingerprint: str,
        severity: str,
        retryable: bool,
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Legacy API used by HandlerAgent.

        Returns only the retry count, while internally using the richer decision path.
        """
        decision = self.decide(
            fingerprint=fingerprint,
            severity=severity,
            retryable=retryable,
            telemetry_history=telemetry_history,
        )
        return decision.retries

    def retries_for_failure(
        self,
        normalized_failure: Mapping[str, Any],
        *,
        telemetry_history: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        sla: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Return the adaptive retry budget for a normalized failure payload."""
        return self.decide(
            normalized_failure=normalized_failure,
            telemetry_history=telemetry_history,
            context=context,
            sla=sla,
        ).retries

    def should_retry(
        self,
        attempted_retries: int,
        *,
        normalized_failure: Optional[Mapping[str, Any]] = None,
        fingerprint: Optional[str] = None,
        severity: Optional[str] = None,
        retryable: Optional[bool] = None,
        telemetry_history: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        sla: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return an executable retry-gating payload for one attempted retry count."""
        decision = self.decide(
            normalized_failure=normalized_failure,
            fingerprint=fingerprint,
            severity=severity,
            retryable=retryable,
            telemetry_history=telemetry_history,
            context=context,
            sla=sla,
        )
        attempted = coerce_int(attempted_retries, 0, minimum=0)
        allowed = decision.allowed and attempted < decision.retries
        return {
            "allowed": allowed,
            "attempted_retries": attempted,
            "remaining_retries": max(0, decision.retries - attempted),
            "next_delay_seconds": self.delay_for_attempt(attempted, fingerprint=decision.fingerprint) if allowed else 0.0,
            "decision": decision.to_dict(),
        }

    def decide(
        self,
        *,
        normalized_failure: Optional[Mapping[str, Any]] = None,
        fingerprint: Optional[str] = None,
        severity: Optional[str] = None,
        retryable: Optional[bool] = None,
        telemetry_history: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        sla: Optional[Mapping[str, Any]] = None,
        recovery_action: Optional[str] = None,
        category: Optional[str] = None,
        memory: Any = None,
    ) -> RetryDecision:
        """Build an explainable adaptive retry decision."""
        try:
            failure = self._resolve_failure(
                normalized_failure=normalized_failure,
                fingerprint=fingerprint,
                severity=severity,
                retryable=retryable,
                recovery_action=recovery_action,
                category=category,
            )
            context_map = coerce_mapping(context)
            resolved_fingerprint = str(failure.get("context_hash") or fingerprint or "")
            if not resolved_fingerprint:
                resolved_fingerprint = stable_hash({"failure": failure, "context": select_keys(context_map, ("route", "agent", "task_id"))}, length=16)

            resolved_severity = normalize_severity(failure.get("severity") or severity)
            resolved_retryable = coerce_bool(failure.get("retryable") if retryable is None else retryable, default=False)
            resolved_category = normalize_identifier(failure.get("category") or category or classify_failure_category(failure.get("type"), failure.get("message")), default="runtime")
            resolved_action = normalize_recovery_action(failure.get("policy_action") or failure.get("action") or recovery_action)

            history = self._resolve_history(
                fingerprint=resolved_fingerprint,
                telemetry_history=telemetry_history,
                memory=memory,
            )
            stats = self.fingerprint_stats(
                fingerprint=resolved_fingerprint,
                telemetry_history=history,
            )
            budget_seconds = self._remaining_budget_seconds(context=context_map, sla=sla)

            if not self.enabled:
                retries = self._clamp_retries(self.base_max_retries if resolved_retryable else 0)
                return self._decision(
                    fingerprint=resolved_fingerprint,
                    retries=retries,
                    retryable=resolved_retryable,
                    severity=resolved_severity,
                    category=resolved_category,
                    action=resolved_action,
                    reason="adaptive_policy_disabled",
                    stats=stats,
                    budget_seconds=budget_seconds,
                    budget_limited=False,
                    modifiers={},
                )

            if not resolved_retryable:
                return self._decision(
                    fingerprint=resolved_fingerprint,
                    retries=0,
                    retryable=False,
                    severity=resolved_severity,
                    category=resolved_category,
                    action=resolved_action,
                    reason="failure_marked_non_retryable",
                    stats=stats,
                    budget_seconds=budget_seconds,
                    budget_limited=False,
                    modifiers={},
                )

            if resolved_action in self.non_retryable_actions:
                return self._decision(
                    fingerprint=resolved_fingerprint,
                    retries=0,
                    retryable=False,
                    severity=resolved_severity,
                    category=resolved_category,
                    action=resolved_action,
                    reason=f"policy_action_{resolved_action}_is_non_retryable",
                    stats=stats,
                    budget_seconds=budget_seconds,
                    budget_limited=False,
                    modifiers={},
                )

            if resolved_category in self.fail_fast_categories:
                return self._decision(
                    fingerprint=resolved_fingerprint,
                    retries=0,
                    retryable=False,
                    severity=resolved_severity,
                    category=resolved_category,
                    action=resolved_action,
                    reason=f"category_{resolved_category}_requires_fail_fast",
                    stats=stats,
                    budget_seconds=budget_seconds,
                    budget_limited=False,
                    modifiers={},
                )

            modifiers = self._modifiers_for(
                severity=resolved_severity,
                category=resolved_category,
                action=resolved_action,
                stats=stats,
            )
            retries = self.base_max_retries + sum(modifiers.values())
            reason = "base_policy"

            if stats.total >= self.min_samples:
                if stats.success_rate >= self.high_success_rate_threshold and self._confidence(stats) >= self.min_confidence_for_retry_bonus:
                    retries += 1
                    modifiers["historical_success_bonus"] = modifiers.get("historical_success_bonus", 0) + 1
                    reason = "historical_success_rate_bonus"
                elif stats.success_rate <= self.low_success_rate_threshold:
                    retries -= 1
                    modifiers["historical_failure_penalty"] = modifiers.get("historical_failure_penalty", 0) - 1
                    reason = "historical_low_success_penalty"

            if self.recent_failure_suppression_enabled and stats.recent_total >= self.min_samples:
                recent_failure_rate = safe_ratio(stats.recent_failed, stats.recent_total, default=0.0)
                if recent_failure_rate >= self.recent_failure_rate_threshold:
                    retries -= 1
                    modifiers["recent_failure_pressure_penalty"] = modifiers.get("recent_failure_pressure_penalty", 0) - 1
                    reason = "recent_failure_pressure_penalty"

            if stats.consecutive_failures >= self.consecutive_failure_threshold:
                retries -= 1
                modifiers["consecutive_failure_penalty"] = modifiers.get("consecutive_failure_penalty", 0) - 1
                reason = "consecutive_failure_penalty"

            if self.low_confidence_penalty_enabled and stats.total > 0 and self._confidence(stats) < self.confidence_floor:
                retries -= 1
                modifiers["low_confidence_penalty"] = modifiers.get("low_confidence_penalty", 0) - 1
                reason = "low_confidence_penalty"

            retries = self._clamp_retries(retries)
            budget_limited = False
            if self.respect_sla_budget:
                budget_retries = self._budget_limited_retries(retries=retries, budget_seconds=budget_seconds)
                if budget_retries < retries:
                    budget_limited = True
                    retries = budget_retries
                    modifiers["sla_budget_limit"] = budget_retries - self.base_max_retries
                    reason = "sla_budget_limited"

            return self._decision(
                fingerprint=resolved_fingerprint,
                retries=retries,
                retryable=retries > 0,
                severity=resolved_severity,
                category=resolved_category,
                action=resolved_action,
                reason=reason,
                stats=stats,
                budget_seconds=budget_seconds,
                budget_limited=budget_limited,
                modifiers=modifiers,
                failure=failure,
            )
        except HandlerError:
            raise
        except Exception as exc:
            raise PolicyError(
                "Adaptive retry policy failed to build retry decision",
                cause=exc,
                context={
                    "fingerprint": fingerprint,
                    "severity": severity,
                    "retryable": retryable,
                    "has_normalized_failure": normalized_failure is not None,
                },
                code="HANDLER_ADAPTIVE_RETRY_DECISION_FAILED",
                policy=self.error_policy,
            ) from exc

    def delay_for_attempt(self, attempt_index: int, *, fingerprint: Optional[str] = None) -> float:
        """Return bounded exponential backoff delay for a zero-based attempt index."""
        attempt = coerce_int(attempt_index, 0, minimum=0)
        delay = self.backoff_initial_seconds * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.backoff_max_seconds) if self.backoff_max_seconds > 0 else delay
        if self.jitter_fraction > 0 and delay > 0:
            jitter = self._deterministic_jitter(fingerprint=fingerprint or "adaptive_retry", attempt_index=attempt)
            delay = delay + (delay * self.jitter_fraction * jitter)
        return round(max(0.0, delay), self.round_delay_digits)

    def delay_schedule(self, retries: int, *, fingerprint: Optional[str] = None) -> List[float]:
        """Return delay schedule for the selected retry budget."""
        count = coerce_int(retries, 0, minimum=0, maximum=self.max_retries)
        return [self.delay_for_attempt(index, fingerprint=fingerprint) for index in range(count)]

    def fingerprint_stats(
        self,
        *,
        fingerprint: str,
        telemetry_history: Optional[Sequence[Mapping[str, Any]]] = None,
        signature: Optional[str] = None,
    ) -> RetryHistoryStats:
        """Return fingerprint-level telemetry statistics."""
        events = [dict(event) for event in coerce_list(telemetry_history) if isinstance(event, Mapping)]
        matched = [event for event in events if event_matches_failure(event, context_hash=fingerprint, signature=signature)]
        now = utc_timestamp()
        recent_cutoff = now - self.recent_window_seconds if self.recent_window_seconds > 0 else None
        recent = [event for event in matched if recent_cutoff is None or coerce_float(event.get("timestamp"), 0.0) >= recent_cutoff]

        total = len(matched)
        recovered = 0
        failed = 0
        degraded = 0
        last_failure_ts: Optional[float] = None
        last_success_ts: Optional[float] = None

        for event in matched:
            status = str(coerce_mapping(event.get("recovery")).get("status", "")).lower()
            event_ts = coerce_float(event.get("timestamp"), 0.0)
            if status == "recovered":
                recovered += 1
                last_success_ts = max(last_success_ts or event_ts, event_ts)
            elif status == "degraded":
                degraded += 1
                recovered += 1
                last_success_ts = max(last_success_ts or event_ts, event_ts)
            else:
                failed += 1
                last_failure_ts = max(last_failure_ts or event_ts, event_ts)

        recent_recovered = 0
        recent_failed = 0
        for event in recent:
            status = str(coerce_mapping(event.get("recovery")).get("status", "")).lower()
            if status in {"recovered", "degraded"}:
                recent_recovered += 1
            else:
                recent_failed += 1

        consecutive_failures = self._consecutive_failures(matched)
        return RetryHistoryStats(
            fingerprint=str(fingerprint),
            total=total,
            recovered=recovered,
            failed=failed,
            degraded=degraded,
            success_rate=round(safe_ratio(recovered, total, default=0.0), 4),
            failure_rate=round(safe_ratio(failed, total, default=0.0), 4),
            recent_total=len(recent),
            recent_recovered=recent_recovered,
            recent_failed=recent_failed,
            recent_success_rate=round(safe_ratio(recent_recovered, len(recent), default=0.0), 4),
            consecutive_failures=consecutive_failures,
            last_failure_ts=last_failure_ts,
            last_success_ts=last_success_ts,
        )

    def summarize(self, telemetry_history: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
        """Summarize policy configuration and optional telemetry history."""
        history = list(telemetry_history or self._history_from_memory(limit=self.default_history_limit))
        stats = success_rate_for_events(history)
        return {
            "schema": "handler.adaptive_retry.summary.v2",
            "timestamp": utc_timestamp(),
            "enabled": self.enabled,
            "base_max_retries": self.base_max_retries,
            "min_retries": self.min_retries,
            "max_retries": self.max_retries,
            "min_samples": self.min_samples,
            "recent_window_seconds": self.recent_window_seconds,
            "telemetry": stats,
        }

    def _resolve_failure(
        self,
        *,
        normalized_failure: Optional[Mapping[str, Any]],
        fingerprint: Optional[str],
        severity: Optional[str],
        retryable: Optional[bool],
        recovery_action: Optional[str],
        category: Optional[str],
    ) -> Dict[str, Any]:
        if isinstance(normalized_failure, Mapping):
            failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
        else:
            failure = {
                "type": DEFAULT_FAILURE_TYPE,
                "message": DEFAULT_FAILURE_MESSAGE,
                "severity": normalize_severity(severity or FailureSeverity.LOW),
                "retryable": coerce_bool(retryable, default=False),
                "context_hash": fingerprint or "",
                "timestamp": utc_timestamp(),
            }
        if fingerprint and not failure.get("context_hash"):
            failure["context_hash"] = fingerprint
        if severity is not None:
            failure["severity"] = normalize_severity(severity)
        if retryable is not None:
            failure["retryable"] = coerce_bool(retryable)
        if recovery_action is not None:
            failure["policy_action"] = normalize_recovery_action(recovery_action)
        if category is not None:
            failure["category"] = normalize_identifier(category, default="runtime")
        return failure

    def _resolve_history(self, *, fingerprint: str, memory: Any = None,
                         telemetry_history: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        if telemetry_history is not None:
            return [dict(event) for event in telemetry_history if isinstance(event, Mapping)]
    
        active_memory = memory or self.memory
        if active_memory is None:
            return []
    
        if hasattr(active_memory, "failure_history") and callable(active_memory.failure_history):
            history = active_memory.failure_history(context_hash=fingerprint, limit=self.default_history_limit)
            if isinstance(history, (list, tuple)):
                return [dict(event) for event in history if isinstance(event, Mapping)]
            # Unexpected non-iterable result; log and continue
            if history is not None:
                logger.warning("failure_history returned non-iterable type: %s", type(history).__name__)
    
        if hasattr(active_memory, "recent_telemetry") and callable(active_memory.recent_telemetry):
            history = active_memory.recent_telemetry(limit=self.default_history_limit)
            if isinstance(history, (list, tuple)):
                return [dict(event) for event in history if isinstance(event, Mapping)]
            if history is not None:
                logger.warning("recent_telemetry returned non-iterable type: %s", type(history).__name__)
    
        return []

    def _history_from_memory(self, *, limit: int) -> List[Dict[str, Any]]:
        if self.memory is None or not hasattr(self.memory, "recent_telemetry"):
            return []
        history = self.memory.recent_telemetry(limit=limit)
        if isinstance(history, (list, tuple)):
            return [dict(event) for event in history if isinstance(event, Mapping)]
        if history is not None:
            logger.warning("recent_telemetry returned non-iterable type: %s", type(history).__name__)
        return []

    def _modifiers_for(self, *, severity: str, category: str, action: str, stats: RetryHistoryStats) -> Dict[str, int]:
        modifiers = {
            "severity": self.severity_modifiers.get(severity, 0),
            "category": self.category_modifiers.get(category, 0),
            "action": self.action_modifiers.get(action, 0),
        }
        if stats.total == 0:
            modifiers["novel_failure_safety"] = 0
        return {key: value for key, value in modifiers.items() if value != 0}

    def _remaining_budget_seconds(self, *, context: Mapping[str, Any], sla: Optional[Mapping[str, Any]]) -> Optional[float]:
        if not self.respect_sla_budget:
            return None
        merged_context = dict(context or {})
        if sla is not None:
            merged_context["sla"] = coerce_mapping(sla)
        return compute_remaining_budget(context=merged_context, default_seconds=self.default_remaining_budget_seconds)

    def _budget_limited_retries(self, *, retries: int, budget_seconds: Optional[float]) -> int:
        if budget_seconds is None:
            return retries
        usable_budget = max(0.0, budget_seconds - self.sla_safety_margin_seconds)
        if self.min_seconds_per_attempt <= 0:
            return retries
        budget_cap = math.floor(usable_budget / self.min_seconds_per_attempt)
        return min(retries, max(0, int(budget_cap)))

    def _decision(
        self,
        *,
        fingerprint: str,
        retries: int,
        retryable: bool,
        severity: str,
        category: str,
        action: str,
        reason: str,
        stats: RetryHistoryStats,
        budget_seconds: Optional[float],
        budget_limited: bool,
        modifiers: Mapping[str, int],
        failure: Optional[Mapping[str, Any]] = None,
    ) -> RetryDecision:
        bounded_retries = self._clamp_retries(retries)
        confidence = self._confidence(stats)
        return RetryDecision(
            fingerprint=fingerprint,
            retries=bounded_retries,
            retryable=bool(retryable) and bounded_retries > 0,
            severity=normalize_severity(severity),
            category=normalize_identifier(category, default="runtime"),
            action=normalize_recovery_action(action),
            reason=reason,
            base_retries=self.base_max_retries,
            min_retries=self.min_retries,
            max_retries=self.max_retries,
            historical_total=stats.total,
            historical_recovered=stats.recovered,
            historical_failed=stats.failed,
            historical_success_rate=stats.success_rate,
            recent_total=stats.recent_total,
            recent_recovered=stats.recent_recovered,
            recent_failed=stats.recent_failed,
            recent_success_rate=stats.recent_success_rate,
            consecutive_failures=stats.consecutive_failures,
            confidence=confidence,
            budget_seconds=round(budget_seconds, 4) if budget_seconds is not None else None,
            budget_limited=budget_limited,
            delay_schedule=self.delay_schedule(bounded_retries, fingerprint=fingerprint),
            modifiers=dict(modifiers),
            metadata=compact_dict(
                {
                    "failure_type": failure.get("type") if isinstance(failure, Mapping) else None,
                    "category": category,
                    "last_failure_ts": stats.last_failure_ts,
                    "last_success_ts": stats.last_success_ts,
                    "recent_window_seconds": self.recent_window_seconds,
                    "min_samples": self.min_samples,
                },
                drop_none=True,
            ),
        )

    def _clamp_retries(self, value: Any) -> int:
        return coerce_int(value, self.base_max_retries, minimum=self.min_retries, maximum=self.max_retries)

    def _confidence(self, stats: RetryHistoryStats) -> float:
        if stats.total <= 0:
            return round(self.confidence_floor, 4)
        sample_factor = min(1.0, safe_ratio(stats.total, max(1, self.min_samples), default=0.0))
        stability = 1.0 - min(1.0, safe_ratio(stats.consecutive_failures, max(1, self.consecutive_failure_threshold), default=0.0) * 0.5)
        value = ((0.65 * stats.success_rate) + (0.35 * sample_factor)) * stability
        return round(coerce_float(value, self.confidence_floor, minimum=self.confidence_floor, maximum=self.confidence_ceiling), 4)

    @staticmethod
    def _consecutive_failures(events: Sequence[Mapping[str, Any]]) -> int:
        count = 0
        for event in reversed(stable_sort_events(events)):
            recovery = coerce_mapping(event.get("recovery"))
            status = str(recovery.get("status", "")).lower()
            if status in {"recovered", "degraded"}:
                break
            count += 1
        return count

    @staticmethod
    def _deterministic_jitter(*, fingerprint: str, attempt_index: int) -> float:
        raw = stable_hash({"fingerprint": fingerprint, "attempt": attempt_index}, length=8)
        value = int(raw, 16) / float(0xFFFFFFFF)
        return (value * 2.0) - 1.0

    @staticmethod
    def _coerce_modifier_mapping(value: Any, *, default: Mapping[str, int]) -> Dict[str, int]:
        source = coerce_mapping(value, default=default)
        return {str(key): coerce_int(item, 0, minimum=-100, maximum=100) for key, item in source.items()}

    def _validate_configuration(self) -> None:
        if self.min_retries > self.max_retries:
            raise ConfigurationError(
                "Adaptive retry min_retries cannot exceed max_retries",
                context={"min_retries": self.min_retries, "max_retries": self.max_retries},
                code="HANDLER_ADAPTIVE_RETRY_INVALID_BOUNDS",
                policy=self.error_policy,
            )
        if self.base_max_retries > self.max_retries:
            raise ConfigurationError(
                "Adaptive retry base_max_retries cannot exceed max_retries",
                context={"base_max_retries": self.base_max_retries, "max_retries": self.max_retries},
                code="HANDLER_ADAPTIVE_RETRY_BASE_EXCEEDS_MAX",
                policy=self.error_policy,
            )
        if self.low_success_rate_threshold > self.high_success_rate_threshold:
            raise ConfigurationError(
                "Adaptive retry low_success_rate_threshold cannot exceed high_success_rate_threshold",
                context={
                    "low_success_rate_threshold": self.low_success_rate_threshold,
                    "high_success_rate_threshold": self.high_success_rate_threshold,
                },
                code="HANDLER_ADAPTIVE_RETRY_INVALID_THRESHOLDS",
                policy=self.error_policy,
            )


if __name__ == "__main__":
    print("\n=== Running Adaptive Retry Policy ===\n")
    printer.status("TEST", "Adaptive Retry Policy initialized", "info")

    from .handler_memory import HandlerMemory

    strict_policy = HandlerErrorPolicy(
        name="adaptive_retry_policy.strict_test",
        expose_internal_messages=False,
        include_context_in_public=False,
        include_context_in_telemetry=True,
        max_message_chars=240,
        max_string_chars=160,
    )

    memory = HandlerMemory(error_policy=strict_policy)
    retry_policy = AdaptiveRetryPolicy(
        config={
            "enabled": True,
            "base_max_retries": 2,
            "min_retries": 0,
            "max_retries_cap": 5,
            "min_samples": 3,
            "recent_window_seconds": 600,
            "high_success_rate_threshold": 0.70,
            "low_success_rate_threshold": 0.20,
            "respect_sla_budget": True,
            "min_seconds_per_attempt": 1.0,
            "backoff_initial_seconds": 0.25,
            "backoff_multiplier": 2.0,
            "backoff_max_seconds": 4.0,
            "jitter_fraction": 0.0,
        },
        memory=memory,
        error_policy=strict_policy,
    )

    context = {
        "task_id": "adaptive-retry-smoke-001",
        "agent": "demo_agent",
        "route": "handler.recovery",
        "correlation_id": "corr-adaptive-retry-test",
        "sla": {"max_recovery_seconds": 12},
        "password": "SuperSecret123",
    }
    failure = build_normalized_failure(
        error=TimeoutError("Upstream timed out with Authorization: Bearer token-123"),
        context=context,
        policy=strict_policy,
        source="handler.adaptive_retry.__main__",
        correlation_id="corr-adaptive-retry-test",
    )

    for index in range(3):
        memory.append_recovery_telemetry(
            failure=failure,
            recovery={
                "status": "recovered" if index < 3 else "failed",
                "strategy": "timeout",
                "attempts": 1,
                "sla": {"remaining_seconds": 10.0},
            },
            context=context,
            insight={"signature": "timeout:test", "recommendation": "retry_with_backoff"},
        )

    telemetry_history = memory.recent_telemetry(limit=10)
    decision = retry_policy.decide(
        normalized_failure=failure,
        telemetry_history=telemetry_history,
        context=context,
    )
    legacy_retries = retry_policy.retries_for_fingerprint(
        fingerprint=failure["context_hash"],
        severity=failure["severity"],
        retryable=failure["retryable"],
        telemetry_history=telemetry_history,
    )
    legacy_decision = retry_policy.decide(
        fingerprint=failure["context_hash"],
        severity=failure["severity"],
        retryable=failure["retryable"],
        telemetry_history=telemetry_history,
    )
    gate = retry_policy.should_retry(
        attempted_retries=0,
        normalized_failure=failure,
        telemetry_history=telemetry_history,
        context=context,
    )
    budget_limited = retry_policy.decide(
        normalized_failure=failure,
        telemetry_history=telemetry_history,
        context={**context, "sla": {"max_recovery_seconds": 1.2}},
    )
    non_retryable = retry_policy.decide(
        fingerprint="security:test",
        severity=FailureSeverity.CRITICAL.value,
        retryable=False,
        category="security",
        recovery_action=HandlerRecoveryAction.QUARANTINE.value,
    )
    summary = retry_policy.summarize(telemetry_history)

    serialized = stable_json_dumps(
        {
            "decision": decision.to_dict(),
            "legacy_retries": legacy_retries,
            "legacy_decision": legacy_decision.to_dict(),
            "gate": gate,
            "budget_limited": budget_limited.to_dict(),
            "non_retryable": non_retryable.to_dict(),
            "summary": summary,
            "telemetry": telemetry_history,
        }
    )

    assert decision.retries >= retry_policy.base_max_retries
    assert legacy_retries == legacy_decision.retries
    assert gate["allowed"] is True
    assert gate["remaining_retries"] == decision.retries
    assert budget_limited.budget_limited is True
    assert budget_limited.retries <= decision.retries
    assert non_retryable.retries == 0
    assert non_retryable.allowed is False
    assert summary["telemetry"]["total"] == 3
    assert "SuperSecret123" not in serialized
    assert "token-123" not in serialized

    printer.pretty("Retry decision", decision.to_dict(), "success")
    printer.pretty("Retry gate", gate, "success")
    printer.pretty("Retry summary", summary, "success")
    print("\n=== Test ran successfully ===\n")
