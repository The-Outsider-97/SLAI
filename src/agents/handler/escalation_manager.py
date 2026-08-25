from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.handler_error import *
from .utils.handler_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Escalation Manager")
printer = PrettyPrinter()


@dataclass(frozen=True)
class EscalationDecision:
    """
    Explainable routing decision produced by EscalationManager.

    The Manager still exposes build_handoff_payload(...) for HandlerAgent compatibility,
    but advanced callers can inspect decide(...) for the selected target, priority,
    acknowledgement requirements, and policy evidence before building the handoff.
    """

    target_agent: str
    reason: str
    severity: str
    retryable: bool
    action: str
    priority: str
    escalation_level: str
    require_ack: bool
    matrix_key: str
    routed_by: str
    confidence: float = 0.0
    deadline_seconds: Optional[float] = None
    correlation_id: Optional[str] = None
    labels: Tuple[str, ...] = field(default_factory=tuple)
    evidence: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=utc_timestamp)

    @property
    def should_escalate(self) -> bool:
        return self.target_agent not in {"", "none", "handler_agent"} or self.action in {
            HandlerRecoveryAction.ESCALATE.value,
            HandlerRecoveryAction.FAIL_FAST.value,
            HandlerRecoveryAction.QUARANTINE.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "schema": "handler.escalation_decision.v2",
                "target_agent": self.target_agent,
                "reason": self.reason,
                "severity": self.severity,
                "retryable": self.retryable,
                "action": self.action,
                "priority": self.priority,
                "escalation_level": self.escalation_level,
                "require_ack": self.require_ack,
                "matrix_key": self.matrix_key,
                "routed_by": self.routed_by,
                "confidence": self.confidence,
                "deadline_seconds": self.deadline_seconds,
                "correlation_id": self.correlation_id,
                "labels": list(self.labels),
                "evidence": [dict(item) for item in self.evidence],
                "metadata": dict(self.metadata),
                "timestamp": self.timestamp,
                "should_escalate": self.should_escalate,
            },
            drop_none=True,
            drop_empty=True,
        )


class EscalationManager:
    """
    Production escalation manager for HandlerAgent recovery orchestration.

    Scope:
    - keeps the legacy build_handoff_payload(...) API used by HandlerAgent
    - resolves escalation targets from policy action, recommendation, category,
      severity/retryability matrix, SLA pressure, and failure-intelligence hints
    - emits secure, bounded, schema-versioned handoff payloads
    - optionally appends escalation telemetry to HandlerMemory-like objects
    - does not execute recovery, own retry budgets, or duplicate helper logic

    Configuration is read from handler_config.yaml:
    - primary section: escalation_manager
    - backward-compatible matrix fallback: policy.escalation_matrix
    """

    DEFAULT_MATRIX: Mapping[str, Mapping[str, str]] = {
        FailureSeverity.CRITICAL.value: {"retryable": "safety_agent", "non_retryable": "planning_agent"},
        FailureSeverity.HIGH.value: {"retryable": "planning_agent", "non_retryable": "evaluation_agent"},
        FailureSeverity.MEDIUM.value: {"retryable": "handler_agent", "non_retryable": "planning_agent"},
        FailureSeverity.LOW.value: {"retryable": "handler_agent", "non_retryable": "handler_agent"},
    }
    DEFAULT_ACTION_ROUTES: Mapping[str, str] = {
        HandlerRecoveryAction.QUARANTINE.value: "safety_agent",
        HandlerRecoveryAction.FAIL_FAST.value: "planning_agent",
        HandlerRecoveryAction.ESCALATE.value: "planning_agent",
        HandlerRecoveryAction.DEGRADE.value: "handler_agent",
        HandlerRecoveryAction.RETRY.value: "handler_agent",
        HandlerRecoveryAction.NONE.value: "planning_agent",
    }
    DEFAULT_CATEGORY_ROUTES: Mapping[str, str] = {
        "security": "safety_agent",
        "dependency": "planning_agent",
        "memory": "planning_agent",
        "sla": "planning_agent",
        "validation": "evaluation_agent",
    }
    DEFAULT_RECOMMENDATION_ROUTES: Mapping[str, str] = {
        "quarantine_security_failure": "safety_agent",
        "immediate_escalation": "planning_agent",
        "collect_context_and_escalate": "planning_agent",
        "open_circuit_or_failover": "planning_agent",
        "validate_runtime_dependencies": "planning_agent",
        "validate_payload_and_schema": "evaluation_agent",
        "preserve_sla_and_fast_failover": "planning_agent",
    }
    DEFAULT_PRIORITY_BY_SEVERITY: Mapping[str, str] = {
        FailureSeverity.CRITICAL.value: "p0",
        FailureSeverity.HIGH.value: "p1",
        FailureSeverity.MEDIUM.value: "p2",
        FailureSeverity.LOW.value: "p3",
    }
    DEFAULT_LEVEL_BY_PRIORITY: Mapping[str, str] = {
        "p0": "page",
        "p1": "urgent_handoff",
        "p2": "standard_handoff",
        "p3": "informational",
    }
    DEFAULT_ACK_BY_PRIORITY: Mapping[str, bool] = {
        "p0": True,
        "p1": True,
        "p2": False,
        "p3": False,
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
        escalation_cfg = get_config_section("escalation_manager")

        merged = deep_merge(policy_cfg, escalation_cfg)
        if isinstance(config, Mapping):
            merged = deep_merge(merged, config)

        policy_config = merged.get("error_policy") if isinstance(merged.get("error_policy"), Mapping) else None
        self.error_policy = error_policy or HandlerErrorPolicy.from_mapping(policy_config)
        self.memory = memory

        self.enabled = coerce_bool(merged.get("enabled"), default=True)
        self.schema_version = coerce_str(merged.get("schema_version"), default="handler_escalation.v2")
        self.handoff_type = coerce_str(merged.get("handoff_type"), default="handler_escalation.v2")
        self.default_target_agent = normalize_identifier(merged.get("default_target_agent"), default="planning_agent")
        self.local_handler_agent = normalize_identifier(merged.get("local_handler_agent"), default="handler_agent")
        self.fallback_target_agent = normalize_identifier(merged.get("fallback_target_agent"), default="planning_agent")
        self.default_reason = coerce_str(merged.get("default_reason"), default="recovery_exhausted")
        self.max_reason_chars = coerce_int(merged.get("max_reason_chars"), 240, minimum=32, maximum=4000)
        self.max_message_chars = coerce_int(merged.get("max_message_chars"), 500, minimum=32, maximum=8000)
        self.max_context_fields = coerce_int(merged.get("max_context_fields"), 12, minimum=1, maximum=100)
        self.max_evidence_items = coerce_int(merged.get("max_evidence_items"), 10, minimum=0, maximum=100)
        self.max_payload_chars = coerce_int(merged.get("max_payload_chars"), 120_000, minimum=1024)
        self.emit_to_memory = coerce_bool(merged.get("emit_to_memory"), default=False)
        self.memory_event_type = normalize_identifier(merged.get("memory_event_type"), default="handler_escalation")
        self.include_public_error = coerce_bool(merged.get("include_public_error"), default=True)
        self.include_internal_evidence = coerce_bool(merged.get("include_internal_evidence"), default=True)
        self.recovery_failed_statuses = tuple(
            str(item).lower()
            for item in coerce_list(merged.get("recovery_failed_statuses"), default=("failed", "skipped", "unknown"), split_strings=True)
        )
        self.degraded_statuses = tuple(
            str(item).lower()
            for item in coerce_list(merged.get("degraded_statuses"), default=("degraded",), split_strings=True)
        )
        self.escalate_degraded_high_severity = coerce_bool(merged.get("escalate_degraded_high_severity"), default=True)
        self.low_sla_budget_seconds = coerce_float(merged.get("low_sla_budget_seconds"), 3.0, minimum=0.0)
        self.sla_priority_floor = coerce_str(merged.get("sla_priority_floor"), default="p1").lower()

        matrix = merged.get("escalation_matrix", self.DEFAULT_MATRIX)
        self.matrix = self._load_matrix(matrix)
        self.action_routes = self._load_route_mapping(merged.get("action_routes"), self.DEFAULT_ACTION_ROUTES, normalize_actions=True)
        self.category_routes = self._load_route_mapping(merged.get("category_routes"), self.DEFAULT_CATEGORY_ROUTES)
        self.recommendation_routes = self._load_route_mapping(merged.get("recommendation_routes"), self.DEFAULT_RECOMMENDATION_ROUTES)
        self.priority_by_severity = self._load_simple_mapping(merged.get("priority_by_severity"), self.DEFAULT_PRIORITY_BY_SEVERITY)
        self.level_by_priority = self._load_simple_mapping(merged.get("level_by_priority"), self.DEFAULT_LEVEL_BY_PRIORITY)
        self.ack_by_priority = self._load_bool_mapping(merged.get("ack_by_priority"), self.DEFAULT_ACK_BY_PRIORITY)
        self.target_overrides = self._load_route_mapping(merged.get("target_overrides"), {})

        self._validate_configuration()
        logger.info(
            "Escalation Manager initialized | enabled=%s default_target=%s emit_to_memory=%s",
            self.enabled,
            self.default_target_agent,
            self.emit_to_memory,
        )

    def attach_memory(self, memory: Any) -> None:
        """Attach a HandlerMemory-like object after construction."""
        self.memory = memory

    def should_escalate(
        self,
        normalized_failure: Mapping[str, Any],
        recovery_result: Optional[Mapping[str, Any]] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        insight: Optional[Mapping[str, Any]] = None,
        sla: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Return whether this failure/recovery state should leave local handler-only handling."""
        decision = self.decide(
            normalized_failure=normalized_failure,
            recovery_result=recovery_result,
            context=context,
            insight=insight,
            sla=sla,
        )
        return decision.should_escalate

    def decide(
        self,
        *,
        normalized_failure: Mapping[str, Any],
        recovery_result: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        insight: Optional[Mapping[str, Any]] = None,
        sla: Optional[Mapping[str, Any]] = None,
        strategy_distribution: Optional[Mapping[str, Any]] = None,
    ) -> EscalationDecision:
        """Build an explainable escalation routing decision."""
        try:
            if not isinstance(normalized_failure, Mapping):
                raise ValidationError(
                    "EscalationManager expected normalized_failure to be a mapping",
                    context={"actual_type": type(normalized_failure).__name__},
                    code="HANDLER_ESCALATION_FAILURE_MAPPING_REQUIRED",
                    policy=self.error_policy,
                )

            context_payload = coerce_mapping(context)
            failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
            recovery = normalize_recovery_result(recovery_result or {})
            insight_payload = coerce_mapping(insight)
            sla_payload = coerce_mapping(sla if sla is not None else recovery.get("sla"))
            distribution = summarize_strategy_distribution(strategy_distribution or recovery.get("strategy_distribution"))

            severity = normalize_severity(failure.get("severity"))
            retryable = coerce_bool(failure.get("retryable"), default=False)
            category = normalize_identifier(failure.get("category") or classify_failure_category(failure.get("type"), failure.get("message")), default="runtime")
            action = normalize_recovery_action(
                failure.get("policy_action")
                or failure.get("action")
                or insight_payload.get("action")
                or self._action_from_recovery(recovery)
            )
            recommendation = self._resolve_reason(failure=failure, recovery=recovery, insight=insight_payload)
            recovery_status = str(recovery.get("status") or "unknown").lower()
            low_sla_budget = self._is_low_sla_budget(context=context_payload, sla=sla_payload)

            if not self.enabled:
                target = self.local_handler_agent
                routed_by = "manager_disabled"
                matrix_key = "disabled"
            else:
                target, routed_by, matrix_key = self._resolve_target(
                    severity=severity,
                    retryable=retryable,
                    action=action,
                    category=category,
                    recommendation=recommendation,
                    recovery_status=recovery_status,
                    low_sla_budget=low_sla_budget,
                )

            priority = self._priority_for(severity=severity, action=action, low_sla_budget=low_sla_budget, recovery_status=recovery_status)
            escalation_level = self.level_by_priority.get(priority, "standard_handoff")
            require_ack = self.ack_by_priority.get(priority, False)
            confidence = self._confidence(
                severity=severity,
                action=action,
                target=target,
                routed_by=routed_by,
                insight=insight_payload,
                strategy_distribution=distribution,
            )
            labels = self._labels(
                severity=severity,
                retryable=retryable,
                category=category,
                action=action,
                recovery_status=recovery_status,
                low_sla_budget=low_sla_budget,
                routed_by=routed_by,
            )
            evidence = self._evidence(
                failure=failure,
                recovery=recovery,
                insight=insight_payload,
                context=context_payload,
                sla=sla_payload,
                strategy_distribution=distribution,
                routed_by=routed_by,
                matrix_key=matrix_key,
                low_sla_budget=low_sla_budget,
            )

            return EscalationDecision(
                target_agent=target,
                reason=recommendation,
                severity=severity,
                retryable=retryable,
                action=action,
                priority=priority,
                escalation_level=escalation_level,
                require_ack=require_ack,
                matrix_key=matrix_key,
                routed_by=routed_by,
                confidence=confidence,
                deadline_seconds=self._deadline_seconds(context=context_payload, sla=sla_payload),
                correlation_id=failure.get("correlation_id") or context_payload.get("correlation_id"),
                labels=tuple(labels),
                evidence=tuple(evidence),
                metadata=compact_dict(
                    {
                        "failure_type": failure.get("type"),
                        "category": category,
                        "context_hash": failure.get("context_hash"),
                        "recovery_status": recovery_status,
                        "strategy": recovery.get("strategy"),
                        "task_id": context_payload.get("task_id"),
                        "route": context_payload.get("route"),
                        "agent": context_payload.get("agent"),
                    },
                    drop_none=True,
                    drop_empty=True,
                ),
            )
        except HandlerError:
            raise
        except Exception as exc:
            raise EscalationError(
                "EscalationManager failed to build escalation decision",
                cause=exc,
                context={"has_recovery_result": recovery_result is not None, "has_insight": insight is not None},
                code="HANDLER_ESCALATION_DECISION_FAILED",
                policy=self.error_policy,
            ) from exc

    def build_handoff_payload(
        self,
        normalized_failure: Mapping[str, Any],
        recovery_result: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
        strategy_distribution: Optional[Mapping[str, Any]] = None,
        sla: Optional[Mapping[str, Any]] = None,
        insight: Optional[Mapping[str, Any]] = None,
        *,
        emit: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Build a typed escalation handoff payload.

        This method intentionally preserves the original HandlerAgent-facing method name
        and core fields while adding v2 decision, audit, policy, and routing metadata.
        """
        try:
            context_payload = coerce_mapping(context)
            failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
            recovery = normalize_recovery_result(recovery_result or {})
            sla_payload = coerce_mapping(sla if sla is not None else recovery.get("sla"))
            distribution = summarize_strategy_distribution(strategy_distribution or recovery.get("strategy_distribution"))
            insight_payload = coerce_mapping(insight)

            decision = self.decide(
                normalized_failure=failure,
                recovery_result=recovery,
                context=context_payload,
                insight=insight_payload,
                sla=sla_payload,
                strategy_distribution=distribution,
            )
            safe_context = self._safe_context(context_payload)
            public_error = self._public_error(failure) if self.include_public_error else {}

            payload = compact_dict(
                {
                    "schema": self.schema_version,
                    "handoff_type": self.handoff_type,
                    "timestamp": utc_timestamp(),
                    "correlation_id": decision.correlation_id or context_payload.get("correlation_id") or generate_correlation_id("handler-escalation"),
                    "target_agent": decision.target_agent,
                    "reason": decision.reason,
                    "priority": decision.priority,
                    "escalation_level": decision.escalation_level,
                    "require_ack": decision.require_ack,
                    "decision": decision.to_dict(),
                    "failure": {
                        "type": failure.get("type"),
                        "message": truncate_text(failure.get("message"), self.max_message_chars),
                        "severity": decision.severity,
                        "retryable": decision.retryable,
                        "category": failure.get("category"),
                        "context_hash": failure.get("context_hash"),
                        "policy_action": failure.get("policy_action"),
                        "code": failure.get("code"),
                        "source": failure.get("source"),
                    },
                    "public_error": public_error,
                    "recovery": {
                        "status": recovery.get("status"),
                        "strategy": recovery.get("strategy"),
                        "attempts": recovery.get("attempts", 0),
                        "max_retries": recovery.get("max_retries"),
                        "checkpoint_id": recovery.get("checkpoint_id"),
                        "checkpoint_restored": recovery.get("checkpoint_restored"),
                        "recommendation": recovery.get("recommendation"),
                    },
                    "insight": self._compact_insight(insight_payload),
                    "strategy_distribution": distribution,
                    "sla": make_json_safe(self.error_policy.sanitize_context(sla_payload)),
                    "context": safe_context,
                    "audit": self._audit_payload(
                        failure=failure,
                        recovery=recovery,
                        context=safe_context,
                        decision=decision,
                        insight=insight_payload,
                    ),
                },
                drop_none=True,
                drop_empty=True,
            )
            self._enforce_payload_size(payload)
            if self._should_emit(emit):
                self.emit_escalation_event(payload)
            return payload
        except HandlerError:
            raise
        except Exception as exc:
            raise EscalationError(
                "EscalationManager failed to build handoff payload",
                cause=exc,
                context={"normalized_failure_type": type(normalized_failure).__name__},
                code="HANDLER_ESCALATION_HANDOFF_BUILD_FAILED",
                policy=self.error_policy,
            ) from exc

    def emit_escalation_event(self, handoff_payload: Mapping[str, Any]) -> bool:
        """Append an escalation telemetry event to an attached HandlerMemory-like object when available."""
        if self.memory is None:
            return False
        event = {
            "event_type": self.memory_event_type,
            "timestamp": handoff_payload.get("timestamp", utc_timestamp()),
            "correlation_id": handoff_payload.get("correlation_id"),
            "failure": handoff_payload.get("failure", {}),
            "recovery": handoff_payload.get("recovery", {}),
            "context": handoff_payload.get("context", {}),
            "decision": handoff_payload.get("decision", {}),
            "target_agent": handoff_payload.get("target_agent"),
            "reason": handoff_payload.get("reason"),
            "priority": handoff_payload.get("priority"),
        }
        if hasattr(self.memory, "append_telemetry") and callable(self.memory.append_telemetry):
            self.memory.append_telemetry(event)
            return True
        return False

    def summarize_escalations(self, events: Optional[Iterable[Mapping[str, Any]]] = None, *, limit: int = 500) -> Dict[str, Any]:
        """Summarize escalation telemetry for diagnostics and test assertions."""
        if events is None and self.memory is not None and hasattr(self.memory, "recent_telemetry"):
            events = self.memory.recent_telemetry(limit=limit, event_type=self.memory_event_type)
        stream = [dict(event) for event in coerce_list(events) if isinstance(event, Mapping)]
        target_counts: Counter[str] = Counter()
        priority_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        for event in stream:
            target_counts[str(event.get("target_agent") or deep_get(event, "decision.target_agent", "unknown"))] += 1
            priority_counts[str(event.get("priority") or deep_get(event, "decision.priority", "unknown"))] += 1
            reason_counts[str(event.get("reason") or deep_get(event, "decision.reason", "unknown"))] += 1
            severity_counts[str(deep_get(event, "failure.severity", "unknown"))] += 1
        return {
            "schema": "handler.escalation.summary.v2",
            "timestamp": utc_timestamp(),
            "total_events": len(stream),
            "target_counts": dict(target_counts),
            "priority_counts": dict(priority_counts),
            "reason_counts": dict(reason_counts),
            "severity_counts": dict(severity_counts),
        }

    def _resolve_target(
        self,
        *,
        severity: str,
        retryable: bool,
        action: str,
        category: str,
        recommendation: str,
        recovery_status: str,
        low_sla_budget: bool,
    ) -> Tuple[str, str, str]:
        if recommendation in self.target_overrides:
            return self.target_overrides[recommendation], "target_override", f"override:{recommendation}"
        if action in {HandlerRecoveryAction.QUARANTINE.value, HandlerRecoveryAction.FAIL_FAST.value}:
            return self.action_routes.get(action, self.fallback_target_agent), "action_route", f"action:{action}"
        if recommendation in self.recommendation_routes:
            return self.recommendation_routes[recommendation], "recommendation_route", f"recommendation:{recommendation}"
        if category in self.category_routes and recovery_status in self.recovery_failed_statuses:
            return self.category_routes[category], "category_route", f"category:{category}"
        if low_sla_budget and recovery_status in self.recovery_failed_statuses:
            return self.fallback_target_agent, "sla_pressure", "sla:low_budget"
        retry_key = "retryable" if retryable else "non_retryable"
        target = coerce_mapping(self.matrix.get(severity)).get(retry_key)
        if target:
            return normalize_identifier(target, default=self.fallback_target_agent), "severity_retryability_matrix", f"{severity}:{retry_key}"
        return self.fallback_target_agent, "fallback", "fallback"

    def _priority_for(self, *, severity: str, action: str, low_sla_budget: bool, recovery_status: str) -> str:
        priority = str(self.priority_by_severity.get(severity, "p2")).lower()
        if action == HandlerRecoveryAction.QUARANTINE.value:
            return "p0"
        if low_sla_budget and recovery_status in self.recovery_failed_statuses:
            return self._higher_priority(priority, self.sla_priority_floor)
        if self.escalate_degraded_high_severity and recovery_status in self.degraded_statuses and severity in {FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value}:
            return self._higher_priority(priority, "p1")
        return priority

    @staticmethod
    def _higher_priority(current: str, floor: str) -> str:
        order = {"p0": 0, "p1": 1, "p2": 2, "p3": 3, "p4": 4}
        current_value = order.get(str(current).lower(), 2)
        floor_value = order.get(str(floor).lower(), 2)
        return current if current_value <= floor_value else floor

    def _resolve_reason(self, *, failure: Mapping[str, Any], recovery: Mapping[str, Any], insight: Mapping[str, Any]) -> str:
        reason = (
            recovery.get("recommendation")
            or insight.get("recommendation")
            or failure.get("recommendation")
            or self.default_reason
        )
        return truncate_text(self.error_policy.sanitize_message(reason), self.max_reason_chars)

    @staticmethod
    def _action_from_recovery(recovery: Mapping[str, Any]) -> str:
        recommendation = str(recovery.get("recommendation") or "").lower()
        if "quarantine" in recommendation:
            return HandlerRecoveryAction.QUARANTINE.value
        if "escalat" in recommendation:
            return HandlerRecoveryAction.ESCALATE.value
        if "degrad" in recommendation or "failover" in recommendation:
            return HandlerRecoveryAction.DEGRADE.value
        if "retry" in recommendation:
            return HandlerRecoveryAction.RETRY.value
        return HandlerRecoveryAction.NONE.value

    def _is_low_sla_budget(self, *, context: Mapping[str, Any], sla: Mapping[str, Any]) -> bool:
        if sla and "remaining_seconds" in sla:
            remaining = coerce_float(sla.get("remaining_seconds"), self.low_sla_budget_seconds + 1.0, minimum=0.0)
        else:
            merged_context = dict(context or {})
            if sla:
                merged_context["sla"] = dict(sla)
            remaining = compute_remaining_budget(context=merged_context, default_seconds=self.low_sla_budget_seconds + 1.0)
        return remaining <= self.low_sla_budget_seconds

    @staticmethod
    def _deadline_seconds(*, context: Mapping[str, Any], sla: Mapping[str, Any]) -> Optional[float]:
        if sla and "remaining_seconds" in sla:
            return coerce_float(sla.get("remaining_seconds"), 0.0, minimum=0.0)
        merged_context = dict(context or {})
        if sla:
            merged_context["sla"] = dict(sla)
        remaining = compute_remaining_budget(context=merged_context, default_seconds=-1.0)
        return remaining if remaining >= 0.0 else None

    def _confidence(self, *, severity: str, action: str, target: str, routed_by: str, insight: Mapping[str, Any], strategy_distribution: Mapping[str, float]) -> float:
        value = 0.55
        if routed_by in {"action_route", "recommendation_route", "target_override"}:
            value += 0.18
        if severity in {FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value}:
            value += 0.08
        if action in {HandlerRecoveryAction.FAIL_FAST.value, HandlerRecoveryAction.QUARANTINE.value, HandlerRecoveryAction.ESCALATE.value}:
            value += 0.08
        if insight.get("confidence") is not None:
            value = (value + coerce_float(insight.get("confidence"), value, minimum=0.0, maximum=1.0)) / 2.0
        if strategy_distribution:
            value += min(0.05, max(strategy_distribution.values()) * 0.05)
        if target == self.local_handler_agent:
            value -= 0.05
        return round(coerce_float(value, 0.55, minimum=0.0, maximum=0.99), 4)

    def _labels(
        self,
        *,
        severity: str,
        retryable: bool,
        category: str,
        action: str,
        recovery_status: str,
        low_sla_budget: bool,
        routed_by: str,
    ) -> List[str]:
        labels = [f"severity:{severity}", f"category:{category}", f"action:{action}", f"status:{recovery_status}", f"route:{routed_by}"]
        labels.append("retryable" if retryable else "non_retryable")
        if low_sla_budget:
            labels.append("sla:low_budget")
        return labels

    def _evidence(
        self,
        *,
        failure: Mapping[str, Any],
        recovery: Mapping[str, Any],
        insight: Mapping[str, Any],
        context: Mapping[str, Any],
        sla: Mapping[str, Any],
        strategy_distribution: Mapping[str, Any],
        routed_by: str,
        matrix_key: str,
        low_sla_budget: bool,
    ) -> List[Dict[str, Any]]:
        if not self.include_internal_evidence or self.max_evidence_items <= 0:
            return []
        evidence = [
            {"signal": "routing", "routed_by": routed_by, "matrix_key": matrix_key},
            {"signal": "failure", "type": failure.get("type"), "severity": failure.get("severity"), "category": failure.get("category")},
            {"signal": "recovery", "status": recovery.get("status"), "strategy": recovery.get("strategy"), "attempts": recovery.get("attempts")},
        ]
        if insight:
            evidence.append({"signal": "insight", "signature": insight.get("signature"), "recommendation": insight.get("recommendation"), "risk_score": insight.get("risk_score")})
        if strategy_distribution:
            evidence.append({"signal": "strategy_distribution", "distribution": strategy_distribution})
        if sla:
            evidence.append({"signal": "sla", "remaining_seconds": sla.get("remaining_seconds"), "mode": sla.get("mode"), "low_budget": low_sla_budget})
        if context:
            evidence.append({"signal": "context", "context": self._safe_context(context)})
        return [make_json_safe(self.error_policy.sanitize_context(item)) for item in evidence[: self.max_evidence_items]]  # type: ignore[list-item]

    def _safe_context(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        preferred = build_escalation_context(context)
        if len(preferred) < self.max_context_fields:
            for key, value in context.items():
                if key in preferred:
                    continue
                preferred[str(key)] = value
                if len(preferred) >= self.max_context_fields:
                    break
        sanitized = make_json_safe(self.error_policy.sanitize_context(preferred))
        return sanitized if isinstance(sanitized, dict) else {}

    def _public_error(self, failure: Mapping[str, Any]) -> Dict[str, Any]:
        handler_error = HandlerError.from_mapping(
            {
                "message": failure.get("message"),
                "error_type": failure.get("type"),
                "severity": failure.get("severity"),
                "retryable": failure.get("retryable"),
                "context": {"context_hash": failure.get("context_hash")},
                "action": failure.get("policy_action"),
                "code": failure.get("code"),
                "source": failure.get("source"),
                "correlation_id": failure.get("correlation_id"),
            },
            policy=self.error_policy,
        )
        return handler_error.to_public_dict()

    @staticmethod
    def _compact_insight(insight: Mapping[str, Any]) -> Dict[str, Any]:
        return compact_dict(
            select_keys(
                insight,
                (
                    "signature",
                    "confidence",
                    "category",
                    "recommendation",
                    "severity",
                    "retryable",
                    "action",
                    "risk_score",
                    "anomaly_score",
                    "recurrence_count",
                    "historical_success_rate",
                ),
                include_missing=False,
            ),
            drop_none=True,
            drop_empty=True,
        )

    def _audit_payload(
        self,
        *,
        failure: Mapping[str, Any],
        recovery: Mapping[str, Any],
        context: Mapping[str, Any],
        decision: EscalationDecision,
        insight: Mapping[str, Any],
    ) -> Dict[str, Any]:
        audit = {
            "schema": "handler.escalation_audit.v2",
            "audit_id": stable_hash(
                {
                    "failure": failure.get("context_hash"),
                    "target": decision.target_agent,
                    "reason": decision.reason,
                    "timestamp": int(decision.timestamp * 1000),
                },
                length=16,
                policy=self.error_policy,
            ),
            "created_at": utc_iso_timestamp(),
            "policy": self.error_policy.name,
            "target_agent": decision.target_agent,
            "priority": decision.priority,
            "require_ack": decision.require_ack,
            "routed_by": decision.routed_by,
            "matrix_key": decision.matrix_key,
            "failure_context_hash": failure.get("context_hash"),
            "recovery_status": recovery.get("status"),
            "insight_signature": insight.get("signature"),
            "task_id": context.get("task_id"),
            "route": context.get("route"),
            "agent": context.get("agent"),
        }
        return compact_dict(make_json_safe(self.error_policy.sanitize_context(audit)), drop_none=True, drop_empty=True)  # type: ignore[arg-type]

    def _should_emit(self, emit: Optional[bool]) -> bool:
        return self.emit_to_memory if emit is None else bool(emit)

    def _enforce_payload_size(self, payload: Mapping[str, Any]) -> None:
        serialized_length = len(stable_json_dumps(payload))
        if serialized_length > self.max_payload_chars:
            raise SerializationError(
                "Escalation handoff payload exceeds configured size limit",
                context={"serialized_chars": serialized_length, "max_payload_chars": self.max_payload_chars},
                code="HANDLER_ESCALATION_PAYLOAD_TOO_LARGE",
                policy=self.error_policy,
            )

    def _load_matrix(self, value: Any) -> Dict[str, Dict[str, str]]:
        matrix: Dict[str, Dict[str, str]] = {}
        source = value if isinstance(value, Mapping) else self.DEFAULT_MATRIX
        for severity, row in source.items():
            normalized_severity = normalize_severity(severity)
            row_map = coerce_mapping(row)
            retryable_target = normalize_identifier(row_map.get("retryable"), default=self.local_handler_agent)
            non_retryable_target = normalize_identifier(row_map.get("non_retryable"), default=self.fallback_target_agent)
            matrix[normalized_severity] = {"retryable": retryable_target, "non_retryable": non_retryable_target}
        for severity, row in self.DEFAULT_MATRIX.items():
            matrix.setdefault(severity, dict(row))
        return matrix

    def _load_route_mapping(self, configured: Any, default: Mapping[str, str], *, normalize_actions: bool = False) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for key, target in default.items():
            route_key = normalize_recovery_action(key) if normalize_actions else normalize_identifier(key, default=str(key))
            values[route_key] = normalize_identifier(target, default=self.fallback_target_agent)
        if isinstance(configured, Mapping):
            for key, target in configured.items():
                route_key = normalize_recovery_action(key) if normalize_actions else normalize_identifier(key, default=str(key))
                values[route_key] = normalize_identifier(target, default=self.fallback_target_agent)
        return values

    @staticmethod
    def _load_simple_mapping(configured: Any, default: Mapping[str, str]) -> Dict[str, str]:
        values = {str(key).lower(): str(value).lower() for key, value in default.items()}
        if isinstance(configured, Mapping):
            for key, value in configured.items():
                values[str(key).lower()] = str(value).lower()
        return values

    @staticmethod
    def _load_bool_mapping(configured: Any, default: Mapping[str, bool]) -> Dict[str, bool]:
        values = {str(key).lower(): bool(value) for key, value in default.items()}
        if isinstance(configured, Mapping):
            for key, value in configured.items():
                values[str(key).lower()] = coerce_bool(value)
        return values

    def _validate_configuration(self) -> None:
        if not self.default_target_agent:
            raise ConfigurationError(
                "EscalationManager default_target_agent is required",
                code="HANDLER_ESCALATION_DEFAULT_TARGET_REQUIRED",
                policy=self.error_policy,
            )
        for severity in (FailureSeverity.LOW.value, FailureSeverity.MEDIUM.value, FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value):
            if severity not in self.matrix:
                raise ConfigurationError(
                    "EscalationManager escalation_matrix missing severity row",
                    context={"missing_severity": severity},
                    code="HANDLER_ESCALATION_MATRIX_SEVERITY_MISSING",
                    policy=self.error_policy,
                )
            row = self.matrix[severity]
            if "retryable" not in row or "non_retryable" not in row:
                raise ConfigurationError(
                    "EscalationManager escalation_matrix rows require retryable and non_retryable targets",
                    context={"severity": severity, "row": row},
                    code="HANDLER_ESCALATION_MATRIX_ROW_INVALID",
                    policy=self.error_policy,
                )


if __name__ == "__main__":
    print("\n=== Running Escalation Manager ===\n")
    printer.status("TEST", "Escalation Manager initialized", "info")

    from .handler_memory import HandlerMemory

    strict_policy = HandlerErrorPolicy(
        name="escalation_manager.strict_test",
        expose_internal_messages=False,
        include_context_in_public=False,
        include_context_in_telemetry=True,
        max_message_chars=240,
        max_string_chars=160,
    )

    memory = HandlerMemory(error_policy=strict_policy)
    manager = EscalationManager(
        config={
            "emit_to_memory": True,
            "low_sla_budget_seconds": 3.0,
            "max_payload_chars": 120_000,
        },
        memory=memory,
        error_policy=strict_policy,
    )

    context = {
        "task_id": "handler-escalation-smoke-001",
        "route": "handler.recovery",
        "agent": "demo_agent",
        "correlation_id": "corr-handler-escalation-test",
        "password": "SuperSecret123",
    }

    failure = build_normalized_failure(
        error=TimeoutError("Upstream timed out with Authorization: Bearer token-123"),
        context=context,
        policy=strict_policy,
        source="handler.escalation.__main__",
        correlation_id="corr-handler-escalation-test",
    )
    recovery = {
        "status": "failed",
        "strategy": "timeout",
        "attempts": 2,
        "max_retries": 2,
        "checkpoint_id": "handler:checkpoint:test",
        "recommendation": "open_circuit_or_failover",
        "sla": {"remaining_seconds": 2.0, "mode": "fast_failover"},
        "strategy_distribution": {"timeout": 0.75, "runtime": 0.25},
    }
    insight = {
        "signature": "timeout:abc123",
        "confidence": 0.81,
        "category": "timeout",
        "recommendation": "open_circuit_or_failover",
        "action": "escalate",
        "risk_score": 0.76,
        "historical_success_rate": 0.1,
    }

    decision = manager.decide(
        normalized_failure=failure,
        recovery_result=recovery,
        context=context,
        insight=insight,
        sla=recovery["sla"],
        strategy_distribution=recovery["strategy_distribution"],
    )
    handoff = manager.build_handoff_payload(
        normalized_failure=failure,
        recovery_result=recovery,
        context=context,
        strategy_distribution=recovery["strategy_distribution"],
        sla=recovery["sla"],
        insight=insight,
    )

    security_failure = build_normalized_failure(
        error=PermissionError("Unauthorized token=abc123 with password=SuperSecret123"),
        error_info={"error_type": HandlerErrorType.SECURITY.value, "severity": FailureSeverity.CRITICAL.value, "retryable": False},
        context=context,
        policy=strict_policy,
        source="handler.escalation.__main__",
        correlation_id="corr-handler-escalation-security-test",
    )
    security_handoff = manager.build_handoff_payload(
        normalized_failure=security_failure,
        recovery_result={"status": "failed", "strategy": "security", "attempts": 0, "recommendation": "quarantine_security_failure"},
        context=context,
        insight={"signature": "security:def456", "recommendation": "quarantine_security_failure", "action": "quarantine", "confidence": 0.93},
    )

    summary = manager.summarize_escalations(limit=10)
    serialized = stable_json_dumps({"decision": decision.to_dict(), "handoff": handoff, "security_handoff": security_handoff, "summary": summary})

    assert decision.target_agent == "planning_agent"
    assert decision.priority in {"p0", "p1"}
    assert handoff["target_agent"] == "planning_agent"
    assert handoff["decision"]["should_escalate"] is True
    assert security_handoff["target_agent"] == "safety_agent"
    assert security_handoff["priority"] == "p0"
    assert summary["total_events"] >= 2
    assert "SuperSecret123" not in serialized
    assert "token-123" not in serialized

    printer.pretty("Escalation decision", decision.to_dict(), "success")
    printer.pretty("Escalation handoff", handoff, "success")
    printer.pretty("Escalation summary", summary, "success")
    print("\n=== Test ran successfully ===\n")
