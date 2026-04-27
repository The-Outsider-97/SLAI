"""
Production-grade policy engine for the collaborative agent subsystem.

This module centralizes policy evaluation for SLAI's collaborative runtime. It
keeps orchestration, routing, task contracts, registry discovery, and reliability
state transitions in their own modules while providing a stable policy layer for
allow/deny/review decisions.

Design goals
------------
1. Preserve the existing public API: ``PolicyDecision``, ``PolicyRule``,
   ``PolicyEvaluation``, ``PolicyEngine.add_rule``, ``add_simple_rule``,
   ``evaluate``, and ``list_rules`` remain available and compatible.
2. Support both programmatic predicates and config-backed declarative rules.
3. Use collaborative helpers for normalization, serialization, redaction,
   audit payloads, result payloads, ids, timestamps, and defensive diagnostics.
4. Use collaboration errors for policy-engine boundary failures instead of
   raising unstructured exceptions.
5. Keep rule evaluation deterministic, thread-safe, inspectable, and suitable
   for production telemetry.
"""

from __future__ import annotations

import fnmatch
import re
import threading

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.collaboration_error import *
from .utils.collaborative_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Policy Engine")
printer = PrettyPrinter()


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_REVIEW = "require_review"


PolicyPredicate = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], bool]
PolicyDecisionInput = Union[PolicyDecision, str]


_DEFAULT_POLICY_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "default_decision": PolicyDecision.ALLOW.value,
    "fail_closed": False,
    "allow_duplicate_rule_ids": False,
    "stop_on_first_deny": False,
    "stop_on_first_review": False,
    "max_rules": 1000,
    "audit_enabled": True,
    "audit_key": "collaboration:policy_events",
    "audit_max_events": 1000,
    "include_rule_metadata": True,
    "include_unmatched_rules": False,
    "redact_evaluation_payloads": True,
    "rule_error_effect": PolicyDecision.REQUIRE_REVIEW.value,
    "configured_rules": [],
}

_ALLOWED_CONDITION_OPERATORS = {"exists", "missing", "eq", "equals", "ne", "not_equals", "falsy",
                                "gt", "gte", "lt", "lte", "in", "not_in", "contains", "glob",
                                "not_contains", "intersects", "includes_all", "regex", "truthy",
                                }

_MISSING = object()


@dataclass(frozen=True)
class PolicyCondition:
    """Declarative condition for config-backed or code-built policy rules.

    ``path`` is resolved against one of three scopes: ``task``, ``agent`` or
    ``context``. A path can also include the scope prefix directly, for example
    ``task.risk_score`` or ``agent.capabilities``.
    """

    path: str
    operator: str = "exists"
    value: Any = None
    values: Tuple[Any, ...] = ()
    source: str = "task"
    case_sensitive: bool = True
    default: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PolicyCondition":
        data = ensure_mapping(payload, field_name="policy_condition")
        raw_values = data.get("values", ())
        if raw_values is None:
            normalized_values: Tuple[Any, ...] = ()
        elif isinstance(raw_values, (str, bytes)):
            normalized_values = (raw_values,)
        elif isinstance(raw_values, Iterable):
            normalized_values = tuple(raw_values)
        else:
            normalized_values = (raw_values,)

        return cls(
            path=require_non_empty_string(data.get("path"), "condition.path"),
            operator=normalize_condition_operator(data.get("operator", data.get("op", "exists"))),
            value=data.get("value"),
            values=normalized_values,
            source=normalize_condition_source(data.get("source", "task")),
            case_sensitive=coerce_bool(data.get("case_sensitive"), default=True),
            default=data.get("default"),
            metadata=normalize_metadata(data.get("metadata"), drop_none=True),
        )

    def to_dict(self) -> Dict[str, Any]:
        return redact_mapping(prune_none(asdict(self)))


@dataclass(frozen=True)
class PolicyRuleEvaluation:
    """Detailed per-rule evaluation record."""

    rule_id: str
    matched: bool
    effect: PolicyDecision
    description: str
    priority: int
    enabled: bool
    duration_ms: float = 0.0
    reason: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = {
            "rule_id": self.rule_id,
            "matched": self.matched,
            "effect": self.effect.value,
            "description": self.description,
            "priority": self.priority,
            "enabled": self.enabled,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
            "error": self.error,
            "metadata": json_safe(self.metadata),
        }
        payload = prune_none(payload, drop_empty=True)
        return redact_mapping(payload) if redact else payload


@dataclass
class PolicyRule:
    rule_id: str
    description: str
    effect: PolicyDecision
    priority: int = 100
    enabled: bool = True
    predicate: Optional[PolicyPredicate] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    conditions: Tuple[PolicyCondition, ...] = ()
    condition_match: str = "all"
    tags: Tuple[str, ...] = ()
    owner: Optional[str] = None
    created_at: float = field(default_factory=epoch_seconds)
    updated_at: Optional[float] = None
    expires_at: Optional[float] = None
    stop_processing: bool = False
    audit: bool = True

    def __post_init__(self) -> None:
        self.rule_id = normalize_rule_id(self.rule_id)
        self.description = require_non_empty_string(self.description, "description")
        self.effect = normalize_policy_decision(self.effect)
        self.priority = coerce_int(self.priority, default=100)
        self.enabled = coerce_bool(self.enabled, default=True)
        self.metadata = normalize_metadata(self.metadata, drop_none=True)
        self.conditions = tuple(
            item if isinstance(item, PolicyCondition) else PolicyCondition.from_mapping(item)
            for item in (self.conditions or ())
        )
        self.condition_match = normalize_condition_match(self.condition_match)
        self.tags = normalize_tags(self.tags)
        self.owner = str(self.owner).strip() if self.owner is not None and str(self.owner).strip() else None
        self.created_at = coerce_float(self.created_at, default=epoch_seconds(), minimum=0.0)
        self.updated_at = coerce_float(self.updated_at, default=0.0, minimum=0.0) or None
        self.expires_at = coerce_float(self.expires_at, default=0.0, minimum=0.0) or None
        self.stop_processing = coerce_bool(self.stop_processing, default=False)
        self.audit = coerce_bool(self.audit, default=True)

        if self.predicate is None and not self.conditions:
            raise _policy_error(
                "PolicyRule requires either a predicate or at least one condition.",
                context={"rule_id": self.rule_id},
                severity="medium",
            )

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and epoch_seconds() >= self.expires_at

    def evaluate(self, task: Dict[str, Any], agent_meta: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not self.enabled or self.expired:
            return False
        predicate_match = True
        if self.predicate is not None:
            predicate_match = bool(self.predicate(task, agent_meta, context))
        condition_match = True
        if self.conditions:
            condition_match = evaluate_conditions(
                self.conditions,
                task=task,
                agent_meta=agent_meta,
                context=context,
                match=self.condition_match,
            )
        return bool(predicate_match and condition_match)

    def evaluate_detailed(self, task: Dict[str, Any], agent_meta: Dict[str, Any], context: Dict[str, Any]) -> PolicyRuleEvaluation:
        start_ms = monotonic_ms()
        matched = self.evaluate(task, agent_meta, context)
        return PolicyRuleEvaluation(
            rule_id=self.rule_id,
            matched=matched,
            effect=self.effect,
            description=self.description,
            priority=self.priority,
            enabled=self.enabled,
            duration_ms=elapsed_ms(start_ms),
            reason=self.description if matched else None,
            metadata=self.metadata,
        )

    def to_dict(self, *, include_predicate: bool = False, redact: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "rule_id": self.rule_id,
            "description": self.description,
            "effect": self.effect.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "condition_match": self.condition_match,
            "tags": list(self.tags),
            "owner": self.owner,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "expired": self.expired,
            "stop_processing": self.stop_processing,
            "audit": self.audit,
        }
        if include_predicate:
            payload["predicate"] = describe_predicate(self.predicate)
        payload = prune_none(payload, drop_empty=True)
        return redact_mapping(payload) if redact else payload


@dataclass
class PolicyEvaluation:
    decision: PolicyDecision
    reasons: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    evaluated_rules: List[Dict[str, Any]] = field(default_factory=list)
    denied_rules: List[str] = field(default_factory=list)
    review_rules: List[str] = field(default_factory=list)
    allow_rules: List[str] = field(default_factory=list)
    skipped_rules: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("policy"))
    evaluated_at: float = field(default_factory=epoch_seconds)
    evaluated_at_utc: str = field(default_factory=utc_timestamp)
    duration_ms: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.decision == PolicyDecision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == PolicyDecision.DENY

    @property
    def requires_review(self) -> bool:
        return self.decision == PolicyDecision.REQUIRE_REVIEW

    def to_dict(self) -> Dict[str, Any]:
        return redact_mapping(
            prune_none(
                {
                    "decision": self.decision.value,
                    "reasons": list(self.reasons),
                    "matched_rules": list(self.matched_rules),
                    "evaluated_rules": list(self.evaluated_rules),
                    "denied_rules": list(self.denied_rules),
                    "review_rules": list(self.review_rules),
                    "allow_rules": list(self.allow_rules),
                    "skipped_rules": list(self.skipped_rules),
                    "errors": list(self.errors),
                    "warnings": list(self.warnings),
                    "metadata": json_safe(self.metadata),
                    "correlation_id": self.correlation_id,
                    "evaluated_at": self.evaluated_at,
                    "evaluated_at_utc": self.evaluated_at_utc,
                    "duration_ms": self.duration_ms,
                    "allowed": self.allowed,
                    "denied": self.denied,
                    "requires_review": self.requires_review,
                },
                drop_empty=True,
            )
        )

    def to_result(self, *, action: str = "policy_evaluation") -> Dict[str, Any]:
        if self.decision == PolicyDecision.ALLOW:
            return success_result(action=action, message="Policy allowed task", data=self.to_dict(), correlation_id=self.correlation_id)
        if self.decision == PolicyDecision.REQUIRE_REVIEW:
            return review_result(action=action, message="Policy requires review", reasons=self.reasons, data=self.to_dict(), correlation_id=self.correlation_id)
        return error_result(
            action=action,
            message="Policy denied task",
            error={"decision": self.decision.value, "reasons": self.reasons, "matched_rules": self.matched_rules},
            data=self.to_dict(),
            correlation_id=self.correlation_id,
        )


class PolicyEngine:
    """Thread-safe policy registry and evaluator for collaborative tasks."""

    def __init__(self, *, shared_memory: Any = None, rules: Optional[Iterable[PolicyRule]] = None,
                 load_config_rules: bool = True):
        self._rules: List[PolicyRule] = []
        self._rule_index: "OrderedDict[str, PolicyRule]" = OrderedDict()
        self._lock = threading.RLock()
        self.shared_memory = shared_memory
        self.config = load_global_config()
        self.policy_config = get_config_section("policy_engine") or {}

        self.enabled = coerce_bool(self.policy_config.get("enabled"), default=True)
        self.default_decision = normalize_policy_decision(self.policy_config.get("default_decision", PolicyDecision.ALLOW.value))
        self.fail_closed = coerce_bool(self.policy_config.get("fail_closed"), default=False)
        self.allow_duplicate_rule_ids = coerce_bool(self.policy_config.get("allow_duplicate_rule_ids"), default=False)
        self.stop_on_first_deny = coerce_bool(self.policy_config.get("stop_on_first_deny"), default=False)
        self.stop_on_first_review = coerce_bool(self.policy_config.get("stop_on_first_review"), default=False)
        self.max_rules = coerce_int(self.policy_config.get("max_rules"), default=1000, minimum=1)
        self.audit_enabled = coerce_bool(self.policy_config.get("audit_enabled"), default=True)
        self.audit_key = str(self.policy_config.get("audit_key") or "collaboration:policy_events")
        self.audit_max_events = coerce_int(self.policy_config.get("audit_max_events"), default=1000, minimum=1)
        self.include_rule_metadata = coerce_bool(self.policy_config.get("include_rule_metadata"), default=True)
        self.include_unmatched_rules = coerce_bool(self.policy_config.get("include_unmatched_rules"), default=False)
        self.redact_evaluation_payloads = coerce_bool(self.policy_config.get("redact_evaluation_payloads"), default=True)
        self.rule_error_effect = normalize_policy_decision(self.policy_config.get("rule_error_effect", PolicyDecision.REQUIRE_REVIEW.value))

        if load_config_rules:
            self.load_rules_from_config()
        for rule in rules or ():
            self.add_rule(rule)

        logger.info("Policy Engine initialized")

    def add_rule(self, rule: PolicyRule) -> None:
        normalized = ensure_policy_rule(rule)
        with self._lock:
            if len(self._rules) >= self.max_rules and normalized.rule_id not in self._rule_index:
                raise _policy_error(
                    "Policy rule limit exceeded.",
                    context={"max_rules": self.max_rules, "rule_id": normalized.rule_id},
                    severity="high",
                )
            if normalized.rule_id in self._rule_index:
                if not self.allow_duplicate_rule_ids:
                    raise _policy_error(
                        "Duplicate policy rule id is not allowed.",
                        context={"rule_id": normalized.rule_id},
                        severity="medium",
                    )
                generated_id = f"{normalized.rule_id}:{stable_hash(normalized.to_dict(redact=False), length=8)}"
                normalized = clone_rule(normalized, rule_id=generated_id)

            self._rule_index[normalized.rule_id] = normalized
            self._rules = sorted(self._rule_index.values(), key=lambda item: (item.priority, item.created_at, item.rule_id))
            logger.debug("Policy rule registered: %s", normalized.rule_id)

    def add_simple_rule(self, *, rule_id: str, description: str, effect: PolicyDecision, priority: int,
                        predicate: PolicyPredicate, enabled: bool = True,
                        metadata: Optional[Dict[str, Any]] = None) -> PolicyRule:
        rule = PolicyRule(
            rule_id=rule_id,
            description=description,
            effect=normalize_policy_decision(effect),
            priority=priority,
            enabled=enabled,
            predicate=predicate,
            metadata=metadata or {},
        )
        self.add_rule(rule)
        return rule

    def add_condition_rule(self, *, rule_id: str, description: str, effect: PolicyDecisionInput,
                           conditions: Iterable[Union[PolicyCondition, Mapping[str, Any]]],
                           priority: int = 100, enabled: bool = True, condition_match: str = "all",
                           metadata: Optional[Mapping[str, Any]] = None, tags: Optional[Iterable[Any]] = None,
                           owner: Optional[str] = None, stop_processing: bool = False) -> PolicyRule:
        rule = PolicyRule(
            rule_id=rule_id,
            description=description,
            effect=normalize_policy_decision(effect),
            priority=priority,
            enabled=enabled,
            conditions=tuple(
                item if isinstance(item, PolicyCondition) else PolicyCondition.from_mapping(item)
                for item in conditions
            ),
            condition_match=condition_match,
            metadata=normalize_metadata(metadata, drop_none=True),
            tags=normalize_tags(tags),
            owner=owner,
            stop_processing=stop_processing,
        )
        self.add_rule(rule)
        return rule

    def load_rules_from_config(self) -> int:
        configured = self.policy_config.get("configured_rules", self.policy_config.get("rules", []))
        if configured is None:
            return 0
        if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
            raise _policy_error(
                "policy_engine.configured_rules must be a sequence of mappings.",
                context={"received_type": type(configured).__name__},
                severity="medium",
            )
        count = 0
        for item in configured:
            if not item:
                continue
            rule = build_rule_from_config(item)
            self.add_rule(rule)
            count += 1
        return count

    def remove_rule(self, rule_id: str) -> bool:
        normalized_id = normalize_rule_id(rule_id)
        with self._lock:
            existed = normalized_id in self._rule_index
            if existed:
                del self._rule_index[normalized_id]
                self._rules = sorted(self._rule_index.values(), key=lambda item: (item.priority, item.created_at, item.rule_id))
            return existed

    def clear_rules(self) -> None:
        with self._lock:
            self._rules.clear()
            self._rule_index.clear()

    def get_rule(self, rule_id: str) -> Optional[PolicyRule]:
        return self._rule_index.get(normalize_rule_id(rule_id))

    def enable_rule(self, rule_id: str) -> bool:
        return self._set_rule_enabled(rule_id, True)

    def disable_rule(self, rule_id: str) -> bool:
        return self._set_rule_enabled(rule_id, False)

    def _set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        normalized_id = normalize_rule_id(rule_id)
        with self._lock:
            current = self._rule_index.get(normalized_id)
            if current is None:
                return False
            updated = clone_rule(current, enabled=enabled, updated_at=epoch_seconds())
            self._rule_index[normalized_id] = updated
            self._rules = sorted(self._rule_index.values(), key=lambda item: (item.priority, item.created_at, item.rule_id))
            return True

    def evaluate(self, task: Dict[str, Any], agent_meta: Optional[Dict[str, Any]] = None,
                 context: Optional[Dict[str, Any]] = None) -> PolicyEvaluation:
        start_ms = monotonic_ms()
        correlation_id = generate_correlation_id("policy")
        warnings: List[str] = []
        errors: List[Dict[str, Any]] = []
        evaluated_rules: List[Dict[str, Any]] = []
        skipped_rules: List[str] = []
        matched_rules: List[str] = []
        deny_hits: List[PolicyRule] = []
        review_hits: List[PolicyRule] = []
        allow_hits: List[PolicyRule] = []

        normalized_task = normalize_task_payload(task, allow_none=False, redact=False)
        normalized_agent = normalize_metadata(agent_meta, drop_none=True)
        normalized_context = normalize_metadata(context, drop_none=True)

        if not self.enabled:
            evaluation = PolicyEvaluation(
                decision=self.default_decision,
                reasons=["Policy engine is disabled."],
                warnings=["policy_engine.enabled is false"],
                correlation_id=correlation_id,
                duration_ms=elapsed_ms(start_ms),
                metadata={"enabled": False},
            )
            self._record_evaluation(evaluation, normalized_task, normalized_agent, normalized_context)
            return evaluation

        with self._lock:
            rules = list(self._rules)

        for rule in rules:
            if not rule.enabled:
                skipped_rules.append(rule.rule_id)
                if self.include_unmatched_rules:
                    evaluated_rules.append(
                        PolicyRuleEvaluation(
                            rule_id=rule.rule_id,
                            matched=False,
                            effect=rule.effect,
                            description=rule.description,
                            priority=rule.priority,
                            enabled=False,
                            reason="rule disabled",
                            metadata=rule.metadata if self.include_rule_metadata else {},
                        ).to_dict(redact=self.redact_evaluation_payloads)
                    )
                continue
            if rule.expired:
                skipped_rules.append(rule.rule_id)
                if self.include_unmatched_rules:
                    evaluated_rules.append(
                        PolicyRuleEvaluation(
                            rule_id=rule.rule_id,
                            matched=False,
                            effect=rule.effect,
                            description=rule.description,
                            priority=rule.priority,
                            enabled=rule.enabled,
                            reason="rule expired",
                            metadata=rule.metadata if self.include_rule_metadata else {},
                        ).to_dict(redact=self.redact_evaluation_payloads)
                    )
                continue

            try:
                rule_eval = rule.evaluate_detailed(normalized_task, normalized_agent, normalized_context)
                if not self.include_rule_metadata:
                    rule_eval = PolicyRuleEvaluation(
                        rule_id=rule_eval.rule_id,
                        matched=rule_eval.matched,
                        effect=rule_eval.effect,
                        description=rule_eval.description,
                        priority=rule_eval.priority,
                        enabled=rule_eval.enabled,
                        duration_ms=rule_eval.duration_ms,
                        reason=rule_eval.reason,
                        error=rule_eval.error,
                        metadata={},
                    )
            except Exception as exc:
                payload = exception_to_error_payload(exc, action="policy_rule_evaluation").get("error", {"message": str(exc)})
                errors.append(
                    redact_mapping(
                        {
                            "rule_id": rule.rule_id,
                            "effect": self.rule_error_effect.value,
                            "error": payload,
                        }
                    )
                )
                warnings.append(f"Policy rule '{rule.rule_id}' failed: {type(exc).__name__}")
                synthetic = PolicyRuleEvaluation(
                    rule_id=rule.rule_id,
                    matched=True,
                    effect=self.rule_error_effect if self.fail_closed else PolicyDecision.REQUIRE_REVIEW,
                    description=f"Policy rule '{rule.rule_id}' failed during evaluation.",
                    priority=rule.priority,
                    enabled=rule.enabled,
                    duration_ms=elapsed_ms(start_ms),
                    reason="rule evaluation error",
                    error=payload,
                    metadata=rule.metadata if self.include_rule_metadata else {},
                )
                rule_eval = synthetic

            if rule_eval.matched or self.include_unmatched_rules:
                evaluated_rules.append(rule_eval.to_dict(redact=self.redact_evaluation_payloads))

            if not rule_eval.matched:
                continue

            matched_rules.append(rule.rule_id)
            if rule_eval.effect == PolicyDecision.DENY:
                deny_hits.append(rule)
                if self.stop_on_first_deny or rule.stop_processing:
                    break
            elif rule_eval.effect == PolicyDecision.REQUIRE_REVIEW:
                review_hits.append(rule)
                if self.stop_on_first_review or rule.stop_processing:
                    break
            elif rule_eval.effect == PolicyDecision.ALLOW:
                allow_hits.append(rule)
                if rule.stop_processing:
                    break

        if deny_hits:
            decision = PolicyDecision.DENY
            reasons = [rule.description for rule in deny_hits]
        elif review_hits:
            decision = PolicyDecision.REQUIRE_REVIEW
            reasons = [rule.description for rule in review_hits]
        elif errors and self.fail_closed:
            decision = PolicyDecision.DENY
            reasons = ["Policy evaluation failed and fail_closed is enabled."]
        elif errors:
            decision = PolicyDecision.REQUIRE_REVIEW
            reasons = ["Policy evaluation encountered one or more rule errors."]
        else:
            decision = self.default_decision
            reasons = [] if decision == PolicyDecision.ALLOW else ["Default policy decision applied."]

        evaluation = PolicyEvaluation(
            decision=decision,
            reasons=reasons,
            matched_rules=matched_rules,
            evaluated_rules=evaluated_rules,
            denied_rules=[rule.rule_id for rule in deny_hits],
            review_rules=[rule.rule_id for rule in review_hits],
            allow_rules=[rule.rule_id for rule in allow_hits],
            skipped_rules=skipped_rules,
            errors=errors,
            warnings=warnings,
            metadata={
                "rule_count": len(rules),
                "evaluated_rule_count": len(evaluated_rules),
                "matched_rule_count": len(matched_rules),
                "default_decision": self.default_decision.value,
                "fail_closed": self.fail_closed,
            },
            correlation_id=correlation_id,
            duration_ms=elapsed_ms(start_ms),
        )
        self._record_evaluation(evaluation, normalized_task, normalized_agent, normalized_context)
        return evaluation

    def _record_evaluation(
        self,
        evaluation: PolicyEvaluation,
        task: Mapping[str, Any],
        agent_meta: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> None:
        if not self.audit_enabled or self.shared_memory is None:
            return
        try:
            event = build_audit_event(
                "policy_evaluated",
                f"Policy decision: {evaluation.decision.value}",
                severity="warning" if evaluation.decision != PolicyDecision.ALLOW else "info",
                component="policy_engine",
                correlation_id=evaluation.correlation_id,
                context={
                    "task": task,
                    "agent_meta": agent_meta,
                    "context": context,
                    "evaluation": evaluation.to_dict(),
                },
            )
            append_audit_event(self.shared_memory, event, key=self.audit_key, max_events=self.audit_max_events)
        except Exception as exc:
            logger.warning("Failed to record policy audit event: %s", sanitize_for_logging(exc))

    def list_rules(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [rule.to_dict(include_predicate=True, redact=True) for rule in self._rules]

    def explain(self, task: Dict[str, Any], agent_meta: Optional[Dict[str, Any]] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        evaluation = self.evaluate(task, agent_meta=agent_meta, context=context)
        return evaluation.to_dict()

    def evaluate_to_result(self, task: Dict[str, Any], agent_meta: Optional[Dict[str, Any]] = None,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate(task, agent_meta=agent_meta, context=context).to_result()

    def validate_task(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            normalized = normalize_task_payload(task, allow_none=False, redact=False)
            return success_result(action="policy_task_validation", message="Task payload is valid", data={"task": normalized})
        except Exception as exc:
            return error_result(action="policy_task_validation", message="Task payload is invalid", error=exc)

    def rules_for_effect(self, effect: PolicyDecisionInput) -> List[Dict[str, Any]]:
        decision = normalize_policy_decision(effect)
        with self._lock:
            return [rule.to_dict(include_predicate=True) for rule in self._rules if rule.effect == decision]

    def export_rules(self, path: Optional[Union[str, Path]] = None) -> Union[List[Dict[str, Any]], Path]:
        payload = self.list_rules()
        if path is None:
            return payload
        return export_json_file(path, {"policy_engine": {"configured_rules": payload}}, pretty=True)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            rules = list(self._rules)
        by_effect = {decision.value: 0 for decision in PolicyDecision}
        enabled_count = 0
        expired_count = 0
        for rule in rules:
            by_effect[rule.effect.value] += 1
            if rule.enabled:
                enabled_count += 1
            if rule.expired:
                expired_count += 1
        return redact_mapping(
            {
                "enabled": self.enabled,
                "rule_count": len(rules),
                "enabled_rule_count": enabled_count,
                "disabled_rule_count": len(rules) - enabled_count,
                "expired_rule_count": expired_count,
                "by_effect": by_effect,
                "default_decision": self.default_decision.value,
                "fail_closed": self.fail_closed,
                "audit_enabled": self.audit_enabled,
            }
        )


# ---------------------------------------------------------------------------
# Normalization, rule construction, and condition evaluation helpers
# ---------------------------------------------------------------------------
def normalize_policy_decision(value: PolicyDecisionInput) -> PolicyDecision:
    if isinstance(value, PolicyDecision):
        return value
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "allow": PolicyDecision.ALLOW,
        "allowed": PolicyDecision.ALLOW,
        "pass": PolicyDecision.ALLOW,
        "deny": PolicyDecision.DENY,
        "denied": PolicyDecision.DENY,
        "block": PolicyDecision.DENY,
        "blocked": PolicyDecision.DENY,
        "reject": PolicyDecision.DENY,
        "require_review": PolicyDecision.REQUIRE_REVIEW,
        "review": PolicyDecision.REQUIRE_REVIEW,
        "manual_review": PolicyDecision.REQUIRE_REVIEW,
        "needs_review": PolicyDecision.REQUIRE_REVIEW,
    }
    if text in aliases:
        return aliases[text]
    raise _policy_error("Unknown policy decision.", context={"decision": value}, severity="medium")


def normalize_rule_id(rule_id: Any) -> str:
    return normalize_identifier_component(require_non_empty_string(rule_id, "rule_id"), lowercase=False, separator="_")


def normalize_condition_operator(operator: Any) -> str:
    text = str(operator or "exists").strip().lower().replace("-", "_")
    aliases = {
        "=": "eq",
        "==": "eq",
        "equals": "eq",
        "!=": "ne",
        "not_equals": "ne",
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
        "present": "exists",
        "absent": "missing",
        "matches": "regex",
        "pattern": "regex",
        "wildcard": "glob",
    }
    normalized = aliases.get(text, text)
    if normalized not in _ALLOWED_CONDITION_OPERATORS:
        raise _policy_error(
            "Unsupported policy condition operator.",
            context={"operator": operator, "allowed": sorted(_ALLOWED_CONDITION_OPERATORS)},
            severity="medium",
        )
    return normalized


def normalize_condition_source(source: Any) -> str:
    text = str(source or "task").strip().lower()
    aliases = {"agent_meta": "agent", "agent_metadata": "agent", "payload": "task"}
    normalized = aliases.get(text, text)
    if normalized not in {"task", "agent", "context"}:
        raise _policy_error("Unsupported policy condition source.", context={"source": source}, severity="medium")
    return normalized


def normalize_condition_match(value: Any) -> str:
    text = str(value or "all").strip().lower()
    if text in {"all", "and"}:
        return "all"
    if text in {"any", "or"}:
        return "any"
    if text in {"none", "not_any"}:
        return "none"
    raise _policy_error("Unsupported condition_match value.", context={"condition_match": value}, severity="medium")


def ensure_policy_rule(rule: Union[PolicyRule, Mapping[str, Any]]) -> PolicyRule:
    if isinstance(rule, PolicyRule):
        return rule
    if isinstance(rule, Mapping):
        return build_rule_from_config(rule)
    raise _policy_error("Expected PolicyRule or mapping.", context={"received_type": type(rule).__name__}, severity="medium")


def clone_rule(rule: PolicyRule, **updates: Any) -> PolicyRule:
    payload = rule.to_dict(include_predicate=False, redact=False)
    payload.update(updates)
    return PolicyRule(
        rule_id=payload["rule_id"],
        description=payload["description"],
        effect=normalize_policy_decision(payload["effect"]),
        priority=payload.get("priority", 100),
        enabled=payload.get("enabled", True),
        predicate=rule.predicate,
        metadata=payload.get("metadata", {}),
        conditions=tuple(PolicyCondition.from_mapping(item) for item in payload.get("conditions", [])),
        condition_match=payload.get("condition_match", "all"),
        tags=tuple(payload.get("tags", ())),
        owner=payload.get("owner"),
        created_at=payload.get("created_at", rule.created_at),
        updated_at=payload.get("updated_at"),
        expires_at=payload.get("expires_at"),
        stop_processing=payload.get("stop_processing", False),
        audit=payload.get("audit", True),
    )


def build_rule_from_config(payload: Mapping[str, Any]) -> PolicyRule:
    data = ensure_mapping(payload, field_name="policy_rule")
    raw_conditions = data.get("conditions") or data.get("when") or []
    if isinstance(raw_conditions, Mapping):
        raw_conditions = [raw_conditions]
    if raw_conditions and (not isinstance(raw_conditions, Sequence) or isinstance(raw_conditions, (str, bytes))):
        raise _policy_error(
            "Policy rule conditions must be a mapping or sequence of mappings.",
            context={"rule_id": data.get("rule_id", data.get("id")), "received_type": type(raw_conditions).__name__},
            severity="medium",
        )
    conditions = tuple(PolicyCondition.from_mapping(item) for item in raw_conditions)
    rule_id = require_non_empty_string(data.get("rule_id", data.get("id")), "rule_id")
    return PolicyRule(
        rule_id=rule_id,
        description=data.get("description", data.get("reason", "Configured policy rule")),
        effect=normalize_policy_decision(data.get("effect", data.get("decision", PolicyDecision.REQUIRE_REVIEW.value))),
        priority=coerce_int(data.get("priority"), default=100),
        enabled=coerce_bool(data.get("enabled"), default=True),
        predicate=None,
        metadata=normalize_metadata(data.get("metadata"), drop_none=True),
        conditions=conditions,
        condition_match=data.get("condition_match", data.get("match", "all")),
        tags=normalize_tags(data.get("tags")),
        owner=data.get("owner"),
        created_at=coerce_float(data.get("created_at"), default=epoch_seconds(), minimum=0.0),
        updated_at=coerce_float(data.get("updated_at"), default=0.0, minimum=0.0) or None,
        expires_at=coerce_float(data.get("expires_at"), default=0.0, minimum=0.0) or None,
        stop_processing=coerce_bool(data.get("stop_processing"), default=False),
        audit=coerce_bool(data.get("audit"), default=True),
    )


def evaluate_conditions(
    conditions: Iterable[PolicyCondition],
    *,
    task: Mapping[str, Any],
    agent_meta: Mapping[str, Any],
    context: Mapping[str, Any],
    match: str = "all",
) -> bool:
    resolved_match = normalize_condition_match(match)
    results = [evaluate_condition(condition, task=task, agent_meta=agent_meta, context=context) for condition in conditions]
    if not results:
        return True
    if resolved_match == "any":
        return any(results)
    if resolved_match == "none":
        return not any(results)
    return all(results)


def evaluate_condition(
    condition: PolicyCondition,
    *,
    task: Mapping[str, Any],
    agent_meta: Mapping[str, Any],
    context: Mapping[str, Any],
) -> bool:
    scope, path = split_condition_path(condition.path, condition.source)
    source = {"task": task, "agent": agent_meta, "context": context}[scope]
    actual = resolve_path(source, path, default=_MISSING)
    if actual is _MISSING and condition.default is not None:
        actual = condition.default
    expected = condition.value
    candidates = condition.values if condition.values else (() if expected is None else (expected,))
    return compare_condition_values(
        actual,
        operator=condition.operator,
        expected=expected,
        candidates=candidates,
        case_sensitive=condition.case_sensitive,
    )


def split_condition_path(path: str, default_source: str) -> Tuple[str, str]:
    text = require_non_empty_string(path, "condition.path")
    if "." in text:
        first, rest = text.split(".", 1)
        lowered = first.lower()
        if lowered in {"task", "agent", "agent_meta", "context", "payload"}:
            return normalize_condition_source(lowered), rest
    return normalize_condition_source(default_source), text


def resolve_path(source: Any, path: str, *, default: Any = None) -> Any:
    if not path:
        return source
    current = source
    for part in str(path).split("."):
        if current is _MISSING:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                current = current[int(part)]
            except Exception:
                return default
        else:
            current = getattr(current, part, _MISSING)
        if current is _MISSING:
            return default
    return current


def compare_condition_values(
    actual: Any,
    *,
    operator: str,
    expected: Any = None,
    candidates: Sequence[Any] = (),
    case_sensitive: bool = True,
) -> bool:
    op = normalize_condition_operator(operator)
    if op == "exists":
        return actual is not _MISSING and actual is not None
    if op == "missing":
        return actual is _MISSING or actual is None
    if actual is _MISSING:
        return False

    normalized_actual = _case_normalize(actual, case_sensitive=case_sensitive)
    normalized_expected = _case_normalize(expected, case_sensitive=case_sensitive)
    normalized_candidates = [_case_normalize(item, case_sensitive=case_sensitive) for item in candidates]

    if op == "eq":
        return normalized_actual == normalized_expected
    if op == "ne":
        return normalized_actual != normalized_expected
    if op in {"gt", "gte", "lt", "lte"}:
        return _numeric_compare(normalized_actual, normalized_expected, op)
    if op == "in":
        return normalized_actual in normalized_candidates
    if op == "not_in":
        return normalized_actual not in normalized_candidates
    if op == "contains":
        return _contains(normalized_actual, normalized_expected)
    if op == "not_contains":
        return not _contains(normalized_actual, normalized_expected)
    if op == "intersects":
        actual_set = set(_as_iterable_for_compare(normalized_actual))
        return bool(actual_set.intersection(set(normalized_candidates)))
    if op == "includes_all":
        actual_set = set(_as_iterable_for_compare(normalized_actual))
        return set(normalized_candidates).issubset(actual_set)
    if op == "regex":
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.search(str(expected), str(actual), flags=flags) is not None
    if op == "glob":
        if case_sensitive:
            return fnmatch.fnmatchcase(str(actual), str(expected))
        return fnmatch.fnmatch(str(actual).lower(), str(expected).lower())
    if op == "truthy":
        return bool(actual)
    if op == "falsy":
        return not bool(actual)
    return False


def _case_normalize(value: Any, *, case_sensitive: bool) -> Any:
    if case_sensitive:
        return value
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return [_case_normalize(item, case_sensitive=False) for item in value]
    if isinstance(value, tuple):
        return tuple(_case_normalize(item, case_sensitive=False) for item in value)
    if isinstance(value, set):
        return {_case_normalize(item, case_sensitive=False) for item in value}
    return value


def _numeric_compare(actual: Any, expected: Any, operator: str) -> bool:
    try:
        left = float(actual)
        right = float(expected)
    except Exception:
        return False
    if operator == "gt":
        return left > right
    if operator == "gte":
        return left >= right
    if operator == "lt":
        return left < right
    if operator == "lte":
        return left <= right
    return False


def _contains(container: Any, needle: Any) -> bool:
    if container is None:
        return False
    if isinstance(container, Mapping):
        return needle in container or str(needle) in container
    if isinstance(container, (list, tuple, set, frozenset)):
        return needle in container
    return str(needle) in str(container)


def _as_iterable_for_compare(value: Any) -> Tuple[Any, ...]:
    if value is None or value is _MISSING:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(value.keys())
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def describe_predicate(predicate: Optional[PolicyPredicate]) -> Optional[Dict[str, Any]]:
    if predicate is None:
        return None
    return prune_none(
        {
            "name": getattr(predicate, "__name__", None),
            "module": getattr(predicate, "__module__", None),
            "repr": truncate_text(repr(predicate), 300),
        },
        drop_empty=True,
    )


def _policy_error(message: str, *, context: Optional[Mapping[str, Any]] = None, severity: str = "high") -> Exception:
    payload = merge_mappings({"component": "policy_engine"}, context, deep=True, drop_none=True)
    return CollaborationError(
        CollaborationErrorType.ROUTING_FAILURE, # type: ignore
        message,
        severity=severity,
        context=payload,
        remediation_guidance="Review policy configuration, rule predicates, and collaborative runtime inputs.",
    ) # type: ignore


if __name__ == "__main__":
    print("\n=== Running Policy Engine ===\n")
    printer.status("TEST", "Policy Engine initialized", "info")

    class _Memory:
        def __init__(self):
            self._store: Dict[str, Any] = {}

        def get(self, key, default=None):
            return self._store.get(key, default)

        def set(self, key, value, **kwargs):
            self._store[key] = value
            return True

    memory = _Memory()
    engine = PolicyEngine(shared_memory=memory, load_config_rules=False)

    engine.add_simple_rule(
        rule_id="deny_high_risk",
        description="Deny tasks above critical risk threshold.",
        effect=PolicyDecision.DENY,
        priority=10,
        predicate=lambda task, agent, context: float(task.get("risk_score", 0.0)) >= 0.95,
        metadata={"threshold": 0.95},
    )

    engine.add_condition_rule(
        rule_id="review_sensitive_task",
        description="Require review for sensitive tasks.",
        effect=PolicyDecision.REQUIRE_REVIEW,
        priority=20,
        conditions=[{"path": "task.sensitive", "operator": "eq", "value": True}],
        metadata={"category": "safety"},
    )

    engine.add_condition_rule(
        rule_id="allow_translation",
        description="Allow translation tasks by default.",
        effect=PolicyDecision.ALLOW,
        priority=100,
        conditions=[{"path": "task.task_type", "operator": "eq", "value": "translate"}],
    )

    allow_eval = engine.evaluate({"task_type": "translate", "risk_score": 0.1}, agent_meta={"capabilities": ["translate"]})
    assert allow_eval.decision == PolicyDecision.ALLOW
    assert allow_eval.allowed

    review_eval = engine.evaluate({"task_type": "summarize", "risk_score": 0.2, "sensitive": True})
    assert review_eval.decision == PolicyDecision.REQUIRE_REVIEW
    assert "review_sensitive_task" in review_eval.matched_rules

    deny_eval = engine.evaluate({"task_type": "execute", "risk_score": 0.99, "token": "secret-value"})
    assert deny_eval.decision == PolicyDecision.DENY
    assert "deny_high_risk" in deny_eval.matched_rules
    assert deny_eval.to_result()["status"] == "error"

    assert engine.disable_rule("deny_high_risk") is True
    disabled_eval = engine.evaluate({"task_type": "execute", "risk_score": 0.99})
    assert disabled_eval.decision == PolicyDecision.ALLOW
    assert "deny_high_risk" in disabled_eval.skipped_rules
    assert engine.enable_rule("deny_high_risk") is True

    config_rule = build_rule_from_config(
        {
            "rule_id": "review_external_agent",
            "description": "Require review for external agents.",
            "effect": "require_review",
            "priority": 15,
            "conditions": [{"path": "agent.trust_tier", "operator": "eq", "value": "external"}],
            "metadata": {"source": "unit-test"},
        }
    )
    engine.add_rule(config_rule)
    external_eval = engine.evaluate({"task_type": "classify"}, agent_meta={"trust_tier": "external"})
    assert external_eval.decision == PolicyDecision.REQUIRE_REVIEW
    assert "review_external_agent" in external_eval.matched_rules

    rules = engine.list_rules()
    assert len(rules) == 4
    assert engine.summary()["rule_count"] == 4
    assert isinstance(engine.export_rules(), list)
    assert memory.get(engine.audit_key), "audit events should be recorded"

    print("\n=== Test ran successfully ===\n")
