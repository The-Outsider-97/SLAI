from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Policy Engine")
printer = PrettyPrinter

class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_REVIEW = "require_review"


PolicyPredicate = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], bool]


@dataclass
class PolicyRule:
    rule_id: str
    description: str
    effect: PolicyDecision
    priority: int = 100
    enabled: bool = True
    predicate: Optional[PolicyPredicate] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, task: Dict[str, Any], agent_meta: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        if self.predicate is None:
            return False
        return bool(self.predicate(task, agent_meta, context))


@dataclass
class PolicyEvaluation:
    decision: PolicyDecision
    reasons: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "matched_rules": list(self.matched_rules),
        }


class PolicyEngine:
    def __init__(self):
        self._rules: List[PolicyRule] = []

        logger.info("Policy Engine initialized")

    def add_rule(self, rule: PolicyRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda item: item.priority)

    def add_simple_rule(
        self,
        *,
        rule_id: str,
        description: str,
        effect: PolicyDecision,
        priority: int,
        predicate: PolicyPredicate,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PolicyRule:
        rule = PolicyRule(
            rule_id=rule_id,
            description=description,
            effect=effect,
            priority=priority,
            enabled=enabled,
            predicate=predicate,
            metadata=metadata or {},
        )
        self.add_rule(rule)
        return rule

    def evaluate(
        self,
        task: Dict[str, Any],
        agent_meta: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyEvaluation:
        agent_meta = agent_meta or {}
        context = context or {}

        review_hits: List[PolicyRule] = []
        deny_hits: List[PolicyRule] = []

        for rule in self._rules:
            if not rule.evaluate(task, agent_meta, context):
                continue
            if rule.effect == PolicyDecision.DENY:
                deny_hits.append(rule)
            elif rule.effect == PolicyDecision.REQUIRE_REVIEW:
                review_hits.append(rule)

        if deny_hits:
            return PolicyEvaluation(
                decision=PolicyDecision.DENY,
                reasons=[rule.description for rule in deny_hits],
                matched_rules=[rule.rule_id for rule in deny_hits],
            )

        if review_hits:
            return PolicyEvaluation(
                decision=PolicyDecision.REQUIRE_REVIEW,
                reasons=[rule.description for rule in review_hits],
                matched_rules=[rule.rule_id for rule in review_hits],
            )

        return PolicyEvaluation(decision=PolicyDecision.ALLOW)

    def list_rules(self) -> List[Dict[str, Any]]:
        return [
            {
                "rule_id": rule.rule_id,
                "description": rule.description,
                "effect": rule.effect.value,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "metadata": dict(rule.metadata),
            }
            for rule in self._rules
        ]
