from __future__ import annotations

__version__ = "2.0.0"

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.handler_error import *
from .utils.handler_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Probabilistic Strategy Selector")
printer = PrettyPrinter()


@dataclass(frozen=True)
class StrategyHistoryStats:
    """Telemetry-derived performance statistics for one recovery strategy."""

    strategy: str
    total: int = 0
    recovered: int = 0
    failed: int = 0
    degraded: int = 0
    skipped: int = 0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    recent_total: int = 0
    recent_recovered: int = 0
    recent_failed: int = 0
    recent_success_rate: float = 0.0
    avg_attempts: float = 0.0
    last_seen: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "strategy": self.strategy,
                "total": self.total,
                "recovered": self.recovered,
                "failed": self.failed,
                "degraded": self.degraded,
                "skipped": self.skipped,
                "success_rate": self.success_rate,
                "failure_rate": self.failure_rate,
                "recent_total": self.recent_total,
                "recent_recovered": self.recent_recovered,
                "recent_failed": self.recent_failed,
                "recent_success_rate": self.recent_success_rate,
                "avg_attempts": self.avg_attempts,
                "last_seen": self.last_seen,
            },
            drop_none=True,
        )


@dataclass(frozen=True)
class StrategyScore:
    """Explainable score for one candidate recovery strategy."""

    strategy: str
    score: float
    probability: float
    prior: float
    likelihood: float
    historical_success_rate: float
    recent_success_rate: float
    confidence: float
    evidence_count: int
    penalty: float = 0.0
    bonus: float = 0.0
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    stats: StrategyHistoryStats = field(default_factory=lambda: StrategyHistoryStats(strategy="unknown"))

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "strategy": self.strategy,
                "score": self.score,
                "probability": self.probability,
                "prior": self.prior,
                "likelihood": self.likelihood,
                "historical_success_rate": self.historical_success_rate,
                "recent_success_rate": self.recent_success_rate,
                "confidence": self.confidence,
                "evidence_count": self.evidence_count,
                "penalty": self.penalty,
                "bonus": self.bonus,
                "reasons": list(self.reasons),
                "stats": self.stats.to_dict(),
            },
            drop_none=True,
            drop_empty=True,
        )


@dataclass(frozen=True)
class StrategySelection:
    """Production strategy-selection result.

    The public ``to_dict`` shape intentionally contains the legacy keys consumed by
    HandlerAgent: selected_strategy, distribution, and candidates.
    """

    selected_strategy: str
    distribution: Dict[str, float]
    candidates: List[str]
    scores: List[StrategyScore]
    category: str
    severity: str
    retryable: bool
    policy_action: str
    reason: str
    confidence: float
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=utc_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "schema": "handler.strategy_selection.v2",
                "selected_strategy": self.selected_strategy,
                "distribution": dict(self.distribution),
                "candidates": list(self.candidates),
                "scores": [score.to_dict() for score in self.scores],
                "category": self.category,
                "severity": self.severity,
                "retryable": self.retryable,
                "policy_action": self.policy_action,
                "reason": self.reason,
                "confidence": self.confidence,
                "fallback_used": self.fallback_used,
                "metadata": dict(self.metadata),
                "timestamp": self.timestamp,
            },
            drop_none=True,
            drop_empty=True,
        )


class ProbabilisticStrategySelector:
    """
    Production recovery strategy selector for HandlerAgent.

    Scope:
    - preserves the legacy ``select(normalized_failure, telemetry_history)`` API
    - selects from HandlerAgent-compatible recovery strategy names
    - combines configured priors, failure category evidence, policy action hints,
      historical strategy performance, recent strategy pressure, and optional insight
    - consumes telemetry_history or HandlerMemory-like objects without importing memory
    - emits explainable selection payloads for telemetry and learning loops

    This selector does not execute recovery, mutate retry budgets, or open circuits.
    Those responsibilities remain with HandlerAgent, AdaptiveRetryPolicy, and
    HandlerPolicy respectively.
    """

    DEFAULT_PRIORS: Mapping[str, float] = {
        "network": 0.20,
        "timeout": 0.20,
        "memory": 0.15,
        "runtime": 0.25,
        "dependency": 0.10,
        "resource": 0.07,
        "unicode": 0.03,
    }

    DEFAULT_ALLOWED_STRATEGIES: Tuple[str, ...] = (
        "runtime",
        "network",
        "timeout",
        "memory",
        "dependency",
        "resource",
        "unicode",
    )

    DEFAULT_CATEGORY_CANDIDATES: Mapping[str, Tuple[str, ...]] = {
        "timeout": ("timeout", "network", "runtime"),
        "network": ("network", "timeout", "runtime"),
        "memory": ("memory", "resource", "runtime"),
        "dependency": ("dependency", "runtime"),
        "resource": ("resource", "memory", "runtime"),
        "unicode": ("unicode", "runtime"),
        "validation": ("runtime",),
        "sla": ("runtime", "resource"),
        "security": ("runtime",),
        "runtime": ("runtime",),
    }

    DEFAULT_ACTION_CANDIDATES: Mapping[str, Tuple[str, ...]] = {
        HandlerRecoveryAction.RETRY.value: ("timeout", "network", "unicode", "runtime"),
        HandlerRecoveryAction.DEGRADE.value: ("resource", "memory", "runtime"),
        HandlerRecoveryAction.ESCALATE.value: ("runtime", "dependency"),
        HandlerRecoveryAction.FAIL_FAST.value: ("runtime",),
        HandlerRecoveryAction.QUARANTINE.value: ("runtime",),
        HandlerRecoveryAction.NONE.value: ("runtime",),
    }

    DEFAULT_STRATEGY_RULES: Mapping[str, Tuple[str, ...]] = {
        "network": ("network", "connection", "socket", "dns", "http", "ssl", "tls"),
        "timeout": ("timeout", "timed out", "deadline exceeded"),
        "memory": ("memory", "oom", "outofmemory", "out of memory", "cuda"),
        "dependency": ("dependency", "import", "module", "dll", "package", "no module named", "cannot import"),
        "resource": ("resource", "busy", "quota", "rate limit", "cpu", "gpu", "disk"),
        "unicode": ("unicode", "codec", "encode", "decode"),
        "runtime": ("runtime", "exception", "error", "unknown"),
    }

    DEFAULT_SEVERITY_MODIFIERS: Mapping[str, Mapping[str, float]] = {
        FailureSeverity.LOW.value: {"timeout": 0.04, "network": 0.04, "runtime": 0.01},
        FailureSeverity.MEDIUM.value: {"runtime": 0.02},
        FailureSeverity.HIGH.value: {"memory": 0.04, "dependency": 0.04, "resource": 0.02},
        FailureSeverity.CRITICAL.value: {"runtime": 0.08, "dependency": 0.03},
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *, memory: Any = None,
                 error_policy: Optional[HandlerErrorPolicy] = None ):
        self.config = load_global_config()
        policy_cfg = get_config_section("policy")
        selector_cfg = get_config_section("strategy_selector")

        # ``policy`` is retained only for legacy fallback keys such as strategy_priors.
        merged = deep_merge(policy_cfg, selector_cfg)
        if isinstance(config, Mapping):
            merged = deep_merge(merged, config)

        policy_config = merged.get("error_policy") if isinstance(merged.get("error_policy"), Mapping) else None
        self.error_policy = error_policy or HandlerErrorPolicy.from_mapping(policy_config)
        self.memory = memory

        self.enabled = coerce_bool(merged.get("enabled"), default=True)
        self.default_strategy = normalize_identifier(merged.get("default_strategy"), default="runtime")
        self.fallback_strategy = normalize_identifier(merged.get("fallback_strategy"), default="runtime")
        self.allowed_strategies = tuple(
            self._unique_strategies(coerce_list(merged.get("allowed_strategies"), default=self.DEFAULT_ALLOWED_STRATEGIES, split_strings=True))
        )
        if self.default_strategy not in self.allowed_strategies:
            self.default_strategy = "runtime" if "runtime" in self.allowed_strategies else self.allowed_strategies[0]
        if self.fallback_strategy not in self.allowed_strategies:
            self.fallback_strategy = self.default_strategy

        self.priors = self._normalize_priors(
            deep_merge(self.DEFAULT_PRIORS, coerce_mapping(merged.get("strategy_priors")))
        )
        self.category_candidates = self._load_strategy_tuple_mapping(
            merged.get("category_candidates"),
            default=self.DEFAULT_CATEGORY_CANDIDATES,
        )
        self.action_candidates = self._load_strategy_tuple_mapping(
            merged.get("action_candidates"),
            default=self.DEFAULT_ACTION_CANDIDATES,
        )
        self.strategy_rules = self._load_strategy_tuple_mapping(
            merged.get("strategy_rules"),
            default=self.DEFAULT_STRATEGY_RULES,
            allow_non_strategy_keys=False,
        )
        self.severity_modifiers = self._load_nested_float_mapping(
            merged.get("severity_modifiers"),
            default=self.DEFAULT_SEVERITY_MODIFIERS,
        )

        self.prior_weight = coerce_float(merged.get("prior_weight"), 0.45, minimum=0.0, maximum=1.0)
        self.history_weight = coerce_float(merged.get("history_weight"), 0.35, minimum=0.0, maximum=1.0)
        self.rule_weight = coerce_float(merged.get("rule_weight"), 0.20, minimum=0.0, maximum=1.0)
        self._normalize_weights()

        self.default_unknown_success_rate = coerce_float(merged.get("default_unknown_success_rate"), 0.5, minimum=0.0, maximum=1.0)
        self.min_score = coerce_float(merged.get("min_score"), 0.001, minimum=0.0, maximum=1.0)
        self.min_confidence = coerce_float(merged.get("min_confidence"), 0.35, minimum=0.0, maximum=1.0)
        self.max_confidence = coerce_float(merged.get("max_confidence"), 0.95, minimum=0.0, maximum=1.0)
        self.min_samples = coerce_int(merged.get("min_samples"), 3, minimum=0, maximum=100_000)
        self.default_history_limit = coerce_int(merged.get("default_history_limit"), 500, minimum=1, maximum=1_000_000)
        self.recent_window_seconds = coerce_float(merged.get("recent_window_seconds"), 300.0, minimum=0.0)
        self.max_candidates = coerce_int(merged.get("max_candidates"), 5, minimum=1, maximum=max(1, len(self.allowed_strategies)))
        self.include_runtime_fallback = coerce_bool(merged.get("include_runtime_fallback"), default=True)
        self.penalize_failed_recent_strategy = coerce_bool(merged.get("penalize_failed_recent_strategy"), default=True)
        self.recent_failure_penalty = coerce_float(merged.get("recent_failure_penalty"), 0.08, minimum=0.0, maximum=1.0)
        self.low_success_penalty = coerce_float(merged.get("low_success_penalty"), 0.06, minimum=0.0, maximum=1.0)
        self.high_success_bonus = coerce_float(merged.get("high_success_bonus"), 0.06, minimum=0.0, maximum=1.0)
        self.high_success_rate_threshold = coerce_float(merged.get("high_success_rate_threshold"), 0.70, minimum=0.0, maximum=1.0)
        self.low_success_rate_threshold = coerce_float(merged.get("low_success_rate_threshold"), 0.20, minimum=0.0, maximum=1.0)
        self.round_probabilities = coerce_int(merged.get("round_probabilities"), 6, minimum=0, maximum=10)
        self.emit_to_memory = coerce_bool(merged.get("emit_to_memory"), default=False)
        self.memory_event_type = normalize_identifier(merged.get("memory_event_type"), default="handler_strategy_selection")

        self._validate_configuration()
        logger.info(
            "Probabilistic Strategy Selector initialized | enabled=%s default=%s allowed=%s",
            self.enabled,
            self.default_strategy,
            len(self.allowed_strategies),
        )

    def attach_memory(self, memory: Any) -> None:
        """Attach a HandlerMemory-like object after construction."""
        self.memory = memory

    def select(self, normalized_failure: Mapping[str, Any],
               telemetry_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Legacy HandlerAgent API: select a strategy and return a dict."""
        return self.decide(normalized_failure=normalized_failure, telemetry_history=telemetry_history).to_dict()

    def decide(
        self,
        *,
        normalized_failure: Mapping[str, Any],
        telemetry_history: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        insight: Optional[Mapping[str, Any]] = None,
        available_strategies: Optional[Sequence[str]] = None,
        memory: Any = None,
    ) -> StrategySelection:
        """Build an explainable probabilistic strategy selection."""
        try:
            if not isinstance(normalized_failure, Mapping):
                raise ValidationError(
                    "ProbabilisticStrategySelector expected normalized_failure to be a mapping",
                    context={"actual_type": type(normalized_failure).__name__},
                    code="HANDLER_STRATEGY_FAILURE_MAPPING_REQUIRED",
                    policy=self.error_policy,
                )

            failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
            context_map = coerce_mapping(context)
            insight_map = coerce_mapping(insight)
            history = self._resolve_history(telemetry_history=telemetry_history, memory=memory, failure=failure)
            active_allowed = tuple(self._unique_strategies(available_strategies or self.allowed_strategies))
            if not active_allowed:
                raise ConfigurationError(
                    "No recovery strategies are available for selection",
                    code="HANDLER_STRATEGY_NO_AVAILABLE_STRATEGIES",
                    policy=self.error_policy,
                )

            severity = normalize_severity(failure.get("severity"))
            retryable = coerce_bool(failure.get("retryable"), default=False)
            category = normalize_identifier(
                insight_map.get("category") or failure.get("category") or classify_failure_category(failure.get("type"), failure.get("message")),
                default="runtime",
            )
            policy_action = normalize_recovery_action(
                insight_map.get("action") or failure.get("policy_action") or failure.get("action")
            )

            if not self.enabled:
                return self._fallback_selection(
                    failure=failure,
                    category=category,
                    severity=severity,
                    retryable=retryable,
                    policy_action=policy_action,
                    reason="strategy_selector_disabled",
                    active_allowed=active_allowed,
                )

            candidates, candidate_reasons = self.infer_candidates(
                failure=failure,
                category=category,
                severity=severity,
                retryable=retryable,
                policy_action=policy_action,
                insight=insight_map,
                active_allowed=active_allowed,
            )
            if not candidates:
                return self._fallback_selection(
                    failure=failure,
                    category=category,
                    severity=severity,
                    retryable=retryable,
                    policy_action=policy_action,
                    reason="no_candidates_after_filtering",
                    active_allowed=active_allowed,
                )

            raw_scores = [
                self.score_strategy(
                    strategy=strategy,
                    failure=failure,
                    category=category,
                    severity=severity,
                    retryable=retryable,
                    policy_action=policy_action,
                    telemetry_history=history,
                    candidate_reasons=candidate_reasons.get(strategy, ()),
                )
                for strategy in candidates
            ]
            scored = self._normalize_scores(raw_scores)
            selected = max(scored, key=lambda item: (item.probability, item.score, self.priors.get(item.strategy, 0.0))).strategy
            distribution = {score.strategy: round(score.probability, self.round_probabilities) for score in scored}
            distribution = summarize_strategy_distribution(distribution, precision=self.round_probabilities) if distribution else {}
            confidence = self._selection_confidence(scored=scored, selected=selected)

            selection = StrategySelection(
                selected_strategy=selected,
                distribution=distribution,
                candidates=list(candidates),
                scores=scored,
                category=category,
                severity=severity,
                retryable=retryable,
                policy_action=policy_action,
                reason="probabilistic_score_max",
                confidence=confidence,
                fallback_used=False,
                metadata=compact_dict(
                    {
                        "failure_type": failure.get("type"),
                        "context_hash": failure.get("context_hash"),
                        "route": context_map.get("route"),
                        "agent": context_map.get("agent"),
                        "task_id": context_map.get("task_id"),
                        "history_events": len(history),
                        "selector_version": __version__,
                    },
                    drop_none=True,
                    drop_empty=True,
                ),
            )
            self._emit_selection(selection=selection, failure=failure, context=context_map)
            return selection
        except HandlerError:
            raise
        except Exception as exc:
            raise PolicyError(
                "Probabilistic strategy selection failed",
                cause=exc,
                context={"normalized_failure_type": type(normalized_failure).__name__},
                code="HANDLER_STRATEGY_SELECTION_FAILED",
                policy=self.error_policy,
            ) from exc

    def infer_candidates(
        self,
        *,
        failure: Mapping[str, Any],
        category: Optional[str] = None,
        severity: Optional[str] = None,
        retryable: Optional[bool] = None,
        policy_action: Optional[str] = None,
        insight: Optional[Mapping[str, Any]] = None,
        active_allowed: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], Dict[str, Tuple[str, ...]]]:
        """Infer strategy candidates and explain the signals that included them."""
        allowed = tuple(self._unique_strategies(active_allowed or self.allowed_strategies))
        allowed_set = set(allowed)
        failure_type = coerce_str(failure.get("type"), default=DEFAULT_FAILURE_TYPE, max_chars=240).lower()
        message = coerce_str(failure.get("message"), default=DEFAULT_FAILURE_MESSAGE, max_chars=4000).lower()
        category = normalize_identifier(category or failure.get("category") or classify_failure_category(failure_type, message), default="runtime")
        severity = normalize_severity(severity or failure.get("severity"))
        retryable = coerce_bool(failure.get("retryable") if retryable is None else retryable, default=False)
        action = normalize_recovery_action(policy_action or failure.get("policy_action") or failure.get("action"))
        insight = coerce_mapping(insight)

        candidates: List[str] = []
        reasons: Dict[str, List[str]] = {}

        def add(strategy: Any, reason: str) -> None:
            normalized = normalize_identifier(strategy, default="")
            if not normalized or normalized not in allowed_set:
                return
            if normalized not in candidates:
                candidates.append(normalized)
            reasons.setdefault(normalized, []).append(reason)

        for strategy in self.category_candidates.get(category, (self.default_strategy,)):
            add(strategy, f"category:{category}")

        for strategy in self.action_candidates.get(action, ()):  # policy-action hint is secondary
            add(strategy, f"action:{action}")

        insight_strategy = insight.get("strategy") or insight.get("recommended_strategy")
        if insight_strategy:
            add(insight_strategy, "insight_strategy")

        lowered = f"{failure_type} {message}"
        for strategy, tokens in self.strategy_rules.items():
            if any(token and token in lowered for token in tokens):
                add(strategy, "message_rule")

        if severity == FailureSeverity.CRITICAL.value:
            add(self.default_strategy, "critical_safe_default")
        if not retryable and action in {HandlerRecoveryAction.FAIL_FAST.value, HandlerRecoveryAction.QUARANTINE.value}:
            add(self.default_strategy, "non_retryable_safe_default")
        if self.include_runtime_fallback:
            add(self.fallback_strategy, "fallback")

        if not candidates:
            add(self.default_strategy, "default")

        candidates = candidates[: self.max_candidates]
        return candidates, {strategy: tuple(reasons.get(strategy, ())) for strategy in candidates}

    def score_strategy(
        self,
        *,
        strategy: str,
        failure: Mapping[str, Any],
        category: str,
        severity: str,
        retryable: bool,
        policy_action: str,
        telemetry_history: Sequence[Mapping[str, Any]],
        candidate_reasons: Sequence[str] = (),
    ) -> StrategyScore:
        """Score one candidate strategy using priors, rules, and historical outcomes."""
        normalized_strategy = normalize_identifier(strategy, default=self.default_strategy)
        stats = self.strategy_stats(strategy=normalized_strategy, telemetry_history=telemetry_history, category=category)
        prior = coerce_float(self.priors.get(normalized_strategy), 0.01, minimum=0.0, maximum=1.0)
        historical = stats.success_rate if stats.total >= self.min_samples else self.default_unknown_success_rate
        recent = stats.recent_success_rate if stats.recent_total >= self.min_samples else historical
        evidence_count = len(candidate_reasons)
        rule_likelihood = min(1.0, 0.45 + (0.15 * evidence_count)) if evidence_count else 0.35

        score = (self.prior_weight * prior) + (self.history_weight * historical) + (self.rule_weight * rule_likelihood)
        penalty = 0.0
        bonus = 0.0
        reasons = list(candidate_reasons)

        severity_bonus = coerce_float(self.severity_modifiers.get(severity, {}).get(normalized_strategy), 0.0, minimum=-1.0, maximum=1.0)
        if severity_bonus >= 0:
            bonus += severity_bonus
        else:
            penalty += abs(severity_bonus)
        if severity_bonus != 0:
            reasons.append(f"severity_modifier:{severity}")

        if stats.total >= self.min_samples:
            if stats.success_rate >= self.high_success_rate_threshold:
                bonus += self.high_success_bonus
                reasons.append("historical_success_bonus")
            elif stats.success_rate <= self.low_success_rate_threshold:
                penalty += self.low_success_penalty
                reasons.append("historical_low_success_penalty")

        if self.penalize_failed_recent_strategy and stats.recent_total >= self.min_samples:
            recent_failure_rate = safe_ratio(stats.recent_failed, stats.recent_total, default=0.0)
            if recent_failure_rate >= (1.0 - self.low_success_rate_threshold):
                penalty += self.recent_failure_penalty
                reasons.append("recent_failure_pressure_penalty")

        if not retryable and normalized_strategy in {"timeout", "network", "unicode"}:
            penalty += self.low_success_penalty
            reasons.append("non_retryable_retry_strategy_penalty")

        if policy_action in {HandlerRecoveryAction.QUARANTINE.value, HandlerRecoveryAction.FAIL_FAST.value} and normalized_strategy != self.default_strategy:
            penalty += self.low_success_penalty
            reasons.append("fail_fast_default_strategy_bias")

        likelihood = coerce_float((0.55 * historical) + (0.25 * recent) + (0.20 * rule_likelihood), 0.0, minimum=0.0, maximum=1.0)
        confidence = self._strategy_confidence(stats=stats, evidence_count=evidence_count)
        final_score = max(self.min_score, score + bonus - penalty)
        return StrategyScore(
            strategy=normalized_strategy,
            score=round(final_score, 8),
            probability=0.0,
            prior=round(prior, 6),
            likelihood=round(likelihood, 6),
            historical_success_rate=stats.success_rate,
            recent_success_rate=stats.recent_success_rate,
            confidence=confidence,
            evidence_count=evidence_count,
            penalty=round(penalty, 6),
            bonus=round(bonus, 6),
            reasons=tuple(dict.fromkeys(reasons)),
            stats=stats,
        )

    def strategy_stats(
        self,
        *,
        strategy: str,
        telemetry_history: Optional[Sequence[Mapping[str, Any]]] = None,
        category: Optional[str] = None,
    ) -> StrategyHistoryStats:
        """Return historical performance stats for a strategy."""
        normalized_strategy = normalize_identifier(strategy, default=self.default_strategy)
        history = [dict(event) for event in coerce_list(telemetry_history) if isinstance(event, Mapping)]
        now = utc_timestamp()
        recent_cutoff = now - self.recent_window_seconds if self.recent_window_seconds > 0 else None

        matched: List[Mapping[str, Any]] = []
        for event in history:
            recovery = coerce_mapping(event.get("recovery"))
            event_strategy = strategy_base_name(recovery.get("strategy"))
            if event_strategy != normalized_strategy:
                continue
            if category:
                failure = coerce_mapping(event.get("failure"))
                insight = coerce_mapping(event.get("insight"))
                event_category = insight.get("category") or failure.get("category")
                if event_category and str(event_category) != str(category):
                    continue
            matched.append(event)

        total = len(matched)
        recovered = failed = degraded = skipped = 0
        attempts_total = 0
        attempts_count = 0
        timestamps: List[float] = []

        for event in matched:
            recovery = coerce_mapping(event.get("recovery"))
            status = str(recovery.get("status") or "unknown").lower()
            if status == "recovered":
                recovered += 1
            elif status == "degraded":
                degraded += 1
                recovered += 1
            elif status == "skipped":
                skipped += 1
                failed += 1
            else:
                failed += 1
            attempts = recovery.get("attempts")
            if attempts is not None:
                attempts_total += coerce_int(attempts, 0, minimum=0)
                attempts_count += 1
            timestamps.append(coerce_float(event.get("timestamp"), 0.0, minimum=0.0))

        recent = [event for event in matched if recent_cutoff is None or coerce_float(event.get("timestamp"), 0.0) >= recent_cutoff]
        recent_recovered = recent_failed = 0
        for event in recent:
            status = str(coerce_mapping(event.get("recovery")).get("status") or "unknown").lower()
            if status in {"recovered", "degraded"}:
                recent_recovered += 1
            else:
                recent_failed += 1

        return StrategyHistoryStats(
            strategy=normalized_strategy,
            total=total,
            recovered=recovered,
            failed=failed,
            degraded=degraded,
            skipped=skipped,
            success_rate=round(safe_ratio(recovered, total, default=0.0, minimum=0.0, maximum=1.0), 4),
            failure_rate=round(safe_ratio(failed, total, default=0.0, minimum=0.0, maximum=1.0), 4),
            recent_total=len(recent),
            recent_recovered=recent_recovered,
            recent_failed=recent_failed,
            recent_success_rate=round(safe_ratio(recent_recovered, len(recent), default=0.0, minimum=0.0, maximum=1.0), 4),
            avg_attempts=round(safe_ratio(attempts_total, attempts_count, default=0.0), 4),
            last_seen=max(timestamps) if timestamps else None,
        )

    def summarize(self, telemetry_history: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
        """Summarize selector configuration and optional telemetry history."""
        history = list(telemetry_history or self._history_from_memory(limit=self.default_history_limit))
        strategy_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        for event in history:
            recovery = coerce_mapping(event.get("recovery"))
            strategy_counts[strategy_base_name(recovery.get("strategy"))] += 1
            status_counts[str(recovery.get("status") or "unknown").lower()] += 1

        return {
            "schema": "handler.strategy_selector.summary.v2",
            "timestamp": utc_timestamp(),
            "enabled": self.enabled,
            "default_strategy": self.default_strategy,
            "fallback_strategy": self.fallback_strategy,
            "allowed_strategies": list(self.allowed_strategies),
            "strategy_priors": dict(self.priors),
            "weights": {
                "prior": self.prior_weight,
                "history": self.history_weight,
                "rule": self.rule_weight,
            },
            "telemetry": success_rate_for_events(history),
            "strategy_counts": dict(strategy_counts),
            "status_counts": dict(status_counts),
        }

    def _fallback_selection(
        self,
        *,
        failure: Mapping[str, Any],
        category: str,
        severity: str,
        retryable: bool,
        policy_action: str,
        reason: str,
        active_allowed: Sequence[str],
    ) -> StrategySelection:
        strategy = self.fallback_strategy if self.fallback_strategy in active_allowed else self.default_strategy
        if strategy not in active_allowed:
            strategy = active_allowed[0]
        score = StrategyScore(
            strategy=strategy,
            score=1.0,
            probability=1.0,
            prior=coerce_float(self.priors.get(strategy), 0.01),
            likelihood=1.0,
            historical_success_rate=0.0,
            recent_success_rate=0.0,
            confidence=self.min_confidence,
            evidence_count=0,
            reasons=(reason,),
            stats=StrategyHistoryStats(strategy=strategy),
        )
        return StrategySelection(
            selected_strategy=strategy,
            distribution={strategy: 1.0},
            candidates=[strategy],
            scores=[score],
            category=category,
            severity=severity,
            retryable=retryable,
            policy_action=policy_action,
            reason=reason,
            confidence=self.min_confidence,
            fallback_used=True,
            metadata=compact_dict(
                {
                    "failure_type": failure.get("type"),
                    "context_hash": failure.get("context_hash"),
                },
                drop_none=True,
            ),
        )

    def _resolve_history(self, *, telemetry_history: Optional[Sequence[Mapping[str, Any]]],
                         memory: Any = None, failure: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        if telemetry_history is not None:
            return [dict(event) for event in telemetry_history if isinstance(event, Mapping)]

        active_memory = memory or self.memory
        if active_memory is None:
            return []

        if failure is not None and hasattr(active_memory, "failure_history") and callable(active_memory.failure_history):
            history = active_memory.failure_history(context_hash=failure.get("context_hash"), limit=self.default_history_limit)
            if isinstance(history, (list, tuple)):
                return [dict(event) for event in history if isinstance(event, Mapping)]
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

    def _emit_selection(self, *, selection: StrategySelection, failure: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        if not self.emit_to_memory or self.memory is None:
            return
        event = {
            "event_type": self.memory_event_type,
            "timestamp": selection.timestamp,
            "failure": normalize_failure_payload(failure, policy=self.error_policy),
            "recovery": {"status": "selected", "strategy": selection.selected_strategy, "attempts": 0},
            "context": select_keys(context, ("route", "agent", "task_id", "priority", "correlation_id")),
            "strategy_selection": selection.to_dict(),
            "strategy_distribution": selection.distribution,
        }
        if hasattr(self.memory, "append_telemetry") and callable(self.memory.append_telemetry):
            self.memory.append_telemetry(event)

    def _normalize_scores(self, scores: Sequence[StrategyScore]) -> List[StrategyScore]:
        total = sum(max(self.min_score, score.score) for score in scores) or 1.0
        normalized: List[StrategyScore] = []
        for score in scores:
            probability = max(self.min_score, score.score) / total
            normalized.append(
                StrategyScore(
                    strategy=score.strategy,
                    score=score.score,
                    probability=round(probability, self.round_probabilities),
                    prior=score.prior,
                    likelihood=score.likelihood,
                    historical_success_rate=score.historical_success_rate,
                    recent_success_rate=score.recent_success_rate,
                    confidence=score.confidence,
                    evidence_count=score.evidence_count,
                    penalty=score.penalty,
                    bonus=score.bonus,
                    reasons=score.reasons,
                    stats=score.stats,
                )
            )
        return normalized

    def _selection_confidence(self, *, scored: Sequence[StrategyScore], selected: str) -> float:
        if not scored:
            return self.min_confidence
        by_strategy = {score.strategy: score for score in scored}
        selected_score = by_strategy.get(selected)
        if selected_score is None:
            return self.min_confidence
        sorted_probs = sorted((score.probability for score in scored), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        value = (0.55 * selected_score.confidence) + (0.45 * min(1.0, margin * 3.0))
        return round(coerce_float(value, self.min_confidence, minimum=self.min_confidence, maximum=self.max_confidence), 4)

    def _strategy_confidence(self, *, stats: StrategyHistoryStats, evidence_count: int) -> float:
        sample_factor = min(1.0, safe_ratio(stats.total, max(1, self.min_samples * 2), default=0.0))
        evidence_factor = min(1.0, evidence_count / 4.0)
        recency_factor = 0.1 if stats.recent_total > 0 else 0.0
        value = self.min_confidence + (0.38 * sample_factor) + (0.22 * evidence_factor) + recency_factor
        return round(coerce_float(value, self.min_confidence, minimum=self.min_confidence, maximum=self.max_confidence), 4)

    def _normalize_priors(self, priors: Mapping[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for strategy in self.allowed_strategies:
            normalized[strategy] = coerce_float(priors.get(strategy), 0.01, minimum=0.0)
        total = sum(normalized.values())
        if total <= 0:
            even = 1.0 / max(1, len(normalized))
            return {strategy: round(even, 6) for strategy in normalized}
        return {strategy: round(value / total, 6) for strategy, value in normalized.items()}

    def _normalize_weights(self) -> None:
        total = self.prior_weight + self.history_weight + self.rule_weight
        if total <= 0:
            self.prior_weight = 0.45
            self.history_weight = 0.35
            self.rule_weight = 0.20
            return
        self.prior_weight = round(self.prior_weight / total, 6)
        self.history_weight = round(self.history_weight / total, 6)
        self.rule_weight = round(self.rule_weight / total, 6)

    def _load_strategy_tuple_mapping(self, configured: Any, *, default: Mapping[str, Sequence[str]],
                                     allow_non_strategy_keys: bool = True) -> Dict[str, Tuple[str, ...]]:
        result: Dict[str, Tuple[str, ...]] = {}
        for key, values in default.items():
            result[str(key)] = tuple(self._unique_strategies(values, allow_unknown=allow_non_strategy_keys))
        if isinstance(configured, Mapping):
            for key, values in configured.items():
                normalized_values = tuple(self._unique_strategies(coerce_list(values, split_strings=True), allow_unknown=allow_non_strategy_keys))
                if normalized_values:
                    result[str(key)] = normalized_values
        return result

    def _load_nested_float_mapping(self, configured: Any, *, default: Mapping[str, Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {str(key): {str(k): float(v) for k, v in value.items()} for key, value in default.items()}
        if isinstance(configured, Mapping):
            for group, mapping in configured.items():
                if not isinstance(mapping, Mapping):
                    continue
                result[str(group)] = {str(key): coerce_float(value, 0.0, minimum=-1.0, maximum=1.0) for key, value in mapping.items()}
        return result

    def _unique_strategies(self, values: Iterable[Any], *, allow_unknown: bool = False) -> List[str]:
        unique: List[str] = []
        for value in values:
            strategy = normalize_identifier(value, default="")
            if not strategy:
                continue
            if not allow_unknown and strategy not in self.DEFAULT_ALLOWED_STRATEGIES and strategy not in getattr(self, "allowed_strategies", self.DEFAULT_ALLOWED_STRATEGIES):
                continue
            if strategy not in unique:
                unique.append(strategy)
        return unique

    def _validate_configuration(self) -> None:
        if self.min_confidence > self.max_confidence:
            raise ConfigurationError(
                "ProbabilisticStrategySelector min_confidence cannot exceed max_confidence",
                context={"min_confidence": self.min_confidence, "max_confidence": self.max_confidence},
                code="HANDLER_STRATEGY_CONFIDENCE_BOUNDS_INVALID",
                policy=self.error_policy,
            )
        if not self.allowed_strategies:
            raise ConfigurationError(
                "ProbabilisticStrategySelector requires at least one allowed strategy",
                code="HANDLER_STRATEGY_ALLOWED_EMPTY",
                policy=self.error_policy,
            )
        unknown_priors = [strategy for strategy in self.priors if strategy not in self.allowed_strategies]
        if unknown_priors:
            logger.warning("Strategy priors include strategies outside allowed_strategies: %s", unknown_priors)


if __name__ == "__main__":
    print("\n=== Running Probabilistic Strategy Selector ===\n")
    printer.status("TEST", "Probabilistic Strategy Selector initialized", "info")

    from .handler_memory import HandlerMemory

    strict_policy = HandlerErrorPolicy(
        name="strategy_selector.strict_test",
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
            "mirror_to_shared_memory": False,
        },
        error_policy=strict_policy,
    )

    failure = build_normalized_failure(
        error=TimeoutError("Network timeout while calling upstream with Authorization: Bearer token-123"),
        context={
            "route": "handler.recovery",
            "agent": "demo_agent",
            "task_id": "strategy-selector-smoke-001",
            "password": "SuperSecret123",
        },
        policy=strict_policy,
        source="handler.strategy_selector.__main__",
        correlation_id="corr-strategy-selector-test",
    )

    for index in range(4):
        memory.append_recovery_telemetry(
            failure=failure,
            recovery={
                "status": "recovered" if index < 3 else "failed",
                "strategy": "timeout",
                "attempts": 1 + index,
                "sla": {"remaining_seconds": 10.0},
            },
            context={
                "route": "handler.recovery",
                "agent": "demo_agent",
                "task_id": "strategy-selector-smoke-001",
                "correlation_id": "corr-strategy-selector-test",
            },
            insight={"signature": "timeout:test", "category": "timeout"},
        )

    selector = ProbabilisticStrategySelector(
        config={
            "max_candidates": 4,
            "min_samples": 2,
            "emit_to_memory": True,
            "strategy_priors": {"runtime": 0.15, "timeout": 0.35, "network": 0.25, "memory": 0.10, "dependency": 0.05, "resource": 0.05, "unicode": 0.05},
        },
        memory=memory,
        error_policy=strict_policy,
    )

    selection = selector.decide(
        normalized_failure=failure,
        context={"route": "handler.recovery", "agent": "demo_agent", "task_id": "strategy-selector-smoke-001"},
        insight={"category": "timeout", "action": HandlerRecoveryAction.RETRY.value},
    )
    legacy = selector.select(failure, telemetry_history=memory.recent_telemetry(limit=20))
    stats = selector.strategy_stats(strategy="timeout", telemetry_history=memory.recent_telemetry(limit=20), category="timeout")
    summary = selector.summarize(memory.recent_telemetry(limit=20))

    security_failure = build_normalized_failure(
        error=PermissionError("Security policy violation with token=abc123"),
        error_info={"severity": "critical", "retryable": False, "policy_action": HandlerRecoveryAction.QUARANTINE.value},
        context={"route": "handler.recovery", "agent": "demo_agent", "password": "SuperSecret123"},
        policy=strict_policy,
        correlation_id="corr-strategy-selector-security-test",
    )
    security_selection = selector.decide(normalized_failure=security_failure, context={"route": "handler.recovery", "agent": "demo_agent"})

    serialized = stable_json_dumps(
        {
            "selection": selection.to_dict(),
            "legacy": legacy,
            "stats": stats.to_dict(),
            "summary": summary,
            "security_selection": security_selection.to_dict(),
            "memory": memory.recent_telemetry(limit=20),
        }
    )

    assert selection.selected_strategy in {"timeout", "network", "runtime"}
    assert "selected_strategy" in legacy
    assert "distribution" in legacy
    assert "candidates" in legacy
    assert stats.total >= 4
    assert summary["telemetry"]["total"] >= 4
    assert security_selection.selected_strategy == "runtime"
    assert "SuperSecret123" not in serialized
    assert "token-123" not in serialized
    assert "abc123" not in serialized

    printer.pretty("Strategy selection", selection.to_dict(), "success")
    printer.pretty("Strategy stats", stats.to_dict(), "success")
    printer.pretty("Strategy summary", summary, "success")
    print("\n=== Test ran successfully ===\n")
