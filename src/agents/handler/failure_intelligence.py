from __future__ import annotations

import re

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.handler_error import *
from .utils.handler_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Failure Intelligence")
printer = PrettyPrinter()


@dataclass(frozen=True)
class FailureHistoryStats:
    """Historical recovery statistics for one failure signature/context."""

    total: int = 0
    recovered: int = 0
    failed: int = 0
    degraded: int = 0
    skipped: int = 0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "total": self.total,
                "recovered": self.recovered,
                "failed": self.failed,
                "degraded": self.degraded,
                "skipped": self.skipped,
                "success_rate": self.success_rate,
                "failure_rate": self.failure_rate,
                "first_seen": self.first_seen,
                "last_seen": self.last_seen,
            },
            drop_none=True,
        )


@dataclass(frozen=True)
class FailureInsight:
    """
    Production insight returned by FailureIntelligence.

    The first four fields intentionally preserve the legacy API used by HandlerAgent:
    signature, confidence, category, recommendation.
    """

    signature: str
    confidence: float
    category: str
    recommendation: str
    severity: str = FailureSeverity.LOW.value
    retryable: bool = False
    action: str = HandlerRecoveryAction.NONE.value
    risk_score: float = 0.0
    anomaly_score: float = 0.0
    recurrence_count: int = 0
    historical_total: int = 0
    historical_recovered: int = 0
    historical_failed: int = 0
    historical_success_rate: float = 0.0
    context_hash: Optional[str] = None
    root_cause: Optional[str] = None
    labels: Tuple[str, ...] = field(default_factory=tuple)
    evidence: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    related_signatures: Tuple[str, ...] = field(default_factory=tuple)
    next_actions: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "handler.failure_insight.v2"

    def to_dict(self) -> Dict[str, Any]:
        return compact_dict(
            {
                "schema": self.schema,
                "signature": self.signature,
                "confidence": self.confidence,
                "category": self.category,
                "recommendation": self.recommendation,
                "severity": self.severity,
                "retryable": self.retryable,
                "action": self.action,
                "risk_score": self.risk_score,
                "anomaly_score": self.anomaly_score,
                "recurrence_count": self.recurrence_count,
                "historical_total": self.historical_total,
                "historical_recovered": self.historical_recovered,
                "historical_failed": self.historical_failed,
                "historical_success_rate": self.historical_success_rate,
                "context_hash": self.context_hash,
                "root_cause": self.root_cause,
                "labels": list(self.labels),
                "evidence": [dict(item) for item in self.evidence],
                "related_signatures": list(self.related_signatures),
                "next_actions": list(self.next_actions),
                "metadata": dict(self.metadata),
            },
            drop_none=True,
            drop_empty=True,
        )


class FailureIntelligence:
    """
    Production failure intelligence for HandlerAgent.

    Scope:
    - consumes normalized Handler failure payloads
    - categorizes failures with configurable rules plus helper fallback
    - creates stable, redacted signatures for de-duplication and routing
    - estimates confidence, recurrence, anomaly, and operational risk
    - produces recommendation/action hints for recovery, escalation, and learning loops
    - integrates with HandlerMemory-like objects without importing them at module load time

    This module intentionally does not own shared helpers, telemetry storage, or recovery execution.
    """

    DEFAULT_RECOMMENDATION_BY_CATEGORY: Mapping[str, str] = {
        "security": "quarantine_security_failure",
        "timeout": "retry_with_backoff",
        "network": "retry_with_backoff",
        "memory": "degrade_and_reduce_resource_pressure",
        "dependency": "validate_runtime_dependencies",
        "resource": "degrade_and_failover_resource",
        "unicode": "sanitize_encoding_and_retry",
        "sla": "preserve_sla_and_fast_failover",
        "validation": "validate_payload_and_schema",
        "runtime": "collect_context_and_escalate",
    }

    DEFAULT_ACTION_BY_CATEGORY: Mapping[str, str] = {
        "security": HandlerRecoveryAction.QUARANTINE.value,
        "timeout": HandlerRecoveryAction.RETRY.value,
        "network": HandlerRecoveryAction.RETRY.value,
        "memory": HandlerRecoveryAction.DEGRADE.value,
        "dependency": HandlerRecoveryAction.ESCALATE.value,
        "resource": HandlerRecoveryAction.DEGRADE.value,
        "unicode": HandlerRecoveryAction.RETRY.value,
        "sla": HandlerRecoveryAction.DEGRADE.value,
        "validation": HandlerRecoveryAction.DEGRADE.value,
        "runtime": HandlerRecoveryAction.ESCALATE.value,
    }

    DEFAULT_NEXT_ACTIONS_BY_RECOMMENDATION: Mapping[str, Tuple[str, ...]] = {
        "quarantine_security_failure": (
            "quarantine_context",
            "block_retry",
            "escalate_to_safety_or_planning",
            "preserve_audit_payload",
        ),
        "immediate_escalation": (
            "stop_recovery_loop",
            "escalate_to_planning_or_safety",
            "attach_failure_context",
        ),
        "retry_with_backoff": (
            "retry_with_adaptive_backoff",
            "check_circuit_breaker",
            "preserve_checkpoint_before_retry",
        ),
        "open_circuit_or_failover": (
            "open_or_extend_circuit_breaker",
            "failover_or_degrade",
            "escalate_if_sla_budget_low",
        ),
        "degrade_and_reduce_resource_pressure": (
            "switch_to_lightweight_mode",
            "reduce_batch_or_context_size",
            "retry_only_if_sla_allows",
        ),
        "degrade_and_failover_resource": (
            "reduce_resource_pressure",
            "failover_resource_pool",
            "retry_after_cooldown",
        ),
        "validate_runtime_dependencies": (
            "validate_import_path_and_package_version",
            "capture_environment_snapshot",
            "escalate_to_maintenance_or_planning",
        ),
        "sanitize_encoding_and_retry": (
            "sanitize_payload_encoding",
            "remove_unsupported_unicode_tokens",
            "retry_once_with_clean_payload",
        ),
        "preserve_sla_and_fast_failover": (
            "skip_expensive_retry",
            "degrade_or_fast_failover",
            "emit_sla_postmortem",
        ),
        "validate_payload_and_schema": (
            "validate_required_fields",
            "reject_or_repair_payload",
            "avoid_blind_retry",
        ),
        "collect_context_and_escalate": (
            "collect_runtime_context",
            "attempt_safe_fallback_if_retryable",
            "escalate_with_postmortem",
        ),
    }

    DEFAULT_CATEGORY_RULES: Mapping[str, Tuple[str, ...]] = {
        "security": (
            "security",
            "unauthorized",
            "forbidden",
            "permission denied",
            "credential",
            "token",
            "policy violation",
        ),
        "timeout": ("timeout", "timed out", "deadline exceeded"),
        "network": ("network", "connection", "socket", "dns", "http", "ssl", "tls"),
        "memory": ("memory", "oom", "outofmemory", "out of memory", "cuda"),
        "dependency": ("dependency", "import", "module", "dll", "package", "no module named", "cannot import"),
        "resource": ("resource", "busy", "quota", "rate limit", "cpu", "gpu", "disk"),
        "unicode": ("unicode", "codec", "encode", "decode"),
        "sla": ("sla", "deadline", "latency budget", "budget exhausted"),
        "validation": ("validation", "invalid", "schema", "missing required", "malformed"),
    }

    DEFAULT_CATEGORY_RISK: Mapping[str, float] = {
        "security": 0.95,
        "memory": 0.76,
        "dependency": 0.70,
        "resource": 0.62,
        "sla": 0.68,
        "runtime": 0.58,
        "timeout": 0.42,
        "network": 0.40,
        "validation": 0.44,
        "unicode": 0.30,
    }

    DEFAULT_SEVERITY_RISK: Mapping[str, float] = {
        FailureSeverity.LOW.value: 0.18,
        FailureSeverity.MEDIUM.value: 0.42,
        FailureSeverity.HIGH.value: 0.72,
        FailureSeverity.CRITICAL.value: 0.96,
    }

    _VOLATILE_TEXT_PATTERNS: Tuple[Tuple[str, str], ...] = (
        (r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<uuid>"),
        (r"\b[0-9a-f]{16,}\b", "<hex>"),
        (r"\b\d{4,}\b", "<number>"),
        (r"\b\d+(?:\.\d+)?\b", "<number>"),
        (r"[A-Za-z]:\\[^\s]+", "<path>"),
        (r"/(?:[^\s/]+/){2,}[^\s]+", "<path>"),
        (r"\s+", " "),
    )

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        memory: Any = None,
        error_policy: Optional[HandlerErrorPolicy] = None,
    ):
        self.config = load_global_config()
        self.intelligence_config = get_config_section("intelligence")

        merged = deep_merge(self.intelligence_config, config or {})
        self.enabled = coerce_bool(merged.get("enabled"), default=True)
        self.max_message_chars = coerce_int(merged.get("max_message_chars"), 280, minimum=40, maximum=8000)
        self.max_evidence_items = coerce_int(merged.get("max_evidence_items"), 8, minimum=1, maximum=50)
        self.max_related_signatures = coerce_int(merged.get("max_related_signatures"), 5, minimum=0, maximum=50)
        self.signature_length = coerce_int(merged.get("signature_length"), 16, minimum=8, maximum=64)
        self.signature_mode = coerce_str(merged.get("signature_mode"), default="structural").lower()
        self.signature_context_fields = tuple(
            coerce_list(
                merged.get("signature_context_fields", ("route", "agent", "task_id")),
                split_strings=True,
            )
        )
        self.default_history_limit = coerce_int(merged.get("default_history_limit"), 500, minimum=1, maximum=100_000)
        self.recurrence_window_seconds = coerce_float(merged.get("recurrence_window_seconds"), 3600.0, minimum=1.0)
        self.recurrence_threshold = coerce_int(merged.get("recurrence_threshold"), 3, minimum=1)
        self.min_historical_samples = coerce_int(merged.get("min_historical_samples"), 3, minimum=1)
        self.low_success_rate_threshold = coerce_float(merged.get("low_success_rate_threshold"), 0.25, minimum=0.0, maximum=1.0)
        self.high_risk_threshold = coerce_float(merged.get("high_risk_threshold"), 0.72, minimum=0.0, maximum=1.0)
        self.high_anomaly_threshold = coerce_float(merged.get("high_anomaly_threshold"), 0.68, minimum=0.0, maximum=1.0)
        self.confidence_floor = coerce_float(merged.get("confidence_floor"), 0.35, minimum=0.0, maximum=1.0)
        self.confidence_ceiling = coerce_float(merged.get("confidence_ceiling"), 0.92, minimum=0.0, maximum=1.0)
        if self.confidence_floor > self.confidence_ceiling:
            raise ConfigurationError(
                "FailureIntelligence confidence_floor cannot exceed confidence_ceiling",
                context={"confidence_floor": self.confidence_floor, "confidence_ceiling": self.confidence_ceiling},
                code="HANDLER_INTELLIGENCE_CONFIDENCE_BOUNDS_INVALID",
                policy=error_policy or HandlerErrorPolicy(),
            )

        self.category_rules = self._load_category_rules(merged.get("category_rules"))
        self.recommendation_by_category = deep_merge(self.DEFAULT_RECOMMENDATION_BY_CATEGORY, coerce_mapping(merged.get("recommendation_by_category")))
        self.action_by_category = deep_merge(self.DEFAULT_ACTION_BY_CATEGORY, coerce_mapping(merged.get("action_by_category")))
        self.next_actions_by_recommendation = self._load_next_actions(merged.get("next_actions_by_recommendation"))
        self.category_risk = deep_merge(self.DEFAULT_CATEGORY_RISK, coerce_mapping(merged.get("category_risk")))
        self.severity_risk = deep_merge(self.DEFAULT_SEVERITY_RISK, coerce_mapping(merged.get("severity_risk")))
        self.memory = memory
        policy_config = merged.get("error_policy")
        self.error_policy = error_policy or HandlerErrorPolicy.from_mapping(policy_config if isinstance(policy_config, Mapping) else None)

        logger.info(
            "Failure Intelligence initialized | enabled=%s signature_mode=%s history_limit=%s",
            self.enabled,
            self.signature_mode,
            self.default_history_limit,
        )

    def analyze(
        self,
        normalized_failure: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
    ) -> FailureInsight:
        """Analyze a normalized Handler failure and return a production insight."""
        if not self.enabled:
            return self._disabled_insight(normalized_failure=normalized_failure, context=context)

        try:
            if not isinstance(normalized_failure, Mapping):
                raise ValidationError(
                    "FailureIntelligence expected normalized_failure to be a mapping",
                    context={"actual_type": type(normalized_failure).__name__},
                    code="HANDLER_INTELLIGENCE_FAILURE_MAPPING_REQUIRED",
                    policy=self.error_policy,
                )

            context_payload = coerce_mapping(context)
            failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
            history = self._resolve_history(telemetry_history=telemetry_history, failure=failure)

            failure_type = coerce_str(failure.get("type"), default=DEFAULT_FAILURE_TYPE, max_chars=240)
            message = truncate_text(self.error_policy.sanitize_message(failure.get("message", "")), self.max_message_chars)
            severity = normalize_severity(failure.get("severity"))
            retryable = coerce_bool(failure.get("retryable"), default=False)
            category, category_evidence = self._categorize(
                failure_type=failure_type,
                failure_message=message,
                fallback=failure.get("category"),
            )
            context_hash = failure.get("context_hash")
            signature = self._signature(
                failure_type=failure_type,
                failure_message=message,
                category=category,
                severity=severity,
                context=context_payload,
            )
            stats = self._history_stats(
                telemetry_history=history,
                signature=signature,
                context_hash=context_hash,
                category=category,
                context=context_payload,
            )
            recurrence_count = self._recurrence_count(history=history, signature=signature, context_hash=context_hash)
            related_signatures = self._related_signatures(history=history, category=category, signature=signature, context=context_payload)
            risk_score = self._risk_score(
                category=category,
                severity=severity,
                retryable=retryable,
                stats=stats,
                recurrence_count=recurrence_count,
            )
            anomaly_score = self._anomaly_score(
                severity=severity,
                stats=stats,
                recurrence_count=recurrence_count,
                related_signature_count=len(related_signatures),
            )
            confidence = self._confidence(
                category=category,
                severity=severity,
                retryable=retryable,
                stats=stats,
                evidence_count=len(category_evidence),
                recurrence_count=recurrence_count,
            )
            recommendation = self._recommend(
                category=category,
                severity=severity,
                retryable=retryable,
                stats=stats,
                risk_score=risk_score,
                anomaly_score=anomaly_score,
            )
            action = self._action_for(category=category, severity=severity, retryable=retryable, recommendation=recommendation)
            next_actions = self._next_actions_for(recommendation)
            labels = self._labels(
                category=category,
                severity=severity,
                retryable=retryable,
                stats=stats,
                recurrence_count=recurrence_count,
                risk_score=risk_score,
                anomaly_score=anomaly_score,
            )
            evidence = self._evidence(
                failure=failure,
                context=context_payload,
                category_evidence=category_evidence,
                stats=stats,
                recurrence_count=recurrence_count,
                risk_score=risk_score,
                anomaly_score=anomaly_score,
            )

            return FailureInsight(
                signature=signature,
                confidence=confidence,
                category=category,
                recommendation=recommendation,
                severity=severity,
                retryable=retryable,
                action=action,
                risk_score=risk_score,
                anomaly_score=anomaly_score,
                recurrence_count=recurrence_count,
                historical_total=stats.total,
                historical_recovered=stats.recovered,
                historical_failed=stats.failed,
                historical_success_rate=stats.success_rate,
                context_hash=str(context_hash) if context_hash else None,
                root_cause=self._root_cause(category=category, failure_type=failure_type, message=message),
                labels=tuple(labels),
                evidence=tuple(evidence),
                related_signatures=tuple(related_signatures),
                next_actions=tuple(next_actions),
                metadata=compact_dict(
                    {
                        "failure_type": failure_type,
                        "route": context_payload.get("route"),
                        "agent": context_payload.get("agent"),
                        "task_id": context_payload.get("task_id"),
                        "policy_action": failure.get("policy_action"),
                        "signature_mode": self.signature_mode,
                        "history_window_seconds": self.recurrence_window_seconds,
                    },
                    drop_none=True,
                    drop_empty=True,
                ),
            )
        except HandlerError:
            raise
        except Exception as exc:
            raise IntelligenceError(
                "FailureIntelligence failed to analyze handler failure",
                cause=exc,
                context={"normalized_failure_type": type(normalized_failure).__name__},
                code="HANDLER_INTELLIGENCE_ANALYZE_FAILED",
                policy=self.error_policy,
            ) from exc

    def analyze_exception(
        self,
        error: BaseException,
        *,
        error_info: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> FailureInsight:
        """Normalize and analyze a raw exception without duplicating HandlerAgent normalization."""
        failure = build_normalized_failure(
            error=error,
            error_info=error_info,
            context=context,
            policy=self.error_policy,
            source=source,
            correlation_id=correlation_id,
        )
        return self.analyze(failure, context=context, telemetry_history=telemetry_history)

    def analyze_many(
        self,
        failures: Iterable[Mapping[str, Any]],
        *,
        context: Optional[Mapping[str, Any]] = None,
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[FailureInsight]:
        """Analyze multiple failures against one telemetry history snapshot."""
        history = self._resolve_history(telemetry_history=telemetry_history)
        return [self.analyze(failure, context=context, telemetry_history=history) for failure in failures if isinstance(failure, Mapping)]

    def summarize_history(
        self,
        telemetry_history: Optional[List[Dict[str, Any]]] = None,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarize failure intelligence signals from telemetry history."""
        history = self._resolve_history(telemetry_history=telemetry_history)
        if limit is not None:
            history = recent_events(history, limit=limit)

        category_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        recommendation_counts: Counter[str] = Counter()
        signature_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()

        for event in history:
            failure = coerce_mapping(event.get("failure"))
            insight = coerce_mapping(event.get("insight"))
            recovery = coerce_mapping(event.get("recovery"))
            category_counts[str(insight.get("category") or failure.get("category") or "unknown")] += 1
            severity_counts[str(failure.get("severity") or "unknown")] += 1
            recommendation_counts[str(insight.get("recommendation") or "unknown")] += 1
            if insight.get("signature"):
                signature_counts[str(insight.get("signature"))] += 1
            status_counts[str(recovery.get("status") or "unknown")] += 1

        return {
            "schema": "handler.failure_intelligence.summary.v2",
            "timestamp": utc_timestamp(),
            "total_events": len(history),
            "recovery": success_rate_for_events(history),
            "category_counts": dict(category_counts),
            "severity_counts": dict(severity_counts),
            "recommendation_counts": dict(recommendation_counts),
            "status_counts": dict(status_counts),
            "top_signatures": dict(signature_counts.most_common(10)),
        }

    def route_decision(self, insight: FailureInsight) -> Dict[str, Any]:
        """Build a compact routing hint for HandlerAgent/escalation consumers."""
        action = normalize_recovery_action(insight.action)
        target = "handler_agent"
        if action == HandlerRecoveryAction.QUARANTINE.value:
            target = "safety_agent"
        elif action in {HandlerRecoveryAction.FAIL_FAST.value, HandlerRecoveryAction.ESCALATE.value}:
            target = "planning_agent"
        elif action == HandlerRecoveryAction.DEGRADE.value:
            target = "handler_agent"

        return {
            "schema": "handler.failure_intelligence.route_decision.v2",
            "signature": insight.signature,
            "category": insight.category,
            "severity": insight.severity,
            "action": action,
            "target_agent": target,
            "recommendation": insight.recommendation,
            "risk_score": insight.risk_score,
            "confidence": insight.confidence,
            "next_actions": list(insight.next_actions),
        }

    def attach_memory(self, memory: Any) -> None:
        """Attach a HandlerMemory-like object after construction."""
        self.memory = memory

    def _disabled_insight(self, *, normalized_failure: Mapping[str, Any], context: Optional[Mapping[str, Any]]) -> FailureInsight:
        failure = normalize_failure_payload(normalized_failure, policy=self.error_policy)
        category = failure.get("category") or classify_failure_category(failure.get("type"), failure.get("message"))
        signature = self._signature(
            failure_type=failure.get("type", DEFAULT_FAILURE_TYPE),
            failure_message=failure.get("message", ""),
            category=category,
            severity=normalize_severity(failure.get("severity")),
            context=coerce_mapping(context),
        )
        return FailureInsight(
            signature=signature,
            confidence=self.confidence_floor,
            category=category,
            recommendation="failure_intelligence_disabled",
            severity=normalize_severity(failure.get("severity")),
            retryable=coerce_bool(failure.get("retryable")),
            context_hash=failure.get("context_hash"),
            labels=("disabled",),
            next_actions=("use_default_handler_policy",),
        )

    def _load_category_rules(self, configured_rules: Any) -> Dict[str, Tuple[str, ...]]:
        rules: Dict[str, Tuple[str, ...]] = {key: tuple(values) for key, values in self.DEFAULT_CATEGORY_RULES.items()}
        if isinstance(configured_rules, Mapping):
            for category, values in configured_rules.items():
                tokens = tuple(str(item).lower() for item in coerce_list(values, split_strings=True) if str(item).strip())
                if tokens:
                    rules[str(category).lower()] = tokens
        return rules

    def _load_next_actions(self, configured: Any) -> Dict[str, Tuple[str, ...]]:
        actions: Dict[str, Tuple[str, ...]] = {key: tuple(value) for key, value in self.DEFAULT_NEXT_ACTIONS_BY_RECOMMENDATION.items()}
        if isinstance(configured, Mapping):
            for recommendation, values in configured.items():
                normalized_values = tuple(str(item) for item in coerce_list(values, split_strings=True) if str(item).strip())
                if normalized_values:
                    actions[str(recommendation)] = normalized_values
        return actions

    def _resolve_history(self, *, telemetry_history: Optional[List[Dict[str, Any]]],
                         failure: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        if telemetry_history is not None:
            return [dict(event) for event in telemetry_history if isinstance(event, Mapping)]
    
        if self.memory is None:
            return []
    
        if failure is not None and hasattr(self.memory, "failure_history") and callable(self.memory.failure_history):
            history = self.memory.failure_history(context_hash=failure.get("context_hash"), limit=self.default_history_limit)
            if isinstance(history, (list, tuple)):
                return [dict(event) for event in history if isinstance(event, Mapping)]
            # Log unexpected type but continue
            if history is not None:
                logger.warning("failure_history returned non-iterable type: %s", type(history).__name__)
    
        if hasattr(self.memory, "recent_telemetry") and callable(self.memory.recent_telemetry):
            history = self.memory.recent_telemetry(limit=self.default_history_limit)
            if isinstance(history, (list, tuple)):
                return [dict(event) for event in history if isinstance(event, Mapping)]
            if history is not None:
                logger.warning("recent_telemetry returned non-iterable type: %s", type(history).__name__)
    
        return []

    def _categorize(self, *, failure_type: str, failure_message: str, fallback: Optional[Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        lowered = f"{failure_type} {failure_message}".lower()
        evidence: List[Dict[str, Any]] = []

        for category, tokens in self.category_rules.items():
            matched = [token for token in tokens if token and token in lowered]
            if matched:
                evidence.append({"signal": "category_rule", "category": category, "matched_terms": matched[:6]})
                return str(category), evidence

        fallback_category = coerce_str(fallback, default="")
        if fallback_category:
            evidence.append({"signal": "provided_category", "category": fallback_category})
            return fallback_category, evidence

        category = classify_failure_category(failure_type, failure_message)
        evidence.append({"signal": "helper_classifier", "category": category})
        return category, evidence

    def _signature(self, *, failure_type: str, failure_message: str, category: str,
                   severity: str, context: Mapping[str, Any]) -> str:
        message_fingerprint = self._canonical_message(failure_message)
        context_payload = select_keys(context, self.signature_context_fields, include_missing=False)
        payload = compact_dict(
            {
                "type": failure_type.lower(),
                "category": category,
                "message": message_fingerprint,
                "severity": severity if self.signature_mode == "strict" else None,
                "context": context_payload,
            },
            drop_none=True,
            drop_empty=True,
        )
        return f"{category}:{stable_hash(payload, length=self.signature_length, policy=self.error_policy)}"

    def _canonical_message(self, message: Any) -> str:
        text = coerce_str(message, default="", max_chars=self.max_message_chars).lower()
        if self.signature_mode in {"exact", "strict"}:
            return text
        for pattern, replacement in self._VOLATILE_TEXT_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text.strip()

    def _history_stats(self, *, telemetry_history: Sequence[Mapping[str, Any]], signature: str,
                       context_hash: Optional[Any], category: str, context: Mapping[str, Any]) -> FailureHistoryStats:
        total = recovered = failed = degraded = skipped = 0
        timestamps: List[float] = []
        route = context.get("route")
        agent = context.get("agent")

        for event in telemetry_history:
            if not isinstance(event, Mapping):
                continue
            failure = coerce_mapping(event.get("failure"))
            insight = coerce_mapping(event.get("insight"))
            event_context = coerce_mapping(event.get("context"))
            exact_match = event_matches_failure(event, context_hash=str(context_hash) if context_hash else None, signature=signature)
            same_signature = insight.get("signature") == signature or failure.get("signature") == signature
            same_category = (insight.get("category") or failure.get("category")) == category
            same_route_agent = True
            if route and event_context.get("route") and event_context.get("route") != route:
                same_route_agent = False
            if agent and event_context.get("agent") and event_context.get("agent") != agent:
                same_route_agent = False

            if not (exact_match or same_signature or (same_category and same_route_agent)):
                continue

            total += 1
            recovery = coerce_mapping(event.get("recovery"))
            status = str(recovery.get("status") or "unknown").lower()
            if status == "recovered":
                recovered += 1
            elif status == "degraded":
                degraded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
            timestamps.append(coerce_float(event.get("timestamp"), 0.0, minimum=0.0))

        return FailureHistoryStats(
            total=total,
            recovered=recovered,
            failed=failed,
            degraded=degraded,
            skipped=skipped,
            success_rate=round(safe_ratio(recovered, total, default=0.0, minimum=0.0, maximum=1.0), 4),
            failure_rate=round(safe_ratio(failed, total, default=0.0, minimum=0.0, maximum=1.0), 4),
            first_seen=min(timestamps) if timestamps else None,
            last_seen=max(timestamps) if timestamps else None,
        )

    def _recurrence_count(self, *, history: Sequence[Mapping[str, Any]], signature: str,
                          context_hash: Optional[Any]) -> int:
        now = utc_timestamp()
        cutoff = now - self.recurrence_window_seconds
        count = 0
        for event in history:
            if coerce_float(event.get("timestamp"), 0.0) < cutoff:
                continue
            if event_matches_failure(event, context_hash=str(context_hash) if context_hash else None, signature=signature):
                count += 1
        return count

    def _related_signatures(self, *, history: Sequence[Mapping[str, Any]], category: str,
                            signature: str, context: Mapping[str, Any]) -> List[str]:
        if self.max_related_signatures <= 0:
            return []
        route = context.get("route")
        agent = context.get("agent")
        counts: Counter[str] = Counter()
        for event in history:
            insight = coerce_mapping(event.get("insight"))
            event_context = coerce_mapping(event.get("context"))
            event_signature = insight.get("signature")
            event_category = insight.get("category") or coerce_mapping(event.get("failure")).get("category")
            if not event_signature or event_signature == signature or event_category != category:
                continue
            if route and event_context.get("route") and event_context.get("route") != route:
                continue
            if agent and event_context.get("agent") and event_context.get("agent") != agent:
                continue
            counts[str(event_signature)] += 1
        return [item for item, _count in counts.most_common(self.max_related_signatures)]

    def _risk_score(self, *, category: str, severity: str, retryable: bool,
                    stats: FailureHistoryStats, recurrence_count: int) -> float:
        category_component = coerce_float(self.category_risk.get(category), 0.55, minimum=0.0, maximum=1.0)
        severity_component = coerce_float(self.severity_risk.get(severity), 0.5, minimum=0.0, maximum=1.0)
        failure_component = stats.failure_rate if stats.total >= self.min_historical_samples else 0.35
        recurrence_component = min(1.0, recurrence_count / max(1, self.recurrence_threshold * 2))
        retry_component = 0.0 if retryable else 0.18
        score = (0.34 * category_component) + (0.34 * severity_component) + (0.18 * failure_component) + (0.14 * recurrence_component) + retry_component
        return round(coerce_float(score, 0.0, minimum=0.0, maximum=1.0), 4)

    def _anomaly_score(self, *, severity: str, stats: FailureHistoryStats, recurrence_count: int,
                       related_signature_count: int ) -> float:
        novelty_component = 0.75 if stats.total == 0 else max(0.0, 1.0 - min(1.0, stats.total / max(1, self.min_historical_samples * 3)))
        recurrence_component = min(1.0, recurrence_count / max(1, self.recurrence_threshold))
        severity_component = coerce_float(self.severity_risk.get(severity), 0.5, minimum=0.0, maximum=1.0)
        related_component = min(1.0, related_signature_count / max(1, self.max_related_signatures or 1))
        score = (0.30 * novelty_component) + (0.30 * recurrence_component) + (0.25 * severity_component) + (0.15 * related_component)
        return round(coerce_float(score, 0.0, minimum=0.0, maximum=1.0), 4)

    def _confidence(self, *, category: str, severity: str, retryable: bool, stats: FailureHistoryStats,
                    evidence_count: int, recurrence_count: int) -> float:
        history_confidence = min(1.0, stats.total / max(1, self.min_historical_samples * 2))
        evidence_confidence = min(1.0, evidence_count / max(1, self.max_evidence_items))
        severity_confidence = 0.08 if severity in {FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value} else 0.0
        recurrence_confidence = min(0.12, recurrence_count / max(1, self.recurrence_threshold) * 0.12)
        retry_penalty = -0.03 if not retryable else 0.02
        category_confidence = 0.72 if category in self.category_rules else 0.55
        value = (0.45 * category_confidence) + (0.30 * history_confidence) + (0.15 * evidence_confidence) + severity_confidence + recurrence_confidence + retry_penalty
        return round(coerce_float(value, 0.5, minimum=self.confidence_floor, maximum=self.confidence_ceiling), 3)

    def _recommend(self, *, category: str, severity: str, retryable: bool, stats: FailureHistoryStats,
                   risk_score: float, anomaly_score: float) -> str:
        if category == "security":
            return "quarantine_security_failure"
        if severity == FailureSeverity.CRITICAL.value and not retryable:
            return "immediate_escalation"
        if stats.total >= self.min_historical_samples and stats.success_rate <= self.low_success_rate_threshold:
            if category in {"timeout", "network", "resource"}:
                return "open_circuit_or_failover"
            return "collect_context_and_escalate"
        if risk_score >= self.high_risk_threshold and not retryable:
            return "immediate_escalation"
        if anomaly_score >= self.high_anomaly_threshold and severity in {FailureSeverity.HIGH.value, FailureSeverity.CRITICAL.value}:
            return "collect_context_and_escalate"
        return str(self.recommendation_by_category.get(category, "collect_context_and_escalate"))

    def _action_for(self, *, category: str, severity: str, retryable: bool, recommendation: str) -> str:
        if recommendation == "quarantine_security_failure":
            return HandlerRecoveryAction.QUARANTINE.value
        if recommendation == "immediate_escalation":
            return HandlerRecoveryAction.FAIL_FAST.value
        if recommendation in {"collect_context_and_escalate", "open_circuit_or_failover"} and not retryable:
            return HandlerRecoveryAction.ESCALATE.value
        if retryable and recommendation in {"retry_with_backoff", "sanitize_encoding_and_retry"}:
            return HandlerRecoveryAction.RETRY.value
        if severity == FailureSeverity.CRITICAL.value:
            return HandlerRecoveryAction.FAIL_FAST.value
        return normalize_recovery_action(self.action_by_category.get(category), default=HandlerRecoveryAction.ESCALATE)

    def _next_actions_for(self, recommendation: str) -> Tuple[str, ...]:
        return tuple(self.next_actions_by_recommendation.get(recommendation, self.DEFAULT_NEXT_ACTIONS_BY_RECOMMENDATION["collect_context_and_escalate"]))

    def _labels(
        self,
        *,
        category: str,
        severity: str,
        retryable: bool,
        stats: FailureHistoryStats,
        recurrence_count: int,
        risk_score: float,
        anomaly_score: float,
    ) -> List[str]:
        labels = [category, severity]
        if retryable:
            labels.append("retryable")
        else:
            labels.append("non_retryable")
        if stats.total == 0:
            labels.append("novel")
        elif stats.total >= self.min_historical_samples:
            labels.append("known_pattern")
        if recurrence_count >= self.recurrence_threshold:
            labels.append("recurrent")
        if risk_score >= self.high_risk_threshold:
            labels.append("high_risk")
        if anomaly_score >= self.high_anomaly_threshold:
            labels.append("anomalous")
        if stats.total >= self.min_historical_samples and stats.success_rate <= self.low_success_rate_threshold:
            labels.append("low_recovery_success")
        return list(dict.fromkeys(labels))

    def _evidence(
        self,
        *,
        failure: Mapping[str, Any],
        context: Mapping[str, Any],
        category_evidence: Sequence[Mapping[str, Any]],
        stats: FailureHistoryStats,
        recurrence_count: int,
        risk_score: float,
        anomaly_score: float,
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = [dict(item) for item in category_evidence]
        evidence.extend(
            [
                {"signal": "severity", "value": failure.get("severity")},
                {"signal": "retryable", "value": failure.get("retryable")},
                {"signal": "history", **stats.to_dict()},
                {"signal": "recurrence", "window_seconds": self.recurrence_window_seconds, "count": recurrence_count},
                {"signal": "risk", "risk_score": risk_score, "anomaly_score": anomaly_score},
                {"signal": "context", **select_keys(context, ("route", "agent", "task_id", "priority"), include_missing=False)},
            ]
        )
        return [make_json_safe(self.error_policy.sanitize_context(item)) for item in evidence[: self.max_evidence_items]]  # type: ignore[list-item]

    @staticmethod
    def _root_cause(*, category: str, failure_type: str, message: str) -> str:
        if category == "security":
            return "policy_or_credential_boundary"
        if category == "timeout":
            return "deadline_or_upstream_latency"
        if category == "network":
            return "connectivity_or_transport_failure"
        if category == "memory":
            return "memory_pressure_or_resource_exhaustion"
        if category == "dependency":
            return "runtime_dependency_or_import_failure"
        if category == "resource":
            return "resource_contention_or_quota_pressure"
        if category == "unicode":
            return "encoding_or_payload_character_set"
        if category == "sla":
            return "sla_budget_exhaustion"
        if category == "validation":
            return "payload_contract_or_schema_failure"
        lowered = f"{failure_type} {message}".lower()
        if "runtime" in lowered:
            return "unhandled_runtime_failure"
        return "undetermined_runtime_pattern"


if __name__ == "__main__":
    print("\n=== Running Failure Intelligence ===\n")
    printer.status("TEST", "Failure Intelligence initialized", "info")

    from .handler_memory import HandlerMemory

    strict_policy = HandlerErrorPolicy(
        name="failure_intelligence.strict_test",
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
            "max_postmortems": 10,
            "sanitize_payloads": True,
            "mirror_to_shared_memory": False,
        },
        error_policy=strict_policy,
    )

    context = {
        "task_id": "failure-intelligence-smoke-001",
        "route": "handler.recovery",
        "agent": "demo_agent",
        "priority": "high",
        "correlation_id": "corr-failure-intelligence-test",
        "password": "SuperSecret123",
    }

    failure = build_normalized_failure(
        error=TimeoutError("Upstream timed out with Authorization: Bearer token-123 after 31000ms"),
        context=context,
        policy=strict_policy,
        source="handler.failure_intelligence.__main__",
        correlation_id="corr-failure-intelligence-test",
    )

    previous_recovery = {
        "status": "failed",
        "strategy": "timeout",
        "attempts": 2,
        "sla": {"remaining_seconds": 1.5, "mode": "fast_failover"},
        "strategy_distribution": {"timeout": 0.80, "runtime": 0.20},
    }

    bootstrap_intelligence = FailureIntelligence(
        config={
            "default_history_limit": 10,
            "recurrence_threshold": 2,
            "recurrence_window_seconds": 3600,
        },
        memory=None,
        error_policy=strict_policy,
    )
    bootstrap_insight = bootstrap_intelligence.analyze(failure, context=context, telemetry_history=[])

    memory.append_recovery_telemetry(
        failure=failure,
        recovery=previous_recovery,
        context=context,
        insight=bootstrap_insight.to_dict(),
    )
    memory.append_recovery_telemetry(
        failure=failure,
        recovery={**previous_recovery, "status": "recovered", "attempts": 1},
        context=context,
        insight=bootstrap_insight.to_dict(),
    )

    intelligence = FailureIntelligence(
        config={
            "default_history_limit": 10,
            "recurrence_threshold": 2,
            "recurrence_window_seconds": 3600,
            "signature_context_fields": ["route", "agent", "task_id"],
        },
        memory=memory,
        error_policy=strict_policy,
    )

    insight = intelligence.analyze(failure, context=context)
    route = intelligence.route_decision(insight)
    summary = intelligence.summarize_history(memory.recent_telemetry(limit=10))

    converted = intelligence.analyze_exception(
        TimeoutError("Network connection timed out while calling upstream"),
        context={"route": "handler.recovery", "agent": "demo_agent", "authorization": "Bearer secret-token-456"},
        telemetry_history=memory.recent_telemetry(limit=10),
        source="handler.failure_intelligence.__main__",
    )

    serialized = stable_json_dumps(
        {
            "failure": failure,
            "bootstrap_insight": bootstrap_insight.to_dict(),
            "insight": insight.to_dict(),
            "route": route,
            "summary": summary,
            "converted": converted.to_dict(),
            "telemetry": memory.recent_telemetry(limit=10),
        }
    )

    assert insight.category == "timeout"
    assert insight.signature.startswith("timeout:")
    assert insight.recommendation in {"retry_with_backoff", "open_circuit_or_failover", "collect_context_and_escalate"}
    assert insight.recurrence_count >= 2
    assert insight.historical_total >= 2
    assert 0.0 <= insight.confidence <= 1.0
    assert route["signature"] == insight.signature
    assert summary["total_events"] == 2
    assert converted.category in {"timeout", "network"}
    assert "SuperSecret123" not in serialized
    assert "token-123" not in serialized
    assert "secret-token-456" not in serialized

    printer.pretty("Failure insight", insight.to_dict(), "success")
    printer.pretty("Route decision", route, "success")
    printer.pretty("History summary", summary, "success")
    print("\n=== Test ran successfully ===\n")
