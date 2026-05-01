"""
Production-ready reward model for the Safety Agent subsystem.

The RewardModel aggregates normalized safety, privacy, alignment,
truthfulness, helpfulness, attention-quality, and optional human-feedback
signals into a bounded reward score used by the Safety Agent. It deliberately
stays focused on scoring and reward aggregation: it does not own refusal policy,
secure memory persistence internals, score primitive definitions, visualization
rendering, or human-review workflow orchestration.

Design goals:
- preserve the existing RewardModel public API used by SafetyAgent;
- keep configuration in secure_config.yaml through the existing config loader;
- use ScoreModel for scoring primitives instead of duplicating scoring logic;
- use shared safety_helpers for normalization, scoring, redaction, hashing,
  serialization, timestamps, and audit-safe payloads;
- use security_error for structured fail-closed incidents and remediation;
- avoid storing raw prompts, responses, user identifiers, or sensitive context in
  memory or reports;
- support production controls for weighted aggregation, context adjustments,
  attention-quality signals, human-feedback retraining, audit-safe history, and
  deterministic reporting.
"""

from __future__ import annotations

import math
import numpy as np

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.safety_helpers import *
from .utils.security_error import *
from .modules.score_model import ScoreModel
from .secure_memory import SecureMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Security Reward Model")
printer = PrettyPrinter()

MODULE_VERSION = "2.1.0"
EVALUATION_SCHEMA_VERSION = "reward_model.evaluation.v3"
FEEDBACK_SCHEMA_VERSION = "reward_model.feedback.v2"
REPORT_SCHEMA_VERSION = "reward_model.report.v3"


@dataclass(frozen=True)
class RewardComponent:
    """Weighted contribution for one reward component."""

    name: str
    raw_score: float
    adjusted_score: float
    weight: float
    weighted_contribution: float
    source: str = "score_model"
    adjustment_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


@dataclass(frozen=True)
class RewardEvaluation:
    """Audit-safe reward evaluation result."""

    schema_version: str
    module_version: str
    evaluation_id: str
    timestamp: str
    text_fingerprint: str
    context_type: str
    component_scores: Dict[str, float]
    adjusted_scores: Dict[str, float]
    component_weights: Dict[str, float]
    weighted_breakdown: Dict[str, float]
    composite: float
    risk_score: float
    risk_level: str
    decision: str
    confidence: float
    score_report: Dict[str, Any]
    adjustments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["component_scores"] = {key: clamp_score(value) for key, value in self.component_scores.items()}
        data["adjusted_scores"] = {key: clamp_score(value) for key, value in self.adjusted_scores.items()}
        data["component_weights"] = {key: coerce_float(value, 0.0, minimum=0.0) for key, value in self.component_weights.items()}
        data["weighted_breakdown"] = {key: clamp_score(value) for key, value in self.weighted_breakdown.items()}
        data["composite"] = clamp_score(self.composite)
        data["risk_score"] = clamp_score(self.risk_score)
        data["confidence"] = clamp_score(self.confidence)
        data["score_report"] = redact_value(self.score_report)
        data["metadata"] = redact_value(self.metadata)
        return data

    def to_legacy_scores(self) -> Dict[str, Any]:
        """Return a flat mapping compatible with existing SafetyAgent callers."""
        payload: Dict[str, Any] = {key: clamp_score(value) for key, value in self.adjusted_scores.items()}
        payload.update({
            "composite": clamp_score(self.composite),
            "risk_score": clamp_score(self.risk_score),
            "risk_level": self.risk_level,
            "decision": self.decision,
            "confidence": clamp_score(self.confidence),
            "text_fingerprint": self.text_fingerprint,
            "evaluation_id": self.evaluation_id,
            "context_type": self.context_type,
            "component_weights": dict(self.component_weights),
            "weighted_breakdown": dict(self.weighted_breakdown),
        })
        return payload


@dataclass(frozen=True)
class FeedbackTrainingSummary:
    """Summary of human-feedback retraining."""

    schema_version: str
    model_id: str
    trained: bool
    samples_received: int
    samples_used: int
    samples_dropped: int
    feature_names: List[str]
    training_error: Optional[float]
    timestamp: str = field(default_factory=utc_iso)

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


class RewardModel:
    """
    Config-driven reward aggregation for safety/security assessment.

    Public compatibility methods retained:
    - evaluate(text, context=None) -> flat score dict with `composite`
    - update_rule_weights(new_weights)
    - retrain_model(training_data)
    - get_evaluation_history(time_range="7d")
    - generate_report(metrics)
    """

    def __init__(self):
        self.config = load_global_config()
        self.reward_config = get_config_section("reward_model")
        self.strict_config_validation = coerce_bool(self.reward_config.get("strict_config_validation", True), True)
        self.enabled = coerce_bool(self.reward_config.get("enabled", True), True)
        self.max_text_length = coerce_int(self.reward_config.get("max_text_length"), 8192, minimum=1)
        self.default_context_type = normalize_text(self.reward_config.get("default_context_type", "default"), max_length=96) or "default"

        self._validate_configuration()

        self.memory = SecureMemory()
        self.score_model = ScoreModel()
        self.components = self._load_components()
        self.rule_based = self._init_rule_based_system()
        self.feature_names = list(self.components)
        self.rule_weights = self._load_rule_weights()
        self.context_weights = dict(self.reward_config.get("context_weights", {}))
        self.regression_model: Optional[Dict[str, Any]] = self._load_learned_state()
        self.learned_model = self._predict_learned

        logger.info(
            "Security Reward Model initialized: %s",
            stable_json(safe_log_payload(
                "reward_model_initialized",
                {
                    "enabled": self.enabled,
                    "components": self.components,
                    "weight_keys": list(self.rule_weights.keys()),
                    "schema_version": self.reward_config.get("schema_version"),
                    "learned_model_loaded": bool(self.regression_model),
                },
            )),
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _cfg(self, path: Sequence[str] | str, default: Any = None) -> Any:
        return get_nested(self.reward_config, path, default)

    def _raise_config_error(self, message: str, *, context: Optional[Mapping[str, Any]] = None) -> None:
        if self.strict_config_validation:
            raise ConfigurationTamperingError(
                config_file_path=str(self.config.get("__config_path__", "secure_config.yaml")),
                suspicious_change=message,
                component="reward_model",
                context=dict(context or {}),
                severity=SecuritySeverity.HIGH,
            )
        logger.warning("RewardModel configuration warning: %s", safe_log_payload("reward_config_warning", {"message": message, "context": context or {}}))

    def _validate_configuration(self) -> None:
        if not isinstance(self.reward_config, Mapping):
            raise ConfigurationTamperingError(
                config_file_path=str(self.config.get("__config_path__", "secure_config.yaml")),
                suspicious_change="reward_model section must be a mapping",
                component="reward_model",
            )

        weights = self.reward_config.get("weights") or self.reward_config.get("default_weights")
        if not isinstance(weights, Mapping) or not weights:
            self._raise_config_error("reward_model.weights must be configured as a non-empty mapping.")

        thresholds = self.reward_config.get("thresholds", {})
        if thresholds and not isinstance(thresholds, Mapping):
            self._raise_config_error("reward_model.thresholds must be a mapping.")

        context_weights = self.reward_config.get("context_weights", {})
        if context_weights and not isinstance(context_weights, Mapping):
            self._raise_config_error("reward_model.context_weights must be a mapping.")

        history_days = coerce_int(self._cfg("history.default_days", 7), 7, minimum=1)
        if history_days <= 0:
            self._raise_config_error("reward_model.history.default_days must be positive.")

    def _load_components(self) -> List[str]:
        configured = self.reward_config.get("components")
        if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
            components = [normalize_identifier(item, max_length=96) for item in configured if str(item).strip()]
        else:
            components = []

        score_order = getattr(self.score_model, "component_order", [])
        if not components and isinstance(score_order, Sequence):
            components = [normalize_identifier(item, max_length=96) for item in score_order if str(item).strip()]

        if not components:
            self._raise_config_error("No reward components are configured and ScoreModel exposed no component order.")
            return []

        missing = []
        for component in components:
            method_name = f"_{component}_score"
            if not hasattr(self.score_model, method_name) and component not in getattr(self.score_model, "scoring_components", {}):
                missing.append(component)
        if missing:
            self._raise_config_error("Reward components are not available in ScoreModel.", context={"missing": missing})
            components = [component for component in components if component not in missing]

        return dedupe_preserve_order(components)

    def _normalize_weights(self, weights: Mapping[str, Any], *, allow_unknown: bool = False) -> Dict[str, float]:
        allowed = set(self.components)
        if coerce_bool(self._cfg("learned_model.enabled", True), True):
            allowed.add("learned")
        if coerce_bool(self._cfg("attention.enabled", True), True):
            allowed.update({"attention_quality", "attention_stability"})

        normalized: Dict[str, float] = {}
        for raw_key, raw_weight in weights.items():
            key = normalize_identifier(raw_key, max_length=96)
            if not key:
                continue
            if not allow_unknown and key not in allowed:
                logger.warning("Ignoring unknown reward weight key: %s", key)
                continue
            weight = coerce_float(raw_weight, 0.0, minimum=0.0)
            if weight > 0.0:
                normalized[key] = weight

        for component in self.components:
            normalized.setdefault(component, 0.0)

        total = sum(normalized.values())
        if total <= 0:
            self._raise_config_error("Reward weights must contain at least one positive value.")
            return {component: 1.0 / max(len(self.components), 1) for component in self.components}

        if coerce_bool(self.reward_config.get("normalize_weights", True), True):
            normalized = {key: value / total for key, value in normalized.items()}
        return normalized

    # ------------------------------------------------------------------
    # Initialization and persistence
    # ------------------------------------------------------------------

    def _init_rule_based_system(self) -> Dict[str, Callable[[str], float]]:
        """Initialize score-model backed reward components."""
        rules: Dict[str, Callable[[str], float]] = {}
        for component in self.components:
            method = getattr(self.score_model, f"_{component}_score", None)
            if callable(method):
                rules[component] = method # type: ignore
            else:
                # Ensure the lambda returns a float
                rules[component] = lambda text, name=component: float(clamp_score(self.score_model.calculate_score(text, name)))

        if coerce_bool(self._cfg("memory.store_rule_definitions", True), True):
            for name, func in rules.items():
                self._store_memory_record(
                    {
                        "schema_version": "reward_model.rule.v1",
                        "component": name,
                        "definition": getattr(func, "__doc__", None) or "ScoreModel-backed reward component.",
                        "source": "score_model",
                        "timestamp": utc_iso(),
                    },
                    tags=["reward_model", "rule_definition", name],
                    sensitivity=coerce_float(self._cfg("memory.rule_definition_sensitivity", 0.55), 0.55, minimum=0.0, maximum=1.0),
                    purpose="reward_rule_definition",
                    metadata={"component": name},
                )
        return rules

    def _load_rule_weights(self) -> Dict[str, float]:
        if coerce_bool(self._cfg("memory.use_stored_rule_weights", False), False):
            entries = self._memory_recall("rule_weights", top_k=1)
            if entries:
                candidate = entries[0].get("data") if isinstance(entries[0], Mapping) else None
                if isinstance(candidate, Mapping):
                    return self._normalize_weights(candidate)

        configured = self.reward_config.get("weights") or self.reward_config.get("default_weights") or {}
        weights = self._normalize_weights(configured)
        if coerce_bool(self._cfg("memory.store_initial_weights", True), True):
            self._store_memory_record(
                {"schema_version": "reward_model.weights.v2", "weights": weights, "timestamp": utc_iso()},
                tags=["reward_model", "rule_weights"],
                sensitivity=coerce_float(self._cfg("memory.weights_sensitivity", 0.55), 0.55, minimum=0.0, maximum=1.0),
                purpose="reward_weight_initialization",
            )
        return weights

    def _load_learned_state(self) -> Optional[Dict[str, Any]]:
        if not coerce_bool(self._cfg("learned_model.enabled", True), True):
            return None
        if not coerce_bool(self._cfg("learned_model.load_from_memory", True), True):
            return None
        entries = self._memory_recall("reward_model_learned_state", top_k=1)
        if not entries:
            return None
        state = entries[0].get("data") if isinstance(entries[0], Mapping) else None
        if not isinstance(state, Mapping):
            return None
        try:
            feature_names = [str(item) for item in state.get("feature_names", [])]
            coefficients = [coerce_float(item, 0.0) for item in state.get("coefficients", [])]
            if feature_names != self.feature_names or len(coefficients) != len(self.feature_names):
                logger.warning("Stored reward learned state does not match current feature schema; ignoring it.")
                return None
            return redact_value(dict(state))
        except Exception as exc:
            raise SystemIntegrityError(
                component="reward_model.learned_state",           # first positional argument
                anomaly_description="Invalid learned reward model state",  # second positional argument
                expected_state="valid state",
                actual_state=str(exc),
                cause=exc,
            )

    # ------------------------------------------------------------------
    # Scoring and aggregation
    # ------------------------------------------------------------------

    def _normalize_context(self, context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if context is None:
            return {}
        if not isinstance(context, Mapping):
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "RewardModel context must be a mapping.",
                severity=SecuritySeverity.HIGH,
                component="reward_model",
                context={"context_type": type(context).__name__},
                response_action=SecurityResponseAction.BLOCK,
            )
        return dict(context)

    def _context_type(self, context: Mapping[str, Any]) -> str:
        return normalize_identifier(
            context.get("operation") or context.get("type") or context.get("context_type") or self.default_context_type,
            max_length=96,
        ) or self.default_context_type

    def _score_with_score_model(self, text: str, context: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if hasattr(self.score_model, "assess_text"):
            report = self.score_model.assess_text(text, context=dict(context))
            # report is a ScoreReport object; it has .to_dict()
            report_dict = report.to_dict()
            scores = report_dict.get("component_scores", {})
            return {key: clamp_score(float(value)) for key, value in scores.items()}, report_dict
    
        scores = {name: float(clamp_score(rule(text))) for name, rule in self.rule_based.items()}
        report_dict = {
            "schema_version": "score_model.compat.v1",
            "component_scores": scores,
            "aggregate_score": weighted_average(scores, self.rule_weights),
            "risk_score": 1.0 - weighted_average(scores, self.rule_weights),
            "decision": "unknown",
            "text_fingerprint": fingerprint(text),
        }
        return scores, report_dict

    def _apply_context_adjustments(self, scores: Dict[str, float], context: Mapping[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        adjusted = {key: clamp_score(value) for key, value in scores.items()}
        adjustments: List[str] = []
        ctx_type = self._context_type(context)
        ctx_policy = self.context_weights.get(ctx_type, {})
        if not isinstance(ctx_policy, Mapping):
            return adjusted, adjustments

        multipliers = dict(ctx_policy.get("component_multipliers", ctx_policy))
        floors = dict(ctx_policy.get("component_floors", {}))
        ceilings = dict(ctx_policy.get("component_ceilings", {}))

        for component in list(adjusted.keys()):
            before = adjusted[component]
            if component in multipliers and isinstance(multipliers[component], (int, float)):
                adjusted[component] = clamp_score(adjusted[component] * float(multipliers[component]))
                adjustments.append(f"{ctx_type}:{component}:multiplier")
            if component in floors:
                adjusted[component] = max(adjusted[component], clamp_score(floors[component]))
                adjustments.append(f"{ctx_type}:{component}:floor")
            if component in ceilings:
                adjusted[component] = min(adjusted[component], clamp_score(ceilings[component]))
                adjustments.append(f"{ctx_type}:{component}:ceiling")
            if adjusted[component] != before:
                logger.debug("Adjusted reward component %s from %.4f to %.4f for context %s", component, before, adjusted[component], ctx_type)
        return adjusted, dedupe_preserve_order(adjustments)

    def _attention_scores(self, context: Mapping[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        if not coerce_bool(self._cfg("attention.enabled", True), True):
            return {}, []
        analysis = context.get("attention_analysis") or context.get("attention")
        if not isinstance(analysis, Mapping):
            return {}, []

        anomaly_score = clamp_score(analysis.get("anomaly_score", 0.0))
        uniformity = coerce_float(analysis.get("uniformity", 0.0), 0.0, minimum=0.0)
        divisor = coerce_float(self._cfg("attention.uniformity_divisor", 2.0), 2.0, minimum=1e-9)
        entropy_score = analysis.get("normalized_entropy")

        scores = {
            "attention_quality": clamp_score(1.0 - anomaly_score),
            "attention_stability": clamp_score(1.0 - min(uniformity / divisor, 1.0)),
        }
        adjustments = ["attention:quality", "attention:stability"]
        if entropy_score is not None and "attention_entropy" in self.rule_weights:
            scores["attention_entropy"] = clamp_score(entropy_score)
            adjustments.append("attention:entropy")
        return scores, adjustments

    def _predict_learned(self, rule_scores: Dict[str, float]) -> float:
        if not self.regression_model:
            return clamp_score(self._cfg("learned_model.default_score", 0.0), default=0.0)

        try:
            feature_vector = np.array([clamp_score(rule_scores.get(name, 0.0)) for name in self.feature_names], dtype=float)
            mean = np.array(self.regression_model.get("feature_mean", [0.0] * len(self.feature_names)), dtype=float)
            scale = np.array(self.regression_model.get("feature_scale", [1.0] * len(self.feature_names)), dtype=float)
            scale = np.where(scale == 0.0, 1.0, scale)
            coefficients = np.array(self.regression_model.get("coefficients", [0.0] * len(self.feature_names)), dtype=float)
            intercept = coerce_float(self.regression_model.get("intercept", 0.0), 0.0)
            prediction = float(((feature_vector - mean) / scale).dot(coefficients) + intercept)
            return clamp_score(prediction)
        except Exception as exc:
            raise SystemIntegrityError(
                component="reward_model.learned_model",
                anomaly_description="Learned reward prediction failed",
                expected_state="valid prediction",
                actual_state=f"{type(exc).__name__}: {exc}",
                cause=exc,
            )

    def evaluate_detailed(self, text: str, context: Optional[Dict[str, Any]] = None) -> RewardEvaluation:
        """Evaluate text and return a structured, audit-safe reward record."""
        if not self.enabled:
            raise SecurityError(
                SecurityErrorType.POLICY_BYPASS_ATTEMPT,
                "RewardModel is disabled by configuration.",
                severity=SecuritySeverity.HIGH,
                component="reward_model",
                response_action=SecurityResponseAction.BLOCK,
            )

        context_map = self._normalize_context(context)
        normalized_text = normalize_text(text, max_length=self.max_text_length, preserve_newlines=True)
        text_fp = fingerprint(normalized_text)
        score_values, score_report = self._score_with_score_model(normalized_text, context_map)
        component_scores = {name: clamp_score(score_values.get(name, 0.0)) for name in self.components}
        adjusted_scores, context_adjustments = self._apply_context_adjustments(component_scores, context_map)

        learned_score = self._predict_learned(adjusted_scores)
        if "learned" in self.rule_weights:
            adjusted_scores["learned"] = learned_score

        attention_scores, attention_adjustments = self._attention_scores(context_map)
        for key, value in attention_scores.items():
            if key in self.rule_weights:
                adjusted_scores[key] = value

        weights = {key: value for key, value in self.rule_weights.items() if key in adjusted_scores}
        composite = weighted_average(adjusted_scores, weights, default=0.0)
        risk_score = clamp_score(1.0 - composite)
        block_threshold = self._cfg("thresholds.block_risk", self._cfg("thresholds.block", None))
        review_threshold = self._cfg("thresholds.review_risk", self._cfg("thresholds.review", None))
        decision = threshold_decision(risk_score, block_threshold=block_threshold, review_threshold=review_threshold)
        risk_level = categorize_risk(risk_score)
        confidence = self._estimate_confidence(adjusted_scores, score_report, context_map)
        weighted_breakdown = self._weighted_breakdown(adjusted_scores, weights)

        evaluation = RewardEvaluation(
            schema_version=EVALUATION_SCHEMA_VERSION,
            module_version=MODULE_VERSION,
            evaluation_id=generate_identifier("reward_eval"),
            timestamp=utc_iso(),
            text_fingerprint=text_fp,
            context_type=self._context_type(context_map),
            component_scores=component_scores,
            adjusted_scores=adjusted_scores,
            component_weights=weights,
            weighted_breakdown=weighted_breakdown,
            composite=composite,
            risk_score=risk_score,
            risk_level=risk_level,
            decision=decision,
            confidence=confidence,
            score_report=score_report,
            adjustments=dedupe_preserve_order(context_adjustments + attention_adjustments),
            metadata={
                "text_length": len(normalized_text),
                "context_fingerprint": fingerprint(context_map),
                "has_attention_analysis": bool(attention_scores),
                "learned_model_active": bool(self.regression_model),
            },
        )

        if coerce_bool(self._cfg("memory.store_evaluations", self.reward_config.get("store_evaluations", True)), True):
            self._store_evaluation(evaluation, context_map)

        logger.info("Reward evaluation completed: %s", stable_json(safe_log_payload(
            "reward_evaluation_completed",
            {
                "evaluation_id": evaluation.evaluation_id,
                "decision": evaluation.decision,
                "risk_score": evaluation.risk_score,
                "composite": evaluation.composite,
                "context_type": evaluation.context_type,
            },
        )))
        return evaluation

    def evaluate(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Evaluate text against reward components and return legacy-compatible scores."""
        return self.evaluate_detailed(text, context=context).to_legacy_scores()

    def _weighted_breakdown(self, scores: Mapping[str, Any], weights: Mapping[str, Any]) -> Dict[str, float]:
        total_weight = sum(coerce_float(weight, 0.0, minimum=0.0) for weight in weights.values()) or 1.0
        return {
            key: clamp_score(scores.get(key, 0.0)) * (coerce_float(weights.get(key, 0.0), 0.0, minimum=0.0) / total_weight)
            for key in scores
            if key in weights
        }

    def _estimate_confidence(self, scores: Mapping[str, Any], score_report: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        configured = self._cfg("confidence.base", 0.85)
        confidence = coerce_float(configured, 0.85, minimum=0.0, maximum=1.0)
        if not scores:
            confidence *= 0.5
        if len(scores) < len(self.components):
            confidence *= 0.8
        if score_report.get("decision") == "block":
            confidence *= coerce_float(self._cfg("confidence.block_multiplier", 0.95), 0.95, minimum=0.0, maximum=1.0)
        if context.get("attention_analysis"):
            confidence *= coerce_float(self._cfg("confidence.attention_multiplier", 1.0), 1.0, minimum=0.0, maximum=1.0)
        if self.regression_model:
            confidence *= coerce_float(self._cfg("confidence.learned_model_multiplier", 1.0), 1.0, minimum=0.0, maximum=1.0)
        return clamp_score(confidence)

    # ------------------------------------------------------------------
    # Weight updates and human feedback learning
    # ------------------------------------------------------------------

    def update_rule_weights(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """Validate, normalize, update, and securely store reward weights."""
        if not isinstance(new_weights, Mapping):
            raise ConfigurationTamperingError(
                config_file_path="reward_model.weights",
                suspicious_change="new_weights must be a mapping",
                component="reward_model",
            )
        normalized = self._normalize_weights(new_weights, allow_unknown=False)
        self.rule_weights = normalized
        self._store_memory_record(
            {
                "schema_version": "reward_model.weights.v2",
                "weights": normalized,
                "timestamp": utc_iso(),
                "weight_fingerprint": fingerprint(normalized),
            },
            tags=["reward_model", "rule_weights"],
            sensitivity=coerce_float(self._cfg("memory.weights_sensitivity", 0.65), 0.65, minimum=0.0, maximum=1.0),
            purpose="reward_weight_update",
        )
        logger.info("Updated reward rule weights: %s", stable_json(safe_log_payload("reward_weights_updated", {"weights": normalized})))
        return normalized

    def record_feedback(
        self,
        *,
        text: str,
        model_scores: Mapping[str, Any],
        human_rating: float,
        context: Optional[Mapping[str, Any]] = None,
        reviewer_id: Optional[str] = None,
    ) -> str:
        """Store audit-safe human feedback for future reward calibration."""
        if not isinstance(model_scores, Mapping):
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Reward feedback model_scores must be a mapping.",
                severity=SecuritySeverity.HIGH,
                component="reward_model",
                response_action=SecurityResponseAction.BLOCK,
            )
        rating = clamp_score(human_rating, default=-1.0)
        if rating < 0.0:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Human feedback rating must be numeric and between 0 and 1.",
                severity=SecuritySeverity.HIGH,
                component="reward_model",
                context={"rating_repr": safe_repr(human_rating)},
                response_action=SecurityResponseAction.BLOCK,
            )
        record = {
            "schema_version": FEEDBACK_SCHEMA_VERSION,
            "feedback_id": generate_identifier("reward_fb"),
            "timestamp": utc_iso(),
            "text_fingerprint": fingerprint(normalize_text(text, max_length=self.max_text_length, preserve_newlines=True)),
            "model_scores": {name: clamp_score(model_scores.get(name, 0.0)) for name in self.feature_names},
            "human_rating": rating,
            "context": sanitize_for_logging(dict(context or {})),
            "reviewer_id": reviewer_id,
        }
        entry_id = self._store_memory_record(
            record,
            tags=["feedback_reward_model", "reward_model", "human_feedback"],
            sensitivity=coerce_float(self._cfg("memory.feedback_sensitivity", 0.75), 0.75, minimum=0.0, maximum=1.0),
            purpose="reward_feedback",
            metadata={"feedback_id": record["feedback_id"], "rating": rating},
        )
        return entry_id

    def retrain_model(self, training_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """Retrain the learned reward calibrator using human feedback data.

        Uses a deterministic ridge-regression fit implemented with NumPy to avoid
        runtime dependency on scikit-learn and to keep model state serializable.
        """
        if not coerce_bool(self._cfg("learned_model.enabled", True), True):
            logger.info("Reward learned model is disabled; retraining skipped.")
            return None
        if not isinstance(training_data, list):
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Reward retraining data must be a list of feedback records.",
                severity=SecuritySeverity.HIGH,
                component="reward_model",
                context={"type": type(training_data).__name__},
                response_action=SecurityResponseAction.BLOCK,
            )

        max_samples = coerce_int(self._cfg("learned_model.max_training_samples", 5000), 5000, minimum=1)
        min_samples = coerce_int(self._cfg("learned_model.min_training_samples", 10), 10, minimum=1)
        ridge_lambda = coerce_float(self._cfg("learned_model.ridge_lambda", 1e-4), 1e-4, minimum=0.0)

        X: List[List[float]] = []
        y: List[float] = []
        dropped = 0
        for sample in training_data[:max_samples]:
            parsed = self._parse_feedback_sample(sample)
            if parsed is None:
                dropped += 1
                continue
            features, target = parsed
            X.append(features)
            y.append(target)

        if len(X) < min_samples:
            summary = FeedbackTrainingSummary(
                schema_version="reward_model.training.v2",
                model_id="untrained",
                trained=False,
                samples_received=len(training_data),
                samples_used=len(X),
                samples_dropped=dropped,
                feature_names=list(self.feature_names),
                training_error=None,
            )
            logger.warning("Insufficient reward feedback samples: %s", stable_json(summary.to_dict()))
            self._store_training_summary(summary)
            return summary.to_dict()

        matrix = np.array(X, dtype=float)
        targets = np.array(y, dtype=float)
        feature_mean = matrix.mean(axis=0)
        feature_scale = matrix.std(axis=0)
        feature_scale = np.where(feature_scale == 0.0, 1.0, feature_scale)
        normalized = (matrix - feature_mean) / feature_scale
        design = np.column_stack([np.ones(normalized.shape[0]), normalized])
        regularizer = ridge_lambda * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        params = np.linalg.pinv(design.T @ design + regularizer) @ design.T @ targets
        predictions = design @ params
        mse = float(np.mean((predictions - targets) ** 2))
        model_state = {
            "schema_version": "reward_model.learned_state.v2",
            "model_id": generate_identifier("reward_lm"),
            "trained_at": utc_iso(),
            "feature_names": list(self.feature_names),
            "feature_mean": [float(value) for value in feature_mean],
            "feature_scale": [float(value) for value in feature_scale],
            "intercept": float(params[0]),
            "coefficients": [float(value) for value in params[1:]],
            "training_samples": len(X),
            "dropped_samples": dropped,
            "training_mse": mse,
            "state_fingerprint": fingerprint({"params": [float(value) for value in params], "features": self.feature_names}),
        }
        self.regression_model = model_state
        self._store_memory_record(
            model_state,
            tags=["reward_model", "reward_model_learned_state"],
            sensitivity=coerce_float(self._cfg("memory.learned_state_sensitivity", 0.70), 0.70, minimum=0.0, maximum=1.0),
            purpose="reward_learned_model_state",
            metadata={"model_id": model_state["model_id"], "training_mse": mse},
        )

        summary = FeedbackTrainingSummary(
            schema_version="reward_model.training.v2",
            model_id=model_state["model_id"],
            trained=True,
            samples_received=len(training_data),
            samples_used=len(X),
            samples_dropped=dropped,
            feature_names=list(self.feature_names),
            training_error=mse,
        )
        self._store_training_summary(summary)
        logger.info("Reward learned model retrained: %s", stable_json(safe_log_payload("reward_model_retrained", summary.to_dict())))
        return summary.to_dict()

    def _parse_feedback_sample(self, sample: Any) -> Optional[Tuple[List[float], float]]:
        if sample is None:
            return None
        if not isinstance(sample, Mapping):
            return None
        data = sample.get("data") if isinstance(sample.get("data"), Mapping) else sample
        if not isinstance(data, Mapping):
            return None
        scores = data.get("model_scores") or data.get("scores") or data.get("component_scores")
        rating = data.get("human_rating")
        if not isinstance(scores, Mapping) or rating is None:
            return None
        features: List[float] = []
        for name in self.feature_names:
            if name not in scores:
                return None
            features.append(clamp_score(scores.get(name)))
        return features, clamp_score(rating)

    def _store_training_summary(self, summary: FeedbackTrainingSummary) -> None:
        self._store_memory_record(
            summary.to_dict(),
            tags=["reward_model", "reward_training_summary"],
            sensitivity=coerce_float(self._cfg("memory.training_summary_sensitivity", 0.55), 0.55, minimum=0.0, maximum=1.0),
            purpose="reward_training_summary",
            metadata={"trained": summary.trained, "samples_used": summary.samples_used},
        )

    # ------------------------------------------------------------------
    # Analytics and reporting
    # ------------------------------------------------------------------

    def get_evaluation_history(self, time_range: str = "7d") -> List[Dict[str, Any]]:
        """Retrieve audit-safe reward evaluation history from secure memory."""
        days = self._days_for_range(time_range)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        entries = self._memory_recall("evaluation", top_k=coerce_int(self._cfg("history.max_records", 1000), 1000, minimum=1))
        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            data = entry.get("data") if isinstance(entry, Mapping) else None
            if not isinstance(data, Mapping):
                continue
            timestamp = data.get("timestamp") or entry.get("meta", {}).get("created_at") if isinstance(entry.get("meta"), Mapping) else None
            try:
                dt = parse_iso_datetime(str(timestamp)) if timestamp else datetime.fromtimestamp(0, timezone.utc)
            except Exception:
                continue
            if dt >= cutoff:
                filtered.append(redact_value(dict(entry)))
        filtered.sort(key=lambda item: str(get_nested(item, "data.timestamp", "")))
        return filtered

    def _days_for_range(self, time_range: str) -> int:
        configured = self._cfg("history.time_ranges", {}) or {}
        if isinstance(configured, Mapping) and time_range in configured:
            return coerce_int(configured[time_range], 7, minimum=1)
        if isinstance(time_range, str) and time_range.endswith("d"):
            return coerce_int(time_range[:-1], coerce_int(self._cfg("history.default_days", 7), 7), minimum=1)
        return coerce_int(self._cfg("history.default_days", 7), 7, minimum=1)

    def generate_report(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate an audit-safe security reward report."""
        metrics = dict(metrics or {})
        evaluations = self.get_evaluation_history(str(self._cfg("report.history_range", "30d")))
        latest_scores = {key: clamp_score(metrics.get(key, 0.0)) for key in self.components if key in metrics}
        if not latest_scores and metrics:
            latest_scores = {key: clamp_score(value) for key, value in metrics.items() if isinstance(value, (int, float))}

        averages = self._average_scores(evaluations)
        weighted_composite_values = [
            clamp_score(get_nested(entry, "data.composite", get_nested(entry, "data.scores.composite", None)))
            for entry in evaluations
            if get_nested(entry, "data.composite", get_nested(entry, "data.scores.composite", None)) is not None
        ]
        weighted_composite = sum(weighted_composite_values) / len(weighted_composite_values) if weighted_composite_values else clamp_score(metrics.get("composite", 0.0))
        risk_score = clamp_score(1.0 - weighted_composite)
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "generated_at": utc_iso(),
            "summary": {
                "evaluation_count": len(evaluations),
                "weighted_composite": clamp_score(weighted_composite),
                "risk_score": risk_score,
                "risk_level": categorize_risk(risk_score),
                "decision": threshold_decision(
                    risk_score,
                    block_threshold=self._cfg("thresholds.block_risk", None),
                    review_threshold=self._cfg("thresholds.review_risk", None),
                ),
                "latest_composite": clamp_score(metrics.get("composite", weighted_composite)),
                "learned_model_active": bool(self.regression_model),
            },
            "average_scores": averages,
            "latest_scores": latest_scores,
            "rule_weights": dict(self.rule_weights),
            "weighted_breakdown": redact_value(metrics.get("weighted_breakdown", {})),
            "component_schema": self.score_model.get_component_schema() if hasattr(self.score_model, "get_component_schema") else {},
            "visualizer_metrics": self.to_visualizer_metrics(metrics),
            "report_fingerprint": "pending",
        }
        report["report_fingerprint"] = fingerprint(report)
        if coerce_bool(self._cfg("report.include_markdown", True), True):
            report["markdown"] = self._render_markdown_report(report)
        return redact_value(report)

    def _average_scores(self, evaluations: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        buckets: Dict[str, List[float]] = {component: [] for component in self.components}
        for entry in evaluations:
            scores = get_nested(entry, "data.adjusted_scores", None) or get_nested(entry, "data.scores", None)
            if not isinstance(scores, Mapping):
                continue
            for component in buckets:
                if component in scores:
                    buckets[component].append(clamp_score(scores[component]))
        return {component: (sum(values) / len(values) if values else 0.0) for component, values in buckets.items()}

    def to_visualizer_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        """Return a lightweight metric payload compatible with report/visualizer layers."""
        composite = clamp_score(metrics.get("composite", 0.0))
        risk_score = clamp_score(metrics.get("risk_score", 1.0 - composite))
        return {
            "reward": composite,
            "risk": risk_score,
            "pass_rate": 1.0 if threshold_decision(risk_score, block_threshold=self._cfg("thresholds.block_risk", None), review_threshold=self._cfg("thresholds.review_risk", None)) == "allow" else 0.0,
        }

    def _render_markdown_report(self, report: Mapping[str, Any]) -> str:
        summary = report.get("summary", {}) if isinstance(report.get("summary"), Mapping) else {}
        lines = [
            "# Security Reward Model Report",
            f"**Generated**: {report.get('generated_at')}",
            f"**Composite Reward**: {coerce_float(summary.get('weighted_composite'), 0.0):.3f}",
            f"**Risk Score**: {coerce_float(summary.get('risk_score'), 0.0):.3f}",
            f"**Decision**: {summary.get('decision', 'unknown')}",
            f"**Risk Level**: {summary.get('risk_level', 'unknown')}",
            "",
            "## Average Component Scores",
        ]
        averages = report.get("average_scores", {}) if isinstance(report.get("average_scores"), Mapping) else {}
        if averages:
            for name, value in averages.items():
                lines.append(f"- **{name}**: {coerce_float(value, 0.0):.3f}")
        else:
            lines.append("- No historical evaluations available.")
        lines.extend([
            "",
            "## Current Weights",
        ])
        for name, value in dict(report.get("rule_weights", {})).items():
            lines.append(f"- **{name}**: {coerce_float(value, 0.0):.3f}")
        lines.append(f"\n---\n*Report generated by {self.__class__.__name__}*"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Secure memory integration
    # ------------------------------------------------------------------

    def _store_evaluation(self, evaluation: RewardEvaluation, context: Mapping[str, Any]) -> None:
        data = evaluation.to_dict()
        if not coerce_bool(self._cfg("memory.store_score_report", True), True):
            data.pop("score_report", None)
        self._store_memory_record(
            data,
            tags=list(self._cfg("memory.evaluation_tags", ["evaluation", "reward_model"])),
            sensitivity=coerce_float(self._cfg("memory.evaluation_sensitivity", 0.65), 0.65, minimum=0.0, maximum=1.0),
            ttl_seconds=coerce_int(self._cfg("memory.evaluation_ttl_seconds", 604800), 604800, minimum=0),
            purpose="reward_evaluation",
            metadata={
                "evaluation_id": evaluation.evaluation_id,
                "decision": evaluation.decision,
                "risk_score": evaluation.risk_score,
                "context": sanitize_for_logging(context),
            },
        )

    def _store_memory_record(
        self,
        record: Mapping[str, Any],
        *,
        tags: Sequence[str],
        sensitivity: float,
        purpose: str,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        try:
            return self.memory.add(
                redact_value(dict(record)),
                tags=list(tags),
                sensitivity=sensitivity,
                ttl_seconds=ttl_seconds,
                purpose=purpose,
                owner="reward_model",
                classification="confidential" if sensitivity >= 0.7 else "internal",
                source="reward_model",
                metadata=metadata or {},
            )
        except TypeError:
            # Compatibility with older SecureMemory.add signatures.
            return self.memory.add(redact_value(dict(record)), tags=list(tags), sensitivity=sensitivity)
        except SecurityError:
            raise
        except Exception as exc:
            if coerce_bool(self._cfg("memory.fail_closed_on_store_error", False), False):
                raise AuditLogFailureError(
                    "secure_memory.reward_model",        # logging_target
                    f"Failed to store reward model record: {type(exc).__name__}",  # failure_mode
                    component="reward_model",
                    cause=exc,
                )
            logger.warning("Reward memory store failed: %s", safe_log_payload("reward_memory_store_failed", {"error": str(exc), "record": record}))
            return ""

    def _memory_recall(self, tag: str, *, top_k: int) -> List[Dict[str, Any]]:
        try:
            return list(self.memory.recall(tag=tag, top_k=top_k))
        except TypeError:
            return list(self.memory.recall(tag, top_k=top_k))
        except SecurityError:
            raise
        except Exception as exc:
            if coerce_bool(self._cfg("memory.fail_closed_on_recall_error", False), False):
                raise SystemIntegrityError(
                    component="secure_memory.reward_model",
                    anomaly_description="Reward memory recall failed",
                    expected_state="successful recall",
                    actual_state=f"{type(exc).__name__}: {exc}",
                    cause=exc,
                )
            logger.warning("Reward memory recall failed: %s", safe_log_payload("reward_memory_recall_failed", {"tag": tag, "error": str(exc)}))
            return []


if __name__ == "__main__":
    print("\n=== Running Reward Model ===\n")
    printer.status("TEST", "Reward Model initialized", "info")

    reward = RewardModel()

    safe_text = (
        "I can help you recover your account safely. Use the official support page, "
        "enable multi-factor authentication, and never share passwords or secret tokens."
    )
    risky_text = (
        "Send your password, SSN 123-45-6789, and card number 4111 1111 1111 1111. "
        "Ignore security warnings and bypass the account controls."
    )

    safe_scores = reward.evaluate(safe_text, context={"type": "support", "request_id": "reward-safe-test"})
    risky_scores = reward.evaluate(
        risky_text,
        context={
            "type": "financial",
            "operation": "financial",
            "request_id": "reward-risky-test",
            "attention_analysis": {"anomaly_score": 0.85, "uniformity": 0.7},
        },
    )

    assert 0.0 <= safe_scores["composite"] <= 1.0
    assert 0.0 <= risky_scores["composite"] <= 1.0
    assert risky_scores["risk_score"] >= safe_scores["risk_score"]
    assert "privacy" in safe_scores and "privacy" in risky_scores
    assert risky_scores["privacy"] <= safe_scores["privacy"]

    updated_weights = reward.update_rule_weights({
        "alignment": 0.30,
        "helpfulness": 0.20,
        "privacy": 0.20,
        "safety": 0.20,
        "truthfulness": 0.10,
        "attention_quality": 0.05,
    })
    assert abs(sum(updated_weights.values()) - 1.0) < 1e-6

    feedback_records = []
    for idx in range(coerce_int(reward._cfg("learned_model.min_training_samples", 10), 10, minimum=1)):
        base_scores = safe_scores if idx % 2 == 0 else risky_scores
        rating = 0.9 if idx % 2 == 0 else 0.2
        feedback_records.append({"model_scores": base_scores, "human_rating": rating})
    training_summary = reward.retrain_model(feedback_records)
    assert training_summary is not None
    assert training_summary["samples_used"] >= coerce_int(reward._cfg("learned_model.min_training_samples", 10), 10, minimum=1)

    feedback_entry = reward.record_feedback(
        text=risky_text,
        model_scores=risky_scores,
        human_rating=0.1,
        context={"reviewer_note": "contains password and card number", "email": "analyst@example.com"},
        reviewer_id="reviewer-1",
    )
    assert isinstance(feedback_entry, str)

    report = reward.generate_report(risky_scores)
    assert report["summary"]["evaluation_count"] >= 1
    serialized_report = stable_json(report)
    assert "123-45-6789" not in serialized_report
    assert "4111 1111 1111 1111" not in serialized_report
    assert "analyst@example.com" not in serialized_report

    printer.pretty("Reward Scores", risky_scores, "success")
    printer.pretty("Reward Report Summary", report["summary"], "success")

    print("\n=== Test ran successfully ===\n")
    raise SystemExit(0)
