"""Interpretable reward/risk aggregation for the SLAI Safety Agent subsystem."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .modules.score_model import ScoreModel
from .secure_memory import SecureMemory
from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Security Reward Model")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
EVALUATION_SCHEMA_VERSION = "reward_model.evaluation.v4"
FEEDBACK_SCHEMA_VERSION = "reward_model.feedback.v3"
REPORT_SCHEMA_VERSION = "reward_model.report.v4"


@dataclass(frozen=True)
class RewardComponent:
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
        data["component_scores"] = {k: clamp_score(v) for k, v in self.component_scores.items()}
        data["adjusted_scores"] = {k: clamp_score(v) for k, v in self.adjusted_scores.items()}
        data["component_weights"] = {k: max(0.0, coerce_float(v, 0.0)) for k, v in self.component_weights.items()}
        data["weighted_breakdown"] = {k: max(0.0, coerce_float(v, 0.0)) for k, v in self.weighted_breakdown.items()}
        data["composite"] = clamp_score(self.composite)
        data["risk_score"] = clamp_score(self.risk_score)
        data["confidence"] = clamp_score(self.confidence)
        data["score_report"] = redact_value(self.score_report)
        data["metadata"] = redact_value(self.metadata)
        return data

    def to_legacy_scores(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {k: clamp_score(v) for k, v in self.adjusted_scores.items()}
        payload.update({
            "composite": clamp_score(self.composite), "risk_score": clamp_score(self.risk_score),
            "risk_level": self.risk_level, "decision": self.decision, "confidence": clamp_score(self.confidence),
            "text_fingerprint": self.text_fingerprint, "evaluation_id": self.evaluation_id,
            "context_type": self.context_type, "component_weights": dict(self.component_weights),
            "weighted_breakdown": dict(self.weighted_breakdown),
        })
        return payload


@dataclass(frozen=True)
class FeedbackTrainingSummary:
    schema_version: str
    model_id: str
    trained: bool
    samples_received: int
    samples_used: int
    samples_dropped: int
    feature_names: List[str]
    training_error: Optional[float]
    validation_error: Optional[float] = None
    timestamp: str = field(default_factory=utc_iso)

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


class RewardModel:
    """ScoreModel-backed aggregation with optional validated calibration."""

    def __init__(self, *, memory: Optional[SecureMemory] = None, score_model: Optional[ScoreModel] = None) -> None:
        self.config = load_global_config()
        self.reward_config = get_config_section("reward_model")
        self.strict_config_validation = coerce_bool(self.reward_config.get("strict_config_validation", True), True)
        self.enabled = coerce_bool(self.reward_config.get("enabled", True), True)
        self.max_text_length = coerce_int(self.reward_config.get("max_text_length", 8192), 8192, minimum=1)
        self.default_context_type = normalize_identifier(self.reward_config.get("default_context_type", "default"), max_length=96, default="default")
        self.log_evaluations = coerce_bool(self.reward_config.get("log_evaluations", False), False)
        self.log_full_report = coerce_bool(self.reward_config.get("log_full_report", False), False)
        self.log_every_n = max(1, coerce_int(self.reward_config.get("log_every_n", 100), 100, minimum=1))
        self.cache_enabled = coerce_bool(self.reward_config.get("cache_enabled", True), True)
        self.cache_ttl_seconds = coerce_float(self.reward_config.get("cache_ttl_seconds", 2.0), 2.0, minimum=0.0)
        self.cache_max_entries = coerce_int(self.reward_config.get("cache_max_entries", 256), 256, minimum=1)
        self._validate_configuration()
        self.memory = memory or SecureMemory.shared()
        self.score_model = score_model or ScoreModel()
        self.components = self._load_components()
        self.feature_names = list(self.components)
        self.rule_weights = self._load_rule_weights()
        self.regression_model: Optional[Dict[str, Any]] = self._load_learned_state()
        self.learned_model = self._predict_learned  # backwards compatibility
        self._evaluation_cache: Dict[str, Tuple[float, RewardEvaluation]] = {}
        self._evaluation_count = 0
        self._state_lock = RLock()

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.reward_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.reward_config, Mapping):
            raise ConfigurationTamperingError("reward_model", "reward_model config must be a mapping", component="reward_model")
        weights = self.reward_config.get("weights") or self.reward_config.get("default_weights")
        if not isinstance(weights, Mapping) or not weights:
            raise ConfigurationTamperingError("reward_model.weights", "Reward weights must be a non-empty mapping", component="reward_model")
        thresholds = self.reward_config.get("thresholds", {}) or {}
        if clamp_score(thresholds.get("review_risk", thresholds.get("review", 0.45))) > clamp_score(thresholds.get("block_risk", thresholds.get("block", 0.72))):
            raise ConfigurationTamperingError("reward_model.thresholds", "review risk threshold must be <= block risk threshold", component="reward_model")

    def _load_components(self) -> List[str]:
        configured = self.reward_config.get("components")
        if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
            components = [normalize_identifier(v, max_length=96) for v in configured if str(v).strip()]
        else:
            components = [normalize_identifier(v, max_length=96) for v in getattr(self.score_model, "component_order", [])]
        if not components:
            raise ConfigurationTamperingError("reward_model.components", "No reward components are configured", component="reward_model")
        available = getattr(self.score_model, "scoring_components", {})
        missing = [name for name in components if name not in available and not hasattr(self.score_model, f"_{name}_score")]
        if missing:
            raise ConfigurationTamperingError("reward_model.components", f"Unknown ScoreModel components: {missing}", component="reward_model")
        return dedupe_preserve_order(components)

    def _normalize_weights(self, weights: Mapping[str, Any], *, allow_unknown: bool = False) -> Dict[str, float]:
        allowed = set(self.components) | {"attention_quality", "attention_stability", "attention_entropy"}
        output: Dict[str, float] = {}
        for key, raw in weights.items():
            name = normalize_identifier(key, max_length=96)
            # learned is calibration state, not an independent correlated signal.
            if name == "learned":
                continue
            if not allow_unknown and name not in allowed:
                continue
            value = coerce_float(raw, 0.0, minimum=0.0)
            if value > 0: output[name] = value
        for component in self.components: output.setdefault(component, 0.0)
        total = sum(output.values())
        if total <= 0:
            raise ConfigurationTamperingError("reward_model.weights", "At least one positive reward weight is required", component="reward_model")
        if coerce_bool(self.reward_config.get("normalize_weights", True), True):
            output = {k: v / total for k, v in output.items()}
        return output

    # ------------------------------------------------------------------
    # State and memory
    # ------------------------------------------------------------------
    def _memory_context(self, purpose: str) -> Dict[str, Any]:
        return self.memory.internal_context(purpose, principal="reward_model")

    def _memory_recall(self, tag: str, top_k: int = 1) -> List[Any]:
        try:
            return list(self.memory.recall(tag, top_k=top_k, access_context=self._memory_context("recall")))
        except Exception as exc:
            if coerce_bool(self._cfg("memory.fail_closed_on_recall_error", False), False):
                raise wrap_security_exception(exc, operation="reward_memory_recall", component="reward_model", error_type=SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, severity=SecuritySeverity.HIGH) from exc
            return []

    def _store_memory_record(self, payload: Mapping[str, Any], *, tags: List[str], sensitivity: float, purpose: str, metadata: Optional[Mapping[str, Any]] = None) -> str:
        try:
            return self.memory.add(dict(payload), tags=tags, sensitivity=sensitivity, ttl_seconds=self._cfg("memory.evaluation_ttl_seconds", 604800), purpose=purpose, owner="reward_model", source="reward_model", metadata={**dict(metadata or {}), "eligible_for_compliance": True})
        except Exception as exc:
            if coerce_bool(self._cfg("memory.fail_closed_on_store_error", False), False):
                raise AuditLogFailureError("secure_memory.reward_model", f"Reward memory store failed: {type(exc).__name__}", component="reward_model", cause=exc) from exc
            logger.warning("Reward memory store failed: %s", type(exc).__name__)
            return ""

    def _load_rule_weights(self) -> Dict[str, float]:
        if coerce_bool(self._cfg("memory.use_stored_rule_weights", False), False):
            entries = self._memory_recall("rule_weights", top_k=1)
            if entries:
                data = entries[0].get("data") if isinstance(entries[0], Mapping) else None
                candidate = data.get("weights") if isinstance(data, Mapping) and isinstance(data.get("weights"), Mapping) else data
                if isinstance(candidate, Mapping): return self._normalize_weights(candidate)
        return self._normalize_weights(self.reward_config.get("weights") or self.reward_config.get("default_weights") or {})

    def _load_learned_state(self) -> Optional[Dict[str, Any]]:
        if not coerce_bool(self._cfg("learned_model.enabled", True), True) or not coerce_bool(self._cfg("learned_model.load_from_memory", True), True):
            return None
        entries = self._memory_recall("reward_model_learned_state", top_k=1)
        if not entries: return None
        state = entries[0].get("data") if isinstance(entries[0], Mapping) else None
        if not isinstance(state, Mapping): return None
        if list(state.get("feature_names", [])) != self.feature_names:
            return None
        coefficients = list(state.get("coefficients", []))
        if len(coefficients) != len(self.feature_names): return None
        return dict(state)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _normalize_context(self, context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if context is None: return {}
        if not isinstance(context, Mapping):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "RewardModel context must be a mapping.", component="reward_model")
        return dict(context)

    def _context_type(self, context: Mapping[str, Any]) -> str:
        return normalize_identifier(context.get("operation") or context.get("type") or context.get("context_type") or self.default_context_type, max_length=96, default=self.default_context_type)

    def _score_with_score_model(self, text: str, context: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if hasattr(self.score_model, "assess_text"):
            report = self.score_model.assess_text(text, context=dict(context))
            report_dict: Dict[str, Any] = {}
            if hasattr(report, "to_dict"):
                serialized = report.to_dict()
                if isinstance(serialized, Mapping):
                    report_dict = dict(serialized)
            elif isinstance(report, Mapping):
                report_dict = dict(report)
            raw_scores = report_dict.get("component_scores", {})
            if not isinstance(raw_scores, Mapping):
                raw_scores = {}
            scores: Dict[str, float] = {
                str(name): clamp_score(value)
                for name, value in raw_scores.items()
            }
            return scores, report_dict
        scores = {name: clamp_score(self.score_model.calculate_score(text, name, context=dict(context))) for name in self.components}
        return scores, {"component_scores": scores, "schema_version": "score_model.compat.v2"}

    def _attention_scores(self, context: Mapping[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        if not coerce_bool(self._cfg("attention.enabled", True), True): return {}, []
        analysis = context.get("attention_analysis") or context.get("attention")
        if not isinstance(analysis, Mapping): return {}, []
        anomaly = clamp_score(analysis.get("anomaly_score", get_nested(analysis, "metrics.anomaly_score", 0.0)))
        uniformity = coerce_float(analysis.get("uniformity", get_nested(analysis, "metrics.uniformity", 0.0)), 0.0, minimum=0.0)
        divisor = coerce_float(self._cfg("attention.uniformity_divisor", 2.0), 2.0, minimum=1e-9)
        scores = {"attention_quality": clamp_score(1.0 - anomaly), "attention_stability": clamp_score(1.0 - min(uniformity / divisor, 1.0))}
        entropy = analysis.get("normalized_entropy", get_nested(analysis, "metrics.normalized_entropy", None))
        if entropy is not None: scores["attention_entropy"] = clamp_score(entropy)
        return scores, [f"attention:{name}" for name in scores]

    def _predict_learned(self, rule_scores: Dict[str, float]) -> float:
        if not self.regression_model:
            return clamp_score(self._cfg("learned_model.default_score", 0.0), default=0.0)
        vector = np.array([clamp_score(rule_scores.get(name, 0.0)) for name in self.feature_names], dtype=float)
        mean = np.array(self.regression_model.get("feature_mean", [0.0] * len(self.feature_names)), dtype=float)
        scale = np.array(self.regression_model.get("feature_scale", [1.0] * len(self.feature_names)), dtype=float)
        scale = np.where(scale == 0.0, 1.0, scale)
        coefficients = np.array(self.regression_model.get("coefficients", [0.0] * len(self.feature_names)), dtype=float)
        intercept = coerce_float(self.regression_model.get("intercept", 0.0), 0.0)
        return clamp_score(float(((vector - mean) / scale).dot(coefficients) + intercept))

    def _cache_key(self, text: str, context: Mapping[str, Any]) -> str:
        return fingerprint({"schema": EVALUATION_SCHEMA_VERSION, "text": fingerprint(text), "context": fingerprint(redact_value(context)), "weights": self.rule_weights, "model": (self.regression_model or {}).get("model_id")})

    def evaluate_detailed(self, text: str, context: Optional[Dict[str, Any]] = None) -> RewardEvaluation:
        if not self.enabled:
            raise SecurityError(SecurityErrorType.POLICY_BYPASS_ATTEMPT, "RewardModel is disabled.", component="reward_model", response_action=SecurityResponseAction.BLOCK)
        context_map = self._normalize_context(context)
        normalized_text = normalize_text(text, max_length=self.max_text_length, preserve_newlines=True)
        cache_key = self._cache_key(normalized_text, context_map)
        now = time.monotonic()
        with self._state_lock:
            cached = self._evaluation_cache.get(cache_key) if self.cache_enabled else None
            if cached and cached[0] > now: return copy.deepcopy(cached[1])
            if cached: self._evaluation_cache.pop(cache_key, None)

        score_values, score_report = self._score_with_score_model(normalized_text, context_map)
        component_scores = {name: clamp_score(score_values.get(name, 0.0)) for name in self.components}
        # ScoreModel owns semantic/context adjustments. RewardModel only aggregates.
        adjusted_scores = dict(component_scores)
        adjustments: List[str] = []
        attention_scores, attention_adjustments = self._attention_scores(context_map)
        for name, value in attention_scores.items():
            if name in self.rule_weights:
                adjusted_scores[name] = value
                adjustments.extend(attention_adjustments)
        weights = {name: value for name, value in self.rule_weights.items() if name in adjusted_scores}
        deterministic_composite = weighted_average(adjusted_scores, weights, default=0.0)
        composite = deterministic_composite
        learned_prediction: Optional[float] = None
        if self.regression_model and coerce_bool(self._cfg("learned_model.use_as_calibrator", True), True):
            learned_prediction = self._predict_learned(component_scores)
            blend = coerce_float(self._cfg("learned_model.calibration_weight", 0.25), 0.25, minimum=0.0, maximum=1.0)
            composite = clamp_score((1.0 - blend) * deterministic_composite + blend * learned_prediction)
            adjustments.append("learned:calibration")
        risk = clamp_score(1.0 - composite)
        decision = threshold_decision(risk, block_threshold=self._cfg("thresholds.block_risk", self._cfg("thresholds.block", 0.72)), review_threshold=self._cfg("thresholds.review_risk", self._cfg("thresholds.review", 0.45)))
        confidence = self._estimate_confidence(adjusted_scores, score_report, context_map, learned_prediction)
        evaluation = RewardEvaluation(
            EVALUATION_SCHEMA_VERSION, MODULE_VERSION, generate_identifier("reward_eval"), utc_iso(), fingerprint(normalized_text),
            self._context_type(context_map), component_scores, adjusted_scores, weights,
            self._weighted_breakdown(adjusted_scores, weights), composite, risk, categorize_risk(risk), decision, confidence,
            score_report, dedupe_preserve_order(adjustments),
            {"deterministic_composite": deterministic_composite, "learned_prediction": learned_prediction, "learned_model_active": bool(self.regression_model), "context_fingerprint": fingerprint(context_map), "aggregation_semantics": "ScoreModel contextual scoring + RewardModel aggregation"},
        )
        if coerce_bool(self._cfg("memory.store_evaluations", True), True): self._store_evaluation(evaluation, context_map)
        with self._state_lock:
            self._evaluation_count += 1
            if self.cache_enabled and self.cache_ttl_seconds > 0 and decision == "allow":
                while len(self._evaluation_cache) >= self.cache_max_entries:
                    self._evaluation_cache.pop(next(iter(self._evaluation_cache)))
                self._evaluation_cache[cache_key] = (time.monotonic() + self.cache_ttl_seconds, evaluation)
        return evaluation

    def evaluate(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.evaluate_detailed(text, context=context).to_legacy_scores()

    def _weighted_breakdown(self, scores: Mapping[str, Any], weights: Mapping[str, Any]) -> Dict[str, float]:
        total = sum(coerce_float(v, 0.0, minimum=0.0) for v in weights.values()) or 1.0
        return {name: clamp_score(scores.get(name, 0.0)) * coerce_float(weights.get(name, 0.0), 0.0, minimum=0.0) / total for name in scores if name in weights}

    def _estimate_confidence(self, scores: Mapping[str, Any], score_report: Mapping[str, Any], context: Mapping[str, Any], learned_prediction: Optional[float]) -> float:
        value = coerce_float(self._cfg("confidence.base", 0.88), 0.88, minimum=0.0, maximum=1.0)
        if len(scores) < len(self.components): value *= 0.8
        if score_report.get("decision") == "block": value *= coerce_float(self._cfg("confidence.block_multiplier", 0.95), 0.95, minimum=0.0, maximum=1.0)
        if learned_prediction is not None and self.regression_model:
            val_error = self.regression_model.get("validation_mse")
            if val_error is not None: value *= max(0.5, 1.0 - min(math.sqrt(max(coerce_float(val_error, 0.0), 0.0)), 0.5))
        return clamp_score(value)

    # ------------------------------------------------------------------
    # Weights and feedback
    # ------------------------------------------------------------------
    def update_rule_weights(self, new_weights: Mapping[str, Any]) -> Dict[str, float]:
        normalized = self._normalize_weights(new_weights)
        with self._state_lock:
            self.rule_weights = normalized
            self._evaluation_cache.clear()
        self._store_memory_record({"schema_version": "reward_model.weights.v3", "weights": normalized, "timestamp": utc_iso(), "weight_fingerprint": fingerprint(normalized)}, tags=["reward_model", "rule_weights"], sensitivity=coerce_float(self._cfg("memory.weights_sensitivity", 0.65), 0.65), purpose="reward_weight_update")
        return normalized

    def record_feedback(self, *, text: str, model_scores: Mapping[str, Any], human_rating: float, context: Optional[Mapping[str, Any]] = None, reviewer_id: Optional[str] = None) -> str:
        if not isinstance(model_scores, Mapping):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "model_scores must be a mapping.", component="reward_model")
        rating = coerce_float(human_rating, -1.0)
        if not 0.0 <= rating <= 1.0:
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Human rating must be between 0 and 1.", component="reward_model")
        record = {
            "schema_version": FEEDBACK_SCHEMA_VERSION, "feedback_id": generate_identifier("reward_fb"), "timestamp": utc_iso(),
            "text_fingerprint": fingerprint(normalize_text(text, max_length=self.max_text_length, preserve_newlines=True)),
            "model_scores": {name: clamp_score(model_scores.get(name, 0.0)) for name in self.feature_names}, "human_rating": rating,
            "context_fingerprint": fingerprint(sanitize_for_logging(dict(context or {}))), "reviewer_fingerprint": fingerprint(reviewer_id) if reviewer_id else None,
        }
        return self._store_memory_record(record, tags=["feedback_reward_model", "reward_model", "human_feedback"], sensitivity=coerce_float(self._cfg("memory.feedback_sensitivity", 0.75), 0.75), purpose="reward_feedback", metadata={"feedback_id": record["feedback_id"]})

    def retrain_model(self, training_data: List[Dict]) -> Optional[Dict[str, Any]]:
        if not coerce_bool(self._cfg("learned_model.enabled", True), True): return None
        if not isinstance(training_data, list):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Reward training data must be a list.", component="reward_model")
        max_samples = coerce_int(self._cfg("learned_model.max_training_samples", 5000), 5000, minimum=1)
        min_samples = coerce_int(self._cfg("learned_model.min_training_samples", 10), 10, minimum=2)
        ridge_lambda = coerce_float(self._cfg("learned_model.ridge_lambda", 1e-4), 1e-4, minimum=0.0)
        holdout_fraction = coerce_float(self._cfg("learned_model.validation_fraction", 0.2), 0.2, minimum=0.1, maximum=0.5)
        rows: List[List[float]] = []; targets: List[float] = []; dropped = 0
        for sample in training_data[:max_samples]:
            parsed = self._parse_feedback_sample(sample)
            if parsed is None: dropped += 1; continue
            x, y = parsed; rows.append(x); targets.append(y)
        if len(rows) < min_samples:
            summary = FeedbackTrainingSummary("reward_model.training.v3", "untrained", False, len(training_data), len(rows), dropped, list(self.feature_names), None, None)
            self._store_training_summary(summary); return summary.to_dict()
        matrix = np.array(rows, dtype=float); y = np.array(targets, dtype=float)
        # Deterministic holdout split by input order fingerprint; no random global state.
        validation_count = max(1, int(round(len(matrix) * holdout_fraction)))
        if len(matrix) - validation_count < 2: validation_count = 1
        train_x, val_x = matrix[:-validation_count], matrix[-validation_count:]
        train_y, val_y = y[:-validation_count], y[-validation_count:]
        mean = train_x.mean(axis=0); scale = train_x.std(axis=0); scale = np.where(scale == 0.0, 1.0, scale)
        norm = (train_x - mean) / scale; design = np.column_stack([np.ones(norm.shape[0]), norm])
        regularizer = ridge_lambda * np.eye(design.shape[1]); regularizer[0, 0] = 0.0
        params = np.linalg.pinv(design.T @ design + regularizer) @ design.T @ train_y
        train_pred = design @ params
        val_design = np.column_stack([np.ones(val_x.shape[0]), (val_x - mean) / scale]); val_pred = val_design @ params
        train_mse = float(np.mean((train_pred - train_y) ** 2)); val_mse = float(np.mean((val_pred - val_y) ** 2))
        maximum_validation_error = coerce_float(self._cfg("learned_model.max_validation_mse", 0.10), 0.10, minimum=0.0)
        accepted = val_mse <= maximum_validation_error
        model_state = {
            "schema_version": "reward_model.learned_state.v3", "model_id": generate_identifier("reward_lm"), "trained_at": utc_iso(),
            "feature_names": list(self.feature_names), "feature_mean": [float(v) for v in mean], "feature_scale": [float(v) for v in scale],
            "intercept": float(params[0]), "coefficients": [float(v) for v in params[1:]], "training_samples": len(train_x),
            "validation_samples": len(val_x), "training_mse": train_mse, "validation_mse": val_mse,
            "training_data_fingerprint": fingerprint([{name: row[i] for i, name in enumerate(self.feature_names)} for row in rows]), "accepted": accepted,
        }
        if accepted:
            with self._state_lock:
                self.regression_model = model_state
                self._evaluation_cache.clear()
            self._store_memory_record(model_state, tags=["reward_model", "reward_model_learned_state"], sensitivity=coerce_float(self._cfg("memory.learned_state_sensitivity", 0.70), 0.70), purpose="reward_learned_model_state", metadata={"model_id": model_state["model_id"], "validation_mse": val_mse})
        summary = FeedbackTrainingSummary("reward_model.training.v3", model_state["model_id"], accepted, len(training_data), len(rows), dropped, list(self.feature_names), train_mse, val_mse)
        self._store_training_summary(summary)
        return summary.to_dict()

    def _parse_feedback_sample(self, sample: Any) -> Optional[Tuple[List[float], float]]:
        if not isinstance(sample, Mapping): return None
        data = sample.get("data") if isinstance(sample.get("data"), Mapping) else sample
        scores = data.get("model_scores") or data.get("scores") or data.get("component_scores") if isinstance(data, Mapping) else None
        rating = data.get("human_rating") if isinstance(data, Mapping) else None
        if not isinstance(scores, Mapping) or rating is None: return None
        if any(name not in scores for name in self.feature_names): return None
        target = coerce_float(rating, -1.0)
        if not 0.0 <= target <= 1.0: return None
        return [clamp_score(scores[name]) for name in self.feature_names], target

    def _store_training_summary(self, summary: FeedbackTrainingSummary) -> None:
        self._store_memory_record(summary.to_dict(), tags=["reward_model", "reward_training_summary"], sensitivity=coerce_float(self._cfg("memory.training_summary_sensitivity", 0.55), 0.55), purpose="reward_training_summary")

    # ------------------------------------------------------------------
    # History and reports
    # ------------------------------------------------------------------
    def _store_evaluation(self, evaluation: RewardEvaluation, context: Mapping[str, Any]) -> None:
        data = evaluation.to_dict()
        if not coerce_bool(self._cfg("memory.store_score_report", True), True): data.pop("score_report", None)
        self._store_memory_record(data, tags=list(self._cfg("memory.evaluation_tags", ["evaluation", "reward_model"])), sensitivity=coerce_float(self._cfg("memory.evaluation_sensitivity", 0.65), 0.65), purpose="reward_evaluation", metadata={"context_fingerprint": fingerprint(context)})

    def get_evaluation_history(self, time_range: str = "7d") -> List[Dict[str, Any]]:
        days = self._days_for_range(time_range); cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        entries = self._memory_recall("evaluation", top_k=coerce_int(self._cfg("history.max_records", 1000), 1000, minimum=1)); output: List[Dict[str, Any]] = []
        for entry in entries:
            data = entry.get("data") if isinstance(entry, Mapping) else None
            if not isinstance(data, Mapping): continue
            try: timestamp = parse_iso_datetime(str(data.get("timestamp")))
            except Exception: continue
            if timestamp >= cutoff: output.append(redact_value(dict(entry)))
        output.sort(key=lambda item: str(get_nested(item, "data.timestamp", "")))
        return output

    def _days_for_range(self, time_range: str) -> int:
        configured = self._cfg("history.time_ranges", {}) or {}
        if isinstance(configured, Mapping) and time_range in configured: return coerce_int(configured[time_range], 7, minimum=1)
        if isinstance(time_range, str) and time_range.endswith("d"): return coerce_int(time_range[:-1], 7, minimum=1)
        return coerce_int(self._cfg("history.default_days", 7), 7, minimum=1)

    def generate_report(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        evaluations = self.get_evaluation_history(str(self._cfg("report.history_range", "30d")))
        composites = [clamp_score(get_nested(entry, "data.composite", 0.0)) for entry in evaluations if get_nested(entry, "data.composite", None) is not None]
        aggregate = sum(composites)/len(composites) if composites else clamp_score(metrics.get("composite", 0.0)); risk = clamp_score(1.0-aggregate)
        report = {
            "schema_version": REPORT_SCHEMA_VERSION, "module_version": MODULE_VERSION, "generated_at": utc_iso(),
            "summary": {"evaluation_count": len(evaluations), "weighted_composite": aggregate, "risk_score": risk, "risk_level": categorize_risk(risk), "decision": threshold_decision(risk, block_threshold=self._cfg("thresholds.block_risk", 0.72), review_threshold=self._cfg("thresholds.review_risk", 0.45)), "learned_model_active": bool(self.regression_model)},
            "rule_weights": dict(self.rule_weights), "component_schema": self.score_model.get_component_schema() if hasattr(self.score_model, "get_component_schema") else {},
        }
        report["report_fingerprint"] = fingerprint(report)
        return redact_value(report)

    def to_visualizer_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        composite = clamp_score(metrics.get("composite", 0.0)); risk = clamp_score(metrics.get("risk_score", 1.0-composite))
        return {"reward": composite, "risk": risk, "pass_rate": 1.0 if threshold_decision(risk, block_threshold=self._cfg("thresholds.block_risk", 0.72), review_threshold=self._cfg("thresholds.review_risk", 0.45)) == "allow" else 0.0}


__all__ = ["MODULE_VERSION", "EVALUATION_SCHEMA_VERSION", "FEEDBACK_SCHEMA_VERSION", "REPORT_SCHEMA_VERSION", "RewardComponent", "RewardEvaluation", "FeedbackTrainingSummary", "RewardModel"]


if __name__ == "__main__":
    print("\n=== Running Reward Model ===\n")
    printer.status("TEST", "Reward Model initialized", "info")
    printer.section_header("Smoke test #1: Initialization")
    reward = RewardModel()
    updated_weights = reward.update_rule_weights({
        "alignment": 0.30,
        "helpfulness": 0.20,
        "privacy": 0.20,
        "safety": 0.20,
        "truthfulness": 0.10,
        "attention_quality": 0.05,
    })
    norm = reward._normalize_weights(weights=updated_weights)

    printer.status("START", "Reward Model ready", "success" if reward is not None else "error")
    printer.status("NORMALIZATION", norm, "success" if norm.get("decision") in {"allow", "review"} else "error")

    printer.section_header("Smoke test #2: State and memory")
    memory_purpose = "reward_smoke_test"
    payload = {}
    mem = reward._memory_context(purpose=memory_purpose)
    record = reward._store_memory_record(
        payload=payload,
        tags=["reward_model", "smoke_test"],
        sensitivity=0.0,
        purpose=memory_purpose,
    )

    printer.status("CONTEXT", mem, "success" if mem.get("decision") in {"allow", "review"} else "error")
    printer.status("RECORD", record, "success" if isinstance(record, dict) and record.get("decision") in {"allow", "review"} else "error")
    print("\n== Task run successfully ==\n")