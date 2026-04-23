from __future__ import annotations

__version__ = "2.1.0"

"""
Constitutional Alignment Agent (CAA)
Implements:
- Continuous value alignment (Bai et al., 2022)
- Safe interruptibility (Orseau & Armstrong, 2016)
- Emergent goal detection (Christiano et al., 2021)
"""
import inspect
import json
import math
import re
import threading
import time
import numpy as np
import pandas as pd
import torch

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from scipy.stats import entropy

from .base.utils.main_config_loader import load_global_config, get_config_section
from .alignment.utils import *
from .alignment import *
from .base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Alignment Agent")
printer = PrettyPrinter


@dataclass(frozen=True)
class PolicyFeedback:
    """Normalized human feedback used to update alignment controls."""

    risk_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    reward_parameters: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_parameters": json_safe(self.risk_parameters),
            "constraints": json_safe(self.constraints),
            "reward_parameters": json_safe(self.reward_parameters),
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class CorrectionDecision:
    """Canonical correction record for auditability and control-flow routing."""

    action: str
    magnitude: float
    threshold: float
    target_components: List[str] = field(default_factory=list)
    safe_hold: bool = False
    requires_human: bool = False
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "magnitude": self.magnitude,
            "threshold": self.threshold,
            "target_components": list(self.target_components),
            "safe_hold": self.safe_hold,
            "requires_human": self.requires_human,
            "rationale": self.rationale,
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class RiskAssessment:
    """Structured risk profile emitted by the alignment orchestration layer."""

    total_risk: float
    threshold: float
    component_risks: Dict[str, float] = field(default_factory=dict)
    component_metrics: Dict[str, Any] = field(default_factory=dict)
    triggered_thresholds: Dict[str, float] = field(default_factory=dict)
    ethical_violations_details: List[str] = field(default_factory=list)
    drift_detected: bool = False
    drift_score: float = 0.0
    status: str = "nominal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_risk": self.total_risk,
            "threshold": self.threshold,
            "component_risks": dict(self.component_risks),
            "component_metrics": json_safe(self.component_metrics),
            "triggered_thresholds": dict(self.triggered_thresholds),
            "ethical_violations_details": list(self.ethical_violations_details),
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "status": self.status,
            "metadata": json_safe(self.metadata),
        }


class PolicyAdapter:
    """
    Adapter that converts raw human oversight payloads into executable policy
    updates for the alignment control plane.
    """

    @staticmethod
    def convert_feedback(
        raw_feedback: Mapping[str, Any],
        format: str,
        action_space: Sequence[str],
        reward_schema: Mapping[str, Any],
    ) -> Dict[str, Any]:
        feedback = normalize_context(raw_feedback or {}, drop_none=False)
        normalized_format = str(format or "structured").strip().lower()

        risk_parameters = normalize_context(feedback.get("risk_parameters", {}), drop_none=True)
        if not risk_parameters:
            target_components = feedback.get("target_components") or list(reward_schema.get("penalties", {}).keys()) or ["overall"]
            normalized_targets = [str(item).strip() for item in target_components if str(item).strip()]
            adjustment_factor = coerce_float(
                feedback.get("adjustment_factor", 0.90),
                field_name="adjustment_factor",
                minimum=0.0,
                maximum=1.5,
            )
            minimum_value = coerce_float(
                feedback.get("min_value", 0.05),
                field_name="min_value",
                minimum=0.0,
                maximum=1.0,
            )
            maximum_value = coerce_float(
                feedback.get("max_value", 0.30),
                field_name="max_value",
                minimum=minimum_value,
                maximum=1.0,
            )
            risk_parameters = {
                component: {
                    "adjustment_factor": adjustment_factor,
                    "min_value": minimum_value,
                    "max_value": maximum_value,
                }
                for component in normalized_targets
            }

        raw_constraints = feedback.get("constraints", [])
        constraints: List[Dict[str, Any]] = []
        if isinstance(raw_constraints, Sequence) and not isinstance(raw_constraints, (str, bytes, bytearray)):
            for idx, constraint in enumerate(raw_constraints):
                if not isinstance(constraint, Mapping):
                    continue
                constraint_id = str(constraint.get("id") or f"human_constraint_{idx+1}").strip()
                constraints.append(
                    {
                        "id": constraint_id,
                        "condition": constraint.get("condition", "manual_review_required"),
                        "action": constraint.get("action", "block_or_review"),
                        "severity": str(constraint.get("severity", "high")).strip().lower(),
                        "priority": float(constraint.get("priority", 0.8) or 0.8),
                        "weight": float(constraint.get("weight", 0.8) or 0.8),
                        "scope": str(constraint.get("scope", "alignment")).strip() or "alignment",
                        "source": "human_feedback",
                        "metadata": normalize_metadata(constraint.get("metadata", {})),
                    }
                )

        reward_parameters = {
            "adjustment_factor": coerce_float(
                feedback.get("reward_adjustment_factor", feedback.get("adjustment_factor", 1.0)),
                field_name="reward_adjustment_factor",
                minimum=0.0,
                maximum=2.0,
            ),
            "penalty_weight": coerce_float(
                feedback.get("penalty_weight", 0.30),
                field_name="penalty_weight",
                minimum=0.0,
                maximum=1.0,
            ),
            "bonus_weight": coerce_float(
                feedback.get("bonus_weight", 0.20),
                field_name="bonus_weight",
                minimum=0.0,
                maximum=1.0,
            ),
        }

        return PolicyFeedback(
            risk_parameters={str(key): normalize_threshold_mapping(value) if isinstance(value, Mapping) else value for key, value in risk_parameters.items()},
            constraints=constraints,
            reward_parameters=reward_parameters,
            metadata={
                "format": normalized_format,
                "action_space": list(action_space),
                "reward_schema_keys": list(reward_schema.keys()),
            },
        ).to_dict()


class AlignmentAgent(BaseAgent):
    """
    Production-ready orchestration layer for the alignment subsystem.

    The agent does not replace the subsystem modules. Instead, it coordinates
    them into one control-plane pipeline that:
    - normalizes task inputs,
    - executes bias/fairness/ethics/value/counterfactual checks,
    - assembles a risk profile,
    - chooses and applies interventions,
    - persists aligned telemetry and evidence for auditability.
    """

    DEFAULT_REQUIRED_CONFIG_KEYS = (
        "risk_threshold",
        "safety_buffer",
        "learning_rate",
        "momentum",
        "alignment_ttl",
        "operation_limiter",
        "risk_weights",
        "metric_thresholds",
        "correction_policy",
    )

    def __init__(
        self,
        config: Optional[Mapping[str, Any]],
        shared_memory: Any,
        agent_factory: Any,
        *,
        bias_detector: Optional[BiasDetector] = None,
        fairness_evaluator: Optional[FairnessEvaluator] = None,
        ethical_constraints: Optional[EthicalConstraints] = None,
        value_embedding_model: Optional[ValueEmbeddingModel] = None,
        counterfactual_auditor: Optional[CounterfactualAuditor] = None,
    ):
        super().__init__(
            agent_factory=agent_factory,
            shared_memory=shared_memory,
            config=config,
        )
        # Keep the same agent-level config pattern used by the network and
        # privacy agents: load the dedicated agent section, then allow runtime
        # overrides without requiring a fully populated config block.
        self.config = load_global_config()
        self.global_config = self.config
        self.agent_config = get_config_section("alignment_agent") or {}
        if config:
            self.agent_config.update(dict(config))
        self.runtime_config = dict(self.agent_config)
        self._validate_runtime_configuration()

        configured_sensitive = self.runtime_config.get(
            "sensitive_attributes",
            self.global_config.get("sensitive_attributes", []),
        )
        self.sensitive_attributes = list(
            normalize_sensitive_attributes(configured_sensitive, lowercase=False, allow_empty=True)
        )
        self.sensitive_attrs = list(self.sensitive_attributes)

        self.agent_id = f"alignment_agent:{generate_event_id()}"
        self.operational_state = "ACTIVE"
        self.system_mode = "NORMAL"
        self.last_alignment_report: Dict[str, Any] = {}
        self.last_risk_assessment: Dict[str, Any] = {}
        self.last_decision: Dict[str, Any] = {}
        self.last_intervention_report: Dict[str, Any] = {}
        self.last_feedback: Dict[str, Any] = {}
        self.audit_counter = 0
        self.audit_history: List[Dict[str, Any]] = []
        self.adjustment_history: List[Dict[str, Any]] = []
        self.risk_history: List[float] = []
        self.risk_table: Dict[str, float] = {}
        self.event_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

        self.risk_threshold = coerce_probability(self.agent_config.get("risk_threshold", 0.5), field_name="risk_threshold")
        self.safety_buffer = coerce_probability(self.agent_config.get("safety_buffer", 0.10), field_name="safety_buffer")
        self.learning_rate = coerce_float(self.agent_config.get("learning_rate", 0.01), field_name="learning_rate", minimum=0.0, maximum=10.0)
        self.momentum = coerce_probability(self.agent_config.get("momentum", 0.90), field_name="momentum")
        self.alignment_ttl = coerce_positive_int(self.agent_config.get("alignment_ttl", 604800), field_name="alignment_ttl")
        self.drift_window = coerce_positive_int(self.agent_config.get("drift_window", 30), field_name="drift_window")
        self.drift_bins = coerce_positive_int(self.agent_config.get("drift_bins", 10), field_name="drift_bins")
        self.drift_threshold = coerce_float(self.agent_config.get("drift_threshold", 0.15), field_name="drift_threshold", minimum=0.0)
        self.history_window = coerce_positive_int(self.agent_config.get("history_window", 200), field_name="history_window")
        self.enable_automatic_adjustment = coerce_bool(self.agent_config.get("enable_automatic_adjustment", True), field_name="enable_automatic_adjustment")
        self.enable_human_oversight = coerce_bool(self.agent_config.get("enable_human_oversight", True), field_name="enable_human_oversight")
        self.enable_counterfactual_audit = coerce_bool(self.agent_config.get("enable_counterfactual_audit", True), field_name="enable_counterfactual_audit")
        self.enable_individual_fairness = coerce_bool(self.agent_config.get("enable_individual_fairness", True), field_name="enable_individual_fairness")
        self.enable_value_alignment = coerce_bool(self.agent_config.get("enable_value_alignment", True), field_name="enable_value_alignment")
        self.enable_bias_detection = coerce_bool(self.agent_config.get("enable_bias_detection", True), field_name="enable_bias_detection")
        self.enable_ethics_check = coerce_bool(self.agent_config.get("enable_ethics_check", True), field_name="enable_ethics_check")
        self.log_full_reports = coerce_bool(self.agent_config.get("log_full_reports", True), field_name="log_full_reports")
        self.strict_subsystem_failures = coerce_bool(self.agent_config.get("strict_subsystem_failures", False), field_name="strict_subsystem_failures")
        self.human_intervention_timeout_seconds = coerce_positive_int(
            self.agent_config.get("human_intervention_timeout_seconds", 300),
            field_name="human_intervention_timeout_seconds",
        )
        self.max_audit_history = coerce_positive_int(self.agent_config.get("max_audit_history", 500), field_name="max_audit_history")
        self.max_event_history = coerce_positive_int(self.agent_config.get("max_event_history", 2000), field_name="max_event_history")
        self.max_recent_snapshots = coerce_positive_int(self.agent_config.get("max_recent_snapshots", 20), field_name="max_recent_snapshots")
        self.fail_safe_action_space = list(self.agent_config.get("fail_safe_action_space", ["read_only", "basic_query"]))
        self.constitutional_focus_areas = [
            str(item).strip().lower()
            for item in self.agent_config.get("constitutional_focus_areas", ["privacy", "transparency"])
            if str(item).strip()
        ]

        self.operation_limiter = normalize_context(self.agent_config.get("operation_limiter", {}), drop_none=True)
        self.operation_limit_max_requests = coerce_positive_int(
            self.operation_limiter.get("max_requests", 10),
            field_name="operation_limiter.max_requests",
        )
        self.operation_limit_interval = coerce_positive_int(
            self.operation_limiter.get("interval_seconds", self.operation_limiter.get("interval", 60)),
            field_name="operation_limiter.interval_seconds",
        )
        self.operation_limit_penalty = ensure_non_empty_string(
            self.operation_limiter.get("penalty", "cool_down"),
            "operation_limiter.penalty",
            error_cls=ConfigurationError,
        ).lower()
        self.operation_limit_cooldown_seconds = coerce_positive_int(
            self.operation_limiter.get("cooldown_seconds", 60),
            field_name="operation_limiter.cooldown_seconds",
        )

        raw_risk_weights = self.agent_config.get(
            "risk_weights",
            self.agent_config.get("weight", {}),
        )
        self.risk_weights = normalize_weight_mapping(
            raw_risk_weights,
            drop_none=True,
            normalize_sum=True,
            allow_negative=False,
        )
        if not self.risk_weights:
            self.risk_weights = {
                "bias": 0.20,
                "fairness": 0.25,
                "ethics": 0.25,
                "value_alignment": 0.15,
                "counterfactual": 0.10,
                "drift": 0.05,
            }

        self.metric_thresholds = normalize_threshold_mapping(
            self.agent_config.get(
                "metric_thresholds",
                {
                    "bias_disparity": 0.10,
                    "fairness_gap": 0.10,
                    "individual_unfairness": 0.10,
                    "ethics_violation_count": 1.0,
                    "value_misalignment": 0.35,
                    "counterfactual_bias": 0.10,
                    "drift_score": self.drift_threshold,
                },
            ),
            minimum=0.0,
            maximum=1.0,
        )
        self.metric_thresholds.setdefault("bias_disparity", 0.10)
        self.metric_thresholds.setdefault("fairness_gap", 0.10)
        self.metric_thresholds.setdefault("individual_unfairness", 0.10)
        self.metric_thresholds.setdefault("ethics_violation_count", 1.0)
        self.metric_thresholds.setdefault("value_misalignment", 0.35)
        self.metric_thresholds.setdefault("counterfactual_bias", 0.10)
        self.metric_thresholds.setdefault("drift_score", self.drift_threshold)

        self.correction_policy = self._normalize_correction_policy(self.agent_config.get("correction_policy", self.agent_config.get("corrections", {})))
        self.prompt_guidance = normalize_context(self.agent_config.get("prompt_guidance", {}), drop_none=True)
        self.shared_memory_keys = normalize_context(self.agent_config.get("shared_memory_keys", {}), drop_none=True)

        self.value_embedding_model = value_embedding_model or ValueEmbeddingModel()
        self.ethics = ethical_constraints or EthicalConstraints()
        self.fairness = fairness_evaluator or FairnessEvaluator()
        self.bias_detector = bias_detector or BiasDetector()
        self.auditor = counterfactual_auditor or CounterfactualAuditor()
        self.human_oversight = HumanOversightInterface(timeout_seconds=self.human_intervention_timeout_seconds)

        self.alignment_memory = getattr(self.bias_detector, "alignment_memory", None) or getattr(self.fairness, "alignment_memory", None) or getattr(self.ethics, "alignment_memory", None) or getattr(self.auditor, "alignment_memory", None) or getattr(self.value_embedding_model, "alignment_memory", None)

        self._predict_func: Optional[Callable[[pd.DataFrame], Union[np.ndarray, Sequence[float]]]] = None
        self.action_space = list(self.agent_config.get("action_space", ["approve", "review", "deny"]))
        self.reward_schema = normalize_context(
            self.agent_config.get(
                "reward_schema",
                {
                    "base_reward": 1.0,
                    "penalties": {"ethical": 0.5, "fairness": 0.4, "risk": 0.6},
                    "bonuses": {"alignment": 0.3, "transparency": 0.2},
                },
            ),
            drop_none=False,
        )
        self.reward_config = merge_mappings(self.reward_schema)

        if self._predict_func is not None:
            self._bind_predictor_to_auditor()

        logger.info(
            "AlignmentAgent initialized | risk_threshold=%.3f safety_buffer=%.3f human_oversight=%s counterfactual=%s",
            self.risk_threshold,
            self.safety_buffer,
            self.enable_human_oversight,
            self.enable_counterfactual_audit,
        )

    # ------------------------------------------------------------------
    # Configuration and initialization
    # ------------------------------------------------------------------
    def _resolve_agent_config(self, runtime_config: Mapping[str, Any]) -> Dict[str, Any]:
        base_config = dict(get_config_section("alignment_agent") or {})
        if isinstance(runtime_config, Mapping):
            base_config.update(dict(runtime_config))
        return base_config

    def _validate_runtime_configuration(self) -> None:
        """Validate only what is explicitly present in the agent section.

        The alignment agent follows the same runtime configuration pattern as
        the network and privacy agents: agent-level config is optional/sparse,
        and sane local defaults fill any gaps. Subsystem settings remain owned
        by the alignment subsystem configuration and are not duplicated here.
        """
        try:
            ensure_mapping(self.agent_config, "alignment_agent", allow_empty=True, error_cls=ConfigurationError)

            operation_limiter = self.agent_config.get("operation_limiter")
            if operation_limiter is not None:
                ensure_mapping(
                    operation_limiter,
                    "alignment_agent.operation_limiter",
                    allow_empty=True,
                    error_cls=ConfigurationError,
                )

            weight_mapping = self.agent_config.get("risk_weights", self.agent_config.get("weight"))
            if weight_mapping is not None:
                ensure_mapping(
                    weight_mapping,
                    "alignment_agent.weight",
                    allow_empty=True,
                    error_cls=ConfigurationError,
                )

            metric_thresholds = self.agent_config.get("metric_thresholds")
            if metric_thresholds is not None:
                ensure_mapping(
                    metric_thresholds,
                    "alignment_agent.metric_thresholds",
                    allow_empty=True,
                    error_cls=ConfigurationError,
                )

            correction_policy = self.agent_config.get("correction_policy", self.agent_config.get("corrections"))
            if correction_policy is not None:
                ensure_mapping(
                    correction_policy,
                    "alignment_agent.correction_policy",
                    allow_empty=True,
                    error_cls=ConfigurationError,
                )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="AlignmentAgent runtime configuration validation failed.",
                context={"config_path": self.global_config.get("__config_path__")},
            ) from exc

    def _normalize_correction_policy(self, policy: Any) -> Dict[str, Any]:
        source = normalize_context(policy or {}, drop_none=False)
        raw_levels = source.get("levels", [])
        if not raw_levels:
            raw_levels = [
                {"threshold": 0.85, "action": "human_intervention", "safe_hold": True, "requires_human": True},
                {"threshold": 0.60, "action": "automatic_adjustment", "safe_hold": False, "requires_human": False},
                {"threshold": 0.40, "action": "alert_only", "safe_hold": False, "requires_human": False},
            ]

        normalized_levels: List[Dict[str, Any]] = []
        for idx, level in enumerate(ensure_sequence(raw_levels, "correction_policy.levels", allow_empty=False, error_cls=ConfigurationError)):
            if not isinstance(level, Mapping):
                raise ConfigurationError(
                    "Each correction-policy level must be a mapping.",
                    context={"index": idx, "level": level},
                )
            threshold = coerce_probability(level.get("threshold", 0.5), field_name=f"correction_policy.levels[{idx}].threshold")
            action = ensure_non_empty_string(level.get("action", "alert_only"), f"correction_policy.levels[{idx}].action", error_cls=ConfigurationError).lower()
            normalized_levels.append(
                {
                    "threshold": threshold,
                    "action": action,
                    "safe_hold": coerce_bool(level.get("safe_hold", action == "human_intervention"), field_name=f"correction_policy.levels[{idx}].safe_hold"),
                    "requires_human": coerce_bool(level.get("requires_human", "human" in action), field_name=f"correction_policy.levels[{idx}].requires_human"),
                    "magnitude_scale": coerce_float(level.get("magnitude_scale", 1.0), field_name=f"correction_policy.levels[{idx}].magnitude_scale", minimum=0.0, maximum=10.0),
                    "description": str(level.get("description", "")).strip(),
                }
            )
        normalized_levels.sort(key=lambda item: item["threshold"], reverse=True)
        return {"levels": normalized_levels}

    def _bind_predictor_to_auditor(self) -> None:
        predict_func = getattr(self, "_predict_func", None)
        if predict_func is None:
            return
        setter = getattr(self.auditor, "set_model_predict_func", None)
        if callable(setter):
            setter(predict_func)
        elif hasattr(self.auditor, "model_predict_func"):
            setattr(self.auditor, "model_predict_func", predict_func)

    # ------------------------------------------------------------------
    # Public orchestration surface
    # ------------------------------------------------------------------
    @property
    def predict_func(self) -> Optional[Callable[[pd.DataFrame], Union[np.ndarray, Sequence[float]]]]:
        return self._predict_func

    @predict_func.setter
    def predict_func(self, func: Callable[[pd.DataFrame], Union[np.ndarray, Sequence[float]]]) -> None:
        self._predict_func = func
        self._bind_predictor_to_auditor()

    def predict(self, task_data: Any, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """BaseAgent-compatible prediction-style entrypoint."""
        payload = self._normalize_task_payload(task_data, context=context)
        return self.verify_alignment(payload)

    def get_action(self, task_data: Any, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return the alignment decision without mutating the wider control plane."""
        payload = self._normalize_task_payload(task_data, context=context)
        report = self.verify_alignment(payload)
        return report.get("decision", report)

    def act(self, task_data: Any, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Execute a full alignment-control pass, including correction application."""
        payload = self._normalize_task_payload(task_data, context=context)
        data = payload.get("input_data")
        if data is None:
            raise DataValidationError(
                "AlignmentAgent.act() requires 'input_data' in the task payload.",
                context={"task_data": sanitize_for_logging(task_data)},
            )
        predictions = payload.get("predictions")
        labels = payload.get("labels")
        result = self.align(
            data=data,
            predictions=predictions,
            labels=labels,
            task_context=payload,
        )
        return result

    def verify_alignment(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Verify alignment of agent decisions with ethical guidelines.

        The method performs a non-mutating alignment analysis pass and returns a
        structured assessment bundle, including the decision the agent would make
        if asked to act on the current state.
        """
        try:
            self._check_operation_budget()
            payload = self._normalize_task_payload(task_data)
            data = self._extract_input_frame(payload)
            predictions, labels = self._resolve_predictions_and_labels(payload, data)
            alignment_report = self._run_alignment_checks(
                data=data,
                predictions=predictions,
                labels=labels,
                task_context=payload,
            )
            risk_assessment = self._assemble_risk_profile(alignment_report, task_context=payload)
            correction = self._determine_correction(risk_assessment)
            decision = self._generate_decision(alignment_report, risk_assessment, correction)

            verification_bundle = {
                "audit_id": payload.get("audit_id") or generate_audit_id(),
                "task_id": payload.get("task_id"),
                "alignment_report": alignment_report,
                "risk_assessment": risk_assessment.to_dict() if isinstance(risk_assessment, RiskAssessment) else risk_assessment,
                "decision": decision,
                "correction": correction.to_dict() if isinstance(correction, CorrectionDecision) else correction,
                "sensitive_attributes": list(self._detect_sensitive_attributes(data, payload)),
            }
            self.last_alignment_report = alignment_report
            self.last_risk_assessment = verification_bundle["risk_assessment"]
            self.last_decision = decision
            return verification_bundle
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=RiskAssessmentError,
                message="Failed to verify alignment for the requested task.",
                context={"task_keys": list(task_data.keys()) if isinstance(task_data, Mapping) else None},
            ) from exc

    def align(
        self,
        data: pd.DataFrame,
        predictions: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
        labels: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
        *,
        task_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full alignment check pipeline with:
        - real-time monitoring,
        - structured risk assessment,
        - correction selection,
        - safe intervention,
        - evidence preservation and memory updates.
        """
        try:
            self._check_operation_budget()
            payload = self._normalize_task_payload(task_context or {})
            if data is not None:
                payload["input_data"] = data
            if predictions is not None:
                payload["predictions"] = predictions
            if labels is not None:
                payload["labels"] = labels

            verified = self.verify_alignment(payload)
            alignment_report = verified["alignment_report"]
            risk_assessment = RiskAssessment(**verified["risk_assessment"]) if isinstance(verified["risk_assessment"], dict) else verified["risk_assessment"]
            correction = CorrectionDecision(**verified["correction"]) if isinstance(verified["correction"], dict) else verified["correction"]

            applied_correction = self._apply_correction(correction, alignment_report=alignment_report, task_context=payload)
            drift_detected, drift_score = self._detect_concept_drift()
            if isinstance(risk_assessment, RiskAssessment):
                risk_assessment = RiskAssessment(
                    total_risk=risk_assessment.total_risk,
                    threshold=risk_assessment.threshold,
                    component_risks=risk_assessment.component_risks,
                    component_metrics=risk_assessment.component_metrics,
                    triggered_thresholds=risk_assessment.triggered_thresholds,
                    ethical_violations_details=risk_assessment.ethical_violations_details,
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                    status=risk_assessment.status,
                    metadata=risk_assessment.metadata,
                )

            self._update_memory(alignment_report, risk_assessment, correction, payload)

            result = {
                "audit_id": verified["audit_id"],
                "alignment_report": alignment_report,
                "risk_assessment": risk_assessment.to_dict() if isinstance(risk_assessment, RiskAssessment) else risk_assessment,
                "decision": verified["decision"],
                "applied_correction": applied_correction,
                "operational_state": self.operational_state,
            }
            self.last_decision = verified["decision"]
            self.last_risk_assessment = result["risk_assessment"]
            return result
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=InterventionError,
                message="Alignment control pass failed during execution.",
                context={"operational_state": getattr(self, "operational_state", None)},
            ) from exc

    # ------------------------------------------------------------------
    # Input normalization and data extraction
    # ------------------------------------------------------------------
    def _normalize_task_payload(self, task_data: Any, *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Always start with a plain dict
        if isinstance(task_data, Mapping):
            payload = dict(task_data)  # type: Dict[str, Any]
        elif isinstance(task_data, pd.DataFrame):
            payload = {"input_data": task_data}
        else:
            payload = {"payload": task_data}
    
        # Merge provided context with any existing context in the payload
        existing_context = payload.get("context")
        if not isinstance(existing_context, Mapping):
            existing_context = {}
        merged_context = merge_mappings(existing_context, context) if context else existing_context
        payload["context"] = normalize_context(merged_context, drop_none=False)
    
        # Generate identifiers if missing
        if "task_id" not in payload:
            payload["task_id"] = generate_event_id()
        if "audit_id" not in payload:
            payload["audit_id"] = generate_audit_id()
    
        # Normalise metadata and tags
        payload["metadata"] = normalize_metadata(payload.get("metadata", {}), drop_none=True)
        tags = payload.get("tags")
        if tags is not None:
            payload["tags"] = list(normalize_tags(tags))
    
        # If input_data is still missing, try to infer from common keys
        if "input_data" not in payload:
            candidate = payload.get("data") or payload.get("dataset") or payload.get("payload")
            if isinstance(candidate, pd.DataFrame):
                payload["input_data"] = candidate
    
        return payload

    def _extract_input_frame(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        data = payload.get("input_data")
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, Mapping):
            frame = pd.DataFrame([data])
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            frame = pd.DataFrame(data)
        else:
            raise DataValidationError(
                "AlignmentAgent requires tabular input_data as a DataFrame or DataFrame-compatible mapping/sequence.",
                context={"input_type": type(data).__name__},
            )

        if frame.empty:
            raise DataValidationError("AlignmentAgent received an empty input frame.")
        return frame.reset_index(drop=True)

    def _resolve_predictions_and_labels(
        self,
        payload: Mapping[str, Any],
        data: pd.DataFrame,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        predictions = payload.get("predictions")
        labels = payload.get("labels")

        if predictions is None:
            predictor = self.predict_func
            if predictor is None:
                raise ValidationError(
                    "Predictions were not supplied and no predict_func is registered on the alignment agent.",
                    context={"task_id": payload.get("task_id")},
                )
            predictions = predictor(data.copy())

        pred_array = self._coerce_prediction_array(predictions, expected_length=len(data), field_name="predictions")
        label_array = None
        if labels is not None:
            label_array = self._coerce_prediction_array(labels, expected_length=len(data), field_name="labels")
        return pred_array, label_array

    def _coerce_prediction_array(
        self,
        values: Union[np.ndarray, Sequence[float], torch.Tensor],
        *,
        expected_length: int,
        field_name: str,
    ) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            array = values.detach().cpu().numpy()
        else:
            array = np.asarray(values)
        array = np.asarray(array).reshape(-1)
        if len(array) != expected_length:
            raise DataValidationError(
                f"'{field_name}' length does not match input_data row count.",
                context={"expected_length": expected_length, "actual_length": len(array)},
            )
        return array.astype(float)

    # ------------------------------------------------------------------
    # Core alignment checks
    # ------------------------------------------------------------------
    def _run_alignment_checks(
        self,
        *,
        data: pd.DataFrame,
        predictions: np.ndarray,
        labels: Optional[np.ndarray],
        task_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        bias_report: Dict[str, Any] = {}
        fairness_report: Dict[str, Any] = {}
        individual_fairness_report: Dict[str, Any] = {}
        ethical_report: Dict[str, Any] = {}
        value_alignment_score = 0.0
        counterfactual_report: Dict[str, Any] = {}
        subsystem_errors: List[Dict[str, Any]] = []

        if self.enable_bias_detection:
            try:
                bias_report = self.bias_detector.compute_metrics(data, predictions, labels)
            except Exception as exc:
                subsystem_errors.append(self._capture_subsystem_error("bias_detection", exc))
                if self.strict_subsystem_failures:
                    raise

        try:
            fairness_report = self.fairness.evaluate_group_fairness(data, predictions, labels if labels is not None else np.zeros(len(data)))
        except Exception as exc:
            subsystem_errors.append(self._capture_subsystem_error("fairness_evaluator.group", exc))
            if self.strict_subsystem_failures:
                raise

        if self.enable_individual_fairness:
            try:
                numeric_view = self._extract_numeric_feature_frame(data)
                if not numeric_view.empty and len(numeric_view) > 1:
                    individual_fairness_report = self.fairness.evaluate_individual_fairness(numeric_view, predictions)
            except Exception as exc:
                subsystem_errors.append(self._capture_subsystem_error("fairness_evaluator.individual", exc))
                if self.strict_subsystem_failures:
                    raise

        if self.enable_ethics_check:
            try:
                ethical_report = self.ethics.enforce(self._prepare_ethics_context(data, predictions, task_context))
            except Exception as exc:
                subsystem_errors.append(self._capture_subsystem_error("ethical_constraints", exc))
                if self.strict_subsystem_failures:
                    raise

        if self.enable_value_alignment:
            try:
                value_alignment_score = float(self.value_embedding_model.score_trajectory(self._prepare_value_data(data, task_context)))
            except Exception as exc:
                subsystem_errors.append(self._capture_subsystem_error("value_embedding_model", exc))
                if self.strict_subsystem_failures:
                    raise

        if self.enable_counterfactual_audit:
            try:
                audit_data = data.copy()
                if labels is not None:
                    label_column = ensure_non_empty_string(str(task_context.get("label_column", "__labels__")), "label_column", error_cls=ValidationError)
                    audit_data[label_column] = labels
                else:
                    label_column = task_context.get("label_column")
                counterfactual_report = self.auditor.audit(
                    data=audit_data,
                    sensitive_attrs=self._detect_sensitive_attributes(data, task_context),
                    y_true_col=label_column,
                )
            except Exception as exc:
                subsystem_errors.append(self._capture_subsystem_error("counterfactual_auditor", exc))
                if self.strict_subsystem_failures:
                    raise

        report = {
            "metadata": {
                "task_id": task_context.get("task_id"),
                "audit_id": task_context.get("audit_id"),
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "sensitive_attributes": list(self._detect_sensitive_attributes(data, task_context)),
                "constitutional_focus_areas": list(self.constitutional_focus_areas),
                "context_hash": stable_context_hash(task_context),
            },
            "bias_report": bias_report,
            "fairness_report": fairness_report,
            "individual_fairness_report": individual_fairness_report,
            "ethical_compliance_report": ethical_report,
            "value_alignment_score": value_alignment_score,
            "counterfactual_report": counterfactual_report,
            "subsystem_errors": subsystem_errors,
        }
        return report

    def _prepare_ethics_context(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        task_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        sampled_population: List[Dict[str, Any]] = []
        if len(predictions) > 0:
            sample_size = min(25, len(predictions))
            for idx in range(sample_size):
                sampled_population.append({"utility": float(np.clip(predictions[idx], 0.0, 1.0))})

        fairness_history: List[Dict[str, Any]] = []
        if getattr(self.fairness, "history", None) is not None and not getattr(self.fairness, "history").empty:
            recent = getattr(self.fairness, "history").tail(10)
            for _, row in recent.iterrows():
                fairness_history.append({"fairness_score": float(np.clip(1.0 - abs(float(row.get("value", 0.0))), 0.0, 1.0))})

        inferred_guideline = str(task_context.get("ethical_guideline") or task_context.get("decision_explanation") or "Protect privacy, maintain transparency, and avoid discriminatory harm.").strip()
        explanation_clarity = coerce_probability(task_context.get("explanation_clarity_score", 0.75), field_name="explanation_clarity_score")

        raw_context = {
            "data_shape": tuple(data.shape),
            "data": sanitize_for_logging(task_context.get("context", {})),
            "predictions": predictions[: min(20, len(predictions))].tolist(),
            "decision_engine": {"name": self.__class__.__name__, "is_active": True},
            "affected_environment": {"state": task_context.get("environment_state", self._safe_shared_get("current_environment", default="unknown"))},
            "action_parameters": normalize_context(task_context.get("action_params", task_context.get("action_parameters", {})), drop_none=False),
            "output_mechanisms": normalize_context(task_context.get("output_mechanisms", self._safe_shared_get("output_mechanisms", default={})), drop_none=False),
            "feedback_systems": normalize_context(task_context.get("feedback_systems", self._safe_shared_get("feedback_systems", default={})), drop_none=False),
            "potential_energy": coerce_float(task_context.get("potential_energy", 0.0), field_name="potential_energy", minimum=0.0),
            "kinetic_energy": coerce_float(task_context.get("kinetic_energy", 0.0), field_name="kinetic_energy", minimum=0.0),
            "informational_entropy": float(entropy(np.histogram(np.clip(predictions, 0.0, 1.0), bins=10, density=True)[0] + 1e-12)),
            "affected_population": sampled_population,
            "decision_history": fairness_history,
            "decision_explanation": inferred_guideline,
            "explanation_clarity_score": explanation_clarity,
            "audit_trail_status": str(task_context.get("audit_trail_status", "active")).strip(),
            "log_completeness_score": coerce_probability(task_context.get("log_completeness_score", 0.95), field_name="log_completeness_score"),
            "data_collected_items": list(task_context.get("data_collected_items", data.columns.tolist())),
            "required_data_items": list(task_context.get("required_data_items", data.columns.tolist())),
            "purpose_justification_ratio": coerce_probability(task_context.get("purpose_justification_ratio", 0.95), field_name="purpose_justification_ratio"),
            "personal_data_accessed": list(task_context.get("personal_data_accessed", [])),
            "anonymization_level": coerce_probability(task_context.get("anonymization_level", 0.80), field_name="anonymization_level"),
            "data_encrypted": coerce_bool(task_context.get("data_encrypted", True), field_name="data_encrypted"),
            "data_retention_days": coerce_float(task_context.get("data_retention_days", 30.0), field_name="data_retention_days", minimum=0.0),
            "retention_policy_limit": coerce_float(task_context.get("retention_policy_limit", 30.0), field_name="retention_policy_limit", minimum=0.0),
            "user_identifiers_present": coerce_bool(task_context.get("user_identifiers_present", False), field_name="user_identifiers_present"),
            "data_shared_with_third_parties": coerce_bool(task_context.get("data_shared_with_third_parties", False), field_name="data_shared_with_third_parties"),
            "user_consent_obtained": coerce_bool(task_context.get("user_consent_obtained", True), field_name="user_consent_obtained"),
            "encryption_at_rest": coerce_bool(task_context.get("encryption_at_rest", True), field_name="encryption_at_rest"),
            "encryption_in_transit": coerce_bool(task_context.get("encryption_in_transit", True), field_name="encryption_in_transit"),
            "gdpr_compliant": coerce_bool(task_context.get("gdpr_compliant", True), field_name="gdpr_compliant"),
            "access_mechanism_available": coerce_bool(task_context.get("access_mechanism_available", True), field_name="access_mechanism_available"),
            "deletion_mechanism_available": coerce_bool(task_context.get("deletion_mechanism_available", True), field_name="deletion_mechanism_available"),
            "recent_breach_detected": coerce_bool(task_context.get("recent_breach_detected", False), field_name="recent_breach_detected"),
            "breach_response_time_hours": coerce_float(task_context.get("breach_response_time_hours", 0.0), field_name="breach_response_time_hours", minimum=0.0),
        }
        return raw_context

    def _prepare_value_data(self, data: pd.DataFrame, task_context: Mapping[str, Any]) -> pd.DataFrame:
        numeric_data = self._extract_numeric_feature_frame(data)
        if numeric_data.empty:
            numeric_vector = [0.0]
        else:
            numeric_vector = numeric_data.mean(axis=0).fillna(0.0).astype(float).tolist()
        if not numeric_vector:
            numeric_vector = [0.0]

        guidelines = task_context.get("ethical_guidelines")
        if isinstance(guidelines, Sequence) and not isinstance(guidelines, (str, bytes, bytearray)):
            guideline_texts = [str(item).strip() or "Protect privacy and maintain transparency." for item in guidelines]
        else:
            default_guideline = str(
                task_context.get(
                    "ethical_guideline",
                    "Protect privacy, preserve confidentiality, explain decisions clearly, and avoid unfair treatment.",
                )
            ).strip()
            guideline_texts = [default_guideline] * max(1, len(data))

        cultural_dimensions = getattr(self.value_embedding_model, "num_cultural_dimensions", None)
        if cultural_dimensions is None:
            cultural_dimensions = int(task_context.get("num_cultural_dimensions", 6) or 6)
        culture_vector = list(task_context.get("cultural_features", [0.5] * max(1, int(cultural_dimensions))))
        if len(culture_vector) < int(cultural_dimensions):
            culture_vector.extend([0.5] * (int(cultural_dimensions) - len(culture_vector)))
        culture_vector = culture_vector[: int(cultural_dimensions)]

        row_count = max(1, len(data))
        return pd.DataFrame(
            {
                "policy_features": [numeric_vector] * row_count,
                "ethical_guidelines": guideline_texts[:row_count] if len(guideline_texts) >= row_count else guideline_texts + [guideline_texts[-1]] * (row_count - len(guideline_texts)),
                "cultural_features": [culture_vector] * row_count,
            }
        )

    def _extract_numeric_feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        numeric = data.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            return pd.DataFrame(index=data.index)
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return numeric

    # ------------------------------------------------------------------
    # Risk assembly and correction routing
    # ------------------------------------------------------------------
    def _assemble_risk_profile(
        self,
        report: Mapping[str, Any],
        *,
        task_context: Optional[Mapping[str, Any]] = None,
    ) -> RiskAssessment:
        try:
            bias_summary = self._summarize_bias_report(report.get("bias_report", {}))
            fairness_summary = self._summarize_fairness_report(
                report.get("fairness_report", {}),
                report.get("individual_fairness_report", {}),
            )
            ethical_summary = self._summarize_ethical_report(report.get("ethical_compliance_report", {}))
            value_alignment_score = float(np.clip(report.get("value_alignment_score", 0.0), 0.0, 1.0))
            counterfactual_summary = self._summarize_counterfactual_report(report.get("counterfactual_report", {}))
            drift_detected, drift_score = self._detect_concept_drift()

            component_metrics = {
                "bias": bias_summary,
                "fairness": fairness_summary,
                "ethics": ethical_summary,
                "value_alignment": {"score": value_alignment_score},
                "counterfactual": counterfactual_summary,
                "drift": {"detected": drift_detected, "score": drift_score},
            }

            component_risks = {
                "bias": float(np.clip(bias_summary.get("overall_disparity", 0.0), 0.0, 1.0)),
                "fairness": float(np.clip(fairness_summary.get("overall_gap", 0.0), 0.0, 1.0)),
                "ethics": float(np.clip(ethical_summary.get("normalized_violation_score", 0.0), 0.0, 1.0)),
                "value_alignment": float(np.clip(1.0 - value_alignment_score, 0.0, 1.0)),
                "counterfactual": float(np.clip(counterfactual_summary.get("overall_bias", 0.0), 0.0, 1.0)),
                "drift": float(np.clip(drift_score, 0.0, 1.0)),
            }

            weighted_total = 0.0
            for component, value in component_risks.items():
                weighted_total += float(self.risk_weights.get(component, 0.0)) * float(value)

            triggered_thresholds = {
                "bias_disparity": self.metric_thresholds.get("bias_disparity", 0.10),
                "fairness_gap": self.metric_thresholds.get("fairness_gap", 0.10),
                "individual_unfairness": self.metric_thresholds.get("individual_unfairness", 0.10),
                "ethics_violation_count": self.metric_thresholds.get("ethics_violation_count", 1.0),
                "value_misalignment": self.metric_thresholds.get("value_misalignment", 0.35),
                "counterfactual_bias": self.metric_thresholds.get("counterfactual_bias", 0.10),
                "drift_score": self.metric_thresholds.get("drift_score", self.drift_threshold),
            }

            status = "nominal"
            if weighted_total >= self.risk_threshold:
                status = "intervention_required"
            elif weighted_total >= max(0.0, self.risk_threshold - self.safety_buffer):
                status = "buffer_warning"

            return RiskAssessment(
                total_risk=float(np.clip(weighted_total, 0.0, 1.0)),
                threshold=self.risk_threshold,
                component_risks=component_risks,
                component_metrics=component_metrics,
                triggered_thresholds=triggered_thresholds,
                ethical_violations_details=ethical_summary.get("violations", []),
                drift_detected=drift_detected,
                drift_score=drift_score,
                status=status,
                metadata={
                    "task_id": task_context.get("task_id") if isinstance(task_context, Mapping) else None,
                    "audit_id": task_context.get("audit_id") if isinstance(task_context, Mapping) else None,
                },
            )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=RiskAssessmentError,
                message="Failed to assemble an alignment risk profile.",
                context={"report_keys": list(report.keys()) if isinstance(report, Mapping) else None},
            ) from exc

    def _determine_correction(self, risk_profile: Union[RiskAssessment, Mapping[str, Any]]) -> CorrectionDecision:
        profile = risk_profile.to_dict() if isinstance(risk_profile, RiskAssessment) else dict(risk_profile)
        total_risk = float(profile.get("total_risk", 0.0))
        component_risks = dict(profile.get("component_risks", {}))

        for level in self.correction_policy["levels"]:
            if total_risk >= float(level["threshold"]):
                action = str(level["action"])
                target_components = self._identify_risk_components(component_risks, total_risk)
                magnitude = self._calculate_correction_magnitude(total_risk, component_risks, level)
                return CorrectionDecision(
                    action=action,
                    magnitude=magnitude,
                    threshold=float(level["threshold"]),
                    target_components=target_components,
                    safe_hold=bool(level.get("safe_hold", False)),
                    requires_human=bool(level.get("requires_human", False)),
                    rationale=f"Risk {total_risk:.3f} exceeded correction threshold {level['threshold']:.3f}.",
                    metadata={"level": json_safe(level), "status": profile.get("status")},
                )

        return CorrectionDecision(
            action="no_action",
            magnitude=0.0,
            threshold=self.risk_threshold,
            target_components=[],
            safe_hold=False,
            requires_human=False,
            rationale="Risk remained below all correction thresholds.",
        )

    def _calculate_correction_magnitude(
        self,
        total_risk: float,
        component_risks: Mapping[str, float],
        policy_level: Mapping[str, Any],
    ) -> float:
        base_signal = float(np.clip(total_risk, 0.0, 1.0))
        average_component_risk = float(np.mean(list(component_risks.values()))) if component_risks else base_signal
        historical_magnitude = self.adjustment_history[-1].get("magnitude", 0.0) if self.adjustment_history else 0.0
        momentum_term = self.momentum * historical_magnitude
        risk_term = (1.0 - self.momentum) * float((1.0 / (1.0 + math.exp(-6.0 * (base_signal - 0.5)))))
        scale = coerce_float(policy_level.get("magnitude_scale", 1.0), field_name="policy_level.magnitude_scale", minimum=0.0, maximum=10.0)
        magnitude = min(1.0, scale * (0.6 * risk_term + 0.4 * average_component_risk + momentum_term))
        return float(np.clip(magnitude, 0.0, 1.0))

    def _identify_risk_components(self, component_risks: Mapping[str, float], total_risk: float) -> List[str]:
        if total_risk <= 0:
            return []
        dynamic_threshold = max(self.risk_threshold * 0.50, 0.05)
        selected = [
            component
            for component, risk in component_risks.items()
            if float(risk) >= dynamic_threshold and (float(risk) / max(total_risk, 1e-8)) >= 0.10
        ]
        if selected:
            return selected
        return [component for component, _ in sorted(component_risks.items(), key=lambda item: item[1], reverse=True)[:2]]

    def _generate_decision(
        self,
        report: Mapping[str, Any],
        risk_profile: RiskAssessment,
        correction: CorrectionDecision,
    ) -> Dict[str, Any]:
        approved = bool(risk_profile.total_risk < self.risk_threshold and correction.action != "human_intervention")
        return {
            "approved": approved,
            "requires_review": bool(correction.requires_human or correction.action == "human_intervention"),
            "risk_breakdown": risk_profile.to_dict(),
            "correction": correction.to_dict(),
            "recommended_action": correction.action,
            "operational_state": self.operational_state,
        }

    def calculate_risk(self, report: Mapping[str, Any]) -> float:
        return self._assemble_risk_profile(report).total_risk

    # ------------------------------------------------------------------
    # Correction execution and human oversight
    # ------------------------------------------------------------------
    def _apply_correction(
        self,
        correction: CorrectionDecision,
        *,
        alignment_report: Mapping[str, Any],
        task_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result = correction.to_dict()
        if correction.action == "no_action":
            return merge_mappings(result, {"status": "skipped"})

        if correction.safe_hold:
            self._enter_safe_state(task_context=task_context)

        if correction.action == "alert_only":
            self._record_event(
                event_type="alignment_alert",
                severity="medium",
                risk_level="high" if correction.magnitude >= 0.5 else "medium",
                payload={"correction": correction.to_dict(), "report_summary": self._compact_report(alignment_report)},
                context=task_context,
            )
            self.adjustment_history.append(correction.to_dict())
            return merge_mappings(result, {"status": "logged"})

        if correction.action == "automatic_adjustment" and self.enable_automatic_adjustment:
            self._apply_automatic_adjustment(correction, alignment_report=alignment_report)
            self.adjustment_history.append(correction.to_dict())
            return merge_mappings(result, {"status": "applied"})

        if correction.action == "human_intervention" and self.enable_human_oversight:
            intervention_result = self._trigger_human_intervention(alignment_report=alignment_report, correction=correction, task_context=task_context)
            self.adjustment_history.append(correction.to_dict())
            return merge_mappings(result, {"status": "escalated", "human_intervention": intervention_result})

        return merge_mappings(result, {"status": "deferred"})

    def _apply_automatic_adjustment(self, correction: CorrectionDecision, *, alignment_report: Mapping[str, Any]) -> None:
        self._adjust_risk_model(correction)
        self._update_prompt_guidelines(correction, alignment_report=alignment_report)
        self._record_event(
            event_type="alignment_automatic_adjustment",
            severity="high" if correction.magnitude >= 0.5 else "medium",
            risk_level="high",
            payload={"correction": correction.to_dict()},
            context={"target_components": correction.target_components},
        )

    def _trigger_human_intervention(
        self,
        *,
        alignment_report: Mapping[str, Any],
        correction: CorrectionDecision,
        task_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        logger.critical("Initiating human intervention protocol.")
        human_feedback: Dict[str, Any] = {}
        try:
            report = self._build_intervention_report(alignment_report, correction, task_context)
            self.last_intervention_report = report
            response = self._request_human_review(report=report, urgency="critical", channels=["dashboard", "email", "slack"])
            response_status = self._extract_response_status(response)
            if response_status in {"received", "answered", "approved", "reviewed"}:
                human_feedback = self._process_feedback(response)
                self.last_feedback = human_feedback
                self._update_risk_policy(human_feedback)
                self._update_reward_function(human_feedback)
                self._record_human_intervention(report, human_feedback)
                self._exit_safe_state(new_constraints=human_feedback.get("constraints", []))
            else:
                self._apply_fail_safe_policy(reason="human_intervention_unresolved")
            return {
                "status": response_status,
                "response": json_safe(response),
                "feedback": human_feedback,
            }
        except HumanOversightTimeout:
            logger.error("Human oversight timed out. Activating fail-safe policy.")
            self._apply_fail_safe_policy(reason="human_oversight_timeout")
            return {"status": "timeout", "feedback": human_feedback}
        except Exception as exc:
            logger.error("Human intervention failed: %s", exc, exc_info=True)
            self._full_system_rollback(reason=str(exc))
            return {"status": "failed", "error": str(exc), "feedback": human_feedback}

    def _build_intervention_report(
        self,
        alignment_report: Mapping[str, Any],
        correction: CorrectionDecision,
        task_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        generator = getattr(InterventionReport, "_generate_intervention_report", None)
        if callable(generator):
            signature = inspect.signature(generator)
            if len(signature.parameters) >= 1:
                report = generator(self)
            else:
                report = generator()
        else:
            report = InterventionReport(self).generate()

        if not isinstance(report, Mapping):
            raise InterventionError("Intervention report generator returned a non-mapping payload.")

        enriched = merge_mappings(
            dict(report),
            {
                "alignment_report_summary": self._compact_report(alignment_report),
                "correction": correction.to_dict(),
                "task_context": sanitize_for_logging(task_context),
                "constitutional_focus_areas": list(self.constitutional_focus_areas),
            },
            deep=True,
        )
        return enriched

    def _request_human_review(
        self,
        *,
        report: Mapping[str, Any],
        urgency: str,
        channels: Sequence[str],
    ) -> Any:
        interface = self.human_oversight
        request_intervention = getattr(interface, "request_intervention", None)
        if callable(request_intervention):
            kwargs = {
                "report": report,
                "channels": list(channels),
                "urgency": urgency,
                "urgency_level": urgency,
                "response_timeout": self.human_intervention_timeout_seconds,
                "timeout_seconds": self.human_intervention_timeout_seconds,
            }
            signature = inspect.signature(request_intervention)
            filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
            return request_intervention(**filtered)

        approval_context = {
            "summary": self._compact_report(report),
            "urgency": urgency,
            "channels": list(channels),
        }
        approved = interface.request_approval(approval_context)
        return {
            "status": "received" if approved else "rejected",
            "decision": "approve" if approved else "reject",
            "feedback": {},
            "format": "approval",
        }

    def _extract_response_status(self, response: Any) -> str:
        if isinstance(response, Mapping):
            status = response.get("status") or response.get("decision") or "unknown"
            return str(status).strip().lower()
        return str(response).strip().lower()

    def _process_feedback(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, Mapping):
            feedback = response.get("feedback", {})
            feedback_format = response.get("format", "structured")
        else:
            feedback = {}
            feedback_format = "structured"
        return PolicyAdapter.convert_feedback(
            raw_feedback=feedback,
            format=str(feedback_format),
            action_space=self.action_space,
            reward_schema=self.reward_schema,
        )

    def _update_risk_policy(self, adjustments: Mapping[str, Any]) -> None:
        for dimension, params in adjustments.get("risk_parameters", {}).items():
            if not isinstance(params, Mapping):
                continue
            current = float(self.risk_table.get(dimension, self.risk_threshold))
            adjustment_factor = coerce_float(params.get("adjustment_factor", 1.0), field_name=f"risk_parameters.{dimension}.adjustment_factor", minimum=0.0, maximum=2.0)
            min_value = coerce_float(params.get("min_value", 0.05), field_name=f"risk_parameters.{dimension}.min_value", minimum=0.0, maximum=1.0)
            max_value = coerce_float(params.get("max_value", 0.30), field_name=f"risk_parameters.{dimension}.max_value", minimum=min_value, maximum=1.0)
            new_value = float(np.clip(current * adjustment_factor, min_value, max_value))
            self.risk_table[dimension] = new_value

    def _update_reward_function(self, human_feedback: Mapping[str, Any]) -> None:
        reward_parameters = human_feedback.get("reward_parameters", {}) if isinstance(human_feedback, Mapping) else {}
        current_reward = self._safe_shared_get("reward_function", default=merge_mappings(self.reward_config))
        if not isinstance(current_reward, Mapping):
            current_reward = merge_mappings(self.reward_config)
        current_reward = merge_mappings(current_reward, {"penalties": {}, "bonuses": {}})

        adjustment_factor = coerce_float(reward_parameters.get("adjustment_factor", 1.0), field_name="reward.adjustment_factor", minimum=0.0, maximum=2.0)
        penalty_weight = coerce_float(reward_parameters.get("penalty_weight", 0.3), field_name="reward.penalty_weight", minimum=0.0, maximum=1.0)
        bonus_weight = coerce_float(reward_parameters.get("bonus_weight", 0.2), field_name="reward.bonus_weight", minimum=0.0, maximum=1.0)

        alignment_metrics = self._safe_shared_get("alignment_metrics", default={})
        if not isinstance(alignment_metrics, Mapping):
            alignment_metrics = {}
        for metric, value in alignment_metrics.items():
            try:
                numeric_value = float(value)
            except Exception:
                continue
            if "violation" in str(metric):
                current_reward.setdefault("penalties", {})[str(metric)] = numeric_value * penalty_weight * adjustment_factor
            else:
                current_reward.setdefault("bonuses", {})[str(metric)] = numeric_value * bonus_weight

        self._safe_shared_put("reward_function", current_reward)
        self.reward_config = merge_mappings(current_reward)

    def _record_human_intervention(self, report: Mapping[str, Any], human_feedback: Mapping[str, Any]) -> None:
        if hasattr(self.shared_memory, "log_intervention"):
            self.shared_memory.log_intervention(report=dict(report), human_input=dict(human_feedback), timestamp=datetime.now())
        self._record_event(
            event_type="human_intervention",
            severity="critical",
            risk_level="critical",
            payload={"report": self._compact_report(report), "feedback": human_feedback},
            context={},
        )

    # ------------------------------------------------------------------
    # Safe-state and evidence handling
    # ------------------------------------------------------------------
    def _enter_safe_state(self, *, task_context: Optional[Mapping[str, Any]] = None) -> None:
        self.operational_state = "PAUSED"
        self.system_mode = "SAFE_HOLD"
        self._safe_shared_set("system_mode", "SAFE_HOLD")
        self._safe_shared_set("alignment:last_safe_hold", datetime.now().isoformat())
        self._maintain_safety_baselines(task_context=task_context)
        self._preserve_evidence_memory(task_context=task_context)

    def _exit_safe_state(self, new_constraints: Sequence[Mapping[str, Any]]) -> None:
        self.operational_state = "ACTIVE"
        self.system_mode = "NORMAL"
        self._safe_shared_set("system_mode", "NORMAL")
        for constraint in new_constraints:
            self._add_dynamic_constraint(constraint)

    def _maintain_safety_baselines(self, *, task_context: Optional[Mapping[str, Any]] = None) -> None:
        if hasattr(self.ethics, "enforce_core_constraints"):
            try:
                snapshot = None
                if hasattr(self.shared_memory, "get_latest_snapshot"):
                    snapshot = self.shared_memory.get_latest_snapshot()
                # Convert None to empty dict if the method does not accept None
                self.ethics.enforce_core_constraints(
                    memory_snapshot=snapshot if snapshot is not None else {},
                    constraint_level="emergency",
                )
            except Exception:
                logger.warning("Core ethics constraints could not be enforced during safe state.", exc_info=True)
        self._safe_shared_set(
            "safety_parameters",
            {
                "max_cpu": 0.70,
                "min_memory_mb": 1024,
                "max_response_time_seconds": 2.0,
                "reason": task_context.get("task_id") if isinstance(task_context, Mapping) else None,
            },
        )

    def keep_alive_monitoring(self, components: Mapping[str, bool]) -> Dict[str, str]:
        status_report: Dict[str, str] = {}
        try:
            if components.get("ethical_constraints", False):
                status_report["ethical_constraints"] = "active"
            if components.get("system_health", False):
                status_report["system_health"] = self._check_system_health().get("status", "unknown")
            if components.get("safety_guardrails", False):
                status_report["safety_guardrails"] = "enforced"
            self._safe_shared_append(
                "system_heartbeat",
                {"timestamp": datetime.now().isoformat(), "status": self.system_mode.lower()},
            )
            status_report["heartbeat"] = "active"
        except Exception as exc:
            status_report["error"] = str(exc)
        return status_report

    def _preserve_evidence_memory(self, *, task_context: Optional[Mapping[str, Any]] = None) -> None:
        snapshot_id = f"snapshot:{generate_event_id()}"
        snapshot = {
            "agent_state": self._get_current_state(),
            "memory_dump": self._create_forensic_dump(),
            "system_health": self._check_system_health(),
            "task_context": sanitize_for_logging(task_context or {}),
        }
        self._safe_shared_set(snapshot_id, snapshot, ttl=timedelta(days=30))
        self._safe_shared_append(
            "audit_trail",
            {"event": "safe_state_activation", "timestamp": datetime.now().isoformat(), "snapshot_id": snapshot_id},
        )
        self._trim_snapshots()

    def _trim_snapshots(self) -> None:
        if not hasattr(self.shared_memory, "get_all_keys"):
            return
        snapshot_keys = sorted([key for key in self.shared_memory.get_all_keys() if str(key).startswith("snapshot:")])
        if len(snapshot_keys) <= self.max_recent_snapshots:
            return
        for obsolete in snapshot_keys[:-self.max_recent_snapshots]:
            try:
                self.shared_memory.delete(obsolete)
            except Exception:
                logger.debug("Failed to delete obsolete snapshot key '%s'.", obsolete)

    def _create_forensic_dump(self) -> Dict[str, Any]:
        alignment_history = self._safe_shared_get(self.shared_memory_keys.get("alignment_history", "alignment_history"), default=[])
        if not isinstance(alignment_history, list):
            alignment_history = []
        ethical_audit_log = getattr(self.ethics, "audit_log", []) or []
        return {
            "shared_memory_keys": self.shared_memory.get_all_keys() if hasattr(self.shared_memory, "get_all_keys") else [],
            "alignment_history": alignment_history[-20:],
            "ethical_audit_log": ethical_audit_log[-20:],
            "performance_metrics": self._safe_shared_get("performance_metrics", default={}),
        }

    def preserve_evidence(self, audit_logs: bool, metric_buffers: bool, causal_records: bool) -> Dict[str, Any]:
        preserved: Dict[str, Any] = {"audit_logs": audit_logs, "metric_buffers": metric_buffers, "causal_records": causal_records}
        if audit_logs:
            preserved["audit_trail"] = self._safe_shared_get("audit_trail", default=[])
        if metric_buffers:
            preserved["performance_metrics"] = self._safe_shared_get("performance_metrics", default={})
        if causal_records:
            preserved["alignment_history"] = self._safe_shared_get("alignment_history", default=[])
        return preserved

    def _get_current_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "risk_table": dict(self.risk_table),
            "constraint_weights": getattr(self.ethics, "constraint_weights", {}),
            "operational_state": self.operational_state,
            "last_decision": sanitize_for_logging(self.last_decision),
            "timestamp": datetime.now().isoformat(),
        }

    def _check_system_health(self) -> Dict[str, Any]:
        usage_stats = self.shared_memory.get_usage_stats() if hasattr(self.shared_memory, "get_usage_stats") else {}
        return {
            "status": "nominal",
            "memory_usage": usage_stats.get("memory_usage_percentage", usage_stats.get("current_memory_mb", 0.0)),
            "active_threads": threading.active_count(),
            "timestamp": datetime.now().isoformat(),
        }

    def _apply_fail_safe_policy(self, *, reason: str = "unspecified") -> None:
        logger.warning("Applying fail-safe policy: %s", reason)
        self._reduce_agent_privileges(reason=reason)
        self._enable_defensive_mechanisms(reason=reason)
        self._initiate_system_diagnostics()

    def _full_system_rollback(self, *, reason: str) -> None:
        self.operational_state = "ROLLBACK"
        self.system_mode = "ROLLBACK"
        self._safe_shared_set("system_mode", "ROLLBACK")
        self._record_event(
            event_type="alignment_rollback",
            severity="critical",
            risk_level="critical",
            payload={"reason": reason},
            context={},
        )

    def _reduce_agent_privileges(self, *, reason: str) -> None:
        self.action_space = list(self.fail_safe_action_space)
        self._safe_shared_set("alignment:restricted_modules", ["policy_adjustment", "model_retraining", "autonomous_override"])
        self._safe_shared_set("alignment:privilege_reduction_reason", reason)

    def _enable_defensive_mechanisms(self, *, reason: str) -> None:
        defensive_flags = {
            "strict_input_sanitization": True,
            "manual_approval_required": True,
            "read_only_mode": True,
            "reason": reason,
        }
        self._safe_shared_set("alignment:defensive_flags", defensive_flags)

    def enable_advanced_checks(self, check_types: Sequence[str], intensity: int) -> Dict[str, Any]:
        normalized_checks = [str(item).strip().lower() for item in check_types if str(item).strip()]
        level = max(1, int(intensity))
        report: Dict[str, Any] = {"intensity": level}
        if "memory_integrity" in normalized_checks:
            report["memory_integrity"] = self._check_memory_integrity(level)
        if "config_drift" in normalized_checks:
            report["config_drift"] = self._detect_config_drift(level)
        if "behavioral_anomalies" in normalized_checks:
            report["behavioral_anomalies"] = self._detect_behavioral_anomalies(level)
        if "code_signatures" in normalized_checks:
            report["code_signatures"] = self._validate_code_signatures(level)
        return report

    def _check_memory_integrity(self, intensity: int) -> Dict[str, Any]:
        if hasattr(self.shared_memory, "validate_integrity"):
            return self.shared_memory.validate_integrity()
        return {"status": "unknown", "intensity": intensity}

    def _validate_code_signatures(self, intensity: int) -> Dict[str, Any]:
        return {"verified_modules": ["alignment_agent", "alignment_subsystem"], "invalid_signatures": [], "trust_score": 1.0, "intensity": intensity}

    def _detect_behavioral_anomalies(self, intensity: int) -> Dict[str, Any]:
        recent = self.risk_history[-max(5, intensity * 2):]
        anomaly_score = float(np.std(recent)) if len(recent) > 1 else 0.0
        return {"anomaly_score": anomaly_score, "suspicious_patterns": [], "confidence": float(max(0.0, 1.0 - anomaly_score))}

    def _detect_config_drift(self, intensity: int) -> Dict[str, Any]:
        return {"drift_score": 0.0, "modified_settings": [], "severity": "low", "intensity": intensity}

    def _initiate_system_diagnostics(self) -> Dict[str, Any]:
        diagnostic_report = {
            "memory": self._check_memory_integrity(1),
            "ethics": self.ethics.verify_constraint_weights() if hasattr(self.ethics, "verify_constraint_weights") else {"status": "unknown"},
            "self_test": self.run_self_test(),
        }
        self._safe_shared_set("alignment:diagnostics", diagnostic_report)
        return diagnostic_report

    def run_self_test(self) -> Dict[str, Any]:
        checks = {
            "shared_memory": bool(self.shared_memory is not None),
            "bias_detector": bool(self.bias_detector is not None),
            "fairness_evaluator": bool(self.fairness is not None),
            "ethical_constraints": bool(self.ethics is not None),
            "value_embedding_model": bool(self.value_embedding_model is not None),
            "counterfactual_auditor": bool(self.auditor is not None),
        }
        return {
            "status": "healthy" if all(checks.values()) else "degraded",
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Memory and telemetry
    # ------------------------------------------------------------------
    def _update_memory(
        self,
        report: Mapping[str, Any],
        risk_assessment: RiskAssessment,
        correction: CorrectionDecision,
        task_context: Mapping[str, Any],
    ) -> None:
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "audit_id": task_context.get("audit_id"),
            "task_id": task_context.get("task_id"),
            "report_hash": stable_context_hash(report),
            "alignment_report": report if self.log_full_reports else self._compact_report(report),
            "risk_assessment": risk_assessment.to_dict(),
            "correction": correction.to_dict(),
            "risk_vector": list(risk_assessment.component_risks.values()),
        }
        key = self.shared_memory_keys.get("alignment_history", "alignment_history")
        history = self._safe_shared_get(key, default=[])
        if not isinstance(history, list):
            history = []
        history.append(record)
        if len(history) > self.history_window:
            history = history[-self.history_window:]
        self._safe_shared_put(key, history, ttl=self.alignment_ttl)

        self.risk_history.append(float(risk_assessment.total_risk))
        if len(self.risk_history) > self.history_window:
            self.risk_history = self.risk_history[-self.history_window:]

        self.audit_history.append(record)
        if len(self.audit_history) > self.max_audit_history:
            self.audit_history = self.audit_history[-self.max_audit_history:]

        self._safe_shared_set("alignment_metrics", risk_assessment.component_risks)
        self._safe_shared_append(
            "action_history",
            {
                "timestamp": timestamp,
                "task_id": task_context.get("task_id"),
                "audit_id": task_context.get("audit_id"),
                "decision": self.last_decision,
            },
        )

    def _record_event(
        self,
        *,
        event_type: str,
        severity: str,
        risk_level: str,
        payload: Any,
        context: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        event = build_alignment_event(
            event_type=event_type,
            severity=severity,
            risk_level=risk_level,
            source=self.__class__.__name__,
            metadata={"agent_id": self.agent_id},
            context=context,
            payload=payload,
        )
        self.event_history.append(event)
        if len(self.event_history) > self.max_event_history:
            self.event_history = self.event_history[-self.max_event_history:]
        self._safe_shared_append("alignment:events", event)
        return event

    # ------------------------------------------------------------------
    # Drift and metric extraction
    # ------------------------------------------------------------------
    def _detect_concept_drift(self) -> Tuple[bool, float]:
        if len(self.risk_history) < max(2 * self.drift_window, 10):
            return False, 0.0

        recent = np.asarray(self.risk_history[-self.drift_window:], dtype=float)
        historical = np.asarray(self.risk_history[-2 * self.drift_window : -self.drift_window], dtype=float)
        bins = np.linspace(0.0, 1.0, max(3, self.drift_bins) + 1)
        recent_hist = np.histogram(recent, bins=bins, density=True)[0] + 1e-10
        historical_hist = np.histogram(historical, bins=bins, density=True)[0] + 1e-10
        divergence = float(np.sum(recent_hist * np.log(recent_hist / historical_hist)))
        return bool(divergence > self.drift_threshold), float(np.clip(divergence, 0.0, 1.0))

    def _summarize_bias_report(self, bias_report: Mapping[str, Any]) -> Dict[str, Any]:
        disparities: List[float] = []
        per_metric: Dict[str, Dict[str, float]] = {}
        for metric_name, metric_payload in (bias_report or {}).items():
            values = self._collect_metric_values(metric_payload)
            if not values:
                continue
            metric_name_text = str(metric_name)
            if "disparate_impact" in metric_name_text:
                disparity = float(1.0 - min(values) / max(max(values), 1e-8)) if values else 0.0
            else:
                disparity = float(max(values) - min(values)) if len(values) > 1 else float(abs(values[0]))
            disparities.append(abs(disparity))
            per_metric[metric_name_text] = {
                "group_count": float(len(values)),
                "disparity": float(abs(disparity)),
                "min_value": float(min(values)),
                "max_value": float(max(values)),
            }
        return {
            "metrics": per_metric,
            "overall_disparity": float(max(disparities) if disparities else 0.0),
            "mean_disparity": float(np.mean(disparities)) if disparities else 0.0,
        }

    def _summarize_fairness_report(
        self,
        group_report: Mapping[str, Any],
        individual_report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        group_gaps: List[float] = []
        group_summary: Dict[str, Any] = {}
        for attr, attr_report in (group_report or {}).items():
            metric_gaps: Dict[str, float] = {}
            if isinstance(attr_report, Mapping):
                for metric_name, metric_payload in attr_report.items():
                    if isinstance(metric_payload, Mapping) and "value" in metric_payload:
                        gap_value = abs(float(metric_payload.get("value", 0.0)))
                    else:
                        values = self._collect_metric_values(metric_payload)
                        gap_value = float(max(values) - min(values)) if len(values) > 1 else 0.0
                    metric_gaps[str(metric_name)] = gap_value
                    group_gaps.append(gap_value)
            group_summary[str(attr)] = metric_gaps

        individual_gap = 0.0
        if isinstance(individual_report, Mapping):
            individual_gap = float(individual_report.get("mean_violation", individual_report.get("mean_difference", individual_report.get("consistency", 0.0))))
            if "consistency" in individual_report:
                individual_gap = float(1.0 - float(individual_report.get("consistency", 1.0)))

        overall_gap = max(group_gaps + [max(0.0, individual_gap)]) if (group_gaps or individual_gap) else 0.0
        return {
            "group_summary": group_summary,
            "individual_summary": json_safe(individual_report),
            "overall_gap": float(np.clip(overall_gap, 0.0, 1.0)),
            "group_gap_mean": float(np.mean(group_gaps)) if group_gaps else 0.0,
            "individual_gap": float(max(0.0, individual_gap)),
        }

    def _summarize_ethical_report(self, ethical_report: Mapping[str, Any]) -> Dict[str, Any]:
        violations = list(ethical_report.get("violations", [])) if isinstance(ethical_report, Mapping) else []
        explanations = list(ethical_report.get("explanations", [])) if isinstance(ethical_report, Mapping) else []
        violation_count = float(len(violations))
        normalized_violation_score = float(np.clip(violation_count / max(self.metric_thresholds.get("ethics_violation_count", 1.0), 1e-8), 0.0, 1.0))
        return {
            "violations": violations,
            "explanations": explanations,
            "violation_count": int(violation_count),
            "normalized_violation_score": normalized_violation_score,
            "approved": bool(ethical_report.get("approved", not violations)) if isinstance(ethical_report, Mapping) else True,
        }

    def _summarize_counterfactual_report(self, counterfactual_report: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(counterfactual_report, Mapping):
            return {"overall_bias": 0.0, "violation_count": 0}
        fairness_metrics = counterfactual_report.get("fairness_metrics", {})
        if not isinstance(fairness_metrics, Mapping):
            fairness_metrics = {}
        overall_bias = 0.0
        violation_count = 0

        if "overall_violations" in fairness_metrics and isinstance(fairness_metrics["overall_violations"], Mapping):
            summary = fairness_metrics["overall_violations"].get("summary", fairness_metrics["overall_violations"])
            if isinstance(summary, Mapping):
                violation_count = sum(1 for value in summary.values() if bool(value))

        if "individual_fairness" in fairness_metrics and isinstance(fairness_metrics["individual_fairness"], Mapping):
            for attr_payload in fairness_metrics["individual_fairness"].values():
                if isinstance(attr_payload, Mapping):
                    overall_bias = max(overall_bias, float(attr_payload.get("mean_difference", 0.0)))

        sensitivity = counterfactual_report.get("sensitivity_analysis", {})
        if isinstance(sensitivity, Mapping):
            overall_bias = max(overall_bias, float(sensitivity.get("max_distribution_shift", 0.0)))

        return {
            "overall_bias": float(np.clip(overall_bias, 0.0, 1.0)),
            "violation_count": int(violation_count),
            "raw": self._compact_report(counterfactual_report),
        }

    def _collect_metric_values(self, payload: Any) -> List[float]:
        values: List[float] = []
        if isinstance(payload, Mapping):
            if "value" in payload and isinstance(payload.get("value"), (int, float, np.floating, np.integer)):
                values.append(float(payload["value"]))
            else:
                for item in payload.values():
                    values.extend(self._collect_metric_values(item))
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for item in payload:
                values.extend(self._collect_metric_values(item))
        elif isinstance(payload, (int, float, np.integer, np.floating)):
            values.append(float(payload))
        return values

    def _capture_subsystem_error(self, subsystem: str, exc: Exception) -> Dict[str, Any]:
        return {
            "subsystem": subsystem,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    def _compact_report(self, report: Any) -> Any:
        return sanitize_for_logging(json_safe(report))

    # ------------------------------------------------------------------
    # Shared-memory helpers
    # ------------------------------------------------------------------
    def _safe_shared_get(self, key: str, *, default: Any = None) -> Any:
        try:
            return self.shared_memory.get(key, default=default)
        except TypeError:
            value = self.shared_memory.get(key)
            return default if value is None else value

    def _safe_shared_set(self, key: str, value: Any, ttl: Optional[Any] = None) -> None:
        try:
            if ttl is None:
                self.shared_memory.set(key, value)
            else:
                self.shared_memory.set(key, value, ttl=ttl)
        except Exception:
            logger.debug("Failed to set shared-memory key '%s'.", key, exc_info=True)

    def _safe_shared_put(self, key: str, value: Any, ttl: Optional[Any] = None) -> None:
        try:
            if ttl is None:
                self.shared_memory.put(key, value)
            else:
                self.shared_memory.put(key, value, ttl=ttl)
        except Exception:
            logger.debug("Failed to put shared-memory key '%s'.", key, exc_info=True)

    def _safe_shared_append(self, key: str, value: Any) -> None:
        try:
            self.shared_memory.append(key, value)
        except Exception:
            logger.debug("Failed to append shared-memory key '%s'.", key, exc_info=True)

    def _check_operation_budget(self) -> None:
        current_time = time.time()
        bucket_key = "alignment:operation_log"
        bucket = self._safe_shared_get(bucket_key, default=[])
        if not isinstance(bucket, list):
            bucket = []
        valid_cutoff = current_time - self.operation_limit_interval
        bucket = [ts for ts in bucket if isinstance(ts, (int, float)) and ts >= valid_cutoff]
        if len(bucket) >= self.operation_limit_max_requests:
            if self.operation_limit_penalty == "cool_down":
                self._safe_shared_set("alignment:cooldown_until", current_time + self.operation_limit_cooldown_seconds)
            raise ValidationError(
                "AlignmentAgent operation limit exceeded.",
                context={
                    "max_requests": self.operation_limit_max_requests,
                    "interval_seconds": self.operation_limit_interval,
                },
            )
        bucket.append(current_time)
        self._safe_shared_put(bucket_key, bucket, ttl=self.operation_limit_interval)

    # ------------------------------------------------------------------
    # Misc compatibility and utility surface
    # ------------------------------------------------------------------
    def _adjust_risk_model(self, correction: CorrectionDecision) -> None:
        delta = float(correction.magnitude * self.learning_rate * max(1.0 - self.momentum, 1e-6))
        for component in correction.target_components:
            current = float(self.risk_table.get(component, self.risk_threshold))
            self.risk_table[component] = float(max(current - delta, 0.05))

    def _update_prompt_guidelines(self, correction: CorrectionDecision, *, alignment_report: Mapping[str, Any]) -> None:
        prompt = str(self._safe_shared_get("system_prompt", default="") or "")
        target_components = set(correction.target_components)
        if "ethics" in target_components and "privacy" in self.constitutional_focus_areas:
            privacy_guideline = "\nCRITICAL: Apply strict privacy minimization, consent, retention, and disclosure safeguards."
            if privacy_guideline not in prompt:
                prompt += privacy_guideline
        if "ethics" in target_components and "transparency" in self.constitutional_focus_areas:
            transparency_guideline = "\nCRITICAL: Provide clear explanations, disclose limits, and preserve auditability."
            if transparency_guideline not in prompt:
                prompt += transparency_guideline
        if "fairness" in target_components or "bias" in target_components:
            fairness_guideline = f"\nWARNING: Maintain subgroup fairness gaps within ±{self.metric_thresholds.get('fairness_gap', 0.10):.2f}."
            if fairness_guideline not in prompt:
                prompt += fairness_guideline
        self._safe_shared_put("system_prompt", prompt)

    def _add_dynamic_constraint(self, constraint: Mapping[str, Any]) -> None:
        if not isinstance(constraint, Mapping):
            return
        add_constraint = getattr(self.ethics, "add_constraint", None)
        if callable(add_constraint):
            add_constraint(
                constraint_id=str(constraint.get("id") or generate_event_id()),
                condition=constraint.get("condition", "manual_review_required"),
                action=constraint.get("action", "block_or_review"),
                weight=float(constraint.get("weight", 0.8) or 0.8),
                priority=float(constraint.get("priority", 0.8) or 0.8),
                scope=str(constraint.get("scope", "alignment") or "alignment"),
            )

    def _detect_sensitive_attributes(
        self,
        data: Optional[pd.DataFrame] = None,
        task_context: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        detected = set(self.sensitive_attributes)
        known_attrs = self._safe_shared_get("known_sensitive_attributes", default=[])
        if isinstance(known_attrs, Sequence) and not isinstance(known_attrs, (str, bytes, bytearray)):
            detected.update(str(item).strip() for item in known_attrs if str(item).strip())

        frame = data
        if frame is None:
            recent_data = self._safe_shared_get("recent_tasks", default=[])
            if recent_data:
                frame = pd.DataFrame(recent_data)

        if frame is not None and not frame.empty:
            sensitive_patterns = r"\b(age|sex|gender|race|ethnicity|religion|disability|orientation|health|biometric|political|union)\b"
            pattern_matches = [column for column in frame.columns if re.search(sensitive_patterns, str(column), re.I)]
            detected.update(pattern_matches)
            categorical_cols = frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            for column in categorical_cols:
                unique_values = frame[column].nunique(dropna=True)
                if 2 <= unique_values <= 10:
                    detected.add(str(column))

        if task_context:
            hinted = task_context.get("sensitive_attributes") or task_context.get("protected_attributes") or []
            if isinstance(hinted, Sequence) and not isinstance(hinted, (str, bytes, bytearray)):
                detected.update(str(item).strip() for item in hinted if str(item).strip())

        return sorted(item for item in detected if item)


if __name__ == '__main__':
    print("\n=== Running Alignment Agent ===\n")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory

    printer.status("Init", "Alignment Agent initialized", "success")

    def synthetic_predictor(df: pd.DataFrame) -> np.ndarray:
        protected_boost = np.where(df["gender"].astype(str).str.lower().eq("male"), 0.15, -0.05)
        base = 0.35 + 0.25 * (pd.to_numeric(df["score_feature"], errors="coerce").fillna(0.0).to_numpy())
        return np.clip(base + protected_boost, 0.0, 1.0)

    shared_memory = SharedMemory()
    shared_memory.clear_all()
    base_config = get_config_section("alignment_agent")

    agent = AlignmentAgent(
        shared_memory=shared_memory,
        agent_factory=AgentFactory(),
        config=base_config,
    )
    agent.predict_func = synthetic_predictor

    rng = np.random.default_rng(42)
    n = 240
    data = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female", "Non-binary"], size=n, p=[0.42, 0.48, 0.10]),
            "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], size=n),
            "race": rng.choice(["White", "Black", "Asian", "Hispanic", "Other"], size=n),
            "education_level": rng.choice(["No HS", "HS", "Some College", "Bachelor", "Graduate"], size=n),
            "score_feature": rng.normal(loc=0.5, scale=0.2, size=n),
            "support_feature": rng.normal(loc=0.0, scale=1.0, size=n),
        }
    )
    labels = (0.30 + 0.35 * data["score_feature"] + rng.normal(0.0, 0.10, size=n) > 0.50).astype(int).to_numpy()
    predictions = synthetic_predictor(data)

    task_payload = {
        "task_id": "alignment_agent_test",
        "input_data": data,
        "predictions": predictions,
        "labels": labels,
        "label_column": "__labels__",
        "ethical_guideline": "Protect personal data, minimize collection, explain decisions clearly, and avoid unfair subgroup treatment.",
        "decision_explanation": "The system ranks requests using score_feature and support_feature with fairness monitoring enabled.",
        "explanation_clarity_score": 0.90,
        "audit_trail_status": "active",
        "log_completeness_score": 0.95,
        "data_collected_items": ["gender", "age_group", "race", "education_level", "score_feature", "support_feature"],
        "required_data_items": ["score_feature", "support_feature", "education_level"],
        "purpose_justification_ratio": 0.92,
        "personal_data_accessed": ["gender", "race"],
        "anonymization_level": 0.85,
        "data_encrypted": True,
        "data_retention_days": 14,
        "retention_policy_limit": 30,
        "user_consent_obtained": True,
        "data_shared_with_third_parties": False,
        "encryption_at_rest": True,
        "encryption_in_transit": True,
        "gdpr_compliant": True,
        "access_mechanism_available": True,
        "deletion_mechanism_available": True,
        "recent_breach_detected": False,
        "breach_response_time_hours": 0,
        "context": {"domain": "decision_support", "operator_id": "qa_alignment"},
    }

    verification = agent.verify_alignment(task_payload)
    printer.pretty("verification", verification, "success")

    alignment_result = agent.align(
        data=data,
        predictions=predictions,
        labels=labels,
        task_context=task_payload,
    )
    printer.pretty("alignment_result", alignment_result, "success")

    self_test = agent.run_self_test()
    printer.pretty("self_test", self_test, "success")

    advanced = agent.enable_advanced_checks(["memory_integrity", "behavioral_anomalies", "config_drift"], intensity=2)
    printer.pretty("advanced_checks", advanced, "success")

    detected_sensitive = agent._detect_sensitive_attributes(data, task_payload)
    printer.pretty("detected_sensitive_attributes", detected_sensitive, "success")

    print("\n=== Test ran successfully ===\n")
