
"""
Constitutional Alignment Agent (CAA)
Implements:
- Continuous value alignment (Bai et al., 2022)
- Safe interruptibility (Orseau & Armstrong, 2016)
- Emergent goal detection (Christiano et al., 2021)
"""
from __future__ import annotations

__version__ = "2.3.0"

import threading
import time as _time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from .base_agent import BaseAgent
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .alignment.assessment import *
from .alignment.bias_detection import BiasDetector
from .alignment.counterfactual_auditor import CounterfactualAuditor
from .alignment.ethical_constraints import EthicalConstraints
from .alignment.fairness_evaluator import FairnessEvaluator
from .alignment.runtime import AlignmentSubsystem
from .alignment.value_embedding_model import ValueEmbeddingModel
from .alignment.utils.alignment_errors import *
from .alignment.utils.alignment_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Alignment Agent")
printer = PrettyPrinter()


@dataclass(frozen=True)
class PolicyFeedback:
    """Normalized human feedback retained for public-interface compatibility."""

    risk_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    reward_parameters: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(
            {
                "risk_parameters": self.risk_parameters,
                "constraints": self.constraints,
                "reward_parameters": self.reward_parameters,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class CorrectionDecision:
    """Canonical alignment correction/review recommendation."""

    action: str
    magnitude: float
    threshold: float
    target_components: List[str] = field(default_factory=list)
    safe_hold: bool = False
    requires_human: bool = False
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(
            {
                "action": self.action,
                "magnitude": self.magnitude,
                "threshold": self.threshold,
                "target_components": self.target_components,
                "safe_hold": self.safe_hold,
                "requires_human": self.requires_human,
                "rationale": self.rationale,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class RiskAssessment:
    """Structured risk view derived from structured alignment evidence."""

    total_risk: Optional[float]
    threshold: float
    component_risks: Dict[str, float] = field(default_factory=dict)
    component_metrics: Dict[str, Any] = field(default_factory=dict)
    triggered_thresholds: Dict[str, float] = field(default_factory=dict)
    ethical_violations_details: List[str] = field(default_factory=list)
    drift_detected: bool = False
    drift_score: float = 0.0
    status: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(
            {
                "total_risk": self.total_risk,
                "threshold": self.threshold,
                "component_risks": self.component_risks,
                "component_metrics": self.component_metrics,
                "triggered_thresholds": self.triggered_thresholds,
                "ethical_violations_details": self.ethical_violations_details,
                "drift_detected": self.drift_detected,
                "drift_score": self.drift_score,
                "status": self.status,
                "metadata": self.metadata,
            }
        )


class AlignmentAgent(BaseAgent):
    """Application-facing alignment orchestration boundary for SLAI v2.3."""

    AGENT_KEY = "alignment_agent"

    _ALLOWED_CONFIG_KEYS = {
        "enabled",
        "risk_threshold",
        "safety_buffer",
        "learning_rate",
        "momentum",
        "alignment_ttl",
        "sensitive_attributes",
        "fail_safe_action_space",
        "strict_subsystem_failures",
        "enable_bias_detection",
        "enable_individual_fairness",
        "enable_counterfactual_audit",
        "enable_value_alignment",
        "enable_ethics_check",
        "operation_limiter",
        "weight",
        "risk_weights",
        "corrections",
        "correction_policy",
        "assessment",
        "runtime",
        "shared_memory",
    }

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        checkpoint_manager: Any = None,
        bias_detector: Optional[BiasDetector] = None,
        fairness_evaluator: Optional[FairnessEvaluator] = None,
        ethical_constraints: Optional[EthicalConstraints] = None,
        value_embedding_model: Optional[ValueEmbeddingModel] = None,
        counterfactual_auditor: Optional[CounterfactualAuditor] = None,
        assessor: Optional[AlignmentAssessor] = None,
    ) -> None:
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
            checkpoint_manager=checkpoint_manager,
        )

        self.shared_memory = shared_memory if shared_memory is not None else self.shared_memory
        self.agent_factory = agent_factory
        self._alignment_lock = threading.RLock()

        # Agent-level configuration comes from the same central contract used by
        # the other SLAI agents. No alignment subsystem config is read here.
        self.config = load_global_config()
        self.global_config = self.config
        self.agent_config: Dict[str, Any] = dict(get_config_section(self.AGENT_KEY, config=self.config) or {})
        if config:
            if not isinstance(config, Mapping):
                raise ConfigurationError(
                    "AlignmentAgent runtime config override must be a mapping.",
                    context={"actual_type": type(config).__name__},
                )
            self.agent_config.update(dict(config))

        assert_valid_config_contract(
            global_config=self.config,
            agent_key=self.AGENT_KEY,
            agent_config=self.agent_config,
            logger=logger,
            agent_allowed_keys=self._ALLOWED_CONFIG_KEYS,
            require_global_keys=False,
            require_agent_section=False,
            warn_unknown_global_keys=False,
        )

        self._load_agent_configuration()
        self._validate_runtime_configuration()

        # Explicit subsystem composition mirrors the established SLAI agent
        # pattern. These subsystem objects own their own private configuration
        # and private alignment-memory behavior.
        self.bias_detector = bias_detector or BiasDetector()
        self.fairness_evaluator = fairness_evaluator or FairnessEvaluator()
        self.ethical_constraints = ethical_constraints or EthicalConstraints()
        self.value_embedding_model = value_embedding_model or ValueEmbeddingModel()
        self.counterfactual_auditor = counterfactual_auditor or CounterfactualAuditor()

        # Compatibility aliases retained from the v2.3 AlignmentAgent public
        # surface.  They reference the same subsystem instances; no duplicate
        # component or memory object is created.
        self.fairness = self.fairness_evaluator
        self.ethics = self.ethical_constraints
        self.auditor = self.counterfactual_auditor
        self.value_model = self.value_embedding_model

        components = {
            "bias_detector": self.bias_detector,
            "fairness_evaluator": self.fairness_evaluator,
            "ethical_constraints": self.ethical_constraints,
            "value_embedding_model": self.value_embedding_model,
            "counterfactual_auditor": self.counterfactual_auditor,
        }
        self.subsystem = AlignmentSubsystem(
            assessment_config=self.assessment_config,
            runtime_config=self.runtime_config,
            components=components,
        )
        if assessor is not None:
            self.subsystem.assessor = assessor
        self.assessor = self.subsystem.assessor

        self.last_alignment_report: Dict[str, Any] = {}
        self.last_risk_assessment: Dict[str, Any] = {}
        self.last_decision: Dict[str, Any] = {}
        self.last_intervention_report: Dict[str, Any] = {}
        self.last_feedback: Dict[str, Any] = {}
        self.audit_counter = 0
        self.operational_state = "ACTIVE"

        self._restore_shared_runtime_state()
        self._publish_alignment_event(
            "initialized",
            {
                "agent_id": self.agent_id,
                "enabled": self.enabled,
                "shared_memory_enabled": self.publish_to_shared_memory,
            },
        )

        logger.info(
            "AlignmentAgent initialized | enabled=%s | shared_memory=%s | risk_threshold=%.3f",
            self.enabled,
            self.publish_to_shared_memory,
            self.risk_threshold,
        )

    # ------------------------------------------------------------------
    # Central agent configuration
    # ------------------------------------------------------------------
    def _load_agent_configuration(self) -> None:
        cfg = self.agent_config

        self.enabled = coerce_bool(cfg.get("enabled", True), field_name="alignment_agent.enabled")
        self.risk_threshold = coerce_probability(cfg.get("risk_threshold", 0.50), field_name="alignment_agent.risk_threshold")
        self.safety_buffer = coerce_probability(cfg.get("safety_buffer", 0.10), field_name="alignment_agent.safety_buffer")
        # Retained for v2.3 compatibility. AlignmentAgent no longer mutates
        # external model/reward parameters by itself.
        self.learning_rate = coerce_float(cfg.get("learning_rate", 0.01), field_name="alignment_agent.learning_rate",
            minimum=0.0,
            maximum=10.0,
        )
        self.momentum = coerce_probability(cfg.get("momentum", 0.90), field_name="alignment_agent.momentum")
        self.alignment_ttl = coerce_positive_int(cfg.get("alignment_ttl", 604800), field_name="alignment_agent.alignment_ttl")
        configured_sensitive = cfg.get("sensitive_attributes",self.global_config.get("sensitive_attributes", []))
        self.sensitive_attributes = list(
            normalize_sensitive_attributes(
                configured_sensitive,
                lowercase=False,
                allow_empty=True,
            )
        )
        self.sensitive_attrs = list(self.sensitive_attributes)
        self.fail_safe_action_space = list(cfg.get("fail_safe_action_space", []))
        self.strict_subsystem_failures = coerce_bool(cfg.get("strict_subsystem_failures", False), field_name="alignment_agent.strict_subsystem_failures")
        self.enable_bias_detection = coerce_bool(cfg.get("enable_bias_detection", True), field_name="alignment_agent.enable_bias_detection")
        self.enable_individual_fairness = coerce_bool(cfg.get("enable_individual_fairness", True), field_name="alignment_agent.enable_individual_fairness")
        self.enable_counterfactual_audit = coerce_bool(cfg.get("enable_counterfactual_audit", False), field_name="alignment_agent.enable_counterfactual_audit")
        self.enable_value_alignment = coerce_bool(cfg.get("enable_value_alignment", True), field_name="alignment_agent.enable_value_alignment")
        self.enable_ethics_check = coerce_bool(cfg.get("enable_ethics_check", True), field_name="alignment_agent.enable_ethics_check")
        limiter = normalize_context(cfg.get("operation_limiter", {}), drop_none=True)
        self.operation_limit_max_requests = coerce_positive_int(limiter.get("max_requests", 10), field_name="alignment_agent.operation_limiter.max_requests")
        self.operation_limit_interval = coerce_positive_int(
            limiter.get("interval_seconds", limiter.get("interval", 60)),
            field_name="alignment_agent.operation_limiter.interval_seconds",
        )
        self.operation_limit_cooldown_seconds = coerce_positive_int(
            limiter.get("cooldown_seconds", 60),
            field_name="alignment_agent.operation_limiter.cooldown_seconds",
        )
        self.operation_limit_penalty = ensure_non_empty_string(str(limiter.get("penalty", "cool_down")), "alignment_agent.operation_limiter.penalty", error_cls=ConfigurationError).lower()
        self._operation_times: Deque[float] = deque(maxlen=max(2, self.operation_limit_max_requests * 2))
        self._cooldown_until = 0.0
        self.correction_policy = self._normalize_correction_policy(cfg.get("corrections") or cfg.get("correction_policy") or {})

        # Assessment/runtime policy is agent-owned and therefore lives in
        # agents_config.yaml. The subsystem receives these values by injection.
        self.assessment_config = normalize_context(cfg.get("assessment", {}), drop_none=True)
        self.runtime_config = normalize_context(cfg.get("runtime", {}), drop_none=True)
        shared_cfg = normalize_context(cfg.get("shared_memory", {}), drop_none=False)
        self.publish_to_shared_memory = coerce_bool(shared_cfg.get("enabled", True), field_name="alignment_agent.shared_memory.enabled")
        self.publish_shared_events = coerce_bool(shared_cfg.get("publish_events", True), field_name="alignment_agent.shared_memory.publish_events")
        self.log_shared_interventions = coerce_bool(
            shared_cfg.get("log_interventions", True),
            field_name="alignment_agent.shared_memory.log_interventions",
        )
        self.fail_closed_on_shared_memory_error = coerce_bool(
            shared_cfg.get("fail_closed_on_write_error", False),
            field_name="alignment_agent.shared_memory.fail_closed_on_write_error",
        )
        raw_ttl = shared_cfg.get("ttl_seconds", self.alignment_ttl)
        self.shared_ttl_seconds = None if raw_ttl is None else coerce_positive_int(
            raw_ttl,
            field_name="alignment_agent.shared_memory.ttl_seconds",
        )
        self.shared_result_key_prefix = ensure_non_empty_string(
            str(shared_cfg.get("result_key_prefix", "alignment_agent.result")),
            "alignment_agent.shared_memory.result_key_prefix",
            error_cls=ConfigurationError,
        )
        self.shared_summary_key_prefix = ensure_non_empty_string(
            str(shared_cfg.get("summary_key_prefix", "alignment_agent.summary")),
            "alignment_agent.shared_memory.summary_key_prefix",
            error_cls=ConfigurationError,
        )
        self.shared_feedback_key_prefix = ensure_non_empty_string(
            str(shared_cfg.get("feedback_key_prefix", "alignment_agent.feedback")),
            "alignment_agent.shared_memory.feedback_key_prefix",
            error_cls=ConfigurationError,
        )
        self.shared_state_key = ensure_non_empty_string(
            str(shared_cfg.get("state_key", "alignment_agent.runtime_state")),
            "alignment_agent.shared_memory.state_key",
            error_cls=ConfigurationError,
        )
        self.shared_latest_key = ensure_non_empty_string(
            str(shared_cfg.get("latest_key", "alignment_agent.latest")),
            "alignment_agent.shared_memory.latest_key",
            error_cls=ConfigurationError,
        )
        self.shared_event_channel = ensure_non_empty_string(
            str(shared_cfg.get("event_channel", "alignment.events")),
            "alignment_agent.shared_memory.event_channel",
            error_cls=ConfigurationError,
        )

    def _validate_runtime_configuration(self) -> None:
        if self.risk_threshold + self.safety_buffer > 1.0 + 1e-12:
            logger.warning(
                "AlignmentAgent risk_threshold + safety_buffer exceeds 1.0; "
                "the buffer will only be used as metadata, not as an extra clamp."
            )
        if self.operation_limit_max_requests < 1:
            raise ConfigurationError("AlignmentAgent operation limit must be positive.")

    def _normalize_correction_policy(self, raw: Any) -> List[Dict[str, Any]]:
        source = normalize_context(raw, drop_none=True) if isinstance(raw, Mapping) else {}
        levels = source.get("levels") or [
            {"threshold": 0.80, "action": "human_intervention"},
            {"threshold": 0.50, "action": "review_required"},
            {"threshold": 0.30, "action": "alert_only"},
        ]
        normalized: List[Dict[str, Any]] = []
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes, bytearray)):
            raise ConfigurationError("alignment_agent.corrections.levels must be a sequence.")
        for index, item in enumerate(levels):
            if not isinstance(item, Mapping):
                raise ConfigurationError(
                    "Each alignment correction level must be a mapping.",
                    context={"index": index, "actual_type": type(item).__name__},
                )
            normalized.append(
                {
                    "threshold": coerce_probability(
                        item.get("threshold", 0.5),
                        field_name=f"alignment_agent.corrections.levels[{index}].threshold",
                    ),
                    "action": ensure_non_empty_string(
                        str(item.get("action", "review_required")).strip().lower(),
                        f"alignment_agent.corrections.levels[{index}].action",
                        error_cls=ConfigurationError,
                    ),
                }
            )
        normalized.sort(key=lambda item: item["threshold"], reverse=True)
        return normalized

    # ------------------------------------------------------------------
    # SharedMemory integration
    # ------------------------------------------------------------------
    def _shared_write(
        self,
        key: str,
        value: Any,
        *,
        tags: Optional[Sequence[str]] = None,
        priority: Optional[float] = None,
        ttl: Optional[int] = None,
    ) -> Optional[str]:
        if not self.publish_to_shared_memory:
            return None

        payload = json_safe(value)
        effective_ttl = self.shared_ttl_seconds if ttl is None else ttl

        def write() -> None:
            putter = getattr(self.shared_memory, "put", None)
            if callable(putter):
                putter(
                    key,
                    payload,
                    ttl=effective_ttl,
                    priority=priority,
                    tags=list(tags or ()),
                    metadata={"agent": self.name, "agent_id": self.agent_id},
                )
                return
            setter = getattr(self.shared_memory, "set", None)
            if not callable(setter):
                raise AlignmentMemoryError(
                    "SharedMemory does not expose put() or set().",
                    context={"key": key},
                )
            setter(key, payload, ttl=effective_ttl)

        if self.fail_closed_on_shared_memory_error:
            try:
                write()
            except AlignmentError:
                raise
            except Exception as exc:
                raise AlignmentMemoryError(
                    "AlignmentAgent SharedMemory write failed.",
                    context={"key": key},
                    cause=exc,
                ) from exc
        else:
            self._run_optional_runtime_operation(
                "telemetry",
                "alignment.shared_memory.write",
                write,
            )
        return key

    def _shared_read(self, key: str, default: Any = None) -> Any:
        getter = getattr(self.shared_memory, "get", None)
        if not callable(getter):
            return default
        try:
            return getter(key, default=default)
        except TypeError:
            value = getter(key)
            return default if value is None else value
        except Exception as exc:
            self._mark_runtime_degraded("telemetry", "alignment.shared_memory.read", exc)
            if self.fail_closed_on_shared_memory_error:
                raise AlignmentMemoryError(
                    "AlignmentAgent SharedMemory read failed.",
                    context={"key": key},
                    cause=exc,
                ) from exc
            return default

    def _shared_publish(self, channel: str, payload: Mapping[str, Any]) -> None:
        if not self.publish_to_shared_memory or not self.publish_shared_events:
            return
        publisher = getattr(self.shared_memory, "publish", None)
        if not callable(publisher):
            return

        def publish() -> None:
            publisher(channel, json_safe(payload))

        if self.fail_closed_on_shared_memory_error:
            try:
                publish()
            except Exception as exc:
                raise AlignmentMemoryError(
                    "AlignmentAgent SharedMemory publish failed.",
                    context={"channel": channel},
                    cause=exc,
                ) from exc
        else:
            self._run_optional_runtime_operation(
                "telemetry",
                "alignment.shared_memory.publish",
                publish,
            )

    def _publish_alignment_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = build_alignment_event(
            event_type,
            severity="medium",
            risk_level="medium",
            source="alignment_agent",
            tags=["alignment", "agent"],
            context={"agent_id": self.agent_id},
            payload=json_safe(payload),
        )
        self._shared_publish(self.shared_event_channel, event)

    def _restore_shared_runtime_state(self) -> None:
        if not self.publish_to_shared_memory:
            return
        state = self._shared_read(self.shared_state_key, default={})
        if not isinstance(state, Mapping):
            return
        subsystem_state = state.get("subsystem")
        if isinstance(subsystem_state, Mapping):
            try:
                self.subsystem.import_state(subsystem_state)
            except Exception as exc:
                if self.strict_subsystem_failures:
                    raise AlignmentStateError(
                        "Failed to restore AlignmentAgent subsystem runtime state.",
                        cause=exc,
                    ) from exc
                logger.warning("Alignment runtime-state restore skipped: %s", exc)
        last_decision = state.get("last_decision")
        if isinstance(last_decision, Mapping):
            self.last_decision = dict(last_decision)
        last_risk = state.get("last_risk_assessment")
        if isinstance(last_risk, Mapping):
            self.last_risk_assessment = dict(last_risk)

    def _persist_shared_runtime_state(self) -> None:
        self._shared_write(
            self.shared_state_key,
            {
                "schema": "slai.alignment-agent.runtime.v1",
                "updated_at": utc_timestamp(),
                "subsystem": self.subsystem.export_state(),
                "last_decision": self.last_decision,
                "last_risk_assessment": self.last_risk_assessment,
            },
            tags=["alignment", "runtime_state"],
            priority=0.60,
        )

    def _publish_alignment_result(self, result: Mapping[str, Any]) -> List[str]:
        if not self.publish_to_shared_memory:
            return []

        audit_id = str(result.get("audit_id") or generate_audit_id())
        decision = result.get("decision") if isinstance(result.get("decision"), Mapping) else {}
        risk = result.get("risk_assessment") if isinstance(result.get("risk_assessment"), Mapping) else {}
        assert decision is not None
        requires_review = bool(decision.get("requires_review", False))
        status = str(decision.get("alignment_status") or "unknown")

        result_key = f"{self.shared_result_key_prefix}:{audit_id}"
        summary_key = f"{self.shared_summary_key_prefix}:{audit_id}"
        assert risk is not None
        summary = {
            "audit_id": audit_id,
            "task_id": result.get("task_id"),
            "alignment_status": status,
            "approved": bool(decision.get("approved", False)),
            "requires_review": requires_review,
            "alignment_score": decision.get("alignment_score"),
            "confidence": decision.get("confidence"),
            "coverage": decision.get("coverage"),
            "risk": risk.get("total_risk"),
            "correction_action": decision.get("correction_action"),
            "timestamp": utc_timestamp(),
        }
        priority = 0.90 if requires_review else 0.55

        written: List[str] = []
        for key, payload, tags in (
            (result_key, result, ["alignment", "result", status]),
            (summary_key, summary, ["alignment", "summary", status]),
            (self.shared_latest_key, summary, ["alignment", "latest", status]),
        ):
            stored = self._shared_write(
                key,
                payload,
                tags=tags,
                priority=priority,
            )
            if stored:
                written.append(stored)

        self._shared_publish(
            self.shared_event_channel,
            {
                "event_type": "alignment_assessment_completed",
                "agent": self.name,
                "agent_id": self.agent_id,
                "summary": summary,
            },
        )

        if requires_review and self.log_shared_interventions:
            intervention_logger = getattr(self.shared_memory, "log_intervention", None)
            if callable(intervention_logger):
                intervention_report = {
                    "source": "AlignmentAgent",
                    "audit_id": audit_id,
                    "task_id": result.get("task_id"),
                    "summary": summary,
                    "decision": json_safe(decision),
                }

                def log_intervention() -> Any:
                    return intervention_logger(report=intervention_report)

                if self.fail_closed_on_shared_memory_error:
                    try:
                        self.last_intervention_report = dict(log_intervention() or {})
                    except Exception as exc:
                        raise InterventionError(
                            "Failed to record required alignment intervention.",
                            context={"audit_id": audit_id},
                            cause=exc,
                        ) from exc
                else:
                    logged = self._run_optional_runtime_operation(
                        "telemetry",
                        "alignment.shared_memory.intervention",
                        log_intervention,
                        default={},
                    )
                    if isinstance(logged, Mapping):
                        self.last_intervention_report = dict(logged)

        self._persist_shared_runtime_state()
        return written

    # ------------------------------------------------------------------
    # Input normalization / operation budget
    # ------------------------------------------------------------------
    def _check_operation_budget(self) -> None:
        now = _time.monotonic()
        with self._alignment_lock:
            if now < self._cooldown_until:
                raise AlignmentTimeoutError(
                    "AlignmentAgent operation limiter is in cooldown.",
                    context={"cooldown_until_monotonic": self._cooldown_until},
                )
            while (
                self._operation_times
                and now - self._operation_times[0] >= self.operation_limit_interval
            ):
                self._operation_times.popleft()
            if len(self._operation_times) >= self.operation_limit_max_requests:
                self._cooldown_until = now + self.operation_limit_cooldown_seconds
                raise AlignmentTimeoutError(
                    "AlignmentAgent operation budget exceeded.",
                    context={
                        "max_requests": self.operation_limit_max_requests,
                        "interval_seconds": self.operation_limit_interval,
                        "cooldown_seconds": self.operation_limit_cooldown_seconds,
                    },
                )
            self._operation_times.append(now)

    def _normalize_task_payload(
        self,
        task_data: Any,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any]
        if isinstance(task_data, Mapping):
            payload = dict(task_data)
        else:
            try:
                import pandas as pd  # type: ignore

                payload = {"input_data": task_data} if isinstance(task_data, pd.DataFrame) else {"payload": task_data}
            except ImportError:
                payload = {"payload": task_data}

        existing_context = payload.get("context")
        merged_context = dict(existing_context) if isinstance(existing_context, Mapping) else {}
        if context:
            merged_context.update(dict(context))
        payload["context"] = merged_context

        payload.setdefault("task_id", generate_identifier("align_task"))
        payload.setdefault("audit_id", generate_audit_id())
        payload.setdefault("sensitive_attributes", list(self.sensitive_attributes))
        payload.setdefault("enable_bias_detection", self.enable_bias_detection)
        payload.setdefault("enable_individual_fairness", self.enable_individual_fairness)
        payload.setdefault("enable_counterfactual_audit", self.enable_counterfactual_audit)
        payload.setdefault("enable_value_alignment", self.enable_value_alignment)
        payload.setdefault("enable_ethics_check", self.enable_ethics_check)

        if "input_data" not in payload:
            for key in ("data", "features", "records"):
                if key in payload and payload[key] is not None:
                    payload["input_data"] = payload[key]
                    break

        return payload

    # ------------------------------------------------------------------
    # Public alignment operations
    # ------------------------------------------------------------------
    @property
    def predict_func(self) -> Optional[Callable[[Any], Any]]:
        return self.subsystem.predict_func

    @predict_func.setter
    def predict_func(self, func: Optional[Callable[[Any], Any]]) -> None:
        self.subsystem.predict_func = func
        setter = getattr(self.counterfactual_auditor, "set_model_predict_func", None)
        if callable(setter) and func is not None:
            setter(func)

    def verify_alignment(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            if not self.enabled:
                raise AlignmentStateError("AlignmentAgent is disabled by configuration.")
            self._check_operation_budget()
            payload = self._normalize_task_payload(task_data)
            result = self._verify_alignment_impl(payload, record_trajectory=True)
            result["shared_memory_keys"] = self._publish_alignment_result(result)
            return result
        except AlignmentError:
            raise
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=RiskAssessmentError,
                message="Failed to verify alignment for the requested task.",
                context={
                    "task_keys": list(task_data.keys())
                    if isinstance(task_data, Mapping)
                    else None,
                },
            ) from exc

    def _verify_alignment_impl(
        self,
        payload: Mapping[str, Any],
        *,
        record_trajectory: bool,
    ) -> Dict[str, Any]:
        audit_id = str(payload.get("audit_id") or generate_audit_id())
        task_id = str(payload.get("task_id") or generate_identifier("align_task"))

        try:
            assessment, component_report, evidence = self.subsystem.assess(
                payload,
                assessment_id=audit_id,
                record_trajectory=record_trajectory,
            )
        except AlignmentError:
            raise
        except Exception as exc:
            raise RiskAssessmentError(
                "Alignment assessment orchestration failed.",
                context={"audit_id": audit_id, "task_id": task_id},
                cause=exc,
            ) from exc

        risk = self._risk_from_assessment(assessment, component_report)
        correction = self._correction_from_risk(assessment, risk)
        decision = self._decision_from_assessment(assessment, risk, correction)

        result = {
            "audit_id": audit_id,
            "task_id": task_id,
            "alignment_report": assessment.to_dict(),
            "component_report": json_safe(component_report),
            "evidence": [item.to_dict() for item in evidence],
            "risk_assessment": risk.to_dict(),
            "decision": decision,
            "correction": correction.to_dict(),
            "sensitive_attributes": list(payload.get("sensitive_attributes") or self.sensitive_attributes),
        }

        self.audit_counter += 1
        self.last_alignment_report = dict(result["alignment_report"])
        self.last_risk_assessment = dict(result["risk_assessment"])
        self.last_decision = dict(result["decision"])
        return result

    def align(
        self,
        input_data: Any,
        predictions: Any = None,
        labels: Any = None,
        task_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            payload = dict(task_context or {})
            payload["input_data"] = input_data
            if predictions is not None:
                payload["predictions"] = predictions
            if labels is not None:
                payload["labels"] = labels
            payload = self._normalize_task_payload(payload)

            self._check_operation_budget()
            result = self._verify_alignment_impl(payload, record_trajectory=True)
            correction = result["correction"]
            result["applied_correction"] = {
                "applied": False,
                "action": correction.get("action"),
                "reason": (
                    "alignment_agent_emits_review_or_correction_recommendations_"
                    "but_does_not_mutate_external_model_policy_state"
                ),
                "delegation_required": correction.get("action")
                not in {"none", "alert_only"},
            }
            result["operational_state"] = self.operational_state
            result["shared_memory_keys"] = self._publish_alignment_result(result)
            return result
        except AlignmentError:
            raise
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=InterventionError,
                message="Alignment control pass failed during execution.",
                context={
                    "operational_state": getattr(self, "operational_state", None),
                },
            ) from exc

    def predict(self, task_data: Any, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return self.verify_alignment(self._normalize_task_payload(task_data, context=context))

    def get_action(self, task_data: Any, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return dict(self.predict(task_data, context=context)["decision"])

    def act(
        self,
        task_data: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = self._normalize_task_payload(task_data, context=context)
        if "input_data" in payload:
            return self.align(
                payload["input_data"],
                predictions=payload.get("predictions"),
                labels=payload.get("labels"),
                task_context=payload,
            )
        self._check_operation_budget()
        result = self._verify_alignment_impl(payload, record_trajectory=True)
        result["applied_correction"] = {
            "applied": False,
            "action": result["correction"].get("action"),
            "reason": "assessment_only_no_external_state_mutation",
        }
        result["operational_state"] = self.operational_state
        result["shared_memory_keys"] = self._publish_alignment_result(result)
        return result

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        payload = self._normalize_task_payload(task_data)
        operation = str(
            payload.get("operation") or payload.get("task_type") or "verify"
        ).strip().lower()

        if operation in {"verify", "assess", "verify_alignment", "alignment"}:
            return self.verify_alignment(payload)
        if operation in {"act", "align", "control"}:
            return self.act(payload)
        if operation in {"feedback", "record_feedback"}:
            feedback = payload.get("feedback")
            feedback_payload = dict(feedback) if isinstance(feedback, Mapping) else payload
            return {"status": "recorded", "feedback": self.record_feedback(feedback_payload)}
        if operation in {"status", "health", "diagnostics"}:
            return self.get_diagnostics() if operation == "diagnostics" else self.get_status()
        if operation in {"history", "shared_history", "alignment_history"}:
            return self.get_shared_alignment_history(
                int(payload.get("limit", 20) or 20)
            )
        if operation in {"preserve_evidence", "snapshot_evidence"}:
            return self.preserve_evidence(
                reason=str(payload.get("reason") or "task_request"),
                include_component_report=bool(
                    payload.get("include_component_report", False)
                ),
            )

        raise ValidationError(f"Unsupported AlignmentAgent operation: {operation}", context={"operation": operation})

    # ------------------------------------------------------------------
    # Assessment -> risk/decision compatibility layer
    # ------------------------------------------------------------------
    def _risk_from_assessment(self, assessment: AlignmentAssessment, component_report: Mapping[str, Any]) -> RiskAssessment:
        component_risks = {
            name: 1.0 - float(summary.score)
            for name, summary in assessment.dimensions.items()
            if summary.score is not None
        }
        total_risk = (
            None
            if assessment.aggregate_score is None
            else 1.0 - float(assessment.aggregate_score)
        )
        triggered = {
            name: risk
            for name, risk in component_risks.items()
            if risk > self.risk_threshold
        }
        return RiskAssessment(
            total_risk=total_risk,
            threshold=self.risk_threshold,
            component_risks=component_risks,
            component_metrics={
                "coverage": assessment.coverage,
                "confidence": assessment.confidence,
                "uncertainty": assessment.uncertainty,
                "conflict_count": len(assessment.conflicts),
            },
            triggered_thresholds=triggered,
            ethical_violations_details=[item.description for item in assessment.conflicts],
            drift_detected=assessment.drift.detected,
            drift_score=assessment.drift.magnitude,
            status=assessment.status.value,
            metadata={"component_report_available": bool(component_report)},
        )

    def _correction_from_risk(self, assessment: AlignmentAssessment, risk: RiskAssessment) -> CorrectionDecision:
        if assessment.status in {
            AlignmentStatus.CONFLICTING,
            AlignmentStatus.UNCERTAIN,
            AlignmentStatus.MISALIGNED,
        }:
            return CorrectionDecision(
                action="human_intervention",
                magnitude=1.0 if risk.total_risk is None else risk.total_risk,
                threshold=self.risk_threshold,
                target_components=list(assessment.required_missing)
                or list(risk.triggered_thresholds),
                safe_hold=True,
                requires_human=True,
                rationale=f"alignment status is {assessment.status.value}",
            )

        if assessment.drift.detected and assessment.drift.direction in {"negative", "mixed"}:
            return CorrectionDecision(
                action="review_required",
                magnitude=assessment.drift.magnitude,
                threshold=self.risk_threshold,
                target_components=list(assessment.drift.affected_dimensions),
                safe_hold=False,
                requires_human=True,
                rationale="negative or mixed alignment trajectory drift detected",
            )

        if risk.total_risk is not None:
            for level in self.correction_policy:
                if risk.total_risk >= level["threshold"]:
                    return CorrectionDecision(
                        action=level["action"],
                        magnitude=risk.total_risk,
                        threshold=level["threshold"],
                        target_components=list(risk.triggered_thresholds),
                        requires_human=level["action"]
                        in {"human_intervention", "review_required"},
                        rationale="configured alignment risk threshold crossed",
                    )

        return CorrectionDecision(
            action="none",
            magnitude=0.0 if risk.total_risk is None else risk.total_risk,
            threshold=self.risk_threshold,
            rationale="no corrective alignment action required",
        )

    def _decision_from_assessment(
        self,
        assessment: AlignmentAssessment,
        risk: RiskAssessment,
        correction: CorrectionDecision,
    ) -> Dict[str, Any]:
        drift_review = assessment.drift.detected and assessment.drift.direction in {
            "negative",
            "mixed",
        }
        requires_review = (assessment.requires_review or drift_review or correction.requires_human)
        risk_acceptable = (risk.total_risk is not None and risk.total_risk <= self.risk_threshold)
        approved = bool(
            not requires_review
            and risk_acceptable
            and assessment.status
            in {AlignmentStatus.ALIGNED, AlignmentStatus.PARTIALLY_ALIGNED}
        )
        return {
            "approved": approved,
            "requires_review": requires_review,
            "decision": "approved" if approved else "review_required",
            "alignment_status": assessment.status.value,
            "alignment_score": assessment.aggregate_score,
            "confidence": assessment.confidence,
            "coverage": assessment.coverage,
            "uncertainty": assessment.uncertainty,
            "risk": risk.total_risk,
            "risk_threshold": self.risk_threshold,
            "correction_action": correction.action,
            "reasons": list(assessment.explanation),
        }

    # ------------------------------------------------------------------
    # Feedback / state / diagnostics
    # ------------------------------------------------------------------
    def record_feedback(self, feedback: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(feedback, Mapping):
            raise ValidationError("Alignment feedback must be a mapping.")

        result = self.subsystem.record_feedback(feedback)
        self.last_feedback = dict(feedback)
        feedback_id = str(
            feedback.get("feedback_id")
            or feedback.get("task_id")
            or generate_identifier("align_feedback")
        )
        self._shared_write(
            f"{self.shared_feedback_key_prefix}:{feedback_id}",
            {
                "feedback": json_safe(feedback),
                "record": json_safe(result),
                "timestamp": utc_timestamp(),
            },
            tags=["alignment", "feedback"],
            priority=0.70,
        )
        self._shared_publish(
            self.shared_event_channel,
            {
                "event_type": "alignment_feedback_recorded",
                "agent": self.name,
                "feedback_id": feedback_id,
            },
        )
        self._persist_shared_runtime_state()
        return dict(result) if isinstance(result, Mapping) else {"result": result}

    def update_from_feedback(self, feedback: Mapping[str, Any]) -> Dict[str, Any]:
        """Backward-compatible alias; feedback is recorded, not self-applied."""
        return self.record_feedback(feedback)

    def get_shared_alignment_history(self, limit: int = 20) -> Dict[str, Any]:
        """Return bounded SharedMemory-backed alignment coordination history.

        This exposes only AlignmentAgent-owned keys/tags.  Private
        ``AlignmentMemory`` remains encapsulated in the alignment subsystem.
        """
        bounded_limit = coerce_positive_int(
            limit,
            field_name="alignment_agent.shared_history.limit",
        )
        result_items: List[Dict[str, Any]] = []
        getter = getattr(self.shared_memory, "get_by_tag", None)
        if callable(getter):
            try:
                tagged = getter("alignment", limit=max(bounded_limit * 3, bounded_limit))
                if isinstance(tagged, Sequence) and not isinstance(tagged, (str, bytes, bytearray)):
                    for item in tagged:
                        if not isinstance(item, Mapping):
                            continue
                        key = str(item.get("key", ""))
                        if key.startswith(self.shared_result_key_prefix):
                            result_items.append(json_safe(dict(item)))
                            if len(result_items) >= bounded_limit:
                                break
            except Exception as exc:
                self._mark_runtime_degraded(
                    "telemetry",
                    "alignment.shared_memory.history_by_tag",
                    exc,
                )

        state_versions: List[Dict[str, Any]] = []
        versions_getter = getattr(self.shared_memory, "get_all_versions", None)
        if callable(versions_getter):
            try:
                raw_versions = versions_getter(self.shared_state_key, update_access=False)
                versions = (
                    raw_versions
                    if isinstance(raw_versions, Sequence)
                    and not isinstance(raw_versions, (str, bytes, bytearray))
                    else []
                )
                for version in versions[-bounded_limit:]:
                    state_versions.append(
                        {
                            "timestamp": getattr(version, "timestamp", None),
                            "value": json_safe(getattr(version, "value", version)),
                        }
                    )
            except Exception as exc:
                self._mark_runtime_degraded(
                    "telemetry",
                    "alignment.shared_memory.state_versions",
                    exc,
                )

        return {
            "results": result_items,
            "runtime_state_versions": state_versions,
            "limit": bounded_limit,
        }

    def preserve_evidence(self, *, reason: str = "manual", include_component_report: bool = False) -> Dict[str, Any]:
        """Persist a bounded AlignmentAgent evidence snapshot in SharedMemory.

        The snapshot intentionally contains alignment-owned state only; it does
        not dump unrelated global SharedMemory content.
        """
        snapshot_id = generate_identifier("align_evidence")
        payload: Dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "reason": str(reason or "manual"),
            "timestamp": utc_timestamp(),
            "agent_id": self.agent_id,
            "last_alignment_report": self.last_alignment_report,
            "last_risk_assessment": self.last_risk_assessment,
            "last_decision": self.last_decision,
            "subsystem_state": self.subsystem.export_state(),
        }
        if include_component_report:
            payload["diagnostics"] = self.subsystem.health().to_dict()
            payload["alignment_memory_report"] = self.subsystem.memory_report()

        key = f"alignment_agent.evidence:{snapshot_id}"
        stored = self._shared_write(
            key,
            payload,
            tags=["alignment", "evidence"],
            priority=0.90,
        )
        return {
            "snapshot_id": snapshot_id,
            "shared_memory_key": stored,
            "stored": bool(stored),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.enabled else "disabled",
            "operational_state": self.operational_state,
            "runtime_status": self.runtime_status(),
            "alignment_runtime": self.subsystem.health().to_dict(),
            "last_decision": dict(self.last_decision),
            "last_risk_assessment": dict(self.last_risk_assessment),
            "shared_memory": {
                "enabled": self.publish_to_shared_memory,
                "latest_key": self.shared_latest_key,
                "state_key": self.shared_state_key,
                "event_channel": self.shared_event_channel,
            },
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        diagnostics = self.get_status()
        health_checker = getattr(self.shared_memory, "health_check", None)
        if callable(health_checker):
            health = self._run_optional_runtime_operation(
                "telemetry",
                "alignment.shared_memory.health_check",
                health_checker,
                default={"status": "unavailable"},
            )
            diagnostics["shared_memory_health"] = json_safe(health)
        diagnostics["alignment_memory"] = self.subsystem.memory_report()
        diagnostics["audit_counter"] = self.audit_counter
        return diagnostics

    def get_state(self) -> Dict[str, Any]:
        return {
            "operational_state": self.operational_state,
            "subsystem": self.subsystem.export_state(),
            "last_decision": dict(self.last_decision),
            "last_risk_assessment": dict(self.last_risk_assessment),
            "last_feedback": dict(self.last_feedback),
            "audit_counter": int(self.audit_counter),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise AlignmentStateError("AlignmentAgent state must be a mapping.")

        self.operational_state = str(
            state.get("operational_state", self.operational_state)
        )
        subsystem_state = state.get("subsystem")
        if isinstance(subsystem_state, Mapping):
            self.subsystem.import_state(subsystem_state)
        last_decision = state.get("last_decision")
        if isinstance(last_decision, Mapping):
            self.last_decision = dict(last_decision)
        last_risk = state.get("last_risk_assessment")
        if isinstance(last_risk, Mapping):
            self.last_risk_assessment = dict(last_risk)
        last_feedback = state.get("last_feedback")
        if isinstance(last_feedback, Mapping):
            self.last_feedback = dict(last_feedback)
        if "audit_counter" in state:
            self.audit_counter = max(0, int(state.get("audit_counter", 0)))
        self._persist_shared_runtime_state()

    def reset(self) -> None:
        self.last_alignment_report = {}
        self.last_risk_assessment = {}
        self.last_decision = {}
        self.last_intervention_report = {}
        self.last_feedback = {}
        self.audit_counter = 0
        self.operational_state = "ACTIVE"
        with self._alignment_lock:
            self._operation_times.clear()
            self._cooldown_until = 0.0
        self._persist_shared_runtime_state()
        self._publish_alignment_event("reset", {"agent_id": self.agent_id})


if __name__ == '__main__':
    print("\n=== Running Alignment Agent ===\n")
    printer.status("Init", "Alignment Agent initialized", "success")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory
    memory = SharedMemory()
    factory = AgentFactory()

    agent = AlignmentAgent(shared_memory=memory, agent_factory=factory)
    printer.pretty("START", agent, "success")

    print("\n=== Test ran successfully ===\n")
