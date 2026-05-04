from __future__ import annotations

__version__ = "2.1.0"

"""
Production-grade Collaborative Agent built on the shared BaseAgent architecture.

The CollaborativeAgent is the agent-facing orchestration layer for SLAI's
collaborative runtime. It coordinates safety assessment, multi-agent assignment,
manager-backed delegation, shared-memory coordination, and operator-facing
state publication while keeping low-level subsystem ownership in the dedicated
collaborative modules.

Responsibilities
----------------
- Provide a stable BaseAgent-compatible interface for collaborative operations.
- Assess task and agent risk through an adaptive Bayesian risk model.
- Coordinate task-to-agent assignments when the caller supplies explicit agents.
- Delegate executable tasks through CollaborationManager without importing or
  owning TaskRouter internals.
- Publish bounded assessments, coordination decisions, metrics, snapshots, and
  audit events into shared memory.
- Serialize and restore agent-local coordination state for warm starts.
- Use collaboration errors and collaborative helpers consistently at runtime
  boundaries.

Design principles
-----------------
1. Direct local imports: project-local BaseAgent, config, manager, policy,
   contracts, error, and helper imports remain explicit and unwrapped.
2. No circular router ownership: this agent imports CollaborationManager as the
   subsystem facade and does not import TaskRouter directly.
3. Separation of concerns: registry discovery, route execution, contract rule
   internals, policy condition semantics, strategy scoring, reliability circuit
   transitions, and shared-memory storage remain in their dedicated modules.
4. Config-backed behavior: CollaborativeAgent tuning belongs in
   agents_config.yaml under ``collaborative_agent``.
5. Operational transparency: all diagnostics are JSON-safe, redacted by default,
   bounded, and suitable for logs, shared memory, test assertions, and incident
   reports.
"""

import inspect
import json
import threading
import time

from collections import deque
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .collaborative.collaboration_manager import CollaborationManager
from .collaborative.policy_engine import PolicyDecision, PolicyEngine
from .collaborative.task_contracts import TaskContractRegistry
from .collaborative.utils.collaboration_error import *
from .collaborative.utils.collaborative_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Collaborative Agent")
printer = PrettyPrinter()


class RiskLevel(str, Enum):
    """Normalized risk labels emitted by safety assessments."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CollaborativeAgentMode(str, Enum):
    """Supported perform_task modes."""

    ASSESS = "assess"
    ASSESS_TASK = "assess_task"
    COORDINATE = "coordinate"
    DELEGATE = "delegate"
    REGISTER_AGENT = "register_agent"
    EXPLAIN = "explain"
    SNAPSHOT = "snapshot"
    HEALTH = "health"
    METRICS = "metrics"
    SHARED_GET = "shared_get"
    SHARED_SET = "shared_set"
    SHARED_UPDATE = "shared_update"


class CollaborativeAgentEventType(str, Enum):
    """Shared audit/lifecycle event labels for this agent."""

    INITIALIZED = "collaborative_agent_initialized"
    RISK_ASSESSED = "risk_assessed"
    TASK_ASSESSED = "task_assessed"
    COORDINATION_STARTED = "coordination_started"
    COORDINATION_COMPLETED = "coordination_completed"
    COORDINATION_FAILED = "coordination_failed"
    TASK_DELEGATED = "task_delegated"
    DELEGATION_FAILED = "delegation_failed"
    AGENT_REGISTERED = "agent_registered"
    SHARED_MEMORY_UPDATED = "shared_memory_updated"
    STATE_SAVED = "state_saved"
    STATE_LOADED = "state_loaded"


@dataclass(frozen=True)
class SafetyAssessment:
    """Stable risk assessment artifact shared across collaborative modules."""

    risk_score: float
    threshold: float
    risk_level: RiskLevel
    recommended_action: str
    confidence: float
    source_agent: str = "unknown"
    task_type: str = "general"
    timestamp: float = field(default_factory=epoch_seconds)
    assessment_id: str = field(default_factory=lambda: generate_uuid("assessment", length=24))
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("assessment"))
    factors: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return self.risk_level == RiskLevel.CRITICAL

    @property
    def requires_review(self) -> bool:
        return self.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = asdict(self)
        payload["risk_level"] = self.risk_level.value if isinstance(self.risk_level, RiskLevel) else str(self.risk_level)
        payload["timestamp_utc"] = utc_timestamp()
        return redact_mapping(prune_none(payload, drop_empty=True)) if redact else prune_none(payload, drop_empty=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SafetyAssessment":
        data = dict(payload)
        data.pop("timestamp_utc", None)
        data["risk_level"] = normalize_risk_level(data.get("risk_level", RiskLevel.MODERATE.value))
        data["risk_score"] = clamp_score(data.get("risk_score", 0.5))
        data["threshold"] = clamp_score(data.get("threshold", 0.7))
        data["confidence"] = clamp_score(data.get("confidence", 0.0))
        data["factors"] = normalize_metadata(data.get("factors"), drop_none=True)
        data["metadata"] = normalize_metadata(data.get("metadata"), drop_none=True)
        return cls(**data)

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return stable_json_dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> "SafetyAssessment":
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class CoordinationAssignment:
    """One task assignment/rejection record from local coordination."""

    task_id: str
    status: str
    task_type: str = "general"
    agent: Optional[str] = None
    optimization_score: Optional[float] = None
    safety: Optional[Dict[str, Any]] = None
    policy: Optional[Dict[str, Any]] = None
    contract: Optional[Dict[str, Any]] = None
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class CoordinationResult:
    """Stable result artifact for a coordination run."""

    coordination_id: str
    status: str
    assignments: Dict[str, Dict[str, Any]]
    task_count: int
    assigned_count: int
    delegated_count: int
    rejected_count: int
    unassigned_count: int
    started_at: float
    finished_at: float
    duration_ms: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    optimization_goals: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        payload["started_at_utc"] = utc_timestamp()
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class DelegationRecord:
    """Bounded history entry for one manager-backed delegation."""

    delegation_id: str
    task_type: str
    status: str
    started_at: float
    finished_at: float
    duration_ms: float
    result_fingerprint: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self), drop_empty=True)
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class CollaborativeAgentConfig:
    """Runtime configuration loaded from agents_config.yaml."""

    enabled: bool = True
    use_collaboration_manager: bool = True
    initialize_local_policy_engine: bool = True
    initialize_local_contract_registry: bool = True
    register_default_task_contracts: bool = True
    register_default_policy_rules: bool = True
    delegate_during_coordination: bool = True
    local_fallback_when_delegation_fails: bool = True
    deny_critical_risk: bool = True
    review_high_risk: bool = True
    max_concurrent_tasks: int = 100
    load_factor: float = 0.75
    max_tasks_per_agent: int = DEFAULT_AGENT_TASK_MULTIPLIER
    risk_threshold: float = 0.70
    critical_risk_multiplier: float = 1.40
    moderate_risk_multiplier: float = 0.60
    policy_deny_risk_threshold: float = 0.98
    policy_review_risk_threshold: float = 0.85
    bayes_prior_alpha: float = 1.0
    bayes_prior_beta: float = 1.0
    bayes_decay: float = 1.0
    minimum_dynamic_threshold: float = 0.05
    maximum_dynamic_threshold: float = 0.95
    optimization_weight_capability: float = 0.50
    optimization_weight_load: float = 0.30
    optimization_weight_risk: float = 0.20
    optimization_weight_priority: float = 0.05
    default_task_type: str = "general"
    default_source_agent: str = "unknown"
    default_retries: int = 1
    coordination_history_limit: int = 500
    assessment_history_limit: int = 1000
    delegation_history_limit: int = 500
    metric_history_limit: int = 500
    shared_memory_enabled: bool = True
    audit_enabled: bool = True
    audit_key: str = "collaborative_agent:audit_events"
    audit_max_events: int = DEFAULT_MAX_AUDIT_EVENTS
    assessment_key: str = "collaborative_agent:last_assessment"
    assessment_history_key: str = "collaborative_agent:assessment_history"
    coordination_key: str = "collaborative_agent:last_coordination"
    coordination_history_key: str = "collaborative_agent:coordination_history"
    delegation_history_key: str = "collaborative_agent:delegation_history"
    status_key: str = "collaborative_agent:status"
    metrics_key: str = "collaborative_agent:metrics"
    health_key: str = "collaborative_agent:health"
    state_key: str = "collaborative_agent:state"
    publish_status: bool = True
    redact_shared_payloads: bool = True
    result_preview_length: int = 500

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "CollaborativeAgentConfig":
        source = dict(config or {})
        shared_memory_cfg = source.get("shared_memory") if isinstance(source.get("shared_memory"), Mapping) else {}
        risk_cfg = source.get("risk") if isinstance(source.get("risk"), Mapping) else {}
        policy_cfg = source.get("policy") if isinstance(source.get("policy"), Mapping) else {}
        coordination_cfg = source.get("coordination") if isinstance(source.get("coordination"), Mapping) else {}
        history_cfg = source.get("history") if isinstance(source.get("history"), Mapping) else {}
        audit_cfg = source.get("audit") if isinstance(source.get("audit"), Mapping) else {}
        merged = merge_mappings(source, risk_cfg, policy_cfg, coordination_cfg, history_cfg, shared_memory_cfg, audit_cfg, deep=True, drop_none=True)
        return cls(
            enabled=coerce_bool(merged.get("enabled"), default=cls.enabled),
            use_collaboration_manager=coerce_bool(merged.get("use_collaboration_manager"), default=cls.use_collaboration_manager),
            initialize_local_policy_engine=coerce_bool(merged.get("initialize_local_policy_engine"), default=cls.initialize_local_policy_engine),
            initialize_local_contract_registry=coerce_bool(merged.get("initialize_local_contract_registry"), default=cls.initialize_local_contract_registry),
            register_default_task_contracts=coerce_bool(merged.get("register_default_task_contracts"), default=cls.register_default_task_contracts),
            register_default_policy_rules=coerce_bool(merged.get("register_default_policy_rules"), default=cls.register_default_policy_rules),
            delegate_during_coordination=coerce_bool(merged.get("delegate_during_coordination"), default=cls.delegate_during_coordination),
            local_fallback_when_delegation_fails=coerce_bool(merged.get("local_fallback_when_delegation_fails"), default=cls.local_fallback_when_delegation_fails),
            deny_critical_risk=coerce_bool(merged.get("deny_critical_risk"), default=cls.deny_critical_risk),
            review_high_risk=coerce_bool(merged.get("review_high_risk"), default=cls.review_high_risk),
            max_concurrent_tasks=coerce_int(merged.get("max_concurrent_tasks"), default=cls.max_concurrent_tasks, minimum=1),
            load_factor=coerce_float(merged.get("load_factor"), default=cls.load_factor, minimum=0.0),
            max_tasks_per_agent=coerce_int(merged.get("max_tasks_per_agent"), default=cls.max_tasks_per_agent, minimum=1),
            risk_threshold=clamp_score(merged.get("risk_threshold", cls.risk_threshold)),
            critical_risk_multiplier=coerce_float(merged.get("critical_risk_multiplier"), default=cls.critical_risk_multiplier, minimum=1.0),
            moderate_risk_multiplier=coerce_float(merged.get("moderate_risk_multiplier"), default=cls.moderate_risk_multiplier, minimum=0.0, maximum=1.0),
            policy_deny_risk_threshold=clamp_score(merged.get("policy_deny_risk_threshold", cls.policy_deny_risk_threshold)),
            policy_review_risk_threshold=clamp_score(merged.get("policy_review_risk_threshold", cls.policy_review_risk_threshold)),
            bayes_prior_alpha=coerce_float(merged.get("bayes_prior_alpha"), default=cls.bayes_prior_alpha, minimum=0.01),
            bayes_prior_beta=coerce_float(merged.get("bayes_prior_beta"), default=cls.bayes_prior_beta, minimum=0.01),
            bayes_decay=coerce_float(merged.get("bayes_decay"), default=cls.bayes_decay, minimum=0.0, maximum=1.0),
            minimum_dynamic_threshold=clamp_score(merged.get("minimum_dynamic_threshold", cls.minimum_dynamic_threshold)),
            maximum_dynamic_threshold=clamp_score(merged.get("maximum_dynamic_threshold", cls.maximum_dynamic_threshold)),
            optimization_weight_capability=coerce_float(merged.get("optimization_weight_capability"), default=cls.optimization_weight_capability, minimum=0.0),
            optimization_weight_load=coerce_float(merged.get("optimization_weight_load"), default=cls.optimization_weight_load, minimum=0.0),
            optimization_weight_risk=coerce_float(merged.get("optimization_weight_risk"), default=cls.optimization_weight_risk, minimum=0.0),
            optimization_weight_priority=coerce_float(merged.get("optimization_weight_priority"), default=cls.optimization_weight_priority),
            default_task_type=str(merged.get("default_task_type", cls.default_task_type)).strip() or cls.default_task_type,
            default_source_agent=str(merged.get("default_source_agent", cls.default_source_agent)).strip() or cls.default_source_agent,
            default_retries=coerce_int(merged.get("default_retries"), default=cls.default_retries, minimum=1),
            coordination_history_limit=coerce_int(merged.get("coordination_history_limit"), default=cls.coordination_history_limit, minimum=1),
            assessment_history_limit=coerce_int(merged.get("assessment_history_limit"), default=cls.assessment_history_limit, minimum=1),
            delegation_history_limit=coerce_int(merged.get("delegation_history_limit"), default=cls.delegation_history_limit, minimum=1),
            metric_history_limit=coerce_int(merged.get("metric_history_limit"), default=cls.metric_history_limit, minimum=1),
            shared_memory_enabled=coerce_bool(merged.get("shared_memory_enabled", merged.get("enabled")), default=cls.shared_memory_enabled),
            audit_enabled=coerce_bool(merged.get("audit_enabled", merged.get("enabled")), default=cls.audit_enabled),
            audit_key=str(merged.get("audit_key", merged.get("key", cls.audit_key))).strip() or cls.audit_key,
            audit_max_events=coerce_int(merged.get("audit_max_events", merged.get("max_events")), default=cls.audit_max_events, minimum=1),
            assessment_key=str(merged.get("assessment_key", cls.assessment_key)).strip() or cls.assessment_key,
            assessment_history_key=str(merged.get("assessment_history_key", cls.assessment_history_key)).strip() or cls.assessment_history_key,
            coordination_key=str(merged.get("coordination_key", cls.coordination_key)).strip() or cls.coordination_key,
            coordination_history_key=str(merged.get("coordination_history_key", cls.coordination_history_key)).strip() or cls.coordination_history_key,
            delegation_history_key=str(merged.get("delegation_history_key", cls.delegation_history_key)).strip() or cls.delegation_history_key,
            status_key=str(merged.get("status_key", cls.status_key)).strip() or cls.status_key,
            metrics_key=str(merged.get("metrics_key", cls.metrics_key)).strip() or cls.metrics_key,
            health_key=str(merged.get("health_key", cls.health_key)).strip() or cls.health_key,
            state_key=str(merged.get("state_key", cls.state_key)).strip() or cls.state_key,
            publish_status=coerce_bool(merged.get("publish_status"), default=cls.publish_status),
            redact_shared_payloads=coerce_bool(merged.get("redact_shared_payloads"), default=cls.redact_shared_payloads),
            result_preview_length=coerce_int(merged.get("result_preview_length"), default=cls.result_preview_length, minimum=1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BayesianRiskModel:
    """Thread-safe beta-binomial model for adaptive risk thresholds."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, *, decay: float = 1.0):
        self._default_alpha = max(float(alpha), 0.01)
        self._default_beta = max(float(beta), 0.01)
        self._decay = max(0.0, min(1.0, float(decay)))
        self._posterior: Dict[str, List[float]] = {}
        self._event_counts: Dict[str, int] = {}
        self._updated_at: Dict[str, float] = {}
        self._lock = threading.RLock()

    def _ensure_key(self, key: str) -> None:
        normalized = normalize_risk_key(key)
        if normalized not in self._posterior:
            self._posterior[normalized] = [self._default_alpha, self._default_beta]
            self._event_counts[normalized] = 0
            self._updated_at[normalized] = epoch_seconds()

    def update(self, key: str, event_was_safe: bool, *, weight: float = 1.0) -> None:
        """Update the posterior with one safe/unsafe observation."""

        normalized = normalize_risk_key(key)
        with self._lock:
            self._ensure_key(normalized)
            alpha, beta = self._posterior[normalized]
            if self._decay < 1.0:
                alpha = self._default_alpha + ((alpha - self._default_alpha) * self._decay)
                beta = self._default_beta + ((beta - self._default_beta) * self._decay)
            if event_was_safe:
                alpha += max(0.0, float(weight))
            else:
                beta += max(0.0, float(weight))
            self._posterior[normalized] = [alpha, beta]
            self._event_counts[normalized] = int(self._event_counts.get(normalized, 0)) + 1
            self._updated_at[normalized] = epoch_seconds()

    def update_from_assessment(self, assessment: SafetyAssessment) -> None:
        safe = assessment.risk_level in {RiskLevel.LOW, RiskLevel.MODERATE}
        weight = 1.0 + abs(assessment.risk_score - assessment.threshold)
        self.update(f"task:{assessment.task_type}", safe, weight=weight)
        self.update(f"agent:{assessment.source_agent}", safe, weight=weight)

    def threshold(self, key: str, fallback: float = 0.7, *, minimum: float = 0.05, maximum: float = 0.95) -> float:
        """Return an adaptive threshold for a key.

        Higher observed safe rates make the threshold more permissive while
        remaining bounded by configured operational limits.
        """

        normalized = normalize_risk_key(key)
        with self._lock:
            self._ensure_key(normalized)
            alpha, beta = self._posterior[normalized]
            total = alpha + beta
            safe_rate = alpha / total if total else fallback
            value = 0.25 + (0.70 * safe_rate)
            return clamp_score(value, minimum=minimum, maximum=maximum)

    def unsafe_probability(self, key: str) -> float:
        normalized = normalize_risk_key(key)
        with self._lock:
            self._ensure_key(normalized)
            alpha, beta = self._posterior[normalized]
            total = alpha + beta
            return clamp_score(beta / total if total else 0.5)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "posterior": {key: [values[0], values[1]] for key, values in self._posterior.items()},
                "event_counts": dict(self._event_counts),
                "updated_at": dict(self._updated_at),
                "default_alpha": self._default_alpha,
                "default_beta": self._default_beta,
                "decay": self._decay,
            }

    def restore(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            posterior = payload.get("posterior", payload)
            if isinstance(posterior, Mapping):
                for key, values in posterior.items():
                    if isinstance(values, Sequence) and len(values) == 2:
                        self._posterior[normalize_risk_key(key)] = [max(float(values[0]), 0.01), max(float(values[1]), 0.01)]
            event_counts = payload.get("event_counts", {}) if isinstance(payload, Mapping) else {}
            if isinstance(event_counts, Mapping):
                self._event_counts.update({normalize_risk_key(k): coerce_int(v, default=0, minimum=0) for k, v in event_counts.items()})
            updated_at = payload.get("updated_at", {}) if isinstance(payload, Mapping) else {}
            if isinstance(updated_at, Mapping):
                self._updated_at.update({normalize_risk_key(k): coerce_float(v, default=epoch_seconds()) for k, v in updated_at.items()})


def clamp_score(value: Any, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Coerce a value into a bounded score."""

    return float(clamp(coerce_float(value, default=0.0), minimum, maximum))


def normalize_risk_level(value: Any) -> RiskLevel:
    text = normalize_whitespace(value).lower()
    aliases = {
        "medium": RiskLevel.MODERATE.value,
        "review": RiskLevel.HIGH.value,
        "block": RiskLevel.CRITICAL.value,
    }
    text = aliases.get(text, text)
    try:
        return RiskLevel(text)
    except Exception:
        return RiskLevel.MODERATE


def normalize_risk_key(value: Any) -> str:
    return normalize_identifier_component(value, default="unknown", lowercase=True, separator=":")


def _error_payload(exc: BaseException, *, action: str = "collaborative_agent") -> Dict[str, Any]:
    return exception_to_error_payload(exc, action=action).get("error", {"type": type(exc).__name__, "message": str(exc)})


def _make_agent_exception(class_name: str, message: str, *, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None, **kwargs: Any) -> Exception:
    """Construct a collaboration exception while remaining compatible with older constructors."""

    cls = globals().get(class_name)
    if isinstance(cls, type) and issubclass(cls, Exception):
        attempts = (
            lambda: cls(message=message, context=context, cause=cause, **kwargs), # type: ignore
            lambda: cls(message, context=context, cause=cause, **kwargs), # type: ignore
            lambda: cls(message, context=context, **kwargs), # type: ignore
            lambda: cls(context.get("task_type", "unknown") if context else "unknown", message),
            lambda: cls(message),
        )
        for build in attempts:
            try:
                return build()
            except TypeError:
                continue
    return make_collaboration_exception(class_name, message, context=context, cause=cause, **kwargs)


def _policy_decision_value(policy_evaluation: Any) -> str:
    data = policy_evaluation_to_dict(policy_evaluation) or {}
    decision = data.get("decision", "allow")
    if isinstance(decision, Enum):
        decision = decision.value
    return str(decision or "allow").strip().lower()


class CollaborativeAgent(BaseAgent):
    """Production-grade orchestrator for collaborative safety and coordination."""

    capabilities = [
        "coordination",
        "safety_assessment",
        "shared_memory",
        "task_delegation",
        "collaboration_management",
    ]

    def __init__(self, shared_memory=None, agent_factory=None, config: Optional[Mapping[str, Any]] = None, **kwargs: Any):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.name = "CollaborativeAgent"
        self._lock = threading.RLock()

        self.config = load_global_config()
        self.global_config = self.config
        self.collaborative_config: Dict[str, Any] = dict(get_config_section("collaborative_agent") or {})
        if config:
            self.collaborative_config.update(dict(config))
        if kwargs:
            self.collaborative_config.update(dict(kwargs))
        self.runtime_config = CollaborativeAgentConfig.from_config(self.collaborative_config)

        self.enabled = self.runtime_config.enabled
        self._risk_model = BayesianRiskModel(
            alpha=self.runtime_config.bayes_prior_alpha,
            beta=self.runtime_config.bayes_prior_beta,
            decay=self.runtime_config.bayes_decay,
        )
        self._assessment_history: Deque[Dict[str, Any]] = deque(maxlen=self.runtime_config.assessment_history_limit)
        self._coordination_history: Deque[Dict[str, Any]] = deque(maxlen=self.runtime_config.coordination_history_limit)
        self._delegation_history: Deque[Dict[str, Any]] = deque(maxlen=self.runtime_config.delegation_history_limit)
        self._metric_events: Deque[Dict[str, Any]] = deque(maxlen=self.runtime_config.metric_history_limit)

        self._metrics: Dict[str, float] = {
            "assessments_completed": 0.0,
            "high_risk_interventions": 0.0,
            "critical_risk_blocks": 0.0,
            "tasks_coordinated": 0.0,
            "coordination_runs": 0.0,
            "coordination_failures": 0.0,
            "avg_assessment_latency_ms": 0.0,
            "avg_coordination_latency_ms": 0.0,
            "delegated_tasks": 0.0,
            "delegation_failures": 0.0,
            "contract_validation_failures": 0.0,
            "policy_denials": 0.0,
            "policy_review_flags": 0.0,
            "local_assignments": 0.0,
            "unassigned_tasks": 0.0,
            "shared_memory_writes": 0.0,
            "shared_memory_failures": 0.0,
        }

        self.policy_engine: Optional[PolicyEngine] = PolicyEngine() if self.runtime_config.initialize_local_policy_engine else None
        self.task_contracts: Optional[TaskContractRegistry] = TaskContractRegistry() if self.runtime_config.initialize_local_contract_registry else None

        self.collaboration_manager: Optional[CollaborationManager] = None
        if self.runtime_config.use_collaboration_manager:
            self.collaboration_manager = self._build_collaboration_manager()
            self._adopt_manager_components()

        if self.task_contracts is not None and self.runtime_config.register_default_task_contracts:
            self._register_default_task_contracts()
        if self.policy_engine is not None and self.runtime_config.register_default_policy_rules:
            self._register_default_policy_rules()

        self._record_event(
            CollaborativeAgentEventType.INITIALIZED.value,
            "Collaborative agent initialized.",
            metadata={"config": self.runtime_config.to_dict(), "manager_enabled": self.collaboration_manager is not None},
        )
        self._publish_status()
        logger.info("Collaborative Agent initialized")

    # ------------------------------------------------------------------
    # Manager and component wiring
    # ------------------------------------------------------------------
    def _build_collaboration_manager(self) -> CollaborationManager:
        signature = inspect.signature(CollaborationManager.__init__)
        kwargs: Dict[str, Any] = {"shared_memory": self.shared_memory}
        if "config" in signature.parameters:
            kwargs["config"] = self.collaborative_config.get("manager_config", {})
        if "policy_engine" in signature.parameters and self.policy_engine is not None:
            kwargs["policy_engine"] = self.policy_engine
        if "contract_registry" in signature.parameters and self.task_contracts is not None:
            kwargs["contract_registry"] = self.task_contracts
        return CollaborationManager(**kwargs)

    def _adopt_manager_components(self) -> None:
        """Reuse manager-owned policy/contracts when exposed, avoiding duplicates."""

        if self.collaboration_manager is None:
            return
        manager_contracts = getattr(self.collaboration_manager, "contract_registry", None)
        if manager_contracts is None:
            manager_contracts = getattr(self.collaboration_manager, "task_contracts", None)
        if manager_contracts is not None:
            self.task_contracts = manager_contracts
        manager_policy = getattr(self.collaboration_manager, "policy_engine", None)
        if manager_policy is not None:
            self.policy_engine = manager_policy

    # ------------------------------------------------------------------
    # Shared memory wrappers
    # ------------------------------------------------------------------
    def shared_get(self, key: str, default: Any = None) -> Any:
        return memory_get(self.shared_memory, key, default=default)

    def shared_set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        if not self.runtime_config.shared_memory_enabled:
            return False
        payload = redact_mapping(value) if self.runtime_config.redact_shared_payloads and isinstance(value, Mapping) else json_safe(value)
        ok = memory_set(self.shared_memory, key, payload, ttl=ttl)
        self._update_metric("shared_memory_writes" if ok else "shared_memory_failures", 1)
        return ok

    def shared_update(self, key: str, updates: Mapping[str, Any], ttl: Optional[float] = None) -> Dict[str, Any]:
        current = self.shared_get(key, default={})
        if not isinstance(current, Mapping):
            current = {"value": current}
        merged = merge_mappings(current, updates, deep=True, drop_none=False)
        if not self.shared_set(key, merged, ttl=ttl):
            raise _make_agent_exception(
                "SharedMemoryFailureError",
                f"Failed to update shared memory key '{key}'.",
                context={"key": key, "updates": updates},
            )
        self._record_event(CollaborativeAgentEventType.SHARED_MEMORY_UPDATED.value, f"Updated shared memory key '{key}'.", metadata={"key": key})
        return merged

    def shared_append(self, key: str, value: Any, *, ttl: Optional[float] = None, priority: Optional[int] = None) -> bool:
        if not self.runtime_config.shared_memory_enabled:
            return False
        ok = memory_append(self.shared_memory, key, value, ttl=ttl, priority=priority)
        self._update_metric("shared_memory_writes" if ok else "shared_memory_failures", 1)
        return ok

    # ------------------------------------------------------------------
    # Default contracts and policies
    # ------------------------------------------------------------------
    def _register_default_task_contracts(self) -> None:
        if self.task_contracts is None:
            return
        self.task_contracts.register_contract(
            task_type="general",
            required_fields=["id", "type"],
            optional_fields=["requirements", "priority", "deadline", "estimated_risk", "payload", "metadata"],
            field_types={
                "id": (str, int),
                "type": (str,),
                "requirements": (list, tuple),
                "priority": (int, float),
                "estimated_risk": (int, float),
                "payload": (dict,),
                "metadata": (dict,),
            },
            allow_unknown_fields=True,
        )
        self.task_contracts.register_contract(
            task_type="coordination",
            required_fields=["tasks", "available_agents"],
            optional_fields=["optimization_goals", "constraints"],
            field_types={
                "tasks": (list,),
                "available_agents": (dict,),
                "optimization_goals": (list, tuple),
                "constraints": (dict,),
            },
            allow_unknown_fields=True,
        )

    def _register_default_policy_rules(self) -> None:
        if self.policy_engine is None:
            return
        existing = {str(rule.get("rule_id")) for rule in self.policy_engine.list_rules()} if hasattr(self.policy_engine, "list_rules") else set()
        if "deny_critical_risk" not in existing:
            self.policy_engine.add_simple_rule(
                rule_id="deny_critical_risk",
                description=f"Task denied because estimated risk is above {self.runtime_config.policy_deny_risk_threshold}.",
                effect=PolicyDecision.DENY,
                priority=10,
                predicate=lambda task, _agent, _ctx: clamp_score(task.get("estimated_risk", 0.0)) >= self.runtime_config.policy_deny_risk_threshold,
            )
        if "review_high_risk" not in existing:
            self.policy_engine.add_simple_rule(
                rule_id="review_high_risk",
                description=f"Task requires review because estimated risk is above {self.runtime_config.policy_review_risk_threshold}.",
                effect=PolicyDecision.REQUIRE_REVIEW,
                priority=20,
                predicate=lambda task, _agent, _ctx: clamp_score(task.get("estimated_risk", 0.0)) >= self.runtime_config.policy_review_risk_threshold,
            )

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------
    def assess_risk(
        self,
        risk_score: float,
        task_type: str = "general",
        source_agent: str = "unknown",
        context: Optional[Mapping[str, Any]] = None,
    ) -> SafetyAssessment:
        start_ms = monotonic_ms()
        try:
            normalized_task_type = normalize_task_type(task_type or self.runtime_config.default_task_type, allow_empty=False)
            normalized_agent = normalize_agent_name(source_agent or self.runtime_config.default_source_agent)
            normalized_risk = clamp_score(risk_score)
            context_map = normalize_metadata(context, drop_none=True)

            base_threshold = self.runtime_config.risk_threshold
            task_threshold = self._risk_model.threshold(
                f"task:{normalized_task_type}",
                base_threshold,
                minimum=self.runtime_config.minimum_dynamic_threshold,
                maximum=self.runtime_config.maximum_dynamic_threshold,
            )
            agent_threshold = self._risk_model.threshold(
                f"agent:{normalized_agent}",
                base_threshold,
                minimum=self.runtime_config.minimum_dynamic_threshold,
                maximum=self.runtime_config.maximum_dynamic_threshold,
            )
            contextual_threshold = self._contextual_threshold(base_threshold, context_map)
            dynamic_threshold = min(base_threshold, task_threshold, agent_threshold, contextual_threshold)

            if normalized_risk >= dynamic_threshold * self.runtime_config.critical_risk_multiplier:
                level = RiskLevel.CRITICAL
                action = "halt_and_escalate"
            elif normalized_risk >= dynamic_threshold:
                level = RiskLevel.HIGH
                action = "human_review"
            elif normalized_risk >= dynamic_threshold * self.runtime_config.moderate_risk_multiplier:
                level = RiskLevel.MODERATE
                action = "proceed_with_guardrails"
            else:
                level = RiskLevel.LOW
                action = "proceed"

            confidence = clamp_score(1.0 - abs(normalized_risk - dynamic_threshold))
            assessment = SafetyAssessment(
                risk_score=normalized_risk,
                threshold=dynamic_threshold,
                risk_level=level,
                recommended_action=action,
                confidence=confidence,
                source_agent=normalized_agent,
                task_type=normalized_task_type,
                factors={
                    "base_threshold": base_threshold,
                    "task_threshold": task_threshold,
                    "agent_threshold": agent_threshold,
                    "contextual_threshold": contextual_threshold,
                    "unsafe_probability_task": self._risk_model.unsafe_probability(f"task:{normalized_task_type}"),
                    "unsafe_probability_agent": self._risk_model.unsafe_probability(f"agent:{normalized_agent}"),
                },
                metadata={"context": context_map, "duration_ms": elapsed_ms(start_ms)},
            )
            self._risk_model.update_from_assessment(assessment)
            self._record_assessment(assessment)
            self._update_metric("assessments_completed", 1)
            self._rolling_metric("avg_assessment_latency_ms", elapsed_ms(start_ms), "assessments_completed")
            if assessment.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
                self._update_metric("high_risk_interventions", 1)
            if assessment.risk_level == RiskLevel.CRITICAL:
                self._update_metric("critical_risk_blocks", 1)
            return assessment
        except Exception as exc:
            raise _make_agent_exception(
                "DelegationFailureError",
                "Risk assessment failed.",
                context={"risk_score": risk_score, "task_type": task_type, "source_agent": source_agent},
                cause=exc,
            ) from exc

    def assess_task(self, task: Mapping[str, Any], *, source_agent: str = "unknown", context: Optional[Mapping[str, Any]] = None) -> SafetyAssessment:
        payload = normalize_task_payload(task, allow_none=False)
        task_type = str(payload.get("type", payload.get("task_type", self.runtime_config.default_task_type)))
        risk = payload.get("estimated_risk", payload.get("risk_score", self._estimate_payload_risk(payload)))
        return self.assess_risk(risk, task_type=task_type, source_agent=source_agent, context=merge_mappings(context, {"task": payload}, deep=True))

    def _contextual_threshold(self, base_threshold: float, context: Mapping[str, Any]) -> float:
        adjustment = 0.0
        if coerce_bool(context.get("external_user"), default=False):
            adjustment -= 0.05
        if coerce_bool(context.get("writes_to_memory"), default=False):
            adjustment -= 0.05
        if coerce_bool(context.get("requires_privileged_action"), default=False):
            adjustment -= 0.10
        if str(context.get("data_sensitivity", "")).lower() in {"high", "restricted", "secret"}:
            adjustment -= 0.10
        return clamp_score(base_threshold + adjustment, minimum=self.runtime_config.minimum_dynamic_threshold, maximum=self.runtime_config.maximum_dynamic_threshold)

    def _estimate_payload_risk(self, payload: Mapping[str, Any]) -> float:
        score = 0.25
        text = stable_json_dumps(payload).lower()
        high_risk_terms = ("delete", "credential", "secret", "payment", "irreversible", "shutdown", "external")
        score += min(0.5, 0.08 * sum(1 for term in high_risk_terms if term in text))
        if payload.get("priority") is not None:
            score += min(0.1, max(0.0, coerce_float(payload.get("priority"), default=0.0) / 100.0))
        return clamp_score(score)

    def _record_assessment(self, assessment: SafetyAssessment) -> None:
        payload = assessment.to_dict(redact=self.runtime_config.redact_shared_payloads)
        self._assessment_history.append(payload)
        if self.runtime_config.shared_memory_enabled:
            self.shared_set(self.runtime_config.assessment_key, payload)
            self.shared_set(self.runtime_config.assessment_history_key, list(self._assessment_history))
        self._record_event(
            CollaborativeAgentEventType.RISK_ASSESSED.value,
            f"Risk assessed for task type '{assessment.task_type}'.",
            severity="warning" if assessment.requires_review else "info",
            metadata={"assessment": payload},
        )

    # ------------------------------------------------------------------
    # Coordination and delegation
    # ------------------------------------------------------------------
    def coordinate_tasks(
        self,
        tasks: Sequence[Mapping[str, Any]],
        available_agents: Mapping[str, Mapping[str, Any]],
        optimization_goals: Optional[Iterable[Any]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        started_at = epoch_seconds()
        start_ms = monotonic_ms()
        coordination_id = generate_uuid("coordination", length=24)
        constraints_map = normalize_metadata(constraints, drop_none=True)
        goals = normalize_tags(optimization_goals or ["minimize_risk", "balance_load", "maximize_capability_match"])
        assignments: Dict[str, Dict[str, Any]] = {}
        agent_pool = self._normalize_available_agents(available_agents)
        agent_loads = {name: coerce_int(meta.get("current_load", meta.get("active_tasks", 0)), default=0, minimum=0) for name, meta in agent_pool.items()}
        max_tasks_per_agent = coerce_int(constraints_map.get("max_tasks_per_agent"), default=self.runtime_config.max_tasks_per_agent, minimum=1)

        self._record_event(
            CollaborativeAgentEventType.COORDINATION_STARTED.value,
            "Coordination started.",
            metadata={"coordination_id": coordination_id, "task_count": len(list(tasks or [])), "agent_count": len(agent_pool)},
        )

        try:
            if not tasks or not agent_pool:
                self._update_metric("coordination_failures", 1)
                error = _make_agent_exception(
                    "NoCapableAgentError",
                    "tasks and available_agents are required for coordination.",
                    context={"task_count": len(list(tasks or [])), "agent_count": len(agent_pool)},
                )
                result = self._finalize_coordination(
                    coordination_id=coordination_id,
                    status="error",
                    assignments={},
                    task_count=len(list(tasks or [])),
                    started_at=started_at,
                    duration_ms=elapsed_ms(start_ms),
                    goals=goals,
                    error=error,
                    metadata={"constraints": constraints_map},
                )
                return result

            sorted_tasks = sorted(
                [normalize_task_payload(task, allow_none=False) for task in tasks],
                key=lambda item: (coerce_float(item.get("deadline"), default=float("inf")), -coerce_float(item.get("priority"), default=0.0)),
            )
            for index, task in enumerate(sorted_tasks, start=1):
                task_id = normalize_whitespace(task.get("id") or task.get("task_id") or f"task-{index}")
                assignment = self._coordinate_one_task(task_id, task, agent_pool, agent_loads, max_tasks_per_agent, constraints_map)
                assignments[task_id] = assignment.to_dict()
                assigned_agent = assignment.agent
                if assigned_agent:
                    agent_loads[assigned_agent] = agent_loads.get(assigned_agent, 0) + 1
                    if agent_loads[assigned_agent] >= max_tasks_per_agent:
                        agent_pool[assigned_agent]["_saturated"] = True

            self._update_metric("coordination_runs", 1)
            self._update_metric("tasks_coordinated", len(sorted_tasks))
            self._rolling_metric("avg_coordination_latency_ms", elapsed_ms(start_ms), "coordination_runs")
            return self._finalize_coordination(
                coordination_id=coordination_id,
                status="success",
                assignments=assignments,
                task_count=len(sorted_tasks),
                started_at=started_at,
                duration_ms=elapsed_ms(start_ms),
                goals=goals,
                metadata={"constraints": constraints_map},
            )
        except Exception as exc:
            self._update_metric("coordination_failures", 1)
            result = self._finalize_coordination(
                coordination_id=coordination_id,
                status="error",
                assignments=assignments,
                task_count=len(list(tasks or [])),
                started_at=started_at,
                duration_ms=elapsed_ms(start_ms),
                goals=goals,
                error=exc,
                metadata={"constraints": constraints_map},
            )
            return result

    def _coordinate_one_task(
        self,
        task_id: str,
        task: Dict[str, Any],
        available_agents: Dict[str, Dict[str, Any]],
        agent_loads: Dict[str, int],
        max_tasks_per_agent: int,
        constraints: Mapping[str, Any],
    ) -> CoordinationAssignment:
        task_type = str(task.get("type", task.get("task_type", self.runtime_config.default_task_type)))
        contract_result = self._validate_task_contract(task_type, task)
        if not contract_is_valid(contract_result):
            self._update_metric("contract_validation_failures", 1)
            return CoordinationAssignment(
                task_id=task_id,
                task_type=task_type,
                status="rejected_invalid_contract",
                contract=contract_validation_to_dict(contract_result),
            )

        if self.runtime_config.delegate_during_coordination:
            delegated = self._try_manager_delegation(task_type, task)
            if delegated is not None:
                return CoordinationAssignment(
                    task_id=task_id,
                    task_type=task_type,
                    status="delegated",
                    result=delegated.get("result"),
                    metadata={"delegation": delegated},
                )

        selected_agent, selected_score = self._select_best_agent(task, available_agents, agent_loads, max_tasks_per_agent=max_tasks_per_agent)
        if selected_agent is None:
            self._update_metric("unassigned_tasks", 1)
            return CoordinationAssignment(task_id=task_id, task_type=task_type, status="unassigned", reason="no_capable_agent")

        agent_meta = available_agents.get(selected_agent, {})
        policy_evaluation = self._evaluate_policy(task, agent_meta, constraints)
        policy_payload = policy_evaluation_to_dict(policy_evaluation)
        decision = _policy_decision_value(policy_evaluation)
        if decision == "deny":
            self._update_metric("policy_denials", 1)
            return CoordinationAssignment(
                task_id=task_id,
                task_type=task_type,
                status="rejected_policy",
                agent=selected_agent,
                policy=policy_payload,
            )

        assessment = self.assess_task(task, source_agent=selected_agent, context=constraints)
        if assessment.risk_level == RiskLevel.CRITICAL and self.runtime_config.deny_critical_risk:
            return CoordinationAssignment(
                task_id=task_id,
                task_type=task_type,
                status="rejected_high_risk",
                agent=selected_agent,
                optimization_score=round(selected_score, 6),
                safety=assessment.to_dict(),
                policy=policy_payload,
            )

        status = "assigned"
        if decision in {"require_review", "review", "manual_review"} or (assessment.requires_review and self.runtime_config.review_high_risk):
            self._update_metric("policy_review_flags", 1)
            status = "assigned_with_review"
        self._update_metric("local_assignments", 1)
        return CoordinationAssignment(
            task_id=task_id,
            task_type=task_type,
            status=status,
            agent=selected_agent,
            optimization_score=round(selected_score, 6),
            safety=assessment.to_dict(),
            policy=policy_payload,
            contract=contract_validation_to_dict(contract_result),
            metadata={"max_tasks_per_agent": max_tasks_per_agent},
        )

    def _try_manager_delegation(self, task_type: str, task: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if self.collaboration_manager is None:
            return None
        started_at = epoch_seconds()
        start_ms = monotonic_ms()
        delegation_id = generate_uuid("delegation", length=24)
        try:
            result = self.collaboration_manager.run_task(task_type, dict(task), retries=self.runtime_config.default_retries)
            normalized = normalize_result(result, action="manager_delegation", default_message="Task delegated through CollaborationManager")
            record = DelegationRecord(
                delegation_id=delegation_id,
                task_type=task_type,
                status="success",
                started_at=started_at,
                finished_at=epoch_seconds(),
                duration_ms=elapsed_ms(start_ms),
                result_fingerprint=stable_hash(normalized, length=16),
            ).to_dict()
            self._delegation_history.append(record)
            self._update_metric("delegated_tasks", 1)
            self._publish_delegation_history()
            self._record_event(CollaborativeAgentEventType.TASK_DELEGATED.value, f"Delegated task '{task_type}'.", metadata=record)
            return {"status": "delegated", "task_type": task_type, "result": result, "record": record}
        except Exception as exc:
            record = DelegationRecord(
                delegation_id=delegation_id,
                task_type=task_type,
                status="error",
                started_at=started_at,
                finished_at=epoch_seconds(),
                duration_ms=elapsed_ms(start_ms),
                error=_error_payload(exc, action="manager_delegation"),
            ).to_dict()
            self._delegation_history.append(record)
            self._update_metric("delegation_failures", 1)
            self._publish_delegation_history()
            self._record_event(CollaborativeAgentEventType.DELEGATION_FAILED.value, f"Delegation failed for task '{task_type}'.", severity="warning", error=exc, metadata=record)
            if self.runtime_config.local_fallback_when_delegation_fails:
                return None
            raise _make_agent_exception(
                "DelegationFailureError",
                f"Delegation failed for task type '{task_type}'.",
                context={"task_type": task_type, "task": task},
                cause=exc,
            ) from exc

    def delegate_task(self, task_type: str, task_data: Mapping[str, Any], *, retries: Optional[int] = None) -> Any:
        if self.collaboration_manager is None:
            raise _make_agent_exception("DelegationFailureError", "CollaborationManager is disabled.", context={"task_type": task_type})
        try:
            return self.collaboration_manager.run_task(
                normalize_task_type(task_type),
                normalize_task_payload(task_data, allow_none=False),
                retries=coerce_int(retries, default=self.runtime_config.default_retries, minimum=1),
            )
        except Exception as exc:
            raise _make_agent_exception(
                "DelegationFailureError",
                f"Delegation failed for task type '{task_type}'.",
                context={"task_type": task_type, "task_data": task_data},
                cause=exc,
            ) from exc

    def _select_best_agent(
        self,
        task: Mapping[str, Any],
        available_agents: Mapping[str, Mapping[str, Any]],
        agent_loads: Mapping[str, int],
        *,
        max_tasks_per_agent: Optional[int] = None,
    ) -> Tuple[Optional[str], float]:
        required = set(normalize_capabilities(task.get("requirements") or task.get("required_capabilities") or task.get("capabilities")))
        best_agent: Optional[str] = None
        best_score = float("-inf")
        for agent_name, meta in available_agents.items():
            if coerce_bool(meta.get("_saturated"), default=False):
                continue
            active_tasks = coerce_int(agent_loads.get(agent_name, meta.get("current_load", meta.get("active_tasks", 0))), default=0, minimum=0)
            if max_tasks_per_agent is not None and active_tasks >= max_tasks_per_agent:
                continue
            capabilities = set(extract_agent_capabilities(meta))
            if required and not required.issubset(capabilities):
                continue
            capability_score = len(required & capabilities) / max(1, len(required)) if required else 1.0
            load_score = 1.0 / (1.0 + float(active_tasks))
            risk_score = self._risk_model.threshold(f"agent:{agent_name}", self.runtime_config.risk_threshold)
            priority_score = coerce_float(task.get("priority"), default=0.0) / 100.0
            score = (
                self.runtime_config.optimization_weight_capability * capability_score
                + self.runtime_config.optimization_weight_load * load_score
                + self.runtime_config.optimization_weight_risk * risk_score
                + self.runtime_config.optimization_weight_priority * priority_score
            )
            if score > best_score:
                best_score = score
                best_agent = agent_name
        return best_agent, best_score

    def _validate_task_contract(self, task_type: str, task: Mapping[str, Any]) -> Any:
        if self.task_contracts is None:
            return None
        return self.task_contracts.validate(task_type, dict(task))

    def _evaluate_policy(self, task: Mapping[str, Any], agent_meta: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> Any:
        if self.policy_engine is None:
            return {"decision": "allow", "reasons": [], "matched_rules": []}
        return self.policy_engine.evaluate(dict(task), dict(agent_meta), dict(context or {}))

    def _normalize_available_agents(self, agents: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, meta in dict(agents or {}).items():
            agent_name = normalize_agent_name(name)
            row = ensure_mapping(meta, field_name=f"available_agents[{agent_name}]", allow_none=True)
            row["capabilities"] = list(extract_agent_capabilities(row))
            normalized[agent_name] = row
        return normalized

    def _finalize_coordination(
        self,
        *,
        coordination_id: str,
        status: str,
        assignments: Dict[str, Dict[str, Any]],
        task_count: int,
        started_at: float,
        duration_ms: float,
        goals: Tuple[str, ...],
        metadata: Optional[Mapping[str, Any]] = None,
        error: Optional[BaseException] = None,
    ) -> Dict[str, Any]:
        assigned_count = sum(1 for item in assignments.values() if str(item.get("status")) in {"assigned", "assigned_with_review"})
        delegated_count = sum(1 for item in assignments.values() if str(item.get("status")) == "delegated")
        rejected_count = sum(1 for item in assignments.values() if str(item.get("status", "")).startswith("rejected"))
        unassigned_count = sum(1 for item in assignments.values() if str(item.get("status")) == "unassigned")
        result = CoordinationResult(
            coordination_id=coordination_id,
            status=status,
            assignments=assignments,
            task_count=task_count,
            assigned_count=assigned_count,
            delegated_count=delegated_count,
            rejected_count=rejected_count,
            unassigned_count=unassigned_count,
            started_at=started_at,
            finished_at=epoch_seconds(),
            duration_ms=duration_ms,
            metrics=self.get_metrics(),
            optimization_goals=goals,
            metadata=normalize_metadata(metadata, drop_none=True),
            error=_error_payload(error) if error else None,
        ).to_dict()
        self._coordination_history.append(result)
        if self.runtime_config.shared_memory_enabled:
            self.shared_set(self.runtime_config.coordination_key, result)
            self.shared_set(self.runtime_config.coordination_history_key, list(self._coordination_history))
        self._record_event(
            CollaborativeAgentEventType.COORDINATION_COMPLETED.value if status == "success" else CollaborativeAgentEventType.COORDINATION_FAILED.value,
            f"Coordination {status}.",
            severity="info" if status == "success" else "error",
            error=error,
            metadata={"coordination_id": coordination_id, "task_count": task_count, "assigned_count": assigned_count, "delegated_count": delegated_count},
        )
        self._publish_status()
        return result

    # ------------------------------------------------------------------
    # BaseAgent execution interface
    # ------------------------------------------------------------------
    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        if not isinstance(task_data, Mapping):
            raise _make_agent_exception(
                "RoutingFailureError",
                "CollaborativeAgent task_data must be a mapping.",
                context={"received_type": type(task_data).__name__},
            )
        payload = normalize_task_payload(task_data, allow_none=False)
        mode = normalize_whitespace(payload.get("mode") or payload.get("operation") or CollaborativeAgentMode.COORDINATE.value).lower()
        try:
            if mode in {CollaborativeAgentMode.ASSESS.value, "risk", "risk_assessment"}:
                assessment = self.assess_risk(
                    risk_score=payload.get("risk_score", payload.get("estimated_risk", 0.5)),
                    task_type=payload.get("task_type", payload.get("type", self.runtime_config.default_task_type)),
                    source_agent=payload.get("source_agent", self.runtime_config.default_source_agent),
                    context=payload.get("context"),
                )
                return success_result(action="assess_risk", message="Risk assessment completed", data={"assessment": assessment.to_dict()})
            if mode in {CollaborativeAgentMode.ASSESS_TASK.value, "task_assessment"}:
                assessment = self.assess_task(payload.get("task", payload), source_agent=str(payload.get("source_agent", self.runtime_config.default_source_agent)), context=payload.get("context"))
                return success_result(action="assess_task", message="Task assessment completed", data={"assessment": assessment.to_dict()})
            if mode in {CollaborativeAgentMode.DELEGATE.value, "route", "run_task"}:
                task_type = str(payload.get("task_type", payload.get("type", self.runtime_config.default_task_type)))
                task_payload = payload.get("task_data", payload.get("payload", payload))
                result = self.delegate_task(task_type, ensure_mapping(task_payload, field_name="task_data", allow_none=True), retries=payload.get("retries"))
                return normalize_result(result, action="delegate_task", default_message="Delegation completed")
            if mode in {CollaborativeAgentMode.REGISTER_AGENT.value, "register"}:
                self.register_agent(payload["agent_name"], payload["agent_instance"], payload.get("capabilities", []))
                return success_result(action="register_agent", message="Agent registered", data={"agent_name": payload["agent_name"]})
            if mode == CollaborativeAgentMode.EXPLAIN.value:
                report = self.explain_task(str(payload.get("task_type", payload.get("type", self.runtime_config.default_task_type))), ensure_mapping(payload.get("task_data", payload.get("payload", {})), field_name="task_data", allow_none=True))
                return success_result(action="explain", message="Task explanation generated", data=report)
            if mode == CollaborativeAgentMode.SNAPSHOT.value:
                return success_result(action="snapshot", message="Snapshot generated", data=self.snapshot())
            if mode == CollaborativeAgentMode.HEALTH.value:
                return success_result(action="health", message="Health report generated", data=self.health_report())
            if mode == CollaborativeAgentMode.METRICS.value:
                return success_result(action="metrics", message="Metrics returned", data={"metrics": self.get_metrics()})
            if mode == CollaborativeAgentMode.SHARED_GET.value:
                value = self.shared_get(str(payload["key"]), default=payload.get("default"))
                return success_result(action="shared_get", message="Shared value read", data={"key": payload["key"], "value": json_safe(value)})
            if mode == CollaborativeAgentMode.SHARED_SET.value:
                ok = self.shared_set(str(payload["key"]), payload.get("value"), ttl=payload.get("ttl"))
                return success_result(action="shared_set", message="Shared value written", data={"key": payload["key"], "ok": ok})
            if mode == CollaborativeAgentMode.SHARED_UPDATE.value:
                updated = self.shared_update(str(payload["key"]), ensure_mapping(payload.get("updates"), field_name="updates", allow_none=True), ttl=payload.get("ttl"))
                return success_result(action="shared_update", message="Shared value updated", data={"key": payload["key"], "value": updated})

            return self.coordinate_tasks(
                tasks=ensure_sequence(payload.get("tasks", []), field_name="tasks", allow_none=True),
                available_agents=ensure_mapping(payload.get("available_agents"), field_name="available_agents", allow_none=True),
                optimization_goals=payload.get("optimization_goals"),
                constraints=payload.get("constraints"),
            )
        except Exception as exc:
            return error_result(action=f"collaborative_agent.{mode}", message="Collaborative agent task failed", error=exc)

    def _perform_task(self, task_input: Mapping[str, Any]) -> Dict[str, Any]:
        """Backward-compatible alias for older callers."""

        return self.perform_task(task_input)

    # ------------------------------------------------------------------
    # Manager facade and operational APIs
    # ------------------------------------------------------------------
    def register_agent(self, agent_name: str, agent_instance: Any, capabilities: Iterable[Any]) -> None:
        if self.collaboration_manager is None:
            raise _make_agent_exception("RegistrationFailureError", "CollaborationManager is disabled.", context={"agent_name": agent_name})
        self.collaboration_manager.register_agent(agent_name, agent_instance, capabilities)
        self._record_event(CollaborativeAgentEventType.AGENT_REGISTERED.value, f"Registered agent '{agent_name}'.", metadata={"capabilities": list(normalize_capabilities(capabilities))})
        self._publish_status()

    def explain_task(self, task_type: str, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        if self.collaboration_manager is not None and hasattr(self.collaboration_manager, "explain_task"):
            return normalize_metadata(getattr(self.collaboration_manager, "explain_task")(task_type, dict(task_data)), drop_none=True)
        return {
            "task_type": normalize_task_type(task_type),
            "task_data": normalize_task_payload(task_data, allow_none=True, redact=True),
            "local_policy": policy_evaluation_to_dict(self._evaluate_policy(task_data, {}, {})),
            "risk_estimate": self.assess_task(task_data, context={"explain_only": True}).to_dict(),
        }

    def list_agents(self) -> Dict[str, Any]:
        if self.collaboration_manager is None:
            return {}
        return json_safe(self.collaboration_manager.list_agents())  # type: ignore[return-value]

    def get_agent_stats(self) -> Dict[str, Any]:
        if self.collaboration_manager is not None and hasattr(self.collaboration_manager, "get_agent_stats"):
            return json_safe(self.collaboration_manager.get_agent_stats())  # type: ignore[return-value]
        return get_agent_stats(self.shared_memory)

    def get_reliability_status(self) -> Dict[str, Any]:
        if self.collaboration_manager is not None and hasattr(self.collaboration_manager, "get_reliability_status"):
            return json_safe(self.collaboration_manager.get_reliability_status())  # type: ignore[return-value]
        return {}

    def snapshot(self) -> Dict[str, Any]:
        manager_snapshot = None
        if self.collaboration_manager is not None and hasattr(self.collaboration_manager, "snapshot"):
            manager_snapshot = self.collaboration_manager.snapshot()
        payload = {
            "component": "collaborative_agent",
            "name": self.name,
            "agent_id": getattr(self, "agent_id", None),
            "enabled": self.enabled,
            "version": __version__,
            "captured_at": epoch_seconds(),
            "captured_at_utc": utc_timestamp(),
            "config": self.runtime_config.to_dict(),
            "metrics": self.get_metrics(),
            "risk_model": self._risk_model.snapshot(),
            "assessment_history_size": len(self._assessment_history),
            "coordination_history_size": len(self._coordination_history),
            "delegation_history_size": len(self._delegation_history),
            "last_assessment": self._assessment_history[-1] if self._assessment_history else None,
            "last_coordination": self._coordination_history[-1] if self._coordination_history else None,
            "last_delegation": self._delegation_history[-1] if self._delegation_history else None,
            "manager_snapshot": manager_snapshot,
        }
        return redact_mapping(prune_none(payload, drop_empty=True))

    def health_report(self) -> Dict[str, Any]:
        manager_health = None
        if self.collaboration_manager is not None and hasattr(self.collaboration_manager, "health_report"):
            manager_health = self.collaboration_manager.health_report()
        metrics = self.get_metrics()
        status = "healthy"
        if not self.enabled:
            status = "disabled"
        elif manager_health and str(manager_health.get("status", "healthy")).lower() not in {"healthy", "ok"}:
            status = "degraded"
        elif metrics.get("coordination_failures", 0) or metrics.get("delegation_failures", 0):
            status = "degraded"
        report = {
            "status": status,
            "component": "collaborative_agent",
            "captured_at": epoch_seconds(),
            "captured_at_utc": utc_timestamp(),
            "summary": {
                "assessments_completed": metrics.get("assessments_completed", 0),
                "tasks_coordinated": metrics.get("tasks_coordinated", 0),
                "delegated_tasks": metrics.get("delegated_tasks", 0),
                "coordination_failures": metrics.get("coordination_failures", 0),
                "delegation_failures": metrics.get("delegation_failures", 0),
            },
            "manager_health": manager_health,
        }
        if self.runtime_config.shared_memory_enabled:
            self.shared_set(self.runtime_config.health_key, report)
        return redact_mapping(report)

    def get_assessment_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._assessment_history)
        return items[-max(0, int(limit)):] if limit is not None else items

    def get_coordination_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._coordination_history)
        return items[-max(0, int(limit)):] if limit is not None else items

    def get_delegation_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = list(self._delegation_history)
        return items[-max(0, int(limit)):] if limit is not None else items

    # ------------------------------------------------------------------
    # Serialization and persistence
    # ------------------------------------------------------------------
    def serialize_state(self, *, indent: Optional[int] = None) -> str:
        payload = {
            "name": self.name,
            "version": __version__,
            "config": self.collaborative_config,
            "runtime_config": self.runtime_config.to_dict(),
            "metrics": self.get_metrics(),
            "risk_model": self._risk_model.snapshot(),
            "assessment_history": list(self._assessment_history),
            "coordination_history": list(self._coordination_history),
            "delegation_history": list(self._delegation_history),
            "task_contracts": self.task_contracts.list_contracts() if self.task_contracts is not None else {},
            "policy_rules": self.policy_engine.list_rules() if self.policy_engine is not None else [],
            "timestamp": epoch_seconds(),
            "timestamp_utc": utc_timestamp(),
        }
        return stable_json_dumps(payload, indent=indent)

    @classmethod
    def deserialize_state(cls, raw: str, shared_memory=None, agent_factory=None) -> "CollaborativeAgent":
        payload = json.loads(raw)
        agent = cls(shared_memory=shared_memory, agent_factory=agent_factory, config=payload.get("config", {}))
        agent._metrics.update({str(k): float(v) for k, v in dict(payload.get("metrics", {})).items() if isinstance(v, (int, float))})
        risk_model = payload.get("risk_model", {})
        if isinstance(risk_model, Mapping):
            agent._risk_model.restore(risk_model)
        for item in ensure_list(payload.get("assessment_history")):
            if isinstance(item, Mapping):
                agent._assessment_history.append(dict(item))
        for item in ensure_list(payload.get("coordination_history")):
            if isinstance(item, Mapping):
                agent._coordination_history.append(dict(item))
        for item in ensure_list(payload.get("delegation_history")):
            if isinstance(item, Mapping):
                agent._delegation_history.append(dict(item))
        agent._record_event(CollaborativeAgentEventType.STATE_LOADED.value, "Collaborative agent state restored from serialized payload.")
        return agent

    def save_state(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
        tmp_path.write_text(self.serialize_state(indent=2), encoding="utf-8")
        tmp_path.replace(path_obj)
        self.shared_set(self.runtime_config.state_key, {"path": str(path_obj), "saved_at": epoch_seconds()})
        self._record_event(CollaborativeAgentEventType.STATE_SAVED.value, f"State saved to '{path_obj}'.", metadata={"path": str(path_obj)})

    @classmethod
    def load_state(cls, path: str, shared_memory=None, agent_factory=None) -> "CollaborativeAgent":
        return cls.deserialize_state(Path(path).read_text(encoding="utf-8"), shared_memory=shared_memory, agent_factory=agent_factory)

    # ------------------------------------------------------------------
    # Metrics, audit, and status
    # ------------------------------------------------------------------
    def _update_metric(self, key: str, delta: float) -> None:
        with self._lock:
            self._metrics[key] = float(self._metrics.get(key, 0.0) + delta)
            self._metric_events.append({"key": key, "delta": delta, "value": self._metrics[key], "timestamp": epoch_seconds()})

    def _rolling_metric(self, key: str, latest: float, count_key: str) -> None:
        with self._lock:
            count = max(1.0, float(self._metrics.get(count_key, 1.0)))
            old = float(self._metrics.get(key, 0.0))
            self._metrics[key] = old + ((float(latest) - old) / count)

    def get_metrics(self) -> Dict[str, float]:
        with self._lock:
            return {key: round(float(value), 6) for key, value in self._metrics.items()}

    def reset_metrics(self) -> None:
        with self._lock:
            for key in self._metrics:
                self._metrics[key] = 0.0
            self._metric_events.clear()
        self._publish_status()

    def _record_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        error: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.runtime_config.audit_enabled or not self.runtime_config.shared_memory_enabled:
            return None
        event = build_audit_event(
            event_type,
            message,
            severity=severity,
            component="collaborative_agent",
            state={"metrics": self.get_metrics()},
            error=error,
            metadata=metadata,
        )
        append_audit_event(
            self.shared_memory,
            event,
            key=self.runtime_config.audit_key,
            max_events=self.runtime_config.audit_max_events,
        )
        return event

    def _publish_status(self) -> None:
        if not self.runtime_config.publish_status or not self.runtime_config.shared_memory_enabled:
            return
        status = {
            "component": "collaborative_agent",
            "name": self.name,
            "agent_id": getattr(self, "agent_id", None),
            "enabled": self.enabled,
            "manager_enabled": self.collaboration_manager is not None,
            "metrics": self.get_metrics(),
            "updated_at": epoch_seconds(),
            "updated_at_utc": utc_timestamp(),
        }
        self.shared_set(self.runtime_config.status_key, status)
        self.shared_set(self.runtime_config.metrics_key, {"metrics": self.get_metrics(), "events": list(self._metric_events)})

    def _publish_delegation_history(self) -> None:
        if self.runtime_config.shared_memory_enabled:
            self.shared_set(self.runtime_config.delegation_history_key, list(self._delegation_history))

    def shutdown(self) -> None:
        if self.collaboration_manager is not None and hasattr(self.collaboration_manager, "shutdown"):
            self.collaboration_manager.shutdown()
        self._publish_status()


if __name__ == "__main__":
    print("\n=== Running  Collaborative agent ===\n")
    printer.status("TEST", " Collaborative agent initialized", "info")
    from .collaborative.shared_memory import SharedMemory
    from.agent_factory import AgentFactory

    memory = SharedMemory()
    agent = CollaborativeAgent(
        shared_memory=memory,
        agent_factory=AgentFactory(),
        config={
            "use_collaboration_manager": False,
            "risk_threshold": 0.6,
            "policy_review_risk_threshold": 0.8,
            "policy_deny_risk_threshold": 0.97,
            "shared_memory_enabled": True,
        },
    )

    assessment = agent.assess_risk(0.9, task_type="analysis", source_agent="AgentA", context={"request_id": "r1"})
    assert assessment.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert memory.get(agent.runtime_config.assessment_key) is not None

    task_assessment = agent.assess_task({"id": "a1", "type": "analysis", "estimated_risk": 0.2}, source_agent="AgentA")
    assert task_assessment.risk_level in {RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL}

    available_agents = {
        "AgentA": {"capabilities": ["analysis", "nlp"], "current_load": 0},
        "AgentB": {"capabilities": ["vision"], "current_load": 0},
    }
    tasks = [
        {"id": "t1", "type": "analysis", "requirements": ["analysis"], "priority": 2, "estimated_risk": 0.2},
        {"id": "t2", "type": "analysis", "requirements": ["analysis"], "priority": 1, "estimated_risk": 0.95},
    ]
    result = agent.coordinate_tasks(tasks, available_agents)
    assert result["status"] == "success"
    assert "t1" in result["assignments"] and "t2" in result["assignments"]
    assert result["assignments"]["t1"]["agent"] == "AgentA"

    invalid_result = agent.coordinate_tasks([{"type": "general", "estimated_risk": 0.1}], available_agents)
    assert invalid_result["assignments"]["task-1"]["status"] == "rejected_invalid_contract"

    denied_result = agent.coordinate_tasks([{"id": "t3", "type": "general", "risk_score": 0.99}], available_agents)
    assert denied_result["assignments"]["t3"]["status"] == "rejected_policy"

    task_result = agent.perform_task({
        "mode": "assess",
        "risk_score": 0.1,
        "task_type": "general",
        "source_agent": "AgentA",
    })
    assert task_result["status"] == "success"

    coordination_result = agent.perform_task({
        "mode": "coordinate",
        "tasks": [{"id": "t4", "type": "analysis", "requirements": ["analysis"], "estimated_risk": 0.1}],
        "available_agents": available_agents,
    })
    assert coordination_result["status"] == "success"

    shared_result = agent.perform_task({"mode": "shared_update", "key": "collaborative:test", "updates": {"ok": True}})
    assert shared_result["status"] == "success"
    shared_value = memory.get("collaborative:test")
    assert isinstance(shared_value, dict) and shared_value.get("ok") is True

    snapshot = agent.snapshot()
    assert snapshot["component"] == "collaborative_agent"
    health = agent.health_report()
    assert health["status"] in {"healthy", "degraded", "disabled"}

    state = agent.serialize_state()
    restored = CollaborativeAgent.deserialize_state(state, shared_memory=memory, agent_factory=AgentFactory())
    assert restored.get_metrics()["assessments_completed"] >= 1

    printer.status("TEST", " Collaborative agent checks passed", "success")
    print("\n=== Test ran successfully ===\n")
