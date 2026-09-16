"""Production Safety Agent orchestration for SLAI.

The agent owns orchestration policy only. Agent policy is loaded exclusively
from agents_config.yaml. Safety subsystem modules retain ownership of their own
private configuration, models, and memory implementation.
"""

from __future__ import annotations

__version__ = "2.3.0"

import re
import time

from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .base_agent import BaseAgent
from .base.utils.base_errors import BaseRuntimeError, BaseValidationError
from .base.utils.base_helpers import *
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .safety.adaptive_security import AdaptiveSecurity
from .safety.attention_monitor import AttentionMonitor
from .safety.compliance_checker import ComplianceChecker
from .safety.cyber_safety import CyberSafetyModule
from .safety.reward_model import RewardModel
from .safety.safety_guard import SafetyGuard
from .safety.secure_hacker import SecureHacker
from .safety.secure_stpa import SecureSTPA
from .safety.utils.security_error import SecurityError
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Safety Agent")
printer = PrettyPrinter()

MODULE_VERSION = __version__
ASSESSMENT_SCHEMA_VERSION = "safety_agent.assessment.v4"
ACTION_VALIDATION_SCHEMA_VERSION = "safety_agent.action_validation.v4"
AUDIT_SCHEMA_VERSION = "safety_agent.audit.v3"
POSTURE_SCHEMA_VERSION = "safety_agent.posture.v3"
CORRELATION_SCHEMA_VERSION = "safety_agent.correlation.v1"


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if numeric != numeric:
        numeric = default
    return max(0.0, min(1.0, numeric))


def _nested(mapping: Mapping[str, Any], path: Union[str, Sequence[str]], default: Any = None) -> Any:
    keys = path.split(".") if isinstance(path, str) else list(path)
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _dedupe(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = json_dumps(to_json_safe(value), sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _weighted_average(values: Mapping[str, Any], weights: Mapping[str, Any], *, default: float = 0.0) -> float:
    numerator = 0.0
    denominator = 0.0
    for key, raw_value in values.items():
        weight = max(0.0, coerce_float(weights.get(key, 0.0), 0.0))
        if weight <= 0.0:
            continue
        numerator += _clamp_score(raw_value) * weight
        denominator += weight
    return _clamp_score(numerator / denominator) if denominator > 0.0 else _clamp_score(default)


def _risk_level(score: Any) -> str:
    value = _clamp_score(score)
    if value >= 0.90:
        return "critical"
    if value >= 0.75:
        return "high"
    if value >= 0.50:
        return "medium"
    if value >= 0.25:
        return "low"
    return "minimal"


def _threshold_decision(score: Any, *, review: Any, block: Any) -> str:
    value = _clamp_score(score)
    if value >= _clamp_score(block, 0.75):
        return "block"
    if value >= _clamp_score(review, 0.45):
        return "review"
    return "allow"


def _noisy_or(scores: Iterable[Any]) -> float:
    survival = 1.0
    observed = False
    for score in scores:
        observed = True
        survival *= 1.0 - _clamp_score(score)
    return _clamp_score(1.0 - survival) if observed else 0.0


@dataclass(frozen=True)
class ComponentOutcome:
    component: str
    success: bool
    report: Dict[str, Any]
    risk_score: float
    decision: str
    blocked: bool
    degraded: bool
    duration_ms: int
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "success": bool(self.success),
            "risk_score": _clamp_score(self.risk_score),
            "decision": self.decision,
            "blocked": bool(self.blocked),
            "degraded": bool(self.degraded),
            "duration_ms": max(0, int(self.duration_ms)),
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class SafetyAuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    assessment_id: Optional[str]
    decision: str
    risk_score: float
    risk_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "assessment_id": self.assessment_id,
            "decision": self.decision,
            "risk_score": _clamp_score(self.risk_score),
            "risk_level": self.risk_level,
            "component": "safety_agent",
            "metadata": to_json_safe(self.metadata),
        }


@dataclass(frozen=True)
class SafetyAssessment:
    assessment_id: str
    timestamp: str
    input_type: str
    input_fingerprint: str
    context_type: str
    sanitized_text: str
    reports: Dict[str, Any]
    component_risks: Dict[str, float]
    family_risks: Dict[str, float]
    component_weights: Dict[str, float]
    blockers: List[str]
    warnings: List[str]
    degraded_components: List[str]
    constitutional_violations: List[Dict[str, Any]]
    final_safety_score: float
    risk_score: float
    risk_level: str
    decision: str
    overall_recommendation: str
    is_safe: bool
    aggregation_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "assessment_id": self.assessment_id,
            "timestamp": self.timestamp,
            "input_type": self.input_type,
            "input_fingerprint": self.input_fingerprint,
            "context_type": self.context_type,
            "sanitized_text": self.sanitized_text,
            "reports": to_json_safe(self.reports),
            "component_risks": {k: _clamp_score(v) for k, v in self.component_risks.items()},
            "family_risks": {k: _clamp_score(v) for k, v in self.family_risks.items()},
            "component_weights": {k: max(0.0, coerce_float(v, 0.0)) for k, v in self.component_weights.items()},
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "degraded_components": list(self.degraded_components),
            "constitutional_violations": to_json_safe(self.constitutional_violations),
            "final_safety_score": _clamp_score(self.final_safety_score),
            "risk_score": _clamp_score(self.risk_score),
            "risk_level": self.risk_level,
            "decision": self.decision,
            "overall_recommendation": self.overall_recommendation,
            "is_safe": bool(self.is_safe),
            "aggregation_method": self.aggregation_method,
            "metadata": to_json_safe(self.metadata),
        }


@dataclass(frozen=True)
class ActionValidationResult:
    validation_id: str
    timestamp: str
    action_name: str
    action_fingerprint: str
    approved: bool
    decision: str
    risk_score: float
    risk_level: str
    component_risks: Dict[str, float]
    family_risks: Dict[str, float]
    details: List[str]
    corrections: List[Dict[str, Any]]
    reports: Dict[str, Any]
    degraded_components: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ACTION_VALIDATION_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "validation_id": self.validation_id,
            "timestamp": self.timestamp,
            "action_name": self.action_name,
            "action_fingerprint": self.action_fingerprint,
            "approved": bool(self.approved),
            "decision": self.decision,
            "risk_score": _clamp_score(self.risk_score),
            "risk_level": self.risk_level,
            "component_risks": {k: _clamp_score(v) for k, v in self.component_risks.items()},
            "family_risks": {k: _clamp_score(v) for k, v in self.family_risks.items()},
            "details": list(self.details),
            "corrections": to_json_safe(self.corrections),
            "reports": to_json_safe(self.reports),
            "degraded_components": list(self.degraded_components),
            "metadata": to_json_safe(self.metadata),
        }


class SafetyAgent(BaseAgent):
    """Safety orchestration boundary with subsystem-private configuration/memory."""

    DEFAULT_CONTEXT_TYPE = "general"
    TEXT_CONTENT_KEYS: Tuple[str, ...] = ("text", "message", "prompt", "query", "content", "body", "input")
    EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    PAYMENT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    SECRET_ASSIGNMENT_RE = re.compile(
        r"(?i)\b(?:api[_-]?key|apikey|secret|token|password|passwd|pwd|credential|authorization|auth|cookie|session|private[_-]?key|access[_-]?key|refresh[_-]?token|client[_-]?secret)\s*[:=]\s*['\"]?(?P<value>[A-Za-z0-9._~+/=:@-]{6,})"
    )
    IDENTIFIER_CONTEXT_KEYS = {
        "actor_id", "client_ip", "correlation_id", "ip", "principal", "request_id",
        "session_id", "source_ip", "tenant_id", "trace_id", "user_id",
    }
    _COMPONENT_CLASS_MAP: Dict[str, Any] = {
        "safety_guard": SafetyGuard,
        "cyber_safety": CyberSafetyModule,
        "adaptive_security": AdaptiveSecurity,
        "attention_monitor": AttentionMonitor,
        "reward_model": RewardModel,
        "compliance_checker": ComplianceChecker,
        "secure_stpa": SecureSTPA,
    }

    def __init__(self, agent_factory: Any = None, shared_memory: Any = None, config: Optional[Mapping[str, Any]] = None) -> None:
        # Role-specific runtime overrides are intentionally not passed into BaseAgent.
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=None)
        self.name = "Safety_Agent"
        self._state_lock = RLock()

        self.config = load_global_config()
        self.safety_config: Dict[str, Any] = dict(get_config_section("safety_agent") or {})
        if config:
            logger.warning("SafetyAgent runtime config override ignored; use agents_config.yaml:safety_agent.")

        assert_valid_config_contract(
            global_config=self.config,
            agent_key="safety_agent",
            agent_config=self.safety_config,
            logger=logger,
            require_global_keys=False,
            require_agent_section=True,
            warn_unknown_global_keys=False,
        )
        self._validate_configuration()

        self.audit_level = coerce_int(self._cfg("audit_level"), 2, minimum=0, maximum=5)
        self.collect_feedback_enabled = coerce_bool(self._cfg("collect_feedback"), False)
        self.fail_closed_on_component_error = coerce_bool(self._cfg("fail_closed_on_component_error"), True)
        self.store_assessments = coerce_bool(self._cfg("shared_memory.store_assessments"), True)
        self.store_audit_events = coerce_bool(self._cfg("shared_memory.store_audit_events"), True)
        self.assessment_history_limit = coerce_int(self._cfg("assessment_history_limit"), 500, minimum=10)
        self.audit_trail_limit = coerce_int(self._cfg("audit_trail_limit"), 1000, minimum=50)
        self.verbose_console = coerce_bool(self._cfg("verbose_console"), False)
        self.enable_learnable_aggregation = coerce_bool(self._cfg("enable_learnable_aggregation"), False)
        self.risk_thresholds = dict(self._cfg("risk_thresholds") or {})
        self.aggregation_weights = dict(self._cfg("aggregation_weights") or {})
        self.enabled_components = dict(self._cfg("components") or {})
        self.correlation_config = dict(self._cfg("correlation") or {})
        self.adversarial_config = dict(self._cfg("adversarial_validation") or {})
        self.architecture_map = dict(self._cfg("architecture_map") or {})
        self.system_models = dict(self._cfg("system_models") or {})
        self.known_hazards = list(self._cfg("known_hazards") or [])
        self.global_losses = list(self._cfg("global_losses") or [])
        self.safety_policies = list(self._cfg("safety_policies") or [])
        self.formal_specs = dict(self._cfg("formal_specs") or {})
        self.fault_tree_config = dict(self._cfg("fault_tree_config") or {})

        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.component_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "errors": 0, "degraded": 0, "total_duration_ms": 0, "max_duration_ms": 0}
        )
        self.safety_guard: Optional[SafetyGuard] = None
        self.cyber_safety: Optional[CyberSafetyModule] = None
        self.adaptive_security: Optional[AdaptiveSecurity] = None
        self.attention_monitor: Optional[AttentionMonitor] = None
        self.reward_model: Optional[RewardModel] = None
        self.compliance_checker: Optional[ComplianceChecker] = None
        self.secure_stpa: Optional[SecureSTPA] = None
        self.secure_hacker: Optional[SecureHacker] = None
        self._initialize_components()

        self.training_data: List[Dict[str, Any]] = []
        self.risk_table: Dict[str, Any] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        self.calls = 0
        self._utilities: Dict[str, Any] = {}
        self.learning_factory = None
        self.risk_aggregator = None
        self.constitution = {"checks": self._constitutional_checks()}
        self._init_learning_factory()

        self._publish_agent_event(
            "safety_agent.initialized",
            {
                "schema_version": self._cfg("schema_version"),
                "audit_level": self.audit_level,
                "component_status": self.component_status,
                "config_source": "agents_config.yaml",
                "subsystem_memory_accessed_by_agent": False,
            },
            decision="allow",
            risk_score=0.0,
        )

    # ------------------------------------------------------------------
    # Configuration and component lifecycle
    # ------------------------------------------------------------------

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return _nested(self.safety_config, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.safety_config, Mapping):
            raise BaseValidationError("safety_agent configuration must be a mapping.", None, component="SafetyAgent")
        required = (
            "components", "risk_thresholds", "aggregation_weights", "shared_memory",
            "correlation", "orchestration", "performance", "constitutional", "adversarial_validation",
        )
        missing = [key for key in required if not isinstance(self.safety_config.get(key), Mapping)]
        if missing:
            raise BaseValidationError(
                "SafetyAgent configuration is missing required mapping sections.",
                None,
                component="SafetyAgent",
                context={"missing_sections": missing},
            )
        review = _clamp_score(_nested(self.safety_config, "risk_thresholds.review_threshold", 0.45))
        block = _clamp_score(_nested(self.safety_config, "risk_thresholds.block_threshold", 0.75))
        if review > block:
            raise BaseValidationError("review_threshold must be <= block_threshold.", None, component="SafetyAgent")
        if not self.safety_config.get("aggregation_weights"):
            raise BaseValidationError("aggregation_weights must not be empty.", None, component="SafetyAgent")
        if not _nested(self.safety_config, "correlation.family_weights", {}):
            raise BaseValidationError("correlation.family_weights must not be empty.", None, component="SafetyAgent")
        for name, spec in self.safety_config.get("components", {}).items():
            if not isinstance(spec, Mapping):
                raise BaseValidationError("Each component entry must be a mapping.", None, context={"component": name})
            authority = str(spec.get("authority", "corroborating")).lower()
            if authority not in {"enforcing", "corroborating", "advisory"}:
                raise BaseValidationError("Unsupported component authority.", None, context={"component": name, "authority": authority})

    def _component_spec(self, component: str) -> Dict[str, Any]:
        raw = self.enabled_components.get(component, {})
        return dict(raw) if isinstance(raw, Mapping) else {"enabled": coerce_bool(raw, True)}

    def _component_enabled(self, component: str) -> bool:
        return coerce_bool(self._component_spec(component).get("enabled"), True)

    def _component_required(self, component: str) -> bool:
        return coerce_bool(self._component_spec(component).get("required"), False)

    def _component_authority(self, component: str) -> str:
        authority = str(self._component_spec(component).get("authority", "corroborating")).lower()
        return authority if authority in {"enforcing", "corroborating", "advisory"} else "corroborating"

    def _initialize_components(self) -> None:
        for name in (
            "safety_guard", "cyber_safety", "adaptive_security", "attention_monitor",
            "reward_model", "compliance_checker", "secure_stpa",
        ):
            self._initialize_component(name)

        if not self._component_enabled("secure_hacker"):
            self.component_status["secure_hacker"] = {"initialized": False, "enabled": False, "required": False, "degraded": False, "reason": "disabled"}
            return

        dependencies = {
            "safety_guard": self.safety_guard,
            "cyber_safety": self.cyber_safety,
            "adaptive_security": self.adaptive_security,
        }
        missing = [name for name, value in dependencies.items() if value is None]
        if missing:
            self.component_status["secure_hacker"] = {
                "initialized": False,
                "enabled": True,
                "required": self._component_required("secure_hacker"),
                "degraded": True,
                "reason": "dependency_unavailable",
                "missing_dependencies": missing,
            }
            if self._component_required("secure_hacker") and coerce_bool(self._cfg("orchestration.fail_startup_on_required_component_error"), True):
                raise BaseRuntimeError("SecureHacker required dependencies are unavailable.", None, context={"missing": missing})
            return

        try:
            self.secure_hacker = SecureHacker(
                safety_guard=self.safety_guard,
                cyber_safety=self.cyber_safety,
                adaptive_security=self.adaptive_security,
            )
            self.component_status["secure_hacker"] = {
                "initialized": True,
                "enabled": True,
                "required": self._component_required("secure_hacker"),
                "degraded": False,
            }
        except Exception as exc:
            self._record_component_init_failure("secure_hacker", exc)

    def _initialize_component(self, name: str) -> None:
        if not self._component_enabled(name):
            self.component_status[name] = {"initialized": False, "enabled": False, "required": False, "degraded": False, "reason": "disabled"}
            return
        try:
            setattr(self, name, self._COMPONENT_CLASS_MAP[name]())
            self.component_status[name] = {
                "initialized": True,
                "enabled": True,
                "required": self._component_required(name),
                "degraded": False,
            }
        except Exception as exc:
            self._record_component_init_failure(name, exc)

    def _record_component_init_failure(self, name: str, exc: BaseException) -> None:
        required = self._component_required(name)
        self.component_status[name] = {
            "initialized": False,
            "enabled": True,
            "required": required,
            "degraded": True,
            "reason": "initialization_error",
            "error_type": type(exc).__name__,
        }
        if required and coerce_bool(self._cfg("orchestration.fail_startup_on_required_component_error"), True):
            raise BaseRuntimeError.wrap(
                exc,
                message=f"Required SafetyAgent component failed to initialize: {name}",
                component="SafetyAgent",
                operation="initialize_components",
                context={"component_name": name},
            ) from exc
        logger.error("Optional SafetyAgent component failed to initialize: %s (%s)", name, type(exc).__name__)


    def _init_learning_factory(self) -> None:
        """Compatibility hook; aggregation learning must be injected externally."""
        self.learning_factory = None
        self.risk_aggregator = None
        if self.enable_learnable_aggregation:
            logger.warning(
                "SafetyAgent learnable aggregation requested, but model ownership "
                "remains external to the agent. RewardModel calibration is used instead."
            )

    def _load_constitution(self) -> Dict[str, Any]:
        """Return the inline agents_config.yaml constitutional policy."""
        return {"checks": self._constitutional_checks()}

    # ------------------------------------------------------------------
    # Agent-boundary sanitization and SharedMemory publication
    # ------------------------------------------------------------------

    def _normalize_input_text(self, value: Any) -> str:
        max_length = coerce_int(self._cfg("max_input_text_length"), 12000, minimum=1)
        if isinstance(value, str):
            text = value
        elif isinstance(value, Mapping):
            text = ""
            for key in self.TEXT_CONTENT_KEYS:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    text = candidate
                    break
            if not text:
                text = json_dumps(self._safe_payload(value))
        elif isinstance(value, (list, tuple, set, frozenset)):
            text = json_dumps(self._safe_payload(list(value)))
        else:
            text = str(value)
        return truncate_string(normalize_text(text, collapse_whitespace=False), max_length=max_length, suffix="")

    def _fingerprint(self, value: Any) -> str:
        return stable_fingerprint(value, length=24)

    def _safe_text(self, value: Any, *, max_length: Optional[int] = None) -> str:
        text = str(value)
        text = self.SECRET_ASSIGNMENT_RE.sub("[REDACTED:secret_assignment]", text)
        text = self.EMAIL_RE.sub("[REDACTED:email]", text)
        text = self.PAYMENT_CARD_RE.sub("[REDACTED:payment_card]", text)
        text = self.SSN_RE.sub("[REDACTED:ssn]", text)
        # Fallback layer for anything the targeted patterns missed.
        text = redact_text(text)
        limit = max_length or coerce_int(
            self._cfg("max_output_text_length"),
            coerce_int(self._cfg("max_input_text_length"), 12000, minimum=1),
            minimum=1,
        )
        return truncate_string(text, max_length=limit, suffix="")

    def _safe_payload(self, value: Any, *, key_hint: str = "") -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if key_hint.lower() in self.IDENTIFIER_CONTEXT_KEYS and value:
                return f"fp:{self._fingerprint(value)}"
            return self._safe_text(value)
        if isinstance(value, bytes):
            return {"type": "bytes", "length": len(value), "fingerprint": self._fingerprint(value.hex())}
        if is_dataclass(value) and not isinstance(value, type):
            return self._safe_payload(asdict(value), key_hint=key_hint)
        if isinstance(value, Mapping):
            redacted = redact_mapping(value)
            return {str(key): self._safe_payload(item, key_hint=str(key))
                    for key, item in redacted.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            max_items = coerce_int(self._cfg("output.max_collection_items"), 250, minimum=1)
            sequence = list(value)
            result = [self._safe_payload(item, key_hint=key_hint) for item in sequence[:max_items]]
            if len(sequence) > max_items:
                result.append({"truncated_items": len(sequence) - max_items})
            return result
        return self._safe_text(repr(value))

    def _public_context(self, runtime_context: Mapping[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in runtime_context.items():
            key_text = str(key)
            if key_text == "attention_matrix":
                shape = getattr(value, "shape", None)
                result[key_text] = {
                    "present": value is not None,
                    "type": type(value).__name__,
                    "shape": list(shape) if shape is not None else None,
                }
                continue
            if key_text in {"request", "raw_request", "http_request"}:
                result[key_text] = {"present": value is not None, "type": type(value).__name__}
                continue
            result[key_text] = self._safe_payload(value, key_hint=key_text)
        return result

    def _result_to_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_dict") and callable(value.to_dict):
            result = value.to_dict()
        elif hasattr(value, "to_legacy_scores") and callable(value.to_legacy_scores):
            result = value.to_legacy_scores()
        elif isinstance(value, Mapping):
            result = dict(value)
        else:
            result = {"value": to_json_safe(value)}
        return dict(result) if isinstance(result, Mapping) else {"value": result}

    def _shared_enabled(self) -> bool:
        return coerce_bool(self._cfg("shared_memory.enabled"), True)

    def _shared_ttl(self, key: str, default: int) -> int:
        return coerce_int(self._cfg(["shared_memory", key], default), default, minimum=1)

    def _shared_key(self, kind: str, identifier: str) -> str:
        prefix = str(self._cfg("shared_memory.key_prefix", "safety_agent"))
        kind_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(kind)).strip("_")
        ident = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(identifier)).strip("_")
        return f"{prefix}:{kind_id}:{ident}"

    def _shared_put(
        self,
        kind: str,
        identifier: str,
        value: Mapping[str, Any],
        *,
        ttl: Optional[int] = None,
        tags: Optional[Sequence[str]] = None,
        priority: Optional[float] = None,
    ) -> None:
        if not self._shared_enabled() or self.shared_memory is None or not hasattr(self.shared_memory, "put"):
            return
        try:
            self.shared_memory.put(
                self._shared_key(kind, identifier),
                self._safe_payload(dict(value)),
                ttl=ttl or self._shared_ttl("assessment_ttl_seconds", 604800),
                priority=priority,
                tags=_dedupe(list(tags or []) + ["safety_agent", kind]),
                metadata={"component": "safety_agent", "kind": kind},
            )
        except Exception as exc:
            logger.error("SafetyAgent SharedMemory publication failed: %s (%s)", kind, type(exc).__name__)
            if coerce_bool(self._cfg("shared_memory.fail_closed_on_write_error"), False):
                raise BaseRuntimeError.wrap(
                    exc,
                    message="SafetyAgent SharedMemory publication failed.",
                    component="SafetyAgent",
                    operation="shared_put",
                    context={"kind": kind},
                ) from exc

    def _publish_agent_event(
        self,
        event_type: str,
        metadata: Mapping[str, Any],
        *,
        assessment_id: Optional[str] = None,
        decision: str = "observe",
        risk_score: float = 0.0,
    ) -> None:
        event = SafetyAuditEvent(
            event_id=generate_request_id("safety_evt"),
            timestamp=utc_now_iso(),
            event_type=str(event_type),
            assessment_id=assessment_id,
            decision=str(decision),
            risk_score=_clamp_score(risk_score),
            risk_level=_risk_level(risk_score),
            metadata=dict(metadata),
        ).to_dict()
        with self._state_lock:
            self.audit_trail.append(event)
            if len(self.audit_trail) > self.audit_trail_limit:
                del self.audit_trail[:-self.audit_trail_limit]
        if self.store_audit_events:
            self._shared_put(
                "event",
                event["event_id"],
                event,
                ttl=self._shared_ttl("audit_ttl_seconds", 2592000),
                tags=["audit", str(event_type)],
                priority=_clamp_score(risk_score),
            )

    # ------------------------------------------------------------------
    # Component execution
    # ------------------------------------------------------------------

    def _execute_component(
        self,
        component: str,
        operation: Callable[[], Any],
        *,
        fallback_risk: Optional[float] = None,
    ) -> ComponentOutcome:
        started = time.monotonic()
        required = self._component_required(component)
        try:
            report = self._result_to_dict(operation())
            risk = self._extract_risk(report, default=0.0)
            decision = self._extract_decision(report, risk)
            degraded = self._report_is_degraded(report)
            blocked = self._report_blocks(report)
            success = True
            error_type = None
        except SecurityError as exc:
            report = self._security_error_report(exc)
            default_risk = fallback_risk
            if default_risk is None:
                default_risk = (
                    coerce_float(self._cfg("orchestration.required_error_risk"), 1.0)
                    if required
                    else coerce_float(self._cfg("orchestration.optional_error_risk"), 0.55)
                )
            risk = _clamp_score(getattr(exc, "risk_score", None), default=default_risk)
            blocked = bool(getattr(exc, "blocked", False))
            decision = "block" if blocked else "review"
            degraded = True
            success = False
            error_type = type(exc).__name__
        except Exception as exc:
            report = self._generic_error_report(exc, component)
            default_risk = fallback_risk
            if default_risk is None:
                default_risk = (
                    coerce_float(self._cfg("orchestration.required_error_risk"), 1.0)
                    if required
                    else coerce_float(self._cfg("orchestration.optional_error_risk"), 0.55)
                )
            risk = _clamp_score(default_risk)
            blocked = bool(required and self.fail_closed_on_component_error)
            decision = "block" if blocked else "review"
            degraded = True
            success = False
            error_type = type(exc).__name__

        duration_ms = int((time.monotonic() - started) * 1000)
        self._record_component_metrics(component, duration_ms=duration_ms, error=not success, degraded=degraded)
        return ComponentOutcome(
            component=component,
            success=success,
            report=report,
            risk_score=risk,
            decision=decision,
            blocked=blocked,
            degraded=degraded,
            duration_ms=duration_ms,
            error_type=error_type,
        )

    def _record_component_metrics(self, component: str, *, duration_ms: int, error: bool, degraded: bool) -> None:
        with self._state_lock:
            metrics = self.component_metrics[component]
            metrics["calls"] += 1
            metrics["errors"] += int(bool(error))
            metrics["degraded"] += int(bool(degraded))
            metrics["total_duration_ms"] += max(0, duration_ms)
            metrics["max_duration_ms"] = max(metrics["max_duration_ms"], max(0, duration_ms))

    def _security_error_report(self, error: SecurityError) -> Dict[str, Any]:
        public = None
        log_record = None
        audit = None
        method = getattr(error, "to_public_response", None)
        if callable(method):
            try:
                public = method()
            except Exception:
                pass
        method = getattr(error, "to_log_record", None)
        if callable(method):
            try:
                log_record = method()
            except Exception:
                pass
        method = getattr(error, "to_audit_format", None)
        if callable(method):
            try:
                audit = method(include_sensitive=False)
            except Exception:
                pass
        return self._safe_payload(
            {
                "status": "security_error",
                "error_type": type(error).__name__,
                "public": public,
                "log": log_record,
                "audit": audit,
            }
        )

    def _generic_error_report(self, exc: BaseException, operation: str) -> Dict[str, Any]:
        return {
            "status": "component_error",
            "operation": str(operation),
            "error_type": type(exc).__name__,
            "message": "Safety component execution failed.",
        }

    def _slow_component_warning(self, outcome: ComponentOutcome, warnings: List[str]) -> None:
        threshold_ms = coerce_int(self._cfg("performance.slow_component_ms"), 750, minimum=1)
        if outcome.duration_ms >= threshold_ms:
            warnings.append(f"slow_component:{outcome.component}")

    # ------------------------------------------------------------------
    # Main assessment pipeline
    # ------------------------------------------------------------------

    def perform_task(self, task_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        runtime_context: Dict[str, Any] = dict(context or {})
        public_context = self._public_context(runtime_context)
        with self._state_lock:
            self.calls += 1
            call_count = self.calls

        assessment_id = generate_request_id("safety_assess")
        started = time.monotonic()
        input_text = self._normalize_input_text(task_data)
        sensitive_spans = self._extract_secret_spans(input_text)
        input_fingerprint = self._fingerprint(input_text)
        context_type = self._context_type(runtime_context)

        reports: Dict[str, Any] = {}
        outcomes: Dict[str, ComponentOutcome] = {}
        component_risks: Dict[str, float] = {}
        blockers: List[str] = []
        warnings: List[str] = []
        degraded_components: List[str] = []
        sanitized_text = self._safe_text(input_text)

        if self.verbose_console:
            printer.status("SAFETY", f"Assessment {assessment_id}: {context_type}", "info")

        # Guard is deliberately first: downstream content analysis consumes the sanitized form.
        if self.safety_guard is not None:
            depth = str(runtime_context.get("sanitization_depth") or self._cfg("sanitization_depth", "full"))
            safety_guard = self.safety_guard
            guard = self._execute_component(
                "safety_guard",
                lambda: safety_guard.analyze(input_text, context=runtime_context, depth=depth),
            )
            outcomes["safety_guard"] = guard
            reports["safety_guard"] = guard.report
            component_risks["safety_guard"] = guard.risk_score
            if guard.success and isinstance(guard.report.get("sanitized_text"), str):
                sanitized_text = str(guard.report["sanitized_text"])
            self._apply_component_authority(guard, blockers, warnings, degraded_components)
            self._slow_component_warning(guard, warnings)

        if self.cyber_safety is not None:
            cyber_context = str(runtime_context.get("cyber_context") or context_type)
            cyber_safety = self.cyber_safety
            cyber = self._execute_component(
                "cyber_safety",
                lambda: cyber_safety.analyze_input(sanitized_text, context=cyber_context),
            )
            outcomes["cyber_safety"] = cyber
            reports["cyber_safety"] = cyber.report
            component_risks["cyber_safety"] = cyber.risk_score
            self._apply_component_authority(cyber, blockers, warnings, degraded_components)
            self._slow_component_warning(cyber, warnings)

        adaptive = self._run_adaptive_analysis(task_data, runtime_context)
        if adaptive is not None:
            outcomes["adaptive_security"] = adaptive
            reports["adaptive_security"] = adaptive.report
            component_risks["adaptive_security"] = adaptive.risk_score
            self._apply_component_authority(adaptive, blockers, warnings, degraded_components)
            self._slow_component_warning(adaptive, warnings)

        attention: Optional[ComponentOutcome] = None
        if self.attention_monitor is not None and runtime_context.get("attention_matrix") is not None:
            attention_context = {key: value for key, value in runtime_context.items() if key != "attention_matrix"}
            attention_monitor = self.attention_monitor
            attention = self._execute_component(
                "attention_monitor",
                lambda: attention_monitor.analyze_attention(
                    runtime_context["attention_matrix"],
                    context=attention_context,
                ),
            )
            outcomes["attention_monitor"] = attention
            reports["attention_analysis"] = attention.report
            component_risks["attention_monitor"] = attention.risk_score
            self._apply_component_authority(attention, blockers, warnings, degraded_components)
            self._slow_component_warning(attention, warnings)

        # RewardModel receives computed attention evidence, not the raw attention tensor.
        if self.reward_model is not None:
            reward_context = dict(runtime_context)
            reward_context.pop("attention_matrix", None)
            if attention is not None and attention.success:
                reward_context["attention_analysis"] = attention.report
            reward_model = self.reward_model
            reward = self._execute_component(
                "reward_model",
                lambda: reward_model.evaluate_detailed(sanitized_text, context=reward_context),
            )
            reward_risk = self._extract_reward_risk(reward.report)
            reward = ComponentOutcome(
                component=reward.component,
                success=reward.success,
                report=reward.report,
                risk_score=reward_risk,
                decision=reward.decision,
                blocked=reward.blocked,
                degraded=reward.degraded,
                duration_ms=reward.duration_ms,
                error_type=reward.error_type,
            )
            outcomes["reward_model"] = reward
            reports["reward_model"] = reward.report
            component_risks["reward_model"] = reward_risk
            self._apply_component_authority(reward, blockers, warnings, degraded_components)
            self._slow_component_warning(reward, warnings)

        if self.compliance_checker is not None and self._should_evaluate_compliance(runtime_context):
            compliance = self._execute_component("compliance_checker", self.compliance_checker.evaluate_compliance)
            compliance_risk = self._extract_compliance_risk(compliance.report)
            compliance = ComponentOutcome(
                component="compliance_checker",
                success=compliance.success,
                report=compliance.report,
                risk_score=compliance_risk,
                decision=self._compliance_decision(compliance.report),
                blocked=compliance.blocked or self._compliance_is_blocking(compliance.report),
                degraded=compliance.degraded,
                duration_ms=compliance.duration_ms,
                error_type=compliance.error_type,
            )
            outcomes["compliance_checker"] = compliance
            reports["compliance"] = compliance.report
            component_risks["compliance"] = compliance_risk
            self._apply_component_authority(
                compliance,
                blockers,
                warnings,
                degraded_components,
                blocker_name="compliance_blocker",
            )
            self._slow_component_warning(compliance, warnings)

        constitutional_violations = self._check_constitutional_violations(sanitized_text, include_metadata=True)
        constitutional_metadata = [
            violation for violation in constitutional_violations
            if isinstance(violation, Mapping)
        ]
        constitutional_risk = self._constitutional_risk(constitutional_metadata)
        reports["constitutional"] = {
            "status": "violation" if constitutional_violations else "pass",
            "violations": constitutional_violations,
            "risk_score": constitutional_risk,
        }
        component_risks["constitutional"] = constitutional_risk
        if constitutional_violations and coerce_bool(self._cfg("constitutional.block_on_violation"), True):
            blockers.append("constitutional_violation")

        preliminary_risk, preliminary_families, _ = self._correlate_risks(
            component_risks,
            degraded_components=degraded_components,
        )
        hacker = self._run_secure_hacker_if_needed(
            sanitized_text,
            runtime_context,
            preliminary_risk,
            blockers,
        )
        if hacker is not None:
            regression_risk = self._secure_hacker_regression_risk(hacker.report)
            hacker = ComponentOutcome(
                component="secure_hacker",
                success=hacker.success,
                report=hacker.report,
                risk_score=regression_risk,
                decision=hacker.decision,
                blocked=False,
                degraded=hacker.degraded,
                duration_ms=hacker.duration_ms,
                error_type=hacker.error_type,
            )
            outcomes["secure_hacker"] = hacker
            reports["secure_hacker"] = hacker.report
            component_risks["secure_hacker"] = regression_risk
            self._apply_component_authority(hacker, blockers, warnings, degraded_components)
            self._slow_component_warning(hacker, warnings)

        risk_score, family_risks, correlation_report = self._correlate_risks(
            component_risks,
            degraded_components=degraded_components,
        )
        reports["correlation"] = correlation_report

        if blockers:
            risk_score = max(risk_score, _clamp_score(self.risk_thresholds.get("block_threshold"), 0.75))

        elapsed_ms = int((time.monotonic() - started) * 1000)
        max_assessment_ms = coerce_int(self._cfg("performance.max_assessment_ms"), 5000, minimum=1)
        if elapsed_ms > max_assessment_ms:
            warnings.append("assessment_budget_exceeded")
            if coerce_bool(self._cfg("performance.review_on_budget_exceeded"), False):
                risk_score = max(risk_score, _clamp_score(self.risk_thresholds.get("review_threshold"), 0.45))

        decision = self._decision_from_risk(
            risk_score,
            blockers=blockers,
            degraded_components=degraded_components,
        )
        assessment = SafetyAssessment(
            assessment_id=assessment_id,
            timestamp=utc_now_iso(),
            input_type=type(task_data).__name__,
            input_fingerprint=input_fingerprint,
            context_type=context_type,
            sanitized_text=self._safe_text(sanitized_text),
            reports=reports,
            component_risks=component_risks,
            family_risks=family_risks,
            component_weights=self.aggregation_weights,
            blockers=_dedupe(blockers),
            warnings=_dedupe(warnings),
            degraded_components=_dedupe(degraded_components),
            constitutional_violations=constitutional_violations, # type: ignore
            final_safety_score=_clamp_score(1.0 - risk_score),
            risk_score=risk_score,
            risk_level=_risk_level(risk_score),
            decision=decision,
            overall_recommendation=self._recommendation_from_decision(decision, blockers=blockers),
            is_safe=decision == "allow",
            aggregation_method="family_correlated_weighted_risk",
            metadata={
                "duration_ms": elapsed_ms,
                "call_count": call_count,
                "context": public_context,
                "component_execution": {name: outcome.to_dict() for name, outcome in outcomes.items()},
                "preliminary_risk": preliminary_risk,
                "preliminary_family_risks": preliminary_families,
                "subsystem_memory_accessed_by_agent": False,
                "config_source": "agents_config.yaml",
            },
        )
        result = self._safe_payload(assessment.to_dict())
        result = self._scrub_known_secrets(result, sensitive_spans)
        self._store_assessment(result)
        self._publish_agent_event(
            "safety_agent.assessment_completed",
            {
                "input_type": result.get("input_type"),
                "blockers": result.get("blockers"),
                "warnings": result.get("warnings"),
                "duration_ms": elapsed_ms,
                "degraded_components": result.get("degraded_components"),
            },
            assessment_id=assessment_id,
            decision=decision,
            risk_score=risk_score,
        )

        if self.collect_feedback_enabled and runtime_context.get("human_feedback") is not None:
            self.collect_human_feedback(
                text=sanitized_text,
                model_scores=reports.get("reward_model", {}),
                human_rating=runtime_context["human_feedback"],
                context={**public_context, "assessment_id": assessment_id},
            )
        return result

    # ------------------------------------------------------------------
    # Specialized component routing
    # ------------------------------------------------------------------

    def _run_adaptive_analysis(
        self,
        data: Any,
        runtime_context: Mapping[str, Any],
    ) -> Optional[ComponentOutcome]:
        if self.adaptive_security is None:
            return None
        client_ip = runtime_context.get("client_ip") or runtime_context.get("source_ip")
        if isinstance(data, Mapping) and self._is_email_payload(data):
            adaptive_security = self.adaptive_security
            assert adaptive_security is not None
            return self._execute_component(
                "adaptive_security",
                lambda: adaptive_security.analyze_email(
                    dict(data),
                    client_ip=str(client_ip) if client_ip else None,
                    context=runtime_context,
                ),
            )
        if isinstance(data, str) and re.match(r"(?i)^\s*https?://", data):
            url = data.strip()
            adaptive_security = self.adaptive_security
            assert adaptive_security is not None
            return self._execute_component(
                "adaptive_security",
                lambda: adaptive_security.analyze_url(
                    url,
                    client_ip=str(client_ip) if client_ip else None,
                    context=runtime_context,
                ),
            )
        return None

    def _is_email_payload(self, data: Mapping[str, Any]) -> bool:
        keys = {str(key).lower() for key in data}
        return {"subject", "body"}.issubset(keys) or ("from" in keys and ("subject" in keys or "body" in keys))

    def _run_secure_hacker_if_needed(
        self,
        text: str,
        runtime_context: Mapping[str, Any],
        preliminary_risk: float,
        blockers: Sequence[str],
    ) -> Optional[ComponentOutcome]:
        if self.secure_hacker is None or not coerce_bool(self.adversarial_config.get("enabled"), True):
            return None
        explicit = coerce_bool(runtime_context.get("run_adversarial_validation"), False)
        auto_run = coerce_bool(self.adversarial_config.get("auto_run_on_risk"), True)
        trigger = _clamp_score(self.adversarial_config.get("trigger_risk"), 0.55)
        if not explicit and not (auto_run and (preliminary_risk >= trigger or bool(blockers))):
            return None
        techniques = self.adversarial_config.get("techniques", [])
        selected = list(techniques) if isinstance(techniques, Sequence) and not isinstance(techniques, (str, bytes)) else None
        secure_hacker = self.secure_hacker
        assert secure_hacker is not None
        return self._execute_component(
            "secure_hacker",
            lambda: secure_hacker.run_adversarial_validation(
                text,
                context=runtime_context,
                techniques=selected,
            ),
        )

    def _secure_hacker_regression_risk(self, report: Mapping[str, Any]) -> float:
        robustness = report.get("robustness_score")
        if robustness is not None:
            return _clamp_score(1.0 - _clamp_score(robustness))
        metadata = report.get("metadata")
        if isinstance(metadata, Mapping):
            count = coerce_int(metadata.get("case_count"), 0, minimum=0)
            regressions = coerce_int(metadata.get("regression_count"), 0, minimum=0)
            if count > 0:
                return _clamp_score(regressions / count)
        return 0.0

    # ------------------------------------------------------------------
    # Risk extraction, authority, and correlation
    # ------------------------------------------------------------------

    def _extract_risk(self, report: Any, default: float = 0.0) -> float:
        if not isinstance(report, Mapping):
            return _clamp_score(default)
        for key in ("risk_score", "normalized_risk", "anomaly_score", "phishing_score"):
            if key in report:
                return _clamp_score(report.get(key), default)
        assessment = report.get("security_assessment")
        if isinstance(assessment, Mapping):
            return self._extract_risk(assessment, default)
        nested_scores = [self._extract_risk(value, default=0.0) for value in report.values() if isinstance(value, Mapping)]
        return max(nested_scores, default=_clamp_score(default))

    def _extract_reward_risk(self, report: Mapping[str, Any]) -> float:
        if "risk_score" in report:
            return _clamp_score(report.get("risk_score"))
        composite = report.get("composite", report.get("aggregate_score"))
        return _clamp_score(1.0 - _clamp_score(composite)) if composite is not None else self._extract_risk(report, default=0.5)

    def _extract_compliance_risk(self, report: Mapping[str, Any]) -> float:
        mandatory = report.get("mandatory_failures")
        if isinstance(mandatory, Sequence) and not isinstance(mandatory, (str, bytes)) and mandatory:
            return _clamp_score(self._cfg("risk_thresholds.mandatory_compliance_failure_risk"), 0.95)
        status = str(report.get("status", "unknown")).lower()
        status_map = {
            "compliant": 0.05, "pass": 0.05, "passed": 0.05, "ok": 0.05,
            "warning": 0.45, "partial": 0.45, "conditional": 0.45,
            "critical": 0.95, "fail": 0.90, "failed": 0.90, "error": 0.90, "blocked": 0.95,
        }
        if status in status_map:
            return status_map[status]
        score = report.get("overall_score", report.get("score"))
        return _clamp_score(1.0 - _clamp_score(score)) if score is not None else 0.5

    def _extract_decision(self, report: Mapping[str, Any], risk_score: float) -> str:
        raw = report.get("decision", report.get("overall_recommendation"))
        if raw is not None and str(raw).lower() in {"allow", "review", "block"}:
            return str(raw).lower()
        return _threshold_decision(
            risk_score,
            review=self.risk_thresholds.get("review_threshold", 0.45),
            block=self.risk_thresholds.get("block_threshold", 0.75),
        )

    def _report_blocks(self, report: Mapping[str, Any]) -> bool:
        if report.get("is_phishing") is True:
            return True
        return str(report.get("decision", report.get("overall_recommendation", ""))).lower() == "block"

    def _report_is_degraded(self, report: Mapping[str, Any]) -> bool:
        if report.get("degraded_mode") is True:
            return True
        metadata = report.get("metadata")
        return bool(
            isinstance(metadata, Mapping)
            and (metadata.get("degraded_mode") is True or metadata.get("model_evidentiary") is False)
        )

    def _apply_component_authority(
        self,
        outcome: ComponentOutcome,
        blockers: List[str],
        warnings: List[str],
        degraded_components: List[str],
        *,
        blocker_name: Optional[str] = None,
    ) -> None:
        authority = self._component_authority(outcome.component)
        if outcome.degraded:
            degraded_components.append(outcome.component)
            warnings.append(f"component_degraded:{outcome.component}")
        if outcome.blocked or outcome.decision == "block":
            if authority == "enforcing":
                blockers.append(blocker_name or f"{outcome.component}_block")
            else:
                warnings.append(f"{outcome.component}_block_signal_{authority}")
        if not outcome.success:
            if self._component_required(outcome.component) and self.fail_closed_on_component_error:
                blockers.append(f"{outcome.component}_required_component_error")
            else:
                warnings.append(f"{outcome.component}_component_error")

    def _correlate_risks(
        self,
        component_risks: Mapping[str, Any],
        *,
        degraded_components: Sequence[str],
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        if not component_risks:
            risk = _clamp_score(self._cfg("empty_assessment_risk"), 0.5)
            return risk, {}, {
                "schema_version": CORRELATION_SCHEMA_VERSION,
                "family_risks": {},
                "risk_score": risk,
                "reason": "no_component_evidence",
            }

        if not coerce_bool(self.correlation_config.get("enabled"), True):
            risk = _weighted_average(
                component_risks,
                self.aggregation_weights,
                default=_clamp_score(self._cfg("empty_assessment_risk"), 0.5),
            )
            return risk, {}, {"schema_version": CORRELATION_SCHEMA_VERSION, "enabled": False, "risk_score": risk}

        family_map = self.correlation_config.get("component_families", {})
        family_weights = self.correlation_config.get("family_weights", {})
        corroboration_factor = _clamp_score(self.correlation_config.get("corroboration_factor"), 0.20)
        grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for component, raw_score in component_risks.items():
            family = str(family_map.get(component, component) if isinstance(family_map, Mapping) else component)
            grouped[family].append((component, _clamp_score(raw_score)))

        family_risks: Dict[str, float] = {}
        family_evidence: Dict[str, Any] = {}
        for family, signals in grouped.items():
            scores = [score for _, score in signals]
            maximum = max(scores, default=0.0)
            independent_upper = _noisy_or(scores)
            correlated = _clamp_score(maximum + corroboration_factor * max(0.0, independent_upper - maximum))
            family_risks[family] = correlated
            family_evidence[family] = {
                "risk_score": correlated,
                "maximum_signal": maximum,
                "signals": [{"component": name, "risk_score": score} for name, score in signals],
            }

        risk = _weighted_average(
            family_risks,
            family_weights if isinstance(family_weights, Mapping) else {},
            default=max(family_risks.values(), default=0.0),
        )
        degraded_required = [name for name in degraded_components if self._component_required(name)]
        if degraded_required:
            uplift = coerce_float(
                self.correlation_config.get("degraded_uncertainty_uplift"),
                0.08,
                minimum=0.0,
                maximum=1.0,
            )
            max_components = coerce_int(self.correlation_config.get("max_degraded_uplift_components"), 3, minimum=1)
            risk = _clamp_score(risk + uplift * min(len(set(degraded_required)), max_components))

        return risk, family_risks, {
            "schema_version": CORRELATION_SCHEMA_VERSION,
            "enabled": True,
            "risk_score": risk,
            "family_risks": family_risks,
            "families": family_evidence,
            "corroboration_factor": corroboration_factor,
            "degraded_required_components": _dedupe(degraded_required),
            "semantics": "Within-family corroboration is bounded to reduce correlated-signal double counting.",
        }

    def _decision_from_risk(
        self,
        risk_score: float,
        *,
        blockers: Optional[Sequence[str]] = None,
        degraded_components: Optional[Sequence[str]] = None,
    ) -> str:
        if blockers:
            return "block"
        decision = _threshold_decision(
            risk_score,
            review=self.risk_thresholds.get("review_threshold", 0.45),
            block=self.risk_thresholds.get("block_threshold", 0.75),
        )
        if (
            decision == "allow"
            and degraded_components
            and coerce_bool(self._cfg("orchestration.review_on_required_component_degraded"), False)
            and any(self._component_required(name) for name in degraded_components)
        ):
            return "review"
        return decision

    def _recommendation_from_decision(self, decision: str, *, blockers: Optional[Sequence[str]] = None) -> str:
        if decision == "allow":
            return "proceed"
        if decision == "review":
            return "human_review"
        return "block_due_to_enforcing_control" if blockers else "block_or_review"

    # ------------------------------------------------------------------
    # Compliance policy
    # ------------------------------------------------------------------

    def _should_evaluate_compliance(self, runtime_context: Mapping[str, Any]) -> bool:
        if not self._component_enabled("compliance_checker"):
            return False
        explicit = runtime_context.get("run_compliance")
        if explicit is not None:
            return coerce_bool(explicit, False)
        every = coerce_int(self._cfg("compliance.evaluate_every_n_calls"), 0, minimum=0)
        if every > 0:
            with self._state_lock:
                call_count = self.calls
            if call_count % every == 0:
                return True
        contexts = self._cfg("compliance.evaluate_contexts", [])
        if isinstance(contexts, Sequence) and not isinstance(contexts, (str, bytes)):
            if self._context_type(runtime_context) in {str(item).lower() for item in contexts}:
                return True
        return coerce_bool(self._cfg("compliance.evaluate_on_task"), False)

    def _compliance_is_blocking(self, report: Mapping[str, Any]) -> bool:
        if not coerce_bool(self.risk_thresholds.get("compliance_failure_is_blocker"), True):
            return False
        mandatory = report.get("mandatory_failures")
        if isinstance(mandatory, Sequence) and not isinstance(mandatory, (str, bytes)) and mandatory:
            return True
        return str(report.get("status", "")).lower() in {"critical", "fail", "failed", "blocked", "error"}

    def _compliance_decision(self, report: Mapping[str, Any]) -> str:
        if self._compliance_is_blocking(report):
            return "block"
        return _threshold_decision(
            self._extract_compliance_risk(report),
            review=self.risk_thresholds.get("review_threshold", 0.45),
            block=self.risk_thresholds.get("block_threshold", 0.75),
        )

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def validate_action(
        self,
        action_params: Dict[str, Any],
        action_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(action_params, Mapping):
            raise BaseValidationError("Action validation requires a mapping.", None, component="SafetyAgent")

        runtime_context: Dict[str, Any] = dict(action_context or {})
        public_context = self._public_context(runtime_context)
        validation_id = generate_request_id("safety_action")
        action_name = self._safe_identifier(action_params.get("action_name") or action_params.get("name") or "unknown_action")
        action_text = json_dumps(self._safe_payload(dict(action_params)), sort_keys=True)
        action_fingerprint = self._fingerprint(action_params)

        reports: Dict[str, Any] = {}
        component_risks: Dict[str, float] = {}
        blockers: List[str] = []
        warnings: List[str] = []
        details: List[str] = []
        degraded_components: List[str] = []

        if self.safety_guard is not None:
            safety_guard = self.safety_guard
            guard = self._execute_component(
                "safety_guard",
                lambda: safety_guard.analyze(
                    action_text,
                    context={**runtime_context, "type": "action_validation"},
                    depth=str(self._cfg("sanitization_depth", "full")),
                ),
            )
            reports["safety_guard"] = guard.report
            component_risks["safety_guard"] = guard.risk_score
            self._apply_component_authority(guard, blockers, warnings, degraded_components)

        if self.cyber_safety is not None:
            cyber_safety = self.cyber_safety
            cyber = self._execute_component(
                "cyber_safety",
                lambda: cyber_safety.analyze_input(action_text, context="action_validation"),
            )
            reports["cyber_safety"] = cyber.report
            component_risks["cyber_safety"] = cyber.risk_score
            self._apply_component_authority(cyber, blockers, warnings, degraded_components)
            if cyber.risk_score >= _clamp_score(self.risk_thresholds.get("cyber_risk"), 0.70):
                details.append("Cyber risk exceeds the configured action threshold.")

        if self.reward_model is not None:
            reward_model = self.reward_model
            reward = self._execute_component(
                "reward_model",
                lambda: reward_model.evaluate_detailed(
                    action_text,
                    context={**runtime_context, "type": "action_validation"},
                ),
            )
            reward_risk = self._extract_reward_risk(reward.report)
            reward = ComponentOutcome(
                component=reward.component,
                success=reward.success,
                report=reward.report,
                risk_score=reward_risk,
                decision=reward.decision,
                blocked=reward.blocked,
                degraded=reward.degraded,
                duration_ms=reward.duration_ms,
                error_type=reward.error_type,
            )
            reports["reward_model"] = reward.report
            component_risks["reward_model"] = reward_risk
            self._apply_component_authority(reward, blockers, warnings, degraded_components)

        if self.secure_stpa is not None:
            stpa = self._execute_component(
                "secure_stpa",
                lambda: self._validate_action_with_stpa(action_name, runtime_context),
            )
            reports["secure_stpa"] = stpa.report
            component_risks["secure_stpa"] = stpa.risk_score
            self._apply_component_authority(stpa, blockers, warnings, degraded_components)
            if stpa.report.get("matched_uca_risks"):
                details.append("Action matches one or more unsafe control-action contexts.")

        constitutional = self._check_constitutional_violations(action_text, include_metadata=True)
        constitutional_records = [
            violation for violation in constitutional if isinstance(violation, Mapping)
        ]
        constitutional_risk = self._constitutional_risk(constitutional_records)
        reports["constitutional"] = {"violations": constitutional, "risk_score": constitutional_risk}
        component_risks["constitutional"] = constitutional_risk
        if constitutional and coerce_bool(self._cfg("constitutional.block_on_violation"), True):
            blockers.append("constitutional_violation")
            details.append("Action description triggers an enforcing constitutional rule.")

        preliminary_risk, _, _ = self._correlate_risks(component_risks, degraded_components=degraded_components)
        hacker = self._run_secure_hacker_if_needed(
            action_text,
            {**runtime_context, "type": "action_validation"},
            preliminary_risk,
            blockers,
        )
        if hacker is not None:
            reports["secure_hacker"] = hacker.report
            component_risks["secure_hacker"] = self._secure_hacker_regression_risk(hacker.report)
            if component_risks["secure_hacker"] > 0.0:
                details.append("Bounded adversarial validation found detector-regression risk.")
                warnings.append("adversarial_regression_detected")

        risk_score, family_risks, correlation = self._correlate_risks(
            component_risks,
            degraded_components=degraded_components,
        )
        reports["correlation"] = correlation
        if blockers:
            risk_score = max(
                risk_score,
                _clamp_score(self._cfg("action_validation.block_above", self.risk_thresholds.get("block_threshold", 0.75))),
            )
        decision = self._decision_from_risk(
            risk_score,
            blockers=blockers,
            degraded_components=degraded_components,
        )
        if decision == "allow" and risk_score >= _clamp_score(self._cfg("action_validation.require_human_review_above", 0.45)):
            decision = "review"
        approved = decision == "allow"
        corrections = [] if approved else self.apply_corrections(
            dict(action_params),
            {"risk_score": risk_score, "details": details, "reports": reports, "decision": decision},
        )

        result = self._safe_payload(
            ActionValidationResult(
                validation_id=validation_id,
                timestamp=utc_now_iso(),
                action_name=action_name,
                action_fingerprint=action_fingerprint,
                approved=approved,
                decision=decision,
                risk_score=risk_score,
                risk_level=_risk_level(risk_score),
                component_risks=component_risks,
                family_risks=family_risks,
                details=_dedupe(details),
                corrections=corrections,
                reports=reports,
                degraded_components=_dedupe(degraded_components),
                metadata={
                    "context": public_context,
                    "blockers": _dedupe(blockers),
                    "warnings": _dedupe(warnings),
                    "subsystem_memory_accessed_by_agent": False,
                },
            ).to_dict()
        )
        self._shared_put(
            "action_validation",
            validation_id,
            result,
            ttl=self._shared_ttl("validation_ttl_seconds", 604800),
            tags=["action_validation", decision, str(result.get("risk_level"))],
            priority=risk_score,
        )
        self._publish_agent_event(
            "safety_agent.action_validation_completed",
            {"action_name": action_name, "approved": approved, "blockers": blockers, "details": details},
            assessment_id=validation_id,
            decision=decision,
            risk_score=risk_score,
        )
        return result

    def _validate_action_with_stpa(self, action_name: str, action_context: Mapping[str, Any]) -> Dict[str, Any]:
        if self.secure_stpa is None:
            return {"status": "disabled", "risk_score": 0.0}
        # Action analyses are intentionally isolated; no prior STPA state is reused.
        self.secure_stpa.reset_analysis()
        self.secure_stpa.define_analysis_scope(
            losses=self.global_losses or ["Loss of safe, secure, or compliant operation"],
            hazards=self.known_hazards or ["Unsafe or unauthorized system action"],
            constraints=self.safety_policies or ["Actions must satisfy configured safety policy before execution"],
            system_boundary=str(self._cfg("system_boundary", "Safety Agent orchestration boundary")),
        )
        structure = self.architecture_map or {
            "Safety_Agent": {
                "inputs": ["task_request", "shared_memory_context"],
                "outputs": [action_name],
                "process_vars": ["risk_score", "decision"],
            }
        }
        self.secure_stpa.model_control_structure(structure=structure, process_models=self.system_models)
        ucas = self.secure_stpa.identify_unsafe_control_actions()
        context_tables = self.secure_stpa.build_context_tables(
            formal_spec=self.formal_specs,
            fta_config=self.fault_tree_config,
        )
        threshold = _clamp_score(self._cfg("action_validation.stpa_match_threshold", 0.70))
        matched: List[Dict[str, Any]] = []
        for entries in context_tables.values():
            if not isinstance(entries, Sequence):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                if self._safe_identifier(entry.get("control_action", "unknown")) != action_name:
                    continue
                contextual_match = self._assess_contextual_match(entry, action_context)
                stpa_risk = _clamp_score(entry.get("risk_score"), 0.0)
                combined = max(
                    contextual_match * stpa_risk,
                    stpa_risk if contextual_match >= threshold else 0.0,
                )
                if contextual_match >= threshold or combined >= threshold:
                    matched.append(
                        {
                            "entry": self._safe_payload(entry),
                            "context_match": contextual_match,
                            "stpa_risk": stpa_risk,
                            "combined_risk": _clamp_score(combined),
                        }
                    )
        risk = max((_clamp_score(item["combined_risk"]) for item in matched), default=0.0)
        return {
            "status": "evaluated",
            "uca_count": len(ucas),
            "matched_uca_risks": matched,
            "risk_score": risk,
            "decision": _threshold_decision(
                risk,
                review=self.risk_thresholds.get("review_threshold", 0.45),
                block=self.risk_thresholds.get("block_threshold", 0.75),
            ),
            "analysis_state_reused": False,
        }

    def _assess_contextual_match(self, entry: Mapping[str, Any], execution_state: Mapping[str, Any]) -> float:
        if not execution_state:
            return 0.0
        state_text = normalize_text(json_dumps(self._safe_payload(execution_state)), lowercase=True)
        terms: List[str] = []
        for key in ("process_variables", "state_constraints"):
            values = entry.get(key, [])
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                terms.extend(str(item).lower() for item in values)
        terms.extend(str(entry.get("guideword", "")).lower().split())
        terms = [term for term in _dedupe(terms) if term and len(term) > 2]
        if not terms:
            return 0.0
        return _clamp_score(sum(1 for term in terms if term in state_text) / len(terms))

    # ------------------------------------------------------------------
    # Constitutional policy: agents_config.yaml only
    # ------------------------------------------------------------------

    def _constitutional_checks(self) -> List[Dict[str, Any]]:
        checks = self._cfg("constitutional.checks", [])
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
            return []
        return [dict(item) for item in checks if isinstance(item, Mapping)]

    def _check_constitutional_violations(
        self,
        text: str,
        *,
        include_metadata: bool = False,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        normalized = normalize_text(
            truncate_string(
                text,
                max_length=coerce_int(self._cfg("max_input_text_length"), 12000, minimum=1),
                suffix="",
            ),
            lowercase=True,
            collapse_whitespace=False,
        )
        if not normalized:
            return []
        violations: List[Dict[str, Any]] = []
        for check in self._constitutional_checks():
            patterns = check.get("patterns", [])
            if not isinstance(patterns, Sequence) or isinstance(patterns, (str, bytes)):
                continue
            for raw_pattern in patterns:
                try:
                    matched = re.search(str(raw_pattern), normalized, flags=re.IGNORECASE)
                except re.error as exc:
                    raise BaseValidationError(
                        "Invalid constitutional regex in agents_config.yaml.",
                        None,
                        component="SafetyAgent",
                        operation="constitutional_check",
                        context={
                            "rule_id": str(check.get("rule_id", "unknown")),
                            "pattern_fingerprint": self._fingerprint(str(raw_pattern)),
                        },
                        cause=exc,
                    ) from exc
                if matched:
                    violations.append(
                        {
                            "rule_id": str(check.get("rule_id", check.get("id", "constitutional_rule"))),
                            "category": str(check.get("category", "constitutional")),
                            "severity": str(check.get("severity", "medium")),
                            "description": self._safe_text(
                                check.get("description", "Configured constitutional rule matched."),
                                max_length=512,
                            ),
                            "pattern_fingerprint": self._fingerprint(str(raw_pattern)),
                        }
                    )
                    break
        if include_metadata:
            return violations
        return [f"{item['category']}:{item['rule_id']}" for item in violations]

    def _constitutional_risk(self, violations: Sequence[Mapping[str, Any]]) -> float:
        if not violations:
            return 0.0
        scores = self._cfg(
            "constitutional.severity_risk",
            {"low": 0.35, "medium": 0.55, "high": 0.75, "critical": 0.95},
        )
        if not isinstance(scores, Mapping):
            scores = {}
        return max(
            (_clamp_score(scores.get(str(item.get("severity", "medium")).lower(), 0.55)) for item in violations),
            default=0.0,
        )

    def _apply_constitutional_rules(self, output: str, assessment: Optional[Dict[str, Any]] = None) -> str:
        violations = self._check_constitutional_violations(output, include_metadata=True)
        if violations and coerce_bool(self._cfg("constitutional.redact_on_violation"), True):
            risk = self._constitutional_risk(violations) # type: ignore
            self._publish_agent_event(
                "safety_agent.constitutional_violation",
                {"violations": violations},
                decision="block",
                risk_score=risk,
            )
            return str(self._cfg("constitutional.block_message", "[SAFETY_BLOCK] Output violates configured safety policy."))
        return output

    # ------------------------------------------------------------------
    # Feedback and compatibility methods
    # ------------------------------------------------------------------


    def _request_human_feedback(
        self,
        input_data: Any,
        assessment: Mapping[str, Any],
        context: Optional[Mapping[str, Any]],
    ) -> None:
        feedback = _nested(context or {}, "human_feedback", None)
        if feedback is None:
            return
        self.collect_human_feedback(
            text=self._normalize_input_text(input_data),
            model_scores=_nested(assessment, "reports.reward_model", {}),
            human_rating=feedback,
            context={**dict(context or {}), "assessment_id": assessment.get("assessment_id")},
        )

    def _extract_features_from_assessment(self, assessment: Mapping[str, Any]) -> List[float]:
        risks = assessment.get("component_risks", {}) if isinstance(assessment, Mapping) else {}
        if not isinstance(risks, Mapping):
            risks = {}
        return [
            _clamp_score(risks.get("safety_guard", 0.0)),
            _clamp_score(risks.get("cyber_safety", 0.0)),
            _clamp_score(risks.get("adaptive_security", 0.0)),
            _clamp_score(risks.get("reward_model", 0.0)),
            _clamp_score(risks.get("attention_monitor", 0.0)),
            _clamp_score(risks.get("compliance", 0.0)),
            _clamp_score(assessment.get("risk_score", 0.0)),
        ]

    def _update_risk_aggregator(self, features: Sequence[float], human_rating: float) -> None:
        """Compatibility hook: persist feedback without duplicating RewardModel learning."""
        self._shared_put(
            "risk_feedback",
            generate_request_id("risk_fb"),
            {
                "features": [_clamp_score(value) for value in features],
                "human_rating": _clamp_score(human_rating),
                "timestamp": utc_now_iso(),
            },
            ttl=self._shared_ttl("feedback_ttl_seconds", 2592000),
            tags=["feedback", "risk_aggregation"],
        )

    def _contains_block_decision(self, value: Any) -> bool:
        if isinstance(value, Mapping):
            if value.get("is_phishing") is True:
                return True
            decision = str(value.get("decision", value.get("overall_recommendation", ""))).lower()
            if decision == "block":
                return True
            return any(self._contains_block_decision(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(self._contains_block_decision(item) for item in value)
        return False

    def _detect_violation(self, text: str, rule_text: str) -> bool:
        terms = {
            term
            for term in re.findall(r"\b[a-zA-Z][a-zA-Z-]{4,}\b", normalize_text(rule_text, lowercase=True))
        }
        if not terms:
            return False
        lowered = normalize_text(text, lowercase=True)
        hits = sum(1 for term in terms if term in lowered)
        return hits >= max(1, min(3, len(terms) // 5))

    def _finalize_agent_payload(self, payload: Any, *sensitive_sources: Any) -> Any:
        """Backward-compatible name for the agent's final safe-output barrier."""
        return self._safe_payload(payload)

    def _context_type(self, context: Mapping[str, Any]) -> str:
        return self._safe_identifier(
            context.get("type") or context.get("context_type") or context.get("source") or self.DEFAULT_CONTEXT_TYPE,
            default=self.DEFAULT_CONTEXT_TYPE,
        )

    def _safe_identifier(self, value: Any, *, default: str = "unknown") -> str:
        try:
            return normalize_identifier(value, max_length=128)
        except Exception:
            return default

    def reset_calls(self) -> None:
        with self._state_lock:
            self.calls = 0

    def collect_human_feedback(
        self,
        text: str,
        model_scores: Dict[str, Any],
        human_rating: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        feedback_id = generate_request_id("reward_feedback")
        record = self._safe_payload(
            {
                "feedback_id": feedback_id,
                "timestamp": utc_now_iso(),
                "text_fingerprint": self._fingerprint(text),
                "model_scores": model_scores,
                "human_rating": _clamp_score(human_rating),
                "context": self._public_context(context or {}),
            }
        )
        with self._state_lock:
            self.training_data.append(record)
            limit = coerce_int(self._cfg("feedback.local_buffer_limit"), 1000, minimum=1)
            if len(self.training_data) > limit:
                del self.training_data[:-limit]
        if coerce_bool(self._cfg("feedback.store_feedback"), True):
            self._shared_put(
                "reward_feedback",
                feedback_id,
                record,
                ttl=self._shared_ttl("feedback_ttl_seconds", 2592000),
                tags=["feedback", "reward_feedback"],
                priority=1.0 - _clamp_score(human_rating),
            )
        if self.reward_model is not None and hasattr(self.reward_model, "record_feedback"):
            try:
                self.reward_model.record_feedback(
                    text=text,
                    model_scores=model_scores,
                    human_rating=human_rating,
                    context=context or {},
                )
            except Exception as exc:
                logger.warning("RewardModel feedback recording failed: %s", type(exc).__name__)
        return record

    def update_reward_model(self, min_samples: Optional[int] = None) -> bool:
        if self.reward_model is None:
            return False
        minimum = coerce_int(
            min_samples if min_samples is not None else self._cfg("feedback.min_samples_for_retrain", 25),
            25,
            minimum=1,
        )
        with self._state_lock:
            samples = list(self.training_data)
        if self.shared_memory is not None and hasattr(self.shared_memory, "get_by_tag"):
            try:
                for item in self.shared_memory.get_by_tag("reward_feedback", limit=minimum * 4):
                    value = item.get("value") if isinstance(item, Mapping) else None
                    if isinstance(value, Mapping):
                        samples.append(dict(value))
            except Exception as exc:
                logger.warning("Shared feedback retrieval failed: %s", type(exc).__name__)
        normalized: List[Dict[str, Any]] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            scores = sample.get("model_scores")
            rating = sample.get("human_rating")
            if isinstance(scores, Mapping) and rating is not None:
                normalized.append({"model_scores": dict(scores), "human_rating": _clamp_score(rating)})
        if len(normalized) < minimum:
            logger.info("Not enough SafetyAgent feedback samples: %d/%d", len(normalized), minimum)
            return False
        return self.reward_model.retrain_model(normalized) is not None

    def train_embedded_models(self, training_cycle_id: str) -> Dict[str, Any]:
        result = {
            "training_cycle_id": self._safe_identifier(training_cycle_id, default="training_cycle"),
            "timestamp": utc_now_iso(),
            "reward_model_updated": self.update_reward_model(),
        }
        self._shared_put("training", result["training_cycle_id"], result, tags=["training", "safety_agent"])
        return result

    def _calculate_risk(self, data_str: str, risk_type: str, context: Optional[Dict[str, Any]] = None) -> float:
        normalized_type = str(risk_type).lower()
        if normalized_type == "pii":
            denominator = max(coerce_int(self._cfg("compatibility.pii_count_for_max_risk"), 5, minimum=1), 1)
            return _clamp_score(self._detect_pii(data_str) / denominator)
        if normalized_type == "adversarial":
            return _clamp_score(self._cfg("compatibility.adversarial_match_risk"), 0.75) if self._detect_adversarial_patterns(data_str) else 0.0
        if normalized_type == "constitutional":
            constitutional_violations = self._check_constitutional_violations(data_str, include_metadata=True)
            metadata_violations: List[Mapping[str, Any]] = [
                violation
                for violation in constitutional_violations
                if isinstance(violation, Mapping)
            ]
            return self._constitutional_risk(metadata_violations)
        return _clamp_score((context or {}).get("risk"), 0.0)

    def _detect_pii(self, data_str: str) -> int:
        return (
            len(self.EMAIL_RE.findall(data_str or ""))
            + len(self.PAYMENT_CARD_RE.findall(data_str or ""))
            + len(self.SSN_RE.findall(data_str or ""))
            + len(self.SECRET_ASSIGNMENT_RE.findall(data_str or ""))
        )

    def _detect_adversarial_patterns(self, text: str) -> bool:
        patterns = self._cfg("adversarial_patterns", [])
        if not isinstance(patterns, Sequence) or isinstance(patterns, (str, bytes)):
            return False
        for pattern in patterns:
            try:
                if re.search(str(pattern), text or "", flags=re.IGNORECASE):
                    return True
            except re.error:
                logger.warning("Invalid SafetyAgent adversarial regex ignored: %s", self._fingerprint(str(pattern)))
        return False

    def _generate_self_critique(self, output_text: str, original_prompt: Optional[str] = None) -> str:
        assessment = self.perform_task(
            output_text,
            context={"type": "self_critique", "run_compliance": False, "run_adversarial_validation": False},
        )
        lines = [
            "Self-Critique:",
            f"- Output fingerprint: {self._fingerprint(output_text)}",
            f"- Decision: {assessment.get('decision')}",
            f"- Risk level: {assessment.get('risk_level')}",
            f"- Final safety score: {assessment.get('final_safety_score')}",
        ]
        if original_prompt:
            lines.append(f"- Prompt fingerprint: {self._fingerprint(original_prompt)}")
        violations = assessment.get("constitutional_violations", [])
        if violations:
            lines.append("- Constitutional concerns detected:")
            lines.extend(
                f"  • {item.get('category')}:{item.get('rule_id')}"
                for item in violations
                if isinstance(item, Mapping)
            )
        else:
            lines.append("- No configured constitutional rule match detected.")
        return "\n".join(lines)

    def analyze_attention_matrix(self, attention_tensor: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.attention_monitor is None:
            return {"status": "disabled", "reason": "attention_monitor unavailable"}
        assert self.attention_monitor is not None
        outcome = self._execute_component(
            "attention_monitor",
            lambda: self.attention_monitor.analyze_attention(attention_tensor, context=dict(context or {})), # type: ignore
        )
        if outcome.risk_score >= _clamp_score(self.risk_thresholds.get("review_threshold"), 0.45):
            self._publish_agent_event(
                "safety_agent.attention_anomaly",
                {"attention_report": self._safe_payload(outcome.report)},
                decision="review",
                risk_score=outcome.risk_score,
            )
        return self._safe_payload(outcome.report)

    def suggest_correction(self, current_assessment: Dict[str, Any], task_data: Any) -> Dict[str, Any]:
        decision = str(current_assessment.get("decision", current_assessment.get("overall_recommendation", "review"))).lower()
        risk = _clamp_score(current_assessment.get("risk_score"), 0.5)
        suggestions: List[str] = []
        if decision == "block":
            suggestions.append("Do not execute until the enforcing blockers are remediated.")
        if risk >= _clamp_score(self.risk_thresholds.get("review_threshold"), 0.45):
            suggestions.append("Reduce unnecessary sensitive content and route the operation through human review.")
        if current_assessment.get("constitutional_violations"):
            suggestions.append("Revise the output or action to satisfy the cited agent policy rules.")
        return self._safe_payload(
            {
                "correction_id": generate_request_id("safety_corr"),
                "timestamp": utc_now_iso(),
                "risk_score": risk,
                "suggestions": suggestions or ["No correction required."],
                "task_fingerprint": self._fingerprint(self._normalize_input_text(task_data)),
            }
        )

    def apply_corrections(self, action_params: Dict[str, Any], validation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        corrections: List[Dict[str, Any]] = []
        risk = _clamp_score(validation_result.get("risk_score"), 0.0)
        if risk >= _clamp_score(self._cfg("action_validation.require_human_review_above", 0.45)):
            corrections.append(
                {
                    "type": "human_review_required",
                    "reason": "Validation risk exceeds the configured human-review threshold.",
                    "action_fingerprint": self._fingerprint(action_params),
                }
            )
        # Inspect the original parameters; checking only the redacted serialization
        # would turn every redaction marker into a false credential finding.
        raw_serialized = json_dumps(to_json_safe(action_params)).lower()
        if any(marker in raw_serialized for marker in ("api_key", "password", "secret", "token")):
            corrections.append(
                {
                    "type": "secret_redaction_required",
                    "reason": "Action parameters appear to contain credential material.",
                }
            )
        return self._safe_payload(corrections)

    # ------------------------------------------------------------------
    # Incident, posture, audit, and BaseAgent compatibility
    # ------------------------------------------------------------------

    def _store_assessment(self, result: Mapping[str, Any]) -> None:
        if not self.store_assessments:
            return
        self._shared_put(
            "assessment",
            str(result.get("assessment_id")),
            dict(result),
            ttl=self._shared_ttl("assessment_ttl_seconds", 604800),
            tags=["assessment", str(result.get("decision")), str(result.get("risk_level"))],
            priority=_clamp_score(result.get("risk_score")),
        )

    def _trigger_alert(
        self,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        risk = {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 0.95}.get(str(severity).lower(), 0.50)
        alert = self._safe_payload(
            {
                "alert_id": generate_request_id("safety_alert"),
                "timestamp": utc_now_iso(),
                "severity": severity,
                "message": self._safe_text(message, max_length=512),
                "details": details or {},
            }
        )
        self._shared_put(
            "alert",
            alert["alert_id"],
            alert,
            ttl=self._shared_ttl("audit_ttl_seconds", 2592000),
            tags=["alert", severity],
            priority=risk,
        )
        self._publish_agent_event("safety_agent.alert", alert, decision="review", risk_score=risk)
        return alert

    def handle_incident(self, category: str, incident_details: Dict[str, Any]) -> Dict[str, Any]:
        response_plan = self._cfg(["incident_response", category], None)
        if not isinstance(response_plan, Sequence) or isinstance(response_plan, (str, bytes)):
            response_plan = self._cfg(
                "incident_response.default",
                ["Contain operation", "Preserve audit evidence", "Route for human review"],
            )
        incident = self._safe_payload(
            {
                "incident_id": generate_request_id("safety_inc"),
                "timestamp": utc_now_iso(),
                "category": str(category),
                "details": incident_details,
                "response_plan": list(response_plan),
            }
        )
        self._shared_put("incident", incident["incident_id"], incident, tags=["incident", str(category)], priority=0.9)
        if self.shared_memory is not None and hasattr(self.shared_memory, "log_intervention"):
            try:
                self.shared_memory.log_intervention(report=incident, human_input={"required": True})
            except Exception as exc:
                logger.warning("Shared intervention logging failed: %s", type(exc).__name__)
        return incident

    def assess_risk(self, overall_score: float, task_type: str = "general") -> bool:
        threshold = _clamp_score(
            self._cfg(["task_thresholds", task_type], self.risk_thresholds.get("overall_safety", 0.75))
        )
        return _clamp_score(overall_score) >= threshold

    def evaluate_overall_safety_posture(self) -> Dict[str, Any]:
        component_health = {
            name: {**dict(status), "metrics": self._component_metrics_snapshot(name)}
            for name, status in self.component_status.items()
        }
        latest: List[Any] = []
        if self.shared_memory is not None and hasattr(self.shared_memory, "get_by_tag"):
            try:
                latest = list(
                    self.shared_memory.get_by_tag(
                        "assessment",
                        limit=coerce_int(self._cfg("posture.assessment_sample_size"), 25, minimum=1),
                    )
                )
            except Exception as exc:
                logger.warning("Posture assessment retrieval failed: %s", type(exc).__name__)

        risks: List[float] = []
        decisions: List[str] = []
        for item in latest:
            value = item.get("value") if isinstance(item, Mapping) else None
            if isinstance(value, Mapping):
                risks.append(_clamp_score(value.get("risk_score"), 0.0))
                decisions.append(str(value.get("decision", "unknown")))
        average_risk = sum(risks) / len(risks) if risks else 0.0
        max_risk = max(risks, default=0.0)
        block_rate = sum(1 for value in decisions if value == "block") / len(decisions) if decisions else 0.0
        review_rate = sum(1 for value in decisions if value == "review") / len(decisions) if decisions else 0.0
        required_unhealthy = [
            name
            for name, status in self.component_status.items()
            if status.get("required") and not status.get("initialized")
        ]
        health_risk = _clamp_score(self._cfg("posture.required_component_failure_risk"), 0.80) if required_unhealthy else 0.0

        compliance_status = "not_evaluated"
        compliance_risk = 0.0
        if self.compliance_checker is not None and coerce_bool(self._cfg("posture.include_compliance"), True):
            compliance = self._execute_component("compliance_checker", self.compliance_checker.evaluate_compliance)
            compliance_status = str(compliance.report.get("status", "unknown"))
            compliance_risk = self._extract_compliance_risk(compliance.report)

        observed_risk = _clamp_score(
            0.45 * average_risk + 0.25 * max_risk + 0.15 * block_rate + 0.15 * review_rate
        )
        posture_risk = max(observed_risk, health_risk, compliance_risk)
        result = self._safe_payload(
            {
                "schema_version": POSTURE_SCHEMA_VERSION,
                "module_version": MODULE_VERSION,
                "timestamp": utc_now_iso(),
                "component_health": component_health,
                "required_unhealthy_components": required_unhealthy,
                "recent_assessment_count": len(risks),
                "average_recent_risk": average_risk,
                "maximum_recent_risk": max_risk,
                "block_rate": block_rate,
                "review_rate": review_rate,
                "compliance_status": compliance_status,
                "posture_risk": posture_risk,
                "risk_level": _risk_level(posture_risk),
                "decision": _threshold_decision(
                    posture_risk,
                    review=self.risk_thresholds.get("review_threshold", 0.45),
                    block=self.risk_thresholds.get("block_threshold", 0.75),
                ),
                "subsystem_memory_accessed_by_agent": False,
                "config_source": "agents_config.yaml",
            }
        )
        self._shared_put(
            "posture",
            "latest",
            result,
            ttl=self._shared_ttl("posture_ttl_seconds", 604800),
            tags=["posture"],
            priority=posture_risk,
        )
        return result

    def _component_metrics_snapshot(self, component: str) -> Dict[str, Any]:
        with self._state_lock:
            raw = dict(self.component_metrics.get(component, {}))
        calls = coerce_int(raw.get("calls"), 0, minimum=0)
        total = coerce_int(raw.get("total_duration_ms"), 0, minimum=0)
        raw["average_duration_ms"] = total / calls if calls else 0.0
        return raw

    def _log_audit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        risk = _clamp_score(event_data.get("risk_score", event_data.get("risk", 0.0)))
        self._publish_agent_event(
            event_type,
            event_data,
            decision=_threshold_decision(
                risk,
                review=self.risk_thresholds.get("review_threshold", 0.45),
                block=self.risk_thresholds.get("block_threshold", 0.75),
            ),
            risk_score=risk,
        )

    def export_audit_log(self, path: str = "output/safety/safety_agent_audit_log.jsonl") -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._state_lock:
            events = list(self.audit_trail)
        with output_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json_dumps(self._safe_payload(event), sort_keys=True) + "\n")
        return str(output_path)

    def register_utility(self, name: str, utility: Any) -> None:
        self._utilities[self._safe_identifier(name, default="utility")] = utility

    def _get_timestamp(self) -> int:
        return int(time.time())

    def predict(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.perform_task(input_data, context)

    def act(self, action_params: Dict[str, Any], action_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.validate_action(action_params, action_context)

    def get_action(self, observation: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.predict(observation, context)

    # =======================================================================================================
    # Local Helpers
    # =======================================================================================================

    def _extract_secret_spans(self, text: str) -> set[str]:
        """Return concrete substrings that must never reach agent output.
    
        The spans are captured from the raw input *before* any downstream
        component can strip the key prefix. Substrings are stored in both full
        and trailing-punctuation-stripped form so that
        ``password=SuperSecret123.`` also protects the bare value
        ``SuperSecret123`` if a downstream component reports only the value.
        """
        spans: set[str] = set()
        for match in self.SECRET_ASSIGNMENT_RE.finditer(text or ""):
            value = match.group("value")
            if not value:
                continue
            spans.add(value)
            stripped = value.rstrip(".,;:!?")
            if stripped and stripped != value and len(stripped) >= 4:
                spans.add(stripped)
        # Return only spans long enough to be meaningful.
        return {span for span in spans if len(span) >= 6}
    
    
    def _scrub_known_secrets(self, value: Any, secrets: set[str]) -> Any:
        """Recursively remove known-secret substrings from any structure."""
        if not secrets:
            return value
        if isinstance(value, str):
            for secret in secrets:
                if secret in value:
                    value = value.replace(secret, "[REDACTED:known_secret]")
            return value
        if isinstance(value, Mapping):
            return {
                (self._scrub_known_secrets(k, secrets) if isinstance(k, str) else k):
                self._scrub_known_secrets(v, secrets)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._scrub_known_secrets(item, secrets) for item in value]
        if isinstance(value, (set, frozenset)):
            return {self._scrub_known_secrets(item, secrets) for item in value}
        if is_dataclass(value) and not isinstance(value, type):
            return self._scrub_known_secrets(asdict(value), secrets)
        return value


if __name__ == "__main__":
    print("\n=== Running Safety Agent ===\n")
    printer.status("TEST", "Safety Agent initialization", "info")
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    safety_agent = SafetyAgent(agent_factory=None, shared_memory=shared_memory)

    assessment = safety_agent.perform_task(
        "Please explain how to keep an account secure without exposing credentials.",
        context={
            "type": "cyber_security",
            "source": "self_test",
            "run_compliance": False,
            "run_adversarial_validation": False,
        },
    )
    assert assessment["decision"] in {"allow", "review", "block"}
    assert assessment["metadata"]["subsystem_memory_accessed_by_agent"] is False

    pii = safety_agent.perform_task(
        "Contact tester@example.com; password=SuperSecret123.",
        context={
            "type": "privacy",
            "source": "self_test",
            "run_compliance": False,
            "run_adversarial_validation": False,
        },
    )
    serialized = json_dumps(pii)
    def _find_leak(obj, needle, path="root"):
        if isinstance(obj, str):
            if needle in obj:
                i = obj.find(needle)
                print(f"LEAK at {path}: ...{obj[max(0,i-40):i+len(needle)+40]!r}...")
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                _find_leak(v, needle, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _find_leak(v, needle, f"{path}[{i}]")
    
    _find_leak(pii, "SuperSecret123")
    assert "tester@example.com" not in serialized
    assert "SuperSecret123" not in serialized

    validation = safety_agent.validate_action(
        {
            "action_name": "deploy_service",
            "parameters": {
                "image": "registry.example/app:latest",
                "config": "api_key=abc123SECRETtoken",
            },
        },
        action_context={"source": "self_test", "run_adversarial_validation": False},
    )
    assert validation["decision"] in {"allow", "review", "block"}

    posture = safety_agent.evaluate_overall_safety_posture()
    assert posture["decision"] in {"allow", "review", "block"}
    print("\n=== Safety Agent self-test completed ===\n")
