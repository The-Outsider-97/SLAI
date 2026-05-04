from __future__ import annotations

__version__ = "2.1.0"

"""
Production-grade Safety Agent for the SLAI multi-agent runtime.

The SafetyAgent is the top-level orchestration layer for the safety subsystem.
It coordinates specialized safety components, aggregates their signals, writes
cross-agent evidence to collaborative SharedMemory, and produces a bounded
allow/review/block decision for incoming content or proposed actions.
"""

import json
import math
import os
import re
import time
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/slai_matplotlib_cache")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .base_agent import BaseAgent
from .base.utils.main_config_loader import load_global_config, get_config_section
from .safety.secure_stpa import SecureSTPA
from .safety.reward_model import RewardModel
from .safety.cyber_safety import CyberSafetyModule
from .safety.safety_guard import SafetyGuard
from .safety.compliance_checker import ComplianceChecker
from .safety.attention_monitor import AttentionMonitor
from .safety.adaptive_security import AdaptiveSecurity
from .safety.utils.security_error import *
from .safety.utils.safety_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Safety Agent")
printer = PrettyPrinter()

MODULE_VERSION = __version__
ASSESSMENT_SCHEMA_VERSION = "safety_agent.assessment.v3"
ACTION_VALIDATION_SCHEMA_VERSION = "safety_agent.action_validation.v3"
AUDIT_SCHEMA_VERSION = "safety_agent.audit.v2"
POSTURE_SCHEMA_VERSION = "safety_agent.posture.v2"


@dataclass(frozen=True)
class SafetyAuditEvent:
    """Audit-safe event emitted by the SafetyAgent into SharedMemory."""

    event_id: str
    timestamp: str
    event_type: str
    assessment_id: Optional[str]
    decision: str
    risk_score: float
    risk_level: str
    component: str = "safety_agent"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging({
            "schema_version": AUDIT_SCHEMA_VERSION,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "assessment_id": self.assessment_id,
            "decision": self.decision,
            "risk_score": clamp_score(self.risk_score),
            "risk_level": self.risk_level,
            "component": self.component,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class SafetyAssessment:
    """Structured SafetyAgent result for an input/content assessment."""

    schema_version: str
    module_version: str
    assessment_id: str
    timestamp: str
    input_type: str
    input_fingerprint: str
    context_type: str
    sanitized_text: str
    reports: Dict[str, Any]
    component_risks: Dict[str, float]
    component_weights: Dict[str, float]
    blockers: List[str]
    warnings: List[str]
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
        return sanitize_for_logging({
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "assessment_id": self.assessment_id,
            "timestamp": self.timestamp,
            "input_type": self.input_type,
            "input_fingerprint": self.input_fingerprint,
            "context_type": self.context_type,
            "sanitized_text": redact_text(self.sanitized_text, max_length=4096),
            "reports": self.reports,
            "component_risks": {key: clamp_score(value) for key, value in self.component_risks.items()},
            "component_weights": {key: max(0.0, coerce_float(value, 0.0)) for key, value in self.component_weights.items()},
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "constitutional_violations": self.constitutional_violations,
            "final_safety_score": clamp_score(self.final_safety_score),
            "risk_score": clamp_score(self.risk_score),
            "risk_level": self.risk_level,
            "decision": self.decision,
            "overall_recommendation": self.overall_recommendation,
            "is_safe": bool(self.is_safe),
            "aggregation_method": self.aggregation_method,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class ActionValidationResult:
    """Structured result for pre-execution action validation."""

    schema_version: str
    validation_id: str
    timestamp: str
    action_name: str
    action_fingerprint: str
    approved: bool
    decision: str
    risk_score: float
    risk_level: str
    component_risks: Dict[str, float]
    details: List[str]
    corrections: List[Dict[str, Any]]
    reports: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_logging({
            "schema_version": self.schema_version,
            "validation_id": self.validation_id,
            "timestamp": self.timestamp,
            "action_name": self.action_name,
            "action_fingerprint": self.action_fingerprint,
            "approved": bool(self.approved),
            "decision": self.decision,
            "risk_score": clamp_score(self.risk_score),
            "risk_level": self.risk_level,
            "component_risks": {key: clamp_score(value) for key, value in self.component_risks.items()},
            "details": list(self.details),
            "corrections": redact_value(self.corrections),
            "reports": self.reports,
            "metadata": self.metadata,
        })


class SafetyAgent(BaseAgent):
    """
    Holistic safety management agent integrating specialized safety modules.

    The class keeps orchestration concerns at the agent boundary: module routing,
    cross-module aggregation, shared-memory publication, constitutional checks,
    action validation, incident escalation, and safety posture reporting. Pattern
    matching, cyber-safety analysis, reward scoring, STPA modeling, compliance,
    attention analysis, and secure persistence remain owned by their subsystem
    modules.
    """

    DEFAULT_CONTEXT_TYPE = "general"
    TEXT_CONTENT_KEYS: Tuple[str, ...] = ("text", "message", "prompt", "query", "content", "body", "input")
    SENSITIVE_ASSIGNMENT_RE = re.compile(
        r"(?i)\b(?:api[_-]?key|apikey|secret|token|password|passwd|pwd|credential|authorization|auth|cookie|session|private[_-]?key|access[_-]?key|refresh[_-]?token|client[_-]?secret)\s*[:=]\s*['\"]?(?P<value>[A-Za-z0-9._~+/=:-]{6,})"
    )
    EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    PAYMENT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    RAW_SECRET_MIN_LENGTH = 6


    def __init__(self, agent_factory: Any = None, shared_memory: Any = None, config: Optional[Mapping[str, Any]] = None):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.name = "Safety_Agent"
        self.config = load_global_config()
        self.safety_config = get_config_section("safety_agent")
        if config:
            self.safety_config = {**dict(self.safety_config or {}), **dict(config)}
        self._validate_configuration()

        self.audit_level = coerce_int(self._cfg("audit_level", 2), 2, minimum=0, maximum=5)
        self.collect_feedback_enabled = coerce_bool(self._cfg("collect_feedback", False), False)
        self.enable_learnable_aggregation = coerce_bool(self._cfg("enable_learnable_aggregation", False), False)
        self.fail_closed_on_component_error = coerce_bool(self._cfg("fail_closed_on_component_error", True), True)
        self.store_assessments = coerce_bool(self._cfg("shared_memory.store_assessments", True), True)
        self.store_audit_events = coerce_bool(self._cfg("shared_memory.store_audit_events", True), True)
        self.assessment_history_limit = coerce_int(self._cfg("assessment_history_limit", 500), 500, minimum=10)
        self.audit_trail_limit = coerce_int(self._cfg("audit_trail_limit", 1000), 1000, minimum=50)
        self.risk_thresholds = dict(self._cfg("risk_thresholds", {}))
        self.aggregation_weights = dict(self._cfg("aggregation_weights", {}))
        self.component_timeouts = dict(self._cfg("component_timeouts", {}))
        self.enabled_components = dict(self._cfg("components", {}))
        self.architecture_map = dict(self._cfg("architecture_map", {}))
        self.system_models = dict(self._cfg("system_models", {}))
        self.known_hazards = list(self._cfg("known_hazards", []))
        self.global_losses = list(self._cfg("global_losses", []))
        self.safety_policies = list(self._cfg("safety_policies", []))
        self.formal_specs = dict(self._cfg("formal_specs", {}))
        self.fault_tree_config = dict(self._cfg("fault_tree_config", {}))

        self.reward_model = RewardModel()
        self.attention_monitor = AttentionMonitor()
        self.safety_guard = SafetyGuard()
        self.secure_stpa = SecureSTPA()
        self.compliance_checker = ComplianceChecker()
        self.adaptive_security = AdaptiveSecurity()
        self.cyber_safety = CyberSafetyModule()

        self.constitution = self._load_constitution()
        self.training_data: List[Dict[str, Any]] = []
        self.risk_table: Dict[str, Any] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        self.calls = 0
        self.learning_factory = None
        self.risk_aggregator = None
        self._init_learning_factory()

        self._publish_agent_event(
            "safety_agent.initialized",
            {
                "enabled_components": self.enabled_components,
                "audit_level": self.audit_level,
                "schema_version": self._cfg("schema_version"),
            },
            decision="allow",
            risk_score=0.0,
        )
        logger.info(
            "Safety Agent initialized: %s",
            stable_json(safe_log_payload("safety_agent.initialized", {"components": list(self.enabled_components)})),
        )

    # ------------------------------------------------------------------
    # Configuration and shared-memory helpers
    # ------------------------------------------------------------------

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.safety_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.safety_config, Mapping):
            raise ConfigurationTamperingError(
                "agents_config.yaml:safety_agent",
                "safety_agent section must be a mapping",
                component="safety_agent",
            )
        if not isinstance(self.safety_config.get("risk_thresholds", {}), Mapping):
            raise ConfigurationTamperingError(
                "agents_config.yaml:safety_agent.risk_thresholds",
                "risk_thresholds must be a mapping",
                component="safety_agent",
            )
        if not isinstance(self.safety_config.get("aggregation_weights", {}), Mapping):
            raise ConfigurationTamperingError(
                "agents_config.yaml:safety_agent.aggregation_weights",
                "aggregation_weights must be a mapping",
                component="safety_agent",
            )
        block_threshold = clamp_score(get_nested(self.safety_config, "risk_thresholds.block_threshold", 0.75))
        review_threshold = clamp_score(get_nested(self.safety_config, "risk_thresholds.review_threshold", 0.45))
        if review_threshold > block_threshold:
            raise ConfigurationTamperingError(
                "agents_config.yaml:safety_agent.risk_thresholds",
                "review_threshold must be <= block_threshold",
                component="safety_agent",
            )

    def _shared_ttl(self, key: str, default: int = 604800) -> int:
        return coerce_int(self._cfg(["shared_memory", key], default), default, minimum=1)

    def _shared_key(self, kind: str, identifier: str) -> str:
        prefix = normalize_identifier(self._cfg("shared_memory.key_prefix", "safety_agent"), max_length=80)
        kind_clean = normalize_identifier(kind, max_length=80)
        ident_clean = normalize_identifier(identifier, max_length=160)
        return f"{prefix}:{kind_clean}:{ident_clean}"

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
        if self.shared_memory is None or not hasattr(self.shared_memory, "put"):
            return
        try:
            key = self._shared_key(kind, identifier)
            safe_value = self._finalize_agent_payload(dict(value))
            shared_tags = list(tags or []) + ["safety_agent", kind]
            self.shared_memory.put(
                key,
                safe_value,
                ttl=ttl or self._shared_ttl("assessment_ttl_seconds"),
                priority=priority,
                tags=dedupe_preserve_order(shared_tags),
                metadata={"component": "safety_agent", "kind": kind},
            )
        except Exception as exc:
            raise AuditLogFailureError(
                "safety_agent.shared_memory",
                f"Failed to write {kind} to shared memory: {type(exc).__name__}",
                component="safety_agent",
                cause=exc,
            )

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
            event_id=generate_identifier("safety_evt"),
            timestamp=utc_iso(),
            event_type=event_type,
            assessment_id=assessment_id,
            decision=decision,
            risk_score=clamp_score(risk_score),
            risk_level=categorize_risk(risk_score),
            metadata=dict(metadata),
        ).to_dict()
        self.audit_trail.append(event)
        if len(self.audit_trail) > self.audit_trail_limit:
            self.audit_trail = self.audit_trail[-self.audit_trail_limit:]
        if self.store_audit_events:
            self._shared_put(
                "event",
                event["event_id"],
                event,
                ttl=self._shared_ttl("audit_ttl_seconds", 2592000),
                tags=["audit", event_type],
                priority=risk_score,
            )

    def _extract_text(self, data: Any) -> str:
        if isinstance(data, str):
            return normalize_text(data, max_length=coerce_int(self._cfg("max_input_text_length", 12000), 12000), preserve_newlines=True)
        if isinstance(data, Mapping):
            for key in self.TEXT_CONTENT_KEYS:
                if key in data and isinstance(data[key], str):
                    return normalize_text(data[key], max_length=coerce_int(self._cfg("max_input_text_length", 12000), 12000), preserve_newlines=True)
            return stable_json(redact_value(data))
        if isinstance(data, (list, tuple, set)):
            return stable_json(redact_value(list(data)))
        return normalize_text(str(data), max_length=coerce_int(self._cfg("max_input_text_length", 12000), 12000), preserve_newlines=True)

    def _result_to_dict(self, value: Any) -> Any:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        if hasattr(value, "to_legacy_scores") and callable(value.to_legacy_scores):
            return value.to_legacy_scores()
        if isinstance(value, Mapping):
            return sanitize_for_logging(dict(value))
        if isinstance(value, (list, tuple)):
            return [self._result_to_dict(item) for item in value]
        return redact_value(value)

    def _extract_sensitive_literals(self, *sources: Any) -> List[str]:
        """
        Extract exact high-risk literals from raw inputs so the agent boundary can
        scrub values even if a downstream subsystem returns raw evidence.

        This is intentionally narrow: it only captures concrete PII/credential
        material, not every token, so reports remain useful while secrets,
        account identifiers, and contact data are never returned or stored.
        """

        literals: List[str] = []

        def visit(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, Mapping):
                for raw_key, raw_value in value.items():
                    key = str(raw_key)
                    if get_sensitive_key_regex().search(key) and raw_value not in (None, ""):
                        raw = str(raw_value)
                        literals.append(raw)
                        for match in self.SENSITIVE_ASSIGNMENT_RE.finditer(f"{key}={raw}"):
                            candidate = match.group("value").strip("\"'`;,. )]}")
                            if len(candidate) >= self.RAW_SECRET_MIN_LENGTH:
                                literals.append(candidate)
                    visit(raw_value)
                return
            if isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    visit(item)
                return

            text_value = str(value)
            if not text_value:
                return

            for regex in (self.EMAIL_RE, self.PAYMENT_CARD_RE, self.SSN_RE):
                for match in regex.finditer(text_value):
                    literals.append(match.group(0))

            for match in self.SENSITIVE_ASSIGNMENT_RE.finditer(text_value):
                assignment = match.group(0).strip()
                secret_value = match.group("value").strip("\"'`;,. )]}")
                literals.append(assignment)
                if len(secret_value) >= self.RAW_SECRET_MIN_LENGTH:
                    literals.append(secret_value)

        for source in sources:
            visit(source)

        safe_literals = []
        for literal in literals:
            cleaned = str(literal).strip()
            if len(cleaned) >= 3:
                safe_literals.append(cleaned)
        return sorted(dedupe_preserve_order(safe_literals), key=len, reverse=True)

    def _scrub_text_with_literals(self, text: str, sensitive_literals: Sequence[str]) -> str:
        scrubbed = redact_text(text, max_length=coerce_int(self._cfg("max_output_text_length", 12000), 12000))

        def replace_assignment(match: re.Match[str]) -> str:
            return f"[REDACTED:secret_assignment:{fingerprint(match.group(0))}]"

        scrubbed = self.SENSITIVE_ASSIGNMENT_RE.sub(replace_assignment, scrubbed)

        for literal in sensitive_literals:
            if not literal:
                continue
            replacement = f"[REDACTED:literal:{fingerprint(literal)}]"
            scrubbed = scrubbed.replace(literal, replacement)
        return scrubbed

    def _scrub_payload_with_literals(self, payload: Any, sensitive_literals: Sequence[str]) -> Any:
        if isinstance(payload, str):
            return self._scrub_text_with_literals(payload, sensitive_literals)
        if isinstance(payload, Mapping):
            return {
                str(key): self._scrub_payload_with_literals(value, sensitive_literals)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [self._scrub_payload_with_literals(item, sensitive_literals) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self._scrub_payload_with_literals(item, sensitive_literals) for item in payload)
        if isinstance(payload, set):
            return {self._scrub_payload_with_literals(item, sensitive_literals) for item in payload}
        return payload

    def _finalize_agent_payload(self, payload: Any, *sensitive_sources: Any) -> Any:
        """
        Final redaction barrier for all data leaving SafetyAgent.

        Component modules already sanitize their own reports, but SafetyAgent is
        the orchestration boundary. This final pass prevents raw source secrets
        from escaping through nested reports, metadata, exception payloads,
        shared-memory entries, self-test serialization, or future integrations.
        """

        sensitive_literals = self._extract_sensitive_literals(*sensitive_sources)
        sanitized = sanitize_for_logging(payload)
        return self._scrub_payload_with_literals(sanitized, sensitive_literals)

    def _security_error_report(self, error: SecurityError) -> Dict[str, Any]:
        return {
            "status": "security_error",
            "public": error.to_public_response(),
            "log": error.to_log_record(),
            "audit": error.to_audit_format(include_sensitive=False),
        }

    def _generic_error_report(self, exc: BaseException, operation: str, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        wrapped = SecurityError.from_exception(
            exc,
            error_type=SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
            message=f"SafetyAgent operation failed: {operation}",
            component="safety_agent",
            severity=SecuritySeverity.HIGH if self.fail_closed_on_component_error else SecuritySeverity.MEDIUM,
            context={"operation": operation, "payload": sanitize_for_logging(payload or {})},
        )
        return self._security_error_report(wrapped)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _resolve_config_path(self, raw_path: Optional[str]) -> Optional[Path]:
        if not raw_path:
            return None
        candidate = Path(str(raw_path)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate
        possible = [candidate, Path.cwd() / candidate]
        config_path = self.config.get("__config_path__")
        if config_path:
            parent = Path(str(config_path)).resolve().parent
            possible.extend([parent / candidate, parent.parent / candidate, parent.parent.parent / candidate])
        possible.append(Path("/mnt/data") / candidate.name)
        for path in possible:
            if path.exists():
                return path
        return candidate

    def _load_constitution(self) -> Dict[str, Any]:
        inline = self._cfg("constitutional_rules")
        if isinstance(inline, Mapping) and inline:
            return sanitize_for_logging(dict(inline))
        path = self._resolve_config_path(self._cfg("constitutional_rules_path"))
        if not path:
            return {}
        try:
            raw = load_text_file(path, max_bytes=coerce_int(self._cfg("max_constitution_bytes", 1_048_576), 1_048_576))
            loaded = json.loads(raw)
            if not isinstance(loaded, Mapping):
                raise ValueError("constitutional rules must be a JSON object")
            return sanitize_for_logging(dict(loaded))
        except SecurityError:
            raise
        except Exception as exc:
            raise ConfigurationTamperingError(
                "agents_config.yaml:safety_agent.constitutional_rules_path",
                f"Failed to load constitutional rules: {type(exc).__name__}",
                component="safety_agent",
                cause=exc,
                context={"path": str(path)},
            )

    def _init_learning_factory(self) -> None:
        # The agent-specific flag remains, but the default production path avoids
        # importing learning subsystems unless explicitly enabled by orchestration.
        if not self.enable_learnable_aggregation:
            self.learning_factory = None
            self.risk_aggregator = None
            return
        self._publish_agent_event(
            "safety_agent.learnable_aggregation_requested",
            {"reason": "External learning factory integration should be injected by orchestrator."},
            decision="review",
            risk_score=0.25,
        )
        self.learning_factory = None
        self.risk_aggregator = None

    # ------------------------------------------------------------------
    # Main content assessment
    # ------------------------------------------------------------------

    def perform_task(self, data_to_assess: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform a comprehensive safety/security assessment for input data."""

        context = sanitize_for_logging(dict(context or {}))
        self.calls += 1
        assessment_id = generate_identifier("safety_assess")
        started = time.monotonic()
        input_text = self._extract_text(data_to_assess)
        input_fingerprint = fingerprint(input_text)
        context_type = normalize_identifier(context.get("type") or context.get("context_type") or context.get("source") or self.DEFAULT_CONTEXT_TYPE, max_length=96)
        reports: Dict[str, Any] = {}
        component_risks: Dict[str, float] = {}
        blockers: List[str] = []
        warnings: List[str] = []
        sanitized_text = input_text

        printer.status("SAFETY", "Performing comprehensive safety assessment", "info")

        # SafetyGuard is the first gate because it sanitizes content and detects fail-closed prompt/content risks.
        if self._component_enabled("safety_guard"):
            try:
                depth = str(context.get("sanitization_depth") or self._cfg("sanitization_depth", "full"))
                if hasattr(self.safety_guard, "analyze"):
                    guard_analysis = self.safety_guard.analyze(input_text, depth=depth, context=context)
                    guard_report = self._result_to_dict(guard_analysis)
                    sanitized_text = guard_report.get("sanitized_text", input_text) if isinstance(guard_report, Mapping) else input_text
                else:
                    sanitized_text = self.safety_guard.sanitize(input_text, depth=depth)
                    guard_report = self.safety_guard.get_protection_report(input_text)
                reports["safety_guard"] = guard_report
                guard_risk = self._extract_risk(guard_report, default=0.0)
                component_risks["safety_guard"] = guard_risk
                guard_decision = str(guard_report.get("decision", "allow")).lower() if isinstance(guard_report, Mapping) else "allow"
                if "[SAFETY_BLOCK]" in str(sanitized_text) or guard_decision == "block":
                    blockers.append("safety_guard_block")
            except SecurityError as exc:
                reports["safety_guard"] = self._security_error_report(exc)
                component_risks["safety_guard"] = clamp_score(exc.risk_score or 1.0)
                blockers.append("safety_guard_security_error")
            except Exception as exc:
                reports["safety_guard"] = self._generic_error_report(exc, "safety_guard", {"assessment_id": assessment_id})
                component_risks["safety_guard"] = 1.0 if self.fail_closed_on_component_error else 0.5
                if self.fail_closed_on_component_error:
                    blockers.append("safety_guard_component_error")

        if self._component_enabled("cyber_safety"):
            try:
                cyber_context = str(context.get("cyber_context", context_type))
                cyber_report = self._result_to_dict(self.cyber_safety.analyze_input(sanitized_text, context=cyber_context))
                reports["cyber_safety"] = cyber_report
                component_risks["cyber_safety"] = self._extract_risk(cyber_report, default=0.0)
                if component_risks["cyber_safety"] >= clamp_score(self.risk_thresholds.get("cyber_risk", 0.70)):
                    warnings.append("cyber_risk_above_threshold")
            except SecurityError as exc:
                reports["cyber_safety"] = self._security_error_report(exc)
                component_risks["cyber_safety"] = clamp_score(exc.risk_score or 1.0)
                if exc.blocked:
                    blockers.append("cyber_safety_security_error")
            except Exception as exc:
                reports["cyber_safety"] = self._generic_error_report(exc, "cyber_safety", {"assessment_id": assessment_id})
                component_risks["cyber_safety"] = 1.0 if self.fail_closed_on_component_error else 0.5
                if self.fail_closed_on_component_error:
                    blockers.append("cyber_safety_component_error")

        if self._component_enabled("adaptive_security"):
            adaptive_report = self._run_adaptive_analysis(data_to_assess)
            if adaptive_report:
                reports["adaptive_security"] = adaptive_report
                component_risks["adaptive_security"] = self._extract_risk(adaptive_report, default=0.0)
                if self._contains_block_decision(adaptive_report):
                    blockers.append("adaptive_security_block")

        if self._component_enabled("attention_monitor") and context.get("attention_matrix") is not None:
            try:
                attention_report = self._result_to_dict(self.attention_monitor.analyze_attention(context["attention_matrix"], context=context))
                reports["attention_analysis"] = attention_report
                component_risks["attention_monitor"] = self._extract_risk(attention_report, default=0.0)
                if self._contains_block_decision(attention_report):
                    blockers.append("attention_monitor_block")
            except SecurityError as exc:
                reports["attention_analysis"] = self._security_error_report(exc)
                component_risks["attention_monitor"] = clamp_score(exc.risk_score or 0.75)
                if exc.blocked:
                    blockers.append("attention_monitor_security_error")
            except Exception as exc:
                reports["attention_analysis"] = self._generic_error_report(exc, "attention_monitor", {"assessment_id": assessment_id})
                component_risks["attention_monitor"] = 1.0 if self.fail_closed_on_component_error else 0.5
                if self.fail_closed_on_component_error:
                    blockers.append("attention_monitor_component_error")

        if self._component_enabled("reward_model"):
            try:
                if hasattr(self.reward_model, "evaluate_detailed"):
                    reward_report = self._result_to_dict(self.reward_model.evaluate_detailed(sanitized_text, context=context))
                else:
                    reward_report = self._result_to_dict(self.reward_model.evaluate(sanitized_text, context=context))
                reports["reward_model"] = reward_report
                component_risks["reward_model"] = self._extract_reward_risk(reward_report)
            except SecurityError as exc:
                reports["reward_model"] = self._security_error_report(exc)
                component_risks["reward_model"] = clamp_score(exc.risk_score or 0.75)
                if exc.blocked:
                    blockers.append("reward_model_security_error")
            except Exception as exc:
                reports["reward_model"] = self._generic_error_report(exc, "reward_model", {"assessment_id": assessment_id})
                component_risks["reward_model"] = 1.0 if self.fail_closed_on_component_error else 0.5
                if self.fail_closed_on_component_error:
                    blockers.append("reward_model_component_error")

        if self._should_evaluate_compliance(context):
            try:
                compliance_report = self._result_to_dict(self.compliance_checker.evaluate_compliance())
                reports["compliance"] = compliance_report
                component_risks["compliance"] = self._extract_compliance_risk(compliance_report)
                if self.risk_thresholds.get("compliance_failure_is_blocker", True) and str(compliance_report.get("status", "")).lower() in {"critical", "fail", "failed", "blocked", "error"}:
                    blockers.append("compliance_blocker")
            except SecurityError as exc:
                reports["compliance"] = self._security_error_report(exc)
                component_risks["compliance"] = clamp_score(exc.risk_score or 0.75)
                if exc.blocked:
                    blockers.append("compliance_security_error")
            except Exception as exc:
                reports["compliance"] = self._generic_error_report(exc, "compliance", {"assessment_id": assessment_id})
                component_risks["compliance"] = 1.0 if self.fail_closed_on_component_error else 0.5
                if self.fail_closed_on_component_error:
                    blockers.append("compliance_component_error")

        constitutional_violations = self._check_constitutional_violations(sanitized_text, include_metadata=True)
        if constitutional_violations:
            reports["constitutional"] = {"violations": constitutional_violations, "status": "violation"}
            component_risks["constitutional"] = self._constitutional_risk(constitutional_violations)
            if coerce_bool(self._cfg("constitutional.block_on_violation", True), True):
                blockers.append("constitutional_violation")
        else:
            reports["constitutional"] = {"violations": [], "status": "pass"}
            component_risks.setdefault("constitutional", 0.0)

        aggregation_method = "weighted_risk"
        risk_score = self._aggregate_component_risks(component_risks)
        if blockers:
            risk_score = max(risk_score, clamp_score(self.risk_thresholds.get("block_threshold", 0.75)))
        final_safety_score = clamp_score(1.0 - risk_score)
        decision = self._decision_from_risk(risk_score, blockers=blockers)
        risk_level = categorize_risk(risk_score)
        overall_recommendation = self._recommendation_from_decision(decision, blockers=blockers)
        is_safe = decision == "allow"

        duration_ms = int((time.monotonic() - started) * 1000)
        assessment = SafetyAssessment(
            schema_version=ASSESSMENT_SCHEMA_VERSION,
            module_version=MODULE_VERSION,
            assessment_id=assessment_id,
            timestamp=utc_iso(),
            input_type=type(data_to_assess).__name__,
            input_fingerprint=input_fingerprint,
            context_type=context_type,
            sanitized_text=sanitized_text,
            reports=reports,
            component_risks=component_risks,
            component_weights=self.aggregation_weights,
            blockers=dedupe_preserve_order(blockers),
            warnings=dedupe_preserve_order(warnings),
            constitutional_violations=constitutional_violations,
            final_safety_score=final_safety_score,
            risk_score=risk_score,
            risk_level=risk_level,
            decision=decision,
            overall_recommendation=overall_recommendation,
            is_safe=is_safe,
            aggregation_method=aggregation_method,
            metadata={
                "duration_ms": duration_ms,
                "call_count": self.calls,
                "context": context,
            },
        )
        result = self._finalize_agent_payload(
            assessment.to_dict(),
            data_to_assess,
            input_text,
            sanitized_text,
            context,
            reports,
        )
        self._store_assessment(result)
        self._publish_agent_event(
            "safety_agent.assessment_completed",
            {"input_type": result["input_type"], "blockers": result["blockers"], "duration_ms": duration_ms},
            assessment_id=assessment_id,
            decision=decision,
            risk_score=risk_score,
        )

        if self.collect_feedback_enabled and context.get("human_feedback") is not None:
            self.collect_human_feedback(
                text=sanitized_text,
                model_scores=reports.get("reward_model", {}),
                human_rating=context.get("human_feedback"),
                context={**context, "assessment_id": assessment_id},
            )

        return result

    def _component_enabled(self, component: str) -> bool:
        raw = self.enabled_components.get(component, True)
        if isinstance(raw, Mapping):
            return coerce_bool(raw.get("enabled", True), True)
        return coerce_bool(raw, True)

    def _run_adaptive_analysis(self, data: Any) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        try:
            if isinstance(data, Mapping) and {"subject", "body"}.issubset(set(data.keys())):
                report["email_analysis"] = self._result_to_dict(self.adaptive_security.analyze_email(dict(data)))
            elif isinstance(data, str) and re.match(r"(?i)^https?://", data.strip()):
                report["url_analysis"] = self._result_to_dict(self.adaptive_security.analyze_url(data.strip()))
        except SecurityError as exc:
            report["security_error"] = self._security_error_report(exc)
        except Exception as exc:
            report["error"] = self._generic_error_report(exc, "adaptive_security")
        return sanitize_for_logging(report)

    def _store_assessment(self, result: Mapping[str, Any]) -> None:
        if not self.store_assessments:
            return
        self._shared_put(
            "assessment",
            str(result.get("assessment_id")),
            dict(result),
            ttl=self._shared_ttl("assessment_ttl_seconds", 604800),
            tags=["assessment", str(result.get("decision")), str(result.get("risk_level"))],
            priority=coerce_float(result.get("risk_score"), 0.0),
        )

    def _should_evaluate_compliance(self, context: Mapping[str, Any]) -> bool:
        if context.get("run_compliance") is not None:
            return coerce_bool(context.get("run_compliance"), False)
        every = coerce_int(self._cfg("compliance.evaluate_every_n_calls", 0), 0, minimum=0)
        if every > 0 and self.calls % every == 0:
            return True
        return coerce_bool(self._cfg("compliance.evaluate_on_task", False), False)

    # ------------------------------------------------------------------
    # Risk extraction and aggregation
    # ------------------------------------------------------------------

    def _extract_risk(self, report: Any, default: float = 0.0) -> float:
        if not isinstance(report, Mapping):
            return clamp_score(default)
        for key in ("risk_score", "normalized_risk", "anomaly_score", "phishing_score"):
            if key in report and isinstance(report[key], (int, float, str)):
                return clamp_score(report[key])
        if "security_assessment" in report and isinstance(report["security_assessment"], Mapping):
            return self._extract_risk(report["security_assessment"], default=default)
        nested_scores = []
        for value in report.values():
            if isinstance(value, Mapping):
                nested_scores.append(self._extract_risk(value, default=None))
        nested_scores = [score for score in nested_scores if score is not None]
        return max(nested_scores) if nested_scores else clamp_score(default)

    def _extract_reward_risk(self, reward_report: Any) -> float:
        if not isinstance(reward_report, Mapping):
            return 0.5
        if "risk_score" in reward_report:
            return clamp_score(reward_report["risk_score"])
        composite = reward_report.get("composite", reward_report.get("aggregate_score"))
        if composite is not None:
            return clamp_score(1.0 - clamp_score(composite))
        return 0.5

    def _extract_compliance_risk(self, compliance_report: Mapping[str, Any]) -> float:
        status = str(compliance_report.get("status", "unknown")).lower()
        if status in {"compliant", "pass", "passed", "ok"}:
            return 0.05
        if status in {"warning", "partial", "conditional"}:
            return 0.45
        if status in {"critical", "fail", "failed", "error", "blocked"}:
            return 0.9
        score = compliance_report.get("overall_score", compliance_report.get("score"))
        if score is not None:
            return clamp_score(1.0 - clamp_score(score))
        return 0.5

    def _constitutional_risk(self, violations: Sequence[Mapping[str, Any]]) -> float:
        if not violations:
            return 0.0
        severity_scores = {"low": 0.35, "medium": 0.55, "high": 0.75, "critical": 0.95}
        return max(severity_scores.get(str(item.get("severity", "medium")).lower(), 0.55) for item in violations)

    def _aggregate_component_risks(self, component_risks: Mapping[str, Any]) -> float:
        if not component_risks:
            return clamp_score(self._cfg("empty_assessment_risk", 0.5))
        return weighted_average(component_risks, self.aggregation_weights, default=0.5)

    def _decision_from_risk(self, risk_score: float, *, blockers: Optional[Sequence[str]] = None) -> str:
        if blockers:
            return "block"
        return threshold_decision(
            risk_score,
            block_threshold=self.risk_thresholds.get("block_threshold", self.risk_thresholds.get("cyber_risk", 0.75)),
            review_threshold=self.risk_thresholds.get("review_threshold", 0.45),
        )

    def _recommendation_from_decision(self, decision: str, *, blockers: Optional[Sequence[str]] = None) -> str:
        if decision == "allow":
            return "proceed"
        if decision == "review":
            return "human_review"
        if blockers:
            return "block_due_to_policy_or_security_control"
        return "block_or_review"

    def _contains_block_decision(self, value: Any) -> bool:
        if isinstance(value, Mapping):
            decision = str(value.get("decision", value.get("overall_recommendation", ""))).lower()
            if decision == "block" or value.get("is_phishing") is True:
                return True
            return any(self._contains_block_decision(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(self._contains_block_decision(item) for item in value)
        return False

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def validate_action(self, action_params: Dict[str, Any], action_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate a proposed action before execution."""

        if not isinstance(action_params, Mapping):
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Action validation requires a mapping of action parameters.",
                severity=SecuritySeverity.HIGH,
                component="safety_agent",
                response_action=SecurityResponseAction.BLOCK,
            )
        action_context = sanitize_for_logging(dict(action_context or {}))
        validation_id = generate_identifier("safety_action")
        action_name = normalize_identifier(action_params.get("action_name") or action_params.get("name") or "unknown_action", max_length=128)
        action_text = stable_json(redact_value(dict(action_params)))
        action_fingerprint = fingerprint(action_text)
        reports: Dict[str, Any] = {}
        component_risks: Dict[str, float] = {}
        details: List[str] = []

        printer.status("SAFETY", "Validating action", "info")

        try:
            cyber_report = self._result_to_dict(self.cyber_safety.analyze_input(action_text, context=action_context.get("cyber_context", "action_validation")))
            reports["cyber_safety"] = cyber_report
            component_risks["cyber_safety"] = self._extract_risk(cyber_report, default=0.0)
            if component_risks["cyber_safety"] >= clamp_score(self.risk_thresholds.get("cyber_risk", 0.70)):
                details.append("Cyber risk exceeds configured threshold.")
        except SecurityError as exc:
            reports["cyber_safety"] = self._security_error_report(exc)
            component_risks["cyber_safety"] = clamp_score(exc.risk_score or 1.0)
            details.append("Cyber safety security control raised an incident.")
        except Exception as exc:
            reports["cyber_safety"] = self._generic_error_report(exc, "validate_action.cyber_safety")
            component_risks["cyber_safety"] = 1.0 if self.fail_closed_on_component_error else 0.5
            details.append("Cyber safety validation failed.")

        try:
            reward_report = self._result_to_dict(self.reward_model.evaluate(action_text, context={**action_context, "type": "action_validation"}))
            reports["reward_model"] = reward_report
            component_risks["reward_model"] = self._extract_reward_risk(reward_report)
            if clamp_score(reward_report.get("composite", 0.0) if isinstance(reward_report, Mapping) else 0.0) < clamp_score(self.risk_thresholds.get("overall_safety", 0.75)):
                details.append("Reward/safety score below configured threshold.")
        except SecurityError as exc:
            reports["reward_model"] = self._security_error_report(exc)
            component_risks["reward_model"] = clamp_score(exc.risk_score or 0.75)
            details.append("Reward model security control raised an incident.")
        except Exception as exc:
            reports["reward_model"] = self._generic_error_report(exc, "validate_action.reward_model")
            component_risks["reward_model"] = 1.0 if self.fail_closed_on_component_error else 0.5
            details.append("Reward model validation failed.")

        stpa_report = self._validate_action_with_stpa(action_name, action_context)
        reports["secure_stpa"] = stpa_report
        component_risks["secure_stpa"] = self._extract_risk(stpa_report, default=0.0)
        if stpa_report.get("matched_uca_risks"):
            details.append("Action matches one or more unsafe control-action contexts.")

        constitutional_violations = self._check_constitutional_violations(action_text, include_metadata=True)
        if constitutional_violations:
            reports["constitutional"] = {"violations": constitutional_violations}
            component_risks["constitutional"] = self._constitutional_risk(constitutional_violations)
            details.append("Action description triggered constitutional review rules.")

        risk_score = self._aggregate_component_risks(component_risks)
        decision = self._decision_from_risk(risk_score, blockers=[d for d in details if "security control" in d.lower()] if risk_score >= clamp_score(self.risk_thresholds.get("block_threshold", 0.75)) else [])
        approved = decision == "allow" and not details
        corrections = [] if approved else self.apply_corrections(dict(action_params), {"risk_score": risk_score, "details": details, "reports": reports})
        if corrections and decision == "allow":
            decision = "review"
            approved = False

        result = self._finalize_agent_payload(
            ActionValidationResult(
                schema_version=ACTION_VALIDATION_SCHEMA_VERSION,
                validation_id=validation_id,
                timestamp=utc_iso(),
                action_name=action_name,
                action_fingerprint=action_fingerprint,
                approved=approved,
                decision=decision,
                risk_score=risk_score,
                risk_level=categorize_risk(risk_score),
                component_risks=component_risks,
                details=dedupe_preserve_order(details),
                corrections=corrections,
                reports=reports,
                metadata={"context": action_context},
            ).to_dict(),
            action_params,
            action_context,
            reports,
        )
        self._shared_put(
            "action_validation",
            validation_id,
            result,
            ttl=self._shared_ttl("validation_ttl_seconds", 604800),
            tags=["action_validation", decision, result["risk_level"]],
            priority=result["risk_score"],
        )
        self._publish_agent_event(
            "safety_agent.action_validation_completed",
            {"action_name": action_name, "approved": approved, "details": details},
            assessment_id=validation_id,
            decision=decision,
            risk_score=risk_score,
        )
        return result

    def _validate_action_with_stpa(self, action_name: str, action_context: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            if not self.secure_stpa.hazards:
                self.secure_stpa.define_analysis_scope(
                    losses=self.global_losses or ["Loss of safe, secure, or compliant operation"],
                    hazards=self.known_hazards or ["Unsafe or unauthorized system action"],
                    constraints=self.safety_policies or ["Actions must satisfy configured safety policy before execution"],
                    system_boundary=str(self._cfg("system_boundary", "Safety Agent orchestration boundary")),
                )
            if not self.secure_stpa.control_structure:
                structure = self.architecture_map or {
                    "Safety_Agent": {
                        "inputs": ["task_request", "shared_memory_context"],
                        "outputs": [action_name],
                        "process_vars": ["risk_score", "decision"],
                    }
                }
                self.secure_stpa.model_control_structure(structure=structure, process_models=self.system_models)
            ucas = self.secure_stpa.identify_unsafe_control_actions()
            context_tables = self.secure_stpa.build_context_tables(formal_spec=self.formal_specs, fta_config=self.fault_tree_config)
            matched: List[Dict[str, Any]] = []
            for entries in context_tables.values():
                for entry in entries:
                    if normalize_identifier(entry.get("control_action", ""), max_length=128) == action_name:
                        score = self._assess_contextual_match(entry, action_context)
                        if score >= clamp_score(self._cfg("action_validation.stpa_match_threshold", 0.70)):
                            matched.append({"entry": sanitize_for_logging(entry), "match_score": score})
            return sanitize_for_logging({
                "status": "evaluated",
                "uca_count": len(ucas),
                "matched_uca_risks": matched,
                "risk_score": max([item["match_score"] for item in matched], default=0.0),
            })
        except SecurityError as exc:
            return self._security_error_report(exc)
        except Exception as exc:
            return self._generic_error_report(exc, "secure_stpa.action_validation")

    def _assess_contextual_match(self, entry: Mapping[str, Any], execution_state: Mapping[str, Any]) -> float:
        if not execution_state:
            return 0.0
        state_text = normalize_text(stable_json(redact_value(execution_state)), lowercase=True)
        candidate_terms = []
        candidate_terms.extend(str(item).lower() for item in entry.get("process_variables", []) or [])
        candidate_terms.extend(str(item).lower() for item in entry.get("state_constraints", []) or [])
        candidate_terms.extend(str(entry.get("guideword", "")).lower().split())
        candidate_terms = [term for term in candidate_terms if term and len(term) > 2]
        if not candidate_terms:
            return 0.0
        hits = sum(1 for term in candidate_terms if term in state_text)
        return clamp_score(hits / max(len(candidate_terms), 1))

    # ------------------------------------------------------------------
    # Compatibility and feedback methods
    # ------------------------------------------------------------------

    def reset_calls(self) -> None:
        self.calls = 0

    def _request_human_feedback(self, input_data: Any, assessment: Mapping[str, Any], context: Optional[Mapping[str, Any]]) -> None:
        feedback = get_nested(context or {}, "human_feedback", None)
        if feedback is None:
            return
        self.collect_human_feedback(
            text=self._extract_text(input_data),
            model_scores=get_nested(assessment, "reports.reward_model", {}),
            human_rating=feedback,
            context={**dict(context or {}), "assessment_id": assessment.get("assessment_id")},
        )

    def _extract_features_from_assessment(self, assessment: Mapping[str, Any]) -> List[float]:
        risks = assessment.get("component_risks", {}) if isinstance(assessment, Mapping) else {}
        return [
            clamp_score(get_nested(risks, "safety_guard", 0.0)),
            clamp_score(get_nested(risks, "cyber_safety", 0.0)),
            clamp_score(get_nested(risks, "reward_model", 0.0)),
            clamp_score(get_nested(risks, "attention_monitor", 0.0)),
            clamp_score(get_nested(risks, "compliance", 0.0)),
            clamp_score(assessment.get("risk_score", 0.0) if isinstance(assessment, Mapping) else 0.0),
        ]

    def _update_risk_aggregator(self, features: Sequence[float], human_rating: float) -> None:
        # Reserved for externally injected learning modules. Store feedback safely for later use.
        self._shared_put(
            "risk_feedback",
            generate_identifier("risk_fb"),
            {"features": [clamp_score(v) for v in features], "human_rating": clamp_score(human_rating), "timestamp": utc_iso()},
            ttl=self._shared_ttl("feedback_ttl_seconds", 2592000),
            tags=["feedback", "risk_aggregation"],
        )

    def collect_human_feedback(self, text: str, model_scores: Dict[str, Any], human_rating: float, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        feedback_id = generate_identifier("reward_feedback")
        record = self._finalize_agent_payload(
            {
                "feedback_id": feedback_id,
                "timestamp": utc_iso(),
                "text_fingerprint": fingerprint(text),
                "model_scores": model_scores,
                "human_rating": clamp_score(human_rating),
                "context": context or {},
            },
            text,
            context or {},
            model_scores,
        )
        self.training_data.append(record)
        self._shared_put(
            "reward_feedback",
            feedback_id,
            record,
            ttl=self._shared_ttl("feedback_ttl_seconds", 2592000),
            tags=["feedback", "reward_feedback"],
            priority=1.0 - clamp_score(human_rating),
        )
        if hasattr(self.reward_model, "record_feedback"):
            self.reward_model.record_feedback(text=text, model_scores=model_scores, human_rating=human_rating, context=context or {})
        return record

    def update_reward_model(self, min_samples: Optional[int] = None) -> bool:
        min_samples = coerce_int(min_samples if min_samples is not None else self._cfg("feedback.min_samples_for_retrain", 25), 25, minimum=1)
        samples = list(self.training_data)
        if hasattr(self.shared_memory, "get_by_tag"):
            for item in self.shared_memory.get_by_tag("reward_feedback", limit=min_samples * 4):
                value = item.get("value") if isinstance(item, Mapping) else None
                if isinstance(value, Mapping):
                    samples.append(dict(value))
        normalized = []
        for sample in samples:
            scores = sample.get("model_scores") if isinstance(sample, Mapping) else None
            rating = sample.get("human_rating") if isinstance(sample, Mapping) else None
            if isinstance(scores, Mapping) and rating is not None:
                normalized.append({"model_scores": dict(scores), "human_rating": clamp_score(rating)})
        if len(normalized) < min_samples:
            logger.info("Not enough feedback samples for reward retraining: %d/%d", len(normalized), min_samples)
            return False
        self.reward_model.retrain_model(normalized)
        return True

    # ------------------------------------------------------------------
    # Risk, incident, correction, posture, and constitutional helpers
    # ------------------------------------------------------------------

    def _calculate_risk(self, data_str: str, risk_type: str, context: Optional[Dict[str, Any]] = None) -> float:
        text = normalize_text(data_str, max_length=coerce_int(self._cfg("max_input_text_length", 12000), 12000), lowercase=True)
        if risk_type == "pii":
            return clamp_score(self._detect_pii(text) / 5.0)
        if risk_type == "adversarial":
            return 0.75 if self._detect_adversarial_patterns(text) else 0.0
        if risk_type == "constitutional":
            return self._constitutional_risk(self._check_constitutional_violations(text, include_metadata=True))
        return clamp_score(context.get("risk", 0.0) if context else 0.0)

    def _detect_pii(self, data_str: str) -> int:
        report = self.safety_guard.get_protection_report(data_str)
        pii = report.get("pii_detected", []) if isinstance(report, Mapping) else []
        return len(pii)

    def _detect_adversarial_patterns(self, text: str) -> bool:
        patterns = list(self._cfg("adversarial_patterns", []))
        if not patterns:
            patterns = [
                r"ignore\s+(?:all\s+)?previous\s+instructions",
                r"developer\s+mode",
                r"jailbreak",
                r"system\s+prompt",
            ]
        return any(re.search(str(pattern), text, flags=re.IGNORECASE) for pattern in patterns)

    def _trigger_alert(self, severity: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        risk = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 0.95}.get(severity.lower(), 0.5)
        alert = {
            "alert_id": generate_identifier("safety_alert"),
            "timestamp": utc_iso(),
            "severity": severity,
            "message": redact_text(message, max_length=512),
            "details": sanitize_for_logging(details or {}),
        }
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
        response_plan = list(get_nested(self.safety_config, ["incident_response", category], []))
        if not response_plan:
            response_plan = list(get_nested(self.safety_config, "incident_response.default", ["Contain operation", "Preserve audit evidence", "Route for human review"]))
        incident = self._finalize_agent_payload(
            {
                "incident_id": generate_identifier("safety_inc"),
                "timestamp": utc_iso(),
                "category": category,
                "details": incident_details,
                "response_plan": response_plan,
            },
            incident_details,
        )
        self._shared_put("incident", incident["incident_id"], incident, tags=["incident", category], priority=0.9)
        if hasattr(self.shared_memory, "log_intervention"):
            self.shared_memory.log_intervention(report=incident, human_input={"required": True})
        return incident

    def assess_risk(self, overall_score: float, task_type: str = "general") -> bool:
        threshold = clamp_score(get_nested(self.safety_config, ["task_thresholds", task_type], self.risk_thresholds.get("overall_safety", 0.75)))
        return clamp_score(overall_score) >= threshold

    def train_embedded_models(self, training_cycle_id: str) -> Dict[str, Any]:
        reward_updated = self.update_reward_model()
        result = {
            "training_cycle_id": normalize_identifier(training_cycle_id, max_length=128),
            "timestamp": utc_iso(),
            "reward_model_updated": reward_updated,
            "learnable_aggregation_enabled": bool(self.enable_learnable_aggregation),
        }
        self._shared_put("training", result["training_cycle_id"], result, tags=["training", "safety_agent"])
        return result

    def evaluate_overall_safety_posture(self) -> Dict[str, Any]:
        component_health = {
            "safety_guard": self.safety_guard is not None,
            "reward_model": self.reward_model is not None,
            "cyber_safety": self.cyber_safety is not None,
            "adaptive_security": self.adaptive_security is not None,
            "compliance_checker": self.compliance_checker is not None,
            "attention_monitor": self.attention_monitor is not None,
            "secure_stpa": self.secure_stpa is not None,
        }
        latest_assessments = []
        if hasattr(self.shared_memory, "get_by_tag"):
            latest_assessments = self.shared_memory.get_by_tag("assessment", limit=coerce_int(self._cfg("posture.assessment_sample_size", 25), 25, minimum=1))
        risks = []
        for item in latest_assessments:
            value = item.get("value") if isinstance(item, Mapping) else None
            if isinstance(value, Mapping):
                risks.append(clamp_score(value.get("risk_score", 0.0)))
        average_recent_risk = sum(risks) / len(risks) if risks else 0.0
        compliance_status = "not_evaluated"
        if coerce_bool(self._cfg("posture.include_compliance", True), True):
            try:
                compliance_status = str(self.compliance_checker.evaluate_compliance().get("status", "unknown"))
            except Exception as exc:
                compliance_status = f"error:{type(exc).__name__}"
        posture_risk = combine_risk_scores(average_recent_risk, 0.0 if all(component_health.values()) else 0.5, 0.75 if compliance_status in {"critical", "fail", "error"} else 0.0, method="weighted_high")
        result = sanitize_for_logging({
            "schema_version": POSTURE_SCHEMA_VERSION,
            "timestamp": utc_iso(),
            "component_health": component_health,
            "recent_assessment_count": len(latest_assessments),
            "average_recent_risk": average_recent_risk,
            "posture_risk": posture_risk,
            "risk_level": categorize_risk(posture_risk),
            "decision": threshold_decision(posture_risk, block_threshold=self.risk_thresholds.get("block_threshold", 0.75), review_threshold=self.risk_thresholds.get("review_threshold", 0.45)),
            "compliance_status": compliance_status,
        })
        self._shared_put("posture", "latest", result, ttl=self._shared_ttl("posture_ttl_seconds", 604800), tags=["posture"])
        return result

    def _log_audit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        risk = clamp_score(event_data.get("risk_score", event_data.get("risk", 0.0))) if isinstance(event_data, Mapping) else 0.0
        self._publish_agent_event(event_type, event_data, decision=threshold_decision(risk), risk_score=risk)

    def export_audit_log(self, path: str = "src/agents/safety/safety_agent_audit_log.jsonl") -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for event in self.audit_trail:
                handle.write(stable_json(event) + "\n")
        return str(output_path)

    def suggest_correction(self, current_assessment: Dict[str, Any], task_data: Any) -> Dict[str, Any]:
        decision = str(current_assessment.get("decision", current_assessment.get("overall_recommendation", "review"))).lower()
        risk = clamp_score(current_assessment.get("risk_score", 0.5))
        suggestions = []
        if decision == "block":
            suggestions.append("Do not execute the requested operation until the blockers are remediated.")
        if risk >= 0.5:
            suggestions.append("Reduce sensitive content, remove credentials/PII, and retry with explicit safe intent.")
        if current_assessment.get("constitutional_violations"):
            suggestions.append("Revise the output or action to satisfy the cited constitutional rules.")
        return self._finalize_agent_payload(
            {
                "correction_id": generate_identifier("safety_corr"),
                "timestamp": utc_iso(),
                "risk_score": risk,
                "suggestions": suggestions or ["No correction required."],
                "task_fingerprint": fingerprint(self._extract_text(task_data)),
            },
            task_data,
            current_assessment,
        )

    def apply_corrections(self, action_params: Dict[str, Any], validation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        corrections = []
        risk = clamp_score(validation_result.get("risk_score", 0.0))
        if risk >= clamp_score(self.risk_thresholds.get("review_threshold", 0.45)):
            corrections.append({
                "type": "human_review_required",
                "reason": "Validation risk exceeds review threshold.",
                "action_fingerprint": fingerprint(stable_json(redact_value(action_params))),
            })
        serialized = stable_json(redact_value(action_params)).lower()
        if any(marker in serialized for marker in ("api_key", "password", "secret", "token")):
            corrections.append({"type": "secret_redaction_required", "reason": "Action parameters appear to contain credential material."})
        return sanitize_for_logging(corrections)

    def _apply_constitutional_rules(self, output: str, assessment: Optional[Dict[str, Any]] = None) -> str:
        violations = self._check_constitutional_violations(output, include_metadata=True)
        if violations and coerce_bool(self._cfg("constitutional.redact_on_violation", True), True):
            self._publish_agent_event("safety_agent.constitutional_violation", {"violations": violations}, decision="block", risk_score=self._constitutional_risk(violations))
            return str(self._cfg("constitutional.block_message", "[SAFETY_BLOCK] Output violates constitutional safety policy."))
        return output

    def _check_constitutional_violations(self, text: str, *, include_metadata: bool = False) -> Union[List[str], List[Dict[str, Any]]]:
        normalized = normalize_text(text, max_length=coerce_int(self._cfg("max_input_text_length", 12000), 12000), lowercase=True)
        if not normalized:
            return []
        checks = self._constitutional_checks()
        violations: List[Dict[str, Any]] = []
        for check in checks:
            patterns = list(check.get("patterns", []))
            for pattern in patterns:
                if re.search(str(pattern), normalized, flags=re.IGNORECASE):
                    violations.append({
                        "rule_id": str(check.get("rule_id", check.get("id", "constitutional_rule"))),
                        "category": str(check.get("category", "constitutional")),
                        "severity": str(check.get("severity", "medium")),
                        "description": str(check.get("description", "Configured constitutional rule matched.")),
                        "pattern_fingerprint": fingerprint(str(pattern)),
                    })
                    break
        if include_metadata:
            return violations
        return [f"{item['category']}:{item['rule_id']}" for item in violations]

    def _constitutional_checks(self) -> List[Dict[str, Any]]:
        configured = self._cfg("constitutional.checks", [])
        if isinstance(configured, list) and configured:
            return [dict(item) for item in configured if isinstance(item, Mapping)]
        checks: List[Dict[str, Any]] = []
        for category, section in (self.constitution or {}).items():
            if not isinstance(section, Mapping):
                continue
            for rule in section.get("rules", []) or []:
                if not isinstance(rule, Mapping):
                    continue
                rule_text = normalize_text(rule.get("rule", ""), lowercase=True)
                patterns = []
                for keyword in self._cfg("constitutional.default_keywords", ["self-harm", "credential", "pii", "unauthorized", "discriminate", "violence"]):
                    if str(keyword).lower() in rule_text:
                        patterns.append(re.escape(str(keyword)))
                if patterns:
                    checks.append({
                        "rule_id": rule.get("id", "unknown"),
                        "category": category,
                        "severity": "medium",
                        "description": rule.get("rule", "Constitutional rule"),
                        "patterns": patterns,
                    })
        return checks

    def _detect_violation(self, text: str, rule_text: str) -> bool:
        rule_terms = {term for term in re.findall(r"\b[a-zA-Z][a-zA-Z-]{4,}\b", normalize_text(rule_text, lowercase=True))}
        if not rule_terms:
            return False
        lowered = normalize_text(text, lowercase=True)
        hits = sum(1 for term in rule_terms if term in lowered)
        return hits >= max(1, min(3, len(rule_terms) // 5))

    def _generate_self_critique(self, output_text: str, original_prompt: Optional[str] = None) -> str:
        assessment = self.perform_task(output_text, context={"type": "self_critique", "run_compliance": False})
        violations = assessment.get("constitutional_violations", [])
        reward_scores = assessment.get("reports", {}).get("reward_model", {})
        lines = [
            "Self-Critique:",
            f"- Output fingerprint: {fingerprint(output_text)}",
            f"- Decision: {assessment.get('decision')}",
            f"- Risk level: {assessment.get('risk_level')}",
            f"- Final safety score: {assessment.get('final_safety_score')}",
        ]
        if original_prompt:
            lines.append(f"- Prompt fingerprint: {fingerprint(original_prompt)}")
        if violations:
            lines.append("- Constitutional concerns detected:")
            lines.extend(f"  • {item.get('category')}:{item.get('rule_id')}" for item in violations if isinstance(item, Mapping))
        else:
            lines.append("- No configured constitutional rule match detected.")
        if isinstance(reward_scores, Mapping):
            composite = reward_scores.get("composite", reward_scores.get("aggregate_score"))
            if composite is not None:
                lines.append(f"- Reward composite: {clamp_score(composite):.3f}")
        return "\n".join(lines)

    def analyze_attention_matrix(self, attention_tensor: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._component_enabled("attention_monitor"):
            return {"status": "disabled", "reason": "attention_monitor disabled in agents_config.yaml"}
        report = self._result_to_dict(self.attention_monitor.analyze_attention(attention_tensor, context=context or {}))
        risk = self._extract_risk(report, default=0.0)
        if risk >= clamp_score(self.risk_thresholds.get("review_threshold", 0.45)):
            self._publish_agent_event("safety_agent.attention_anomaly", {"attention_report": report}, decision=threshold_decision(risk), risk_score=risk)
        return report

    def register_utility(self, name: str, utility: Any) -> None:
        if not hasattr(self, "_utilities"):
            self._utilities: Dict[str, Any] = {}
        self._utilities[normalize_identifier(name, max_length=128)] = utility

    def _get_timestamp(self) -> int:
        return int(time.time())

    def predict(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.perform_task(input_data, context)

    def act(self, action_params: Dict[str, Any], action_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.validate_action(action_params, action_context)

    def get_action(self, observation: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.predict(observation, context)


if __name__ == "__main__":
    print("\n=== Running Safety Agent ===\n")
    printer.status("TEST", "Safety Agent initialized", "info")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory
    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    safety_agent = SafetyAgent(
        agent_factory=agent_factory,
        shared_memory=shared_memory,
        config={"collect_feedback": False}
        )

    safe_assessment = safety_agent.perform_task(
        "Please explain how to keep my account secure without sharing any private credentials.",
        context={"type": "cyber_security", "source": "self_test", "run_compliance": False},
    )
    assert safe_assessment["decision"] in {"allow", "review", "block"}
    assert "assessment_id" in safe_assessment

    pii_assessment = safety_agent.perform_task(
        "Contact me at tester@example.com and do not expose this password=SuperSecret123.",
        context={"type": "privacy", "source": "self_test", "run_compliance": False},
    )
    serialized_pii = stable_json(pii_assessment)
    assert "tester@example.com" not in serialized_pii
    assert "SuperSecret123" not in serialized_pii
    assert "password=SuperSecret123" not in serialized_pii

    action_validation = safety_agent.validate_action(
        {
            "action_name": "deploy_service",
            "parameters": {"image": "registry.example/app:latest", "config": "api_key=abc123SECRETtoken"},
        },
        action_context={"source": "self_test", "risk_context": 0.4},
    )
    assert action_validation["decision"] in {"allow", "review", "block"}
    assert "validation_id" in action_validation

    critique = safety_agent._generate_self_critique("This response avoids secrets and respects privacy.")
    assert "Self-Critique" in critique

    posture = safety_agent.evaluate_overall_safety_posture()
    assert posture["decision"] in {"allow", "review", "block"}

    audit_target = Path(tempfile.mkdtemp(prefix="safety_agent_test_")) / "safety_agent_audit_log.jsonl"
    audit_path = safety_agent.export_audit_log(str(audit_target))
    assert Path(audit_path).exists()

    print("\n=== Test ran successfully ===\n")
