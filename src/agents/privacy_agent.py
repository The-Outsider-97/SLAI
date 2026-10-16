"""
Privacy Agent orchestration for SLAI.

The agent is the composition/orchestration boundary for privacy evaluation. It
coordinates privacy subsystem services but does not own subsystem policy,
subsystem configuration, or subsystem memory.

Interfaces and dependencies
Inputs:
- Raw user prompts and uploads
- Reader/browser extracted content
- Memory writes/reads
- External tool invocation payloads

Outputs:
- Allow/modify/block decision
- Sanitized payloads
- Retention/deletion tasks
- Audit event records

KPIs
- PII leakage incident rate
- Redaction precision/recall
- Policy violation prevention count
- Deletion SLA compliance
- Audit completeness score

Failure modes & mitigations
- Over-redaction harming utility: context-aware exceptions and tiered masking.
- Under-redaction risk: ensemble detectors + conservative defaults.
- Policy drift: versioned policy packs and periodic validation.
"""

from __future__ import annotations

__version__ = "2.3.0"

import time
import uuid

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .privacy import DataConsent, DataID, DataMinimization, DataRetention, PrivacyAuditability
from .privacy.modules import PrivacyFlowAnalyzer, PrivacyRiskEngine, ResidualExposureAnalyzer
from .privacy.utils.privacy_error import *
from .privacy.utils.privacy_helpers import combine_privacy_decisions
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Privacy Agent")
printer = PrettyPrinter()


@dataclass(slots=True)
class PrivacyExecutionReport:
    """Serializable request-scoped result emitted by ``PrivacyAgent``."""

    request_id: str
    decision: str
    summary: str
    policy_id: str
    policy_version: str
    subject_id: Optional[str]
    record_id: Optional[str]
    purpose: str
    source_context: str
    destination_context: str
    action: str
    audit_trail_ref: str
    decision_sources: Dict[str, str]
    stages: Dict[str, Dict[str, Any]]
    sanitized_payload: Dict[str, Any]
    retention: Dict[str, Any]
    residual_exposure: Dict[str, Any]
    flow_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    evidence_bundle: Optional[Dict[str, Any]]
    evidence_completeness: float
    shared_memory_keys: Sequence[str]
    duration_ms: float
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at_iso"] = datetime.fromtimestamp(
            self.created_at,
            tz=timezone.utc,
        ).isoformat()
        return payload


class PrivacyAgent(BaseAgent):
    """Privacy-domain orchestration boundary.

    The agent coordinates public privacy subsystem APIs only. It intentionally
    has no dependency on ``privacy_memory.py`` and does not access ``.memory``
    attributes exposed by subsystem objects.
    """

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        # ``config`` is forwarded to BaseAgent for the standard agent lifecycle.
        # PrivacyAgent-specific runtime configuration is deliberately loaded only
        # from agents_config.yaml below; no subsystem config is read here.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
        )
        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.privacy_config = get_config_section(
            "privacy_agent",
            config=self.config,
        )
        self._load_agent_configuration()
        self._validate_runtime_configuration()

        # ------------------------------------------------------------------
        # Subsystem composition
        # ------------------------------------------------------------------
        # Each subsystem owns its own configuration and internal state. The
        # agent neither injects subsystem configuration nor reaches into
        # privacy-memory internals.
        self.data_consent = DataConsent()
        self.data_id = DataID()
        self.data_min = DataMinimization()
        self.data_retention = DataRetention()
        self.private_audit = PrivacyAuditability()

        # Evidence-driven intelligence modules. These modules consume public
        # stage evidence and have no import dependency on PrivacyAgent.
        self.residual_exposure = ResidualExposureAnalyzer()
        self.flow_analyzer = PrivacyFlowAnalyzer()
        self.risk_engine = PrivacyRiskEngine()

        logger.info(
            "Privacy Agent initialized | enabled=%s | policy=%s@%s",
            self.enabled,
            self.default_policy_id,
            self.default_policy_version,
        )

    # ------------------------------------------------------------------
    # Agent configuration: agents_config.yaml only
    # ------------------------------------------------------------------
    def _load_agent_configuration(self) -> None:
        cfg = self.privacy_config
        if not cfg:
            raise ValueError("Missing 'privacy_agent' section in src/agents/base/configs/agents_config.yaml")

        self.enabled = self._require_bool(cfg, "enabled")
        self.fail_closed_on_subsystem_error = self._require_bool(cfg, "fail_closed_on_subsystem_error")
        self.include_evidence_bundle = self._require_bool(cfg, "include_evidence_bundle")
        self.include_sanitized_payload_on_block = self._require_bool(cfg, "include_sanitized_payload_on_block")

        self.default_policy_id = self._require_text(cfg, "default_policy_id")
        self.default_policy_version = self._require_text(cfg, "default_policy_version")
        self.default_source_context = self._require_text(cfg, "default_source_context")
        self.default_destination_context = self._require_text(cfg, "default_destination_context")
        self.default_action = self._require_text(cfg, "default_action")
        self.default_purpose = self._require_text(cfg, "default_purpose")
        self.default_retention_days = self._require_positive_int(cfg, "default_retention_days")

        self.post_minimization_stage = self._require_text(cfg, "post_minimization_stage")
        self.runtime_gate_stage = self._require_text(cfg, "runtime_gate_stage")
        self.evidence_bundle_name = self._require_text(cfg, "evidence_bundle_name")

        shared_cfg = self._require_mapping(cfg, "shared_memory")
        self.publish_to_shared_memory = self._require_bool(shared_cfg, "enabled")
        self.publish_notifications = self._require_bool(shared_cfg, "publish_notifications")
        self.include_stage_evidence_in_shared_memory = self._require_bool(shared_cfg, "include_stage_evidence")
        self.include_sanitized_payload_in_shared_memory = self._require_bool(shared_cfg, "include_sanitized_payload")
        self.include_evidence_bundle_in_shared_memory = self._require_bool(shared_cfg, "include_evidence_bundle")

        ttl = shared_cfg.get("ttl_seconds")
        if ttl is None:
            self.shared_ttl_seconds: Optional[int] = None
        else:
            try:
                self.shared_ttl_seconds = int(ttl)
            except (TypeError, ValueError) as exc:
                raise ValueError("privacy_agent.shared_memory.ttl_seconds must be an integer or null") from exc

        self.result_key_prefix = self._require_text(shared_cfg, "result_key_prefix")
        self.summary_key_prefix = self._require_text(shared_cfg, "summary_key_prefix")
        self.error_key_prefix = self._require_text(shared_cfg, "error_key_prefix")
        self.shared_event_channel = self._require_text(shared_cfg, "event_channel")

    @staticmethod
    def _require_mapping(mapping: Mapping[str, Any], key: str) -> Dict[str, Any]:
        value = mapping.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"privacy_agent.{key} must be a mapping")
        return dict(value)

    @staticmethod
    def _require_bool(mapping: Mapping[str, Any], key: str) -> bool:
        if key not in mapping or not isinstance(mapping[key], bool):
            raise ValueError(f"privacy_agent.{key} must be explicitly set to true/false")
        return bool(mapping[key])

    @staticmethod
    def _require_text(mapping: Mapping[str, Any], key: str) -> str:
        value = str(mapping.get(key) or "").strip()
        if not value:
            raise ValueError(f"privacy_agent.{key} must be a non-empty string")
        return value

    @staticmethod
    def _require_positive_int(mapping: Mapping[str, Any], key: str) -> int:
        raw_value = mapping.get(key)
        if raw_value is None:
            raise ValueError(f"privacy_agent.{key} must be an integer")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"privacy_agent.{key} must be an integer") from exc
        if value <= 0:
            raise ValueError(f"privacy_agent.{key} must be > 0")
        return value

    def _validate_runtime_configuration(self) -> None:
        if self.shared_ttl_seconds is not None and self.shared_ttl_seconds < 0:
            raise ValueError(
                "privacy_agent.shared_memory.ttl_seconds must be >= 0 or null"
            )

    # ------------------------------------------------------------------
    # Generic orchestration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _nonempty(value: Optional[str], fallback: str) -> str:
        normalized = str(value or "").strip()
        return normalized or fallback

    @staticmethod
    def _safe_mapping(
        value: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not value:
            return {}
        return dict(value)

    @staticmethod
    def _decision_from(
        stage: Mapping[str, Any],
        *,
        default: str = PrivacyDecision.ALLOW.value,
    ) -> str:
        raw = stage.get("decision")
        if raw is None:
            return default
        return str(getattr(raw, "value", raw)).strip().lower() or default

    @staticmethod
    def _unique_strings(values: Sequence[Any]) -> Tuple[str, ...]:
        result: List[str] = []
        seen = set()
        for raw in values:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return tuple(result)

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:16]}"

    def _normalize_error(
        self,
        exc: Exception,
        *,
        stage: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PrivacyError:
        if isinstance(exc, PrivacyError):
            return exc
        return normalize_privacy_exception(
            exc,
            stage=stage,
            context=self._safe_mapping(context),
        )

    @staticmethod
    def _normalize_lineage_events(
        lineage_events: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if lineage_events is None:
            return []
        if isinstance(lineage_events, (str, bytes, bytearray)) or not isinstance(
            lineage_events,
            Sequence,
        ):
            raise TypeError("lineage_events must be a sequence of mappings")

        normalized: List[Dict[str, Any]] = []
        for index, event in enumerate(lineage_events):
            if not isinstance(event, Mapping):
                raise TypeError(
                    f"lineage_events[{index}] must be a mapping, "
                    f"got {type(event).__name__}"
                )
            normalized.append(dict(event))
        return normalized

    @staticmethod
    def _flow_event_equivalent(
        event: Mapping[str, Any],
        *,
        request_id: str,
        source_context: str,
        destination_context: str,
        purpose: str,
        operation: str,
    ) -> bool:
        payload = event.get("payload")
        source = payload if isinstance(payload, Mapping) else event
        return (
            str(source.get("request_id") or event.get("request_id") or "")
            == request_id
            and str(source.get("source_context") or event.get("source_context") or "")
            == source_context
            and str(
                source.get("destination_context")
                or event.get("destination_context")
                or ""
            )
            == destination_context
            and str(source.get("purpose") or event.get("purpose") or "") == purpose
            and str(source.get("operation") or event.get("operation") or "")
            == operation
        )

    def _with_current_flow_event(
        self,
        lineage_events: Optional[Sequence[Mapping[str, Any]]],
        *,
        request_id: str,
        record_id: str,
        subject_id: str,
        source_context: str,
        destination_context: str,
        purpose: str,
        operation: str,
    ) -> List[Dict[str, Any]]:
        events = self._normalize_lineage_events(lineage_events)
        if not any(
            self._flow_event_equivalent(
                event,
                request_id=request_id,
                source_context=source_context,
                destination_context=destination_context,
                purpose=purpose,
                operation=operation,
            )
            for event in events
        ):
            events.append(
                {
                    "event_id": f"{request_id}:privacy_agent_flow",
                    "request_id": request_id,
                    "record_id": record_id,
                    "subject_id": subject_id,
                    "source_context": source_context,
                    "destination_context": destination_context,
                    "purpose": purpose,
                    "operation": operation,
                    "timestamp": time.time(),
                }
            )
        return events

    def _authorized_contexts_from_consent(
        self,
        consent_result: Mapping[str, Any],
    ) -> Tuple[str, ...]:
        candidates: List[Any] = []

        consent_snapshot = consent_result.get("consent")
        if isinstance(consent_snapshot, Mapping):
            candidates.extend(consent_snapshot.get("allowed_contexts") or [])

        purpose_validation = consent_result.get("purpose_validation")
        if isinstance(purpose_validation, Mapping):
            binding = purpose_validation.get("binding")
            if isinstance(binding, Mapping):
                candidates.extend(binding.get("allowed_contexts") or [])

        sharing_validation = consent_result.get("sharing_validation")
        if isinstance(sharing_validation, Mapping):
            sharing_consent = sharing_validation.get("consent")
            if isinstance(sharing_consent, Mapping):
                candidates.extend(sharing_consent.get("allowed_contexts") or [])
            sharing_purpose = sharing_validation.get("purpose_validation")
            if isinstance(sharing_purpose, Mapping):
                binding = sharing_purpose.get("binding")
                if isinstance(binding, Mapping):
                    candidates.extend(binding.get("allowed_contexts") or [])

        return self._unique_strings(candidates)

    @staticmethod
    def _summary_for_decision(decision: str) -> str:
        if decision == PrivacyDecision.BLOCK.value:
            return (
                "Privacy controls blocked ordinary downstream processing because "
                "at least one authoritative privacy stage reported a blocking condition."
            )
        if decision == PrivacyDecision.ESCALATE.value:
            return (
                "Privacy evaluation requires explicit review before broader "
                "processing, storage, or transfer continues."
            )
        if decision == PrivacyDecision.MODIFY.value:
            return (
                "Processing may continue only with the sanitized payload and "
                "the privacy modifications identified by the runtime controls."
            )
        return "Payload approved by the configured runtime privacy controls."

    # ------------------------------------------------------------------
    # SharedMemory publication (agent-level SharedMemory only)
    # ------------------------------------------------------------------
    def _publish(self, key: str, value: Mapping[str, Any]) -> None:
        if not self.publish_to_shared_memory or self.shared_memory is None:
            return
        payload = deepcopy(dict(value))
        if self.shared_ttl_seconds is None:
            self.shared_memory.set(key, payload)
        else:
            self.shared_memory.set(
                key,
                payload,
                ttl=self.shared_ttl_seconds,
            )

    def _publish_event(self, payload: Mapping[str, Any]) -> None:
        if self.publish_notifications and self.shared_memory is not None:
            self.shared_memory.publish(
                self.shared_event_channel,
                deepcopy(dict(payload)),
            )

    def _shared_result_projection(
        self,
        report: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return a bounded projection suitable for SLAI SharedMemory.

        Detailed detection evidence can contain previews or path-level metadata.
        It is therefore excluded by default and only included when explicitly
        enabled in the agent configuration.
        """

        retention = report.get("retention") or {}
        residual = report.get("residual_exposure") or {}
        flow = report.get("flow_analysis") or {}
        risk = report.get("risk_assessment") or {}

        result = {
            "request_id": report.get("request_id"),
            "decision": report.get("decision"),
            "summary": report.get("summary"),
            "policy_id": report.get("policy_id"),
            "policy_version": report.get("policy_version"),
            "purpose": report.get("purpose"),
            "source_context": report.get("source_context"),
            "destination_context": report.get("destination_context"),
            "action": report.get("action"),
            "audit_trail_ref": report.get("audit_trail_ref"),
            "decision_sources": deepcopy(report.get("decision_sources") or {}),
            "retention": {
                "decision": retention.get("decision"),
                "status": retention.get("status"),
                "is_due": retention.get("is_due"),
                "is_overdue": retention.get("is_overdue"),
                "legal_hold": retention.get("legal_hold"),
                "deletion_sla_breached": retention.get("deletion_sla_breached"),
            },
            "residual_exposure": {
                "decision": residual.get("decision"),
                "status": residual.get("status"),
                "post_sensitivity": residual.get("post_sensitivity"),
                "exposure_reduction": residual.get("exposure_reduction"),
                "under_redaction": residual.get("under_redaction"),
                "over_redaction": residual.get("over_redaction"),
                "residual_risk_score": residual.get("residual_risk_score"),
            },
            "flow_analysis": {
                "decision": flow.get("decision"),
                "status": flow.get("status"),
                "event_count": flow.get("event_count"),
                "unique_destination_count": flow.get("unique_destination_count"),
                "cycle_detected": flow.get("cycle_detected"),
                "post_deletion_processing_count": flow.get(
                    "post_deletion_processing_count"
                ),
                "flow_risk_score": flow.get("flow_risk_score"),
            },
            "risk_assessment": {
                "decision": risk.get("decision"),
                "authoritative_decision": risk.get("authoritative_decision"),
                "score_decision": risk.get("score_decision"),
                "overall_risk_score": risk.get("overall_risk_score"),
                "confidence": risk.get("confidence"),
                "evidence_completeness": risk.get("evidence_completeness"),
                "missing_evidence_sources": deepcopy(
                    risk.get("missing_evidence_sources") or []
                ),
            },
            "evidence_completeness": report.get("evidence_completeness"),
            "duration_ms": report.get("duration_ms"),
            "created_at": report.get("created_at"),
            "created_at_iso": report.get("created_at_iso"),
        }
        if self.include_stage_evidence_in_shared_memory:
            result["stages"] = deepcopy(report.get("stages") or {})
        if self.include_sanitized_payload_in_shared_memory:
            result["sanitized_payload"] = deepcopy(
                report.get("sanitized_payload") or {}
            )
        if self.include_evidence_bundle_in_shared_memory:
            result["evidence_bundle"] = deepcopy(report.get("evidence_bundle"))
        return result

    # ------------------------------------------------------------------
    # Public consent/purpose facade
    # ------------------------------------------------------------------
    def register_consent(
        self,
        *,
        subject_id: str,
        purpose: str,
        status: str,
        artifact_ref: str,
        legal_basis: Optional[str] = None,
        allowed_contexts: Optional[Sequence[str]] = None,
        allowed_processors: Optional[Sequence[str]] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.data_consent.register_consent_artifact(
            subject_id=subject_id,
            purpose=purpose,
            status=status,
            artifact_ref=artifact_ref,
            legal_basis=legal_basis,
            allowed_contexts=allowed_contexts,
            allowed_processors=allowed_processors,
            policy_id=policy_id or self.default_policy_id,
            policy_version=policy_version or self.default_policy_version,
            metadata=self._safe_mapping(metadata),
        )

    def bind_purpose(
        self,
        *,
        subject_id: str,
        purpose: str,
        source_context: str,
        allowed_contexts: Optional[Sequence[str]] = None,
        allowed_actions: Optional[Sequence[str]] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.data_consent.bind_purpose(
            subject_id=subject_id,
            purpose=purpose,
            source_context=source_context,
            allowed_contexts=allowed_contexts,
            allowed_actions=allowed_actions,
            policy_id=policy_id or self.default_policy_id,
            policy_version=policy_version or self.default_policy_version,
            metadata=self._safe_mapping(metadata),
        )

    # ------------------------------------------------------------------
    # Main privacy runtime
    # ------------------------------------------------------------------
    def evaluate_privacy(
        self,
        payload: Mapping[str, Any],
        *,
        request_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        record_id: Optional[str] = None,
        purpose: Optional[str] = None,
        action: Optional[str] = None,
        source_context: Optional[str] = None,
        destination_context: Optional[str] = None,
        required_processor: Optional[str] = None,
        allowed_fields: Optional[Sequence[str]] = None,
        required_fields: Optional[Sequence[str]] = None,
        sensitive_fields: Optional[Sequence[str]] = None,
        field_strategies: Optional[Mapping[str, Any]] = None,
        retention_days: Optional[int] = None,
        record_tags: Optional[Sequence[str]] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        lineage_events: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")

        if not self.enabled:
            req = self._nonempty(request_id, self._new_id("privacy_req"))
            return {
                "request_id": req,
                "decision": PrivacyDecision.ALLOW.value,
                "summary": "Privacy Agent disabled; passing through without mutation.",
                "sanitized_payload": dict(payload),
                "decision_sources": {},
                "stages": {},
            }

        started_at = time.time()
        req_id = self._nonempty(request_id, self._new_id("privacy_req"))
        rec_id = self._nonempty(record_id, self._new_id("privacy_rec"))
        subject = self._nonempty(subject_id, "anonymous_subject")
        use_purpose = self._nonempty(purpose, self.default_purpose)
        use_action = self._nonempty(action, self.default_action)
        source = self._nonempty(source_context, self.default_source_context)
        destination = self._nonempty(
            destination_context,
            self.default_destination_context,
        )
        resolved_policy_id = self._nonempty(policy_id, self.default_policy_id)
        resolved_policy_version = self._nonempty(
            policy_version,
            self.default_policy_version,
        )
        audit_ref = self._new_id("audit")

        use_retention_days = (
            self.default_retention_days
            if retention_days is None
            else int(retention_days)
        )
        if use_retention_days <= 0:
            raise ValueError("retention_days must be > 0")

        runtime_context = self._safe_mapping(context)
        stages: Dict[str, Dict[str, Any]] = {}
        shared_keys: List[str] = []
        retention_snapshot: Dict[str, Any] = {}
        residual_assessment: Dict[str, Any] = {}
        flow_assessment: Dict[str, Any] = {}
        risk_assessment: Dict[str, Any] = {}
        evidence_bundle: Optional[Dict[str, Any]] = None

        try:
            # 1. Pre-control identification.
            identification = self.data_id.identify_entities(
                payload,
                request_id=req_id,
                record_id=rec_id,
                subject_id=subject,
                source_context=source,
                purpose=use_purpose,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                audit_trail_ref=audit_ref,
                context=runtime_context,
            )
            stages["identification"] = identification

            # 2. Consent / purpose / transfer authorization.
            consent = self.data_consent.evaluate_request(
                request_id=req_id,
                subject_id=subject,
                purpose=use_purpose,
                source_context=source,
                destination_context=destination,
                action=use_action,
                required_processor=required_processor,
                record_id=rec_id,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                sensitivity_score=identification.get("sensitivity_score"),
                detected_entities=identification.get("detected_entities"),
                audit_trail_ref=audit_ref,
                context=runtime_context,
            )
            stages["consent"] = consent

            # 3. Least-data-required transformation.
            minimization = self.data_min.minimize_payload(
                payload,
                purpose=use_purpose,
                request_id=req_id,
                subject_id=subject,
                record_id=rec_id,
                source_context=source,
                destination_context=destination,
                allowed_fields=allowed_fields,
                required_fields=required_fields,
                sensitive_fields=sensitive_fields,
                field_strategies=field_strategies,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                audit_trail_ref=audit_ref,
                context={
                    "upstream_sensitivity_score": identification.get(
                        "sensitivity_score"
                    ),
                    "upstream_entity_count": identification.get("entity_count"),
                    **runtime_context,
                },
            )
            stages["minimization"] = minimization

            sanitized_payload = minimization.get("sanitized_payload")
            if not isinstance(sanitized_payload, Mapping):
                raise TypeError(
                    "DataMinimization returned a non-mapping sanitized_payload"
                )
            sanitized_payload = dict(sanitized_payload)

            # 4. Verify the transformed payload with an independent second pass
            # through the *existing* DataID detector. No duplicate detector is
            # implemented by the agent or residual analyzer.
            post_identification = self.data_id.identify_entities(
                sanitized_payload,
                request_id=self._new_id("privacy_post_scan"),
                record_id=rec_id,
                subject_id=subject,
                source_context=destination,
                purpose=use_purpose,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                stage=self.post_minimization_stage,
                audit_trail_ref=audit_ref,
                context={
                    "analysis_phase": "post_minimization",
                    "parent_request_id": req_id,
                    **runtime_context,
                },
            )
            stages["post_minimization_identification"] = post_identification

            # 5. Compare before/after evidence and verify control effectiveness.
            residual_assessment = self.residual_exposure.assess(
                before_identification=identification,
                after_identification=post_identification,
                minimization=minimization,
                request_id=req_id,
                required_fields=required_fields,
                context={
                    "record_id": rec_id,
                    "purpose": use_purpose,
                    "source_context": source,
                    "destination_context": destination,
                },
            )
            stages["residual_exposure"] = residual_assessment

            # 6. Retention obligation and runtime enforcement.
            obligation = self.data_retention.create_retention_obligation(
                record_id=rec_id,
                subject_id=subject,
                request_id=req_id,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                retention_days=use_retention_days,
                metadata={
                    "source_context": source,
                    "destination_context": destination,
                    "action": use_action,
                    **runtime_context,
                },
                record_tags=list(record_tags or []),
                audit_trail_ref=audit_ref,
            )
            stages["retention_obligation"] = obligation

            retention_snapshot = self.data_retention.enforce_retention(
                record_id=rec_id,
                request_id=req_id,
                audit_trail_ref=audit_ref,
            )
            stages["retention_enforcement"] = retention_snapshot

            # 7. Privacy flow analysis. Historical lineage is accepted only as
            # explicit request evidence. The agent never reads privacy_memory.
            flow_events = self._with_current_flow_event(
                lineage_events,
                request_id=req_id,
                record_id=rec_id,
                subject_id=subject,
                source_context=source,
                destination_context=destination,
                purpose=use_purpose,
                operation=use_action,
            )
            authorized_contexts = self._authorized_contexts_from_consent(consent)
            flow_assessment = self.flow_analyzer.analyze(
                lineage_events=flow_events,
                request_id=req_id,
                record_id=rec_id,
                subject_id=subject,
                authorized_contexts=authorized_contexts,
                retention=retention_snapshot,
                context={
                    "policy_id": resolved_policy_id,
                    "policy_version": resolved_policy_version,
                    "current_event_included": True,
                    "historical_event_count": max(len(flow_events) - 1, 0),
                },
            )
            stages["flow_analysis"] = flow_assessment

            # 8. Explainable privacy-specific risk aggregation.
            risk_assessment = self.risk_engine.assess(
                identification=identification,
                consent=consent,
                minimization=minimization,
                residual_exposure=residual_assessment,
                retention=retention_snapshot,
                flow_analysis=flow_assessment,
                request_id=req_id,
                context={
                    "record_id": rec_id,
                    "purpose": use_purpose,
                    "policy_id": resolved_policy_id,
                    "policy_version": resolved_policy_version,
                },
            )
            stages["risk_assessment"] = risk_assessment

            # 9. Agent-level deterministic fusion is an orchestration invariant.
            # It does not depend on subsystem risk weights and therefore cannot
            # weaken an authoritative block/escalation produced upstream.
            decision_sources = {
                "identification": self._decision_from(identification),
                "consent": self._decision_from(consent),
                "minimization": self._decision_from(minimization),
                "post_minimization_identification": self._decision_from(
                    post_identification
                ),
                "residual_exposure": self._decision_from(residual_assessment),
                "retention": self._decision_from(retention_snapshot),
                "flow": self._decision_from(flow_assessment),
                "risk": self._decision_from(risk_assessment),
            }
            final_decision = combine_privacy_decisions(
                *decision_sources.values(),
                default=PrivacyDecision.ALLOW.value,
            )
            summary = self._summary_for_decision(final_decision)

            # 10. Record the actual final decision, not a preliminary checkpoint.
            checkpoint = self.private_audit.record_decision_checkpoint(
                request_id=req_id,
                stage=self.runtime_gate_stage,
                summary=summary,
                subject_id=subject,
                record_id=rec_id,
                purpose=use_purpose,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                context={
                    "final_decision": final_decision,
                    "decision_sources": decision_sources,
                    "pre_sensitivity": identification.get("sensitivity_score"),
                    "post_sensitivity": post_identification.get(
                        "sensitivity_score"
                    ),
                    "residual_risk_score": residual_assessment.get(
                        "residual_risk_score"
                    ),
                    "flow_risk_score": flow_assessment.get("flow_risk_score"),
                    "overall_risk_score": risk_assessment.get(
                        "overall_risk_score"
                    ),
                    "risk_confidence": risk_assessment.get("confidence"),
                    "evidence_completeness": risk_assessment.get(
                        "evidence_completeness"
                    ),
                    "retention_status": retention_snapshot.get("status"),
                },
            )
            stages["audit_checkpoint"] = checkpoint

            if self.include_evidence_bundle:
                evidence_bundle = self.private_audit.generate_audit_evidence_bundle(
                    request_id=req_id,
                    record_id=rec_id,
                    subject_id=subject,
                    purpose=use_purpose,
                    policy_id=resolved_policy_id,
                    policy_version=resolved_policy_version,
                    bundle_name=self.evidence_bundle_name,
                    context={
                        "agent": self.name,
                        "source_context": source,
                        "destination_context": destination,
                        "final_decision": final_decision,
                    },
                )
                stages["evidence_bundle"] = {
                    "bundle_id": evidence_bundle.get("bundle_id"),
                    "fingerprint": evidence_bundle.get("fingerprint"),
                }

            output_payload = sanitized_payload
            if (
                final_decision == PrivacyDecision.BLOCK.value
                and not self.include_sanitized_payload_on_block
            ):
                output_payload = {}

            report = PrivacyExecutionReport(
                request_id=req_id,
                decision=final_decision,
                summary=summary,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                subject_id=subject,
                record_id=rec_id,
                purpose=use_purpose,
                source_context=source,
                destination_context=destination,
                action=use_action,
                audit_trail_ref=audit_ref,
                decision_sources=decision_sources,
                stages=stages,
                sanitized_payload=output_payload,
                retention=retention_snapshot,
                residual_exposure=residual_assessment,
                flow_analysis=flow_assessment,
                risk_assessment=risk_assessment,
                evidence_bundle=evidence_bundle,
                evidence_completeness=float(
                    risk_assessment.get("evidence_completeness") or 0.0
                ),
                shared_memory_keys=[],
                duration_ms=round((time.time() - started_at) * 1000.0, 3),
                created_at=time.time(),
            )
            report_payload = report.to_dict()

            result_key = f"{self.result_key_prefix}.{req_id}"
            summary_key = f"{self.summary_key_prefix}.{req_id}"

            self._publish(
                result_key,
                self._shared_result_projection(report_payload),
            )
            self._publish(
                summary_key,
                {
                    "request_id": req_id,
                    "decision": final_decision,
                    "summary": summary,
                    "decision_sources": decision_sources,
                    "pre_sensitivity_score": identification.get(
                        "sensitivity_score"
                    ),
                    "post_sensitivity_score": post_identification.get(
                        "sensitivity_score"
                    ),
                    "masked_field_count": minimization.get(
                        "masked_field_count",
                        0,
                    ),
                    "removed_field_count": minimization.get(
                        "removed_field_count",
                        0,
                    ),
                    "residual_risk_score": residual_assessment.get(
                        "residual_risk_score"
                    ),
                    "flow_risk_score": flow_assessment.get("flow_risk_score"),
                    "overall_risk_score": risk_assessment.get(
                        "overall_risk_score"
                    ),
                    "risk_confidence": risk_assessment.get("confidence"),
                    "evidence_completeness": risk_assessment.get(
                        "evidence_completeness"
                    ),
                    "retention_status": retention_snapshot.get("status"),
                    "audit_trail_ref": audit_ref,
                    "policy_id": resolved_policy_id,
                    "policy_version": resolved_policy_version,
                    "created_at": report.created_at,
                },
            )

            if self.publish_to_shared_memory and self.shared_memory is not None:
                shared_keys.extend([result_key, summary_key])
            report_payload["shared_memory_keys"] = shared_keys

            self._publish_event(
                {
                    "event_type": "privacy.runtime.decision",
                    "request_id": req_id,
                    "decision": final_decision,
                    "policy_id": resolved_policy_id,
                    "policy_version": resolved_policy_version,
                    "audit_trail_ref": audit_ref,
                    "overall_risk_score": risk_assessment.get(
                        "overall_risk_score"
                    ),
                    "risk_confidence": risk_assessment.get("confidence"),
                    "timestamp": time.time(),
                }
            )

            return report_payload

        except Exception as exc:
            normalized = self._normalize_error(
                exc,
                stage="privacy_agent.evaluate_privacy",
                context={
                    "request_id": req_id,
                    "record_id": rec_id,
                    "subject_id": subject,
                    "purpose": use_purpose,
                },
            )

            error_decision = (
                PrivacyDecision.BLOCK.value
                if self.fail_closed_on_subsystem_error
                else PrivacyDecision.ESCALATE.value
            )
            assert normalized.severity is not None
            error_payload = {
                "request_id": req_id,
                "record_id": rec_id,
                "subject_id": subject,
                "purpose": use_purpose,
                "policy_id": resolved_policy_id,
                "policy_version": resolved_policy_version,
                "decision": error_decision,
                "message": normalized.message,
                "error_type": normalized.error_type.value,
                "error_code": normalized.error_code,
                "severity": normalized.severity.value,
                "retryable": normalized.retryable,
                "completed_stages": list(stages),
                "audit_trail_ref": audit_ref,
                "timestamp": time.time(),
            }

            error_key = f"{self.error_key_prefix}.{req_id}"
            self._publish(error_key, error_payload)
            assert normalized.severity is not None
            self._publish_event(
                {
                    "event_type": "privacy.runtime.error",
                    "request_id": req_id,
                    "decision": error_decision,
                    "error_type": normalized.error_type.value,
                    "error_code": normalized.error_code,
                    "severity": normalized.severity.value,
                    "policy_id": resolved_policy_id,
                    "policy_version": resolved_policy_version,
                    "timestamp": time.time(),
                }
            )

            if self.fail_closed_on_subsystem_error:
                raise normalized from exc
            return error_payload

    # ------------------------------------------------------------------
    # BaseAgent task entry point
    # ------------------------------------------------------------------
    def perform_task_privacy(
        self,
        input_data: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_context = self._safe_mapping(context)

        if isinstance(input_data, Mapping):
            has_explicit_payload = "payload" in input_data
            payload_value = input_data.get("payload")

            if has_explicit_payload:
                if isinstance(payload_value, Mapping):
                    payload = dict(payload_value)
                else:
                    payload = {"content": payload_value}
                runtime_context = {
                    **normalized_context,
                    **self._safe_mapping(input_data.get("context")),
                }
                lineage_events = input_data.get("lineage_events")
            else:
                # Preserve backward compatibility: a mapping without an explicit
                # envelope is interpreted as the payload itself.
                payload = dict(input_data)
                runtime_context = normalized_context
                lineage_events = None

            return self.evaluate_privacy(
                payload,
                request_id=input_data.get("request_id") if has_explicit_payload else None,
                subject_id=input_data.get("subject_id") if has_explicit_payload else None,
                record_id=input_data.get("record_id") if has_explicit_payload else None,
                purpose=input_data.get("purpose") if has_explicit_payload else None,
                action=input_data.get("action") if has_explicit_payload else None,
                source_context=(
                    input_data.get("source_context") if has_explicit_payload else None
                ),
                destination_context=(
                    input_data.get("destination_context")
                    if has_explicit_payload
                    else None
                ),
                required_processor=(
                    input_data.get("required_processor") if has_explicit_payload else None
                ),
                allowed_fields=(
                    input_data.get("allowed_fields") if has_explicit_payload else None
                ),
                required_fields=(
                    input_data.get("required_fields") if has_explicit_payload else None
                ),
                sensitive_fields=(
                    input_data.get("sensitive_fields") if has_explicit_payload else None
                ),
                field_strategies=(
                    input_data.get("field_strategies") if has_explicit_payload else None
                ),
                retention_days=(
                    input_data.get("retention_days") if has_explicit_payload else None
                ),
                record_tags=(
                    input_data.get("record_tags") if has_explicit_payload else None
                ),
                policy_id=input_data.get("policy_id") if has_explicit_payload else None,
                policy_version=(
                    input_data.get("policy_version") if has_explicit_payload else None
                ),
                lineage_events=lineage_events,
                context=runtime_context,
            )

        return self.evaluate_privacy(
            {"content": input_data},
            context=normalized_context,
        )


if __name__ == "__main__":
    print("\n=== Running Privacy agent ===\n")
    printer.status("TEST", "Privacy agent initialized", "info")

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    agent = PrivacyAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
    )

    if not isinstance(agent, PrivacyAgent):
        printer.status("INIT", "PrivacyAgent failed to construct", "error")
        raise SystemExit(1)

    printer.status(
        "INIT",
        f"PrivacyAgent ready (enabled={agent.enabled}, "
        f"policy={agent.default_policy_id}@{agent.default_policy_version}, "
        f"fail_closed={agent.fail_closed_on_subsystem_error})",
        "success",
    )

    # -----------------------------------------------------------------
    # Scenario A: consent + purpose binding + full runtime evaluation
    # -----------------------------------------------------------------
    printer.section_header("Scenario A: fully authorized runtime request")

    subject_id = "subject-smoke-001"
    record_id = "record-smoke-001"
    request_id = "req-smoke-001"
    purpose = "support_resolution"
    source_ctx = "chat_runtime"
    dest_ctx = "ticketing_connector"
    action = "share"

    consent_record = agent.register_consent(
        subject_id=subject_id,
        purpose=purpose,
        status="granted",
        artifact_ref="consent-artifact-smoke-001",
        legal_basis="explicit_consent",
        allowed_contexts=[source_ctx, dest_ctx, "privacy_review"],
        allowed_processors=["ticketing_connector", "execution_agent"],
        metadata={"channel": "chat", "collector": "smoke_test"},
    )
    printer.status(
        "A",
        f"consent registered (artifact={consent_record.get('artifact_ref')})",
        "success",
    )

    binding = agent.bind_purpose(
        subject_id=subject_id,
        purpose=purpose,
        source_context=source_ctx,
        allowed_contexts=[dest_ctx, "privacy_review"],
        allowed_actions=["read", "redact", "share", "respond"],
        metadata={"scope": "support_only"},
    )
    printer.status(
        "A",
        f"purpose bound (contexts={binding.get('allowed_contexts')})",
        "success",
    )

    payload_a = {
        "ticket_id": "T-SMOKE-001",
        "issue_summary": "Customer cannot complete login after password reset.",
        "email": "jane.doe@example.com",
        "phone": "+1 202-555-0141",
        "full_name": "Jane Doe",
        "account_id": "ACC-8827139",
        "metadata": {
            "browser": "Firefox",
            "ip_address": "203.0.113.5",
        },
    }

    result_a = agent.evaluate_privacy(
        payload_a,
        request_id=request_id,
        subject_id=subject_id,
        record_id=record_id,
        purpose=purpose,
        action=action,
        source_context=source_ctx,
        destination_context=dest_ctx,
        required_processor="ticketing_connector",
        allowed_fields=[
            "ticket_id",
            "issue_summary",
            "email",
            "phone",
            "full_name",
            "metadata.browser",
        ],
        required_fields=["ticket_id", "issue_summary"],
        sensitive_fields=["metadata.ip_address"],
        field_strategies={
            "email": "partial_mask",
            "phone": "last4",
            "full_name": "tokenize",
        },
        policy_id="smoke-policy",
        policy_version="2026.04",
        context={"channel": "chat", "entrypoint": "evaluate_privacy"},
    )
    printer.status(
        "A",
        f"decision={result_a['decision']} "
        f"risk={result_a.get('risk_assessment', {}).get('overall_risk_score')} "
        f"audit={result_a.get('audit_trail_ref')}",
        "success",
    )

    # -----------------------------------------------------------------
    # Scenario B: same payload, no consent -> must not silently pass
    # -----------------------------------------------------------------
    printer.section_header("Scenario B: unauthenticated request must not pass")

    try:
        result_b = agent.evaluate_privacy(
            payload_a,
            request_id="req-smoke-002",
            subject_id="subject-without-consent",
            record_id="record-smoke-002",
            purpose=purpose,
            action=action,
            source_context=source_ctx,
            destination_context=dest_ctx,
            required_processor="ticketing_connector",
            policy_id="smoke-policy",
            policy_version="2026.04",
        )
        if result_b.get("decision") == "allow":
            printer.status("B", "unexpected allow without consent", "error")
        else:
            printer.status(
                "B",
                f"decision={result_b['decision']} (correctly restrictive)",
                "success",
            )
    except PrivacyError as exc:
        printer.status(
            "B",
            f"blocked by fail-closed policy: {exc.error_code}",
            "success",
        )

    # -----------------------------------------------------------------
    # Scenario C: explicit lineage evidence spanning multiple contexts
    # -----------------------------------------------------------------
    printer.section_header("Scenario C: lineage-aware evaluation")

    agent.register_consent(
        subject_id="subject-smoke-002",
        purpose=purpose,
        status="granted",
        artifact_ref="consent-artifact-smoke-002",
        legal_basis="explicit_consent",
        allowed_contexts=[source_ctx, dest_ctx, "analytics_workspace", "privacy_review"],
        allowed_processors=["ticketing_connector", "analytics_connector"],
    )
    agent.bind_purpose(
        subject_id="subject-smoke-002",
        purpose=purpose,
        source_context=source_ctx,
        allowed_contexts=[dest_ctx, "analytics_workspace", "privacy_review"],
        allowed_actions=["read", "share", "redact", "respond"],
    )

    lineage_events = [
        {
            "event_id": "lineage-001",
            "request_id": "req-smoke-003",
            "record_id": "record-smoke-003",
            "subject_id": "subject-smoke-002",
            "source_context": source_ctx,
            "destination_context": dest_ctx,
            "purpose": purpose,
            "operation": action,
            "timestamp": time.time() - 60,
        },
        {
            "event_id": "lineage-002",
            "request_id": "req-smoke-003",
            "record_id": "record-smoke-003",
            "subject_id": "subject-smoke-002",
            "source_context": dest_ctx,
            "destination_context": "analytics_workspace",
            "purpose": purpose,
            "operation": "share",
            "timestamp": time.time() - 30,
        },
    ]

    result_c = agent.evaluate_privacy(
        payload_a,
        request_id="req-smoke-003",
        subject_id="subject-smoke-002",
        record_id="record-smoke-003",
        purpose=purpose,
        action=action,
        source_context=source_ctx,
        destination_context=dest_ctx,
        required_processor="ticketing_connector",
        allowed_fields=["ticket_id", "issue_summary", "email"],
        required_fields=["ticket_id", "issue_summary"],
        field_strategies={"email": "partial_mask"},
        policy_id="smoke-policy",
        policy_version="2026.04",
        lineage_events=lineage_events,
    )
    flow = result_c.get("flow_analysis", {})
    printer.status(
        "C",
        f"decision={result_c['decision']} "
        f"flow_risk={flow.get('flow_risk_score')} "
        f"cross_context={flow.get('cross_context_transition_count')}",
        "success",
    )

    # -----------------------------------------------------------------
    # Scenario D: perform_task_privacy with an explicit envelope
    # -----------------------------------------------------------------
    printer.section_header("Scenario D: BaseAgent entry (explicit envelope)")

    envelope = {
        "payload": {
            "ticket_id": "T-SMOKE-004",
            "issue_summary": "Password reset link expired.",
            "email": "user@example.com",
        },
        "request_id": "req-smoke-004",
        "subject_id": subject_id,
        "record_id": "record-smoke-004",
        "purpose": purpose,
        "action": action,
        "source_context": source_ctx,
        "destination_context": dest_ctx,
        "required_processor": "ticketing_connector",
        "allowed_fields": ["ticket_id", "issue_summary", "email"],
        "required_fields": ["ticket_id", "issue_summary"],
        "field_strategies": {"email": "partial_mask"},
        "context": {"channel": "chat", "entrypoint": "envelope"},
    }
    result_d = agent.perform_task_privacy(envelope)
    printer.status(
        "D",
        f"decision={result_d['decision']} "
        f"sanitized_keys={sorted(result_d.get('sanitized_payload', {}).keys())}",
        "success",
    )

    # -----------------------------------------------------------------
    # Scenario E: perform_task_privacy with a raw mapping (back-compat)
    # -----------------------------------------------------------------
    printer.section_header("Scenario E: BaseAgent entry (raw mapping)")

    # The raw-mapping path carries no identity/context/action, so the agent
    # falls back to the configured defaults:
    #   subject_id          -> "anonymous_subject"
    #   purpose             -> agent.default_purpose      ("general_processing")
    #   action              -> agent.default_action       ("process")
    #   source/destination  -> agent.default_*_context    ("runtime")
    #
    # data_consent.require_purpose_binding defaults to True, so a bare consent
    # artifact is NOT enough — a purpose binding is mandatory. The binding must
    # also allow the default action, because enforce_action_allowlist defaults
    # to True as well.
    agent.register_consent(
        subject_id="anonymous_subject",
        purpose=agent.default_purpose,
        status="granted",
        artifact_ref="consent-artifact-anon",
        legal_basis="legitimate_interest",
        allowed_contexts=[
            agent.default_source_context,
            agent.default_destination_context,
        ],
        allowed_processors=["execution_agent"],
    )
    agent.bind_purpose(
        subject_id="anonymous_subject",
        purpose=agent.default_purpose,
        source_context=agent.default_source_context,
        allowed_contexts=[agent.default_destination_context],
        allowed_actions=["read", "process", "share", "respond"],
        metadata={"scope": "backward-compatible default path"},
    )

    result_e = agent.perform_task_privacy(
        {"note": "no PII here at all"},
        context={"source_context": agent.default_source_context},
    )
    printer.status(
        "E",
        f"decision={result_e['decision']} (backward-compatible path)",
        "success",
    )

    # -----------------------------------------------------------------
    # Scenario F: SLAI SharedMemory publication sanity check
    # -----------------------------------------------------------------
    printer.section_header("Scenario F: SLAI SharedMemory publication")

    if agent.publish_to_shared_memory and shared_memory is not None:
        result_key = f"{agent.result_key_prefix}.{result_a['request_id']}"
        summary_key = f"{agent.summary_key_prefix}.{result_a['request_id']}"
        cached_result = shared_memory.get(result_key)
        cached_summary = shared_memory.get(summary_key)
        ok = bool(cached_result) and bool(cached_summary)
        printer.status(
            "F",
            f"result={'present' if cached_result else 'missing'} "
            f"summary={'present' if cached_summary else 'missing'}",
            "success" if ok else "warning",
        )
    else:
        printer.status("F", "shared-memory publication disabled", "info")

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
    printer.section_header("Smoke test summary")

    printer.table(
        ["scenario", "decision", "audit_ref"],
        [
            ["A: authorized pipeline", result_a["decision"], str(result_a.get("audit_trail_ref", ""))],
            ["C: lineage-aware", result_c["decision"], str(result_c.get("audit_trail_ref", ""))],
            ["D: task entry (envelope)", result_d["decision"], str(result_d.get("audit_trail_ref", ""))],
            ["E: task entry (raw)", result_e["decision"], str(result_e.get("audit_trail_ref", ""))],
        ],
    )

    print("\n== Task run successfully ==\n")
