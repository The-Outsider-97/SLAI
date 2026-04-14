"""
The Privacy Agent enforces privacy-by-design across data collection, processing, storage, sharing, and deletion.
It operationalizes PII/PHI controls, retention rules, and purpose constraints as runtime decisions.

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

__version__ = "2.1.0"

import time
import uuid

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .base_agent import BaseAgent
from .base.utils.main_config_loader import load_global_config, get_config_section
from .privacy import DataID, DataMinimization, DataRetention, PrivacyAuditability, DataConsent
from .privacy.utils.privacy_error import PrivacyDecision, PrivacyError, normalize_privacy_exception
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Privacy Agent")
printer = PrettyPrinter


@dataclass(slots=True)
class PrivacyExecutionReport:
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
    stages: Dict[str, Dict[str, Any]]
    sanitized_payload: Dict[str, Any]
    retention: Dict[str, Any]
    evidence_bundle: Optional[Dict[str, Any]]
    shared_memory_keys: Sequence[str]
    duration_ms: float
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return payload


class PrivacyAgent(BaseAgent):
    """Production privacy orchestration layer for identification, consent, minimization, retention, and auditability."""

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.privacy_config = get_config_section("privacy_agent") or {}
        if config:
            self.privacy_config.update(dict(config))

        self.enabled = bool(self.privacy_config.get("enabled", True))
        self.fail_closed_on_subsystem_error = bool(self.privacy_config.get("fail_closed_on_subsystem_error", True))
        self.include_evidence_bundle = bool(self.privacy_config.get("include_evidence_bundle", True))
        self.publish_to_shared_memory = bool(self.privacy_config.get("publish_to_shared_memory", True))
        self.publish_notifications = bool(self.privacy_config.get("publish_notifications", True))

        self.default_policy_id = str(self.privacy_config.get("default_policy_id", "privacy-runtime-policy")).strip()
        self.default_policy_version = str(self.privacy_config.get("default_policy_version", "v1")).strip()
        self.default_source_context = str(self.privacy_config.get("default_source_context", "runtime")).strip()
        self.default_destination_context = str(
            self.privacy_config.get("default_destination_context", "runtime")
        ).strip()
        self.default_action = str(self.privacy_config.get("default_action", "process")).strip()
        self.default_purpose = str(self.privacy_config.get("default_purpose", "general_processing")).strip()
        self.default_retention_days = int(self.privacy_config.get("default_retention_days", 30))

        self.shared_key_prefix = str(self.privacy_config.get("shared_key_prefix", "privacy_agent")).strip() or "privacy_agent"
        self.shared_event_channel = str(self.privacy_config.get("shared_event_channel", "privacy.events")).strip() or "privacy.events"
        self.shared_ttl_seconds = self.privacy_config.get("shared_ttl_seconds", 86400)

        self._validate_runtime_configuration()

        # Subsystem initialization
        self.data_consent = DataConsent()
        self.data_id = DataID()
        self.data_min = DataMinimization()
        self.data_retention = DataRetention()
        self.private_audit = PrivacyAuditability()

        logger.info("Privacy Agent initialized | enabled=%s | policy=%s@%s", self.enabled, self.default_policy_id, self.default_policy_version)

    def _validate_runtime_configuration(self) -> None:
        if self.default_retention_days <= 0:
            raise ValueError("privacy_agent.default_retention_days must be > 0")
        if self.shared_ttl_seconds is not None and int(self.shared_ttl_seconds) < 0:
            raise ValueError("privacy_agent.shared_ttl_seconds must be >= 0")

    @staticmethod
    def _nonempty(value: Optional[str], fallback: str) -> str:
        normalized = str(value or "").strip()
        return normalized or fallback

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:16]}"

    @staticmethod
    def _safe_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not value:
            return {}
        return dict(value)

    def _normalize_error(self, exc: Exception, *, stage: str, context: Optional[Mapping[str, Any]] = None) -> PrivacyError:
        if isinstance(exc, PrivacyError):
            return exc
        return normalize_privacy_exception(exc, stage=stage, context=self._safe_mapping(context))

    def _publish(self, key: str, value: Mapping[str, Any]) -> None:
        if not self.publish_to_shared_memory or self.shared_memory is None:
            return
        payload = deepcopy(dict(value))
        if self.shared_ttl_seconds is None:
            self.shared_memory.set(key, payload)
        else:
            self.shared_memory.set(key, payload, ttl=int(self.shared_ttl_seconds))

    def _publish_event(self, payload: Mapping[str, Any]) -> None:
        if self.publish_notifications and self.shared_memory is not None:
            self.shared_memory.publish(self.shared_event_channel, deepcopy(dict(payload)))

    def register_consent(self, *, subject_id: str, purpose: str, status: str, artifact_ref: str,
                         legal_basis: Optional[str] = None,
                         allowed_contexts: Optional[Sequence[str]] = None,
                         allowed_processors: Optional[Sequence[str]] = None,
                         policy_id: Optional[str] = None,
                         policy_version: Optional[str] = None,
                         metadata: Optional[Mapping[str, Any]] = None,) -> Dict[str, Any]:
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

    def bind_purpose(self, *, subject_id: str, purpose: str, source_context: str,
                     allowed_contexts: Optional[Sequence[str]] = None,
                     allowed_actions: Optional[Sequence[str]] = None,
                     policy_id: Optional[str] = None,
                     policy_version: Optional[str] = None,
                     metadata: Optional[Mapping[str, Any]] = None,) -> Dict[str, Any]:
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

    def evaluate_privacy(self, payload: Mapping[str, Any], *, request_id: Optional[str] = None,
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
                         context: Optional[Mapping[str, Any]] = None,) -> Dict[str, Any]:
        if not self.enabled:
            req = self._nonempty(request_id, self._new_id("privacy_req"))
            return {
                "request_id": req,
                "decision": PrivacyDecision.ALLOW.value,
                "summary": "Privacy Agent disabled; passing through without mutation.",
                "sanitized_payload": dict(payload),
                "stages": {},
            }

        started_at = time.time()
        req_id = self._nonempty(request_id, self._new_id("privacy_req"))
        rec_id = self._nonempty(record_id, self._new_id("privacy_rec"))
        subject = self._nonempty(subject_id, "anonymous_subject")
        use_purpose = self._nonempty(purpose, self.default_purpose)
        use_action = self._nonempty(action, self.default_action)
        source = self._nonempty(source_context, self.default_source_context)
        destination = self._nonempty(destination_context, self.default_destination_context)
        resolved_policy_id = self._nonempty(policy_id, self.default_policy_id)
        resolved_policy_version = self._nonempty(policy_version, self.default_policy_version)
        audit_ref = self._new_id("audit")

        stages: Dict[str, Dict[str, Any]] = {}
        shared_keys = []
        retention_snapshot: Dict[str, Any] = {}
        evidence_bundle: Optional[Dict[str, Any]] = None

        try:
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
                context=context,
            )
            stages["identification"] = identification

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
                context=context,
            )
            stages["consent"] = consent

            minimization = self.data_min.minimize_payload(
                payload,
                purpose=use_purpose,
                request_id=req_id,
                subject_id=subject,
                record_id=rec_id,
                stage="minimization.runtime_gate",
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
                    "upstream_sensitivity_score": identification.get("sensitivity_score"),
                    "upstream_entity_count": identification.get("entity_count"),
                    **self._safe_mapping(context),
                },
            )
            stages["minimization"] = minimization

            obligation = self.data_retention.create_retention_obligation(
                record_id=rec_id,
                subject_id=subject,
                request_id=req_id,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                retention_days=int(retention_days or self.default_retention_days),
                metadata={
                    "source_context": source,
                    "destination_context": destination,
                    "action": use_action,
                    **self._safe_mapping(context),
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

            checkpoint = self.private_audit.record_decision_checkpoint(
                request_id=req_id,
                stage="privacy_agent.runtime_gate",
                summary="Privacy runtime gate completed across identification, consent, minimization, and retention.",
                subject_id=subject,
                record_id=rec_id,
                purpose=use_purpose,
                policy_id=resolved_policy_id,
                policy_version=resolved_policy_version,
                context={
                    "decision_candidates": {
                        "identification": identification.get("decision"),
                        "consent": consent.get("decision"),
                        "minimization": minimization.get("decision"),
                    },
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
                    bundle_name="privacy_runtime_bundle",
                    context={"agent": self.name, "source_context": source, "destination_context": destination},
                )
                stages["evidence_bundle"] = {
                    "bundle_id": evidence_bundle.get("bundle_id"),
                    "fingerprint": evidence_bundle.get("fingerprint"),
                }

            final_decision = PrivacyDecision.MODIFY.value if minimization.get("decision") == PrivacyDecision.MODIFY.value else PrivacyDecision.ALLOW.value
            summary = "Payload approved with runtime privacy controls applied."

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
                stages=stages,
                sanitized_payload=minimization.get("sanitized_payload", {}),
                retention=retention_snapshot,
                evidence_bundle=evidence_bundle,
                shared_memory_keys=[],
                duration_ms=round((time.time() - started_at) * 1000.0, 3),
                created_at=time.time(),
            )
            report_payload = report.to_dict()

            result_key = f"{self.shared_key_prefix}.result.{req_id}"
            summary_key = f"{self.shared_key_prefix}.summary.{req_id}"
            self._publish(result_key, report_payload)
            self._publish(summary_key, {
                "request_id": req_id,
                "decision": report.decision,
                "summary": report.summary,
                "sensitivity_score": identification.get("sensitivity_score"),
                "masked_field_count": minimization.get("masked_field_count", 0),
                "removed_field_count": minimization.get("removed_field_count", 0),
                "retention_status": retention_snapshot.get("status"),
                "audit_trail_ref": audit_ref,
                "policy_id": resolved_policy_id,
                "policy_version": resolved_policy_version,
                "created_at": report.created_at,
            })
            shared_keys.extend([result_key, summary_key])
            report_payload["shared_memory_keys"] = shared_keys

            self._publish_event({
                "event_type": "privacy.runtime.decision",
                "request_id": req_id,
                "decision": report.decision,
                "policy_id": resolved_policy_id,
                "policy_version": resolved_policy_version,
                "audit_trail_ref": audit_ref,
                "timestamp": time.time(),
            })

            return report_payload
        except Exception as exc:
            normalized = self._normalize_error(
                exc,
                stage="privacy_agent.evaluate_privacy",
                context={"request_id": req_id, "record_id": rec_id, "subject_id": subject, "purpose": use_purpose},
            )

            error_payload = {
                "request_id": req_id,
                "record_id": rec_id,
                "subject_id": subject,
                "purpose": use_purpose,
                "policy_id": resolved_policy_id,
                "policy_version": resolved_policy_version,
                "decision": PrivacyDecision.BLOCK.value if self.fail_closed_on_subsystem_error else PrivacyDecision.ESCALATE.value,
                "message": normalized.message,
                "error_type": normalized.error_type.value,
                "error_code": normalized.error_code,
                "severity": normalized.severity.value,
                "retryable": normalized.retryable,
                "stages": stages,
                "audit_trail_ref": audit_ref,
                "timestamp": time.time(),
            }

            error_key = f"{self.shared_key_prefix}.error.{req_id}"
            self._publish(error_key, error_payload)
            self._publish_event({
                "event_type": "privacy.runtime.error",
                "request_id": req_id,
                "error_type": normalized.error_type.value,
                "error_code": normalized.error_code,
                "severity": normalized.severity.value,
                "policy_id": resolved_policy_id,
                "policy_version": resolved_policy_version,
                "timestamp": time.time(),
            })

            if self.fail_closed_on_subsystem_error:
                raise normalized from exc
            return error_payload

    def perform_task_privacy(self, input_data: Any, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        normalized_context = self._safe_mapping(context)
    
        if isinstance(input_data, Mapping):
            payload_value = input_data.get("payload")
            if isinstance(payload_value, Mapping):
                payload = dict(payload_value)          # payload_value is a Mapping
            else:
                payload = dict(input_data)             # input_data is a Mapping
            runtime_context = {**normalized_context, **self._safe_mapping(input_data.get("context"))}
            return self.evaluate_privacy(
                payload,
                request_id=input_data.get("request_id"),
                subject_id=input_data.get("subject_id"),
                record_id=input_data.get("record_id"),
                purpose=input_data.get("purpose"),
                action=input_data.get("action"),
                source_context=input_data.get("source_context"),
                destination_context=input_data.get("destination_context"),
                required_processor=input_data.get("required_processor"),
                allowed_fields=input_data.get("allowed_fields"),
                required_fields=input_data.get("required_fields"),
                sensitive_fields=input_data.get("sensitive_fields"),
                field_strategies=input_data.get("field_strategies"),
                retention_days=input_data.get("retention_days"),
                record_tags=input_data.get("record_tags"),
                policy_id=input_data.get("policy_id"),
                policy_version=input_data.get("policy_version"),
                context=runtime_context,
            )
    
        wrapped_payload = {"content": input_data}
        return self.evaluate_privacy(wrapped_payload, context=normalized_context)
        
        
if __name__ == "__main__":
    print("\n=== Running Privacy agent ===\n")
    printer.status("TEST", "Privacy agent initialized", "info")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    agent = PrivacyAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
    )
    printer.status("CONFIG", f"Loaded privacy_agent config from {agent.config.get('__config_path__', 'unknown')}", "success")
