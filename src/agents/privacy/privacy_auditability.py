"""
- Decision trace for each privacy action.
- Audit evidence bundle generation.
- Policy versioning metadata.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid

from threading import RLock
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .utils import (get_config_section, load_global_config,
                    # Error
                    AuditEvidenceGenerationError, AuditReportGenerationError, PrivacyError, PrivacyDecision,
                    AuditLogWriteError, PrivacyConfigurationError, PolicyEvaluationError, PrivacyMemoryError,
                    normalize_privacy_exception, sanitize_privacy_context)
from .privacy_memory import PrivacyMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Privacy Auditability")
printer = PrettyPrinter


class PrivacyAuditability:
    """Privacy audit orchestration for evidence, reports, and policy metadata.

    PrivacyMemory remains the operational source of truth for consent, retention,
    lineage, redaction, and decision state. PrivacyAuditability builds on top of
    that state to create durable audit events, policy metadata snapshots, evidence
    bundles, and operator-facing reports.
    """

    EVENT_STATUSES = {"info", "success", "warning", "blocked", "failed"}

    def __init__(self, memory: Optional[PrivacyMemory] = None) -> None:
        self.config = load_global_config()
        self.audit_config = get_config_section("privacy_auditability")
        self._lock = RLock()

        self.enabled = bool(self.audit_config.get("enabled", True))
        self.strict_mode = bool(self.audit_config.get("strict_mode", True))
        self.sanitize_freeform_context = bool(self.audit_config.get("sanitize_freeform_context", True))
        self.require_policy_metadata = bool(self.audit_config.get("require_policy_metadata", True))
        self.require_bundle_identifiers = bool(self.audit_config.get("require_bundle_identifiers", True))
        self.include_memory_evidence_by_default = bool(
            self.audit_config.get("include_memory_evidence_by_default", True)
        )
        self.include_related_events_by_default = bool(
            self.audit_config.get("include_related_events_by_default", True)
        )
        self.include_config_metadata_in_reports = bool(
            self.audit_config.get("include_config_metadata_in_reports", True)
        )
        self.emit_events_to_registered_sinks = bool(
            self.audit_config.get("emit_events_to_registered_sinks", True)
        )
        self.allow_state_export = bool(self.audit_config.get("allow_state_export", True))
        self.default_policy_version = str(self.audit_config.get("default_policy_version", "v1"))
        self.default_bundle_name = str(self.audit_config.get("default_bundle_name", "privacy_audit_bundle"))
        self.max_audit_events = int(self.audit_config.get("max_audit_events", 500))
        self.max_bundle_events = int(self.audit_config.get("max_bundle_events", 200))
        self.max_events_per_request = int(self.audit_config.get("max_events_per_request", 150))
        self.max_events_per_record = int(self.audit_config.get("max_events_per_record", 150))
        self.max_events_per_subject = int(self.audit_config.get("max_events_per_subject", 200))
        self.max_policy_snapshots = int(self.audit_config.get("max_policy_snapshots", 50))

        self._validate_config()

        self.memory = memory or PrivacyMemory()
        self._event_log: List[Dict[str, Any]] = []
        self._request_index: Dict[str, List[Dict[str, Any]]] = {}
        self._record_index: Dict[str, List[Dict[str, Any]]] = {}
        self._subject_index: Dict[str, List[Dict[str, Any]]] = {}
        self._bundle_registry: Dict[str, Dict[str, Any]] = {}
        self._policy_registry: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._sinks: List[Callable[[Dict[str, Any]], None]] = []

        logger.info("PrivacyAuditability initialized with production-ready evidence and report orchestration.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        validators = {
            "max_audit_events": self.max_audit_events,
            "max_bundle_events": self.max_bundle_events,
            "max_events_per_request": self.max_events_per_request,
            "max_events_per_record": self.max_events_per_record,
            "max_events_per_subject": self.max_events_per_subject,
            "max_policy_snapshots": self.max_policy_snapshots,
        }
        for field_name, value in validators.items():
            if value <= 0:
                raise PrivacyConfigurationError(
                    section="privacy_auditability",
                    details=f"'{field_name}' must be a positive integer, received {value!r}",
                )

    def _require_enabled(self, operation: str) -> None:
        if not self.enabled:
            raise PrivacyMemoryError(
                operation=operation,
                details="privacy_auditability is disabled by configuration",
                retryable=False,
                context={"config_section": "privacy_auditability"},
            )

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _deepcopy(value: Any) -> Any:
        return copy.deepcopy(value)

    @staticmethod
    def _trim_history(history: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        if len(history) <= limit:
            return history
        return history[-limit:]

    @staticmethod
    def _normalize_identity(value: str, field_name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"'{field_name}' must be a non-empty string")
        return normalized

    def _normalize_context(self, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not payload:
            return {}
        if self.sanitize_freeform_context:
            return sanitize_privacy_context(payload)
        return self._deepcopy(dict(payload))

    def _new_ref(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:16]}"

    def _resolve_policy_version(self, policy_version: Optional[str]) -> str:
        return str(policy_version or self.default_policy_version)

    def _require_policy_fields(
        self,
        *,
        policy_id: Optional[str],
        policy_version: Optional[str],
        stage: str,
    ) -> Tuple[Optional[str], str]:
        if not policy_id:
            if self.require_policy_metadata:
                raise PolicyEvaluationError(
                    stage=stage,
                    details="policy_id is required by privacy_auditability configuration",
                )
            return None, self._resolve_policy_version(policy_version)
        return str(policy_id).strip(), self._resolve_policy_version(policy_version)

    @staticmethod
    def _fingerprint(payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _emit_event(self, payload: Dict[str, Any]) -> None:
        if not self.emit_events_to_registered_sinks:
            return
        for sink in list(self._sinks):
            try:
                sink(self._deepcopy(payload))
            except Exception as exc:
                raise AuditLogWriteError(
                    channel=getattr(sink, "__name__", sink.__class__.__name__),
                    details=exc,
                    context={"event_id": payload.get("event_id")},
                    request_id=payload.get("request_id"),
                ) from exc

    def _index_event(self, event: Dict[str, Any]) -> None:
        self._event_log = self._trim_history(self._event_log + [event], self.max_audit_events)

        request_id = event.get("request_id")
        if request_id:
            request_history = self._request_index.setdefault(request_id, [])
            self._request_index[request_id] = self._trim_history(
                request_history + [event], self.max_events_per_request
            )

        record_id = event.get("record_id")
        if record_id:
            record_history = self._record_index.setdefault(record_id, [])
            self._record_index[record_id] = self._trim_history(
                record_history + [event], self.max_events_per_record
            )

        subject_id = event.get("subject_id")
        if subject_id:
            subject_history = self._subject_index.setdefault(subject_id, [])
            self._subject_index[subject_id] = self._trim_history(
                subject_history + [event], self.max_events_per_subject
            )

    def _build_base_event(
        self,
        *,
        category: str,
        action: str,
        status: str,
        summary: str,
        request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        purpose: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        evidence_refs: Optional[Sequence[str]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_category = self._normalize_identity(category, "category")
        normalized_action = self._normalize_identity(action, "action")
        normalized_status = self._normalize_identity(status, "status").lower()
        if normalized_status not in self.EVENT_STATUSES:
            raise ValueError(
                f"'status' must be one of {sorted(self.EVENT_STATUSES)}, received {normalized_status!r}"
            )
        normalized_summary = self._normalize_identity(summary, "summary")
        resolved_policy_id, resolved_policy_version = self._require_policy_fields(
            policy_id=policy_id,
            policy_version=policy_version,
            stage=f"privacy_auditability.{normalized_action}",
        )

        normalized_evidence_refs: List[str] = []
        for item in evidence_refs or []:
            value = str(item).strip()
            if value and value not in normalized_evidence_refs:
                normalized_evidence_refs.append(value)

        event = {
            "event_id": self._new_ref("audit-event"),
            "timestamp": self._now(),
            "category": normalized_category,
            "action": normalized_action,
            "status": normalized_status,
            "summary": normalized_summary,
            "request_id": request_id,
            "record_id": record_id,
            "subject_id": subject_id,
            "purpose": purpose,
            "policy_id": resolved_policy_id,
            "policy_version": resolved_policy_version,
            "audit_trail_ref": audit_trail_ref or self._new_ref("audit-trail"),
            "evidence_refs": normalized_evidence_refs,
            "context": self._normalize_context(context),
        }
        event["fingerprint"] = self._fingerprint(event)
        return event

    def _handle_exception(
        self,
        exc: Exception,
        *,
        stage: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PrivacyError:
        if isinstance(exc, PrivacyError):
            return exc
        return normalize_privacy_exception(exc, stage=stage, context=context)

    # ------------------------------------------------------------------
    # Sink management and policy snapshots
    # ------------------------------------------------------------------
    def register_audit_sink(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        operation = "register_audit_sink"
        try:
            self._require_enabled(operation)
            if not callable(callback):
                raise TypeError("'callback' must be callable")
            with self._lock:
                if callback not in self._sinks:
                    self._sinks.append(callback)
        except Exception as exc:
            raise self._handle_exception(exc, stage="privacy_auditability.register_audit_sink") from exc

    def unregister_audit_sink(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        operation = "unregister_audit_sink"
        try:
            self._require_enabled(operation)
            with self._lock:
                self._sinks = [sink for sink in self._sinks if sink is not callback]
        except Exception as exc:
            raise self._handle_exception(exc, stage="privacy_auditability.unregister_audit_sink") from exc

    def capture_policy_snapshot(
        self,
        *,
        policy_id: str,
        policy_version: Optional[str],
        stage: str,
        metadata: Optional[Mapping[str, Any]] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "capture_policy_snapshot"
        try:
            self._require_enabled(operation)
            normalized_policy_id = self._normalize_identity(policy_id, "policy_id")
            normalized_stage = self._normalize_identity(stage, "stage")
            resolved_policy_version = self._resolve_policy_version(policy_version)
            snapshot = {
                "snapshot_id": self._new_ref("policy-snapshot"),
                "policy_id": normalized_policy_id,
                "policy_version": resolved_policy_version,
                "stage": normalized_stage,
                "captured_at": self._now(),
                "audit_trail_ref": audit_trail_ref or self._new_ref("audit-policy"),
                "metadata": self._normalize_context(metadata),
            }
            snapshot["fingerprint"] = self._fingerprint(snapshot)

            with self._lock:
                registry_key = (normalized_policy_id, resolved_policy_version)
                entry = self._policy_registry.setdefault(
                    registry_key,
                    {
                        "policy_id": normalized_policy_id,
                        "policy_version": resolved_policy_version,
                        "snapshots": [],
                        "last_snapshot_at": None,
                    },
                )
                entry["snapshots"] = self._trim_history(entry["snapshots"] + [snapshot], self.max_policy_snapshots)
                entry["last_snapshot_at"] = snapshot["captured_at"]
                return self._deepcopy(entry)
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="privacy_auditability.capture_policy_snapshot",
                context={"policy_id": policy_id, "policy_version": policy_version, "stage": stage},
            ) from exc

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------
    def record_audit_event(
        self,
        *,
        category: str,
        action: str,
        status: str,
        summary: str,
        request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        purpose: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        evidence_refs: Optional[Sequence[str]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "record_audit_event"
        try:
            self._require_enabled(operation)
            event = self._build_base_event(
                category=category,
                action=action,
                status=status,
                summary=summary,
                request_id=request_id,
                record_id=record_id,
                subject_id=subject_id,
                purpose=purpose,
                policy_id=policy_id,
                policy_version=policy_version,
                audit_trail_ref=audit_trail_ref,
                evidence_refs=evidence_refs,
                context=context,
            )
            with self._lock:
                self._index_event(event)
                self._emit_event(event)
                return self._deepcopy(event)
        except Exception as exc:
            if isinstance(exc, AuditLogWriteError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="privacy_auditability.record_audit_event",
                context={"category": category, "action": action, "request_id": request_id},
            ) from exc

    def record_decision_checkpoint(self, *, request_id: str, stage: str, summary: str,
        policy_id: Optional[str],
        policy_version: Optional[str],
        subject_id: Optional[str] = None,
        record_id: Optional[str] = None,
        purpose: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "record_decision_checkpoint"
        try:
            self._require_enabled(operation)
            normalized_request_id = self._normalize_identity(request_id, "request_id")
            normalized_stage = self._normalize_identity(stage, "stage")
            trace = self.memory.privacy_decision_trace(normalized_request_id)
            latest_decision = trace.get("latest_decision") or "unknown"
            evidence_refs = []
            if trace.get("latest_audit_trail_ref"):
                evidence_refs.append(trace["latest_audit_trail_ref"])
            if trace.get("redaction") and trace["redaction"].get("latest_transformation_ref"):
                evidence_refs.append(trace["redaction"]["latest_transformation_ref"])

            snapshot = self.capture_policy_snapshot(
                policy_id=policy_id or "privacy-runtime-policy",
                policy_version=policy_version,
                stage=normalized_stage,
                metadata={"request_id": normalized_request_id, "latest_decision": latest_decision},
            )
            evidence_refs.extend([item["snapshot_id"] for item in snapshot["snapshots"][-1:]])

            return self.record_audit_event(
                category="decision_trace",
                action="decision_checkpoint_recorded",
                status="success",
                summary=summary,
                request_id=normalized_request_id,
                record_id=record_id,
                subject_id=subject_id or trace.get("subject_id"),
                purpose=purpose,
                policy_id=policy_id or trace.get("policy_id") or "privacy-runtime-policy",
                policy_version=policy_version or trace.get("policy_version"),
                evidence_refs=evidence_refs,
                context={
                    "stage": normalized_stage,
                    "latest_decision": latest_decision,
                    "shared_contract": trace.get("shared_contract"),
                    "lineage_present": bool(trace.get("lineage")),
                    "redaction_present": bool(trace.get("redaction")),
                    **dict(context or {}),
                },
            )
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="privacy_auditability.record_decision_checkpoint",
                context={"request_id": request_id, "stage": stage},
            ) from exc

    # ------------------------------------------------------------------
    # Evidence bundles and reports
    # ------------------------------------------------------------------
    def generate_audit_evidence_bundle(self, *, request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        purpose: Optional[str] = None,
        policy_id: Optional[str],
        policy_version: Optional[str],
        bundle_name: Optional[str] = None,
        include_memory_bundle: Optional[bool] = None,
        include_related_events: Optional[bool] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "generate_audit_evidence_bundle"
        try:
            self._require_enabled(operation)
            if self.require_bundle_identifiers and not any([request_id, record_id, subject_id]):
                raise ValueError("At least one of 'request_id', 'record_id', or 'subject_id' must be provided")

            resolved_policy_id, resolved_policy_version = self._require_policy_fields(
                policy_id=policy_id,
                policy_version=policy_version,
                stage="privacy_auditability.generate_audit_evidence_bundle",
            )
            normalized_bundle_name = str(bundle_name or self.default_bundle_name).strip() or self.default_bundle_name
            use_memory_bundle = self.include_memory_evidence_by_default if include_memory_bundle is None else bool(include_memory_bundle)
            use_related_events = self.include_related_events_by_default if include_related_events is None else bool(include_related_events)

            memory_bundle = None
            if use_memory_bundle:
                memory_bundle = self.memory.build_audit_evidence_bundle(
                    request_id=request_id,
                    record_id=record_id,
                    subject_id=subject_id,
                    purpose=purpose,
                )

            with self._lock:
                related_events: List[Dict[str, Any]] = []
                if use_related_events:
                    if request_id:
                        related_events.extend(self._deepcopy(self._request_index.get(request_id, [])))
                    if record_id:
                        related_events.extend(self._deepcopy(self._record_index.get(record_id, [])))
                    if subject_id:
                        related_events.extend(self._deepcopy(self._subject_index.get(subject_id, [])))

                deduped: Dict[str, Dict[str, Any]] = {}
                for event in related_events:
                    deduped[event["event_id"]] = event
                related_events = list(deduped.values())[-self.max_bundle_events :]

                snapshot = self.capture_policy_snapshot(
                    policy_id=resolved_policy_id or "privacy-audit-policy",
                    policy_version=resolved_policy_version,
                    stage="evidence_bundle_generation",
                    metadata={
                        "bundle_name": normalized_bundle_name,
                        "request_id": request_id,
                        "record_id": record_id,
                        "subject_id": subject_id,
                    },
                )

                bundle = {
                    "bundle_id": self._new_ref("audit-bundle"),
                    "bundle_name": normalized_bundle_name,
                    "generated_at": self._now(),
                    "request_id": request_id,
                    "record_id": record_id,
                    "subject_id": subject_id,
                    "purpose": purpose,
                    "policy_id": resolved_policy_id,
                    "policy_version": resolved_policy_version,
                    "policy_snapshot": self._deepcopy(snapshot["snapshots"][-1]),
                    "memory_evidence": self._deepcopy(memory_bundle),
                    "audit_events": related_events,
                    "context": self._normalize_context(context),
                }
                bundle["fingerprint"] = self._fingerprint(bundle)
                self._bundle_registry[bundle["bundle_id"]] = self._deepcopy(bundle)

                self.record_audit_event(
                    category="evidence_bundle",
                    action="audit_evidence_bundle_generated",
                    status="success",
                    summary=f"Generated audit evidence bundle '{normalized_bundle_name}'.",
                    request_id=request_id,
                    record_id=record_id,
                    subject_id=subject_id,
                    purpose=purpose,
                    policy_id=resolved_policy_id,
                    policy_version=resolved_policy_version,
                    evidence_refs=[bundle["bundle_id"], bundle["policy_snapshot"]["snapshot_id"]],
                    context={"bundle_fingerprint": bundle["fingerprint"], "audit_events": len(related_events)},
                )
                return self._deepcopy(bundle)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise AuditEvidenceGenerationError(
                    bundle_name=bundle_name or self.default_bundle_name,
                    details=exc,
                    context={"request_id": request_id, "record_id": record_id, "subject_id": subject_id},
                    request_id=request_id,
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_auditability.generate_audit_evidence_bundle",
                context={"request_id": request_id, "record_id": record_id, "subject_id": subject_id, "purpose": purpose},
            ) from exc

    def generate_decision_trace_report(self, *, request_id: str,
        policy_id: Optional[str],
        policy_version: Optional[str],
        include_bundle: bool = True,
    ) -> Dict[str, Any]:
        operation = "generate_decision_trace_report"
        try:
            self._require_enabled(operation)
            normalized_request_id = self._normalize_identity(request_id, "request_id")
            trace = self.memory.privacy_decision_trace(normalized_request_id)
            if not trace.get("history") and self.strict_mode:
                raise AuditReportGenerationError(
                    report_name="decision_trace_report",
                    details=f"No privacy decision trace exists for request '{normalized_request_id}'.",
                    request_id=normalized_request_id,
                )

            bundle = None
            if include_bundle:
                bundle = self.generate_audit_evidence_bundle(
                    request_id=normalized_request_id,
                    record_id=trace.get("record_id"),
                    subject_id=trace.get("subject_id"),
                    policy_id=policy_id or trace.get("policy_id") or "privacy-runtime-policy",
                    policy_version=policy_version or trace.get("policy_version"),
                    bundle_name="decision_trace_bundle",
                )

            report = {
                "report_type": "decision_trace_report",
                "generated_at": self._now(),
                "request_id": normalized_request_id,
                "latest_decision": trace.get("latest_decision"),
                "latest_stage": trace.get("latest_stage"),
                "latest_audit_trail_ref": trace.get("latest_audit_trail_ref"),
                "subject_id": trace.get("subject_id"),
                "record_id": trace.get("record_id"),
                "policy_id": policy_id or trace.get("policy_id"),
                "policy_version": policy_version or trace.get("policy_version"),
                "shared_contract": self._deepcopy(trace.get("shared_contract")),
                "lineage": self._deepcopy(trace.get("lineage")),
                "redaction": self._deepcopy(trace.get("redaction")),
                "history": self._deepcopy(trace.get("history", [])),
                "bundle_ref": bundle.get("bundle_id") if bundle else None,
            }
            if self.include_config_metadata_in_reports:
                report["config_path"] = self.config.get("__config_path__")
            report["fingerprint"] = self._fingerprint(report)

            self.record_audit_event(
                category="audit_report",
                action="decision_trace_report_generated",
                status="success",
                summary="Generated privacy decision trace report.",
                request_id=normalized_request_id,
                record_id=trace.get("record_id"),
                subject_id=trace.get("subject_id"),
                policy_id=policy_id or trace.get("policy_id") or "privacy-runtime-policy",
                policy_version=policy_version or trace.get("policy_version"),
                evidence_refs=[bundle["bundle_id"]] if bundle else [],
                context={"history_length": len(report["history"]), "fingerprint": report["fingerprint"]},
            )
            return report
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise AuditReportGenerationError(
                    report_name="decision_trace_report",
                    details=exc,
                    request_id=request_id,
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_auditability.generate_decision_trace_report",
                context={"request_id": request_id},
            ) from exc

    def generate_subject_audit_report(self, *, subject_id: str,
        purpose: Optional[str],
        policy_id: Optional[str],
        policy_version: Optional[str],
    ) -> Dict[str, Any]:
        operation = "generate_subject_audit_report"
        try:
            self._require_enabled(operation)
            normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
            consent = None
            if purpose:
                consent = self.memory.consent_status(normalized_subject_id, purpose)

            with self._lock:
                events = self._deepcopy(self._subject_index.get(normalized_subject_id, []))
                report = {
                    "report_type": "subject_audit_report",
                    "generated_at": self._now(),
                    "subject_id": normalized_subject_id,
                    "purpose": purpose,
                    "consent": consent,
                    "events": events,
                    "policy_id": policy_id,
                    "policy_version": self._resolve_policy_version(policy_version),
                }
                if self.include_config_metadata_in_reports:
                    report["config_path"] = self.config.get("__config_path__")
                report["fingerprint"] = self._fingerprint(report)

            self.record_audit_event(
                category="audit_report",
                action="subject_audit_report_generated",
                status="success",
                summary="Generated subject-level privacy audit report.",
                subject_id=normalized_subject_id,
                purpose=purpose,
                policy_id=policy_id or "privacy-subject-audit-policy",
                policy_version=policy_version,
                context={"event_count": len(events), "fingerprint": report["fingerprint"]},
            )
            return report
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise AuditReportGenerationError(
                    report_name="subject_audit_report",
                    details=exc,
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_auditability.generate_subject_audit_report",
                context={"subject_id": subject_id, "purpose": purpose},
            ) from exc

    def generate_record_audit_report(self, *, record_id: str,
        policy_id: Optional[str],
        policy_version: Optional[str],
    ) -> Dict[str, Any]:
        operation = "generate_record_audit_report"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            retention = self.memory.retention_obligation(normalized_record_id)
            with self._lock:
                events = self._deepcopy(self._record_index.get(normalized_record_id, []))
                report = {
                    "report_type": "record_audit_report",
                    "generated_at": self._now(),
                    "record_id": normalized_record_id,
                    "retention": retention,
                    "events": events,
                    "policy_id": policy_id or retention.get("policy_id"),
                    "policy_version": policy_version or retention.get("policy_version"),
                }
                if self.include_config_metadata_in_reports:
                    report["config_path"] = self.config.get("__config_path__")
                report["fingerprint"] = self._fingerprint(report)

            self.record_audit_event(
                category="audit_report",
                action="record_audit_report_generated",
                status="success",
                summary="Generated record-level privacy audit report.",
                record_id=normalized_record_id,
                subject_id=retention.get("subject_id"),
                policy_id=policy_id or retention.get("policy_id") or "privacy-record-audit-policy",
                policy_version=policy_version or retention.get("policy_version"),
                context={"event_count": len(events), "fingerprint": report["fingerprint"]},
            )
            return report
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise AuditReportGenerationError(
                    report_name="record_audit_report",
                    details=exc,
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_auditability.generate_record_audit_report",
                context={"record_id": record_id},
            ) from exc

    # ------------------------------------------------------------------
    # Operational helpers
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        operation = "stats"
        try:
            self._require_enabled(operation)
            with self._lock:
                return {
                    "enabled": self.enabled,
                    "strict_mode": self.strict_mode,
                    "audit_events": len(self._event_log),
                    "request_entries": len(self._request_index),
                    "record_entries": len(self._record_index),
                    "subject_entries": len(self._subject_index),
                    "bundles": len(self._bundle_registry),
                    "policy_snapshots": sum(len(entry["snapshots"]) for entry in self._policy_registry.values()),
                    "registered_sinks": len(self._sinks),
                    "state_export_allowed": self.allow_state_export,
                    "memory_stats": self.memory.stats(),
                    "config_path": self.config.get("__config_path__"),
                }
        except Exception as exc:
            raise self._handle_exception(exc, stage="privacy_auditability.stats") from exc

    def export_state(self) -> Dict[str, Any]:
        operation = "export_state"
        try:
            self._require_enabled(operation)
            if not self.allow_state_export:
                raise PrivacyMemoryError(
                    operation=operation,
                    details="state export is disabled by configuration",
                    retryable=False,
                )
            with self._lock:
                return {
                    "event_log": self._deepcopy(self._event_log),
                    "request_index": self._deepcopy(self._request_index),
                    "record_index": self._deepcopy(self._record_index),
                    "subject_index": self._deepcopy(self._subject_index),
                    "bundle_registry": self._deepcopy(self._bundle_registry),
                    "policy_registry": self._deepcopy(self._policy_registry),
                }
        except Exception as exc:
            raise self._handle_exception(exc, stage="privacy_auditability.export_state") from exc


if __name__ == "__main__":
    print("\n=== Running Privacy Auditability===\n")
    printer.status("TEST", "Privacy Auditability initialized", "info")

    captured_events: List[Dict[str, Any]] = []

    def _test_sink(payload: Dict[str, Any]) -> None:
        captured_events.append(payload)

    auditability = PrivacyAuditability()
    auditability.register_audit_sink(_test_sink)
    printer.status("TEST", "Audit sink registered", "info")

    auditability.memory.register_consent_artifact(
        subject_id="subject-100",
        purpose="support_resolution",
        status="granted",
        artifact_ref="consent-artifact-100",
        legal_basis="explicit_consent",
        granted_at=time.time(),
        allowed_contexts=["support_workspace", "privacy_review"],
        allowed_processors=["knowledge_agent", "execution_agent"],
        metadata={"collector": "privacy_agent", "channel": "chat"},
        policy_id="consent-policy",
        policy_version="2026.04",
    )
    printer.status("TEST", "Consent artifact registered", "info")

    auditability.memory.bind_purpose(
        subject_id="subject-100",
        purpose="support_resolution",
        source_context="support_workspace",
        allowed_contexts=["support_workspace", "privacy_review"],
        allowed_actions=["read", "redact", "respond"],
        metadata={"cross_context_sharing": "restricted"},
        policy_id="purpose-policy",
        policy_version="2026.04",
    )
    printer.status("TEST", "Purpose binding created", "info")

    auditability.memory.record_privacy_decision(
        request_id="request-100",
        stage="execution.pre_tool_call",
        decision=PrivacyDecision.MODIFY,
        rationale="Sensitive fields present; payload minimized before downstream tool invocation.",
        subject_id="subject-100",
        record_id="record-100",
        purpose="support_resolution",
        sensitivity_score=0.94,
        detected_entities=["email", "phone", "account_id"],
        redaction_actions=["mask_email", "tokenize_phone"],
        retention_policy_id="retention-standard-30d",
        consent_status={"status": "granted", "artifact_ref": "consent-artifact-100"},
        policy_id="privacy-runtime-policy",
        policy_version="2026.04",
        context={"tool_name": "ticketing_connector", "payload": {"email": "user@example.com"}},
    )
    printer.status("TEST", "Privacy decision recorded", "info")

    auditability.memory.record_lineage_event(
        request_id="request-100",
        parent_request_id="request-root-100",
        record_id="record-100",
        operation="payload_transfer",
        stage="execution.pre_tool_call",
        source_context="chat_runtime",
        destination_context="ticketing_connector",
        subject_id="subject-100",
        purpose="support_resolution",
        context={"transformation": "minimized_payload"},
    )
    printer.status("TEST", "Lineage event recorded", "info")

    auditability.memory.store_redaction_transformation(
        request_id="request-100",
        record_id="record-100",
        masked_fields=["email", "phone"],
        strategy="tokenization",
        reason="least_data_required_for_external_tool",
        original_field_count=6,
        retained_field_count=4,
        context={"executor": "execution_agent"},
    )
    printer.status("TEST", "Redaction transformation stored", "info")

    auditability.memory.create_retention_obligation(
        record_id="record-100",
        subject_id="subject-100",
        policy_id="retention-standard-30d",
        retention_days=30,
        metadata={"data_class": "support_case"},
        policy_version="2026.04",
    )
    auditability.memory.schedule_deletion(
        record_id="record-100",
        reason="retention_period_elapsed",
        requested_by="privacy_agent",
    )
    printer.status("TEST", "Retention and deletion workflow staged", "info")

    auditability.capture_policy_snapshot(
        policy_id="privacy-runtime-policy",
        policy_version="2026.04",
        stage="execution.pre_tool_call",
        metadata={"actor": "privacy_agent"},
    )
    printer.status("TEST", "Policy snapshot captured", "info")

    auditability.record_decision_checkpoint(
        request_id="request-100",
        stage="execution.pre_tool_call",
        summary="Decision checkpoint recorded for minimized external payload transfer.",
        subject_id="subject-100",
        record_id="record-100",
        purpose="support_resolution",
        policy_id="privacy-runtime-policy",
        policy_version="2026.04",
        context={"checkpoint": "pre_tool_call"},
    )
    printer.status("TEST", "Decision checkpoint recorded", "info")

    auditability.record_audit_event(
        category="compliance",
        action="privacy_review_completed",
        status="success",
        summary="Privacy review completed successfully for runtime request.",
        request_id="request-100",
        record_id="record-100",
        subject_id="subject-100",
        purpose="support_resolution",
        policy_id="privacy-runtime-policy",
        policy_version="2026.04",
        evidence_refs=["consent-artifact-100"],
        context={"reviewer": "privacy_agent", "result": "approved_with_modification"},
    )
    printer.status("TEST", "Custom audit event recorded", "info")

    bundle = auditability.generate_audit_evidence_bundle(
        request_id="request-100",
        record_id="record-100",
        subject_id="subject-100",
        purpose="support_resolution",
        policy_id="privacy-runtime-policy",
        policy_version="2026.04",
        bundle_name="support_resolution_bundle",
    )
    printer.status("TEST", "Audit evidence bundle generated", "info")

    decision_report = auditability.generate_decision_trace_report(
        request_id="request-100",
        policy_id="privacy-runtime-policy",
        policy_version="2026.04",
    )
    printer.status("TEST", "Decision trace report generated", "info")

    subject_report = auditability.generate_subject_audit_report(
        subject_id="subject-100",
        purpose="support_resolution",
        policy_id="privacy-subject-audit-policy",
        policy_version="2026.04",
    )
    printer.status("TEST", "Subject audit report generated", "info")

    record_report = auditability.generate_record_audit_report(
        record_id="record-100",
        policy_id="privacy-record-audit-policy",
        policy_version="2026.04",
    )
    printer.status("TEST", "Record audit report generated", "info")

    print(bundle)
    print(decision_report)
    print(subject_report)
    print(record_report)
    print(auditability.stats())
    print({"captured_sink_events": len(captured_events)})

    print("\n=== Test ran successfully ===\n")