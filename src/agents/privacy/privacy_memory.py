"""
- Stores policy decisions with immutable audit references.
- Tracks consent artifacts, purpose bindings, and data-handling lineage.
- Maintains retention/deletion schedules and fulfillment status.
- Preserves redaction transformation metadata (what was masked and why).
- Exposes retrieval APIs for:
    - consent_status(subject_id, purpose)
    - retention_obligation(record_id)
    - privacy_decision_trace(request_id)
"""

from __future__ import annotations

import copy
import json
import time
import uuid

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from threading import RLock

from .utils import (get_config_section, load_global_config,
                    # Error
                    PolicyEvaluationError, PrivacyConfigurationError, PrivacyDecision, PrivacyError,
                    PrivacyMemoryError, PrivacyMemoryWriteError, RetentionObligationMissingError,
                    normalize_privacy_exception, sanitize_privacy_context)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Privacy Memory")
printer = PrettyPrinter


class PrivacyMemory:
    """Stateful privacy memory for consent, lineage, retention, and audit traceability.

    The memory layer is intentionally structured around append-only event capture with
    materialized indexes for fast retrieval. This gives the privacy subsystem a stable,
    auditable backbone for runtime policy decisions without collapsing all state into a
    single mutable blob.
    """

    CONSENT_STATUSES = {"granted", "denied", "revoked", "expired", "pending"}
    RETENTION_STATUSES = {"active", "due", "expired", "deleted", "legal_hold"}
    DELETION_STATUSES = {"scheduled", "in_progress", "completed", "failed", "cancelled", "legal_hold"}

    def __init__(self) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("privacy_memory")
        self._lock = RLock()

        self.enabled = bool(self.memory_config.get("enabled", True))
        self.strict_mode = bool(self.memory_config.get("strict_mode", True))
        self.sanitize_freeform_context = bool(self.memory_config.get("sanitize_freeform_context", True))
        self.require_audit_refs = bool(self.memory_config.get("require_audit_refs", True))
        self.auto_create_tombstone_on_delete = bool(
            self.memory_config.get("auto_create_tombstone_on_delete", True)
        )
        self.allow_state_export = bool(self.memory_config.get("allow_state_export", True))

        self.default_policy_version = str(self.memory_config.get("default_policy_version", "v1"))
        self.default_deletion_workflow = str(
            self.memory_config.get("default_deletion_workflow", "privacy_deletion_orchestrator")
        )
        self.max_history_per_subject_purpose = int(
            self.memory_config.get("max_history_per_subject_purpose", 50)
        )
        self.max_history_per_record = int(self.memory_config.get("max_history_per_record", 100))
        self.max_decision_trace_length = int(self.memory_config.get("max_decision_trace_length", 200))
        self.max_redaction_history = int(self.memory_config.get("max_redaction_history", 100))
        self.max_lineage_parents = int(self.memory_config.get("max_lineage_parents", 20))
        self.deletion_grace_period_seconds = float(
            self.memory_config.get("deletion_grace_period_seconds", 0)
        )

        self._validate_config()

        self._event_log: List[Dict[str, Any]] = []
        self._consent_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._purpose_binding_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._decision_index: Dict[str, Dict[str, Any]] = {}
        self._retention_index: Dict[str, Dict[str, Any]] = {}
        self._deletion_index: Dict[str, Dict[str, Any]] = {}
        self._redaction_index: Dict[str, Dict[str, Any]] = {}
        self._lineage_index: Dict[str, Dict[str, Any]] = {}
        self._shared_contract_index: Dict[str, Dict[str, Any]] = {}
        self._artifact_index: Dict[str, Dict[str, Any]] = {}

        logger.info("PrivacyMemory initialized with production-ready in-memory indexes.")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _validate_config(self) -> None:
        validators = {
            "max_history_per_subject_purpose": self.max_history_per_subject_purpose,
            "max_history_per_record": self.max_history_per_record,
            "max_decision_trace_length": self.max_decision_trace_length,
            "max_redaction_history": self.max_redaction_history,
            "max_lineage_parents": self.max_lineage_parents,
        }
        for field_name, value in validators.items():
            if value <= 0:
                raise PrivacyConfigurationError(
                    section="privacy_memory",
                    details=f"'{field_name}' must be a positive integer, received {value!r}",
                )

        if self.deletion_grace_period_seconds < 0:
            raise PrivacyConfigurationError(
                section="privacy_memory",
                details=(
                    "'deletion_grace_period_seconds' must be greater than or equal to zero, "
                    f"received {self.deletion_grace_period_seconds!r}"
                ),
            )

    def _require_enabled(self, operation: str) -> None:
        if not self.enabled:
            raise PrivacyMemoryError(
                operation=operation,
                details="privacy_memory is disabled by configuration",
                retryable=False,
                context={"config_section": "privacy_memory"},
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

    @staticmethod
    def _normalize_optional_timestamp(value: Optional[float], field_name: str) -> Optional[float]:
        if value is None:
            return None
        normalized = float(value)
        if normalized < 0:
            raise ValueError(f"'{field_name}' must be >= 0")
        return normalized

    @staticmethod
    def _normalize_sequence(values: Optional[Sequence[Any]]) -> List[str]:
        if not values:
            return []
        normalized: List[str] = []
        seen = set()
        for value in values:
            item = str(value).strip()
            if not item or item in seen:
                continue
            normalized.append(item)
            seen.add(item)
        return normalized

    def _normalize_context(self, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not payload:
            return {}
        if self.sanitize_freeform_context:
            return sanitize_privacy_context(payload)
        return self._deepcopy(dict(payload))

    def _new_ref(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:16]}"

    def _resolve_audit_ref(self, audit_trail_ref: Optional[str], *, category: str) -> str:
        if audit_trail_ref:
            return str(audit_trail_ref).strip()
        if self.require_audit_refs:
            return self._new_ref(f"audit-{category}")
        return ""

    def _resolve_policy_version(self, policy_version: Optional[str]) -> str:
        return str(policy_version or self.default_policy_version)

    def _append_event(self, *, category: str, action: str,
        payload: Mapping[str, Any],
        audit_trail_ref: Optional[str] = None,
        request_id: Optional[str] = None,
        subject_id: Optional[str] = None,
        record_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = {
            "event_id": self._new_ref("evt"),
            "timestamp": self._now(),
            "category": category,
            "action": action,
            "audit_trail_ref": self._resolve_audit_ref(audit_trail_ref, category=category),
            "request_id": request_id,
            "subject_id": subject_id,
            "record_id": record_id,
            "policy_id": policy_id,
            "policy_version": self._resolve_policy_version(policy_version),
            "payload": self._normalize_context(payload),
        }
        self._event_log.append(event)
        return self._deepcopy(event)

    def _ensure_consent_entry(self, subject_id: str, purpose: str) -> Dict[str, Any]:
        key = (subject_id, purpose)
        if key not in self._consent_index:
            self._consent_index[key] = {
                "subject_id": subject_id,
                "purpose": purpose,
                "current_status": "unknown",
                "artifact_ref": None,
                "policy_id": None,
                "policy_version": self.default_policy_version,
                "granted_at": None,
                "expires_at": None,
                "revoked_at": None,
                "allowed_contexts": [],
                "allowed_processors": [],
                "legal_basis": None,
                "metadata": {},
                "history": [],
                "last_event_id": None,
                "last_updated_at": None,
                "audit_trail_ref": None,
            }
        return self._consent_index[key]

    def _ensure_binding_entry(self, subject_id: str, purpose: str) -> Dict[str, Any]:
        key = (subject_id, purpose)
        if key not in self._purpose_binding_index:
            self._purpose_binding_index[key] = {
                "subject_id": subject_id,
                "purpose": purpose,
                "binding_ref": None,
                "source_context": None,
                "allowed_contexts": [],
                "allowed_actions": [],
                "policy_id": None,
                "policy_version": self.default_policy_version,
                "expires_at": None,
                "metadata": {},
                "history": [],
                "last_event_id": None,
                "last_updated_at": None,
                "audit_trail_ref": None,
            }
        return self._purpose_binding_index[key]

    def _ensure_decision_entry(self, request_id: str) -> Dict[str, Any]:
        if request_id not in self._decision_index:
            self._decision_index[request_id] = {
                "request_id": request_id,
                "latest_decision": None,
                "latest_stage": None,
                "latest_audit_trail_ref": None,
                "subject_id": None,
                "record_id": None,
                "policy_id": None,
                "policy_version": self.default_policy_version,
                "history": [],
                "last_updated_at": None,
            }
        return self._decision_index[request_id]

    def _ensure_retention_entry(self, record_id: str) -> Dict[str, Any]:
        if record_id not in self._retention_index:
            self._retention_index[record_id] = {
                "record_id": record_id,
                "subject_id": None,
                "policy_id": None,
                "policy_version": self.default_policy_version,
                "retention_days": None,
                "created_at": None,
                "delete_after_ts": None,
                "status": None,
                "legal_hold": False,
                "hold_ref": None,
                "metadata": {},
                "history": [],
                "last_event_id": None,
                "last_updated_at": None,
                "audit_trail_ref": None,
            }
        return self._retention_index[record_id]

    def _ensure_deletion_entry(self, record_id: str) -> Dict[str, Any]:
        if record_id not in self._deletion_index:
            self._deletion_index[record_id] = {
                "record_id": record_id,
                "status": None,
                "due_at": None,
                "workflow": self.default_deletion_workflow,
                "requested_by": None,
                "reason": None,
                "retry_at": None,
                "completed_at": None,
                "proof_ref": None,
                "tombstone_ref": None,
                "history": [],
                "last_event_id": None,
                "last_updated_at": None,
                "audit_trail_ref": None,
            }
        return self._deletion_index[record_id]

    def _ensure_shared_contract(self, request_id: str) -> Dict[str, Any]:
        if request_id not in self._shared_contract_index:
            self._shared_contract_index[request_id] = {
                "request_id": request_id,
                "privacy.sensitivity_score": None,
                "privacy.detected_entities": [],
                "privacy.redaction_actions": [],
                "privacy.retention_policy_id": None,
                "privacy.consent_status": None,
                "privacy.audit_trail_ref": None,
                "last_updated_at": None,
            }
        return self._shared_contract_index[request_id]
    
    
    def _ensure_redaction_entry(self, request_id: str) -> Dict[str, Any]:
        if request_id not in self._redaction_index:
            self._redaction_index[request_id] = {
                "request_id": request_id,
                "record_id": None,
                "latest_transformation_ref": None,
                "history": [],
                "last_updated_at": None,
            }
        return self._redaction_index[request_id]
    
    
    def _ensure_lineage_entry(self, request_id: str) -> Dict[str, Any]:
        if request_id not in self._lineage_index:
            self._lineage_index[request_id] = {
                "request_id": request_id,
                "record_ids": [],
                "parent_request_ids": [],
                "history": [],
                "last_updated_at": None,
            }
        return self._lineage_index[request_id]

    def _derive_consent_active(self, consent_entry: Mapping[str, Any], *, at_timestamp: Optional[float]) -> bool:
        status = str(consent_entry.get("current_status") or "unknown").lower()
        if status != "granted":
            return False

        expires_at = consent_entry.get("expires_at")
        if expires_at is not None:
            evaluation_ts = at_timestamp if at_timestamp is not None else self._now()
            if float(expires_at) <= float(evaluation_ts):
                return False
        return True

    def _derive_retention_state(self, obligation: Mapping[str, Any]) -> Dict[str, Any]:
        delete_after_ts = obligation.get("delete_after_ts")
        status = str(obligation.get("status") or "unknown")
        now_ts = self._now()
        overdue = bool(delete_after_ts is not None and delete_after_ts <= now_ts and status not in {"deleted", "legal_hold"})
        return {
            "is_overdue": overdue,
            "seconds_until_due": None if delete_after_ts is None else max(0.0, float(delete_after_ts) - now_ts),
        }

    def _handle_exception(self, exc: Exception, *, stage: str,
                          context: Optional[Dict[str, Any]] = None,
                          ) -> PrivacyError:
        if isinstance(exc, PrivacyError):
            return exc
        return normalize_privacy_exception(exc, stage=stage, context=context)

    # ---------------------------------------------------------------------
    # Write APIs
    # ---------------------------------------------------------------------
    def register_consent_artifact(self, *, subject_id: str, purpose: str, status: str, artifact_ref: str,
            legal_basis: Optional[str] = None,
            granted_at: Optional[float] = None,
            expires_at: Optional[float] = None,
            revoked_at: Optional[float] = None,
            allowed_contexts: Optional[Sequence[str]] = None,
            allowed_processors: Optional[Sequence[str]] = None,
            metadata: Optional[Mapping[str, Any]] = None,
            policy_id: Optional[str] = None,
            policy_version: Optional[str] = None,
            audit_trail_ref: Optional[str] = None,
        ) -> Dict[str, Any]:
            operation = "register_consent_artifact"
            try:
                self._require_enabled(operation)
                normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
                normalized_purpose = self._normalize_identity(purpose, "purpose")
                normalized_status = self._normalize_identity(status, "status").lower()
                normalized_artifact_ref = self._normalize_identity(artifact_ref, "artifact_ref")
    
                if normalized_status not in self.CONSENT_STATUSES:
                    raise ValueError(
                        f"'status' must be one of {sorted(self.CONSENT_STATUSES)}, received {normalized_status!r}"
                    )
    
                normalized_granted_at = self._normalize_optional_timestamp(granted_at, "granted_at")
                normalized_expires_at = self._normalize_optional_timestamp(expires_at, "expires_at")
                normalized_revoked_at = self._normalize_optional_timestamp(revoked_at, "revoked_at")
    
                if (
                    normalized_granted_at is not None
                    and normalized_expires_at is not None
                    and normalized_expires_at < normalized_granted_at
                ):
                    raise ValueError("'expires_at' cannot be earlier than 'granted_at'")
    
                with self._lock:
                    entry = self._ensure_consent_entry(normalized_subject_id, normalized_purpose)
                    event = self._append_event(
                        category="consent",
                        action="consent_artifact_registered",
                        payload={
                            "status": normalized_status,
                            "artifact_ref": normalized_artifact_ref,
                            "legal_basis": legal_basis,
                            "granted_at": normalized_granted_at,
                            "expires_at": normalized_expires_at,
                            "revoked_at": normalized_revoked_at,
                            "allowed_contexts": list(self._normalize_sequence(allowed_contexts)),
                            "allowed_processors": list(self._normalize_sequence(allowed_processors)),
                            "metadata": dict(metadata or {}),
                        },
                        audit_trail_ref=audit_trail_ref,
                        subject_id=normalized_subject_id,
                        policy_id=policy_id,
                        policy_version=policy_version,
                    )
    
                    entry.update(
                        {
                            "current_status": normalized_status,
                            "artifact_ref": normalized_artifact_ref,
                            "policy_id": policy_id,
                            "policy_version": self._resolve_policy_version(policy_version),
                            "granted_at": normalized_granted_at,
                            "expires_at": normalized_expires_at,
                            "revoked_at": normalized_revoked_at,
                            "allowed_contexts": self._normalize_sequence(allowed_contexts),
                            "allowed_processors": self._normalize_sequence(allowed_processors),
                            "legal_basis": legal_basis,
                            "metadata": self._normalize_context(metadata),
                            "last_event_id": event["event_id"],
                            "last_updated_at": event["timestamp"],
                            "audit_trail_ref": event["audit_trail_ref"],
                        }
                    )
                    entry["history"] = self._trim_history(entry["history"] + [event], self.max_history_per_subject_purpose)
    
                    self._artifact_index[normalized_artifact_ref] = {
                        "artifact_ref": normalized_artifact_ref,
                        "subject_id": normalized_subject_id,
                        "purpose": normalized_purpose,
                        "status": normalized_status,
                        "policy_id": policy_id,
                        "policy_version": self._resolve_policy_version(policy_version),
                        "audit_trail_ref": event["audit_trail_ref"],
                        "last_updated_at": event["timestamp"],
                    }
                    return self._deepcopy(entry)
            except Exception as exc:
                if isinstance(exc, PrivacyError):
                    raise PrivacyMemoryWriteError(
                        operation=operation,
                        details=exc,
                        context={"subject_id": subject_id, "purpose": purpose},
                    ) from exc
                raise self._handle_exception(
                    exc,
                    stage="privacy_memory.register_consent_artifact",
                    context={"subject_id": subject_id, "purpose": purpose},
                ) from exc

    def bind_purpose(self, *, subject_id: str, purpose: str, source_context: str,
            allowed_contexts: Optional[Sequence[str]] = None,
            allowed_actions: Optional[Sequence[str]] = None,
            binding_ref: Optional[str] = None,
            expires_at: Optional[float] = None,
            metadata: Optional[Mapping[str, Any]] = None,
            policy_id: Optional[str] = None,
            policy_version: Optional[str] = None,
            audit_trail_ref: Optional[str] = None,
        ) -> Dict[str, Any]:
            operation = "bind_purpose"
            try:
                self._require_enabled(operation)
                normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
                normalized_purpose = self._normalize_identity(purpose, "purpose")
                normalized_source_context = self._normalize_identity(source_context, "source_context")
                normalized_expires_at = self._normalize_optional_timestamp(expires_at, "expires_at")
    
                with self._lock:
                    entry = self._ensure_binding_entry(normalized_subject_id, normalized_purpose)
                    effective_binding_ref = binding_ref or self._new_ref("purpose-binding")
                    event = self._append_event(
                        category="purpose_binding",
                        action="purpose_bound",
                        payload={
                            "binding_ref": effective_binding_ref,
                            "source_context": normalized_source_context,
                            "allowed_contexts": self._normalize_sequence(allowed_contexts),
                            "allowed_actions": self._normalize_sequence(allowed_actions),
                            "expires_at": normalized_expires_at,
                            "metadata": dict(metadata or {}),
                        },
                        audit_trail_ref=audit_trail_ref,
                        subject_id=normalized_subject_id,
                        policy_id=policy_id,
                        policy_version=policy_version,
                    )
    
                    entry.update(
                        {
                            "binding_ref": effective_binding_ref,
                            "source_context": normalized_source_context,
                            "allowed_contexts": self._normalize_sequence(allowed_contexts),
                            "allowed_actions": self._normalize_sequence(allowed_actions),
                            "policy_id": policy_id,
                            "policy_version": self._resolve_policy_version(policy_version),
                            "expires_at": normalized_expires_at,
                            "metadata": self._normalize_context(metadata),
                            "last_event_id": event["event_id"],
                            "last_updated_at": event["timestamp"],
                            "audit_trail_ref": event["audit_trail_ref"],
                        }
                    )
                    entry["history"] = self._trim_history(
                        entry["history"] + [event],
                        self.max_history_per_subject_purpose,
                    )
                    return self._deepcopy(entry)
            except Exception as exc:
                if isinstance(exc, PrivacyError):
                    raise PrivacyMemoryWriteError(
                        operation=operation,
                        details=exc,
                        context={
                            "subject_id": subject_id,
                            "purpose": purpose,
                            "source_context": source_context,
                        },
                    ) from exc
                raise self._handle_exception(
                    exc,
                    stage="privacy_memory.bind_purpose",
                    context={
                        "subject_id": subject_id,
                        "purpose": purpose,
                        "source_context": source_context,
                    },
                ) from exc

    def write_request_contract(self, *, request_id: str,
        sensitivity_score: Optional[float] = None,
        detected_entities: Optional[Sequence[Any]] = None,
        redaction_actions: Optional[Sequence[Any]] = None,
        retention_policy_id: Optional[str] = None,
        consent_status: Optional[Any] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "write_request_contract"
        try:
            self._require_enabled(operation)
            normalized_request_id = self._normalize_identity(request_id, "request_id")
            normalized_sensitivity_score = None
            if sensitivity_score is not None:
                normalized_sensitivity_score = float(sensitivity_score)
                if normalized_sensitivity_score < 0.0 or normalized_sensitivity_score > 1.0:
                    raise ValueError("'sensitivity_score' must be between 0.0 and 1.0")
    
            with self._lock:
                contract = self._ensure_shared_contract(normalized_request_id)
                if sensitivity_score is not None:
                    contract["privacy.sensitivity_score"] = normalized_sensitivity_score
                if detected_entities is not None:
                    contract["privacy.detected_entities"] = self._normalize_context(
                        {"detected_entities": list(detected_entities)}
                    ).get("detected_entities", [])
                if redaction_actions is not None:
                    contract["privacy.redaction_actions"] = self._normalize_context(
                        {"redaction_actions": list(redaction_actions)}
                    ).get("redaction_actions", [])
                if retention_policy_id is not None:
                    contract["privacy.retention_policy_id"] = str(retention_policy_id)
                if consent_status is not None:
                    contract["privacy.consent_status"] = self._normalize_context(
                        {"consent_status": consent_status}
                    ).get("consent_status")
                if audit_trail_ref:
                    contract["privacy.audit_trail_ref"] = str(audit_trail_ref)
                contract["last_updated_at"] = self._now()
                return self._deepcopy(contract)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"request_id": request_id},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.write_request_contract",
                context={"request_id": request_id},
            ) from exc

    def record_privacy_decision(self, *, request_id: str, stage: str,
        decision: PrivacyDecision | str,
        rationale: str,
        subject_id: Optional[str] = None,
        record_id: Optional[str] = None,
        purpose: Optional[str] = None,
        sensitivity_score: Optional[float] = None,
        detected_entities: Optional[Sequence[Any]] = None,
        redaction_actions: Optional[Sequence[Any]] = None,
        retention_policy_id: Optional[str] = None,
        consent_status: Optional[Any] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        lineage_parent_request_id: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "record_privacy_decision"
        try:
            self._require_enabled(operation)
            normalized_request_id = self._normalize_identity(request_id, "request_id")
            normalized_stage = self._normalize_identity(stage, "stage")
            normalized_rationale = self._normalize_identity(rationale, "rationale")
    
            if isinstance(decision, PrivacyDecision):
                normalized_decision = decision.value
            else:
                normalized_decision = self._normalize_identity(str(decision), "decision").lower()
                if normalized_decision not in {item.value for item in PrivacyDecision}:
                    raise ValueError(
                        f"'decision' must be one of {[item.value for item in PrivacyDecision]}, "
                        f"received {normalized_decision!r}"
                    )
    
            contract = self.write_request_contract(
                request_id=normalized_request_id,
                sensitivity_score=sensitivity_score,
                detected_entities=detected_entities,
                redaction_actions=redaction_actions,
                retention_policy_id=retention_policy_id,
                consent_status=consent_status,
                audit_trail_ref=audit_trail_ref,
            )
    
            with self._lock:
                entry = self._ensure_decision_entry(normalized_request_id)
                event = self._append_event(
                    category="decision",
                    action="privacy_decision_recorded",
                    payload={
                        "stage": normalized_stage,
                        "decision": normalized_decision,
                        "rationale": normalized_rationale,
                        "purpose": purpose,
                        "shared_contract": contract,
                        "context": dict(context or {}),
                    },
                    audit_trail_ref=audit_trail_ref,
                    request_id=normalized_request_id,
                    subject_id=subject_id,
                    record_id=record_id,
                    policy_id=policy_id,
                    policy_version=policy_version,
                )
    
                entry.update(
                    {
                        "latest_decision": normalized_decision,
                        "latest_stage": normalized_stage,
                        "latest_audit_trail_ref": event["audit_trail_ref"],
                        "subject_id": subject_id,
                        "record_id": record_id,
                        "policy_id": policy_id,
                        "policy_version": self._resolve_policy_version(policy_version),
                        "last_updated_at": event["timestamp"],
                    }
                )
                entry["history"] = self._trim_history(
                    entry["history"] + [event],
                    self.max_decision_trace_length,
                )
    
                return self._deepcopy(entry)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"request_id": request_id, "stage": stage},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.record_privacy_decision",
                context={"request_id": request_id, "stage": stage},
            ) from exc

    def record_lineage_event(self, *, request_id: str, operation: str, stage: str,
        parent_request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        source_context: Optional[str] = None,
        destination_context: Optional[str] = None,
        subject_id: Optional[str] = None,
        purpose: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation_name = "record_lineage_event"
        try:
            self._require_enabled(operation_name)
            normalized_request_id = self._normalize_identity(request_id, "request_id")
            normalized_operation = self._normalize_identity(operation, "operation")
            normalized_stage = self._normalize_identity(stage, "stage")
    
            with self._lock:
                entry = self._ensure_lineage_entry(normalized_request_id)
                if parent_request_id:
                    normalized_parent_request_id = self._normalize_identity(parent_request_id, "parent_request_id")
                    parent_ids = list(entry["parent_request_ids"])
                    if normalized_parent_request_id not in parent_ids:
                        parent_ids.append(normalized_parent_request_id)
                    entry["parent_request_ids"] = parent_ids[-self.max_lineage_parents:]
                else:
                    normalized_parent_request_id = None
    
                if record_id:
                    normalized_record_id = self._normalize_identity(record_id, "record_id")
                    record_ids = list(entry["record_ids"])
                    if normalized_record_id not in record_ids:
                        record_ids.append(normalized_record_id)
                    entry["record_ids"] = record_ids[-self.max_history_per_record:]
                else:
                    normalized_record_id = None
    
                event = self._append_event(
                    category="lineage",
                    action="lineage_event_recorded",
                    payload={
                        "operation": normalized_operation,
                        "stage": normalized_stage,
                        "parent_request_id": normalized_parent_request_id,
                        "source_context": source_context,
                        "destination_context": destination_context,
                        "purpose": purpose,
                        "context": dict(context or {}),
                    },
                    audit_trail_ref=audit_trail_ref,
                    request_id=normalized_request_id,
                    subject_id=subject_id,
                    record_id=normalized_record_id,
                )
    
                entry["history"] = self._trim_history(
                    entry["history"] + [event],
                    self.max_history_per_record,
                )
                entry["last_updated_at"] = event["timestamp"]
                return self._deepcopy(entry)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"request_id": request_id, "stage": stage},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.record_lineage_event",
                context={"request_id": request_id, "stage": stage},
            ) from exc

    def store_redaction_transformation(self, *, request_id: str, record_id: str,
        masked_fields: Sequence[str],
        strategy: str,
        reason: str,
        transformation_ref: Optional[str] = None,
        original_field_count: Optional[int] = None,
        retained_field_count: Optional[int] = None,
        context: Optional[Mapping[str, Any]] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "store_redaction_transformation"
        try:
            self._require_enabled(operation)
            normalized_request_id = self._normalize_identity(request_id, "request_id")
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_strategy = self._normalize_identity(strategy, "strategy")
            normalized_reason = self._normalize_identity(reason, "reason")
            normalized_masked_fields = self._normalize_sequence(masked_fields)
            effective_transformation_ref = transformation_ref or self._new_ref("redaction")
    
            with self._lock:
                entry = self._ensure_redaction_entry(normalized_request_id)
                event = self._append_event(
                    category="redaction",
                    action="redaction_transformation_recorded",
                    payload={
                        "transformation_ref": effective_transformation_ref,
                        "masked_fields": normalized_masked_fields,
                        "strategy": normalized_strategy,
                        "reason": normalized_reason,
                        "original_field_count": original_field_count,
                        "retained_field_count": retained_field_count,
                        "context": dict(context or {}),
                    },
                    audit_trail_ref=audit_trail_ref,
                    request_id=normalized_request_id,
                    record_id=normalized_record_id,
                )
    
                entry["record_id"] = normalized_record_id
                entry["latest_transformation_ref"] = effective_transformation_ref
                entry["history"] = self._trim_history(
                    entry["history"] + [event],
                    self.max_redaction_history,
                )
                entry["last_updated_at"] = event["timestamp"]
    
                contract = self._ensure_shared_contract(normalized_request_id)
                contract["privacy.redaction_actions"] = self._normalize_context(
                    {"redaction_actions": normalized_masked_fields}
                ).get("redaction_actions", [])
                contract["privacy.audit_trail_ref"] = event["audit_trail_ref"]
                contract["last_updated_at"] = event["timestamp"]
    
                return self._deepcopy(entry)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"request_id": request_id, "record_id": record_id},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.store_redaction_transformation",
                context={"request_id": request_id, "record_id": record_id},
            ) from exc

    def create_retention_obligation(self, *, record_id: str, policy_id: str, retention_days: int,
        subject_id: Optional[str] = None,
        created_at: Optional[float] = None,
        delete_after_ts: Optional[float] = None,
        status: str = "active",
        legal_hold: bool = False,
        hold_ref: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "create_retention_obligation"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_policy_id = self._normalize_identity(policy_id, "policy_id")
            normalized_retention_days = int(retention_days)
            if normalized_retention_days <= 0:
                raise ValueError("'retention_days' must be a positive integer")
    
            normalized_created_at = self._normalize_optional_timestamp(created_at, "created_at") or self._now()
            normalized_delete_after_ts = self._normalize_optional_timestamp(delete_after_ts, "delete_after_ts")
            if normalized_delete_after_ts is None:
                normalized_delete_after_ts = normalized_created_at + (normalized_retention_days * 86400)
            normalized_status = self._normalize_identity(status, "status").lower()
            if normalized_status not in self.RETENTION_STATUSES:
                raise ValueError(
                    f"'status' must be one of {sorted(self.RETENTION_STATUSES)}, received {normalized_status!r}"
                )
    
            with self._lock:
                entry = self._ensure_retention_entry(normalized_record_id)
                event = self._append_event(
                    category="retention",
                    action="retention_obligation_created",
                    payload={
                        "retention_days": normalized_retention_days,
                        "created_at": normalized_created_at,
                        "delete_after_ts": normalized_delete_after_ts,
                        "status": normalized_status,
                        "legal_hold": bool(legal_hold),
                        "hold_ref": hold_ref,
                        "metadata": dict(metadata or {}),
                    },
                    audit_trail_ref=audit_trail_ref,
                    record_id=normalized_record_id,
                    subject_id=subject_id,
                    policy_id=normalized_policy_id,
                    policy_version=policy_version,
                )
    
                entry.update(
                    {
                        "subject_id": subject_id,
                        "policy_id": normalized_policy_id,
                        "policy_version": self._resolve_policy_version(policy_version),
                        "retention_days": normalized_retention_days,
                        "created_at": normalized_created_at,
                        "delete_after_ts": normalized_delete_after_ts,
                        "status": normalized_status,
                        "legal_hold": bool(legal_hold),
                        "hold_ref": hold_ref,
                        "metadata": self._normalize_context(metadata),
                        "last_event_id": event["event_id"],
                        "last_updated_at": event["timestamp"],
                        "audit_trail_ref": event["audit_trail_ref"],
                    }
                )
                entry["history"] = self._trim_history(
                    entry["history"] + [event],
                    self.max_history_per_record,
                )
                return self._deepcopy(entry)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"record_id": record_id, "policy_id": policy_id},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.create_retention_obligation",
                context={"record_id": record_id, "policy_id": policy_id},
            ) from exc

    def update_retention_status(self, *, record_id: str, status: str,
        reason: Optional[str] = None,
        hold_ref: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "update_retention_status"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_status = self._normalize_identity(status, "status").lower()
            if normalized_status not in self.RETENTION_STATUSES:
                raise ValueError(
                    f"'status' must be one of {sorted(self.RETENTION_STATUSES)}, received {normalized_status!r}"
                )
    
            with self._lock:
                entry = self._retention_index.get(normalized_record_id)
                if entry is None:
                    raise RetentionObligationMissingError(
                        record_id=normalized_record_id,
                        details="No retention obligation exists to update.",
                    )
    
                event = self._append_event(
                    category="retention",
                    action="retention_status_updated",
                    payload={
                        "status": normalized_status,
                        "reason": reason,
                        "hold_ref": hold_ref,
                    },
                    audit_trail_ref=audit_trail_ref,
                    record_id=normalized_record_id,
                    subject_id=entry.get("subject_id"),
                    policy_id=entry.get("policy_id"),
                    policy_version=entry.get("policy_version"),
                )
    
                entry["status"] = normalized_status
                entry["legal_hold"] = normalized_status == "legal_hold"
                if hold_ref is not None:
                    entry["hold_ref"] = hold_ref
                entry["last_event_id"] = event["event_id"]
                entry["last_updated_at"] = event["timestamp"]
                entry["audit_trail_ref"] = event["audit_trail_ref"]
                entry["history"] = self._trim_history(
                    entry["history"] + [event],
                    self.max_history_per_record,
                )
                return self._deepcopy(entry)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"record_id": record_id, "status": status},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.update_retention_status",
                context={"record_id": record_id, "status": status},
            ) from exc

    def schedule_deletion(self, *, record_id: str, reason: str,
        due_at: Optional[float] = None,
        workflow: Optional[str] = None,
        requested_by: str = "privacy_agent",
        retry_at: Optional[float] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "schedule_deletion"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_reason = self._normalize_identity(reason, "reason")
            normalized_requested_by = self._normalize_identity(requested_by, "requested_by")
            normalized_due_at = self._normalize_optional_timestamp(due_at, "due_at")
            normalized_retry_at = self._normalize_optional_timestamp(retry_at, "retry_at")
            normalized_workflow = str(workflow or self.default_deletion_workflow).strip() or self.default_deletion_workflow
    
            with self._lock:
                obligation = self._retention_index.get(normalized_record_id)
                if obligation is None:
                    raise RetentionObligationMissingError(
                        record_id=normalized_record_id,
                        details="Deletion cannot be scheduled without an existing retention obligation.",
                    )
    
                entry = self._ensure_deletion_entry(normalized_record_id)
                effective_due_at = normalized_due_at
                if effective_due_at is None:
                    effective_due_at = float(obligation.get("delete_after_ts") or self._now()) + self.deletion_grace_period_seconds
    
                event = self._append_event(
                    category="deletion",
                    action="deletion_scheduled",
                    payload={
                        "reason": normalized_reason,
                        "due_at": effective_due_at,
                        "workflow": normalized_workflow,
                        "requested_by": normalized_requested_by,
                        "retry_at": normalized_retry_at,
                    },
                    audit_trail_ref=audit_trail_ref,
                    record_id=normalized_record_id,
                    subject_id=obligation.get("subject_id"),
                    policy_id=obligation.get("policy_id"),
                    policy_version=obligation.get("policy_version"),
                )
    
                entry.update(
                    {
                        "status": "scheduled",
                        "due_at": effective_due_at,
                        "workflow": normalized_workflow,
                        "requested_by": normalized_requested_by,
                        "reason": normalized_reason,
                        "retry_at": normalized_retry_at,
                        "last_event_id": event["event_id"],
                        "last_updated_at": event["timestamp"],
                        "audit_trail_ref": event["audit_trail_ref"],
                    }
                )
                entry["history"] = self._trim_history(
                    entry["history"] + [event],
                    self.max_history_per_record,
                )
                return self._deepcopy(entry)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"record_id": record_id},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.schedule_deletion",
                context={"record_id": record_id},
            ) from exc

    def mark_deletion_completed(self, *, record_id: str,
        proof_ref: Optional[str] = None,
        tombstone_ref: Optional[str] = None,
        completed_at: Optional[float] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "mark_deletion_completed"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_completed_at = self._normalize_optional_timestamp(completed_at, "completed_at") or self._now()
    
            with self._lock:
                obligation = self._retention_index.get(normalized_record_id)
                if obligation is None:
                    raise RetentionObligationMissingError(
                        record_id=normalized_record_id,
                        details="Deletion completion requires an existing retention obligation.",
                    )
    
                entry = self._ensure_deletion_entry(normalized_record_id)
                effective_tombstone_ref = tombstone_ref
                if self.auto_create_tombstone_on_delete and not effective_tombstone_ref:
                    effective_tombstone_ref = self._new_ref("tombstone")
    
                event = self._append_event(
                    category="deletion",
                    action="deletion_completed",
                    payload={
                        "proof_ref": proof_ref,
                        "tombstone_ref": effective_tombstone_ref,
                        "completed_at": normalized_completed_at,
                    },
                    audit_trail_ref=audit_trail_ref,
                    record_id=normalized_record_id,
                    subject_id=obligation.get("subject_id"),
                    policy_id=obligation.get("policy_id"),
                    policy_version=obligation.get("policy_version"),
                )
    
                entry.update(
                    {
                        "status": "completed",
                        "completed_at": normalized_completed_at,
                        "proof_ref": proof_ref,
                        "tombstone_ref": effective_tombstone_ref,
                        "last_event_id": event["event_id"],
                        "last_updated_at": event["timestamp"],
                        "audit_trail_ref": event["audit_trail_ref"],
                    }
                )
                entry["history"] = self._trim_history(
                    entry["history"] + [event],
                    self.max_history_per_record,
                )
    
                obligation["status"] = "deleted"
                obligation["last_event_id"] = event["event_id"]
                obligation["last_updated_at"] = event["timestamp"]
                obligation["audit_trail_ref"] = event["audit_trail_ref"]
                obligation["history"] = self._trim_history(
                    obligation["history"] + [event],
                    self.max_history_per_record,
                )
                return self._deepcopy(entry)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"record_id": record_id},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.mark_deletion_completed",
                context={"record_id": record_id},
            ) from exc

    def mark_deletion_failed(self, *, record_id: str, reason: str,
        retry_at: Optional[float] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "mark_deletion_failed"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_reason = self._normalize_identity(reason, "reason")
            normalized_retry_at = self._normalize_optional_timestamp(retry_at, "retry_at")

            with self._lock:
                obligation = self._retention_index.get(normalized_record_id)
                if obligation is None:
                    raise RetentionObligationMissingError(
                        record_id=normalized_record_id,
                        details="Deletion failure cannot be recorded without a retention obligation.",
                    )

                entry = self._ensure_deletion_entry(normalized_record_id)
                event = self._append_event(
                    category="deletion",
                    action="deletion_failed",
                    payload={
                        "reason": normalized_reason,
                        "retry_at": normalized_retry_at,
                    },
                    audit_trail_ref=audit_trail_ref,
                    record_id=normalized_record_id,
                    subject_id=obligation.get("subject_id"),
                    policy_id=obligation.get("policy_id"),
                    policy_version=obligation.get("policy_version"),
                )

                entry.update(
                    {
                        "status": "failed",
                        "reason": normalized_reason,
                        "retry_at": normalized_retry_at,
                        "last_event_id": event["event_id"],
                        "last_updated_at": event["timestamp"],
                        "audit_trail_ref": event["audit_trail_ref"],
                    }
                )
                entry["history"] = self._trim_history(
                    entry["history"] + [event],
                    self.max_history_per_record,
                )
                return self._deepcopy(entry)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"record_id": record_id},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.mark_deletion_failed",
                context={"record_id": record_id},
            ) from exc

    # ---------------------------------------------------------------------
    # Retrieval APIs
    # ---------------------------------------------------------------------
    def consent_status(self, subject_id: str, purpose: str, *,
            at_timestamp: Optional[float] = None) -> Dict[str, Any]:
            operation = "consent_status"
            try:
                self._require_enabled(operation)
                normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
                normalized_purpose = self._normalize_identity(purpose, "purpose")
                normalized_at_timestamp = self._normalize_optional_timestamp(at_timestamp, "at_timestamp")
    
                with self._lock:
                    consent_entry = self._consent_index.get((normalized_subject_id, normalized_purpose))
                    binding_entry = self._purpose_binding_index.get((normalized_subject_id, normalized_purpose))
                    if consent_entry is None:
                        return {
                            "subject_id": normalized_subject_id,
                            "purpose": normalized_purpose,
                            "status": "unknown",
                            "is_active": False,
                            "artifact_ref": None,
                            "policy_id": None,
                            "policy_version": self.default_policy_version,
                            "audit_trail_ref": None,
                            "purpose_binding": self._deepcopy(binding_entry) if binding_entry else None,
                        }
    
                    return {
                        "subject_id": normalized_subject_id,
                        "purpose": normalized_purpose,
                        "status": consent_entry["current_status"],
                        "is_active": self._derive_consent_active(
                            consent_entry,
                            at_timestamp=normalized_at_timestamp,
                        ),
                        "artifact_ref": consent_entry.get("artifact_ref"),
                        "legal_basis": consent_entry.get("legal_basis"),
                        "granted_at": consent_entry.get("granted_at"),
                        "expires_at": consent_entry.get("expires_at"),
                        "revoked_at": consent_entry.get("revoked_at"),
                        "allowed_contexts": self._deepcopy(consent_entry.get("allowed_contexts", [])),
                        "allowed_processors": self._deepcopy(consent_entry.get("allowed_processors", [])),
                        "policy_id": consent_entry.get("policy_id"),
                        "policy_version": consent_entry.get("policy_version"),
                        "audit_trail_ref": consent_entry.get("audit_trail_ref"),
                        "purpose_binding": self._deepcopy(binding_entry) if binding_entry else None,
                    }
            except Exception as exc:
                raise self._handle_exception(
                    exc,
                    stage="privacy_memory.consent_status",
                    context={"subject_id": subject_id, "purpose": purpose},
                ) from exc

    def retention_obligation(self, record_id: str) -> Dict[str, Any]:
        operation = "retention_obligation"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            with self._lock:
                obligation = self._retention_index.get(normalized_record_id)
                if obligation is None:
                    error = RetentionObligationMissingError(
                        record_id=normalized_record_id,
                        details="No retention obligation was found for the requested record.",
                    )
                    if self.strict_mode:
                        raise error
                    return {
                        "record_id": normalized_record_id,
                        "status": "missing",
                        "error": error.to_public_dict(),
                    }
    
                derived_state = self._derive_retention_state(obligation)
                return {
                    **self._deepcopy(obligation),
                    **derived_state,
                    "deletion": self._deepcopy(self._deletion_index.get(normalized_record_id)),
                }
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise PrivacyMemoryWriteError(
                    operation=operation,
                    details=exc,
                    context={"record_id": record_id},
                ) from exc
            raise self._handle_exception(
                exc,
                stage="privacy_memory.retention_obligation",
                context={"record_id": record_id},
            ) from exc

    def privacy_decision_trace(self, request_id: str) -> Dict[str, Any]:
            operation = "privacy_decision_trace"
            try:
                self._require_enabled(operation)
                normalized_request_id = self._normalize_identity(request_id, "request_id")
                with self._lock:
                    trace = self._decision_index.get(normalized_request_id)
                    if trace is None:
                        return {
                            "request_id": normalized_request_id,
                            "history": [],
                            "shared_contract": self._deepcopy(
                                self._shared_contract_index.get(normalized_request_id)
                            ),
                            "lineage": self._deepcopy(self._lineage_index.get(normalized_request_id)),
                            "redaction": self._deepcopy(self._redaction_index.get(normalized_request_id)),
                        }
    
                    return {
                        **self._deepcopy(trace),
                        "shared_contract": self._deepcopy(
                            self._shared_contract_index.get(normalized_request_id)
                        ),
                        "lineage": self._deepcopy(self._lineage_index.get(normalized_request_id)),
                        "redaction": self._deepcopy(self._redaction_index.get(normalized_request_id)),
                    }
            except Exception as exc:
                raise self._handle_exception(
                    exc,
                    stage="privacy_memory.privacy_decision_trace",
                    context={"request_id": request_id},
                ) from exc

    # ---------------------------------------------------------------------
    # Evidence, summaries, and operational helpers
    # ---------------------------------------------------------------------
    def build_audit_evidence_bundle(self, *,
            request_id: Optional[str] = None,
            record_id: Optional[str] = None,
            subject_id: Optional[str] = None,
            purpose: Optional[str] = None,
        ) -> Dict[str, Any]:
            operation = "build_audit_evidence_bundle"
            try:
                self._require_enabled(operation)
                if not any([request_id, record_id, subject_id]):
                    raise ValueError("At least one of 'request_id', 'record_id', or 'subject_id' must be provided")
        
                normalized_request_id = self._normalize_identity(request_id, "request_id") if request_id else None
                normalized_record_id = self._normalize_identity(record_id, "record_id") if record_id else None
                normalized_subject_id = self._normalize_identity(subject_id, "subject_id") if subject_id else None
                normalized_purpose = self._normalize_identity(purpose, "purpose") if purpose else None
        
                with self._lock:
                    consent_bundle = None
                    if normalized_subject_id and normalized_purpose:
                        consent_bundle = self.consent_status(normalized_subject_id, normalized_purpose)
        
                    record_bundle = None
                    if normalized_record_id and normalized_record_id in self._retention_index:
                        record_bundle = self.retention_obligation(normalized_record_id)
        
                    request_bundle = None
                    if normalized_request_id:
                        request_bundle = self.privacy_decision_trace(normalized_request_id)
        
                    related_events = []
                    for event in self._event_log:
                        if normalized_request_id and event.get("request_id") == normalized_request_id:
                            related_events.append(self._deepcopy(event))
                            continue
                        if normalized_record_id and event.get("record_id") == normalized_record_id:
                            related_events.append(self._deepcopy(event))
                            continue
                        if normalized_subject_id and event.get("subject_id") == normalized_subject_id:
                            related_events.append(self._deepcopy(event))
        
                    return {
                        "request_id": normalized_request_id,
                        "record_id": normalized_record_id,
                        "subject_id": normalized_subject_id,
                        "purpose": normalized_purpose,
                        "consent": consent_bundle,
                        "retention": record_bundle,
                        "decision_trace": request_bundle,
                        "related_events": related_events,
                        "bundle_generated_at": self._now(),
                    }
            except Exception as exc:
                raise self._handle_exception(
                    exc,
                    stage="privacy_memory.build_audit_evidence_bundle",
                    context={
                        "request_id": request_id,
                        "record_id": record_id,
                        "subject_id": subject_id,
                        "purpose": purpose,
                    },
                ) from exc

    def stats(self) -> Dict[str, Any]:
        operation = "stats"
        try:
            self._require_enabled(operation)
            with self._lock:
                return {
                    "enabled": self.enabled,
                    "strict_mode": self.strict_mode,
                    "consent_entries": len(self._consent_index),
                    "purpose_bindings": len(self._purpose_binding_index),
                    "decision_traces": len(self._decision_index),
                    "retention_obligations": len(self._retention_index),
                    "deletion_records": len(self._deletion_index),
                    "redaction_records": len(self._redaction_index),
                    "lineage_records": len(self._lineage_index),
                    "events": len(self._event_log),
                    "state_export_allowed": self.allow_state_export,
                    "config_path": self.config.get("__config_path__"),
                }
        except Exception as exc:
            if isinstance(exc, (PrivacyError, ValueError, TypeError)):
                raise self._handle_exception(exc, stage="privacy_memory.stats") from exc
            raise

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
                    "consent_index": self._deepcopy(self._consent_index),
                    "purpose_binding_index": self._deepcopy(self._purpose_binding_index),
                    "decision_index": self._deepcopy(self._decision_index),
                    "retention_index": self._deepcopy(self._retention_index),
                    "deletion_index": self._deepcopy(self._deletion_index),
                    "redaction_index": self._deepcopy(self._redaction_index),
                    "lineage_index": self._deepcopy(self._lineage_index),
                    "shared_contract_index": self._deepcopy(self._shared_contract_index),
                    "artifact_index": self._deepcopy(self._artifact_index),
                    "event_log": self._deepcopy(self._event_log),
                }
        except Exception as exc:
            if isinstance(exc, (PrivacyError, ValueError, TypeError)):
                raise self._handle_exception(exc, stage="privacy_memory.export_state") from exc
            raise


if __name__ == "__main__":
    print("\n=== Running Privacy Memory ===\n")
    printer.status("TEST", "Privacy Memory initialized", "info")

    memory = PrivacyMemory()

    consent = memory.register_consent_artifact(
        subject_id="subject-001",
        purpose="support_resolution",
        status="granted",
        artifact_ref="consent-artifact-001",
        legal_basis="explicit_consent",
        granted_at=time.time(),
        allowed_contexts=["support_workspace", "privacy_review"],
        allowed_processors=["knowledge_agent", "execution_agent"],
        metadata={"channel": "chat", "collector": "privacy_agent"},
        policy_id="privacy-consent-policy",
        policy_version="2026.04",
    )
    printer.status("TEST", "Consent artifact registered", "info")

    purpose_binding = memory.bind_purpose(
        subject_id="subject-001",
        purpose="support_resolution",
        source_context="support_workspace",
        allowed_contexts=["support_workspace", "privacy_review"],
        allowed_actions=["read", "redact", "respond"],
        metadata={"cross_context_sharing": "restricted"},
        policy_id="purpose-binding-policy",
        policy_version="2026.04",
    )
    printer.status("TEST", "Purpose binding created", "info")

    decision = memory.record_privacy_decision(
        request_id="request-001",
        stage="execution.pre_tool_call",
        decision=PrivacyDecision.MODIFY,
        rationale="Sensitive fields present; payload minimized before downstream tool invocation.",
        subject_id="subject-001",
        record_id="record-001",
        purpose="support_resolution",
        sensitivity_score=0.92,
        detected_entities=["email", "phone", "account_id"],
        redaction_actions=["mask_email", "tokenize_phone"],
        retention_policy_id="retention-standard-30d",
        consent_status={"status": "granted", "artifact_ref": "consent-artifact-001"},
        policy_id="privacy-runtime-policy",
        policy_version="2026.04",
        context={"tool_name": "ticketing_connector", "payload": {"email": "user@example.com"}},
    )
    printer.status("TEST", "Privacy decision recorded", "info")

    lineage = memory.record_lineage_event(
        request_id="request-001",
        parent_request_id="request-root-001",
        record_id="record-001",
        operation="payload_transfer",
        stage="execution.pre_tool_call",
        source_context="chat_runtime",
        destination_context="ticketing_connector",
        subject_id="subject-001",
        purpose="support_resolution",
        context={"transformation": "minimized_payload"},
    )
    printer.status("TEST", "Lineage event recorded", "info")

    redaction = memory.store_redaction_transformation(
        request_id="request-001",
        record_id="record-001",
        masked_fields=["email", "phone"],
        strategy="tokenization_and_partial_masking",
        reason="least-data-required for downstream processor",
        original_field_count=8,
        retained_field_count=4,
        context={"token_map_ref": "token-map-001", "payload": {"phone": "+1-000-000-0000"}},
    )
    printer.status("TEST", "Redaction transformation stored", "info")

    retention = memory.create_retention_obligation(
        record_id="record-001",
        subject_id="subject-001",
        policy_id="retention-standard-30d",
        retention_days=30,
        metadata={"data_class": "customer_support_case", "origin": "chat_runtime"},
        policy_version="2026.04",
    )
    printer.status("TEST", "Retention obligation created", "info")

    deletion = memory.schedule_deletion(
        record_id="record-001",
        reason="Scheduled disposal after retention expiry",
        requested_by="privacy_agent",
    )
    printer.status("TEST", "Deletion scheduled", "info")

    deletion_complete = memory.mark_deletion_completed(
        record_id="record-001",
        proof_ref="deletion-proof-001",
    )
    printer.status("TEST", "Deletion completion recorded", "info")

    consent_snapshot = memory.consent_status("subject-001", "support_resolution")
    retention_snapshot = memory.retention_obligation("record-001")
    trace_snapshot = memory.privacy_decision_trace("request-001")
    evidence_bundle = memory.build_audit_evidence_bundle(
        request_id="request-001",
        record_id="record-001",
        subject_id="subject-001",
        purpose="support_resolution",
    )

    print(json.dumps({
        "consent": consent_snapshot,
        "purpose_binding": purpose_binding,
        "decision": decision,
        "lineage": lineage,
        "redaction": redaction,
        "retention": retention,
        "deletion": deletion,
        "deletion_complete": deletion_complete,
        "trace": trace_snapshot,
        "evidence_bundle": evidence_bundle,
        "stats": memory.stats(),
    }, indent=2, default=str))

    print("\n=== Test ran successfully ===\n")