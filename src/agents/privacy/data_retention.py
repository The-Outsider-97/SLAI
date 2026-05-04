"""
- Time-bound retention enforcement.
- Deletion workflows and tombstone tracking.
- “Right to be forgotten” action hooks.
"""

from __future__ import annotations

import json
import time

from typing import Any, Dict, Mapping, Optional, Sequence

from .utils import (get_config_section, load_global_config,
                    # Error
                    DeletionSlaViolationError, RetentionObligationMissingError, PrivacyError,
                    DeletionWorkflowError, PolicyEvaluationError, PrivacyDecision,
                    PrivacyConfigurationError, PrivacyMemoryWriteError, RetentionViolationError,
                    normalize_privacy_exception, sanitize_privacy_context)
from .privacy_memory import PrivacyMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Data Retention and Deletion Governance")
printer = PrettyPrinter


class DataRetention:
    """Retention governance runtime for time-bound storage and deletion controls.

    The retention layer translates policy into runtime behavior across four related
    responsibilities:
    1. establishing retention obligations for records,
    2. continuously evaluating whether a record may still be retained,
    3. orchestrating deletion workflows and tombstone/proof tracking, and
    4. handling right-to-be-forgotten requests in a policy-aware way.

    The module is intentionally state-light and delegates durable state to
    ``PrivacyMemory`` so consent, auditability, and retention decisions converge on a
    shared privacy evidence trail.
    """

    RETENTION_STATUSES = {"active", "due", "expired", "deleted", "legal_hold", "missing"}
    DELETION_STATUSES = {"scheduled", "in_progress", "completed", "failed", "cancelled", "legal_hold"}

    def __init__(self) -> None:
        self.config = load_global_config()
        self.retention_config = get_config_section("data_retention")
        self.memory = PrivacyMemory()

        self.enabled = bool(self.retention_config.get("enabled", True))
        self.strict_mode = bool(self.retention_config.get("strict_mode", True))
        self.sanitize_freeform_context = bool(self.retention_config.get("sanitize_freeform_context", True))
        self.record_decisions_in_memory = bool(self.retention_config.get("record_decisions_in_memory", True))
        self.write_shared_contract = bool(self.retention_config.get("write_shared_contract", True))
        self.auto_update_due_status = bool(self.retention_config.get("auto_update_due_status", True))
        self.auto_schedule_due_deletion = bool(self.retention_config.get("auto_schedule_due_deletion", True))
        self.auto_update_deleted_status = bool(self.retention_config.get("auto_update_deleted_status", True))
        self.allow_legal_hold_override = bool(self.retention_config.get("allow_legal_hold_override", False))
        self.right_to_be_forgotten_requires_reason = bool(
            self.retention_config.get("right_to_be_forgotten_requires_reason", True)
        )
        self.immediate_forget_request_deletion = bool(
            self.retention_config.get("immediate_forget_request_deletion", True)
        )
        self.require_subject_id_for_obligation = bool(
            self.retention_config.get("require_subject_id_for_obligation", False)
        )
        self.require_policy_id = bool(self.retention_config.get("require_policy_id", True))

        self.default_policy_version = str(self.retention_config.get("default_policy_version", "v1"))
        self.default_decision_stage = str(
            self.retention_config.get("default_decision_stage", "retention.runtime_gate")
        )
        self.default_deletion_reason = str(
            self.retention_config.get("default_deletion_reason", "retention_expired")
        )
        self.default_requested_by = str(
            self.retention_config.get("default_requested_by", "privacy_agent")
        )
        self.default_deletion_workflow = str(
            self.retention_config.get(
                "default_deletion_workflow",
                self.memory.default_deletion_workflow,
            )
        )

        self.max_metadata_fields = int(self.retention_config.get("max_metadata_fields", 100))
        self.max_record_tags = int(self.retention_config.get("max_record_tags", 100))
        self.max_related_records = int(self.retention_config.get("max_related_records", 100))
        self.deletion_sla_seconds = float(self.retention_config.get("deletion_sla_seconds", 86400))
        self.overdue_grace_seconds = float(self.retention_config.get("overdue_grace_seconds", 0))

        self._validate_config()
        logger.info("DataRetention initialized with production-grade retention governance controls.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        numeric_fields = {
            "max_metadata_fields": self.max_metadata_fields,
            "max_record_tags": self.max_record_tags,
            "max_related_records": self.max_related_records,
        }
        for field_name, value in numeric_fields.items():
            if value <= 0:
                raise PrivacyConfigurationError(
                    section="data_retention",
                    details=f"'{field_name}' must be a positive integer, received {value!r}",
                )

        float_fields = {
            "deletion_sla_seconds": self.deletion_sla_seconds,
            "overdue_grace_seconds": self.overdue_grace_seconds,
        }
        for field_name, value in float_fields.items():
            if value < 0:
                raise PrivacyConfigurationError(
                    section="data_retention",
                    details=f"'{field_name}' must be greater than or equal to zero, received {value!r}",
                )

        if not self.default_decision_stage.strip():
            raise PrivacyConfigurationError(
                section="data_retention",
                details="'default_decision_stage' must be a non-empty string.",
            )
        if not self.default_deletion_reason.strip():
            raise PrivacyConfigurationError(
                section="data_retention",
                details="'default_deletion_reason' must be a non-empty string.",
            )

    def _require_enabled(self, operation: str) -> None:
        if not self.enabled:
            raise PolicyEvaluationError(
                stage=operation,
                details="data_retention is disabled by configuration",
                context={"config_section": "data_retention"},
            )

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _normalize_identity(value: Any, field_name: str) -> str:
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
    def _normalize_sequence(values: Optional[Sequence[Any]], *, limit: int, field_name: str) -> list[str]:
        if not values:
            return []
        normalized: list[str] = []
        seen = set()
        for value in values:
            item = str(value).strip()
            if not item or item in seen:
                continue
            normalized.append(item)
            seen.add(item)
            if len(normalized) > limit:
                raise ValueError(f"'{field_name}' exceeds the configured limit of {limit}")
        return normalized

    def _normalize_metadata(self, metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not metadata:
            return {}
        if len(metadata) > self.max_metadata_fields:
            raise ValueError(
                f"'metadata' exceeds the configured limit of {self.max_metadata_fields} fields"
            )
        normalized = dict(metadata)
        if self.sanitize_freeform_context:
            return sanitize_privacy_context(normalized)
        return normalized

    def _normalize_context(self, context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not context:
            return {}
        if self.sanitize_freeform_context:
            return sanitize_privacy_context(context)
        return dict(context)

    def _handle_exception(self, exc: Exception, *, stage: str,
                          context: Optional[Dict[str, Any]] = None) -> PrivacyError:
        if isinstance(exc, PrivacyError):
            return exc
        return normalize_privacy_exception(exc, stage=stage, context=context)

    def _derive_snapshot(self, obligation: Mapping[str, Any], *, as_of: Optional[float] = None) -> Dict[str, Any]:
        now_ts = self._now() if as_of is None else float(as_of)
        created_at = obligation.get("created_at")
        delete_after_ts = obligation.get("delete_after_ts")
        retention_days = obligation.get("retention_days")
        status = str(obligation.get("status") or "missing").lower()
        deletion = obligation.get("deletion") or {}
        deletion_status = str((deletion or {}).get("status") or "none").lower()
        legal_hold = bool(obligation.get("legal_hold", False)) or deletion_status == "legal_hold"

        age_seconds = None
        age_days = None
        if created_at is not None:
            age_seconds = max(0.0, now_ts - float(created_at))
            age_days = age_seconds / 86400.0

        seconds_until_due = None
        is_due = False
        is_overdue = False
        if delete_after_ts is not None:
            seconds_until_due = float(delete_after_ts) - now_ts
            is_due = seconds_until_due <= 0
            is_overdue = seconds_until_due <= (-1.0 * self.overdue_grace_seconds)

        scheduled_due_at = deletion.get("due_at") if deletion else None
        deletion_sla_breached = False
        if scheduled_due_at is not None and deletion_status not in {"completed", "cancelled"}:
            deletion_sla_breached = now_ts > (float(scheduled_due_at) + self.deletion_sla_seconds)

        decision = PrivacyDecision.ALLOW
        rationale = "Retention obligation is active and within the allowed retention window."
        actions: list[str] = []

        if status == "deleted" or deletion_status == "completed":
            decision = PrivacyDecision.BLOCK
            rationale = "Record has already been deleted and must not be processed further."
        elif legal_hold:
            decision = PrivacyDecision.ESCALATE
            rationale = "Record is under legal hold; deletion is blocked until the hold is released."
        elif is_overdue:
            decision = PrivacyDecision.BLOCK
            rationale = "Record has exceeded its retention window and must be queued for deletion."
            actions.append("schedule_deletion")
        elif is_due:
            decision = PrivacyDecision.MODIFY
            rationale = "Record has reached its retention boundary and should transition into deletion handling."
            actions.append("schedule_deletion")
        elif status == "due":
            decision = PrivacyDecision.MODIFY
            rationale = "Record is marked due for deletion and should not be retained indefinitely."
        elif deletion_sla_breached:
            decision = PrivacyDecision.ESCALATE
            rationale = "Deletion is scheduled but the deletion SLA has been breached."
            actions.append("escalate_deletion_sla")

        return {
            **dict(obligation),
            "age_seconds": age_seconds,
            "age_days": age_days,
            "seconds_until_due": seconds_until_due,
            "is_due": is_due,
            "is_overdue": is_overdue,
            "deletion_sla_breached": deletion_sla_breached,
            "decision": decision.value,
            "rationale": rationale,
            "recommended_actions": actions,
            "evaluated_at": now_ts,
        }

    def _write_retention_contract(self, *, request_id: Optional[str], policy_id: Optional[str],
                                  audit_trail_ref: Optional[str]) -> None:
        if not self.write_shared_contract or not request_id:
            return
        try:
            self.memory.write_request_contract(
                request_id=request_id,
                retention_policy_id=policy_id,
                audit_trail_ref=audit_trail_ref,
            )
        except Exception as exc:
            raise PrivacyMemoryWriteError(
                operation="write_request_contract",
                details=exc,
                context={"request_id": request_id, "policy_id": policy_id},
            ) from exc

    def _record_runtime_decision(self, *,
        request_id: Optional[str],
        record_id: str,
        subject_id: Optional[str],
        policy_id: Optional[str],
        policy_version: Optional[str],
        stage: str,
        decision: PrivacyDecision,
        rationale: str,
        retention_policy_id: Optional[str],
        context: Optional[Mapping[str, Any]] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> None:
        if not self.record_decisions_in_memory or not request_id:
            return
        try:
            self.memory.record_privacy_decision(
                request_id=request_id,
                stage=stage,
                decision=decision,
                rationale=rationale,
                subject_id=subject_id,
                record_id=record_id,
                retention_policy_id=retention_policy_id,
                policy_id=policy_id,
                policy_version=policy_version,
                context=self._normalize_context(context),
                audit_trail_ref=audit_trail_ref,
            )
        except Exception as exc:
            raise PrivacyMemoryWriteError(
                operation="record_privacy_decision",
                details=exc,
                context={"request_id": request_id, "record_id": record_id, "stage": stage},
            ) from exc

    # ------------------------------------------------------------------
    # Write APIs
    # ------------------------------------------------------------------
    def create_retention_obligation(self, *,
        record_id: str,
        policy_id: str,
        retention_days: int,
        subject_id: Optional[str] = None,
        request_id: Optional[str] = None,
        created_at: Optional[float] = None,
        delete_after_ts: Optional[float] = None,
        status: str = "active",
        legal_hold: bool = False,
        hold_ref: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        record_tags: Optional[Sequence[Any]] = None,
        related_records: Optional[Sequence[Any]] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        decision_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "create_retention_obligation"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_policy_id = self._normalize_identity(policy_id, "policy_id") if policy_id is not None else ""
            if self.require_policy_id and not normalized_policy_id:
                raise ValueError("'policy_id' is required by configuration")
            if self.require_subject_id_for_obligation and not str(subject_id or "").strip():
                raise ValueError("'subject_id' is required by configuration for retention obligations")

            normalized_metadata = self._normalize_metadata(metadata)
            normalized_tags = self._normalize_sequence(
                record_tags,
                limit=self.max_record_tags,
                field_name="record_tags",
            )
            normalized_related_records = self._normalize_sequence(
                related_records,
                limit=self.max_related_records,
                field_name="related_records",
            )

            obligation = self.memory.create_retention_obligation(
                record_id=normalized_record_id,
                policy_id=normalized_policy_id,
                retention_days=int(retention_days),
                subject_id=subject_id,
                created_at=created_at,
                delete_after_ts=delete_after_ts,
                status=status,
                legal_hold=legal_hold,
                hold_ref=hold_ref,
                metadata={
                    **normalized_metadata,
                    "record_tags": normalized_tags,
                    "related_records": normalized_related_records,
                },
                policy_version=policy_version or self.default_policy_version,
                audit_trail_ref=audit_trail_ref,
            )

            self._write_retention_contract(
                request_id=request_id,
                policy_id=normalized_policy_id,
                audit_trail_ref=obligation.get("audit_trail_ref") or audit_trail_ref,
            )
            self._record_runtime_decision(
                request_id=request_id,
                record_id=normalized_record_id,
                subject_id=subject_id,
                policy_id=normalized_policy_id,
                policy_version=policy_version or self.default_policy_version,
                stage=decision_stage or "retention.obligation_create",
                decision=PrivacyDecision.ALLOW,
                rationale="Retention obligation created and bound to the record.",
                retention_policy_id=normalized_policy_id,
                context={
                    "retention_days": int(retention_days),
                    "delete_after_ts": obligation.get("delete_after_ts"),
                    "status": obligation.get("status"),
                },
                audit_trail_ref=obligation.get("audit_trail_ref") or audit_trail_ref,
            )
            return obligation
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.create_retention_obligation",
                context={"record_id": record_id, "policy_id": policy_id, "request_id": request_id},
            ) from exc

    def retention_status(self, *, record_id: str,
        request_id: Optional[str] = None,
        as_of: Optional[float] = None,
        auto_remediate: Optional[bool] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "retention_status"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            snapshot = self.memory.retention_obligation(normalized_record_id)
            derived = self._derive_snapshot(snapshot, as_of=as_of)
            effective_auto_remediate = self.auto_schedule_due_deletion if auto_remediate is None else bool(auto_remediate)

            if derived["is_due"] and self.auto_update_due_status and derived.get("status") == "active":
                updated = self.memory.update_retention_status(
                    record_id=normalized_record_id,
                    status="due",
                    reason="retention boundary reached during runtime evaluation",
                    audit_trail_ref=audit_trail_ref,
                )
                snapshot = self.memory.retention_obligation(normalized_record_id)
                derived = self._derive_snapshot(snapshot, as_of=as_of)
                derived["status_update"] = updated

            deletion = derived.get("deletion") or {}
            deletion_status = str((deletion or {}).get("status") or "none").lower()
            if (
                effective_auto_remediate
                and derived["is_due"]
                and not derived.get("legal_hold")
                and deletion_status not in {"scheduled", "in_progress", "completed"}
            ):
                scheduled = self.memory.schedule_deletion(
                    record_id=normalized_record_id,
                    reason=self.default_deletion_reason,
                    requested_by=self.default_requested_by,
                    workflow=self.default_deletion_workflow,
                    audit_trail_ref=audit_trail_ref,
                )
                snapshot = self.memory.retention_obligation(normalized_record_id)
                derived = self._derive_snapshot(snapshot, as_of=as_of)
                derived["deletion_schedule"] = scheduled

            self._write_retention_contract(
                request_id=request_id,
                policy_id=derived.get("policy_id"),
                audit_trail_ref=derived.get("audit_trail_ref") or audit_trail_ref,
            )
            self._record_runtime_decision(
                request_id=request_id,
                record_id=normalized_record_id,
                subject_id=derived.get("subject_id"),
                policy_id=derived.get("policy_id"),
                policy_version=derived.get("policy_version"),
                stage=self.default_decision_stage,
                decision=PrivacyDecision(derived["decision"]),
                rationale=derived["rationale"],
                retention_policy_id=derived.get("policy_id"),
                context={
                    "is_due": derived["is_due"],
                    "is_overdue": derived["is_overdue"],
                    "deletion_sla_breached": derived["deletion_sla_breached"],
                    "recommended_actions": derived["recommended_actions"],
                },
                audit_trail_ref=derived.get("audit_trail_ref") or audit_trail_ref,
            )
            return derived
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.retention_status",
                context={"record_id": record_id, "request_id": request_id},
            ) from exc

    def enforce_retention(self, *,
        record_id: str,
        request_id: Optional[str] = None,
        as_of: Optional[float] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "enforce_retention"
        try:
            self._require_enabled(operation)
            evaluation = self.retention_status(
                record_id=record_id,
                request_id=request_id,
                as_of=as_of,
                auto_remediate=True,
                audit_trail_ref=audit_trail_ref,
            )

            if evaluation.get("deletion_sla_breached"):
                raise DeletionSlaViolationError(
                    record_id=self._normalize_identity(record_id, "record_id"),
                    due_timestamp=float((evaluation.get("deletion") or {}).get("due_at") or self._now()),
                    status=str((evaluation.get("deletion") or {}).get("status") or "scheduled"),
                    context={
                        "policy_id": evaluation.get("policy_id"),
                        "request_id": request_id,
                    },
                )

            if evaluation.get("is_overdue") and not evaluation.get("legal_hold"):
                raise RetentionViolationError(
                    record_id=self._normalize_identity(record_id, "record_id"),
                    policy_id=str(evaluation.get("policy_id") or "unknown-policy"),
                    retention_days=int(evaluation.get("retention_days") or 0),
                    age_days=int((evaluation.get("age_days") or 0.0)),
                    context={
                        "request_id": request_id,
                        "delete_after_ts": evaluation.get("delete_after_ts"),
                    },
                )

            return evaluation
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.enforce_retention",
                context={"record_id": record_id, "request_id": request_id},
            ) from exc

    def apply_legal_hold(self, *,
        record_id: str, hold_ref: str, reason: str,
        request_id: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "apply_legal_hold"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_hold_ref = self._normalize_identity(hold_ref, "hold_ref")
            normalized_reason = self._normalize_identity(reason, "reason")

            updated = self.memory.update_retention_status(
                record_id=normalized_record_id,
                status="legal_hold",
                reason=normalized_reason,
                hold_ref=normalized_hold_ref,
                audit_trail_ref=audit_trail_ref,
            )
            self._record_runtime_decision(
                request_id=request_id,
                record_id=normalized_record_id,
                subject_id=updated.get("subject_id"),
                policy_id=updated.get("policy_id"),
                policy_version=updated.get("policy_version"),
                stage="retention.legal_hold_apply",
                decision=PrivacyDecision.ESCALATE,
                rationale="Legal hold applied; deletion must remain blocked until the hold is released.",
                retention_policy_id=updated.get("policy_id"),
                context={"hold_ref": normalized_hold_ref, "reason": normalized_reason},
                audit_trail_ref=updated.get("audit_trail_ref") or audit_trail_ref,
            )
            return self.memory.retention_obligation(normalized_record_id)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.apply_legal_hold",
                context={"record_id": record_id, "request_id": request_id},
            ) from exc

    def release_legal_hold(self, *, record_id: str, reason: str,
        request_id: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "release_legal_hold"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_reason = self._normalize_identity(reason, "reason")
            current = self.memory.retention_obligation(normalized_record_id)
            if str(current.get("status") or "").lower() != "legal_hold":
                raise PolicyEvaluationError(
                    stage=operation,
                    details="record is not currently under legal hold",
                    context={"record_id": normalized_record_id, "status": current.get("status")},
                )

            updated = self.memory.update_retention_status(
                record_id=normalized_record_id,
                status="active",
                reason=normalized_reason,
                hold_ref=None,
                audit_trail_ref=audit_trail_ref,
            )
            self._record_runtime_decision(
                request_id=request_id,
                record_id=normalized_record_id,
                subject_id=updated.get("subject_id"),
                policy_id=updated.get("policy_id"),
                policy_version=updated.get("policy_version"),
                stage="retention.legal_hold_release",
                decision=PrivacyDecision.MODIFY,
                rationale="Legal hold released; retention returns to policy-based evaluation.",
                retention_policy_id=updated.get("policy_id"),
                context={"reason": normalized_reason},
                audit_trail_ref=updated.get("audit_trail_ref") or audit_trail_ref,
            )
            return self.memory.retention_obligation(normalized_record_id)
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.release_legal_hold",
                context={"record_id": record_id, "request_id": request_id},
            ) from exc

    def schedule_deletion(self, *, record_id: str,
        reason: Optional[str] = None,
        request_id: Optional[str] = None,
        workflow: Optional[str] = None,
        requested_by: Optional[str] = None,
        due_at: Optional[float] = None,
        retry_at: Optional[float] = None,
        allow_legal_hold_override: bool = False,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "schedule_deletion"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            snapshot = self.memory.retention_obligation(normalized_record_id)
            if snapshot.get("status") == "missing":
                raise RetentionObligationMissingError(
                    record_id=normalized_record_id,
                    details="Deletion scheduling requires an existing retention obligation.",
                )
            if snapshot.get("legal_hold") and not (allow_legal_hold_override and self.allow_legal_hold_override):
                raise DeletionWorkflowError(
                    record_id=normalized_record_id,
                    workflow=str(workflow or self.default_deletion_workflow),
                    details="Cannot schedule deletion while a legal hold is active.",
                    context={"status": snapshot.get("status")},
                )

            scheduled = self.memory.schedule_deletion(
                record_id=normalized_record_id,
                reason=str(reason or self.default_deletion_reason),
                due_at=self._normalize_optional_timestamp(due_at, "due_at"),
                workflow=str(workflow or self.default_deletion_workflow),
                requested_by=str(requested_by or self.default_requested_by),
                retry_at=self._normalize_optional_timestamp(retry_at, "retry_at"),
                audit_trail_ref=audit_trail_ref,
            )

            if self.auto_update_due_status and str(snapshot.get("status") or "").lower() == "active":
                self.memory.update_retention_status(
                    record_id=normalized_record_id,
                    status="due",
                    reason="Deletion scheduled from retention governance layer.",
                    audit_trail_ref=audit_trail_ref,
                )

            updated = self.memory.retention_obligation(normalized_record_id)
            self._record_runtime_decision(
                request_id=request_id,
                record_id=normalized_record_id,
                subject_id=updated.get("subject_id"),
                policy_id=updated.get("policy_id"),
                policy_version=updated.get("policy_version"),
                stage="retention.deletion_schedule",
                decision=PrivacyDecision.ESCALATE,
                rationale="Deletion workflow scheduled under retention governance.",
                retention_policy_id=updated.get("policy_id"),
                context={
                    "deletion": scheduled,
                    "reason": reason or self.default_deletion_reason,
                },
                audit_trail_ref=scheduled.get("audit_trail_ref") or audit_trail_ref,
            )
            return updated
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.schedule_deletion",
                context={"record_id": record_id, "request_id": request_id},
            ) from exc

    def mark_deletion_completed(self, *, record_id: str,
        proof_ref: Optional[str] = None,
        tombstone_ref: Optional[str] = None,
        completed_at: Optional[float] = None,
        request_id: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "mark_deletion_completed"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            completed = self.memory.mark_deletion_completed(
                record_id=normalized_record_id,
                proof_ref=proof_ref,
                tombstone_ref=tombstone_ref,
                completed_at=self._normalize_optional_timestamp(completed_at, "completed_at"),
                audit_trail_ref=audit_trail_ref,
            )
            updated = self.memory.retention_obligation(normalized_record_id)
            if self.auto_update_deleted_status and str(updated.get("status") or "").lower() != "deleted":
                updated = self.memory.retention_obligation(normalized_record_id)

            self._record_runtime_decision(
                request_id=request_id,
                record_id=normalized_record_id,
                subject_id=updated.get("subject_id"),
                policy_id=updated.get("policy_id"),
                policy_version=updated.get("policy_version"),
                stage="retention.deletion_complete",
                decision=PrivacyDecision.BLOCK,
                rationale="Deletion completed; the record must not be retained or processed further.",
                retention_policy_id=updated.get("policy_id"),
                context={"deletion": completed},
                audit_trail_ref=completed.get("audit_trail_ref") or audit_trail_ref,
            )
            return updated
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.mark_deletion_completed",
                context={"record_id": record_id, "request_id": request_id},
            ) from exc

    def mark_deletion_failed(self, *, record_id: str, reason: str,
        request_id: Optional[str] = None,
        retry_at: Optional[float] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "mark_deletion_failed"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_reason = self._normalize_identity(reason, "reason")
            failure = self.memory.mark_deletion_failed(
                record_id=normalized_record_id,
                reason=normalized_reason,
                retry_at=self._normalize_optional_timestamp(retry_at, "retry_at"),
                audit_trail_ref=audit_trail_ref,
            )
            updated = self.memory.retention_obligation(normalized_record_id)
            self._record_runtime_decision(
                request_id=request_id,
                record_id=normalized_record_id,
                subject_id=updated.get("subject_id"),
                policy_id=updated.get("policy_id"),
                policy_version=updated.get("policy_version"),
                stage="retention.deletion_failed",
                decision=PrivacyDecision.ESCALATE,
                rationale="Deletion workflow failed and requires remediation.",
                retention_policy_id=updated.get("policy_id"),
                context={"deletion": failure, "reason": normalized_reason},
                audit_trail_ref=failure.get("audit_trail_ref") or audit_trail_ref,
            )
            return updated
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.mark_deletion_failed",
                context={"record_id": record_id, "request_id": request_id},
            ) from exc

    def process_forget_request(self, *, record_id: str, request_id: str,
        subject_id: Optional[str] = None,
        reason: Optional[str] = None,
        allow_legal_hold_override: bool = False,
        audit_trail_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        operation = "process_forget_request"
        try:
            self._require_enabled(operation)
            normalized_record_id = self._normalize_identity(record_id, "record_id")
            normalized_request_id = self._normalize_identity(request_id, "request_id")
            normalized_reason = str(reason or "").strip()
            if self.right_to_be_forgotten_requires_reason and not normalized_reason:
                raise ValueError("'reason' is required for right-to-be-forgotten workflows")
            effective_reason = normalized_reason or "data_subject_deletion_request"

            snapshot = self.memory.retention_obligation(normalized_record_id)
            if subject_id and snapshot.get("subject_id") and str(snapshot.get("subject_id")) != str(subject_id):
                raise PolicyEvaluationError(
                    stage=operation,
                    details="subject_id does not match the existing retention obligation",
                    context={
                        "record_id": normalized_record_id,
                        "expected_subject_id": snapshot.get("subject_id"),
                        "provided_subject_id": subject_id,
                    },
                )

            if str(snapshot.get("status") or "").lower() != "due":
                self.memory.update_retention_status(
                    record_id=normalized_record_id,
                    status="due",
                    reason="Right-to-be-forgotten request received.",
                    audit_trail_ref=audit_trail_ref,
                )

            due_at = self._now() if self.immediate_forget_request_deletion else None
            updated = self.schedule_deletion(
                record_id=normalized_record_id,
                reason=effective_reason,
                request_id=normalized_request_id,
                requested_by="privacy_subject_request",
                due_at=due_at,
                allow_legal_hold_override=allow_legal_hold_override,
                audit_trail_ref=audit_trail_ref,
            )
            self._record_runtime_decision(
                request_id=normalized_request_id,
                record_id=normalized_record_id,
                subject_id=updated.get("subject_id") or subject_id,
                policy_id=updated.get("policy_id"),
                policy_version=updated.get("policy_version"),
                stage="retention.right_to_be_forgotten",
                decision=PrivacyDecision.BLOCK,
                rationale="Right-to-be-forgotten request accepted; retention must yield to deletion workflow.",
                retention_policy_id=updated.get("policy_id"),
                context={"reason": effective_reason, "immediate_due": due_at is not None},
                audit_trail_ref=updated.get("audit_trail_ref") or audit_trail_ref,
            )
            return updated
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage="data_retention.process_forget_request",
                context={"record_id": record_id, "request_id": request_id},
            ) from exc


if __name__ == "__main__":
    print("\n=== Running data retantion===\n")
    printer.status("TEST", "data retantion initialized", "info")

    retention = DataRetention()

    obligation = retention.create_retention_obligation(
        record_id="record-rtbf-001",
        subject_id="subject-001",
        policy_id="retention-standard-30d",
        retention_days=30,
        request_id="request-retention-001",
        metadata={
            "data_class": "customer_support_case",
            "storage_location": "regional_store_eu",
            "payload": {"email": "user@example.com", "phone": "+1-555-0100"},
        },
        record_tags=["support", "customer", "case"],
        related_records=["record-rtbf-000"],
        policy_version="2026.04",
    )
    printer.status("TEST", "Retention obligation created", "info")

    status_active = retention.retention_status(
        record_id="record-rtbf-001",
        request_id="request-retention-002",
    )
    printer.status("TEST", "Retention status evaluated", "info")

    hold_state = retention.apply_legal_hold(
        record_id="record-rtbf-001",
        hold_ref="legal-hold-001",
        reason="Open litigation hold",
        request_id="request-retention-003",
    )
    printer.status("TEST", "Legal hold applied", "info")

    released_state = retention.release_legal_hold(
        record_id="record-rtbf-001",
        reason="Legal hold lifted after case closure",
        request_id="request-retention-004",
    )
    printer.status("TEST", "Legal hold released", "info")

    forget_request = retention.process_forget_request(
        record_id="record-rtbf-001",
        request_id="request-retention-005",
        subject_id="subject-001",
        reason="Data subject requested erasure.",
    )
    printer.status("TEST", "Right-to-be-forgotten request processed", "info")

    failed_deletion = retention.mark_deletion_failed(
        record_id="record-rtbf-001",
        reason="Deletion worker unavailable",
        request_id="request-retention-006",
        retry_at=time.time() + 1800,
    )
    printer.status("TEST", "Deletion failure recorded", "info")

    rescheduled = retention.schedule_deletion(
        record_id="record-rtbf-001",
        reason="Retry deletion after worker restoration",
        request_id="request-retention-007",
        retry_at=time.time() + 900,
    )
    printer.status("TEST", "Deletion rescheduled", "info")

    completed = retention.mark_deletion_completed(
        record_id="record-rtbf-001",
        proof_ref="deletion-proof-001",
        request_id="request-retention-008",
    )
    printer.status("TEST", "Deletion completion recorded", "info")

    evidence = retention.memory.build_audit_evidence_bundle(
        request_id="request-retention-008",
        record_id="record-rtbf-001",
        subject_id="subject-001",
    )
    printer.status("TEST", "Audit evidence bundle generated", "info")

    print(json.dumps({
        "obligation": obligation,
        "status_active": status_active,
        "hold_state": hold_state,
        "released_state": released_state,
        "forget_request": forget_request,
        "failed_deletion": failed_deletion,
        "rescheduled": rescheduled,
        "completed": completed,
        "evidence": evidence,
        "memory_stats": retention.memory.stats(),
    }, indent=2, default=str))

    print("\n=== Test ran successfully ===\n")