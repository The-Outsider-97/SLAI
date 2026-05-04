"""
- Consent artifact checks.
- Purpose limitation validation.
- Cross-context sharing constraints.
"""

from __future__ import annotations

import json
import time

from typing import Any, Dict, Mapping, Optional, Sequence

from .utils import (load_global_config, get_config_section,
                    # privacy error
                    ConsentArtifactMissingError, ConsentValidationError, CrossContextSharingError,
                    PolicyEvaluationError, PrivacyConfigurationError, PrivacyMemoryWriteError,
                    PrivacyDecision, PrivacyError, PurposeLimitationError, normalize_privacy_exception,
                    sanitize_privacy_context)
from .privacy_memory import PrivacyMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Data Consent and Purpose Binding")
printer = PrettyPrinter


class DataConsent:
    """Consent and purpose-binding runtime enforcement for the Privacy Agent.

    The consent layer does more than store consent metadata. It acts as the
    runtime gate for whether data may be processed, transferred, or reused for
    a given purpose and context. The implementation is intentionally opinionated
    and fail-closed by default so that missing consent artifacts, invalid
    purpose bindings, and unauthorized context propagation become explicit
    privacy decisions instead of implicit assumptions.
    """

    def __init__(self, memory: Optional[PrivacyMemory] = None) -> None:
        self.config = load_global_config()
        self.consent_config = get_config_section("data_consent")
        self.memory = memory or PrivacyMemory()

        self.enabled = bool(self.consent_config.get("enabled", True))
        self.strict_mode = bool(self.consent_config.get("strict_mode", True))
        self.sanitize_freeform_context = bool(
            self.consent_config.get("sanitize_freeform_context", True)
        )
        self.record_decisions_in_memory = bool(
            self.consent_config.get("record_decisions_in_memory", True)
        )
        self.record_stage_decisions = bool(
            self.consent_config.get("record_stage_decisions", False)
        )
        self.write_shared_contract = bool(
            self.consent_config.get("write_shared_contract", True)
        )

        self.require_artifact_for_processing = bool(
            self.consent_config.get("require_artifact_for_processing", True)
        )
        self.require_active_consent = bool(
            self.consent_config.get("require_active_consent", True)
        )
        self.require_purpose_binding = bool(
            self.consent_config.get("require_purpose_binding", True)
        )
        self.require_legal_basis_for_granted_consent = bool(
            self.consent_config.get("require_legal_basis_for_granted_consent", True)
        )
        self.enforce_source_context_match = bool(
            self.consent_config.get("enforce_source_context_match", True)
        )
        self.enforce_destination_context_allowlist = bool(
            self.consent_config.get("enforce_destination_context_allowlist", True)
        )
        self.enforce_action_allowlist = bool(
            self.consent_config.get("enforce_action_allowlist", True)
        )
        self.allow_same_context_without_explicit_destination_binding = bool(
            self.consent_config.get(
                "allow_same_context_without_explicit_destination_binding",
                True,
            )
        )

        self.default_policy_version = str(
            self.consent_config.get("default_policy_version", "v1")
        )
        self.default_decision_stage = str(
            self.consent_config.get("default_decision_stage", "consent.runtime_gate")
        )
        self.max_allowed_contexts = int(
            self.consent_config.get("max_allowed_contexts", 50)
        )
        self.max_allowed_processors = int(
            self.consent_config.get("max_allowed_processors", 50)
        )
        self.max_allowed_actions = int(
            self.consent_config.get("max_allowed_actions", 50)
        )
        self.max_allowed_purposes = int(
            self.consent_config.get("max_allowed_purposes", 100)
        )

        self._validate_config()
        logger.info("DataConsent initialized with production-ready consent controls.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        validators = {
            "max_allowed_contexts": self.max_allowed_contexts,
            "max_allowed_processors": self.max_allowed_processors,
            "max_allowed_actions": self.max_allowed_actions,
            "max_allowed_purposes": self.max_allowed_purposes,
        }
        for field_name, value in validators.items():
            if value <= 0:
                raise PrivacyConfigurationError(
                    section="data_consent",
                    details=f"'{field_name}' must be a positive integer, received {value!r}",
                )

        if not self.default_decision_stage.strip():
            raise PrivacyConfigurationError(
                section="data_consent",
                details="'default_decision_stage' must be a non-empty string.",
            )

    def _require_enabled(self, operation: str) -> None:
        if not self.enabled:
            raise PolicyEvaluationError(
                stage=operation,
                details="data_consent is disabled by configuration",
                context={"config_section": "data_consent"},
            )

    @staticmethod
    def _normalize_identity(value: Any, field_name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"'{field_name}' must be a non-empty string")
        return normalized

    @staticmethod
    def _normalize_optional_timestamp(
        value: Optional[float],
        field_name: str,
    ) -> Optional[float]:
        if value is None:
            return None
        normalized = float(value)
        if normalized < 0:
            raise ValueError(f"'{field_name}' must be >= 0")
        return normalized

    @staticmethod
    def _deduplicate(values: Optional[Sequence[Any]], *, limit: int) -> list[str]:
        if not values:
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip()
            if not item or item in seen:
                continue
            normalized.append(item)
            seen.add(item)
            if len(normalized) >= limit:
                break
        return normalized

    def _normalize_context(self, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not payload:
            return {}
        if self.sanitize_freeform_context:
            return sanitize_privacy_context(payload)
        return dict(payload)

    def _resolve_policy_version(self, policy_version: Optional[str]) -> str:
        return str(policy_version or self.default_policy_version)

    def _record_decision(self, *, request_id: Optional[str], stage: str, decision: PrivacyDecision, rationale: str,
                         subject_id: Optional[str], purpose: Optional[str], record_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        consent_status: Optional[Any] = None,
        context: Optional[Mapping[str, Any]] = None,
        audit_trail_ref: Optional[str] = None,
        best_effort: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not self.record_decisions_in_memory or not request_id:
            return None

        try:
            return self.memory.record_privacy_decision(
                request_id=request_id,
                stage=stage,
                decision=decision,
                rationale=rationale,
                subject_id=subject_id,
                record_id=record_id,
                purpose=purpose,
                consent_status=consent_status,
                policy_id=policy_id,
                policy_version=self._resolve_policy_version(policy_version),
                context=self._normalize_context(context),
                audit_trail_ref=audit_trail_ref,
            )
        except PrivacyError as exc:
            if best_effort:
                logger.error(
                    "Best-effort privacy decision recording failed for request '%s': %s",
                    request_id,
                    exc,
                )
                return None
            raise PrivacyMemoryWriteError(
                operation="data_consent.record_decision",
                details=exc,
                request_id=request_id,
                context={
                    "stage": stage,
                    "subject_id": subject_id,
                    "purpose": purpose,
                },
            ) from exc

    def _handle_exception(self, exc: Exception, *, stage: str,
                          context: Optional[Dict[str, Any]] = None,) -> PrivacyError:
        if isinstance(exc, PrivacyError):
            return exc
        return normalize_privacy_exception(exc, stage=stage, context=context)

    def _maybe_record_stage_success(
        self,
        *,
        request_id: Optional[str],
        stage: str,
        rationale: str,
        subject_id: Optional[str],
        purpose: Optional[str],
        record_id: Optional[str],
        policy_id: Optional[str],
        policy_version: Optional[str],
        consent_status: Optional[Any],
        context: Optional[Mapping[str, Any]] = None,
        audit_trail_ref: Optional[str] = None,
    ) -> None:
        if not self.record_stage_decisions:
            return
        self._record_decision(
            request_id=request_id,
            stage=stage,
            decision=PrivacyDecision.ALLOW,
            rationale=rationale,
            subject_id=subject_id,
            purpose=purpose,
            record_id=record_id,
            policy_id=policy_id,
            policy_version=policy_version,
            consent_status=consent_status,
            context=context,
            audit_trail_ref=audit_trail_ref,
            best_effort=False,
        )

    def _maybe_record_stage_failure(self, *, request_id: Optional[str], stage: str,
        subject_id: Optional[str],
        purpose: Optional[str],
        record_id: Optional[str],
        policy_id: Optional[str],
        policy_version: Optional[str],
        context: Optional[Mapping[str, Any]],
        audit_trail_ref: Optional[str],
        exc: PrivacyError,
    ) -> None:
        if not self.record_stage_decisions:
            return
        self._record_decision(
            request_id=request_id,
            stage=stage,
            decision=PrivacyDecision.BLOCK,
            rationale=exc.message,
            subject_id=subject_id,
            purpose=purpose,
            record_id=record_id,
            policy_id=policy_id,
            policy_version=policy_version,
            consent_status={
                "error_code": exc.error_code,
                "error_type": exc.error_type.value,
                "severity": exc.severity.value,
            },
            context=context,
            audit_trail_ref=audit_trail_ref,
            best_effort=True,
        )

    # ------------------------------------------------------------------
    # Consent and purpose-binding registration
    # ------------------------------------------------------------------
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
        operation = "data_consent.register_consent_artifact"
        try:
            self._require_enabled(operation)
            normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
            normalized_purpose = self._normalize_identity(purpose, "purpose")
            normalized_status = self._normalize_identity(status, "status").lower()
            normalized_artifact_ref = self._normalize_identity(artifact_ref, "artifact_ref")
            normalized_legal_basis = (
                self._normalize_identity(legal_basis, "legal_basis")
                if legal_basis is not None and str(legal_basis).strip()
                else None
            )
            normalized_granted_at = self._normalize_optional_timestamp(granted_at, "granted_at")
            normalized_expires_at = self._normalize_optional_timestamp(expires_at, "expires_at")
            normalized_revoked_at = self._normalize_optional_timestamp(revoked_at, "revoked_at")

            if normalized_status == "granted" and self.require_legal_basis_for_granted_consent:
                if not normalized_legal_basis:
                    raise ConsentValidationError(
                        normalized_subject_id,
                        normalized_purpose,
                        details="Granted consent must include legal basis metadata.",
                        policy_id=policy_id,
                        policy_version=self._resolve_policy_version(policy_version),
                    )

            if (
                normalized_granted_at is not None
                and normalized_expires_at is not None
                and normalized_expires_at < normalized_granted_at
            ):
                raise ValueError("'expires_at' cannot be earlier than 'granted_at'")

            return self.memory.register_consent_artifact(
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                status=normalized_status,
                artifact_ref=normalized_artifact_ref,
                legal_basis=normalized_legal_basis,
                granted_at=normalized_granted_at,
                expires_at=normalized_expires_at,
                revoked_at=normalized_revoked_at,
                allowed_contexts=self._deduplicate(
                    allowed_contexts,
                    limit=self.max_allowed_contexts,
                ),
                allowed_processors=self._deduplicate(
                    allowed_processors,
                    limit=self.max_allowed_processors,
                ),
                metadata=self._normalize_context(metadata),
                policy_id=policy_id,
                policy_version=self._resolve_policy_version(policy_version),
                audit_trail_ref=audit_trail_ref,
            )
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage=operation,
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
        operation = "data_consent.bind_purpose"
        try:
            self._require_enabled(operation)
            normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
            normalized_purpose = self._normalize_identity(purpose, "purpose")
            normalized_source_context = self._normalize_identity(source_context, "source_context")
            normalized_expires_at = self._normalize_optional_timestamp(expires_at, "expires_at")

            return self.memory.bind_purpose(
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                source_context=normalized_source_context,
                allowed_contexts=self._deduplicate(
                    allowed_contexts,
                    limit=self.max_allowed_contexts,
                ),
                allowed_actions=self._deduplicate(
                    allowed_actions,
                    limit=self.max_allowed_actions,
                ),
                binding_ref=binding_ref,
                expires_at=normalized_expires_at,
                metadata=self._normalize_context(metadata),
                policy_id=policy_id,
                policy_version=self._resolve_policy_version(policy_version),
                audit_trail_ref=audit_trail_ref,
            )
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise exc
            raise self._handle_exception(
                exc,
                stage=operation,
                context={
                    "subject_id": subject_id,
                    "purpose": purpose,
                    "source_context": source_context,
                },
            ) from exc

    # ------------------------------------------------------------------
    # Runtime validation APIs
    # ------------------------------------------------------------------
    def validate_consent(self, *, subject_id: str, purpose: str,
        required_context: Optional[str] = None,
        required_processor: Optional[str] = None,
        at_timestamp: Optional[float] = None,
        require_active: Optional[bool] = None,
        require_artifact: Optional[bool] = None,
        require_legal_basis: Optional[bool] = None,
        request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "data_consent.validate_consent"
        normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
        normalized_purpose = self._normalize_identity(purpose, "purpose")

        try:
            self._require_enabled(operation)
            normalized_required_context = (
                self._normalize_identity(required_context, "required_context")
                if required_context is not None and str(required_context).strip()
                else None
            )
            normalized_required_processor = (
                self._normalize_identity(required_processor, "required_processor")
                if required_processor is not None and str(required_processor).strip()
                else None
            )
            normalized_at_timestamp = self._normalize_optional_timestamp(
                at_timestamp,
                "at_timestamp",
            )
            require_active = (
                self.require_active_consent
                if require_active is None
                else bool(require_active)
            )
            require_artifact = (
                self.require_artifact_for_processing
                if require_artifact is None
                else bool(require_artifact)
            )
            require_legal_basis = (
                self.require_legal_basis_for_granted_consent
                if require_legal_basis is None
                else bool(require_legal_basis)
            )

            snapshot = self.memory.consent_status(
                normalized_subject_id,
                normalized_purpose,
                at_timestamp=normalized_at_timestamp,
            )

            if require_artifact and not snapshot.get("artifact_ref"):
                raise ConsentArtifactMissingError(
                    normalized_subject_id,
                    normalized_purpose,
                    artifact_type="consent_record",
                    context={
                        "required_context": normalized_required_context,
                        "required_processor": normalized_required_processor,
                        **self._normalize_context(context),
                    },
                    policy_id=policy_id,
                )

            if require_active and not bool(snapshot.get("is_active")):
                raise ConsentValidationError(
                    normalized_subject_id,
                    normalized_purpose,
                    details=(
                        f"Consent is not active for processing (status={snapshot.get('status')!r})."
                    ),
                    context={
                        "consent_snapshot": snapshot,
                        **self._normalize_context(context),
                    },
                    policy_id=policy_id,
                    policy_version=self._resolve_policy_version(policy_version),
                    audit_trail_ref=snapshot.get("audit_trail_ref") or audit_trail_ref,
                )

            if require_legal_basis and not snapshot.get("legal_basis"):
                raise ConsentValidationError(
                    normalized_subject_id,
                    normalized_purpose,
                    details="Consent is active but legal basis metadata is missing.",
                    context={
                        "consent_snapshot": snapshot,
                        **self._normalize_context(context),
                    },
                    policy_id=policy_id,
                    policy_version=self._resolve_policy_version(policy_version),
                    audit_trail_ref=snapshot.get("audit_trail_ref") or audit_trail_ref,
                )

            allowed_contexts = list(snapshot.get("allowed_contexts") or [])
            if normalized_required_context and allowed_contexts:
                if normalized_required_context not in allowed_contexts:
                    raise ConsentValidationError(
                        normalized_subject_id,
                        normalized_purpose,
                        details=(
                            "Required processing context is not covered by the consent artifact."
                        ),
                        context={
                            "required_context": normalized_required_context,
                            "allowed_contexts": allowed_contexts,
                            **self._normalize_context(context),
                        },
                        policy_id=policy_id,
                        policy_version=self._resolve_policy_version(policy_version),
                        audit_trail_ref=snapshot.get("audit_trail_ref") or audit_trail_ref,
                    )

            allowed_processors = list(snapshot.get("allowed_processors") or [])
            if normalized_required_processor and allowed_processors:
                if normalized_required_processor not in allowed_processors:
                    raise ConsentValidationError(
                        normalized_subject_id,
                        normalized_purpose,
                        details=(
                            "Requested processor is not covered by the consent artifact."
                        ),
                        context={
                            "required_processor": normalized_required_processor,
                            "allowed_processors": allowed_processors,
                            **self._normalize_context(context),
                        },
                        policy_id=policy_id,
                        policy_version=self._resolve_policy_version(policy_version),
                        audit_trail_ref=snapshot.get("audit_trail_ref") or audit_trail_ref,
                    )

            self._maybe_record_stage_success(
                request_id=request_id,
                stage="consent.validate",
                rationale="Consent artifact and runtime consent requirements validated successfully.",
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                consent_status=snapshot,
                context={
                    "required_context": normalized_required_context,
                    "required_processor": normalized_required_processor,
                    **self._normalize_context(context),
                },
                audit_trail_ref=snapshot.get("audit_trail_ref") or audit_trail_ref,
            )
            return snapshot
        except Exception as exc:
            normalized_exc = self._handle_exception(
                exc,
                stage=operation,
                context={
                    "subject_id": normalized_subject_id,
                    "purpose": normalized_purpose,
                    "required_context": required_context,
                    "required_processor": required_processor,
                },
            )
            self._maybe_record_stage_failure(
                request_id=request_id,
                stage="consent.validate",
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                context=context,
                audit_trail_ref=audit_trail_ref,
                exc=normalized_exc,
            )
            raise normalized_exc from exc

    def validate_purpose_limitation(
        self,
        *,
        subject_id: str,
        purpose: str,
        source_context: str,
        action: Optional[str] = None,
        destination_context: Optional[str] = None,
        at_timestamp: Optional[float] = None,
        request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "data_consent.validate_purpose_limitation"
        normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
        normalized_purpose = self._normalize_identity(purpose, "purpose")
        normalized_source_context = self._normalize_identity(source_context, "source_context")

        try:
            self._require_enabled(operation)
            normalized_action = (
                self._normalize_identity(action, "action")
                if action is not None and str(action).strip()
                else None
            )
            normalized_destination_context = (
                self._normalize_identity(destination_context, "destination_context")
                if destination_context is not None and str(destination_context).strip()
                else None
            )
            normalized_at_timestamp = self._normalize_optional_timestamp(
                at_timestamp,
                "at_timestamp",
            )

            snapshot = self.memory.consent_status(
                normalized_subject_id,
                normalized_purpose,
                at_timestamp=normalized_at_timestamp,
            )
            binding = snapshot.get("purpose_binding")

            if self.require_purpose_binding and not binding:
                raise PurposeLimitationError(
                    normalized_purpose,
                    subject_id=normalized_subject_id,
                    allowed_purposes=[normalized_purpose],
                    context={
                        "reason": "purpose binding is missing",
                        "source_context": normalized_source_context,
                        "destination_context": normalized_destination_context,
                        "action": normalized_action,
                        **self._normalize_context(context),
                    },
                    policy_id=policy_id,
                    policy_version=self._resolve_policy_version(policy_version),
                )

            if not binding:
                result = {
                    "subject_id": normalized_subject_id,
                    "purpose": normalized_purpose,
                    "source_context": normalized_source_context,
                    "destination_context": normalized_destination_context,
                    "action": normalized_action,
                    "binding": None,
                    "is_allowed": True,
                }
                self._maybe_record_stage_success(
                    request_id=request_id,
                    stage="purpose.validate",
                    rationale="No explicit purpose binding required for this request under current configuration.",
                    subject_id=normalized_subject_id,
                    purpose=normalized_purpose,
                    record_id=record_id,
                    policy_id=policy_id,
                    policy_version=policy_version,
                    consent_status=snapshot,
                    context=result,
                    audit_trail_ref=audit_trail_ref,
                )
                return result

            binding_expires_at = binding.get("expires_at")
            evaluation_ts = normalized_at_timestamp or time.time()
            if binding_expires_at is not None and float(binding_expires_at) <= evaluation_ts:
                raise PurposeLimitationError(
                    normalized_purpose,
                    subject_id=normalized_subject_id,
                    allowed_purposes=[normalized_purpose],
                    context={
                        "reason": "purpose binding has expired",
                        "binding": binding,
                        **self._normalize_context(context),
                    },
                    policy_id=policy_id,
                    policy_version=self._resolve_policy_version(policy_version),
                )

            expected_source_context = binding.get("source_context")
            if self.enforce_source_context_match and expected_source_context:
                if normalized_source_context != expected_source_context:
                    raise PurposeLimitationError(
                        normalized_purpose,
                        subject_id=normalized_subject_id,
                        allowed_purposes=[normalized_purpose],
                        context={
                            "reason": "source context does not match purpose binding",
                            "expected_source_context": expected_source_context,
                            "actual_source_context": normalized_source_context,
                            **self._normalize_context(context),
                        },
                        policy_id=policy_id,
                        policy_version=self._resolve_policy_version(policy_version),
                    )

            allowed_actions = list(binding.get("allowed_actions") or [])
            if (
                normalized_action
                and self.enforce_action_allowlist
                and allowed_actions
                and normalized_action not in allowed_actions
            ):
                raise PurposeLimitationError(
                    normalized_purpose,
                    subject_id=normalized_subject_id,
                    allowed_purposes=[normalized_purpose],
                    context={
                        "reason": "requested action is not allowed for the bound purpose",
                        "requested_action": normalized_action,
                        "allowed_actions": allowed_actions,
                        **self._normalize_context(context),
                    },
                    policy_id=policy_id,
                    policy_version=self._resolve_policy_version(policy_version),
                )

            allowed_contexts = list(binding.get("allowed_contexts") or [])
            if normalized_destination_context and normalized_destination_context != normalized_source_context:
                if (
                    self.enforce_destination_context_allowlist
                    and not allowed_contexts
                    and not self.allow_same_context_without_explicit_destination_binding
                ):
                    raise CrossContextSharingError(
                        normalized_source_context,
                        normalized_destination_context,
                        subject_id=normalized_subject_id,
                        purpose=normalized_purpose,
                        context={
                            "reason": "no destination contexts are authorized by purpose binding",
                            **self._normalize_context(context),
                        },
                    )
                if (
                    self.enforce_destination_context_allowlist
                    and allowed_contexts
                    and normalized_destination_context not in allowed_contexts
                ):
                    raise CrossContextSharingError(
                        normalized_source_context,
                        normalized_destination_context,
                        subject_id=normalized_subject_id,
                        purpose=normalized_purpose,
                        context={
                            "reason": "destination context is not authorized by purpose binding",
                            "allowed_contexts": allowed_contexts,
                            **self._normalize_context(context),
                        },
                    )

            result = {
                "subject_id": normalized_subject_id,
                "purpose": normalized_purpose,
                "source_context": normalized_source_context,
                "destination_context": normalized_destination_context,
                "action": normalized_action,
                "binding": binding,
                "is_allowed": True,
            }
            self._maybe_record_stage_success(
                request_id=request_id,
                stage="purpose.validate",
                rationale="Purpose limitation and purpose binding checks passed.",
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                consent_status=snapshot,
                context=result,
                audit_trail_ref=binding.get("audit_trail_ref") or audit_trail_ref,
            )
            return result
        except Exception as exc:
            normalized_exc = self._handle_exception(
                exc,
                stage=operation,
                context={
                    "subject_id": normalized_subject_id,
                    "purpose": normalized_purpose,
                    "source_context": normalized_source_context,
                    "destination_context": destination_context,
                    "action": action,
                },
            )
            self._maybe_record_stage_failure(
                request_id=request_id,
                stage="purpose.validate",
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                context=context,
                audit_trail_ref=audit_trail_ref,
                exc=normalized_exc,
            )
            raise normalized_exc from exc

    def validate_cross_context_sharing(
        self,
        *,
        subject_id: str,
        purpose: str,
        source_context: str,
        destination_context: str,
        action: Optional[str] = None,
        required_processor: Optional[str] = None,
        at_timestamp: Optional[float] = None,
        request_id: Optional[str] = None,
        record_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "data_consent.validate_cross_context_sharing"
        normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
        normalized_purpose = self._normalize_identity(purpose, "purpose")
        normalized_source_context = self._normalize_identity(source_context, "source_context")
        normalized_destination_context = self._normalize_identity(
            destination_context,
            "destination_context",
        )

        try:
            self._require_enabled(operation)
            consent_snapshot = self.validate_consent(
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                required_context=normalized_destination_context,
                required_processor=required_processor,
                at_timestamp=at_timestamp,
                request_id=request_id,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                audit_trail_ref=audit_trail_ref,
                context=context,
            )
            purpose_validation = self.validate_purpose_limitation(
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                source_context=normalized_source_context,
                destination_context=normalized_destination_context,
                action=action,
                at_timestamp=at_timestamp,
                request_id=request_id,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                audit_trail_ref=audit_trail_ref,
                context=context,
            )

            if normalized_source_context == normalized_destination_context:
                result = {
                    "subject_id": normalized_subject_id,
                    "purpose": normalized_purpose,
                    "source_context": normalized_source_context,
                    "destination_context": normalized_destination_context,
                    "cross_context": False,
                    "consent": consent_snapshot,
                    "purpose_validation": purpose_validation,
                    "is_allowed": True,
                }
            else:
                result = {
                    "subject_id": normalized_subject_id,
                    "purpose": normalized_purpose,
                    "source_context": normalized_source_context,
                    "destination_context": normalized_destination_context,
                    "cross_context": True,
                    "consent": consent_snapshot,
                    "purpose_validation": purpose_validation,
                    "is_allowed": True,
                }

            self._maybe_record_stage_success(
                request_id=request_id,
                stage="sharing.validate",
                rationale="Cross-context sharing is authorized by consent and purpose binding.",
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                consent_status=consent_snapshot,
                context=result,
                audit_trail_ref=consent_snapshot.get("audit_trail_ref") or audit_trail_ref,
            )
            return result
        except Exception as exc:
            normalized_exc = self._handle_exception(
                exc,
                stage=operation,
                context={
                    "subject_id": normalized_subject_id,
                    "purpose": normalized_purpose,
                    "source_context": normalized_source_context,
                    "destination_context": normalized_destination_context,
                    "action": action,
                    "required_processor": required_processor,
                },
            )
            self._maybe_record_stage_failure(
                request_id=request_id,
                stage="sharing.validate",
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                context=context,
                audit_trail_ref=audit_trail_ref,
                exc=normalized_exc,
            )
            raise normalized_exc from exc

    def evaluate_request(
        self,
        *,
        request_id: str,
        subject_id: str,
        purpose: str,
        source_context: str,
        action: str,
        destination_context: Optional[str] = None,
        required_processor: Optional[str] = None,
        record_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        sensitivity_score: Optional[float] = None,
        detected_entities: Optional[Sequence[Any]] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "data_consent.evaluate_request"
        normalized_request_id = self._normalize_identity(request_id, "request_id")
        normalized_subject_id = self._normalize_identity(subject_id, "subject_id")
        normalized_purpose = self._normalize_identity(purpose, "purpose")
        normalized_source_context = self._normalize_identity(source_context, "source_context")
        normalized_action = self._normalize_identity(action, "action")
        normalized_destination_context = (
            self._normalize_identity(destination_context, "destination_context")
            if destination_context is not None and str(destination_context).strip()
            else normalized_source_context
        )

        try:
            self._require_enabled(operation)

            consent_snapshot = self.validate_consent(
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                required_context=normalized_destination_context,
                required_processor=required_processor,
                request_id=normalized_request_id,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                audit_trail_ref=audit_trail_ref,
                context=context,
            )
            purpose_validation = self.validate_purpose_limitation(
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                source_context=normalized_source_context,
                action=normalized_action,
                destination_context=normalized_destination_context,
                request_id=normalized_request_id,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                audit_trail_ref=audit_trail_ref,
                context=context,
            )

            sharing_validation = None
            if normalized_destination_context != normalized_source_context or required_processor:
                sharing_validation = self.validate_cross_context_sharing(
                    subject_id=normalized_subject_id,
                    purpose=normalized_purpose,
                    source_context=normalized_source_context,
                    destination_context=normalized_destination_context,
                    action=normalized_action,
                    required_processor=required_processor,
                    request_id=normalized_request_id,
                    record_id=record_id,
                    policy_id=policy_id,
                    policy_version=policy_version,
                    audit_trail_ref=audit_trail_ref,
                    context=context,
                )

            if self.write_shared_contract:
                self.memory.write_request_contract(
                    request_id=normalized_request_id,
                    sensitivity_score=sensitivity_score,
                    detected_entities=detected_entities,
                    consent_status={
                        "status": consent_snapshot.get("status"),
                        "is_active": consent_snapshot.get("is_active"),
                        "artifact_ref": consent_snapshot.get("artifact_ref"),
                        "policy_id": consent_snapshot.get("policy_id"),
                        "audit_trail_ref": consent_snapshot.get("audit_trail_ref"),
                    },
                    audit_trail_ref=consent_snapshot.get("audit_trail_ref") or audit_trail_ref,
                )

            decision_record = self._record_decision(
                request_id=normalized_request_id,
                stage=self.default_decision_stage,
                decision=PrivacyDecision.ALLOW,
                rationale="Consent, purpose limitation, and sharing constraints passed for the requested operation.",
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                consent_status=consent_snapshot,
                context={
                    "source_context": normalized_source_context,
                    "destination_context": normalized_destination_context,
                    "action": normalized_action,
                    "required_processor": required_processor,
                    "purpose_validation": purpose_validation,
                    "sharing_validation": sharing_validation,
                    **self._normalize_context(context),
                },
                audit_trail_ref=consent_snapshot.get("audit_trail_ref") or audit_trail_ref,
                best_effort=False,
            )

            return {
                "decision": PrivacyDecision.ALLOW.value,
                "request_id": normalized_request_id,
                "subject_id": normalized_subject_id,
                "purpose": normalized_purpose,
                "source_context": normalized_source_context,
                "destination_context": normalized_destination_context,
                "action": normalized_action,
                "required_processor": required_processor,
                "consent": consent_snapshot,
                "purpose_validation": purpose_validation,
                "sharing_validation": sharing_validation,
                "decision_trace": decision_record,
            }
        except Exception as exc:
            normalized_exc = self._handle_exception(
                exc,
                stage=operation,
                context={
                    "request_id": normalized_request_id,
                    "subject_id": normalized_subject_id,
                    "purpose": normalized_purpose,
                    "source_context": normalized_source_context,
                    "destination_context": normalized_destination_context,
                    "action": normalized_action,
                    "required_processor": required_processor,
                },
            )
            self._record_decision(
                request_id=normalized_request_id,
                stage=self.default_decision_stage,
                decision=PrivacyDecision.BLOCK,
                rationale=normalized_exc.message,
                subject_id=normalized_subject_id,
                purpose=normalized_purpose,
                record_id=record_id,
                policy_id=policy_id,
                policy_version=policy_version,
                consent_status={
                    "error_code": normalized_exc.error_code,
                    "error_type": normalized_exc.error_type.value,
                    "severity": normalized_exc.severity.value,
                },
                context={
                    "source_context": normalized_source_context,
                    "destination_context": normalized_destination_context,
                    "action": normalized_action,
                    "required_processor": required_processor,
                    **self._normalize_context(context),
                },
                audit_trail_ref=audit_trail_ref,
                best_effort=True,
            )
            raise normalized_exc from exc


if __name__ == "__main__":
    print("\n=== Running data consent===\n")
    printer.status("TEST", "data consent initialized", "info")

    consent = DataConsent()

    consent_record = consent.register_consent_artifact(
        subject_id="subject-001",
        purpose="support_resolution",
        status="granted",
        artifact_ref="consent-artifact-001",
        legal_basis="explicit_consent",
        granted_at=time.time(),
        expires_at=time.time() + 86400,
        allowed_contexts=["chat_runtime", "privacy_review", "ticketing_connector"],
        allowed_processors=["execution_agent", "ticketing_connector"],
        metadata={"collector": "privacy_agent", "channel": "chat"},
        policy_id="consent-policy-001",
        policy_version="2026.04",
    )
    printer.status("TEST", "Consent artifact registered", "info")

    purpose_binding = consent.bind_purpose(
        subject_id="subject-001",
        purpose="support_resolution",
        source_context="chat_runtime",
        allowed_contexts=["privacy_review", "ticketing_connector"],
        allowed_actions=["read", "redact", "share", "respond"],
        metadata={"sharing_rule": "restricted", "notes": "Support-only scope"},
        policy_id="purpose-binding-policy-001",
        policy_version="2026.04",
    )
    printer.status("TEST", "Purpose binding registered", "info")

    consent_snapshot = consent.validate_consent(
        subject_id="subject-001",
        purpose="support_resolution",
        required_context="ticketing_connector",
        required_processor="ticketing_connector",
        request_id="request-001",
        record_id="record-001",
        policy_id="runtime-consent-policy",
        policy_version="2026.04",
        context={"intent": "create_support_ticket", "payload": {"email": "user@example.com"}},
    )
    printer.status("TEST", "Consent validated", "info")

    purpose_snapshot = consent.validate_purpose_limitation(
        subject_id="subject-001",
        purpose="support_resolution",
        source_context="chat_runtime",
        destination_context="ticketing_connector",
        action="share",
        request_id="request-001",
        record_id="record-001",
        policy_id="runtime-purpose-policy",
        policy_version="2026.04",
        context={"reason": "handoff to external support processor"},
    )
    printer.status("TEST", "Purpose limitation validated", "info")

    sharing_snapshot = consent.validate_cross_context_sharing(
        subject_id="subject-001",
        purpose="support_resolution",
        source_context="chat_runtime",
        destination_context="ticketing_connector",
        action="share",
        required_processor="ticketing_connector",
        request_id="request-001",
        record_id="record-001",
        policy_id="runtime-sharing-policy",
        policy_version="2026.04",
        context={"reason": "approved support escalation"},
    )
    printer.status("TEST", "Cross-context sharing validated", "info")

    evaluation = consent.evaluate_request(
        request_id="request-001",
        subject_id="subject-001",
        purpose="support_resolution",
        source_context="chat_runtime",
        destination_context="ticketing_connector",
        action="share",
        required_processor="ticketing_connector",
        record_id="record-001",
        policy_id="runtime-gating-policy",
        policy_version="2026.04",
        sensitivity_score=0.81,
        detected_entities=["email", "phone", "customer_id"],
        context={"ticket_priority": "high", "payload": {"phone": "+1-000-000-0000"}},
    )
    printer.status("TEST", "Request evaluation completed", "info")

    blocked_error = None
    try:
        consent.validate_cross_context_sharing(
            subject_id="subject-001",
            purpose="support_resolution",
            source_context="chat_runtime",
            destination_context="marketing_analytics",
            action="share",
            required_processor="analytics_connector",
            request_id="request-002",
            record_id="record-001",
            policy_id="runtime-sharing-policy",
            policy_version="2026.04",
            context={"reason": "expected negative-path test"},
        )
    except PrivacyError as exc:
        blocked_error = exc.to_public_dict()
        printer.status("TEST", "Expected sharing block captured", "warning")

    trace = consent.memory.privacy_decision_trace("request-001")

    print(
        json.dumps(
            {
                "consent_record": consent_record,
                "purpose_binding": purpose_binding,
                "consent_snapshot": consent_snapshot,
                "purpose_snapshot": purpose_snapshot,
                "sharing_snapshot": sharing_snapshot,
                "evaluation": evaluation,
                "blocked_error": blocked_error,
                "trace": trace,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print("\n=== Test ran successfully ===\n")
