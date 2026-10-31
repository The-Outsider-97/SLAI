"""Standards metadata, clause references and contextual applicability.

This registry does NOT fetch proprietary standards, check publisher websites,
interpret legal priority, or certify conformance. Attestations are explicit,
permissioned and must retain external evidence. It reuses a single canonical
KnowledgeRepository rather than starting a second standards database.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .contracts import *
from .repository import KnowledgeRepository
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.knowledge_errors import *
from ..utils.knowledge_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Assurance Standards")
printer = PrettyPrinter()


STANDARD_STATES = frozenset({"draft", "published", "amended", "withdrawn", "superseded", "unknown"})
REQUIREMENT_MODALITY = frozenset({"mandatory", "advisory", "conditional", "informative", "undetermined"})
EVIDENCE_DIMENSIONS = frozenset({"authenticity", "edition", "integrity", "currency", "applicability", "interpretation"})
EVIDENCE_VERDICTS = frozenset({"supported", "rejected", "inconclusive"})


def _contextual_applicability(criteria: Mapping[str, Any], context: Mapping[str, Any]) -> Applicability:
    """No guesswork: missing context is UNDETERMINED, not applicable."""
    reasons: list[str] = []
    missing: list[str] = []
    for key in ("jurisdiction", "domain", "activity", "project_class"):
        allowed = criteria.get(key)
        if allowed is None:
            continue
        if not isinstance(allowed, list) or not allowed or not all(isinstance(x, str) for x in allowed):
            raise ValidationFailure(f"Invalid applicability selector {key}")
        provided = context.get(key)
        if provided is None:
            missing.append(key)
        elif provided not in allowed:
            reasons.append(f"{key}: supplied value is outside declared applicability")
    effective_at = criteria.get("effective_at")
    if effective_at:
        if not isinstance(effective_at, Mapping):
            raise ValidationFailure("effective_at must be an object")
        at = context.get("at")
        if at is None:
            missing.append("at")
        else:
            normalized_at = normalize_time(str(at), "context.at")
            starts = normalize_time(effective_at.get("from"), "effective_at.from")
            ends = normalize_time(effective_at.get("until"), "effective_at.until")
            if normalized_at is not None and starts is not None and normalized_at < starts:
                reasons.append("Context precedes the specified effective period")
            if normalized_at is not None and ends is not None and normalized_at >= ends:
                reasons.append("Context follows the specified effective period")
    if reasons:
        return Applicability("not_applicable", tuple(reasons))
    if missing:
        return Applicability("undetermined", ("Missing context: " + ", ".join(sorted(set(missing))),))
    return Applicability("applicable", ("Declared selectors match supplied context",))


class StandardsRegistry:
    """Metadata and structured requirements. No automatic external authenticity claims."""

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

        logger.info(f"Standard registry initialized")

    def register_edition(self, *, standard_id: str, edition: str,
                         title: str, publisher: str, source_id: str,
                         access: AccessContext, scope: Scope = Scope.GLOBAL,
                         project_id: str | None = None,
                         organization_id: str | None = None,
                         publication_status: str = "unknown",
                         publication_date: str | None = None,
                         licence: str = "metadata_only",
                         applicability: Mapping[str, Any] | None = None,
                         supersedes_record_id: str | None = None,
                         record_id: str | None = None) -> RecordRevision:
        validate_id(standard_id, "standard_id")
        validate_id(source_id, "source_id")
        if publication_status not in STANDARD_STATES:
            raise ValidationFailure("Unknown standards publication status")
        if licence not in {"metadata_only", "authorized_extract", "licensed_content", "public_domain", "unknown"}:
            raise ValidationFailure("Unknown content rights classification")
        _contextual_applicability(applicability or {}, {})  # validate selectors before persistence
        payload: dict[str, Any] = {
            "standard_id": standard_id, "edition": edition.strip(), "title": title.strip(),
            "publisher": publisher.strip(), "source_id": source_id,
            "publication_date": publication_date, "publication_status": publication_status,
            "licence": licence, "applicability": dict(applicability or {}),
            "supersedes_record_id": supersedes_record_id,
            "verification": {},  # Independent assertions inserted only with attest().
            "registered_by": access.principal_id,
        }
        _nonblank(payload, "edition")
        _nonblank(payload, "title", 512)
        _nonblank(payload, "publisher")
        draft = RecordDraft(kind="standard_edition", scope=scope, project_id=project_id,
                            organization_id=organization_id, payload=payload,
                            source_ids=(source_id,), status=Lifecycle.DRAFT)
        registered = self.repository.create(draft, access, record_id=record_id,
                                            reason="standard edition metadata registered; unverified")
        if supersedes_record_id:
            self.repository.link(registered.record_id, supersedes_record_id, access, relation_type="supersedes")
        return registered

    def attest(self, record_id: str, expected_revision: int, *, dimension: str,
               verdict: str, source_id: str, evidence_id: str,
               method: str, access: AccessContext,
               verified_at: str | None = None, details: str = "") -> RecordRevision:
        """Record review evidence; a verified label is not generated from AI confidence."""
        if not access.can_attest or not access.can_write or not access.can_review:
            raise AuthorizationDenied("Independent attestation permission required", action="attest")
        if dimension not in EVIDENCE_DIMENSIONS or verdict not in EVIDENCE_VERDICTS:
            raise ValidationFailure("Invalid verification dimension or outcome")
        for name, value in (("source_id", source_id), ("evidence_id", evidence_id)):
            validate_id(value, name)
        if not isinstance(method, str) or not method.strip():
            raise ValidationFailure("Explicit verification method required")
        rec = self.repository.get(record_id, access)
        if rec is None or rec.draft.kind not in {"standard_edition", "requirement"}:
            raise ValidationFailure("Attestation target is not an accessible standard/requirement")
        data = dict(rec.draft.payload)
        if data.get("registered_by", data.get("extracted_by")) == access.principal_id:
            raise AuthorizationDenied("Independent reviewer must differ from the original author", action="attest")
        previous = dict(data.get("verification", {}))
        attestation = {"dimension": dimension, "verdict": verdict,
                       "source_id": source_id, "evidence_id": evidence_id,
                       "method": method.strip(), "verifier": access.principal_id,
                       "verified_at": normalize_time(verified_at or utc_now(), "verified_at"),
                       "details": details}
        previous[dimension] = attestation
        data["verification"] = previous
        draft = replace(rec.draft, payload=data,
                        evidence=(*rec.draft.evidence, attestation),
                        source_ids=tuple(sorted(set((*rec.draft.source_ids, source_id)))))
        return self.repository.revise(record_id, expected_revision, draft, access,
                                      reason=f"attestation {dimension}: {verdict}")

    def register_requirement(self, *, standard_record_id: str, clause_locator: str,
                             paraphrase: str, modality: str,
                             access: AccessContext,
                             applicability: Mapping[str, Any] | None = None,
                             rationale: str = "extracted candidate; not independently interpreted",
                             record_id: str | None = None) -> RecordRevision:
        """Requires an authorized clause locator; stores no verbatim protected text."""
        if modality not in REQUIREMENT_MODALITY:
            raise ValidationFailure("Unsupported requirement modality")
        standard = self.repository.get(standard_record_id, access)
        if standard is None or standard.draft.kind != "standard_edition":
            raise ValidationFailure("Accessible standard edition required")
        if not clause_locator.strip() or not paraphrase.strip() or len(paraphrase) > 2000:
            raise ValidationFailure("Clause locator and bounded candidate paraphrase required")
        _contextual_applicability(applicability or {}, {})  # validate selectors before persistence
        payload = {"standard_record_id": standard_record_id,
                   "standard_edition": standard.draft.payload["edition"],
                   "clause_locator": clause_locator, "candidate_paraphrase": paraphrase,
                   "modality": modality,
                   "applicability": dict(applicability or {}),
                   "extraction_note": rationale,
                   "extracted_by": access.principal_id,
                   "verification": {}}
        draft = RecordDraft(kind="requirement", scope=standard.draft.scope,
                            project_id=standard.draft.project_id,
                            organization_id=standard.draft.organization_id,
                            payload=payload, source_ids=standard.draft.source_ids,
                            status=Lifecycle.DRAFT)
        rec = self.repository.create(draft, access, record_id=record_id,
                                     reason="candidate requirement extracted; requires independent review")
        self.repository.link(rec.record_id, standard_record_id, access)
        return rec

    def applicable_requirements(self, access: AccessContext | None, *,
                                context: Mapping[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        """Candidates are not assertions of legal force or conformance."""
        outputs = []
        candidates = self.repository.query(access, kind="requirement", limit=limit)
        for requirement in candidates:
            std = self.repository.get(requirement.draft.payload["standard_record_id"], access)
            if std is None:
                continue
            requirement_check = _contextual_applicability(requirement.draft.payload.get("applicability", {}), context)
            standard_check = _contextual_applicability(std.draft.payload.get("applicability", {}), context)
            if "not_applicable" in {requirement_check.outcome, standard_check.outcome}:
                status = "not_applicable"
            elif "undetermined" in {requirement_check.outcome, standard_check.outcome}:
                status = "undetermined"
            else:
                status = "applicable_candidate"
            if std.draft.payload.get("publication_status") in {"withdrawn", "superseded", "unknown"}:
                status = "undetermined" if status != "not_applicable" else status
            outputs.append({"requirement_id": requirement.record_id,
                            "requirement_revision": requirement.revision,
                            "standard_record_id": std.record_id,
                            "standard_revision": std.revision,
                            "edition": std.draft.payload["edition"],
                            "clause_locator": requirement.draft.payload["clause_locator"],
                            "modality": requirement.draft.payload["modality"],
                            "applicability": status,
                            "reasons": (*requirement_check.reasons, *standard_check.reasons),
                            "verification": requirement.draft.payload.get("verification", {}),
                            "standard_verification": std.draft.payload.get("verification", {}),
                            "disclaimer": "Candidate match; not an assessment of legal obligation or compliance"})
        return outputs


__all__ = [
    "STANDARD_STATES",
    "REQUIREMENT_MODALITY",
    "EVIDENCE_DIMENSIONS",
    "EVIDENCE_VERDICTS",
    "_contextual_applicability",
    "StandardsRegistry",
]