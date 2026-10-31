"""Project-owned knowledge is canonical, scoped and reviewable (not chat memory).

This service deliberately delegates persistence, authorization, historical
queries, and impact tracking to KnowledgeRepository. It does not create a
second project database, project management system, or independent provenance.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .contracts import (AccessContext, Lifecycle, RecordDraft, RecordRevision,
                        Scope, ValidationFailure, validate_id)
from .repository import KnowledgeRepository


PROJECT_KINDS = frozenset({
    "project_requirement", "project_decision", "project_assumption",
    "project_constraint", "project_risk", "project_question",
    "project_exception", "project_validation", "project_specification",
})


class ProjectKnowledgeService:
    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

    def register(self, *, project_id: str, kind: str, statement: str,
                 access: AccessContext, rationale: str = "", status: Lifecycle = Lifecycle.DRAFT,
                 source_ids: tuple[str, ...] = (),
                 evidence: tuple[dict[str, Any], ...] = (),
                 organization_id: str | None = None,
                 details: Mapping[str, Any] | None = None,
                 record_id: str | None = None) -> RecordRevision:
        validate_id(project_id, "project_id")
        if kind not in PROJECT_KINDS:
            raise ValidationFailure("Unrecognized project knowledge kind")
        if not isinstance(statement, str) or not statement.strip() or len(statement) > 4000:
            raise ValidationFailure("Project statement must be nonempty and bounded")
        if kind in {"project_decision", "project_exception"} and not rationale.strip():
            raise ValidationFailure("Decisions and exceptions require explicit rationale")
        payload = {"statement": statement.strip(), "rationale": rationale.strip(),
                   "details": dict(details or {}), "author": access.principal_id,
                   "approval": None}
        draft = RecordDraft(kind=kind, scope=Scope.PROJECT, project_id=project_id,
                            organization_id=organization_id, payload=payload,
                            source_ids=source_ids, evidence=evidence, status=status)
        return self.repository.create(draft, access, record_id=record_id,
                                      reason="project-scoped knowledge registered")

    def approve(self, record_id: str, expected_revision: int, access: AccessContext,
                *, approval_evidence_id: str, reason: str) -> RecordRevision:
        if not access.can_review:
            from .contracts import AuthorizationDenied
            raise AuthorizationDenied("Approval permission required")
        validate_id(approval_evidence_id, "approval_evidence_id")
        existing = self.repository.get(record_id, access)
        if existing is None or existing.draft.kind not in PROJECT_KINDS:
            raise ValidationFailure("Accessible project knowledge required")
        if existing.draft.payload.get("author") == access.principal_id:
            from .contracts import AuthorizationDenied
            raise AuthorizationDenied("Approval requires a reviewer distinct from original author")
        updated = dict(existing.draft.payload)
        updated["approval"] = {"reviewer": access.principal_id,
                               "evidence_id": approval_evidence_id,
                               "reason": reason}
        draft = replace(existing.draft, payload=updated, status=Lifecycle.ACTIVE)
        return self.repository.revise(record_id, expected_revision, draft, access,
                                      reason=f"independently approved: {reason}")

    def revise(self, record_id: str, expected_revision: int, statement: str,
               access: AccessContext, *, reason: str) -> RecordRevision:
        existing = self.repository.get(record_id, access)
        if existing is None or existing.draft.kind not in PROJECT_KINDS:
            raise ValidationFailure("Accessible project knowledge required")
        if not isinstance(statement, str) or not statement.strip() or len(statement) > 4000:
            raise ValidationFailure("New statement must be nonempty and bounded")
        payload = dict(existing.draft.payload)
        payload["statement"] = statement.strip()
        payload["approval"] = None  # Changing an approved statement invalidates its approval.
        return self.repository.revise(record_id, expected_revision,
                                      replace(existing.draft, payload=payload,
                                              status=Lifecycle.DRAFT),
                                      access, reason=reason)

    def project_records(self, project_id: str, access: AccessContext, *,
                        kind: str | None = None, limit: int = 100) -> list[RecordRevision]:
        validate_id(project_id, "project_id")
        return self.repository.query(access, kind=kind, project_id=project_id, limit=limit)

    def affected_by(self, record_id: str, access: AccessContext) -> list[dict[str, Any]]:
        return self.repository.impacts(record_id, access)

    def inherited_rule_candidates(self, project_id: str, access: AccessContext,
                                  *, limit: int = 200) -> list[dict[str, Any]]:
        """Return visible candidates; NEVER silently override law/safety priority."""
        validate_id(project_id, "project_id")
        result: list[dict[str, Any]] = []
        for rec in self.repository.query(access, kind="design_rule", limit=limit):
            if rec.draft.scope == Scope.PROJECT and rec.draft.project_id != project_id:
                continue
            if rec.draft.status not in {Lifecycle.ACTIVE, Lifecycle.REVIEWED}:
                continue
            result.append({"rule_id": rec.record_id, "revision": rec.revision,
                           "source_scope": rec.draft.scope.value,
                           "precedence": "unresolved",
                           "verification_required": True})
        return result
