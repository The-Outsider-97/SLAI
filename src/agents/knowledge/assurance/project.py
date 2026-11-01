"""Project-owned knowledge is canonical, scoped and reviewable (not chat memory).

This service deliberately delegates persistence, authorization, historical
queries, and impact tracking to KnowledgeRepository. It does not create a
second project database, project management system, or independent provenance.
"""
from __future__ import annotations
from .contracts import AuthorizationDenied, NotFound, json_object, normalize_time, utc_now
from .acl import can_read, require_write
import uuid
import sqlite3
from dataclasses import dataclass

from dataclasses import replace
from typing import Any, Mapping

from .contracts import AccessContext, Lifecycle, RecordDraft, RecordRevision, Scope
from .repository import KnowledgeRepository
from src.agents.knowledge.utils.config_loader import get_config_section, load_global_config
from src.agents.knowledge.utils.knowledge_errors import *
from src.agents.knowledge.utils.knowledge_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Assurance Project")
printer = PrettyPrinter()


PROJECT_KINDS = frozenset({
    "project_requirement", "project_decision", "project_assumption",
    "project_constraint", "project_risk", "project_question",
    "project_exception", "project_validation", "project_specification",
})


class ProjectKnowledgeService:
    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

        logger.info(f"Project Knowledge Service initialized")

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
        return self.repository.create(draft, access, record_id=record_id, reason="project-scoped knowledge registered")

    def approve(self, record_id: str, expected_revision: int, access: AccessContext,
                *, approval_evidence_id: str, reason: str) -> RecordRevision:
        if not access.can_review:
            from .contracts import AuthorizationDenied
            raise AuthorizationDenied("Approval permission required", action="approve project knowledge")
        validate_id(approval_evidence_id, "approval_evidence_id")
        existing = self.repository.get(record_id, access)
        if existing is None or existing.draft.kind not in PROJECT_KINDS:
            raise ValidationFailure("Accessible project knowledge required")
        if existing.draft.payload.get("author") == access.principal_id:
            from .contracts import AuthorizationDenied
            raise AuthorizationDenied("Approval requires a reviewer distinct from original author",
                                       action="approve project knowledge")
        updated = dict(existing.draft.payload)
        updated["approval"] = {"reviewer": access.principal_id, "evidence_id": approval_evidence_id, "reason": reason}
        draft = replace(existing.draft, payload=updated, status=Lifecycle.ACTIVE)
        return self.repository.revise(record_id, expected_revision, draft, access, reason=f"independently approved: {reason}")

    def revise(self, record_id: str, expected_revision: int, statement: str, access: AccessContext, *, reason: str) -> RecordRevision:
        existing = self.repository.get(record_id, access)
        if existing is None or existing.draft.kind not in PROJECT_KINDS:
            raise ValidationFailure("Accessible project knowledge required")
        if not isinstance(statement, str) or not statement.strip() or len(statement) > 4000:
            raise ValidationFailure("New statement must be nonempty and bounded")
        payload = dict(existing.draft.payload)
        payload["statement"] = statement.strip()
        payload["approval"] = None  # Changing an approved statement invalidates its approval.
        return self.repository.revise(record_id, expected_revision,
                                      replace(existing.draft, payload=payload, status=Lifecycle.DRAFT),
                                      access, reason=reason)

    def project_records(self, project_id: str, access: AccessContext, *, kind: str | None = None, limit: int = 100) -> list[RecordRevision]:
        validate_id(project_id, "project_id")
        return self.repository.query(access, kind=kind, project_id=project_id, limit=limit)

    def affected_by(self, record_id: str, access: AccessContext) -> list[dict[str, Any]]:
        return self.repository.impacts(record_id, access)

    def inherited_rule_candidates(self, project_id: str, access: AccessContext, *, limit: int = 200) -> list[dict[str, Any]]:
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


__all__ = ["ProjectKnowledgeService"]

@dataclass(frozen=True, slots=True)
class AdoptionResolution:
    project_id: str
    standard_id: str
    outcome: str  # adopted_candidate / conflict / no_adoption / undetermined
    edition_record_ids: tuple[str, ...]
    reasons: tuple[str, ...]


class ProjectAdoptionService:
    """Independently reviewed project-specific selection of an edition."""

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

    def propose(self, *, project_id: str, edition_record_id: str,
                authority_evidence_id: str, basis: str, access: AccessContext,
                valid_from: str | None = None, valid_until: str | None = None,
                record_id: str | None = None) -> RecordRevision:
        validate_id(project_id, "project_id")
        validate_id(edition_record_id, "edition_record_id")
        validate_id(authority_evidence_id, "authority_evidence_id")
        if not basis.strip() or len(basis) > 2000:
            raise ValidationFailure("An explicit contract/legal-adoption rationale is required")
        edition = self.repository.get(edition_record_id, access)
        if edition is None or edition.draft.kind != "standard_edition":
            raise NotFound("standard_edition", edition_record_id, details={"reason": "Accessible standard edition required"})
        if project_id not in access.project_ids:
            raise AuthorizationDenied(access.principal_id if access is not None else "anonymous", "knowledge_access", details={"reason": "Project membership required"})
        draft = RecordDraft(
            kind="project_standard_adoption", scope=Scope.PROJECT, project_id=project_id,
            status=Lifecycle.DRAFT,
            valid_from=valid_from, valid_until=valid_until,
            payload={"edition_record_id": edition_record_id,
                     "standard_id": edition.draft.payload["standard_id"],
                     "edition": edition.draft.payload["edition"],
                     "basis": basis.strip(),
                     "authority_evidence_id": authority_evidence_id,
                     "proposed_by": access.principal_id,
                     "approval": None},
            evidence=({"evidence_ref": authority_evidence_id, "role": "adoption_basis"},),
            source_ids=edition.draft.source_ids,
        )
        candidate = self.repository.create(draft, access, record_id=record_id,
                                           reason="project-specific edition adoption proposed")
        self.repository.link(candidate.record_id, edition_record_id, access)
        return candidate

    def approve(self, record_id: str, expected_revision: int, *, reviewer_evidence_id: str,
                access: AccessContext, reason: str) -> RecordRevision:
        validate_id(reviewer_evidence_id, "reviewer_evidence_id")
        if not access.can_review:
            raise AuthorizationDenied(access.principal_id if access is not None else "anonymous", "knowledge_access", details={"reason": "Independent reviewer required"})
        if not reason.strip():
            raise ValidationFailure("Review justification required")
        rec = self.repository.get(record_id, access)
        if rec is None or rec.draft.kind != "project_standard_adoption":
            raise NotFound("project_standard_adoption", record_id, details={"reason": "Project adoption candidate unavailable"})
        if rec.draft.payload["proposed_by"] == access.principal_id:
            raise AuthorizationDenied(access.principal_id if access is not None else "anonymous", "knowledge_access", details={"reason": "Adoption must be independently reviewed"})
        data = dict(rec.draft.payload)
        data["approval"] = {"reviewer": access.principal_id,
                            "reviewer_evidence_id": reviewer_evidence_id,
                            "reason": reason.strip()}
        new = replace(rec.draft, payload=data, status=Lifecycle.ACTIVE,
                      evidence=(*rec.draft.evidence,
                                {"evidence_ref": reviewer_evidence_id, "role": "adoption_approval"}))
        return self.repository.revise(record_id, expected_revision, new, access,
                                      reason=f"edition adoption approved: {reason.strip()}")

    def resolve(self, *, project_id: str, standard_id: str, access: AccessContext,
                valid_at: str, as_known: str | None = None) -> AdoptionResolution:
        validate_id(project_id, "project_id")
        validate_id(standard_id, "standard_id")
        valid_at = normalize_time(valid_at, "valid_at")
        if valid_at is None:
            raise ValidationFailure("valid_at required")
        if project_id not in access.project_ids:
            raise AuthorizationDenied(access.principal_id if access is not None else "anonymous", "knowledge_access", details={"reason": "Project membership required"})
        records: list[RecordRevision] = []
        offset = 0
        while offset < 10000:
            page = self.repository.query(access, kind="project_standard_adoption",
                                         project_id=project_id, as_known=as_known,
                                         as_valid=valid_at, limit=500, offset=offset)
            records.extend(rec for rec in page if rec.draft.status == Lifecycle.ACTIVE and
                           rec.draft.payload.get("standard_id") == standard_id)
            if len(page) < 500:
                break
            offset += 500
        if offset >= 10000:
            return AdoptionResolution(project_id, standard_id, "undetermined", (),
                                      ("Adoption query exceeded bounded review window",))
        ids = sorted({r.draft.payload["edition_record_id"] for r in records})
        if not ids:
            return AdoptionResolution(project_id, standard_id, "no_adoption", (),
                                      ("No independently approved project adoption in effect",))
        # Edition availability and scope are still checked under the caller's ACL.
        for ref in ids:
            if self.repository.get(ref, access, as_known=as_known) is None:
                return AdoptionResolution(project_id, standard_id, "undetermined", tuple(ids),
                                          ("A cited edition is not available at the requested knowledge time",))
        if len(ids) != 1:
            return AdoptionResolution(project_id, standard_id, "conflict", tuple(ids),
                                      ("Multiple approved editions overlap; human priority decision required",))
        return AdoptionResolution(project_id, standard_id, "adopted_candidate", tuple(ids),
                                  ("Independently reviewed project selection; legal force not certified",))


class HistoricalLineageService:
    """Reconstructs relations as-known without exposing inaccessible endpoints."""

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

    def links(self, record_id: str, access: AccessContext, *,
              as_known: str | None = None, as_valid: str | None = None,
              direction: str = "outgoing", limit: int = 500) -> list[dict[str, Any]]:
        validate_id(record_id, "record_id")
        as_known = normalize_time(as_known, "as_known")
        as_valid = normalize_time(as_valid, "as_valid")
        if direction not in {"incoming", "outgoing"}:
            raise ValidationFailure("direction must be incoming or outgoing")
        if not 1 <= limit <= 1000:
            raise ValidationFailure("Invalid limit")
        if self.repository.get(record_id, access, as_known=as_known, as_valid=as_valid) is None:
            return []
        column = "source_id" if direction == "outgoing" else "target_id"
        with self.repository._connect() as db:
            # Latest event per directed edge at the requested transaction time.
            events = db.execute(f"""SELECT e.* FROM relation_events e
                WHERE e.{column}=?
                  AND (? IS NULL OR e.recorded_at<=?)
                  AND e.rowid=(SELECT MAX(x.rowid) FROM relation_events x
                      WHERE x.source_id=e.source_id AND x.target_id=e.target_id
                      AND x.relation_type=e.relation_type
                      AND (? IS NULL OR x.recorded_at<=?))
                ORDER BY e.recorded_at DESC LIMIT ?""",
                (record_id, as_known, as_known, as_known, as_known, limit)).fetchall()
        result = []
        for event in events:
            if not event["active"]:
                continue
            other_id = event["target_id"] if direction == "outgoing" else event["source_id"]
            other = self.repository.get(other_id, access, as_known=as_known, as_valid=as_valid)
            if other is not None:
                result.append({"relation_type": event["relation_type"],
                               "record_id": other_id, "revision": other.revision,
                               "recorded_at": event["recorded_at"]})
        return result


class KnowledgePromotionService:
    """Creates a NEW public/org summary from private knowledge after review.

    No hidden origin ID, raw source quote, or private details are included in
    the published record. The mapping remains in a protected internal audit
    table, transactionally inserted together with the new global record.
    """

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

    def promote(self, *, origin_record_id: str, expected_revision: int,
                public_statement: str, access: AccessContext,
                privacy_evidence_id: str, licence_evidence_id: str,
                target_scope: Scope = Scope.GLOBAL,
                target_organization_id: str | None = None,
                new_record_id: str | None = None) -> RecordRevision:
        for name, value in (("origin_record_id", origin_record_id),
                            ("privacy_evidence_id", privacy_evidence_id),
                            ("licence_evidence_id", licence_evidence_id)):
            validate_id(value, name)
        if target_scope not in {Scope.GLOBAL, Scope.ORGANIZATION}:
            raise ValidationFailure("Promotion target must be organization or global")
        if not access.can_write or not access.can_review:
            raise AuthorizationDenied(access.principal_id if access is not None else "anonymous", "knowledge_access", details={"reason": "Independent reviewer and write authority required"})
        if not isinstance(public_statement, str) or not public_statement.strip() or len(public_statement) > 1000:
            raise ValidationFailure("Explicit bounded public summary required")
        origin = self.repository.get(origin_record_id, access)
        if origin is None or origin.draft.scope != Scope.PROJECT:
            raise NotFound("project_record", origin_record_id, details={"reason": "Readable private origin required"})
        if origin.revision != expected_revision:
            raise ValidationFailure("Origin version changed")
        if origin.draft.status != Lifecycle.ACTIVE:
            raise ValidationFailure("Only independently approved active knowledge can be proposed for promotion")
        # Do not promote an arbitrary 'active' row without the existing
        # ProjectKnowledgeService's independent maker-checker evidence.
        approval = origin.draft.payload.get("approval")
        origin_author = origin.draft.payload.get("author")
        if (origin.draft.kind not in PROJECT_KINDS or not isinstance(origin_author, str)
                or not isinstance(approval, dict) or not approval.get("evidence_id")
                or not approval.get("reviewer") or approval.get("reviewer") == origin_author):
            raise ValidationFailure("An independently approved project item is required")
        if origin_author == access.principal_id:
            raise AuthorizationDenied(access.principal_id if access is not None else "anonymous", "knowledge_access", details={"reason": "Independent reviewer must differ from source author"})
        # References are review-record locators, not proof of completed external reviews.
        draft = RecordDraft(kind="knowledge_summary", scope=target_scope,
                            organization_id=target_organization_id,
                            status=Lifecycle.REVIEWED,
                            payload={"statement": public_statement.strip(),
                                     "review_disclaimer": "Explicitly reviewed summary; not authoritative source verification"})
        require_write(draft, access)
        new_record_id = validate_id(new_record_id or f"promotion:{uuid.uuid4().hex}", "new_record_id")
        now = utc_now()
        with self.repository._lock, self.repository._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                current = self.repository._latest_row(db, origin_record_id)
                if current is None or current["revision"] != expected_revision:
                    raise ValidationFailure("Origin changed before promotion commit")
                db.execute("INSERT INTO knowledge_heads VALUES(?,1)", (new_record_id,))
                self.repository._insert_revision(db,
                    self.repository._values(draft, new_record_id, 1, now,
                                            access.principal_id, "independent reviewed promotion"))
                db.execute("INSERT INTO promotion_audit VALUES(?,?,?,?,?,?,?)",
                    (new_record_id, origin_record_id, expected_revision,
                     access.principal_id, privacy_evidence_id, licence_evidence_id, now))
                db.commit()
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise ValidationFailure("Promotion identifier already exists") from exc
            except Exception:
                db.rollback()
                raise
        record = self.repository.get(new_record_id, access)
        assert record is not None
        return record

    def origin_for_reviewer(self, public_record_id: str, access: AccessContext) -> dict[str, Any] | None:
        """Only a reviewer with access to the original project can see the mapping."""
        validate_id(public_record_id, "public_record_id")
        if not access.can_review:
            raise AuthorizationDenied(access.principal_id if access is not None else "anonymous", "knowledge_access", details={"reason": "Independent reviewer required"})
        with self.repository._connect() as db:
            event = db.execute("SELECT * FROM promotion_audit WHERE public_record_id=?",
                               (public_record_id,)).fetchone()
        if event is None or self.repository.get(event["origin_record_id"], access) is None:
            return None
        return {"origin_record_id": event["origin_record_id"],
                "origin_revision": event["origin_revision"],
                "privacy_evidence_id": event["privacy_evidence_id"],
                "licence_evidence_id": event["licence_evidence_id"],
                "reviewer_id": event["reviewer_id"]}
