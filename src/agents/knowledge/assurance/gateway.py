"""Application-independent façade for the three P0/P1 assurance capabilities.

This is an opt-in shared service, NOT a replacement Knowledge Agent.
Authoritative caller identity is an external prerequisite. In production,
applications should build AccessContext from their established auth/session
boundary; source documents must never supply their own project permission sets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import AccessContext, RecordDraft, RecordRevision, RuleEvaluation
from .project import ProjectKnowledgeService
from .repository import KnowledgeRepository
from .rules import RulesRegistry
from .standards import StandardsRegistry


class KnowledgeAssuranceGateway:
    """Single entry point for canonical knowledge records, standards and rules."""

    def __init__(self, sqlite_path: str | Path, *, provenance_agent: Any = None,
                 verification_agent: Any = None):
        self.repository = KnowledgeRepository(sqlite_path)
        self.standards = StandardsRegistry(self.repository)
        self.rules = RulesRegistry(self.repository)
        self.projects = ProjectKnowledgeService(self.repository)
        # Explicit peer agents remain owners of their own functions.
        self.provenance_agent = provenance_agent
        self.verification_agent = verification_agent

    def put_knowledge(self, draft: RecordDraft, access: AccessContext,
                      *, record_id: str | None = None) -> RecordRevision:
        return self.repository.create(draft, access, record_id=record_id)

    def revise_knowledge(self, record_id: str, expected_revision: int,
                         draft: RecordDraft, access: AccessContext,
                         *, reason: str) -> RecordRevision:
        return self.repository.revise(record_id, expected_revision, draft, access, reason=reason)

    def find_knowledge(self, access: AccessContext | None, *, kind: str | None = None,
                       query: str | None = None, as_known: str | None = None,
                       as_valid: str | None = None, limit: int = 100) -> list[RecordRevision]:
        return self.repository.query(access, kind=kind, text=query,
                                     as_known=as_known, as_valid=as_valid, limit=limit)

    def resolve_applicable_requirements(self, access: AccessContext | None,
                                        context: Mapping[str, Any]) -> list[dict[str, Any]]:
        return self.standards.applicable_requirements(access, context=context)

    def evaluate_rule(self, record_id: str, inputs: dict[str, Any],
                      access: AccessContext | None) -> RuleEvaluation:
        return self.rules.evaluate(record_id, inputs=inputs, access=access)

    def register_provenance_source(self, source_id: str, source_info: Mapping[str, Any]) -> dict[str, Any]:
        """Delegate identity/digest registration to the existing Provenance Agent."""
        if self.provenance_agent is None:
            raise RuntimeError("ProvenanceAgent not attached; cannot register authoritative source reference")
        return self.provenance_agent.register_source(source_id, source_info)

    def delegate_formal_verification(self, request: Any) -> Any:
        """No implicit assertion-to-proof conversion or compliance certification."""
        if self.verification_agent is None:
            raise RuntimeError("VerificationAgent not attached; cannot verify formal property")
        return self.verification_agent.verify(request)
