"""Shared helpers for compliance tests. Test doubles implement the provider *interfaces*
(consent/restriction state); everything under test is the real production code."""

from __future__ import annotations

import tempfile
import time
import unittest

from typing import Any, Dict, Mapping, Optional

from src.agents.governance.core import GovernanceCore
from src.agents.governance.processing_context import (
    Art9Condition,
    DataClass,
    IdentityContext,
    LawfulBasis,
    Operation,
    ProcessingContext,
)
from src.agents.governance.processing_registry import ProcessingActivity

KEY = b"unit-test-master-key-0123456789abcdef-XYZ"  # >= 32 bytes; tests only


class FakeConsent:
    def __init__(self) -> None:
        self.state: Dict[tuple, Dict[str, Any]] = {}

    def grant(self, subject: str, purpose: str, **extra: Any) -> None:
        self.state[(subject, purpose)] = {"active": True, "withdrawn": False, **extra}

    def withdraw(self, subject: str, purpose: str) -> None:
        self.state[(subject, purpose)] = {"active": False, "withdrawn": True}

    def consent_snapshot(self, subject_id: str, purpose: str) -> Mapping[str, Any]:
        return self.state.get((subject_id, purpose), {"active": False})


class FakeRestrictions:
    def __init__(self) -> None:
        self.state: Dict[str, Dict[str, Any]] = {}

    def subject_state(self, subject_id: str) -> Mapping[str, Any]:
        return self.state.get(subject_id, {})


REVIEW = {"reviewed_by": "dpo.jansen", "reviewed_at": "2026-11-01"}


def evidence_for(basis: LawfulBasis) -> Dict[str, Any]:
    base: Dict[LawfulBasis, Dict[str, Any]] = {
        LawfulBasis.CONSENT: {"consent_mechanism_ref": "mech-001", "notice_ref": "notice-v3", "withdrawal_mechanism_ref": "withdraw-ui"},
        LawfulBasis.CONTRACT: {"contract_ref": "contract-77", "necessity_assessment_ref": "nec-77", "data_subject_is_party": True},
        LawfulBasis.LEGAL_OBLIGATION: {"legal_provision": "Algemene wet inzake rijksbelastingen art. 52", "provision_scope_ref": "scope-1"},
        LawfulBasis.VITAL_INTERESTS: {"emergency_ref": "emerg-1", "no_alternative_basis_ref": "alt-1", "subject_incapable_of_consent": True},
        LawfulBasis.PUBLIC_TASK: {"task_in_law_ref": "law-ref-1", "authority_attribution_ref": "attr-1", "controller_vested_with_task": True},
        LawfulBasis.LEGITIMATE_INTERESTS: {
            "lia_ref": "lia-9", "purpose_test_passed": True, "necessity_test_passed": True, "balancing_test_passed": True,
        },
    }
    return {**base[basis], **REVIEW}


def make_activity(activity_id: str = "act-1", **over: Any) -> ProcessingActivity:
    basis = over.pop("basis", LawfulBasis.CONTRACT)
    defaults: Dict[str, Any] = dict(
        activity_id=activity_id, name="Document drafting", controller="SLAI Controller BV",
        purposes=("document_editing",), data_categories=(DataClass.PERSONAL,), subject_categories=("users",),
        basis=basis, basis_evidence=evidence_for(basis), operations=(Operation.PROCESS, Operation.STORE, Operation.RETRIEVE),
        retention_days=30, security_measures=("encryption_at_rest",), recipients=("processor-a",),
    )
    defaults.update(over)
    return ProcessingActivity(**defaults)


class GovernanceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.consent = FakeConsent()
        self.restrictions = FakeRestrictions()
        GovernanceCore.reset_shared()
        self.core = GovernanceCore(self._tmp.name, master_key=KEY, consent=self.consent, restrictions=self.restrictions)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(self.core.close)

    # identities --------------------------------------------------------
    def ident(self, principal: str = "svc-writer", tenant: Optional[str] = "tenant-a", actor: str = "service", roles=(), ttl: float = 600.0, now: Optional[float] = None) -> IdentityContext:
        token = self.core.identity.issue(principal, tenant, actor_type=actor, roles=roles, ttl_seconds=ttl, now=now)
        return self.core.identity.verify(token, now=now)

    def human(self, principal: str, roles=("oversight_reviewer",), tenant: Optional[str] = "tenant-a") -> IdentityContext:
        return self.ident(principal, tenant, actor="human", roles=roles)

    def ctx(self, **over: Any) -> ProcessingContext:
        defaults: Dict[str, Any] = dict(
            operation=Operation.PROCESS, purpose="document_editing", data_class=DataClass.PERSONAL,
            classification_verified=True, activity_id="act-1", subject_ids=("subj-1",), tenant_id="tenant-a",
            identity=self.ident(),
        )
        defaults.update(over)
        return ProcessingContext(**defaults)

    def register(self, **over: Any) -> ProcessingActivity:
        return self.core.activities.register(make_activity(**over), actor="dpo.jansen")

    def rule_ids(self, decision) -> set:
        return {r.rule_id for r in decision.blocking_rules}
