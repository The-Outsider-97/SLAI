import time
import unittest

from src.agents.governance.human_oversight import ContestationOutcome, DecisionDossier, OversightError, required_oversight_level, OversightLevel
from src.agents.governance.processing_context import DataClass, Decision, IdentityContext, Operation, ProcessingContext
from tests.compliance._fixtures import GovernanceTestCase

DOSSIER = DecisionDossier(
    summary="Reject loan application based on affordability model", evidence_refs=("model-card-1", "features-7"),
    uncertainty="model confidence 0.71; borderline", alternatives=("approve with lower limit", "manual underwriting"),
    risks=("indirect discrimination",), limitations=("trained on 2019-2023 data",), explanation="income-to-debt ratio dominated the score")


class HumanOversightTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.core.activities.register(
            __import__("tests.compliance._fixtures", fromlist=["make_activity"]).make_activity("act-1", art22_exception="contract"), actor="dpo")
        self.requester = self.ident("svc-decider")
        self.t0 = time.time()

    def decision_ctx(self, **over):
        return self.ctx(automated_decision=True, significant_effect=True, identity=self.requester, **over)

    def approve(self, c, reviewer=None, *, present_at=None, decide_at=None, digest=None):
        reviewer = reviewer or self.human("reviewer-1")
        aid = self.core.oversight.request_approval(c, DOSSIER, self.requester, now=self.t0)
        self.core.oversight.mark_presented(aid, reviewer, now=present_at if present_at is not None else self.t0 + 1)
        self.core.oversight.decide(aid, reviewer, approve=True, rationale="Reviewed affordability evidence and alternatives",
                                   reviewed_dossier_digest=digest or DOSSIER.digest(), now=decide_at if decide_at is not None else self.t0 + 60)
        return aid

    def with_approval(self, c, aid):
        return ProcessingContext(**{**{f: getattr(c, f) for f in c.__dataclass_fields__}, "approval_id": aid})

    def test_significant_automated_decision_requires_human_review(self):
        d = self.core.authorize(self.decision_ctx())
        self.assertEqual(d.decision, Decision.REVIEW_REQUIRED)

    def test_routine_lower_risk_operation_needs_no_human(self):
        self.assertEqual(self.core.authorize(self.ctx()).decision, Decision.ALLOW)
        self.assertEqual(required_oversight_level(self.ctx()), OversightLevel.NONE)

    def test_valid_human_approval_unlocks_once_only(self):
        c = self.decision_ctx()
        aid = self.approve(c)
        c2 = self.with_approval(c, aid)
        d = self.core.authorize(c2)
        self.assertEqual(d.decision, Decision.ALLOW_WITH_CONTROLS, d.to_dict())
        self.assertEqual(self.core.authorize(c2).decision, Decision.REVIEW_REQUIRED)  # single use

    def test_ai_agent_approval_is_not_human_oversight(self):
        c = self.decision_ctx()
        aid = self.core.oversight.request_approval(c, DOSSIER, self.requester)
        agent = self.ident("review-agent", actor="agent", roles=("oversight_reviewer",))
        with self.assertRaises(OversightError) as cm:
            self.core.oversight.decide(aid, agent, approve=True, rationale="agent says fine fine fine", reviewed_dossier_digest=DOSSIER.digest())
        self.assertIn("ai_agent", cm.exception.code)

    def test_forged_human_identity_is_rejected(self):
        c = self.decision_ctx()
        aid = self.core.oversight.request_approval(c, DOSSIER, self.requester)
        fake = IdentityContext("reviewer-x", "tenant-a", verified=True, actor_type="human", roles=("oversight_reviewer",), token="a.b")
        with self.assertRaises(OversightError):
            self.core.oversight.decide(aid, fake, approve=True, rationale="I am totally a human", reviewed_dossier_digest=DOSSIER.digest())

    def test_roles_come_from_the_signed_token_not_the_caller(self):
        c = self.decision_ctx()
        aid = self.core.oversight.request_approval(c, DOSSIER, self.requester)
        weak = self.human("reviewer-2", roles=("viewer",))
        escalated = IdentityContext(weak.principal_id, weak.tenant_id, True, "human", ("governance_admin",), weak.expires_at, weak.token)
        with self.assertRaises(OversightError) as cm:
            self.core.oversight.decide(aid, escalated, approve=True, rationale="self-asserted admin role", reviewed_dossier_digest=DOSSIER.digest())
        self.assertEqual(cm.exception.code, "reviewer_not_qualified")

    def test_four_eyes_and_qualification(self):
        c = self.decision_ctx()
        aid = self.core.oversight.request_approval(c, DOSSIER, self.requester)
        self_human = self.ident("svc-decider", actor="human", roles=("oversight_reviewer",))
        with self.assertRaises(OversightError) as cm:
            self.core.oversight.decide(aid, self_human, approve=True, rationale="approving my own request", reviewed_dossier_digest=DOSSIER.digest())
        self.assertEqual(cm.exception.code, "four_eyes_violation")

    def test_material_change_invalidates_approval(self):
        c = self.decision_ctx()
        aid = self.approve(c)
        changed = self.with_approval(self.decision_ctx(subject_ids=("subj-1", "subj-2")), aid)
        self.assertEqual(self.core.authorize(changed).decision, Decision.REVIEW_REQUIRED)
        self.assertIn("material_change", self.core.oversight.verify(aid, changed)[1])

    def test_expired_and_revoked_approvals_are_rejected(self):
        c = self.decision_ctx()
        aid = self.approve(c)
        self.assertFalse(self.core.oversight.verify(aid, c, now=self.t0 + 7200)[0])
        aid2 = self.approve(c)
        self.core.oversight.revoke(aid2, self.human("reviewer-1"), "no longer valid")
        self.assertFalse(self.core.oversight.verify(aid2, c)[0])

    def test_dossier_must_be_complete_and_actually_reviewed(self):
        with self.assertRaises(OversightError):
            self.core.oversight.request_approval(self.decision_ctx(), DecisionDossier("short", (), "", (), (), ()), self.requester)
        c = self.decision_ctx()
        aid = self.core.oversight.request_approval(c, DOSSIER, self.requester)
        reviewer = self.human("reviewer-1")
        self.core.oversight.mark_presented(aid, reviewer, now=self.t0)
        with self.assertRaises(OversightError) as cm:
            self.core.oversight.decide(aid, reviewer, approve=True, rationale="rubber stamp attempt here", reviewed_dossier_digest="0" * 64, now=self.t0 + 60)
        self.assertEqual(cm.exception.code, "dossier_not_reviewed")

    def test_automation_bias_guards(self):
        c = self.decision_ctx()
        reviewer = self.human("reviewer-1")
        aid = self.core.oversight.request_approval(c, DOSSIER, self.requester, now=self.t0)
        with self.assertRaises(OversightError) as cm:  # never shown
            self.core.oversight.decide(aid, reviewer, approve=True, rationale="approve without looking", reviewed_dossier_digest=DOSSIER.digest(), now=self.t0 + 60)
        self.assertEqual(cm.exception.code, "dossier_not_presented")
        self.core.oversight.mark_presented(aid, reviewer, now=self.t0 + 10)
        with self.assertRaises(OversightError) as cm:  # too fast
            self.core.oversight.decide(aid, reviewer, approve=True, rationale="approve instantly please", reviewed_dossier_digest=DOSSIER.digest(), now=self.t0 + 11)
        self.assertEqual(cm.exception.code, "review_too_fast")

    def test_rejection_and_cancellation_leave_the_action_blocked(self):
        c = self.decision_ctx()
        reviewer = self.human("reviewer-1")
        aid = self.core.oversight.request_approval(c, DOSSIER, self.requester)
        self.core.oversight.decide(aid, reviewer, approve=False, rationale="insufficient evidence supplied", reviewed_dossier_digest=DOSSIER.digest())
        self.assertEqual(self.core.authorize(self.with_approval(c, aid)).decision, Decision.REVIEW_REQUIRED)
        aid2 = self.approve(c)
        self.core.oversight.intervene(aid2, reviewer, "cancel", "operator stopped the action")
        self.assertEqual(self.core.authorize(self.with_approval(c, aid2)).decision, Decision.REVIEW_REQUIRED)

    def test_emergency_termination_halts_scoped_processing_and_needs_authority(self):
        ctx = self.ctx()
        self.assertEqual(self.core.authorize(ctx).decision, Decision.ALLOW)
        with self.assertRaises(OversightError):
            self.core.oversight.activate_emergency("tenant:tenant-a", self.human("r", roles=("oversight_reviewer",)), "nope")
        officer = self.human("officer", roles=("safety_officer",))
        self.core.oversight.activate_emergency("tenant:tenant-a", officer, "runaway agent detected")
        d = self.core.authorize(ctx)
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-EMG-15", self.rule_ids(d))
        other = self.core.authorize(self.ctx(tenant_id="tenant-b", identity=self.ident(tenant="tenant-b")))
        self.assertNotIn("GOV-EMG-15", self.rule_ids(other))  # scope is per tenant
        self.core.oversight.release_emergency("tenant:tenant-a", officer, "root cause fixed and verified")
        self.assertEqual(self.core.authorize(ctx).decision, Decision.ALLOW)
        with self.assertRaises(OversightError):
            self.core.oversight.activate_emergency("global", self.ident("bot", actor="agent", roles=("safety_officer",)), "agent trying kill switch")

    def test_contestation_workflow_requires_independent_human_reviewer(self):
        c = self.decision_ctx()
        aid = self.approve(c)  # approver: reviewer-1
        case = self.core.oversight.file_contestation("subj-1", aid, "I dispute the automated rejection")
        with self.assertRaises(OversightError) as cm:
            self.core.oversight.resolve_contestation(case, self.human("reviewer-1"), ContestationOutcome.UPHELD, "same person re-reviewing own decision")
        self.assertEqual(cm.exception.code, "independent_reviewer_required")
        self.core.oversight.resolve_contestation(case, self.human("reviewer-2"), ContestationOutcome.OVERTURNED, "affordability data was outdated")
        self.assertEqual(self.core.oversight.overdue_contestations(), [])
        late = self.core.oversight.file_contestation("subj-2", "none", "dispute", now=self.t0 - 40 * 86400)
        self.assertEqual(len(self.core.oversight.overdue_contestations()), 1)

    def test_approvals_survive_restart(self):
        c = self.decision_ctx()
        aid = self.approve(c)
        path = self.core.state_dir
        self.core.close()
        from src.agents.governance.core import GovernanceCore
        from tests.compliance._fixtures import KEY
        again = GovernanceCore(path, master_key=KEY)
        try:
            self.assertTrue(again.oversight.verify(aid, c)[0])
        finally:
            again.close()


if __name__ == "__main__":
    unittest.main()
