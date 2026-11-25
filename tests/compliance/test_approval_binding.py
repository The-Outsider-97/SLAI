"""GOV-P0-02: approvals and grants are bound to every decision-relevant attribute; single use is atomic; old evidence is invalidated."""
from __future__ import annotations

import dataclasses
import json
import threading
import time
import unittest

from src.agents.governance.human_oversight import ApprovalStatus, DecisionDossier, OversightError, OversightManager
from src.agents.governance.processing_context import DataClass, Decision, FINGERPRINT_VERSION, Operation, ProcessingContext
from src.agents.governance.regulatory_context import AuthorizationGrant, GrantError
from src.agents.governance.risk_classification import AISystemProfile
from src.agents.governance.policy_registry import Capability, Policy
from tests.compliance._fixtures import GovernanceTestCase, make_activity

DOSSIER = DecisionDossier(
    summary="Reject loan application based on affordability model", evidence_refs=("model-card-1", "features-7"),
    uncertainty="model confidence 0.71; borderline", alternatives=("approve with lower limit", "manual underwriting"),
    risks=("indirect discrimination",), limitations=("trained on 2019-2023 data",), explanation="income-to-debt ratio dominated the score")


class ApprovalBindingTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.core.activities.register(make_activity("act-1", art22_exception="contract", recipients=("processor-a", "processor-b")), actor="dpo")
        self.requester = self.ident("svc-decider")
        self.reviewer = self.human("reviewer-1")
        self.t0 = time.time()

    def base(self, **over):
        return self.ctx(automated_decision=True, significant_effect=True, identity=self.requester,
                        application_id="app-1", agent_id="governance", **over)

    def approve(self, c, *, oversight=None, conditions=(), single_use=True, max_uses=None):
        om = oversight or self.core.oversight
        aid = om.request_approval(c, DOSSIER, self.requester, now=self.t0, single_use=single_use, max_uses=max_uses)
        om.mark_presented(aid, self.reviewer, now=self.t0 + 1)
        om.decide(aid, self.reviewer, approve=True, rationale="Reviewed affordability evidence and alternatives",
                  reviewed_dossier_digest=DOSSIER.digest(), conditions=conditions, now=self.t0 + 60)
        return aid

    @staticmethod
    def with_approval(c, aid, **over):
        return dataclasses.replace(c, approval_id=aid, **over)

    # ------------------------------------------------------------ every material attribute
    def test_each_material_attribute_change_invalidates_the_approval(self):
        c = self.base(recipients=("processor-a",), destination_country="NL", resource="loan-77", capability=None)
        aid = self.approve(c)
        self.assertEqual(self.core.oversight.verify(aid, c, now=self.t0 + 61), (True, "ok"))
        other_tenant = self.ident("svc-decider", tenant="tenant-b")
        changes = {
            "purpose": dict(purpose="advertising"),
            "operation": dict(operation=Operation.SHARE),
            "data_class": dict(data_class=DataClass.SPECIAL_CATEGORY),
            "subjects": dict(subject_ids=("subj-1", "subj-2")),
            "tenant": dict(tenant_id="tenant-b", identity=other_tenant),
            "principal": dict(identity=self.ident("svc-other")),
            "roles": dict(identity=self.ident("svc-decider", roles=("governance_admin",))),
            "application": dict(application_id="app-2"),
            "agent": dict(agent_id="other-agent"),
            "capability": dict(capability="stem.solve"),
            "resource": dict(resource="loan-78"),
            "recipients": dict(recipients=("processor-a", "processor-b")),
            "destination": dict(destination_country="US"),
            "ai_system": dict(ai_system_id="sys-9"),
            "ai_practices": dict(ai_practices=("social_scoring",)),
            "automated_decision": dict(automated_decision=False),
            "significant_effect": dict(significant_effect=False),
            "child": dict(subject_is_child=True),
            "provenance": dict(provenance_ref="prov-2"),
            "controller": dict(controller_id="ctrl-2"),
            "processor": dict(processor_id="proc-2"),
            "activity": dict(activity_id="act-2"),
            "classification_evidence": dict(classification_evidence="x" * 8 + ".sig", data_digest="abc"),
            "retention_condition": dict(attributes={"retention_days": 3650}),
        }
        for label, change in changes.items():
            with self.subTest(label):
                ok, reason = self.core.oversight.verify(aid, dataclasses.replace(c, **change), now=self.t0 + 61)
                self.assertFalse(ok, f"{label} change was not detected")
                self.assertTrue(reason.startswith("material_change:"), reason)

    def test_reason_names_the_changed_fields_but_never_their_values(self):
        c = self.base(subject_ids=("subject-with-secret-id",))
        aid = self.approve(c)
        ok, reason = self.core.oversight.verify(aid, dataclasses.replace(c, purpose="advertising"), now=self.t0 + 61)
        self.assertIn("request.purpose", reason)
        self.assertNotIn("advertising", reason)
        ok, reason = self.core.oversight.verify(aid, dataclasses.replace(c, subject_ids=("another-subject",)), now=self.t0 + 61)
        self.assertNotIn("another-subject", reason)

    def test_volatile_transport_attributes_and_renewed_identity_keep_the_approval(self):
        c = self.base(attributes={"trace_id": "t-1", "timestamp": 1.0})
        aid = self.approve(c)
        renewed = dataclasses.replace(c, attributes={"trace_id": "t-2", "timestamp": 2.0}, identity=self.ident("svc-decider", ttl=3000.0),
                                      request_id="gov_req_other", requested_at=time.time() + 5)
        self.assertEqual(self.core.oversight.verify(aid, renewed, now=self.t0 + 61), (True, "ok"))

    # ------------------------------------------------------------ regulatory environment
    def test_policy_or_profile_change_invalidates_the_approval(self):
        self.reg_profile(AISystemProfile("sys-1", "loan scorer", intended_purpose="credit decisions"))
        c = self.base(ai_system_id="sys-1")
        aid = self.approve(c)
        self.assertTrue(self.core.oversight.verify(aid, c, now=self.t0 + 61)[0])
        self.reg_profile(AISystemProfile("sys-1", "loan scorer", intended_purpose="credit decisions",
                                                    annex_iii_areas=("essential_services",)))
        ok, reason = self.core.oversight.verify(aid, c, now=self.t0 + 61)
        self.assertFalse(ok)
        self.assertIn("environment.profile", reason)

    def test_policy_set_change_invalidates_the_approval(self):
        c = self.base()
        aid = self.approve(c)
        self.core.policies.register(Policy("ORG-RET-01", 1, "organisational retention rule", ("internal",), mandatory=False))
        ok, reason = self.core.oversight.verify(aid, c, now=self.t0 + 61)
        self.assertFalse(ok)
        self.assertIn("environment.policy_version", reason)

    # ------------------------------------------------------------ substitution / replay / cross tenant
    def test_approval_for_one_request_cannot_be_substituted_for_another(self):
        a, b = self.base(resource="loan-1"), self.base(resource="loan-2")
        aid_a = self.approve(a)
        d = self.core.authorize(self.with_approval(b, aid_a))
        self.assertEqual(d.decision, Decision.REVIEW_REQUIRED)
        self.assertEqual(self.core.oversight.status(aid_a), ApprovalStatus.APPROVED)  # the failed substitution did not burn it
        self.assertEqual(self.core.authorize(self.with_approval(a, aid_a)).decision, Decision.ALLOW_WITH_CONTROLS)

    def test_consumed_approval_cannot_be_replayed(self):
        c = self.base()
        aid = self.approve(c)
        self.assertTrue(self.core.authorize(self.with_approval(c, aid)).allowed)
        self.assertFalse(self.core.authorize(self.with_approval(c, aid)).allowed)

    def test_cross_tenant_approval_reuse_is_refused(self):
        c = self.base()
        aid = self.approve(c)
        foreign = self.ident("svc-decider", tenant="tenant-b")
        d = self.core.authorize(self.with_approval(c, aid, tenant_id="tenant-b", identity=foreign))
        self.assertFalse(d.allowed)
        self.assertEqual(self.core.oversight.status(aid), ApprovalStatus.APPROVED)

    def test_approval_is_not_consumed_for_an_operation_that_cannot_proceed(self):
        c = self.base()
        aid = self.approve(c)
        d = self.core.authorize(self.with_approval(c, aid), precondition_failure=("GOV-GATE-X", "boundary_precondition_failed"))
        self.assertFalse(d.allowed)
        self.assertEqual(self.core.oversight.status(aid), ApprovalStatus.APPROVED)
        self.assertTrue(self.core.authorize(self.with_approval(c, aid)).allowed)

    # ------------------------------------------------------------ explicit, bounded reuse
    def test_reuse_must_be_explicit_and_bounded(self):
        c = self.base()
        with self.assertRaises(OversightError) as cm:
            self.core.oversight.request_approval(c, DOSSIER, self.requester, single_use=False)
        self.assertEqual(cm.exception.code, "reuse_requires_explicit_max_uses")
        aid = self.approve(c, single_use=False, max_uses=2)
        approved = self.with_approval(c, aid)
        self.assertTrue(self.core.authorize(approved).allowed)
        self.assertTrue(self.core.authorize(approved).allowed)
        self.assertFalse(self.core.authorize(approved).allowed)
        self.assertEqual(self.core.oversight.status(aid), ApprovalStatus.CONSUMED)

    def test_reviewer_conditions_become_controls_of_the_decision(self):
        c = self.base()
        aid = self.approve(c, conditions=("redact_free_text", "notify_dpo"))
        d = self.core.authorize(self.with_approval(c, aid))
        self.assertIn("approved_condition:redact_free_text", d.conditions)
        self.assertIn("approved_condition:notify_dpo", d.conditions)

    # ------------------------------------------------------------ atomic single use
    def test_single_use_is_atomic_across_threads_and_independent_connections(self):
        c = self.base()
        aid = self.approve(c)
        managers = [self.core.oversight] + [
            OversightManager(self.core.state_dir / "oversight.db", authority=self.core.identity,
                             environment_binding=self.core._environment_binding) for _ in range(3)]
        self.addCleanup(lambda: [m.close() for m in managers[1:]])
        results, barrier = [], threading.Barrier(len(managers) * 2)

        def spend(manager):
            barrier.wait()
            try:
                manager.consume(aid, c, now=self.t0 + 61)
                results.append("ok")
            except OversightError as exc:
                results.append(exc.code)

        threads = [threading.Thread(target=spend, args=(m,)) for m in managers for _ in range(2)]
        [t.start() for t in threads]
        [t.join(20) for t in threads]
        self.assertEqual(results.count("ok"), 1, results)
        self.assertEqual(results.count("approval_invalid"), len(threads) - 1, results)

    def test_concurrent_revoke_and_consume_never_both_succeed_inconsistently(self):
        c = self.base()
        aid = self.approve(c)
        outcome = {}

        def consume():
            try:
                self.core.oversight.consume(aid, c, now=self.t0 + 61)
                outcome["consume"] = True
            except OversightError:
                outcome["consume"] = False

        def revoke():
            self.core.oversight.revoke(aid, self.reviewer, "customer withdrew", now=self.t0 + 61)

        threads = [threading.Thread(target=f) for f in (consume, revoke)]
        [t.start() for t in threads]
        [t.join(20) for t in threads]
        status = self.core.oversight.status(aid)
        self.assertIn(status, (ApprovalStatus.CONSUMED, ApprovalStatus.REVOKED))
        # A revocation that lands after consumption withdraws the approval from the (running) operation it backs: the final
        # state is REVOKED in every interleaving that revoked at all. Whatever happened, the approval can never be used again.
        self.assertEqual(status, ApprovalStatus.REVOKED)
        with self.assertRaises(OversightError):
            self.core.oversight.consume(aid, c, now=self.t0 + 62)
        self.assertEqual(self.core.oversight.status(aid), ApprovalStatus.REVOKED)  # no resurrection

    # ------------------------------------------------------------ migration
    def test_persisted_approvals_with_an_incompatible_binding_are_invalidated_and_logged(self):
        c = self.base()
        legacy = self.approve(c)
        rec = json.loads(self.core.oversight._db.execute("SELECT record_json FROM approvals WHERE approval_id=?", (legacy,)).fetchone()[0])
        for key in ("binding_version", "binding"):
            rec.pop(key)  # the shape written before binding v2
        self.core.oversight._db.execute("UPDATE approvals SET record_json=? WHERE approval_id=?", (json.dumps(rec), legacy))
        self.assertEqual(self.core.oversight.verify(legacy, c, now=self.t0 + 61), (False, "approval_binding_version_incompatible"))
        self.assertEqual(self.core.oversight.invalidate_incompatible(now=self.t0 + 62), 1)
        self.assertEqual(self.core.oversight.status(legacy), ApprovalStatus.REVOKED)
        events = [r[0] for r in self.core.oversight._db.execute("SELECT event FROM oversight_log WHERE approval_id=?", (legacy,))]
        self.assertIn("invalidated", events)
        self.assertEqual(self.core.oversight.invalidate_incompatible(), 0)  # idempotent
        current = self.approve(c)
        self.assertEqual(self.core.oversight.status(current), ApprovalStatus.APPROVED)

    def test_restart_keeps_current_version_approvals(self):
        import tempfile
        from src.agents.governance.core import GovernanceCore
        c = self.base()
        aid = self.approve(c)
        state = self.core.state_dir
        self.core.close()
        reopened = GovernanceCore(state, master_key=b"unit-test-master-key-0123456789abcdef-XYZ", consent=self.consent, restrictions=self.restrictions)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.migration_report["approvals_invalidated"], 0)
        self.assertEqual(reopened.oversight.verify(aid, c, now=self.t0 + 61), (True, "ok"))

    # ------------------------------------------------------------ grants
    def grant_ctx(self, **over):
        self.core.activities.register(make_activity("act-g", operations=(Operation.EXECUTE, Operation.PROCESS, Operation.EXPORT)), actor="dpo")
        self.core.capabilities.register(Capability("stem.solve", "stem", "solve", allowed_data_classes=frozenset({DataClass.PERSONAL})))
        return self.ctx(operation=Operation.EXECUTE, capability="stem.solve", application_id="app-1", activity_id="act-g", **over)

    def test_grant_is_bound_to_principal_tenant_application_and_decision_characteristics(self):
        c = self.grant_ctx()
        d = self.core.authorize(c)
        self.assertTrue(d.allowed, d.to_dict())
        token = self.core.issue_grant(c, d)
        self.assertEqual(self.core.verify_grant(token, c).principal_id, "svc-writer")
        mismatches = {
            "principal_mismatch": dataclasses.replace(c, identity=self.ident("svc-other")),
            "tenant_mismatch": dataclasses.replace(c, tenant_id="tenant-b", identity=self.ident("svc-writer", tenant="tenant-b")),
            "application_mismatch": dataclasses.replace(c, application_id="app-2"),
            "ai_system_mismatch": dataclasses.replace(c, ai_system_id="sys-1"),
            "decision_characteristics_exceed_grant": dataclasses.replace(c, automated_decision=True, significant_effect=True),
            "operation_not_granted": dataclasses.replace(c, operation=Operation.EXPORT),
            "controller_processor_mismatch": dataclasses.replace(c, controller_id="c-2"),
        }
        for code, ctx in mismatches.items():
            with self.subTest(code):
                with self.assertRaises(GrantError) as cm:
                    self.core.verify_grant(token, ctx)
                self.assertEqual(cm.exception.code, code)

    def test_grant_dies_when_the_regulatory_profile_changes(self):
        self.reg_profile(AISystemProfile("sys-1", "tool", intended_purpose="drafting"))
        c = self.grant_ctx(ai_system_id="sys-1")
        d = self.core.authorize(c)
        self.assertTrue(d.allowed, d.to_dict())
        token = self.core.issue_grant(c, d)
        self.core.verify_grant(token, c)
        self.reg_profile(AISystemProfile("sys-1", "tool", intended_purpose="drafting", interacts_with_natural_persons=True))
        with self.assertRaises(GrantError) as cm:
            self.core.verify_grant(token, c)
        self.assertEqual(cm.exception.code, "environment_changed")

    def test_grants_issued_under_an_older_binding_version_are_rejected_not_silently_accepted(self):
        c = self.grant_ctx()
        d = self.core.authorize(c)
        legacy = AuthorizationGrant._from_payload({
            "grant_id": "grant_legacy", "request_id": c.request_id, "principal_id": "svc-writer", "tenant_id": "tenant-a",
            "intended_purpose": c.purpose, "processing_activity": "act-1", "policy_version": d.policy_version,
            "authorization_expiry": time.time() + 600, "issued_at": time.time(), "capabilities": ["stem.solve"]})
        self.assertEqual(legacy.scope_version, 1)
        token = self.core.grants._seal(legacy)
        with self.assertRaises(GrantError) as cm:
            self.core.verify_grant(token, c)
        self.assertEqual(cm.exception.code, "incompatible_grant_version")

    def test_fingerprint_is_deterministic_and_versioned(self):
        a = self.ctx(subject_ids=("b", "a"), recipients=("y", "x"), ai_practices=("Social_Scoring",))
        b = self.ctx(subject_ids=("a", "b"), recipients=("x", "y"), ai_practices=("social_scoring",), request_id="other", requested_at=1.0)
        self.assertEqual(a.canonical_binding()["v"], FINGERPRINT_VERSION)
        self.assertEqual(dataclasses.replace(a, identity=self.ident("svc-writer")).fingerprint(), dataclasses.replace(b, identity=self.ident("svc-writer")).fingerprint())


if __name__ == "__main__":
    unittest.main()
