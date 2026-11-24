import time
import unittest

from src.agents.governance.policy_registry import Capability, MandatoryPolicyError, Policy
from src.agents.governance.processing_context import DataClass, Decision, GovernanceDenied, LawfulBasis, Operation, ProcessingContext
from src.agents.governance.regulatory_context import GrantError
from src.agents.governance.policy_engine import _r

from tests.compliance._fixtures import GovernanceTestCase


class ProcessingAuthorityTests(GovernanceTestCase):
    def test_personal_data_without_registered_activity_is_blocked(self):
        d = self.core.authorize(self.ctx(activity_id=None))
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-LAW-04", self.rule_ids(d))

    def test_registered_activity_allows_matching_request(self):
        self.register()
        d = self.core.authorize(self.ctx())
        self.assertEqual(d.decision, Decision.ALLOW, d.to_dict())

    def test_purpose_and_operation_must_match_activity(self):
        self.register()
        self.assertIn("GOV-PUR-05", self.rule_ids(self.core.authorize(self.ctx(purpose="advertising"))))
        self.assertIn("GOV-PUR-05", self.rule_ids(self.core.authorize(self.ctx(operation=Operation.SHARE))))

    def test_unclassified_is_quarantined_not_allowed(self):
        d = self.core.authorize(self.ctx(data_class=DataClass.UNCLASSIFIED, classification_verified=False))
        self.assertEqual(d.decision, Decision.QUARANTINE)

    def test_unverified_non_personal_claim_is_treated_as_unclassified(self):
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_verified=False)
        self.assertEqual(c.effective_data_class, DataClass.UNCLASSIFIED)
        self.assertNotEqual(self.core.authorize(c).decision, Decision.ALLOW)

    def test_verified_non_personal_is_not_burdened(self):
        c = ProcessingContext(Operation.PROCESS, "geometry_conversion", data_class=DataClass.NON_PERSONAL, classification_verified=True)
        self.assertEqual(self.core.authorize(c).decision, Decision.ALLOW)

    def test_inactive_or_suspended_activity_is_blocked(self):
        from src.agents.governance.processing_registry import ActivityStatus
        self.register()
        self.core.activities.set_status("act-1", ActivityStatus.SUSPENDED, actor="dpo", reason="review")
        self.assertIn("GOV-LAW-04", self.rule_ids(self.core.authorize(self.ctx())))

    def test_recipient_must_be_registered(self):
        self.register()
        self.assertIn("GOV-TRF-09", self.rule_ids(self.core.authorize(self.ctx(recipients=("unknown-processor",)))))

    def test_missing_forged_expired_and_mismatched_identity(self):
        self.register()
        # missing
        self.assertIn("GOV-ID-01", self.rule_ids(self.core.authorize(self.ctx(identity=None))))
        # hand-built "verified" context without a signed token
        from src.agents.governance.processing_context import IdentityContext
        fake = IdentityContext("svc-writer", "tenant-a", verified=True, token=None)
        self.assertIn("GOV-ID-01", self.rule_ids(self.core.authorize(self.ctx(identity=fake))))
        # forged: valid-looking token with a tampered signature
        good = self.ident()
        forged = IdentityContext(good.principal_id, good.tenant_id, True, token=good.token[:-4] + "AAAA")
        self.assertIn("GOV-ID-01", self.rule_ids(self.core.authorize(self.ctx(identity=forged))))
        # expired
        old = self.ident(now=time.time() - 7200, ttl=60)
        self.assertIn("GOV-ID-01", self.rule_ids(self.core.authorize(self.ctx(identity=old))))
        # tenant mismatch
        other = self.ident(tenant="tenant-b")
        self.assertIn("GOV-ID-01", self.rule_ids(self.core.authorize(self.ctx(identity=other))))

    def test_identity_is_fail_closed_without_signing_key(self):
        from src.agents.governance.core import GovernanceCore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            keyless = GovernanceCore(d, master_key=None)
            try:
                self.assertFalse(keyless.health()["ready"])
                self.assertFalse(keyless.identity.available)
                c = ProcessingContext(Operation.PROCESS, "x", data_class=DataClass.PERSONAL, classification_verified=True)
                self.assertFalse(keyless.authorize(c).allowed)
            finally:
                keyless.close()

    def test_enforce_raises_and_modification_decisions_are_not_plain_allows(self):
        with self.assertRaises(GovernanceDenied):
            self.core.enforce(self.ctx(activity_id=None))
        self.assertFalse(Decision.REDACT.permits_execution)
        self.assertFalse(Decision.RESTRICT.permits_execution)

    def test_advisory_rules_cannot_relax_a_mandatory_block(self):
        from src.agents.governance.policy_engine import PolicyEngine
        engine = self.core.engine
        engine._rules = tuple(engine._rules) + (lambda c, e: [_r("ADV-1", Decision.ALLOW, "learned model says fine")],)
        self.assertEqual(self.core.authorize(self.ctx(activity_id=None)).decision, Decision.BLOCK)

    def test_rule_exception_fails_closed(self):
        def boom(c, e):
            raise RuntimeError("bug")
        self.core.engine._rules = tuple(self.core.engine._rules) + (boom,)
        d = self.core.authorize(ProcessingContext(Operation.PROCESS, "x", data_class=DataClass.NON_PERSONAL, classification_verified=True))
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-ERR-99", self.rule_ids(d))

    def test_mandatory_policies_cannot_be_deactivated_or_downgraded(self):
        with self.assertRaises(MandatoryPolicyError):
            self.core.policies.deactivate("GOV-LAW-04")
        with self.assertRaises(MandatoryPolicyError):
            self.core.policies.register(Policy("GOV-LAW-04", 2, "weakened", (), mandatory=False))
        with self.assertRaises(MandatoryPolicyError):
            self.core.policies.register(Policy("GOV-LAW-04", 1, "same version", ()))

    def test_unregistered_capability_is_blocked_and_registered_one_is_scoped(self):
        c = ProcessingContext(Operation.EXECUTE, "calc", data_class=DataClass.NON_PERSONAL, classification_verified=True, capability="stem.solve", identity=self.ident())
        self.assertIn("GOV-CAP-13", self.rule_ids(self.core.authorize(c)))
        self.core.capabilities.register(Capability("stem.solve", "stem", "solve"))
        self.assertEqual(self.core.authorize(c).decision, Decision.ALLOW)
        personal = ProcessingContext(Operation.EXECUTE, "calc", data_class=DataClass.PERSONAL, classification_verified=True, capability="stem.solve", identity=self.ident())
        self.assertIn("GOV-CAP-13", self.rule_ids(self.core.authorize(personal)))

    def test_decisions_are_recorded_without_payload(self):
        self.core.authorize(self.ctx(activity_id=None))
        log = self.core.decision_log(1)[0]
        self.assertEqual(set(log), {"request_id", "fingerprint", "decision", "rules", "policy_version", "at"})

    # ------------------------------------------------------------ grants
    def test_grant_roundtrip_and_scope_binding(self):
        self.register()
        c = self.ctx()
        d = self.core.authorize(c)
        token = self.core.issue_grant(c, d)
        grant = self.core.verify_grant(token, c)
        self.assertEqual(grant.processing_activity, "act-1")
        self.assertEqual(grant.to_contract()["intended_purpose"], "document_editing")
        with self.assertRaises(GrantError) as cm:
            self.core.verify_grant(token, self.ctx(purpose="advertising"))
        self.assertEqual(cm.exception.code, "purpose_mismatch")

    def test_grant_forged_expired_revoked_and_policy_bound(self):
        self.register()
        c = self.ctx()
        d = self.core.authorize(c)
        token = self.core.issue_grant(c, d, ttl=60)
        body, sig = token.split(".")
        with self.assertRaises(GrantError) as cm:
            self.core.verify_grant(body + "." + sig[:-3] + "AAA")
        self.assertEqual(cm.exception.code, "forged")
        with self.assertRaises(GrantError) as cm:
            self.core.grants.verify(token, now=time.time() + 3600)
        self.assertEqual(cm.exception.code, "expired")
        self.core.grants.revoke(self.core.verify_grant(token).grant_id, "test")
        with self.assertRaises(GrantError) as cm:
            self.core.verify_grant(token)
        self.assertEqual(cm.exception.code, "revoked")
        # policy change invalidates outstanding grants
        token2 = self.core.issue_grant(c, d)
        self.core.policies.register(Policy("EXTRA-1", 1, "stricter extra", ("test",), mandatory=False))
        with self.assertRaises(GrantError) as cm:
            self.core.verify_grant(token2)
        self.assertEqual(cm.exception.code, "policy_version_changed")

    def test_blocked_decision_cannot_be_turned_into_a_grant(self):
        c = self.ctx(activity_id=None)
        d = self.core.authorize(c)
        with self.assertRaises(GrantError):
            self.core.issue_grant(c, d)

    def test_delegation_only_narrows_and_is_depth_limited(self):
        self.core.capabilities.register(Capability("stem.solve", "stem", "solve"))
        self.core.capabilities.register(Capability("stem.plot", "stem", "plot"))
        c = ProcessingContext(Operation.EXECUTE, "calc", data_class=DataClass.NON_PERSONAL, classification_verified=True, capability="stem.solve", identity=self.ident())
        token = self.core.issue_grant(c, self.core.authorize(c))
        child = self.core.grants.delegate(token, to_agent="sub", capabilities=["stem.solve"])
        with self.assertRaises(GrantError) as cm:
            self.core.grants.delegate(token, to_agent="sub", capabilities=["stem.plot"])
        self.assertEqual(cm.exception.code, "delegation_exceeds_scope")
        grandchild = self.core.grants.delegate(child, to_agent="sub2", capabilities=["stem.solve"])
        with self.assertRaises(GrantError) as cm:
            self.core.grants.delegate(grandchild, to_agent="sub3", capabilities=["stem.solve"])
        self.assertEqual(cm.exception.code, "delegation_not_permitted")
        self.core.grants.revoke(self.core.verify_grant(token).grant_id, "parent revoked")
        with self.assertRaises(GrantError):
            self.core.verify_grant(child)

    def test_state_survives_restart(self):
        self.register()
        path = self.core.state_dir
        self.core.close()
        from src.agents.governance.core import GovernanceCore
        from tests.compliance._fixtures import KEY
        again = GovernanceCore(path, master_key=KEY, consent=self.consent, restrictions=self.restrictions)
        try:
            self.assertIsNotNone(again.activities.get("act-1"))
            self.assertEqual(again.authorize(self.ctx()).decision, Decision.ALLOW)
        finally:
            again.close()


if __name__ == "__main__":
    unittest.main()
