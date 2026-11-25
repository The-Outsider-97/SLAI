"""GOV-P1-03: the regulatory-profile registry is durable, versioned, authorised and tamper-evident."""
from __future__ import annotations

import dataclasses
import threading
import unittest

from src.agents.governance import risk_classification as rc
from src.agents.governance.core import GovernanceCore
from src.agents.governance.processing_context import Decision
from src.agents.governance.risk_classification import AISystemProfile, ProfileError, ProfileRegistry, ProfileStatus
from tests.compliance._fixtures import GovernanceTestCase, KEY


def profile(**over):
    base = dict(system_id="sys-1", name="Writer", intended_purpose="drafting", is_gpai_model=False, interacts_with_natural_persons=True)
    base.update(over)
    return AISystemProfile(**base)


class ProfileRegistryTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.admin = self.profile_admin()
        self.reg = self.core.profiles

    def put(self, p=None, **kw):
        return self.reg.register(p or profile(), self.admin, kw.pop("reason", "registration"), **kw)

    # ------------------------------------------------------------ authorisation
    def test_changes_require_a_verified_identity_with_a_profile_admin_role(self):
        with self.assertRaises(ProfileError):
            self.reg.register(profile())
        with self.assertRaises(ProfileError):
            self.reg.register(profile(), self.ident("svc", roles=("oversight_reviewer",)), "x")
        forged = dataclasses.replace(self.admin, token=self.admin.token[:-3] + "AAA")
        with self.assertRaises(ProfileError):
            self.reg.register(profile(), forged, "x")
        self.assertIsNone(self.reg.profile("sys-1"))
        self.assertEqual(self.reg.history("sys-1"), [])

    def test_lifecycle_changes_need_a_documented_reason(self):
        self.put()
        for action in (self.reg.suspend, self.reg.retire):
            with self.assertRaises(ProfileError):
                action("sys-1", self.admin, "no")
        self.assertEqual(self.reg.status("sys-1"), ProfileStatus.ACTIVE)

    # ------------------------------------------------------------ versions
    def test_each_change_is_a_new_version_and_identical_content_is_not(self):
        self.put()
        self.put()
        self.assertEqual(self.reg.version("sys-1"), 1)
        self.put(profile(intended_purpose="drafting and editing"))
        self.assertEqual(self.reg.version("sys-1"), 2)
        self.assertEqual([h["version"] for h in self.reg.history("sys-1")], [1, 2])
        self.assertTrue(self.reg.verify())

    def test_stale_writer_is_rejected_with_a_version_conflict(self):
        self.put()
        self.put(profile(intended_purpose="v2"), expected_version=1)
        with self.assertRaises(ProfileError) as cm:
            self.put(profile(intended_purpose="v3"), expected_version=1)
        self.assertEqual(cm.exception.code, "version_conflict")
        self.assertEqual(self.reg.version("sys-1"), 2)

    def test_concurrent_updates_are_serialised_without_gaps_or_duplicates(self):
        self.put()
        errors = []

        def writer(i):
            try:
                self.reg.register(profile(intended_purpose=f"purpose-{i}"), self.admin, "concurrent update")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
        self.assertEqual(errors, [])
        versions = [h["version"] for h in self.reg.history("sys-1")]
        self.assertEqual(versions, list(range(1, 10)))
        self.assertTrue(self.reg.verify())

    # ------------------------------------------------------------ lifecycle effects on decisions
    def test_suspended_and_retired_profiles_block_and_resume_restores(self):
        self.register()
        self.put()
        ctx = self.ctx(ai_system_id="sys-1")
        active = self.core.authorize(ctx)
        self.assertTrue(active.allowed, active)  # a usable profile lets the request through (with its controls)
        self.reg.suspend("sys-1", self.admin, "investigating a model change")
        d = self.core.authorize(ctx)
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-AIS-12", self.rule_ids(d))
        self.assertIn("ai_system_profile_suspended", [r.reason for r in d.results])
        self.reg.resume("sys-1", self.admin, "investigation finished, no change needed")
        self.assertEqual(self.reg.status("sys-1"), ProfileStatus.ACTIVE)
        self.reg.retire("sys-1", self.admin, "system withdrawn from service")
        self.assertIn("ai_system_profile_retired", [r.reason for r in self.core.authorize(ctx).results])
        with self.assertRaises(ProfileError):  # silently re-registering a retired system is not a restore
            self.put()

    def test_restore_creates_a_new_version_with_the_old_content(self):
        self.put(profile(intended_purpose="first purpose"))
        self.put(profile(intended_purpose="second purpose"))
        self.reg.restore("sys-1", self.admin, "second purpose was registered in error", 1)
        self.assertEqual(self.reg.profile("sys-1").intended_purpose, "first purpose")
        hist = self.reg.history("sys-1")
        self.assertEqual([h["version"] for h in hist], [1, 2, 3])
        self.assertEqual(hist[-1]["restored_from"], 1)
        with self.assertRaises(ProfileError):
            self.reg.restore("sys-1", self.admin, "restoring a version that never existed", 9)

    def test_profile_changes_notify_running_operations_and_invalidate_approvals(self):
        self.register()
        self.put()
        d = self.core.authorize(self.ctx(ai_system_id="sys-1"))
        lease = self.core.authorization(d)
        self.assertIsNotNone(lease)
        before = self.core.oversight.binding_for(self.ctx(ai_system_id="sys-1"))[0]
        self.reg.suspend("sys-1", self.admin, "model retrained, re-assessing risk")
        self.assertNotEqual(self.core.oversight.binding_for(self.ctx(ai_system_id="sys-1"))[0], before)
        from src.agents.governance.reassessment import AuthorizationRevoked
        with self.assertRaises(AuthorizationRevoked):  # the running operation rested on the suspended profile
            lease.check()
        self.assertIn("profile_changed", [r["trigger"] for r in self.core.reassessment.log()])

    # ------------------------------------------------------------ durability and integrity
    def test_state_survives_restart_and_history_is_intact(self):
        self.put()
        self.reg.suspend("sys-1", self.admin, "paused for a review of intended purpose")
        state = self.core.state_dir
        self.core.close()
        GovernanceCore.reset_shared()
        again = GovernanceCore(state, master_key=KEY)
        self.addCleanup(again.close)
        self.assertEqual(again.profiles.status("sys-1"), ProfileStatus.SUSPENDED)
        self.assertEqual(again.profiles.version("sys-1"), 2)
        self.assertTrue(again.profiles.verify())
        self.assertEqual(again.profiles.recovery["tampered"], [])

    def test_tampered_record_is_never_trusted(self):
        self.put()
        self.core.profiles._db.execute("UPDATE profile_versions SET profile=replace(profile,'drafting','surveillance') WHERE version=1")
        self.assertFalse(self.reg.verify())
        self.assertFalse(self.core.health()["ready"])
        state = self.core.state_dir
        self.core.close()
        GovernanceCore.reset_shared()
        again = GovernanceCore(state, master_key=KEY)
        self.addCleanup(again.close)
        self.assertEqual(again.profiles.status("sys-1"), ProfileStatus.TAMPERED)
        self.assertIsNone(again.profiles.classification("sys-1"))
        self.assertEqual(again.profiles.recovery["tampered"], ["sys-1"])
        self.assertFalse(again.health()["ready"])

    def test_forged_signature_is_detected_even_with_a_recomputed_hash(self):
        self.put()
        self.assertTrue(self.reg.verify())
        self.core.profiles._db.execute("UPDATE profile_versions SET signature='forged' WHERE version=1")
        self.assertFalse(self.reg.verify())  # the hash chain alone still looks fine: the HMAC is what catches this

    def test_ruleset_change_is_reclassified_as_a_new_recorded_version_exactly_once(self):
        self.put(profile(annex_iii_areas=("employment",)))
        before = self.reg.classification("sys-1")

        class NewRuleset(rc.RiskClassifier):
            def classify(self, p):
                return dataclasses.replace(super().classify(p), ruleset_version="TEST-RULESET-2")

        self.reg._classifier = NewRuleset()
        self.assertEqual(self.core.reclassify_profiles(), ["sys-1"])
        self.assertEqual(self.reg.classification("sys-1").ruleset_version, "TEST-RULESET-2")
        self.assertNotEqual(self.reg.classification("sys-1").ruleset_version, before.ruleset_version)
        last = self.reg.history("sys-1")[-1]
        self.assertEqual((last["version"], last["actor"], last["reason"]), (2, "governance:system", "ruleset_change"))
        self.assertEqual(self.core.reclassify_profiles(), [])  # idempotent
        self.assertEqual(self.reg.version("sys-1"), 2)

    def test_config_bootstrap_is_idempotent_and_reports_errors(self):
        entry = {"system_id": "cfg-1", "name": "Config system", "intended_purpose": "drafting", "is_gpai_model": False,
                 "practices": [], "annex_iii_areas": []}
        out = self.core.bootstrap_profiles([entry, {"name": "missing system id"}])
        self.assertEqual(out["registered"], ["cfg-1"])
        self.assertEqual(len(out["errors"]), 1)
        self.core.bootstrap_profiles([entry])
        self.assertEqual(self.reg.version("cfg-1"), 1)
        self.assertEqual(self.reg.history("cfg-1")[0]["actor"], "governance:system")

    def test_in_memory_registry_without_authorisation_still_works_for_embedding(self):
        reg = ProfileRegistry()
        self.assertEqual(reg.register(profile()).system_id, "sys-1")
        self.assertEqual(reg.version("sys-1"), 1)
        self.assertTrue(reg.verify())


if __name__ == "__main__":
    unittest.main()
