"""GOV-P1-04: controlled adaptive governance (real GovernanceCore, real identities, real persistence)."""

import sqlite3
import unittest

from src.agents.governance.adaptive import AdaptiveError, AdaptiveGovernance, default_bucket
from src.agents.governance.core import GovernanceCore
from src.agents.governance.processing_context import Decision, Operation
from tests.compliance._fixtures import KEY, GovernanceTestCase, structural_test_classifier

ADAPTIVE_CFG = {"mode": "enforce", "min_samples": 10, "min_sources": 2, "max_per_source": 10, "min_validation": 10,
                "max_false_positive_rate": 0.2, "max_escalation_rate": 0.6, "min_live_outcomes": 10, "drift_tolerance": 0.15}


class AdaptiveBase(GovernanceTestCase):
    cfg = ADAPTIVE_CFG

    def setUp(self):
        super().setUp()
        self.core.close()
        GovernanceCore.reset_shared()
        self.core = GovernanceCore(self._tmp.name, master_key=KEY, consent=self.consent, restrictions=self.restrictions, config={"adaptive": dict(self.cfg)})
        self.core.classification.attach_classifier(structural_test_classifier)
        self.addCleanup(self.core.close)
        self.register()
        self.admin = self.ident("gov-admin", None, actor="human", roles=("governance_admin",))
        self.n = 0

    # helpers --------------------------------------------------------------
    def reviewer(self, i):
        return self.ident(f"reviewer-{i}", "tenant-a", actor="human", roles=("oversight_reviewer",))

    def req(self, op=Operation.PROCESS):
        self.n += 1
        return self.ctx(operation=op, resource=f"res-{self.n}")

    def label(self, op, label, source, requester=None):
        c = self.req(op)
        return self.core.record_outcome(c, label, self.reviewer(source), evidence_ref=f"case-{self.n}", requester=requester)

    def seed(self, op, escalate_frac, sources=(1, 2, 3), per_source=12):
        for s in sources:
            for i in range(per_source):
                self.label(op, "should_escalate" if i < round(escalate_frac * per_source) else "routine", s)

    def live(self, version=1):
        p = self.core.adaptive.propose(self.admin)
        self.assertTrue(p["validation"]["passed"], p["validation"])
        self.core.adaptive.promote(p["version"], self.admin, "validated candidate promoted", expected_digest=p["digest"])
        return p


class OutcomeIntegrityTests(AdaptiveBase):
    def test_only_verified_identities_with_an_outcome_role_can_label(self):
        c = self.req()
        with self.assertRaises(AdaptiveError) as e:
            self.core.record_outcome(c, "routine", None, evidence_ref="x")
        self.assertEqual(e.exception.code, "not_authorised")
        with self.assertRaises(AdaptiveError):
            self.core.record_outcome(c, "routine", self.ident("rando", roles=("viewer",)), evidence_ref="x")
        forged = type("I", (), {"token": "forged.token.value"})()
        with self.assertRaises(AdaptiveError):
            self.core.record_outcome(c, "routine", forged, evidence_ref="x")

    def test_a_principal_cannot_label_its_own_request(self):
        with self.assertRaises(AdaptiveError) as e:
            self.core.record_outcome(self.req(), "routine", self.reviewer(1), evidence_ref="x", requester="reviewer-1")
        self.assertEqual(e.exception.code, "self_labelling")

    def test_evidence_reference_and_label_are_required_and_validated(self):
        with self.assertRaises(AdaptiveError):
            self.core.record_outcome(self.req(), "maybe", self.reviewer(1), evidence_ref="x")
        with self.assertRaises(AdaptiveError):
            self.core.record_outcome(self.req(), "routine", self.reviewer(1), evidence_ref="  ")

    def test_replayed_fingerprint_is_not_counted_twice(self):
        c = self.req()
        self.assertTrue(self.core.record_outcome(c, "routine", self.reviewer(1), evidence_ref="a"))
        self.assertFalse(self.core.record_outcome(c, "should_escalate", self.reviewer(2), evidence_ref="b"))  # also cannot flip it
        self.assertEqual(self.core.adaptive._db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0], 1)

    def test_requests_in_these_tests_have_distinct_fingerprints(self):
        self.assertNotEqual(self.req().fingerprint(), self.req().fingerprint())

    def test_per_source_rate_limit(self):
        self.core.adaptive._cfg["rate"] = 5
        for _ in range(5):
            self.label(Operation.PROCESS, "routine", 1)
        with self.assertRaises(AdaptiveError) as e:
            self.label(Operation.PROCESS, "routine", 1)
        self.assertEqual(e.exception.code, "rate_limited")


class LifecycleTests(AdaptiveBase):
    def test_nothing_is_live_before_a_human_promotes_a_validated_candidate(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.assertEqual(self.core.adaptive.status()["state"], "none")
        d = self.core.authorize(self.ctx())
        self.assertTrue(d.allowed, d.to_dict())  # outcomes alone change nothing

    def test_promoted_model_escalates_only_where_verified_outcomes_say_so(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        hot = self.core.authorize(self.ctx(operation=Operation.PROCESS, resource="new-1"))
        cold = self.core.authorize(self.ctx(operation=Operation.STORE, resource="new-2"))
        self.assertEqual(hot.decision, Decision.REVIEW_REQUIRED)
        self.assertIn("GOV-ADP-01", {r.rule_id for r in hot.results})
        self.assertEqual(hot.resolution["kind"], "review")
        self.assertTrue(cold.allowed, cold.to_dict())

    def test_shadow_mode_measures_but_never_changes_a_decision(self):
        self.core.adaptive.mode = type(self.core.adaptive.mode)("shadow")
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        d = self.core.authorize(self.ctx(resource="shadow-1"))
        self.assertTrue(d.allowed, d.to_dict())
        self.assertEqual(self.core.adaptive.stats["shadow_would_escalate"], 1)
        self.assertEqual(self.core.adaptive.stats["escalated"], 0)

    def test_a_candidate_that_failed_validation_cannot_be_promoted(self):
        for i in range(4):
            self.label(Operation.PROCESS, "should_escalate", 1 + i % 3)
        p = self.core.adaptive.propose(self.admin)
        self.assertFalse(p["validation"]["passed"])
        self.assertIn("validation_set_too_small", p["validation"]["failed_gates"])
        with self.assertRaises(AdaptiveError) as e:
            self.core.adaptive.promote(p["version"], self.admin, "trying to force this one", expected_digest=p["digest"])
        self.assertEqual(e.exception.code, "validation_failed")

    def test_promotion_is_bound_to_the_reviewed_digest_role_and_reason(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        p = self.core.adaptive.propose(self.admin)
        with self.assertRaises(AdaptiveError) as e:
            self.core.adaptive.promote(p["version"], self.admin, "reviewed and approved", expected_digest="0" * 64)
        self.assertEqual(e.exception.code, "digest_mismatch")
        with self.assertRaises(AdaptiveError) as e:
            self.core.adaptive.promote(p["version"], self.reviewer(1), "reviewers cannot promote", expected_digest=p["digest"])
        self.assertEqual(e.exception.code, "not_authorised")
        with self.assertRaises(AdaptiveError) as e:
            self.core.adaptive.promote(p["version"], self.admin, "short", expected_digest=p["digest"])
        self.assertEqual(e.exception.code, "reason_required")

    def test_rollback_to_an_earlier_version_and_to_the_mandatory_floor(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        v1 = self.live()
        self.seed(Operation.PROCESS, 1.0, per_source=10)
        p2 = self.core.adaptive.propose(self.admin)
        if p2["validation"]["passed"]:
            self.core.adaptive.promote(p2["version"], self.admin, "second version promoted", expected_digest=p2["digest"])
            self.assertEqual(self.core.adaptive.rollback(self.admin, "second version misbehaves", to_version=v1["version"]), v1["version"])
        self.assertEqual(self.core.adaptive.rollback(self.admin, "return to the mandatory floor"), None)
        self.assertTrue(self.core.authorize(self.ctx(resource="after-rollback")).allowed)
        with self.assertRaises(AdaptiveError):
            self.core.adaptive.rollback(self.admin, "to a version never live", to_version=99)

    def test_every_change_is_in_a_verified_hash_chain(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        self.core.adaptive.rollback(self.admin, "audit trail demonstration")
        events = [h["event"] for h in self.core.adaptive.history()]
        self.assertEqual(events, ["proposed", "promoted", "disabled"])
        self.assertTrue(self.core.adaptive.verify())

    def test_live_version_survives_restart(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        self.core.close()
        GovernanceCore.reset_shared()
        self.core = GovernanceCore(self._tmp.name, master_key=KEY, consent=self.consent, restrictions=self.restrictions, config={"adaptive": dict(self.cfg)})
        self.core.classification.attach_classifier(structural_test_classifier)
        self.addCleanup(self.core.close)
        self.assertEqual(self.core.adaptive.status()["live_version"], 1)
        self.assertEqual(self.core.authorize(self.ctx(resource="after-restart")).decision, Decision.REVIEW_REQUIRED)


class MandatoryControlsTests(AdaptiveBase):
    def test_adaptive_results_can_only_ever_be_review_required(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        for c in (self.ctx(resource="a"), self.ctx(resource="b", activity_id="nope"), self.ctx(resource="c", tenant_id="other")):
            for r in self.core.authorize(c).results:
                if r.rule_id == "GOV-ADP-01":
                    self.assertIs(r.decision, Decision.REVIEW_REQUIRED)

    def test_adaptation_never_relaxes_a_mandatory_outcome(self):
        """Over contexts spanning ALLOW .. BLOCK: the decision with the adaptive layer is never less severe than without it."""
        contexts = lambda: [self.ctx(resource="m1"), self.ctx(resource="m2", activity_id="nope"), self.ctx(resource="m3", purpose="marketing"),
                            self.ctx(resource="m4", operation=Operation.STORE), self.ctx(resource="m5", tenant_id="tenant-z")]
        baseline = [self.core.engine.evaluate(c).decision for c in contexts()]
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        for c, before in zip(contexts(), baseline):
            self.assertGreaterEqual(self.core.engine.evaluate(c).decision.severity, before.severity)

    def test_blocked_requests_stay_blocked_exactly_as_before(self):
        before = self.core.engine.evaluate(self.ctx(resource="b1", activity_id="nope")).decision
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        self.assertEqual(self.core.engine.evaluate(self.ctx(resource="b2", activity_id="nope")).decision, before)

    def test_an_adaptive_review_can_be_resolved_by_a_verified_human_like_any_review(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        c = self.ctx(resource="reviewable")
        r = [x for x in self.core.engine.evaluate(c).results if x.rule_id == "GOV-ADP-01"][0]
        self.assertTrue(r.overridable)

    def test_adaptive_failure_falls_back_to_the_mandatory_floor(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()

        def boom(ctx):
            raise RuntimeError("bucket bug")

        self.core.adaptive._bucket_of = boom
        d = self.core.authorize(self.ctx(resource="floor"))
        self.assertTrue(d.allowed, d.to_dict())
        self.assertGreaterEqual(self.core.adaptive.stats["errors"], 1)

    def test_unknown_mode_has_no_effect(self):
        a = AdaptiveGovernance(mode="aggressive")
        self.assertEqual(a.mode.value, "off")
        self.assertEqual(a.rule(self.req()), [])


class PoisoningTests(AdaptiveBase):
    def test_one_source_flooding_routine_labels_cannot_suppress_a_real_escalation(self):
        self.seed(Operation.PROCESS, 1.0, sources=(1, 2, 3), per_source=12)  # legitimate reviewers: all should escalate
        for i in range(90):  # attacker floods mostly-opposite labels, mixed enough to stay under the label-flip detector
            self.label(Operation.PROCESS, "should_escalate" if i % 10 < 3 else "routine", 9)
        self.seed(Operation.STORE, 0.0)
        params, _ = self.core.adaptive._fit()
        b = params["buckets"][default_bucket(self.core.engine.resolve(self.req(Operation.PROCESS)))]
        self.assertLessEqual(b["n"], 3 * 10 + 10)  # the attacker contributes at most max_per_source
        self.assertGreaterEqual(b["lcb"], 0.5)  # and the verdict stands

    def test_one_source_flooding_escalate_labels_cannot_create_an_escalation(self):
        self.seed(Operation.PROCESS, 0.0, sources=(1, 2, 3), per_source=12)  # reviewers: routine
        for _ in range(90):
            self.label(Operation.PROCESS, "should_escalate", 9)
        params, _ = self.core.adaptive._fit()
        b = list(params["buckets"].values())[0]
        self.assertLess(b["lcb"], 0.5)

    def test_a_single_source_is_never_enough_to_escalate(self):
        for _ in range(40):
            self.label(Operation.PROCESS, "should_escalate", 1)
        params, _ = self.core.adaptive._fit()
        self.assertFalse(self.core.adaptive._escalates(params, default_bucket(self.core.engine.resolve(self.req()))))

    def test_a_source_that_contradicts_every_other_source_is_excluded(self):
        for s in (1, 2, 3):
            for _ in range(10):
                self.label(Operation.PROCESS, "should_escalate", s)
        for _ in range(10):
            self.label(Operation.PROCESS, "routine", 7)  # consistent contrarian
        params, excluded = self.core.adaptive._fit()
        self.assertTrue(any(e.endswith(":reviewer-7") for e in excluded), excluded)

    def test_held_out_outcomes_are_never_used_to_fit(self):
        self.core.adaptive._cfg["max_per_source"] = 10_000  # remove the cap so only the split limits what is fitted
        self.seed(Operation.PROCESS, 1.0, per_source=20)
        held = self.core.adaptive._db.execute("SELECT COUNT(*) FROM outcomes WHERE fold=0").fetchone()[0]
        total = self.core.adaptive._db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        params, _ = self.core.adaptive._fit()
        fitted = sum(b["n"] for b in params["buckets"].values())
        self.assertGreater(held, 0)
        self.assertEqual(fitted, total - held)

    def test_a_flood_cannot_dominate_the_validation_set_either(self):
        self.seed(Operation.PROCESS, 1.0, per_source=30)
        self.seed(Operation.STORE, 0.0, per_source=30)
        before = self.core.adaptive.propose(self.admin)["validation"]["validation_size"]
        for i in range(150):  # attacker floods mixed, mostly-opposite labels (30% escalate: below the label-flip detector)
            self.label(Operation.PROCESS, "should_escalate" if i % 10 < 3 else "routine", 9)
        after = self.core.adaptive.propose(self.admin)["validation"]["validation_size"]
        self.assertLessEqual(after, before + self.core.adaptive._cfg["max_per_source"])

    def test_a_candidate_that_would_escalate_too_much_is_rejected(self):
        self.core.adaptive._cfg["max_esc"] = 0.05
        self.seed(Operation.PROCESS, 1.0, per_source=14)
        p = self.core.adaptive.propose(self.admin)
        self.assertIn("escalation_rate_above_ceiling", p["validation"]["failed_gates"])
        self.assertFalse(p["validation"]["passed"])


class DriftAndTamperTests(AdaptiveBase):
    def test_live_version_that_starts_missing_real_escalations_is_auto_suspended(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        for i in range(30):  # new verified outcomes: a bucket the model does not escalate keeps needing review
            self.label(Operation.RETRIEVE, "should_escalate", 1 + i % 3)
        st = self.core.adaptive.status()
        self.assertEqual(st["state"], "suspended")
        self.assertEqual(self.core.adaptive.history()[-1]["actor"], "governance:auto")
        self.assertTrue(self.core.authorize(self.ctx(resource="post-suspend")).allowed)  # back on the mandatory floor

    def test_tampering_with_the_log_disables_adaptation_and_is_reported(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        self.core.adaptive._db.execute("UPDATE log SET actor='mallory' WHERE event='promoted'")
        self.assertFalse(self.core.adaptive.verify())
        self.core.adaptive._reload()
        self.assertEqual(self.core.adaptive.status()["state"], "tampered")
        self.assertTrue(self.core.authorize(self.ctx(resource="post-tamper")).allowed)
        self.assertFalse(self.core.health()["components"]["adaptive_log_intact"])
        with self.assertRaises(AdaptiveError):
            self.core.adaptive.rollback(self.admin, "cannot operate on a tampered log")

    def test_a_modified_model_no_longer_matches_its_reviewed_digest(self):
        self.seed(Operation.PROCESS, 1.0)
        self.seed(Operation.STORE, 0.0)
        self.live()
        self.core.adaptive._db.execute("UPDATE versions SET params=replace(params,'0.5','0.0') WHERE version=1")
        self.core.adaptive._db.execute("UPDATE versions SET params='{\"threshold\":0.0,\"min_samples\":0,\"min_sources\":0,\"buckets\":{}}' WHERE version=1")
        self.core.adaptive._reload()
        self.assertEqual(self.core.adaptive.status()["state"], "tampered")
        self.assertEqual(self.core.adaptive.rule(self.req()), [])


if __name__ == "__main__":
    unittest.main()
