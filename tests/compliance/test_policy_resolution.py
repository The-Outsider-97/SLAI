"""GOV-P1-05: policy applicability and deterministic conflict resolution."""

import random
import re
import unittest
from pathlib import Path

from src.agents.governance import policy_resolution as PR
from src.agents.governance.policy_resolution import BASES, ResolutionKind, resolve
from src.agents.governance.processing_context import DataClass, Decision, Operation, RuleResult
from src.agents.governance.risk_classification import AISystemProfile
from tests.compliance._fixtures import GovernanceTestCase

GOV_DIR = Path(PR.__file__).parent
_FACTS_PERSONAL = {"personal": True, "ai_system": False, "protected": True, "transfer_like": False, "automated_decision": False}
_FACTS_NONE = {"personal": False, "ai_system": False, "protected": False, "transfer_like": False, "automated_decision": False}


def rr(rid, decision, reason="r", **kw):
    return RuleResult(rid, decision, reason, **kw)


class ResolutionUnitTests(unittest.TestCase):
    def test_every_rule_id_emitted_by_production_code_has_a_registered_basis(self):
        emitted = set()
        for path in GOV_DIR.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            emitted |= set(re.findall(r"(?:_r|RuleResult)\(\s*[\"'](GOV-[A-Z0-9-]+)[\"']", text))
        self.assertTrue(emitted, "scan found nothing: the scan itself is broken")
        self.assertEqual(sorted(emitted - set(BASES)), [], "rule ids without a registered legal basis")

    def test_every_basis_cites_at_least_one_instrument_and_a_known_applicability(self):
        for b in BASES.values():
            self.assertTrue(b.instruments, b.rule_id)
            self.assertIn(b.applies_when, PR._APPLICABLE, b.rule_id)

    def test_prohibition_outranks_a_permission_and_the_overruled_permission_is_recorded(self):
        out = resolve(Decision.BLOCK, [
            rr("GOV-LAW-04", Decision.ALLOW_WITH_CONTROLS, "adequate"),
            rr("GOV-AI5-03", Decision.BLOCK, "prohibited"),
            rr("GOV-CAP-13", Decision.BLOCK, "capability_not_registered"),
        ])
        self.assertEqual(out.kind, ResolutionKind.DENY)
        self.assertEqual(out.governing[0], "GOV-AI5-03")  # stratum 0 leads the same-severity tie
        self.assertEqual(out.stratum, 0)
        self.assertEqual([c.permitting_rule for c in out.conflicts], ["GOV-LAW-04"])
        self.assertEqual(out.conflicts[0].governs, "GOV-AI5-03")

    def test_resolution_is_independent_of_rule_order(self):
        results = [
            rr("GOV-TRF-09", Decision.BLOCK, "no_transfer_mechanism_for:US"),
            rr("GOV-LAW-04", Decision.BLOCK, "no_registered_processing_activity"),
            rr("GOV-ID-01", Decision.ALLOW_WITH_CONTROLS, "ok"),
            rr("GOV-CAP-13", Decision.BLOCK, "x"),
            rr("GOV-AI5-03", Decision.BLOCK, "p"),
        ]
        base = resolve(Decision.BLOCK, results, _FACTS_PERSONAL).to_dict()
        rng = random.Random(7)
        for _ in range(50):
            shuffled = results[:]
            rng.shuffle(shuffled)
            self.assertEqual(resolve(Decision.BLOCK, shuffled, _FACTS_PERSONAL).to_dict(), base)

    def test_resolver_can_only_tighten_never_relax(self):
        rng = random.Random(11)
        ids = list(BASES)
        decisions = list(Decision)
        for _ in range(300):
            rs = [rr(rng.choice(ids), rng.choice(decisions), f"r{i}") for i in range(rng.randint(0, 6))]
            top = max((r.decision for r in rs), key=lambda d: d.severity, default=Decision.ALLOW)
            claimed = rng.choice(decisions)  # an engine bug could claim anything
            out = resolve(claimed, rs)
            self.assertGreaterEqual(out.decision.severity, top.severity)
            self.assertGreaterEqual(out.decision.severity, claimed.severity)

    def test_kinds(self):
        cases = [
            (Decision.ALLOW, [], ResolutionKind.AUTHORIZE),
            (Decision.ALLOW_WITH_CONTROLS, [rr("GOV-HO-14", Decision.ALLOW_WITH_CONTROLS)], ResolutionKind.AUTHORIZE_WITH_CONDITIONS),
            (Decision.REDACT, [rr("GOV-CAP-13", Decision.REDACT)], ResolutionKind.REMEDIATE),
            (Decision.RESTRICT, [rr("GOV-CAP-13", Decision.RESTRICT)], ResolutionKind.REMEDIATE),
            (Decision.REVIEW_REQUIRED, [rr("GOV-HO-14", Decision.REVIEW_REQUIRED)], ResolutionKind.REVIEW),
            (Decision.QUARANTINE, [rr("GOV-CLS-02", Decision.QUARANTINE, "unclassified_data")], ResolutionKind.CLARIFY),
            (Decision.QUARANTINE, [rr("GOV-AIS-12", Decision.QUARANTINE, "ai_system_has_no_registered_regulatory_profile")], ResolutionKind.CLARIFY),
            (Decision.QUARANTINE, [rr("GOV-CON-07", Decision.QUARANTINE, "consent_provider_unavailable")], ResolutionKind.SUSPEND),
            (Decision.QUARANTINE, [rr("GOV-AIS-12", Decision.QUARANTINE, "high_risk_obligations_applicable_and_release_gate_not_passed")], ResolutionKind.SUSPEND),
            (Decision.BLOCK, [rr("GOV-LAW-04", Decision.BLOCK)], ResolutionKind.DENY),
        ]
        for decision, results, kind in cases:
            with self.subTest(kind=kind, reasons=[r.reason for r in results]):
                self.assertEqual(resolve(decision, results).kind, kind)

    def test_unregistered_rule_still_restricts_and_is_reported(self):
        out = resolve(Decision.BLOCK, [rr("DEPLOY-X-1", Decision.BLOCK, "site_policy")])
        self.assertEqual(out.decision, Decision.BLOCK)
        self.assertEqual(out.unregistered_rules, ("DEPLOY-X-1",))
        self.assertEqual(out.stratum, int(PR.Stratum.EXTENSION))

    def test_a_rule_that_does_not_apply_is_flagged_but_never_relaxes(self):
        out = resolve(Decision.BLOCK, [rr("GOV-LAW-04", Decision.BLOCK, "no_registered_processing_activity")], _FACTS_NONE)
        self.assertIn("GOV-LAW-04:not_applicable_but_emitted:personal_data", out.applicability_anomalies)
        self.assertEqual(out.decision, Decision.BLOCK)

    def test_steps_come_from_the_governing_rules_only(self):
        out = resolve(Decision.BLOCK, [rr("GOV-LAW-04", Decision.BLOCK), rr("GOV-ID-01", Decision.ALLOW_WITH_CONTROLS)])
        self.assertTrue(any("processing activity" in s for s in out.steps))
        self.assertFalse(any("identity token" in s for s in out.steps))

    def test_cumulative_instruments_are_all_named(self):
        out = resolve(Decision.BLOCK, [rr("GOV-CON-07", Decision.BLOCK, "child_consent_without_parental_authorisation")])
        self.assertEqual(set(out.instruments), {"GDPR", "UAVG"})  # UAVG art. 5 supplements, never replaces, GDPR Art. 8


class ResolutionIntegrationTests(GovernanceTestCase):
    def test_real_decisions_carry_a_resolution_that_matches_the_decision(self):
        self.register()
        d = self.core.authorize(self.ctx())
        self.assertIsNotNone(d.resolution)
        self.assertEqual(d.resolution["decision"], d.decision.value)
        self.assertEqual(d.to_dict()["resolution"]["basis_version"], PR.BASIS_VERSION)

    def test_missing_activity_resolves_to_deny_with_remediation_step(self):
        d = self.core.authorize(self.ctx(activity_id="nope"))
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertEqual(d.resolution["kind"], "deny")
        self.assertIn("GOV-LAW-04", d.resolution["governing"])
        self.assertTrue(any("processing activity" in s for s in d.resolution["steps"]))

    def test_unclassified_data_resolves_to_clarify(self):
        self.register()
        d = self.core.authorize(self.ctx(data_class=DataClass.UNCLASSIFIED))
        self.assertEqual(d.decision, Decision.QUARANTINE)
        self.assertEqual(d.resolution["kind"], "clarify")

    def test_prohibited_practice_resolves_in_the_safety_stratum_over_everything_else(self):
        self.reg_profile(AISystemProfile("sys-1", "t", intended_purpose="t"))
        c = self.np_ctx(Operation.PROCESS, "analysis", ai_system_id="sys-1", ai_practices=("social_scoring",), identity=self.ident())
        d = self.core.authorize(c)
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertEqual(d.resolution["stratum"], 0)
        self.assertEqual(d.resolution["governing"][0], "GOV-AI5-03")

    def test_no_production_decision_uses_an_unregistered_rule_id(self):
        self.register()
        for c in (self.ctx(), self.ctx(activity_id="nope"), self.ctx(tenant_id="tenant-b"), self.ctx(purpose="marketing")):
            d = self.core.authorize(c)
            self.assertEqual(d.resolution["unregistered_rules"], [], d.to_dict())
            self.assertEqual(d.resolution["applicability_anomalies"], [], d.to_dict())


if __name__ == "__main__":
    unittest.main()
