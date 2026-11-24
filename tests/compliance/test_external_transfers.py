import unittest

from src.agents.governance.processing_context import Decision, Operation
from tests.compliance._fixtures import GovernanceTestCase, make_activity


class TransferTests(GovernanceTestCase):
    def reg(self, transfers):
        self.core.activities.register(make_activity("act-1", operations=(Operation.PROCESS, Operation.TRANSFER, Operation.SHARE), transfers=tuple(transfers)), actor="dpo")

    def req(self, country, **over):
        return self.ctx(operation=Operation.TRANSFER, destination_country=country, **over)

    def test_eea_destination_needs_no_transfer_mechanism(self):
        self.reg([])
        self.assertEqual(self.core.authorize(self.req("DE")).decision, Decision.ALLOW)

    def test_unregistered_third_country_is_blocked(self):
        self.reg([])
        d = self.core.authorize(self.req("IN"))
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-TRF-09", self.rule_ids(d))

    def test_adequacy_decision_allows_with_controls_and_us_requires_dpf(self):
        self.reg([{"country": "JP", "mechanism": "adequacy", "ref": "ec-2019-419"}, {"country": "US", "mechanism": "adequacy", "ref": "dpf"}])
        self.assertEqual(self.core.authorize(self.req("JP")).decision, Decision.ALLOW_WITH_CONTROLS)
        self.assertEqual(self.core.authorize(self.req("US")).decision, Decision.BLOCK)

    def test_us_with_certified_recipient_is_allowed(self):
        self.reg([{"country": "US", "mechanism": "adequacy", "ref": "dpf", "dpf_certified": True}])
        self.assertEqual(self.core.authorize(self.req("US")).decision, Decision.ALLOW_WITH_CONTROLS)

    def test_scc_requires_transfer_impact_assessment(self):
        self.reg([{"country": "IN", "mechanism": "scc", "ref": "scc-2021"}])
        self.assertEqual(self.core.authorize(self.req("IN")).decision, Decision.BLOCK)
        self.core.activities.update(make_activity("act-1", operations=(Operation.PROCESS, Operation.TRANSFER), transfers=({"country": "IN", "mechanism": "scc", "ref": "scc-2021", "tia_ref": "tia-5"},)), expected_version=1, actor="dpo")
        d = self.core.authorize(self.req("IN"))
        self.assertEqual(d.decision, Decision.ALLOW_WITH_CONTROLS)
        self.assertIn("apply_supplementary_measures", d.conditions)

    def test_art49_derogation_is_never_automatic(self):
        self.reg([{"country": "BR", "mechanism": "art49_derogation", "ref": "d-1", "derogation_ground": "explicit_consent_after_risk_notice"}])
        self.assertEqual(self.core.authorize(self.req("BR")).decision, Decision.REVIEW_REQUIRED)

    def test_unknown_destination_for_disclosure_is_quarantined(self):
        self.reg([])
        self.assertEqual(self.core.authorize(self.ctx(operation=Operation.SHARE, destination_country=None)).decision, Decision.QUARANTINE)

    def test_invalid_transfer_entries_cannot_be_registered(self):
        from src.agents.governance.processing_registry import ActivityRegistrationError
        with self.assertRaises(ActivityRegistrationError):
            self.reg([{"country": "IN", "mechanism": "trust_me", "ref": "x"}])


if __name__ == "__main__":
    unittest.main()
