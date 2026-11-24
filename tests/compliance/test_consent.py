import unittest

from src.agents.governance.processing_context import Art9Condition, DataClass, Decision, LawfulBasis
from tests.compliance._fixtures import GovernanceTestCase, evidence_for


class ConsentTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.register(basis=LawfulBasis.CONSENT)

    def test_no_consent_blocks_consent_based_processing(self):
        self.assertIn("GOV-CON-07", self.rule_ids(self.core.authorize(self.ctx())))

    def test_active_consent_allows_and_withdrawal_stops_further_processing(self):
        self.consent.grant("subj-1", "document_editing")
        self.assertEqual(self.core.authorize(self.ctx()).decision, Decision.ALLOW)
        self.consent.withdraw("subj-1", "document_editing")
        d = self.core.authorize(self.ctx())
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-CON-07", self.rule_ids(d))

    def test_withdrawal_revokes_outstanding_grants(self):
        self.consent.grant("subj-1", "document_editing")
        c = self.ctx()
        token = self.core.issue_grant(c, self.core.authorize(c))
        self.assertEqual(self.core.grants.revoke_for_subject("subj-1", "consent withdrawn"), 1)
        from src.agents.governance.regulatory_context import GrantError
        with self.assertRaises(GrantError):
            self.core.verify_grant(token)

    def test_consent_must_be_specific_to_purpose(self):
        self.consent.grant("subj-1", "document_editing", purposes=["something_else"])
        self.assertIn("GOV-CON-07", self.rule_ids(self.core.authorize(self.ctx())))

    def test_consent_without_identified_subject_or_provider_fails_closed(self):
        self.assertIn("GOV-CON-07", self.rule_ids(self.core.authorize(self.ctx(subject_ids=()))))
        self.core.engine.consent = None
        d = self.core.authorize(self.ctx())
        self.assertEqual(d.decision, Decision.QUARANTINE)

    def test_child_requires_parental_authorisation(self):
        self.consent.grant("subj-1", "document_editing")
        self.assertIn("GOV-CON-07", self.rule_ids(self.core.authorize(self.ctx(subject_is_child=True))))
        self.consent.grant("subj-1", "document_editing", parental_authorisation=True)
        self.assertEqual(self.core.authorize(self.ctx(subject_is_child=True)).decision, Decision.ALLOW)

    def test_consent_is_not_a_substitute_for_a_registered_activity(self):
        self.consent.grant("subj-1", "document_editing")
        self.assertIn("GOV-LAW-04", self.rule_ids(self.core.authorize(self.ctx(activity_id="no-such-activity"))))

    def test_explicit_consent_required_for_special_category_under_art9a(self):
        self.core.activities.register(
            __import__("tests.compliance._fixtures", fromlist=["make_activity"]).make_activity(
                "sp", basis=LawfulBasis.CONSENT, data_categories=(DataClass.SPECIAL_CATEGORY,),
                art9_condition=Art9Condition.EXPLICIT_CONSENT,
                art9_evidence={"explicit_consent_mechanism_ref": "mech-explicit", "reviewed_by": "dpo", "reviewed_at": "2026-11-01"}),
            actor="dpo")
        self.consent.grant("subj-1", "document_editing")  # ordinary, not explicit
        c = self.ctx(activity_id="sp", data_class=DataClass.SPECIAL_CATEGORY)
        self.assertIn("GOV-CON-07", self.rule_ids(self.core.authorize(c)))
        self.consent.grant("subj-1", "document_editing", explicit=True)
        self.assertEqual(self.core.authorize(c).decision, Decision.ALLOW)


if __name__ == "__main__":
    unittest.main()


class LedgerBackedConsentTests(GovernanceTestCase):
    """The durable PrivacyLedger is the authority for consent and restriction (audit 4.2)."""

    def setUp(self):
        super().setUp()
        self.register(basis=LawfulBasis.CONSENT)
        self.core.attach_providers(consent=self.core.ledger.provider(), restrictions=self.core.ledger.provider())

    def test_ledger_consent_drives_decisions_per_tenant(self):
        self.assertEqual(self.core.authorize(self.ctx()).decision, Decision.BLOCK)
        self.core.ledger.record_consent("tenant-a", "subj-1", "document_editing", explicit=False, notice_version="v1", evidence_ref="e1")
        self.assertEqual(self.core.authorize(self.ctx()).decision, Decision.ALLOW)
        self.core.ledger.withdraw_consent("tenant-a", "subj-1", "document_editing", evidence_ref="e2")
        self.assertEqual(self.core.authorize(self.ctx()).decision, Decision.BLOCK)

    def test_other_tenants_consent_is_not_visible(self):
        self.core.ledger.record_consent("tenant-b", "subj-1", "document_editing", explicit=False, notice_version="v1", evidence_ref="e1")
        self.assertEqual(self.core.authorize(self.ctx()).decision, Decision.BLOCK)

    def test_ledger_restriction_blocks_processing(self):
        self.core.ledger.record_consent("tenant-a", "subj-1", "document_editing", explicit=False, notice_version="v1", evidence_ref="e1")
        self.core.ledger.set_restriction("tenant-a", "subj-1", True, evidence_ref="r1")
        d = self.core.authorize(self.ctx())
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-RTS-08", self.rule_ids(d))
