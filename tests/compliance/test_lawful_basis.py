import unittest

from src.agents.governance.lawful_basis import validate_art9_evidence, validate_basis_evidence
from src.agents.governance.processing_context import Art9Condition, DataClass, Decision, LawfulBasis, Operation
from src.agents.governance.processing_registry import ActivityRegistrationError, ConcurrencyError
from tests.compliance._fixtures import GovernanceTestCase, evidence_for, make_activity


class LawfulBasisTests(GovernanceTestCase):
    def test_all_six_bases_accept_complete_evidence(self):
        for basis in LawfulBasis:
            with self.subTest(basis=basis):
                self.assertEqual(validate_basis_evidence(basis, evidence_for(basis)), [])
                self.core.activities.register(make_activity(f"act-{basis.value}", basis=basis), actor="dpo")

    def test_arbitrary_non_empty_strings_are_not_evidence(self):
        for basis in LawfulBasis:
            with self.subTest(basis=basis):
                self.assertTrue(validate_basis_evidence(basis, {"anything": "yes"}))
                with self.assertRaises(ActivityRegistrationError):
                    self.core.activities.register(make_activity(f"bad-{basis.value}", basis=basis, basis_evidence={"legal_basis": "legitimate interest"}), actor="dpo")

    def test_placeholder_values_are_rejected(self):
        ev = evidence_for(LawfulBasis.CONTRACT)
        ev["contract_ref"] = "TODO"
        self.assertTrue(validate_basis_evidence(LawfulBasis.CONTRACT, ev))

    def test_legitimate_interest_needs_all_three_tests_and_not_public_authority_tasks(self):
        ev = evidence_for(LawfulBasis.LEGITIMATE_INTERESTS)
        ev["balancing_test_passed"] = False
        self.assertTrue(validate_basis_evidence(LawfulBasis.LEGITIMATE_INTERESTS, ev))
        ev = evidence_for(LawfulBasis.LEGITIMATE_INTERESTS)
        ev["public_authority_task_processing"] = True
        self.assertTrue(validate_basis_evidence(LawfulBasis.LEGITIMATE_INTERESTS, ev, controller_is_public_authority=True))

    def test_vital_interests_and_contract_require_their_attestations(self):
        ev = evidence_for(LawfulBasis.VITAL_INTERESTS); ev["subject_incapable_of_consent"] = False
        self.assertTrue(validate_basis_evidence(LawfulBasis.VITAL_INTERESTS, ev))
        ev = evidence_for(LawfulBasis.CONTRACT); ev["data_subject_is_party"] = False
        self.assertTrue(validate_basis_evidence(LawfulBasis.CONTRACT, ev))

    def test_special_category_activity_requires_art9_condition_and_evidence(self):
        with self.assertRaises(ActivityRegistrationError):
            self.core.activities.register(make_activity("sp", data_categories=(DataClass.SPECIAL_CATEGORY,)), actor="dpo")
        with self.assertRaises(ActivityRegistrationError):
            self.core.activities.register(make_activity("sp", data_categories=(DataClass.SPECIAL_CATEGORY,), art9_condition=Art9Condition.HEALTH_CARE, art9_evidence={"law_ref": "x"}), actor="dpo")
        ok = make_activity("sp", data_categories=(DataClass.SPECIAL_CATEGORY,), art9_condition=Art9Condition.HEALTH_CARE,
                           art9_evidence={"law_ref": "Wgbo art. 7:457", "professional_secrecy_ref": "secrecy-1", "reviewed_by": "dpo", "reviewed_at": "2026-11-01"})
        self.core.activities.register(ok, actor="dpo")

    def test_special_category_runtime_requires_activity_to_cover_class(self):
        self.register()  # covers PERSONAL only
        d = self.core.authorize(self.ctx(data_class=DataClass.SPECIAL_CATEGORY))
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-PUR-05", self.rule_ids(d))

    def test_pseudonymised_data_is_still_personal_data(self):
        self.register(activity_id="act-1")
        self.assertEqual(self.core.authorize(self.ctx(data_class=DataClass.PSEUDONYMISED)).decision, Decision.ALLOW)
        self.assertEqual(self.core.authorize(self.ctx(data_class=DataClass.PSEUDONYMISED, activity_id=None)).decision, Decision.BLOCK)

    def test_transaction_and_concurrency_control(self):
        a = self.register()
        with self.assertRaises(ConcurrencyError):
            self.core.activities.register(make_activity(), actor="dpo")
        self.core.activities.update(make_activity(name="renamed"), expected_version=1, actor="dpo")
        with self.assertRaises(ConcurrencyError):
            self.core.activities.update(make_activity(name="stale writer"), expected_version=1, actor="dpo")
        self.assertEqual([h["change"] for h in self.core.activities.history("act-1")], ["register", "update"])

    def test_ropa_export_contains_art30_fields(self):
        self.register()
        rec = self.core.activities.export_ropa()[0]
        for key in ("controller", "purposes", "data_categories", "subject_categories", "recipients", "transfers", "retention_days", "security_measures"):
            self.assertIn(key, rec)


if __name__ == "__main__":
    unittest.main()
