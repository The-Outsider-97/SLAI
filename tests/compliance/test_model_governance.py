"""Model release rules: classification, erasure impacts, lineage and GPAI documentation."""
from __future__ import annotations

import unittest

from src.agents.governance.learning_governance import LearningGovernanceError, LearningRegistry
from src.agents.governance.risk_classification import AISystemProfile, RiskClassifier

SALT = b"d" * 32


class Decision_:
    from src.agents.governance.processing_context import Decision as _D

    decision = _D.ALLOW
    request_id = "req-1"
    context_fingerprint = "fp"


GPAI_DOCS = {
    "technical_documentation_ref": "t", "downstream_provider_information_ref": "d", "copyright_policy_ref": "c",
    "training_content_summary_ref": "s",
}
SYSTEMIC_DOCS = {
    "model_evaluation_ref": "e", "systemic_risk_mitigation_ref": "m", "serious_incident_process_ref": "i", "cybersecurity_protection_ref": "c",
}


class ModelGovernanceTests(unittest.TestCase):
    def setUp(self):
        self.reg = LearningRegistry(":memory:", digest_salt=SALT)
        self.addCleanup(self.reg.close)
        self.clf = RiskClassifier()

    def trained(self, model="m", version="1", **kw):
        self.reg.register_model(model, version, system_id="sys", **kw)
        self.reg.register_dataset(f"ds-{model}-{version}", "t1", purpose="model_training", activity_id="a", decision=Decision_, created_by="ml", subject_ids=["s1"])
        self.reg.register_training_run(f"run-{model}-{version}", model, version, "t1", dataset_ids=[f"ds-{model}-{version}"], purpose="model_training", decision=Decision_)

    def minimal(self):
        return self.clf.classify(AISystemProfile(system_id="sys", name="Minimal", intended_purpose="document drafting", is_gpai_model=False))

    def test_undeclared_purpose_leaves_open_questions_and_blocks_approval(self):
        self.trained()
        vague = self.clf.classify(AISystemProfile(system_id="sys", name="x", is_gpai_model=False))
        with self.assertRaisesRegex(LearningGovernanceError, "classification_open_questions"):
            self.reg.approve_model("m", "1", classification=vague, approved_by="reviewer")

    def test_model_without_lineage_cannot_be_approved(self):
        self.reg.register_model("m", "1", system_id="sys")
        with self.assertRaisesRegex(LearningGovernanceError, "no_registered_training_lineage"):
            self.reg.approve_model("m", "1", classification=self.minimal(), approved_by="reviewer")

    def test_prohibited_classification_blocks_approval(self):
        self.trained()
        bad = self.clf.classify(AISystemProfile(system_id="sys", name="x", practices=("harmful_manipulation",), is_gpai_model=False))
        with self.assertRaisesRegex(LearningGovernanceError, "prohibited_classification"):
            self.reg.approve_model("m", "1", classification=bad, approved_by="reviewer")

    def test_open_erasure_impact_blocks_approval_and_degrades_status(self):
        self.trained()
        mc = self.minimal()
        self.assertEqual(mc.open_questions, ())
        self.reg.approve_model("m", "1", classification=mc, approved_by="reviewer")
        self.assertEqual(self.reg.model_status("m", "1"), "approved")
        self.reg.open_erasure_impacts("t1", "s1")
        self.assertEqual(self.reg.model_status("m", "1"), "remediation_required")
        self.trained("m", "2")
        self.reg.open_erasure_impacts("t1", "s1")  # no-op for v2: membership already removed
        iid = self.reg.open_impacts(model_id="m", version="1")[0]["impact_id"]
        self.reg.resolve_impact(iid, resolution="exempt_documented", evidence_ref="dpo-memo-3", reviewed_by="dpo")
        self.assertEqual(self.reg.model_status("m", "1"), "approved")

    def test_gpai_documentation_gaps_block_approval(self):
        self.trained("g", "1", gpai=True, training_flops=1e22, docs=dict(list(GPAI_DOCS.items())[:2]))
        a = self.reg.gpai_assessment("g", "1")
        self.assertEqual(sorted(a["missing"]), sorted(set(GPAI_DOCS) - set(list(GPAI_DOCS)[:2])))
        self.assertFalse(a["systemic_risk"])
        with self.assertRaisesRegex(LearningGovernanceError, "gpai_documentation_missing"):
            self.reg.approve_model("g", "1", classification=self.minimal(), approved_by="r")

    def test_unknown_compute_is_assumed_systemic(self):
        self.reg.register_model("g", "2", system_id="sys", gpai=True, docs=GPAI_DOCS)
        a = self.reg.gpai_assessment("g", "2")
        self.assertTrue(a["systemic_risk"] and a["systemic_assumed_because_compute_unknown"])
        self.assertEqual(sorted(a["missing"]), sorted(SYSTEMIC_DOCS))

    def test_complete_gpai_documentation_passes(self):
        self.reg.register_model("g", "3", system_id="sys", gpai=True, training_flops=1e26, docs={**GPAI_DOCS, **SYSTEMIC_DOCS})
        a = self.reg.gpai_assessment("g", "3")
        self.assertTrue(a["systemic_risk"] and a["complete"])

    def test_non_gpai_model_has_no_gpai_obligations(self):
        self.reg.register_model("p", "1", system_id="sys")
        self.assertFalse(self.reg.gpai_assessment("p", "1")["gpai"])


if __name__ == "__main__":
    unittest.main()
