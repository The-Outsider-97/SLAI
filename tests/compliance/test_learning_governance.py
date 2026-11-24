"""Section 10: learning is a separate purpose; deleting source records is not unlearning."""
from __future__ import annotations

import os
import tempfile
import unittest

from src.agents.governance.deletion_adapters import subject_tag
from src.agents.governance.learning_governance import LearningDeletionAdapter, LearningGovernanceError, LearningRegistry
from src.agents.governance.privacy_ledger import PrivacyLedger
from src.agents.governance.processing_context import Decision, LawfulBasis, Operation
from tests.compliance._fixtures import GovernanceTestCase

SALT = b"d" * 32


class GovernedLearningTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.reg = LearningRegistry(":memory:", digest_salt=SALT)
        self.addCleanup(self.reg.close)
        # Serving purpose and learning purpose are registered as SEPARATE activities.
        self.register(activity_id="serve", purposes=("document_editing",), basis=LawfulBasis.CONTRACT)
        self.register(activity_id="train", purposes=("model_training",), basis=LawfulBasis.CONTRACT,
                      operations=(Operation.PROCESS, Operation.STORE, Operation.RETRIEVE, Operation.TRAIN) if hasattr(Operation, "TRAIN") else (Operation.PROCESS, Operation.STORE, Operation.RETRIEVE))

    def decision(self, purpose="model_training", activity="train"):
        return self.core.authorize(self.ctx(purpose=purpose, activity_id=activity))

    def test_serving_approval_cannot_authorise_learning(self):
        serving = self.decision("document_editing", "serve")
        self.assertTrue(serving.allowed)
        with self.assertRaises(LearningGovernanceError):
            self.reg.register_dataset("ds1", "tenant-a", purpose="document_editing", activity_id="serve", decision=serving, created_by="ml")

    def test_blocked_decision_cannot_start_learning(self):
        blocked = self.core.authorize(self.ctx(purpose="model_training", activity_id="no-such-activity"))
        self.assertEqual(blocked.decision, Decision.BLOCK)
        with self.assertRaises(LearningGovernanceError):
            self.reg.register_dataset("ds1", "tenant-a", purpose="model_training", activity_id="train", decision=blocked, created_by="ml")

    def test_lineage_is_complete_or_the_run_is_refused(self):
        ok = self.decision()
        self.assertTrue(ok.allowed, [r.reason for r in ok.blocking_rules])
        self.reg.register_model("m", "1", system_id="sys")
        with self.assertRaises(LearningGovernanceError):
            self.reg.register_training_run("run1", "m", "1", "tenant-a", dataset_ids=["ghost"], purpose="model_training", decision=ok)
        with self.assertRaises(LearningGovernanceError):
            self.reg.register_training_run("run1", "m", "1", "tenant-a", dataset_ids=[], purpose="model_training", decision=ok)

    def test_cross_tenant_dataset_is_refused(self):
        ok = self.decision()
        self.reg.register_dataset("ds1", "tenant-a", purpose="model_training", activity_id="train", decision=ok, created_by="ml")
        self.reg.register_model("m", "1", system_id="sys")
        with self.assertRaises(LearningGovernanceError):
            self.reg.register_training_run("run1", "m", "1", "tenant-b", dataset_ids=["ds1"], purpose="model_training", decision=ok)

    def test_erasure_opens_impacts_and_stays_incomplete_until_remediation_evidence(self):
        ok = self.decision()
        self.reg.register_dataset("ds1", "tenant-a", purpose="model_training", activity_id="train", decision=ok, created_by="ml", subject_ids=["subj-1", "subj-2"])
        self.reg.register_model("m", "1", system_id="sys")
        self.reg.register_training_run("run1", "m", "1", "tenant-a", dataset_ids=["ds1"], purpose="model_training", decision=ok)
        self.assertEqual([m["model_id"] for m in self.reg.models_trained_on("tenant-a", "subj-1")], ["m"])

        ledger = PrivacyLedger(":memory:", tombstone_salt=SALT)
        ledger.register_adapter(LearningDeletionAdapter(self.reg))
        ver = {"verified": True, "method": "m", "verified_by": "idp", "tenant_id": "tenant-a", "subject_id": "subj-1"}
        ledger.request_erasure("r1", "tenant-a", "subj-1", verification=ver)
        out = ledger.execute_erasure("r1")
        # Source membership is gone, but the model still carries the subject's influence: NOT complete.
        self.assertEqual(out.status, "incomplete")
        self.assertFalse(self.reg.subject_in_learning_data("tenant-a", "subj-1"))
        self.assertTrue(self.reg.subject_in_learning_data("tenant-a", "subj-2"))
        impacts = self.reg.open_impacts(model_id="m")
        self.assertEqual(len(impacts), 1)

        with self.assertRaises(LearningGovernanceError):
            self.reg.resolve_impact(impacts[0]["impact_id"], resolution="deleted_the_source_row", evidence_ref="e", reviewed_by="ml")
        with self.assertRaises(LearningGovernanceError):
            self.reg.resolve_impact(impacts[0]["impact_id"], resolution="unlearning_verified", evidence_ref="", reviewed_by="ml")
        self.reg.resolve_impact(impacts[0]["impact_id"], resolution="retrained_without_subject", evidence_ref="retrain-run-9", reviewed_by="ml.lead")
        self.assertEqual(ledger.execute_erasure("r1").status, "completed")

    def test_other_subjects_and_tenants_unaffected(self):
        ok = self.decision()
        self.reg.register_dataset("ds1", "tenant-a", purpose="model_training", activity_id="train", decision=ok, created_by="ml", subject_ids=["s1"])
        self.assertFalse(self.reg.subject_in_learning_data("tenant-b", "s1"))
        self.assertEqual(self.reg.open_erasure_impacts("tenant-b", "s1"), 0)
        self.assertTrue(self.reg.subject_in_learning_data("tenant-a", "s1"))

    def test_membership_stores_digests_not_identifiers(self):
        ok = self.decision()
        self.reg.register_dataset("ds1", "tenant-a", purpose="model_training", activity_id="train", decision=ok, created_by="ml", subject_ids=["jan.devries"])
        rows = self.reg._db.execute("SELECT subject_digest FROM dataset_members").fetchall()
        self.assertTrue(all("jan" not in r[0] for r in rows))


if __name__ == "__main__":
    unittest.main()
