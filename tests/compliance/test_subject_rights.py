import unittest

from src.agents.governance.processing_context import Decision, Operation
from tests.compliance._fixtures import GovernanceTestCase, make_activity


class SubjectRightsStateTests(GovernanceTestCase):
    """Engine-side enforcement of restriction (Art. 18), objection (Art. 21) and pending erasure (Art. 17)."""

    def setUp(self):
        super().setUp()
        self.core.activities.register(make_activity("act-1", operations=(Operation.PROCESS, Operation.STORE, Operation.LEARN, Operation.SHARE), allows_learning=True), actor="dpo")

    def test_restriction_blocks_processing_but_not_storage(self):
        self.restrictions.state["subj-1"] = {"restricted": True}
        self.assertIn("GOV-RTS-08", self.rule_ids(self.core.authorize(self.ctx())))
        self.assertEqual(self.core.authorize(self.ctx(operation=Operation.STORE)).decision, Decision.ALLOW)

    def test_objection_to_purpose_blocks_that_purpose_only(self):
        self.restrictions.state["subj-1"] = {"objected_purposes": ["document_editing"]}
        self.assertIn("GOV-RTS-08", self.rule_ids(self.core.authorize(self.ctx())))

    def test_pending_erasure_blocks_learning_and_sharing(self):
        self.restrictions.state["subj-1"] = {"erasure_pending": True}
        for op in (Operation.LEARN, Operation.SHARE, Operation.PROCESS):
            self.assertIn("GOV-RTS-08", self.rule_ids(self.core.authorize(self.ctx(operation=op))), op)

    def test_missing_restriction_state_provider_fails_closed(self):
        self.core.engine.restrictions = None
        self.assertEqual(self.core.authorize(self.ctx()).decision, Decision.QUARANTINE)

    def test_learning_requires_an_activity_that_authorises_it(self):
        self.core.activities.update(make_activity("act-1", operations=(Operation.PROCESS, Operation.LEARN), allows_learning=False), expected_version=1, actor="dpo")
        self.assertIn("GOV-LRN-11", self.rule_ids(self.core.authorize(self.ctx(operation=Operation.LEARN))))

    def test_solely_automated_significant_decision_needs_art22_exception(self):
        d = self.core.authorize(self.ctx(automated_decision=True, significant_effect=True))
        self.assertIn("GOV-ADM-10", self.rule_ids(d))


if __name__ == "__main__":
    unittest.main()
