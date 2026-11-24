import unittest

from src.agents.governance.processing_context import DataClass, Decision, Operation, ProcessingContext
from src.agents.governance.risk_classification import PROHIBITED_PRACTICES, AISystemProfile
from tests.compliance._fixtures import GovernanceTestCase


class ProhibitedPracticeTests(GovernanceTestCase):
    def _ctx(self, tags, system="sys-1"):
        return ProcessingContext(Operation.PROCESS, "analysis", data_class=DataClass.NON_PERSONAL, classification_verified=True,
                                 ai_system_id=system, ai_practices=tags, identity=self.ident())

    def test_every_art5_practice_is_blocked_without_exception(self):
        self.core.profiles.register(AISystemProfile("sys-1", "test system", intended_purpose="test"))
        for tag in PROHIBITED_PRACTICES:
            with self.subTest(tag=tag):
                d = self.core.authorize(self._ctx((tag,)))
                self.assertEqual(d.decision, Decision.BLOCK)
                self.assertIn("GOV-AI5-03", self.rule_ids(d))

    def test_human_approval_cannot_unblock_a_prohibited_practice(self):
        self.core.profiles.register(AISystemProfile("sys-1", "t", intended_purpose="t"))
        c = self._ctx(("social_scoring",))
        reviewer = self.human("reviewer-1")
        requester = self.ident("svc-1")
        from src.agents.governance.human_oversight import DecisionDossier
        dossier = DecisionDossier("Score persons across contexts", ("ev-1",), "high uncertainty", ("do nothing",), ("discrimination",), ("biased data",))
        aid = self.core.oversight.request_approval(c, dossier, requester)
        self.core.oversight.mark_presented(aid, reviewer, now=1000.0)
        self.core.oversight.decide(aid, reviewer, approve=True, rationale="I approve this scoring request", reviewed_dossier_digest=dossier.digest(), now=1100.0)
        approved = ProcessingContext(**{**{f: getattr(c, f) for f in c.__dataclass_fields__}, "approval_id": aid})
        self.assertEqual(self.core.authorize(approved).decision, Decision.BLOCK)

    def test_unknown_practice_tag_is_quarantined_not_ignored(self):
        self.core.profiles.register(AISystemProfile("sys-1", "t", intended_purpose="t"))
        self.assertEqual(self.core.authorize(self._ctx(("mystery_practice",))).decision, Decision.QUARANTINE)

    def test_complete_statutory_exception_in_registered_profile_is_honoured(self):
        # Art. 5(1)(f): emotion recognition for medical or safety reasons is outside the prohibition.
        self.core.profiles.register(AISystemProfile(
            "med-1", "driver fatigue", intended_purpose="fatigue detection for safety", practices=("emotion_recognition_workplace_education",),
            practice_details={"emotion_recognition_workplace_education": {"medical_or_safety_purpose": True}}))
        d = self.core.authorize(self._ctx(("emotion_recognition_workplace_education",), "med-1"))
        self.assertTrue(d.allowed, d.to_dict())  # allowed, but carries open-question controls
        self.assertNotIn("GOV-AI5-03", self.rule_ids(d))

    def test_exception_cannot_be_claimed_by_the_request_only_by_registered_profile(self):
        self.core.profiles.register(AISystemProfile("sys-1", "t", intended_purpose="t"))
        c = ProcessingContext(Operation.PROCESS, "x", data_class=DataClass.NON_PERSONAL, classification_verified=True, ai_system_id="sys-1",
                              ai_practices=("emotion_recognition_workplace_education",), attributes={"medical_or_safety_purpose": True}, identity=self.ident())
        self.assertEqual(self.core.authorize(c).decision, Decision.BLOCK)

    def test_incomplete_rbi_exception_stays_prohibited(self):
        p = AISystemProfile("rbi", "rbi", intended_purpose="x", practices=("realtime_remote_biometric_id_law_enforcement",),
                            practice_details={"realtime_remote_biometric_id_law_enforcement": {"authorised_objective": True}})
        self.core.profiles.register(p)
        self.assertEqual(self.core.profiles.classification("rbi").tier.value, "prohibited")

    def test_ai_system_without_profile_is_quarantined(self):
        self.assertEqual(self.core.authorize(self._ctx((), "ghost-system")).decision, Decision.QUARANTINE)


if __name__ == "__main__":
    unittest.main()
