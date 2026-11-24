import unittest

from src.agents.governance.incident_registry import BreachRisk, IncidentError, IncidentKind, IncidentRegistry, IncidentState

HOUR = 3600.0
DAY = 86400.0
T0 = 1_800_000_000.0


class IncidentTests(unittest.TestCase):
    def setUp(self):
        self.reg = IncidentRegistry()

    def breach(self, **kw):
        return self.reg.detect(IncidentKind.PERSONAL_DATA_BREACH, reporter="observability", description="Unencrypted export left on shared drive", now=T0, **kw)

    def test_gdpr_72h_deadline_runs_from_awareness_not_detection(self):
        iid = self.breach(aware_at=T0 - 10 * HOUR)
        self.assertEqual(self.reg.deadlines(iid)["supervisory_authority"]["due_at"], T0 - 10 * HOUR + 72 * HOUR)

    def test_unlikely_risk_needs_no_notification_but_is_still_registered(self):
        iid = self.breach()
        self.reg.assess_breach_risk(iid, BreachRisk.UNLIKELY, "data was encrypted and key not exposed", actor="dpo")
        self.assertNotIn("supervisory_authority", self.reg.deadlines(iid))
        self.assertEqual([b["incident_id"] for b in self.reg.breach_register()], [iid])

    def test_high_risk_requires_data_subject_communication_unless_exempt(self):
        iid = self.breach()
        self.reg.assess_breach_risk(iid, BreachRisk.HIGH, "special category data exposed in clear", actor="dpo")
        self.assertIn("data_subjects", self.reg.deadlines(iid))
        iid2 = self.breach()
        self.reg.assess_breach_risk(iid2, BreachRisk.HIGH, "encrypted data lost; key safe", actor="dpo", subjects_protected_by_encryption=True)
        self.assertNotIn("data_subjects", self.reg.deadlines(iid2))

    def test_processor_notifies_controller_not_authority(self):
        iid = self.breach(is_processor=True)
        self.assertEqual(set(self.reg.deadlines(iid)), {"controller_notification"})

    def test_lifecycle_cannot_skip_steps_or_close_with_unverified_actions(self):
        iid = self.breach()
        with self.assertRaises(IncidentError):
            self.reg.advance(iid, IncidentState.CONTAINED, actor="handler")
        self.reg.advance(iid, IncidentState.CLASSIFIED, actor="handler")
        with self.assertRaises(IncidentError):
            self.reg.advance(iid, IncidentState.CONTAINED, actor="handler")  # no containment recorded
        self.reg.record_containment(iid, "revoked export link and rotated credentials", actor="handler")
        self.reg.advance(iid, IncidentState.CONTAINED, actor="handler")
        with self.assertRaises(IncidentError):
            self.reg.advance(iid, IncidentState.RISK_ASSESSED, actor="dpo")
        self.reg.assess_breach_risk(iid, BreachRisk.RISK, "names and emails exposed to 40 recipients", actor="dpo")
        self.reg.advance(iid, IncidentState.RISK_ASSESSED, actor="dpo")
        self.reg.advance(iid, IncidentState.ESCALATED, actor="dpo")
        with self.assertRaises(IncidentError):  # notification required but not prepared
            self.reg.advance(iid, IncidentState.NOTIFICATION_DECIDED, actor="dpo")
        nid = self.reg.prepare_notification(iid, "supervisory_authority", {"nature": "email export exposure", "contact": "dpo@example.invalid"}, actor="dpo")
        self.reg.advance(iid, IncidentState.NOTIFICATION_DECIDED, actor="dpo")
        self.reg.advance(iid, IncidentState.REMEDIATED, actor="handler")
        idx = self.reg.add_corrective_action(iid, "enforce encryption on exports", owner="eng", due_at=T0 + DAY)
        self.reg.advance(iid, IncidentState.VERIFIED, actor="verification")
        with self.assertRaises(IncidentError):
            self.reg.advance(iid, IncidentState.CLOSED, actor="dpo")
        with self.assertRaises(IncidentError):
            self.reg.verify_corrective_action(iid, idx, verifier="eng", evidence_ref="test-1")
        self.reg.verify_corrective_action(iid, idx, verifier="verification_agent", evidence_ref="test-1")
        self.reg.advance(iid, IncidentState.CLOSED, actor="dpo")
        self.assertEqual(self.reg.get(iid)["state"], "closed")
        self.assertGreaterEqual(len(self.reg.timeline(iid)), 8)

    def test_notifications_are_drafts_and_only_authorised_humans_can_record_submission(self):
        iid = self.breach()
        self.reg.assess_breach_risk(iid, BreachRisk.RISK, "emails exposed to external party", actor="dpo")
        nid = self.reg.prepare_notification(iid, "supervisory_authority", {"nature": "exposure"}, actor="agent")
        self.assertEqual(self.reg.get(iid)["notifications"][0]["status"], "draft")
        with self.assertRaises(IncidentError):
            self.reg.record_submission(iid, nid, ("engineer",), "eng", "AP-123")
        with self.assertRaises(IncidentError):
            self.reg.record_submission(iid, nid, ("dpo",), "dpo", " ")
        self.reg.record_submission(iid, nid, ("dpo",), "dpo", "AP-2026-000123", now=T0 + HOUR)
        self.assertEqual(self.reg.get(iid)["notifications"][0]["status"], "submitted")

    def test_overdue_tracking(self):
        iid = self.breach()
        self.reg.assess_breach_risk(iid, BreachRisk.RISK, "emails exposed to external party", actor="dpo")
        self.assertEqual(self.reg.overdue(now=T0 + 71 * HOUR), [])
        self.assertEqual(self.reg.overdue(now=T0 + 73 * HOUR)[0]["audience"], "supervisory_authority")

    def test_ai_serious_incident_deadlines_are_separate_from_gdpr(self):
        iid = self.reg.detect(IncidentKind.AI_SERIOUS_INCIDENT, reporter="safety", description="High-risk system caused serious harm", now=T0)
        self.assertEqual(self.reg.deadlines(iid)["market_surveillance_authority"]["days"], 15)
        self.assertNotIn("supervisory_authority", self.reg.deadlines(iid))
        self.reg.record_ai_flags(iid, widespread_or_critical_infrastructure=True, actor="safety")
        self.assertEqual(self.reg.deadlines(iid)["market_surveillance_authority"]["days"], 2)
        self.reg.record_ai_flags(iid, death=True, actor="safety")
        self.assertEqual(self.reg.deadlines(iid)["market_surveillance_authority"]["days"], 10)

    def test_incidents_survive_restart(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "inc.db")
            r1 = IncidentRegistry(p)
            iid = r1.detect(IncidentKind.GOVERNANCE_FAILURE, reporter="gov", description="Policy tamper attempt detected", now=T0)
            r2 = IncidentRegistry(p)
            self.assertEqual(r2.get(iid)["state"], "detected")


if __name__ == "__main__":
    unittest.main()
