"""Handler/Observability -> IncidentRegistry bridge (section 12)."""
from __future__ import annotations

import unittest

from src.agents.governance.incident_bridge import IncidentIntake
from src.agents.governance.incident_registry import IncidentKind, IncidentRegistry, IncidentState


class Bridge(unittest.TestCase):
    def setUp(self):
        self.t = [1_000_000.0]
        self.reg = IncidentRegistry(":memory:")
        self.intake = IncidentIntake(self.reg, clock=lambda: self.t[0], dedupe_window_seconds=600)

    def test_irrelevant_events_are_ignored(self):
        self.assertIsNone(self.intake.ingest({"incident_id": "o1", "summary": "latency spike on cache", "severity": "warning"}))
        self.assertIsNone(self.intake.ingest({}))
        self.assertIsNone(self.intake.ingest("not a mapping"))
        self.assertEqual(self.intake.ignored, 3)

    def test_breach_marker_opens_incident_and_starts_72h_clock_at_detection(self):
        iid = self.intake.ingest({"incident_id": "o2", "category": "unauthorized_access", "summary": "tenant_isolation violated for jan@example.nl",
                                  "detected_at": self.t[0] - 3600, "affected_subject_count": 12})
        rec = self.reg.get(iid)
        self.assertEqual(rec["kind"], IncidentKind.PERSONAL_DATA_BREACH.value)
        self.assertEqual(rec["aware_at"], self.t[0] - 3600)
        self.assertNotIn("jan@example.nl", rec["description"])  # sanitised on the way in
        self.assertEqual(rec["evidence_refs"], ["observability:o2"])
        self.assertTrue(self.reg.deadlines(iid))

    def test_future_or_bogus_detection_time_falls_back_to_now(self):
        iid = self.intake.ingest({"incident_id": "o3", "category": "pii_leak", "detected_at": self.t[0] + 99999})
        self.assertEqual(self.reg.get(iid)["aware_at"], self.t[0])

    def test_duplicates_collapse_within_window_and_reopen_after(self):
        e = {"incident_id": "o4", "category": "data_breach"}
        first = self.intake.ingest(e)
        self.assertEqual(self.intake.ingest(e), first)
        self.t[0] += 601
        self.assertNotEqual(self.intake.ingest(e), first)

    def test_explicit_kind_wins_and_unknown_kind_is_not_dropped(self):
        self.assertEqual(self.reg.get(self.intake.ingest({"incident_id": "a", "governance_kind": "ai_serious_incident"}))["kind"], "ai_serious_incident")
        self.assertEqual(self.reg.get(self.intake.ingest({"incident_id": "b", "governance_kind": "weird"}))["kind"], "governance_failure")

    def test_prohibited_practice_takes_precedence(self):
        self.assertEqual(self.intake.classify({"summary": "prohibited_practice attempt: pii"}), IncidentKind.PROHIBITED_PRACTICE_ATTEMPT)

    def test_bridge_never_submits_to_an_authority(self):
        iid = self.intake.ingest({"incident_id": "o5", "category": "data_breach"})
        self.assertEqual(self.reg.get(iid)["notifications"], [])
        self.assertEqual(self.reg._load(iid)[0], IncidentState.DETECTED)

    def test_shared_memory_subscription_contains_failures(self):
        class FakeSM:
            def __init__(self):
                self.cbs = []

            def subscribe(self, pattern, cb, once=True):
                assert once is False
                self.cbs.append((pattern, cb))

        sm = FakeSM()
        self.intake.attach(sm)
        pattern, cb = sm.cbs[0]
        self.assertEqual(pattern, "*:latest_incident")
        cb({"incident_id": "o6", "category": "data_breach"})
        self.assertEqual(len(self.reg.breach_register()), 1)
        # the real SharedMemory sends a change event (key as repr), not the value: the bridge must read the value back
        class Store(FakeSM):
            def get(self, key):
                return {"incident_id": "o8", "category": "data_breach"} if key == "obs:latest_incident" else None

        real = Store()
        self.intake.attach(real)
        _, cb2 = real.cbs[0]
        cb2({"event_type": "put", "key": "'obs:latest_incident'", "timestamp": "t", "metadata": {}, "value_fingerprint": "f"})
        cb2({"event_type": "delete", "key": "'obs:latest_incident'"})  # deletions are not incidents
        self.assertEqual(len(self.reg.breach_register()), 2)
        cb(object())  # malformed payload must not raise into the publisher
        cb({"incident_id": "o7", "governance_kind": None, "category": "data_breach", "affected_subject_count": "NaN"})


if __name__ == "__main__":
    unittest.main()
