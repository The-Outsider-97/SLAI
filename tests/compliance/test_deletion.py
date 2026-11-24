"""Audit findings 4.2 / 4.4: authoritative ledger and verified distributed deletion."""
from __future__ import annotations

import unittest

from src.agents.governance.privacy_ledger import LedgerError, PrivacyLedger

SALT = b"s" * 32
VER = {"verified": True, "method": "authenticated_session", "verified_by": "idp", "tenant_id": "t1", "subject_id": "s1"}


class Store:
    """A real in-memory data store used as a deletion target."""

    def __init__(self, name, *, leaves_residue=False, raises=False):
        self.name, self.leaves_residue, self.raises = name, leaves_residue, raises
        self.data = {("t1", "s1"): "x", ("t2", "s1"): "other tenant", ("t1", "s2"): "other subject"}

    def delete(self, tenant_id, subject_id):
        if self.raises:
            raise RuntimeError("store down")
        if not self.leaves_residue:
            self.data.pop((tenant_id, subject_id), None)
        return {"deleted": 1}

    def verify_absent(self, tenant_id, subject_id):
        return (tenant_id, subject_id) not in self.data


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.l = PrivacyLedger(":memory:", tombstone_salt=SALT)

    def test_consent_lifecycle_and_tenant_scope(self):
        self.assertFalse(self.l.consent_snapshot("t1", "s1", "p")["active"])
        self.l.record_consent("t1", "s1", "p", explicit=True, notice_version="v1", evidence_ref="e1")
        self.assertTrue(self.l.consent_snapshot("t1", "s1", "p")["active"])
        self.assertFalse(self.l.consent_snapshot("t2", "s1", "p")["active"])
        self.l.withdraw_consent("t1", "s1", "p", evidence_ref="e2")
        snap = self.l.consent_snapshot("t1", "s1", "p")
        self.assertFalse(snap["active"]); self.assertTrue(snap["withdrawn"])

    def test_restriction_objection_and_view(self):
        self.l.set_restriction("t1", "s1", True, evidence_ref="r")
        self.l.set_objection("t1", "s1", "marketing", True, evidence_ref="o")
        st = self.l.view("t1").subject_state("s1")
        self.assertTrue(st["restricted"]); self.assertEqual(st["objected_purposes"], ["marketing"])
        self.assertFalse(self.l.view("t2").subject_state("s1")["restricted"])
        self.l.set_objection("t1", "s1", "marketing", False, evidence_ref="o2")
        self.assertEqual(self.l.view("t1").subject_state("s1")["objected_purposes"], [])

    def test_chain_integrity_detects_tampering(self):
        self.l.record_consent("t1", "s1", "p", explicit=True, notice_version="v1", evidence_ref="e1")
        self.assertTrue(self.l.verify_chain())
        self.l._db.execute("UPDATE ledger_events SET payload='{}' WHERE seq=1")
        self.assertFalse(self.l.verify_chain())

    def test_erasure_requires_verification_not_reason(self):
        with self.assertRaises(LedgerError):
            self.l.request_erasure("r1", "t1", "s1", verification={})
        with self.assertRaises(LedgerError):
            self.l.request_erasure("r1", "t1", "s1", verification={**VER, "subject_id": "someone-else"})
        self.assertEqual(self.l.request_erasure("r1", "t1", "s1", verification=VER), "r1")  # no reason argument exists

    def test_completed_only_when_every_store_verifies_absence(self):
        a, b = Store("a"), Store("b")
        self.l.register_adapter(a); self.l.register_adapter(b)
        self.l.request_erasure("r1", "t1", "s1", verification=VER)
        out = self.l.execute_erasure("r1")
        self.assertEqual(out.status, "completed")
        self.assertNotIn(("t1", "s1"), a.data)
        self.assertIn(("t2", "s1"), a.data)  # other tenant untouched
        self.assertIn(("t1", "s2"), b.data)  # other subject untouched

    def test_residue_or_failure_never_reports_completed(self):
        self.l.register_adapter(Store("good")); self.l.register_adapter(Store("residue", leaves_residue=True))
        self.l.register_adapter(Store("down", raises=True))
        self.l.request_erasure("r1", "t1", "s1", verification=VER)
        out = self.l.execute_erasure("r1")
        self.assertEqual(out.status, "incomplete")
        self.assertEqual(out.targets["good"], "verified")
        self.assertEqual(out.targets["residue"], "failed"); self.assertEqual(out.targets["down"], "failed")
        self.assertTrue(self.l.subject_state("t1", "s1")["erasure_pending"])
        self.assertFalse(self.l.must_reerase_after_restore("t1", "s1"))

    def test_retry_after_recovery_completes(self):
        flaky = Store("flaky", raises=True)
        self.l.register_adapter(flaky)
        self.l.request_erasure("r1", "t1", "s1", verification=VER)
        self.assertEqual(self.l.execute_erasure("r1").status, "incomplete")
        flaky.raises = False
        self.assertEqual(self.l.execute_erasure("r1").status, "completed")

    def test_legal_hold_blocks_erasure_until_released(self):
        store = Store("a"); self.l.register_adapter(store)
        self.l.place_hold("h1", "t1", "s1", basis="Art. 17(3)(e) litigation", placed_by="counsel")
        self.l.request_erasure("r1", "t1", "s1", verification=VER)
        out = self.l.execute_erasure("r1")
        self.assertEqual(out.status, "held"); self.assertIn(("t1", "s1"), store.data)
        self.l.release_hold("h1", released_by="counsel")
        self.assertEqual(self.l.execute_erasure("r1").status, "completed")

    def test_no_adapters_cannot_complete(self):
        self.l.request_erasure("r1", "t1", "s1", verification=VER)
        with self.assertRaises(LedgerError):
            self.l.execute_erasure("r1")

    def test_backup_resurrection_protection_and_recipient_notices(self):
        self.l.register_adapter(Store("a")); self.l.register_recipient("t1", "processor-x")
        self.l.request_erasure("r1", "t1", "s1", verification=VER)
        self.l.execute_erasure("r1")
        self.assertTrue(self.l.must_reerase_after_restore("t1", "s1"))
        self.assertFalse(self.l.must_reerase_after_restore("t2", "s1"))
        self.assertEqual(self.l.pending_recipient_notices("r1"), ["processor-x"])
        self.l.confirm_recipient_notice("r1", "processor-x")
        self.assertEqual(self.l.pending_recipient_notices("r1"), [])
        digests = [r[0] for r in self.l._db.execute("SELECT subject_digest FROM tombstones")]
        self.assertTrue(all("s1" not in d for d in digests))  # only salted hashes are kept


if __name__ == "__main__":
    unittest.main()
