"""Concurrency, adversarial-input and failure-injection tests for the governance layer."""
from __future__ import annotations

import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from src.agents.governance.human_oversight import DecisionDossier
from src.agents.governance.learning_governance import LearningGovernanceError, LearningRegistry
from src.agents.governance.privacy_ledger import LedgerError, PrivacyLedger
from src.agents.governance.processing_context import Decision, ProcessingContext
from tests.compliance._fixtures import GovernanceTestCase, make_activity
from tests.compliance.test_human_oversight import DOSSIER

VER = {"verified": True, "method": "m", "verified_by": "idp", "tenant_id": "t1", "subject_id": "s1"}


class Store:
    name = "s"

    def __init__(self):
        self.deleted = 0
        self.present = True
        self.lock = threading.Lock()

    def delete(self, t, s):
        with self.lock:
            self.deleted += 1
            self.present = False
        return {}

    def verify_absent(self, t, s):
        return not self.present


class LedgerConcurrency(unittest.TestCase):
    def test_parallel_writes_keep_the_hash_chain_valid_and_lose_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            ledger = PrivacyLedger(os.path.join(tmp, "l.db"), tombstone_salt=b"s" * 32)
            n = 200
            with ThreadPoolExecutor(16) as ex:
                list(ex.map(lambda i: ledger.record_consent("t1", f"s{i}", "p", explicit=False, notice_version="v", evidence_ref="e"), range(n)))
            self.assertEqual(ledger._db.execute("SELECT COUNT(*) FROM ledger_events").fetchone()[0], n)
            self.assertTrue(ledger.verify_chain())
            ledger.close()

    def test_concurrent_erasure_executions_end_completed_exactly_once(self):
        ledger = PrivacyLedger(":memory:", tombstone_salt=b"s" * 32)
        ledger.register_adapter(Store())
        ledger.request_erasure("r1", "t1", "s1", verification=VER)
        with ThreadPoolExecutor(8) as ex:
            outcomes = list(ex.map(lambda _: ledger.execute_erasure("r1").status, range(8)))
        self.assertEqual(set(outcomes), {"completed"})
        self.assertEqual(ledger._db.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0], 1)

    def test_duplicate_request_ids_are_rejected(self):
        ledger = PrivacyLedger(":memory:", tombstone_salt=b"s" * 32)
        ledger.request_erasure("r1", "t1", "s1", verification=VER)
        with self.assertRaises(Exception):
            ledger.request_erasure("r1", "t1", "s1", verification=VER)


class AdversarialInput(unittest.TestCase):
    def setUp(self):
        self.ledger = PrivacyLedger(":memory:", tombstone_salt=b"s" * 32)

    def test_sql_injection_strings_are_inert(self):
        evil = "x'); DROP TABLE ledger_events;--"
        self.ledger.record_consent("t1", evil, evil, explicit=True, notice_version="v", evidence_ref=evil)
        self.assertTrue(self.ledger.consent_snapshot("t1", evil, evil)["active"])
        self.assertTrue(self.ledger.verify_chain())

    def test_blank_and_non_string_identifiers_are_rejected(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(LedgerError):
                self.ledger.record_consent("t1", bad, "p", explicit=True, notice_version="v", evidence_ref="e")

    def test_cross_tenant_verification_cannot_be_replayed(self):
        with self.assertRaises(LedgerError):
            self.ledger.request_erasure("r1", "t2", "s1", verification=VER)  # verification belongs to t1

    def test_huge_payloads_do_not_break_chain(self):
        self.ledger.record_consent("t1", "s1", "p" * 5000, explicit=True, notice_version="v" * 5000, evidence_ref="e" * 5000)
        self.assertTrue(self.ledger.verify_chain())


class FailureInjection(unittest.TestCase):
    def test_adapter_crash_midway_never_completes_and_retry_converges(self):
        ledger = PrivacyLedger(":memory:", tombstone_salt=b"s" * 32)
        calls = {"n": 0}

        class Crashy(Store):
            name = "crashy"

            def delete(self, t, s):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("disk full")
                return super().delete(t, s)

        ledger.register_adapter(Store())
        ledger.register_adapter(Crashy())
        ledger.request_erasure("r1", "t1", "s1", verification=VER)
        self.assertEqual(ledger.execute_erasure("r1").status, "incomplete")
        self.assertFalse(ledger.must_reerase_after_restore("t1", "s1"))
        self.assertEqual(ledger.execute_erasure("r1").status, "completed")

    def test_adapter_lying_about_delete_is_caught_by_independent_verification(self):
        class Liar:
            name = "liar"

            def delete(self, t, s):
                return {"deleted": 999}

            def verify_absent(self, t, s):
                return False

        ledger = PrivacyLedger(":memory:", tombstone_salt=b"s" * 32)
        ledger.register_adapter(Liar())
        ledger.request_erasure("r1", "t1", "s1", verification=VER)
        self.assertEqual(ledger.execute_erasure("r1").status, "incomplete")


class OversightRace(GovernanceTestCase):
    def test_single_use_approval_is_consumed_exactly_once_under_contention(self):
        self.core.activities.register(make_activity("act-1", art22_exception="contract"), actor="dpo")
        requester = self.ident("svc-decider")
        t0 = time.time()
        c = self.ctx(automated_decision=True, significant_effect=True, identity=requester)
        reviewer = self.human("reviewer-1")
        aid = self.core.oversight.request_approval(c, DOSSIER, requester, now=t0)
        self.core.oversight.mark_presented(aid, reviewer, now=t0 + 1)
        self.core.oversight.decide(aid, reviewer, approve=True, rationale="Reviewed affordability evidence and alternatives",
                                   reviewed_dossier_digest=DOSSIER.digest(), now=t0 + 60)
        c2 = ProcessingContext(**{**{f: getattr(c, f) for f in c.__dataclass_fields__}, "approval_id": aid})
        with ThreadPoolExecutor(12) as ex:
            decisions = list(ex.map(lambda _: self.core.authorize(c2), range(12)))
        granted = [d for d in decisions if d.decision is Decision.ALLOW_WITH_CONTROLS]
        self.assertEqual(len(granted), 1)  # single use holds under contention
        losers = [d for d in decisions if d.decision is not Decision.ALLOW_WITH_CONTROLS]
        # Losers either see the approval already spent (REVIEW_REQUIRED) or lose the consume race (BLOCK, fail closed).
        self.assertTrue(all(d.decision in (Decision.REVIEW_REQUIRED, Decision.BLOCK) and not d.allowed for d in losers))


class PolicyFailClosed(GovernanceTestCase):
    def test_exploding_rule_blocks_instead_of_allowing(self):
        self.register()
        self.core.engine._rules = tuple(self.core.engine._rules) + (lambda ctx, env: (_ for _ in ()).throw(RuntimeError("boom")),)
        d = self.core.authorize(self.ctx())
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-ERR-99", {r.rule_id for r in d.blocking_rules})

    def test_garbage_context_values_do_not_allow(self):
        self.register()
        for op in ("rm -rf", None, 7):
            with self.assertRaises(Exception):
                self.core.authorize(self.ctx(operation=op))


if __name__ == "__main__":
    unittest.main()
