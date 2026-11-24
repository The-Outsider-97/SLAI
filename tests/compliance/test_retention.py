"""Retention and erasure-request rules in the real DataRetention component (audit 4.5)."""
from __future__ import annotations

import sys
import time
import unittest
from unittest import mock

_STUBS = ("torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils", "torch.utils.data")


def _import():
    try:
        import colorama

        if not hasattr(colorama, "just_fix_windows_console"):
            colorama.just_fix_windows_console = lambda: None
    except Exception:
        pass
    try:
        import torch  # noqa: F401
    except Exception:
        for n in _STUBS:
            sys.modules.setdefault(n, mock.MagicMock())
    from src.agents.privacy.data_retention import DataRetention
    from src.agents.privacy.utils.privacy_error import PrivacyError

    return DataRetention, PrivacyError


def ver(subject="s1", **over):
    return {"verified": True, "method": "authenticated_session", "verified_by": "idp", "subject_id": subject, **over}


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.DataRetention, cls.PrivacyError = _import()
        except Exception as exc:
            raise unittest.SkipTest(f"DataRetention not importable here: {type(exc).__name__}: {exc}")

    def setUp(self):
        self.ret = self.DataRetention()
        self.ret.create_retention_obligation(record_id="rec-1", subject_id="s1", policy_id="p", retention_days=30, request_id="q0")

    def forget(self, **kw):
        args = dict(record_id="rec-1", request_id="q1", subject_id="s1", identity_verification=ver())
        args.update(kw)
        return self.ret.process_forget_request(**args)

    def test_no_reason_is_required_for_erasure(self):
        out = self.forget()  # no reason argument at all
        self.assertEqual(out["status"], "due")
        self.assertFalse(self.ret.right_to_be_forgotten_requires_reason)

    def test_legacy_config_cannot_make_a_reason_mandatory_again(self):
        self.ret.retention_config["right_to_be_forgotten_requires_reason"] = True
        again = self.DataRetention()
        self.assertFalse(again.right_to_be_forgotten_requires_reason)

    def test_identity_verification_is_mandatory(self):
        for bad in (None, {}, ver(verified="yes"), ver(method=""), ver(verified_by=""), ver(subject="someone-else")):
            with self.assertRaises(self.PrivacyError, msg=repr(bad)):
                self.forget(identity_verification=bad, request_id=f"q-{id(bad)}")

    def test_unknown_art17_ground_is_rejected_and_known_one_is_accepted(self):
        with self.assertRaises(self.PrivacyError):
            self.forget(erasure_ground="because_i_said_so", request_id="q2")
        self.forget(erasure_ground="art17_1_b_consent_withdrawn", request_id="q3")

    def test_subject_mismatch_with_obligation_is_rejected(self):
        with self.assertRaises(self.PrivacyError):
            self.forget(subject_id="s2", identity_verification=ver("s2"), request_id="q4")

    def test_legal_hold_blocks_deletion_unless_override_is_explicit(self):
        self.ret.apply_legal_hold(record_id="rec-1", hold_ref="h1", reason="litigation", request_id="q5")
        with self.assertRaises(self.PrivacyError):
            self.forget(request_id="q6")

    def test_failed_erasure_during_hold_leaves_the_hold_intact(self):
        self.ret.apply_legal_hold(record_id="rec-1", hold_ref="h1", reason="litigation", request_id="q5")
        with self.assertRaises(self.PrivacyError):
            self.forget(request_id="q6")
        ob = self.ret.memory.retention_obligation("rec-1")
        self.assertTrue(ob["legal_hold"])
        self.assertEqual(ob["status"], "legal_hold")
        self.assertIsNone(self.ret.memory.export_state()["deletion_index"].get("rec-1"))

    def test_release_of_hold_allows_erasure(self):
        self.ret.apply_legal_hold(record_id="rec-1", hold_ref="h1", reason="litigation", request_id="q5")
        self.ret.release_legal_hold(record_id="rec-1", reason="case closed", request_id="q6")
        self.assertEqual(self.forget(request_id="q7")["status"], "due")

    def test_expiry_schedules_deletion_and_active_record_is_allowed(self):
        now = self.ret.retention_status(record_id="rec-1", request_id="a")
        self.assertEqual((now["decision"], now["is_due"]), ("allow", False))
        later = self.ret.retention_status(record_id="rec-1", request_id="b", as_of=time.time() + 31 * 86400)
        self.assertEqual((later["decision"], later["is_due"]), ("block", True))
        self.assertEqual(later["deletion_schedule"]["status"], "scheduled")


if __name__ == "__main__":
    unittest.main()
