"""Audit finding 4.4: real deletion adapters, end to end through the PrivacyLedger.

The real SharedMemory, SecureMemory and PrivacyMemory classes are used. Only environment gaps of the
test machine (missing torch, old colorama/jsonschema) are shimmed, and only when absent.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.governance.deletion_adapters import (
    PrivacyMemoryDeletionAdapter, SecureMemoryDeletionAdapter, SharedMemoryDeletionAdapter, subject_tag,
)
from src.agents.governance.privacy_ledger import PrivacyLedger

_STUBS = ("torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils", "torch.utils.data")
KEY = "k" * 40


def _shim_and_import():
    try:
        import colorama

        if not hasattr(colorama, "just_fix_windows_console"):
            colorama.just_fix_windows_console = lambda: None
    except Exception:
        pass
    try:
        import jsonschema

        if not hasattr(jsonschema, "Draft202012Validator"):
            jsonschema.Draft202012Validator = mock.MagicMock()
    except Exception:
        pass
    try:
        import torch  # noqa: F401
    except Exception:
        for n in _STUBS:
            sys.modules.setdefault(n, mock.MagicMock())
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.privacy.privacy_memory import PrivacyMemory
    from src.agents.safety.secure_memory import SecureMemory

    return SharedMemory, SecureMemory, PrivacyMemory


def ver(tenant, subject):
    return {"verified": True, "method": "session", "verified_by": "idp", "tenant_id": tenant, "subject_id": subject}


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.SharedMemory, cls.SecureMemory, cls.PrivacyMemory = _shim_and_import()
        except Exception as exc:
            raise unittest.SkipTest(f"runtime classes not importable here: {type(exc).__name__}: {exc}")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["SLAI_SHARED_MEMORY_KEY"] = KEY
        self.addCleanup(os.environ.pop, "SLAI_SHARED_MEMORY_KEY", None)
        self.ledger = PrivacyLedger(":memory:", tombstone_salt=b"s" * 32)
        self.tag_a1 = subject_tag("t1", "s1")
        self.tag_a2 = subject_tag("t1", "s2")
        self.tag_b1 = subject_tag("t2", "s1")

    def run_erasure(self, tenant="t1", subject="s1"):
        self.ledger.request_erasure("r1", tenant, subject, verification=ver(tenant, subject))
        return self.ledger.execute_erasure("r1")

    # ---------------------------------------------------------------- SharedMemory
    def test_shared_memory_erasure_all_versions_and_persisted_file(self):
        mem = self.SharedMemory()
        mem.put("doc:1", {"text": "v1"}, tags=[self.tag_a1])
        mem.put("doc:1", {"text": "v2"}, tags=[self.tag_a1])
        mem.put("doc:2", {"text": "other subject"}, tags=[self.tag_a2])
        mem.put("doc:3", {"text": "other tenant"}, tags=[self.tag_b1])
        path = Path(self.tmp.name) / "state.slaimem"
        mem.save_to_file(str(path))
        self.ledger.register_adapter(SharedMemoryDeletionAdapter(mem, persist_path=str(path)))
        out = self.run_erasure()
        self.assertEqual(out.status, "completed")
        self.assertIsNone(mem.get("doc:1"))
        self.assertIsNotNone(mem.get("doc:2"))
        self.assertIsNotNone(mem.get("doc:3"))
        reloaded = self.SharedMemory()
        reloaded.load_from_file(str(path))
        self.assertIsNone(reloaded.get("doc:1"))
        self.assertIsNotNone(reloaded.get("doc:2"))

    def test_shared_memory_stale_persisted_file_fails_verification(self):
        mem = self.SharedMemory()
        mem.put("doc:1", {"text": "x"}, tags=[self.tag_a1])
        path = Path(self.tmp.name) / "state.slaimem"
        mem.save_to_file(str(path))
        adapter = SharedMemoryDeletionAdapter(mem, persist_path=str(path))
        mem.delete("doc:1")  # in-memory copy removed but the file still holds it
        self.assertFalse(adapter.verify_absent("t1", "s1"))

    def test_shared_memory_without_key_cannot_complete(self):
        mem = self.SharedMemory()
        mem.put("doc:1", {"text": "x"}, tags=[self.tag_a1])
        os.environ.pop("SLAI_SHARED_MEMORY_KEY")
        self.ledger.register_adapter(SharedMemoryDeletionAdapter(mem, persist_path=str(Path(self.tmp.name) / "s.slaimem")))
        self.assertEqual(self.run_erasure().status, "incomplete")  # fail closed, never "completed"

    # ---------------------------------------------------------------- SecureMemory
    def test_secure_memory_tenant_scoped_erasure_and_checkpoint_purge(self):
        mem = self.SecureMemory(namespace=f"adapt-{id(self)}")
        a1 = mem.add({"v": "a1"}, tags=[self.tag_a1], tenant_id="t1")
        a2 = mem.add({"v": "a2"}, tags=[self.tag_a2], tenant_id="t1")
        b1 = mem.add({"v": "b1"}, tags=[self.tag_b1], tenant_id="t2")
        with mock.patch.object(type(mem), "_checkpoint_dir", return_value=Path(self.tmp.name) / "ck"):
            (Path(self.tmp.name) / "ck").mkdir()
            self.assertTrue(mem.create_checkpoint("before"))
            self.assertTrue(any(self.tag_a1 in p.read_text() for p in mem.checkpoint_files()))
            self.ledger.register_adapter(SecureMemoryDeletionAdapter(mem))
            out = self.run_erasure()
            self.assertEqual(out.status, "completed")
            self.assertNotIn(a1, mem.store)
            self.assertIn(a2, mem.store)
            self.assertIn(b1, mem.store)
            self.assertFalse(any(self.tag_a1 in p.read_text() for p in mem.checkpoint_files()))
            self.assertTrue(any(self.tag_a2 in p.read_text() for p in mem.checkpoint_files()))  # fresh checkpoint keeps others

    def test_secure_memory_entry_level_legal_hold_keeps_request_incomplete(self):
        mem = self.SecureMemory(namespace=f"adapt-h-{id(self)}")
        eid = mem.add({"v": "x"}, tags=[self.tag_a1], tenant_id="t1")
        mem.store[eid]["meta"]["legal_hold"] = True
        with mock.patch.object(type(mem), "_checkpoint_dir", return_value=Path(self.tmp.name) / "ck"):
            (Path(self.tmp.name) / "ck").mkdir(exist_ok=True)
            self.ledger.register_adapter(SecureMemoryDeletionAdapter(mem))
            self.assertEqual(self.run_erasure().status, "incomplete")
            self.assertIn(eid, mem.store)

    # ---------------------------------------------------------------- PrivacyMemory (retention store)
    def test_privacy_memory_erasure(self):
        pm = self.PrivacyMemory()
        pm.register_consent_artifact(subject_id="s1", purpose="p", status="granted", artifact_ref="a1")
        pm.register_consent_artifact(subject_id="s2", purpose="p", status="granted", artifact_ref="a2")
        pm.create_retention_obligation(record_id="rec-1", policy_id="pol", retention_days=30, subject_id="s1")
        pm.create_retention_obligation(record_id="rec-2", policy_id="pol", retention_days=30, subject_id="s2")
        self.assertTrue(pm.contains_subject("s1"))
        self.ledger.register_adapter(PrivacyMemoryDeletionAdapter(pm, tenant_id="t1"))
        self.assertEqual(self.run_erasure().status, "completed")
        self.assertFalse(pm.contains_subject("s1"))
        self.assertTrue(pm.contains_subject("s2"))
        state = pm.export_state()
        self.assertNotIn("rec-1", state["retention_index"])
        self.assertIn("rec-2", state["retention_index"])
        self.assertFalse(any(e.get("subject_id") == "s1" for e in state["event_log"]))

    def test_privacy_memory_other_tenant_is_noop(self):
        pm = self.PrivacyMemory()
        pm.register_consent_artifact(subject_id="s1", purpose="p", status="granted", artifact_ref="a1")
        adapter = PrivacyMemoryDeletionAdapter(pm, tenant_id="t1")
        adapter.delete("t2", "s1")
        self.assertTrue(pm.contains_subject("s1"))

    # ---------------------------------------------------------------- all stores together
    def test_all_adapters_together_and_one_failure_blocks_completion(self):
        shared = self.SharedMemory()
        shared.put("k", {"v": 1}, tags=[self.tag_a1])
        secure = self.SecureMemory(namespace=f"adapt-all-{id(self)}")
        secure.add({"v": 1}, tags=[self.tag_a1], tenant_id="t1")
        pm = self.PrivacyMemory()
        pm.register_consent_artifact(subject_id="s1", purpose="p", status="granted", artifact_ref="a")

        class Broken:
            name = "external_provider"

            def delete(self, tenant_id, subject_id):
                raise RuntimeError("provider API down")

            def verify_absent(self, tenant_id, subject_id):
                return False

        with mock.patch.object(type(secure), "_checkpoint_dir", return_value=Path(self.tmp.name) / "ck"):
            (Path(self.tmp.name) / "ck").mkdir(exist_ok=True)
            for a in (SharedMemoryDeletionAdapter(shared), SecureMemoryDeletionAdapter(secure),
                      PrivacyMemoryDeletionAdapter(pm, tenant_id="t1"), Broken()):
                self.ledger.register_adapter(a)
            out = self.run_erasure()
            self.assertEqual(out.status, "incomplete")
            self.assertEqual(out.targets["external_provider"], "failed")
            self.assertEqual({k: v for k, v in out.targets.items() if k != "external_provider"},
                             {"shared_memory": "verified", "secure_memory": "verified", "privacy_memory": "verified"})
            self.assertFalse(self.ledger.must_reerase_after_restore("t1", "s1"))

    def test_subject_tag_is_pii_free_and_stable(self):
        self.assertNotIn("s1", self.tag_a1)
        self.assertEqual(self.tag_a1, subject_tag("t1", "s1"))
        self.assertNotEqual(self.tag_a1, self.tag_b1)


if __name__ == "__main__":
    unittest.main()
