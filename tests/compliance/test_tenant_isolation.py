"""Audit finding 4.7: SecureMemory must isolate tenants and fail closed."""
from __future__ import annotations

import sys
import unittest
from unittest import mock

_STUBS = ("torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils", "torch.utils.data")


def _import_secure_memory():
    """Import SecureMemory; heavy ML packages are stubbed ONLY if absent (the module under test does not use them)."""
    saved = {}
    try:
        import colorama

        if not hasattr(colorama, "just_fix_windows_console"):
            colorama.just_fix_windows_console = lambda: None
    except Exception:
        pass
    try:
        import torch  # noqa: F401
    except Exception:
        for name in _STUBS:
            saved[name] = sys.modules.get(name)
            sys.modules[name] = mock.MagicMock()
    try:
        from src.agents.safety.secure_memory import SecureMemory

        return SecureMemory
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class TenantIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.SecureMemory = _import_secure_memory()
        except Exception as exc:  # environment cannot import the safety package
            raise unittest.SkipTest(f"SecureMemory not importable here: {type(exc).__name__}: {exc}")

    def setUp(self):
        self.mem = self.SecureMemory(namespace=f"tenant-test-{id(self)}")
        self.a = self.mem.add({"secret": "tenant A data"}, tags=["t"], tenant_id="tenant-a")
        self.shared = self.mem.add({"public": True}, tags=["t"])

    def ctx(self, tenant=None):
        return self.mem.internal_context("read", tenant_id=tenant)

    def test_owner_can_read(self):
        self.assertIsNotNone(self.mem.get(self.a, self.ctx("tenant-a")))

    def test_other_tenant_denied(self):
        self.assertIsNone(self.mem.get(self.a, self.ctx("tenant-b")))

    def test_missing_caller_tenant_is_not_a_wildcard(self):
        self.assertIsNone(self.mem.get(self.a, self.ctx(None)))

    def test_tenant_caller_cannot_read_untenanted_entries_by_default(self):
        self.assertIsNone(self.mem.get(self.shared, self.ctx("tenant-a")))

    def test_recall_and_search_do_not_leak_across_tenants(self):
        ids = {r["meta"]["entry_id"] for r in self.mem.recall("t", access_context=self.ctx("tenant-b"))}
        self.assertNotIn(self.a, ids)
        found = self.mem.search_secure("tenant a data", access_context=self.ctx("tenant-b"))
        self.assertEqual(found, [])

    def test_cross_tenant_update_and_delete_denied(self):
        self.assertFalse(self.mem.update(self.a, {"x": 1}, self.ctx("tenant-b")))
        self.assertFalse(self.mem.delete(self.a, self.ctx("tenant-b")))
        self.assertIsNotNone(self.mem.get(self.a, self.ctx("tenant-a")))

    def test_tenant_cannot_be_forged_via_metadata(self):
        forged = self.mem.add({"v": 1}, tags=["t"], metadata={"tenant_id": "tenant-a"}, tenant_id="tenant-b")
        self.assertIsNone(self.mem.get(forged, self.ctx("tenant-a")))
        self.assertIsNotNone(self.mem.get(forged, self.ctx("tenant-b")))

    def test_update_cannot_change_or_drop_tenant(self):
        self.assertTrue(self.mem.update(self.a, {"v": 2}, self.ctx("tenant-a"), metadata={"tenant_id": "tenant-b"}))
        self.assertIsNone(self.mem.get(self.a, self.ctx("tenant-b")))
        self.assertTrue(self.mem.update(self.a, {"v": 3}, self.ctx("tenant-a"), metadata={}))
        self.assertIsNone(self.mem.get(self.a, self.ctx(None)))


if __name__ == "__main__":
    unittest.main()
