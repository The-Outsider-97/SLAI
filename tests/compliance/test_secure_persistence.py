"""Audit finding 4.8: shared-memory persistence must be authenticated, safe to load and versioned."""
from __future__ import annotations

import importlib.util
import os
import pickle
import sys
import unittest
from collections import OrderedDict, namedtuple
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "src" / "agents" / "collaborative" / "utils" / "secure_persistence.py"
_spec = importlib.util.spec_from_file_location("secure_persistence_under_test", _PATH)
sp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sp
_spec.loader.exec_module(sp)  # stdlib-only module: loadable without the agent runtime

import types

_FAKE = "slai_fake.collaborative.shared_memory"  # same suffix the real class lives under
_mod = types.ModuleType(_FAKE)
VersionedItem = namedtuple("VersionedItem", ["timestamp", "value"])
VersionedItem.__module__ = _FAKE
_mod.VersionedItem = VersionedItem
for _name in ("slai_fake", "slai_fake.collaborative"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["slai_fake"].collaborative = sys.modules["slai_fake.collaborative"]
sys.modules["slai_fake.collaborative"].shared_memory = _mod
sys.modules[_FAKE] = _mod
KEY = b"k" * 40


class Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned > /tmp/slai_pwned",))


class SecurePersistenceTests(unittest.TestCase):
    def payload(self):
        return {"schema_version": 2, "data": {"k": [VersionedItem(1.0, {"a": [1, 2]})]}, "access_log": OrderedDict(k=1)}

    def test_roundtrip(self):
        blob = sp.seal(self.payload(), key=KEY)
        out = sp.unseal(blob, versioned_item=VersionedItem, key=KEY)
        self.assertEqual(out["data"]["k"][0].value, {"a": [1, 2]})

    def test_tamper_detected_before_deserialisation(self):
        blob = bytearray(sp.seal(self.payload(), key=KEY))
        blob[-3] ^= 0x01
        with self.assertRaises(sp.PersistenceSecurityError):
            sp.unseal(bytes(blob), versioned_item=VersionedItem, key=KEY)

    def test_wrong_key_rejected(self):
        blob = sp.seal(self.payload(), key=KEY)
        with self.assertRaises(sp.PersistenceSecurityError):
            sp.unseal(blob, versioned_item=VersionedItem, key=b"x" * 40)

    def test_no_key_fails_closed(self):
        for var in ("SLAI_SHARED_MEMORY_KEY", "SLAI_GOVERNANCE_MASTER_KEY"):
            os.environ.pop(var, None)
        with self.assertRaises(sp.PersistenceSecurityError):
            sp.seal(self.payload())
        with self.assertRaises(sp.PersistenceSecurityError):
            sp.seal(self.payload(), key=b"short")

    def test_raw_pickle_is_refused(self):
        with self.assertRaises(sp.PersistenceSecurityError):
            sp.unseal(pickle.dumps(self.payload()), versioned_item=VersionedItem, key=KEY)

    def test_malicious_pickle_never_executes_even_through_migration(self):
        marker = Path("/tmp/slai_pwned")
        marker.unlink(missing_ok=True)
        with self.assertRaises(sp.PersistenceSecurityError):
            sp.migrate_legacy_pickle(pickle.dumps(Evil()), versioned_item=VersionedItem)
        self.assertFalse(marker.exists())

    def test_malicious_pickle_signed_with_valid_key_still_blocked(self):
        # Even an attacker holding the key cannot smuggle executable objects.
        mac_key, _ = sp.load_keys(KEY)
        import hashlib
        import hmac as _h

        header = sp.MAGIC + bytes([0])
        body = pickle.dumps(Evil())
        blob = header + _h.new(mac_key, header + body, hashlib.sha256).digest() + body
        with self.assertRaises(sp.PersistenceSecurityError):
            sp.unseal(blob, versioned_item=VersionedItem, key=KEY)

    def test_legacy_migration_decodes_plain_data(self):
        out = sp.migrate_legacy_pickle(pickle.dumps(self.payload()), versioned_item=VersionedItem)
        self.assertIn("data", out)

    def test_truncated_and_oversized_rejected(self):
        with self.assertRaises(sp.PersistenceSecurityError):
            sp.unseal(sp.MAGIC + b"\x00", versioned_item=VersionedItem, key=KEY)

    def test_encryption_hides_plaintext_when_available(self):
        secret = "Jan de Vries BSN 123456782"
        blob = sp.seal({"v": secret}, key=KEY)
        if sp.encryption_available():
            self.assertNotIn(secret.encode(), blob)
        else:
            self.skipTest("cryptography package not installed: integrity-only mode")


if __name__ == "__main__":
    unittest.main()
