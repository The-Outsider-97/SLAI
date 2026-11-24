"""Section 10 / audit 4.4: runtime registration covers every reachable store, end to end (erasure + access + incidents)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.agents.governance.deletion_adapters import subject_tag
from src.agents.governance.registrations import register_runtime
from src.agents.governance.subject_rights import RightsError

from ._fixtures import GovernanceTestCase
from .test_deletion_adapters import _shim_and_import, ver

T, S = "t1", "s1"


class RegistrationTests(GovernanceTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.SharedMemory, cls.SecureMemory, cls.PrivacyMemory = _shim_and_import()
        except Exception as exc:
            raise unittest.SkipTest(f"runtime classes not importable here: {type(exc).__name__}: {exc}")

    def setUp(self):
        super().setUp()
        os.environ["SLAI_SHARED_MEMORY_KEY"] = "k" * 40
        self.addCleanup(os.environ.pop, "SLAI_SHARED_MEMORY_KEY", None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tag = subject_tag(T, S)
        self.shared = self.SharedMemory()
        self.secure = self.SecureMemory(namespace=f"reg-{id(self)}")
        self.files = Path(self.tmp.name) / "files"
        self.files.mkdir()
        self.pm1, self.pm2 = self.PrivacyMemory(), self.PrivacyMemory()  # two privacy components, two instances

    def populate(self):
        self.shared.put("doc:1", {"text": "shared"}, tags=[self.tag])
        self.secure.add({"v": "secure"}, tags=[self.tag], tenant_id=T)
        (self.files / f"{self.tag}.txt").write_text("file content", encoding="utf-8")
        self.pm1.register_consent_artifact(subject_id=S, purpose="p", status="granted", artifact_ref="a1")
        self.pm2.create_retention_obligation(record_id="rec-1", policy_id="pol", retention_days=30, subject_id=S)

    def register(self, **cfg):
        config = {"privacy_memory_tenant_id": T, "file_store_roots": [str(self.files)], **cfg}
        return register_runtime(self.core, shared_memory=self.shared, secure_memory=self.secure, config=config)

    def test_everything_reachable_is_registered_and_gaps_are_named(self):
        report = self.register()
        self.assertEqual(set(report["registered_now"]), {"shared_memory", "secure_memory", "privacy_memory", f"file_store:{self.files}"})
        self.assertTrue(set(report["registered_now"]) <= set(report["deletion_adapters"]))
        self.assertTrue(set(report["registered_now"]) <= set(report["rights_providers"]))
        self.assertIn("learning", " ".join(report["deletion_adapters"]))  # the learning pipeline adapter is registered by the core
        self.assertTrue(report["incident_bridge_attached"])
        self.assertTrue(report["known_uncovered_by_design"])

    def test_unconfigured_stores_are_reported_not_silently_skipped(self):
        report = register_runtime(self.core, shared_memory=None, secure_memory=self.secure, config={})
        stores = {s["store"]: s["reason"] for s in report["not_covered"]}
        self.assertIn("shared_memory", stores)
        self.assertIn("privacy_memory", stores)  # no tenant configured: PrivacyMemory is not tenant-aware, never guessed
        self.assertIn("incident_bridge", stores)

    def test_registration_is_idempotent(self):
        self.register()
        n_a, n_p = len(self.core.ledger.adapter_names()), len(self.core.rights.provider_names())
        self.register()
        self.assertEqual((n_a, n_p), (len(self.core.ledger.adapter_names()), len(self.core.rights.provider_names())))

    def test_erasure_completes_across_all_stores_including_every_privacy_memory_instance(self):
        self.populate()
        self.register()
        self.core.ledger.request_erasure("e1", T, S, verification=ver(T, S))
        out = self.core.ledger.execute_erasure("e1")
        self.assertEqual(out.status, "completed")
        self.assertEqual(self.shared.get_by_tag(self.tag), [])
        self.assertFalse(self.secure.tag_index.get(self.tag))
        self.assertEqual(list(self.files.iterdir()), [])
        self.assertFalse(self.pm1.contains_subject(S))
        self.assertFalse(self.pm2.contains_subject(S))  # the second component's instance was reached too

    def test_a_privacy_memory_created_after_registration_is_still_covered(self):
        self.register()
        late = self.PrivacyMemory()
        late.register_consent_artifact(subject_id=S, purpose="p", status="granted", artifact_ref="late")
        self.core.ledger.request_erasure("e2", T, S, verification=ver(T, S))
        self.assertEqual(self.core.ledger.execute_erasure("e2").status, "completed")
        self.assertFalse(late.contains_subject(S))

    def test_access_request_exports_from_every_registered_store_and_only_for_that_subject(self):
        self.populate()
        self.shared.put("doc:other", {"text": "someone else"}, tags=[subject_tag(T, "s2")])
        self.register()
        self.core.rights.submit("a1", T, S, "access", verification=ver(T, S))
        out = self.core.rights.fulfil("a1")["bundle"]["sections"]
        self.assertEqual([i["key"] for i in out["shared_memory"]["items"]], ["doc:1"])
        self.assertEqual(out["secure_memory"]["entries"][0]["data"]["v"], "secure")
        self.assertEqual(out[f"file_store:{self.files}"]["files"][0]["content_utf8"], "file content")
        self.assertEqual(len(out["privacy_memory"]["instances"]), len(self.PrivacyMemory.live_instances()))
        self.assertNotIn("someone else", str(out))

    def test_secure_memory_export_never_crosses_tenants(self):
        self.secure.add({"v": "other tenant"}, tags=[subject_tag("t2", S)], tenant_id="t2")
        self.register()
        self.core.rights.submit("a2", "t2", S, "access", verification=ver("t2", S))
        out = self.core.rights.fulfil("a2")["bundle"]["sections"]["secure_memory"]["entries"]
        self.assertEqual([e["data"]["v"] for e in out], ["other tenant"])
        self.assertNotIn("other tenant", str(self.core.rights.provider_names()))
        self.core.rights.submit("a3", T, S, "access", verification=ver(T, S))
        self.assertEqual(self.core.rights.fulfil("a3")["bundle"]["sections"]["secure_memory"]["entries"], [])

    def test_portability_exports_only_stores_explicitly_declared_portable(self):
        self.populate()
        self.register(portable_stores=["shared_memory"])
        self.core.rights.submit("p1", T, S, "portability", verification=ver(T, S))
        sections = self.core.rights.fulfil("p1")["bundle"]["sections"]
        self.assertEqual(set(sections), {"shared_memory"})  # Art. 20 eligibility is never inferred

    def test_large_files_are_listed_but_not_inlined(self):
        (self.files / f"{self.tag}.big").write_bytes(b"x" * 50)
        self.register()
        from src.agents.governance.rights_providers import FileStoreRightsProvider

        row = FileStoreRightsProvider(str(self.files), max_inline_bytes=10).export(T, S)["files"][0]
        self.assertIn("omitted", row["content"])
        self.assertEqual(row["bytes"], 50)

    def _wait_for(self, n, timeout=3.0):
        import time

        end = time.time() + timeout
        while time.time() < end and len(self.core.incidents.breach_register()) < n:
            time.sleep(0.02)
        return len(self.core.incidents.breach_register())

    def test_incident_published_to_shared_memory_reaches_the_registry(self):
        self.register()
        self.shared.set("observability_agent:latest_incident", {"incident_id": "obs-1", "category": "data_breach"})
        self.assertEqual(self._wait_for(1), 1)  # real SharedMemory (asynchronous callbacks) delivered it to the intake
        self.shared.set("observability_agent:latest_incident", {"incident_id": "obs-2", "category": "latency"})  # not a governance incident
        self.assertEqual(self._wait_for(2, timeout=0.5), 1)


if __name__ == "__main__":
    unittest.main()
