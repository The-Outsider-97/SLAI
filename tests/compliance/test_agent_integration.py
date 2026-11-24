"""Section 9: every factory agent is covered by governed capabilities; file stores can be erased."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agents.governance.agent_manifest import build_manifest, factory_specs, integration_report, register_all
from src.agents.governance.deletion_adapters import FileStoreDeletionAdapter, subject_tag
from src.agents.governance.delegation_guard import DelegationDenied, check_delegable, install_authorizer
from src.agents.governance.policy_registry import CapabilityRegistry
from src.agents.governance.privacy_ledger import PrivacyLedger
from src.agents.governance.processing_context import DataClass

AGENTS = Path(__file__).resolve().parents[2] / "src" / "agents"


class CentralGateCoverageTests(unittest.TestCase):
    def test_every_factory_agent_inherits_base_agent_and_none_replaces_execute(self):
        report = integration_report(AGENTS)
        self.assertEqual(report["summary"]["not_covered_by_central_gate"], [])  # a replaced execute() would bypass the gate
        self.assertEqual(report["summary"]["covered_by_central_gate"], report["summary"]["agents"])
        self.assertIn("static analysis", report["basis"])

    def test_inherited_execute_is_a_registered_capability_for_every_agent(self):
        reg = CapabilityRegistry()
        counts = register_all(reg, AGENTS)
        self.assertEqual([a for a in counts if reg.resolve(a, "execute") is None], [])


class ManifestTests(unittest.TestCase):
    def test_every_factory_agent_is_discovered_and_resolved(self):
        specs = factory_specs(AGENTS)
        manifest = build_manifest(AGENTS)
        self.assertGreaterEqual(len(specs), 30)
        self.assertEqual(set(specs), set(manifest))
        self.assertEqual([a for a, m in manifest.items() if not m["resolved"]], [])
        self.assertIn("governance", manifest)

    def test_registration_is_conservative_by_default(self):
        reg = CapabilityRegistry()
        counts = register_all(reg, AGENTS)
        self.assertTrue(all(n > 0 for n in counts.values()))
        cap = next(c for c in reg.list() if c.owner_agent == "reader")
        self.assertEqual(cap.allowed_data_classes, frozenset({DataClass.NON_PERSONAL, DataClass.ANONYMISED}))
        self.assertFalse(cap.tenant_bound)

    def test_personal_data_agents_are_an_explicit_opt_in_and_tenant_bound(self):
        reg = CapabilityRegistry()
        register_all(reg, AGENTS, personal_data_agents=["privacy"], human_approval_actions=["privacy.evaluate_privacy"])
        cap = reg.resolve("privacy", "evaluate_privacy")
        self.assertIn(DataClass.PERSONAL, cap.allowed_data_classes)
        self.assertTrue(cap.tenant_bound)
        self.assertTrue(cap.requires_human_approval)
        self.assertNotIn(DataClass.PERSONAL, reg.resolve("reader", "perform_task").allowed_data_classes)

    def test_private_and_unknown_actions_are_not_registered(self):
        reg = CapabilityRegistry()
        register_all(reg, AGENTS)
        self.assertIsNone(reg.resolve("privacy", "_disabled_privacy_decision"))
        self.assertIsNone(reg.resolve("privacy", "no_such_action"))
        self.assertIsNone(reg.resolve("no_such_agent", "perform_task"))

    def test_registry_enforcement_through_the_delegation_guard(self):
        reg = CapabilityRegistry()
        register_all(reg, AGENTS)

        def authorizer(caller, target, action):
            if reg.resolve(target, action) is None:
                raise DelegationDenied("not a registered capability")

        class Reader:
            def perform_task(self):
                return "ok"

            def other(self):
                return "no"

        install_authorizer(authorizer)
        self.addCleanup(install_authorizer, None)
        self.assertEqual(check_delegable(Reader(), target_name="reader", action="perform_task")(), "ok")
        with self.assertRaises(DelegationDenied):
            check_delegable(Reader(), target_name="reader", action="other")
        with self.assertRaises(DelegationDenied):
            check_delegable(Reader(), target_name="unknown_agent", action="perform_task")

    def test_integration_report_is_honest_about_what_it_is(self):
        reg = CapabilityRegistry()
        register_all(reg, AGENTS)
        rep = integration_report(AGENTS, reg)
        self.assertIn("not proof of runtime behaviour", rep["basis"])
        self.assertEqual(rep["summary"]["capabilities_registered_for"], rep["summary"]["agents"])
        self.assertTrue(all(r["delegation_guarded"] for r in rep["agents"].values()))


class FileStoreAdapterTests(unittest.TestCase):
    def test_file_store_erasure_through_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t1s1, t1s2, t2s1 = subject_tag("t1", "s1"), subject_tag("t1", "s2"), subject_tag("t2", "s1")
            (root / f"{t1s1}.json").write_text("{}")
            (root / t1s1 / "nested").mkdir(parents=True)
            (root / t1s1 / "nested" / "a.txt").write_text("x")
            (root / f"{t1s2}.json").write_text("{}")
            (root / f"{t2s1}.json").write_text("{}")
            ledger = PrivacyLedger(":memory:", tombstone_salt=b"s" * 32)
            ledger.register_adapter(FileStoreDeletionAdapter(tmp))
            ledger.request_erasure("r1", "t1", "s1", verification={"verified": True, "method": "m", "verified_by": "idp", "tenant_id": "t1", "subject_id": "s1"})
            self.assertEqual(ledger.execute_erasure("r1").status, "completed")
            self.assertFalse((root / f"{t1s1}.json").exists())
            self.assertFalse((root / t1s1).exists())
            self.assertTrue((root / f"{t1s2}.json").exists())
            self.assertTrue((root / f"{t2s1}.json").exists())


if __name__ == "__main__":
    unittest.main()
