"""The real GovernanceAgent and AgentFactory, started end to end (only missing third-party libraries are shimmed)."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

from src.agents.governance.core import GovernanceCore, load_master_key
from src.agents.governance.delegation_guard import check_delegable, install_authorizer
from src.agents.governance.agent_gate import install_gate, installed_gate
from src.agents.governance.processing_context import DataClass, GovernanceDenied, Operation, ProcessingContext

from .test_deletion_adapters import _shim_and_import


class MasterKeyTests(unittest.TestCase):
    def test_base64_key_of_32_bytes(self):
        import base64

        self.assertEqual(load_master_key({"SLAI_GOVERNANCE_MASTER_KEY": base64.b64encode(b"x" * 32).decode()}), b"x" * 32)

    def test_alphanumeric_passphrase_that_is_valid_base64_but_decodes_short_is_still_a_key(self):
        raw = "m" * 40  # valid base64 charset, decodes to 30 bytes
        self.assertEqual(load_master_key({"SLAI_GOVERNANCE_MASTER_KEY": raw}), raw.encode())

    def test_too_short_or_missing_is_no_key(self):
        self.assertIsNone(load_master_key({"SLAI_GOVERNANCE_MASTER_KEY": "short"}))
        self.assertIsNone(load_master_key({}))

    def test_key_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as fh:
            fh.write("p" * 33 + "!\n")
        self.addCleanup(os.unlink, fh.name)
        self.assertEqual(load_master_key({"SLAI_GOVERNANCE_MASTER_KEY_FILE": fh.name}), ("p" * 33 + "!").encode())


class DatabaseLifecycleTests(unittest.TestCase):
    @staticmethod
    def connections(core):
        return [core.activities._db, core.oversight._db, core.grants._db,
                core.evidence._db, core.incidents._db, core.ledger._db,
                core.learning._db, core.rights._db, core._decisions,
                core.classification._db, core.reassessment._db, core.conditions._db, core.profiles._db, core.adaptive._db]

    @staticmethod
    def discovered_connections(core):
        """Every sqlite3 connection reachable from the core's components (so a new store cannot be forgotten by the list above)."""
        found, seen = [], set()
        for owner in [core, *vars(core).values()]:
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            try:
                attributes = vars(owner).values()
            except TypeError:
                continue
            found.extend(v for v in attributes if isinstance(v, sqlite3.Connection))
        return list({id(c): c for c in found}.values())

    def test_every_reachable_connection_is_listed_closed_and_the_state_directory_is_deletable(self):
        """Windows cannot delete a SQLite file that still has an open handle: removing the state directory after close() is the
        portable proof that no store leaked a connection (this is the check the Windows CI job exists for)."""
        import shutil
        tmp = tempfile.mkdtemp()
        try:
            core = GovernanceCore(tmp, master_key=b"x" * 32)
            core.authorize(ProcessingContext(Operation.PROCESS, "document_editing", data_class=DataClass.PERSONAL))
            listed = self.connections(core)
            discovered = self.discovered_connections(core)
            self.assertEqual({id(c) for c in discovered} - {id(c) for c in listed}, set(), "a SQLite connection is not covered by the lifecycle test")
            self.assertGreaterEqual(len(discovered), 13)
            core.close()
            self.assert_closed(discovered)
            if sys.platform.startswith("linux"):
                held = [os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd") if os.path.exists(f"/proc/self/fd/{fd}")]
                self.assertEqual([h for h in held if h.startswith(tmp)], [])
            shutil.rmtree(tmp)  # raises PermissionError on Windows if any handle is still open
            self.assertFalse(os.path.exists(tmp))
        finally:
            GovernanceCore.reset_shared()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_state_survives_a_restart_and_reopen_does_not_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = GovernanceCore(tmp, master_key=b"x" * 32)
            first.ledger.record_consent("tenant-a", "subj-1", "document_editing", explicit=False, notice_version="v1", evidence_ref="e1")
            first.close()
            second = GovernanceCore(tmp, master_key=b"x" * 32)
            try:
                self.assertTrue(second.ledger.consent_snapshot("tenant-a", "subj-1", "document_editing")["active"])
                self.assertTrue(second.ledger.verify_chain())
                self.assertTrue(second.health()["ready"])
            finally:
                second.close()

    def assert_closed(self, connections):
        for connection in connections:
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_close_releases_every_database_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            core = GovernanceCore(tmp, master_key=b"x" * 32)
            try:
                connections = self.connections(core)
            finally:
                core.close()
            core.close()
            self.assert_closed(connections)

    def test_reset_shared_closes_connections_and_creates_fresh_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                core = GovernanceCore.shared(tmp, master_key=b"x" * 32)
                self.assertIs(core, GovernanceCore.shared(tmp))
                connections = self.connections(core)
                GovernanceCore.reset_shared()
                self.assert_closed(connections)
                fresh = GovernanceCore.shared(tmp, master_key=b"x" * 32)
                self.assertIsNot(core, fresh)
                self.assertTrue(fresh.health()["ready"])
                fresh.close()
                self.assert_closed(self.connections(fresh))
                self.assertIsNot(fresh, GovernanceCore.shared(tmp, master_key=b"x" * 32))
            finally:
                GovernanceCore.reset_shared()


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.SharedMemory, _, _ = _shim_and_import()
            from src.agents.base_agent import BaseAgent
            from src.agents.governance_agent import GovernanceAgent
        except Exception as exc:
            raise unittest.SkipTest(f"runtime not importable here: {type(exc).__name__}: {exc}")
        cls.BaseAgent, cls.GovernanceAgent = BaseAgent, GovernanceAgent

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)  # state_dir 'data/governance' is relative
        for k, v in {"SLAI_GOVERNANCE_MASTER_KEY": "m" * 40, "SLAI_SHARED_MEMORY_KEY": "k" * 40}.items():
            self.addCleanup(os.environ.pop, k, None)
            os.environ[k] = v
        GovernanceCore.reset_shared()
        self.addCleanup(self._cleanup)
        self.agent = self.GovernanceAgent(shared_memory=self.SharedMemory(), agent_factory=None)

    def _cleanup(self):
        install_gate(None)
        install_authorizer(None)
        GovernanceCore.reset_shared()
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_starting_the_agent_installs_the_gate_and_is_honest_about_its_mode(self):
        h = self.agent.health()
        self.assertIs(installed_gate(), self.agent.agent_gate)
        self.assertTrue(h["governance"]["ready"], h["governance"]["problems"])
        self.assertEqual(h["gate"]["mode"], "observe")
        self.assertFalse(h["gate"]["enforcing"])  # observe is reported as not enforcing
        self.assertEqual(h["delegation_audit"]["mode"], "audit")

    def test_registration_report_names_what_is_and_is_not_covered(self):
        reg = self.agent.health()["registration"]
        self.assertEqual({"shared_memory", "secure_memory"}, set(reg["registered_now"]))
        self.assertIn("privacy_memory", {s["store"] for s in reg["not_covered"]})  # tenant not configured
        self.assertTrue(reg["incident_bridge_attached"])

    def test_agents_started_afterwards_are_evaluated_by_the_gate(self):
        class ReaderAgent(self.BaseAgent):
            def perform_task(self, task):
                return {"status": "success"}

        agent = ReaderAgent(shared_memory=self.SharedMemory(), agent_factory=None, config={})
        self.assertEqual(agent.execute({"x": 1}), {"status": "success"})
        self.assertEqual(self.agent.agent_gate.status()["stats"]["evaluated"], 1)

    def test_class_names_resolve_to_factory_keys(self):
        self.assertEqual(self.agent.agent_gate.aliases.get("LanguageAgent"), "language")
        self.assertEqual(self.agent.agent_gate.aliases.get("LearningAgent"), "learning")

    def test_inherited_actions_are_registered_for_every_agent(self):
        caps = self.agent.core.capabilities
        for agent in ("reader", "planning", "knowledge", "evaluation", "language"):
            self.assertIsNotNone(caps.resolve(agent, "execute"), agent)
            self.assertIsNotNone(caps.resolve(agent, "perform_task"), agent)

    def test_delegation_audit_counts_would_be_denials_without_denying(self):
        class Target:
            def mystery_action(self):
                return "ran"

        method = check_delegable(Target(), target_name="reader", action="mystery_action", caller="orchestrator")
        self.assertEqual(method(), "ran")  # audit mode never denies
        self.assertIn("reader.mystery_action", self.agent.delegation_audit["would_deny"])

    def test_enforce_capability_registry_true_denies_unregistered_delegations(self):
        GovernanceCore.reset_shared()
        install_authorizer(None)
        from src.agents.governance import delegation_guard as dg

        with mock.patch("src.agents.governance_agent.get_config_section") as cfg:
            real = {"enabled": True, "state_dir": "data/gov2", "enforce_capability_registry": True}
            cfg.return_value = real
            agent = self.GovernanceAgent(shared_memory=self.SharedMemory(), agent_factory=None)

        class Target:
            def mystery_action(self):
                return "ran"

        with self.assertRaises(dg.DelegationDenied):
            check_delegable(Target(), target_name="reader", action="mystery_action", caller="o")
        self.assertEqual(agent.delegation_audit["mode"], "enforce")

    def test_enforce_mode_blocks_agents_before_they_run(self):
        self.agent.agent_gate.mode = "enforce"

        class ReaderAgent(self.BaseAgent):
            ran = False

            def perform_task(self, task):
                type(self).ran = True
                return {"status": "success"}

        agent = ReaderAgent(shared_memory=self.SharedMemory(), agent_factory=None, config={})
        # Free text cannot be proven non-personal, so it stays UNCLASSIFIED and is refused before the agent runs.
        with self.assertRaises(GovernanceDenied):
            agent.execute({"text": "please summarise the attached meeting notes"})
        self.assertFalse(ReaderAgent.ran)
        self.assertEqual(self.agent.agent_gate.status()["stats"]["default_unclassified"], 1)

    def test_enforce_mode_admits_provably_non_personal_internal_calls_inside_one_scope(self):
        from src.agents.governance.effect_boundary import current_scope, guard_effect

        self.agent.agent_gate.mode = "enforce"
        seen = {}

        class ReaderAgent(self.BaseAgent):
            def perform_task(self, task):
                scope = current_scope()
                seen["scope"] = scope.auth_id if scope else None
                seen["permit"] = guard_effect("file_write", "report.txt").outcome  # inherits the scope: no second decision
                return {"status": "success"}

        agent = ReaderAgent(shared_memory=self.SharedMemory(), agent_factory=None, config={})
        self.assertEqual(agent.execute({"x": 1}), {"status": "success"})
        self.assertIsNotNone(seen["scope"])
        self.assertEqual(seen["permit"], "scoped")
        stats = self.agent.agent_gate.status()["stats"]
        self.assertEqual((stats["evaluated"], stats["default_classified"]), (1, 1))
        self.assertIsNone(current_scope())  # the scope never outlives the execution
        self.assertEqual(self.agent.core.reassessment.active(), [])  # and its lease is completed

    def test_side_effect_outside_any_execution_is_refused_in_enforce_mode(self):
        from src.agents.governance.effect_boundary import guard_effect

        self.agent.agent_gate.mode = "enforce"
        with self.assertRaises(GovernanceDenied):
            guard_effect("subprocess", "rm")


class FactoryBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.SharedMemory, _, _ = _shim_and_import()
            from src.agents.agent_factory import AgentFactory
        except Exception as exc:
            raise unittest.SkipTest(f"factory not importable here: {type(exc).__name__}: {exc}")
        cls.AgentFactory = AgentFactory

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        for k, v in {"SLAI_GOVERNANCE_MASTER_KEY": "m" * 40, "SLAI_SHARED_MEMORY_KEY": "k" * 40}.items():
            self.addCleanup(os.environ.pop, k, None)
            os.environ[k] = v
        GovernanceCore.reset_shared()
        install_gate(None)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        install_gate(None)
        install_authorizer(None)
        GovernanceCore.reset_shared()
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_creating_any_agent_starts_governance_first(self):
        factory = self.AgentFactory()
        self.assertIsNone(installed_gate())
        factory.create("reader", self.SharedMemory())
        self.assertIsNotNone(installed_gate())
        self.assertIsNone(getattr(factory, "governance_bootstrap_error", None))

    def test_a_governance_start_failure_is_critical_logged_and_only_fatal_when_required(self):
        factory = self.AgentFactory()
        real_create = factory.create

        def failing(agent_type, *a, **k):
            if agent_type == "governance":
                raise RuntimeError("cannot start")
            return real_create(agent_type, *a, **k)

        with mock.patch.object(factory, "create", side_effect=failing):
            factory._ensure_governance("reader", None)
        self.assertEqual(factory.governance_bootstrap_error, "RuntimeError")  # recorded, agents run un-gated and that is visible
        self.assertIsNone(installed_gate())


if __name__ == "__main__":
    unittest.main()
