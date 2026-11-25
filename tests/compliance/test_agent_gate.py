"""Section 9 / audit 4.6: the central gate is wired into BaseAgent.execute and behaves as documented."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from src.agents.governance.agent_gate import AgentGate, install_gate, installed_gate
from src.agents.governance.processing_context import DataClass, Decision, GovernanceDenied, Operation, ProcessingContext, ProcessingDecision, RuleResult, fingerprint
from src.agents.governance.policy_registry import Policy, PolicyRegistry

from ._fixtures import GovernanceTestCase
from .test_deletion_adapters import _shim_and_import


def make_gate(core, **kw):
    return AgentGate(core, **kw)


class GateUnitTests(GovernanceTestCase):
    def test_observe_records_but_never_blocks(self):
        gate = make_gate(self.core, mode="observe")
        before = len(self.core.decision_log(1000))
        decision = gate.before("reader", {"x": 1})
        self.assertIsNotNone(decision)
        self.assertFalse(decision.decision.permits_execution)  # UNCLASSIFIED default context is not allowed...
        self.assertEqual(gate.status()["stats"]["would_block"], 1)  # ...but observe only counts it
        self.assertGreater(len(self.core.decision_log(1000)), before)  # durably recorded
        self.assertFalse(gate.status()["enforcing"])

    def test_enforce_blocks_uncontextualised_execution_before_the_agent_runs(self):
        gate = make_gate(self.core, mode="enforce")
        with self.assertRaises(GovernanceDenied):
            gate.before("reader", {"x": 1})
        self.assertEqual(gate.status()["stats"]["blocked"], 1)

    def test_explicit_context_is_enforced_even_in_observe_mode(self):
        gate = make_gate(self.core, mode="observe")
        bad = self.ctx(activity_id=None)  # personal data without a registered processing activity
        with self.assertRaises(GovernanceDenied):
            gate.before("reader", {"governance_context": bad})
        self.assertEqual(gate.status()["stats"]["explicit_context"], 1)

    def test_enforce_agents_are_enforced_regardless_of_mode(self):
        gate = make_gate(self.core, mode="observe", enforce_agents=["reader"])
        with self.assertRaises(GovernanceDenied):
            gate.before("reader", {})
        self.assertIsNotNone(gate.before("other", {}))  # not listed: observe

    def test_a_fully_authorised_explicit_context_runs(self):
        self.register()
        gate = make_gate(self.core, mode="enforce")
        decision = gate.before("reader", {"governance_context": self.ctx()})
        self.assertTrue(decision.decision.permits_execution)
        self.assertEqual(gate.status()["stats"]["allowed"], 1)

    def test_off_mode_evaluates_nothing_but_explicit_contexts_still_count(self):
        gate = make_gate(self.core, mode="off")
        self.assertIsNone(gate.before("reader", {}))
        self.assertEqual(gate.status()["stats"]["evaluated"], 0)

    def test_governance_agent_is_exempt(self):
        gate = make_gate(self.core, mode="enforce")
        self.assertIsNone(gate.before("governance", {}))

    def test_aliases_map_class_names_to_factory_keys(self):
        gate = make_gate(self.core, mode="enforce", enforce_agents=(), aliases={"GovernanceAgent": "governance"})
        self.assertIsNone(gate.before("GovernanceAgent", {}))

    def test_gate_failure_fails_closed_when_enforcing_and_is_contained_when_observing(self):
        with mock.patch.object(self.core, "authorize", side_effect=RuntimeError("boom")):
            with self.assertRaises(GovernanceDenied) as cm:
                make_gate(self.core, mode="enforce").before("reader", {})
            self.assertIn("GOV-GATE-ERR-01", {r.rule_id for r in cm.exception.decision.blocking_rules})
            obs = make_gate(self.core, mode="observe")
            self.assertIsNone(obs.before("reader", {}))
            self.assertEqual(obs.status()["stats"]["gate_errors"], 1)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            make_gate(self.core, mode="permissive")

    def test_learning_requires_registered_dataset_lineage_when_enforcing(self):
        gate = make_gate(self.core, mode="enforce", learning_agents=["learning"])
        ctx = self.ctx(operation=Operation.LEARN, purpose="model_training")
        with self.assertRaises(GovernanceDenied) as cm:
            gate.before("learning", {"mode": "train", "dataset_ids": ["ds-unknown"], "governance_context": ctx})
        self.assertIn("GOV-GATE-LRN-01", {r.rule_id for r in cm.exception.decision.blocking_rules})
        with self.assertRaises(GovernanceDenied) as cm2:
            gate.before("learning", {"mode": "train", "governance_context": ctx})  # no dataset_ids at all
        self.assertIn("GOV-GATE-LRN-01", {r.rule_id for r in cm2.exception.decision.blocking_rules})

    def test_default_learning_context_uses_the_learn_operation_and_training_purpose(self):
        gate = make_gate(self.core, mode="observe", learning_agents=["learning"])
        ctx = gate._default_context("learning", {"mode": "train"})
        self.assertIs(ctx.operation, Operation.LEARN)
        self.assertEqual(ctx.purpose, "model_training")
        self.assertIs(gate._default_context("learning", {"mode": "health"}).operation, Operation.EXECUTE)

    def test_mapping_context_is_coerced_and_unknown_keys_ignored(self):
        gate = make_gate(self.core, mode="observe")
        ctx = gate._explicit({"governance_context": {"operation": "process", "purpose": "p", "data_class": "personal", "subject_ids": ["a"], "bogus": 1}})
        self.assertIs(ctx.operation, Operation.PROCESS)
        self.assertIs(ctx.data_class, DataClass.PERSONAL)
        with self.assertRaises(ValueError):
            gate._explicit({"governance_context": "nope"})

    def test_malformed_explicit_context_fails_closed_in_all_modes(self):
        for mode in ("observe", "off", "enforce"):
            for raw in (None, "invalid", {"operation": "invalid", "purpose": "test"}):
                with self.subTest(mode=mode, raw=raw):
                    gate = make_gate(self.core, mode=mode)
                    with self.assertRaises(GovernanceDenied) as cm:
                        gate.before("reader", {"governance_context": raw})
                    self.assertEqual(gate.status()["stats"]["blocked"], 1)
                    self.assertEqual(self.core.decision_log(1)[0]["request_id"], cm.exception.decision.request_id)

    def test_lineage_failure_is_the_durable_final_decision(self):
        self.register(purposes=("model_training",), operations=(Operation.LEARN,), allows_learning=True)
        ctx = self.ctx(operation=Operation.LEARN, purpose="model_training")
        gate = make_gate(self.core, mode="enforce")
        with mock.patch.object(self.core.oversight, "consume", wraps=self.core.oversight.consume) as consume:
            with self.assertRaises(GovernanceDenied):
                gate.before("learning", {"mode": "train", "dataset_ids": ["unknown"], "governance_context": ctx})
            consume.assert_not_called()
        row = self.core.decision_log(1)[0]
        self.assertEqual(row["request_id"], ctx.request_id)
        self.assertEqual(row["decision"], Decision.BLOCK.value)
        self.assertIn("GOV-GATE-LRN-01", row["rules"])

    def test_failed_precondition_does_not_consume_an_approval_on_an_otherwise_allowed_decision(self):
        ctx = self.ctx()
        # Isolate the commit boundary from the independently tested policy/oversight evaluation.
        candidate = ProcessingDecision(ctx.request_id, Decision.ALLOW_WITH_CONTROLS,
            (RuleResult("GOV-HO-14", Decision.ALLOW_WITH_CONTROLS, "approved"),),
            self.core.policies.active_version(), ctx.fingerprint(), conditions=("consume_approval:approval-1",))
        with mock.patch.object(self.core.engine, "evaluate", return_value=candidate), mock.patch.object(self.core.oversight, "consume") as consume:
            denied = self.core.authorize(ctx, precondition_failure=("GOV-GATE-LRN-01", "lineage missing"))
            consume.assert_not_called()
        self.assertEqual(denied.decision, Decision.BLOCK)
        self.assertEqual(self.core.decision_log(1)[0]["decision"], Decision.BLOCK.value)

    def test_training_cannot_be_disguised_as_an_ordinary_processing_operation(self):
        self.register()
        with self.assertRaises(GovernanceDenied) as cm:
            make_gate(self.core).before("learning", {"mode": "train", "governance_context": self.ctx()})
        self.assertIn("GOV-GATE-LRN-01", self.rule_ids(cm.exception.decision))

    def test_explicit_context_cannot_claim_a_different_execution_agent(self):
        self.register()
        with self.assertRaises(GovernanceDenied) as cm:
            make_gate(self.core).before("reader", {"governance_context": self.ctx(agent_id="language")})
        self.assertIn("GOV-CAP-13", self.rule_ids(cm.exception.decision))

    def test_lineage_dataset_lookup_deduplicates_without_changing_scope(self):
        gate = make_gate(self.core)
        ctx = self.ctx(operation=Operation.LEARN)
        with mock.patch.object(self.core.learning, "dataset_registered", return_value=True) as registered:
            self.assertIsNone(gate._lineage_failure(ctx, {"dataset_ids": ["ds", "ds"]}))
            registered.assert_called_once_with("ds", ctx.tenant_id)
            self.assertIsNotNone(gate._lineage_failure(ctx, {"dataset_ids": "ds"}))

    def test_policy_version_reuses_digest_and_invalidates_on_mutation(self):
        registry = PolicyRegistry()
        with mock.patch("src.agents.governance.policy_registry.fingerprint", wraps=fingerprint) as digest:
            original = registry.active_version()
            self.assertEqual(registry.active_version(), original)
            self.assertEqual(digest.call_count, 1)
            registry.register(Policy("LOCAL-TEST", 1, "Test policy", (), mandatory=False))
            self.assertNotEqual(registry.active_version(), original)
            self.assertEqual(digest.call_count, 2)
            registry.deactivate("LOCAL-TEST")
            self.assertEqual(registry.active_version(), original)
            self.assertEqual(digest.call_count, 3)

    # ---------------------------------------------------------------- Art. 50 marking
    SPEC = {"language": {"system_id": "slai-language", "name": "SLAI Language Agent", "model": "m", "interacts_with_natural_persons": True,
                         "generates_synthetic_content": True}}

    def test_mapping_results_of_marking_agents_carry_a_verifiable_marking(self):
        gate = make_gate(self.core, mode="observe", marking_agents=self.SPEC)
        original = {"status": "success", "text": "hallo"}
        marked = gate.after("language", {}, original)
        self.assertNotIn("ai_content_marking", original)  # caller's object untouched
        m = marked["ai_content_marking"]
        self.assertIn("machine_readable_marking", m["measures"])
        self.assertTrue(m["disclosures"])
        self.assertTrue(m["limitations"])  # known limits travel with the marking
        import json
        data = json.dumps(original, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
        self.assertTrue(self.core.transparency.verify(data, m["manifest"]))
        self.assertEqual(gate.status()["stats"]["marked"], 1)

    def test_non_mapping_results_are_returned_unchanged_and_counted(self):
        gate = make_gate(self.core, mode="observe", marking_agents=self.SPEC)
        self.assertEqual(gate.after("language", {}, "plain string"), "plain string")
        self.assertEqual(gate.status()["stats"]["marking_unattached"], 1)

    def test_explicit_context_enforces_marking_even_in_observe_mode(self):
        gate = make_gate(self.core, mode="observe", marking_agents=self.SPEC)
        input_data = {"governance_context": self.ctx()}
        with mock.patch.object(self.core.transparency, "release", side_effect=RuntimeError("no key")):
            with self.assertRaises(GovernanceDenied):
                gate.after("language", input_data, {"a": 1})
        with self.assertRaises(GovernanceDenied):
            gate.after("language", input_data, "plain string")
        self.assertEqual(gate.status()["stats"]["marking_errors"], 2)

    def test_unlisted_agents_are_not_marked(self):
        gate = make_gate(self.core, mode="observe", marking_agents=self.SPEC)
        r = {"a": 1}
        self.assertIs(gate.after("reader", {}, r), r)

    def test_marking_failure_blocks_release_when_enforcing_but_is_counted_when_observing(self):
        with mock.patch.object(self.core.transparency, "release", side_effect=RuntimeError("no key")):
            with self.assertRaises(GovernanceDenied):
                make_gate(self.core, mode="enforce", marking_agents=self.SPEC).after("language", {}, {"a": 1})
            obs = make_gate(self.core, mode="observe", marking_agents=self.SPEC)
            self.assertEqual(obs.after("language", {}, {"a": 1}), {"a": 1})
            self.assertEqual(obs.status()["stats"]["marking_errors"], 1)


class BaseAgentWiringTests(GovernanceTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.SharedMemory, _, _ = _shim_and_import()
            from src.agents.base_agent import BaseAgent
        except Exception as exc:
            raise unittest.SkipTest(f"BaseAgent not importable here: {type(exc).__name__}: {exc}")

        class LanguageAgent(BaseAgent):  # class name maps to factory key 'language' through the aliases
            calls = 0

            def perform_task(self, task):
                type(self).calls += 1
                return {"status": "success", "text": "hello"}

        cls.Agent = LanguageAgent

    def setUp(self):
        super().setUp()
        os.environ["SLAI_SHARED_MEMORY_KEY"] = "k" * 40
        self.addCleanup(os.environ.pop, "SLAI_SHARED_MEMORY_KEY", None)
        self.addCleanup(install_gate, None)
        self.agent = self.Agent(shared_memory=self.SharedMemory(), agent_factory=None, config={})
        type(self.agent).calls = 0

    def test_without_an_installed_gate_nothing_changes(self):
        install_gate(None)
        self.assertEqual(self.agent.execute({"a": 1}), {"status": "success", "text": "hello"})

    def test_enforce_blocks_before_perform_task_runs(self):
        install_gate(make_gate(self.core, mode="enforce", aliases={"LanguageAgent": "language"}))
        with self.assertRaises(GovernanceDenied):
            self.agent.execute({"a": 1})
        self.assertEqual(type(self.agent).calls, 0)  # the agent never ran

    def test_observe_runs_the_agent_and_records_the_decision(self):
        gate = make_gate(self.core, mode="observe", aliases={"LanguageAgent": "language"})
        install_gate(gate)
        self.assertEqual(self.agent.execute({"a": 1})["text"], "hello")
        self.assertEqual(type(self.agent).calls, 1)
        self.assertEqual(gate.status()["stats"]["evaluated"], 1)

    def test_marking_is_applied_after_execution_for_configured_agents(self):
        install_gate(make_gate(self.core, mode="observe", aliases={"LanguageAgent": "language"}, marking_agents=GateUnitTests.SPEC))
        out = self.agent.execute({"a": 1})
        self.assertIn("ai_content_marking", out)
        self.assertEqual(type(self.agent).calls, 1)  # marking never re-executes the agent

    def test_marking_error_in_enforce_mode_does_not_rerun_the_agent(self):
        install_gate(make_gate(self.core, mode="observe", enforce_agents=["language"], aliases={"LanguageAgent": "language"},
                               marking_agents=GateUnitTests.SPEC))
        explicit = self.ctx(activity_id=None)
        with mock.patch.object(self.core.transparency, "release", side_effect=RuntimeError("x")):
            with self.assertRaises(GovernanceDenied):
                self.agent.execute({"governance_context": explicit})
        self.assertLessEqual(type(self.agent).calls, 1)

    def test_installed_gate_accessor(self):
        gate = make_gate(self.core)
        install_gate(gate)
        self.assertIs(installed_gate(), gate)


if __name__ == "__main__":
    unittest.main()
