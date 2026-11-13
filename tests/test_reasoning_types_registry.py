"""Registry, alias, keyword-policy, result-contract and combined-execution tests for ReasoningTypes.

Each test pins a defect found by the SLAI v2.4 Reasoning Types audit (see the docstring of the
test for the failing behaviour it guards against). Run from the repository root:
    python -m unittest tests.test_reasoning_types_registry
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.reasoning.reasoning_types import ReasoningTypes
from src.agents.reasoning.strategy_selector import STRATEGIES
from src.agents.reasoning.types import (
    BaseReasoning, ReasoningAbduction, ReasoningCauseAndEffect, ReasoningDeductive,
)
from src.agents.reasoning.utils.config_loader import load_global_config
from src.agents.reasoning.utils.reasoning_errors import InferenceExecutionError, ReasoningTypeError

from tests.test_reasoning_strategy_selector import mini_model

CANONICAL = sorted(STRATEGIES)


class _Boom(BaseReasoning):
    def perform_reasoning(self, input_data=None, context=None, **kwargs):  # noqa: D401
        raise ValueError("boom")


class _Echo(BaseReasoning):
    def perform_reasoning(self, input_data=None, context=None, **kwargs):
        return {"echo": input_data, "metrics": {"success": True}}


class _Mutator(BaseReasoning):
    """Tries to corrupt the previous step's result through the shared context."""

    def perform_reasoning(self, input_data=None, context=None, **kwargs):
        prev = (context or {}).get("prev_step_result")
        if isinstance(prev, dict):
            prev.setdefault("items", []).append("MUTATED")
        return {"saw": copy.deepcopy(prev)}


class _Producer(BaseReasoning):
    def perform_reasoning(self, input_data=None, context=None, **kwargs):
        return {"items": ["original"]}


class RegistryTests(unittest.TestCase):
    def test_registry_is_instance_scoped(self) -> None:
        """register() used to write into the class-level dict, leaking into every instance."""
        first, second = ReasoningTypes(), ReasoningTypes()
        first.register("echo", _Echo)
        self.assertIn("echo", first._TASK_TYPES)
        self.assertNotIn("echo", second._TASK_TYPES)
        self.assertNotIn("echo", ReasoningTypes._TASK_TYPES)

    def test_register_rejects_non_strategies(self) -> None:
        types = ReasoningTypes()
        for bad_name, bad_cls in (("", _Echo), ("x", object), ("a+b", _Echo), ("y", "not-a-class")):
            with self.assertRaises(ReasoningTypeError):
                types.register(bad_name, bad_cls)  # type: ignore[arg-type]

    def test_aliases_resolve_without_polluting_registry(self) -> None:
        """discover_task_types() used to add 'deductive', 'inductive', 'causeandeffect', ... as new keys."""
        types = ReasoningTypes()
        cases = {
            "deductive": ReasoningDeductive,
            "Cause and Effect": ReasoningCauseAndEffect,
            "CAUSE-EFFECT": ReasoningCauseAndEffect,
            "causal": ReasoningCauseAndEffect,
            "abductive": ReasoningAbduction,
        }
        for alias, expected in cases.items():
            self.assertIsInstance(types.create(alias), expected, alias)
        self.assertEqual(sorted(types._TASK_TYPES), CANONICAL)
        self.assertEqual(sorted(ReasoningTypes._TASK_TYPES), CANONICAL)

    def test_unknown_type_is_a_typed_error_and_does_not_pollute(self) -> None:
        types = ReasoningTypes()
        with self.assertRaises(ReasoningTypeError):
            types.create("telepathy")
        with self.assertRaises(ReasoningTypeError):
            types.create("")
        self.assertEqual(sorted(types._TASK_TYPES), CANONICAL)

    def test_combined_expression_uses_canonical_names_and_limit(self) -> None:
        types = ReasoningTypes()
        self.assertEqual(types._parse_combined_types("abductive + Deductive"), ["abduction", "deduction"])
        with self.assertRaises(ReasoningTypeError):
            types.create("abduction+induction+deduction+cause_effect")
        with self.assertRaises(ReasoningTypeError):
            types.create("abduction+nonsense")

    def test_normalize_result_canonicalizes_strategy_name(self) -> None:
        """normalize_result('deductive', ...) skipped every deduction extractor (conclusion/confidence None)."""
        types = ReasoningTypes()
        raw = ReasoningDeductive().perform_reasoning(
            ["all humans are mortal", "socrates is human"], "socrates is mortal", {}
        )
        normalized = types.normalize_result("deductive", raw)
        self.assertEqual(normalized["strategy"], "deduction")
        self.assertEqual(normalized["outcome"], "supported")
        self.assertIsNotNone(normalized["confidence"]["value"])


class KeywordPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.types = ReasoningTypes()

    def strategy(self, problem) -> str:
        return self.types.select_reasoning_strategy(problem)["strategy"]

    def test_substring_false_positives_are_gone(self) -> None:
        """'cause' matched 'because', 'similar' matched 'dissimilar', 'why' matched 'somewhere'."""
        default = self.types.default_strategy
        for text in ("Because the server crashed", "The two are dissimilar", "He went somewhere else"):
            selection = self.types.select_reasoning_strategy(text)
            self.assertEqual(selection["method"], "keyword_policy_default", text)
            self.assertEqual(selection["strategy"], default, text)

    def test_inflected_keywords_still_match(self) -> None:
        self.assertEqual(self.strategy("What are the side effects of this change?"), "cause_effect")
        self.assertIn("abduction", self.strategy("Explaining why the grass is wet").split("+"))

    def test_multiword_keyword_tolerates_whitespace(self) -> None:
        self.assertIn("decompositional", self.strategy("please break   down the system").split("+"))

    def test_cap_keeps_strongest_evidence_not_alphabetical_order(self) -> None:
        """The old policy cut alphabetically, silently dropping e.g. cause_effect/induction."""
        types = ReasoningTypes()
        types.max_combined_types = 2
        selection = types._select_by_keywords("prove it, then find the pattern and predict the trend, also compare")
        chosen = selection["strategy"].split("+")
        self.assertEqual(len(chosen), 2)
        self.assertIn("induction", chosen)  # three distinct induction keywords beat the single-hit strategies

    def test_structured_problem_is_matched_on_text_fields_not_keys(self) -> None:
        selection = self.types.select_reasoning_strategy({"causes": ["x"], "values": [1, 2]})
        self.assertEqual(selection["method"], "keyword_policy_default")


class SelectorCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "model.json"
        self.path.write_text(json.dumps(mini_model()), encoding="utf-8")

    def make(self) -> ReasoningTypes:
        cfg = copy.deepcopy(load_global_config())
        cfg.setdefault("reasoning_types", {})["learned_selector"] = {
            "enabled": True, "model_path": str(self.path), "allow_experimental": True,
        }
        return ReasoningTypes(config=cfg)

    def test_label_space_matches_registry(self) -> None:
        self.assertEqual(sorted(STRATEGIES), sorted(ReasoningTypes._TASK_TYPES))

    def test_selector_refused_when_registry_cannot_dispatch_its_labels(self) -> None:
        """A checkpoint whose labels are not all dispatchable must be refused, never 'fixed up'."""
        remaining = {k: v for k, v in ReasoningTypes._TASK_TYPES.items() if k != "induction"}
        with mock.patch.dict(ReasoningTypes._TASK_TYPES, remaining, clear=True):
            types = self.make()
        self.assertTrue(types.get_stats()["learned_selector"].startswith("incompatible_registry"))
        self.assertIsNone(types._learned_selector)

    def test_shipped_checkpoint_stays_experimental_and_off_by_default(self) -> None:
        types = ReasoningTypes()
        self.assertEqual(types.get_stats()["learned_selector"], "disabled")


class ResultContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.types = ReasoningTypes()

    def test_cause_effect_normalization_reads_valid_relationships(self) -> None:
        """The normalizer read 'validated_relationships' (never produced): outcome was always
        'indeterminate' with no confidence and no evidence for every successful causal run."""
        events = [
            {"id": "rain", "description": "Heavy rainfall", "timestamp": 1,
             "attributes": {"intensity": "high", "causes": ["flood"]}},
            {"id": "river", "description": "River water level rises", "timestamp": 2,
             "attributes": {"intensity": "high", "effects": ["flood"]}},
            {"id": "warning", "description": "Flood warning issued", "timestamp": 3, "attributes": {"severity": "high"}},
        ]
        result = self.types.reason("cause_effect", events, {"conditions": {"intensity": "high"}, "max_time_gap": 5})
        self.assertEqual(result["outcome"], "supported")
        self.assertIsNotNone(result["confidence"]["value"])
        self.assertFalse(result["confidence"]["calibrated"])
        self.assertTrue(result["evidence"])

    def test_association_only_causal_finding_is_not_reported_as_supported(self) -> None:
        raw = {"valid_relationships": [{"type": "sequential", "confidence": 0.9, "mechanism_score": 0.0}]}
        normalized = self.types.normalize_result("cause_effect", raw)
        self.assertEqual(normalized["outcome"], "partial")
        self.assertEqual(normalized["stop_reason"], "association_only_no_mechanism")

    def test_declared_mechanism_is_supported(self) -> None:
        raw = {"valid_relationships": [{"type": "declared_mechanism", "confidence": 0.8}]}
        self.assertEqual(self.types.normalize_result("cause_effect", raw)["outcome"], "supported")

    def test_deduction_outcomes_supported_disproven_indeterminate(self) -> None:
        premises = ["all humans are mortal", "socrates is human"]
        supported = self.types.reason("deduction", "x", {"premises": premises, "hypothesis": "socrates is mortal"})
        self.assertEqual(supported["outcome"], "supported")
        disproven = self.types.reason("deduction", "x", {"premises": premises, "hypothesis": "socrates is not mortal"})
        self.assertEqual(disproven["outcome"], "disproven")
        self.assertEqual(disproven["stop_reason"], "hypothesis_refuted")
        undetermined = self.types.reason("deduction", "x", {"premises": premises, "hypothesis": "plato is wise"})
        self.assertEqual(undetermined["outcome"], "indeterminate")
        self.assertEqual(undetermined["confidence"]["value"], 0.0)

    def test_abduction_template_hypothesis_is_unsupported(self) -> None:
        """A template that restates the observation 'explained' 100% of it and was reported supported."""
        result = self.types.reason("abduction", "The grass is wet", {})
        self.assertEqual(result["outcome"], "unsupported")
        self.assertEqual(result["stop_reason"], "no_grounded_hypothesis")
        self.assertFalse(result["output"]["metrics"]["grounded"])

    def test_abduction_with_declared_explanation_is_supported(self) -> None:
        context = {"seed_hypotheses": [{"hypothesis": "It rained overnight", "prior": 0.8, "explains": [0]}]}
        result = self.types.reason("abduction", "The grass is wet", context)
        self.assertEqual(result["outcome"], "supported")
        self.assertEqual(result["conclusion"], "It rained overnight")

    def test_abduction_contradiction_is_surfaced_and_blocks_selection(self) -> None:
        context = {
            "seed_hypotheses": [{"hypothesis": "power failure", "explains": [0]}],
            "evidence_sources": [{"content": "power failure ruled out", "type": "contradiction",
                                  "confidence": 0.95, "source": "audit"}],
        }
        result = self.types.reason("abduction", "lights out", context)
        self.assertNotEqual(result["outcome"], "supported")
        self.assertTrue(result["contradictions"])

    def test_decompositional_free_text_is_not_reported_completed(self) -> None:
        result = self.types.reason("decompositional", "break down the quarterly planning problem", {})
        self.assertEqual(result["outcome"], "indeterminate")
        self.assertEqual(result["stop_reason"], "not_decomposable")

    def test_decompositional_structured_system_completes(self) -> None:
        system = {"name": "shop", "components": {"auth": {"functions": ["login"]}, "db": {"functions": ["store"]}}}
        result = self.types.reason("decompositional", system, {})
        self.assertEqual(result["outcome"], "completed")

    def test_confidence_is_never_marked_calibrated(self) -> None:
        for strategy, problem, ctx in (
            ("deduction", "x", {"premises": ["a is b"], "hypothesis": "a is b"}),
            ("abduction", "x y", {}),
            ("induction", [1, 2, 3, 4, 5, 6], {}),
        ):
            normalized = self.types.reason(strategy, problem, ctx)
            self.assertFalse(normalized["confidence"]["calibrated"], strategy)
            self.assertIsNone(normalized["confidence"]["calibration_method"], strategy)

    def test_normalized_result_is_json_serialisable(self) -> None:
        result = self.types.reason("induction", [{"content": {"t": i}} for i in range(8)], {})
        json.dumps(result)


class CombinedExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.types = ReasoningTypes()
        for name, cls in (("boom", _Boom), ("echo", _Echo), ("mutator", _Mutator), ("producer", _Producer)):
            self.types.register(name, cls)

    def test_failing_step_yields_partial_result_not_total_loss(self) -> None:
        """One step raising used to abort the chain and lose every completed step's output."""
        result = self.types.reason("echo+boom+echo", "p", {})
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["outcome"], "partial")
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["errors"][0]["strategy"], "boom")
        self.assertEqual(result["errors"][0]["error_type"], "ValueError")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["step_count"], 2)
        self.assertEqual(self.types.get_stats()["combined_step_failures"], 1)

    def test_all_steps_failing_raises_typed_error(self) -> None:
        with self.assertRaises(InferenceExecutionError) as caught:
            self.types.reason("boom+boom", "p", {})
        self.assertIn("step_errors", caught.exception.context)

    def test_fail_fast_stops_at_first_failure(self) -> None:
        raw = self.types.execute("echo+boom+echo", "p", {"combined_fail_fast": True})
        self.assertEqual(list(raw["combined_result"]), ["step_1_echo"])
        self.assertEqual(list(raw["step_errors"]), ["step_2_boom"])

    def test_later_step_cannot_mutate_earlier_result(self) -> None:
        raw = self.types.execute("producer+mutator", "p", {})
        self.assertEqual(raw["combined_result"]["step_1_producer"], {"items": ["original"]})
        self.assertIn("MUTATED", raw["combined_result"]["step_2_mutator"]["saw"]["items"])

    def test_original_context_is_not_modified(self) -> None:
        context = {"premises": ["a"], "nested": {"k": [1]}}
        snapshot = copy.deepcopy(context)
        self.types.execute("echo+echo", "p", context)
        self.assertEqual(context, snapshot)

    def test_aggregate_outcome_rules(self) -> None:
        agg = ReasoningTypes._aggregate_combined_outcome
        self.assertEqual(agg(["supported", "disproven"], has_errors=False), "indeterminate")
        self.assertEqual(agg(["supported"], has_errors=True), "partial")
        self.assertEqual(agg(["indeterminate", "supported"], has_errors=False), "partial")
        self.assertEqual(agg(["supported", "supported"], has_errors=False), "supported")
        self.assertEqual(agg(["disproven"], has_errors=False), "disproven")
        self.assertEqual(agg([], has_errors=True), "failed")

    def test_conflicting_steps_are_reported_indeterminate(self) -> None:
        premises = ["all humans are mortal", "socrates is human"]
        raw_support = ReasoningDeductive().perform_reasoning(premises, "socrates is mortal", {})
        raw_refute = ReasoningDeductive().perform_reasoning(premises, "socrates is not mortal", {})
        normalized = self.types.normalize_result(
            "deduction+deduction",
            {"combined_result": {"step_1_deduction": raw_support, "step_2_deduction": raw_refute},
             "final_output": raw_refute},
        )
        self.assertEqual(normalized["outcome"], "indeterminate")

    def test_combined_contradictions_are_unioned(self) -> None:
        premises = ["all humans are mortal", "socrates is human"]
        refute = ReasoningDeductive().perform_reasoning(premises, "socrates is not mortal", {})
        normalized = self.types.normalize_result(
            "echo+deduction", {"combined_result": {"step_1_echo": {"x": 1}, "step_2_deduction": refute}},
        )
        self.assertTrue(normalized["contradictions"])

    def test_stats_count_executions(self) -> None:
        before = self.types.get_stats()["total_executions"]
        self.types.execute("echo", "p", {})
        self.types.execute("echo+echo", "p", {})
        self.assertEqual(self.types.get_stats()["total_executions"], before + 2)


try:  # ReasoningAgent needs heavy optional dependencies (torch, pgmpy, ...)
    from src.agents.reasoning_agent import ReasoningAgent
except Exception:  # pragma: no cover - environment dependent
    ReasoningAgent = None  # type: ignore[assignment]


@unittest.skipIf(ReasoningAgent is None, "ReasoningAgent dependencies are not installed")
class AgentIntegrationTests(unittest.TestCase):
    def _fake_agent(self):
        import types as pytypes
        return pytypes.SimpleNamespace(
            types=ReasoningTypes(), reasoning_type_aliases={"logic": "deduction"},
            strategy_default="deduction", low_confidence_threshold=0.5,
        )

    def test_auto_selection_receives_raw_problem_and_context(self) -> None:
        agent = self._fake_agent()
        resolved, selection = ReasoningAgent._resolve_reasoning_selection(  # type: ignore[union-attr]
            agent, "auto", {"text": "Explain why it failed", "causes": ["x"]}, {"premises": ["a"]}
        )
        self.assertEqual(resolved, "abduction")
        self.assertEqual(selection["method"], "keyword_policy")

    def test_explicit_alias_is_canonicalized(self) -> None:
        agent = self._fake_agent()
        resolved, _ = ReasoningAgent._resolve_reasoning_selection(agent, "deductive", "p", {})  # type: ignore[union-attr]
        self.assertEqual(resolved, "deduction")

    def test_unsettled_outcomes_trigger_refinement(self) -> None:
        agent = self._fake_agent()
        for outcome in ("indeterminate", "unsupported", "partial", "failed"):
            reasons = ReasoningAgent._refinement_reasons(agent, {"outcome": outcome})  # type: ignore[union-attr]
            self.assertTrue(reasons, outcome)
        for outcome in ("supported", "disproven"):
            self.assertEqual(ReasoningAgent._refinement_reasons(agent, {"outcome": outcome}), [])  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
