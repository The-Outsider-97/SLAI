"""Tests for the learned ReasoningStrategySelector and its ReasoningTypes integration.

Integration tests use a tiny deterministic checkpoint built in a temp directory, so they do not
depend on what the trained model currently predicts. Only the contract test at the end uses the
shipped checkpoint (skipped when it has not been trained). Run from the repository root:
    python -m pytest tests/test_reasoning_strategy_selector.py
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agents.reasoning.reasoning_types import ReasoningTypes
from src.agents.reasoning.strategy_selector import (
    DEFAULT_MODEL_PATH, DENSE_FEATURES, MODEL_SCHEMA, STRATEGIES, ReasoningStrategySelector,
    choose_strategies, order_sequence,
)
from src.agents.reasoning.utils.config_loader import load_global_config

K = len(STRATEGIES)
KEYWORD_PROBES = {
    "Explain why the grass is wet": "abduction",
    "Prove that Socrates is mortal given premises": "deduction",
}


def mini_model(promotion: str = "experimental", min_confidence: float = 0.5) -> dict:
    """One hidden unit lit by the word 'why': abduction ~0.99 with it, ~0.27 without; others ~0.12."""
    n_inputs = 1 + len(DENSE_FEATURES)
    w1 = [[0.0] for _ in range(n_inputs)]
    w1[0][0] = 3.0
    return {
        "schema": MODEL_SCHEMA, "version": "test", "promotion": promotion, "strategies": list(STRATEGIES),
        "features": {"vocab": ["u:why"], "dense": list(DENSE_FEATURES), "keywords": {"deduction": ["prove"]}},
        "network": {"W1": w1, "b1": [0.0]},
        "heads": {"probability": {"W": [[2.0] if s == "abduction" else [0.0] for s in STRATEGIES],
                                  "b": [-1.0 if s == "abduction" else -2.0 for s in STRATEGIES],
                                  "calibration": [[1.0, 0.0]] * K}, "latency": None},
        "selection": {"threshold": 0.5, "min_confidence": min_confidence, "latency_weight": 0.0,
                      "latency_cap_ms": 10000.0, "max_strategies": 3},
        "precedence": [[0.5] * K for _ in range(K)],
    }


class LearnedSelectorIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write_model(self, **kwargs) -> str:
        path = self.dir / "model.json"
        path.write_text(json.dumps(mini_model(**kwargs)), encoding="utf-8")
        return str(path)

    @staticmethod
    def make_types(**selector_cfg) -> ReasoningTypes:
        cfg = copy.deepcopy(load_global_config())
        cfg.setdefault("reasoning_types", {})["learned_selector"] = selector_cfg
        return ReasoningTypes(config=cfg)

    def test_disabled_by_default_keeps_keyword_policy(self) -> None:
        types = ReasoningTypes()
        self.assertEqual(types.get_stats()["learned_selector"], "disabled")
        for problem, expected in KEYWORD_PROBES.items():
            selection = types.select_reasoning_strategy(problem)
            self.assertEqual((selection["strategy"], selection["method"]), (expected, "keyword_policy"))

    def test_induction_is_registered_and_dispatchable(self) -> None:
        types = ReasoningTypes()
        self.assertIn("induction", types._TASK_TYPES)
        self.assertIsNotNone(types.create("induction"))
        self.assertIsNotNone(types.create("abduction+induction"))

    def test_enabled_uses_learned_choice(self) -> None:
        types = self.make_types(enabled=True, model_path=self.write_model(), allow_experimental=True)
        self.assertEqual(types.get_stats()["learned_selector"], "active")
        selection = types.select_reasoning_strategy("Why did it fail?")
        self.assertEqual((selection["strategy"], selection["method"]), ("abduction", "learned_selector"))
        self.assertIn("abduction", selection["matches"])
        self.assertEqual(types.determine_reasoning_strategy("Why did it fail?"), "abduction")

    def test_experimental_checkpoint_is_refused_without_flag(self) -> None:
        types = self.make_types(enabled=True, model_path=self.write_model())
        self.assertTrue(types.get_stats()["learned_selector"].startswith("error"))
        self.assertEqual(types.select_reasoning_strategy("Why did it fail?"),
                         ReasoningTypes().select_reasoning_strategy("Why did it fail?"))

    def test_approved_checkpoint_loads_without_flag(self) -> None:
        types = self.make_types(enabled=True, model_path=self.write_model(promotion="approved"))
        self.assertEqual(types.get_stats()["learned_selector"], "active")

    def test_missing_checkpoint_falls_back_to_keywords(self) -> None:
        types = self.make_types(enabled=True, model_path=str(self.dir / "absent.json"), allow_experimental=True)
        self.assertEqual(types.get_stats()["learned_selector"], "missing_checkpoint")
        for problem, expected in KEYWORD_PROBES.items():
            self.assertEqual(types.select_reasoning_strategy(problem)["strategy"], expected)

    def test_low_confidence_defers_only_to_keyword_policy_with_evidence(self) -> None:
        types = self.make_types(enabled=True, model_path=self.write_model(min_confidence=0.5), allow_experimental=True)
        evidence = types.select_reasoning_strategy("prove it")  # no 'why': learned p~0.27 < 0.5; keyword hit 'prove'
        self.assertEqual((evidence["method"], evidence["strategy"]), ("learned_selector_fallback", "deduction"))
        blind = types.select_reasoning_strategy("zzz")  # keyword policy only has its blind default
        self.assertEqual(blind["method"], "learned_selector")
        self.assertTrue(blind["low_confidence"])

    def test_selection_is_limited_to_registered_strategies(self) -> None:
        # min_confidence=0 keeps the keyword fallback out of the way so only `allowed` is under test.
        types = self.make_types(enabled=True, model_path=self.write_model(min_confidence=0.0), allow_experimental=True)
        remaining = {k: v for k, v in ReasoningTypes._TASK_TYPES.items() if k != "abduction"}
        with mock.patch.dict(ReasoningTypes._TASK_TYPES, remaining, clear=True):
            selection = types.select_reasoning_strategy("Why did it fail?")
        self.assertNotIn("abduction", selection["strategy"].split("+"))

    def test_selector_error_never_breaks_selection(self) -> None:
        types = self.make_types(enabled=True, model_path=self.write_model(), allow_experimental=True)
        with mock.patch.object(ReasoningStrategySelector, "select", side_effect=RuntimeError("boom")):
            selection = types.select_reasoning_strategy("Explain why the grass is wet")
        self.assertEqual((selection["strategy"], selection["method"]), ("abduction", "keyword_policy"))


class StrategySelectorCheckpointTests(unittest.TestCase):
    def test_experimental_checkpoint_needs_explicit_opt_in(self) -> None:
        with self.assertRaises(ValueError):
            ReasoningStrategySelector.from_model(mini_model(), allow_experimental=False)
        ReasoningStrategySelector.from_model(mini_model(), allow_experimental=True)

    def test_tampered_checkpoints_are_rejected(self) -> None:
        def broken(mutate):
            model = mini_model()
            mutate(model)
            return model
        for label, model in {
            "schema": broken(lambda m: m.update(schema="other")),
            "strategy order": broken(lambda m: m.update(strategies=list(reversed(m["strategies"])))),
            "short W1": broken(lambda m: m["network"]["W1"].pop()),
            "non-finite weight": broken(lambda m: m["network"]["W1"][0].__setitem__(0, float("inf"))),
            "calibration slope": broken(lambda m: m["heads"]["probability"]["calibration"][0].__setitem__(0, -1.0)),
            "max_strategies": broken(lambda m: m["selection"].update(max_strategies=K + 1)),
            "precedence of 1": broken(lambda m: m["precedence"][0].__setitem__(1, 1.0)),
        }.items():
            with self.subTest(label), self.assertRaises(ValueError):
                ReasoningStrategySelector.from_model(model)

    def test_odd_inputs_never_raise_or_produce_nan(self) -> None:
        selector = ReasoningStrategySelector.from_model(mini_model())
        for problem in (None, "", "   ", "?", "x" * 50_000, "日本語 why?", ["a", "b"], {"problem": "why"}, {}, 123):
            result = selector.select(problem)
            self.assertTrue(result["sequence"])
            self.assertTrue(all(math.isfinite(s["probability"]) for s in result["scores"].values()))

    def test_selection_rule_and_ordering(self) -> None:
        self.assertEqual(choose_strategies({"a": 0.9, "b": 0.6, "c": 0.4}, 3, 0.5), ["a", "b"])
        self.assertEqual(choose_strategies({"a": 0.1, "b": 0.05}, 3, 0.5), ["a"])  # best is always kept
        self.assertEqual(len(choose_strategies({"a": 1.0, "b": 1.0, "c": 1.0}, 2, 0.0)), 2)
        precedence = [[0.5] * K for _ in range(K)]
        a, b = STRATEGIES.index("decompositional"), STRATEGIES.index("abduction")
        precedence[a][b], precedence[b][a] = 0.9, 0.1
        self.assertEqual(order_sequence(["abduction", "decompositional"], precedence), ["decompositional", "abduction"])

    @unittest.skipUnless(DEFAULT_MODEL_PATH.is_file(), "RSS checkpoint not trained (python train_rss.py all)")
    def test_shipped_checkpoint_selections_are_executable(self) -> None:
        selector = ReasoningStrategySelector(DEFAULT_MODEL_PATH, allow_experimental=True)
        types = ReasoningTypes()
        for problem in ("If every A is B, and X is A, is X also B?", "Why did this system suddenly fail?",
                        "My application started failing after I updated the database. What is the likeliest cause?"):
            selection = selector.select(problem, allowed=types._TASK_TYPES, max_strategies=types.max_combined_types,
                                        fallback=types.select_reasoning_strategy)
            names = selection["strategy"].split("+")
            self.assertTrue(set(names) <= set(types._TASK_TYPES))
            self.assertLessEqual(len(names), types.max_combined_types)
            self.assertEqual(types.reason(selection["strategy"], problem, {})["status"], "success")


if __name__ == "__main__":
    unittest.main()
