"""Regression tests for defects found while integrating an external application (GangUP) with the real agents.

Every test runs without torch: behaviour is exercised on the isolated class/function source, and the import-level
facts are asserted on the parsed source so a refactor cannot silently reintroduce the defect.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
import typing
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tree(rel: str) -> ast.Module:
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def _class_source(rel: str, name: str) -> str:
    text = (ROOT / rel).read_text(encoding="utf-8")
    for node in _tree(rel).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{name} not found in {rel}")


class BaseTokenizerImports(unittest.TestCase):
    def test_literal_is_imported_where_it_is_used(self):
        rel = "src/agents/base/modules/base_tokenizer.py"
        tree = _tree(rel)
        imported = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "typing" for a in n.names}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertIn("Literal", used)
        self.assertIn("Literal", imported)


class BytePairEncoderSingleSymbol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = _class_source("src/agents/perception/modules/tokenizer.py", "BytePairEncoder")
        ns: dict = {k: getattr(typing, k) for k in ("List", "Dict", "Tuple", "Optional")}
        exec(compile(textwrap.dedent(src), "BytePairEncoder", "exec"), ns)
        cls.BPE = ns["BytePairEncoder"]

    def test_merges_that_fold_a_word_into_one_symbol_do_not_crash(self):
        enc = self.BPE([("a", "b"), ("ab", "</w>")], {}, "[UNK]")
        self.assertEqual(enc.bpe("ab"), ["ab</w>"])

    def test_ordinary_merges_unchanged(self):
        enc = self.BPE([("l", "o"), ("lo", "w"), ("low", "</w>")], {}, "[UNK]")
        self.assertEqual(enc.bpe("low"), ["low</w>"])
        partial = self.BPE([("l", "o")], {}, "[UNK]")
        self.assertEqual(partial.bpe("lox"), ["lo", "x"])
        self.assertEqual(partial.bpe("x"), ["x"])


class PlanningAnyBinding(unittest.TestCase):
    def test_planning_agent_rebinds_its_own_any_after_star_imports(self):
        tree = _tree("src/agents/planning_agent.py")
        order = []
        for i, n in enumerate(tree.body):
            if isinstance(n, ast.ImportFrom):
                if any(a.name == "*" for a in n.names) and (n.module or "").endswith("planning_errors"):
                    order.append(("star_errors", i))
                if (n.module or "").endswith("planning.planning_types") and any(a.name == "Any" and a.asname is None for a in n.names):
                    order.append(("any", i))
        star = [i for k, i in order if k == "star_errors"]
        anys = [i for k, i in order if k == "any"]
        self.assertTrue(star and anys)
        self.assertGreater(max(anys), max(star), "SLAI's Any must be (re)imported after `planning_errors import *`")


class ResponseQualityModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("response_quality", ROOT / "src/utils/helpers/response_quality.py")
        cls.rq = importlib.util.module_from_spec(spec)
        sys.modules["response_quality"] = cls.rq
        spec.loader.exec_module(cls.rq)

    def test_loops_and_echo(self):
        rq = self.rq
        self.assertTrue(rq.has_repetition_loop("meta meta meta meta meta meta"))
        self.assertTrue(rq.has_repetition_loop("the team the team the team the team"))
        self.assertFalse(rq.has_repetition_loop("Hi there, how can I help you today?"))
        self.assertIn("echoes_user", rq.degenerate_reasons("what is the capital of france", user_text="what is the capital of france"))
        self.assertIn("repeats_previous_assistant", rq.degenerate_reasons("same answer here friend", previous_assistant=["same answer here friend"]))
        self.assertEqual(rq.degenerate_reasons("The capital is Paris, as far as I know."), [])

    def test_sentence_case(self):
        self.assertEqual(self.rq.restore_sentence_case("hi! i'm here. how can i help?"), "Hi! I'm here. How can I help?")

    def test_health_separates_collapse_from_healthy(self):
        prompts = ["hi", "how are you"] * 6
        healthy = self.rq.generation_health(prompts, [f"answer number {i} is fine" for i in range(12)])
        collapsed = self.rq.generation_health(prompts, ["meta meta meta meta meta"] * 12)
        self.assertTrue(healthy["healthy"])
        self.assertFalse(collapsed["healthy"])
        self.assertLess(healthy["penalty"], collapsed["penalty"])

    def test_templated_sentences_need_many_distinct_replies(self):
        filler = "Start with the part that matters most to you."
        texts = [f"Reply {i} is about subject {i} in detail. {filler}" for i in range(100)]
        self.assertEqual(list(self.rq.find_templated_sentences(texts)), ["start with the part that matters most to you"])
        self.assertEqual(self.rq.find_templated_sentences([f"Unique reply number {i} about topic {i}." for i in range(100)]), {})


class UncertaintyAwareHeuristicReadinessCache(unittest.TestCase):
    """The heuristic must not re-read the planning database on every prediction when the data is known to be insufficient."""

    def test_source_keeps_a_negative_readiness_cache(self):
        src = (ROOT / "src/agents/planning/heuristics/uncertainty_aware_heuristic.py").read_text(encoding="utf-8")
        for needle in ("_readiness_signature", "_training_source_signature", "invalidate_training_cache", "def readiness"):
            self.assertIn(needle, src)

    def test_cached_verdict_skips_reload_and_warns_once(self):
        try:
            from src.agents.planning.heuristics.uncertainty_aware_heuristic import UncertaintyAwareHeuristic as UAH
        except Exception as exc:  # torch / optional stack not installed
            self.skipTest(f"planning stack unavailable: {type(exc).__name__}")
        import numpy as np
        uah = object.__new__(UAH)
        import threading
        uah._lock = threading.RLock(); uah.trained = False
        uah._readiness_signature = None; uah._readiness_reason = None; uah._warned_signature = None
        calls = {"n": 0}
        uah.load_training_data = lambda: (calls.__setitem__("n", calls["n"] + 1) or (np.zeros((4, 4)), np.array([1, 0, 1, 1])))
        uah._training_source_signature = lambda: ("db", 1, 1)
        for _ in range(7):
            self.assertFalse(uah._ensure_model_ready())
        self.assertEqual(calls["n"], 1)
        self.assertTrue(uah.readiness["reason"].startswith("insufficient_training_data"))
        uah._training_source_signature = lambda: ("db", 2, 9)  # the data changed: one fresh attempt is allowed
        uah._ensure_model_ready()
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
