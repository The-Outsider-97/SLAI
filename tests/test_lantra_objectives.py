"""Offline tests for src/training/lantra_objectives.py (pure Python, no torch)."""
from __future__ import annotations

import dataclasses
import random
import sys
import unittest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import lantra_objectives as lo  # noqa: E402
from src.utils.helpers.response_quality import (  # noqa: E402
    degenerate_reasons,
    find_templated_sentences,
    generation_health,
    has_repetition_loop,
    restore_sentence_case,
)

BOS, EOS, UNK, PAD = 1, 2, 3, 0


def ids(n: int, start: int = 10):
    return [BOS, *range(start, start + n), EOS]


@dataclasses.dataclass(frozen=True)
class Ex:
    task: str
    target: str
    example_id: str = ""


class ObjectiveTests(unittest.TestCase):
    def test_mix_parsing(self):
        mix = lo.parse_objective_mix("continuation=3,span_infill=1")
        self.assertAlmostEqual(sum(mix.values()), 1.0)
        self.assertAlmostEqual(mix["continuation"], 0.75)
        self.assertEqual(set(lo.parse_objective_mix(lo.DEFAULT_OBJECTIVE_MIX)), set(lo.RAW_OBJECTIVES))
        for bad in ("nonsense=1", "continuation=-1", "continuation=0"):
            with self.assertRaises(ValueError):
                lo.parse_objective_mix(bad)

    def test_choose_objective_respects_weights(self):
        mix = lo.parse_objective_mix("continuation=0.8,denoise=0.2")
        rng = random.Random(0)
        draws = [lo.choose_objective(rng, mix) for _ in range(4000)]
        self.assertAlmostEqual(draws.count("continuation") / 4000, 0.8, delta=0.03)
        self.assertNotIn("span_infill", draws)

    def test_continuation_partitions_the_text(self):
        rng = random.Random(1)
        original = ids(40)
        for _ in range(50):
            src, tgt = lo.continuation_pair(original, bos=BOS, eos=EOS, rng=rng)
            self.assertEqual((src[0], src[-1], tgt[0], tgt[-1]), (BOS, EOS, BOS, EOS))
            self.assertEqual(src[1:-1] + tgt[1:-1], original[1:-1])
            self.assertGreaterEqual(len(src) - 2, 6)
            self.assertGreaterEqual(len(tgt) - 2, 6)
            self.assertFalse(set(src[1:-1]) & set(tgt[1:-1]))  # nothing to copy from the source
        self.assertIsNone(lo.continuation_pair(ids(5), bos=BOS, eos=EOS, rng=rng))

    def test_span_infill_is_invertible_and_not_a_copy_task(self):
        rng = random.Random(2)
        original = ids(60)
        for _ in range(100):
            pair = lo.span_infill_pair(original, bos=BOS, eos=EOS, mask_id=UNK, protected=[PAD], rng=rng)
            self.assertIsNotNone(pair)
            src, tgt = pair
            self.assertEqual((src[0], src[-1], tgt[0], tgt[-1]), (BOS, EOS, BOS, EOS))
            masks = [i for i, t in enumerate(src) if t == UNK]
            self.assertTrue(masks)
            self.assertTrue(all(b - a > 1 for a, b in zip(masks, masks[1:])))  # never adjacent
            spans, current = [], []
            for token in tgt[1:-1]:
                if token == UNK:
                    spans.append(current)
                    current = []
                else:
                    current.append(token)
            spans.append(current)
            self.assertEqual(len(spans), len(masks))
            rebuilt, k = [], 0
            for token in src[1:-1]:
                if token == UNK:
                    rebuilt.extend(spans[k])
                    k += 1
                else:
                    rebuilt.append(token)
            self.assertEqual(rebuilt, original[1:-1])
            self.assertLess(len(src), len(original))  # shorter source: content is missing, not copyable

    def test_build_raw_pair_falls_back(self):
        rng = random.Random(3)
        corrupt = lambda seq: [UNK if i == 2 else t for i, t in enumerate(seq)]
        short = ids(4)
        src, tgt = lo.build_raw_pair(short, objective="continuation", bos=BOS, eos=EOS, mask_id=UNK,
                                     protected=[], rng=rng, corrupt=corrupt)
        self.assertEqual(tgt, short)
        self.assertEqual(src[2], UNK)
        with self.assertRaises(ValueError):
            lo.build_raw_pair(short, objective="bogus", bos=BOS, eos=EOS, mask_id=UNK, protected=[], rng=rng, corrupt=corrupt)

    def test_padding_and_teacher_forcing(self):
        padded, valid = lo.pad_sequences([[1, 2, 3], [4]], PAD)
        self.assertEqual(padded, [[1, 2, 3], [4, 0, 0]])
        self.assertEqual(valid, [[True] * 3, [True, False, False]])
        self.assertEqual(lo.teacher_forcing_split([BOS, 7, 8, EOS]), ([BOS, 7, 8], [7, 8, EOS]))
        with self.assertRaises(ValueError):
            lo.teacher_forcing_split([BOS])

    def test_micro_batches(self):
        items = list(range(10))
        self.assertEqual([len(b) for b in lo.micro_batches(items, max_items=4)], [4, 4, 2])
        lengths = {i: 10 * (i + 1) for i in items}
        batches = lo.micro_batches(items, max_items=100, max_tokens=100, length=lambda i: lengths[i])
        self.assertEqual([i for b in batches for i in b], items)
        for b in batches[:-1]:
            self.assertLessEqual(max(lengths[i] for i in b) * len(b), 100)


class AuditTests(unittest.TestCase):
    def make(self, n=300):
        filler = "Start with the part that matters most to you."
        examples = []
        for i in range(n):
            core = f"Reply number {i} talks about subject {i % 37} in detail word{i}."
            tail = f" Extra context sentence {i % 11} here." if i % 3 == 0 else ""
            examples.append(Ex("dialogue", f"{core}{tail} {filler}" if i % 2 == 0 else core, f"e{i}"))
        examples.append(Ex("generation", "Not dialogue. Start with the part that matters most to you."))
        return examples

    def test_strips_filler_but_leaves_other_tasks_alone(self):
        examples = self.make()
        kept, audit, findings = lo.audit_dialogue_examples(examples)
        self.assertEqual(len(findings), 1)
        self.assertGreater(audit.stripped_examples, 100)
        self.assertTrue(all("matters most" not in e.target for e in kept if e.task == "dialogue"))
        self.assertIn("matters most", [e for e in kept if e.task == "generation"][0].target)
        self.assertEqual(find_templated_sentences([e.target for e in kept if e.task == "dialogue"]), {})

    def test_duplicate_cap_and_legitimate_greetings(self):
        examples = [Ex("dialogue", "Hi! How can I help you?") for _ in range(40)]
        examples += [Ex("dialogue", f"Hello there, number {i}. What would you like to ask about topic {i}?") for i in range(40)]
        kept, audit, findings = lo.audit_dialogue_examples(examples, duplicate_cap=12)
        self.assertEqual(sum(1 for e in kept if e.target == "Hi! How can I help you?"), 12)
        self.assertEqual(audit.capped_duplicates, 28)
        self.assertEqual(findings, {})  # a legitimate follow-up question is not "filler"

    def test_selection_objective_penalises_collapse(self):
        healthy = generation_health(["hi", "how are you"] * 6, [f"answer number {i} is fine" for i in range(12)])
        collapsed = generation_health(["hi", "how are you"] * 6, ["meta meta meta meta meta"] * 12)
        self.assertTrue(lo.is_healthy(healthy))
        self.assertFalse(lo.is_healthy(collapsed))
        good = lo.selection_objective(0.30, healthy["penalty"])
        bad = lo.selection_objective(0.20, collapsed["penalty"])  # lower loss, collapsed outputs
        self.assertLess(good, bad)
        self.assertEqual(lo.selection_objective(0.3, None), 0.3)
        with self.assertRaises(ValueError):
            lo.selection_objective(float("nan"), 0.0)


class ResponseQualityTests(unittest.TestCase):
    def test_loops_and_echo(self):
        self.assertTrue(has_repetition_loop("meta meta meta meta meta meta"))
        self.assertTrue(has_repetition_loop("the team the team the team the team"))
        self.assertFalse(has_repetition_loop("Hi there, how can I help you today?"))
        self.assertIn("echoes_user", degenerate_reasons("what is the capital of france", user_text="what is the capital of france"))
        self.assertIn("repeats_previous_assistant", degenerate_reasons("same answer here friend", previous_assistant=["same answer here friend"]))

    def test_sentence_case(self):
        self.assertEqual(restore_sentence_case("hi! i'm here. how can i help?"), "Hi! I'm here. How can I help?")


if __name__ == "__main__":
    unittest.main(verbosity=2)
