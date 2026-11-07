"""Offline tests for src/training/dialogue_teacher.py (no network, no torch)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

from collections import Counter
from pathlib import Path

try:  # old colorama on some machines lacks this helper; the logger only needs it to exist
    import colorama

    if not hasattr(colorama, "just_fix_windows_console"):
        colorama.just_fix_windows_console = lambda *a, **k: None
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["OPENAI_API_KEY"] = "test-openai"
os.environ["OPENROUTER_API_KEY"] = "test-openrouter"
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

from src.training import dialogue_teacher as dt  # noqa: E402
from src.utils.helpers.response_quality import find_templated_sentences  # noqa: E402

TAXONOMY = ROOT / "src/training/configs/lantra_dialogue_taxonomy.yaml"

_SUBJECTS = ["bridge", "garden", "engine", "recipe", "planet", "letter", "river", "market", "robot", "ticket",
             "violin", "castle", "laptop", "storm", "museum", "bicycle", "harbor", "lantern", "compass", "meadow"]
_VERBS = ["inspect", "repair", "describe", "compare", "borrow", "polish", "measure", "photograph", "label", "sketch"]


class FakeTeacher:
    """Deterministic stand-in for an OpenAI-compatible endpoint."""

    def __init__(self, *, bad_every: int = 4, filler: str = "") -> None:
        self.counter = 0
        self.bad_every = bad_every
        self.filler = filler
        self.calls: Counter[str] = Counter()
        self.providers_seen: Counter[str] = Counter()

    def _dialogue(self, family: str, lo: int, hi: int, bad: bool) -> dict:
        self.counter += 1
        n = self.counter
        turns = []
        users = max(lo, min(hi, 1 + n % 2))
        for u in range(users):
            s, v = _SUBJECTS[(n + u) % 20], _VERBS[(n * 3 + u) % 10]
            turns.append({"role": "user", "content": f"could you {v} the {s} number {n}{'abcdefg'[u]} please"})
            reply = f"Sure, the {s} {v} job {n}{'abcdefg'[u]} takes about {n % 7 + 2} minutes if you start with the basics."
            if u == users - 1:
                if bad == "closer":
                    reply += " Let me know if you need anything else."
                elif bad == "markdown":
                    reply = "See https://example.com/x for details. " + reply
                elif bad == "fact":
                    reply += " BADFACT"
                elif bad == "long":
                    reply = " ".join(["word"] * 90)
                if self.filler:
                    reply += " " + self.filler
            turns.append({"role": "assistant", "content": reply})
        return {"turns": turns}

    def __call__(self, url, headers, body, timeout):
        request = json.loads(body.decode("utf-8"))
        prompt = json.loads(request["messages"][1]["content"])
        task = prompt["task"]
        self.calls[task] += 1
        self.providers_seen["openrouter" if "openrouter" in url else "openai"] += 1
        if task == "write_dialogues":
            fam = prompt["family"]
            rng = prompt["user_turns_per_dialogue"]
            out = []
            for i in range(prompt["count"]):
                bad = False
                if self.bad_every and i % self.bad_every == self.bad_every - 1:
                    self.bad_n = getattr(self, "bad_n", -1) + 1
                    bad = ["closer", "markdown", "fact", "long"][self.bad_n % 4]
                out.append(self._dialogue(fam, rng["min"], rng["max"], bad))
            content = {"dialogues": out}
        elif task == "score_dialogues":
            content = {"verdicts": [
                {"id": d["id"], "score": 2 if any("BADFACT" in t for t in d["turns"]) else 5, "issue": "bad fact"}
                for d in prompt["dialogues"]
            ]}
        elif task == "repair_student_replies":
            content = {"replies": [
                {"id": c["id"], "reply": f"Here is a direct answer number {c['id']}{self.counter} for that request."}
                for c in prompt["cases"]
            ]}
            self.counter += 1
        else:  # pragma: no cover
            raise AssertionError(task)
        payload = {"choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}]}
        return 200, json.dumps(payload).encode("utf-8")


def make_config(tmp: Path, **overrides) -> dict:
    gen = {
        "taxonomy_path": str(TAXONOMY),
        "output_dir": str(tmp / "dialogue_v4"),
        "state_path": str(tmp / "state.json"),
        "probe_report_path": str(tmp / "no_report.json"),
        "target_examples": 120,
        "batch_size": 8,
        "max_calls": 200,
        "critic": True,
        "critic_min_score": 4,
    }
    gen.update(overrides)
    return {
        "dialogue_generation": gen,
        "supervised_generation": {
            "external_llm_enabled": True,
            "provider_order": ["openai", "openrouter"],
            "scheduler_state_path": str(tmp / "scheduler.sqlite3"),
            "provider_limits": {"daily_requests": 10000, "total_requests": 10000, "requests_per_minute": 100000,
                                "total_token_budget": 100000000, "max_concurrency": 1},
            "provider_retry_limit": 0,
            "provider_backoff_seconds": 0,
            "providers": [
                {"name": "openai", "api_key_env": "OPENAI_API_KEY", "default_model": "m1", "url_env": "TEST_NO_URL_OPENAI"},
                {"name": "openrouter", "api_key_env": "OPENROUTER_API_KEY", "default_model": "m2", "url_env": "TEST_NO_URL_OR"},
            ],
        },
    }


def read(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class TaxonomyTests(unittest.TestCase):
    def test_load_and_gold(self):
        tax = dt.load_taxonomy(TAXONOMY)
        self.assertEqual(len(tax.families), 20)
        gold = dt.gold_records(tax)
        splits = Counter(r["split"] for r in gold)
        self.assertEqual(splits, {"train": 77, "validation": 20, "test": 20})
        for r in gold:
            self.assertEqual(r["history"][0]["role"], "user")
            self.assertEqual(r["history"][-1]["role"], "user")
            self.assertTrue(r["target"])
            self.assertEqual(r["metadata"]["origin"], "curated_gold_v1")
        ids = [r["id"] for r in gold]
        self.assertEqual(len(ids), len(set(ids)))

    def test_gold_obeys_own_gate(self):
        tax = dt.load_taxonomy(TAXONOMY)
        gate = dt.DialogueGate(tax, 10000)
        problems = []
        for family in tax.families.values():
            for d in family.gold:
                try:
                    gate.validate_turns(family, d.turns, enforce_turn_range=False)
                except dt.DialogueRejected as exc:
                    problems.append((family.name, str(exc), d.turns[-1][1]))
        self.assertEqual(problems, [])

    def test_quotas(self):
        tax = dt.load_taxonomy(TAXONOMY)
        q = dt.family_quotas(tax, 1000)
        self.assertEqual(sum(q.values()), 1000)
        boosted = dt.family_quotas(tax, 1000, failure_rates={"greeting": 1.0})
        self.assertEqual(sum(boosted.values()), 1000)
        self.assertGreater(boosted["greeting"], q["greeting"] * 2)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tax = dt.load_taxonomy(TAXONOMY)
        self.family = self.tax.families["greeting"]

    def test_rejections(self):
        gate = dt.DialogueGate(self.tax, 1000)
        cases = {
            "generic_closer_or_meta": [("user", "hi there"), ("assistant", "Hello! Let me know if you need anything else.")],
            "markdown_or_url": [("user", "hi there"), ("assistant", "Hello friend, see https://example.com now")],
            "roles_not_alternating": [("user", "hi there"), ("user", "hello again")],
            "must_end_with_assistant": [("user", "hi there"), ("assistant", "Hello to you."), ("user", "hello again")],
            "reply_length": [("user", "hi there"), ("assistant", " ".join(["alpha"] * 100))],
        }
        for expected, turns in cases.items():
            with self.assertRaises(dt.DialogueRejected, msg=expected) as ctx:
                gate.accept_dialogue(self.family, turns)
            self.assertEqual(str(ctx.exception), expected)

    def test_duplicates_and_loop(self):
        gate = dt.DialogueGate(self.tax, 1000)
        gate.accept_dialogue(self.family, [("user", "hey"), ("assistant", "Hello, nice to see you today.")])
        with self.assertRaises(dt.DialogueRejected) as ctx:
            gate.accept_dialogue(self.family, [("user", "hello you"), ("assistant", "Hello, nice to see you today.")])
        self.assertEqual(str(ctx.exception), "duplicate_reply")
        with self.assertRaises(dt.DialogueRejected) as ctx:
            gate.accept_dialogue(self.family, [("user", "hey there"), ("assistant", "meta meta meta meta meta meta")])
        self.assertEqual(str(ctx.exception), "repetition_loop")

    def test_sentence_budget_blocks_recurring_filler(self):
        gate = dt.DialogueGate(self.tax, 100)  # cap = max(3, ceil(.01*100)) = 3
        ok = 0
        reasons = Counter()
        for i in range(12):
            turns = [("user", f"question about topic {i} alpha{i}"),
                     ("assistant", f"Answer number {i} with unique detail zeta{i}. Start with the part that matters.")]
            try:
                gate.accept_dialogue(self.family, turns)
                ok += 1
            except dt.DialogueRejected as exc:
                reasons[str(exc)] += 1
        self.assertEqual(ok, 3)
        self.assertEqual(reasons["recurring_filler_sentence"], 9)

    def test_final_turn_only_family(self):
        fam = self.tax.families["correction_feedback"]
        gate = dt.DialogueGate(self.tax, 1000)
        turns = [("user", "what is two plus two"), ("assistant", "It is five."),
                 ("user", "that is wrong, check again"), ("assistant", "You are right, two plus two is four.")]
        examples = gate.accept_dialogue(fam, turns)
        self.assertEqual(len(examples), 1)
        self.assertEqual(len(examples[0][0]), 3)

    def test_multi_turn_yields_one_example_per_reply(self):
        fam = self.tax.families["multi_turn_followup"]
        gate = dt.DialogueGate(self.tax, 1000)
        turns = [("user", "tell me a fruit"), ("assistant", "A mango is a good choice."),
                 ("user", "what colour is it"), ("assistant", "A ripe mango is orange and yellow.")]
        lo, hi = fam.user_turns
        if lo <= 2 <= hi:
            self.assertEqual(len(gate.accept_dialogue(fam, turns)), 2)


class BuildTests(unittest.TestCase):
    def test_end_to_end_with_critic(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fake = FakeTeacher()
            cfg = make_config(tmp)
            res = dt.build_dialogue_dataset(cfg, transport=fake)
            self.assertEqual(res.stopped, "quota_reached", res)
            out = Path(res.output_dir)
            gold = read(out / dt.GOLD_FILE)
            teacher = read(out / dt.TEACHER_FILE)
            self.assertEqual(len(gold), 117)
            self.assertGreaterEqual(len(teacher), 120)
            for r in teacher:
                m = r["metadata"]
                self.assertEqual(r["split"], "train")
                self.assertEqual(m["origin"], dt.ORIGIN_TEACHER)
                self.assertEqual(m["validation_method"], dt.METHOD_GENERATED)
                self.assertTrue(m["teacher_validated"])
                self.assertNotIn("BADFACT", r["target"])
                self.assertNotIn("Let me know", r["target"])
                self.assertNotIn("http", r["target"])
            self.assertGreater(res.rejected.get("generic_closer_or_meta", 0), 0)
            self.assertGreater(res.rejected.get("markdown_or_url", 0), 0)
            self.assertGreater(res.rejected.get("critic_rejected", 0), 0)
            self.assertGreater(fake.calls["score_dialogues"], 0)
            # critic must use a different provider than the generator
            self.assertEqual(set(fake.providers_seen), {"openai", "openrouter"})
            self.assertEqual(len({r["id"] for r in teacher}), len(teacher))
            manifest = json.loads(Path(res.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["accepted_total"], len(teacher))

    def test_resume_is_incremental(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fake = FakeTeacher()
            cfg = make_config(tmp)
            first = dt.build_dialogue_dataset(cfg, transport=fake, max_calls=6)
            self.assertEqual(first.stopped, "call_budget_reached")
            n1 = len(read(Path(first.output_dir) / dt.TEACHER_FILE))
            second = dt.build_dialogue_dataset(cfg, transport=fake)
            n2 = len(read(Path(second.output_dir) / dt.TEACHER_FILE))
            self.assertGreater(n2, n1)
            third = dt.build_dialogue_dataset(cfg, transport=fake)
            self.assertEqual(len(read(Path(third.output_dir) / dt.TEACHER_FILE)), n2)
            self.assertEqual(third.calls, 0)

    def test_statistical_filler_is_blocked_at_the_source(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fake = FakeTeacher(bad_every=0, filler="Start with the part that matters most to you.")
            res = dt.build_dialogue_dataset(make_config(tmp, critic=False, target_examples=100), transport=fake)
            teacher = read(Path(res.output_dir) / dt.TEACHER_FILE)
            self.assertGreater(res.rejected.get("recurring_filler_sentence", 0), 0)
            found = find_templated_sentences([r["target"] for r in teacher])
            self.assertEqual(found, {})

    def test_teacher_never_duplicates_or_leaks_into_gold(self):
        tax = dt.load_taxonomy(TAXONOMY)
        gold = dt.gold_records(tax)
        gate = dt.DialogueGate(tax, 1000)
        gate.reserve(gold)
        held_out = next(r for r in gold if r["split"] == "test")
        turns = [(t["role"], t["content"]) for t in held_out["history"]] + [("assistant", held_out["target"])]
        family = tax.families[held_out["metadata"]["family"]]
        with self.assertRaises(dt.DialogueRejected):
            gate.accept_dialogue(family, turns, enforce_turn_range=False)

    def test_no_providers(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = make_config(Path(d))
            cfg["supervised_generation"]["external_llm_enabled"] = False
            res = dt.build_dialogue_dataset(cfg)
            self.assertEqual(res.stopped, "no_providers_configured")
            self.assertEqual(res.gold_records, 117)

    def test_repair_loop(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fake = FakeTeacher()
            cfg = make_config(tmp)
            tax = dt.load_taxonomy(TAXONOMY)
            failures = tmp / "failures.jsonl"
            rows = []
            self.assertTrue(tax.probes("monitor") and tax.probes("tutor"))
            self.assertEqual(len(tax.probes("tutor")) + len(tax.probes("monitor")), len(tax.probes()))
            tutor_prompts, monitor_prompts, n = [], [], 0
            while len(tutor_prompts) < 10 or len(monitor_prompts) < 2:  # unique prompts, both roles
                n += 1
                prompt = f"repair test question number {n} about gardens"
                (tutor_prompts if dt.probe_role(prompt) == "tutor" else monitor_prompts).append(prompt)
            rows = []
            for fam, prompt in zip(list(tax.families) * 2, tutor_prompts[:10] + monitor_prompts[:2]):
                rows.append({"family": fam, "history": [{"role": "user", "content": prompt}],
                             "student_output": "meta meta meta meta meta", "reasons": ["repetition_loop"]})
            failures.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            res = dt.build_repair_dataset(cfg, failures, transport=fake)
            self.assertEqual(res.accepted, 10, res)  # the 2 monitor probes were not tutored
            repaired = read(Path(res.output_dir) / dt.REPAIR_FILE)
            self.assertEqual(repaired[0]["metadata"]["validation_method"], dt.METHOD_REPAIR)
            self.assertEqual(repaired[0]["split"], "train")
            again = dt.build_repair_dataset(cfg, failures, transport=fake)
            self.assertEqual(again.accepted, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
