"""Offline tests for the source-grounded teacher (src/training/supervised_llm_builder.py)."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

from collections import Counter
from pathlib import Path

try:
    import colorama

    if not hasattr(colorama, "just_fix_windows_console"):
        colorama.just_fix_windows_console = lambda *a, **k: None
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["OPENAI_API_KEY"] = "test-openai"
os.environ["OPENROUTER_API_KEY"] = "test-openrouter"
for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(name, None)

from src.training import supervised_llm_builder as sb  # noqa: E402
from src.training.enrichment_contracts import CurriculumQualityError, SourceDocument  # noqa: E402
from src.training.lantra_phase_scheduler import TEACHER_VALIDATION_METHODS  # noqa: E402

THEMES = [
    ("history", "war", "empire", "century", "treaty", "kingdom"),
    ("geography", "river", "region", "climate", "valley", "coast"),
    ("natural_science", "biology", "species", "cells", "ecology", "organism"),
    ("technology_engineering", "machine", "engineering", "software", "circuit", "engine"),
]


def make_document(index: int, split: str = "train") -> SourceDocument:
    label, a, b, c, d, e = THEMES[index % len(THEMES)]
    sentences = [
        f"The {a} shaped the early {label} of the northern {b} during the {c} in 18{index}0.",
        f"Many scholars argue that the {d} changed how the {e} was governed after {index + 3} long years.",
        f"A detailed survey of the {a} and the {b} recorded {index + 40} separate observations in {c} notes.",
        f"The {d} remained important because every {e} depended upon the {a} for supplies.",
        f"Later accounts of the {c} mention that the {b} was rebuilt twice before the {d} ended.",
        f"Records from the {e} describe how the {a} was measured by teams of {index + 6} trained observers.",
        f"By the end of the period the {b} and the {d} were studied across {index + 12} different schools.",
        f"The final report on the {c} concluded that the {a} had been misunderstood for generations.",
    ]
    text = " ".join(sentences)
    return SourceDocument(
        document_id=f"doc-{index}", source_path=f"/x/{index}.txt", source_type="txt",
        source_sha256=f"{index:064d}", normalized_text_sha256=f"{index:064d}", text=text,
        title=f"Document {index} on {label}", extractor="test", logical_index=index, split=split,
    )


def sentences_of(text: str):
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]


class FakeTeacher:
    def __init__(self, hallucinate: bool = True):
        self.calls = 0
        self.hallucinate = hallucinate

    def __call__(self, url, headers, body, timeout):
        self.calls += 1
        request = json.loads(body.decode("utf-8"))
        prompt = json.loads(request["messages"][1]["content"])
        source = prompt["source"]
        sents = sentences_of(source)
        out = []
        for task in prompt["tasks"]:
            for k in range(2):
                s = sents[(k * 2 + 1) % len(sents)]
                words = [w for w in re.findall(r"[a-z]{5,}", s.lower())]
                if task == "generation":
                    out.append({"task": task, "input": f"What does the passage say about item number {k}?", "target": s})
                elif task == "classification":
                    label = "history" if "war" in source or "empire" in source else "natural_science"
                    out.append({"task": task, "input": " ".join(sents[:3]), "label": label})
                elif task == "translation":
                    out.append({"task": task, "source": s, "target": "In plain words, " + s.replace("The ", "That ", 1)})
                elif task == "summarization":
                    out.append({"task": task, "target": " ".join(sents[:3])})
                elif task == "embedding":
                    out.append({"task": task, "anchor": f"Which passage discusses {words[1]} and {words[2]}?", "positive": s})
                elif task == "reranking":
                    out.append({"task": task, "query": f"Where is the {words[1]} described in the records?", "positive": s})
        if self.hallucinate:
            out.append({"task": "generation", "input": "What is the capital of Atlantis?", "target": "The capital of Atlantis is Poseidonis, founded by traders."})
            out.append({"task": "classification", "input": " ".join(sents[:3]), "label": "astrology"})
            out.append({"task": "dialogue", "input": "hi", "target": "hello there my friend"})
        payload = {"choices": [{"message": {"content": json.dumps({"examples": out})}, "finish_reason": "stop"}]}
        return 200, json.dumps(payload).encode("utf-8")


def config(tmp: Path, **extra) -> dict:
    cfg = {
        "output_dir": str(tmp / "teacher_v3"),
        "state_path": str(tmp / "state.json"),
        "manifest_path": str(tmp / "manifest.json"),
        "cache_path": str(tmp / "cache.sqlite3"),
        "scheduler_state_path": str(tmp / "scheduler.sqlite3"),
        "review_path": str(tmp / "review.jsonl"),
        "tasks": ["generation", "classification", "translation", "summarization", "embedding", "reranking"],
        "examples_per_task": 2,
        "minimum_train_examples_per_task": 6,
        "max_source_characters": 1800,
        "provider_order": ["openai", "openrouter"],
        "provider_limits": {"daily_requests": 10000, "total_requests": 10000, "requests_per_minute": 100000,
                            "total_token_budget": 100000000, "max_concurrency": 1},
        "provider_retry_limit": 0,
        "provider_backoff_seconds": 0,
        "providers": [
            {"name": "openai", "api_key_env": "OPENAI_API_KEY", "default_model": "m1", "url_env": "TEST_NO_URL_A"},
            {"name": "openrouter", "api_key_env": "OPENROUTER_API_KEY", "default_model": "m2", "url_env": "TEST_NO_URL_B"},
        ],
    }
    cfg.update(extra)
    return {"supervised_generation": cfg}


def read_all(directory: Path):
    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


class CertificationTests(unittest.TestCase):
    def setUp(self):
        self.doc = make_document(0)
        self.other = make_document(1)
        self.source = sb._excerpt(self.doc, 1800)
        self.negative_source = sb._excerpt(self.other, 1800)
        self.sents = sentences_of(self.source)

    def _meta(self):
        return {"teacher_provider": "openai", "source_document_ids": ["doc-0"]}

    def test_methods_are_known_to_the_scheduler(self):
        for method in (sb.METHOD_EXTRACT, sb.METHOD_LABEL, sb.METHOD_REWRITE, sb.METHOD_PAIR):
            self.assertIn(method, TEACHER_VALIDATION_METHODS)

    def test_generation_requires_verbatim_target(self):
        rec = {"task": "generation", "split": "train", "input": "What happened with item one?", "target": self.sents[1], "metadata": self._meta()}
        ok = sb.validate_record(rec, source_text=self.source)
        self.assertEqual(ok["metadata"]["validation_method"], sb.METHOD_EXTRACT)
        self.assertIn("source:\n", ok["input"])
        rec["target"] = "The capital of Atlantis is Poseidonis, founded by traders."
        with self.assertRaises(CurriculumQualityError):
            sb.validate_record(rec, source_text=self.source)

    def test_classification_closed_set(self):
        text = " ".join(self.sents[:3])
        rec = {"task": "classification", "split": "train", "input": text, "label": "history", "metadata": self._meta()}
        self.assertEqual(sb.validate_record(rec, source_text=self.source)["metadata"]["validation_method"], sb.METHOD_LABEL)
        for bad in ("astrology", "geography"):  # outside set / no lexical support
            with self.assertRaises(CurriculumQualityError):
                sb.validate_record({**rec, "label": bad}, source_text=self.source)

    def test_rewrite_checks(self):
        s = self.sents[0]
        base = {"task": "translation", "split": "train", "source": s, "metadata": self._meta(),
                "source_language": "English (source register)", "target_language": "English (plain register)"}
        good = sb.validate_record({**base, "target": "In plain words, " + s.replace("The ", "That ", 1)}, source_text=self.source)
        self.assertEqual(good["metadata"]["validation_method"], sb.METHOD_REWRITE)
        for bad in (s, "A rewrite that forgot everything important about it all.", s.replace("1800", "")):
            with self.assertRaises(CurriculumQualityError):
                sb.validate_record({**base, "target": bad}, source_text=self.source)

    def test_pair_requires_other_document_negative(self):
        positive = self.sents[1]
        negative = sb._hard_negative(self.negative_source, positive)
        rec = {"task": "embedding", "split": "train", "anchor": "Which passage discusses the treaty and kingdom?",
               "positive": positive, "negative": negative, "metadata": self._meta()}
        ok = sb.validate_record(rec, source_text=self.source, negative_source_text=self.negative_source)
        self.assertEqual(ok["metadata"]["validation_method"], sb.METHOD_PAIR)
        with self.assertRaises(CurriculumQualityError):  # negative from the same document
            sb.validate_record({**rec, "negative": self.sents[3]}, source_text=self.source, negative_source_text=self.negative_source)
        with self.assertRaises(CurriculumQualityError):  # query copied from source
            sb.validate_record({**rec, "anchor": self.sents[2]}, source_text=self.source, negative_source_text=self.negative_source)

    def test_dialogue_is_refused(self):
        rec = {"task": "dialogue", "split": "train", "input": "user: hi\nassistant:", "target": "hello there my friend again",
               "metadata": self._meta()}
        with self.assertRaises(CurriculumQualityError):
            sb.validate_record(rec, source_text=self.source)

    def test_deterministic_fallback_has_no_dialogue(self):
        records = sb.deterministic_examples(self.doc, negative_document=self.other, max_source_characters=1800)
        self.assertNotIn("dialogue", {r["task"] for r in records})


class BuildTests(unittest.TestCase):
    def test_build_certifies_every_source_task(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            docs = [make_document(i) for i in range(8)]
            # fake transport is injected through providers_from_config
            fake = FakeTeacher()
            real = sb.providers_from_config
            sb.providers_from_config = lambda cfg, **kw: real(cfg, transport=fake)
            try:
                result = sb.build_supervised_dataset(docs, pipeline_config=config(tmp))
            finally:
                sb.providers_from_config = real
            rows = read_all(Path(result.output_dir))
            by_task = Counter(r["task"] for r in rows)
            for task in sb.SOURCE_GROUNDED_TASKS:
                self.assertGreaterEqual(by_task[task], 6, (task, by_task, result.coverage))
            self.assertNotIn("dialogue", by_task)
            methods = {r["metadata"]["validation_method"] for r in rows}
            self.assertEqual(methods, {sb.METHOD_EXTRACT, sb.METHOD_LABEL, sb.METHOD_REWRITE, sb.METHOD_PAIR})
            self.assertTrue(all(r["split"] == "train" for r in rows))
            self.assertTrue(all(r["metadata"]["teacher_validated"] for r in rows))
            self.assertGreater(result.rejected, 0)
            # pair negatives always come from a different document
            for r in rows:
                if r["task"] in {"embedding", "reranking"}:
                    self.assertEqual(len(r["metadata"]["source_document_ids"]), 2)
            self.assertEqual(len({r["id"] for r in rows}), len(rows))

    def test_held_out_documents_get_held_out_records_first(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            docs = [make_document(i, "train") for i in range(6)] + [make_document(6 + i, "validation") for i in range(3)] \
                + [make_document(9 + i, "test") for i in range(3)]
            fake = FakeTeacher()
            real = sb.providers_from_config
            sb.providers_from_config = lambda cfg, **kw: real(cfg, transport=fake)
            try:
                result = sb.build_supervised_dataset(
                    docs, pipeline_config=config(tmp, minimum_validation_examples_per_task=2, minimum_test_examples_per_task=2))
            finally:
                sb.providers_from_config = real
            rows = read_all(Path(result.output_dir))
            seen = Counter((r["task"], r["split"]) for r in rows)
            for task in sb.SOURCE_GROUNDED_TASKS:
                self.assertGreaterEqual(seen[(task, "validation")], 2, (task, seen))
                self.assertGreaterEqual(seen[(task, "test")], 2, (task, seen))
            # a record's split is always its source document's split
            by_doc = {doc.document_id: doc.split for doc in docs}
            for r in rows:
                self.assertEqual(r["split"], by_doc[r["metadata"]["source_document_ids"][0]])

    def test_stale_state_does_not_hide_documents(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            docs = [make_document(i) for i in range(6)]
            cfg = config(tmp)
            sb.atomic_write_json(Path(cfg["supervised_generation"]["state_path"]), {
                "schema": "slai.lantra.supervised-build-state.v1",
                "processed_documents": [doc.document_id for doc in docs], "accepted_ids": [],
            })
            fake = FakeTeacher()
            real = sb.providers_from_config
            sb.providers_from_config = lambda c, **kw: real(c, transport=fake)
            try:
                result = sb.build_supervised_dataset(docs, pipeline_config=cfg)
            finally:
                sb.providers_from_config = real
            self.assertGreater(fake.calls, 0)
            self.assertGreater(sum(v["examples"] for v in result.coverage.values()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
