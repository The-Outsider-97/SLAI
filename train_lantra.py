#!/usr/bin/env python3
"""SLAI Language Transformer (LANTRA) multi-task trainer.

This is a real staged training entry point for SLAI's existing LanguageTransformer
and LanguageTokenizer. It does not define a parallel transformer implementation.

Training stages
---------------
0. BPE bootstrap      - load SLAI's configured BPE tokenizer/vocabulary
1. GloVe bootstrap    - initialize token embeddings and distill static semantics
2. Raw-text pretrain  - denoising encoder-decoder self-supervised pretraining
3. Supervised tune    - optional seven-task multi-task specialization

Supported objectives
--------------------
1. generation      - sequence-to-sequence conditional generation
2. classification  - text-to-text label generation
3. translation     - source-language to target-language generation
4. summarization   - document to summary generation
5. dialogue        - dialogue context to assistant response generation
6. embedding       - trainable encoder triplet objective
7. reranking       - query/document pairwise ranking objective

Dependency boundary
-------------------
This file imports only Python standard-library modules and SLAI modules.  It does
not directly import torch, numpy, pandas, sklearn, transformers, datasets, or any
other third-party library.  SLAI's LanguageTransformer/LanguageTokenizer already
use PyTorch (and the tokenizer currently uses ``regex``) internally; those are
therefore transitive SLAI runtime dependencies and cannot be removed here without
replacing SLAI's current language subsystem itself.

Default invocation
------------------
    py -m train_lantra

No subcommand is required.  By default the trainer discovers supervised data from:
    data/processed/lantra/
    data/raw/lantra/
    data/lantra_training.jsonl
    data/language_training.jsonl

and raw/document pretraining corpora from:
    data/library/                 (TXT/MD/HTML/DOCX/EPUB/PDF/JSON/JSONL)
    data/raw/lantra_corpus/
    data/raw/language_corpus/
    data/raw/language/
    data/lantra_corpus/
    data/language_corpus/

or from ``SLAI_LANTRA_DATA``. Supervised data is optional: when it is absent,
the trainer can still perform genuine GloVe semantic bootstrap training and, when
raw text is available, self-supervised denoising pretraining. No task-labelled
examples are fabricated.

JSONL task schema
-----------------
Each line must be a JSON object with ``task`` and the fields below.  ``id`` and
``split`` (train|validation|test) are optional.

Generation:
    {"task":"generation", "input":"prompt", "target":"completion"}
Classification:
    {"task":"classification", "input":"text", "label":"class_name"}
Translation:
    {"task":"translation", "source":"...", "target":"...",
     "source_language":"en", "target_language":"nl"}
Summarization:
    {"task":"summarization", "input":"document", "target":"summary"}
Dialogue:
    {"task":"dialogue",
     "history":[{"role":"user","content":"..."}],
     "target":"assistant reply"}
Embedding:
    {"task":"embedding", "anchor":"...", "positive":"...", "negative":"..."}
Reranking:
    {"task":"reranking", "query":"...", "positive":"...", "negative":"..."}

The model checkpoint is a native LanguageTransformer checkpoint produced by
``LanguageTransformer.save_language_model`` and can therefore be reloaded by
SLAI through ``LanguageTransformer.load_language_model``.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import importlib
import json
import math
import os
import platform
import posixpath
import random
import re
import sys
import time
import traceback
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from logs.logger import PrettyPrinter, configure_logging, get_logger
from src.agents.language.modules.language_tokenizer import LanguageTokenizer
from src.agents.language.modules.language_transformer import LanguageTransformer


LOGGER = get_logger("LANTRA Trainer")
PRINTER = PrettyPrinter

SUPPORTED_TASKS: Tuple[str, ...] = (
    "generation",
    "classification",
    "translation",
    "summarization",
    "dialogue",
    "embedding",
    "reranking",
)
SEQ2SEQ_TASKS = frozenset({
    "generation",
    "classification",
    "translation",
    "summarization",
    "dialogue",
})
PAIRWISE_TASKS = frozenset({"embedding", "reranking"})
VALID_SPLITS = frozenset({"train", "validation", "test"})
DEFAULT_DATA_CANDIDATES: Tuple[str, ...] = (
    "data/processed/lantra",
    "data/raw/lantra",
    "data/lantra_training.jsonl",
    "data/language_training.jsonl",
)
DEFAULT_OUTPUT_DIR = "src/agents/language/checkpoints/lantra"
DEFAULT_REPORT_DIR = "src/agents/language/artifacts/training/lantra"
DEFAULT_RAW_TEXT_CANDIDATES: Tuple[str, ...] = (
    "data/library",
    "data/raw/lantra_corpus",
    "data/raw/language_corpus",
    "data/raw/language",
    "data/lantra_corpus",
    "data/language_corpus",
)
DEFAULT_GLOVE_CANDIDATES: Tuple[str, ...] = (
    "data/embeddings/glove.6B.200d.json",
    "data/embeddings/glove.6B.100d.json",
    "data/embeddings/glove.6B.300d.json",
)
RAW_TEXT_EXTENSIONS = frozenset({
    ".txt", ".text", ".md", ".markdown",
    ".html", ".htm", ".xhtml",
    ".docx", ".epub", ".pdf",
    ".json", ".jsonl",
})

OBJECTIVE_CONTRACT: Dict[str, Any] = {
    "generation": {"mode": "seq2seq", "source_prefix": "task: generation\nprompt:", "target": "completion"},
    "classification": {"mode": "seq2seq", "source_prefix": "task: classification\ntext:", "target": "label text"},
    "translation": {"mode": "seq2seq", "source_prefix": "task: translation", "target": "translated text"},
    "summarization": {"mode": "seq2seq", "source_prefix": "task: summarization\ndocument:", "target": "summary"},
    "dialogue": {"mode": "seq2seq", "source_prefix": "task: dialogue\nconversation:", "target": "assistant response"},
    "embedding": {
        "mode": "triplet_cosine_margin",
        "text_prefix": "task: embedding\ntext:",
        "inference": "mean-pool encoder memory over non-padding tokens, then L2-normalize",
    },
    "reranking": {
        "mode": "pairwise_cosine_margin",
        "query_prefix": "task: reranking\nquery:",
        "document_prefix": "task: reranking\ndocument:",
        "inference": "rank candidate documents by cosine(query_embedding, document_embedding)",
    },
}


class LantraTrainingError(RuntimeError):
    """Raised for an actionable LANTRA training/configuration failure."""


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    task: str
    split: Optional[str]
    source: Optional[str] = None
    target: Optional[str] = None
    anchor: Optional[str] = None
    positive: Optional[str] = None
    negative: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: str = ""

    def canonical_payload(self) -> Dict[str, Any]:
        return {
            "id": self.example_id,
            "task": self.task,
            "split": self.split,
            "source": self.source,
            "target": self.target,
            "anchor": self.anchor,
            "positive": self.positive,
            "negative": self.negative,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DatasetSplit:
    train: Tuple[TrainingExample, ...]
    validation: Tuple[TrainingExample, ...]
    test: Tuple[TrainingExample, ...]
    files: Tuple[str, ...]
    fingerprint: str
    duplicate_count: int

    def counts(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for name, records in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            by_task = collections.Counter(item.task for item in records)
            payload[name] = {
                "total": len(records),
                "tasks": {task: int(by_task.get(task, 0)) for task in SUPPORTED_TASKS},
            }
        return payload


@dataclass(frozen=True)
class TrainerConfig:
    data_paths: Tuple[str, ...]
    raw_text_paths: Tuple[str, ...]
    glove_path: Optional[str]
    output_dir: str
    report_dir: str
    epochs: int
    gradient_accumulation: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    clip_grad: float
    label_smoothing: float
    source_max_length: int
    target_max_length: int
    embedding_max_length: int
    raw_max_length: int
    validation_fraction: float
    test_fraction: float
    patience: int
    min_delta: float
    seed: int
    device: str
    embedding_margin: float
    reranking_margin: float
    max_eval_records: int
    generation_eval_records: int
    log_every: int
    min_task_samples: int
    balance_tasks: bool
    retrain_tokenizer: bool
    tokenizer_vocab_size: int
    tokenizer_min_frequency: int
    init_from: Optional[str]
    require_all_supervised_tasks: bool
    glove_bootstrap: bool
    glove_epochs: int
    glove_batch_size: int
    glove_max_tokens: int
    glove_learning_rate: float
    raw_pretrain_epochs: int
    raw_segments_per_epoch: int
    raw_max_segments: int
    raw_validation_fraction: float
    raw_test_fraction: float
    raw_corruption_probability: float
    raw_min_chars: int
    raw_chunk_chars: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class EpochStats:
    weighted_loss_sum: float = 0.0
    raw_loss_sum: float = 0.0
    examples: int = 0
    optimizer_steps: int = 0
    task_loss_sum: Dict[str, float] = field(default_factory=lambda: collections.defaultdict(float))
    task_count: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    pairwise_correct: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    pairwise_total: Dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    grad_norm_sum: float = 0.0

    def add_loss(self, task: str, raw_loss: float, weighted_loss: float) -> None:
        self.raw_loss_sum += raw_loss
        self.weighted_loss_sum += weighted_loss
        self.examples += 1
        self.task_loss_sum[task] += raw_loss
        self.task_count[task] += 1

    def add_pairwise(self, task: str, correct: bool) -> None:
        self.pairwise_total[task] += 1
        if correct:
            self.pairwise_correct[task] += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "examples": self.examples,
            "optimizer_steps": self.optimizer_steps,
            "mean_raw_loss": self.raw_loss_sum / max(1, self.examples),
            "mean_weighted_loss": self.weighted_loss_sum / max(1, self.examples),
            "mean_gradient_norm": self.grad_norm_sum / max(1, self.optimizer_steps),
            "tasks": {
                task: {
                    "examples": int(self.task_count.get(task, 0)),
                    "mean_loss": (
                        self.task_loss_sum.get(task, 0.0) / max(1, self.task_count.get(task, 0))
                    ),
                    "pairwise_accuracy": (
                        self.pairwise_correct.get(task, 0) / max(1, self.pairwise_total.get(task, 0))
                        if self.pairwise_total.get(task, 0)
                        else None
                    ),
                }
                for task in SUPPORTED_TASKS
            },
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_text(value: Any, field_name: str, *, max_chars: int = 500_000) -> str:
    if not isinstance(value, str):
        raise LantraTrainingError(f"Field '{field_name}' must be a string, got {type(value).__name__}.")
    text = value.strip()
    if not text:
        raise LantraTrainingError(f"Field '{field_name}' cannot be empty.")
    if len(text) > max_chars:
        raise LantraTrainingError(
            f"Field '{field_name}' contains {len(text)} characters, exceeding the safety limit {max_chars}."
        )
    return text


def first_text(record: Mapping[str, Any], names: Sequence[str], field_name: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return require_text(value, name)
    raise LantraTrainingError(
        f"Missing required text for '{field_name}'. Accepted keys: {', '.join(names)}."
    )


def normalize_task(value: Any) -> str:
    task = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "classify": "classification",
        "classifier": "classification",
        "translate": "translation",
        "summary": "summarization",
        "summarize": "summarization",
        "chat": "dialogue",
        "conversation": "dialogue",
        "embeddings": "embedding",
        "embed": "embedding",
        "rerank": "reranking",
        "ranking": "reranking",
    }
    task = aliases.get(task, task)
    if task not in SUPPORTED_TASKS:
        raise LantraTrainingError(
            f"Unsupported task {value!r}. Supported tasks: {', '.join(SUPPORTED_TASKS)}."
        )
    return task


def normalize_split(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    split = str(value).strip().lower()
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    split = aliases.get(split, split)
    if split not in VALID_SPLITS:
        raise LantraTrainingError(
            f"Unsupported split {value!r}; expected train, validation, or test."
        )
    return split


def format_dialogue_history(value: Any) -> str:
    if isinstance(value, str):
        return require_text(value, "history")
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise LantraTrainingError("Dialogue 'history' must be a string or a sequence of role/content objects.")
    turns: List[str] = []
    for index, turn in enumerate(value):
        if not isinstance(turn, Mapping):
            raise LantraTrainingError(f"Dialogue history item {index} must be an object.")
        role = str(turn.get("role", "unknown")).strip().lower() or "unknown"
        content = first_text(turn, ("content", "text", "message"), f"history[{index}].content")
        turns.append(f"{role}: {content}")
    if not turns:
        raise LantraTrainingError("Dialogue history cannot be empty.")
    return "\n".join(turns)


def normalize_record(record: Mapping[str, Any], provenance: str, line_number: int) -> TrainingExample:
    task = normalize_task(record.get("task"))
    split = normalize_split(record.get("split"))
    metadata = record.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise LantraTrainingError(f"metadata must be an object at {provenance}:{line_number}.")
    normalized_metadata: Dict[str, Any] = dict(metadata)

    source: Optional[str] = None
    target: Optional[str] = None
    anchor: Optional[str] = None
    positive: Optional[str] = None
    negative: Optional[str] = None

    if task == "generation":
        source = first_text(record, ("input", "prompt", "source", "text"), "generation input")
        target = first_text(record, ("target", "output", "completion", "response"), "generation target")
    elif task == "classification":
        source = first_text(record, ("input", "text", "source"), "classification input")
        target = first_text(record, ("label", "target", "class"), "classification label")
    elif task == "translation":
        source = first_text(record, ("source", "input", "text"), "translation source")
        target = first_text(record, ("target", "translation", "output"), "translation target")
        for key in ("source_language", "target_language"):
            if record.get(key) not in (None, ""):
                normalized_metadata[key] = str(record[key]).strip()
    elif task == "summarization":
        source = first_text(record, ("input", "document", "text", "source"), "summarization document")
        target = first_text(record, ("target", "summary", "output"), "summary target")
    elif task == "dialogue":
        if "history" in record:
            source = format_dialogue_history(record["history"])
        else:
            source = first_text(record, ("input", "context", "source", "text"), "dialogue context")
        target = first_text(record, ("target", "response", "output", "reply"), "dialogue response")
    elif task == "embedding":
        anchor = first_text(record, ("anchor", "query", "input"), "embedding anchor")
        positive = first_text(record, ("positive", "positive_text", "match"), "embedding positive")
        negative = first_text(record, ("negative", "negative_text", "non_match"), "embedding negative")
    elif task == "reranking":
        anchor = first_text(record, ("query", "anchor", "input"), "reranking query")
        positive = first_text(record, ("positive", "positive_document", "relevant"), "reranking positive document")
        negative = first_text(record, ("negative", "negative_document", "irrelevant"), "reranking negative document")

    id_value = record.get("id")
    if id_value in (None, ""):
        id_basis = {
            "task": task,
            "source": source,
            "target": target,
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
        }
        example_id = sha256_payload(id_basis)[:24]
    else:
        example_id = str(id_value).strip()
        if not example_id:
            raise LantraTrainingError(f"Record id cannot be blank at {provenance}:{line_number}.")

    return TrainingExample(
        example_id=example_id,
        task=task,
        split=split,
        source=source,
        target=target,
        anchor=anchor,
        positive=positive,
        negative=negative,
        metadata=normalized_metadata,
        provenance=f"{provenance}:{line_number}",
    )


def iter_json_records(path: Path) -> Iterator[Tuple[Mapping[str, Any], int]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise LantraTrainingError(
                        f"Invalid JSON at {path}:{line_number}: {exc.msg}."
                    ) from exc
                if not isinstance(item, Mapping):
                    raise LantraTrainingError(f"JSONL record at {path}:{line_number} must be an object.")
                yield item, line_number
        return

    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise LantraTrainingError(f"Invalid JSON in {path}: {exc.msg}.") from exc
        if isinstance(payload, Mapping):
            records = payload.get("records", payload.get("examples", payload.get("data")))
            if records is None:
                records = [payload]
        else:
            records = payload
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
            raise LantraTrainingError(f"JSON file {path} must contain a record array or a records/examples/data array.")
        for index, item in enumerate(records, 1):
            if not isinstance(item, Mapping):
                raise LantraTrainingError(f"Record {index} in {path} must be an object.")
            yield item, index
        return

    raise LantraTrainingError(f"Unsupported training data extension: {path}")


def discover_data_files(configured_paths: Sequence[str]) -> List[Path]:
    """Discover optional supervised JSON/JSONL data.

    Absence is not an error because LANTRA can bootstrap from GloVe and/or raw
    text before supervised specialization is available.
    """
    paths: List[Path] = []
    if configured_paths:
        candidates = [Path(value) for value in configured_paths]
    else:
        env_path = os.getenv("SLAI_LANTRA_DATA", "").strip()
        candidates = [Path(env_path)] if env_path else [Path(value) for value in DEFAULT_DATA_CANDIDATES]

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.is_file():
            discovered = [candidate] if candidate.suffix.lower() in {".json", ".jsonl"} else []
        else:
            discovered = sorted(
                item for item in candidate.rglob("*")
                if item.is_file() and item.suffix.lower() in {".json", ".jsonl"}
            )
        for item in discovered:
            resolved = str(item.resolve())
            if resolved not in seen:
                seen.add(resolved)
                paths.append(item)
    return sorted(paths, key=lambda item: str(item))


def load_examples(paths: Sequence[Path]) -> Tuple[List[TrainingExample], int, str]:
    examples: List[TrainingExample] = []
    seen_payloads: Dict[str, TrainingExample] = {}
    duplicate_count = 0
    file_fingerprints: List[Dict[str, str]] = []

    for path in paths:
        file_fingerprints.append({"path": str(path), "sha256": sha256_file(path)})
        for raw_record, line_number in iter_json_records(path):
            try:
                example = normalize_record(raw_record, str(path), line_number)
            except LantraTrainingError as exc:
                raise LantraTrainingError(f"{exc} [source={path}:{line_number}]") from exc
            dedupe_key = sha256_payload({
                "task": example.task,
                "source": example.source,
                "target": example.target,
                "anchor": example.anchor,
                "positive": example.positive,
                "negative": example.negative,
            })
            previous = seen_payloads.get(dedupe_key)
            if previous is not None:
                if previous.split and example.split and previous.split != example.split:
                    raise LantraTrainingError(
                        "Exact duplicate content occurs in conflicting explicit splits: "
                        f"{previous.provenance} ({previous.split}) vs {example.provenance} ({example.split})."
                    )
                # If exactly one duplicate has an explicit split annotation, keep
                # that record so an unlabelled duplicate cannot silently override
                # the user's train/validation/test boundary by source-file order.
                if previous.split is None and example.split is not None:
                    replacement_index = examples.index(previous)
                    examples[replacement_index] = example
                    seen_payloads[dedupe_key] = example
                duplicate_count += 1
                continue
            seen_payloads[dedupe_key] = example
            examples.append(example)

    if not examples:
        raise LantraTrainingError("Training data files were found, but they contain no usable examples.")

    fingerprint = sha256_payload({
        "files": file_fingerprints,
        "records": [item.canonical_payload() for item in examples],
    })
    return examples, duplicate_count, fingerprint


def stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64 - 1)


def split_examples(
    examples: Sequence[TrainingExample],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    files: Sequence[Path],
    fingerprint: str,
    duplicate_count: int,
) -> DatasetSplit:
    if not (0.0 < validation_fraction < 0.5):
        raise LantraTrainingError("validation_fraction must be > 0 and < 0.5.")
    if not (0.0 < test_fraction < 0.5):
        raise LantraTrainingError("test_fraction must be > 0 and < 0.5.")
    if validation_fraction + test_fraction >= 0.5:
        raise LantraTrainingError("validation_fraction + test_fraction must be < 0.5.")

    train: List[TrainingExample] = []
    validation: List[TrainingExample] = []
    test: List[TrainingExample] = []
    implicit_by_task: Dict[str, List[TrainingExample]] = collections.defaultdict(list)

    for example in examples:
        if example.split == "train":
            train.append(example)
        elif example.split == "validation":
            validation.append(example)
        elif example.split == "test":
            test.append(example)
        else:
            implicit_by_task[example.task].append(example)

    # Stratify implicit records by task and guarantee held-out records whenever
    # a task has enough examples to support a meaningful three-way split.
    for task in SUPPORTED_TASKS:
        records = implicit_by_task.get(task, [])
        ordered = sorted(records, key=lambda item: stable_unit_interval(item.example_id, seed))
        n = len(ordered)
        if n == 0:
            continue
        n_test = max(1, int(round(n * test_fraction))) if n >= 5 else 0
        n_val = max(1, int(round(n * validation_fraction))) if n >= 4 else 0
        while n - n_test - n_val < 2 and (n_test > 0 or n_val > 0):
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
        test.extend(ordered[:n_test])
        validation.extend(ordered[n_test:n_test + n_val])
        train.extend(ordered[n_test + n_val:])

    train = sorted(train, key=lambda item: item.example_id)
    validation = sorted(validation, key=lambda item: item.example_id)
    test = sorted(test, key=lambda item: item.example_id)

    if not train:
        raise LantraTrainingError("No training records remain after split construction.")
    if not validation:
        raise LantraTrainingError(
            "No validation records are available. Provide explicit validation data or enough records for a split."
        )
    if not test:
        raise LantraTrainingError(
            "No test records are available. Provide explicit test data or enough records for a split."
        )

    return DatasetSplit(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        files=tuple(str(path) for path in files),
        fingerprint=fingerprint,
        duplicate_count=duplicate_count,
    )


def validate_task_coverage(
    dataset: DatasetSplit,
    min_task_samples: int,
    *,
    require_all_tasks: bool = False,
) -> Dict[str, Any]:
    """Validate supervised coverage without inventing missing task data.

    By default, any subset of the seven tasks may be used for specialization, but
    each task that is trained must also be represented in validation and test.
    ``require_all_tasks=True`` restores strict seven-task coverage.
    """
    split_records = {
        "train": dataset.train,
        "validation": dataset.validation,
        "test": dataset.test,
    }
    split_counts = {
        name: collections.Counter(item.task for item in records)
        for name, records in split_records.items()
    }
    train_counts = split_counts["train"]
    active_tasks = [task for task in SUPPORTED_TASKS if train_counts.get(task, 0) > 0]
    if not active_tasks:
        raise LantraTrainingError("Supervised files were found but no supported task has training examples.")

    if require_all_tasks:
        missing = [task for task in SUPPORTED_TASKS if task not in active_tasks]
        if missing:
            raise LantraTrainingError(
                "--require-all-supervised-tasks is enabled, but training data is missing: "
                + ", ".join(missing)
            )

    heldout_missing: Dict[str, List[str]] = {}
    for split_name in ("validation", "test"):
        missing = [task for task in active_tasks if split_counts[split_name].get(task, 0) <= 0]
        if missing:
            heldout_missing[split_name] = missing
    if heldout_missing:
        details = "; ".join(f"{name}: {', '.join(tasks)}" for name, tasks in heldout_missing.items())
        raise LantraTrainingError(
            "Every supervised task that is trained must also have held-out validation and test examples. "
            + details
        )

    undersized = {
        task: int(train_counts.get(task, 0))
        for task in active_tasks
        if train_counts.get(task, 0) < min_task_samples
    }
    if undersized:
        detail = ", ".join(f"{task}={count}" for task, count in undersized.items())
        raise LantraTrainingError(
            f"Insufficient real supervised examples ({detail}). Minimum per active task is "
            f"{min_task_samples}. Reduce --min-task-samples only for an intentional pilot run."
        )

    return {
        "active_tasks": active_tasks,
        "missing_tasks": [task for task in SUPPORTED_TASKS if task not in active_tasks],
        "counts": {name: dict(counts) for name, counts in split_counts.items()},
    }


def tokenizer_corpus(examples: Sequence[TrainingExample]) -> List[str]:
    corpus: List[str] = []
    for item in examples:
        for value in (item.source, item.target, item.anchor, item.positive, item.negative):
            if value:
                corpus.append(value)
    return corpus


def initialize_tokenizer(
    config: TrainerConfig,
    dataset: Optional[DatasetSplit],
    raw_segments: Sequence[str],
) -> LanguageTokenizer:
    tokenizer = LanguageTokenizer()
    if config.retrain_tokenizer:
        corpus: List[str] = []
        if dataset is not None:
            # Validation/test remain excluded from tokenizer fitting.
            corpus.extend(tokenizer_corpus(dataset.train))
        corpus.extend(raw_segments)
        if not corpus:
            raise LantraTrainingError(
                "Tokenizer retraining requested but neither supervised training text nor raw text is available."
            )
        LOGGER.info("Retraining SLAI LanguageTokenizer on %d text segments", len(corpus))
        summary = tokenizer.train(
            corpus,
            min_freq=config.tokenizer_min_frequency,
            max_vocab_size=config.tokenizer_vocab_size,
            reset_existing=True,
        )
        artifact_dir = Path(config.report_dir) / "tokenizer"
        saved = tokenizer.save(artifact_dir, name="lantra_tokenizer")
        LOGGER.info("Tokenizer trained and saved: %s", saved)
        PRINTER.pretty(
            "TOKENIZER TRAINING",
            summary.to_dict() if hasattr(summary, "to_dict") else summary,
            "success",
        )

    if not getattr(tokenizer, "is_trained", False):
        raise LantraTrainingError(
            "SLAI LanguageTokenizer is not trained and no usable configured BPE resources were loaded. "
            "Fix language_config.yaml BPE paths or use --retrain-tokenizer with real text."
        )
    vocab_size = len(getattr(tokenizer, "vocab", {}))
    if vocab_size < 8:
        raise LantraTrainingError(f"Tokenizer vocabulary is unexpectedly small: {vocab_size}.")
    return tokenizer


def torch_runtime() -> Any:
    """Return PyTorch already loaded transitively by SLAI, without importing it here."""
    runtime = sys.modules.get("torch")
    if runtime is None:
        raise LantraTrainingError(
            "SLAI LanguageTransformer did not load its PyTorch runtime. "
            "Install SLAI's declared runtime requirements before training."
        )
    return runtime


def seed_runtime(seed: int) -> None:
    random.seed(seed)
    runtime = torch_runtime()
    runtime.manual_seed(seed)
    if getattr(runtime, "cuda", None) is not None and runtime.cuda.is_available():
        runtime.cuda.manual_seed_all(seed)
    try:
        runtime.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            runtime.use_deterministic_algorithms(True)
        except Exception as exc:
            LOGGER.warning("Could not enable deterministic PyTorch algorithms: %s", exc)
    except Exception as exc:
        LOGGER.warning("Could not enable deterministic PyTorch algorithms: %s", exc)
    try:
        cudnn = getattr(getattr(runtime, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.benchmark = False
            cudnn.deterministic = True
    except Exception:
        pass


def resolve_device(requested: str) -> str:
    value = requested.strip().lower()
    runtime = torch_runtime()
    if value != "auto":
        return value
    try:
        if runtime.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        mps = getattr(getattr(runtime, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def initialize_model(config: TrainerConfig, tokenizer: LanguageTokenizer, device: str) -> LanguageTransformer:
    vocab_size = len(tokenizer.vocab)
    overrides = {
        "src_vocab_size": vocab_size,
        "tgt_vocab_size": vocab_size,
        "pad_token_id": int(tokenizer.pad_token_id),
        "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "clip_grad": config.clip_grad,
        "label_smoothing": config.label_smoothing,
        "batch_first": True,
    }

    if config.init_from:
        checkpoint = Path(config.init_from)
        if not checkpoint.is_file():
            raise LantraTrainingError(f"--init-from checkpoint does not exist: {checkpoint}")
        LOGGER.info("Warm-starting LanguageTransformer weights from %s", checkpoint)
        model = LanguageTransformer.load_language_model(
            checkpoint,
            device=device,
            strict=True,
            overrides=overrides,
        )
    else:
        model = LanguageTransformer(**overrides)
        model = model.to(device)

    if not bool(model.config.batch_first):
        raise LantraTrainingError(
            "LANTRA trainer requires batch_first=True because LanguageTransformer.language_forward currently "
            "constructs its automatic causal mask using tgt.size(1)."
        )
    configured_capacity = int(model.config.max_position_embeddings)
    requested_capacity = max(
        config.source_max_length,
        config.target_max_length,
        config.embedding_max_length,
        config.raw_max_length,
    )
    if requested_capacity > configured_capacity:
        raise LantraTrainingError(
            "Requested sequence length exceeds SLAI LanguageTransformer positional capacity: "
            f"requested={requested_capacity}, configured={configured_capacity}. "
            "Reduce --source-max-length/--target-max-length/--embedding-max-length or increase "
            "language_transformer.base_overrides.max_position_embeddings in language_config.yaml."
        )
    if int(model.config.src_vocab_size) != vocab_size or int(model.config.tgt_vocab_size) != vocab_size:
        raise LantraTrainingError(
            "Tokenizer/model vocabulary mismatch: "
            f"tokenizer={vocab_size}, src={model.config.src_vocab_size}, tgt={model.config.tgt_vocab_size}."
        )
    return model



@dataclass(frozen=True)
class RawDocumentProvenance:
    document_id: str
    source_path: str
    source_type: str
    source_sha256: str
    content_sha256: str
    normalized_text_sha256: str
    title: Optional[str]
    extractor: str
    character_count: int
    split: str
    segment_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "source_sha256": self.source_sha256,
            "content_sha256": self.content_sha256,
            "normalized_text_sha256": self.normalized_text_sha256,
            "title": self.title,
            "extractor": self.extractor,
            "character_count": self.character_count,
            "split": self.split,
            "segment_count": self.segment_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RawExtractionFailure:
    source_path: str
    source_type: str
    error_type: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RawCorpus:
    train_segments: Tuple[str, ...]
    validation_segments: Tuple[str, ...]
    test_segments: Tuple[str, ...]
    files: Tuple[str, ...]
    fingerprint: str
    documents: Tuple[RawDocumentProvenance, ...] = ()
    duplicate_files_exact: int = 0
    duplicate_documents_exact: int = 0
    duplicate_documents_normalized: int = 0
    duplicate_segments: int = 0
    extraction_failures: Tuple[RawExtractionFailure, ...] = ()
    manifest_path: Optional[str] = None

    @property
    def segments(self) -> Tuple[str, ...]:
        return self.train_segments + self.validation_segments + self.test_segments

    def to_dict(self) -> Dict[str, Any]:
        by_type = collections.Counter(document.source_type for document in self.documents)
        by_split = collections.Counter(document.split for document in self.documents)
        return {
            "documents": len(self.documents),
            "documents_by_type": dict(sorted(by_type.items())),
            "documents_by_split": {
                split: int(by_split.get(split, 0)) for split in ("train", "validation", "test")
            },
            "segments": {
                "total": len(self.segments),
                "train": len(self.train_segments),
                "validation": len(self.validation_segments),
                "test": len(self.test_segments),
            },
            "files": list(self.files),
            "fingerprint_sha256": self.fingerprint,
            "deduplication": {
                "duplicate_files_exact": self.duplicate_files_exact,
                "duplicate_documents_exact": self.duplicate_documents_exact,
                "duplicate_documents_normalized": self.duplicate_documents_normalized,
                "duplicate_segments": self.duplicate_segments,
            },
            "extraction_failures": [failure.to_dict() for failure in self.extraction_failures],
            "provenance_manifest": self.manifest_path,
        }


@dataclass(frozen=True)
class _ExtractedRawDocument:
    source_path: str
    source_type: str
    source_sha256: str
    text: str
    title: Optional[str]
    extractor: str
    logical_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedRawDocument:
    document_id: str
    source_path: str
    source_type: str
    source_sha256: str
    text: str
    title: Optional[str]
    extractor: str
    content_sha256: str
    normalized_text_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SegmentCandidate:
    text: str
    normalized_sha256: str
    document_id: str
    split: str
    segment_index: int


@dataclass(frozen=True)
class GloveMatch:
    token_id: int
    token: str
    glove_key: str
    vector: Tuple[float, ...]


@dataclass(frozen=True)
class GloveAsset:
    path: str
    dimension: int
    matches: Tuple[GloveMatch, ...]
    file_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "dimension": self.dimension,
            "matched_tokens": len(self.matches),
            "file_sha256": self.file_sha256,
        }


def empty_raw_corpus() -> RawCorpus:
    return RawCorpus(
        train_segments=(),
        validation_segments=(),
        test_segments=(),
        files=(),
        fingerprint=sha256_payload({"files": [], "documents": [], "segments": []}),
    )


def discover_raw_text_files(configured_paths: Sequence[str]) -> List[Path]:
    # data/library is always a first-class local corpus source. Explicit --raw-text
    # paths and SLAI_LANTRA_RAW_TEXT add sources; they do not suppress the library.
    candidates: List[Path] = [Path("data/library")]
    if configured_paths:
        candidates.extend(Path(value) for value in configured_paths)
    else:
        env_path = os.getenv("SLAI_LANTRA_RAW_TEXT", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        else:
            candidates.extend(Path(value) for value in DEFAULT_RAW_TEXT_CANDIDATES if value != "data/library")

    files: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        discovered = [candidate] if candidate.is_file() else sorted(candidate.rglob("*"))
        for item in discovered:
            if not item.is_file() or item.suffix.lower() not in RAW_TEXT_EXTENSIONS:
                continue
            key = str(item.resolve())
            if key not in seen:
                seen.add(key)
                files.append(item)
    return sorted(files, key=lambda item: str(item))


def _decode_document_bytes(payload: bytes) -> str:
    """Decode ordinary text files without introducing a charset dependency."""
    if not payload:
        return ""
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252"):
        try:
            return payload.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return payload.decode("utf-8", errors="replace")


def _normalize_document_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    compact: List[str] = []
    blank = False
    for line in lines:
        if line:
            compact.append(line)
            blank = False
        elif compact and not blank:
            compact.append("")
            blank = True
    return "\n".join(compact).strip()


def _normalized_text_hash(text: str) -> str:
    canonical = " ".join(_normalize_document_text(text).split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _clean_markdown_text(text: str) -> str:
    # Preserve the actual prose/code while removing high-frequency presentation
    # syntax that otherwise becomes corpus noise.
    text = re.sub(r"(?ms)^---\s*$.*?^---\s*$", " ", text, count=1)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
    text = text.replace("```", "").replace("~~~", "")
    return _normalize_document_text(text)


class _VisibleHTMLTextExtractor(HTMLParser):
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "canvas", "template"})
    _BLOCK_TAGS = frozenset({
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table",
        "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if not self._skip_depth and tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = data.strip()
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        self.parts.append(value)
        self.parts.append(" ")

    def result(self) -> Tuple[str, Optional[str]]:
        text = _normalize_document_text("".join(self.parts))
        title = " ".join(self.title_parts).strip() or None
        return text, title


def _extract_html_text(payload: str) -> Tuple[str, Optional[str]]:
    parser = _VisibleHTMLTextExtractor()
    parser.feed(payload)
    parser.close()
    return parser.result()


def _extract_docx(path: Path) -> Tuple[str, Optional[str], Dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        try:
            root = ET.fromstring(archive.read("word/document.xml"))
        except KeyError as exc:
            raise LantraTrainingError(f"DOCX is missing word/document.xml: {path}") from exc
        paragraphs: List[str] = []
        for paragraph in root.iter():
            if not paragraph.tag.endswith("}p"):
                continue
            text_nodes = [node.text or "" for node in paragraph.iter() if node.tag.endswith("}t")]
            paragraph_text = "".join(text_nodes).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)

        title: Optional[str] = None
        try:
            core_root = ET.fromstring(archive.read("docProps/core.xml"))
            for node in core_root.iter():
                if node.tag.endswith("}title") and (node.text or "").strip():
                    title = (node.text or "").strip()
                    break
        except (KeyError, ET.ParseError):
            pass

    return _normalize_document_text("\n\n".join(paragraphs)), title, {"paragraphs": len(paragraphs)}


def _safe_epub_member(base_dir: str, href: str) -> str:
    member = posixpath.normpath(posixpath.join(base_dir, href.split("#", 1)[0]))
    if member.startswith("../") or member.startswith("/"):
        raise LantraTrainingError(f"Unsafe EPUB member path: {href!r}")
    return member


def _extract_epub(path: Path) -> Tuple[str, Optional[str], Dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        opf_path: Optional[str] = None
        if "META-INF/container.xml" in names:
            container_root = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container_root.find(".//{*}rootfile")
            if rootfile is not None:
                opf_path = rootfile.attrib.get("full-path")

        ordered_members: List[str] = []
        title: Optional[str] = None
        if opf_path and opf_path in names:
            package_root = ET.fromstring(archive.read(opf_path))
            base_dir = posixpath.dirname(opf_path)
            manifest: Dict[str, Tuple[str, str, str]] = {}
            for item in package_root.findall(".//{*}manifest/{*}item"):
                item_id = item.attrib.get("id", "")
                href = item.attrib.get("href", "")
                media_type = item.attrib.get("media-type", "")
                properties = item.attrib.get("properties", "")
                if item_id and href:
                    manifest[item_id] = (href, media_type, properties)
            for itemref in package_root.findall(".//{*}spine/{*}itemref"):
                item_id = itemref.attrib.get("idref", "")
                if item_id not in manifest:
                    continue
                href, media_type, properties = manifest[item_id]
                if "nav" in properties.split():
                    continue
                if media_type in {"application/xhtml+xml", "text/html"} or href.lower().endswith((".xhtml", ".html", ".htm")):
                    member = _safe_epub_member(base_dir, href)
                    if member in names:
                        ordered_members.append(member)
            title_node = package_root.find(".//{http://purl.org/dc/elements/1.1/}title")
            if title_node is not None and (title_node.text or "").strip():
                title = (title_node.text or "").strip()

        if not ordered_members:
            ordered_members = sorted(
                name for name in names if name.lower().endswith((".xhtml", ".html", ".htm"))
            )

        chapters: List[str] = []
        seen_members: set[str] = set()
        for member in ordered_members:
            if member in seen_members:
                continue
            seen_members.add(member)
            text, chapter_title = _extract_html_text(_decode_document_bytes(archive.read(member)))
            if text:
                if chapter_title and not title:
                    title = chapter_title
                chapters.append(text)

    return _normalize_document_text("\n\n".join(chapters)), title, {"chapters": len(chapters)}


def _extract_pdf(path: Path) -> Tuple[str, Optional[str], Dict[str, Any]]:
    try:
        pypdf = importlib.import_module("pypdf")
    except ImportError as exc:
        raise LantraTrainingError(
            "PDF corpus ingestion requires pypdf. SLAI's root requirements.txt already declares pypdf; "
            "install the project requirements in the active virtual environment."
        ) from exc

    try:
        reader = pypdf.PdfReader(str(path), strict=False)
    except Exception as exc:
        raise LantraTrainingError(f"Unable to open PDF {path}: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise LantraTrainingError(f"Encrypted PDF cannot be decrypted without a password: {path}") from exc
        if not decrypted:
            raise LantraTrainingError(f"Encrypted PDF requires a password and cannot be used for training: {path}")

    pages: List[str] = []
    failed_pages = 0
    for page_number, page in enumerate(reader.pages, 1):
        try:
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                text = page.extract_text() or ""
        except Exception as exc:
            failed_pages += 1
            LOGGER.warning("PDF text extraction failed for %s page %d: %s", path, page_number, exc)
            continue
        if text.strip():
            # Repair common line-end hyphenation before paragraph normalization.
            text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
            pages.append(text)

    title: Optional[str] = None
    metadata = getattr(reader, "metadata", None)
    if metadata is not None:
        candidate = getattr(metadata, "title", None)
        if candidate is None and isinstance(metadata, Mapping):
            candidate = metadata.get("/Title")
        if candidate:
            title = str(candidate).strip() or None

    return _normalize_document_text("\n\n".join(pages)), title, {
        "pages": len(getattr(reader, "pages", [])),
        "pages_with_text": len(pages),
        "failed_pages": failed_pages,
    }


def _json_text_candidates(value: Any) -> Iterator[Tuple[str, Optional[str], Dict[str, Any]]]:
    """Yield logical text documents from JSON without mistaking task labels for prose."""
    preferred = ("text", "content", "document", "body", "passage", "article", "paragraph")
    if isinstance(value, Mapping):
        emitted = False
        title = value.get("title") if isinstance(value.get("title"), str) else None
        for key in preferred:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                emitted = True
                yield item, title, {"json_field": key}
        if not emitted:
            for key in ("records", "examples", "data", "documents", "items"):
                nested = value.get(key)
                if nested is not None:
                    yield from _json_text_candidates(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _json_text_candidates(item)
    elif isinstance(value, str) and value.strip():
        yield value, None, {"json_field": None}


def extract_raw_documents(path: Path, *, source_sha256: Optional[str] = None) -> Iterator[_ExtractedRawDocument]:
    suffix = path.suffix.lower()
    source_hash = source_sha256 or sha256_file(path)
    source_type = suffix.lstrip(".") or "unknown"

    if suffix in {".txt", ".text"}:
        text = _normalize_document_text(_decode_document_bytes(path.read_bytes()))
        if text:
            yield _ExtractedRawDocument(str(path), source_type, source_hash, text, None, "stdlib-text", 0)
        return

    if suffix in {".md", ".markdown"}:
        text = _clean_markdown_text(_decode_document_bytes(path.read_bytes()))
        if text:
            yield _ExtractedRawDocument(str(path), source_type, source_hash, text, None, "stdlib-markdown", 0)
        return

    if suffix in {".html", ".htm", ".xhtml"}:
        text, title = _extract_html_text(_decode_document_bytes(path.read_bytes()))
        if text:
            yield _ExtractedRawDocument(str(path), source_type, source_hash, text, title, "stdlib-html.parser", 0)
        return

    if suffix == ".docx":
        text, title, metadata = _extract_docx(path)
        if text:
            yield _ExtractedRawDocument(str(path), source_type, source_hash, text, title, "stdlib-zipfile+xml", 0, metadata)
        return

    if suffix == ".epub":
        text, title, metadata = _extract_epub(path)
        if text:
            yield _ExtractedRawDocument(str(path), source_type, source_hash, text, title, "stdlib-epub-zip+xml+html", 0, metadata)
        return

    if suffix == ".pdf":
        text, title, metadata = _extract_pdf(path)
        if text:
            yield _ExtractedRawDocument(str(path), source_type, source_hash, text, title, "pypdf", 0, metadata)
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            logical_index = 0
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LantraTrainingError(
                        f"Invalid raw-text JSONL in {path}:{line_number}: {exc.msg}."
                    ) from exc
                for text, title, metadata in _json_text_candidates(payload):
                    metadata = {**metadata, "jsonl_line": line_number}
                    yield _ExtractedRawDocument(
                        str(path), source_type, source_hash, _normalize_document_text(text), title,
                        "stdlib-json", logical_index, metadata,
                    )
                    logical_index += 1
        return

    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise LantraTrainingError(f"Invalid raw-text JSON in {path}: {exc.msg}.") from exc
        for logical_index, (text, title, metadata) in enumerate(_json_text_candidates(payload)):
            yield _ExtractedRawDocument(
                str(path), source_type, source_hash, _normalize_document_text(text), title,
                "stdlib-json", logical_index, metadata,
            )
        return

    raise LantraTrainingError(f"Unsupported raw corpus document type: {path}")


def segment_raw_document(text: str, *, min_chars: int, chunk_chars: int) -> Iterator[str]:
    normalized = " ".join(_normalize_document_text(text).split())
    if len(normalized) < min_chars:
        return
    if len(normalized) <= chunk_chars:
        yield normalized
        return

    words = normalized.split(" ")
    current: List[str] = []
    current_chars = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and current_chars + addition > chunk_chars:
            chunk = " ".join(current).strip()
            if len(chunk) >= min_chars:
                yield chunk
            current = [word]
            current_chars = len(word)
        else:
            current.append(word)
            current_chars += addition
    if current:
        chunk = " ".join(current).strip()
        if len(chunk) >= min_chars:
            yield chunk


def _assign_document_splits(
    documents: Sequence[_PreparedRawDocument],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, str]:
    """Assign complete documents before chunking to prevent cross-split leakage."""
    if not documents:
        return {}
    ordered = sorted(
        documents,
        key=lambda item: (
            stable_unit_interval(item.normalized_text_sha256, seed),
            item.normalized_text_sha256,
            item.document_id,
        ),
    )
    count = len(ordered)
    n_validation = 0 if validation_fraction <= 0.0 else max(1, int(round(count * validation_fraction)))
    n_test = 0 if test_fraction <= 0.0 else max(1, int(round(count * test_fraction)))

    # Always retain at least one training document. Tiny corpora cannot support
    # three statistically meaningful partitions; the manifest makes this explicit.
    while n_validation + n_test >= count:
        if n_test >= n_validation and n_test > 0:
            n_test -= 1
        elif n_validation > 0:
            n_validation -= 1
        else:
            break

    assignments: Dict[str, str] = {}
    for index, document in enumerate(ordered):
        if index < n_test:
            split = "test"
        elif index < n_test + n_validation:
            split = "validation"
        else:
            split = "train"
        assignments[document.document_id] = split
    return assignments


def load_raw_corpus(files: Sequence[Path], config: TrainerConfig) -> RawCorpus:
    prepared: List[_PreparedRawDocument] = []
    failures: List[RawExtractionFailure] = []
    accepted_files: List[str] = []
    seen_file_hashes: set[str] = set()
    seen_content_hashes: set[str] = set()
    seen_normalized_hashes: set[str] = set()
    duplicate_files_exact = 0
    duplicate_documents_exact = 0
    duplicate_documents_normalized = 0

    for path in files:
        try:
            source_hash = sha256_file(path)
        except OSError as exc:
            failures.append(RawExtractionFailure(str(path), path.suffix.lower().lstrip("."), type(exc).__name__, str(exc)))
            LOGGER.warning("LANTRA corpus could not hash %s: %s", path, exc)
            continue

        if source_hash in seen_file_hashes:
            duplicate_files_exact += 1
            LOGGER.info("LANTRA corpus exact duplicate file skipped: %s", path)
            continue
        seen_file_hashes.add(source_hash)

        try:
            extracted = list(extract_raw_documents(path, source_sha256=source_hash))
        except Exception as exc:
            failures.append(RawExtractionFailure(str(path), path.suffix.lower().lstrip("."), type(exc).__name__, str(exc)))
            LOGGER.warning("LANTRA corpus extraction skipped %s: %s", path, exc)
            continue

        if not extracted:
            failures.append(RawExtractionFailure(
                str(path), path.suffix.lower().lstrip("."), "NoTextExtracted",
                "Document contained no extractable training text (image-only PDFs require OCR before LANTRA ingestion).",
            ))
            continue

        accepted_files.append(str(path))
        for document in extracted:
            normalized = _normalize_document_text(document.text)
            if len(normalized) < config.raw_min_chars:
                continue
            content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
            normalized_hash = _normalized_text_hash(normalized)
            if content_hash in seen_content_hashes:
                duplicate_documents_exact += 1
                continue
            if normalized_hash in seen_normalized_hashes:
                duplicate_documents_normalized += 1
                continue
            seen_content_hashes.add(content_hash)
            seen_normalized_hashes.add(normalized_hash)
            document_id = sha256_payload({
                "source_sha256": document.source_sha256,
                "logical_index": document.logical_index,
                "normalized_text_sha256": normalized_hash,
            })[:24]
            prepared.append(_PreparedRawDocument(
                document_id=document_id,
                source_path=document.source_path,
                source_type=document.source_type,
                source_sha256=document.source_sha256,
                text=normalized,
                title=document.title,
                extractor=document.extractor,
                content_sha256=content_hash,
                normalized_text_sha256=normalized_hash,
                metadata=dict(document.metadata),
            ))

    assignments = _assign_document_splits(
        prepared,
        validation_fraction=config.raw_validation_fraction,
        test_fraction=config.raw_test_fraction,
        seed=config.seed,
    )

    # Chunk only after document assignment. Segment-level deduplication is global
    # across partitions, so repeated boilerplate cannot leak from train into held-out data.
    segment_by_hash: Dict[str, _SegmentCandidate] = {}
    duplicate_segments = 0
    for document in prepared:
        split = assignments[document.document_id]
        for segment_index, segment in enumerate(segment_raw_document(
            document.text,
            min_chars=config.raw_min_chars,
            chunk_chars=config.raw_chunk_chars,
        )):
            segment_hash = _normalized_text_hash(segment)
            candidate = _SegmentCandidate(segment, segment_hash, document.document_id, split, segment_index)
            existing = segment_by_hash.get(segment_hash)
            if existing is not None:
                duplicate_segments += 1
                # Select a deterministic owner independent of filesystem traversal order.
                current_key = stable_unit_interval(existing.document_id + segment_hash, config.seed)
                candidate_key = stable_unit_interval(candidate.document_id + segment_hash, config.seed)
                if candidate_key < current_key:
                    segment_by_hash[segment_hash] = candidate
            else:
                segment_by_hash[segment_hash] = candidate

    retained = sorted(
        segment_by_hash.values(),
        key=lambda item: (
            stable_unit_interval(item.normalized_sha256, config.seed + 71),
            item.normalized_sha256,
        ),
    )
    if config.raw_max_segments > 0:
        retained = retained[: config.raw_max_segments]

    split_segments: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    segment_counts = collections.Counter()
    for item in retained:
        split_segments[item.split].append(item.text)
        segment_counts[item.document_id] += 1

    provenance: List[RawDocumentProvenance] = []
    for document in prepared:
        provenance.append(RawDocumentProvenance(
            document_id=document.document_id,
            source_path=document.source_path,
            source_type=document.source_type,
            source_sha256=document.source_sha256,
            content_sha256=document.content_sha256,
            normalized_text_sha256=document.normalized_text_sha256,
            title=document.title,
            extractor=document.extractor,
            character_count=len(document.text),
            split=assignments[document.document_id],
            segment_count=int(segment_counts.get(document.document_id, 0)),
            metadata=dict(document.metadata),
        ))

    fingerprint = sha256_payload({
        "files": [
            {"path": document.source_path, "source_sha256": document.source_sha256}
            for document in provenance
        ],
        "documents": [
            {
                "document_id": document.document_id,
                "normalized_text_sha256": document.normalized_text_sha256,
                "split": document.split,
                "segment_count": document.segment_count,
            }
            for document in provenance
        ],
        "segments": {
            split: [hashlib.sha256(item.encode("utf-8")).hexdigest() for item in values]
            for split, values in split_segments.items()
        },
    })

    return RawCorpus(
        train_segments=tuple(split_segments["train"]),
        validation_segments=tuple(split_segments["validation"]),
        test_segments=tuple(split_segments["test"]),
        files=tuple(accepted_files),
        fingerprint=fingerprint,
        documents=tuple(sorted(provenance, key=lambda item: item.document_id)),
        duplicate_files_exact=duplicate_files_exact,
        duplicate_documents_exact=duplicate_documents_exact,
        duplicate_documents_normalized=duplicate_documents_normalized,
        duplicate_segments=duplicate_segments,
        extraction_failures=tuple(failures),
    )


def write_raw_corpus_manifest(corpus: RawCorpus, report_dir: Path, run_id: str) -> RawCorpus:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"lantra_corpus_manifest_{run_id}.json"
    enriched = dataclasses.replace(corpus, manifest_path=str(path))
    payload = {
        "schema": "slai.lantra.raw-corpus-manifest.v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "fingerprint_sha256": enriched.fingerprint,
        "summary": enriched.to_dict(),
        "documents": [document.to_dict() for document in enriched.documents],
    }
    atomic_json_write(path, payload)
    return enriched

def _glove_dimension_from_filename(path: Path) -> Optional[int]:
    name = path.name.lower()
    for dimension in (50, 100, 200, 300, 512, 768, 1024):
        if f".{dimension}d" in name or f"_{dimension}d" in name or f"-{dimension}d" in name:
            return dimension
    return None


def discover_glove_path(config: TrainerConfig, model_dimension: int) -> Optional[Path]:
    if not config.glove_bootstrap:
        return None
    explicit = config.glove_path or os.getenv("SLAI_LANTRA_GLOVE", "").strip() or None
    if config.init_from and not explicit:
        LOGGER.info("Warm-start checkpoint supplied; skipping automatic GloVe reinitialization.")
        return None
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise LantraTrainingError(f"Configured GloVe file does not exist: {path}")
        return path

    candidates = [Path(value) for value in DEFAULT_GLOVE_CANDIDATES if Path(value).is_file()]
    if not candidates:
        return None
    compatible: List[Tuple[int, Path]] = []
    unknown: List[Path] = []
    for path in candidates:
        dim = _glove_dimension_from_filename(path)
        if dim is None:
            unknown.append(path)
        elif dim <= model_dimension:
            compatible.append((dim, path))
    if compatible:
        compatible.sort(key=lambda item: item[0], reverse=True)
        return compatible[0][1]
    return unknown[0] if unknown else None


def _glove_key_for_token(token: str, tokenizer: LanguageTokenizer) -> Optional[str]:
    value = str(token).strip()
    if not value or value in set(getattr(tokenizer, "special_tokens", [])):
        return None
    suffix = str(getattr(tokenizer, "end_of_word_suffix", "") or "")
    prefix = str(getattr(tokenizer, "continuation_prefix", "") or "")

    # GloVe is word-level. When SLAI's BPE marks end-of-word explicitly, only
    # align complete-word BPE symbols. Mapping continuation fragments such as
    # ``in`` from inside another word to the standalone GloVe word "in" would
    # inject incorrect lexical semantics.
    if suffix and bool(getattr(tokenizer, "add_end_of_word", True)):
        if not value.endswith(suffix):
            return None
        value = value[:-len(suffix)]
    elif suffix and value.endswith(suffix):
        value = value[:-len(suffix)]
    if prefix and value.startswith(prefix):
        return None
    if value.startswith("##"):
        return None

    value = value.strip().lower()
    if not value or any(ch.isspace() for ch in value):
        return None
    return value


def _coerce_vector(value: Any) -> Optional[Tuple[float, ...]]:
    if isinstance(value, Mapping):
        for key in ("vector", "embedding", "values"):
            if key in value:
                return _coerce_vector(value[key])
        return None
    if isinstance(value, str):
        parts = value.strip().split()
        if not parts:
            return None
        value = parts
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return None
    out: List[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return tuple(out) if out else None


def load_glove_asset(
    path: Path,
    tokenizer: LanguageTokenizer,
    *,
    model_dimension: int,
    max_tokens: int,
) -> GloveAsset:
    desired: Dict[str, List[Tuple[int, str]]] = collections.defaultdict(list)
    for token, token_id in tokenizer.vocab.items():
        key = _glove_key_for_token(token, tokenizer)
        if key:
            desired[key].append((int(token_id), str(token)))

    LOGGER.info("Loading GloVe resource %s for %d candidate lexical keys", path, len(desired))
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LantraTrainingError(f"Failed to load GloVe JSON {path}: {exc}") from exc

    container: Any = payload
    if isinstance(payload, Mapping):
        for key in ("vectors", "embeddings", "data"):
            nested = payload.get(key)
            if isinstance(nested, (Mapping, list, tuple)):
                container = nested
                break

    found: Dict[str, Tuple[float, ...]] = {}
    dimension: Optional[int] = None

    # Also support the common {"words": [...], "vectors": [[...], ...]}
    # representation in addition to word->vector mappings.
    paired_words = payload.get("words") if isinstance(payload, Mapping) else None
    paired_vectors = payload.get("vectors") if isinstance(payload, Mapping) else None
    if (
        isinstance(paired_words, Sequence)
        and not isinstance(paired_words, (str, bytes, bytearray))
        and isinstance(paired_vectors, Sequence)
        and not isinstance(paired_vectors, (str, bytes, bytearray))
        and len(paired_words) == len(paired_vectors)
    ):
        for word, raw_vector in zip(paired_words, paired_vectors):
            key = str(word).strip().lower()
            if key not in desired:
                continue
            vector = _coerce_vector(raw_vector)
            if vector is None:
                continue
            if dimension is None:
                dimension = len(vector)
            if len(vector) == dimension:
                found[key] = vector
    elif isinstance(container, Mapping):
        for word, raw_vector in container.items():
            key = str(word).strip().lower()
            if key not in desired:
                continue
            vector = _coerce_vector(raw_vector)
            if vector is None:
                continue
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                continue
            found[key] = vector
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)):
        for item in container:
            if isinstance(item, Mapping):
                word = item.get("word", item.get("token", item.get("text")))
                if not isinstance(word, str):
                    continue
                key = word.strip().lower()
                raw_vector = item
            elif (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes, bytearray))
                and len(item) >= 2
                and isinstance(item[0], str)
            ):
                key = item[0].strip().lower()
                raw_vector = item[1:]
            else:
                continue
            if key not in desired:
                continue
            vector = _coerce_vector(raw_vector)
            if vector is None:
                continue
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                continue
            found[key] = vector
    else:
        raise LantraTrainingError(
            f"Unsupported GloVe JSON structure in {path}; expected word->vector mapping or record list."
        )

    del payload
    if not found or dimension is None:
        raise LantraTrainingError(
            f"GloVe file {path} loaded, but none of its vectors matched SLAI BPE vocabulary tokens."
        )
    if dimension > model_dimension:
        raise LantraTrainingError(
            f"GloVe dimension {dimension} exceeds LanguageTransformer d_model={model_dimension}. "
            "Use a compatible GloVe file (the repository's 200d file is compatible with d_model=256)."
        )

    matches: List[GloveMatch] = []
    for key in sorted(found):
        vector = found[key]
        for token_id, token in desired[key]:
            matches.append(GloveMatch(token_id, token, key, vector))
            if max_tokens > 0 and len(matches) >= max_tokens:
                break
        if max_tokens > 0 and len(matches) >= max_tokens:
            break
    return GloveAsset(
        path=str(path),
        dimension=dimension,
        matches=tuple(matches),
        file_sha256=sha256_file(path),
    )


def _expand_glove_vector(vector: Sequence[float], model_dimension: int) -> List[float]:
    if len(vector) > model_dimension:
        raise LantraTrainingError(
            f"Cannot losslessly place {len(vector)}d GloVe vector into {model_dimension}d model embedding."
        )
    # Zero padding preserves all original dot products, Euclidean distances in
    # the original subspace, and cosine similarity exactly.
    return [float(value) for value in vector] + [0.0] * (model_dimension - len(vector))


def initialize_embeddings_from_glove(
    model: LanguageTransformer,
    asset: GloveAsset,
) -> Dict[str, Any]:
    runtime = torch_runtime()
    dimension = int(model.config.d_model)
    copied = 0
    with runtime.no_grad():
        for match in asset.matches:
            vector = _expand_glove_vector(match.vector, dimension)
            src_row = model.src_embed.weight[match.token_id]
            tgt_row = model.tgt_embed.weight[match.token_id]
            tensor = src_row.new_tensor(vector)
            src_row.copy_(tensor)
            tgt_row.copy_(tensor)
            copied += 1
    LOGGER.info(
        "Initialized %d SLAI source/target token rows from %dd GloVe into d_model=%d",
        copied,
        asset.dimension,
        dimension,
    )
    return {
        "rows_initialized": copied,
        "glove_dimension": asset.dimension,
        "model_dimension": dimension,
        "strategy": "direct_copy" if asset.dimension == dimension else "zero_pad_geometry_preserving",
    }


def save_phase_checkpoint(
    model: LanguageTransformer,
    optimizer: Optional[Any],
    path: Path,
    *,
    run_id: str,
    phase: str,
    metadata: Mapping[str, Any],
) -> str:
    payload = {
        "trainer": "train_lantra",
        "run_id": run_id,
        "phase": phase,
        "saved_at": utc_now(),
        **dict(metadata),
    }
    return model.save_language_model(path, lang_metadata=payload, optimizer=optimizer)


def semantic_bootstrap_train(
    config: TrainerConfig,
    tokenizer: LanguageTokenizer,
    model: LanguageTransformer,
    asset: GloveAsset,
    device: str,
    run_id: str,
) -> Dict[str, Any]:
    """Distill static GloVe token semantics through SLAI's trainable encoder."""
    if config.glove_epochs <= 0 or not asset.matches:
        return {"status": "skipped", "reason": "glove_epochs=0 or no matches"}

    runtime = torch_runtime()
    functional = runtime.nn.functional
    optimizer = model.configure_optimizer()
    set_optimizer_lr(optimizer, config.glove_learning_rate)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "lantra_glove_bootstrap.pt"
    history: List[Dict[str, Any]] = []
    rng = random.Random(config.seed + 101)
    matches = list(asset.matches)
    model_dimension = int(model.config.d_model)

    for epoch in range(1, config.glove_epochs + 1):
        rng.shuffle(matches)
        model.train()
        loss_sum = 0.0
        cos_sum = 0.0
        mse_sum = 0.0
        batches = 0
        for offset in range(0, len(matches), config.glove_batch_size):
            batch = matches[offset:offset + config.glove_batch_size]
            if not batch:
                continue
            ids = [
                [int(tokenizer.bos_token_id), item.token_id, int(tokenizer.eos_token_id)]
                for item in batch
            ]
            targets = [_expand_glove_vector(item.vector, model_dimension) for item in batch]
            input_ids = runtime.tensor(ids, dtype=runtime.long, device=device)
            target_tensor = runtime.tensor(targets, dtype=model.src_embed.weight.dtype, device=device)

            optimizer.zero_grad(set_to_none=True)
            memory = model.encode(input_ids)
            representation = memory[:, 1, :]
            rep_norm = functional.normalize(representation, p=2, dim=-1)
            tgt_norm = functional.normalize(target_tensor, p=2, dim=-1)
            cosine_loss = (1.0 - (rep_norm * tgt_norm).sum(dim=-1)).mean()
            mse_loss = functional.mse_loss(rep_norm, tgt_norm)
            loss = 0.80 * cosine_loss + 0.20 * mse_loss
            if not bool(runtime.isfinite(loss).all().item()):
                raise LantraTrainingError("Non-finite GloVe semantic bootstrap loss encountered.")
            loss.backward()
            grad_norm = model.clip_gradients(optimizer)
            optimizer.step()

            batches += 1
            loss_sum += float(loss.detach().cpu().item())
            cos_sum += float(cosine_loss.detach().cpu().item())
            mse_sum += float(mse_loss.detach().cpu().item())
            if config.log_every > 0 and batches % config.log_every == 0:
                LOGGER.info(
                    "GloVe bootstrap epoch=%d batch=%d/%d loss=%.6f grad_norm=%.4f",
                    epoch,
                    batches,
                    math.ceil(len(matches) / config.glove_batch_size),
                    loss_sum / batches,
                    grad_norm,
                )

        record = {
            "epoch": epoch,
            "batches": batches,
            "examples": len(matches),
            "mean_loss": loss_sum / max(1, batches),
            "mean_cosine_loss": cos_sum / max(1, batches),
            "mean_normalized_mse": mse_sum / max(1, batches),
        }
        history.append(record)
        PRINTER.pretty("LANTRA GLOVE", record, "success")

    saved = save_phase_checkpoint(
        model,
        optimizer,
        checkpoint,
        run_id=run_id,
        phase="glove_semantic_bootstrap",
        metadata={"asset": asset.to_dict(), "history": history},
    )
    return {
        "status": "completed",
        "asset": asset.to_dict(),
        "history": history,
        "checkpoint": saved,
    }


def _encode_raw_ids(tokenizer: LanguageTokenizer, text: str, max_length: int) -> List[int]:
    payload = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
        return_attention_mask=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_token_metadata=False,
        return_tokens=False,
    )
    values = payload.get("input_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise LantraTrainingError("LanguageTokenizer returned invalid input_ids for raw-text pretraining.")
    return [int(value) for value in values]


def corrupt_token_ids(
    token_ids: Sequence[int],
    tokenizer: LanguageTokenizer,
    *,
    probability: float,
    rng: random.Random,
) -> List[int]:
    values = [int(value) for value in token_ids]
    special_ids = {
        int(getattr(tokenizer, name))
        for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id")
        if hasattr(tokenizer, name) and getattr(tokenizer, name) is not None
    }
    eligible = [index for index, value in enumerate(values) if value not in special_ids]
    if not eligible or probability <= 0.0:
        return values
    selected = [index for index in eligible if rng.random() < probability]
    if not selected:
        selected = [rng.choice(eligible)]

    unk_id = int(getattr(tokenizer, "unk_token_id", tokenizer.pad_token_id))
    vocab_size = max(1, len(tokenizer.vocab))
    for index in selected:
        draw = rng.random()
        if draw < 0.80:
            values[index] = unk_id
        elif draw < 0.90:
            for _ in range(10):
                candidate = rng.randrange(vocab_size)
                if candidate not in special_ids:
                    values[index] = candidate
                    break
        # final 10% intentionally remains unchanged
    return values


def raw_reconstruction_loss(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    text: str,
    config: TrainerConfig,
    device: str,
    *,
    rng: random.Random,
) -> Any:
    runtime = torch_runtime()
    original = _encode_raw_ids(tokenizer, text, config.raw_max_length)
    if len(original) < 3:
        return None
    corrupted = corrupt_token_ids(
        original,
        tokenizer,
        probability=config.raw_corruption_probability,
        rng=rng,
    )
    src = runtime.tensor([corrupted], dtype=runtime.long, device=device)
    target = runtime.tensor([original], dtype=runtime.long, device=device)
    decoder_input = target[:, :-1]
    labels = target[:, 1:]
    output = model.language_forward(src, decoder_input, return_dict=True)
    logits = output["logits"]
    return model.compute_loss(logits, labels, ignore_index=int(tokenizer.pad_token_id))


def evaluate_raw_pretraining(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    segments: Sequence[str],
    config: TrainerConfig,
    device: str,
) -> Optional[float]:
    if not segments:
        return None
    runtime = torch_runtime()
    model.eval()
    total = 0.0
    count = 0
    with runtime.no_grad():
        for text in segments[: min(len(segments), 1000)]:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") ^ config.seed
            loss = raw_reconstruction_loss(
                model,
                tokenizer,
                text,
                config,
                device,
                rng=random.Random(seed),
            )
            if loss is None:
                continue
            total += finite_loss_value(loss)
            count += 1
    return total / count if count else None


def raw_text_pretrain(
    config: TrainerConfig,
    corpus: RawCorpus,
    tokenizer: LanguageTokenizer,
    model: LanguageTransformer,
    device: str,
    run_id: str,
) -> Tuple[LanguageTransformer, Dict[str, Any]]:
    if config.raw_pretrain_epochs <= 0 or not corpus.train_segments:
        return model, {"status": "skipped", "reason": "no raw training segments or raw_pretrain_epochs=0"}

    runtime = torch_runtime()
    train_segments = list(corpus.train_segments)
    validation_segments = list(corpus.validation_segments)
    test_segments = list(corpus.test_segments)
    if not train_segments:
        return model, {"status": "skipped", "reason": "raw corpus produced no train segments"}

    optimizer = model.configure_optimizer()
    per_epoch_count = len(train_segments)
    if config.raw_segments_per_epoch > 0:
        per_epoch_count = min(per_epoch_count, config.raw_segments_per_epoch)
    total_steps = max(
        1,
        math.ceil(per_epoch_count / config.gradient_accumulation) * config.raw_pretrain_epochs,
    )
    warmup = min(config.warmup_steps, max(1, total_steps // 10) if config.warmup_steps else 0)
    global_step = 0
    best_validation = float("inf")
    best_path = Path(config.output_dir) / "lantra_raw_pretrain_best.pt"
    latest_path = Path(config.output_dir) / "lantra_raw_pretrain_latest.pt"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    rng = random.Random(config.seed + 202)

    for epoch in range(1, config.raw_pretrain_epochs + 1):
        order = list(train_segments)
        rng.shuffle(order)
        order = order[:per_epoch_count]
        model.train()
        epoch_loss = 0.0
        example_count = 0
        epoch_steps = 0

        for group_start in range(0, len(order), config.gradient_accumulation):
            group = order[group_start:group_start + config.gradient_accumulation]
            if not group:
                continue
            optimizer.zero_grad(set_to_none=True)
            # Segments are filtered to meaningful text before training, so nearly
            # every item yields >=3 tokens. Backpropagate each graph immediately
            # instead of retaining an entire accumulation group's graphs in memory.
            scale = float(len(group))
            valid_in_group = 0
            for text in group:
                loss = raw_reconstruction_loss(model, tokenizer, text, config, device, rng=rng)
                if loss is None:
                    continue
                (loss / scale).backward()
                value = finite_loss_value(loss)
                epoch_loss += value
                example_count += 1
                valid_in_group += 1
            if valid_in_group == 0:
                optimizer.zero_grad(set_to_none=True)
                continue

            global_step += 1
            lr = scheduled_learning_rate(
                global_step,
                total_steps,
                peak_lr=config.learning_rate,
                min_lr=config.min_learning_rate,
                warmup_steps=warmup,
            )
            set_optimizer_lr(optimizer, lr)
            grad_norm = model.clip_gradients(optimizer)
            optimizer.step()
            epoch_steps += 1
            if config.log_every > 0 and global_step % config.log_every == 0:
                LOGGER.info(
                    "Raw pretrain epoch=%d step=%d/%d examples=%d mean_loss=%.6f lr=%.8f grad_norm=%.4f",
                    epoch,
                    global_step,
                    total_steps,
                    example_count,
                    epoch_loss / max(1, example_count),
                    lr,
                    grad_norm,
                )

        validation_loss = evaluate_raw_pretraining(
            model,
            tokenizer,
            validation_segments,
            config,
            device,
        )
        record = {
            "epoch": epoch,
            "examples": example_count,
            "optimizer_steps": epoch_steps,
            "mean_train_loss": epoch_loss / max(1, example_count),
            "validation_loss": validation_loss,
            "global_optimizer_step": global_step,
        }
        history.append(record)
        metric = validation_loss if validation_loss is not None else record["mean_train_loss"]
        if metric < best_validation:
            best_validation = metric
            save_phase_checkpoint(
                model,
                optimizer,
                best_path,
                run_id=run_id,
                phase="raw_text_denoising_pretraining_best",
                metadata={"corpus": corpus.to_dict(), "record": record},
            )
        save_phase_checkpoint(
            model,
            optimizer,
            latest_path,
            run_id=run_id,
            phase="raw_text_denoising_pretraining_latest",
            metadata={"corpus": corpus.to_dict(), "record": record},
        )
        PRINTER.pretty("LANTRA RAW PRETRAIN", record, "success")

    if best_path.is_file():
        model = LanguageTransformer.load_language_model(best_path, device=device, strict=True)
    test_loss = evaluate_raw_pretraining(
        model,
        tokenizer,
        test_segments,
        config,
        device,
    )
    return model, {
        "status": "completed",
        "corpus": corpus.to_dict(),
        "train_segments": len(train_segments),
        "validation_segments": len(validation_segments),
        "test_segments": len(test_segments),
        "history": history,
        "best_objective": best_validation,
        "heldout_test_loss": test_loss,
        "checkpoint": str(best_path if best_path.is_file() else latest_path),
    }


def encode_text(
    tokenizer: LanguageTokenizer,
    text: str,
    *,
    max_length: int,
    device: str,
    pad_to_max: bool = False,
) -> Tuple[Any, Any]:
    payload = tokenizer.encode(
        text,
        add_special_tokens=True,
        padding="max_length" if pad_to_max else False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_token_metadata=False,
        return_tokens=False,
        return_tensors="pt",
    )
    ids = payload.get("input_ids")
    mask = payload.get("attention_mask")
    if ids is None or mask is None:
        raise LantraTrainingError("LanguageTokenizer.encode did not return input_ids and attention_mask.")
    if ids.dim() != 1 or mask.dim() != 1:
        raise LantraTrainingError(
            f"Unexpected tokenizer tensor shape ids={list(ids.shape)}, mask={list(mask.shape)}."
        )
    return ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)


def build_seq2seq_text(example: TrainingExample) -> Tuple[str, str]:
    assert example.task in SEQ2SEQ_TASKS
    assert example.source is not None and example.target is not None

    if example.task == "generation":
        source = f"task: generation\nprompt:\n{example.source}\nresponse:"
        target = example.target
    elif example.task == "classification":
        source = f"task: classification\ntext:\n{example.source}\nlabel:"
        target = example.target
    elif example.task == "translation":
        source_language = str(example.metadata.get("source_language", "unspecified"))
        target_language = str(example.metadata.get("target_language", "unspecified"))
        source = (
            "task: translation\n"
            f"source_language: {source_language}\n"
            f"target_language: {target_language}\n"
            f"source:\n{example.source}\ntranslation:"
        )
        target = example.target
    elif example.task == "summarization":
        source = f"task: summarization\ndocument:\n{example.source}\nsummary:"
        target = example.target
    elif example.task == "dialogue":
        source = f"task: dialogue\nconversation:\n{example.source}\nassistant:"
        target = example.target
    else:
        raise LantraTrainingError(f"Unsupported seq2seq task: {example.task}")
    return source, target


def seq2seq_loss(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    example: TrainingExample,
    config: TrainerConfig,
    device: str,
) -> Any:
    source_text, target_text = build_seq2seq_text(example)
    src, _src_attention = encode_text(
        tokenizer,
        source_text,
        max_length=config.source_max_length,
        device=device,
    )
    target_full, _target_attention = encode_text(
        tokenizer,
        target_text,
        max_length=config.target_max_length,
        device=device,
    )

    # Correct teacher forcing: decoder consumes BOS..token[n-1] and predicts
    # token[1]..EOS.  Feeding target_full unchanged as both decoder input and
    # label would leak the token being predicted into the decoder input.
    decoder_input = target_full[:, :-1]
    labels = target_full[:, 1:]
    src_padding_mask = model.make_padding_mask(src, pad_token_id=tokenizer.pad_token_id)
    tgt_padding_mask = model.make_padding_mask(decoder_input, pad_token_id=tokenizer.pad_token_id)

    output = model.language_forward(
        src,
        decoder_input,
        src_key_padding_mask=src_padding_mask,
        tgt_key_padding_mask=tgt_padding_mask,
        memory_key_padding_mask=src_padding_mask,
        return_dict=True,
    )
    logits = output["logits"]
    return model.compute_loss(
        logits,
        labels,
        ignore_index=int(tokenizer.pad_token_id),
        reduction="mean",
    )


def trainable_encoder_embedding(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    text: str,
    *,
    prefix: str,
    max_length: int,
    device: str,
) -> Any:
    ids, attention_mask = encode_text(
        tokenizer,
        f"{prefix}\n{text}",
        max_length=max_length,
        device=device,
    )
    padding_mask = attention_mask.eq(0)
    memory = model.encode(ids, src_key_padding_mask=padding_mask)
    if memory.dim() != 3:
        raise LantraTrainingError(f"Encoder memory must be rank 3, got shape {list(memory.shape)}.")

    # Mean pooling over non-padding positions while retaining autograd.
    valid = attention_mask.unsqueeze(-1).to(dtype=memory.dtype)
    pooled = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    norm = pooled.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return pooled / norm


def pairwise_loss(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    example: TrainingExample,
    config: TrainerConfig,
    device: str,
) -> Tuple[Any, bool, float, float]:
    assert example.anchor is not None and example.positive is not None and example.negative is not None
    if example.task == "embedding":
        query_prefix = "task: embedding\ntext:"
        document_prefix = query_prefix
        margin = config.embedding_margin
    elif example.task == "reranking":
        query_prefix = "task: reranking\nquery:"
        document_prefix = "task: reranking\ndocument:"
        margin = config.reranking_margin
    else:
        raise LantraTrainingError(f"Unsupported pairwise task: {example.task}")

    anchor = trainable_encoder_embedding(
        model,
        tokenizer,
        example.anchor,
        prefix=query_prefix,
        max_length=config.embedding_max_length,
        device=device,
    )
    positive = trainable_encoder_embedding(
        model,
        tokenizer,
        example.positive,
        prefix=document_prefix,
        max_length=config.embedding_max_length,
        device=device,
    )
    negative = trainable_encoder_embedding(
        model,
        tokenizer,
        example.negative,
        prefix=document_prefix,
        max_length=config.embedding_max_length,
        device=device,
    )

    positive_similarity = (anchor * positive).sum(dim=-1)
    negative_similarity = (anchor * negative).sum(dim=-1)
    loss = (float(margin) - positive_similarity + negative_similarity).clamp_min(0.0).mean()
    pos_value = float(positive_similarity.detach().cpu().mean().item())
    neg_value = float(negative_similarity.detach().cpu().mean().item())
    return loss, pos_value > neg_value, pos_value, neg_value


def compute_example_loss(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    example: TrainingExample,
    config: TrainerConfig,
    device: str,
) -> Tuple[Any, Dict[str, Any]]:
    if example.task in SEQ2SEQ_TASKS:
        loss = seq2seq_loss(model, tokenizer, example, config, device)
        return loss, {}
    loss, correct, pos, neg = pairwise_loss(model, tokenizer, example, config, device)
    return loss, {
        "pairwise_correct": correct,
        "positive_similarity": pos,
        "negative_similarity": neg,
    }


def example_order(
    records: Sequence[TrainingExample],
    *,
    seed: int,
    epoch: int,
    balance_tasks: bool,
) -> List[TrainingExample]:
    """Return an epoch order without dropping or duplicating real examples.

    With task balancing enabled, task-specific queues are independently shuffled
    and consumed round-robin.  This improves local task mixture while preserving
    the empirical dataset exactly once per epoch.  It intentionally avoids both
    minority oversampling and majority undersampling; stronger balancing belongs
    in dataset design or an explicitly documented weighting policy.
    """
    rng = random.Random(seed + 104729 * epoch)
    if not balance_tasks:
        output = list(records)
        rng.shuffle(output)
        return output

    groups: Dict[str, List[TrainingExample]] = collections.defaultdict(list)
    for item in records:
        groups[item.task].append(item)
    for values in groups.values():
        rng.shuffle(values)

    active_tasks = [task for task in SUPPORTED_TASKS if groups.get(task)]
    if not active_tasks:
        return []
    rng.shuffle(active_tasks)
    positions = {task: 0 for task in active_tasks}
    output: List[TrainingExample] = []

    while len(output) < len(records):
        progressed = False
        for task in active_tasks:
            position = positions[task]
            values = groups[task]
            if position < len(values):
                output.append(values[position])
                positions[task] = position + 1
                progressed = True
        if not progressed:
            break

    if len(output) != len(records):
        raise LantraTrainingError(
            f"Internal epoch-order invariant failed: expected {len(records)} examples, got {len(output)}."
        )
    return output


def scheduled_learning_rate(
    step: int,
    total_steps: int,
    *,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if step <= 0:
        return 0.0 if warmup_steps > 0 else peak_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * (step / warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: Any, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def finite_loss_value(loss: Any) -> float:
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value):
        raise LantraTrainingError(f"Non-finite loss encountered: {value}.")
    return value


def task_weight(task: str) -> float:
    # Balanced sampling already equalizes task exposure.  Keep the objective
    # weights neutral so the report remains interpretable and users can assess
    # raw per-task losses instead of hiding arbitrary scaling constants.
    if task not in SUPPORTED_TASKS:
        raise LantraTrainingError(f"Unknown task weight requested: {task}")
    return 1.0


def train_epoch(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    optimizer: Any,
    records: Sequence[TrainingExample],
    config: TrainerConfig,
    device: str,
    *,
    epoch: int,
    total_optimizer_steps: int,
    effective_warmup_steps: int,
    global_optimizer_step: int,
) -> Tuple[EpochStats, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stats = EpochStats()
    ordered = example_order(
        records,
        seed=config.seed,
        epoch=epoch,
        balance_tasks=config.balance_tasks,
    )
    pending = 0

    for index, example in enumerate(ordered, 1):
        loss, aux = compute_example_loss(model, tokenizer, example, config, device)
        raw_loss = finite_loss_value(loss)
        weight = task_weight(example.task)
        weighted = loss * weight
        # Average each accumulation group by its actual size.  This matters for
        # the final partial group; dividing it by the configured accumulation
        # value would systematically underweight those examples.
        group_start = ((index - 1) // config.gradient_accumulation) * config.gradient_accumulation
        group_size = min(config.gradient_accumulation, len(ordered) - group_start)
        scaled = weighted / float(group_size)
        scaled.backward()
        weighted_value = raw_loss * weight
        stats.add_loss(example.task, raw_loss, weighted_value)
        if "pairwise_correct" in aux:
            stats.add_pairwise(example.task, bool(aux["pairwise_correct"]))
        pending += 1

        should_step = pending >= config.gradient_accumulation or index == len(ordered)
        if should_step:
            global_optimizer_step += 1
            lr = scheduled_learning_rate(
                global_optimizer_step,
                total_optimizer_steps,
                peak_lr=config.learning_rate,
                min_lr=config.min_learning_rate,
                warmup_steps=effective_warmup_steps,
            )
            set_optimizer_lr(optimizer, lr)
            grad_norm = float(model.clip_gradients(optimizer))
            if not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise LantraTrainingError(f"Non-finite gradient norm encountered: {grad_norm}.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            stats.optimizer_steps += 1
            stats.grad_norm_sum += grad_norm
            pending = 0

        if config.log_every > 0 and index % config.log_every == 0:
            LOGGER.info(
                "epoch=%d sample=%d/%d task=%s loss=%.6f mean_loss=%.6f optimizer_step=%d",
                epoch,
                index,
                len(ordered),
                example.task,
                raw_loss,
                stats.raw_loss_sum / max(1, stats.examples),
                global_optimizer_step,
            )

    return stats, global_optimizer_step


def normalize_eval_text(text: str) -> List[str]:
    return " ".join(text.lower().split()).split()


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    # O(min(n,m)) memory dynamic programming.
    if len(b) > len(a):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(current[-1], previous[j]))
        previous = current
    return previous[-1]


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_eval_text(prediction)
    ref = normalize_eval_text(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    pred_counts = collections.Counter(pred)
    ref_counts = collections.Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = normalize_eval_text(prediction)
    ref = normalize_eval_text(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def generate_prediction(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    example: TrainingExample,
    config: TrainerConfig,
    device: str,
) -> str:
    source_text, _ = build_seq2seq_text(example)
    src, _ = encode_text(
        tokenizer,
        source_text,
        max_length=config.source_max_length,
        device=device,
    )
    # encode_text uses padding=False for single-example inference.  The current
    # LanguageTransformer greedy/sample wrapper does not forward a source padding
    # mask into BaseTransformer.inference, so deliberately keep this source
    # unpadded rather than pretending that an ignored mask is honored.
    generated = model.generate(
        src,
        strategy="greedy",
        max_len=config.target_max_length,
        return_dict=False,
    )
    if generated.dim() != 2 or generated.size(0) < 1:
        raise LantraTrainingError(f"Unexpected generated tensor shape: {list(generated.shape)}")
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def balanced_evaluation_subset(
    records: Sequence[TrainingExample],
    max_records: int,
) -> List[TrainingExample]:
    """Select evaluation records round-robin by task without changing their content.

    Early stopping is based on a macro task objective, so a capped validation
    sample must not accidentally omit a task simply because records are ordered
    by identifier or source file.  ``max_records == 0`` means evaluate all.
    """
    if max_records == 0 or max_records >= len(records):
        return list(records)
    buckets: Dict[str, List[TrainingExample]] = {task: [] for task in SUPPORTED_TASKS}
    for example in records:
        buckets[example.task].append(example)
    selected: List[TrainingExample] = []
    offsets = {task: 0 for task in SUPPORTED_TASKS}
    while len(selected) < max_records:
        progressed = False
        for task in SUPPORTED_TASKS:
            idx = offsets[task]
            bucket = buckets[task]
            if idx < len(bucket) and len(selected) < max_records:
                selected.append(bucket[idx])
                offsets[task] = idx + 1
                progressed = True
        if not progressed:
            break
    return selected


def evaluate(
    model: LanguageTransformer,
    tokenizer: LanguageTokenizer,
    records: Sequence[TrainingExample],
    config: TrainerConfig,
    device: str,
    *,
    include_generation_metrics: bool,
) -> Dict[str, Any]:
    runtime = torch_runtime()
    model.eval()
    sampled = balanced_evaluation_subset(records, config.max_eval_records)
    if not sampled:
        raise LantraTrainingError("Evaluation split is empty.")

    loss_sum = 0.0
    task_loss_sum: Dict[str, float] = collections.defaultdict(float)
    task_count: Dict[str, int] = collections.defaultdict(int)
    pair_correct: Dict[str, int] = collections.defaultdict(int)
    pair_total: Dict[str, int] = collections.defaultdict(int)

    with runtime.no_grad():
        for example in sampled:
            loss, aux = compute_example_loss(model, tokenizer, example, config, device)
            value = finite_loss_value(loss)
            loss_sum += value
            task_loss_sum[example.task] += value
            task_count[example.task] += 1
            if "pairwise_correct" in aux:
                pair_total[example.task] += 1
                if aux["pairwise_correct"]:
                    pair_correct[example.task] += 1

    vocab_baseline = max(1.0, math.log(max(2, len(tokenizer.vocab))))
    task_metrics: Dict[str, Any] = {}
    normalized_task_losses: List[float] = []
    for task in SUPPORTED_TASKS:
        count = task_count.get(task, 0)
        mean_task_loss = task_loss_sum.get(task, 0.0) / count if count else None
        pair_accuracy = (
            pair_correct.get(task, 0) / pair_total.get(task, 1)
            if pair_total.get(task, 0)
            else None
        )
        normalized_loss = None
        if mean_task_loss is not None:
            if task in SEQ2SEQ_TASKS:
                normalized_loss = mean_task_loss / vocab_baseline
            elif task == "embedding":
                normalized_loss = mean_task_loss / max(config.embedding_margin, 1e-6)
            elif task == "reranking":
                normalized_loss = mean_task_loss / max(config.reranking_margin, 1e-6)
            normalized_task_losses.append(float(normalized_loss))
        task_metrics[task] = {
            "examples": count,
            "mean_loss": mean_task_loss,
            "normalized_loss": normalized_loss,
            "pairwise_accuracy": pair_accuracy,
        }

    metrics: Dict[str, Any] = {
        "examples": len(sampled),
        "mean_loss": loss_sum / len(sampled),
        "macro_normalized_task_loss": (
            sum(normalized_task_losses) / len(normalized_task_losses)
            if normalized_task_losses else None
        ),
        "tasks": task_metrics,
    }
    seq_losses = [
        task_loss_sum[task] / task_count[task]
        for task in SEQ2SEQ_TASKS
        if task_count.get(task, 0)
    ]
    if seq_losses:
        seq_mean = sum(seq_losses) / len(seq_losses)
        metrics["seq2seq_mean_task_loss"] = seq_mean
        metrics["seq2seq_task_perplexity"] = math.exp(min(50.0, seq_mean))

    if include_generation_metrics and config.generation_eval_records > 0:
        generative: Dict[str, Any] = {}
        for task in SEQ2SEQ_TASKS:
            candidates = [item for item in sampled if item.task == task][: config.generation_eval_records]
            if not candidates:
                continue
            exact = 0
            f1_values: List[float] = []
            rouge_values: List[float] = []
            examples_payload: List[Dict[str, str]] = []
            with runtime.no_grad():
                for example in candidates:
                    assert example.target is not None
                    prediction = generate_prediction(model, tokenizer, example, config, device)
                    reference = example.target
                    pred_normalized = " ".join(normalize_eval_text(prediction))
                    ref_normalized = " ".join(normalize_eval_text(reference))
                    exact += int(pred_normalized == ref_normalized)
                    f1_values.append(token_f1(prediction, reference))
                    rouge_values.append(rouge_l_f1(prediction, reference))
                    examples_payload.append({
                        "id": example.example_id,
                        "prediction": prediction[:1000],
                        "reference": reference[:1000],
                    })
            generative[task] = {
                "examples": len(candidates),
                "exact_match": exact / len(candidates),
                "token_f1": sum(f1_values) / len(f1_values),
                "rouge_l_f1": sum(rouge_values) / len(rouge_values),
                "samples": examples_payload,
            }
        metrics["generation_quality"] = generative

    return metrics


def validation_objective(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("macro_normalized_task_loss")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LantraTrainingError(f"Invalid validation macro normalized task loss: {value!r}")
    return float(value)


def count_optimizer_steps(train_size: int, epochs: int, accumulation: int) -> int:
    per_epoch = int(math.ceil(train_size / accumulation))
    return max(1, per_epoch * epochs)


def save_checkpoint(
    model: LanguageTransformer,
    optimizer: Optional[Any],
    path: Path,
    *,
    run_id: str,
    dataset: DatasetSplit,
    config: TrainerConfig,
    epoch: int,
    global_optimizer_step: int,
    validation_metrics: Optional[Mapping[str, Any]] = None,
    status: str,
) -> str:
    metadata = {
        "trainer": "train_lantra",
        "run_id": run_id,
        "status": status,
        "epoch": epoch,
        "global_optimizer_step": global_optimizer_step,
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_counts": dataset.counts(),
        "trainer_config": config.to_dict(),
        "objective_contract": OBJECTIVE_CONTRACT,
        "validation_metrics": dict(validation_metrics or {}),
        "saved_at": utc_now(),
    }
    return model.save_language_model(
        path,
        lang_metadata=metadata,
        optimizer=optimizer,
    )


def train(
    config: TrainerConfig,
    dataset: DatasetSplit,
    tokenizer: LanguageTokenizer,
    model: LanguageTransformer,
    device: str,
    run_id: str,
) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "lantra_best.pt"
    latest_path = output_dir / "lantra_latest.pt"
    final_path = output_dir / f"lantra_{run_id}.pt"

    optimizer = model.configure_optimizer()
    total_optimizer_steps = count_optimizer_steps(
        len(dataset.train), config.epochs, config.gradient_accumulation
    )
    effective_warmup_steps = min(
        config.warmup_steps,
        max(1, total_optimizer_steps // 10) if config.warmup_steps > 0 else 0,
    )
    global_optimizer_step = 0
    best_validation_objective = float("inf")
    best_epoch = 0
    best_global_optimizer_step = 0
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    LOGGER.info(
        "Training LANTRA: train=%d validation=%d test=%d epochs=%d device=%s total_optimizer_steps=%d warmup_steps=%d",
        len(dataset.train),
        len(dataset.validation),
        len(dataset.test),
        config.epochs,
        device,
        total_optimizer_steps,
        effective_warmup_steps,
    )

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            train_stats, global_optimizer_step = train_epoch(
                model,
                tokenizer,
                optimizer,
                dataset.train,
                config,
                device,
                epoch=epoch,
                total_optimizer_steps=total_optimizer_steps,
                effective_warmup_steps=effective_warmup_steps,
                global_optimizer_step=global_optimizer_step,
            )
            validation_metrics = evaluate(
                model,
                tokenizer,
                dataset.validation,
                config,
                device,
                include_generation_metrics=False,
            )
            val_loss = validation_objective(validation_metrics)
            elapsed = time.perf_counter() - epoch_started
            record = {
                "epoch": epoch,
                "seconds": elapsed,
                "global_optimizer_step": global_optimizer_step,
                "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                "train": train_stats.to_dict(),
                "validation": validation_metrics,
            }
            history.append(record)

            improved = val_loss < best_validation_objective - config.min_delta
            if improved:
                best_validation_objective = val_loss
                best_epoch = epoch
                best_global_optimizer_step = global_optimizer_step
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    optimizer,
                    best_path,
                    run_id=run_id,
                    dataset=dataset,
                    config=config,
                    epoch=epoch,
                    global_optimizer_step=global_optimizer_step,
                    validation_metrics=validation_metrics,
                    status="best",
                )
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                model,
                optimizer,
                latest_path,
                run_id=run_id,
                dataset=dataset,
                config=config,
                epoch=epoch,
                global_optimizer_step=global_optimizer_step,
                validation_metrics=validation_metrics,
                status="latest",
            )

            PRINTER.pretty(
                "LANTRA EPOCH",
                {
                    "epoch": epoch,
                    "seconds": round(elapsed, 3),
                    "train_loss": train_stats.raw_loss_sum / max(1, train_stats.examples),
                    "validation_objective": val_loss,
                    "best_validation_objective": best_validation_objective,
                    "best_epoch": best_epoch,
                    "optimizer_step": global_optimizer_step,
                    "learning_rate": optimizer.param_groups[0].get("lr"),
                    "early_stop_counter": f"{epochs_without_improvement}/{config.patience}",
                },
                "success" if improved else "info",
            )

            if epochs_without_improvement >= config.patience:
                LOGGER.info(
                    "Early stopping after epoch %d; best epoch=%d validation_objective=%.6f",
                    epoch,
                    best_epoch,
                    best_validation_objective,
                )
                break

    except KeyboardInterrupt:
        interrupted_path = output_dir / "lantra_interrupted.pt"
        LOGGER.warning("Training interrupted; saving recoverable model checkpoint to %s", interrupted_path)
        save_checkpoint(
            model,
            optimizer,
            interrupted_path,
            run_id=run_id,
            dataset=dataset,
            config=config,
            epoch=history[-1]["epoch"] if history else 0,
            global_optimizer_step=global_optimizer_step,
            validation_metrics=history[-1]["validation"] if history else None,
            status="interrupted",
        )
        raise

    if not best_path.is_file():
        raise LantraTrainingError("Training completed without producing a best checkpoint.")

    LOGGER.info("Reloading best LANTRA checkpoint for held-out test evaluation: %s", best_path)
    best_model = LanguageTransformer.load_language_model(best_path, device=device, strict=True)
    test_metrics = evaluate(
        best_model,
        tokenizer,
        dataset.test,
        config,
        device,
        include_generation_metrics=True,
    )

    # Save a run-specific deployment checkpoint from the selected best model.
    # Do not attach a freshly initialized optimizer state: the native best
    # checkpoint already contains the real optimizer state from its selected epoch.
    final_saved = save_checkpoint(
        best_model,
        None,
        final_path,
        run_id=run_id,
        dataset=dataset,
        config=config,
        epoch=best_epoch,
        global_optimizer_step=best_global_optimizer_step,
        validation_metrics={
            "selection_metric": "macro_normalized_task_loss",
            "best_validation_objective": best_validation_objective,
        },
        status="final_best",
    )

    return {
        "best_epoch": best_epoch,
        "selection_metric": "macro_normalized_task_loss",
        "best_validation_objective": best_validation_objective,
        "best_global_optimizer_step": best_global_optimizer_step,
        "epochs_completed": len(history),
        "global_optimizer_steps": global_optimizer_step,
        "history": history,
        "test": test_metrics,
        "checkpoints": {
            "best": str(best_path),
            "latest": str(latest_path),
            "final": final_saved,
        },
        "model_stats": best_model.stats().to_dict(),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_lantra",
        description=(
            "Stage-train SLAI's native LanguageTransformer: BPE/GloVe bootstrap, "
            "raw-text self-supervision, then optional seven-task specialization."
        ),
    )
    parser.add_argument(
        "--data",
        action="append",
        default=[],
        help="Optional supervised JSON/JSONL file or directory. Repeatable.",
    )
    parser.add_argument(
        "--raw-text",
        action="append",
        default=[],
        help="Additional raw/document corpus file or directory. Supports TXT/MD/HTML/DOCX/EPUB/PDF/JSON/JSONL; data/library is always included automatically.",
    )
    parser.add_argument(
        "--glove",
        default=None,
        help="Optional GloVe JSON path. Default: auto-select highest compatible file from data/embeddings/.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)

    # Supervised specialization.
    parser.add_argument("--epochs", type=int, default=int(os.getenv("SLAI_LANTRA_EPOCHS", "12")))
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--source-max-length", type=int, default=384)
    parser.add_argument("--target-max-length", type=int, default=160)
    parser.add_argument("--embedding-max-length", type=int, default=192)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=os.getenv("SLAI_LANTRA_DEVICE", "auto"))
    parser.add_argument("--embedding-margin", type=float, default=0.20)
    parser.add_argument("--reranking-margin", type=float, default=0.15)
    parser.add_argument("--max-eval-records", type=int, default=2000)
    parser.add_argument("--generation-eval-records", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--min-task-samples", type=int, default=8)
    parser.add_argument("--no-balance-tasks", action="store_true")
    parser.add_argument(
        "--require-all-supervised-tasks",
        action="store_true",
        help="Fail unless all seven task types are present in train/validation/test.",
    )

    # Tokenizer.
    parser.add_argument("--retrain-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=50_000)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=2)

    # GloVe semantic bootstrap.
    parser.add_argument("--no-glove-bootstrap", action="store_true")
    parser.add_argument("--glove-epochs", type=int, default=1)
    parser.add_argument("--glove-batch-size", type=int, default=128)
    parser.add_argument("--glove-max-tokens", type=int, default=0, help="0 means all vocabulary matches.")
    parser.add_argument("--glove-learning-rate", type=float, default=1e-4)

    # Raw-text self-supervised denoising.
    parser.add_argument("--raw-pretrain-epochs", type=int, default=2)
    parser.add_argument("--raw-segments-per-epoch", type=int, default=20_000, help="0 means all discovered segments.")
    parser.add_argument("--raw-max-segments", type=int, default=100_000, help="0 means no corpus loading cap.")
    parser.add_argument("--raw-validation-fraction", type=float, default=0.02)
    parser.add_argument("--raw-test-fraction", type=float, default=0.02)
    parser.add_argument("--raw-corruption-probability", type=float, default=0.15)
    parser.add_argument("--raw-min-chars", type=int, default=40)
    parser.add_argument("--raw-chunk-chars", type=int, default=2000)
    parser.add_argument("--raw-max-length", type=int, default=256)

    parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "Warm-start from an existing SLAI LanguageTransformer checkpoint. "
            "Automatic GloVe reinitialization is skipped for warm starts unless --glove is explicitly provided."
        ),
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainerConfig:
    config = TrainerConfig(
        data_paths=tuple(args.data),
        raw_text_paths=tuple(args.raw_text),
        glove_path=str(args.glove) if args.glove else None,
        output_dir=str(args.output_dir),
        report_dir=str(args.report_dir),
        epochs=int(args.epochs),
        gradient_accumulation=int(args.gradient_accumulation),
        learning_rate=float(args.learning_rate),
        min_learning_rate=float(args.min_learning_rate),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        clip_grad=float(args.clip_grad),
        label_smoothing=float(args.label_smoothing),
        source_max_length=int(args.source_max_length),
        target_max_length=int(args.target_max_length),
        embedding_max_length=int(args.embedding_max_length),
        raw_max_length=int(args.raw_max_length),
        validation_fraction=float(args.validation_fraction),
        test_fraction=float(args.test_fraction),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        seed=int(args.seed),
        device=str(args.device),
        embedding_margin=float(args.embedding_margin),
        reranking_margin=float(args.reranking_margin),
        max_eval_records=int(args.max_eval_records),
        generation_eval_records=int(args.generation_eval_records),
        log_every=int(args.log_every),
        min_task_samples=int(args.min_task_samples),
        balance_tasks=not bool(args.no_balance_tasks),
        retrain_tokenizer=bool(args.retrain_tokenizer),
        tokenizer_vocab_size=int(args.tokenizer_vocab_size),
        tokenizer_min_frequency=int(args.tokenizer_min_frequency),
        init_from=str(args.init_from) if args.init_from else None,
        require_all_supervised_tasks=bool(args.require_all_supervised_tasks),
        glove_bootstrap=not bool(args.no_glove_bootstrap),
        glove_epochs=int(args.glove_epochs),
        glove_batch_size=int(args.glove_batch_size),
        glove_max_tokens=int(args.glove_max_tokens),
        glove_learning_rate=float(args.glove_learning_rate),
        raw_pretrain_epochs=int(args.raw_pretrain_epochs),
        raw_segments_per_epoch=int(args.raw_segments_per_epoch),
        raw_max_segments=int(args.raw_max_segments),
        raw_validation_fraction=float(args.raw_validation_fraction),
        raw_test_fraction=float(args.raw_test_fraction),
        raw_corruption_probability=float(args.raw_corruption_probability),
        raw_min_chars=int(args.raw_min_chars),
        raw_chunk_chars=int(args.raw_chunk_chars),
    )
    validate_config(config)
    return config


def validate_config(config: TrainerConfig) -> None:
    integer_positive = {
        "epochs": config.epochs,
        "gradient_accumulation": config.gradient_accumulation,
        "source_max_length": config.source_max_length,
        "target_max_length": config.target_max_length,
        "embedding_max_length": config.embedding_max_length,
        "raw_max_length": config.raw_max_length,
        "patience": config.patience,
        "min_task_samples": config.min_task_samples,
        "tokenizer_vocab_size": config.tokenizer_vocab_size,
        "tokenizer_min_frequency": config.tokenizer_min_frequency,
        "glove_batch_size": config.glove_batch_size,
        "raw_min_chars": config.raw_min_chars,
        "raw_chunk_chars": config.raw_chunk_chars,
    }
    for name, value in integer_positive.items():
        if value <= 0:
            raise LantraTrainingError(f"{name} must be > 0, got {value}.")
    nonnegative = {
        "glove_epochs": config.glove_epochs,
        "glove_max_tokens": config.glove_max_tokens,
        "raw_pretrain_epochs": config.raw_pretrain_epochs,
        "raw_segments_per_epoch": config.raw_segments_per_epoch,
        "raw_max_segments": config.raw_max_segments,
        "warmup_steps": config.warmup_steps,
        "max_eval_records": config.max_eval_records,
        "generation_eval_records": config.generation_eval_records,
        "log_every": config.log_every,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise LantraTrainingError(f"{name} cannot be negative.")
    if config.learning_rate <= 0.0 or config.glove_learning_rate <= 0.0:
        raise LantraTrainingError("learning rates must be > 0.")
    if not (0.0 <= config.min_learning_rate <= config.learning_rate):
        raise LantraTrainingError("min_learning_rate must be between 0 and learning_rate.")
    if config.weight_decay < 0.0:
        raise LantraTrainingError("weight_decay cannot be negative.")
    if config.clip_grad <= 0.0:
        raise LantraTrainingError("clip_grad must be > 0.")
    if not (0.0 <= config.label_smoothing < 1.0):
        raise LantraTrainingError("label_smoothing must be >= 0 and < 1.")
    if config.min_delta < 0.0:
        raise LantraTrainingError("min_delta cannot be negative.")
    if config.embedding_margin <= 0.0 or config.reranking_margin <= 0.0:
        raise LantraTrainingError("Pairwise margins must be > 0.")
    if not (0.0 <= config.raw_validation_fraction < 0.5):
        raise LantraTrainingError("raw_validation_fraction must be >= 0 and < 0.5.")
    if not (0.0 <= config.raw_test_fraction < 0.5):
        raise LantraTrainingError("raw_test_fraction must be >= 0 and < 0.5.")
    if config.raw_validation_fraction + config.raw_test_fraction >= 1.0:
        raise LantraTrainingError("raw_validation_fraction + raw_test_fraction must be < 1.0.")
    if not (0.0 < config.raw_corruption_probability < 1.0):
        raise LantraTrainingError("raw_corruption_probability must be > 0 and < 1.")
    if config.raw_chunk_chars < config.raw_min_chars:
        raise LantraTrainingError("raw_chunk_chars must be >= raw_min_chars.")
    if 0 < config.max_eval_records < len(SUPPORTED_TASKS) and config.require_all_supervised_tasks:
        raise LantraTrainingError(
            f"When requiring all tasks, max_eval_records must be 0 or >= {len(SUPPORTED_TASKS)}."
        )


def build_report(
    *,
    run_id: str,
    started_at: str,
    elapsed_seconds: float,
    config: TrainerConfig,
    dataset: Optional[DatasetSplit],
    raw_corpus: RawCorpus,
    glove_asset: Optional[GloveAsset],
    tokenizer: LanguageTokenizer,
    device: str,
    training_result: Mapping[str, Any],
    supervised_coverage: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    tokenizer_stats = tokenizer.stats().to_dict() if hasattr(tokenizer, "stats") else {}
    dataset_payload: Optional[Dict[str, Any]] = None
    if dataset is not None:
        dataset_payload = {
            "files": list(dataset.files),
            "fingerprint_sha256": dataset.fingerprint,
            "duplicates_removed": dataset.duplicate_count,
            "counts": dataset.counts(),
            "coverage": dict(supervised_coverage or {}),
        }
    return {
        "schema": "slai.lantra.training-report.v3",
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "status": "completed",
        "trainer": {
            "module": "train_lantra",
            "training_pipeline": [
                "bpe_tokenizer_bootstrap",
                "glove_semantic_bootstrap",
                "raw_text_denoising_pretraining",
                "supervised_multitask_specialization",
            ],
            "direct_third_party_imports": [],
            "slai_components": [
                "src.agents.language.modules.language_tokenizer.LanguageTokenizer",
                "src.agents.language.modules.language_transformer.LanguageTransformer",
                "logs.logger",
            ],
            "transitive_runtime_note": (
                "LanguageTransformer/LanguageTokenizer retain SLAI's own PyTorch/regex dependencies; "
                "this trainer does not import those packages directly."
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": device,
        },
        "config": config.to_dict(),
        "objective_contract": OBJECTIVE_CONTRACT,
        "supervised_dataset": dataset_payload,
        "raw_corpus": raw_corpus.to_dict(),
        "glove": glove_asset.to_dict() if glove_asset is not None else None,
        "tokenizer": tokenizer_stats,
        "training": dict(training_result),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    started_clock = time.perf_counter()
    started_at = utc_now()
    run_id = safe_run_id()

    try:
        args = parse_args(argv)
        config = config_from_args(args)
        seed_runtime(config.seed)

        # --------------------------------------------------------------
        # Discover optional raw corpus and optional supervised corpus.
        # --------------------------------------------------------------
        raw_files = discover_raw_text_files(config.raw_text_paths)
        raw_corpus = load_raw_corpus(raw_files, config) if raw_files else empty_raw_corpus()
        raw_corpus = write_raw_corpus_manifest(raw_corpus, Path(config.report_dir), run_id)
        LOGGER.info("LANTRA raw/document corpus files: %s", list(raw_corpus.files))
        if raw_corpus.segments:
            PRINTER.pretty("LANTRA RAW CORPUS", raw_corpus.to_dict(), "success")
        else:
            PRINTER.status("LANTRA RAW", "No raw-text corpus found; raw-text pretraining will be skipped.", "info")

        supervised_files = discover_data_files(config.data_paths)
        dataset: Optional[DatasetSplit] = None
        supervised_coverage: Optional[Dict[str, Any]] = None
        if supervised_files:
            LOGGER.info("LANTRA supervised data files: %s", [str(path) for path in supervised_files])
            examples, duplicate_count, fingerprint = load_examples(supervised_files)
            dataset = split_examples(
                examples,
                validation_fraction=config.validation_fraction,
                test_fraction=config.test_fraction,
                seed=config.seed,
                files=supervised_files,
                fingerprint=fingerprint,
                duplicate_count=duplicate_count,
            )
            supervised_coverage = validate_task_coverage(
                dataset,
                config.min_task_samples,
                require_all_tasks=config.require_all_supervised_tasks,
            )
            PRINTER.pretty(
                "LANTRA SUPERVISED DATASET",
                {
                    "fingerprint_sha256": dataset.fingerprint,
                    "duplicates_removed": dataset.duplicate_count,
                    "counts": dataset.counts(),
                    "coverage": supervised_coverage,
                },
                "success",
            )
        else:
            PRINTER.status(
                "LANTRA SUPERVISED",
                "No supervised LANTRA task corpus found; task specialization will be skipped.",
                "info",
            )

        # --------------------------------------------------------------
        # Phase 0: SLAI BPE tokenizer bootstrap.
        # --------------------------------------------------------------
        tokenizer = initialize_tokenizer(config, dataset, raw_corpus.train_segments)
        device = resolve_device(config.device)
        LOGGER.info("Resolved LANTRA device: %s", device)
        model = initialize_model(config, tokenizer, device)

        PRINTER.pretty(
            "LANTRA MODEL",
            {
                "device": device,
                "vocab_size": len(tokenizer.vocab),
                "tokenizer_trained": bool(tokenizer.is_trained),
                "model_stats": model.stats().to_dict(),
                "supported_tasks": list(SUPPORTED_TASKS),
            },
            "success",
        )

        phases: Dict[str, Any] = {
            "bpe": {
                "status": "completed",
                "vocab_size": len(tokenizer.vocab),
                "tokenizer_stats": tokenizer.stats().to_dict() if hasattr(tokenizer, "stats") else {},
            }
        }
        glove_asset: Optional[GloveAsset] = None
        genuine_training_signal = False

        # --------------------------------------------------------------
        # Phase 1: GloVe embedding initialization + semantic distillation.
        # --------------------------------------------------------------
        glove_path = discover_glove_path(config, int(model.config.d_model))
        if glove_path is not None:
            glove_asset = load_glove_asset(
                glove_path,
                tokenizer,
                model_dimension=int(model.config.d_model),
                max_tokens=config.glove_max_tokens,
            )
            initialization = initialize_embeddings_from_glove(model, glove_asset)
            glove_result = semantic_bootstrap_train(
                config,
                tokenizer,
                model,
                glove_asset,
                device,
                run_id,
            )
            phases["glove"] = {
                "initialization": initialization,
                **glove_result,
            }
            genuine_training_signal = genuine_training_signal or glove_result.get("status") == "completed"
            PRINTER.pretty("LANTRA GLOVE ASSET", glove_asset.to_dict(), "success")
        else:
            phases["glove"] = {"status": "skipped", "reason": "no compatible/configured GloVe resource"}
            PRINTER.status("LANTRA GLOVE", "No compatible GloVe bootstrap resource selected.", "info")

        # --------------------------------------------------------------
        # Phase 2: raw-text denoising self-supervised pretraining.
        # --------------------------------------------------------------
        model, raw_result = raw_text_pretrain(
            config,
            raw_corpus,
            tokenizer,
            model,
            device,
            run_id,
        )
        phases["raw_text"] = raw_result
        genuine_training_signal = genuine_training_signal or raw_result.get("status") == "completed"

        # --------------------------------------------------------------
        # Phase 3: supervised specialization when real labelled data exists.
        # --------------------------------------------------------------
        if dataset is not None:
            supervised_result = train(config, dataset, tokenizer, model, device, run_id)
            phases["supervised"] = supervised_result
            genuine_training_signal = True
            training_result: Dict[str, Any] = {
                "mode": "staged_with_supervised_specialization",
                "phases": phases,
                "final_checkpoint": supervised_result["checkpoints"]["final"],
                "model_stats": supervised_result["model_stats"],
            }
        else:
            phases["supervised"] = {"status": "skipped", "reason": "no supervised task corpus"}
            if not genuine_training_signal:
                raise LantraTrainingError(
                    "LANTRA found a valid BPE tokenizer but no genuine model-training signal. "
                    "Provide/restore a compatible GloVe JSON, add raw text via --raw-text/SLAI_LANTRA_RAW_TEXT, "
                    "or add supervised task data. BPE vocabulary alone does not train Transformer weights."
                )
            final_path = Path(config.output_dir) / f"lantra_{run_id}.pt"
            final_saved = save_phase_checkpoint(
                model,
                None,
                final_path,
                run_id=run_id,
                phase="bootstrap_pretrained_final",
                metadata={
                    "phases": phases,
                    "capability_note": (
                        "No supervised task corpus was supplied. This checkpoint contains semantic/bootstrap "
                        "and/or raw-text pretraining, not validated seven-task specialization."
                    ),
                },
            )
            training_result = {
                "mode": "bootstrap_pretraining_only",
                "phases": phases,
                "final_checkpoint": final_saved,
                "model_stats": model.stats().to_dict(),
            }

        elapsed = time.perf_counter() - started_clock
        report = build_report(
            run_id=run_id,
            started_at=started_at,
            elapsed_seconds=elapsed,
            config=config,
            dataset=dataset,
            raw_corpus=raw_corpus,
            glove_asset=glove_asset,
            tokenizer=tokenizer,
            device=device,
            training_result=training_result,
            supervised_coverage=supervised_coverage,
        )
        report_path = Path(config.report_dir) / f"lantra_training_{run_id}.json"
        atomic_json_write(report_path, report)

        PRINTER.pretty(
            "LANTRA COMPLETE",
            {
                "run_id": run_id,
                "elapsed_seconds": round(elapsed, 3),
                "mode": training_result["mode"],
                "final_checkpoint": training_result["final_checkpoint"],
                "report": str(report_path),
                "phases": {
                    name: result.get("status", "completed") if isinstance(result, Mapping) else "completed"
                    for name, result in phases.items()
                },
            },
            "success",
        )
        LOGGER.info("LANTRA staged training completed successfully: %s", training_result["final_checkpoint"])
        return 0

    except LantraTrainingError as exc:
        LOGGER.error("LANTRA training configuration/data failure: %s", exc)
        PRINTER.status("LANTRA", str(exc), "error")
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("LANTRA training interrupted by user.")
        PRINTER.status("LANTRA", "Training interrupted; phase checkpoints are retained when available.", "warning")
        return 130
    except Exception as exc:
        LOGGER.exception("Unexpected LANTRA training failure: %s", exc)
        PRINTER.pretty(
            "LANTRA FAILURE",
            {
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "error",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
