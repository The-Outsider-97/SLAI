"""Validated teacher-supervision builder for LANTRA's seven existing tasks.

The builder supplements, rather than replaces, the deterministic SLAI curriculum.
It emits JSONL records compatible with train_lantra.py and never changes the task
registry. External providers are optional: deterministic source-grounded SLAI
fallback examples keep the build resumable and complete when every provider is
unavailable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from src.training.corpus_dedup import atomic_write_json, normalize_text_for_dedup
from src.training.enrichment_contracts import (
    SUPPORTED_TASKS,
    CurriculumQualityError,
    SourceDocument,
    sha256_payload,
)
from src.training.llm_providers import (
    LLMCache,
    LLMProviderResponseError,
    ProviderPool,
    parse_json_object,
    providers_from_config,
)


LOGGER = logging.getLogger("lantra_supervised_builder")
ROOT = Path(__file__).resolve().parents[2]
VALID_SPLITS = ("train", "validation", "test")
ORIGIN = "validated_teacher_supervision"


@dataclass(frozen=True)
class SupervisedBuildResult:
    output_dir: str
    manifest_path: str
    coverage: Mapping[str, Any]
    providers: tuple[str, ...]
    external_calls_accepted: int
    deterministic_fallback_accepted: int
    rejected: int
    resumed_records: int


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _safe_text(value: Any, *, minimum: int = 1, maximum: int = 20000) -> str:
    if not isinstance(value, str):
        raise CurriculumQualityError("Expected text value.")
    text = " ".join(value.split())
    if len(text) < minimum:
        raise CurriculumQualityError("Text value is empty or too short.")
    if len(text) > maximum:
        raise CurriculumQualityError("Text value exceeds configured safety limit.")
    return text


def _sentences(text: str) -> list[str]:
    normalized = " ".join(str(text).split())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if len(part.split()) >= 5]


def _excerpt(document: SourceDocument, maximum: int) -> str:
    text = " ".join(document.text.split())
    if len(text) <= maximum:
        return text
    cut = text[:maximum]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary >= maximum // 2:
        return cut[: boundary + 1]
    return cut.rstrip()


def _topic_label(document: SourceDocument) -> str:
    text = f"{document.title or ''} {document.text[:5000]}".casefold()
    groups = {
        "history": ("history", "century", "empire", "war", "ancient", "medieval"),
        "social_science": ("society", "social", "politic", "law", "economic", "psycholog"),
        "humanities": ("literature", "philosophy", "art", "language", "religion"),
        "natural_science": ("physics", "chemistry", "biology", "astronomy", "geology", "ecology"),
        "technology_engineering": ("computer", "engineering", "technology", "software", "machine"),
        "geography": ("geography", "region", "country", "river", "climate", "population"),
    }
    scores = {
        label: sum(text.count(signal) for signal in signals)
        for label, signals in groups.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "general_reference"


_INTRALINGUAL = {
    "utilize": "use",
    "approximately": "about",
    "commence": "begin",
    "terminate": "end",
    "demonstrate": "show",
    "subsequent": "later",
    "prior": "earlier",
    "numerous": "many",
    "individuals": "people",
    "whilst": "while",
    "amongst": "among",
    "therefore": "so",
}


def _plain_english(text: str) -> str:
    result = text
    for source, target in _INTRALINGUAL.items():
        result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.I)
    result = result.replace(";", ".")
    return " ".join(result.split())


def _metadata(
    document_ids: Sequence[str],
    *,
    derivation: str,
    provider: str,
    model: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "origin": ORIGIN,
        "synthetic": True,
        "teacher_generated": True,
        "teacher_validated": True,
        "source_grounded": True,
        "derivation": derivation,
        "teacher_provider": provider,
        "source_document_ids": list(dict.fromkeys(str(value) for value in document_ids)),
    }
    if model:
        payload["teacher_model"] = model
    if extra:
        payload.update(dict(extra))
    return payload


def deterministic_examples(
    document: SourceDocument,
    *,
    negative_document: SourceDocument | None,
    max_source_characters: int,
) -> list[dict[str, Any]]:
    """Create conservative source-derived fallback coverage for all seven tasks."""
    excerpt = _excerpt(document, max_source_characters)
    sentences = _sentences(excerpt)
    if len(sentences) < 2:
        return []
    first = sentences[0]
    second = sentences[1]
    summary = " ".join(sentences[: min(3, len(sentences))])
    title = document.title or "the source passage"
    common_meta = {
        "source_title": document.title,
        "fallback": True,
    }
    records: list[dict[str, Any]] = [
        {
            "task": "generation",
            "split": document.split,
            "input": f"Continue the source-grounded explanation about {title}: {first}",
            "target": second,
            "metadata": _metadata(
                [document.document_id],
                derivation="source_continuation",
                provider="slai_deterministic",
                extra=common_meta,
            ),
        },
        {
            "task": "classification",
            "split": document.split,
            "input": excerpt[: min(1800, len(excerpt))],
            "label": _topic_label(document),
            "metadata": _metadata(
                [document.document_id],
                derivation="deterministic_topic_classification",
                provider="slai_deterministic",
                extra=common_meta,
            ),
        },
        {
            "task": "translation",
            "split": document.split,
            "source": first,
            "target": _plain_english(first),
            "source_language": "English (source register)",
            "target_language": "English (plain register)",
            "metadata": _metadata(
                [document.document_id],
                derivation="intralingual_plain_english_translation",
                provider="slai_deterministic",
                extra={**common_meta, "translation_mode": "intralingual"},
            ),
        },
        {
            "task": "summarization",
            "split": document.split,
            "input": excerpt,
            "target": summary,
            "metadata": _metadata(
                [document.document_id],
                derivation="extractive_source_summary",
                provider="slai_deterministic",
                extra=common_meta,
            ),
        },
        {
            "task": "dialogue",
            "split": document.split,
            "input": f"User: What does the source say about {title}?\nAssistant:",
            "target": summary,
            "metadata": _metadata(
                [document.document_id],
                derivation="source_grounded_dialogue",
                provider="slai_deterministic",
                extra=common_meta,
            ),
        },
    ]

    positive = second
    if negative_document is not None:
        negative_excerpt = _excerpt(negative_document, min(1200, max_source_characters))
        negative_sentences = _sentences(negative_excerpt)
        negative = negative_sentences[0] if negative_sentences else negative_excerpt
        negative_ids = [document.document_id, negative_document.document_id]
    else:
        negative = sentences[-1] if len(sentences) > 2 else first
        negative_ids = [document.document_id]

    for task in ("embedding", "reranking"):
        records.append(
            {
                "task": task,
                "split": document.split,
                "anchor" if task == "embedding" else "query": first,
                "positive": positive,
                "negative": negative,
                "metadata": _metadata(
                    negative_ids,
                    derivation="source_contrastive_pair",
                    provider="slai_deterministic",
                    extra=common_meta,
                ),
            }
        )
    return records


def _llm_prompt(document: SourceDocument, excerpt: str) -> tuple[str, str]:
    system = (
        "You create grounded training examples for SLAI LANTRA. Return strict JSON only. "
        "Use only facts present in the supplied source. Do not invent citations or metadata. "
        "Create one useful example for each requested existing LANTRA task."
    )
    prompt = json.dumps(
        {
            "instruction": (
                "Return an object with key examples containing exactly seven objects, one for each "
                "task: generation, classification, translation, summarization, dialogue, embedding, reranking. "
                "Keep every answer grounded in SOURCE. For translation use a faithful translation into Spanish. "
                "For embedding/reranking use anchor-or-query, positive, and negative fields; the negative may be "
                "a plausible but source-unsupported distractor and must not be presented as factual. "
                "For all other tasks use the field names compatible with the task."
            ),
            "source_title": document.title,
            "source": excerpt,
            "schemas": {
                "generation": ["task", "input", "target"],
                "classification": ["task", "input", "label"],
                "translation": ["task", "source", "target", "source_language", "target_language"],
                "summarization": ["task", "input", "target"],
                "dialogue": ["task", "input", "target"],
                "embedding": ["task", "anchor", "positive", "negative"],
                "reranking": ["task", "query", "positive", "negative"],
            },
        },
        ensure_ascii=False,
    )
    return system, prompt


def _second_model_accepts(
    pool: ProviderPool,
    *,
    first_provider: str,
    source_title: str | None,
    source_text: str,
    candidate_payload: Mapping[str, Any],
) -> tuple[bool, str]:
    """Optionally ask a different provider to critique one generated bundle.

    Failure or malformed critique is non-fatal: deterministic SLAI validation still
    runs. A well-formed explicit rejection blocks the external bundle.
    """
    system = (
        "You are a strict training-data critic. Return JSON only. Decide whether the "
        "candidate examples are grounded in the supplied source, non-trivial, and compatible "
        "with their task schemas. Do not rewrite the examples."
    )
    prompt = json.dumps(
        {
            "source_title": source_title,
            "source": source_text,
            "candidate": candidate_payload,
            "required_output": {"accept": True, "issues": ["short reason if any"]},
        },
        ensure_ascii=False,
    )
    critique = pool.complete(system=system, prompt=prompt, exclude=(first_provider,))
    if critique is None:
        return True, "no_secondary_provider_available"
    try:
        parsed = parse_json_object(critique.content)
    except LLMProviderResponseError as exc:
        LOGGER.warning("Secondary provider critique was malformed: %s", exc)
        return True, "malformed_secondary_critique"
    accepted = parsed.get("accept")
    if not isinstance(accepted, bool):
        LOGGER.warning("Secondary provider critique omitted boolean accept; continuing with SLAI validation.")
        return True, "secondary_critique_missing_boolean"
    issues = parsed.get("issues", [])
    reason = "; ".join(str(item) for item in issues[:5]) if isinstance(issues, Sequence) and not isinstance(issues, (str, bytes, bytearray)) else ""
    return accepted, reason or ("accepted" if accepted else "rejected")


def _normalise_candidate(
    raw: Mapping[str, Any],
    *,
    document: SourceDocument,
    provider: str,
    model: str,
) -> dict[str, Any]:
    task = str(raw.get("task", "")).strip().casefold()
    if task not in SUPPORTED_TASKS:
        raise CurriculumQualityError(f"Unsupported generated task: {task!r}.")
    record: dict[str, Any] = {"task": task, "split": document.split}
    if task == "generation":
        record["input"] = _safe_text(raw.get("input"), minimum=8)
        record["target"] = _safe_text(raw.get("target"), minimum=5)
    elif task == "classification":
        record["input"] = _safe_text(raw.get("input"), minimum=8)
        record["label"] = _safe_text(raw.get("label"), maximum=160)
    elif task == "translation":
        record["source"] = _safe_text(raw.get("source"), minimum=5)
        record["target"] = _safe_text(raw.get("target"), minimum=5)
        record["source_language"] = _safe_text(raw.get("source_language", "English"), maximum=80)
        record["target_language"] = _safe_text(raw.get("target_language", "Spanish"), maximum=80)
    elif task in {"summarization", "dialogue"}:
        record["input"] = _safe_text(raw.get("input"), minimum=8)
        record["target"] = _safe_text(raw.get("target"), minimum=5)
    elif task == "embedding":
        record["anchor"] = _safe_text(raw.get("anchor"), minimum=5)
        record["positive"] = _safe_text(raw.get("positive"), minimum=5)
        record["negative"] = _safe_text(raw.get("negative"), minimum=5)
    elif task == "reranking":
        record["query"] = _safe_text(raw.get("query"), minimum=5)
        record["positive"] = _safe_text(raw.get("positive"), minimum=5)
        record["negative"] = _safe_text(raw.get("negative"), minimum=5)

    record["metadata"] = _metadata(
        [document.document_id],
        derivation="external_llm_source_grounded",
        provider=provider,
        model=model,
        extra={"source_title": document.title, "fallback": False},
    )
    return record


def _text_fields(record: Mapping[str, Any]) -> list[str]:
    return [
        str(record[key])
        for key in ("input", "source", "target", "label", "anchor", "query", "positive")
        if isinstance(record.get(key), str)
    ]


def _grounding_score(record: Mapping[str, Any], source: str) -> float:
    source_tokens = set(normalize_text_for_dedup(source).split())
    candidate_tokens: set[str] = set()
    for value in _text_fields(record):
        candidate_tokens.update(normalize_text_for_dedup(value).split())
    if not candidate_tokens:
        return 0.0
    return len(source_tokens & candidate_tokens) / len(candidate_tokens)


def validate_record(
    record: Mapping[str, Any],
    *,
    source_text: str,
    max_record_chars: int = 20000,
) -> dict[str, Any]:
    task = str(record.get("task", "")).strip().casefold()
    if task not in SUPPORTED_TASKS:
        raise CurriculumQualityError("Generated record uses an unsupported task.")
    split = str(record.get("split", "")).strip().casefold()
    if split not in VALID_SPLITS:
        raise CurriculumQualityError("Generated record has an invalid split.")
    payload_chars = len(json.dumps(record, ensure_ascii=False))
    if payload_chars > max_record_chars:
        raise CurriculumQualityError("Generated record is too large.")

    required = {
        "generation": ("input", "target"),
        "classification": ("input", "label"),
        "translation": ("source", "target"),
        "summarization": ("input", "target"),
        "dialogue": ("input", "target"),
        "embedding": ("anchor", "positive", "negative"),
        "reranking": ("query", "positive", "negative"),
    }[task]
    for field in required:
        _safe_text(record.get(field), minimum=1, maximum=max_record_chars)

    serialised = json.dumps(
        {key: record.get(key) for key in required},
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(set(normalize_text_for_dedup(serialised).split())) < 4:
        raise CurriculumQualityError("Generated record is trivial or excessively repetitive.")

    grounding = _grounding_score(record, source_text)
    # Translation targets and classification labels naturally overlap less; input
    # grounding is still measured across the whole record.
    minimum_grounding = 0.08 if task in {"translation", "classification"} else 0.12
    if grounding < minimum_grounding:
        raise CurriculumQualityError(
            f"Generated {task} record is insufficiently source-grounded ({grounding:.3f})."
        )

    normalized = dict(record)
    metadata = normalized.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise CurriculumQualityError("Generated metadata must be an object.")
    metadata = dict(metadata)
    metadata["grounding_score"] = grounding
    metadata["teacher_validated"] = True
    metadata["source_grounded"] = True
    normalized["metadata"] = metadata
    basis = {
        "task": task,
        "split": split,
        "fields": {key: normalized.get(key) for key in required},
        "source_document_ids": metadata.get("source_document_ids", []),
    }
    normalized["id"] = str(normalized.get("id") or sha256_payload(basis)[:24])
    return normalized


def _quality_agent_accepts(quality_agent: Any, record: Mapping[str, Any]) -> bool:
    if quality_agent is None:
        return True
    result = quality_agent.evaluate_batch(
        [record],
        dataset_id="lantra_teacher_supervision",
        source_id=str(record.get("metadata", {}).get("teacher_provider", "unknown")),
        batch_id=str(record.get("id", "candidate")),
        context={"use_case": "lantra_supervised_training"},
    )
    verdict = str(result.get("verdict", result.get("decision", "warn"))).casefold()
    return verdict not in {"reject", "rejected", "block", "blocked", "fail", "failed"}


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _existing_records(output_dir: Path) -> tuple[set[str], Counter[tuple[str, str]]]:
    ids: set[str] = set()
    coverage: Counter[tuple[str, str]] = Counter()
    if not output_dir.exists():
        return ids, coverage
    for path in sorted(output_dir.rglob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    value = json.loads(stripped)
                    if not isinstance(value, Mapping):
                        continue
                    record_id = str(value.get("id", "")).strip()
                    task = str(value.get("task", "")).strip().casefold()
                    split = str(value.get("split", "")).strip().casefold()
                    if record_id:
                        ids.add(record_id)
                    if task in SUPPORTED_TASKS and split in VALID_SPLITS:
                        coverage[(task, split)] += 1
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring unreadable prior generated file %s: %s", path, exc)
    return ids, coverage


def _coverage_summary(
    coverage: Mapping[tuple[str, str], int],
    rejected: Mapping[str, int],
) -> dict[str, Any]:
    return {
        task: {
            "examples": sum(int(coverage.get((task, split), 0)) for split in VALID_SPLITS),
            "valid": {
                split: int(coverage.get((task, split), 0))
                for split in VALID_SPLITS
            },
            "rejected": int(rejected.get(task, 0)),
        }
        for task in SUPPORTED_TASKS
    }


def _needed(
    coverage: Mapping[tuple[str, str], int],
    *,
    train_min: int,
    validation_min: int,
    test_min: int,
) -> bool:
    minima = {"train": train_min, "validation": validation_min, "test": test_min}
    return any(
        int(coverage.get((task, split), 0)) < minima[split]
        for task in SUPPORTED_TASKS
        for split in VALID_SPLITS
    )


def build_supervised_dataset(
    documents: Sequence[SourceDocument],
    *,
    pipeline_config: Mapping[str, Any],
    quality_agent: Any = None,
    enable_external_llm: bool = True,
    force: bool = False,
) -> SupervisedBuildResult:
    cfg = pipeline_config.get("supervised_generation", {})
    if not isinstance(cfg, Mapping):
        raise ValueError("lantra_pipeline.supervised_generation must be a mapping.")
    output_dir = _resolve_path(str(cfg.get("output_dir", "data/processed/lantra/supervised/generated")))
    state_path = _resolve_path(str(cfg.get("state_path", "data/processed/lantra/supervised/.build_state.json")))
    manifest_path = _resolve_path(str(cfg.get("manifest_path", "data/processed/lantra/supervised/generated_manifest.json")))
    cache_path = _resolve_path(str(cfg.get("cache_path", "data/processed/lantra/supervised/.llm_cache.sqlite3")))
    train_min = int(cfg.get("minimum_train_examples_per_task", 8))
    validation_min = int(cfg.get("minimum_validation_examples_per_task", 1))
    test_min = int(cfg.get("minimum_test_examples_per_task", 1))
    max_source_chars = int(cfg.get("max_source_characters", 6000))

    if force and output_dir.exists():
        for path in output_dir.glob("lantra_teacher_*.jsonl"):
            path.unlink(missing_ok=True)

    existing_ids, coverage = _existing_records(output_dir)
    rejected: Counter[str] = Counter()
    resumed_records = len(existing_ids)
    state: dict[str, Any] = {
        "schema": "slai.lantra.supervised-build-state.v1",
        "processed_documents": [],
        "accepted_ids": sorted(existing_ids),
    }
    if state_path.is_file() and not force:
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping) and loaded.get("schema") == state["schema"]:
                state.update(dict(loaded))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Ignoring malformed supervised build state %s.", state_path)
    processed = {str(value) for value in state.get("processed_documents", ())}

    providers = providers_from_config(cfg) if enable_external_llm and bool(cfg.get("external_llm_enabled", True)) else []
    multi_model_critique = bool(cfg.get("multi_model_critique", False)) and len(providers) > 1
    critique_rejections = 0
    external_accepted = 0
    deterministic_accepted = 0

    split_documents: dict[str, list[SourceDocument]] = defaultdict(list)
    for document in documents:
        if document.split in VALID_SPLITS and len(_sentences(document.text)) >= 2:
            split_documents[str(document.split)].append(document)

    minima = {"train": train_min, "validation": validation_min, "test": test_min}

    with LLMCache(cache_path) as cache:
        pool = ProviderPool(
            providers,
            cache=cache,
            retry_limit=int(cfg.get("provider_retry_limit", 2)),
            backoff_seconds=float(cfg.get("provider_backoff_seconds", 2.0)),
        )

        for split in VALID_SPLITS:
            docs = split_documents.get(split, [])
            if not docs:
                continue
            for index, document in enumerate(docs):
                if not _needed(
                    coverage,
                    train_min=train_min,
                    validation_min=validation_min,
                    test_min=test_min,
                ):
                    break

                negative = docs[(index + 1) % len(docs)] if len(docs) > 1 else None
                excerpt = _excerpt(document, max_source_chars)

                # External enhancement: one provider call requests all seven tasks.
                # Invalid/missing provider output never aborts the curriculum build.
                if document.document_id not in processed and providers:
                    system, prompt = _llm_prompt(document, excerpt)
                    result = pool.complete(system=system, prompt=prompt)
                    if result is not None:
                        try:
                            parsed = parse_json_object(result.content)
                            if multi_model_critique:
                                critique_ok, critique_reason = _second_model_accepts(
                                    pool,
                                    first_provider=result.provider,
                                    source_title=document.title,
                                    source_text=excerpt,
                                    candidate_payload=parsed,
                                )
                                if not critique_ok:
                                    critique_rejections += 1
                                    LOGGER.info(
                                        "Secondary-model critique rejected bundle for source %s: %s",
                                        document.document_id,
                                        critique_reason,
                                    )
                                    raise CurriculumQualityError("secondary model rejected generated bundle")
                            raw_examples = parsed.get("examples", [])
                            if not isinstance(raw_examples, Sequence) or isinstance(raw_examples, (str, bytes, bytearray)):
                                raise LLMProviderResponseError("examples must be an array.")
                            for raw in raw_examples:
                                if not isinstance(raw, Mapping):
                                    continue
                                task = str(raw.get("task", "")).strip().casefold()
                                if task not in SUPPORTED_TASKS:
                                    rejected[task or "unknown"] += 1
                                    continue
                                if coverage[(task, split)] >= minima[split]:
                                    continue
                                try:
                                    candidate = _normalise_candidate(
                                        raw,
                                        document=document,
                                        provider=result.provider,
                                        model=result.model,
                                    )
                                    validated = validate_record(
                                        candidate,
                                        source_text=excerpt,
                                    )
                                    if validated["id"] in existing_ids:
                                        continue
                                    if not _quality_agent_accepts(quality_agent, validated):
                                        rejected[task] += 1
                                        continue
                                    _append_jsonl(
                                        output_dir / f"lantra_teacher_{task}.jsonl",
                                        validated,
                                    )
                                    existing_ids.add(str(validated["id"]))
                                    coverage[(task, split)] += 1
                                    external_accepted += 1
                                except (CurriculumQualityError, ValueError) as exc:
                                    rejected[task] += 1
                                    LOGGER.debug("Rejected %s LLM example: %s", task, exc)
                        except (LLMProviderResponseError, CurriculumQualityError) as exc:
                            LOGGER.warning(
                                "External LLM bundle rejected for source %s: %s",
                                document.document_id,
                                exc,
                            )

                # SLAI-only deterministic fallback fills any still-missing task
                # coverage while preserving the source document's existing split.
                for candidate in deterministic_examples(
                    document,
                    negative_document=negative,
                    max_source_characters=max_source_chars,
                ):
                    task = str(candidate["task"])
                    if coverage[(task, split)] >= minima[split]:
                        continue
                    try:
                        validated = validate_record(candidate, source_text=excerpt)
                        if validated["id"] in existing_ids:
                            continue
                        if not _quality_agent_accepts(quality_agent, validated):
                            rejected[task] += 1
                            continue
                        _append_jsonl(
                            output_dir / f"lantra_teacher_{task}.jsonl",
                            validated,
                        )
                        existing_ids.add(str(validated["id"]))
                        coverage[(task, split)] += 1
                        deterministic_accepted += 1
                    except (CurriculumQualityError, ValueError) as exc:
                        rejected[task] += 1
                        LOGGER.debug("Rejected deterministic %s example: %s", task, exc)

                processed.add(document.document_id)
                state["processed_documents"] = sorted(processed)
                state["accepted_ids"] = sorted(existing_ids)
                state["coverage"] = _coverage_summary(coverage, rejected)
                atomic_write_json(state_path, state)

    missing = {
        f"{task}:{split}": minima[split] - int(coverage.get((task, split), 0))
        for task in SUPPORTED_TASKS
        for split in VALID_SPLITS
        if int(coverage.get((task, split), 0)) < minima[split]
    }
    summary = _coverage_summary(coverage, rejected)
    manifest = {
        "schema": "slai.lantra.validated-teacher-supervision.v1",
        "origin": ORIGIN,
        "supported_tasks": list(SUPPORTED_TASKS),
        "coverage": summary,
        "missing_required_coverage": missing,
        "providers_available": [provider.name for provider in providers],
        "multi_model_critique_enabled": multi_model_critique,
        "multi_model_critique_rejections": critique_rejections,
        "external_calls_accepted": external_accepted,
        "deterministic_fallback_accepted": deterministic_accepted,
        "rejected": int(sum(rejected.values())),
        "resume_state": str(state_path),
        "cache": str(cache_path),
    }
    atomic_write_json(manifest_path, manifest)
    state["manifest"] = str(manifest_path)
    state["complete"] = not bool(missing)
    atomic_write_json(state_path, state)

    if missing:
        LOGGER.warning(
            "Teacher-supervision build completed with missing coverage because the source corpus "
            "does not contain enough valid documents for one or more splits: %s",
            missing,
        )

    return SupervisedBuildResult(
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        coverage=summary,
        providers=tuple(provider.name for provider in providers),
        external_calls_accepted=external_accepted,
        deterministic_fallback_accepted=deterministic_accepted,
        rejected=int(sum(rejected.values())),
        resumed_records=resumed_records,
    )


__all__ = [
    "ORIGIN",
    "SupervisedBuildResult",
    "build_supervised_dataset",
    "deterministic_examples",
    "validate_record",
]
