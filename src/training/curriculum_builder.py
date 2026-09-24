"""Offline, provenance-preserving curriculum builder for LANTRA.

The builder coordinates thin adapters over SLAI's existing Knowledge,
Reasoning, and optional Perception agents. It emits only the seven JSONL task
schemas already accepted by ``train_lantra.py`` and assigns explicit splits at
the source-document level before any enriched example is created.

Important boundary: this module prepares deterministic training artifacts. It
never runs inside LANTRA's optimizer loop and never duplicates agent algorithms.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .enrichment_contracts import (
    CURRICULUM_SCHEMA,
    MANIFEST_SCHEMA,
    CurriculumBuildResult,
    CurriculumConfig,
    CurriculumError,
    KnowledgeFact,
    ReasoningValidation,
    RetrievalTriplet,
    SourceDocument,
    SourceSegment,
    sha256_payload,
    stable_unit_interval,
    unique_facts,
)
from .knowledge_adapter import KnowledgeAdapter
from .reasoning_adapter import ReasoningAdapter
from .training_quality_gate import TrainingQualityGate


_PHASE_FILE_NAMES = {
    ("2a", "generation"): "phase_2a_generation.jsonl",
    ("2b", "embedding"): "phase_2b_embedding.jsonl",
    ("2b", "reranking"): "phase_2b_reranking.jsonl",
    ("2c", "generation"): "phase_2c_generation.jsonl",
    ("2c", "classification"): "phase_2c_classification.jsonl",
}


class LantraCurriculumBuilder:
    """Construct phases 2A-2C and optional Perception-assisted 2B/2D metadata."""

    WORK_STATE_SCHEMA = "slai.lantra.curriculum-work.v1"

    def __init__(
        self,
        config: CurriculumConfig,
        *,
        knowledge: KnowledgeAdapter,
        reasoning: Optional[ReasoningAdapter],
        perception: Optional[Any],
        runtime_metadata: Optional[Mapping[str, Any]] = None,
        resume: bool = True,
        checkpoint_every: int = 25,
    ) -> None:
        self.config = config
        self.knowledge = knowledge
        self.reasoning = reasoning
        self.perception = perception
        self.runtime_metadata = dict(runtime_metadata or {})

        if isinstance(checkpoint_every, bool) or int(checkpoint_every) <= 0:
            raise CurriculumError("checkpoint_every must be a positive integer.")

        self.resume = bool(resume)
        self.checkpoint_every = int(checkpoint_every)

        self._active_build_fingerprint = ""
        self._work_state: Dict[str, Any] = {}
        self._work_state_path: Optional[Path] = None

    def _new_work_state(self, build_fingerprint: str) -> Dict[str, Any]:
        return {
            "schema": self.WORK_STATE_SCHEMA,
            "build_fingerprint": build_fingerprint,
            "completed": {
                "2a": False,
                "2b": False,
                "2c": False,
            },
            "cursors": {
                "2a": 0,
                "2b": 0,
                "2c": 0,
            },
        }

    def _work_path(self, build_fingerprint: str) -> Path:
        output_dir = Path(self.config.output_dir)

        return (
            output_dir.parent
            / ".lantra_curriculum_work"
            / f"{build_fingerprint}.json"
        )

    def _load_work_state(self, build_fingerprint: str) -> Dict[str, Any]:
        self._active_build_fingerprint = build_fingerprint
        self._work_state_path = self._work_path(build_fingerprint)
        if not self.resume:
            try:
                self._work_state_path.unlink(missing_ok=True)
            except OSError:
                pass
            return self._new_work_state(build_fingerprint)

        path = self._work_state_path
        if not path.is_file():
            return self._new_work_state(build_fingerprint)

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return self._new_work_state(build_fingerprint)

        if not isinstance(payload, Mapping):
            return self._new_work_state(build_fingerprint)

        if payload.get("schema") != self.WORK_STATE_SCHEMA:
            return self._new_work_state(build_fingerprint)

        if payload.get("build_fingerprint") != build_fingerprint:
            return self._new_work_state(build_fingerprint)

        return dict(payload)

    def _save_work_state(self, records: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]]) -> None:
        if not self.resume:
            return

        if self._work_state_path is None:
            raise CurriculumError("Curriculum work-state path has not been initialized.")

        path = self._work_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        flattened_records: List[Dict[str, Any]] = []

        for key in sorted(records):
            for record in records[key]:
                flattened_records.append(dict(record))

        payload = dict(self._work_state)
        payload["records"] = flattened_records

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )

        os.close(fd)

        temporary_path = Path(temporary_name)
        try:
            self._write_json(temporary_path, payload)
            os.replace(temporary_path, path)

        except Exception:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

            raise

    def _checkpoint_phase(
        self,
        phase: str,
        cursor: int,
        records: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
        *,
        completed: bool = False,
    ) -> None:
        if phase not in {"2a", "2b", "2c"}:
            raise CurriculumError(f"Unsupported resumable curriculum phase: {phase!r}")

        cursors = self._work_state.setdefault( "cursors", {})
        completed_phases = (self._work_state.setdefault("completed", {}))
        cursors[phase] = max(0, int(cursor))

        if completed:
            completed_phases[phase] = True

        self._save_work_state(records)

    def _phase_cursor(self, phase: str) -> int:
        cursors = self._work_state.get("cursors", {})
        if not isinstance(cursors, Mapping):
            return 0

        try:
            return max(0, int(cursors.get(phase, 0)))
        except (TypeError, ValueError):
            return 0

    def _phase_completed(self, phase: str) -> bool:
        completed = self._work_state.get("completed", {})
        return bool(isinstance(completed, Mapping) and completed.get(phase, False))

    def _checkpoint_due(self, completed_items: int) -> bool:
        return (completed_items > 0 and completed_items % self.checkpoint_every == 0)

    def _restore_records(
        self,
        gate: TrainingQualityGate,
        records: MutableMapping[Tuple[str, str], List[Dict[str, Any]]],
    ) -> int:
        raw_records = self._work_state.pop("records", [])
        if not raw_records:
            return 0

        if (
            not isinstance(raw_records, Sequence)
            or isinstance(raw_records, (str, bytes, bytearray))):
            raise CurriculumError("Invalid records section in curriculum work state.")

        restored = 0

        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise CurriculumError("Invalid curriculum record in resumable state.")

            record = dict(raw_record)
            decision = gate.evaluate(record)
            if not decision.accepted:
                raise CurriculumError(
                    "Previously checkpointed curriculum record "
                    f"failed restoration: {decision.reason}"
                )

            metadata = record.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise CurriculumError("Checkpointed curriculum record metadata is invalid.")

            phase = str(metadata.get("curriculum_phase", ""))
            task = str(record.get("task", ""))
            records[(phase, task)].append(record)

            restored += 1

        return restored

    def _clear_work_state(self) -> None:
        path = self._work_state_path
        if path is None:
            return

        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def build(
        self,
        documents: Sequence[SourceDocument],
        segments: Sequence[SourceSegment],
        *,
        source_inventory: Sequence[Mapping[str, str]],
        source_fingerprint: str,
        build_fingerprint: str,
    ) -> CurriculumBuildResult:
        if not documents:
            raise CurriculumError("No canonical source documents were available for curriculum construction.")
        if not segments:
            raise CurriculumError("No canonical source segments were available for curriculum construction.")
        if any(document.split is None for document in documents):
            raise CurriculumError("Every source document must have a split before enrichment begins.")

        output_dir = Path(self.config.output_dir)
        reused = self._try_reuse(output_dir, build_fingerprint)
        if reused is not None:
            return reused

        document_splits = {document.document_id: str(document.split) for document in documents}
        gate = TrainingQualityGate(self.config, document_splits)
        records: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)

        self._work_state = (self._load_work_state(build_fingerprint))
        self._restore_records(gate, records)
        indexed = self.knowledge.index_segments(segments)
        needs_knowledge_facts = (
            not self._phase_completed("2a")
            or (self.reasoning is not None and not self._phase_completed("2c"))
        )

        if needs_knowledge_facts:
            (
                facts_by_segment,
                facts_by_document,
                origin_by_fact,
            ) = self._collect_knowledge_facts(segments)
        else:
            facts_by_segment = {}
            facts_by_document = {}
            origin_by_fact = {}

        if not self._phase_completed("2a"):
            self._build_phase_2a(
                segments,
                facts_by_segment=facts_by_segment,
                gate=gate,
                records=records,
                start_index=self._phase_cursor(
                    "2a"
                ),
            )


        if not self._phase_completed("2b"):
            self._build_phase_2b(
                segments,
                gate=gate,
                records=records,
                start_index=self._phase_cursor(
                    "2b"
                ),
            )


        if self.reasoning is not None:
            if not self._phase_completed("2c"):
                self._build_phase_2c(
                    documents,
                    facts_by_document=facts_by_document,
                    origin_by_fact=origin_by_fact,
                    gate=gate,
                    records=records,
                    start_index=self._phase_cursor(
                        "2c"
                    ),
                )
        else:
            self._checkpoint_phase("2c", len(documents), records, completed=True)

        records, coverage = self._apply_heldout_coverage(records)
        manifest = self._write_artifacts(
            output_dir,
            documents=documents,
            segments=segments,
            records=records,
            gate=gate,
            source_inventory=source_inventory,
            source_fingerprint=source_fingerprint,
            build_fingerprint=build_fingerprint,
            knowledge_indexed_segments=indexed,
            coverage=coverage,
        )
        self._clear_work_state()
        return CurriculumBuildResult(
            output_dir=str(output_dir),
            manifest_path=str(output_dir / "manifest.json"),
            manifest=manifest,
            reused=False,
        )

    # ------------------------------------------------------------------
    # Phase 2A: knowledge-aware denoising/relation completion
    # ------------------------------------------------------------------
    def _build_phase_2a(
        self,
        segments: Sequence[SourceSegment],
        *,
        facts_by_segment: Mapping[str, Sequence[KnowledgeFact]],
        gate: TrainingQualityGate,
        records: MutableMapping[Tuple[str, str], List[Dict[str, Any]]],
        start_index: int = 0,
    ) -> None:
        ordered_segments = sorted(
            segments,
            key=lambda item: (
                stable_unit_interval(item.segment_id, self.config.seed + 211),
                item.segment_id,
            ),
        )
        per_document = collections.Counter()
        for record in records.get(("2a", "generation"), ()):
            metadata = record.get("metadata", {})
            if not isinstance(metadata, Mapping):
                continue

            document_id = metadata.get("group_id")
            if document_id:
                per_document[str(document_id)] += 1

        start_index = min(max(0, int(start_index)), len(ordered_segments))
        for segment_index in range(start_index, len(ordered_segments)):
            segment = ordered_segments[segment_index]

            if per_document[segment.document_id] >= self.config.max_phase_2a_examples_per_document:
                completed_segments = segment_index + 1
                if self._checkpoint_due(completed_segments):
                    self._checkpoint_phase("2a", completed_segments, records)
                continue

            facts = facts_by_segment.get(segment.segment_id, ())
            for fact in facts:
                if per_document[segment.document_id] >= self.config.max_phase_2a_examples_per_document:
                    break

                masked = self._mask_first_case_insensitive(segment.text, fact.subject)
                if masked is None and fact.object:
                    masked = self._mask_first_case_insensitive(segment.text, fact.object)
                if masked is not None:
                    record = self._seq2seq_record(
                        task="generation",
                        split=segment.split,
                        source=(
                            "Reconstruct the missing ontology-linked span using only the supplied text. "
                            "Do not add unsupported information.\n\n"
                            + masked
                        ),
                        target=segment.text,
                        phase="2a",
                        derivation="knowledge_span_denoising",
                        source_document_ids=(segment.document_id,),
                        metadata={
                            "knowledge": {"fact": fact.to_dict()},
                            "source_segment_id": segment.segment_id,
                            "group_id": segment.document_id,
                        },
                    )
                    if self._accept(record, gate, records):
                        per_document[segment.document_id] += 1

                if per_document[segment.document_id] >= self.config.max_phase_2a_examples_per_document:
                    break
                relation_source = (
                    "Context:\n"
                    f"{segment.text}\n\n"
                    "Complete the ontology relation without adding unsupported information:\n"
                    f"{fact.subject} [RELATION] {fact.object}"
                )
                record = self._seq2seq_record(
                    task="generation",
                    split=segment.split,
                    source=relation_source,
                    target=fact.predicate,
                    phase="2a",
                    derivation="knowledge_relation_denoising",
                    source_document_ids=(segment.document_id,),
                    metadata={
                        "knowledge": {"fact": fact.to_dict()},
                        "source_segment_id": segment.segment_id,
                        "group_id": segment.document_id,
                    },
                )
                if self._accept(record, gate, records):
                    per_document[segment.document_id] += 1

            completed_segments = segment_index + 1
            if self._checkpoint_due(completed_segments):
                self._checkpoint_phase("2a", completed_segments, records)

        self._checkpoint_phase("2a", len(ordered_segments), records, completed=True)


    # ------------------------------------------------------------------
    # Phase 2B: retrieval representation learning + hard negatives
    # ------------------------------------------------------------------
    def _build_phase_2b(
        self,
        segments: Sequence[SourceSegment],
        *,
        gate: TrainingQualityGate,
        records: MutableMapping[Tuple[str, str], List[Dict[str, Any]]],
        start_index: int = 0,
    ) -> None:
        by_document: Dict[str, List[SourceSegment]] = collections.defaultdict(list)
        for segment in segments:
            by_document[segment.document_id].append(segment)

        candidate_pairs: List[Tuple[SourceSegment, SourceSegment]] = []
        for document_id, items in by_document.items():
            ordered = sorted(items, key=lambda item: (item.segment_index, item.segment_id))
            for left, right in zip(ordered, ordered[1:]):
                if left.split == right.split:
                    candidate_pairs.append((left, right))

        candidate_pairs.sort(
            key=lambda pair: (
                stable_unit_interval(pair[0].segment_id + ":" + pair[1].segment_id, self.config.seed + 307),
                pair[0].segment_id,
                pair[1].segment_id,
            )
        )
        if self.config.max_retrieval_pairs > 0:
            candidate_pairs = candidate_pairs[: self.config.max_retrieval_pairs]

        start_index = min(max(0, int(start_index)), len(candidate_pairs))
        for pair_index in range(start_index, len(candidate_pairs)):
            anchor, positive = (candidate_pairs[pair_index])
            triplet = self.knowledge.retrieval_triplet(
                anchor,
                positive,
                perception=self.perception,
            )
            if triplet is None:
                completed_pairs = (pair_index + 1)
                if self._checkpoint_due(completed_pairs):
                    self._checkpoint_phase("2b", completed_pairs, records)
                continue

            source_ids = tuple(sorted({
                triplet.anchor.document_id,
                triplet.positive.document_id,
                triplet.negative.document_id,
            }))
            shared_metadata = {
                "retrieval": triplet.to_metadata(),
                "source_segment_ids": [
                    triplet.anchor.segment_id,
                    triplet.positive.segment_id,
                    triplet.negative.segment_id,
                ],
                "group_id": sha256_payload({"source_document_ids": source_ids})[:24],
                "curriculum_stages": ["2b"] + (["2d"] if self.perception is not None else []),
            }
            if self.perception is not None:
                shared_metadata["perception_teacher"] = {
                    "used_for": "semantic_hardness_filter",
                    "checkpoint": dict(getattr(self.perception, "checkpoint_metadata", {}) or {}),
                    "note": (
                        "Current train_lantra.py has no teacher-representation distillation loss; "
                        "the frozen Perception representation is used only to accept/reject hard negatives."
                    ),
                }

            embedding_record = self._pairwise_record(
                task="embedding",
                split=triplet.anchor.split,
                anchor=triplet.anchor.text,
                positive=triplet.positive.text,
                negative=triplet.negative.text,
                phase="2b",
                derivation="knowledge_hard_negative",
                source_document_ids=source_ids,
                metadata=shared_metadata,
            )
            self._accept(embedding_record, gate, records)

            reranking_record = self._pairwise_record(
                task="reranking",
                split=triplet.anchor.split,
                anchor=triplet.anchor.text,
                positive=triplet.positive.text,
                negative=triplet.negative.text,
                phase="2b",
                derivation="knowledge_hard_negative",
                source_document_ids=source_ids,
                metadata=shared_metadata,
            )
            self._accept(reranking_record, gate, records)

            completed_pairs = pair_index + 1
            if self._checkpoint_due(completed_pairs):
                self._checkpoint_phase("2b", completed_pairs, records)

        self._checkpoint_phase("2b", len(candidate_pairs), records, completed=True)

    # ------------------------------------------------------------------
    # Phase 2C: validated factual + reasoning curriculum
    # ------------------------------------------------------------------
    def _build_phase_2c(
        self,
        documents: Sequence[SourceDocument],
        *,
        facts_by_document: Mapping[str, Sequence[KnowledgeFact]],
        origin_by_fact: Mapping[Tuple[str, Tuple[str, str, str]], SourceSegment],
        gate: TrainingQualityGate,
        records: MutableMapping[Tuple[str, str], List[Dict[str, Any]]],
        start_index: int = 0,
    ) -> None:
        assert self.reasoning is not None
        ordered_documents = sorted(documents, key=lambda item: item.document_id)
        start_index = min(max(0, int(start_index)), len(ordered_documents))
        for document_index in range(start_index, len(ordered_documents)):
            document = ordered_documents[document_index]
            facts = tuple(facts_by_document.get(document.document_id, ()))
            if not facts:
                completed_documents = document_index + 1
                if self._checkpoint_due(completed_documents):
                    self._checkpoint_phase("2c", completed_documents, records)
                continue

            validations = self.reasoning.validate_fact_set(facts)
            accepted = [
                fact
                for fact in facts
                if validations.get(fact.tuple) is not None
                and validations[fact.tuple].accepted_gold
            ]
            if not accepted:
                completed_documents = document_index + 1
                if self._checkpoint_due(completed_documents):
                    self._checkpoint_phase("2c", completed_documents, records)
                continue

            accepted = list(unique_facts(accepted))
            created = 0
            for fact in accepted:
                if created >= self.config.max_phase_2c_examples_per_document:
                    break
                validation = validations[fact.tuple]
                origin = origin_by_fact.get((document.document_id, fact.tuple))
                context = origin.text if origin is not None else document.text[: self.config.segment_chunk_chars]

                generation = self._seq2seq_record(
                    task="generation",
                    split=str(document.split),
                    source=(
                        "Evidence context:\n"
                        f"{context}\n\n"
                        "Complete the validated factual relation:\n"
                        f"{fact.subject} --{fact.predicate}-->"
                    ),
                    target=fact.object,
                    phase="2c",
                    derivation="reasoning_validated_fact_generation",
                    source_document_ids=(document.document_id,),
                    metadata={
                        "knowledge": {"fact": fact.to_dict()},
                        "reasoning": self._reasoning_metadata(validation, required_gold=True),
                        "group_id": document.document_id,
                        "source_segment_id": origin.segment_id if origin else None,
                    },
                )
                if self._accept(generation, gate, records):
                    created += 1

                if created >= self.config.max_phase_2c_examples_per_document:
                    break
                classification = self._classification_record(
                    split=str(document.split),
                    source=(
                        "Use only the supplied evidence and validated structured knowledge.\n\n"
                        f"Evidence context:\n{context}\n\n"
                        f"Claim: {fact.subject} | {fact.predicate} | {fact.object}\n"
                        "Is the claim supported by the supplied knowledge?"
                    ),
                    label="supported",
                    phase="2c",
                    derivation="reasoning_consistency_supported",
                    source_document_ids=(document.document_id,),
                    metadata={
                        "knowledge": {"fact": fact.to_dict()},
                        "reasoning": self._reasoning_metadata(validation, required_gold=True),
                        "group_id": document.document_id,
                        "source_segment_id": origin.segment_id if origin else None,
                    },
                )
                if self._accept(classification, gate, records):
                    created += 1

                if created >= self.config.max_phase_2c_examples_per_document:
                    break
                unsupported = self._unsupported_candidate(fact, accepted)
                if unsupported is not None:
                    negative_validation = self.reasoning.classify_candidate(unsupported, accepted)
                    if negative_validation.accepted_unsupported:
                        negative_record = self._classification_record(
                            split=str(document.split),
                            source=(
                                "Use only the supplied validated facts. Lack of support must be labelled unsupported, "
                                "not contradicted.\n\n"
                                f"Known facts:\n{self._facts_as_text(accepted)}\n\n"
                                f"Claim: {unsupported.subject} | {unsupported.predicate} | {unsupported.object}\n"
                                "Is the claim supported by the supplied knowledge?"
                            ),
                            label="unsupported",
                            phase="2c",
                            derivation="reasoning_consistency_unsupported",
                            source_document_ids=(document.document_id,),
                            metadata={
                                "knowledge": {"candidate_fact": unsupported.to_dict()},
                                "reasoning": self._reasoning_metadata(
                                    negative_validation,
                                    required_gold=False,
                                    required_unsupported=True,
                                ),
                                "group_id": document.document_id,
                            },
                        )
                        if self._accept(negative_record, gate, records):
                            created += 1

            # Use configured ReasoningAgent rules when they genuinely infer new
            # facts. No examples are fabricated when forward chaining yields none.
            if created < self.config.max_phase_2c_examples_per_document:
                inferred = self.reasoning.infer_validated_facts(accepted, max_iterations=3)
                for inferred_fact, validation in inferred:
                    if created >= self.config.max_phase_2c_examples_per_document:
                        break
                    if inferred_fact.tuple in {fact.tuple for fact in accepted}:
                        continue
                    record = self._seq2seq_record(
                        task="generation",
                        split=str(document.split),
                        source=(
                            "Given the validated premises below, return only a supported conclusion.\n\n"
                            f"Premises:\n{self._facts_as_text(accepted)}"
                        ),
                        target=(
                            f"{inferred_fact.subject} | {inferred_fact.predicate} | {inferred_fact.object}"
                        ),
                        phase="2c",
                        derivation="reasoning_forward_chaining_conclusion",
                        source_document_ids=(document.document_id,),
                        metadata={
                            "knowledge": {
                                "premises": [fact.to_dict() for fact in accepted],
                                "inferred_fact": inferred_fact.to_dict(),
                            },
                            "reasoning": self._reasoning_metadata(validation, required_gold=True),
                            "group_id": document.document_id,
                        },
                    )
                    if self._accept(record, gate, records):
                        created += 1

            completed_documents = (document_index + 1)
            if self._checkpoint_due(completed_documents):
                self._checkpoint_phase("2c", completed_documents, records)

        self._checkpoint_phase("2c", len(ordered_documents), records, completed=True)


    # ------------------------------------------------------------------
    # Knowledge preparation
    # ------------------------------------------------------------------
    def _collect_knowledge_facts(
        self,
        segments: Sequence[SourceSegment],
    ) -> Tuple[
        Dict[str, Tuple[KnowledgeFact, ...]],
        Dict[str, Tuple[KnowledgeFact, ...]],
        Dict[Tuple[str, Tuple[str, str, str]], SourceSegment],
    ]:
        facts_by_segment: Dict[str, Tuple[KnowledgeFact, ...]] = {}
        raw_by_document: Dict[str, List[KnowledgeFact]] = collections.defaultdict(list)
        origin_by_fact: Dict[Tuple[str, Tuple[str, str, str]], SourceSegment] = {}
        document_fact_cap = max(
            self.config.ontology_max_facts_per_segment,
            self.config.max_phase_2a_examples_per_document
            + self.config.max_phase_2c_examples_per_document,
        )
        ordered_segments = sorted(segments, key=lambda item: (item.document_id, item.segment_index, item.segment_id))
        for segment in ordered_segments:
            existing_keys = {fact.tuple for fact in raw_by_document.get(segment.document_id, [])}
            if len(existing_keys) >= document_fact_cap:
                facts_by_segment[segment.segment_id] = ()
                continue
            facts = self.knowledge.relations_for_segment(segment)
            facts_by_segment[segment.segment_id] = facts
            for fact in facts:
                if fact.tuple in existing_keys:
                    continue
                raw_by_document[segment.document_id].append(fact)
                existing_keys.add(fact.tuple)
                origin_by_fact.setdefault((segment.document_id, fact.tuple), segment)
                if len(existing_keys) >= document_fact_cap:
                    break
        facts_by_document = {
            document_id: unique_facts(facts)
            for document_id, facts in raw_by_document.items()
        }
        return facts_by_segment, facts_by_document, origin_by_fact

    # ------------------------------------------------------------------
    # Record construction / gate
    # ------------------------------------------------------------------
    def _seq2seq_record(
        self,
        *,
        task: str,
        split: str,
        source: str,
        target: str,
        phase: str,
        derivation: str,
        source_document_ids: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base_metadata = self._metadata(
            phase=phase,
            derivation=derivation,
            source_document_ids=source_document_ids,
            extra=metadata,
        )
        payload = {
            "task": task,
            "split": split,
            "input": source,
            "target": target,
            "metadata": base_metadata,
        }
        payload["id"] = sha256_payload(payload)[:24]
        return payload

    def _classification_record(
        self,
        *,
        split: str,
        source: str,
        label: str,
        phase: str,
        derivation: str,
        source_document_ids: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base_metadata = self._metadata(
            phase=phase,
            derivation=derivation,
            source_document_ids=source_document_ids,
            extra=metadata,
        )
        payload = {
            "task": "classification",
            "split": split,
            "input": source,
            "label": label,
            "metadata": base_metadata,
        }
        payload["id"] = sha256_payload(payload)[:24]
        return payload

    def _pairwise_record(
        self,
        *,
        task: str,
        split: str,
        anchor: str,
        positive: str,
        negative: str,
        phase: str,
        derivation: str,
        source_document_ids: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        base_metadata = self._metadata(
            phase=phase,
            derivation=derivation,
            source_document_ids=source_document_ids,
            extra=metadata,
        )
        if task == "reranking":
            payload = {
                "task": task,
                "split": split,
                "query": anchor,
                "positive": positive,
                "negative": negative,
                "metadata": base_metadata,
            }
        else:
            payload = {
                "task": task,
                "split": split,
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "metadata": base_metadata,
            }
        payload["id"] = sha256_payload(payload)[:24]
        return payload

    def _metadata(
        self,
        *,
        phase: str,
        derivation: str,
        source_document_ids: Sequence[str],
        extra: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "schema": CURRICULUM_SCHEMA,
            "origin": "agent_enriched",
            "synthetic": True,
            "curriculum_phase": phase,
            "derivation": derivation,
            "source_document_ids": list(dict.fromkeys(str(item) for item in source_document_ids)),
        }
        payload.update(dict(extra))
        return payload

    def _accept(
        self,
        record: Dict[str, Any],
        gate: TrainingQualityGate,
        records: MutableMapping[Tuple[str, str], List[Dict[str, Any]]],
    ) -> bool:
        decision = gate.evaluate(record)
        if not decision.accepted:
            return False
        metadata = record.get("metadata", {})
        phase = str(metadata.get("curriculum_phase", ""))
        task = str(record.get("task", ""))
        records[(phase, task)].append(record)
        return True

    # ------------------------------------------------------------------
    # Coverage and deterministic output
    # ------------------------------------------------------------------
    def _apply_heldout_coverage(
        self,
        records: Mapping[Tuple[str, str], Sequence[Dict[str, Any]]],
    ) -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], Dict[str, Any]]:
        flattened = [record for items in records.values() for record in items]
        by_task_split: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for record in flattened:
            by_task_split[str(record["task"])][str(record["split"])] += 1

        dropped_tasks: List[str] = []
        active_tasks: List[str] = []
        for task, counts in sorted(by_task_split.items()):
            has_all = all(counts.get(split, 0) > 0 for split in ("train", "validation", "test"))
            if self.config.require_heldout_coverage and not has_all:
                dropped_tasks.append(task)
            else:
                active_tasks.append(task)

        filtered: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for key, items in records.items():
            if key[1] in dropped_tasks:
                continue
            filtered[key] = sorted((dict(item) for item in items), key=lambda item: str(item["id"]))

        return filtered, {
            "active_tasks": active_tasks,
            "dropped_tasks_missing_heldout_coverage": dropped_tasks,
            "counts_before_filter": {
                task: {split: int(counter.get(split, 0)) for split in ("train", "validation", "test")}
                for task, counter in sorted(by_task_split.items())
            },
        }

    def _write_artifacts(
        self,
        output_dir: Path,
        *,
        documents: Sequence[SourceDocument],
        segments: Sequence[SourceSegment],
        records: Mapping[Tuple[str, str], Sequence[Dict[str, Any]]],
        gate: TrainingQualityGate,
        source_inventory: Sequence[Mapping[str, str]],
        source_fingerprint: str,
        build_fingerprint: str,
        knowledge_indexed_segments: int,
        coverage: Mapping[str, Any],
    ) -> Dict[str, Any]:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".lantra_curriculum_", dir=str(output_dir.parent)))
        artifacts: List[Dict[str, Any]] = []
        try:
            for key, filename in _PHASE_FILE_NAMES.items():
                items = records.get(key, ())
                if not items:
                    continue
                path = staging / filename
                self._write_jsonl(path, items)
                artifacts.append({
                    "file": filename,
                    "phase": key[0],
                    "task": key[1],
                    "records": len(items),
                    "sha256": self._sha256_file(path),
                })

            counts = self._record_counts(records)
            manifest: Dict[str, Any] = {
                "schema": MANIFEST_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "build_fingerprint": build_fingerprint,
                "source_fingerprint": source_fingerprint,
                "config": self.config.to_dict(),
                "runtime": dict(self.runtime_metadata),
                "source": {
                    "files": list(source_inventory),
                    "documents": len(documents),
                    "segments": len(segments),
                    "segments_by_split": dict(collections.Counter(item.split for item in segments)),
                    "documents_by_split": dict(collections.Counter(str(item.split) for item in documents)),
                },
                "agents": {
                    "knowledge_indexed_segments": int(knowledge_indexed_segments),
                    "reasoning_enabled": self.reasoning is not None,
                    "perception_enabled": self.perception is not None,
                },
                "records": counts,
                "coverage": dict(coverage),
                "quality_gate": gate.summary(),
                "artifacts": artifacts,
                "documents": [document.provenance() for document in documents],
                "integration": {
                    "trainer_compatible": True,
                    "task_enums_added": False,
                    "explicit_group_safe_splits": True,
                    "current_train_lantra_behavior": (
                        "These JSONL files are directly consumable by the current trainer. "
                        "Until train_lantra.py gains stage-aware supervised scheduling, phases 2A-2C "
                        "are consumed during its supervised multitask stage; curriculum_phase metadata "
                        "preserves the intended future stage boundary."
                    ),
                    "phase_2d": (
                        "When Perception is enabled, frozen representations filter hard-negative pairs and "
                        "their similarities are recorded. True representation-distillation loss is not fabricated "
                        "because current train_lantra.py has no such objective."
                    ),
                },
            }
            manifest_path = staging / "manifest.json"
            self._write_json(manifest_path, manifest)

            output_dir.mkdir(parents=True, exist_ok=True)
            for stale in output_dir.glob("phase_*.jsonl"):
                stale.unlink()
            for staged in sorted(staging.iterdir()):
                os.replace(staged, output_dir / staged.name)
            shutil.rmtree(staging, ignore_errors=True)
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _try_reuse(self, output_dir: Path, build_fingerprint: str) -> Optional[CurriculumBuildResult]:
        if not self.config.reuse_if_unchanged:
            return None
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            return None
        if not isinstance(manifest, Mapping):
            return None
        if manifest.get("schema") != MANIFEST_SCHEMA:
            return None
        if manifest.get("build_fingerprint") != build_fingerprint:
            return None
        artifacts = manifest.get("artifacts", [])
        if not isinstance(artifacts, Sequence):
            return None
        for item in artifacts:
            if not isinstance(item, Mapping):
                return None
            path = output_dir / str(item.get("file", ""))
            if not path.is_file() or self._sha256_file(path) != str(item.get("sha256", "")):
                return None
        return CurriculumBuildResult(
            output_dir=str(output_dir),
            manifest_path=str(manifest_path),
            manifest=dict(manifest),
            reused=True,
        )

    @staticmethod
    def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _record_counts(
        records: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]]
    ) -> Dict[str, Any]:
        by_phase = collections.Counter()
        by_task = collections.Counter()
        by_split = collections.Counter()
        by_derivation = collections.Counter()
        total = 0
        for (phase, task), items in records.items():
            for record in items:
                total += 1
                by_phase[phase] += 1
                by_task[task] += 1
                by_split[str(record.get("split"))] += 1
                metadata = record.get("metadata", {})
                if isinstance(metadata, Mapping):
                    by_derivation[str(metadata.get("derivation", "unknown"))] += 1
        return {
            "total": total,
            "by_phase": dict(sorted(by_phase.items())),
            "by_task": dict(sorted(by_task.items())),
            "by_split": dict(sorted(by_split.items())),
            "by_derivation": dict(sorted(by_derivation.items())),
        }

    # ------------------------------------------------------------------
    # Small deterministic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_first_case_insensitive(text: str, phrase: str) -> Optional[str]:
        phrase = str(phrase).strip()
        if not phrase:
            return None
        lower_text = text.casefold()
        lower_phrase = phrase.casefold()
        index = lower_text.find(lower_phrase)
        if index < 0:
            return None
        return text[:index] + "[MASK]" + text[index + len(phrase):]

    @staticmethod
    def _reasoning_metadata(
        validation: ReasoningValidation,
        *,
        required_gold: bool,
        required_unsupported: bool = False,
    ) -> Dict[str, Any]:
        payload = validation.to_dict()
        payload["required_gold"] = bool(required_gold)
        payload["required_unsupported"] = bool(required_unsupported)
        return payload

    @staticmethod
    def _unsupported_candidate(
        base: KnowledgeFact,
        accepted: Sequence[KnowledgeFact],
    ) -> Optional[KnowledgeFact]:
        accepted_keys = {fact.tuple for fact in accepted}
        objects = sorted({fact.object for fact in accepted if fact.object != base.object})
        for obj in objects:
            candidate = KnowledgeFact(
                base.subject,
                base.predicate,
                obj,
                confidence=0.0,
                source="curriculum_candidate",
            )
            if candidate.tuple not in accepted_keys:
                return candidate
        return None

    @staticmethod
    def _facts_as_text(facts: Sequence[KnowledgeFact]) -> str:
        return "\n".join(
            f"- {fact.subject} | {fact.predicate} | {fact.object}"
            for fact in facts
        )


__all__ = ["LantraCurriculumBuilder"]
