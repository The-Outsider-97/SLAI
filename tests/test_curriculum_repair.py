"""Source-grounding and trainer-contract regressions for SLAI LANTRA repair.

Tests use the production builder/contracts and the real LANTRA trainer. They do
not emulate KnowledgeAgent or the optimizer with fake objects.
"""

from __future__ import annotations

from dataclasses import replace

from src.training.curriculum_builder import LantraCurriculumBuilder
from src.training.enrichment_contracts import (
    CurriculumConfig,
    SourceSegment,
    normalized_text_hash,
)
from src.training.training_quality_gate import TrainingQualityGate


TEXT = (
    "The ecological survey documented a measurable decline in coastal vegetation "
    "during the dry season. Scientific teams independently verified the sample "
    "locations and recorded observations in a standardized research database. "
    "The full field results supported careful analysis of rainfall, salinity, "
    "and the influence of changing groundwater conditions."
)


def _segment(document_id: str, split: str) -> SourceSegment:
    return SourceSegment(
        segment_id=f"segment-{document_id}",
        document_id=document_id,
        split=split,
        segment_index=0,
        text=TEXT,
        normalized_text_sha256=normalized_text_hash(TEXT),
    )


def test_extractive_positive_is_not_a_different_topic() -> None:
    positive = _segment("doc-train", "train")
    query = LantraCurriculumBuilder._extractive_query_view(positive.text)
    assert query is not None
    assert query != positive.text
    assert query in positive.text
    anchor = replace(positive, text=query, normalized_text_sha256=normalized_text_hash(query))
    assert anchor.document_id == positive.document_id
    assert anchor.segment_id == positive.segment_id
    assert anchor.normalized_text_sha256 != positive.normalized_text_sha256


def test_denoising_target_is_exact_verbatim_source_phrase() -> None:
    masked_and_gold = LantraCurriculumBuilder._extractive_denoising_span(TEXT)
    assert masked_and_gold is not None
    masked, gold = masked_and_gold
    assert "[MASK]" in masked
    assert gold in TEXT
    assert masked.replace("[MASK]", gold) == TEXT


def test_real_trainer_accepts_grounded_generation_record() -> None:
    config = CurriculumConfig()
    segment = _segment("doc-train", "train")
    builder = LantraCurriculumBuilder(
        config, knowledge=None, reasoning=None, perception=None,
    )
    masked, gold = LantraCurriculumBuilder._extractive_denoising_span(TEXT)
    record = builder._seq2seq_record(
        task="generation", split="train", source=masked, target=gold,
        phase="2a", derivation="source_grounded_span_denoising",
        source_document_ids=(segment.document_id,),
        metadata={"source_segment_id": segment.segment_id},
    )
    gate = TrainingQualityGate(config, {segment.document_id: "train"})
    assert gate.evaluate(record).accepted is True
    assert gate.evaluate(record).accepted is False  # deduplication


def test_real_gate_rejects_cross_split_training_sources() -> None:
    config = CurriculumConfig()
    builder = LantraCurriculumBuilder(
        config, knowledge=None, reasoning=None, perception=None,
    )
    record = builder._pairwise_record(
        task="embedding", split="train", anchor="anchor excerpt",
        positive="positive source", negative="distinct negative",
        phase="2b", derivation="extractive_positive_scoped_retrieval_negative",
        source_document_ids=("train-doc", "test-doc"), metadata={},
    )
    gate = TrainingQualityGate(config, {"train-doc": "train", "test-doc": "test"})
    result = gate.evaluate(record)
    assert result.accepted is False
    assert result.reason == "cross_split_source_leakage"
