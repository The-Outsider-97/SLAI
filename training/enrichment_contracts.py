"""Typed contracts for LANTRA curriculum enrichment.

This module intentionally contains only data contracts, deterministic hashing,
and configuration validation. It does not import or instantiate SLAI agents.
That separation keeps the curriculum builder testable and prevents training
policy from leaking into Knowledge, Reasoning, or Perception subsystems.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


CURRICULUM_SCHEMA = "slai.lantra.curriculum.v1"
MANIFEST_SCHEMA = "slai.lantra.curriculum-manifest.v1"
SUPPORTED_TASKS: Tuple[str, ...] = (
    "generation",
    "classification",
    "translation",
    "summarization",
    "dialogue",
    "embedding",
    "reranking",
)
VALID_SPLITS = frozenset({"train", "validation", "test"})
VALID_PHASES = frozenset({"2a", "2b", "2c", "2d", "3"})


class CurriculumError(RuntimeError):
    """Raised for actionable curriculum construction failures."""


class CurriculumCompatibilityError(CurriculumError):
    """Raised when the active SLAI runtime does not satisfy the integration contract."""


class CurriculumQualityError(CurriculumError):
    """Raised when a candidate example violates a hard quality invariant."""


def stable_json_bytes(value: Any) -> bytes:
    """Serialize deterministically for content-addressed IDs and fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def stable_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) / float(2**64 - 1)


def normalized_text_hash(text: str) -> str:
    canonical = " ".join(str(text).split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_non_empty_text(value: Any, field_name: str, *, max_chars: int = 1_000_000) -> str:
    if not isinstance(value, str):
        raise CurriculumError(f"{field_name} must be text, got {type(value).__name__}.")
    text = value.strip()
    if not text:
        raise CurriculumError(f"{field_name} cannot be empty.")
    if len(text) > max_chars:
        raise CurriculumError(
            f"{field_name} contains {len(text)} characters, exceeding the safety limit {max_chars}."
        )
    return text


@dataclass(frozen=True)
class SourceDocument:
    """One canonical document extracted through train_lantra.py's extractor."""

    document_id: str
    source_path: str
    source_type: str
    source_sha256: str
    normalized_text_sha256: str
    text: str
    title: Optional[str]
    extractor: str
    logical_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    split: Optional[str] = None

    def with_split(self, split: str) -> "SourceDocument":
        if split not in VALID_SPLITS:
            raise CurriculumError(f"Invalid document split: {split!r}.")
        return dataclasses.replace(self, split=split)

    def provenance(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "source_sha256": self.source_sha256,
            "normalized_text_sha256": self.normalized_text_sha256,
            "title": self.title,
            "extractor": self.extractor,
            "logical_index": self.logical_index,
            "split": self.split,
            "character_count": len(self.text),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SourceSegment:
    """A document-scoped chunk retained after global exact deduplication."""

    segment_id: str
    document_id: str
    split: str
    segment_index: int
    text: str
    normalized_text_sha256: str
    title: Optional[str] = None
    source_path: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source_document_id": self.document_id,
            "source_segment_index": self.segment_index,
            "source_segment_sha256": self.normalized_text_sha256,
            "source_title": self.title,
            "source_path": self.source_path,
            "split": self.split,
        }


@dataclass(frozen=True, order=True)
class KnowledgeFact:
    """Canonical subject/predicate/object fact used at the adapter boundary."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "knowledge_ontology"

    def __post_init__(self) -> None:
        for field_name in ("subject", "predicate", "object"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise CurriculumError(f"KnowledgeFact.{field_name} cannot be empty.")
            object.__setattr__(self, field_name, value)
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise CurriculumError("KnowledgeFact.confidence must be in [0, 1].")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source", str(self.source or "knowledge_ontology"))

    @property
    def tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class RetrievalTriplet:
    anchor: SourceSegment
    positive: SourceSegment
    negative: SourceSegment
    positive_score: float
    negative_score: float
    retrieval_mode: str
    perception_positive_similarity: Optional[float] = None
    perception_negative_similarity: Optional[float] = None

    def to_metadata(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "retrieval_mode": self.retrieval_mode,
            "positive_score": float(self.positive_score),
            "negative_score": float(self.negative_score),
            "retrieval_margin": float(self.positive_score - self.negative_score),
            "anchor_segment_id": self.anchor.segment_id,
            "positive_segment_id": self.positive.segment_id,
            "negative_segment_id": self.negative.segment_id,
        }
        if self.perception_positive_similarity is not None:
            payload["perception_positive_similarity"] = float(self.perception_positive_similarity)
        if self.perception_negative_similarity is not None:
            payload["perception_negative_similarity"] = float(self.perception_negative_similarity)
            payload["perception_similarity_margin"] = float(
                (self.perception_positive_similarity or 0.0)
                - self.perception_negative_similarity
            )
        return payload


@dataclass(frozen=True)
class ReasoningValidation:
    fact: KnowledgeFact
    decision: str
    combined_valid: bool
    validation_status: str
    validation_complete: bool
    has_conflict: bool
    kb_confidence: Optional[float]
    probabilistic_confidence: Optional[float]
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted_gold(self) -> bool:
        return bool(
            self.decision == "valid"
            and self.combined_valid
            and self.validation_status == "success"
            and self.validation_complete
            and not self.has_conflict
        )

    @property
    def accepted_unsupported(self) -> bool:
        return bool(
            self.decision == "invalid"
            and not self.combined_valid
            and self.validation_status == "success"
            and self.validation_complete
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "decision": self.decision,
            "combined_valid": self.combined_valid,
            "validation_status": self.validation_status,
            "validation_complete": self.validation_complete,
            "has_conflict": self.has_conflict,
            "kb_confidence": self.kb_confidence,
            "probabilistic_confidence": self.probabilistic_confidence,
        }


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    dedupe_key: Optional[str] = None


@dataclass(frozen=True)
class CurriculumBuildResult:
    output_dir: str
    manifest_path: str
    manifest: Mapping[str, Any]
    reused: bool = False


@dataclass(frozen=True)
class CurriculumConfig:
    """Validated build policy for the offline LANTRA enrichment pipeline."""

    output_dir: str = "data/processed/lantra/agent_enriched"
    source_paths: Tuple[str, ...] = ()
    seed: int = 42
    validation_fraction: float = 0.10
    test_fraction: float = 0.10

    min_document_chars: int = 180
    segment_min_chars: int = 160
    segment_chunk_chars: int = 1600
    max_documents: int = 0
    max_segments: int = 20_000
    max_segments_per_document: int = 64

    enable_knowledge: bool = True
    enable_reasoning: bool = True
    enable_perception: bool = False
    perception_checkpoint_version: Optional[str] = None
    require_perception_checkpoint: bool = True

    retrieval_top_k: int = 24
    max_retrieval_pairs: int = 2_000
    min_positive_retrieval_score: float = 0.08
    min_negative_retrieval_score: float = 0.03
    min_retrieval_margin: float = 0.01
    min_perception_margin: float = 0.01

    ontology_max_candidate_terms_per_segment: int = 32
    ontology_max_facts_per_segment: int = 8
    max_phase_2a_examples_per_document: int = 8
    max_phase_2c_examples_per_document: int = 12
    reasoning_threshold: float = 0.75

    require_heldout_coverage: bool = True
    reuse_if_unchanged: bool = True
    fail_on_agent_error: bool = True
    max_record_chars: int = 20_000

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "CurriculumConfig":
        raw = dict(value or {})
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known - {"__config_path__"})
        if unknown:
            raise CurriculumError(
                "Unknown lantra curriculum configuration keys: " + ", ".join(unknown)
            )

        tuple_fields = {"source_paths"}
        normalized: Dict[str, Any] = {}
        for key, item in raw.items():
            if key == "__config_path__":
                continue
            if key in tuple_fields:
                if item in (None, ""):
                    normalized[key] = ()
                elif isinstance(item, str):
                    normalized[key] = (item,)
                elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
                    normalized[key] = tuple(str(entry) for entry in item if str(entry).strip())
                else:
                    raise CurriculumError(f"{key} must be a string or sequence of strings.")
            else:
                normalized[key] = item
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.output_dir.strip():
            raise CurriculumError("output_dir cannot be empty.")
        if not (0.0 <= float(self.validation_fraction) < 0.5):
            raise CurriculumError("validation_fraction must be >= 0 and < 0.5.")
        if not (0.0 <= float(self.test_fraction) < 0.5):
            raise CurriculumError("test_fraction must be >= 0 and < 0.5.")
        if float(self.validation_fraction) + float(self.test_fraction) >= 0.5:
            raise CurriculumError("validation_fraction + test_fraction must be < 0.5.")

        positive_int_fields = (
            "min_document_chars",
            "segment_min_chars",
            "segment_chunk_chars",
            "max_segments_per_document",
            "retrieval_top_k",
            "ontology_max_candidate_terms_per_segment",
            "ontology_max_facts_per_segment",
            "max_phase_2a_examples_per_document",
            "max_phase_2c_examples_per_document",
            "max_record_chars",
        )
        for name in positive_int_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise CurriculumError(f"{name} must be a positive integer.")

        nonnegative_int_fields = ("max_documents", "max_segments", "max_retrieval_pairs")
        for name in nonnegative_int_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise CurriculumError(f"{name} must be a non-negative integer.")

        for name in (
            "min_positive_retrieval_score",
            "min_negative_retrieval_score",
            "min_retrieval_margin",
            "min_perception_margin",
            "reasoning_threshold",
        ):
            number = float(getattr(self, name))
            if not 0.0 <= number <= 1.0:
                raise CurriculumError(f"{name} must be in [0, 1].")

        if self.enable_perception and self.require_perception_checkpoint:
            if not str(self.perception_checkpoint_version or "").strip():
                raise CurriculumError(
                    "Perception enrichment is enabled but no perception_checkpoint_version was supplied. "
                    "A trained/restored checkpoint is required by default so an untrained PerceptionAgent "
                    "cannot become a semantic teacher accidentally."
                )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def with_overrides(self, **overrides: Any) -> "CurriculumConfig":
        values = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        return CurriculumConfig.from_mapping(values)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def unique_facts(facts: Iterable[KnowledgeFact]) -> Tuple[KnowledgeFact, ...]:
    by_key: Dict[Tuple[str, str, str], KnowledgeFact] = {}
    for fact in facts:
        existing = by_key.get(fact.tuple)
        if existing is None or fact.confidence > existing.confidence:
            by_key[fact.tuple] = fact
    return tuple(sorted(by_key.values(), key=lambda item: item.tuple))


__all__ = [
    "CURRICULUM_SCHEMA",
    "MANIFEST_SCHEMA",
    "SUPPORTED_TASKS",
    "VALID_SPLITS",
    "VALID_PHASES",
    "CurriculumError",
    "CurriculumCompatibilityError",
    "CurriculumQualityError",
    "SourceDocument",
    "SourceSegment",
    "KnowledgeFact",
    "RetrievalTriplet",
    "ReasoningValidation",
    "GateDecision",
    "CurriculumBuildResult",
    "CurriculumConfig",
    "stable_json_bytes",
    "sha256_payload",
    "stable_unit_interval",
    "normalized_text_hash",
    "require_non_empty_text",
    "unique_facts",
]
