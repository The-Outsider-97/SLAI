"""Thin adapter from canonical LANTRA source segments to KnowledgeAgent.

No retrieval, ontology, caching, or inference algorithm is reimplemented here.
The adapter only validates the public surface and translates LANTRA contracts to
KnowledgeAgent inputs/outputs.
"""

from __future__ import annotations

import collections
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .enrichment_contracts import (
    CurriculumCompatibilityError,
    CurriculumConfig,
    KnowledgeFact,
    RetrievalTriplet,
    SourceSegment,
    unique_facts,
)


_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][\w'’-]*", flags=re.UNICODE)
_CAPITALIZED_SPAN_RE = re.compile(
    r"\b(?:[A-ZÀ-ÖØ-Þ][\w'’-]*(?:\s+|$)){1,4}",
    flags=re.UNICODE,
)


class KnowledgeAdapter:
    """Public-contract adapter for KnowledgeAgent enrichment."""

    def __init__(self, agent: Any, config: CurriculumConfig) -> None:
        self.agent = agent
        self.config = config
        for method in ("add_document", "retrieve"):
            if not callable(getattr(agent, method, None)):
                raise CurriculumCompatibilityError(
                    f"KnowledgeAgent does not expose required method {method}()."
                )
        ontology = getattr(agent, "ontology_manager", None)
        if ontology is None:
            raise CurriculumCompatibilityError(
                "KnowledgeAgent does not expose ontology_manager required by LANTRA enrichment."
            )
        for method in ("get_relations", "get_types"):
            if not callable(getattr(ontology, method, None)):
                raise CurriculumCompatibilityError(
                    f"Knowledge ontology manager does not expose {method}()."
                )
        self.ontology = ontology
        self._segment_by_id: Dict[str, SourceSegment] = {}
        self._relation_cache: Dict[str, Tuple[KnowledgeFact, ...]] = {}
        self._term_fact_cache: Dict[str, Tuple[KnowledgeFact, ...]] = {}

    @property
    def retrieval_mode(self) -> str:
        return str(getattr(self.agent, "retrieval_mode", "unknown"))

    def index_segments(self, segments: Sequence[SourceSegment]) -> int:
        indexed = 0
        for segment in segments:
            self._segment_by_id[segment.segment_id] = segment
            before = len(getattr(self.agent, "doc_index", {}))
            self.agent.add_document(
                segment.text,
                doc_id=segment.segment_id,
                metadata={
                    "type": "lantra_training_segment",
                    "source_document_id": segment.document_id,
                    "source_segment_id": segment.segment_id,
                    "source_segment_index": segment.segment_index,
                    "split": segment.split,
                    "title": segment.title,
                    "source_path": segment.source_path,
                },
            )
            after = len(getattr(self.agent, "doc_index", {}))
            indexed += int(after > before)
        return indexed

    def retrieval_triplet(
        self,
        anchor: SourceSegment,
        positive: SourceSegment,
        *,
        perception: Optional[Any] = None,
    ) -> Optional[RetrievalTriplet]:
        """Mine one hard negative while preserving source-split isolation.

        The positive must be retrieved with sufficient relevance. The hard
        negative must come from a *different document in the same split* and must
        remain less relevant than the positive by the configured minimum margin.
        """

        if anchor.split != positive.split:
            return None
        try:
            results = self.agent.retrieve(anchor.text, k=self.config.retrieval_top_k)
        except Exception:
            if self.config.fail_on_agent_error:
                raise
            return None
        if not isinstance(results, Sequence):
            return None

        positive_score: Optional[float] = None
        negative_candidates: List[Tuple[float, SourceSegment]] = []

        for item in results:
            if not isinstance(item, Sequence) or len(item) != 2:
                continue
            raw_score, raw_doc = item
            if not isinstance(raw_doc, Mapping):
                continue
            doc_id = str(raw_doc.get("doc_id") or "")
            candidate = self._segment_by_id.get(doc_id)
            if candidate is None:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if candidate.segment_id == positive.segment_id:
                positive_score = score
                continue
            if candidate.segment_id == anchor.segment_id:
                continue
            if candidate.split != anchor.split:
                continue
            if candidate.document_id == anchor.document_id:
                # Same-document passages are plausible positives/near-positives;
                # never use them as negative supervision.
                continue
            if score < self.config.min_negative_retrieval_score:
                continue
            negative_candidates.append((score, candidate))

        if positive_score is None or positive_score < self.config.min_positive_retrieval_score:
            return None
        eligible = [
            (score, candidate)
            for score, candidate in negative_candidates
            if positive_score - score >= self.config.min_retrieval_margin
        ]
        if not eligible:
            return None
        eligible.sort(key=lambda item: (-item[0], item[1].segment_id))
        negative_score, negative = eligible[0]

        perception_positive: Optional[float] = None
        perception_negative: Optional[float] = None
        if perception is not None:
            scores = perception.score_triplet(anchor.text, positive.text, negative.text)
            perception_positive = float(scores["positive_similarity"])
            perception_negative = float(scores["negative_similarity"])
            if (
                perception_positive - perception_negative
                < self.config.min_perception_margin
            ):
                return None

        return RetrievalTriplet(
            anchor=anchor,
            positive=positive,
            negative=negative,
            positive_score=float(positive_score),
            negative_score=float(negative_score),
            retrieval_mode=self.retrieval_mode,
            perception_positive_similarity=perception_positive,
            perception_negative_similarity=perception_negative,
        )

    def relations_for_segment(self, segment: SourceSegment) -> Tuple[KnowledgeFact, ...]:
        """Find ontology facts whose subject terms are plausibly present in text.

        Candidate-term extraction is intentionally conservative and bounded. The
        ontology manager remains the authority for whether a relation exists.
        """

        cached = self._relation_cache.get(segment.normalized_text_sha256)
        if cached is not None:
            return cached

        candidates = self._candidate_terms(segment.text)
        facts: List[KnowledgeFact] = []
        for term in candidates[: self.config.ontology_max_candidate_terms_per_segment]:
            term_key = term.casefold()
            cached_term = self._term_fact_cache.get(term_key)
            if cached_term is None:
                term_facts: List[KnowledgeFact] = []
                # Query the canonical spelling first and a case-folded spelling
                # second. OntologyManager remains the authority for matches.
                variants = [term]
                folded = term.casefold()
                if folded != term:
                    variants.append(folded)
                try:
                    for variant in variants:
                        relations = self.ontology.get_relations(variant)
                        types = self.ontology.get_types(variant)
                        if isinstance(relations, Sequence):
                            for relation in relations:
                                if not isinstance(relation, Sequence) or len(relation) < 2:
                                    continue
                                predicate, obj = str(relation[0]).strip(), str(relation[1]).strip()
                                if predicate and obj:
                                    term_facts.append(KnowledgeFact(variant, predicate, obj))
                        if isinstance(types, Sequence):
                            for obj in types:
                                value = str(obj).strip()
                                if value:
                                    term_facts.append(KnowledgeFact(variant, "type", value))
                except Exception:
                    if self.config.fail_on_agent_error:
                        raise
                cached_term = unique_facts(term_facts)
                self._term_fact_cache[term_key] = cached_term
            facts.extend(cached_term)
            if len(facts) >= self.config.ontology_max_facts_per_segment:
                break

        unique = unique_facts(facts)[: self.config.ontology_max_facts_per_segment]
        self._relation_cache[segment.normalized_text_sha256] = unique
        return unique

    @staticmethod
    def _candidate_terms(text: str) -> List[str]:
        """Produce bounded lexical candidates without performing ontology logic."""

        candidates: List[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            clean = " ".join(value.strip().split())
            key = clean.casefold()
            if clean and key not in seen and 1 <= len(clean.split()) <= 4:
                seen.add(key)
                candidates.append(clean)

        for match in _CAPITALIZED_SPAN_RE.finditer(text):
            add(match.group(0))

        tokens = [match.group(0) for match in _TOKEN_RE.finditer(text)]
        # Ontology labels may be lowercase. Prefer informative longer tokens and
        # short local n-grams, while keeping work tightly bounded.
        ranked_tokens = sorted(
            set(tokens),
            key=lambda token: (-len(token), token.casefold()),
        )
        for token in ranked_tokens[:32]:
            if len(token) >= 4:
                add(token)

        for index in range(min(len(tokens), 48)):
            for width in (2, 3):
                if index + width <= len(tokens):
                    phrase = " ".join(tokens[index:index + width])
                    if any(char.isupper() for char in phrase) or len(phrase) >= 10:
                        add(phrase)
        return candidates


__all__ = ["KnowledgeAdapter"]
