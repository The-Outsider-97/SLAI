"""Thin adapter from canonical LANTRA source segments to KnowledgeAgent.

No retrieval, ontology, caching, or inference algorithm is reimplemented here.
The adapter only validates the public surface and translates LANTRA contracts to
KnowledgeAgent inputs/outputs.
"""

from __future__ import annotations

import collections
import hashlib
import math
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
                raise CurriculumCompatibilityError(f"KnowledgeAgent does not expose required method {method}().")
        ontology = getattr(agent, "ontology_manager", None)
        if ontology is None:
            raise CurriculumCompatibilityError("KnowledgeAgent does not expose ontology_manager required by LANTRA enrichment.")
        for method in ("get_relations", "get_types"):
            if not callable(getattr(ontology, method, None)):
                raise CurriculumCompatibilityError(f"Knowledge ontology manager does not expose {method}().")
        self.ontology = ontology
        self._segment_by_id: Dict[str, SourceSegment] = {}
        self._relation_cache: Dict[str, Tuple[KnowledgeFact, ...]] = {}
        self._term_fact_cache: Dict[str, Tuple[KnowledgeFact, ...]] = {}
        self._negative_pool: Dict[str, Tuple[SourceSegment, ...]] = {}
        self._retrieval_diagnostics: collections.Counter[str] = collections.Counter()
        # Use the KnowledgeAgent's own TF-IDF relevance function.  The public
        # retrieve() operation remains the first negative-candidate source.
        self._pair_score = getattr(agent, "score_text_relevance", None)
        if not callable(self._pair_score):
            raise CurriculumCompatibilityError(
                "KnowledgeAgent.score_text_relevance() is required for "
                "direct positive and negative relevance comparison."
            )

    def retrieval_diagnostics(self) -> Dict[str, int]:
        return {key: int(value) for key, value in sorted(self._retrieval_diagnostics.items())}

    @property
    def retrieval_mode(self) -> str:
        return str(getattr(self.agent, "retrieval_mode", "unknown"))

    @property
    def ontology_has_facts(self) -> bool:
        """Use the ontology manager's already-loaded authoritative RDF graph."""
        graph = getattr(self.ontology, "graph", None)
        return True if graph is None else len(graph) > 0

    def index_segments(self, segments: Sequence[SourceSegment]) -> int:
        indexed = 0
        by_split: Dict[str, List[SourceSegment]] = collections.defaultdict(list)
        for segment in segments:
            self._segment_by_id[segment.segment_id] = segment
            by_split[segment.split].append(segment)
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
        finalize = getattr(self.agent, "finalize_retrieval_index", None)

        if callable(finalize):
            finalize()
        self._negative_pool = {
            split: tuple(sorted(items, key=lambda item: item.segment_id))
            for split, items in by_split.items()
        }
        self._retrieval_diagnostics["indexed_segments"] = indexed
        self._retrieval_diagnostics["source_segments"] = len(segments)
        return indexed

    def retrieval_triplet(
        self,
        anchor: SourceSegment,
        positive: SourceSegment,
        *,
        perception: Optional[Any] = None,
    ) -> Optional[RetrievalTriplet]:
        """Mine a source-grounded pair with same-split negative supervision.

        Positives are scored directly by KnowledgeAgent's TF-IDF similarity;
        their presence in retrieve()'s top-k is NOT required.  Retrieval is used
        to propose negatives; a bounded same-split lexical search compensates
        when KnowledgeAgent's global similarity threshold is too restrictive.
        No cross-document fact/entailment labels are manufactured here.
        """
        self._retrieval_diagnostics["pairs_attempted"] += 1
        if anchor.split != positive.split or not anchor.text.strip() or not positive.text.strip():
            self._retrieval_diagnostics["invalid_positive"] += 1
            return None
        if anchor.text.strip() == positive.text.strip():
            self._retrieval_diagnostics["identical_views"] += 1
            return None

        try:
            positive_score = float(self._pair_score(anchor.text, positive.text))
        except Exception:
            self._retrieval_diagnostics["positive_score_error"] += 1
            if self.config.fail_on_agent_error:
                raise
            return None
        if not math.isfinite(positive_score) or positive_score < self.config.min_positive_retrieval_score:
            self._retrieval_diagnostics["weak_positive"] += 1
            return None

        try:
            results = self.agent.retrieve(anchor.text, k=self.config.retrieval_top_k)
        except Exception:
            self._retrieval_diagnostics["retrieval_error"] += 1
            if self.config.fail_on_agent_error:
                raise
            return None
        if not isinstance(results, Sequence):
            self._retrieval_diagnostics["invalid_retrieval_response"] += 1
            return None

        candidates: Dict[str, SourceSegment] = {}
        for item in results:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                continue
            _, raw_doc = item
            if not isinstance(raw_doc, Mapping):
                continue
            candidate = self._segment_by_id.get(str(raw_doc.get("doc_id") or ""))
            if (candidate is not None
                    and candidate.split == anchor.split
                    and candidate.document_id != positive.document_id
                    and candidate.document_id != anchor.document_id):
                candidates[candidate.segment_id] = candidate
        self._retrieval_diagnostics["retriever_scoped_candidates"] += len(candidates)

        negative = self._best_negative(anchor, positive_score, candidates.values())
        if negative is None:
            # The default KnowledgeAgent threshold can eliminate all plausible
            # negatives before LANTRA sees them.  Score a bounded, deterministic
            # subset with *the same* KnowledgeAgent scoring function.
            pool = self._negative_pool.get(anchor.split, ())
            if pool:
                start = int.from_bytes(
                    hashlib.sha256(anchor.segment_id.encode("utf-8")).digest()[:8], "big"
                ) % len(pool)
                stride = max(1, len(pool) // 96)
                sampled: List[SourceSegment] = []
                seen_ids: set[str] = set(candidates)
                for offset in range(min(len(pool), 192)):
                    candidate = pool[(start + offset * stride) % len(pool)]
                    if (candidate.document_id not in {anchor.document_id, positive.document_id}
                            and candidate.segment_id not in seen_ids):
                        sampled.append(candidate)
                        seen_ids.add(candidate.segment_id)
                        if len(sampled) >= 96:
                            break
                self._retrieval_diagnostics["fallback_scoped_candidates"] += len(sampled)
                negative = self._best_negative(anchor, positive_score, sampled)

        if negative is None:
            self._retrieval_diagnostics["no_eligible_negative"] += 1
            return None

        negative_score, negative_segment = negative
        perception_positive: Optional[float] = None
        perception_negative: Optional[float] = None
        if perception is not None:
            scores = perception.score_triplet(anchor.text, positive.text, negative_segment.text)
            perception_positive = float(scores["positive_similarity"])
            perception_negative = float(scores["negative_similarity"])
            if (not math.isfinite(perception_positive)
                    or not math.isfinite(perception_negative)
                    or perception_positive - perception_negative < self.config.min_perception_margin):
                self._retrieval_diagnostics["perception_rejected"] += 1
                return None

        self._retrieval_diagnostics["triplets_accepted"] += 1
        return RetrievalTriplet(
            anchor=anchor,
            positive=positive,
            negative=negative_segment,
            positive_score=positive_score,
            negative_score=negative_score,
            retrieval_mode=f"{self.retrieval_mode}:direct_tfidf_scoring",
            perception_positive_similarity=perception_positive,
            perception_negative_similarity=perception_negative,
        )

    def _best_negative(
        self,
        anchor: SourceSegment,
        positive_score: float,
        candidates: Iterable[SourceSegment],
    ) -> Optional[Tuple[float, SourceSegment]]:
        ranked: List[Tuple[float, SourceSegment]] = []
        for candidate in candidates:
            if candidate.split != anchor.split or candidate.document_id == anchor.document_id:
                continue
            try:
                score = float(self._pair_score(anchor.text, candidate.text))
            except Exception:
                self._retrieval_diagnostics["negative_score_error"] += 1
                if self.config.fail_on_agent_error:
                    raise
                continue
            if (math.isfinite(score)
                    and score >= self.config.min_negative_retrieval_score
                    and positive_score - score >= self.config.min_retrieval_margin):
                ranked.append((score, candidate))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], item[1].segment_id))
        return ranked[0]

    def relations_for_segment(self, segment: SourceSegment) -> Tuple[KnowledgeFact, ...]:
        """Find ontology facts whose subject terms are plausibly present in text.

        Candidate-term extraction is intentionally conservative and bounded. The
        ontology manager remains the authority for whether a relation exists.
        """

        if not self.ontology_has_facts:
            return ()

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
        ranked_tokens = sorted(set(tokens), key=lambda token: (-len(token), token.casefold()))
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
