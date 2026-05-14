"""Analogical reasoning strategy for the reasoning subsystem.

Analogical reasoning transfers useful knowledge from familiar source domains to
an unfamiliar target when they share enough structural, functional, relational,
and semantic alignment.

Pipeline
--------
1. Normalize and validate target/source-domain inputs.
2. Extract candidate source descriptors from dicts, objects, and text.
3. Score candidates using shared ``BaseReasoning`` property extraction plus
   analogical-specific structural, functional, relational, semantic, and
   constraint scores.
4. Build property correspondences and a mapping-quality profile for each
   accepted analogy.
5. Transfer source knowledge to target-oriented properties, adapting values with
   caller constraints and configured transformation policy.
6. Select the best transfer and return an explainable, JSON-safe result.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..reasoning_cache import ReasoningCache
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from .base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Analogical")
printer = PrettyPrinter()


@dataclass(frozen=True)
class AnalogyCandidate:
    """Scored candidate source domain item."""

    item: Any
    index: int
    name: str
    similarity: float
    structural_score: float
    functional_score: float
    relational_score: float
    semantic_score: float
    constraint_score: float
    matching_properties: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_public_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["item"] = self.item
        return data


@dataclass(frozen=True)
class PropertyCorrespondence:
    """A target/source property match used in a mapping."""

    target_property: str
    source_property: str
    similarity: float
    relation: str = "property_alignment"
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalogicalMapping:
    """Structured mapping between target and source."""

    target_name: str
    source_name: str
    target: Any
    source: Any
    correspondences: List[Dict[str, Any]]
    structural_score: float
    functional_score: float
    relational_score: float
    semantic_score: float
    constraint_score: float
    mapping_score: float
    confidence: float
    accepted: bool
    rejection_reason: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransferRecord:
    """Knowledge transfer result from one source mapping."""

    source_name: str
    target_name: str
    source: Any
    original_knowledge: List[Dict[str, Any]]
    adapted_knowledge: List[Dict[str, Any]]
    mapping_score: float
    confidence: float
    transfer_score: float
    validation: Dict[str, Any]
    mapping: Dict[str, Any]

    def to_public_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReasoningAnalogical(BaseReasoning):
    """Production analogical reasoning strategy.

    Public compatibility surface retained:
    - ``perform_reasoning(target, source_domain, context=None)``
    - ``create_mapping(...)``
    - ``transfer_knowledge(...)``
    - ``adapt_knowledge(...)``
    - ``select_best_transfer(...)``

    ``context`` may provide:
    - ``property_constraints``: target property -> allowed source properties.
    - ``constraints``: transfer/adaptation constraints.
    - ``concept_mappings``: source concept -> target concept mapping.
    - ``relation_mappings``: source relation -> target relation mapping.
    - ``source_weights``: source-name/index -> score multiplier.
    - ``protected_target_properties``: properties that must not be overwritten.
    - ``required_target_properties``: target properties that should be mapped.
    - ``reasoning_memory`` or ``memory``: optional memory object for event records.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config: Dict[str, Any] = load_global_config()
        self.analogical_config: Dict[str, Any] = get_config_section("reasoning_analogical") or {}

        self.min_similarity: float = clamp_confidence(self.analogical_config.get("min_similarity", 0.5))
        self.max_analogies: int = bounded_iterations(self.analogical_config.get("max_analogies", 3), minimum=1, maximum=512)
        self.structural_weight: float = self._non_negative_float("structural_weight", 0.35)
        self.functional_weight: float = self._non_negative_float("functional_weight", 0.25)
        self.relational_weight: float = self._non_negative_float("relational_weight", 0.15)
        self.semantic_weight: float = self._non_negative_float("semantic_weight", 0.15)
        self.constraint_weight: float = self._non_negative_float("constraint_weight", 0.10)
        self.adaptation_threshold: float = clamp_confidence(self.analogical_config.get("adaptation_threshold", 0.7))
        self.mapping_acceptance_threshold: float = clamp_confidence(
            self.analogical_config.get("mapping_acceptance_threshold", self.min_similarity)
        )
        self.max_correspondences: int = bounded_iterations(
            self.analogical_config.get("max_correspondences", 64), minimum=1, maximum=4096
        )
        self.max_transfer_items: int = bounded_iterations(
            self.analogical_config.get("max_transfer_items", 64), minimum=1, maximum=4096
        )
        self.max_property_tokens: int = bounded_iterations(
            self.analogical_config.get("max_property_tokens", 256), minimum=4, maximum=10_000
        )
        self.min_correspondence_similarity: float = clamp_confidence(
            self.analogical_config.get("min_correspondence_similarity", self.min_similarity)
        )
        self.name_similarity_weight: float = clamp_confidence(self.analogical_config.get("name_similarity_weight", 0.10))
        self.transfer_confidence_floor: float = clamp_confidence(
            self.analogical_config.get("transfer_confidence_floor", 0.05)
        )
        self.direct_transfer_bonus: float = clamp_confidence(self.analogical_config.get("direct_transfer_bonus", 0.05))
        self.transformation_penalty: float = clamp_confidence(self.analogical_config.get("transformation_penalty", 0.10))
        self.enable_cache: bool = bool(self.analogical_config.get("enable_cache", True))
        self.cache_ttl_seconds: float = self._positive_float("cache_ttl_seconds", 300.0)
        self.record_memory_events: bool = bool(self.analogical_config.get("record_memory_events", False))
        self.memory_event_priority: float = clamp_confidence(self.analogical_config.get("memory_event_priority", 0.35))
        self.strict_inputs: bool = bool(self.analogical_config.get("strict_inputs", True))
        self.return_context: bool = bool(self.analogical_config.get("return_context", False))
        self.include_rejected_mappings: bool = bool(self.analogical_config.get("include_rejected_mappings", True))

        self._weights = self._normalize_weights(
            {
                "structural": self.structural_weight,
                "functional": self.functional_weight,
                "relational": self.relational_weight,
                "semantic": self.semantic_weight,
                "constraint": self.constraint_weight,
            }
        )
        self.cache: Optional[ReasoningCache] = (
            ReasoningCache(namespace="reasoning_analogical", default_ttl_seconds=self.cache_ttl_seconds)
            if self.enable_cache
            else None
        )
        logger.info("ReasoningAnalogical initialized | max_analogies=%s | cache=%s", self.max_analogies, self.enable_cache)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def perform_reasoning(self, target: Any, source_domain: Sequence[Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # type: ignore[override]
        """Perform analogical reasoning between one target and source candidates."""
        started = time.monotonic()
        context = dict(context or {})
        self._log_step("Starting analogical reasoning process")

        target = self._validate_target(target)
        sources = self._validate_source_domain(source_domain)
        cache_key = self._reasoning_cache_key(target, sources, context)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached["metrics"]["cache_hit"] = True
                return cached

        analogies = self.find_analogies(target, list(sources))
        filtered_analogies = self._filter_analogies(analogies)

        mappings: List[Dict[str, Any]] = []
        rejected_mappings: List[Dict[str, Any]] = []
        transfers: List[Dict[str, Any]] = []

        for analogy in filtered_analogies:
            mapping = self.create_mapping(target, analogy["item"], context)
            if mapping.get("accepted", False):
                mappings.append(mapping)
                transfer = self.transfer_knowledge(mapping, context)
                if transfer and transfer.get("adapted_knowledge"):
                    transfers.append(transfer)
            else:
                rejected_mappings.append(mapping)

        best_transfer = self.select_best_transfer(transfers)
        result = self._format_results(
            best_transfer=best_transfer,
            analogies=filtered_analogies,
            context=context,
            mappings=mappings,
            rejected_mappings=rejected_mappings,
            elapsed=elapsed_seconds(started),
        )
        result["metrics"]["cache_hit"] = False

        if self.cache is not None:
            self.cache.set(cache_key, result, ttl_seconds=self.cache_ttl_seconds, metadata={"reasoning_type": "analogical"})
        self._record_memory_event(context, result)
        self._log_step(
            f"Analogical reasoning complete | analogies={len(filtered_analogies)} | transfers={len(transfers)} "
            f"| success={best_transfer is not None}"
        )
        return result

    def find_analogies(self, target: Any, domain: List[Any]) -> List[Dict[str, Any]]: # type: ignore[override]
        """Find and score candidate analogies using multi-dimensional alignment."""
        if not domain:
            return []
        target_profile = self._profile_entity(target)
        candidates: List[AnalogyCandidate] = []
        for idx, item in enumerate(domain):
            source_profile = self._profile_entity(item)
            structural = self._calculate_similarity(target_profile["properties"], source_profile["properties"])
            functional = self._calculate_similarity(target_profile["functions"], source_profile["functions"])
            relational = self._calculate_relation_similarity(target_profile, source_profile)
            semantic = self._semantic_similarity(target_profile, source_profile)
            constraint = 1.0
            score = self._weighted_score(
                structural_score=structural,
                functional_score=functional,
                relational_score=relational,
                semantic_score=semantic,
                constraint_score=constraint,
            )
            matches = sorted(target_profile["properties"] & source_profile["properties"])
            candidate = AnalogyCandidate(
                item=item,
                index=idx,
                name=source_profile["name"],
                similarity=score,
                structural_score=structural,
                functional_score=functional,
                relational_score=relational,
                semantic_score=semantic,
                constraint_score=constraint,
                matching_properties=matches[: self.max_correspondences],
                confidence=score,
            )
            if score >= self.min_similarity:
                candidates.append(candidate)

        candidates.sort(key=lambda cand: (cand.similarity, cand.structural_score, cand.functional_score), reverse=True)
        self._log_step(f"Found {len(candidates)} analogies above threshold {self.min_similarity}")
        return [candidate.to_public_dict() for candidate in candidates]

    def _filter_analogies(self, analogies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter/rank analogies based on configured threshold and count."""
        filtered = [a for a in analogies if clamp_confidence(a.get("similarity", 0.0)) >= self.min_similarity]
        filtered.sort(key=lambda item: float(item.get("similarity", 0.0)), reverse=True)
        return filtered[: self.max_analogies]

    def create_mapping(self, target: Any, source: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured mapping between target and one source item."""
        target_profile = self._profile_entity(target)
        source_profile = self._profile_entity(source)
        self._log_step(f"Creating mapping: {source_profile['name']} -> {target_profile['name']}")

        correspondences = self._build_correspondences(target_profile, source_profile, context)
        structural_score = self._calculate_structural_score(
            target_profile["properties"], source_profile["properties"], correspondences
        )
        functional_score = self._calculate_functional_score(target, source, context)
        relational_score = self._calculate_relation_similarity(target_profile, source_profile)
        semantic_score = self._semantic_similarity(target_profile, source_profile)
        constraint_score = self._constraint_score(target_profile, source_profile, correspondences, context)
        mapping_score = self._weighted_score(
            structural_score=structural_score,
            functional_score=functional_score,
            relational_score=relational_score,
            semantic_score=semantic_score,
            constraint_score=constraint_score,
        )
        confidence = self._mapping_confidence(mapping_score, correspondences)
        accepted = mapping_score >= self.mapping_acceptance_threshold and bool(correspondences)
        reason = None if accepted else self._mapping_rejection_reason(mapping_score, correspondences)
        mapping = AnalogicalMapping(
            target_name=target_profile["name"],
            source_name=source_profile["name"],
            target=target,
            source=source,
            correspondences=correspondences,
            structural_score=structural_score,
            functional_score=functional_score,
            relational_score=relational_score,
            semantic_score=semantic_score,
            constraint_score=constraint_score,
            mapping_score=mapping_score,
            confidence=confidence,
            accepted=accepted,
            rejection_reason=reason,
        )
        return mapping.to_public_dict()

    def _is_corresponding(self, target_prop: str, source_prop: str, context: Dict[str, Any]) -> bool:
        """Determine whether two properties can be aligned."""
        similarity = self._property_similarity(target_prop, source_prop)
        if similarity < self.min_correspondence_similarity:
            return False
        constraints = context.get("property_constraints", {}) or {}
        allowed = constraints.get(target_prop)
        if allowed and source_prop not in set(map(str, allowed)):
            return False
        blocked = set(map(str, context.get("blocked_correspondences", []) or []))
        if f"{target_prop}:{source_prop}" in blocked or f"{source_prop}:{target_prop}" in blocked:
            return False
        return True

    def _property_similarity(self, prop1: str, prop2: str) -> float:
        """Calculate robust lexical similarity between two property labels."""
        p1, p2 = self._tokenize_property(prop1), self._tokenize_property(prop2)
        if not p1 and not p2:
            return 1.0
        if not p1 or not p2:
            return 0.0
        jaccard = len(p1 & p2) / len(p1 | p2)
        sequence = SequenceMatcher(None, str(prop1).lower(), str(prop2).lower()).ratio()
        synonym = self._synonym_overlap_score(prop1, prop2)
        return clamp_confidence(0.55 * jaccard + 0.30 * sequence + 0.15 * synonym)

    def _calculate_structural_score(self, target_props: Set[str], source_props: Set[str], correspondences: List[Dict[str, Any]]) -> float:
        """Calculate structure preservation score from coverage and match quality."""
        if not target_props and not source_props:
            return 1.0
        if not target_props or not source_props:
            return 0.0
        mapped_target = {str(c["target_property"]) for c in correspondences}
        mapped_source = {str(c["source_property"]) for c in correspondences}
        coverage = weighted_confidence(
            [
                len(mapped_target) / max(1, len(target_props)),
                len(mapped_source) / max(1, len(source_props)),
            ],
            [0.6, 0.4],
        )
        avg_match = weighted_confidence([float(c.get("similarity", 0.0)) for c in correspondences]) if correspondences else 0.0
        return clamp_confidence(0.65 * coverage + 0.35 * avg_match)

    def _calculate_functional_score(self, target: Any, source: Any, context: Dict[str, Any]) -> float:
        """Calculate functional similarity from functions/capabilities/goals."""
        target_funcs = set(self._extract_functions(target))
        source_funcs = set(self._extract_functions(source))
        if not target_funcs and not source_funcs:
            return 0.5
        base = self._calculate_similarity(target_funcs, source_funcs)
        goal_score = self._goal_alignment_score(target, source)
        weight_override = context.get("functional_weight_override")
        if weight_override is not None:
            try:
                return clamp_confidence(base * float(weight_override))
            except (TypeError, ValueError) as exc:
                raise ReasoningValidationError(
                    "functional_weight_override must be numeric",
                    cause=exc,
                    context={"functional_weight_override": weight_override},
                ) from exc
        return clamp_confidence(0.75 * base + 0.25 * goal_score)

    def _extract_functions(self, entity: Any) -> List[str]:
        """Extract functional properties without replacing BaseReasoning property extraction."""
        values: List[str] = []
        if isinstance(entity, Mapping):
            for key in ("functions", "capabilities", "behaviors", "actions", "goals", "purpose", "solutions"):
                values.extend(self._coerce_string_list(entity.get(key)))
        elif hasattr(entity, "functions"):
            values.extend(self._coerce_string_list(getattr(entity, "functions")))
        elif hasattr(entity, "capabilities"):
            values.extend(self._coerce_string_list(getattr(entity, "capabilities")))
        return sorted({token for value in values for token in self._tokenize_property(value)})

    def transfer_knowledge(self, mapping: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer source knowledge through accepted correspondences."""
        if not mapping.get("accepted", False):
            return {}
        source = mapping["source"]
        target = mapping["target"]
        source_name = str(mapping.get("source_name", self._entity_name(source)))
        target_name = str(mapping.get("target_name", self._entity_name(target)))
        self._log_step(f"Transferring knowledge from {source_name}")

        transferable = self._extract_transferable_knowledge(mapping, context)
        adapted_knowledge = self.adapt_knowledge(transferable, target, context)
        validation = self._validate_transfer(adapted_knowledge, mapping, context)
        confidence = clamp_confidence(float(mapping.get("confidence", mapping.get("mapping_score", 0.0))))
        transfer_score = self._transfer_score(adapted_knowledge, validation, mapping)
        record = TransferRecord(
            source_name=source_name,
            target_name=target_name,
            source=source,
            original_knowledge=transferable,
            adapted_knowledge=adapted_knowledge,
            mapping_score=clamp_confidence(mapping.get("mapping_score", 0.0)),
            confidence=confidence,
            transfer_score=transfer_score,
            validation=validation,
            mapping=mapping,
        )
        return record.to_public_dict()

    def adapt_knowledge(self, knowledge: List[Dict[str, Any]], target: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt transferred knowledge for the target domain."""
        adapted: List[Dict[str, Any]] = []
        constraints = context.get("constraints", {}) or {}
        for item in knowledge[: self.max_transfer_items]:
            if self._is_directly_applicable(item, target, constraints):
                adapted.append({**item, "adaptation": "direct", "confidence": self._direct_transfer_confidence(item)})
                continue
            transformed = self._apply_transformations(item, target, constraints)
            if transformed:
                adapted.append(transformed)
        return [item for item in adapted if clamp_confidence(item.get("confidence", 0.0)) >= self.transfer_confidence_floor]

    def _is_directly_applicable(self, item: Dict[str, Any], target: Any, constraints: Dict[str, Any]) -> bool:
        """Check whether transferred knowledge can be applied without transformation."""
        prop = str(item.get("target_property", "")).strip()
        if not prop:
            return False
        if prop in set(map(str, constraints.get("no_transfer", []) or [])):
            return False
        if prop in constraints and isinstance(constraints[prop], Mapping) and constraints[prop].get("no_transfer"):
            return False
        protected = set(map(str, constraints.get("protected_target_properties", []) or []))
        if prop in protected:
            return False
        return not self._target_has_property(target, prop)

    def _apply_transformations(self, item: Dict[str, Any], target: Any, constraints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply configured scalar/concept/relation fallback transformations."""
        prop = str(item.get("target_property", "")).strip()
        value = item.get("value")
        confidence = clamp_confidence(float(item.get("confidence", self.adaptation_threshold)) - self.transformation_penalty)

        prop_constraints = constraints.get(prop, {}) if isinstance(constraints.get(prop, {}), Mapping) else {}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scale_factor = float(prop_constraints.get("scale_factor", constraints.get("scale_factor", 1.0)))
            return {
                **item,
                "value": value * scale_factor,
                "adaptation": "scaled",
                "transformation": f"scaled by {scale_factor}",
                "confidence": clamp_confidence(confidence),
            }

        concept_map = constraints.get("concept_mappings", {}) or constraints.get("concept_map", {}) or {}
        if isinstance(value, str) and value in concept_map:
            return {
                **item,
                "value": concept_map[value],
                "adaptation": "concept_mapping",
                "transformation": f"concept mapping: {value} -> {concept_map[value]}",
                "confidence": clamp_confidence(confidence),
            }

        relation_map = constraints.get("relation_mappings", {}) or {}
        relation = item.get("relation")
        if relation in relation_map:
            return {
                **item,
                "relation": relation_map[relation],
                "adaptation": "relation_mapping",
                "transformation": f"relation mapping: {relation} -> {relation_map[relation]}",
                "confidence": clamp_confidence(confidence),
            }

        if self._target_has_property(target, prop):
            return None
        return {
            **item,
            "adaptation": "weak_projection",
            "transformation": "projected without direct source value transformation",
            "confidence": clamp_confidence(confidence * 0.8),
        }

    def select_best_transfer(self, transfers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best transfer by validation-aware transfer score."""
        if not transfers:
            return None
        return max(transfers, key=lambda t: (float(t.get("transfer_score", 0.0)), float(t.get("mapping_score", 0.0))))

    def _format_results(
        self,
        best_transfer: Optional[Dict[str, Any]],
        analogies: List[Dict[str, Any]],
        context: Dict[str, Any],
        mappings: Optional[List[Dict[str, Any]]] = None,
        rejected_mappings: Optional[List[Dict[str, Any]]] = None,
        elapsed: float = 0.0,
    ) -> Dict[str, Any]:
        """Format final results with compatibility keys and richer metadata."""
        mappings = mappings or []
        rejected_mappings = rejected_mappings or []
        result: Dict[str, Any] = {
            "best_transfer": best_transfer,
            "alternative_analogies": analogies,
            "mappings": mappings,
            "metrics": {
                "analogies_considered": len(analogies),
                "mappings_created": len(mappings),
                "rejected_mappings": len(rejected_mappings),
                "min_similarity": self.min_similarity,
                "mapping_score": best_transfer.get("mapping_score", 0.0) if best_transfer else 0.0,
                "transfer_score": best_transfer.get("transfer_score", 0.0) if best_transfer else 0.0,
                "success": best_transfer is not None,
                "elapsed_seconds": elapsed,
                "cache_enabled": self.cache is not None,
            },
            "reasoning_type": "analogical",
        }
        if self.include_rejected_mappings:
            result["rejected_mappings"] = rejected_mappings
        if self.return_context:
            result["context_used"] = context
        return json_safe_reasoning_state(result)

    # ------------------------------------------------------------------
    # Mapping internals
    # ------------------------------------------------------------------
    def _build_correspondences(
        self,
        target_profile: Dict[str, Any],
        source_profile: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        raw: List[PropertyCorrespondence] = []
        target_props = sorted(target_profile["properties"])
        source_props = sorted(source_profile["properties"])
        for t_prop in target_props:
            best: Optional[PropertyCorrespondence] = None
            for s_prop in source_props:
                if not self._is_corresponding(t_prop, s_prop, context):
                    continue
                sim = self._property_similarity(t_prop, s_prop)
                candidate = PropertyCorrespondence(
                    target_property=t_prop,
                    source_property=s_prop,
                    similarity=sim,
                    evidence={"target_tokens": sorted(self._tokenize_property(t_prop)), "source_tokens": sorted(self._tokenize_property(s_prop))},
                )
                if best is None or candidate.similarity > best.similarity:
                    best = candidate
            if best is not None:
                raw.append(best)
        raw.sort(key=lambda c: c.similarity, reverse=True)
        return [asdict(item) for item in raw[: self.max_correspondences]]

    def _extract_transferable_knowledge(self, mapping: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        source = mapping["source"]
        allow_unmapped = bool(context.get("allow_unmapped_transfer", False))
        transferable: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
        for corr in mapping.get("correspondences", [])[: self.max_correspondences]:
            source_prop = str(corr.get("source_property", "")).strip()
            target_prop = str(corr.get("target_property", "")).strip()
            if not source_prop or not target_prop:
                continue
            value_found, value = self._read_property(source, source_prop)
            if not value_found and not allow_unmapped:
                continue
            key = (source_prop, target_prop)
            if key in seen:
                continue
            seen.add(key)
            transferable.append(
                {
                    "source_property": source_prop,
                    "target_property": target_prop,
                    "value": value,
                    "relation": corr.get("relation", "property_alignment"),
                    "similarity": clamp_confidence(corr.get("similarity", 0.0)),
                    "confidence": self._item_confidence(corr, mapping),
                    "source_name": mapping.get("source_name", self._entity_name(source)),
                }
            )
        return transferable[: self.max_transfer_items]

    def _validate_transfer(self, adapted: List[Dict[str, Any]], mapping: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if not adapted:
            return {"valid": False, "coverage": 0.0, "average_confidence": 0.0, "reason": "no_adapted_knowledge"}
        required = set(map(str, context.get("required_target_properties", []) or []))
        adapted_props = {str(item.get("target_property")) for item in adapted}
        coverage = len(adapted_props & required) / len(required) if required else len(adapted) / max(1, len(mapping.get("correspondences", [])))
        avg_conf = weighted_confidence([item.get("confidence", 0.0) for item in adapted])
        valid = avg_conf >= self.adaptation_threshold and coverage > 0.0
        return {
            "valid": valid,
            "coverage": clamp_confidence(coverage),
            "average_confidence": avg_conf,
            "required_properties": sorted(required),
            "adapted_properties": sorted(adapted_props),
        }

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def _weighted_score(self, *, structural_score: float, functional_score: float, relational_score: float, semantic_score: float, constraint_score: float) -> float:
        return clamp_confidence(
            structural_score * self._weights["structural"]
            + functional_score * self._weights["functional"]
            + relational_score * self._weights["relational"]
            + semantic_score * self._weights["semantic"]
            + constraint_score * self._weights["constraint"]
        )

    def _calculate_relation_similarity(self, target_profile: Dict[str, Any], source_profile: Dict[str, Any]) -> float:
        target_rel = set(target_profile.get("relations", set()))
        source_rel = set(source_profile.get("relations", set()))
        if not target_rel and not source_rel:
            return 0.5
        return self._calculate_similarity(target_rel, source_rel)

    def _semantic_similarity(self, target_profile: Dict[str, Any], source_profile: Dict[str, Any]) -> float:
        target_text = " ".join(sorted(target_profile["tokens"]))
        source_text = " ".join(sorted(source_profile["tokens"]))
        lexical = SequenceMatcher(None, target_text, source_text).ratio() if target_text or source_text else 0.0
        overlap = self._calculate_similarity(set(target_profile["tokens"]), set(source_profile["tokens"]))
        name = SequenceMatcher(None, target_profile["name"].lower(), source_profile["name"].lower()).ratio()
        return clamp_confidence((1.0 - self.name_similarity_weight) * (0.55 * overlap + 0.45 * lexical) + self.name_similarity_weight * name)

    def _constraint_score(self, target_profile: Dict[str, Any], source_profile: Dict[str, Any], correspondences: List[Dict[str, Any]], context: Dict[str, Any]) -> float:
        constraints = context.get("property_constraints", {}) or {}
        if not constraints:
            return 1.0
        if not correspondences:
            return 0.0
        satisfied = 0
        checked = 0
        for corr in correspondences:
            target_prop = corr["target_property"]
            allowed = constraints.get(target_prop)
            if not allowed:
                continue
            checked += 1
            if corr["source_property"] in set(map(str, allowed)):
                satisfied += 1
        return 1.0 if checked == 0 else clamp_confidence(satisfied / checked)

    def _goal_alignment_score(self, target: Any, source: Any) -> float:
        target_goals = self._extract_named_values(target, ("goals", "objectives", "constraints", "requirements"))
        source_goals = self._extract_named_values(source, ("goals", "objectives", "principles", "constraints"))
        if not target_goals and not source_goals:
            return 0.5
        return self._calculate_similarity(set(target_goals), set(source_goals))

    def _mapping_confidence(self, mapping_score: float, correspondences: List[Dict[str, Any]]) -> float:
        evidence_factor = min(1.0, math.log1p(len(correspondences)) / math.log1p(max(2, self.max_correspondences)))
        return clamp_confidence(0.75 * mapping_score + 0.25 * evidence_factor)

    def _mapping_rejection_reason(self, mapping_score: float, correspondences: List[Dict[str, Any]]) -> str:
        if not correspondences:
            return "no_correspondences"
        if mapping_score < self.mapping_acceptance_threshold:
            return "mapping_score_below_threshold"
        return "rejected_by_policy"

    def _transfer_score(self, adapted: List[Dict[str, Any]], validation: Dict[str, Any], mapping: Dict[str, Any]) -> float:
        if not adapted:
            return 0.0
        avg_item_conf = weighted_confidence([item.get("confidence", 0.0) for item in adapted])
        return clamp_confidence(
            0.45 * float(mapping.get("mapping_score", 0.0))
            + 0.30 * avg_item_conf
            + 0.25 * float(validation.get("coverage", 0.0))
        )

    def _item_confidence(self, correspondence: Mapping[str, Any], mapping: Mapping[str, Any]) -> float:
        return clamp_confidence(0.65 * float(correspondence.get("similarity", 0.0)) + 0.35 * float(mapping.get("confidence", 0.0)))

    def _direct_transfer_confidence(self, item: Mapping[str, Any]) -> float:
        return clamp_confidence(float(item.get("confidence", 0.0)) + self.direct_transfer_bonus)

    # ------------------------------------------------------------------
    # Entity/profile helpers
    # ------------------------------------------------------------------
    def _profile_entity(self, entity: Any) -> Dict[str, Any]:
        properties = set(list(self._extract_properties(entity))[: self.max_property_tokens])
        functions = set(self._extract_functions(entity))
        relations = set(self._extract_relations(entity))
        tokens = set(properties) | set(functions) | set(relations) | set(self._tokenize_property(self._entity_name(entity)))
        return {
            "name": self._entity_name(entity),
            "properties": properties,
            "functions": functions,
            "relations": relations,
            "tokens": set(list(tokens)[: self.max_property_tokens]),
        }

    def _entity_name(self, entity: Any) -> str:
        if isinstance(entity, Mapping):
            for key in ("name", "id", "title", "label", "type"):
                if entity.get(key):
                    return str(entity[key])
        for attr in ("name", "id", "title", "label"):
            if hasattr(entity, attr):
                value = getattr(entity, attr)
                if value:
                    return str(value)
        text = str(entity)
        return text[:80] + ("..." if len(text) > 80 else "")

    def _extract_relations(self, entity: Any) -> List[str]:
        values: List[str] = []
        if isinstance(entity, Mapping):
            for key in ("relations", "relationships", "dependencies", "interactions", "edges", "links"):
                values.extend(self._coerce_string_list(entity.get(key)))
        elif hasattr(entity, "relations"):
            values.extend(self._coerce_string_list(getattr(entity, "relations")))
        return sorted({token for value in values for token in self._tokenize_property(value)})

    def _extract_named_values(self, entity: Any, keys: Sequence[str]) -> List[str]:
        values: List[str] = []
        if isinstance(entity, Mapping):
            for key in keys:
                values.extend(self._coerce_string_list(entity.get(key)))
        else:
            for key in keys:
                if hasattr(entity, key):
                    values.extend(self._coerce_string_list(getattr(entity, key)))
        return sorted({token for value in values for token in self._tokenize_property(value)})

    def _read_property(self, entity: Any, prop: str) -> Tuple[bool, Any]:
        if isinstance(entity, Mapping):
            if prop in entity:
                return True, entity[prop]
            for key, value in entity.items():
                key_tokens = self._tokenize_property(str(key))
                if prop in key_tokens or self._property_similarity(prop, str(key)) >= self.min_correspondence_similarity:
                    return True, value
        if hasattr(entity, prop):
            return True, getattr(entity, prop)
        return False, None

    def _target_has_property(self, target: Any, prop: str) -> bool:
        if isinstance(target, Mapping):
            if prop in target:
                return True
            return any(self._property_similarity(prop, str(key)) >= 0.95 for key in target.keys())
        return hasattr(target, prop)

    def _tokenize_property(self, value: Any) -> Set[str]:
        text = str(value).replace("_", " ").replace("-", " ").replace("/", " ").lower()
        tokens = {token.strip(".,:;!?()[]{}\"'") for token in text.split()}
        return {token for token in tokens if token and len(token) > 1}

    def _synonym_overlap_score(self, prop1: str, prop2: str) -> float:
        synonyms = self.analogical_config.get("synonym_groups", []) or []
        p1 = self._tokenize_property(prop1)
        p2 = self._tokenize_property(prop2)
        if not p1 or not p2 or not synonyms:
            return 0.0
        hits = 0
        total = 0
        for group in synonyms:
            group_tokens = {str(item).lower() for item in group}
            if p1 & group_tokens or p2 & group_tokens:
                total += 1
                if p1 & group_tokens and p2 & group_tokens:
                    hits += 1
        return hits / total if total else 0.0

    @staticmethod
    def _coerce_string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [str(k) for k in value.keys()] + [str(v) for v in value.values() if isinstance(v, (str, int, float, bool))]
        if isinstance(value, Iterable):
            return [str(item) for item in value]
        return [str(value)]

    # ------------------------------------------------------------------
    # Validation, config, and integration helpers
    # ------------------------------------------------------------------
    def _validate_target(self, target: Any) -> Any:
        if target is None:
            raise ReasoningValidationError("target cannot be None")
        return target

    def _validate_source_domain(self, source_domain: Sequence[Any]) -> List[Any]:
        if source_domain is None:
            raise ReasoningValidationError("source_domain cannot be None")
        if isinstance(source_domain, (str, bytes)) or not isinstance(source_domain, Sequence):
            if self.strict_inputs:
                raise ReasoningValidationError(
                    "source_domain must be a sequence of candidate source items",
                    context={"type": type(source_domain).__name__},
                )
            return [source_domain]
        sources = [item for item in source_domain if item is not None]
        if not sources:
            raise ReasoningValidationError("source_domain must contain at least one candidate")
        return sources

    def _non_negative_float(self, key: str, default: float) -> float:
        value = self.analogical_config.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ReasoningConfigurationError(
                f"reasoning_analogical.{key} must be numeric",
                cause=exc,
                context={"key": key, "value": value},
            ) from exc
        if parsed < 0.0:
            raise ReasoningConfigurationError(
                f"reasoning_analogical.{key} must be non-negative",
                context={"key": key, "value": parsed},
            )
        return parsed

    def _positive_float(self, key: str, default: float) -> float:
        parsed = self._non_negative_float(key, default)
        if parsed <= 0.0:
            raise ReasoningConfigurationError(f"reasoning_analogical.{key} must be positive", context={"key": key, "value": parsed})
        return parsed

    @staticmethod
    def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, float(v)) for v in weights.values())
        if total <= 0:
            raise ReasoningConfigurationError("reasoning_analogical scoring weights cannot all be zero")
        return {key: max(0.0, float(value)) / total for key, value in weights.items()}

    def _reasoning_cache_key(self, target: Any, sources: Sequence[Any], context: Mapping[str, Any]) -> str:
        payload = {
            "target": self._safe_repr(target),
            "sources": [self._safe_repr(src) for src in sources],
            "context": self._cache_safe_context(context),
            "version": "2.1.0",
        }
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _cache_safe_context(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        excluded = {"reasoning_memory", "memory", "cache", "logger"}
        return {str(k): self._safe_repr(v) for k, v in context.items() if str(k) not in excluded}

    @staticmethod
    def _safe_repr(value: Any, max_len: int = 1500) -> str:
        text = repr(value)
        return text[:max_len] + ("..." if len(text) > max_len else "")

    def _record_memory_event(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        if not self.record_memory_events:
            return
        memory = context.get("reasoning_memory") or context.get("memory")
        if memory is None or not hasattr(memory, "add"):
            return
        event = {
            "type": "reasoning_analogical",
            "success": result.get("metrics", {}).get("success", False),
            "mapping_score": result.get("metrics", {}).get("mapping_score", 0.0),
            "transfer_score": result.get("metrics", {}).get("transfer_score", 0.0),
            "created_at_ms": monotonic_timestamp_ms(),
        }
        memory.add(event, priority=self.memory_event_priority, tag="reasoning_analogical")

    def _log_step(self, message: str, level: str = "info") -> None:
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)


if __name__ == "__main__":
    print("\n=== Running Reasoning Analogical ===\n")
    printer.status("TEST", "Reasoning Analogical initialized", "info")

    reasoner = ReasoningAnalogical()
    target = {
        "name": "Urban Traffic Network",
        "functions": ["route packets", "allocate bandwidth", "maintain reliability"],
        "goals": ["efficiency", "resilience"],
        "relations": ["packets use links", "routers guide routing"],
    }
    sources = [
        {
            "name": "Computer Network",
            "functions": ["route packets", "allocate bandwidth", "maintain reliability"],
            "principles": ["efficiency", "resilience"],
            "relations": ["packets use links", "routers guide routing"],
            "solutions": ["load balancing", "priority routing"],
        },
        {
            "name": "Ecosystem",
            "functions": ["cycle resources", "balance populations"],
            "principles": ["diversity", "balance"],
            "solutions": ["keystone support"],
        },
    ]
    context = {
        "required_target_properties": ["functions", "relations"],
        "allow_unmapped_transfer": True,
        "constraints": {"concept_mappings": {"load balancing": "adaptive lane balancing"}},
    }
    result = reasoner.perform_reasoning(target, sources, context=context)
    assert result["reasoning_type"] == "analogical"
    assert result["metrics"]["analogies_considered"] >= 1
    assert result["metrics"]["success"] is True
    assert len(str(result["metrics"])) < 3000
    printer.pretty("Analogical summary", result["metrics"])

    print("\n=== Test ran successfully ===\n")
