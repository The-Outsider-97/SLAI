"""Production cause-and-effect reasoning strategy for the reasoning subsystem.

The strategy identifies candidate causal relationships from event streams,
validates them through temporal, correlational, mechanistic, conditional, and
counterfactual evidence, builds a compact causal graph, and emits predictions.
"""
from __future__ import annotations

import hashlib
import math
import re
import time

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from ..reasoning_cache import ReasoningCache
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from .base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Cause And Effect")
printer = PrettyPrinter()


@dataclass(frozen=True)
class CausalEvent:
    """Normalized event used by the cause-effect pipeline."""

    id: str
    description: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "timestamp": self.timestamp,
            "attributes": dict(self.attributes),
            "confidence": self.confidence,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CausalRelationship:
    """Candidate or validated causal link between two events."""

    cause: Dict[str, Any]
    effect: Dict[str, Any]
    type: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    attribute: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str, str, Optional[str]]:
        return (
            str(self.cause.get("id")),
            str(self.effect.get("id")),
            str(self.type),
            self.attribute,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "cause": dict(self.cause),
            "effect": dict(self.effect),
            "type": self.type,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata),
        }
        if self.attribute is not None:
            payload["attribute"] = self.attribute
        return payload


@dataclass(frozen=True)
class ValidationScores:
    """Decomposed relationship validation scores."""

    temporal: float
    correlation: float
    counterfactual: float
    mechanism: float
    condition: float
    confounder_penalty: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "temporal_score": self.temporal,
            "correlation_score": self.correlation,
            "counterfactual_score": self.counterfactual,
            "mechanism_score": self.mechanism,
            "condition_score": self.condition,
            "confounder_penalty": self.confounder_penalty,
        }


class ReasoningCauseAndEffect(BaseReasoning):
    """Cause-and-effect reasoning over ordered, attributed event streams.

    The class preserves the original public surface while making the internals
    deterministic, bounded, configurable, and explainable.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

    def __init__(self) -> None:
        super().__init__()
        self.config: Dict[str, Any] = load_global_config()
        self.cause_effect_config: Dict[str, Any] = get_config_section("reasoning_cause_effect")

        self.min_confidence: float = clamp_confidence(self.cause_effect_config.get("min_confidence", 0.7))
        self.max_chain_length: int = bounded_iterations(self.cause_effect_config.get("max_chain_length", 3), minimum=1, maximum=1000)
        self.enable_counterfactual: bool = bool(self.cause_effect_config.get("enable_counterfactual", True))
        self.temporal_weight: float = max(0.0, float(self.cause_effect_config.get("temporal_weight", 0.35)))
        self.correlation_weight: float = max(0.0, float(self.cause_effect_config.get("correlation_weight", 0.25)))
        self.counterfactual_weight: float = max(0.0, float(self.cause_effect_config.get("counterfactual_weight", 0.15)))
        self.mechanism_weight: float = max(0.0, float(self.cause_effect_config.get("mechanism_weight", 0.15)))
        self.condition_weight: float = max(0.0, float(self.cause_effect_config.get("condition_weight", 0.10)))
        self.network_mode: str = str(self.cause_effect_config.get("network_mode", "bayesian")).strip().lower()

        self.max_pair_distance: int = bounded_iterations(self.cause_effect_config.get("max_pair_distance", 5), minimum=1, maximum=10000)
        self.max_relationships: int = bounded_iterations(self.cause_effect_config.get("max_relationships", 256), minimum=1, maximum=100000)
        self.max_predictions: int = bounded_iterations(self.cause_effect_config.get("max_predictions", 128), minimum=1, maximum=100000)
        self.max_time_gap: float = max(1e-9, float(self.cause_effect_config.get("max_time_gap", 10.0)))
        self.temporal_decay: float = max(1e-9, float(self.cause_effect_config.get("temporal_decay", 1.0)))
        self.same_timestamp_policy: str = str(self.cause_effect_config.get("same_timestamp_policy", "allow_contextual")).strip().lower()
        self.sequential_base_confidence: float = clamp_confidence(self.cause_effect_config.get("sequential_base_confidence", 0.7))
        self.attribute_base_confidence: float = clamp_confidence(self.cause_effect_config.get("attribute_base_confidence", 0.65))
        self.pattern_base_confidence: float = clamp_confidence(self.cause_effect_config.get("pattern_base_confidence", 0.6))
        self.mechanism_base_confidence: float = clamp_confidence(self.cause_effect_config.get("mechanism_base_confidence", 0.75))
        self.condition_base_confidence: float = clamp_confidence(self.cause_effect_config.get("condition_base_confidence", 0.6))
        self.attribute_match_weight: float = clamp_confidence(self.cause_effect_config.get("attribute_match_weight", 0.35))
        self.token_overlap_weight: float = clamp_confidence(self.cause_effect_config.get("token_overlap_weight", 0.25))
        self.condition_match_weight: float = clamp_confidence(self.cause_effect_config.get("condition_match_weight", 0.25))
        self.counterfactual_default_score: float = clamp_confidence(self.cause_effect_config.get("counterfactual_default_score", 0.3))
        self.counterfactual_change_score: float = clamp_confidence(self.cause_effect_config.get("counterfactual_change_score", 0.8))
        self.confounder_penalty_weight: float = clamp_confidence(self.cause_effect_config.get("confounder_penalty_weight", 0.25))
        self.prediction_decay: float = clamp_confidence(self.cause_effect_config.get("prediction_decay", 0.85))
        self.intervention_default_strength: float = clamp_confidence(self.cause_effect_config.get("intervention_default_strength", 1.0))
        self.enable_cache: bool = bool(self.cause_effect_config.get("enable_cache", True))
        self.cache_ttl_seconds: float = float(self.cause_effect_config.get("cache_ttl_seconds", 300.0))
        self.strict_inputs: bool = bool(self.cause_effect_config.get("strict_inputs", True))
        self.return_context: bool = bool(self.cause_effect_config.get("return_context", False))
        self.include_rejected_relationships: bool = bool(self.cause_effect_config.get("include_rejected_relationships", True))
        self.enable_bayesian_projection: bool = bool(self.cause_effect_config.get("enable_bayesian_projection", True))

        self.causal_cue_map: Dict[str, List[str]] = self._normalize_cue_map(
            self.cause_effect_config.get("causal_cues", {})
        )
        self._cache: Optional[ReasoningCache] = ReasoningCache(
            namespace="reasoning_cause_effect",
            default_ttl_seconds=self.cache_ttl_seconds,
        ) if self.enable_cache else None
        self._last_run_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------
    def perform_reasoning(self, events: List[Any], conditions: Optional[Dict[str, Any]] = None, # type: ignore
                          context: Optional[dict] = None) -> Dict[str, Any]:
        """Perform cause-and-effect reasoning over an event stream."""
        start = time.monotonic()
        logger.info("Starting cause-and-effect reasoning")
        context = dict(context or {})
        conditions = dict(conditions or {})
        cache_key = self._build_cache_key(events, conditions, context)

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached = dict(cached)
                cached.setdefault("metrics", {})["cache_hit"] = True
                return cached

        normalized_events = self._normalize_events(events, context)
        relationships = self.identify_causal_relationships(normalized_events)

        validated_relationships: List[Dict[str, Any]] = []
        rejected_relationships: List[Dict[str, Any]] = []
        for relationship in relationships:
            validated = self.validate_relationship(relationship, conditions, context)
            if validated["is_valid"]:
                validated_relationships.append(validated)
            else:
                rejected_relationships.append(validated)

        causal_model = self.build_causal_model(validated_relationships, conditions)
        predictions = self.predict_outcomes(causal_model, context)
        result = self._format_results(
            validated_relationships,
            causal_model,
            predictions,
            context,
            rejected_relationships=rejected_relationships,
            all_relationships=relationships,
            elapsed=elapsed_seconds(start),
        )

        if self._cache is not None:
            self._cache.set(cache_key, result, ttl_seconds=self.cache_ttl_seconds)
        return result

    # ------------------------------------------------------------------
    # Event preparation
    # ------------------------------------------------------------------
    def _normalize_events(self, events: List[Any], context: Dict) -> List[Dict[str, Any]]:
        """Convert input events to normalized dictionaries sorted by time."""
        if events is None:
            raise ReasoningValidationError("events cannot be None")
        if not isinstance(events, list):
            if self.strict_inputs:
                raise ReasoningValidationError("events must be a list", context={"type": type(events).__name__})
            events = [events]
        if not events:
            raise ReasoningValidationError("events must be non-empty")

        normalized: List[Dict[str, Any]] = []
        for index, event in enumerate(events):
            normalized_event = self._normalize_single_event(event, index, context)
            for normalizer in context.get("event_normalizers", []) or []:
                if not callable(normalizer):
                    raise ReasoningValidationError("event_normalizers must be callable", context={"normalizer": normalizer})
                normalized_event = normalizer(normalized_event)
                normalized_event = self._normalize_single_event(normalized_event, index, context)
            normalized.append(normalized_event)

        normalized.sort(key=lambda item: (float(item["timestamp"]), str(item["id"])))
        seen: Set[str] = set()
        for item in normalized:
            if item["id"] in seen:
                raise ReasoningValidationError("duplicate event id after normalization", context={"id": item["id"]})
            seen.add(item["id"])
        return normalized

    def _normalize_single_event(self, event: Any, index: int, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(event, Mapping):
            raw_id = event.get("id") or event.get("event_id") or f"event_{index}"
            description = event.get("description") or event.get("event") or event.get("name") or str(event)
            timestamp = event.get("timestamp", event.get("time", index))
            attributes = event.get("attributes", {}) or {}
            confidence = event.get("confidence", 1.0)
            source = event.get("source", "input")
            metadata = dict(event.get("metadata", {}) or {})
        else:
            raw_id = f"event_{index}"
            description = str(event)
            timestamp = index
            attributes = {}
            confidence = 1.0
            source = "input"
            metadata = {}

        if not isinstance(attributes, Mapping):
            raise ReasoningValidationError("event attributes must be a mapping", context={"event": raw_id})
        event_obj = CausalEvent(
            id=str(raw_id).strip() or f"event_{index}",
            description=str(description).strip(),
            timestamp=self._normalize_timestamp(timestamp, fallback=float(index)),
            attributes={str(k): v for k, v in dict(attributes).items()},
            confidence=clamp_confidence(confidence),
            source=str(source),
            metadata=metadata,
        )
        if not event_obj.description:
            raise ReasoningValidationError("event description cannot be empty", context={"event": raw_id})
        return event_obj.to_dict()

    def _normalize_timestamp(self, value: Any, *, fallback: float) -> float:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        if isinstance(value, datetime):
            return float(value.timestamp())
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return fallback
            try:
                return float(text)
            except ValueError:
                pass
            try:
                return float(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
            except ValueError:
                if self.strict_inputs:
                    raise ReasoningValidationError("timestamp string is not numeric or ISO-8601", context={"timestamp": value})
        return fallback

    # ------------------------------------------------------------------
    # Candidate discovery
    # ------------------------------------------------------------------
    def identify_causal_relationships(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]: # type: ignore
        """Identify candidate cause-effect relationships with bounded search."""
        if len(events) < 2:
            return []
        candidates: List[CausalRelationship] = []
        candidates.extend(self._identify_sequential_relationships(events))
        candidates.extend(self._identify_pattern_relationships(events))
        candidates.extend(self._identify_attribute_relationships(events))
        candidates.extend(self._identify_condition_relationships(events))
        candidates.extend(self._identify_explicit_relationships(events))

        deduped: Dict[Tuple[str, str, str, Optional[str]], CausalRelationship] = {}
        for candidate in candidates:
            key = candidate.key()
            current = deduped.get(key)
            if current is None or candidate.confidence > current.confidence:
                deduped[key] = candidate

        ordered = sorted(deduped.values(), key=lambda rel: rel.confidence, reverse=True)[: self.max_relationships]
        logger.info("Identified %s candidate causal relationships", len(ordered))
        return [item.to_dict() for item in ordered]

    def _identify_sequential_relationships(self, events: List[Dict[str, Any]]) -> List[CausalRelationship]:
        relationships: List[CausalRelationship] = []
        for index, cause in enumerate(events[:-1]):
            upper = min(len(events), index + self.max_pair_distance + 1)
            for effect in events[index + 1: upper]:
                gap = float(effect["timestamp"]) - float(cause["timestamp"])
                if gap < 0 or (gap == 0 and self.same_timestamp_policy == "reject"):
                    continue
                distance = max(1, events.index(effect) - index)
                confidence = clamp_confidence(self.sequential_base_confidence / math.sqrt(distance))
                relationships.append(CausalRelationship(
                    cause=cause,
                    effect=effect,
                    type="sequential",
                    confidence=confidence,
                    evidence=["cause precedes effect in event order"],
                    metadata={"time_gap": gap, "event_distance": distance},
                ))
        return relationships

    def _identify_pattern_relationships(self, events: List[Dict[str, Any]]) -> List[CausalRelationship]:
        relationships: List[CausalRelationship] = []
        for i, cause in enumerate(events):
            for effect in events[i + 1: min(len(events), i + self.max_pair_distance + 1)]:
                token_score = self._token_similarity(cause["description"], effect["description"])
                cue_score = self._causal_cue_score(cause["description"], effect["description"])
                if max(token_score, cue_score) <= 0.0:
                    continue
                confidence = clamp_confidence(self.pattern_base_confidence * max(token_score, cue_score))
                relationships.append(CausalRelationship(
                    cause=cause,
                    effect=effect,
                    type="pattern_based",
                    confidence=confidence,
                    evidence=["description tokens or causal cue patterns align"],
                    metadata={"token_score": token_score, "cue_score": cue_score},
                ))
        return relationships

    def _identify_attribute_relationships(self, events: List[Dict[str, Any]]) -> List[CausalRelationship]:
        """Identify relationships from shared or shifted attributes."""
        relationships: List[CausalRelationship] = []
        for i, cause in enumerate(events):
            for effect in events[i + 1: min(len(events), i + self.max_pair_distance + 1)]:
                cause_attrs = dict(cause.get("attributes", {}))
                effect_attrs = dict(effect.get("attributes", {}))
                shared_keys = set(cause_attrs) & set(effect_attrs)
                for key in sorted(shared_keys):
                    same_value = cause_attrs.get(key) == effect_attrs.get(key)
                    value_similarity = 1.0 if same_value else self._token_similarity(cause_attrs.get(key), effect_attrs.get(key))
                    if value_similarity <= 0.0:
                        continue
                    relationships.append(CausalRelationship(
                        cause=cause,
                        effect=effect,
                        type="attribute_shared",
                        confidence=clamp_confidence(self.attribute_base_confidence * value_similarity),
                        attribute=key,
                        evidence=[f"shared attribute '{key}' links cause and effect"],
                        metadata={"attribute_similarity": value_similarity},
                    ))
        return relationships

    def _identify_condition_relationships(self, events: List[Dict[str, Any]]) -> List[CausalRelationship]:
        relationships: List[CausalRelationship] = []
        for i, cause in enumerate(events):
            for effect in events[i + 1: min(len(events), i + self.max_pair_distance + 1)]:
                cause_flags = set(map(str, cause.get("attributes", {}).get("causes", []) or []))
                effect_flags = set(map(str, effect.get("attributes", {}).get("effects", []) or []))
                overlaps = cause_flags & effect_flags
                if not overlaps:
                    continue
                relationships.append(CausalRelationship(
                    cause=cause,
                    effect=effect,
                    type="declared_mechanism",
                    confidence=self.mechanism_base_confidence,
                    evidence=[f"declared mechanism overlap: {', '.join(sorted(overlaps))}"],
                    metadata={"mechanisms": sorted(overlaps)},
                ))
        return relationships

    def _identify_explicit_relationships(self, events: List[Dict[str, Any]]) -> List[CausalRelationship]:
        by_id = {event["id"]: event for event in events}
        relationships: List[CausalRelationship] = []
        for event in events:
            links = event.get("metadata", {}).get("causes") or event.get("attributes", {}).get("causes") or []
            if isinstance(links, str):
                links = [links]
            for cause_id in links:
                cause = by_id.get(str(cause_id))
                if cause and cause["id"] != event["id"]:
                    relationships.append(CausalRelationship(
                        cause=cause,
                        effect=event,
                        type="explicit",
                        confidence=self.mechanism_base_confidence,
                        evidence=["explicit cause link supplied by event metadata"],
                    ))
        return relationships

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_relationship(self, relationship: Dict, conditions: Dict, context: Dict) -> Dict:
        """Validate a causal relationship using decomposed evidence scores."""
        self._validate_relationship_shape(relationship)
        cause = relationship["cause"]
        effect = relationship["effect"]
        scores = ValidationScores(
            temporal=self._validate_temporal(cause, effect, context),
            correlation=self._validate_correlation(cause, effect, conditions, context),
            counterfactual=self._counterfactual_analysis(cause, effect, context) if self.enable_counterfactual else 0.0,
            mechanism=self._validate_mechanism(cause, effect, context),
            condition=self._validate_conditions(cause, effect, conditions, context),
            confounder_penalty=self._confounder_penalty(cause, effect, conditions, context),
        )
        weighted_score = self._weighted_validation_score(scores)
        base_confidence = clamp_confidence(relationship.get("confidence", 0.5))
        combined = clamp_confidence((base_confidence * 0.35) + (weighted_score * 0.65))
        adjusted = clamp_confidence(combined * (1.0 - self.confounder_penalty_weight * scores.confounder_penalty))
        is_valid = adjusted >= self.min_confidence

        return {
            **relationship,
            "confidence": adjusted,
            "validated_confidence": adjusted,
            **scores.to_dict(),
            "validation_score": weighted_score,
            "is_valid": is_valid,
            "validation_status": "accepted" if is_valid else "rejected",
        }

    def _validate_relationship_shape(self, relationship: Mapping[str, Any]) -> None:
        for key in ("cause", "effect", "type"):
            if key not in relationship:
                raise ReasoningValidationError("causal relationship missing required field", context={"field": key})
        for role in ("cause", "effect"):
            if not isinstance(relationship[role], Mapping) or "id" not in relationship[role]:
                raise ReasoningValidationError("relationship endpoints must be normalized events", context={"role": role})

    def _validate_temporal(self, cause: Dict, effect: Dict, context: Dict) -> float:
        gap = float(effect.get("timestamp", 0.0)) - float(cause.get("timestamp", 0.0))
        if gap < 0:
            return 0.0
        if gap == 0:
            return 0.5 if self.same_timestamp_policy == "allow_contextual" else 0.0
        max_gap = max(1e-9, float(context.get("max_time_gap", self.max_time_gap)))
        if gap > max_gap:
            return clamp_confidence(math.exp(-(gap - max_gap) / max(1e-9, self.temporal_decay * max_gap)))
        return clamp_confidence(1.0 - (gap / (2.0 * max_gap)))

    def _validate_correlation(self, cause: Dict, effect: Dict, conditions: Dict, context: Dict) -> float:
        cause_attrs = dict(cause.get("attributes", {}))
        effect_attrs = dict(effect.get("attributes", {}))
        shared_keys = set(cause_attrs) & set(effect_attrs)
        matching_values = sum(1 for key in shared_keys if cause_attrs.get(key) == effect_attrs.get(key))
        attr_score = matching_values / max(1, len(shared_keys)) if shared_keys else 0.0
        token_score = self._token_similarity(cause.get("description", ""), effect.get("description", ""))
        condition_score = self._condition_overlap_score(cause_attrs, effect_attrs, conditions)
        external = context.get("correlation_scores", {}) or {}
        external_score = clamp_confidence(external.get((cause.get("id"), effect.get("id")), 0.0)) if isinstance(external, Mapping) else 0.0
        return clamp_confidence(max(external_score, attr_score * self.attribute_match_weight + token_score * self.token_overlap_weight + condition_score * self.condition_match_weight))

    def _counterfactual_analysis(self, cause: Dict, effect: Dict, context: Dict) -> float:
        simulated_effect = self._simulate_absence(cause, effect, context)
        original = str(effect.get("description", ""))
        if simulated_effect != original:
            return self.counterfactual_change_score
        counterfactuals = context.get("counterfactuals", {}) or {}
        key = (cause.get("id"), effect.get("id"))
        if isinstance(counterfactuals, Mapping) and key in counterfactuals:
            return clamp_confidence(counterfactuals[key])
        return self.counterfactual_default_score

    def _simulate_absence(self, cause: Dict, effect: Dict, context: Dict) -> str:
        simulator = context.get("counterfactual_simulator")
        if callable(simulator):
            result = simulator(cause, effect, context)
            return str(result)
        cause_tokens = self._tokens(cause.get("description", ""))
        effect_tokens = self._tokens(effect.get("description", ""))
        if {"rain", "rainfall", "storm"} & cause_tokens and {"wet", "flood", "water"} & effect_tokens:
            return "effect reduced without precipitation cause"
        if {"failure", "error", "overload"} & cause_tokens and {"shutdown", "alarm", "warning"} & effect_tokens:
            return "effect less likely without system fault"
        return str(effect.get("description", ""))

    def _validate_mechanism(self, cause: Dict, effect: Dict, context: Dict) -> float:
        context_mechanisms = context.get("mechanisms", {}) or {}
        key = (cause.get("id"), effect.get("id"))
        if isinstance(context_mechanisms, Mapping) and key in context_mechanisms:
            return clamp_confidence(context_mechanisms[key])
        cue_score = self._causal_cue_score(cause.get("description", ""), effect.get("description", ""))
        declared = self._declared_mechanism_overlap(cause, effect)
        return clamp_confidence(max(cue_score, declared))

    def _validate_conditions(self, cause: Dict, effect: Dict, conditions: Dict, context: Dict) -> float:
        if not conditions:
            return 0.5
        cause_attrs = dict(cause.get("attributes", {}))
        effect_attrs = dict(effect.get("attributes", {}))
        return self._condition_overlap_score(cause_attrs, effect_attrs, conditions)

    def _confounder_penalty(self, cause: Dict, effect: Dict, conditions: Dict, context: Dict) -> float:
        confounders = context.get("confounders", {}) or {}
        if not isinstance(confounders, Mapping):
            return 0.0
        direct = confounders.get((cause.get("id"), effect.get("id")))
        if direct is not None:
            return clamp_confidence(direct)
        cause_attrs = set(map(str, cause.get("attributes", {}).keys()))
        effect_attrs = set(map(str, effect.get("attributes", {}).keys()))
        confounder_keys = {str(k) for k in confounders.keys()}
        return clamp_confidence(len((cause_attrs & effect_attrs) & confounder_keys) / max(1, len(confounder_keys)))

    def _weighted_validation_score(self, scores: ValidationScores) -> float:
        weights = [self.temporal_weight, self.correlation_weight, self.counterfactual_weight, self.mechanism_weight, self.condition_weight]
        values = [scores.temporal, scores.correlation, scores.counterfactual, scores.mechanism, scores.condition]
        total = sum(weights)
        if total <= 0:
            return weighted_confidence(values)
        return clamp_confidence(sum(w * v for w, v in zip(weights, values)) / total)

    # ------------------------------------------------------------------
    # Model construction and prediction
    # ------------------------------------------------------------------
    def build_causal_model(self, relationships: List[Dict], conditions: Dict) -> Dict[str, Any]:
        """Build a graph-like causal model from validated relationships."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        for rel in relationships:
            if not rel.get("is_valid", False):
                continue
            cause = rel["cause"]
            effect = rel["effect"]
            nodes.setdefault(cause["id"], cause)
            nodes.setdefault(effect["id"], effect)
            edge = {
                "source": cause["id"],
                "target": effect["id"],
                "type": rel["type"],
                "confidence": rel["validated_confidence"],
                "evidence": rel.get("evidence", []),
            }
            edges.append(edge)

        edges = self._dedupe_edges(edges)
        chains = self._find_causal_chains(edges)
        adjacency = self._build_adjacency(edges)
        model = {
            "nodes": list(nodes.values()),
            "edges": edges,
            "chains": chains,
            "conditions": dict(conditions),
            "adjacency": adjacency,
            "roots": self._root_nodes(nodes, edges),
            "leaves": self._leaf_nodes(nodes, edges),
            "metrics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "chain_count": len(chains),
                "acyclic": self._is_acyclic(edges),
            },
        }
        if self.network_mode == "bayesian" and self.enable_bayesian_projection:
            model["network"] = self._build_bayesian_network(model)
        return model

    def _find_causal_chains(self, edges: List[Dict]) -> List[List[str]]:
        graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for edge in edges:
            graph[edge["source"]].append((edge["target"], float(edge.get("confidence", 0.0))))

        chains: List[List[str]] = []
        queue: Deque[Tuple[str, List[str]]] = deque((source, [source]) for source in graph.keys())
        while queue and len(chains) < self.max_relationships:
            node, path = queue.popleft()
            if len(path) > self.max_chain_length:
                continue
            for neighbor, _ in graph.get(node, []):
                if neighbor in path:
                    continue
                next_path = path + [neighbor]
                if len(next_path) > 1:
                    chains.append(next_path)
                if len(next_path) < self.max_chain_length:
                    queue.append((neighbor, next_path))
        return chains

    def _build_bayesian_network(self, model: Dict) -> Dict[str, Any]:
        network = {"nodes": {}, "edges": list(model.get("edges", [])), "cpt": {}}
        incoming: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in network["edges"]:
            incoming[edge["target"]].append(edge)
        for node in model.get("nodes", []):
            network["nodes"][node["id"]] = {"states": ["occurred", "not_occurred"], "description": node.get("description", "")}
        for node_id in network["nodes"]:
            parents = incoming.get(node_id, [])
            if not parents:
                network["cpt"][node_id] = {"prior": 0.5}
                continue
            avg_parent_conf = weighted_confidence([edge.get("confidence", 0.5) for edge in parents])
            network["cpt"][node_id] = {
                "parents": [edge["source"] for edge in parents],
                "probabilities": {
                    "parent_occurred": {"occurred": clamp_confidence(avg_parent_conf), "not_occurred": clamp_confidence(1 - avg_parent_conf)},
                    "parent_absent": {"occurred": 0.1, "not_occurred": 0.9},
                },
            }
        return network

    def predict_outcomes(self, model: Dict, context: Dict) -> List[Dict]:
        """Predict downstream outcomes from the causal model."""
        predictions: List[Dict[str, Any]] = []
        edge_lookup = {(edge["source"], edge["target"]): edge for edge in model.get("edges", [])}
        for chain in model.get("chains", []):
            confidence = 1.0
            for source, target in zip(chain, chain[1:]):
                confidence *= float(edge_lookup.get((source, target), {}).get("confidence", 0.5)) * self.prediction_decay
            prediction = {
                "cause": chain[0],
                "effect": chain[-1],
                "chain": chain,
                "type": "chain",
                "confidence": clamp_confidence(confidence),
            }
            predictions.append(self._apply_intervention(prediction, context))

        if self.network_mode == "bayesian" and "network" in model:
            predictions.extend(self._predict_with_bayesian_network(model["network"], context))
        predictions.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        return predictions[: self.max_predictions]

    def _predict_with_bayesian_network(self, network: Dict, context: Dict) -> List[Dict]:
        predictions: List[Dict[str, Any]] = []
        node_ids = set(network.get("nodes", {}).keys())
        targets = {edge["target"] for edge in network.get("edges", [])}
        roots = sorted(node_ids - targets)
        reachable = self._network_reachability(network.get("edges", []))
        for root in roots:
            for effect in sorted(reachable.get(root, set())):
                if root == effect:
                    continue
                predictions.append(self._apply_intervention({
                    "cause": root,
                    "effect": effect,
                    "type": "bayesian_projection",
                    "confidence": 0.75,
                    "path": [root, effect],
                }, context))
        return predictions

    # ------------------------------------------------------------------
    # Formatting and diagnostics
    # ------------------------------------------------------------------
    def _format_results(self, relationships: List[Dict], model: Dict, predictions: List[Dict],
                        context: Dict, *, rejected_relationships: Optional[List[Dict]] = None,
                        all_relationships: Optional[List[Dict]] = None, elapsed: float = 0.0) -> Dict[str, Any]:
        rejected_relationships = rejected_relationships or []
        all_relationships = all_relationships or relationships
        metrics = {
            "total_relationships": len(all_relationships),
            "valid_relationships": len(relationships),
            "rejected_relationships": len(rejected_relationships),
            "predictions_generated": len(predictions),
            "model_complexity": len(model.get("nodes", [])),
            "edge_count": len(model.get("edges", [])),
            "chain_count": len(model.get("chains", [])),
            "success": len(relationships) > 0,
            "elapsed_seconds": elapsed,
            "cache_hit": False,
        }
        self._last_run_metrics = metrics
        result = {
            "valid_relationships": relationships,
            "causal_model": model,
            "predictions": predictions,
            "metrics": metrics,
            "reasoning_type": "cause_effect",
        }
        if self.include_rejected_relationships:
            result["rejected_relationships"] = rejected_relationships
        if self.return_context:
            result["context_used"] = context
        return json_safe_reasoning_state(result)

    def diagnostics(self) -> Dict[str, Any]:
        return json_safe_reasoning_state({
            "min_confidence": self.min_confidence,
            "max_chain_length": self.max_chain_length,
            "network_mode": self.network_mode,
            "last_run_metrics": self._last_run_metrics,
            "cache": self._cache.metrics() if self._cache is not None else None,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _tokens(self, value: Any) -> Set[str]:
        return {token.lower() for token in self._TOKEN_RE.findall(str(value)) if token.strip()}

    def _token_similarity(self, left: Any, right: Any) -> float:
        a, b = self._tokens(left), self._tokens(right)
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b) if a | b else 0.0

    def _normalize_cue_map(self, cue_map: Any) -> Dict[str, List[str]]:
        defaults = {
            "trigger": ["alarm", "warning", "response", "activation"],
            "increase": ["rise", "growth", "surge", "overflow"],
            "failure": ["shutdown", "error", "fault", "degradation"],
            "rain": ["wet", "flood", "water", "saturation"],
        }
        if not isinstance(cue_map, Mapping):
            return defaults
        normalized = {str(k).lower(): [str(v).lower() for v in values] for k, values in cue_map.items() if isinstance(values, Sequence) and not isinstance(values, str)}
        defaults.update(normalized)
        return defaults

    def _causal_cue_score(self, cause_description: Any, effect_description: Any) -> float:
        cause_tokens = self._tokens(cause_description)
        effect_tokens = self._tokens(effect_description)
        if not cause_tokens or not effect_tokens:
            return 0.0
        matches = 0
        possible = 0
        for cause_cue, effect_cues in self.causal_cue_map.items():
            possible += 1
            if cause_cue in cause_tokens and set(effect_cues) & effect_tokens:
                matches += 1
        return clamp_confidence(matches / max(1, possible))

    def _declared_mechanism_overlap(self, cause: Mapping[str, Any], effect: Mapping[str, Any]) -> float:
        cause_mech = set(map(str, cause.get("attributes", {}).get("causes", []) or []))
        effect_mech = set(map(str, effect.get("attributes", {}).get("effects", []) or []))
        if not cause_mech or not effect_mech:
            return 0.0
        return clamp_confidence(len(cause_mech & effect_mech) / len(cause_mech | effect_mech))

    def _condition_overlap_score(self, cause_attrs: Mapping[str, Any], effect_attrs: Mapping[str, Any], conditions: Mapping[str, Any]) -> float:
        if not conditions:
            return 0.0
        matches = 0
        for key, value in conditions.items():
            if cause_attrs.get(key) == value or effect_attrs.get(key) == value:
                matches += 1
        return clamp_confidence(matches / max(1, len(conditions)))

    def _dedupe_edges(self, edges: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for edge in edges:
            key = (str(edge.get("source")), str(edge.get("target")), str(edge.get("type")))
            existing = dedup.get(key)
            if existing is None or float(edge.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                dedup[key] = dict(edge)
        return sorted(dedup.values(), key=lambda item: item.get("confidence", 0.0), reverse=True)

    def _build_adjacency(self, edges: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
        adjacency: Dict[str, List[str]] = defaultdict(list)
        for edge in edges:
            adjacency[str(edge["source"])].append(str(edge["target"]))
        return {key: sorted(set(value)) for key, value in adjacency.items()}

    def _root_nodes(self, nodes: Mapping[str, Dict[str, Any]], edges: Sequence[Mapping[str, Any]]) -> List[str]:
        targets = {str(edge["target"]) for edge in edges}
        return sorted(node for node in nodes if node not in targets)

    def _leaf_nodes(self, nodes: Mapping[str, Dict[str, Any]], edges: Sequence[Mapping[str, Any]]) -> List[str]:
        sources = {str(edge["source"]) for edge in edges}
        return sorted(node for node in nodes if node not in sources)

    def _is_acyclic(self, edges: Sequence[Mapping[str, Any]]) -> bool:
        graph = self._build_adjacency(edges)
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def dfs(node: str) -> bool:
            if node in visiting:
                return False
            if node in visited:
                return True
            visiting.add(node)
            for nxt in graph.get(node, []):
                if not dfs(nxt):
                    return False
            visiting.remove(node)
            visited.add(node)
            return True

        return all(dfs(node) for node in graph)

    def _network_reachability(self, edges: Sequence[Mapping[str, Any]]) -> Dict[str, Set[str]]:
        adjacency = self._build_adjacency(edges)
        reachable: Dict[str, Set[str]] = defaultdict(set)
        for start in adjacency:
            queue = deque(adjacency[start])
            while queue:
                node = queue.popleft()
                if node in reachable[start]:
                    continue
                reachable[start].add(node)
                queue.extend(adjacency.get(node, []))
        return reachable

    def _apply_intervention(self, prediction: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        interventions = context.get("interventions", {}) or {}
        cause = prediction.get("cause")
        intervention = interventions.get(cause) if isinstance(interventions, Mapping) else None
        if intervention is None:
            return prediction
        strength = intervention.get("strength", self.intervention_default_strength) if isinstance(intervention, Mapping) else self.intervention_default_strength
        adjusted = dict(prediction)
        adjusted["confidence"] = clamp_confidence(float(prediction.get("confidence", 0.0)) * clamp_confidence(strength))
        adjusted["intervention"] = intervention
        return adjusted

    def _build_cache_key(self, events: Any, conditions: Any, context: Mapping[str, Any]) -> str:
        safe_context = {k: v for k, v in context.items() if not callable(v) and k not in {"event_normalizers", "counterfactual_simulator"}}
        payload = json_safe_reasoning_state({"events": events, "conditions": conditions, "context": safe_context})
        return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print("\n=== Running Reasoning Cause And Effect ===\n")
    printer.status("TEST", "Reasoning Cause And Effect initialized", "info")

    engine = ReasoningCauseAndEffect()
    events = [
        {"id": "rain", "description": "Heavy rainfall", "timestamp": 1, "attributes": {"intensity": "high", "causes": ["flood"]}},
        {"id": "river", "description": "River water level rises", "timestamp": 2, "attributes": {"intensity": "high", "effects": ["flood"]}},
        {"id": "warning", "description": "Flood warning issued", "timestamp": 3, "attributes": {"severity": "high"}},
        {"id": "evac", "description": "Residents evacuate", "timestamp": 4, "attributes": {"severity": "high"}},
    ]
    result = engine.perform_reasoning(
        events,
        conditions={"intensity": "high"},
        context={"max_time_gap": 5, "interventions": {"rain": {"strength": 0.8}}},
    )
    assert result["metrics"]["success"] is True
    assert result["causal_model"]["metrics"]["edge_count"] >= 1
    assert result["predictions"]
    cached = engine.perform_reasoning(events, conditions={"intensity": "high"}, context={"max_time_gap": 5})
    assert cached["metrics"]["success"] is True
    printer.pretty("Cause Effect Metrics", result["metrics"], "success")
    print("\n=== Test ran successfully ===\n")
