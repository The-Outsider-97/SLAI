"""Inductive reasoning strategy for the reasoning subsystem.

Induction derives general principles from specific observations.  This module
keeps the strategy API stable for ``ReasoningTypes`` while improving runtime
safety, diagnostics, cache integration, validation, and prediction quality.

Pipeline
--------
1. Normalize and validate observations with confidence/source metadata.
2. Identify temporal, numeric, categorical, content, and contextual patterns.
3. Formulate a generalized theory with support, scope, confidence, and risk.
4. Validate the theory against knowledge, counterexamples, and optional holdout.
5. Produce bounded predictions and return an explainable result payload.
"""
from __future__ import annotations

import math
import time

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np # type: ignore

from ..reasoning_cache import ReasoningCache
from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from .base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Inductive")
printer = PrettyPrinter()


@dataclass(frozen=True)
class InductiveObservation:
    """Canonical observation used by the inductive pipeline."""

    id: str
    content: Any
    timestamp: float
    confidence: float
    source: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(asdict(self))


@dataclass(frozen=True)
class PatternRecord:
    """Structured pattern record with support and metadata."""

    type: str
    confidence: float
    support: int
    statement: str
    dimension: str = "general"
    attribute: Optional[str] = None
    value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(asdict(self))


@dataclass(frozen=True)
class TheoryRecord:
    """Generalized theory produced from one or more patterns."""

    theory: str
    confidence: float
    scope: str
    support: int
    generalization_risk: float
    supporting_patterns: List[Dict[str, Any]]
    statements: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return json_safe_reasoning_state(asdict(self))


class ReasoningInductive(BaseReasoning):
    """Production inductive reasoning strategy.

    The public method names intentionally mirror the original implementation so
    existing callers can continue using the strategy through ``ReasoningTypes``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.inductive_config: Dict[str, Any] = get_config_section("reasoning_inductive", self.config)

        self.min_observations: int = bounded_iterations(
            self.inductive_config.get("min_observations", 5), minimum=1, maximum=100_000
        )
        self.max_observations: int = bounded_iterations(
            self.inductive_config.get("max_observations", 10_000), minimum=1, maximum=1_000_000
        )
        self.max_patterns: int = bounded_iterations(
            self.inductive_config.get("max_patterns", 128), minimum=1, maximum=100_000
        )
        self.max_predictions: int = bounded_iterations(
            self.inductive_config.get("max_predictions", 32), minimum=1, maximum=100_000
        )
        self.max_attribute_values: int = bounded_iterations(
            self.inductive_config.get("max_attribute_values", 512), minimum=1, maximum=100_000
        )

        self.confidence_threshold: float = clamp_confidence(
            self.inductive_config.get("confidence_threshold", 0.7)
        )
        self.min_observation_confidence: float = clamp_confidence(
            self.inductive_config.get("min_observation_confidence", 0.0)
        )
        self.min_pattern_support: int = bounded_iterations(
            self.inductive_config.get("min_pattern_support", 2), minimum=1, maximum=100_000
        )
        self.dominant_value_threshold: float = clamp_confidence(
            self.inductive_config.get("dominant_value_threshold", 0.7)
        )
        self.counterexample_tolerance: float = clamp_confidence(
            self.inductive_config.get("counterexample_tolerance", 0.2)
        )
        self.numeric_residual_tolerance: float = max(
            0.0, float(self.inductive_config.get("numeric_residual_tolerance", 0.20))
        )

        self.extrapolation_limit: float = max(0.0, float(self.inductive_config.get("extrapolation_limit", 1.5)))
        self.trend_analysis_weight: float = clamp_confidence(
            self.inductive_config.get("trend_analysis_weight", 0.45)
        )
        self.pattern_analysis_weight: float = clamp_confidence(
            self.inductive_config.get("pattern_analysis_weight", 0.35)
        )
        self.attribute_analysis_weight: float = clamp_confidence(
            self.inductive_config.get("attribute_analysis_weight", 0.15)
        )
        self.diversity_weight: float = clamp_confidence(
            self.inductive_config.get("diversity_weight", 0.05)
        )

        self.strict_inputs: bool = bool(self.inductive_config.get("strict_inputs", True))
        self.include_context: bool = bool(self.inductive_config.get("return_context", False))
        self.include_rejected_observations: bool = bool(
            self.inductive_config.get("include_rejected_observations", True)
        )
        self.enable_cache: bool = bool(self.inductive_config.get("enable_cache", True))
        self.cache_ttl_seconds: float = float(self.inductive_config.get("cache_ttl_seconds", 300.0))
        self.record_memory_events: bool = bool(self.inductive_config.get("record_memory_events", False))

        self._cache: Optional[ReasoningCache] = None
        if self.enable_cache:
            self._cache = ReasoningCache(
                namespace="reasoning_inductive",
                default_ttl_seconds=self.cache_ttl_seconds,
            )
        self._last_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def perform_reasoning(self, observations: List[Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # type: ignore
        """Run inductive reasoning from observations to theory and predictions."""
        started = time.monotonic()
        context = dict(context or {})
        cache_key = self._cache_key(observations, context)

        def _run() -> Dict[str, Any]:
            return self._perform_reasoning_uncached(observations, context, started)

        if self._cache is not None and not context.get("disable_cache", False):
            return self._cache.get_or_set(
                cache_key,
                _run,
                ttl_seconds=self.cache_ttl_seconds,
                metadata={"strategy": "inductive"},
            )
        return _run()

    def diagnostics(self) -> Dict[str, Any]:
        """Return runtime diagnostics for the last run and cache state."""
        return json_safe_reasoning_state(
            {
                "strategy": "reasoning_inductive",
                "last_metrics": self._last_metrics,
                "cache": self._cache.metrics() if self._cache is not None else None,
                "config": {
                    "min_observations": self.min_observations,
                    "confidence_threshold": self.confidence_threshold,
                    "max_patterns": self.max_patterns,
                    "max_predictions": self.max_predictions,
                },
            }
        )

    # ------------------------------------------------------------------
    # Pipeline internals
    # ------------------------------------------------------------------
    def _perform_reasoning_uncached(self, observations: Any, context: Dict[str, Any], started: float) -> Dict[str, Any]:
        logger.info("Starting inductive reasoning process")
        validated_obs = self._validate_observations(observations, context)
        rejected = context.get("_rejected_observations", [])

        if len(validated_obs) < self.min_observations:
            validation = {
                "is_valid": False,
                "confidence": 0.0,
                "tests": [],
                "reasons": [f"Insufficient observations ({len(validated_obs)}/{self.min_observations})"],
            }
            result = self._format_results({}, validated_obs, [], validation, [], context)
            result["rejected_observations"] = rejected if self.include_rejected_observations else []
            result["metrics"]["elapsed_seconds"] = elapsed_seconds(started)
            self._last_metrics = result["metrics"]
            return result

        patterns = self._identify_patterns(validated_obs, context)
        theory = self._formulate_theory(patterns, validated_obs, context)
        validation_result = self._validate_theory(theory, validated_obs, context)
        predictions = self._make_predictions(theory, validated_obs, context)
        result = self._format_results(theory, validated_obs, patterns, validation_result, predictions, context)
        result["rejected_observations"] = rejected if self.include_rejected_observations else []
        result["metrics"]["elapsed_seconds"] = elapsed_seconds(started)
        self._last_metrics = result["metrics"]
        self._record_memory_event(result, context)
        return result

    def _cache_key(self, observations: Any, context: Mapping[str, Any]) -> Tuple[str, str, str]:
        safe_context = {
            key: value
            for key, value in context.items()
            if key not in {"observation_filter", "pattern_generators", "pattern_to_statement"}
            and not callable(value)
        }
        return ("inductive", repr(observations), repr(safe_context))

    # ------------------------------------------------------------------
    # Observation validation
    # ------------------------------------------------------------------
    def _validate_observations(self, observations: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and standardize observations."""
        if observations is None:
            if self.strict_inputs:
                raise ReasoningValidationError("observations cannot be None")
            return []
        raw_items = list(observations) if isinstance(observations, (list, tuple, set)) else [observations]
        if len(raw_items) > self.max_observations:
            raise ReasoningValidationError(
                "Observation count exceeds reasoning_inductive.max_observations",
                context={"count": len(raw_items), "max_observations": self.max_observations},
            )

        validated: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        obs_filter = context.get("observation_filter")
        if obs_filter is not None and not callable(obs_filter):
            raise ReasoningValidationError("context.observation_filter must be callable when provided")

        for idx, raw in enumerate(raw_items):
            try:
                obs = self._standardize_observation(raw, idx)
                if obs["confidence"] < self.min_observation_confidence:
                    rejected.append({"index": idx, "reason": "low_confidence", "observation": obs})
                    continue
                if obs_filter is not None and not bool(obs_filter(obs)):
                    rejected.append({"index": idx, "reason": "filtered", "observation": obs})
                    continue
                validated.append(obs)
            except ReasoningError:
                raise
            except Exception as exc:
                if self.strict_inputs:
                    raise ReasoningValidationError(
                        "Failed to normalize observation",
                        cause=exc,
                        context={"index": idx, "raw": raw},
                    ) from exc
                rejected.append({"index": idx, "reason": f"normalization_error:{exc}", "raw": raw})

        validated.sort(key=lambda item: (item["timestamp"], item["id"]))
        context["_rejected_observations"] = rejected
        return validated

    def _standardize_observation(self, obs: Any, idx: int) -> Dict[str, Any]:
        if isinstance(obs, Mapping):
            content = obs.get("content", obs)
            attributes = self._extract_attributes(content)
            explicit_attrs = obs.get("attributes", {})
            if isinstance(explicit_attrs, Mapping):
                attributes.update(dict(explicit_attrs))
            confidence_source = obs.get("confidence", obs.get("weight", 1.0))
            canonical = InductiveObservation(
                id=str(obs.get("id", f"obs_{idx}")).strip() or f"obs_{idx}",
                content=content,
                timestamp=self._normalize_timestamp(obs.get("timestamp", idx), idx),
                confidence=clamp_confidence(confidence_source),
                source=str(obs.get("source", "unknown")).strip() or "unknown",
                attributes=attributes,
                raw=obs,
            )
        else:
            canonical = InductiveObservation(
                id=f"obs_{idx}",
                content=obs,
                timestamp=float(idx),
                confidence=1.0,
                source="direct",
                attributes=self._extract_attributes(obs),
                raw=obs,
            )
        return canonical.to_dict()

    @staticmethod
    def _normalize_timestamp(value: Any, fallback: int) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    def _extract_attributes(self, content: Any) -> Dict[str, Any]:
        if isinstance(content, Mapping):
            return {str(k): v for k, v in content.items() if not isinstance(v, (dict, list, tuple, set))}
        if hasattr(content, "__dict__"):
            return {str(k): v for k, v in vars(content).items() if not callable(v)}
        if isinstance(content, (int, float)) and not isinstance(content, bool):
            return {"value": float(content)}
        return {}

    # ------------------------------------------------------------------
    # Pattern identification
    # ------------------------------------------------------------------
    def _identify_patterns(self, observations: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        patterns: List[PatternRecord] = []
        patterns.extend(self._identify_temporal_patterns(observations))
        patterns.extend(self._identify_attribute_patterns_records(observations))
        patterns.extend(self._identify_content_patterns(observations))

        generators = context.get("pattern_generators", [])
        if generators:
            if not isinstance(generators, Sequence):
                raise ReasoningValidationError("context.pattern_generators must be a sequence of callables")
            for generator in generators:
                if not callable(generator):
                    raise ReasoningValidationError("Each pattern generator must be callable")
                generated = generator(observations)
                if not isinstance(generated, (list, tuple, set)):
                    generated = [generated]   # treat single item as iterable of one
                for item in generated:
                    record = self._coerce_pattern(item)
                    if record is not None:
                        patterns.append(record)

        dedup: Dict[Tuple[str, str, str, str], PatternRecord] = {}
        for pattern in patterns:
            if pattern.support < self.min_pattern_support:
                continue
            if pattern.confidence < self.confidence_threshold:
                continue
            key = (pattern.type, pattern.dimension, str(pattern.attribute), str(pattern.value))
            existing = dedup.get(key)
            if existing is None or pattern.confidence > existing.confidence:
                dedup[key] = pattern
        ordered = sorted(dedup.values(), key=lambda p: (p.confidence, p.support), reverse=True)
        return [p.to_dict() for p in ordered[: self.max_patterns]]

    def _coerce_pattern(self, item: Any) -> Optional[PatternRecord]:
        if not isinstance(item, Mapping):
            return None
        ptype = str(item.get("type", "contextual")).strip() or "contextual"
        confidence = clamp_confidence(item.get("confidence", self.confidence_threshold))
        support = bounded_iterations(item.get("support", self.min_pattern_support), minimum=1, maximum=1_000_000)
        statement = str(item.get("statement", f"Contextual pattern: {ptype}"))
        return PatternRecord(
            type=ptype,
            confidence=confidence,
            support=support,
            statement=statement,
            dimension=str(item.get("dimension", "contextual")),
            attribute=item.get("attribute"),
            value=item.get("value"),
            metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata", {}), Mapping) else {},
        )

    def _identify_temporal_patterns(self, observations: List[Dict[str, Any]]) -> List[PatternRecord]:
        timestamps = [float(obs["timestamp"]) for obs in observations]
        patterns: List[PatternRecord] = []
        if len(timestamps) < 3:
            return patterns
        diffs = np.diff(np.asarray(timestamps, dtype=float))
        if np.all(diffs > 0):
            regularity = self._regularity_score(diffs)
            patterns.append(
                PatternRecord(
                    type="temporal_order",
                    confidence=clamp_confidence(0.75 + 0.25 * regularity),
                    support=len(timestamps),
                    statement="Observations form a forward temporal sequence",
                    dimension="temporal",
                    metadata={"regularity": regularity},
                )
            )
        if len(diffs) >= 2 and self._regularity_score(diffs) >= 0.75:
            patterns.append(
                PatternRecord(
                    type="periodic",
                    confidence=clamp_confidence(0.70 + 0.25 * self._regularity_score(diffs)),
                    support=len(timestamps),
                    statement=f"Observations recur with an approximate period of {float(np.mean(diffs)):.3f}",
                    dimension="temporal",
                    metadata={"period": float(np.mean(diffs)), "std": float(np.std(diffs))},
                )
            )
        return patterns

    def _identify_attribute_patterns_records(self, observations: List[Dict[str, Any]]) -> List[PatternRecord]:
        patterns: List[PatternRecord] = []
        attribute_keys: Set[str] = set()
        for obs in observations:
            attribute_keys.update(str(k) for k in obs.get("attributes", {}).keys())

        for key in sorted(attribute_keys):
            values = [obs.get("attributes", {}).get(key) for obs in observations if key in obs.get("attributes", {})]
            values = values[: self.max_attribute_values]
            if len(values) < self.min_pattern_support:
                continue
            if all(self._is_number(v) for v in values):
                patterns.extend(self._numeric_attribute_patterns(key, [float(v) for v in values]))
            else:
                patterns.extend(self._categorical_attribute_patterns(key, values))
        return patterns

    # Original public helper retained: returns list[dict]
    def _identify_attribute_patterns(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._identify_attribute_patterns_records(observations)]

    def _numeric_attribute_patterns(self, key: str, values: List[float]) -> List[PatternRecord]:
        patterns: List[PatternRecord] = []
        if len(values) < 2:
            return patterns
        x = np.arange(len(values), dtype=float)
        y = np.asarray(values, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        r2 = self._r_squared(y, predicted)
        direction = "increases" if slope > 0 else "decreases" if slope < 0 else "remains stable"
        confidence = clamp_confidence(0.50 + 0.50 * r2)
        if abs(slope) > 1e-12:
            patterns.append(
                PatternRecord(
                    type="numeric_trend",
                    confidence=confidence,
                    support=len(values),
                    statement=f"Attribute '{key}' {direction} over observations",
                    dimension="attribute",
                    attribute=key,
                    metadata={"slope": float(slope), "intercept": float(intercept), "r_squared": float(r2)},
                )
            )
        if np.std(y) <= max(1e-9, abs(float(np.mean(y))) * 0.05):
            patterns.append(
                PatternRecord(
                    type="stable_numeric_value",
                    confidence=0.85,
                    support=len(values),
                    statement=f"Attribute '{key}' remains stable around {float(np.mean(y)):.3f}",
                    dimension="attribute",
                    attribute=key,
                    value=float(np.mean(y)),
                    metadata={"std": float(np.std(y))},
                )
            )
        return patterns

    def _categorical_attribute_patterns(self, key: str, values: List[Any]) -> List[PatternRecord]:
        counts = Counter(str(v) for v in values)
        total = sum(counts.values())
        if total <= 0:
            return []
        value, count = counts.most_common(1)[0]
        frequency = count / total
        if frequency < self.dominant_value_threshold:
            return []
        return [
            PatternRecord(
                type="dominant_value",
                confidence=clamp_confidence(frequency * 1.1),
                support=count,
                statement=f"Most observations ({frequency*100:.1f}%) have {key} = {value}",
                dimension="attribute",
                attribute=key,
                value=value,
                metadata={"frequency": frequency, "total": total},
            )
        ]

    def _identify_content_patterns(self, observations: List[Dict[str, Any]]) -> List[PatternRecord]:
        tokens: List[str] = []
        for obs in observations:
            tokens.extend(self._tokenize(obs.get("content")))
        if not tokens:
            return []
        counts = Counter(tokens)
        total_obs = max(1, len(observations))
        patterns: List[PatternRecord] = []
        for token, count in counts.most_common(5):
            frequency = count / total_obs
            if frequency >= self.dominant_value_threshold:
                patterns.append(
                    PatternRecord(
                        type="recurring_content_token",
                        confidence=clamp_confidence(frequency),
                        support=count,
                        statement=f"Token '{token}' recurs across observations",
                        dimension="content",
                        value=token,
                        metadata={"frequency": frequency},
                    )
                )
        return patterns

    @staticmethod
    def _tokenize(value: Any) -> List[str]:
        text = str(value).lower().replace("_", " ")
        return [part.strip(".,:;!?()[]{}\"'") for part in text.split() if len(part.strip()) > 2]

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))

    @staticmethod
    def _regularity_score(values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        mean = float(np.mean(values))
        if abs(mean) <= 1e-12:
            return 0.0
        cv = float(np.std(values) / abs(mean))
        return clamp_confidence(1.0 - min(cv, 1.0))

    @staticmethod
    def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        if ss_tot <= 1e-12:
            return 1.0 if ss_res <= 1e-12 else 0.0
        return clamp_confidence(1.0 - ss_res / ss_tot)

    # ------------------------------------------------------------------
    # Theory formulation and validation
    # ------------------------------------------------------------------
    def _formulate_theory(self, patterns: List[Dict[str, Any]], observations: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        if not patterns:
            return TheoryRecord(
                theory="No generalizable patterns found",
                confidence=0.0,
                scope="none",
                support=len(observations),
                generalization_risk=1.0,
                supporting_patterns=[],
                statements=[],
            ).to_dict()

        statements: List[str] = []
        confidences: List[float] = []
        weights: List[float] = []
        for pattern in patterns:
            statement = self._pattern_to_statement(pattern, observations, context)
            if statement:
                statements.append(str(statement["statement"]))
                confidence = clamp_confidence(statement.get("confidence", pattern.get("confidence", 0.0)))
                confidences.append(confidence)
                weights.append(self._pattern_weight(pattern))
        if not statements:
            statements = [str(p.get("statement", p.get("type", "pattern"))) for p in patterns]
            confidences = [clamp_confidence(p.get("confidence", 0.0)) for p in patterns]
            weights = [self._pattern_weight(p) for p in patterns]

        base_conf = weighted_confidence(confidences, weights) if confidences else 0.0
        diversity = self._source_diversity(observations)
        support_ratio = min(1.0, len(observations) / max(1, self.min_observations * 2))
        generalization_risk = clamp_confidence(1.0 - (0.65 * support_ratio + 0.35 * diversity))
        confidence = clamp_confidence(base_conf * (1.0 - 0.35 * generalization_risk) + self.diversity_weight * diversity)
        scope = self._determine_scope(observations, diversity, support_ratio)
        theory_text = " AND ".join(statements[: self.max_patterns])
        return TheoryRecord(
            theory=theory_text,
            confidence=confidence,
            scope=scope,
            support=len(observations),
            generalization_risk=generalization_risk,
            supporting_patterns=patterns,
            statements=statements,
        ).to_dict()

    def _pattern_weight(self, pattern: Mapping[str, Any]) -> float:
        ptype = str(pattern.get("type", ""))
        if ptype in {"numeric_trend", "stable_numeric_value"}:
            return self.trend_analysis_weight
        if ptype in {"dominant_value", "temporal_order", "periodic"}:
            return self.pattern_analysis_weight
        if str(pattern.get("dimension")) == "attribute":
            return self.attribute_analysis_weight
        return max(0.01, 1.0 - self.trend_analysis_weight - self.pattern_analysis_weight)

    def _pattern_to_statement(self, pattern: Dict[str, Any], observations: List[Dict[str, Any]], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "pattern_to_statement" in context:
            converter = context["pattern_to_statement"]
            if not callable(converter):
                raise ReasoningValidationError("context.pattern_to_statement must be callable")
            converted = converter(pattern, observations)
            if isinstance(converted, dict):
                return converted
            elif converted is not None:
                logger.warning(f"pattern_to_statement returned unexpected type {type(converted).__name__}, ignoring")
            return None
        return {
            "statement": pattern.get("statement", f"Pattern detected: {pattern.get('type', 'unknown')}"),
            "confidence": clamp_confidence(pattern.get("confidence", 0.0)),
        }

    def _determine_scope(self, observations: List[Dict[str, Any]], diversity: float, support_ratio: float) -> str:
        if diversity >= 0.7 and support_ratio >= 0.8:
            return "broad"
        if diversity >= 0.4 or support_ratio >= 0.6:
            return "moderate"
        return "narrow"

    @staticmethod
    def _source_diversity(observations: List[Dict[str, Any]]) -> float:
        if not observations:
            return 0.0
        return len({obs.get("source", "unknown") for obs in observations}) / len(observations)

    def _validate_theory(self, theory: Dict[str, Any], observations: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        if clamp_confidence(theory.get("confidence", 0.0)) < self.confidence_threshold:
            return {
                "is_valid": False,
                "confidence": clamp_confidence(theory.get("confidence", 0.0)),
                "tests": [],
                "reasons": ["Theory confidence below threshold"],
            }

        tests: List[Dict[str, Any]] = []
        knowledge_consistency = self._check_knowledge_consistency(theory, context)
        tests.append({"test": "knowledge_consistency", "result": knowledge_consistency["consistent"], "confidence": knowledge_consistency["confidence"], **knowledge_consistency})

        counterexample_check = self._check_counterexamples(theory, observations, context)
        tests.append({"test": "counterexample_check", "result": not counterexample_check["found"], "confidence": counterexample_check["confidence"], **counterexample_check})

        if "holdout_data" in context:
            holdout = self._validate_observations(context["holdout_data"], {k: v for k, v in context.items() if k != "holdout_data"})
            predictive_accuracy = self._test_predictive_accuracy(theory, holdout, context)
            tests.append({"test": "predictive_accuracy", "result": predictive_accuracy["accuracy"] >= self.confidence_threshold, "confidence": predictive_accuracy["confidence"], **predictive_accuracy})

        confidences = [clamp_confidence(test.get("confidence", 0.0)) for test in tests]
        validation_confidence = weighted_confidence(confidences) if confidences else 0.0
        failed = [test["test"] for test in tests if not test.get("result", False)]
        is_valid = validation_confidence >= self.confidence_threshold and not failed
        return {
            "is_valid": is_valid,
            "confidence": validation_confidence,
            "tests": tests,
            "failed_tests": failed,
            "overall_confidence": min(clamp_confidence(theory.get("confidence", 0.0)), validation_confidence),
        }

    def _check_knowledge_consistency(self, theory: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        knowledge_base = context.get("knowledge_base", {}) or {}
        theory_text = str(theory.get("theory", "")).lower()
        contradictions: List[str] = []
        support_score = 0.0
        if isinstance(knowledge_base, Mapping):
            for known in knowledge_base.get("contradictions", []) or []:
                if str(known).lower() in theory_text:
                    contradictions.append(str(known))
            for known in knowledge_base.get("supporting_theories", []) or []:
                if any(token in theory_text for token in self._tokenize(known)):
                    support_score += 0.15
        confidence = clamp_confidence(0.75 + support_score - len(contradictions) * 0.35)
        return {"consistent": len(contradictions) == 0, "confidence": confidence, "contradictions": contradictions, "support_score": support_score}

    def _test_predictive_accuracy(self, theory: Dict[str, Any], holdout_data: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        predictions = self._make_predictions(theory, holdout_data, context)
        if not predictions:
            return {"accuracy": 0.0, "confidence": 0.0, "correct_predictions": 0, "total_predictions": 0}
        correct = sum(1 for p in predictions if p.get("validated", True) and p.get("confidence", 0.0) >= self.confidence_threshold)
        accuracy = correct / len(predictions)
        return {"accuracy": accuracy, "confidence": clamp_confidence(accuracy * 1.1), "correct_predictions": correct, "total_predictions": len(predictions)}

    def _check_counterexamples(self, theory: Dict[str, Any], observations: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        counterexamples = [obs["id"] for obs in observations if not self._observation_conforms(theory, obs, context)]
        ratio = len(counterexamples) / max(1, len(observations))
        return {
            "found": ratio > self.counterexample_tolerance,
            "count": len(counterexamples),
            "ratio": ratio,
            "confidence": clamp_confidence(1.0 - ratio),
            "examples": counterexamples[:25],
        }

    def _observation_conforms(self, theory: Dict[str, Any], observation: Dict[str, Any], context: Dict[str, Any]) -> bool:
        text = str(observation.get("content", "")).lower()
        if "exception" in text or "counterexample" in text:
            return False
        for pattern in theory.get("supporting_patterns", []) or []:
            if pattern.get("type") == "dominant_value":
                attr = pattern.get("attribute")
                if attr in observation.get("attributes", {}):
                    expected = str(pattern.get("value"))
                    if str(observation["attributes"][attr]) != expected:
                        freq = float(pattern.get("metadata", {}).get("frequency", 1.0))
                        if freq >= 0.95:
                            return False
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _make_predictions(self, theory: Dict[str, Any], observations: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        for trend in self._identify_trends(observations):
            prediction = self._extrapolate_trend(trend, context)
            if prediction:
                predictions.append(prediction)
        predictions.extend(self._make_categorical_predictions(theory, observations, context))
        ordered = sorted(predictions, key=lambda p: p.get("confidence", 0.0), reverse=True)
        return ordered[: self.max_predictions]

    def _identify_trends(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trends: List[Dict[str, Any]] = []
        numeric_by_attr: Dict[str, List[float]] = defaultdict(list)
        for obs in observations:
            for attr, value in obs.get("attributes", {}).items():
                if self._is_number(value):
                    numeric_by_attr[str(attr)].append(float(value))
        for attr, values in numeric_by_attr.items():
            if len(values) < max(2, self.min_pattern_support):
                continue
            x = np.arange(len(values), dtype=float)
            y = np.asarray(values, dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            r2 = self._r_squared(y, slope * x + intercept)
            trends.append({"type": "linear", "attribute": attr, "slope": float(slope), "intercept": float(intercept), "confidence": clamp_confidence(0.5 + 0.5 * r2), "r_squared": r2, "observed_count": len(values)})
            if all(v > 0 for v in values):
                log_y = np.log(y)
                exp_slope, exp_intercept = np.polyfit(x, log_y, 1)
                pred = np.exp(exp_slope * x + exp_intercept)
                trends.append({"type": "exponential", "attribute": attr, "growth_rate": float(exp_slope), "intercept": float(exp_intercept), "confidence": clamp_confidence(0.45 + 0.45 * self._r_squared(y, pred)), "observed_count": len(values)})
        return trends

    def _extrapolate_trend(self, trend: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        horizon = float(context.get("extrapolation_factor", context.get("prediction_horizon", self.extrapolation_limit)))
        base_index = float(trend.get("observed_count", 1) - 1 + horizon)
        if trend["type"] == "linear":
            future_value = trend["slope"] * base_index + trend["intercept"]
        elif trend["type"] == "exponential":
            future_value = math.exp(trend["growth_rate"] * base_index + trend.get("intercept", 0.0))
        else:
            return None
        return {
            "type": "numerical",
            "attribute": trend.get("attribute"),
            "prediction": f"{trend.get('attribute', 'value')} will reach {future_value:.3f}",
            "value": future_value,
            "confidence": clamp_confidence(trend.get("confidence", 0.0) * 0.9),
            "extrapolation_factor": horizon,
            "validated": True,
            "basis": trend,
        }

    def _make_categorical_predictions(self, theory: Dict[str, Any], observations: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        for pattern in theory.get("supporting_patterns", []) or []:
            if pattern.get("type") == "dominant_value":
                predictions.append(
                    {
                        "type": "categorical",
                        "attribute": pattern.get("attribute"),
                        "prediction": f"Next observation will likely have {pattern.get('attribute')} = {pattern.get('value')}",
                        "value": pattern.get("value"),
                        "confidence": clamp_confidence(pattern.get("confidence", 0.0) * 0.85),
                        "validated": True,
                        "pattern": pattern,
                    }
                )
        return predictions

    # ------------------------------------------------------------------
    # Formatting and integration
    # ------------------------------------------------------------------
    def _format_results(self, theory: Dict[str, Any], observations: List[Dict[str, Any]], patterns: List[Dict[str, Any]], validation: Dict[str, Any], predictions: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        payload_context = context if self.include_context else {k: v for k, v in context.items() if k.startswith("_") is False and not callable(v)}
        return json_safe_reasoning_state(
            {
                "theory": theory,
                "validation": validation,
                "predictions": predictions,
                "supporting_data": {
                    "observations_used": observations,
                    "patterns_identified": patterns,
                    "context": payload_context,
                },
                "metrics": {
                    "observations_count": len(observations),
                    "patterns_count": len(patterns),
                    "theory_confidence": clamp_confidence(theory.get("confidence", 0.0)) if theory else 0.0,
                    "validation_confidence": clamp_confidence(validation.get("confidence", 0.0)) if validation else 0.0,
                    "predictions_count": len(predictions),
                    "scope": theory.get("scope", "none") if theory else "none",
                    "success": bool(validation.get("is_valid", False)) if validation else False,
                },
                "reasoning_type": "inductive",
            }
        )

    def _record_memory_event(self, result: Dict[str, Any], context: Dict[str, Any]) -> None:
        if not self.record_memory_events:
            return
        memory = context.get("memory") or context.get("reasoning_memory")
        if memory is None or not hasattr(memory, "add"):
            return
        memory.add(
            {
                "type": "reasoning_inductive_result",
                "theory": result.get("theory", {}).get("theory"),
                "metrics": result.get("metrics", {}),
            },
            priority=float(result.get("metrics", {}).get("theory_confidence", 0.5)),
            tag="reasoning_inductive",
        )


if __name__ == "__main__":
    print("\n=== Running Reasoning Inductive ===\n")
    printer.status("TEST", "Reasoning Inductive initialized", "info")

    inductive = ReasoningInductive()
    observations = [
        {"content": {"temp": 22, "weather": "sunny"}, "source": "s1"},
        {"content": {"temp": 24, "weather": "sunny"}, "source": "s1"},
        {"content": {"temp": 26, "weather": "sunny"}, "source": "s2"},
        {"content": {"temp": 28, "weather": "sunny"}, "source": "s2"},
        {"content": {"temp": 30, "weather": "sunny"}, "source": "s3"},
    ]
    context = {
        "knowledge_base": {"supporting_theories": ["temp increases"]},
        "extrapolation_factor": 2.0,
    }
    result = inductive.perform_reasoning(observations, context=context)
    assert result["reasoning_type"] == "inductive"
    assert result["metrics"]["observations_count"] == 5
    assert result["metrics"]["patterns_count"] >= 1
    assert result["predictions"]
    printer.pretty("Inductive Result", result)

    print("\n=== Test ran successfully ===\n")
