"""
Structured, uncertainty-aware agent alignment assessment for SLAI.

This module does not enforce safety policy, plan actions, reason about consequences,
learn rewards, or maintain generic agent memory.  Its responsibility is narrower:
combine alignment-specific evidence from existing SLAI components into a typed,
explainable assessment and track bounded trajectory-level alignment drift.

Scores use the convention ``0.0 = evidence of misalignment`` and
``1.0 = evidence of alignment``.  ``confidence`` is an evidence-quality indicator,
not a calibrated probability.
"""

from __future__ import annotations

import math

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import fmean
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


class AlignmentStatus(str, Enum):
    """Canonical qualitative alignment states."""

    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    CONFLICTING = "conflicting"
    UNCERTAIN = "uncertain"
    MISALIGNED = "misaligned"


class AlignmentDimension(str, Enum):
    """Alignment dimensions that can be supported by runtime evidence."""

    USER_INTENT = "user_intent"
    TASK_OBJECTIVE = "task_objective"
    CONSTRAINT = "constraint"
    ENVIRONMENT = "environment"
    PREFERENCE = "preference"
    TRAJECTORY = "trajectory"
    SELF_CONSTRAINT = "self_constraint"
    MULTI_AGENT = "multi_agent"


class EvidencePolarity(str, Enum):
    """Optional semantic label for evidence provenance."""

    SUPPORTS = "supports"
    OPPOSES = "opposes"
    NEUTRAL = "neutral"


_POSITIVE_STANCES = {"require", "required", "prefer", "preferred", "allow", "positive", "support"}
_NEGATIVE_STANCES = {"avoid", "prohibit", "prohibited", "deny", "negative", "oppose"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_probability(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {numeric}.")
    return numeric


def _optional_probability(value: Any, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    return _finite_probability(value, field_name=field_name)


def _coerce_dimension(value: Any) -> AlignmentDimension:
    if isinstance(value, AlignmentDimension):
        return value
    text = str(value).strip().lower()
    aliases = {
        "intent": AlignmentDimension.USER_INTENT,
        "human_intent": AlignmentDimension.USER_INTENT,
        "user_intention": AlignmentDimension.USER_INTENT,
        "task": AlignmentDimension.TASK_OBJECTIVE,
        "task_fidelity": AlignmentDimension.TASK_OBJECTIVE,
        "objective": AlignmentDimension.TASK_OBJECTIVE,
        "constraints": AlignmentDimension.CONSTRAINT,
        "policy": AlignmentDimension.CONSTRAINT,
        "environmental": AlignmentDimension.ENVIRONMENT,
        "environment_consistency": AlignmentDimension.ENVIRONMENT,
        "values": AlignmentDimension.PREFERENCE,
        "value": AlignmentDimension.PREFERENCE,
        "preference_alignment": AlignmentDimension.PREFERENCE,
        "behavior": AlignmentDimension.TRAJECTORY,
        "behavioral": AlignmentDimension.TRAJECTORY,
        "resource": AlignmentDimension.SELF_CONSTRAINT,
        "resource_constraint": AlignmentDimension.SELF_CONSTRAINT,
        "self": AlignmentDimension.SELF_CONSTRAINT,
        "peer": AlignmentDimension.MULTI_AGENT,
        "multiagent": AlignmentDimension.MULTI_AGENT,
    }
    if text in aliases:
        return aliases[text]
    try:
        return AlignmentDimension(text)
    except ValueError as exc:
        supported = ", ".join(item.value for item in AlignmentDimension)
        raise ValueError(f"Unsupported alignment dimension {value!r}; expected one of: {supported}.") from exc


@dataclass(frozen=True)
class AlignmentEvidence:
    """One provenance-preserving alignment signal."""

    dimension: AlignmentDimension
    source: str
    score: Optional[float]
    confidence: float
    kind: str = "observation"
    required: bool = False
    timestamp: str = field(default_factory=_utc_now_iso)
    polarity: EvidencePolarity = EvidencePolarity.NEUTRAL
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise ValueError("AlignmentEvidence.source must be non-empty.")
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(
            self,
            "score",
            _optional_probability(self.score, field_name="AlignmentEvidence.score"),
        )
        object.__setattr__(
            self,
            "confidence",
            _finite_probability(self.confidence, field_name="AlignmentEvidence.confidence"),
        )
        object.__setattr__(self, "kind", str(self.kind or "observation").strip().lower())
        object.__setattr__(self, "details", dict(self.details or {}))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        default_confidence: float,
        default_source: str = "external",
    ) -> "AlignmentEvidence":
        dimension = _coerce_dimension(payload.get("dimension"))
        raw_score = payload.get("score")
        if raw_score is None and "aligned" in payload:
            raw_score = 1.0 if bool(payload.get("aligned")) else 0.0

        polarity_raw = str(payload.get("polarity", "neutral")).strip().lower()
        try:
            polarity = EvidencePolarity(polarity_raw)
        except ValueError:
            polarity = EvidencePolarity.NEUTRAL

        return cls(
            dimension=dimension,
            source=str(payload.get("source") or default_source),
            score=_optional_probability(raw_score, field_name="alignment_evidence.score"),
            confidence=_finite_probability(
                payload.get("confidence", default_confidence),
                field_name="alignment_evidence.confidence",
            ),
            kind=str(payload.get("kind", "observation")),
            required=bool(payload.get("required", False)),
            timestamp=str(payload.get("timestamp") or _utc_now_iso()),
            polarity=polarity,
            details=dict(payload.get("details") or payload.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "source": self.source,
            "score": self.score,
            "confidence": self.confidence,
            "kind": self.kind,
            "required": self.required,
            "timestamp": self.timestamp,
            "polarity": self.polarity.value,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ConflictRecord:
    """Detected conflict between alignment evidence or explicit objectives."""

    conflict_type: str
    dimension: Optional[AlignmentDimension]
    sources: Tuple[str, ...]
    description: str
    severity: str = "review"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_type": self.conflict_type,
            "dimension": self.dimension.value if self.dimension else None,
            "sources": list(self.sources),
            "description": self.description,
            "severity": self.severity,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DimensionAssessment:
    """Summary for one alignment dimension."""

    dimension: AlignmentDimension
    status: AlignmentStatus
    score: Optional[float]
    confidence: float
    evidence_count: int
    required: bool
    sources: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "status": self.status.value,
            "score": self.score,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "required": self.required,
            "sources": list(self.sources),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DriftAssessment:
    """Descriptive trajectory shift; this is not a causal estimate."""

    detected: bool
    magnitude: float
    direction: str
    affected_dimensions: Tuple[str, ...] = ()
    dimension_shifts: Dict[str, float] = field(default_factory=dict)
    historical_samples: int = 0
    recent_samples: int = 0
    threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "magnitude": self.magnitude,
            "direction": self.direction,
            "affected_dimensions": list(self.affected_dimensions),
            "dimension_shifts": dict(self.dimension_shifts),
            "historical_samples": self.historical_samples,
            "recent_samples": self.recent_samples,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class AlignmentAssessment:
    """Structured agent-level alignment assessment."""

    status: AlignmentStatus
    dimensions: Dict[str, DimensionAssessment]
    conflicts: Tuple[ConflictRecord, ...]
    drift: DriftAssessment
    coverage: float
    uncertainty: float
    required_missing: Tuple[str, ...]
    evidence_count: int
    explanation: Tuple[str, ...]
    aggregate_score: Optional[float] = None
    created_at: str = field(default_factory=_utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def requires_review(self) -> bool:
        return self.status in {
            AlignmentStatus.CONFLICTING,
            AlignmentStatus.UNCERTAIN,
            AlignmentStatus.MISALIGNED,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "dimensions": {
                key: value.to_dict()
                for key, value in self.dimensions.items()
            },
            "conflicts": [item.to_dict() for item in self.conflicts],
            "drift": self.drift.to_dict(),
            "coverage": self.coverage,
            "uncertainty": self.uncertainty,
            "required_missing": list(self.required_missing),
            "evidence_count": self.evidence_count,
            "explanation": list(self.explanation),
            "aggregate_score": self.aggregate_score,
            "requires_review": self.requires_review,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class AlignmentTrajectoryTracker:
    """
    Bounded trajectory state for alignment drift detection.

    The tracker stores only compact dimension summaries.  It is not a replacement
    for AlignmentMemory; persistent/audit storage remains the memory subsystem's
    responsibility.
    """

    def __init__(
        self,
        *,
        history_limit: int,
        window: int,
        drift_threshold: float,
    ) -> None:
        if history_limit < 4:
            raise ValueError("history_limit must be >= 4.")
        if window < 2:
            raise ValueError("window must be >= 2.")
        if history_limit < 2 * window:
            raise ValueError("history_limit must be at least 2 * window.")
        self.history_limit = int(history_limit)
        self.window = int(window)
        self.drift_threshold = _finite_probability(
            drift_threshold,
            field_name="drift_threshold",
        )
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)

    def observe(
        self,
        dimensions: Mapping[str, DimensionAssessment],
        *,
        assessment_id: Optional[str] = None,
    ) -> None:
        scores = {
            name: summary.score
            for name, summary in dimensions.items()
            if summary.score is not None
        }
        self._history.append(
            {
                "timestamp": _utc_now_iso(),
                "assessment_id": assessment_id,
                "scores": scores,
            }
        )

    def detect(self) -> DriftAssessment:
        if len(self._history) < 2 * self.window:
            return DriftAssessment(
                detected=False,
                magnitude=0.0,
                direction="insufficient_history",
                historical_samples=min(len(self._history), self.window),
                recent_samples=max(0, len(self._history) - self.window),
                threshold=self.drift_threshold,
            )

        entries = list(self._history)
        historical = entries[-2 * self.window : -self.window]
        recent = entries[-self.window :]

        all_dimensions = sorted(
            {
                dimension
                for entry in historical + recent
                for dimension in entry["scores"]
            }
        )
        shifts: Dict[str, float] = {}
        for dimension in all_dimensions:
            old_values = [
                float(entry["scores"][dimension])
                for entry in historical
                if dimension in entry["scores"]
            ]
            new_values = [
                float(entry["scores"][dimension])
                for entry in recent
                if dimension in entry["scores"]
            ]
            if len(old_values) < max(2, self.window // 2):
                continue
            if len(new_values) < max(2, self.window // 2):
                continue
            shifts[dimension] = float(fmean(new_values) - fmean(old_values))

        if not shifts:
            return DriftAssessment(
                detected=False,
                magnitude=0.0,
                direction="insufficient_comparable_evidence",
                historical_samples=len(historical),
                recent_samples=len(recent),
                threshold=self.drift_threshold,
            )

        magnitude = max(abs(value) for value in shifts.values())
        affected = tuple(
            name
            for name, value in sorted(shifts.items())
            if abs(value) >= self.drift_threshold
        )
        detected = bool(affected)

        if not detected:
            direction = "stable"
        else:
            affected_values = [shifts[name] for name in affected]
            if all(value < 0 for value in affected_values):
                direction = "negative"
            elif all(value > 0 for value in affected_values):
                direction = "positive"
            else:
                direction = "mixed"

        return DriftAssessment(
            detected=detected,
            magnitude=float(min(1.0, magnitude)),
            direction=direction,
            affected_dimensions=affected,
            dimension_shifts=shifts,
            historical_samples=len(historical),
            recent_samples=len(recent),
            threshold=self.drift_threshold,
        )

    def export_state(self) -> Dict[str, Any]:
        return {
            "history_limit": self.history_limit,
            "window": self.window,
            "drift_threshold": self.drift_threshold,
            "history": list(self._history),
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        raw_history = state.get("history", [])
        if not isinstance(raw_history, Sequence) or isinstance(raw_history, (str, bytes, bytearray)):
            raise ValueError("trajectory state 'history' must be a sequence.")
        self._history.clear()
        for item in raw_history[-self.history_limit :]:
            if not isinstance(item, Mapping):
                continue
            scores = item.get("scores", {})
            if not isinstance(scores, Mapping):
                continue
            normalized_scores: Dict[str, float] = {}
            for key, value in scores.items():
                try:
                    normalized_scores[str(key)] = _finite_probability(
                        value,
                        field_name=f"trajectory.score[{key}]",
                    )
                except ValueError:
                    continue
            self._history.append(
                {
                    "timestamp": str(item.get("timestamp") or _utc_now_iso()),
                    "assessment_id": item.get("assessment_id"),
                    "scores": normalized_scores,
                }
            )


class AlignmentAssessor:
    """
    Synthesize explicit alignment evidence without duplicating Reasoning,
    Safety, Evaluation, Planning, Learning, or generic memory.

    The assessor never attempts semantic task reasoning on its own.  It consumes
    evidence supplied by those owners and adds alignment-specific conflict,
    uncertainty, and trajectory analysis.
    """

    _SCORE_ALIASES: Tuple[Tuple[str, AlignmentDimension], ...] = (
        ("intent_alignment_score", AlignmentDimension.USER_INTENT),
        ("task_fidelity_score", AlignmentDimension.TASK_OBJECTIVE),
        ("environment_alignment_score", AlignmentDimension.ENVIRONMENT),
        ("environment_consistency_score", AlignmentDimension.ENVIRONMENT),
        ("consequence_alignment_score", AlignmentDimension.ENVIRONMENT),
        ("preference_alignment_score", AlignmentDimension.PREFERENCE),
        ("feedback_alignment_score", AlignmentDimension.TRAJECTORY),
        ("trajectory_alignment_score", AlignmentDimension.TRAJECTORY),
        ("self_constraint_alignment_score", AlignmentDimension.SELF_CONSTRAINT),
        ("resource_alignment_score", AlignmentDimension.SELF_CONSTRAINT),
        ("peer_alignment_score", AlignmentDimension.MULTI_AGENT),
        ("multi_agent_alignment_score", AlignmentDimension.MULTI_AGENT),
    )

    _BOOLEAN_ALIASES: Tuple[Tuple[str, AlignmentDimension], ...] = (
        ("intent_satisfied", AlignmentDimension.USER_INTENT),
        ("task_objective_satisfied", AlignmentDimension.TASK_OBJECTIVE),
        ("environment_consistent", AlignmentDimension.ENVIRONMENT),
        ("consequence_consistent", AlignmentDimension.ENVIRONMENT),
        ("feedback_supports_objective", AlignmentDimension.TRAJECTORY),
        ("self_constraints_satisfied", AlignmentDimension.SELF_CONSTRAINT),
        ("resource_constraints_satisfied", AlignmentDimension.SELF_CONSTRAINT),
        ("multi_agent_consistent", AlignmentDimension.MULTI_AGENT),
    )

    def __init__(self, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("AlignmentAssessor config must be a mapping.")

        self.aligned_threshold = _finite_probability(
            config.get("aligned_threshold", 0.80),
            field_name="aligned_threshold",
        )
        self.misaligned_threshold = _finite_probability(
            config.get("misaligned_threshold", 0.30),
            field_name="misaligned_threshold",
        )
        if self.misaligned_threshold >= self.aligned_threshold:
            raise ValueError("misaligned_threshold must be lower than aligned_threshold.")

        self.min_evidence_confidence = _finite_probability(
            config.get("min_evidence_confidence", 0.50),
            field_name="min_evidence_confidence",
        )
        self.conflict_score_gap = _finite_probability(
            config.get("conflict_score_gap", 0.50),
            field_name="conflict_score_gap",
        )
        self.implicit_score_confidence = _finite_probability(
            config.get("implicit_score_confidence", 0.70),
            field_name="implicit_score_confidence",
        )
        self.max_evidence_per_dimension = max(
            1,
            int(config.get("max_evidence_per_dimension", 64)),
        )

        raw_required = config.get("required_dimensions", ["constraint"])
        if not isinstance(raw_required, Sequence) or isinstance(raw_required, (str, bytes, bytearray)):
            raise ValueError("required_dimensions must be a sequence.")
        self.required_dimensions = tuple(
            dict.fromkeys(_coerce_dimension(item) for item in raw_required)
        )

        self.tracker = AlignmentTrajectoryTracker(
            history_limit=max(4, int(config.get("trajectory_history_limit", 128))),
            window=max(2, int(config.get("trajectory_window", 8))),
            drift_threshold=_finite_probability(
                config.get("drift_threshold", 0.20),
                field_name="drift_threshold",
            ),
        )

    def assess(
        self,
        *,
        task_context: Mapping[str, Any],
        component_report: Mapping[str, Any],
        assessment_id: Optional[str] = None,
        record_trajectory: bool = True,
    ) -> AlignmentAssessment:
        context = dict(task_context or {})
        report = dict(component_report or {})

        evidence = self.collect_evidence(
            task_context=context,
            component_report=report,
        )
        conflicts = self.detect_conflicts(
            task_context=context,
            evidence=evidence,
        )
        conflicts_by_dimension: Dict[AlignmentDimension, List[ConflictRecord]] = defaultdict(list)
        for conflict in conflicts:
            if conflict.dimension is not None:
                conflicts_by_dimension[conflict.dimension].append(conflict)

        grouped: Dict[AlignmentDimension, List[AlignmentEvidence]] = defaultdict(list)
        for item in evidence:
            bucket = grouped[item.dimension]
            if len(bucket) < self.max_evidence_per_dimension:
                bucket.append(item)

        all_dimensions = tuple(AlignmentDimension)
        dimensions: Dict[str, DimensionAssessment] = {}
        required_missing: List[str] = []

        for dimension in all_dimensions:
            required = dimension in self.required_dimensions or any(
                item.required for item in grouped.get(dimension, [])
            )
            summary = self._summarize_dimension(
                dimension=dimension,
                evidence=grouped.get(dimension, []),
                conflicts=conflicts_by_dimension.get(dimension, []),
                required=required,
            )
            dimensions[dimension.value] = summary
            if required and summary.score is None:
                required_missing.append(dimension.value)

        prior_drift = self.tracker.detect()
        if record_trajectory:
            self.tracker.observe(dimensions, assessment_id=assessment_id)
            drift = self.tracker.detect()
        else:
            drift = prior_drift

        coverage, uncertainty = self._coverage_and_uncertainty(dimensions)
        status, explanation = self._overall_status(
            dimensions=dimensions,
            conflicts=conflicts,
            required_missing=required_missing,
            drift=drift,
        )

        return AlignmentAssessment(
            status=status,
            dimensions=dimensions,
            conflicts=tuple(conflicts),
            drift=drift,
            coverage=coverage,
            uncertainty=uncertainty,
            required_missing=tuple(required_missing),
            evidence_count=len(evidence),
            explanation=tuple(explanation),
            aggregate_score=None,
            metadata={
                "assessment_id": assessment_id,
                "required_dimensions": [item.value for item in self.required_dimensions],
                "aggregate_score_policy": "not_emitted_without_explicit_external_aggregation",
            },
        )

    def collect_evidence(
        self,
        *,
        task_context: Mapping[str, Any],
        component_report: Mapping[str, Any],
    ) -> List[AlignmentEvidence]:
        evidence: List[AlignmentEvidence] = []

        raw_evidence = task_context.get("alignment_evidence", [])
        if isinstance(raw_evidence, Mapping):
            raw_evidence = [raw_evidence]
        if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, (str, bytes, bytearray)):
            for payload in raw_evidence:
                if not isinstance(payload, Mapping):
                    continue
                try:
                    evidence.append(
                        AlignmentEvidence.from_mapping(
                            payload,
                            default_confidence=self.implicit_score_confidence,
                        )
                    )
                except (TypeError, ValueError):
                    continue

        for key, dimension in self._BOOLEAN_ALIASES:
            if key not in task_context:
                continue
            value = task_context.get(key)
            if value is None:
                continue
            evidence.append(
                AlignmentEvidence(
                    dimension=dimension,
                    source=f"task_context:{key}",
                    score=1.0 if bool(value) else 0.0,
                    confidence=1.0,
                    kind="explicit_boolean",
                    required=False,
                    details={"field": key},
                )
            )

        for key, dimension in self._SCORE_ALIASES:
            if key not in task_context:
                continue
            value = task_context.get(key)
            try:
                score = _finite_probability(value, field_name=key)
            except ValueError:
                continue
            confidence_key = f"{key}_confidence"
            raw_confidence = task_context.get(
                confidence_key,
                self.implicit_score_confidence,
            )
            try:
                confidence = _finite_probability(
                    raw_confidence,
                    field_name=confidence_key,
                )
            except ValueError:
                confidence = self.implicit_score_confidence
            evidence.append(
                AlignmentEvidence(
                    dimension=dimension,
                    source=f"task_context:{key}",
                    score=score,
                    confidence=confidence,
                    kind="upstream_score",
                    details={"field": key},
                )
            )

        self._append_constraint_evidence(evidence, component_report)
        self._append_value_evidence(evidence, component_report)
        self._append_external_assessment_evidence(
            evidence,
            task_context.get("safety_assessment"),
            source="safety_agent",
        )
        self._append_external_assessment_evidence(
            evidence,
            task_context.get("privacy_assessment"),
            source="privacy_agent",
        )
        self._append_external_assessment_evidence(
            evidence,
            task_context.get("evaluation_assessment"),
            source="evaluation_agent",
        )

        return evidence

    def detect_conflicts(
        self,
        *,
        task_context: Mapping[str, Any],
        evidence: Sequence[AlignmentEvidence],
    ) -> List[ConflictRecord]:
        conflicts: List[ConflictRecord] = []

        grouped: Dict[AlignmentDimension, List[AlignmentEvidence]] = defaultdict(list)
        for item in evidence:
            if item.score is not None and item.confidence >= self.min_evidence_confidence:
                grouped[item.dimension].append(item)

        for dimension, items in grouped.items():
            if len(items) < 2:
                continue
            min_item = min(items, key=lambda item: float(item.score or 0.0))
            max_item = max(items, key=lambda item: float(item.score or 0.0))
            if (
                min_item.score is not None
                and max_item.score is not None
                and (max_item.score - min_item.score) >= self.conflict_score_gap
            ):
                conflicts.append(
                    ConflictRecord(
                        conflict_type="evidence_disagreement",
                        dimension=dimension,
                        sources=(min_item.source, max_item.source),
                        description=(
                            f"High-confidence evidence for {dimension.value} disagrees by "
                            f"{max_item.score - min_item.score:.3f}."
                        ),
                        details={
                            "minimum_score": min_item.score,
                            "maximum_score": max_item.score,
                            "minimum_confidence": min_item.confidence,
                            "maximum_confidence": max_item.confidence,
                        },
                    )
                )

        objectives = task_context.get("alignment_objectives", [])
        if isinstance(objectives, Mapping):
            objectives = [objectives]
        if isinstance(objectives, Sequence) and not isinstance(objectives, (str, bytes, bytearray)):
            by_target: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
            for item in objectives:
                if not isinstance(item, Mapping):
                    continue
                target = str(
                    item.get("target")
                    or item.get("id")
                    or item.get("objective")
                    or ""
                ).strip()
                if target:
                    by_target[target].append(item)

            for target, items in by_target.items():
                positive = []
                negative = []
                for item in items:
                    stance = str(
                        item.get("stance")
                        or item.get("polarity")
                        or item.get("mode")
                        or ""
                    ).strip().lower()
                    if stance in _POSITIVE_STANCES:
                        positive.append(item)
                    elif stance in _NEGATIVE_STANCES:
                        negative.append(item)
                if positive and negative:
                    sources = tuple(
                        str(item.get("source") or "alignment_objective")
                        for item in positive + negative
                    )
                    conflicts.append(
                        ConflictRecord(
                            conflict_type="objective_polarity_conflict",
                            dimension=AlignmentDimension.TASK_OBJECTIVE,
                            sources=sources,
                            description=(
                                f"Objective target {target!r} is simultaneously supported "
                                "and opposed by explicit objective records."
                            ),
                            severity="review",
                            details={"target": target},
                        )
                    )

        return conflicts

    def _summarize_dimension(
        self,
        *,
        dimension: AlignmentDimension,
        evidence: Sequence[AlignmentEvidence],
        conflicts: Sequence[ConflictRecord],
        required: bool,
    ) -> DimensionAssessment:
        usable = [
            item
            for item in evidence
            if item.score is not None
        ]

        if conflicts:
            score = self._weighted_score(usable)
            confidence = self._mean_confidence(usable)
            return DimensionAssessment(
                dimension=dimension,
                status=AlignmentStatus.CONFLICTING,
                score=score,
                confidence=confidence,
                evidence_count=len(usable),
                required=required,
                sources=tuple(dict.fromkeys(item.source for item in usable)),
                reasons=tuple(item.description for item in conflicts),
            )

        if not usable:
            return DimensionAssessment(
                dimension=dimension,
                status=AlignmentStatus.UNCERTAIN,
                score=None,
                confidence=0.0,
                evidence_count=0,
                required=required,
                sources=(),
                reasons=("No usable alignment evidence was supplied.",),
            )

        score = self._weighted_score(usable)
        confidence = self._mean_confidence(usable)
        assert score is not None

        if confidence < self.min_evidence_confidence:
            status = AlignmentStatus.UNCERTAIN
            reasons = (
                f"Evidence confidence {confidence:.3f} is below the configured minimum "
                f"{self.min_evidence_confidence:.3f}.",
            )
        elif score <= self.misaligned_threshold:
            status = AlignmentStatus.MISALIGNED
            reasons = (
                f"Alignment score {score:.3f} is at or below the configured "
                f"misalignment threshold {self.misaligned_threshold:.3f}.",
            )
        elif score >= self.aligned_threshold:
            status = AlignmentStatus.ALIGNED
            reasons = (
                f"Alignment score {score:.3f} meets the configured alignment threshold "
                f"{self.aligned_threshold:.3f}.",
            )
        else:
            status = AlignmentStatus.PARTIALLY_ALIGNED
            reasons = (
                f"Alignment score {score:.3f} lies between configured misalignment and "
                "alignment thresholds.",
            )

        return DimensionAssessment(
            dimension=dimension,
            status=status,
            score=score,
            confidence=confidence,
            evidence_count=len(usable),
            required=required,
            sources=tuple(dict.fromkeys(item.source for item in usable)),
            reasons=reasons,
        )

    def _overall_status(
        self,
        *,
        dimensions: Mapping[str, DimensionAssessment],
        conflicts: Sequence[ConflictRecord],
        required_missing: Sequence[str],
        drift: DriftAssessment,
    ) -> Tuple[AlignmentStatus, List[str]]:
        required = [
            summary
            for summary in dimensions.values()
            if summary.required
        ]
        explanations: List[str] = []

        required_conflicts = [
            summary
            for summary in required
            if summary.status == AlignmentStatus.CONFLICTING
        ]
        if required_conflicts:
            explanations.append(
                "One or more required alignment dimensions contain conflicting evidence."
            )
            return AlignmentStatus.CONFLICTING, explanations

        if required_missing:
            explanations.append(
                "Required alignment evidence is missing for: "
                + ", ".join(sorted(required_missing))
                + "."
            )
            return AlignmentStatus.UNCERTAIN, explanations

        required_uncertain = [
            summary
            for summary in required
            if summary.status == AlignmentStatus.UNCERTAIN
        ]
        if required_uncertain:
            explanations.append(
                "At least one required alignment dimension has insufficient evidence confidence."
            )
            return AlignmentStatus.UNCERTAIN, explanations

        required_misaligned = [
            summary
            for summary in required
            if summary.status == AlignmentStatus.MISALIGNED
        ]
        if required_misaligned:
            explanations.append(
                "At least one required alignment dimension contains direct evidence of misalignment."
            )
            return AlignmentStatus.MISALIGNED, explanations

        if conflicts:
            explanations.append(
                "Material alignment evidence or objective conflict is present."
            )
            return AlignmentStatus.CONFLICTING, explanations

        if any(
            summary.status == AlignmentStatus.MISALIGNED
            for summary in dimensions.values()
            if not summary.required and summary.evidence_count > 0
        ):
            explanations.append(
                "A non-required alignment dimension contains evidence of misalignment."
            )
            return AlignmentStatus.PARTIALLY_ALIGNED, explanations

        if any(
            summary.status == AlignmentStatus.PARTIALLY_ALIGNED
            for summary in required
        ):
            explanations.append(
                "At least one required alignment dimension is only partially aligned."
            )
            return AlignmentStatus.PARTIALLY_ALIGNED, explanations

        if drift.detected and drift.direction in {"negative", "mixed"}:
            explanations.append(
                "Trajectory evidence shows a material negative or mixed alignment shift."
            )
            return AlignmentStatus.PARTIALLY_ALIGNED, explanations

        if required and all(
            summary.status == AlignmentStatus.ALIGNED
            for summary in required
        ):
            explanations.append(
                "All required dimensions with available evidence are aligned."
            )
            return AlignmentStatus.ALIGNED, explanations

        explanations.append(
            "No required misalignment was established, but the assessment does not contain "
            "enough required evidence to assert full alignment."
        )
        return AlignmentStatus.UNCERTAIN, explanations

    def _coverage_and_uncertainty(
        self,
        dimensions: Mapping[str, DimensionAssessment],
    ) -> Tuple[float, float]:
        required = [
            summary
            for summary in dimensions.values()
            if summary.required
        ]
        if not required:
            observed = [
                summary for summary in dimensions.values()
                if summary.evidence_count > 0
            ]
            if not observed:
                return 0.0, 1.0
            coverage = len(observed) / max(1, len(dimensions))
            confidence = fmean(summary.confidence for summary in observed)
            return float(coverage), float(max(1.0 - coverage, 1.0 - confidence))

        observed_required = [
            summary for summary in required
            if summary.evidence_count > 0
        ]
        coverage = len(observed_required) / len(required)
        confidence = (
            fmean(summary.confidence for summary in observed_required)
            if observed_required
            else 0.0
        )
        uncertainty = max(1.0 - coverage, 1.0 - confidence)
        return float(coverage), float(min(1.0, max(0.0, uncertainty)))

    @staticmethod
    def _weighted_score(
        evidence: Sequence[AlignmentEvidence],
    ) -> Optional[float]:
        weighted_sum = 0.0
        total_weight = 0.0
        for item in evidence:
            if item.score is None:
                continue
            weight = max(0.0, float(item.confidence))
            if weight <= 0.0:
                continue
            weighted_sum += float(item.score) * weight
            total_weight += weight
        if total_weight <= 0.0:
            return None
        return float(min(1.0, max(0.0, weighted_sum / total_weight)))

    @staticmethod
    def _mean_confidence(
        evidence: Sequence[AlignmentEvidence],
    ) -> float:
        confidences = [
            float(item.confidence)
            for item in evidence
            if item.score is not None
        ]
        return float(fmean(confidences)) if confidences else 0.0

    def _append_constraint_evidence(
        self,
        evidence: List[AlignmentEvidence],
        component_report: Mapping[str, Any],
    ) -> None:
        report = component_report.get("ethical_compliance_report")
        if not isinstance(report, Mapping):
            report = component_report.get("constraint_report")
        if not isinstance(report, Mapping) or not report:
            return

        summary = report.get("summary", {})
        evidence_sufficient = bool(
            report.get("available", False)
            or (
                isinstance(summary, Mapping)
                and summary.get("evidence_sufficient", False)
            )
        )
        approved = report.get("approved")
        if evidence_sufficient and isinstance(approved, bool):
            violations = report.get("violations", [])
            evidence.append(
                AlignmentEvidence(
                    dimension=AlignmentDimension.CONSTRAINT,
                    source="alignment_runtime_constraints",
                    score=1.0 if approved else 0.0,
                    confidence=1.0,
                    kind="alignment_runtime_constraint_compliance",
                    required=True,
                    details={
                        "violation_count": len(violations) if isinstance(violations, Sequence) else None,
                        "scope": report.get("scope"),
                    },
                )
            )

    def _append_value_evidence(
        self,
        evidence: List[AlignmentEvidence],
        component_report: Mapping[str, Any],
    ) -> None:
        payload = component_report.get("value_alignment")
        if isinstance(payload, Mapping):
            if not bool(payload.get("available", False)):
                return
            score = payload.get("score")
            confidence = payload.get("confidence", self.implicit_score_confidence)
            try:
                score_value = _finite_probability(score, field_name="value_alignment.score")
                confidence_value = _finite_probability(
                    confidence,
                    field_name="value_alignment.confidence",
                )
            except ValueError:
                return
            evidence.append(
                AlignmentEvidence(
                    dimension=AlignmentDimension.PREFERENCE,
                    source=str(payload.get("source") or "value_embedding_model"),
                    score=score_value,
                    confidence=confidence_value,
                    kind="preference_model",
                    details={
                        "model_ready": bool(payload.get("model_ready", False)),
                        "provenance": payload.get("provenance"),
                    },
                )
            )
            return

        legacy_score = component_report.get("value_alignment_score")
        if legacy_score is None:
            return
        try:
            score_value = _finite_probability(
                legacy_score,
                field_name="value_alignment_score",
            )
        except ValueError:
            return
        evidence.append(
            AlignmentEvidence(
                dimension=AlignmentDimension.PREFERENCE,
                source="legacy_value_alignment_score",
                score=score_value,
                confidence=self.min_evidence_confidence,
                kind="legacy_unverified_preference_model",
                details={"warning": "Legacy value score has no explicit model-readiness provenance."},
            )
        )

    def _append_external_assessment_evidence(
        self,
        evidence: List[AlignmentEvidence],
        payload: Any,
        *,
        source: str,
    ) -> None:
        if not isinstance(payload, Mapping):
            return

        if source == "safety_agent":
            dimension = AlignmentDimension.CONSTRAINT
        elif source == "privacy_agent":
            dimension = AlignmentDimension.CONSTRAINT
        else:
            raw_dimension = payload.get("alignment_dimension", AlignmentDimension.TASK_OBJECTIVE.value)
            try:
                dimension = _coerce_dimension(raw_dimension)
            except ValueError:
                dimension = AlignmentDimension.TASK_OBJECTIVE

        raw_score = payload.get("alignment_score")
        if raw_score is None:
            compliant = payload.get("compliant")
            if compliant is None:
                decision = str(payload.get("decision", "")).strip().lower()
                if decision in {"allow", "approved", "pass", "safe", "compliant"}:
                    compliant = True
                elif decision in {"block", "deny", "rejected", "unsafe", "non_compliant"}:
                    compliant = False
            if isinstance(compliant, bool):
                raw_score = 1.0 if compliant else 0.0

        if raw_score is None:
            return

        try:
            score = _finite_probability(raw_score, field_name=f"{source}.alignment_score")
            confidence = _finite_probability(
                payload.get("confidence", self.implicit_score_confidence),
                field_name=f"{source}.confidence",
            )
        except ValueError:
            return

        evidence.append(
            AlignmentEvidence(
                dimension=dimension,
                source=source,
                score=score,
                confidence=confidence,
                kind="delegated_assessment",
                required=dimension == AlignmentDimension.CONSTRAINT,
                details={
                    "assessment_id": payload.get("assessment_id"),
                    "decision": payload.get("decision"),
                },
            )
        )


__all__ = [
    "AlignmentStatus",
    "AlignmentDimension",
    "EvidencePolarity",
    "AlignmentEvidence",
    "ConflictRecord",
    "DimensionAssessment",
    "DriftAssessment",
    "AlignmentAssessment",
    "AlignmentTrajectoryTracker",
    "AlignmentAssessor",
]
