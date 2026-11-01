"""Structured runtime alignment assessment for SLAI.

This module owns only alignment-specific evidence aggregation, conflict handling,
uncertainty representation, and bounded trajectory drift detection. It does not
plan actions, enforce safety policy, perform generic evaluation, or learn a policy.

Score convention: 0.0 indicates evidence of misalignment and 1.0 indicates
alignment. ``confidence`` is evidence quality, not a calibrated probability.
"""
from __future__ import annotations

import math

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import fmean
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast


class AlignmentStatus(str, Enum):
    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    CONFLICTING = "conflicting"
    UNCERTAIN = "uncertain"
    MISALIGNED = "misaligned"


class AlignmentDimension(str, Enum):
    USER_INTENT = "user_intent"
    TASK_OBJECTIVE = "task_objective"
    CONSTRAINT = "constraint"
    ENVIRONMENT = "environment"
    PREFERENCE = "preference"
    TRAJECTORY = "trajectory"
    SELF_CONSTRAINT = "self_constraint"
    MULTI_AGENT = "multi_agent"


class EvidencePolarity(str, Enum):
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    NEUTRAL = "neutral"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _probability(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return number


def _optional_probability(value: Any, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    return _probability(value, field_name=field_name)


def _dimension(value: Any) -> AlignmentDimension:
    if isinstance(value, AlignmentDimension):
        return value
    text = str(value or "").strip().lower()
    aliases = {
        "intent": AlignmentDimension.USER_INTENT,
        "human_intent": AlignmentDimension.USER_INTENT,
        "user_intention": AlignmentDimension.USER_INTENT,
        "task": AlignmentDimension.TASK_OBJECTIVE,
        "objective": AlignmentDimension.TASK_OBJECTIVE,
        "task_fidelity": AlignmentDimension.TASK_OBJECTIVE,
        "constraints": AlignmentDimension.CONSTRAINT,
        "policy": AlignmentDimension.CONSTRAINT,
        "environmental": AlignmentDimension.ENVIRONMENT,
        "environment_consistency": AlignmentDimension.ENVIRONMENT,
        "value": AlignmentDimension.PREFERENCE,
        "values": AlignmentDimension.PREFERENCE,
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
        raise ValueError(f"unsupported alignment dimension {value!r}; expected {supported}") from exc


@dataclass(frozen=True)
class AlignmentEvidence:
    dimension: AlignmentDimension
    source: str
    score: Optional[float]
    confidence: float = 1.0
    kind: str = "observation"
    required: bool = False
    timestamp: str = field(default_factory=_utc_now_iso)
    polarity: EvidencePolarity = EvidencePolarity.NEUTRAL
    details: Dict[str, Any] = field(default_factory=dict)
    evidence_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", _dimension(self.dimension))
        source = str(self.source or "").strip()
        if not source:
            raise ValueError("AlignmentEvidence.source must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "score", _optional_probability(self.score, field_name="score"))
        object.__setattr__(self, "confidence", _probability(self.confidence, field_name="confidence"))
        object.__setattr__(self, "kind", str(self.kind or "observation").strip().lower())
        object.__setattr__(self, "details", dict(self.details or {}))
        if self.evidence_id is not None:
            object.__setattr__(self, "evidence_id", str(self.evidence_id).strip() or None)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        default_source: str = "external",
        default_confidence: float = 0.5,
    ) -> "AlignmentEvidence":
        raw_score = payload.get("score")
        if raw_score is None and "aligned" in payload:
            raw_score = 1.0 if payload.get("aligned") is True else 0.0 if payload.get("aligned") is False else None
        polarity_text = str(payload.get("polarity", "neutral")).strip().lower()
        try:
            polarity = EvidencePolarity(polarity_text)
        except ValueError:
            polarity = EvidencePolarity.NEUTRAL
        return cls(
            dimension=_dimension(payload.get("dimension")),
            source=str(payload.get("source") or default_source),
            score=_optional_probability(raw_score, field_name="score"),
            confidence=_probability(payload.get("confidence", default_confidence), field_name="confidence"),
            kind=str(payload.get("kind", "observation")),
            required=bool(payload.get("required", False)),
            timestamp=str(payload.get("timestamp") or _utc_now_iso()),
            polarity=polarity,
            details=dict(payload.get("details") or payload.get("metadata") or {}),
            evidence_id=payload.get("evidence_id") or payload.get("id"),
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
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True)
class ConflictRecord:
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
    confidence: float = 0.0
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
            "dimensions": {key: value.to_dict() for key, value in self.dimensions.items()},
            "conflicts": [item.to_dict() for item in self.conflicts],
            "drift": self.drift.to_dict(),
            "coverage": self.coverage,
            "uncertainty": self.uncertainty,
            "required_missing": list(self.required_missing),
            "evidence_count": self.evidence_count,
            "explanation": list(self.explanation),
            "aggregate_score": self.aggregate_score,
            "confidence": self.confidence,
            "requires_review": self.requires_review,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class AlignmentTrajectoryTracker:
    """Bounded descriptive trajectory state; it is not a causal drift detector."""

    def __init__(self, *, history_limit: int, window: int, drift_threshold: float) -> None:
        if history_limit < 4:
            raise ValueError("history_limit must be >= 4")
        if window < 2:
            raise ValueError("window must be >= 2")
        if history_limit < 2 * window:
            raise ValueError("history_limit must be at least 2 * window")
        self.history_limit = int(history_limit)
        self.window = int(window)
        self.drift_threshold = _probability(drift_threshold, field_name="drift_threshold")
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)

    def observe(self, dimensions: Mapping[str, DimensionAssessment], *, assessment_id: Optional[str] = None) -> None:
        scores = {name: item.score for name, item in dimensions.items() if item.score is not None}
        self._history.append({"timestamp": _utc_now_iso(), "assessment_id": assessment_id, "scores": scores})

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
        dimensions = sorted({name for entry in historical + recent for name in entry["scores"]})
        shifts: Dict[str, float] = {}
        minimum = max(2, self.window // 2)
        for dimension in dimensions:
            old = [float(entry["scores"][dimension]) for entry in historical if dimension in entry["scores"]]
            new = [float(entry["scores"][dimension]) for entry in recent if dimension in entry["scores"]]
            if len(old) >= minimum and len(new) >= minimum:
                shifts[dimension] = float(fmean(new) - fmean(old))
        if not shifts:
            return DriftAssessment(
                detected=False,
                magnitude=0.0,
                direction="insufficient_comparable_evidence",
                historical_samples=len(historical),
                recent_samples=len(recent),
                threshold=self.drift_threshold,
            )
        affected = tuple(name for name, shift in sorted(shifts.items()) if abs(shift) >= self.drift_threshold)
        magnitude = min(1.0, max(abs(value) for value in shifts.values()))
        if not affected:
            direction = "stable"
        else:
            selected = [shifts[name] for name in affected]
            direction = "negative" if all(v < 0 for v in selected) else "positive" if all(v > 0 for v in selected) else "mixed"
        return DriftAssessment(
            detected=bool(affected),
            magnitude=magnitude,
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
        history = state.get("history", [])
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
            raise ValueError("trajectory history must be a sequence")
        self._history.clear()
        for item in history[-self.history_limit :]:
            if not isinstance(item, Mapping) or not isinstance(item.get("scores", {}), Mapping):
                continue
            scores: Dict[str, float] = {}
            for name, value in item.get("scores", {}).items():
                try:
                    scores[str(name)] = _probability(value, field_name=f"trajectory.score[{name}]")
                except ValueError:
                    continue
            self._history.append({
                "timestamp": str(item.get("timestamp") or _utc_now_iso()),
                "assessment_id": item.get("assessment_id"),
                "scores": scores,
            })


class AlignmentAssessor:
    """Aggregate provenance-preserving alignment evidence without hiding conflicts."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        AlignmentDimension.USER_INTENT.value: 1.0,
        AlignmentDimension.TASK_OBJECTIVE.value: 1.0,
        AlignmentDimension.CONSTRAINT.value: 1.0,
        AlignmentDimension.ENVIRONMENT.value: 0.75,
        AlignmentDimension.PREFERENCE.value: 0.75,
        AlignmentDimension.TRAJECTORY.value: 0.75,
        AlignmentDimension.SELF_CONSTRAINT.value: 0.50,
        AlignmentDimension.MULTI_AGENT.value: 0.50,
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.aligned_threshold = _probability(cfg.get("aligned_threshold", 0.75), field_name="aligned_threshold")
        self.misaligned_threshold = _probability(cfg.get("misaligned_threshold", 0.35), field_name="misaligned_threshold")
        if self.misaligned_threshold >= self.aligned_threshold:
            raise ValueError("misaligned_threshold must be below aligned_threshold")
        self.conflict_confidence = _probability(cfg.get("conflict_confidence", 0.60), field_name="conflict_confidence")
        self.min_confidence = _probability(cfg.get("min_confidence", 0.50), field_name="min_confidence")
        self.min_coverage = _probability(cfg.get("min_coverage", 0.50), field_name="min_coverage")
        weights = dict(self.DEFAULT_WEIGHTS)
        configured_weights = cfg.get("dimension_weights", {})
        if isinstance(configured_weights, Mapping):
            for key, value in configured_weights.items():
                dimension = _dimension(key).value
                number = float(value)
                if not math.isfinite(number) or number < 0.0:
                    raise ValueError(f"dimension weight for {dimension} must be finite and >= 0")
                weights[dimension] = number
        self.dimension_weights = weights
        required = cfg.get("required_dimensions", ["user_intent", "task_objective", "constraint"])
        self.required_dimensions = tuple(_dimension(item).value for item in required)
        history_limit = int(cfg.get("trajectory_history_limit", 120))
        window = int(cfg.get("trajectory_window", 8))
        self.tracker = AlignmentTrajectoryTracker(
            history_limit=history_limit,
            window=window,
            drift_threshold=_probability(cfg.get("drift_threshold", 0.15), field_name="drift_threshold"),
        )

    @staticmethod
    def _deduplicate(evidence: Iterable[AlignmentEvidence]) -> List[AlignmentEvidence]:
        output: List[AlignmentEvidence] = []
        by_id: Dict[Tuple[str, str, str], int] = {}
        for item in evidence:
            if item.evidence_id:
                key = (item.dimension.value, item.source, item.evidence_id)
                previous = by_id.get(key)
                if previous is not None:
                    output[previous] = item
                else:
                    by_id[key] = len(output)
                    output.append(item)
            else:
                output.append(item)
        return output

    def _dimension_summary(self, dimension: AlignmentDimension, items: Sequence[AlignmentEvidence]) -> Tuple[DimensionAssessment, List[ConflictRecord]]:
        scored = [item for item in items if item.score is not None]
        required = dimension.value in self.required_dimensions or any(item.required for item in items)
        sources = tuple(sorted({item.source for item in items}))
        conflicts: List[ConflictRecord] = []
        if not scored:
            return (
                DimensionAssessment(
                    dimension=dimension,
                    status=AlignmentStatus.UNCERTAIN,
                    score=None,
                    confidence=0.0,
                    evidence_count=len(items),
                    required=required,
                    sources=sources,
                    reasons=("no scored evidence",),
                ),
                conflicts,
            )

        weight_total = sum(item.confidence for item in scored)
        score = (sum(cast(float, item.score) * item.confidence for item in scored) / weight_total) if weight_total > 0 else None
        confidence = fmean(item.confidence for item in scored) if scored else 0.0
        high = [item for item in scored if item.confidence >= self.conflict_confidence and cast(float, item.score) >= self.aligned_threshold]
        low = [item for item in scored if item.confidence >= self.conflict_confidence and cast(float, item.score) <= self.misaligned_threshold]
        if high and low:
            conflicts.append(ConflictRecord(
                conflict_type="evidence_conflict",
                dimension=dimension,
                sources=tuple(sorted({item.source for item in high + low})),
                description=f"high-confidence evidence disagrees on {dimension.value}",
                details={"supporting": [item.to_dict() for item in high], "opposing": [item.to_dict() for item in low]},
            ))
            status = AlignmentStatus.CONFLICTING
        elif score is None or confidence < self.min_confidence:
            status = AlignmentStatus.UNCERTAIN
        elif score <= self.misaligned_threshold:
            status = AlignmentStatus.MISALIGNED
        elif score >= self.aligned_threshold:
            status = AlignmentStatus.ALIGNED
        else:
            status = AlignmentStatus.PARTIALLY_ALIGNED
        return (
            DimensionAssessment(
                dimension=dimension,
                status=status,
                score=score,
                confidence=confidence,
                evidence_count=len(items),
                required=required,
                sources=sources,
            ),
            conflicts,
        )

    def assess(
        self,
        evidence: Iterable[AlignmentEvidence],
        *,
        assessment_id: Optional[str] = None,
        record_trajectory: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AlignmentAssessment:
        normalized = self._deduplicate(evidence)
        grouped: Dict[AlignmentDimension, List[AlignmentEvidence]] = {dimension: [] for dimension in AlignmentDimension}
        for item in normalized:
            grouped[item.dimension].append(item)

        summaries: Dict[str, DimensionAssessment] = {}
        conflicts: List[ConflictRecord] = []
        for dimension in AlignmentDimension:
            if grouped[dimension] or dimension.value in self.required_dimensions:
                summary, dimension_conflicts = self._dimension_summary(dimension, grouped[dimension])
                summaries[dimension.value] = summary
                conflicts.extend(dimension_conflicts)

        required_missing = tuple(
            name for name in self.required_dimensions
            if name not in summaries or summaries[name].score is None
        )
        required_weight = sum(self.dimension_weights.get(name, 0.0) for name in self.required_dimensions)
        observed_required_weight = sum(
            self.dimension_weights.get(name, 0.0)
            for name in self.required_dimensions
            if name in summaries and summaries[name].score is not None
        )
        optional_present_weight = sum(
            self.dimension_weights.get(name, 0.0)
            for name, summary in summaries.items()
            if name not in self.required_dimensions and summary.score is not None
        )
        # Optional dimensions do not lower coverage merely because they are not
        # applicable to a task. They enter the denominator only when evidence
        # for them is actually supplied.
        expected_weight = required_weight + optional_present_weight
        coverage = (observed_required_weight + optional_present_weight) / expected_weight if expected_weight > 0 else 0.0

        aggregate_numerator = 0.0
        aggregate_denominator = 0.0
        confidence_numerator = 0.0
        for name, summary in summaries.items():
            if summary.score is None:
                continue
            base_weight = self.dimension_weights.get(name, 0.0)
            effective_weight = base_weight * summary.confidence
            aggregate_numerator += float(summary.score) * effective_weight
            aggregate_denominator += effective_weight
            confidence_numerator += summary.confidence * base_weight
        aggregate_score = aggregate_numerator / aggregate_denominator if aggregate_denominator > 0 else None
        represented_weight = sum(self.dimension_weights.get(name, 0.0) for name, summary in summaries.items() if summary.score is not None)
        confidence = confidence_numerator / represented_weight if represented_weight > 0 else 0.0

        uncertainty = max(0.0, min(1.0, 1.0 - (coverage * confidence)))
        if conflicts:
            status = AlignmentStatus.CONFLICTING
        elif required_missing or coverage < self.min_coverage or aggregate_score is None:
            status = AlignmentStatus.UNCERTAIN
        elif confidence < self.min_confidence:
            status = AlignmentStatus.UNCERTAIN
        elif aggregate_score <= self.misaligned_threshold:
            status = AlignmentStatus.MISALIGNED
        elif aggregate_score >= self.aligned_threshold and confidence >= self.min_confidence:
            status = AlignmentStatus.ALIGNED
        else:
            status = AlignmentStatus.PARTIALLY_ALIGNED

        if record_trajectory:
            self.tracker.observe(summaries, assessment_id=assessment_id)
        drift = self.tracker.detect()
        if drift.detected and drift.direction in {"negative", "mixed"} and status == AlignmentStatus.ALIGNED:
            status = AlignmentStatus.PARTIALLY_ALIGNED

        explanation: List[str] = []
        if required_missing:
            explanation.append("required alignment evidence is missing: " + ", ".join(required_missing))
        if conflicts:
            explanation.append(f"{len(conflicts)} evidence conflict(s) require review")
        if drift.detected:
            explanation.append(f"trajectory drift detected ({drift.direction}, magnitude={drift.magnitude:.3f})")
        if aggregate_score is not None:
            explanation.append(f"aggregate score={aggregate_score:.3f}; confidence={confidence:.3f}; coverage={coverage:.3f}")
        else:
            explanation.append("aggregate score unavailable because no scored evidence was supplied")

        return AlignmentAssessment(
            status=status,
            dimensions=summaries,
            conflicts=tuple(conflicts),
            drift=drift,
            coverage=coverage,
            uncertainty=uncertainty,
            required_missing=required_missing,
            evidence_count=len(normalized),
            explanation=tuple(explanation),
            aggregate_score=aggregate_score,
            confidence=confidence,
            metadata=dict(metadata or {}),
        )

__all__ = [
    # Enums
    "AlignmentStatus",
    "AlignmentDimension",
    "EvidencePolarity",
    # Dataclasses
    "AlignmentEvidence",
    "ConflictRecord",
    "DimensionAssessment",
    "DriftAssessment",
    "AlignmentAssessment",
    # Classes
    "AlignmentTrajectoryTracker",
    "AlignmentAssessor",
]