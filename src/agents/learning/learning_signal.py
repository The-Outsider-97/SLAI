"""Typed learning-signal contract for SLAI.

LearningSignal represents evidence from execution, evaluation, human feedback,
validation, errors, or other internal SLAI observations.

It deliberately does not prescribe one global reward function. The semantic
meaning of each observation remains explicit through ``metric`` and
``direction``.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from src.agents.base.utils.base_helpers import stable_fingerprint  # type: ignore
from .utils.learning_helpers import *


VALID_SIGNAL_KINDS = frozenset(
    {
        "outcome",
        "evaluation",
        "human_feedback",
        "correction",
        "validation",
        "error",
        "performance",
        "supervision",
    }
)

VALID_DIRECTIONS = frozenset(
    {
        "higher_better",
        "lower_better",
        "neutral",
    }
)

VALID_PROVENANCE = frozenset(
    {
        "execution",
        "evaluator",
        "human",
        "agent",
        "system",
        "inferred",
    }
)


@dataclass(frozen=True)
class LearningSignal:
    """One traceable observation that may contribute to future learning."""

    signal_id: str
    source_agent: str
    kind: str
    metric: str
    value: Optional[float] = None
    direction: str = "neutral"
    confidence: float = 1.0
    validated: bool = False
    provenance: str = "agent"
    strategy: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.signal_id.strip():
            raise ValueError("LearningSignal.signal_id must not be empty.")

        if not self.source_agent.strip():
            raise ValueError("LearningSignal.source_agent must not be empty.")

        if self.kind not in VALID_SIGNAL_KINDS:
            raise ValueError(
                f"Unsupported learning signal kind {self.kind!r}. "
                f"Expected one of {sorted(VALID_SIGNAL_KINDS)}."
            )

        if not self.metric.strip():
            raise ValueError("LearningSignal.metric must not be empty.")

        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"Unsupported direction {self.direction!r}. "
                f"Expected one of {sorted(VALID_DIRECTIONS)}."
            )

        if self.provenance not in VALID_PROVENANCE:
            raise ValueError(
                f"Unsupported provenance {self.provenance!r}. "
                f"Expected one of {sorted(VALID_PROVENANCE)}."
            )

        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("LearningSignal.value must be finite.")

        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("LearningSignal.confidence must be in [0, 1].")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LearningSignal":
        if not isinstance(payload, Mapping):
            raise TypeError("LearningSignal payload must be a mapping.")

        source_agent = str(
            payload.get("source_agent")
            or payload.get("agent")
            or "unknown"
        ).strip()

        kind = str(payload.get("kind", "outcome")).strip().lower()
        metric = str(payload.get("metric", "")).strip()

        raw_value = payload.get("value")
        value = (
            None
            if raw_value is None
            else coerce_float(raw_value, default=0.0)
        )

        signal_id = str(payload.get("signal_id") or "").strip()
        if not signal_id:
            signal_id = make_learning_id("signal")

        strategy_raw = payload.get("strategy")
        strategy = (
            None
            if strategy_raw is None
            else str(strategy_raw).strip() or None
        )

        context = to_json_safe(dict(payload.get("context") or {}))
        metadata = to_json_safe(dict(payload.get("metadata") or {}))

        return cls(
            signal_id=signal_id,
            source_agent=source_agent,
            kind=kind,
            metric=metric,
            value=value,
            direction=str(payload.get("direction", "neutral")).strip().lower(),
            confidence=coerce_float(
                payload.get("confidence", 1.0),
                default=1.0,
                minimum=0.0,
                maximum=1.0,
            ),
            validated=coerce_bool(payload.get("validated", False), default=False),
            provenance=str(payload.get("provenance", "agent")).strip().lower(),
            strategy=strategy,
            context=dict(context or {}),
            metadata=dict(metadata or {}),
            timestamp=str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat()))

    @property
    def context_key(self) -> str:
        """Deterministic context identity for contextual learning."""
        return stable_fingerprint(self.context, algorithm="sha256", length=24)

    def can_supervise_strategy(self, *, allowed_strategies: set[str], minimum_confidence: float) -> bool:
        """Whether the signal may train the strategy meta-controller.

        Merely observing an outcome does not automatically make that strategy
        the correct supervised label.
        """

        if not self.validated:
            return False

        if self.confidence < minimum_confidence:
            return False

        if self.strategy not in allowed_strategies:
            return False

        if self.kind not in {
            "human_feedback",
            "correction",
            "validation",
            "evaluation",
            "supervision",
        }:
            return False

        return "state" in self.context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "source_agent": self.source_agent,
            "kind": self.kind,
            "metric": self.metric,
            "value": self.value,
            "direction": self.direction,
            "confidence": self.confidence,
            "validated": self.validated,
            "provenance": self.provenance,
            "strategy": self.strategy,
            "context": to_json_safe(self.context),
            "context_key": self.context_key,
            "metadata": to_json_safe(self.metadata),
            "timestamp": self.timestamp,
        }


__all__ = [
    "VALID_SIGNAL_KINDS",
    "VALID_DIRECTIONS",
    "VALID_PROVENANCE",
    "LearningSignal",
]
