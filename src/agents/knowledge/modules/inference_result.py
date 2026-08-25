"""Typed inference result contracts for the Knowledge subsystem.

This module defines the stable data exchanged between rule inference,
orchestration, memory persistence, governance, and learning consumers.

It intentionally contains no inference, persistence, configuration,
logging, governance, or lifecycle behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class InferenceTrace:
    """Provenance for an accepted rule inference contribution.

    Attributes:
        fact:
            The fact produced by the rule.

        confidence:
            The confidence assigned to this contribution after RuleEngine
            applies the rule weight.

        rule:
            The name of the rule that produced the contribution.

        source:
            The configured source of the rule when available.

        sector:
            The sector under which the rule was explicitly executed.
            This may be ``None`` when inference was performed against the
            complete rule set rather than a sector-specific subset.
    """

    fact: Any
    confidence: float
    rule: str
    source: str = "unknown"
    sector: Optional[str] = None


@dataclass
class InferenceResult:
    """Stable result envelope returned by Knowledge rule inference.

    ``facts`` contains the final accepted facts and their resulting
    confidence values.

    ``traces`` contains provenance for rule contributions when tracing was
    requested by the caller.

    ``sector`` represents the explicitly requested or automatically
    detected sector for the inference operation. It does not imply that
    every individual rule was sector-scoped; consult ``InferenceTrace``
    for per-contribution execution provenance.
    """

    facts: Dict[Any, float] = field(default_factory=dict)
    traces: List[InferenceTrace] = field(default_factory=list)
    sector: Optional[str] = None


__all__ = [
    "InferenceTrace",
    "InferenceResult",
]