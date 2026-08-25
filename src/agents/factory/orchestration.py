"""Typed routing and adaptation contracts for AgentFactory orchestration."""

from __future__ import annotations

import copy
import math
import time
import uuid

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


def _names(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    output = []
    seen = set()
    for value in values or ():
        normalized = str(value).strip().lower().replace(" ", "_")
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return tuple(output)


def _finite(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


_QOS_FIELDS = {
    "max_error_rate",
    "max_latency_ms",
    "max_load",
    "min_availability",
    "min_health_score",
}
_CONSTRAINT_FIELDS = {
    "allow_out_of_process",
    "allowed_agents",
    "excluded_agents",
    "metadata_constraints",
    "required_tags",
}


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Declarative request evaluated by the factory routing authority."""

    candidates: Tuple[str, ...] = ()
    required_capabilities: Tuple[str, ...] = ()
    preferred_agents: Tuple[str, ...] = ()
    qos: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    create_if_missing: bool = True
    allow_degraded: bool = True
    require_metrics: bool = False
    require_fresh_metrics: bool = False
    max_metric_age_seconds: float = 30.0
    context: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    requested_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        for field_name in (
            "create_if_missing",
            "allow_degraded",
            "require_metrics",
            "require_fresh_metrics",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        for field_name in ("qos", "constraints", "metrics", "context"):
            if not isinstance(getattr(self, field_name), Mapping):
                raise ValueError(f"{field_name} must be a mapping")
        for field_name in ("candidates", "required_capabilities", "preferred_agents"):
            value = getattr(self, field_name)
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise ValueError(f"{field_name} must be a sequence")
        candidates = _names(self.candidates)
        capabilities = _names(self.required_capabilities)
        preferred = _names(self.preferred_agents)
        metric_age = _finite(self.max_metric_age_seconds, "max_metric_age_seconds")
        if metric_age <= 0:
            raise ValueError("max_metric_age_seconds must be > 0")
        qos = dict(self.qos or {})
        constraints = dict(self.constraints or {})
        unknown_qos = sorted(set(qos) - _QOS_FIELDS)
        if unknown_qos:
            raise ValueError(f"Unsupported QoS fields: {unknown_qos}")
        unknown_constraints = sorted(set(constraints) - _CONSTRAINT_FIELDS)
        if unknown_constraints:
            raise ValueError(f"Unsupported constraint fields: {unknown_constraints}")
        for field_name in ("allowed_agents", "excluded_agents", "required_tags"):
            value = constraints.get(field_name, ())
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise ValueError(f"constraints.{field_name} must be a sequence")
            constraints[field_name] = _names(value)
        metadata_constraints = constraints.get("metadata_constraints", {})
        if not isinstance(metadata_constraints, Mapping):
            raise ValueError("constraints.metadata_constraints must be a mapping")
        constraints["metadata_constraints"] = dict(metadata_constraints)
        if "allow_out_of_process" in constraints and not isinstance(
            constraints["allow_out_of_process"], bool
        ):
            raise ValueError("constraints.allow_out_of_process must be boolean")
        metrics: Dict[str, Dict[str, Any]] = {}
        for name, payload in dict(self.metrics or {}).items():
            if not isinstance(payload, Mapping):
                raise ValueError(f"metrics[{name}] must be a mapping")
            metrics[str(name).strip().lower()] = dict(payload)

        for field_name in ("min_health_score", "min_availability", "max_error_rate", "max_load"):
            if field_name in qos:
                value = _finite(qos[field_name], f"qos.{field_name}")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"qos.{field_name} must be within [0, 1]")
                qos[field_name] = value
        if "max_latency_ms" in qos:
            qos["max_latency_ms"] = _finite(qos["max_latency_ms"], "qos.max_latency_ms")
            if qos["max_latency_ms"] < 0:
                raise ValueError("qos.max_latency_ms must be >= 0")

        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "preferred_agents", preferred)
        object.__setattr__(self, "qos", qos)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "context", dict(self.context or {}))
        object.__setattr__(self, "max_metric_age_seconds", metric_age)
        if not str(self.request_id).strip():
            raise ValueError("request_id must not be empty")
        object.__setattr__(self, "request_id", str(self.request_id).strip())
        object.__setattr__(self, "requested_at", _finite(self.requested_at, "requested_at"))

    def to_dict(self, *, include_metrics: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "request_id": self.request_id,
            "requested_at": self.requested_at,
            "candidates": list(self.candidates),
            "required_capabilities": list(self.required_capabilities),
            "preferred_agents": list(self.preferred_agents),
            "qos": dict(self.qos),
            "constraints": dict(self.constraints),
            "create_if_missing": self.create_if_missing,
            "allow_degraded": self.allow_degraded,
            "require_metrics": self.require_metrics,
            "require_fresh_metrics": self.require_fresh_metrics,
            "max_metric_age_seconds": self.max_metric_age_seconds,
            "context": dict(self.context),
        }
        if include_metrics:
            payload["metrics"] = {name: dict(metrics) for name, metrics in self.metrics.items()}
        return payload


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    agent_type: str
    definition_id: str
    eligible: bool
    score: float
    reasons: Tuple[str, ...] = ()
    score_components: Mapping[str, float] = field(default_factory=dict)
    capabilities: Tuple[str, ...] = ()
    metric_age_seconds: Optional[float] = None
    metrics_fresh: bool = False
    runtime_health: str = "unknown"
    circuit_state: str = "closed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "definition_id": self.definition_id,
            "eligible": self.eligible,
            "score": self.score,
            "reasons": list(self.reasons),
            "score_components": dict(self.score_components),
            "capabilities": list(self.capabilities),
            "metric_age_seconds": self.metric_age_seconds,
            "metrics_fresh": self.metrics_fresh,
            "runtime_health": self.runtime_health,
            "circuit_state": self.circuit_state,
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    request_id: str
    selected_agent: str
    selected_definition_id: str
    selected_score: float
    evaluations: Tuple[CandidateEvaluation, ...]
    decided_at: float = field(default_factory=time.time)
    policy_version: str = "slai.factory.routing.v2.3"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "selected_agent": self.selected_agent,
            "selected_definition_id": self.selected_definition_id,
            "selected_score": self.selected_score,
            "decided_at": self.decided_at,
            "policy_version": self.policy_version,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }


@dataclass(frozen=True, slots=True)
class AdaptationChange:
    agent_type: str
    definition_id: str
    field_name: str
    previous: float
    proposed: float
    source: str

    def __post_init__(self) -> None:
        for field_name in ("agent_type", "definition_id", "field_name"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.source not in {"agent_config", "attribute", "metadata"}:
            raise ValueError("source must be 'agent_config', 'attribute', or 'metadata'")
        object.__setattr__(self, "previous", _finite(self.previous, "previous"))
        object.__setattr__(self, "proposed", _finite(self.proposed, "proposed"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "definition_id": self.definition_id,
            "field_name": self.field_name,
            "previous": self.previous,
            "proposed": self.proposed,
            "source": self.source,
        }


@dataclass(slots=True)
class AdaptationProposal:
    """Auditable, scoped set of proposed metadata changes."""

    target_agents: Tuple[str, ...]
    adjustments: Mapping[str, float]
    changes: Tuple[AdaptationChange, ...]
    evidence: Mapping[str, Any] = field(default_factory=dict)
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    decided_by: Optional[str] = None
    decision_reason: Optional[str] = None
    failure_reason: Optional[str] = None
    applied_changes: Tuple[Mapping[str, Any], ...] = ()
    conflicts: Tuple[Mapping[str, Any], ...] = ()
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.target_agents = _names(self.target_agents)
        if not self.target_agents:
            raise ValueError("target_agents must not be empty")
        self.adjustments = {
            str(name): _finite(value, f"adjustments.{name}")
            for name, value in dict(self.adjustments or {}).items()
        }
        self.changes = tuple(self.changes)
        invalid_changes = sorted(
            {
                change.agent_type
                for change in self.changes
                if change.agent_type not in set(self.target_agents)
            }
        )
        if invalid_changes:
            raise ValueError(
                f"Proposal changes fall outside target_agents: {invalid_changes}"
            )
        change_keys = [
            (change.definition_id, change.field_name)
            for change in self.changes
        ]
        if len(change_keys) != len(set(change_keys)):
            raise ValueError("Proposal contains duplicate definition/field changes")
        self.evidence = copy.deepcopy(dict(self.evidence or {}))
        self.proposal_id = str(self.proposal_id).strip()
        if not self.proposal_id:
            raise ValueError("proposal_id must not be empty")
        if self.status != "pending":
            raise ValueError("New adaptation proposals must start in 'pending' state")

    def approve(self, decided_by: str, reason: Optional[str] = None) -> None:
        with self._lock:
            if self.status != "pending":
                raise ValueError(f"Cannot approve proposal in state '{self.status}'")
            actor = str(decided_by).strip()
            if not actor:
                raise ValueError("Approval actor must not be empty")
            self.status = "approved"
            self.decided_at = time.time()
            self.decided_by = actor
            self.decision_reason = reason

    def reject(self, decided_by: str, reason: str) -> None:
        with self._lock:
            if self.status != "pending":
                raise ValueError(f"Cannot reject proposal in state '{self.status}'")
            actor = str(decided_by).strip()
            rejection_reason = str(reason).strip()
            if not actor:
                raise ValueError("Rejection actor must not be empty")
            if not rejection_reason:
                raise ValueError("Rejection reason must not be empty")
            self.status = "rejected"
            self.decided_at = time.time()
            self.decided_by = actor
            self.decision_reason = rejection_reason

    def mark_applied(
        self,
        applied_changes: Sequence[Mapping[str, Any]],
        conflicts: Sequence[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            if self.status != "approved":
                raise ValueError(f"Cannot apply proposal in state '{self.status}'")
            self.applied_changes = tuple(dict(item) for item in applied_changes)
            self.conflicts = tuple(dict(item) for item in conflicts)
            self.status = "applied" if self.applied_changes else ("conflicted" if self.conflicts else "no_op")

    def mark_failed(self, reason: str) -> None:
        with self._lock:
            if self.status not in {"pending", "approved"}:
                raise ValueError(f"Cannot fail proposal in state '{self.status}'")
            self.status = "failed"
            self.decided_at = self.decided_at or time.time()
            self.failure_reason = str(reason)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "proposal_id": self.proposal_id,
                "status": self.status,
                "target_agents": list(self.target_agents),
                "adjustments": dict(self.adjustments),
                "changes": [change.to_dict() for change in self.changes],
                "evidence": copy.deepcopy(dict(self.evidence)),
                "created_at": self.created_at,
                "decided_at": self.decided_at,
                "decided_by": self.decided_by,
                "decision_reason": self.decision_reason,
                "failure_reason": self.failure_reason,
                "applied_changes": [dict(item) for item in self.applied_changes],
                "conflicts": [dict(item) for item in self.conflicts],
            }


__all__ = [
    "AdaptationChange",
    "AdaptationProposal",
    "AgentRequest",
    "CandidateEvaluation",
    "RouteDecision",
]
