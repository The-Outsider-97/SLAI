"""Canonical value objects for SLAI tuning, evaluation, and promotion.

The types in this module describe facts produced by the tuning lifecycle; they
do not execute models or mutate agent state.  Search strategies, supervised
evaluators, agent evaluators, artifact writers, and promotion policies can
therefore exchange one stable representation without depending on one
another's implementations.

All records use aware UTC datetimes, finite metric values, defensive copies of
mutable inputs, and explicit lifecycle states.  These invariants are kept close
to the data model so invalid trial records cannot silently reach promotion.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .utils.tuning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Tuning Types")
printer = PrettyPrinter()


class TuningStrategy(str, Enum):
    """Search strategies provided by the tuning package."""

    BAYESIAN = "bayesian"
    GRID = "grid"

    @classmethod
    def parse(cls, value: "TuningStrategy | str") -> "TuningStrategy":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold()
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported tuning strategy {value!r}; expected one of: {supported}"
            ) from exc


class ObjectiveDirection(str, Enum):
    """Direction of improvement for a resolved optimization objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @classmethod
    def parse(cls, value: "ObjectiveDirection | str") -> "ObjectiveDirection":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("Objective direction must be 'minimize' or 'maximize'") from exc

    def better(self, candidate: float, incumbent: float) -> bool:
        _require_finite(candidate, "candidate")
        _require_finite(incumbent, "incumbent")
        return candidate < incumbent if self is self.MINIMIZE else candidate > incumbent


class RunStatus(str, Enum):
    """Lifecycle state of a search or complete tuning run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"

    @property
    def is_terminal(self) -> bool:
        return self not in {self.PENDING, self.RUNNING}


class TrialStatus(str, Enum):
    """Lifecycle state of one candidate configuration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PRUNED = "pruned"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self not in {self.PENDING, self.RUNNING}


class AgentStateSource(str, Enum):
    """How the baseline state for an agent trial was obtained."""

    CHECKPOINT = "checkpoint"
    FRESH = "fresh"


class CandidateStateDisposition(str, Enum):
    """Disposition of evaluator-mutated candidate state after a trial."""

    NOT_APPLIED = "not_applied"
    RESTORED = "restored"
    DISCARDED = "discarded"
    RESTORE_FAILED = "restore_failed"
    DISCARD_FAILED = "discard_failed"

    @property
    def is_isolated(self) -> bool:
        return self in {self.NOT_APPLIED, self.RESTORED, self.DISCARDED}


class ConstraintOperator(str, Enum):
    """Comparison used by an eligibility or safety constraint."""

    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    EQUAL = "=="

    def evaluate(self, observed: float, threshold: float) -> bool:
        _require_finite(observed, "observed")
        _require_finite(threshold, "threshold")
        if self is self.LESS_THAN:
            return observed < threshold
        if self is self.LESS_THAN_OR_EQUAL:
            return observed <= threshold
        if self is self.GREATER_THAN:
            return observed > threshold
        if self is self.GREATER_THAN_OR_EQUAL:
            return observed >= threshold
        return observed == threshold


class PromotionDecision(str, Enum):
    """Outcome returned by an explicit candidate-promotion policy."""

    PROMOTE = "promote"
    REJECT = "reject"
    DEFER = "defer"
    ROLLBACK = "rollback"


class ArtifactStatus(str, Enum):
    """Outcome of producing one tuning artifact."""

    WRITTEN = "written"
    FAILED = "failed"
    SKIPPED = "skipped"


def _require_non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_non_negative(value: float | int | None, name: str) -> float | int | None:
    if value is None:
        return None
    numeric = _require_finite(value, name)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _copy_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _copy_metric_mapping(value: Mapping[str, float], name: str) -> dict[str, float]:
    copied: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _require_non_empty(str(raw_key), f"{name} key")
        if key in copied:
            raise ValueError(f"{name} contains duplicate metric {key!r}")
        copied[key] = _require_finite(raw_value, f"{name}.{key}")
    return copied


def _json_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], to_json_safe(payload, redact_sensitive=True))


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Resolved primary objective used to rank successful trials."""

    name: str
    direction: ObjectiveDirection
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty(self.name, "metric name"))
        if not isinstance(self.direction, ObjectiveDirection):
            object.__setattr__(self, "direction", ObjectiveDirection.parse(self.direction))
        if self.unit is not None:
            object.__setattr__(self, "unit", _require_non_empty(self.unit, "metric unit"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "direction": self.direction.value, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class ConstraintEvaluation:
    """Observed result of one explicit eligibility or safety constraint."""

    name: str
    metric_name: str
    operator: ConstraintOperator
    threshold: float
    observed: float
    passed: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty(self.name, "constraint name"))
        object.__setattr__(
            self, "metric_name", _require_non_empty(self.metric_name, "constraint metric_name")
        )
        if not isinstance(self.operator, ConstraintOperator):
            object.__setattr__(self, "operator", ConstraintOperator(self.operator))
        threshold = _require_finite(self.threshold, "constraint threshold")
        observed = _require_finite(self.observed, "constraint observed")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "observed", observed)
        if not isinstance(self.passed, bool):
            raise TypeError("constraint passed must be a bool")
        expected = self.operator.evaluate(observed, threshold)
        if bool(self.passed) is not expected:
            raise ValueError(
                f"Constraint {self.name!r} passed={self.passed!r} contradicts "
                f"{observed} {self.operator.value} {threshold}"
            )
        if self.reason is not None:
            object.__setattr__(self, "reason", _require_non_empty(self.reason, "constraint reason"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "observed": self.observed,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Optional resource observations for an evaluation or whole trial."""

    wall_time_seconds: float | None = None
    cpu_time_seconds: float | None = None
    peak_memory_bytes: int | None = None
    latency_quantiles_seconds: Mapping[str, float] = field(default_factory=dict)
    sample_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wall_time_seconds",
            _require_non_negative(self.wall_time_seconds, "wall_time_seconds"),
        )
        object.__setattr__(
            self,
            "cpu_time_seconds",
            _require_non_negative(self.cpu_time_seconds, "cpu_time_seconds"),
        )
        peak = _require_non_negative(self.peak_memory_bytes, "peak_memory_bytes")
        if peak is not None and not isinstance(peak, int):
            raise TypeError("peak_memory_bytes must be an integer")
        if self.sample_count is not None:
            if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
                raise TypeError("sample_count must be an integer")
            if self.sample_count < 0:
                raise ValueError("sample_count must be non-negative")
        object.__setattr__(
            self,
            "latency_quantiles_seconds",
            _copy_metric_mapping(
                self.latency_quantiles_seconds, "latency_quantiles_seconds"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "wall_time_seconds": self.wall_time_seconds,
                "cpu_time_seconds": self.cpu_time_seconds,
                "peak_memory_bytes": self.peak_memory_bytes,
                "latency_quantiles_seconds": dict(self.latency_quantiles_seconds),
                "sample_count": self.sample_count,
            }
        )


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """Transport-safe reference to a failure without retaining traceback objects."""

    error_type: str
    message: str
    code: str | None = None
    error_id: str | None = None
    retryable: bool = False
    occurred_at: datetime = field(default_factory=utc_now)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "error_type", _require_non_empty(self.error_type, "error_type"))
        object.__setattr__(self, "message", _require_non_empty(self.message, "error message"))
        if self.code is not None:
            object.__setattr__(self, "code", _require_non_empty(self.code, "error code"))
        if self.error_id is not None:
            object.__setattr__(self, "error_id", _require_non_empty(self.error_id, "error_id"))
        _require_aware(self.occurred_at, "occurred_at")
        object.__setattr__(self, "details", _copy_mapping(self.details, "error details"))
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a bool")

    @classmethod
    def from_exception(cls, error: BaseException) -> "ErrorRecord":
        payload: Mapping[str, Any] = {}
        to_dict = getattr(error, "to_dict", None)
        if callable(to_dict):
            try:
                candidate = to_dict(include_traceback=False)
                if isinstance(candidate, Mapping):
                    payload = candidate
            except Exception:
                payload = {}
        raw_code = payload.get("code", getattr(error, "code", None))
        code = getattr(raw_code, "value", raw_code)
        raw_occurred_at = payload.get("timestamp")
        occurred_at = utc_now()
        if isinstance(raw_occurred_at, str):
            try:
                parsed = datetime.fromisoformat(raw_occurred_at.replace("Z", "+00:00"))
                if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                    occurred_at = parsed
            except ValueError:
                pass
        details = payload.get("details", {})
        return cls(
            error_type=error.__class__.__name__,
            message=str(error),
            code=None if code is None else str(code),
            error_id=payload.get("error_id", getattr(error, "error_id", None)),
            retryable=bool(payload.get("retryable", getattr(error, "retryable", False))),
            occurred_at=occurred_at,
            details=details if isinstance(details, Mapping) else {"details": details},
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "error_type": self.error_type,
                "message": self.message,
                "code": self.code,
                "error_id": self.error_id,
                "retryable": self.retryable,
                "occurred_at": self.occurred_at,
                "details": dict(self.details),
            }
        )


@dataclass(frozen=True, slots=True)
class AgentStateRecord:
    """Audit record for transactional state handling during an agent trial."""

    source: AgentStateSource
    transaction_id: str
    candidate_applied: bool
    disposition: CandidateStateDisposition
    baseline_checkpoint_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, AgentStateSource):
            object.__setattr__(self, "source", AgentStateSource(self.source))
        if not isinstance(self.disposition, CandidateStateDisposition):
            object.__setattr__(
                self, "disposition", CandidateStateDisposition(self.disposition)
            )
        object.__setattr__(
            self, "transaction_id", _require_non_empty(self.transaction_id, "transaction_id")
        )
        if not isinstance(self.candidate_applied, bool):
            raise TypeError("candidate_applied must be a bool")
        if self.source is AgentStateSource.CHECKPOINT and not self.baseline_checkpoint_id:
            raise ValueError("checkpoint state source requires baseline_checkpoint_id")
        if self.baseline_checkpoint_id is not None:
            object.__setattr__(
                self,
                "baseline_checkpoint_id",
                _require_non_empty(self.baseline_checkpoint_id, "baseline_checkpoint_id"),
            )
        if self.candidate_applied and self.disposition is CandidateStateDisposition.NOT_APPLIED:
            raise ValueError("applied candidate state cannot have disposition NOT_APPLIED")
        valid_unapplied_dispositions = {
            CandidateStateDisposition.NOT_APPLIED,
            CandidateStateDisposition.RESTORE_FAILED,
            CandidateStateDisposition.DISCARD_FAILED,
        }
        if (
            not self.candidate_applied
            and self.disposition not in valid_unapplied_dispositions
        ):
            raise ValueError(
                "unapplied candidate state must be NOT_APPLIED or record a "
                "failed cleanup"
            )
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "state metadata"))

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "source": self.source.value,
                "transaction_id": self.transaction_id,
                "candidate_applied": self.candidate_applied,
                "disposition": self.disposition.value,
                "baseline_checkpoint_id": self.baseline_checkpoint_id,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class AgentStateAudit:
    """State-isolation evidence for a multi-seed agent trial."""

    transactions: Sequence[AgentStateRecord]
    reset_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transactions = tuple(self.transactions)
        if not transactions:
            raise ValueError("AgentStateAudit requires at least one transaction")
        if any(not isinstance(item, AgentStateRecord) for item in transactions):
            raise TypeError("transactions must contain AgentStateRecord objects")
        identifiers = [item.transaction_id for item in transactions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Agent state transaction identifiers must be unique")
        object.__setattr__(self, "transactions", transactions)
        if isinstance(self.reset_count, bool) or not isinstance(self.reset_count, int):
            raise TypeError("reset_count must be an integer")
        if self.reset_count < 0:
            raise ValueError("reset_count must be non-negative")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "state audit metadata"))

    @property
    def candidate_applied(self) -> bool:
        return all(item.candidate_applied for item in self.transactions)

    @property
    def is_isolated(self) -> bool:
        return all(item.disposition.is_isolated for item in self.transactions)

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "transactions": [item.to_dict() for item in self.transactions],
                "reset_count": self.reset_count,
                "candidate_applied": self.candidate_applied,
                "is_isolated": self.is_isolated,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class EvaluationSlice:
    """Result for one scenario/seed (or supervised fold) within a trial."""

    status: TrialStatus
    metrics: Mapping[str, float] = field(default_factory=dict)
    scenario_id: str | None = None
    seed: int | None = None
    resources: ResourceUsage | None = None
    error: ErrorRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, TrialStatus):
            object.__setattr__(self, "status", TrialStatus(self.status))
        if not self.status.is_terminal:
            raise ValueError("EvaluationSlice must describe a terminal evaluation")
        if self.scenario_id is not None:
            object.__setattr__(
                self, "scenario_id", _require_non_empty(self.scenario_id, "scenario_id")
            )
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "metrics", _copy_metric_mapping(self.metrics, "slice metrics"))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "slice metadata"))
        if self.status is TrialStatus.SUCCEEDED and self.error is not None:
            raise ValueError("A successful evaluation slice cannot contain an error")
        if self.status is TrialStatus.FAILED and self.error is None:
            raise ValueError("A failed evaluation slice must contain an error")

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "status": self.status.value,
                "scenario_id": self.scenario_id,
                "seed": self.seed,
                "metrics": dict(self.metrics),
                "resources": None if self.resources is None else self.resources.to_dict(),
                "error": None if self.error is None else self.error.to_dict(),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """Immutable final record for one candidate configuration."""

    trial_id: str
    run_id: str
    status: TrialStatus
    parameters: Mapping[str, Any]
    started_at: datetime
    completed_at: datetime
    metrics: Mapping[str, float] = field(default_factory=dict)
    objective_value: float | None = None
    evaluations: Sequence[EvaluationSlice] = field(default_factory=tuple)
    constraints: Sequence[ConstraintEvaluation] = field(default_factory=tuple)
    resources: ResourceUsage | None = None
    agent_state: AgentStateRecord | AgentStateAudit | None = None
    error: ErrorRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_id", _require_non_empty(self.trial_id, "trial_id"))
        object.__setattr__(self, "run_id", _require_non_empty(self.run_id, "run_id"))
        if not isinstance(self.status, TrialStatus):
            object.__setattr__(self, "status", TrialStatus(self.status))
        if not self.status.is_terminal:
            raise ValueError("TrialRecord must describe a terminal trial")
        _require_aware(self.started_at, "started_at")
        _require_aware(self.completed_at, "completed_at")
        elapsed_seconds(self.started_at, self.completed_at)
        object.__setattr__(self, "parameters", _copy_mapping(self.parameters, "parameters"))
        object.__setattr__(self, "metrics", _copy_metric_mapping(self.metrics, "trial metrics"))
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if any(not isinstance(item, EvaluationSlice) for item in self.evaluations):
            raise TypeError("evaluations must contain EvaluationSlice objects")
        if any(not isinstance(item, ConstraintEvaluation) for item in self.constraints):
            raise TypeError("constraints must contain ConstraintEvaluation objects")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "trial metadata"))
        if self.agent_state is not None and not isinstance(
            self.agent_state, (AgentStateRecord, AgentStateAudit)
        ):
            raise TypeError("agent_state must be AgentStateRecord or AgentStateAudit")
        if self.objective_value is not None:
            object.__setattr__(self, "objective_value", _require_finite(self.objective_value, "objective_value"))
        if self.status is TrialStatus.SUCCEEDED:
            if self.objective_value is None:
                raise ValueError("A successful trial requires objective_value")
            if self.error is not None:
                raise ValueError("A successful trial cannot contain an error")
            if self.agent_state is not None:
                if not self.agent_state.candidate_applied:
                    raise ValueError("A successful agent trial must apply candidate state")
                isolated = (
                    self.agent_state.disposition.is_isolated
                    if isinstance(self.agent_state, AgentStateRecord)
                    else self.agent_state.is_isolated
                )
                if not isolated:
                    raise ValueError(
                        "A successful agent trial must restore or discard candidate state"
                    )
        if self.status is TrialStatus.FAILED and self.error is None:
            raise ValueError("A failed trial requires an error record")

    @property
    def duration_seconds(self) -> float:
        return elapsed_seconds(self.started_at, self.completed_at)

    @property
    def constraints_passed(self) -> bool:
        return all(item.passed for item in self.constraints)

    @property
    def eligible_for_promotion(self) -> bool:
        return (
            self.status is TrialStatus.SUCCEEDED
            and self.constraints_passed
            and (
                self.agent_state is None
                or (
                    self.agent_state.disposition.is_isolated
                    if isinstance(self.agent_state, AgentStateRecord)
                    else self.agent_state.is_isolated
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "trial_id": self.trial_id,
                "run_id": self.run_id,
                "status": self.status.value,
                "parameters": dict(self.parameters),
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "duration_seconds": self.duration_seconds,
                "metrics": dict(self.metrics),
                "objective_value": self.objective_value,
                "evaluations": [item.to_dict() for item in self.evaluations],
                "constraints": [item.to_dict() for item in self.constraints],
                "constraints_passed": self.constraints_passed,
                "resources": None if self.resources is None else self.resources.to_dict(),
                "agent_state": None if self.agent_state is None else self.agent_state.to_dict(),
                "error": None if self.error is None else self.error.to_dict(),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Final, strategy-independent result returned by a search runner."""

    run_id: str
    strategy: TuningStrategy
    status: RunStatus
    objective: MetricSpec
    trials: Sequence[TrialRecord]
    started_at: datetime
    completed_at: datetime
    best_trial_id: str | None = None
    warnings: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_non_empty(self.run_id, "run_id"))
        if not isinstance(self.strategy, TuningStrategy):
            object.__setattr__(self, "strategy", TuningStrategy.parse(self.strategy))
        if not isinstance(self.status, RunStatus):
            object.__setattr__(self, "status", RunStatus(self.status))
        if not self.status.is_terminal:
            raise ValueError("SearchResult must describe a terminal search")
        if self.status is RunStatus.ROLLED_BACK:
            raise ValueError("ROLLED_BACK is a deployment result, not a search status")
        _require_aware(self.started_at, "started_at")
        _require_aware(self.completed_at, "completed_at")
        elapsed_seconds(self.started_at, self.completed_at)
        if not isinstance(self.objective, MetricSpec):
            raise TypeError("objective must be a MetricSpec")
        trials = tuple(self.trials)
        if any(not isinstance(trial, TrialRecord) for trial in trials):
            raise TypeError("trials must contain TrialRecord objects")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(
            self,
            "warnings",
            tuple(_require_non_empty(str(item), "warning") for item in self.warnings),
        )
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "search metadata"))

        trial_ids: set[str] = set()
        for trial in trials:
            if trial.run_id != self.run_id:
                raise ValueError(f"Trial {trial.trial_id!r} belongs to run {trial.run_id!r}, not {self.run_id!r}")
            if trial.trial_id in trial_ids:
                raise ValueError(f"Duplicate trial_id {trial.trial_id!r}")
            trial_ids.add(trial.trial_id)
        if self.best_trial_id is not None:
            object.__setattr__(self, "best_trial_id", _require_non_empty(self.best_trial_id, "best_trial_id"))
            if self.best_trial_id not in trial_ids:
                raise ValueError("best_trial_id does not reference a returned trial")
            best = self.best_trial
            if best is None or best.status is not TrialStatus.SUCCEEDED:
                raise ValueError("best_trial_id must reference a successful trial")
            if not best.constraints_passed:
                raise ValueError("best_trial_id cannot reference a constraint-violating trial")
        if self.status in {RunStatus.SUCCEEDED, RunStatus.DEGRADED} and self.best_trial_id is None:
            raise ValueError("A successful search requires best_trial_id")

    @property
    def duration_seconds(self) -> float:
        return elapsed_seconds(self.started_at, self.completed_at)

    @property
    def best_trial(self) -> TrialRecord | None:
        if self.best_trial_id is None:
            return None
        return next((trial for trial in self.trials if trial.trial_id == self.best_trial_id), None)

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "run_id": self.run_id,
                "strategy": self.strategy.value,
                "status": self.status.value,
                "objective": self.objective.to_dict(),
                "best_trial_id": self.best_trial_id,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "duration_seconds": self.duration_seconds,
                "trials": [trial.to_dict() for trial in self.trials],
                "warnings": list(self.warnings),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Result of writing one artifact; content ownership stays with artifacts.py."""

    kind: str
    status: ArtifactStatus
    path: Path | None = None
    checksum: str | None = None
    error: ErrorRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _require_non_empty(self.kind, "artifact kind"))
        if not isinstance(self.status, ArtifactStatus):
            object.__setattr__(self, "status", ArtifactStatus(self.status))
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        if self.checksum is not None:
            object.__setattr__(self, "checksum", _require_non_empty(self.checksum, "checksum"))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "artifact metadata"))
        if self.status is ArtifactStatus.WRITTEN and self.path is None:
            raise ValueError("A written artifact requires a path")
        if self.status is ArtifactStatus.FAILED and self.error is None:
            raise ValueError("A failed artifact requires an error record")

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "kind": self.kind,
                "status": self.status.value,
                "path": self.path,
                "checksum": self.checksum,
                "error": None if self.error is None else self.error.to_dict(),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """Auditable outcome of an explicit promotion/rollback policy."""

    policy_name: str
    decision: PromotionDecision
    candidate_trial_id: str
    reason: str
    decided_at: datetime = field(default_factory=utc_now)
    incumbent_checkpoint_id: str | None = None
    promoted_checkpoint_id: str | None = None
    rollback_checkpoint_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_name", _require_non_empty(self.policy_name, "policy_name"))
        if not isinstance(self.decision, PromotionDecision):
            object.__setattr__(self, "decision", PromotionDecision(self.decision))
        object.__setattr__(
            self,
            "candidate_trial_id",
            _require_non_empty(self.candidate_trial_id, "candidate_trial_id"),
        )
        object.__setattr__(self, "reason", _require_non_empty(self.reason, "promotion reason"))
        _require_aware(self.decided_at, "decided_at")
        for name in (
            "incumbent_checkpoint_id",
            "promoted_checkpoint_id",
            "rollback_checkpoint_id",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_non_empty(value, name))
        if self.decision is PromotionDecision.PROMOTE and self.promoted_checkpoint_id is None:
            raise ValueError("PROMOTE requires promoted_checkpoint_id")
        if self.decision is PromotionDecision.ROLLBACK and self.rollback_checkpoint_id is None:
            raise ValueError("ROLLBACK requires rollback_checkpoint_id")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "promotion metadata"))

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "policy_name": self.policy_name,
                "decision": self.decision.value,
                "candidate_trial_id": self.candidate_trial_id,
                "reason": self.reason,
                "decided_at": self.decided_at,
                "incumbent_checkpoint_id": self.incumbent_checkpoint_id,
                "promoted_checkpoint_id": self.promoted_checkpoint_id,
                "rollback_checkpoint_id": self.rollback_checkpoint_id,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True, slots=True)
class TunerSettings:
    """Validated orchestrator settings derived from one config snapshot."""

    strategy: TuningStrategy
    model_type: str
    allow_generate: bool
    output_dir: Path | None = None
    config_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, TuningStrategy):
            object.__setattr__(self, "strategy", TuningStrategy.parse(self.strategy))
        object.__setattr__(self, "model_type", _require_non_empty(self.model_type, "model_type"))
        if not isinstance(self.allow_generate, bool):
            raise TypeError("allow_generate must be a bool")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.config_path is not None:
            object.__setattr__(self, "config_path", Path(self.config_path))

    def to_dict(self) -> dict[str, Any]:
        return _json_dict(
            {
                "strategy": self.strategy.value,
                "model_type": self.model_type,
                "allow_generate": self.allow_generate,
                "output_dir": self.output_dir,
                "config_path": self.config_path,
            }
        )


@dataclass(frozen=True, slots=True)
class TuningRunRequest:
    """Immutable input handed from the tuner to one search runner.

    ``objective`` is optional because the existing v2.2 configuration permits
    ``objective: auto``.  The search runner must resolve that ambiguity and
    return a concrete ``MetricSpec`` in ``SearchResult``.
    """

    run_id: str
    settings: TunerSettings
    config: Mapping[str, Any]
    strategy_config: Mapping[str, Any]
    search_space: Sequence[Mapping[str, Any]]
    config_fingerprint: str
    objective: MetricSpec | None = None
    seeds: Sequence[int] = field(default_factory=tuple)
    scenario_ids: Sequence[str] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_non_empty(self.run_id, "run_id"))
        if not isinstance(self.settings, TunerSettings):
            raise TypeError("settings must be a TunerSettings object")
        object.__setattr__(self, "config", _copy_mapping(self.config, "config"))
        object.__setattr__(self, "strategy_config", _copy_mapping(self.strategy_config, "strategy_config"))
        copied_space: list[dict[str, Any]] = []
        for index, item in enumerate(self.search_space):
            copied_space.append(_copy_mapping(item, f"search_space[{index}]"))
        object.__setattr__(self, "search_space", tuple(copied_space))
        object.__setattr__(self, "config_fingerprint", _require_non_empty(self.config_fingerprint, "config_fingerprint"))
        seeds = tuple(self.seeds)
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise TypeError("Every seed must be an integer")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", seeds)
        scenario_ids = tuple(_require_non_empty(str(item), "scenario_id") for item in self.scenario_ids)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_ids must not contain duplicates")
        object.__setattr__(self, "scenario_ids", scenario_ids)
        _require_aware(self.created_at, "created_at")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "request metadata"))

    def to_dict(self, *, include_config: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "settings": self.settings.to_dict(),
            "strategy_config": dict(self.strategy_config),
            "search_space": [dict(item) for item in self.search_space],
            "config_fingerprint": self.config_fingerprint,
            "objective": None if self.objective is None else self.objective.to_dict(),
            "seeds": list(self.seeds),
            "scenario_ids": list(self.scenario_ids),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
        if include_config:
            payload["config"] = dict(self.config)
        return _json_dict(payload)


@dataclass(frozen=True, slots=True)
class TuningResult:
    """Top-level result spanning search, optional promotion, and artifacts."""

    request: TuningRunRequest
    status: RunStatus
    started_at: datetime
    completed_at: datetime
    search_result: SearchResult | None = None
    promotion: PromotionRecord | None = None
    artifacts: Sequence[ArtifactRecord] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    error: ErrorRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, RunStatus):
            object.__setattr__(self, "status", RunStatus(self.status))
        if not self.status.is_terminal:
            raise ValueError("TuningResult must describe a terminal run")
        _require_aware(self.started_at, "started_at")
        _require_aware(self.completed_at, "completed_at")
        elapsed_seconds(self.started_at, self.completed_at)
        if not isinstance(self.request, TuningRunRequest):
            raise TypeError("request must be a TuningRunRequest")
        if self.search_result is not None and self.search_result.run_id != self.request.run_id:
            raise ValueError("search_result run_id does not match request")
        if self.search_result is not None:
            allowed_statuses = {self.search_result.status}
            if self.search_result.status is RunStatus.SUCCEEDED:
                allowed_statuses.update({RunStatus.DEGRADED, RunStatus.ROLLED_BACK})
            elif self.search_result.status is RunStatus.DEGRADED:
                allowed_statuses.add(RunStatus.ROLLED_BACK)
            if self.status not in allowed_statuses:
                raise ValueError("TuningResult status is inconsistent with SearchResult status")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRecord) for item in artifacts):
            raise TypeError("artifacts must contain ArtifactRecord objects")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "warnings",
            tuple(_require_non_empty(str(item), "warning") for item in self.warnings),
        )
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "result metadata"))
        if self.status in {RunStatus.SUCCEEDED, RunStatus.DEGRADED, RunStatus.ROLLED_BACK}:
            if self.search_result is None or self.search_result.best_trial is None:
                raise ValueError(f"{self.status.value} result requires a successful search")
        if self.status is RunStatus.FAILED and self.error is None and self.search_result is None:
            raise ValueError("A failed result without a search result requires an error")
        if self.promotion is not None:
            search = self.search_result
            if search is None or self.promotion.candidate_trial_id not in {
                trial.trial_id for trial in search.trials
            }:
                raise ValueError("Promotion candidate must reference a returned trial")
        if self.status is RunStatus.ROLLED_BACK and (
            self.promotion is None
            or self.promotion.decision is not PromotionDecision.ROLLBACK
        ):
            raise ValueError("ROLLED_BACK requires an explicit rollback promotion record")

    @property
    def run_id(self) -> str:
        return self.request.run_id

    @property
    def strategy(self) -> str:
        """Backward-compatible v2.2 strategy value."""

        return self.request.settings.strategy.value

    @property
    def model_type(self) -> str:
        """Backward-compatible v2.2 model type."""

        return self.request.settings.model_type

    @property
    def duration_seconds(self) -> float:
        return elapsed_seconds(self.started_at, self.completed_at)

    @property
    def elapsed_seconds(self) -> float:
        """Backward-compatible alias for ``duration_seconds``."""

        return self.duration_seconds

    @property
    def completed_at_utc(self) -> str:
        """Backward-compatible serialized completion timestamp."""

        return utc_iso(self.completed_at)

    @property
    def best_trial(self) -> TrialRecord | None:
        return None if self.search_result is None else self.search_result.best_trial

    @property
    def best_params(self) -> dict[str, Any]:
        return {} if self.best_trial is None else dict(self.best_trial.parameters)

    @property
    def best_score(self) -> float | None:
        return None if self.best_trial is None else self.best_trial.objective_value

    def to_dict(self, *, include_config: bool = True) -> dict[str, Any]:
        return _json_dict(
            {
                "run_id": self.run_id,
                "status": self.status.value,
                "request": self.request.to_dict(include_config=include_config),
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "duration_seconds": self.duration_seconds,
                "search_result": (
                    None if self.search_result is None else self.search_result.to_dict()
                ),
                "promotion": None if self.promotion is None else self.promotion.to_dict(),
                "artifacts": [item.to_dict() for item in self.artifacts],
                "warnings": list(self.warnings),
                "error": None if self.error is None else self.error.to_dict(),
                "metadata": dict(self.metadata),
            }
        )


__all__ = [
    "AgentStateRecord",
    "AgentStateAudit",
    "AgentStateSource",
    "ArtifactRecord",
    "ArtifactStatus",
    "CandidateStateDisposition",
    "ConstraintEvaluation",
    "ConstraintOperator",
    "ErrorRecord",
    "EvaluationSlice",
    "MetricSpec",
    "ObjectiveDirection",
    "PromotionDecision",
    "PromotionRecord",
    "ResourceUsage",
    "RunStatus",
    "SearchResult",
    "TrialRecord",
    "TrialStatus",
    "TunerSettings",
    "TuningResult",
    "TuningRunRequest",
    "TuningStrategy",
]

if __name__ == "__main__":
    from collections.abc import Callable
    print("\n=== Running Tuning Types Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting tuning type tests", "info")

    _failures: list[str] = []

    def _check(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def _run_test(name: str, test: Callable[[], None]) -> None:
        try:
            test()
            printer.status("TEST", name, "success")
        except Exception as exc:
            _failures.append(f"{name}: {type(exc).__name__}: {exc}")
            printer.status("TEST", _failures[-1], "error")

    def _trial(
        trial_id: str,
        value: float,
        *,
        agent_state: AgentStateRecord | AgentStateAudit | None = None,
    ) -> TrialRecord:
        started = utc_now()
        return TrialRecord(
            trial_id=trial_id,
            run_id="types-self-test",
            status=TrialStatus.SUCCEEDED,
            parameters={"x": value},
            started_at=started,
            completed_at=utc_now(),
            metrics={"loss": value},
            objective_value=value,
            agent_state=agent_state,
        )

    def _test_enum_and_metric_semantics() -> None:
        _check(
            TuningStrategy.parse(" GRID ") is TuningStrategy.GRID,
            "strategy normalization failed",
        )
        _check(
            ObjectiveDirection.MINIMIZE.better(1.0, 2.0),
            "minimization ordering failed",
        )
        _check(
            ObjectiveDirection.MAXIMIZE.better(2.0, 1.0),
            "maximization ordering failed",
        )
        try:
            MetricSpec("loss", cast(ObjectiveDirection, "sideways"))
        except ValueError:
            return
        raise AssertionError("invalid objective direction was accepted")

    def _test_defensive_record_copy() -> None:
        parameters = {"depth": 2}
        metrics = {"loss": 0.5}
        started = utc_now()
        trial = TrialRecord(
            trial_id="copy",
            run_id="types-self-test",
            status=TrialStatus.SUCCEEDED,
            parameters=parameters,
            started_at=started,
            completed_at=utc_now(),
            metrics=metrics,
            objective_value=0.5,
        )
        parameters["depth"] = 99
        metrics["loss"] = 99.0
        _check(trial.parameters["depth"] == 2, "parameters were not copied")
        _check(trial.metrics["loss"] == 0.5, "metrics were not copied")

    def _test_agent_isolation_contract() -> None:
        state = AgentStateRecord(
            source=AgentStateSource.FRESH,
            transaction_id="transaction-1",
            candidate_applied=True,
            disposition=CandidateStateDisposition.DISCARDED,
        )
        audit = AgentStateAudit((state,), reset_count=2)
        trial = _trial("isolated", 0.25, agent_state=audit)
        _check(audit.is_isolated, "discarded candidate was not isolated")
        _check(trial.eligible_for_promotion, "isolated trial was not eligible")

        failed_state = AgentStateRecord(
            source=AgentStateSource.FRESH,
            transaction_id="transaction-2",
            candidate_applied=True,
            disposition=CandidateStateDisposition.DISCARD_FAILED,
        )
        try:
            _trial("unsafe", 0.1, agent_state=failed_state)
        except ValueError:
            return
        raise AssertionError("successful trial accepted failed state cleanup")

    def _test_search_result_identity() -> None:
        objective = MetricSpec("loss", ObjectiveDirection.MINIMIZE)
        trials = (_trial("trial-a", 2.0), _trial("trial-b", 1.0))
        result = SearchResult(
            run_id="types-self-test",
            strategy=TuningStrategy.GRID,
            status=RunStatus.SUCCEEDED,
            objective=objective,
            trials=trials,
            started_at=trials[0].started_at,
            completed_at=utc_now(),
            best_trial_id="trial-b",
        )
        _check(result.best_trial is trials[1], "best trial resolution failed")
        _check(result.to_dict()["best_trial_id"] == "trial-b", "serialization failed")

    _run_test("enum parsing and objective ordering", _test_enum_and_metric_semantics)
    _run_test("defensive record copying", _test_defensive_record_copy)
    _run_test("agent-state isolation", _test_agent_isolation_contract)
    _run_test("search-result identity", _test_search_result_identity)

    _all_passed = not _failures
    printer.status(
        "",
        f"{4 - len(_failures)}/4 tuning type tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All tuning type tests passed ===\n")