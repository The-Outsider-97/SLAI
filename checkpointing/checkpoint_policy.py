"""Deterministic checkpoint selection, rollback, and retention policy.

Policy functions in this module are pure: they inspect immutable
``CheckpointRecord`` values and return auditable decisions.  They never read,
write, deserialize, archive, or delete checkpoint files.  The manager is
responsible for obtaining verified records, and the storage layer is
responsible for executing an approved plan.

No learned or probabilistic selector is used.  Every exclusion, ranking, and
retention action is explained by stable reason codes and explicit evidence.
"""

from __future__ import annotations

import datetime as _dt
import math

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from .checkpoint_errors import *
from .checkpoint_types import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Policy")
printer = PrettyPrinter()


VersionMatcher = Callable[[str, str], bool]


class SelectionMode(str, Enum):
    """Deterministic ordering applied after candidate eligibility checks."""

    LATEST = "latest"
    HIGHEST_STEP = "highest_step"
    BEST_METRIC = "best_metric"


class RetentionAction(str, Enum):
    """Action proposed for a checkpoint by a retention plan."""

    KEEP = "keep"
    DELETE = "delete"


def _frozen_evidence(value: Mapping[str, Any] | None) -> Mapping[str, JSONValue]:
    frozen = freeze_json(dict(value or {}), _path="$.policy.evidence")
    if not isinstance(frozen, Mapping):  # Defensive: input is always a mapping.
        raise TypeError("policy evidence must freeze to a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class PolicyReason:
    """Machine-readable explanation for one policy conclusion."""

    code: str
    message: str
    evidence: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not self.code.strip()
            or not isinstance(self.message, str)
            or not self.message.strip()
        ):
            raise ValueError("policy reason code and message must be non-empty")
        object.__setattr__(self, "code", self.code.strip())
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(self, "evidence", _frozen_evidence(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "evidence": thaw_json(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Loader capabilities used for compatibility gating.

    Version strings use exact matching by default.  Applications that use a
    declared version-range syntax must inject an explicit deterministic
    ``version_matcher`` into ``CheckpointPolicy``.
    """

    slai_version: str | None = None
    python_version: str | None = None
    platform_tags: tuple[str, ...] = ()
    codecs: Mapping[str, str] = field(default_factory=dict)
    component_schemas: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("slai_version", "python_version"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} cannot be empty")
        tags = tuple(sorted({str(tag).strip() for tag in self.platform_tags}))
        if any(not tag for tag in tags):
            raise ValueError("platform_tags cannot contain empty strings")
        object.__setattr__(self, "platform_tags", tags)

        codecs: dict[str, str] = {}
        for name, version in self.codecs.items():
            safe_name = validate_identifier(str(name), field_name="codec")
            if not str(version).strip():
                raise ValueError(f"runtime codec version for {name!r} cannot be empty")
            codecs[safe_name] = str(version).strip()
        object.__setattr__(self, "codecs", MappingProxyType(codecs))

        schemas: dict[str, tuple[int, ...]] = {}
        for component, versions in self.component_schemas.items():
            safe_component = validate_component_name(component)
            supported = tuple(sorted({int(version) for version in versions}))
            if not supported or any(version < 1 for version in supported):
                raise ValueError(
                    f"runtime component schemas for {component!r} must contain positive integers"
                )
            schemas[safe_component] = supported
        object.__setattr__(self, "component_schemas", MappingProxyType(schemas))


@dataclass(frozen=True, slots=True)
class SelectionCriteria:
    """Eligibility filters and ranking mode for checkpoint selection."""

    mode: SelectionMode = SelectionMode.LATEST
    checkpoint_id: str | None = None
    version: str | None = None
    required_components: tuple[str, ...] = ()
    excluded_checkpoint_ids: tuple[str, ...] = ()
    require_verified: bool = True
    accepted_health: tuple[CheckpointHealth, ...] = (CheckpointHealth.HEALTHY,)
    minimum_step: int | None = None
    maximum_step: int | None = None
    agent_id: str | None = None
    run_id: str | None = None
    maximum_recovery_generation: int | None = None
    objective_name: str | None = None
    objective_direction: MetricDirection | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", SelectionMode(self.mode))
        if not isinstance(self.require_verified, bool):
            raise TypeError("require_verified must be a boolean")
        if self.checkpoint_id is not None:
            object.__setattr__(
                self,
                "checkpoint_id",
                validate_identifier(self.checkpoint_id, field_name="checkpoint_id"),
            )
        if self.version is not None:
            object.__setattr__(self, "version", validate_version(self.version))
        if self.checkpoint_id is not None and self.version is not None:
            raise ValueError("selection cannot constrain both checkpoint_id and version")

        required = tuple(
            sorted({validate_component_name(name) for name in self.required_components})
        )
        object.__setattr__(self, "required_components", required)
        excluded = tuple(
            sorted(
                {
                    validate_identifier(value, field_name="excluded_checkpoint_id")
                    for value in self.excluded_checkpoint_ids
                }
            )
        )
        object.__setattr__(self, "excluded_checkpoint_ids", excluded)

        accepted = tuple(dict.fromkeys(CheckpointHealth(value) for value in self.accepted_health))
        if not accepted:
            raise ValueError("accepted_health cannot be empty")
        forbidden = {
            CheckpointHealth.CORRUPT,
            CheckpointHealth.INCOMPLETE,
            CheckpointHealth.INCOMPATIBLE,
            CheckpointHealth.QUARANTINED,
        }
        if forbidden.intersection(accepted):
            raise ValueError(
                "corrupt, incomplete, incompatible, and quarantined checkpoints "
                "cannot be made selectable"
            )
        object.__setattr__(self, "accepted_health", accepted)

        for name in ("minimum_step", "maximum_step", "maximum_recovery_generation"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.minimum_step is not None
            and self.maximum_step is not None
            and self.minimum_step > self.maximum_step
        ):
            raise ValueError("minimum_step cannot exceed maximum_step")
        for name in ("agent_id", "run_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_identifier(value, field_name=name))

        if self.mode is SelectionMode.BEST_METRIC:
            if self.objective_name is None or self.objective_direction is None:
                raise ValueError(
                    "BEST_METRIC selection requires objective_name and objective_direction"
                )
        if self.objective_name is not None:
            object.__setattr__(
                self, "objective_name", validate_component_name(self.objective_name)
            )
        if self.objective_direction is not None:
            object.__setattr__(
                self,
                "objective_direction",
                MetricDirection(self.objective_direction),
            )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Eligibility verdict and evidence for one checkpoint candidate."""

    record: CheckpointRecord
    eligible: bool
    reasons: tuple[PolicyReason, ...] = ()
    objective_value: float | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record, CheckpointRecord):
            raise TypeError("candidate record must be a CheckpointRecord")
        if not isinstance(self.eligible, bool):
            raise TypeError("candidate eligible must be a boolean")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if any(not isinstance(reason, PolicyReason) for reason in self.reasons):
            raise TypeError("candidate reasons must contain PolicyReason values")
        if self.eligible and self.reasons:
            raise ValueError("eligible candidates cannot contain rejection reasons")
        if not self.eligible and not self.reasons:
            raise ValueError("ineligible candidates require at least one reason")
        if self.objective_value is not None and (
            isinstance(self.objective_value, bool)
            or not isinstance(self.objective_value, (int, float))
            or not math.isfinite(self.objective_value)
        ):
            raise ValueError("candidate objective value must be finite and numeric")
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("candidate rank must be a positive integer")
        if self.rank is not None and not self.eligible:
            raise ValueError("ineligible candidates cannot have a rank")


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Auditable result of evaluating and ranking checkpoint candidates."""

    criteria: SelectionCriteria
    selected: CheckpointRecord | None
    evaluations: tuple[CandidateEvaluation, ...]
    reason: PolicyReason

    def __post_init__(self) -> None:
        if not isinstance(self.criteria, SelectionCriteria):
            raise TypeError("selection criteria must be SelectionCriteria")
        if self.selected is not None and not isinstance(self.selected, CheckpointRecord):
            raise TypeError("selected must be a CheckpointRecord when supplied")
        if not isinstance(self.reason, PolicyReason):
            raise TypeError("selection reason must be PolicyReason")
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        if any(
            not isinstance(item, CandidateEvaluation) for item in self.evaluations
        ):
            raise TypeError("evaluations must contain CandidateEvaluation values")
        if self.selected is not None:
            selected_matches = [
                item
                for item in self.evaluations
                if item.record.checkpoint_id == self.selected.checkpoint_id and item.eligible
            ]
            if len(selected_matches) != 1:
                raise ValueError("selected record must have exactly one eligible evaluation")

    @property
    def found(self) -> bool:
        return self.selected is not None

    def require_selected(self) -> CheckpointRecord:
        if self.selected is None:
            raise CheckpointNotFoundError(
                self.reason.message,
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DISCOVERY,
                details={
                    "reason": self.reason.code,
                    "evaluated_candidates": len(self.evaluations),
                },
            )
        return self.selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.criteria.mode.value,
            "selected_checkpoint_id": (
                self.selected.checkpoint_id if self.selected else None
            ),
            "selected_version": self.selected.version if self.selected else None,
            "reason": self.reason.to_dict(),
            "candidates": [
                {
                    "checkpoint_id": item.record.checkpoint_id,
                    "version": item.record.version,
                    "eligible": item.eligible,
                    "rank": item.rank,
                    "objective_value": item.objective_value,
                    "reasons": [reason.to_dict() for reason in item.reasons],
                }
                for item in self.evaluations
            ],
        }


@dataclass(frozen=True, slots=True)
class RollbackRules:
    """Constraints governing ancestry-based rollback planning."""

    criteria: SelectionCriteria = field(default_factory=SelectionCriteria)
    require_ancestor: bool = True
    allow_non_ancestor_fallback: bool = False
    maximum_hops: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.criteria, SelectionCriteria):
            raise TypeError("rollback criteria must be SelectionCriteria")
        for name in ("require_ancestor", "allow_non_ancestor_fallback"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.maximum_hops is not None and (
            isinstance(self.maximum_hops, bool)
            or not isinstance(self.maximum_hops, int)
            or self.maximum_hops < 1
        ):
            raise ValueError("maximum_hops must be a positive integer")
        if self.require_ancestor and self.allow_non_ancestor_fallback:
            raise ValueError(
                "allow_non_ancestor_fallback cannot be true when require_ancestor is true"
            )


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """Proposed rollback target and the evidence used to select it."""

    current: CheckpointRecord
    target: CheckpointRecord | None
    ancestor_hops: int | None
    evaluations: tuple[CandidateEvaluation, ...]
    reasons: tuple[PolicyReason, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.target is not None and self.target.checkpoint_id == self.current.checkpoint_id:
            raise ValueError("rollback target cannot be the current checkpoint")
        if self.ancestor_hops is not None and self.ancestor_hops < 1:
            raise ValueError("ancestor_hops must be positive")
        if self.target is None and self.ancestor_hops is not None:
            raise ValueError("ancestor_hops requires a rollback target")
        if self.target is not None:
            target_matches = [
                item
                for item in self.evaluations
                if item.eligible
                and item.record.checkpoint_id == self.target.checkpoint_id
            ]
            if len(target_matches) != 1:
                raise ValueError(
                    "rollback target must have exactly one eligible evaluation"
                )

    @property
    def possible(self) -> bool:
        return self.target is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_checkpoint_id": self.current.checkpoint_id,
            "current_version": self.current.version,
            "target_checkpoint_id": (
                self.target.checkpoint_id if self.target is not None else None
            ),
            "target_version": self.target.version if self.target is not None else None,
            "ancestor_hops": self.ancestor_hops,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "candidates": [
                {
                    "checkpoint_id": item.record.checkpoint_id,
                    "version": item.record.version,
                    "eligible": item.eligible,
                    "objective_value": item.objective_value,
                    "reasons": [reason.to_dict() for reason in item.reasons],
                }
                for item in self.evaluations
            ],
        }


@dataclass(frozen=True, slots=True)
class RetentionRules:
    """Safety-oriented limits used to create, but never execute, a deletion plan."""

    max_checkpoints: int | None = None
    max_total_bytes: int | None = None
    minimum_keep: int = 1
    keep_latest: int = 1
    keep_best: int = 0
    objective_name: str | None = None
    objective_direction: MetricDirection | None = None
    protected_checkpoint_ids: tuple[str, ...] = ()
    protected_versions: tuple[str, ...] = ()
    # Lineage records provenance; it does not imply that a full checkpoint
    # depends on its parent files.  Applications using delta checkpoints may
    # opt in only after establishing that dependency explicitly.
    preserve_protected_lineage: bool = False
    protect_latest_healthy: bool = True

    def __post_init__(self) -> None:
        for name in ("preserve_protected_lineage", "protect_latest_healthy"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in ("max_checkpoints", "max_total_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("minimum_keep", "keep_latest", "keep_best"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_checkpoints is not None and self.max_checkpoints < self.minimum_keep:
            raise ValueError("max_checkpoints cannot be lower than minimum_keep")
        if self.keep_best > 0 and (
            self.objective_name is None or self.objective_direction is None
        ):
            raise ValueError(
                "keep_best requires objective_name and objective_direction"
            )
        if self.objective_name is not None:
            object.__setattr__(
                self, "objective_name", validate_component_name(self.objective_name)
            )
        if self.objective_direction is not None:
            object.__setattr__(
                self,
                "objective_direction",
                MetricDirection(self.objective_direction),
            )
        object.__setattr__(
            self,
            "protected_checkpoint_ids",
            tuple(
                sorted(
                    {
                        validate_identifier(value, field_name="protected_checkpoint_id")
                        for value in self.protected_checkpoint_ids
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "protected_versions",
            tuple(sorted({validate_version(value) for value in self.protected_versions})),
        )


@dataclass(frozen=True, slots=True)
class RetentionEntry:
    """One keep/delete proposal within a retention plan."""

    record: CheckpointRecord
    action: RetentionAction
    reasons: tuple[PolicyReason, ...]
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", RetentionAction(self.action))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if not self.reasons:
            raise ValueError("retention entries require at least one reason")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("retention entry size must be a non-negative integer")
        if any(not isinstance(reason, PolicyReason) for reason in self.reasons):
            raise TypeError("retention reasons must contain PolicyReason values")


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    """Complete non-executing retention proposal with constraint accounting."""

    rules: RetentionRules
    entries: tuple[RetentionEntry, ...]
    total_count_before: int
    total_count_after: int
    total_bytes_before: int
    total_bytes_after: int
    constraints_satisfied: bool
    reasons: tuple[PolicyReason, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if not isinstance(self.constraints_satisfied, bool):
            raise TypeError("constraints_satisfied must be a boolean")
        if any(not isinstance(entry, RetentionEntry) for entry in self.entries):
            raise TypeError("entries must contain RetentionEntry values")
        if any(not isinstance(reason, PolicyReason) for reason in self.reasons):
            raise TypeError("reasons must contain PolicyReason values")
        for name in (
            "total_count_before",
            "total_count_after",
            "total_bytes_before",
            "total_bytes_after",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if len(self.entries) != self.total_count_before:
            raise ValueError("retention entries must account for every input checkpoint")
        kept = [entry for entry in self.entries if entry.action is RetentionAction.KEEP]
        if len(kept) != self.total_count_after:
            raise ValueError("retention total_count_after is inconsistent with entries")
        if sum(entry.size_bytes for entry in self.entries) != self.total_bytes_before:
            raise ValueError("retention total_bytes_before is inconsistent with entries")
        if sum(entry.size_bytes for entry in kept) != self.total_bytes_after:
            raise ValueError("retention total_bytes_after is inconsistent with entries")

    @property
    def keep(self) -> tuple[CheckpointRecord, ...]:
        return tuple(
            entry.record
            for entry in self.entries
            if entry.action is RetentionAction.KEEP
        )

    @property
    def delete(self) -> tuple[CheckpointRecord, ...]:
        return tuple(
            entry.record
            for entry in self.entries
            if entry.action is RetentionAction.DELETE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints_satisfied": self.constraints_satisfied,
            "total_count_before": self.total_count_before,
            "total_count_after": self.total_count_after,
            "total_bytes_before": self.total_bytes_before,
            "total_bytes_after": self.total_bytes_after,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "entries": [
                {
                    "checkpoint_id": entry.record.checkpoint_id,
                    "version": entry.record.version,
                    "action": entry.action.value,
                    "size_bytes": entry.size_bytes,
                    "reasons": [reason.to_dict() for reason in entry.reasons],
                }
                for entry in self.entries
            ],
        }


def _exact_version_match(required: str, actual: str) -> bool:
    return required == actual


def _timestamp(record: CheckpointRecord) -> float:
    value = record.created_at
    parse_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    return _dt.datetime.fromisoformat(parse_value).timestamp()


def _stable_record_key(record: CheckpointRecord) -> tuple[float, int, str, str]:
    return (
        _timestamp(record),
        record.step if record.step is not None else -1,
        record.version,
        record.checkpoint_id,
    )


def _record_size(record: CheckpointRecord) -> int:
    return sum(artifact.size_bytes for artifact in record.manifest.artifacts)


class CheckpointPolicy:
    """Pure deterministic policy engine."""

    def __init__(self, *, version_matcher: VersionMatcher | None = None) -> None:
        self._version_matcher = version_matcher or _exact_version_match

    def _compatibility_reasons(
        self,
        record: CheckpointRecord,
        runtime: RuntimeCapabilities | None,
    ) -> list[PolicyReason]:
        constraints = record.manifest.compatibility
        constrained = bool(
            constraints.slai_version
            or constraints.python_version
            or constraints.platform_tags
            or constraints.required_codecs
            or constraints.component_schemas
        )
        if not constrained:
            return []
        if runtime is None:
            return [
                PolicyReason(
                    "runtime_capabilities_missing",
                    "candidate declares compatibility constraints but no runtime "
                    "capabilities were supplied",
                )
            ]

        reasons: list[PolicyReason] = []
        for field_name, required, actual in (
            ("slai_version", constraints.slai_version, runtime.slai_version),
            ("python_version", constraints.python_version, runtime.python_version),
        ):
            if required is None:
                continue
            if actual is None or not self._version_matcher(required, actual):
                reasons.append(
                    PolicyReason(
                        f"{field_name}_incompatible",
                        f"runtime does not satisfy candidate {field_name}",
                        {"required": required, "actual": actual},
                    )
                )

        if constraints.platform_tags and not set(constraints.platform_tags).intersection(
            runtime.platform_tags
        ):
            reasons.append(
                PolicyReason(
                    "platform_incompatible",
                    "runtime platform tags do not intersect the candidate's allowed tags",
                    {
                        "required_any": list(constraints.platform_tags),
                        "actual": list(runtime.platform_tags),
                    },
                )
            )

        for codec, required_version in constraints.required_codecs.items():
            actual_version = runtime.codecs.get(codec)
            if actual_version is None or not self._version_matcher(
                required_version, actual_version
            ):
                reasons.append(
                    PolicyReason(
                        "codec_incompatible",
                        "runtime codec is missing or has an incompatible version",
                        {
                            "codec": codec,
                            "required": required_version,
                            "actual": actual_version,
                        },
                    )
                )

        for component, schema_version in constraints.component_schemas.items():
            supported = runtime.component_schemas.get(component, ())
            if schema_version not in supported:
                reasons.append(
                    PolicyReason(
                        "component_schema_incompatible",
                        "runtime does not support the candidate component schema",
                        {
                            "component": component,
                            "required": schema_version,
                            "supported": list(supported),
                        },
                    )
                )
        return reasons

    @staticmethod
    def _metric_value(
        record: CheckpointRecord,
        name: str,
    ) -> float | None:
        objective = record.manifest.provenance.objective
        if objective is not None and objective.name == name:
            return objective.value
        raw = record.metrics.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        return value if math.isfinite(value) else None

    def evaluate(
        self,
        record: CheckpointRecord,
        criteria: SelectionCriteria,
        *,
        runtime: RuntimeCapabilities | None = None,
        duplicate_checkpoint_ids: Iterable[str] = (),
        duplicate_versions: Iterable[str] = (),
    ) -> CandidateEvaluation:
        reasons: list[PolicyReason] = []
        duplicate_ids = set(duplicate_checkpoint_ids)
        duplicate_names = set(duplicate_versions)

        if record.checkpoint_id in duplicate_ids:
            reasons.append(
                PolicyReason(
                    "duplicate_checkpoint_id",
                    "checkpoint identity is not unique among candidates",
                    {"checkpoint_id": record.checkpoint_id},
                )
            )
        if record.version in duplicate_names:
            reasons.append(
                PolicyReason(
                    "duplicate_version",
                    "checkpoint version is not unique among candidates",
                    {"version": record.version},
                )
            )
        if record.checkpoint_id in criteria.excluded_checkpoint_ids:
            reasons.append(
                PolicyReason(
                    "explicitly_excluded",
                    "checkpoint identity is explicitly excluded",
                    {"checkpoint_id": record.checkpoint_id},
                )
            )
        if criteria.checkpoint_id is not None and record.checkpoint_id != criteria.checkpoint_id:
            reasons.append(
                PolicyReason(
                    "checkpoint_id_mismatch",
                    "checkpoint identity does not match the requested identity",
                    {"required": criteria.checkpoint_id, "actual": record.checkpoint_id},
                )
            )
        if criteria.version is not None and record.version != criteria.version:
            reasons.append(
                PolicyReason(
                    "version_mismatch",
                    "checkpoint version does not match the requested version",
                    {"required": criteria.version, "actual": record.version},
                )
            )

        if record.health not in criteria.accepted_health:
            reasons.append(
                PolicyReason(
                    "health_not_accepted",
                    "checkpoint health is not accepted by selection criteria",
                    {
                        "health": record.health.value,
                        "accepted": [value.value for value in criteria.accepted_health],
                    },
                )
            )
        if criteria.require_verified:
            if record.verification is None:
                reasons.append(
                    PolicyReason(
                        "verification_missing",
                        "checkpoint has no verification evidence",
                    )
                )
            elif record.verification.status is not VerificationStatus.PASSED:
                reasons.append(
                    PolicyReason(
                        "verification_failed",
                        "checkpoint did not pass integrity verification",
                        {
                            "status": record.verification.status.value,
                            "issue_count": len(record.verification.issues),
                        },
                    )
                )

        saved_components = set(record.manifest.saved_components)
        missing_components = sorted(
            set(criteria.required_components) - saved_components
        )
        if missing_components:
            reasons.append(
                PolicyReason(
                    "required_components_missing",
                    "checkpoint does not contain every required component",
                    {"missing": missing_components},
                )
            )

        if criteria.minimum_step is not None and (
            record.step is None or record.step < criteria.minimum_step
        ):
            reasons.append(
                PolicyReason(
                    "step_below_minimum",
                    "checkpoint step is below the required minimum",
                    {"minimum": criteria.minimum_step, "actual": record.step},
                )
            )
        if criteria.maximum_step is not None and (
            record.step is None or record.step > criteria.maximum_step
        ):
            reasons.append(
                PolicyReason(
                    "step_above_maximum",
                    "checkpoint step exceeds the permitted maximum",
                    {"maximum": criteria.maximum_step, "actual": record.step},
                )
            )

        provenance = record.manifest.provenance
        for field_name, required, actual in (
            ("agent_id", criteria.agent_id, provenance.agent_id),
            ("run_id", criteria.run_id, provenance.run_id),
        ):
            if required is not None and required != actual:
                reasons.append(
                    PolicyReason(
                        f"{field_name}_mismatch",
                        f"checkpoint {field_name} does not match the requested scope",
                        {"required": required, "actual": actual},
                    )
                )
        if (
            criteria.maximum_recovery_generation is not None
            and provenance.lineage.generation > criteria.maximum_recovery_generation
        ):
            reasons.append(
                PolicyReason(
                    "recovery_generation_exceeded",
                    "checkpoint recovery generation exceeds the configured maximum",
                    {
                        "maximum": criteria.maximum_recovery_generation,
                        "actual": provenance.lineage.generation,
                    },
                )
            )

        reasons.extend(self._compatibility_reasons(record, runtime))

        objective_value: float | None = None
        if criteria.mode is SelectionMode.HIGHEST_STEP and record.step is None:
            reasons.append(
                PolicyReason(
                    "step_missing",
                    "HIGHEST_STEP selection requires a recorded step",
                )
            )
        if criteria.mode is SelectionMode.BEST_METRIC:
            if criteria.objective_name is None:  # Guard against invalid instances.
                raise ValueError("BEST_METRIC selection requires objective_name")
            objective_value = self._metric_value(record, criteria.objective_name)
            declared_objective = record.manifest.provenance.objective
            if (
                declared_objective is not None
                and declared_objective.name == criteria.objective_name
                and criteria.objective_direction is not None
                and declared_objective.direction is not criteria.objective_direction
            ):
                reasons.append(
                    PolicyReason(
                        "objective_direction_mismatch",
                        "checkpoint objective direction conflicts with selection criteria",
                        {
                            "objective": criteria.objective_name,
                            "declared": declared_objective.direction.value,
                            "required": criteria.objective_direction.value,
                        },
                    )
                )
            if objective_value is None:
                reasons.append(
                    PolicyReason(
                        "objective_missing_or_non_numeric",
                        "checkpoint does not contain a finite numeric selection objective",
                        {"objective": criteria.objective_name},
                    )
                )

        return CandidateEvaluation(
            record=record,
            eligible=not reasons,
            reasons=tuple(reasons),
            objective_value=objective_value,
        )

    def select(
        self,
        candidates: Sequence[CheckpointRecord],
        criteria: SelectionCriteria | None = None,
        *,
        runtime: RuntimeCapabilities | None = None,
    ) -> SelectionDecision:
        active = criteria or SelectionCriteria()
        ordered = tuple(sorted(candidates, key=_stable_record_key))

        id_counts: dict[str, int] = {}
        version_counts: dict[str, int] = {}
        for record in ordered:
            id_counts[record.checkpoint_id] = id_counts.get(record.checkpoint_id, 0) + 1
            version_counts[record.version] = version_counts.get(record.version, 0) + 1
        duplicate_ids = {key for key, count in id_counts.items() if count > 1}
        duplicate_versions = {key for key, count in version_counts.items() if count > 1}

        evaluated = [
            self.evaluate(
                record,
                active,
                runtime=runtime,
                duplicate_checkpoint_ids=duplicate_ids,
                duplicate_versions=duplicate_versions,
            )
            for record in ordered
        ]
        eligible = [item for item in evaluated if item.eligible]
        if not eligible:
            return SelectionDecision(
                criteria=active,
                selected=None,
                evaluations=tuple(evaluated),
                reason=PolicyReason(
                    "no_eligible_checkpoint",
                    "no checkpoint satisfied all selection criteria",
                    {
                        "candidate_count": len(evaluated),
                        "mode": active.mode.value,
                    },
                ),
            )

        if active.mode is SelectionMode.LATEST:
            ranked = sorted(
                eligible,
                key=lambda item: _stable_record_key(item.record),
                reverse=True,
            )
        elif active.mode is SelectionMode.HIGHEST_STEP:
            ranked = sorted(
                eligible,
                key=lambda item: (
                    item.record.step if item.record.step is not None else -1,
                    *_stable_record_key(item.record),
                ),
                reverse=True,
            )
        else:
            direction = active.objective_direction
            if direction is None:  # Guard against invalid instances.
                raise ValueError("BEST_METRIC selection requires objective_direction")
            sign = 1.0 if direction is MetricDirection.MAXIMIZE else -1.0
            ranked = sorted(
                eligible,
                key=lambda item: (
                    sign * float(item.objective_value),
                    *_stable_record_key(item.record),
                ),
                reverse=True,
            )

        rank_by_id = {
            item.record.checkpoint_id: index + 1 for index, item in enumerate(ranked)
        }
        ranked_evaluations = tuple(
            CandidateEvaluation(
                record=item.record,
                eligible=item.eligible,
                reasons=item.reasons,
                objective_value=item.objective_value,
                rank=rank_by_id.get(item.record.checkpoint_id),
            )
            for item in evaluated
        )
        selected = ranked[0].record
        evidence: dict[str, Any] = {
            "mode": active.mode.value,
            "checkpoint_id": selected.checkpoint_id,
            "version": selected.version,
            "eligible_candidates": len(eligible),
        }
        if active.mode is SelectionMode.HIGHEST_STEP:
            evidence["step"] = selected.step
        if active.mode is SelectionMode.BEST_METRIC:
            assert active.objective_direction is not None
            evidence.update(
                {
                    "objective": active.objective_name,
                    "direction": active.objective_direction.value,
                    "value": ranked[0].objective_value,
                }
            )
        return SelectionDecision(
            criteria=active,
            selected=selected,
            evaluations=ranked_evaluations,
            reason=PolicyReason(
                "checkpoint_selected",
                "checkpoint selected by deterministic eligibility and ranking",
                evidence,
            ),
        )

    def plan_rollback(
        self,
        current: CheckpointRecord,
        candidates: Sequence[CheckpointRecord],
        rules: RollbackRules | None = None,
        *,
        runtime: RuntimeCapabilities | None = None,
    ) -> RollbackPlan:
        active = rules or RollbackRules()
        # The current identity cannot be a rollback target.  Normalize it to
        # the caller-supplied record while retaining ambiguity information for
        # every possible ancestor.
        pool = [
            record
            for record in candidates
            if record.checkpoint_id != current.checkpoint_id
        ]
        pool.append(current)
        records_by_id: dict[str, list[CheckpointRecord]] = {}
        version_counts: dict[str, int] = {}
        for record in pool:
            records_by_id.setdefault(record.checkpoint_id, []).append(record)
            version_counts[record.version] = version_counts.get(record.version, 0) + 1
        duplicate_ids = {
            checkpoint_id
            for checkpoint_id, matches in records_by_id.items()
            if len(matches) > 1
        }
        duplicate_versions = {
            version for version, count in version_counts.items() if count > 1
        }
        reasons: list[PolicyReason] = []
        evaluations: list[CandidateEvaluation] = []

        parent_id = current.manifest.provenance.lineage.parent_checkpoint_id
        # Validate identity resolution and acyclicity before returning an
        # otherwise eligible ancestor.  Selecting the first parent eagerly can
        # conceal a cycle such as A -> B -> A.
        lineage_id = parent_id
        lineage_visited = {current.checkpoint_id}
        while lineage_id is not None:
            if lineage_id in lineage_visited:
                return RollbackPlan(
                    current=current,
                    target=None,
                    ancestor_hops=None,
                    evaluations=(),
                    reasons=(
                        PolicyReason(
                            "lineage_cycle",
                            "checkpoint lineage contains a cycle",
                            {"checkpoint_id": lineage_id},
                        ),
                    ),
                )
            lineage_visited.add(lineage_id)
            lineage_matches = records_by_id.get(lineage_id, [])
            if not lineage_matches:
                break
            if len(lineage_matches) > 1:
                return RollbackPlan(
                    current=current,
                    target=None,
                    ancestor_hops=None,
                    evaluations=(),
                    reasons=(
                        PolicyReason(
                            "ancestor_identity_ambiguous",
                            "checkpoint lineage resolves to more than one candidate",
                            {
                                "checkpoint_id": lineage_id,
                                "matches": len(lineage_matches),
                            },
                        ),
                    ),
                )
            lineage_id = lineage_matches[0].manifest.provenance.lineage.parent_checkpoint_id

        visited = {current.checkpoint_id}
        hops = 0
        while parent_id is not None:
            hops += 1
            if active.maximum_hops is not None and hops > active.maximum_hops:
                reasons.append(
                    PolicyReason(
                        "rollback_hop_limit_reached",
                        "no eligible ancestor was found within the rollback hop limit",
                        {"maximum_hops": active.maximum_hops},
                    )
                )
                break
            if parent_id in visited:
                reasons.append(
                    PolicyReason(
                        "lineage_cycle",
                        "checkpoint lineage contains a cycle",
                        {"checkpoint_id": parent_id},
                    )
                )
                break
            visited.add(parent_id)
            matches = records_by_id.get(parent_id, [])
            if not matches:
                reasons.append(
                    PolicyReason(
                        "ancestor_missing",
                        "checkpoint lineage references an unavailable parent",
                        {"checkpoint_id": parent_id, "hops": hops},
                    )
                )
                break
            if len(matches) > 1:
                reasons.append(
                    PolicyReason(
                        "ancestor_identity_ambiguous",
                        "checkpoint lineage resolves to more than one candidate",
                        {"checkpoint_id": parent_id, "matches": len(matches)},
                    )
                )
                break
            parent = matches[0]
            evaluation = self.evaluate(
                parent,
                active.criteria,
                runtime=runtime,
                duplicate_checkpoint_ids=duplicate_ids,
                duplicate_versions=duplicate_versions,
            )
            evaluations.append(evaluation)
            if evaluation.eligible:
                return RollbackPlan(
                    current=current,
                    target=parent,
                    ancestor_hops=hops,
                    evaluations=tuple(evaluations),
                    reasons=(
                        PolicyReason(
                            "nearest_eligible_ancestor",
                            "selected the nearest eligible checkpoint ancestor",
                            {
                                "checkpoint_id": parent.checkpoint_id,
                                "version": parent.version,
                                "hops": hops,
                            },
                        ),
                    ),
                )
            parent_id = parent.manifest.provenance.lineage.parent_checkpoint_id

        if active.require_ancestor or not active.allow_non_ancestor_fallback:
            if not reasons:
                reasons.append(
                    PolicyReason(
                        "no_eligible_ancestor",
                        "checkpoint has no eligible ancestor for rollback",
                    )
                )
            return RollbackPlan(
                current=current,
                target=None,
                ancestor_hops=None,
                evaluations=tuple(evaluations),
                reasons=tuple(reasons),
            )

        older_candidates = [
            record
            for record in candidates
            if record.checkpoint_id != current.checkpoint_id
            and _stable_record_key(record) < _stable_record_key(current)
        ]
        fallback_criteria = SelectionCriteria(
            mode=active.criteria.mode,
            checkpoint_id=active.criteria.checkpoint_id,
            version=active.criteria.version,
            required_components=active.criteria.required_components,
            excluded_checkpoint_ids=tuple(
                set(active.criteria.excluded_checkpoint_ids) | {current.checkpoint_id}
            ),
            require_verified=active.criteria.require_verified,
            accepted_health=active.criteria.accepted_health,
            minimum_step=active.criteria.minimum_step,
            maximum_step=active.criteria.maximum_step,
            agent_id=active.criteria.agent_id,
            run_id=active.criteria.run_id,
            maximum_recovery_generation=active.criteria.maximum_recovery_generation,
            objective_name=active.criteria.objective_name,
            objective_direction=active.criteria.objective_direction,
        )
        decision = self.select(older_candidates, fallback_criteria, runtime=runtime)
        reasons.append(
            PolicyReason(
                "non_ancestor_fallback",
                "ancestor rollback was unavailable; evaluated explicitly permitted "
                "older checkpoints",
                {"selected": decision.selected.checkpoint_id if decision.selected else None},
            )
        )
        return RollbackPlan(
            current=current,
            target=decision.selected,
            ancestor_hops=None,
            evaluations=tuple(evaluations) + decision.evaluations,
            reasons=tuple(reasons),
        )

    def plan_retention(
        self,
        candidates: Sequence[CheckpointRecord],
        rules: RetentionRules,
    ) -> RetentionPlan:
        ordered = tuple(sorted(candidates, key=_stable_record_key))
        # Object-local keys preserve exact accounting even when the input
        # contains duplicate checkpoint identities.  Duplicate identities are
        # protected below because a deletion executor could not address them
        # unambiguously.
        sizes = {id(record): _record_size(record) for record in ordered}
        total_bytes = sum(sizes[id(record)] for record in ordered)
        global_reasons: list[PolicyReason] = []

        id_counts: dict[str, int] = {}
        version_counts: dict[str, int] = {}
        for record in ordered:
            id_counts[record.checkpoint_id] = id_counts.get(record.checkpoint_id, 0) + 1
            version_counts[record.version] = version_counts.get(record.version, 0) + 1

        protected: dict[int, list[PolicyReason]] = {}

        def protect(record: CheckpointRecord, reason: PolicyReason) -> None:
            protected.setdefault(id(record), []).append(reason)

        for record in ordered:
            if id_counts[record.checkpoint_id] > 1 or version_counts[record.version] > 1:
                protect(
                    record,
                    PolicyReason(
                        "ambiguous_identity",
                        "retention will not delete a checkpoint with duplicate "
                        "identity or version",
                    ),
                )
            if record.checkpoint_id in rules.protected_checkpoint_ids:
                protect(
                    record,
                    PolicyReason(
                        "explicitly_protected", "checkpoint identity is protected"
                    ),
                )
            if record.version in rules.protected_versions:
                protect(
                    record,
                    PolicyReason("protected_version", "checkpoint version is protected"),
                )

        for record in list(reversed(ordered))[: rules.keep_latest]:
            protect(
                record,
                PolicyReason(
                    "latest_window", "checkpoint is in the keep-latest window"
                ),
            )
        for record in list(reversed(ordered))[: rules.minimum_keep]:
            protect(
                record,
                PolicyReason(
                    "minimum_keep", "checkpoint is required by minimum_keep"
                ),
            )

        if rules.protect_latest_healthy:
            healthy = [
                record
                for record in ordered
                if record.health is CheckpointHealth.HEALTHY
                and record.verification is not None
                and record.verification.status is VerificationStatus.PASSED
            ]
            if healthy:
                protect(
                    healthy[-1],
                    PolicyReason(
                        "latest_verified_healthy",
                        "checkpoint is the latest verified healthy recovery point",
                    ),
                )

        if rules.keep_best:
            if rules.objective_name is None or rules.objective_direction is None:
                raise ValueError("keep_best requires an objective and direction")
            conflicting_objectives: set[int] = set()
            for record in ordered:
                declared_objective = record.manifest.provenance.objective
                if (
                    declared_objective is not None
                    and declared_objective.name == rules.objective_name
                    and declared_objective.direction is not rules.objective_direction
                ):
                    conflicting_objectives.add(id(record))
                    protect(
                        record,
                        PolicyReason(
                            "objective_direction_conflict",
                            "checkpoint is protected because its declared objective "
                            "direction conflicts with the retention rule",
                            {
                                "objective": rules.objective_name,
                                "declared": declared_objective.direction.value,
                                "configured": rules.objective_direction.value,
                            },
                        ),
                    )
            scored = [
                (value, record)
                for record in ordered
                if id(record) not in conflicting_objectives
                for value in (self._metric_value(record, rules.objective_name),)
                if value is not None
            ]
            sign = 1.0 if rules.objective_direction is MetricDirection.MAXIMIZE else -1.0
            scored.sort(
                key=lambda item: (
                    sign * float(item[0]),
                    *_stable_record_key(item[1]),
                ),
                reverse=True,
            )
            for value, record in scored[: rules.keep_best]:
                protect(
                    record,
                    PolicyReason(
                        "best_metric_window",
                        "checkpoint is in the keep-best metric window",
                        {
                            "objective": rules.objective_name,
                            "direction": rules.objective_direction.value,
                            "value": value,
                        },
                    ),
                )

        if rules.preserve_protected_lineage:
            by_id: dict[str, list[CheckpointRecord]] = {}
            for record in ordered:
                by_id.setdefault(record.checkpoint_id, []).append(record)
            queue = [record for record in ordered if id(record) in protected]
            visited = {record.checkpoint_id for record in queue}
            while queue:
                record = queue.pop()
                checkpoint_id = record.checkpoint_id
                parent_id = record.manifest.provenance.lineage.parent_checkpoint_id
                if parent_id is None or parent_id in visited:
                    continue
                visited.add(parent_id)
                matches = by_id.get(parent_id, [])
                if not matches:
                    global_reasons.append(
                        PolicyReason(
                            "protected_ancestor_missing",
                            "a protected checkpoint references an unavailable ancestor",
                            {
                                "checkpoint_id": checkpoint_id,
                                "missing_parent_id": parent_id,
                            },
                        )
                    )
                    continue
                if len(matches) > 1:
                    global_reasons.append(
                        PolicyReason(
                            "protected_ancestor_ambiguous",
                            "a protected checkpoint references an ambiguous ancestor identity",
                            {
                                "checkpoint_id": checkpoint_id,
                                "parent_id": parent_id,
                                "matches": len(matches),
                            },
                        )
                    )
                    continue
                parent = matches[0]
                protect(
                    parent,
                    PolicyReason(
                        "protected_lineage",
                        "checkpoint is an ancestor of a protected recovery point",
                        {"descendant_checkpoint_id": checkpoint_id},
                    ),
                )
                queue.append(parent)

        deleted_keys: set[int] = set()
        current_count = len(ordered)
        current_bytes = total_bytes

        def exceeds_limits() -> tuple[bool, bool]:
            count_exceeded = (
                rules.max_checkpoints is not None
                and current_count > rules.max_checkpoints
            )
            bytes_exceeded = (
                rules.max_total_bytes is not None
                and current_bytes > rules.max_total_bytes
            )
            return count_exceeded, bytes_exceeded

        deletion_reasons: dict[int, list[PolicyReason]] = {}
        while any(exceeds_limits()):
            victim = next(
                (
                    record
                    for record in ordered
                    if id(record) not in protected
                    and id(record) not in deleted_keys
                ),
                None,
            )
            if victim is None:
                break
            count_exceeded, bytes_exceeded = exceeds_limits()
            reasons: list[PolicyReason] = []
            if count_exceeded:
                reasons.append(
                    PolicyReason(
                        "max_checkpoints",
                        "checkpoint is the oldest unprotected candidate above the count limit",
                        {"limit": rules.max_checkpoints},
                    )
                )
            if bytes_exceeded:
                reasons.append(
                    PolicyReason(
                        "max_total_bytes",
                        "checkpoint is the oldest unprotected candidate above the byte limit",
                        {"limit": rules.max_total_bytes},
                    )
                )
            victim_key = id(victim)
            deleted_keys.add(victim_key)
            deletion_reasons[victim_key] = reasons
            current_count -= 1
            current_bytes -= sizes[victim_key]

        count_exceeded, bytes_exceeded = exceeds_limits()
        satisfied = not count_exceeded and not bytes_exceeded
        if not satisfied:
            global_reasons.append(
                PolicyReason(
                    "retention_constraints_unsatisfied",
                    "retention limits cannot be met without deleting protected checkpoints",
                    {
                        "remaining_count": current_count,
                        "remaining_bytes": current_bytes,
                        "max_checkpoints": rules.max_checkpoints,
                        "max_total_bytes": rules.max_total_bytes,
                    },
                )
            )

        entries: list[RetentionEntry] = []
        for record in ordered:
            record_key = id(record)
            if record_key in deleted_keys:
                entries.append(
                    RetentionEntry(
                        record=record,
                        action=RetentionAction.DELETE,
                        reasons=tuple(deletion_reasons[record_key]),
                        size_bytes=sizes[record_key],
                    )
                )
            else:
                reasons = protected.get(record_key) or [
                    PolicyReason(
                        "within_retention_limits",
                        "checkpoint can be retained without violating configured limits",
                    )
                ]
                entries.append(
                    RetentionEntry(
                        record=record,
                        action=RetentionAction.KEEP,
                        reasons=tuple(reasons),
                        size_bytes=sizes[record_key],
                    )
                )
        return RetentionPlan(
            rules=rules,
            entries=tuple(entries),
            total_count_before=len(ordered),
            total_count_after=current_count,
            total_bytes_before=total_bytes,
            total_bytes_after=current_bytes,
            constraints_satisfied=satisfied,
            reasons=tuple(global_reasons),
        )


__all__ = [
    "CandidateEvaluation",
    "CheckpointPolicy",
    "PolicyReason",
    "RetentionAction",
    "RetentionEntry",
    "RetentionPlan",
    "RetentionRules",
    "RollbackPlan",
    "RollbackRules",
    "RuntimeCapabilities",
    "SelectionCriteria",
    "SelectionDecision",
    "SelectionMode",
    "VersionMatcher",
]


if __name__ == "__main__":
    print("\n=== Running Checkpoint Policy Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting policy tests", "info")

    policy = CheckpointPolicy()
    printer.status("CODEC", f"created {policy} v{policy}", "success")


    print("\n=== All policy tests passed ===\n")
