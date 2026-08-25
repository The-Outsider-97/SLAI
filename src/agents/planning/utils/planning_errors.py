"""
Planning Errors – Structured exception hierarchy for the Planning Agent subsystem.

Every exception carries structured metadata so callers, loggers, and recovery
handlers can make decisions without parsing message strings. The hierarchy is:

    PlanningError (base)
    ├── PlanningStateError          – world / goal state problems
    │   ├── PreconditionViolation
    │   ├── PostconditionViolation
    │   └── GoalUnreachableError
    ├── PlanningStructureError      – plan / task structure problems
    │   ├── AcademicPlanningError
    │   ├── CyclicDependencyError
    │   ├── DecompositionError
    │   └── MethodSelectionError
    ├── TemporalError               – time / deadline / ordering problems
    │   ├── TemporalViolation
    │   ├── DeadlineExceededError
    │   └── SchedulingConflictError
    ├── ResourceError               – resource availability / safety problems
    │   ├── ResourceViolation
    │   │   ├── ResourceAcquisitionError
    │   │   └── SafetyMarginError
    │   └── ResourceLeakError
    ├── ExecutionError              – runtime execution problems
    │   ├── AdjustmentError
    │   ├── ReplanningError
    │   └── TaskTimeoutError
    └── PlanningConfigError         – configuration / initialisation problems
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_timestamp() -> float:
    """Return the current UTC time as a POSIX timestamp."""
    return time.time()


def _fmt_ts(ts: float) -> str:
    """Format a POSIX timestamp as an ISO-8601 string (UTC, no tzinfo suffix)."""
    import datetime
    return datetime.datetime.utcfromtimestamp(ts).isoformat(timespec="seconds") + "Z"


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class PlanningError(Exception):
    """
    Base class for all Planning Agent exceptions.

    Every subclass must call ``super().__init__(message, **kwargs)`` to ensure
    the shared metadata contract is satisfied.

    Shared attributes
    -----------------
    message : str
        Human-readable description of what went wrong.
    timestamp : float
        POSIX timestamp of when the exception was created.
    context : Dict[str, Any]
        Arbitrary extra key-value pairs attached by the raise site.
    recovery_hints : List[str]
        Ordered list of suggested recovery strategies (most preferred first).
    """

    # Subclasses may override to provide default recovery strategies.
    _default_recovery_hints: List[str] = []

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        recovery_hints: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.timestamp: float = _utc_timestamp()
        self.context: Dict[str, Any] = dict(context or {})
        self.recovery_hints: List[str] = list(
            recovery_hints if recovery_hints is not None else self._default_recovery_hints
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the exception to a JSON-compatible dictionary."""
        return {
            "error_class": type(self).__name__,
            "message": self.message,
            "timestamp_iso": _fmt_ts(self.timestamp),
            "timestamp": self.timestamp,
            "context": self.context,
            "recovery_hints": self.recovery_hints,
            "traceback": traceback.format_exc() or None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the exception to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"timestamp={_fmt_ts(self.timestamp)!r})"
        )

    def __str__(self) -> str:
        return self.message


# ===========================================================================
# PlanningStateError – world / goal state problems
# ===========================================================================

class PlanningStateError(PlanningError):
    """Raised when the world state or goal state is inconsistent or invalid."""

    _default_recovery_hints = [
        "re-evaluate_world_state",
        "trigger_state_resynchronisation",
    ]


class PreconditionViolation(PlanningStateError):
    """
    Raised when a task's preconditions are not satisfied by the current world state.

    Attributes
    ----------
    task_name : str
        Name of the task whose preconditions failed.
    task_id : str
        Unique identifier of the failing task.
    failed_conditions : List[str]
        Human-readable descriptions of each failed precondition.
    world_state_snapshot : Dict[str, Any]
        A snapshot of the world state at the time of the violation.
    """

    _default_recovery_hints = [
        "replan_from_current_state",
        "wait_for_precondition_satisfaction",
        "switch_to_alternative_method",
    ]

    def __init__(
        self,
        message: str,
        *,
        task_name: str = "",
        task_id: str = "",
        failed_conditions: Optional[List[str]] = None,
        world_state_snapshot: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.task_name = task_name
        self.task_id = task_id
        self.failed_conditions: List[str] = list(failed_conditions or [])
        self.world_state_snapshot: Dict[str, Any] = dict(world_state_snapshot or {})

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            task_name=self.task_name,
            task_id=self.task_id,
            failed_conditions=self.failed_conditions,
            world_state_snapshot=self.world_state_snapshot,
        )
        return d


class PostconditionViolation(PlanningStateError):
    """
    Raised when task execution completed but the expected effects were not achieved.

    Attributes
    ----------
    task_name : str
    task_id : str
    expected_state : Dict[str, Any]
        State that was expected after the task.
    actual_state : Dict[str, Any]
        State that was observed after the task.
    divergence_keys : List[str]
        Keys whose values differ between expected and actual state.
    """

    _default_recovery_hints = [
        "rollback_to_last_checkpoint",
        "replan_from_current_state",
        "trigger_state_repair",
    ]

    def __init__(
        self,
        message: str,
        *,
        task_name: str = "",
        task_id: str = "",
        expected_state: Optional[Dict[str, Any]] = None,
        actual_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.task_name = task_name
        self.task_id = task_id
        self.expected_state: Dict[str, Any] = dict(expected_state or {})
        self.actual_state: Dict[str, Any] = dict(actual_state or {})
        self.divergence_keys: List[str] = [
            k for k in set(self.expected_state) | set(self.actual_state)
            if self.expected_state.get(k) != self.actual_state.get(k)
        ]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            task_name=self.task_name,
            task_id=self.task_id,
            expected_state=self.expected_state,
            actual_state=self.actual_state,
            divergence_keys=self.divergence_keys,
        )
        return d


class GoalUnreachableError(PlanningStateError):
    """
    Raised when no valid plan can reach the goal state from the current state.

    Attributes
    ----------
    goal_state : Dict[str, Any]
        The goal that could not be achieved.
    search_depth_reached : int
        Maximum search depth explored before giving up.
    reason : str
        Diagnostic reason (e.g. "no_applicable_methods", "unsatisfiable_constraints").
    """

    _default_recovery_hints = [
        "relax_goal_constraints",
        "request_human_intervention",
        "expand_available_methods",
    ]

    def __init__(
        self,
        message: str,
        *,
        goal_state: Optional[Dict[str, Any]] = None,
        search_depth_reached: int = 0,
        reason: str = "unknown",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.goal_state: Dict[str, Any] = dict(goal_state or {})
        self.search_depth_reached = search_depth_reached
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            goal_state=self.goal_state,
            search_depth_reached=self.search_depth_reached,
            reason=self.reason,
        )
        return d


# ===========================================================================
# PlanningStructureError – plan / task structure problems
# ===========================================================================

class PlanningStructureError(PlanningError):
    """Raised when plan or task structure is invalid."""

    _default_recovery_hints = [
        "validate_plan_structure",
        "rebuild_plan_from_scratch",
    ]


class AcademicPlanningError(PlanningStructureError):
    """
    Raised for type-system and planning-semantics violations.

    Loads additional structured documentation from a JSON metadata file located
    at ``templates/academic_planning_error.json`` relative to the project root.
    The metadata is loaded lazily and cached at the class level.
    """

    _error_metadata_path: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "templates",
        "academic_planning_error.json",
    )
    _metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Lazily load and return the JSON metadata for this error class."""
        if cls._metadata is None:
            try:
                with open(cls._error_metadata_path, "r", encoding="utf-8") as fh:
                    cls._metadata = json.load(fh)
            except Exception as exc:
                cls._metadata = {"error": f"Failed to load metadata: {exc}"}
        # After the above, _metadata is guaranteed to be a dict
        assert cls._metadata is not None
        return cls._metadata

    def __init__(self, message: str = "An academic planning error occurred.", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class CyclicDependencyError(PlanningStructureError):
    """
    Raised when a cyclic dependency is detected in the task graph.

    Attributes
    ----------
    cycle_path : List[str]
        Ordered list of task IDs that form the detected cycle.
    """

    _default_recovery_hints = [
        "remove_cyclic_dependency",
        "reorder_task_dependencies",
        "decompose_into_acyclic_subplans",
    ]

    def __init__(
        self,
        message: str,
        *,
        cycle_path: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.cycle_path: List[str] = list(cycle_path or [])

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["cycle_path"] = self.cycle_path
        return d


class DecompositionError(PlanningStructureError):
    """
    Raised when a task cannot be decomposed into a valid set of subtasks.

    Attributes
    ----------
    task_name : str
    task_id : str
    attempted_methods : List[str]
        Method names that were tried and failed.
    """

    _default_recovery_hints = [
        "try_alternative_decomposition_method",
        "treat_task_as_primitive",
        "request_method_library_update",
    ]

    def __init__(
        self,
        message: str,
        *,
        task_name: str = "",
        task_id: str = "",
        attempted_methods: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.task_name = task_name
        self.task_id = task_id
        self.attempted_methods: List[str] = list(attempted_methods or [])

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            task_name=self.task_name,
            task_id=self.task_id,
            attempted_methods=self.attempted_methods,
        )
        return d


class MethodSelectionError(PlanningStructureError):
    """
    Raised when no suitable method can be selected for a task.

    Attributes
    ----------
    task_name : str
    task_id : str
    candidate_methods : List[str]
        Methods that were considered.
    selection_scores : Dict[str, float]
        Scores assigned to each candidate (if computed).
    """

    _default_recovery_hints = [
        "expand_method_candidates",
        "lower_selection_threshold",
        "fallback_to_default_method",
    ]

    def __init__(
        self,
        message: str,
        *,
        task_name: str = "",
        task_id: str = "",
        candidate_methods: Optional[List[str]] = None,
        selection_scores: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.task_name = task_name
        self.task_id = task_id
        self.candidate_methods: List[str] = list(candidate_methods or [])
        self.selection_scores: Dict[str, float] = dict(selection_scores or {})

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            task_name=self.task_name,
            task_id=self.task_id,
            candidate_methods=self.candidate_methods,
            selection_scores=self.selection_scores,
        )
        return d


# ===========================================================================
# TemporalError – time / deadline / ordering problems
# ===========================================================================

class TemporalError(PlanningError):
    """Base class for all temporal constraint errors."""

    _default_recovery_hints = [
        "reschedule_affected_tasks",
        "compress_non_critical_durations",
    ]


class TemporalViolation(TemporalError):
    """
    Raised when a temporal constraint (ordering, duration, or window) is violated.

    Attributes
    ----------
    violation_type : str
        One of: "ordering", "duration", "window", "dependency_wait".
    task_name : str
    task_id : str
    constraint_details : Dict[str, Any]
        Structured description of the violated constraint.
    time_delta : float
        Signed difference (seconds) between actual and allowed time. Positive
        means late; negative means too early.
    """

    _default_recovery_hints = [
        "reschedule_affected_tasks",
        "extend_deadline_if_possible",
        "trigger_replanning",
    ]

    def __init__(
        self,
        message: str,
        *,
        violation_type: str = "unknown",
        task_name: str = "",
        task_id: str = "",
        constraint_details: Optional[Dict[str, Any]] = None,
        time_delta: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.violation_type = violation_type
        self.task_name = task_name
        self.task_id = task_id
        self.constraint_details: Dict[str, Any] = dict(constraint_details or {})
        self.time_delta = time_delta

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            violation_type=self.violation_type,
            task_name=self.task_name,
            task_id=self.task_id,
            constraint_details=self.constraint_details,
            time_delta=self.time_delta,
        )
        return d


class DeadlineExceededError(TemporalError):
    """
    Raised when a task or plan will not complete before its deadline.

    Attributes
    ----------
    task_name : str
    task_id : str
    deadline : float
        POSIX timestamp of the deadline.
    projected_completion : float
        POSIX timestamp of the projected completion.
    overrun_seconds : float
        How many seconds past the deadline completion is expected.
    """

    _default_recovery_hints = [
        "negotiate_deadline_extension",
        "drop_lower_priority_tasks",
        "parallelise_remaining_work",
        "escalate_to_operator",
    ]

    def __init__(
        self,
        message: str,
        *,
        task_name: str = "",
        task_id: str = "",
        deadline: float = 0.0,
        projected_completion: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.task_name = task_name
        self.task_id = task_id
        self.deadline = deadline
        self.projected_completion = projected_completion
        self.overrun_seconds: float = max(0.0, projected_completion - deadline)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            task_name=self.task_name,
            task_id=self.task_id,
            deadline_iso=_fmt_ts(self.deadline) if self.deadline else None,
            projected_completion_iso=_fmt_ts(self.projected_completion) if self.projected_completion else None,
            overrun_seconds=self.overrun_seconds,
        )
        return d


class SchedulingConflictError(TemporalError):
    """
    Raised when two or more tasks cannot be scheduled without temporal conflict.

    Attributes
    ----------
    conflicting_task_ids : List[str]
        IDs of the tasks in conflict.
    conflict_type : str
        E.g. "resource_overlap", "ordering_violation", "concurrent_exclusion".
    """

    _default_recovery_hints = [
        "serialise_conflicting_tasks",
        "allocate_additional_resources",
        "reorder_task_sequence",
    ]

    def __init__(
        self,
        message: str,
        *,
        conflicting_task_ids: Optional[List[str]] = None,
        conflict_type: str = "unknown",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.conflicting_task_ids: List[str] = list(conflicting_task_ids or [])
        self.conflict_type = conflict_type

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            conflicting_task_ids=self.conflicting_task_ids,
            conflict_type=self.conflict_type,
        )
        return d


# ===========================================================================
# ResourceError – resource availability / safety problems
# ===========================================================================

class ResourceError(PlanningError):
    """Base class for all resource-related errors."""

    _default_recovery_hints = [
        "release_unused_resources",
        "scale_resource_pool",
    ]


class ResourceViolation(ResourceError):
    """
    Raised when a resource constraint cannot be satisfied.

    Attributes
    ----------
    resource_type : str
        E.g. "gpu", "ram", "specialized_hardware".
    requested : Any
        Amount or set of resources requested.
    available : Any
        Amount or set of resources actually available.
    task_id : str
        ID of the task that triggered the violation.
    resolution_strategies : List[str]
        Ordered list of suggested resolution approaches.
    """

    _default_recovery_hints = [
        "resource_scaling",
        "task_reprioritisation",
        "priority_reallocation",
    ]

    def __init__(
        self,
        message: str,
        resource_type: str,
        requested: Any,
        available: Any,
        *,
        task_id: str = "",
        resolution_strategies: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.requested = requested
        # Backward-compatibility alias used by older call sites.
        self.required = requested
        self.available = available
        self.task_id = task_id
        self.resolution_strategies: List[str] = list(
            resolution_strategies if resolution_strategies is not None
            else self._default_recovery_hints
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            resource_type=self.resource_type,
            requested=self.requested,
            available=self.available,
            task_id=self.task_id,
            resolution_strategies=self.resolution_strategies,
        )
        return d


class ResourceAcquisitionError(ResourceViolation):
    """
    Raised when a task cannot acquire the resources it needs to begin execution.

    This is distinct from ResourceViolation in that acquisition failures indicate
    the resource pool is temporarily unavailable, rather than globally exhausted.

    Attributes
    ----------
    retry_after_seconds : float
        Suggested wait time before retrying acquisition (0 = no retry suggested).
    """

    _default_recovery_hints = [
        "retry_resource_acquisition",
        "queue_task_until_resources_available",
        "reduce_resource_request",
    ]

    def __init__(
        self,
        message: str,
        resource_type: str = "unspecified",
        requested: Any = None,
        available: Any = None,
        *,
        retry_after_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            resource_type,
            requested or {},
            available or {},
            **kwargs,
        )
        self.retry_after_seconds = retry_after_seconds

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["retry_after_seconds"] = self.retry_after_seconds
        return d


class SafetyMarginError(ResourceViolation):
    """
    Raised when a resource allocation would breach a configured safety buffer.

    Safety margins are guardrails that prevent the planner from fully exhausting
    any resource. Breaching them does not mean the resource is literally
    unavailable, but that continued allocation is unsafe.

    Attributes
    ----------
    buffer_amount : float
        The safety buffer that was violated (as a fraction, e.g. 0.15 = 15 %).
    measured_utilisation : float
        Observed utilisation at the time of the violation (0–1).
    safe_utilisation_limit : float
        Maximum utilisation considered safe (1.0 – buffer_amount).
    """

    _default_recovery_hints = [
        "reduce_resource_allocation",
        "defer_low_priority_tasks",
        "increase_safety_buffer_threshold",
    ]

    def __init__(
        self,
        message: str,
        resource_type: str,
        *,
        buffer_amount: float = 0.0,
        requested: Any = None,
        available: Any = None,
        measured_utilisation: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            resource_type,
            requested or {},
            available or {},
            **kwargs,
        )
        self.buffer_amount = buffer_amount
        self.measured_utilisation = measured_utilisation
        self.safe_utilisation_limit: float = max(0.0, 1.0 - buffer_amount)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            buffer_amount=self.buffer_amount,
            measured_utilisation=self.measured_utilisation,
            safe_utilisation_limit=self.safe_utilisation_limit,
        )
        return d


class ResourceLeakError(ResourceError):
    """
    Raised when a task terminates without releasing its allocated resources.

    Attributes
    ----------
    task_id : str
        ID of the task that leaked resources.
    leaked_resources : Dict[str, Any]
        Map of resource type to leaked amount.
    """

    _default_recovery_hints = [
        "force_release_leaked_resources",
        "quarantine_task",
        "alert_resource_operator",
    ]

    def __init__(
        self,
        message: str,
        *,
        task_id: str = "",
        leaked_resources: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.task_id = task_id
        self.leaked_resources: Dict[str, Any] = dict(leaked_resources or {})

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(task_id=self.task_id, leaked_resources=self.leaked_resources)
        return d


# ===========================================================================
# ExecutionError – runtime execution problems
# ===========================================================================

class ExecutionError(PlanningError):
    """Base class for errors occurring during plan execution."""

    _default_recovery_hints = [
        "replan_from_current_state",
        "rollback_to_last_checkpoint",
    ]


class AdjustmentError(ExecutionError):
    """
    Raised when an interactive plan adjustment is invalid or unsafe.

    Attributes
    ----------
    adjustment : Dict[str, Any]
        The adjustment dict that failed.
    conflict_details : Dict[str, Any]
        Structured description of why the adjustment was rejected.
    adjustment_type : str
        The ``type`` field from the adjustment dict, for quick introspection.
    """

    _default_recovery_hints = [
        "validate_adjustment_before_apply",
        "retry_with_corrected_parameters",
        "discard_adjustment",
    ]

    def __init__(
        self,
        message: str,
        adjustment: Dict[str, Any],
        conflict_details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.adjustment: Dict[str, Any] = dict(adjustment)
        self.conflict_details: Dict[str, Any] = dict(conflict_details or {})
        self.adjustment_type: str = adjustment.get("type", "unknown")

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            adjustment=self.adjustment,
            adjustment_type=self.adjustment_type,
            conflict_details=self.conflict_details,
        )
        return d


class ReplanningError(ExecutionError):
    """
    Raised when the planner fails to generate a recovery plan after a task failure.

    Attributes
    ----------
    failed_task : Any
        The task object (or task ID string) that triggered replanning.
    failed_task_id : str
        String ID extracted from the failed task for easy logging.
    candidates : List[Any]
        Repair candidates that were considered but rejected.
    failure_reason : str
        Why replanning failed (e.g. "no_valid_candidates", "search_exhausted").
    attempt_count : int
        Number of replanning attempts made.
    """

    _default_recovery_hints = [
        "expand_repair_candidate_pool",
        "relax_plan_constraints",
        "fallback_to_safe_abort",
        "request_operator_guidance",
    ]

    def __init__(
        self,
        message: str,
        failed_task: Any,
        candidates: Optional[List[Any]] = None,
        *,
        failure_reason: str = "unknown",
        attempt_count: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.failed_task = failed_task
        self.failed_task_id: str = (
            getattr(failed_task, "id", None)
            or getattr(failed_task, "name", None)
            or str(failed_task)
        )
        self.candidates: List[Any] = list(candidates or [])
        self.failure_reason = failure_reason
        self.attempt_count = attempt_count

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            failed_task_id=self.failed_task_id,
            failure_reason=self.failure_reason,
            attempt_count=self.attempt_count,
            candidate_count=len(self.candidates),
        )
        return d


class TaskTimeoutError(ExecutionError):
    """
    Raised when a task exceeds its configured execution timeout.

    Attributes
    ----------
    task_name : str
    task_id : str
    timeout_seconds : float
        The timeout limit that was exceeded.
    elapsed_seconds : float
        How long the task had been running.
    """

    _default_recovery_hints = [
        "terminate_task_and_replan",
        "increase_task_timeout",
        "decompose_task_into_shorter_steps",
    ]

    def __init__(
        self,
        message: str,
        *,
        task_name: str = "",
        task_id: str = "",
        timeout_seconds: float = 0.0,
        elapsed_seconds: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.task_name = task_name
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            task_name=self.task_name,
            task_id=self.task_id,
            timeout_seconds=self.timeout_seconds,
            elapsed_seconds=self.elapsed_seconds,
        )
        return d


# ===========================================================================
# PlanningConfigError – configuration / initialisation problems
# ===========================================================================

class PlanningConfigError(PlanningError):
    """
    Raised when the planning subsystem encounters an invalid or missing
    configuration value during initialisation or runtime.

    Attributes
    ----------
    config_key : str
        The configuration key that is missing or invalid.
    config_section : str
        The section of the config file (e.g. "safety_margins").
    expected_type : str
        Human-readable description of the expected type or value.
    """

    _default_recovery_hints = [
        "validate_config_file",
        "restore_default_configuration",
        "contact_system_administrator",
    ]

    def __init__(
        self,
        message: str,
        *,
        config_key: str = "",
        config_section: str = "",
        expected_type: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_section = config_section
        self.expected_type = expected_type

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            config_key=self.config_key,
            config_section=self.config_section,
            expected_type=self.expected_type,
        )
        return d


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Planning Errors – smoke test ===\n")

    errors = [
        PreconditionViolation(
            "Robot arm not in home position",
            task_name="PickObject",
            task_id="task_001",
            failed_conditions=["arm_at_home == True"],
            world_state_snapshot={"arm_at_home": False},
        ),
        PostconditionViolation(
            "Object not placed in target zone after move",
            task_name="PlaceObject",
            task_id="task_002",
            expected_state={"object_in_zone": True},
            actual_state={"object_in_zone": False},
        ),
        DeadlineExceededError(
            "Assembly task will miss deadline by 45 s",
            task_name="AssemblePart",
            task_id="task_003",
            deadline=_utc_timestamp() + 10,
            projected_completion=_utc_timestamp() + 55,
        ),
        SafetyMarginError(
            "GPU utilisation exceeds safety buffer",
            resource_type="gpu",
            buffer_amount=0.15,
            measured_utilisation=0.92,
        ),
        ReplanningError(
            "No valid repair plan found after sensor failure",
            failed_task="task_004",
            failure_reason="no_valid_candidates",
            attempt_count=3,
        ),
        AdjustmentError(
            "Cannot remove task that has active dependents",
            adjustment={"type": "remove_task", "task_id": "task_005"},
            conflict_details={"dependents": ["task_006", "task_007"]},
        ),
        CyclicDependencyError(
            "Dependency cycle detected in plan graph",
            cycle_path=["task_A", "task_B", "task_C", "task_A"],
        ),
    ]

    for err in errors:
        print(f"[{type(err).__name__}]")
        print(f"  message : {err.message}")
        print(f"  hints   : {err.recovery_hints[:2]}")
        print(f"  ts      : {_fmt_ts(err.timestamp)}")
        print()

    print("=== All error classes instantiated successfully ===\n")