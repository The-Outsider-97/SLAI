"""
Planning Helpers – Centralised utility library for the Planning Agent subsystem.

This module is the single shared toolkit imported by every module in the planner.
It is organised into seven cohesive groups:

  1. Type-checking & validation helpers
  2. Task graph helpers           (dependency resolution, cycle detection, DAG traversal)
  3. State helpers                (state manipulation, diffing, merging)
  4. Temporal helpers             (deadline arithmetic, schedule windows)
  5. Resource helpers             (margin calculation, allocation accounting)
  6. Retry / resilience helpers   (exponential back-off, circuit breaker)
  7. Serialisation helpers        (safe JSON encode/decode, dict-flattening)

Design principles
-----------------
- Pure functions only – no global state, no side-effects outside their return value.
- All functions are thread-safe (they operate on local or caller-supplied data).
- Every function has a well-defined return type and raises a specific PlanningError
  subclass (never a bare Exception) when it cannot proceed.
- Parameter validation is strict: callers receive an immediate, descriptive error
  instead of a misleading result.
- Logging is used only for DEBUG messages; errors are surfaced as exceptions.
"""

from __future__ import annotations

import json
import math
import re
import time

from collections import defaultdict, deque
from typing import (Any, Callable, Dict, FrozenSet, Iterable, Iterator,
                    List, Optional, Tuple, TypeVar, Union, Set,)

from .planning_errors import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Planning Helpers")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Section 1 – Type-checking & validation helpers
# ---------------------------------------------------------------------------
def require_type(value: Any, expected: Union[type, Tuple[type, ...]], name: str = "value") -> None:
    """
    Assert that *value* is an instance of *expected* (or one of expected types).

    Raises
    ------
    PlanningConfigError
        If the type check fails.
    """
    if not isinstance(value, expected):
        # Build a readable expected type description
        if isinstance(expected, type):
            expected_name = expected.__name__
        else:
            expected_name = " | ".join(t.__name__ for t in expected)
        raise PlanningConfigError(
            f"'{name}' must be of type {expected_name}, "
            f"got {type(value).__name__}: {value!r}",
            config_key=name,
            expected_type=expected_name,
        )


def require_positive(value: Union[int, float], name: str = "value") -> None:
    """
    Assert that *value* is strictly greater than zero.

    Raises
    ------
    PlanningConfigError
    """
    require_type(value, (int, float), name)
    if value <= 0:
        raise PlanningConfigError(
            f"'{name}' must be positive, got {value}",
            config_key=name,
            expected_type="positive number",
        )


def require_non_negative(value: Union[int, float], name: str = "value") -> None:
    """Assert that *value* is ≥ 0."""
    require_type(value, (int, float), name)
    if value < 0:
        raise PlanningConfigError(
            f"'{name}' must be non-negative, got {value}",
            config_key=name,
            expected_type="non-negative number",
        )


def require_in_range(
    value: Union[int, float],
    lo: float,
    hi: float,
    name: str = "value",
    *,
    inclusive: bool = True,
) -> None:
    """
    Assert that *lo* ≤ *value* ≤ *hi* (or strict when ``inclusive=False``).

    Raises
    ------
    PlanningConfigError
    """
    require_type(value, (int, float), name)
    if inclusive:
        ok = lo <= value <= hi
    else:
        ok = lo < value < hi
    if not ok:
        bracket = "[]" if inclusive else "()"
        raise PlanningConfigError(
            f"'{name}' must be in {bracket[0]}{lo}, {hi}{bracket[1]}, got {value}",
            config_key=name,
            expected_type=f"number in [{lo}, {hi}]",
        )


def require_non_empty(value: Any, name: str = "value") -> None:
    """
    Assert that *value* is a non-empty string, list, dict, set, or tuple.

    Raises
    ------
    PlanningConfigError
    """
    if not value:
        raise PlanningConfigError(
            f"'{name}' must not be empty, got {value!r}",
            config_key=name,
            expected_type="non-empty collection or string",
        )


def is_valid_task_id(task_id: str) -> bool:
    """
    Return True if *task_id* matches the canonical task-ID format.

    A valid ID is a non-empty string composed of alphanumeric characters,
    underscores, and hyphens (no whitespace, no slashes).
    """
    if not isinstance(task_id, str) or not task_id:
        return False
    return bool(re.fullmatch(r"[\w\-]+", task_id))


def validate_task_id(task_id: str, context: str = "") -> None:
    """
    Raise AdjustmentError if *task_id* is not a valid task identifier.

    Parameters
    ----------
    context : str
        Optional human-readable context string for the error message.
    """
    if not is_valid_task_id(task_id):
        prefix = f"[{context}] " if context else ""
        raise AdjustmentError(
            f"{prefix}Invalid task ID: {task_id!r}. "
            "IDs must be non-empty alphanumeric strings (underscores and hyphens allowed).",
            adjustment={"task_id": task_id},
        )


def validate_probability(value: float, name: str = "probability") -> None:
    """Assert that *value* is a valid probability in [0, 1]."""
    require_in_range(value, 0.0, 1.0, name)


def clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* clamped to [*lo*, *hi*]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Section 2 – Task graph helpers
# ---------------------------------------------------------------------------

def topological_sort(
    task_ids: Iterable[str],
    dependency_map: Dict[str, List[str]],
) -> List[str]:
    """
    Return a topological ordering of *task_ids* respecting *dependency_map*.

    Parameters
    ----------
    task_ids :
        All task IDs to sort (must be the complete set of nodes).
    dependency_map :
        Maps each task ID to a list of task IDs it depends on (its prerequisites).
        Tasks with no dependencies should be present with an empty list.

    Returns
    -------
    List[str]
        Task IDs in valid execution order (prerequisites first).

    Raises
    ------
    CyclicDependencyError
        If the graph contains a cycle.
    """
    ids: List[str] = list(task_ids)
    in_degree: Dict[str, int] = {tid: 0 for tid in ids}
    children: Dict[str, List[str]] = {tid: [] for tid in ids}

    for tid in ids:
        for dep in dependency_map.get(tid, []):
            if dep in in_degree:
                in_degree[tid] += 1
                children[dep].append(tid)

    queue: deque[str] = deque(tid for tid in ids if in_degree[tid] == 0)
    result: List[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for child in children.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(result) != len(ids):
        # Identify nodes still in a cycle
        remaining = [tid for tid in ids if tid not in set(result)]
        raise CyclicDependencyError(
            f"Cyclic dependency detected among tasks: {remaining}",
            cycle_path=remaining,
        )

    return result


def detect_cycles(
    task_ids: Iterable[str],
    dependency_map: Dict[str, List[str]],
) -> Optional[List[str]]:
    """
    Return the first cycle path found in the dependency graph, or *None*.

    Uses depth-first search with a recursion stack.

    Returns
    -------
    Optional[List[str]]
        Ordered list of task IDs forming the cycle, or None if no cycle exists.
    """
    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    path: Dict[str, str] = {}  # child -> parent

    def _dfs(node: str) -> Optional[List[str]]:
        visited.add(node)
        rec_stack.add(node)
        for dep in dependency_map.get(node, []):
            if dep not in visited:
                path[dep] = node
                result = _dfs(dep)
                if result is not None:
                    return result
            elif dep in rec_stack:
                # Reconstruct cycle
                cycle = [dep]
                cur = node
                while cur != dep:
                    cycle.append(cur)
                    cur = path.get(cur, dep)
                cycle.append(dep)
                return list(reversed(cycle))
        rec_stack.discard(node)
        return None

    for tid in task_ids:
        if tid not in visited:
            cycle = _dfs(tid)
            if cycle:
                return cycle
    return None


def compute_critical_path(
    task_ids: Iterable[str],
    dependency_map: Dict[str, List[str]],
    duration_map: Dict[str, float],
) -> Tuple[float, List[str]]:
    """
    Compute the Critical Path Method (CPM) length and path.

    Parameters
    ----------
    task_ids :
        All task IDs in the plan.
    dependency_map :
        Maps each task to its list of prerequisite task IDs.
    duration_map :
        Maps each task ID to its estimated duration in seconds.

    Returns
    -------
    Tuple[float, List[str]]
        (critical_path_duration_seconds, ordered_list_of_task_ids_on_path)

    Raises
    ------
    CyclicDependencyError
        If the graph contains a cycle.
    """
    order = topological_sort(task_ids, dependency_map)

    # Build reverse map: task -> tasks that depend on it
    successors: Dict[str, List[str]] = defaultdict(list)
    for tid in order:
        for dep in dependency_map.get(tid, []):
            successors[dep].append(tid)

    # Forward pass – earliest start times
    earliest_finish: Dict[str, float] = {}
    for tid in order:
        deps = dependency_map.get(tid, [])
        start = max((earliest_finish.get(d, 0.0) for d in deps), default=0.0)
        earliest_finish[tid] = start + duration_map.get(tid, 0.0)

    total_duration = max(earliest_finish.values(), default=0.0)

    # Backward pass – latest finish times
    latest_finish: Dict[str, float] = {tid: total_duration for tid in order}
    for tid in reversed(order):
        for succ in successors.get(tid, []):
            lf = latest_finish[succ] - duration_map.get(succ, 0.0)
            latest_finish[tid] = min(latest_finish[tid], lf)

    # Critical path: tasks where slack == 0
    slack: Dict[str, float] = {
        tid: latest_finish[tid] - earliest_finish[tid] for tid in order
    }
    critical = [tid for tid in order if abs(slack.get(tid, 1.0)) < 1e-9]

    return total_duration, critical


def get_all_predecessors(
    task_id: str,
    dependency_map: Dict[str, List[str]],
) -> FrozenSet[str]:
    """
    Return the transitive closure of all predecessors of *task_id*.

    Parameters
    ----------
    task_id :
        The task whose predecessors to collect.
    dependency_map :
        Maps each task ID to its direct prerequisite IDs.

    Returns
    -------
    FrozenSet[str]
        All ancestor task IDs (does not include *task_id* itself).
    """
    result: Set[str] = set()
    stack = list(dependency_map.get(task_id, []))
    while stack:
        node = stack.pop()
        if node not in result:
            result.add(node)
            stack.extend(dependency_map.get(node, []))
    return frozenset(result)


def get_all_successors(
    task_id: str,
    dependency_map: Dict[str, List[str]],
) -> FrozenSet[str]:
    """
    Return the transitive closure of all successors of *task_id*.

    Parameters
    ----------
    dependency_map :
        Maps each task ID to its direct *prerequisite* IDs (upstream edges).
        This function inverts the graph to follow downstream edges.
    """
    # Build forward (successor) map
    successor_map: Dict[str, List[str]] = defaultdict(list)
    for tid, deps in dependency_map.items():
        for dep in deps:
            successor_map[dep].append(tid)

    result: Set[str] = set()
    stack = list(successor_map.get(task_id, []))
    while stack:
        node = stack.pop()
        if node not in result:
            result.add(node)
            stack.extend(successor_map.get(node, []))
    return frozenset(result)


def build_dependency_map(tasks: List[Any]) -> Dict[str, List[str]]:
    """
    Build a dependency map from a list of task objects.

    Each task object must have an ``id`` attribute and a ``dependencies``
    attribute that is a list of prerequisite task IDs.

    Returns
    -------
    Dict[str, List[str]]
        Maps task.id -> list of dependency task IDs.
    """
    dep_map: Dict[str, List[str]] = {}
    for task in tasks:
        tid = getattr(task, "id", None)
        if tid is None:
            continue
        deps = getattr(task, "dependencies", [])
        dep_map[tid] = list(deps) if deps else []
    return dep_map


def iter_tasks_in_execution_order(tasks: List[Any]) -> Iterator[Any]:
    """
    Yield tasks in valid topological execution order.

    Parameters
    ----------
    tasks :
        List of task objects with ``id`` and ``dependencies`` attributes.

    Yields
    ------
    Any
        Task objects in execution order (prerequisites first).

    Raises
    ------
    CyclicDependencyError
    PlanningConfigError
        If any task lacks a valid string id.
    """
    # Validate all task ids first
    task_by_id: Dict[str, Any] = {}
    for task in tasks:
        tid = getattr(task, "id", None)
        if tid is None or not is_valid_task_id(tid):
            raise PlanningConfigError(
                f"Task {task!r} has invalid or missing id: {tid!r}",
                config_key="id",
                expected_type="non‑empty alphanumeric string (underscores/hyphens allowed)",
            )
        task_by_id[tid] = task

    dep_map = build_dependency_map(tasks)   # build_dependency_map already expects valid ids
    ordered = topological_sort(list(task_by_id.keys()), dep_map)
    for tid in ordered:
        yield task_by_id[tid]


# ---------------------------------------------------------------------------
# Section 3 – State helpers
# ---------------------------------------------------------------------------

StateDict = Dict[str, Any]


def diff_states(
    before: StateDict,
    after: StateDict,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Return the symmetric difference between two world-state snapshots.

    Returns
    -------
    Dict[str, Tuple[Any, Any]]
        Maps each changed key to a ``(before_value, after_value)`` pair.
        Keys only present in one snapshot appear with the value ``None`` on the
        missing side.
    """
    all_keys = set(before) | set(after)
    return {
        key: (before.get(key), after.get(key))
        for key in all_keys
        if before.get(key) != after.get(key)
    }


def merge_states(base: StateDict, overlay: StateDict) -> StateDict:
    """
    Return a new state dict that is *base* overwritten by *overlay*.

    Neither input is mutated.
    """
    merged = dict(base)
    merged.update(overlay)
    return merged


def state_satisfies_goal(state: StateDict, goal: StateDict) -> bool:
    """
    Return True if *state* satisfies every key-value pair in *goal*.

    A ``None`` goal value means "key must be absent from the state".
    """
    for key, expected in goal.items():
        if expected is None:
            if key in state:
                return False
        else:
            if state.get(key) != expected:
                return False
    return True


def compute_state_distance(state: StateDict, goal: StateDict) -> float:
    """
    Compute a normalised distance between *state* and *goal* (0 = goal reached, 1 = nothing satisfied).

    Only keys present in *goal* are considered.
    """
    if not goal:
        return 0.0
    unsatisfied = sum(
        1 for k, v in goal.items() if state.get(k) != v
    )
    return unsatisfied / len(goal)


def extract_state_subset(state: StateDict, keys: Iterable[str]) -> StateDict:
    """Return a new dict containing only the *keys* that exist in *state*."""
    return {k: state[k] for k in keys if k in state}


def apply_state_effects(
    state: StateDict,
    effects: List[Callable[[StateDict], None]],
) -> StateDict:
    """
    Apply a list of effect callables to a *copy* of *state* and return the result.

    Each callable receives and mutates the dict in place.
    The original *state* is never modified.
    """
    import copy
    new_state = copy.deepcopy(state)
    for effect in effects:
        try:
            effect(new_state)
        except Exception as exc:
            logger.debug("Effect %s raised %s – skipping", effect, exc)
    return new_state


def check_preconditions(
    state: StateDict,
    preconditions: List[Callable[[StateDict], bool]],
    task_name: str = "",
    task_id: str = "",
) -> None:
    """
    Evaluate all *preconditions* against *state* and raise on first failure.

    Raises
    ------
    PreconditionViolation
    """
    failed: List[str] = []
    for i, cond in enumerate(preconditions):
        try:
            if not cond(state):
                failed.append(f"precondition[{i}] ({getattr(cond, '__name__', 'anonymous')})")
        except Exception as exc:
            failed.append(f"precondition[{i}] raised {exc}")

    if failed:
        raise PreconditionViolation(
            f"Preconditions not satisfied for task '{task_name or task_id}': {failed}",
            task_name=task_name,
            task_id=task_id,
            failed_conditions=failed,
            world_state_snapshot=state,
        )


# ---------------------------------------------------------------------------
# Section 4 – Temporal helpers
# ---------------------------------------------------------------------------

def seconds_until_deadline(deadline: float, now: Optional[float] = None) -> float:
    """
    Return seconds remaining until *deadline*.

    Returns a negative number if the deadline has passed.
    """
    if now is None:
        now = time.time()
    return deadline - now


def is_past_deadline(deadline: float, now: Optional[float] = None) -> bool:
    """Return True if *deadline* has already passed."""
    return seconds_until_deadline(deadline, now) < 0.0


def compute_schedule_window(
    start_not_before: float,
    deadline: float,
    estimated_duration: float,
) -> Tuple[float, float]:
    """
    Compute the latest permissible start time and the available slack.

    Parameters
    ----------
    start_not_before :
        Earliest permitted start time (POSIX timestamp).
    deadline :
        Latest permitted completion time (POSIX timestamp).
    estimated_duration :
        Expected task duration in seconds.

    Returns
    -------
    Tuple[float, float]
        (latest_start_time, slack_seconds). Slack may be negative.

    Raises
    ------
    TemporalViolation
        If the window is too narrow to fit the task (slack < 0 and fatal).
    """
    latest_start = deadline - estimated_duration
    slack = latest_start - start_not_before
    if slack < 0.0:
        raise TemporalViolation(
            f"Task cannot fit in schedule window: "
            f"duration {estimated_duration:.1f}s > window {deadline - start_not_before:.1f}s",
            violation_type="window",
            time_delta=abs(slack),
            constraint_details={
                "start_not_before": start_not_before,
                "deadline": deadline,
                "estimated_duration": estimated_duration,
            },
        )
    return latest_start, slack


def estimate_end_time(
    start_time: float,
    duration: float,
    *,
    time_buffer: float = 0.0,
) -> float:
    """Return ``start_time + duration + time_buffer``."""
    return start_time + duration + time_buffer


def compute_temporal_margin(
    total_duration: float,
    available_time: float,
    *,
    time_buffer: float = 0.0,
) -> float:
    """
    Compute a normalised temporal margin in [0, 1].

    Returns 0.0 if *available_time* is zero or negative, 1.0 if no time will be used.
    The time buffer is subtracted from available time before computing the margin.
    """
    if available_time <= 0.0:
        return 0.0
    effective_available = max(0.0, available_time - time_buffer)
    if effective_available <= 0.0:
        return 0.0
    utilisation = total_duration / effective_available
    return clamp(1.0 - utilisation, 0.0, 1.0)


def sort_tasks_by_deadline(tasks: List[Any]) -> List[Any]:
    """
    Return tasks sorted in Earliest Deadline First (EDF) order.

    Tasks without a deadline are placed at the end, sorted by priority (descending).
    """
    def sort_key(t: Any) -> Tuple[float, int]:
        dl = getattr(t, "deadline", 0.0) or 0.0
        priority = getattr(t, "priority", 0) or 0
        if dl <= 0.0:
            # No deadline: sort by descending priority, after all deadline tasks
            return (float("inf"), -priority)
        return (dl, -priority)

    return sorted(tasks, key=sort_key)


def tasks_with_imminent_deadlines(
    tasks: List[Any],
    horizon_seconds: float,
    now: Optional[float] = None,
) -> List[Any]:
    """
    Return tasks whose deadlines fall within *horizon_seconds* from now.

    Tasks without a deadline (deadline == 0) are excluded.
    """
    if now is None:
        now = time.time()
    return [
        t for t in tasks
        if getattr(t, "deadline", 0.0) > 0.0
        and 0.0 < (getattr(t, "deadline", 0.0) - now) <= horizon_seconds
    ]


# ---------------------------------------------------------------------------
# Section 5 – Resource helpers
# ---------------------------------------------------------------------------

def compute_resource_utilisation(
    requested: float,
    available: float,
) -> float:
    """
    Return utilisation as a fraction in [0, ∞).

    A value > 1 indicates over-subscription.
    Returns 0.0 if *available* is zero and *requested* is also zero.

    Raises
    ------
    ResourceViolation
        If *available* is zero but *requested* is positive.
    """
    if available <= 0.0:
        if requested <= 0.0:
            return 0.0
        raise ResourceViolation(
            f"Resource is unavailable (available=0) but {requested} units requested",
            resource_type="unknown",
            requested=requested,
            available=available,
        )
    return requested / available


def compute_resource_margin(
    requested: float,
    available: float,
    *,
    safety_buffer: float = 0.0,
) -> float:
    """
    Return a normalised resource margin in [0, 1].

    The margin is ``1 - utilisation - safety_buffer``, clamped to [0, 1].
    A safety_buffer of 0.15 means 15 % of capacity is always reserved.

    Returns 1.0 when *requested* == 0 (nothing requested).
    """
    if requested <= 0.0:
        return 1.0
    if available <= 0.0:
        return 0.0
    utilisation = requested / available
    return clamp(1.0 - utilisation - safety_buffer, 0.0, 1.0)


def check_resource_feasibility(
    requirements: Dict[str, float],
    available: Dict[str, float],
    *,
    safety_buffers: Optional[Dict[str, float]] = None,
    task_id: str = "",
) -> Dict[str, float]:
    """
    Check whether *requirements* can be satisfied from *available* resources.

    Parameters
    ----------
    requirements :
        Resource name -> amount required.
    available :
        Resource name -> amount available.
    safety_buffers :
        Optional per-resource safety buffers (fractions, e.g. {"gpu": 0.15}).
    task_id :
        Optional task identifier for error context.

    Returns
    -------
    Dict[str, float]
        Per-resource margins (values in [0, 1]).

    Raises
    ------
    ResourceViolation
        If any resource requirement cannot be met (accounting for buffers).
    """
    buffers = safety_buffers or {}
    margins: Dict[str, float] = {}

    for resource, needed in requirements.items():
        cap = available.get(resource, 0.0)
        buf = buffers.get(resource, 0.0)
        margin = compute_resource_margin(needed, cap, safety_buffer=buf)
        margins[resource] = margin
        if margin <= 0.0:
            raise ResourceViolation(
                f"Insufficient {resource}: requested {needed}, "
                f"available {cap} (buffer {buf:.0%})",
                resource_type=resource,
                requested=needed,
                available=cap,
                task_id=task_id,
            )

    return margins


def aggregate_resource_requirements(tasks: List[Any]) -> Dict[str, float]:
    """
    Sum resource requirements across a list of task objects.

    Each task is expected to have a ``resource_requirements`` attribute whose
    fields are numeric (e.g. ``ResourceProfile`` with ``gpu`` and ``ram``).
    Unknown / non-numeric fields are silently skipped.

    Returns
    -------
    Dict[str, float]
        Aggregated requirements per resource type.
    """
    totals: Dict[str, float] = defaultdict(float)
    for task in tasks:
        req = getattr(task, "resource_requirements", None)
        if req is None:
            continue
        for field_name in vars(req):
            val = getattr(req, field_name, None)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                totals[field_name] += val
    return dict(totals)


def geometric_mean(*values: float) -> float:
    """
    Return the geometric mean of *values*.

    All values must be non-negative. Returns 0.0 if any value is 0.
    """
    if not values:
        return 0.0
    if any(v < 0 for v in values):
        raise PlanningError(
            f"geometric_mean requires non-negative values, got {values}",
        )
    if any(v == 0.0 for v in values):
        return 0.0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


# ---------------------------------------------------------------------------
# Section 6 – Retry / resilience helpers
# ---------------------------------------------------------------------------

def compute_backoff_delay(
    attempt: int,
    *,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 0.0,
) -> float:
    """
    Compute the delay for the *attempt*-th retry using exponential back-off.

    Parameters
    ----------
    attempt :
        1-based attempt index (1 = first retry after first failure).
    base_delay :
        Initial delay in seconds.
    backoff_factor :
        Multiplier applied on each additional attempt.
    max_delay :
        Upper bound on the computed delay.
    jitter :
        If > 0, adds a random fraction of the computed delay (requires
        ``import random`` at the call site; this function uses ``math``
        only and does not add jitter itself – jitter is a documentation hint).

    Returns
    -------
    float
        Delay in seconds.
    """
    require_positive(base_delay, "base_delay")
    require_positive(backoff_factor, "backoff_factor")
    delay = base_delay * (backoff_factor ** (attempt - 1))
    return min(delay, max_delay)


def should_retry(
    attempt: int,
    max_attempts: int,
    error: Optional[Exception] = None,
    *,
    retryable_types: Optional[Tuple[type, ...]] = None,
) -> bool:
    """
    Decide whether to retry based on attempt count and error type.

    Parameters
    ----------
    attempt :
        Current 1-based attempt count.
    max_attempts :
        Maximum number of attempts (including the first).
    error :
        The exception that caused the failure (or None).
    retryable_types :
        If provided, only these exception types trigger a retry.

    Returns
    -------
    bool
    """
    if attempt >= max_attempts:
        return False
    if error is None:
        return True
    if retryable_types is None:
        return True
    return isinstance(error, retryable_types)


class DotDict(dict):
    def __getattr__(self, item):
        return self.get(item)


class CircuitBreaker:
    """
    A simple, stateful circuit breaker for protecting external calls.

    States
    ------
    CLOSED    Normal operation; failures are counted.
    OPEN      All calls are blocked immediately.
    HALF_OPEN One probe call is allowed; success resets to CLOSED.

    This implementation is thread-safe only when a single instance is used from
    a single thread. For multi-threaded use, protect calls with an external lock.

    Usage
    -----
    >>> cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
    >>> if cb.is_callable():
    ...     try:
    ...         result = some_external_call()
    ...         cb.record_success()
    ...     except Exception:
    ...         cb.record_failure()
    """

    _CLOSED = "closed"
    _OPEN = "open"
    _HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "circuit_breaker",
    ) -> None:
        require_positive(failure_threshold, "failure_threshold")
        require_positive(recovery_timeout, "recovery_timeout")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._state = self._CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> str:
        """Current circuit state: "closed", "open", or "half_open"."""
        return self._state

    def is_callable(self) -> bool:
        """Return True if a call should be allowed through."""
        if self._state == self._CLOSED:
            return True
        if self._state == self._OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = self._HALF_OPEN
                logger.debug("CircuitBreaker '%s' -> HALF_OPEN", self.name)
                return True
            return False
        # HALF_OPEN: allow exactly one probe
        return True

    def record_success(self) -> None:
        """Record a successful call; resets to CLOSED."""
        self._failure_count = 0
        if self._state != self._CLOSED:
            logger.debug("CircuitBreaker '%s' -> CLOSED (success)", self.name)
        self._state = self._CLOSED

    def record_failure(self) -> None:
        """Record a failed call; may transition to OPEN."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._state == self._HALF_OPEN or self._failure_count >= self.failure_threshold:
            if self._state != self._OPEN:
                logger.debug(
                    "CircuitBreaker '%s' -> OPEN (failures=%d)",
                    self.name,
                    self._failure_count,
                )
            self._state = self._OPEN

    def reset(self) -> None:
        """Force-reset the breaker to CLOSED."""
        self._state = self._CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0


# ---------------------------------------------------------------------------
# Section 7 – Serialisation helpers
# ---------------------------------------------------------------------------
class _PlanningJSONEncoder(json.JSONEncoder):
    """Extended JSON encoder that handles common planning types."""

    def default(self, o: Any) -> Any:
        # 1. Objects with a to_dict method
        if hasattr(o, "to_dict") and callable(o.to_dict):
            return o.to_dict()

        # 2. Dataclasses (Python 3.7+)
        try:
            from dataclasses import is_dataclass, asdict
            if is_dataclass(o) and not isinstance(o, type):
                return asdict(o)
        except ImportError:
            pass

        # 3. Enums (if enum module is available)
        try:
            from enum import Enum
            if isinstance(o, Enum):
                return o.value
        except ImportError:
            pass

        # 4. Sets / frozensets
        if isinstance(o, (set, frozenset)):
            return list(o)

        # 5. Fallback to superclass (handles dict, list, str, int, etc.)
        return super().default(o)


def safe_json_dumps(obj: Any, *, indent: Optional[int] = None,
                    fallback_str: str = "<unserializable>") -> str:
    """
    Serialise *obj* to a JSON string, falling back gracefully on failure.

    Parameters
    ----------
    obj :
        The object to serialise.
    indent :
        JSON indentation level (None for compact).
    fallback_str :
        String to use if serialisation fails entirely.

    Returns
    -------
    str
    """
    try:
        return json.dumps(obj, cls=_PlanningJSONEncoder, indent=indent, default=str)
    except Exception as exc:
        logger.debug("safe_json_dumps failed: %s", exc)
        return fallback_str


def safe_json_loads(text: str, *, default: Any = None) -> Any:
    """
    Parse *text* as JSON, returning *default* on any parse error.

    Strips common Markdown code-fence wrappers (````json … ````).
    """
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.debug("safe_json_loads failed: %s", exc)
        return default


def flatten_dict(nested: Dict[str, Any], *, separator: str = ".",
                 prefix: str = "") -> Dict[str, Any]:
    """
    Flatten a nested dictionary using *separator* between key segments.

    Example
    -------
    >>> flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
    {"a.b": 1, "a.c.d": 2}
    """
    result: Dict[str, Any] = {}
    for key, value in nested.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, separator=separator, prefix=full_key))
        else:
            result[full_key] = value
    return result


def deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a deep-merged copy of *base* updated with *overlay*.

    Nested dicts are merged recursively; all other types are replaced.
    Neither input is mutated.
    """
    import copy
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def truncate_for_logging(obj: Any, max_chars: int = 256) -> str:
    """
    Return a string representation of *obj* truncated to *max_chars*.

    Useful for safely logging potentially large task parameters or state dicts.
    """
    text = str(obj)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"… (truncated, full length={len(text)})"


if __name__ == "__main__":
    print("\n=== Planning Helpers – smoke test ===\n")

    # --- Task graph ---
    dep_map = {
        "A": [],
        "B": ["A"],
        "C": ["A"],
        "D": ["B", "C"],
    }
    order = topological_sort(["A", "B", "C", "D"], dep_map)
    print(f"Topological order: {order}")
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")

    dur = {"A": 10, "B": 5, "C": 8, "D": 3}
    cp_dur, cp_path = compute_critical_path(["A", "B", "C", "D"], dep_map, dur) # pyright: ignore[reportArgumentType]
    print(f"Critical path duration: {cp_dur}s, path: {cp_path}")

    cycle = detect_cycles(["A", "B", "C"], {"A": ["B"], "B": ["C"], "C": ["A"]})
    print(f"Cycle detected: {cycle}")
    assert cycle is not None

    # --- State helpers ---
    s1 = {"x": 1, "y": 2, "z": 3}
    s2 = {"x": 1, "y": 99, "w": 4}
    diff = diff_states(s1, s2)
    print(f"State diff: {diff}")
    assert "y" in diff and "z" in diff and "w" in diff

    goal = {"x": 1, "y": 2}
    print(f"State satisfies goal: {state_satisfies_goal(s1, goal)}")
    print(f"State distance from goal: {compute_state_distance(s2, goal)}")

    # --- Temporal helpers ---
    now = time.time()
    latest_start, slack = compute_schedule_window(now, now + 100, 60)
    print(f"Schedule window: latest_start T+{latest_start - now:.0f}s, slack={slack:.0f}s")
    margin = compute_temporal_margin(60, 100, time_buffer=10)
    print(f"Temporal margin: {margin:.2f}")

    # --- Resource helpers ---
    reqs = {"gpu": 2.0, "ram": 8.0}
    avail = {"gpu": 8.0, "ram": 32.0}
    margins = check_resource_feasibility(reqs, avail, safety_buffers={"gpu": 0.10}, task_id="t1")
    print(f"Resource margins: {margins}")
    print(f"Geometric mean margin: {geometric_mean(*margins.values()):.3f}")

    # --- Retry helpers ---
    delays = [compute_backoff_delay(i, base_delay=1.0, backoff_factor=2.0) for i in range(1, 6)]
    print(f"Back-off delays: {delays}")
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5.0, name="test_cb")
    cb.record_failure()
    cb.record_failure()
    print(f"Circuit state after 2 failures: {cb.state}")

    # --- Serialisation helpers ---
    payload = {"plan": {"steps": [1, 2, 3]}, "status": "ok"}
    js = safe_json_dumps(payload, indent=2)
    print(f"JSON output (first line): {js.splitlines()[0]}")
    flat = flatten_dict({"a": {"b": 1}, "c": {"d": {"e": 2}}})
    print(f"Flattened: {flat}")

    print("\n=== All helpers exercised successfully ===\n")