"""
Planning Calculations – Centralised service for safety-critical planning computations.

Provides methods to compute resource margins, temporal margins, critical-path
duration, dependency risk, plan cost, probability of success, and composite risk
scores. All public methods are thread-safe and delegate low-level numeric work to
the shared helpers module to avoid duplication.
"""

from __future__ import annotations

import threading
import time

from typing import Any, Dict, List, Optional, Tuple, Union

from ..planning_types import ClusterResources, Task, TaskStatus, TaskType
from .config_loader import get_config_section, load_global_config
from .planning_errors import *
from .planning_helpers import *
from .resource_monitor import ResourceMonitor
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Planning Calculations")
printer = PrettyPrinter()

# ---------------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------------
TaskOrList = Union[Task, List[Task]]
MarginReport = Dict[str, float]


class PlanningCalculations:
    """
    Central service for planning-related calculations.

    All public methods are thread-safe. Expensive results (e.g. critical-path
    length) are memoised with a configurable TTL to avoid redundant recomputation
    when called repeatedly within a single planning cycle.
    """

    # ------------------------------------------------------------------
    # Construction & configuration
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.config = load_global_config()
        self.safety_config = get_config_section("safety_margins")
        self.calc_config  = get_config_section("planning_calculations")

        # --- Resource buffers (fractions reserved as safety headroom) ---
        resource_buffers = self.safety_config.get("resource_buffers", {})
        self.gpu_buffer: float  = float(resource_buffers.get("gpu",  0.15))
        self.ram_buffer: float  = float(resource_buffers.get("ram",  0.20))
        self.hw_buffer:  float  = float(resource_buffers.get("specialized_hardware_buffer", 0.10))

        # --- Temporal configuration ---
        temporal = self.safety_config.get("temporal", {})
        runtime_profile = str(
            self.calc_config.get("runtime_profile")
            or self.config.get("runtime_profile")
            or ""
        ).lower()
        
        profiles = self.safety_config.get("profiles", {})
        if runtime_profile and isinstance(profiles, dict):
            profile = profiles.get(runtime_profile, {})
            if isinstance(profile, dict):
                profile_temporal = profile.get("temporal", {})
                if isinstance(profile_temporal, dict):
                    temporal = {**temporal, **profile_temporal}
        self.time_buffer:       float = float(temporal.get("time_buffer",       120.0))
        self.min_task_duration: float = float(temporal.get("min_task_duration",  30.0))
        self.max_concurrent:    int   = int  (temporal.get("max_concurrent",       5))

        # --- Calculation tuning ---
        self.default_fallback_margin:   float = float(self.calc_config.get("default_fallback_margin",   0.70))
        self.default_success_threshold: float = float(self.calc_config.get("default_success_threshold", 0.90))
        self.cache_ttl_seconds:         float = float(self.calc_config.get("cache_ttl_seconds",         30.0))
        self.risk_weights: Dict[str, float]   = dict (self.calc_config.get("risk_weights", {
            "failure_probability": 0.50,
            "dependency_complexity": 0.30,
            "duration_uncertainty": 0.20,
        }))

        # --- Runtime state ---
        self.resource_monitor: Optional[ResourceMonitor] = None
        self._warned_no_monitor: bool = False

        # Thread-safe result cache: key -> (value, expiry_timestamp)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_lock = threading.RLock()

        self.last_temporal_diagnostics: Dict[str, Any] = {}
        self.last_margin_diagnostics: Dict[str, Any] = {}

        logger.info("PlanningCalculations initialised")

    # ------------------------------------------------------------------
    # Public interface – Resource
    # ------------------------------------------------------------------
    def calculate_resource_margin(self, tasks: TaskOrList,
        resource_state: Optional[ClusterResources] = None,
    ) -> float:
        """
        Compute a normalised resource margin for a task or list of tasks [0, 1].

        The margin reflects how much spare capacity remains after accounting for
        the aggregated requirements of *tasks* and per-resource safety buffers.
        A value of 1.0 means resources are unconstrained; 0.0 means the limit
        has been reached (or breached).

        Parameters
        ----------
        tasks :
            One or more Task objects whose resource_requirements are summed.
        resource_state :
            Live cluster resource snapshot.  If omitted, the injected
            ``ResourceMonitor`` is queried; if that is also absent a
            documented conservative default is returned.

        Returns
        -------
        float
            Geometric mean of per-resource margins, clamped to [0, 1].

        Raises
        ------
        SafetyMarginError
            If any resource breaches its configured safety buffer.
        """
        tasks = self._normalise_task_list(tasks)
        if not tasks:
            return 1.0

        available = self._resolve_resource_state(resource_state)
        if available is None:
            return self.default_fallback_margin

        # Aggregate numeric requirements across all tasks
        totals = aggregate_resource_requirements(tasks)
        gpu_req = totals.get("gpu", 0.0)
        ram_req = totals.get("ram", 0.0)

        # Collect specialised hardware requirements (union across tasks)
        hw_required: List[str] = []
        for task in tasks:
            hw_required = list(set(hw_required) | set(task.resource_requirements.specialized_hardware))

        gpu_margin = compute_resource_margin(gpu_req, float(available.gpu_total),
                                             safety_buffer=self.gpu_buffer)
        ram_margin = compute_resource_margin(ram_req, float(available.ram_total),
                                             safety_buffer=self.ram_buffer)
        hw_margin  = self._hardware_margin(hw_required, available.specialized_hardware_available)

        # Raise SafetyMarginError for any fully-exhausted resource
        for resource, margin, req, cap in [
            ("gpu", gpu_margin, gpu_req, available.gpu_total),
            ("ram", ram_margin, ram_req, available.ram_total),
        ]:
            if margin <= 0.0 and req > 0:
                utilisation = req / max(cap, 1e-9)
                raise SafetyMarginError(
                    f"{resource.upper()} safety buffer breached: "
                    f"utilisation {utilisation:.1%} exceeds safe limit",
                    resource_type=resource,
                    buffer_amount=getattr(self, f"{resource}_buffer"),
                    measured_utilisation=utilisation,
                    requested=req,
                    available=cap,
                )

        return geometric_mean(gpu_margin, ram_margin, hw_margin)

    def _hardware_margin(self, required: List[str], available: List[str]) -> float:
        """Fraction of required hardware items that are available, minus hw_buffer."""
        if not required:
            return 1.0
        if not available:
            return 0.0
        coverage = len(set(required) & set(available)) / len(required)
        return clamp(coverage - self.hw_buffer, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public interface – Temporal
    # ------------------------------------------------------------------
    def calculate_temporal_margin(self, tasks: TaskOrList, current_time: Optional[float] = None) -> float:
        """
        Compute a normalised temporal margin [0, 1] for a task or list of tasks.

        Uses the critical-path duration (via helpers) when dependency information
        is present, falling back to a sequential-sum estimate otherwise.

        Parameters
        ----------
        tasks :
            One or more Task objects.
        current_time :
            Reference timestamp (defaults to ``time.time()``).

        Returns
        -------
        float
            Temporal margin in [0, 1].  0 = no time left; 1 = fully unconstrained.

        Raises
        ------
        DeadlineExceededError
            If the earliest deadline has already passed.
        """
        tasks = self._normalise_task_list(tasks)
        if not tasks:
            return 1.0

        if current_time is None:
            current_time = time.time()

        # Determine total work and binding deadline
        diagnostics = self.calculate_temporal_diagnostics(tasks, current_time=current_time)
        
        available_time = diagnostics["available_time"]
        binding_deadline = diagnostics["binding_deadline"]
        
        if available_time is not None and available_time < 0.0:
            sorted_tasks = sort_tasks_by_deadline(tasks)
            raise DeadlineExceededError(
                f"Binding deadline already passed by {abs(available_time):.1f}s",
                task_name=getattr(sorted_tasks[0], "name", ""),
                task_id=getattr(sorted_tasks[0], "id", ""),
                deadline=binding_deadline,
                projected_completion=current_time + float(diagnostics["total_duration"]),
            )
        
        self.last_temporal_diagnostics = diagnostics
        return float(diagnostics["temporal_margin"])
    
    def calculate_temporal_diagnostics(
        self,
        tasks: TaskOrList,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return audit-safe temporal-margin diagnostics without changing public APIs."""
        tasks = self._normalise_task_list(tasks)
        now = float(current_time if current_time is not None else time.time())
    
        if not tasks:
            return {
                "task_count": 0,
                "current_time": now,
                "binding_deadline": None,
                "available_time": None,
                "total_duration": 0.0,
                "time_buffer": self.time_buffer,
                "critical_path_duration": 0.0,
                "sequential_duration": 0.0,
                "temporal_margin": 1.0,
                "unsafe_reason": None,
                "tasks": [],
            }
    
        dep_map = build_dependency_map(tasks)
        dur_map = {
            t.id: max(float(getattr(t, "duration", 300.0) or 0.0), self.min_task_duration)
            for t in tasks
        }
    
        has_deps = any(deps for deps in dep_map.values())
        critical_path_duration = 0.0
        critical_path_ids: List[str] = []
    
        if has_deps:
            critical_path_duration, critical_path_ids = compute_critical_path(
                list(dep_map.keys()),
                dep_map,
                dur_map,
            )
    
        sequential_duration = sum(dur_map.values())
        total_duration = critical_path_duration if has_deps else sequential_duration
    
        sorted_tasks = sort_tasks_by_deadline(tasks)
        binding_deadline = next(
            (
                float(getattr(t, "deadline", 0.0) or 0.0)
                for t in sorted_tasks
                if float(getattr(t, "deadline", 0.0) or 0.0) > 0.0
            ),
            now + 3600.0,
        )
    
        available_time = binding_deadline - now
        temporal_margin = compute_temporal_margin(
            total_duration,
            available_time,
            time_buffer=self.time_buffer,
        ) if available_time >= 0.0 else 0.0
    
        unsafe_reason = None
        if available_time < 0.0:
            unsafe_reason = "binding_deadline_already_passed"
        elif available_time < total_duration:
            unsafe_reason = "available_time_less_than_plan_duration"
        elif available_time < total_duration + self.time_buffer:
            unsafe_reason = "available_time_cannot_fit_duration_plus_time_buffer"
        elif temporal_margin <= 0.0:
            unsafe_reason = "temporal_margin_exhausted"
    
        return {
            "task_count": len(tasks),
            "current_time": now,
            "binding_deadline": binding_deadline,
            "available_time": available_time,
            "total_duration": total_duration,
            "time_buffer": self.time_buffer,
            "critical_path_duration": critical_path_duration,
            "critical_path_task_ids": critical_path_ids,
            "sequential_duration": sequential_duration,
            "used_critical_path": has_deps,
            "temporal_margin": temporal_margin,
            "unsafe_reason": unsafe_reason,
            "tasks": [
                {
                    "id": getattr(t, "id", ""),
                    "name": getattr(t, "name", ""),
                    "duration": dur_map.get(getattr(t, "id", ""), 0.0),
                    "raw_duration": float(getattr(t, "duration", 0.0) or 0.0),
                    "deadline": float(getattr(t, "deadline", 0.0) or 0.0),
                    "dependencies": list(getattr(t, "dependencies", []) or []),
                }
                for t in tasks
            ],
        }

    def calculate_plan_duration(self, plan: List[Task], *, use_critical_path: bool = True) -> float:
        """
        Estimate total plan duration in seconds.

        When ``use_critical_path=True`` and dependency information is present,
        the critical-path length is used (more accurate for parallel plans).
        Otherwise the sequential sum is returned.

        Parameters
        ----------
        plan :
            Ordered list of tasks.
        use_critical_path :
            Whether to prefer CPM over sequential sum.

        Returns
        -------
        float
            Estimated duration in seconds.
        """
        if not plan:
            return 0.0

        dur_map = {t.id: max(getattr(t, "duration", 300.0), self.min_task_duration)
                   for t in plan}
        dep_map = build_dependency_map(plan)

        if use_critical_path and any(deps for deps in dep_map.values()):
            cp_duration, _ = compute_critical_path(list(dep_map.keys()), dep_map, dur_map)
            return cp_duration

        return sum(dur_map.values())

    def estimate_remaining_time(self, plan: List[Task], current_time: Optional[float] = None) -> float:
        """
        Estimate wall-clock seconds remaining until the plan completes.

        For currently-executing tasks, the already-elapsed portion is subtracted.
        Completed and failed tasks are ignored.

        Parameters
        ----------
        plan :
            Full task list in execution order.
        current_time :
            Reference timestamp (defaults to ``time.time()``).

        Returns
        -------
        float
            Non-negative remaining seconds.
        """
        if not plan:
            return 0.0

        if current_time is None:
            current_time = time.time()

        remaining = 0.0
        for task in plan:
            status = getattr(task, "status", None)
            # Skip completed/failed tasks
            if status is not None and status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
                continue

            duration = max(getattr(task, "duration", 300.0), self.min_task_duration)
            start    = getattr(task, "start_time", 0.0) or 0.0

            if status == TaskStatus.EXECUTING and start > 0.0:
                elapsed = current_time - start
                remaining += max(0.0, duration - elapsed)
            else:
                remaining += duration

        return remaining

    # ------------------------------------------------------------------
    # Public interface – Dependency risk
    # ------------------------------------------------------------------
    def calculate_dependency_risk(self, tasks: TaskOrList) -> float:
        """
        Compute a dependency-graph risk score [0, 1] (1 = safe / low risk).

        Risk is modelled as a weighted combination of:
        - Normalised critical-path length (depth of the dependency chain).
        - Edge density of the dependency graph.

        Parameters
        ----------
        tasks :
            One or more Task objects.

        Returns
        -------
        float
            Risk margin in [0, 1] — higher is safer.
        """
        tasks = self._normalise_task_list(tasks)
        if not tasks:
            return 1.0

        dep_map  = build_dependency_map(tasks)
        dur_map  = {t.id: max(getattr(t, "duration", 300.0), self.min_task_duration)
                    for t in tasks}

        # Nodes that actually participate in any dependency edge
        nodes_with_deps = {tid for tid, deps in dep_map.items() if deps}
        if not nodes_with_deps:
            return 1.0  # No dependencies → no dependency risk

        n = len(nodes_with_deps)
        e = sum(len(dep_map.get(tid, [])) for tid in nodes_with_deps)

        # Normalised edge density (max possible edges = n*(n-1))
        edge_density = e / (n * (n - 1)) if n > 1 else 0.0

        # Normalised critical-path depth
        cp_duration, _ = compute_critical_path(list(dep_map.keys()), dep_map, dur_map)
        total_duration  = sum(dur_map.values()) or 1.0
        cp_depth        = clamp(cp_duration / total_duration, 0.0, 1.0)

        risk = (self.risk_weights.get("dependency_complexity", 0.30) * (
            0.6 * cp_depth + 0.4 * edge_density
        ))
        return clamp(1.0 - risk, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public interface – Probability & risk
    # ------------------------------------------------------------------
    def calculate_probability_of_success(self, task: Task) -> float:
        """
        Estimate the probability that *task* will succeed [0, 1].

        Resolution order:
        1. Explicit ``success_threshold`` attribute on the task (if probabilistic
           actions are present, the maximum action-level success rate is used).
        2. Method-level statistics from the task's history (if available).
        3. Configured ``default_success_threshold`` (fallback).

        Parameters
        ----------
        task :
            The task to evaluate.

        Returns
        -------
        float
            Probability in [0, 1].
        """
        # 1. Probabilistic actions
        if getattr(task, "is_probabilistic", False):
            actions = getattr(task, "probabilistic_actions", [])
            if actions:
                rates = [getattr(a, "success_rate", 0.0) for a in actions]
                best  = max(rates, default=0.0)
                if best > 0.0:
                    return clamp(best, 0.0, 1.0)

        # 2. Historical success rate from task.history
        history = getattr(task, "history", [])
        if history:
            outcomes = [
                h.get("outcome") for h in history
                if isinstance(h, dict) and "outcome" in h
            ]
            if outcomes:
                success_count = sum(1 for o in outcomes if o == "success")
                return clamp(success_count / len(outcomes), 0.0, 1.0)

        # 3. Configured fallback
        return clamp(
            getattr(task, "success_threshold", self.default_success_threshold),
            0.0, 1.0,
        )

    def estimate_risk_score(self, task: Task) -> float:
        """
        Compute a composite risk score for *task* [0, 1] (higher = riskier).

        The score combines:
        - Probability of failure (weight: ``risk_weights.failure_probability``).
        - Pre-computed or estimated ``risk_score`` attribute on the task
          (weight: ``risk_weights.dependency_complexity``).
        - Duration uncertainty expressed as the coefficient of variation of
          actual vs. estimated duration (weight: ``risk_weights.duration_uncertainty``).

        Parameters
        ----------
        task :
            The task to evaluate.

        Returns
        -------
        float
            Risk score in [0, 1].
        """
        w_fail = self.risk_weights.get("failure_probability",    0.50)
        w_dep  = self.risk_weights.get("dependency_complexity",  0.30)
        w_dur  = self.risk_weights.get("duration_uncertainty",   0.20)

        # Component 1 – failure probability
        prob_failure = 1.0 - self.calculate_probability_of_success(task)

        # Component 2 – pre-computed risk attribute (e.g. set by safety planner)
        task_risk = clamp(getattr(task, "risk_score", 0.0), 0.0, 1.0)

        # Component 3 – duration uncertainty: |actual - estimated| / estimated
        estimated = max(getattr(task, "estimated_duration", 0.0), 1e-9)
        actual    = getattr(task, "actual_duration", 0.0)
        dur_uncertainty = clamp(abs(actual - estimated) / estimated, 0.0, 1.0) if actual > 0.0 else 0.0

        score = w_fail * prob_failure + w_dep * task_risk + w_dur * dur_uncertainty
        return clamp(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public interface – Plan-level aggregates
    # ------------------------------------------------------------------

    def calculate_plan_cost(self, plan: List[Task]) -> float:
        """
        Sum the ``cost`` attribute across all tasks in *plan*.

        Parameters
        ----------
        plan :
            List of Task objects.

        Returns
        -------
        float
            Total plan cost (non-negative).
        """
        if not plan:
            return 0.0
        require_type(plan, list, "plan")
        return sum(max(getattr(t, "cost", 1.0), 0.0) for t in plan)

    def check_safety_margins(self, plan: List[Task],
        resources: Optional[ClusterResources] = None,
        *,
        current_time: Optional[float] = None,
    ) -> MarginReport:
        """
        Evaluate all safety margins for *plan* and return a consolidated report.

        The returned dict always contains the keys ``"resource"``, ``"temporal"``,
        and ``"dependency"``.  Values are in [0, 1] (1 = safe).  If any margin
        is at or below 0, the plan is considered unsafe.

        Parameters
        ----------
        plan :
            List of Task objects in execution order.
        resources :
            Optional live resource state; falls back to the injected monitor.
        current_time :
            Reference timestamp for temporal calculations.

        Returns
        -------
        MarginReport
            Dict with keys ``resource``, ``temporal``, ``dependency``.

        Raises
        ------
        SafetyMarginError
            Propagated from ``calculate_resource_margin`` if a buffer is breached.
        DeadlineExceededError
            Propagated from ``calculate_temporal_margin`` if a deadline has passed.
        """
        if current_time is None:
            current_time = time.time()

        temporal_margin = self.calculate_temporal_margin(plan, current_time)
        temporal_diagnostics = getattr(self, "last_temporal_diagnostics", {})
        
        margins: MarginReport = {
            "resource": self.calculate_resource_margin(plan, resources),
            "temporal": temporal_margin,
            "dependency": self.calculate_dependency_risk(plan),
        }
        
        self.last_margin_diagnostics = {
            "temporal": temporal_diagnostics,
            "resource_monitor_attached": self.resource_monitor is not None,
            "task_count": len(plan),
        }

        unsafe = {k: v for k, v in margins.items() if v <= 0.0}
        if unsafe:
            logger.warning(
                "Safety margin(s) exhausted: %s | diagnostics=%s",
                truncate_for_logging(unsafe),
                truncate_for_logging(self.last_margin_diagnostics),
            )

        return margins

    def calculate_plan_risk_profile(self, plan: List[Task]) -> Dict[str, Any]:
        """
        Build a rich risk profile for the entire plan.

        Returns a dict with:
        - ``overall_risk``: composite risk score in [0, 1].
        - ``per_task``: dict mapping task.id -> individual risk score.
        - ``highest_risk_task``: id of the riskiest task (or None).
        - ``plan_cost``: total plan cost.
        - ``plan_duration``: estimated plan duration in seconds.

        Parameters
        ----------
        plan :
            List of Task objects.

        Returns
        -------
        Dict[str, Any]
        """
        if not plan:
            return {
                "overall_risk": 0.0,
                "per_task": {},
                "highest_risk_task": None,
                "plan_cost": 0.0,
                "plan_duration": 0.0,
            }

        per_task: Dict[str, float] = {
            t.id: self.estimate_risk_score(t) for t in plan
        }
        overall_risk = sum(per_task.values()) / len(per_task)
        highest      = max(per_task, key=per_task.__getitem__) if per_task else None

        return {
            "overall_risk":       clamp(overall_risk, 0.0, 1.0),
            "per_task":           per_task,
            "highest_risk_task":  highest,
            "plan_cost":          self.calculate_plan_cost(plan),
            "plan_duration":      self.calculate_plan_duration(plan),
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Evict all memoised calculation results."""
        with self._cache_lock:
            self._cache.clear()
            logger.debug("Calculation cache cleared")

    def _cache_get(self, key: str) -> Optional[Any]:
        """Return a cached value if it exists and has not expired, else None."""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, expiry = entry
            if time.time() > expiry:
                del self._cache[key]
                return None
            return value

    def _cache_set(self, key: str, value: Any) -> None:
        """Store *value* in the cache with the configured TTL."""
        with self._cache_lock:
            self._cache[key] = (value, time.time() + self.cache_ttl_seconds)

    # ------------------------------------------------------------------
    # ResourceMonitor integration
    # ------------------------------------------------------------------
    def set_resource_monitor(self, monitor: ResourceMonitor) -> None:
        """
        Inject a live ``ResourceMonitor`` instance.

        Once set, resource-margin calculations will query the monitor instead
        of returning the conservative default.

        Parameters
        ----------
        monitor :
            Initialised ResourceMonitor.
        """
        require_type(monitor, ResourceMonitor, "monitor")
        self.resource_monitor = monitor
        self._warned_no_monitor = False
        logger.info("ResourceMonitor injected into PlanningCalculations")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_task_list(tasks: TaskOrList) -> List[Task]:
        """Coerce a single Task or list of Tasks to a list; return [] for falsy input."""
        if not tasks:
            return []
        return [tasks] if isinstance(tasks, Task) else list(tasks)

    def _resolve_resource_state(
        self, resource_state: Optional[ClusterResources]
    ) -> Optional[ClusterResources]:
        """
        Return *resource_state* if provided, else query the monitor.
        Returns None (and warns once) if neither is available.
        """
        if resource_state is not None:
            return resource_state
        if self.resource_monitor is not None:
            return self.resource_monitor.get_available_resources()
        if not self._warned_no_monitor:
            logger.warning(
                "No ResourceMonitor set – resource margin will use "
                "conservative default (%.2f)", self.default_fallback_margin
            )
            self._warned_no_monitor = True
        return None


# ---------------------------------------------------------------------------
# __main__ – compact test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Planning Calculations ===\n")
    printer.status("TEST", "PlanningCalculations initialized", "info")

    # Build a small 3-task plan: A → B → C (sequential dependencies)
    t_a = Task("Preprocess",  task_type=TaskType.PRIMITIVE, duration=60.0,  cost=2.0)
    t_b = Task("CoreProcess", task_type=TaskType.PRIMITIVE, duration=120.0, cost=5.0,
               dependencies=[t_a.id])
    t_c = Task("Postprocess", task_type=TaskType.PRIMITIVE, duration=40.0,  cost=1.5,
               dependencies=[t_b.id])

    # Give tasks realistic deadlines
    now = time.time()
    t_a.deadline = now + 600
    t_b.deadline = now + 600
    t_c.deadline = now + 600

    plan = [t_a, t_b, t_c]
    calc = PlanningCalculations()

    # --- Resource margin (no monitor → conservative default) ---
    rm = calc.calculate_resource_margin(plan)
    printer.status("CALC", f"Resource margin (no monitor): {rm:.3f}", "info")
    assert 0.0 <= rm <= 1.0, "Resource margin out of range"

    # --- Temporal margin ---
    tm = calc.calculate_temporal_margin(plan)
    printer.status("CALC", f"Temporal margin: {tm:.3f}", "info")
    assert 0.0 <= tm <= 1.0, "Temporal margin out of range"

    # --- Plan duration (critical path) ---
    dur = calc.calculate_plan_duration(plan)
    printer.status("CALC", f"Plan duration (CPM): {dur:.1f}s", "info")
    assert dur == 220.0, f"Expected 220s sequential, got {dur}s"

    # --- Remaining time (no tasks executing) ---
    rem = calc.estimate_remaining_time(plan)
    printer.status("CALC", f"Remaining time: {rem:.1f}s", "info")
    assert rem == dur

    # --- Remaining time with one task already executing ---
    t_a.status     = TaskStatus.EXECUTING
    t_a.start_time = now - 30.0          # 30s into a 60s task
    rem2 = calc.estimate_remaining_time(plan)
    printer.status("CALC", f"Remaining time (A executing 30s): {rem2:.1f}s", "info")
    assert rem2 < rem, "Remaining time should decrease while a task executes"
    t_a.status     = TaskStatus.PENDING
    t_a.start_time = 0.0

    # --- Dependency risk ---
    dr = calc.calculate_dependency_risk(plan)
    printer.status("CALC", f"Dependency risk margin: {dr:.3f}", "info")
    assert 0.0 <= dr <= 1.0

    # --- Probability of success ---
    t_b.success_threshold = 0.85
    ps = calc.calculate_probability_of_success(t_b)
    printer.status("CALC", f"P(success) for CoreProcess: {ps:.3f}", "info")
    assert ps == 0.85

    # History-based success rate overrides threshold
    t_b.history = [{"outcome": "success"}, {"outcome": "success"}, {"outcome": "failure"}]
    ps_hist = calc.calculate_probability_of_success(t_b)
    printer.status("CALC", f"P(success) from history (2/3): {ps_hist:.3f}", "info")
    assert abs(ps_hist - 2/3) < 1e-9
    t_b.history = []

    # --- Risk score ---
    t_c.risk_score = 0.4
    rs = calc.estimate_risk_score(t_c)
    printer.status("CALC", f"Risk score for Postprocess: {rs:.3f}", "info")
    assert 0.0 <= rs <= 1.0

    # --- Plan cost ---
    cost = calc.calculate_plan_cost(plan)
    printer.status("CALC", f"Plan cost: {cost:.1f}", "info")
    assert cost == 8.5

    # --- Full safety margin report ---
    margins = calc.check_safety_margins(plan)
    printer.status("CALC", f"Safety margins: {margins}", "info")
    assert set(margins.keys()) == {"resource", "temporal", "dependency"}

    # --- Risk profile ---
    profile = calc.calculate_plan_risk_profile(plan)
    printer.status("CALC", f"Overall risk: {profile['overall_risk']:.3f}", "info")
    assert "per_task" in profile and len(profile["per_task"]) == 3

    # --- Cache ---
    calc._cache_set("test_key", 42)
    assert calc._cache_get("test_key") == 42
    calc.clear_cache()
    assert calc._cache_get("test_key") is None
    printer.status("CALC", "Cache set/get/clear: OK", "success")

    print("\n=== Test ran successfully ===\n")
