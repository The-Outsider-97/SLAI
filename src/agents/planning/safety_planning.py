"""
Safety Planning – production-grade safety orchestration for the planning stack.

This module owns high-level safety policy decisions and recovery orchestration.
It deliberately delegates low-level numeric work to PlanningCalculations,
resource telemetry/reservation to ResourceMonitor, persistence to PlanningMemory,
and validation/serialisation primitives to planning_helpers.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
import requests  # type: ignore

from queue import PriorityQueue
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from requests.exceptions import RequestException  # type: ignore

from .utils.config_loader import load_global_config, get_config_section
from .utils.planning_errors import *
from .utils.planning_helpers import *
from .utils.planning_calculations import PlanningCalculations
from .utils.resource_monitor import ResourceMonitor
from .planning_types import (
    ClusterResources,
    RepairCandidate,
    ResourceProfile,
    SafetyMargins,
    SafetyViolation,
    Task,
    TaskStatus,
    TaskType,
)
from .planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Safety Planning")
printer = PrettyPrinter()


class DistributedOrchestrator:
    """Deterministic decomposition helper for distributed repair strategies."""

    def __init__(self) -> None:
        self.do_config = get_config_section("distributed_decomposition") or {}
        self.max_horizontal_splits = int(self.do_config.get("max_horizontal_splits", 4))
        require_positive(self.max_horizontal_splits, "distributed.max_horizontal_splits")
        self.vertical_stage_count = int(self.do_config.get("vertical_stage_count", 3))
        require_positive(self.vertical_stage_count, "distributed.vertical_stage_count")
        self.split_duration_factor = float(self.do_config.get("split_duration_factor", 0.50))
        self.split_resource_factor = float(self.do_config.get("split_resource_factor", 0.50))

        self.resource_monitor = ResourceMonitor()
        validate_probability(clamp(self.split_duration_factor, 0.0, 1.0), "distributed.split_duration_factor")
        validate_probability(clamp(self.split_resource_factor, 0.0, 1.0), "distributed.split_resource_factor")

    def horizontal_split(self, task: Task) -> List[Task]:
        """Split a task into parallel shards with reduced resource pressure."""
        require_type(task, Task, "task")
        shard_count = min(self.max_horizontal_splits, max(2, int(getattr(task, "workload", 0) or 2)))
        result: List[Task] = []
        for index in range(shard_count):
            shard = task.copy()
            shard.id = f"{task.id}_h{index + 1}"
            shard.name = f"{task.name}_horizontal_{index + 1}"
            shard.parent = task
            shard.parent_task = task
            shard.dependencies = list(getattr(task, "dependencies", []) or [])
            shard.duration = max(1.0, float(task.duration or 0.0) * self.split_duration_factor)
            shard.estimated_duration = max(1.0, float(task.estimated_duration or shard.duration) * self.split_duration_factor)
            shard.resource_requirements = self._scaled_resource_profile(
                task.resource_requirements,
                self.split_resource_factor,
            )
            shard.status = TaskStatus.PENDING
            result.append(shard)
        return result

    def vertical_split(self, task: Task) -> List[Task]:
        """Split a task into ordered preparation/execution/verification stages."""
        require_type(task, Task, "task")
        stage_names = ["prepare", "execute", "verify"][: self.vertical_stage_count]
        if len(stage_names) < self.vertical_stage_count:
            stage_names.extend(f"stage_{i}" for i in range(len(stage_names) + 1, self.vertical_stage_count + 1))

        result: List[Task] = []
        previous_id: Optional[str] = None
        duration = max(1.0, float(task.duration or task.estimated_duration or 1.0) / len(stage_names))
        for index, stage in enumerate(stage_names):
            subtask = task.copy()
            subtask.id = f"{task.id}_v{index + 1}"
            subtask.name = f"{task.name}_{stage}"
            subtask.parent = task
            subtask.parent_task = task
            subtask.dependencies = list(getattr(task, "dependencies", []) or [])
            if previous_id:
                subtask.dependencies.append(previous_id)
            subtask.duration = duration
            subtask.estimated_duration = duration
            subtask.resource_requirements = self._scaled_resource_profile(
                task.resource_requirements,
                1.0 if stage == "execute" else self.split_resource_factor,
            )
            subtask.status = TaskStatus.PENDING
            result.append(subtask)
            previous_id = subtask.id
        return result

    def hybrid_split(self, task: Task) -> List[Task]:
        """Combine vertical control stages with horizontal execution shards."""
        vertical = self.vertical_split(task)
        if len(vertical) < 2:
            return self.horizontal_split(task)
        execution_stage = vertical[min(1, len(vertical) - 1)]
        shards = self.horizontal_split(execution_stage)
        shards[0].dependencies = list(vertical[0].dependencies) + [vertical[0].id]
        for shard in shards[1:]:
            shard.dependencies = list(vertical[0].dependencies) + [vertical[0].id]
        if len(vertical) > 2:
            vertical[-1].dependencies = [shard.id for shard in shards]
            return [vertical[0], *shards, vertical[-1]]
        return [vertical[0], *shards]

    def decompose_and_distribute(self, task: Task, strategy: Optional[str] = None) -> List[Task]:
        """Return a decomposition using the requested strategy."""
        selected = (strategy or self.do_config.get("default_strategy", "hybrid")).lower()
        if selected == "horizontal":
            return self.horizontal_split(task)
        if selected == "vertical":
            return self.vertical_split(task)
        if selected == "hybrid":
            return self.hybrid_split(task)
        raise DecompositionError(
            f"Unsupported distributed decomposition strategy: {selected}",
            task_name=getattr(task, "name", ""),
            task_id=getattr(task, "id", ""),
            attempted_methods=[selected],
        )

    @staticmethod
    def _scaled_resource_profile(profile: ResourceProfile, factor: float) -> ResourceProfile:
        return ResourceProfile(
            gpu=max(0.0, float(profile.gpu) * factor),
            ram=max(0.0, float(profile.ram) * factor),
            specialized_hardware=list(profile.specialized_hardware),
        )


class SafetyPlanning:
    """
    Safety orchestration layer for planning and execution.

    Public compatibility surface
    ----------------------------
    - ``safety_check(plan)`` validates a candidate plan.
    - ``interactive_adjustment_handler(adjustment)`` applies safe plan changes.
    - ``dynamic_replanning_pipeline(failed_task)`` produces a repair plan.
    - ``update_allocations(task)`` reserves resources through ResourceMonitor.
    """

    def __init__(self, *, memory: Optional[PlanningMemory] = None,
        calculations: Optional[PlanningCalculations] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        distributed_orchestrator: Optional[DistributedOrchestrator] = None,
    ) -> None:
        self.config = load_global_config()
        self.ram_limit = self.config.get("ram_limit")
        self.gpu_limit = self.config.get("gpu_limit")

        self.safety_config = get_config_section("safety_planning", config=self.config, default={})
        self.margins_config = get_config_section("safety_margins", config=self.config, default={})
        self.resource_buffers = dict(self.margins_config.get("resource_buffers", {}))
        self.temporal = dict(self.margins_config.get("temporal", {}))
        self.safety_margin_model = SafetyMargins.from_config(self.config)

        self._validate_config()

        self.lock = threading.RLock()
        self.memory = memory or PlanningMemory()
        self.calculations = calculations or PlanningCalculations()
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.calculations.set_resource_monitor(self.resource_monitor)

        self.distributed_orchestrator = distributed_orchestrator or DistributedOrchestrator()
        self.adjustment_queue: PriorityQueue[Tuple[int, float, str, Dict[str, Any]]] = PriorityQueue(
            maxsize=int(self.safety_config.get("queue_max_size", 1000))
        )

        self.current_plan: List[Task] = []
        self.task_library: Dict[str, Task] = {}
        self.violation_history: List[SafetyViolation] = []
        self.current_violations: List[SafetyViolation] = []
        self.safety_policies: Dict[str, Callable[[List[Task]], bool]] = {}
        self.base_state: Dict[str, Any] = {"execution_history": []}
        self.last_safety_report: Dict[str, Any] = {}
        self.last_repair_candidates: List[RepairCandidate] = []
        self.last_adjustment_log: List[Dict[str, Any]] = []

        logger.info("Safety Planning successfully initialized")

    @property
    def safety_margins(self) -> Dict[str, Any]:
        """Backward-compatible access to the configured safety margins."""
        return self.margins_config

    # ------------------------------------------------------------------
    # Interactive adjustment flow
    # ------------------------------------------------------------------
    def interactive_adjustment_handler(self, adjustment: Dict[str, Any]) -> None:
        """Validate and apply a runtime plan adjustment when it remains safe."""
        printer.status("SAFETY", "Processing adjustment", "info")
        if not adjustment or not isinstance(adjustment, dict):
            raise AdjustmentError("Adjustment must be a non-empty dictionary", adjustment=adjustment or {})
    
        with self.lock:
            try:
                self._validate_adjustment(adjustment)
                adjusted_plan = self._apply_adjustment(self.current_plan, adjustment)
                self.safety_check(adjusted_plan, raise_on_failure=True)
                self.current_plan = adjusted_plan
                self.log_adjustment(adjustment)
                if self.safety_config.get("auto_checkpoint_on_repair", True):
                    self._checkpoint("adjustment_applied", {"adjustment": adjustment})
            except (AdjustmentError, ResourceViolation, TemporalError, PlanningError) as exc:
                self.handle_adjustment_failure(exc, adjustment)
                raise  # Re-raise after handling, so caller knows it failed
            except Exception as exc:
                self.handle_adjustment_failure(exc, adjustment)
                raise AdjustmentError("Unexpected error during adjustment", adjustment=adjustment, conflict_details={"error": str(exc)}) from exc

    def _apply_adjustment(self, current_plan: List[Task], adjustment: Dict[str, Any]) -> List[Task]:
        """Return a safely modified copy of the current plan."""
        require_type(current_plan, list, "current_plan")
        adj_type = adjustment.get("type")
        adjusted_plan = [self._copy_task(task) for task in current_plan]

        if adj_type == "modify_task":
            task_id = str(adjustment.get("task_id", ""))
            validate_task_id(task_id, "modify_task")
            updates = dict(adjustment.get("updates") or {})
            for task in adjusted_plan:
                if task.id == task_id or task.name == task_id:
                    self._apply_task_updates(task, updates)
                    task.validate()
                    return adjusted_plan
            raise AdjustmentError(
                f"Task {task_id!r} not found for modification",
                adjustment=adjustment,
                conflict_details={"current_task_ids": [task.id for task in adjusted_plan]},
            )

        if adj_type == "add_task":
            new_task = self._coerce_task(adjustment.get("task"), "adjustment.task")
            if any(task.id == new_task.id for task in adjusted_plan):
                raise AdjustmentError(
                    f"Task ID {new_task.id!r} already exists",
                    adjustment=adjustment,
                    conflict_details={"duplicate_task_id": new_task.id},
                )
            self._validate_dependencies_known(new_task, adjusted_plan, allow_existing_only=True)
            adjusted_plan.append(new_task)
            return adjusted_plan

        if adj_type == "remove_task":
            task_id = str(adjustment.get("task_id", ""))
            validate_task_id(task_id, "remove_task")
            cascade = bool(adjustment.get("cascade", False))
            removed_ids = {task_id}
            if cascade:
                removed_ids |= set(self._dependent_ids(adjusted_plan, task_id))
            else:
                dependents = self._dependent_ids(adjusted_plan, task_id)
                if dependents:
                    raise AdjustmentError(
                        f"Cannot remove task {task_id!r}; dependent tasks exist",
                        adjustment=adjustment,
                        conflict_details={"dependents": dependents},
                    )
            result = [task for task in adjusted_plan if task.id not in removed_ids and task.name not in removed_ids]
            if len(result) == len(adjusted_plan):
                raise AdjustmentError(
                    f"Task {task_id!r} not found for removal",
                    adjustment=adjustment,
                    conflict_details={"current_task_ids": [task.id for task in adjusted_plan]},
                )
            return result

        raise AdjustmentError(
            f"Unsupported adjustment type: {adj_type}",
            adjustment=adjustment,
            conflict_details={"allowed": self.safety_config.get("allowed_update_fields", [])},
        )

    def _validate_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """Validate adjustment structure and high-level constraints."""
        require_type(adjustment, dict, "adjustment")
        adj_type = adjustment.get("type")
        if adj_type not in {"modify_task", "add_task", "remove_task"}:
            raise AdjustmentError(
                f"Invalid adjustment type: {adj_type}",
                adjustment=adjustment,
                conflict_details={"allowed": ["modify_task", "add_task", "remove_task"]},
            )

        if adj_type in {"modify_task", "remove_task"}:
            task_id = adjustment.get("task_id")
            if not task_id:
                raise AdjustmentError("Missing task_id", adjustment=adjustment)
            validate_task_id(str(task_id), f"{adj_type}.task_id")

        if adj_type == "modify_task":
            self._validate_modification(adjustment)
        elif adj_type == "add_task":
            self._validate_addition(adjustment)
        elif adj_type == "remove_task":
            self._validate_removal(adjustment)

    def _validate_modification(self, adjustment: Dict[str, Any]) -> None:
        updates = adjustment.get("updates")
        if not isinstance(updates, dict) or not updates:
            raise AdjustmentError("No updates specified for task modification", adjustment=adjustment)
        allowed = set(self.safety_config.get("allowed_update_fields", []))
        invalid = sorted(set(updates) - allowed)
        if invalid:
            raise AdjustmentError(
                f"Invalid update field(s): {invalid}",
                adjustment=adjustment,
                conflict_details={"invalid_fields": invalid, "allowed_fields": sorted(allowed)},
            )
        target = self._find_task(self.current_plan, str(adjustment.get("task_id")))
        if target is None:
            raise AdjustmentError(
                f"Task {adjustment.get('task_id')!r} not found in current plan",
                adjustment=adjustment,
                conflict_details={"current_task_ids": [task.id for task in self.current_plan]},
            )

        if "deadline" in updates and float(updates["deadline"]) > 0:
            self._validate_deadline_value(float(updates["deadline"]), adjustment)
        if "resource_requirements" in updates or "requirements" in updates:
            profile = self._coerce_resource_profile(
                updates.get("resource_requirements", updates.get("requirements"))
            )
            preview = self._copy_task(target)
            preview.resource_requirements = profile
            self._validate_equipment_constraints(preview)
            self._validate_safety_margins(preview)

    def _validate_addition(self, adjustment: Dict[str, Any]) -> None:
        task = self._coerce_task(adjustment.get("task"), "adjustment.task")
        task.validate()
        if self._find_task(self.current_plan, task.id) is not None:
            raise AdjustmentError(
                f"Task ID {task.id!r} already exists",
                adjustment=adjustment,
                conflict_details={"duplicate_task_id": task.id},
            )
        self._validate_dependencies_known(task, self.current_plan, allow_existing_only=True)
        self._validate_temporal_constraints(task)
        self._validate_equipment_constraints(task)
        self._validate_safety_margins(task)

    def _validate_removal(self, adjustment: Dict[str, Any]) -> None:
        task_id = str(adjustment.get("task_id"))
        if self._find_task(self.current_plan, task_id) is None:
            raise AdjustmentError(
                f"Task {task_id!r} not found in current plan",
                adjustment=adjustment,
                conflict_details={"current_task_ids": [task.id for task in self.current_plan]},
            )
        if not bool(adjustment.get("cascade", False)):
            dependents = self._dependent_ids(self.current_plan, task_id)
            if dependents:
                raise AdjustmentError(
                    f"Cannot remove task {task_id!r} with dependents",
                    adjustment=adjustment,
                    conflict_details={"dependents": dependents},
                )

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    def safety_check(self, plan: List[Task], *, raise_on_failure: bool = False) -> bool:
        """
        Validate structural, resource, temporal, dependency, and custom safety rules.

        Returns True when safe. When ``raise_on_failure`` is True, the first
        structured planning error is propagated after violation records are built.
        """
        with self.lock:
            self.current_violations = []
            try:
                safe_plan = self._normalise_plan(plan)
                if not self.safety_config.get("enabled", True):
                    self.last_safety_report = {"enabled": False, "safe": True}
                    return True
    
                try:
                    self._validate_plan_structure(safe_plan)
                except (CyclicDependencyError, DecompositionError, SchedulingConflictError) as e:
                    self._record_violation("structural", e.__class__.__name__, 1.0, 0.0, "", severity="critical")
                    if raise_on_failure:
                        raise
                    return False
    
                try:
                    available = self.resource_monitor.get_available_resources()
                    margin_report = self.calculations.check_safety_margins(safe_plan, available)
                    self._validate_margin_report(margin_report, safe_plan)
                except (ResourceViolation, SafetyMarginError) as e:
                    self._record_violation("resource", e.resource_type, getattr(e, "measured_utilisation", 1.0), 0.0, "")
                    if raise_on_failure:
                        raise
                    return False
    
                for task in safe_plan:
                    try:
                        self._validate_temporal_constraints(task)
                        self._validate_equipment_constraints(task)
                        self._validate_safety_margins(task, available)
                        self._validate_task_risk(task)
                    except (TemporalViolation, DeadlineExceededError, ResourceViolation) as e:
                        self._record_violation("task", task.id, 1.0, 0.0, task.id)
                        if raise_on_failure:
                            raise
                        return False
    
                # Custom policies
                for name, policy in list(self.safety_policies.items()):
                    try:
                        if not bool(policy(safe_plan)):
                            self._record_violation("policy", name, 1.0, 0.0, "")
                            if raise_on_failure:
                                raise SafetyMarginError(f"Policy {name} failed", resource_type="policy")
                    except Exception as e:
                        logger.warning("Safety policy %s raised %s", name, e)
    
                risk_profile = self.calculations.calculate_plan_risk_profile(safe_plan)
                max_risk = float(self.safety_config.get("max_plan_risk_score", 0.90))
                if risk_profile["overall_risk"] > max_risk:
                    self._record_violation("risk", "plan", risk_profile["overall_risk"], max_risk, "")
                    if raise_on_failure:
                        raise SafetyMarginError(f"Plan risk {risk_profile['overall_risk']:.2f} exceeds threshold {max_risk:.2f}", resource_type="risk")
    
                self.last_safety_report = {
                    "safe": not self.current_violations,
                    "timestamp": time.time(),
                    "task_count": len(safe_plan),
                    "margins": margin_report,
                    "risk_profile": risk_profile,
                    "violations": [v.to_dict() for v in self.current_violations],
                }
                return not self.current_violations
            except PlanningError:
                if raise_on_failure:
                    raise
                return False
            except Exception as e:
                logger.error("Unexpected error in safety_check: %s", e, exc_info=True)
                if raise_on_failure:
                    raise PlanningError("Safety check failed unexpectedly", context={"original_error": str(e)}) from e
                return False

    def _validate_plan_structure(self, plan: List[Task]) -> None:
        ids: List[str] = []
        for task in plan:
            task.validate()
            ids.append(task.id)

        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        if duplicates:
            raise PlanningConfigError(
                f"Duplicate task IDs in plan: {duplicates}",
                config_key="plan.task_ids",
                config_section="safety_planning",
                expected_type="unique task IDs",
            )

        dep_map = build_dependency_map(plan)
        cycle = detect_cycles(ids, dep_map)
        if cycle:
            raise CyclicDependencyError("Plan contains cyclic dependencies", cycle_path=cycle)

        known = set(ids)
        missing: Dict[str, List[str]] = {
            task.id: [dep for dep in task.dependencies if dep not in known]
            for task in plan
            if any(dep not in known for dep in task.dependencies)
        }
        if missing:
            raise DecompositionError(
                "Plan contains missing dependencies",
                attempted_methods=["dependency_validation"],
                context={"missing_dependencies": missing},
            )

        max_concurrent = int(self.safety_margin_model.max_concurrent)
        concurrent = sum(1 for task in plan if getattr(task, "status", None) == TaskStatus.EXECUTING)
        if concurrent > max_concurrent:
            raise SchedulingConflictError(
                f"Concurrent task limit exceeded: {concurrent}>{max_concurrent}",
                conflicting_task_ids=[task.id for task in plan if task.status == TaskStatus.EXECUTING],
                conflict_type="max_concurrent",
            )

    def _validate_margin_report(self, margins: Dict[str, float], plan: List[Task]) -> None:
        thresholds = dict(self.safety_config.get("min_margin_thresholds", {}))
        for name, value in margins.items():
            threshold = float(thresholds.get(name, 0.0))
            if float(value) < threshold:
                self._record_violation(
                    "margin",
                    name,
                    float(value),
                    threshold,
                    ",".join(task.id for task in plan[:3]),
                    severity="critical" if value <= 0.0 else "high",
                    corrective_action=f"improve_{name}_margin",
                    impact_analysis={"margins": margins},
                )

    def _validate_temporal_constraints(self, task: Task) -> None:
        """Validate task-level time windows and deadlines."""
        require_type(task, Task, "task")
        now = time.time()
        tc = task.temporal_constraints
        if tc is not None:
            if not tc.validate(now):
                raise TemporalViolation(
                    f"Temporal constraints failed for task {task.name}",
                    violation_type="window",
                    task_name=task.name,
                    task_id=task.id,
                    constraint_details=tc.to_dict() if hasattr(tc, "to_dict") else {},
                )
    
        duration = max(float(getattr(task, "estimated_duration", 0.0) or getattr(task, "duration", 0.0) or 0.0), 0.0)
        min_duration = float(self.safety_margin_model.min_task_duration)
        if duration > 0.0 and duration < min_duration:
            self._record_violation(
                "temporal",
                "min_task_duration",
                duration,
                min_duration,
                task.id,
                severity="low",
                corrective_action="increase_estimated_duration_or_confirm_fast_task",
            )
    
        start = float(getattr(task, "start_time", 0.0) or time.time())
        deadline = float(getattr(task, "deadline", 0.0) or 0.0)
        if deadline > 0.0:
            projected = estimate_end_time(start, duration, time_buffer=float(self.safety_margin_model.time_buffer))
            if projected > deadline:
                raise DeadlineExceededError(
                    f"Task {task.name} would miss its deadline",
                    task_name=task.name,
                    task_id=task.id,
                    deadline=deadline,
                    projected_completion=projected,
                )

    def _validate_safety_margins(
        self,
        task: Task,
        available: Optional[ClusterResources] = None,
    ) -> Dict[str, float]:
        """Validate a single task's numeric resource safety margins."""
        require_type(task, Task, "task")
        available = available or self.resource_monitor.get_available_resources()
        requirements = task.resource_requirements
        margins = check_resource_feasibility(
            {"gpu": float(requirements.gpu), "ram": float(requirements.ram)},
            {"gpu": float(available.gpu_total), "ram": float(available.ram_total)},
            safety_buffers={
                "gpu": float(self.safety_margin_model.gpu_buffer),
                "ram": float(self.safety_margin_model.ram_buffer),
            },
            task_id=task.id,
        )
        for resource, margin in margins.items():
            if margin <= 0.0:
                self._record_violation(
                    "resource",
                    resource,
                    0.0,
                    float(self.safety_config.get("min_margin_thresholds", {}).get("resource", 0.05)),
                    task.id,
                    severity="critical",
                    corrective_action="reduce_resource_request_or_reallocate",
                    impact_analysis={"requirements": requirements.to_dict()},
                )
        return margins

    def _validate_equipment_constraints(self, task: Task) -> None:
        """Validate specialized hardware requirements."""
        require_type(task, Task, "task")
        available = self.resource_monitor.get_available_resources()
        required = set(task.resource_requirements.specialized_hardware or [])
        available_hw = set(available.specialized_hardware_available or [])
        missing = sorted(required - available_hw)
        if missing:
            self._record_violation(
                "resource",
                "specialized_hardware",
                len(missing),
                0.0,
                task.id,
                severity="critical",
                corrective_action="route_task_to_capable_node_or_change_method",
                impact_analysis={"missing": missing, "available": sorted(available_hw)},
            )
            raise ResourceViolation(
                f"Missing specialized hardware for task {task.name}: {missing}",
                resource_type="specialized_hardware",
                requested=missing,
                available=sorted(available_hw),
                task_id=task.id,
            )

    def _validate_task_risk(self, task: Task) -> None:
        limit = float(self.safety_config.get("max_task_risk_score", 0.95))
        risk = float(getattr(task, "risk_score", 0.0) or 0.0)
        if risk > limit:
            self._record_violation(
                "risk",
                "task",
                risk,
                limit,
                task.id,
                severity="high",
                corrective_action="choose_lower_risk_method",
                impact_analysis={"task_name": task.name},
            )

    # ------------------------------------------------------------------
    # Failure handling and diagnostics
    # ------------------------------------------------------------------
    def handle_adjustment_failure(self, e: Exception, adjustment: Dict[str, Any]) -> None:
        """Record diagnostics and route failure to a recovery strategy."""
        printer.status("ADJUST-FAIL", f"Adjustment failed: {str(e)}", "error")
        failure_data = {
            "timestamp": time.time(),
            "adjustment": copy.deepcopy(adjustment),
            "error_type": type(e).__name__,
            "message": str(e),
            "current_plan": [task.id for task in self.current_plan],
            "resource_status": self._cluster_to_dict(self.resource_monitor.get_available_resources()),
        }

        if isinstance(e, ResourceViolation):
            self._handle_resource_failure(e, adjustment)
        elif isinstance(e, TemporalError):
            self._handle_temporal_failure(e, adjustment)
        else:
            self._handle_generic_failure(e, adjustment)

        self._send_alert_notification(failure_data)
        if self.safety_config.get("auto_checkpoint_on_failure", True):
            self._checkpoint(f"adjust_fail_{adjustment.get('type', 'unknown')}", failure_data)

    def _handle_resource_failure(self, e: ResourceViolation, adjustment: Dict[str, Any]) -> None:
        """Queue or decompose adjustments that cannot currently acquire resources."""
        if self._attempt_resource_scaling(e.requested, e.available):
            adjustment["_retry_count"] = int(adjustment.get("_retry_count", 0)) + 1
            self._queue_for_later_execution(adjustment)
            return

        if adjustment.get("type") == "add_task" and adjustment.get("task") is not None:
            try:
                task = self._coerce_task(adjustment.get("task"), "adjustment.task")
                subtasks = self.distributed_decomposition(task)
                if subtasks:
                    queued = {
                        "type": "add_subtasks",
                        "subtasks": subtasks,
                        "origin": adjustment.get("origin", "safety_repair"),
                        "priority": adjustment.get("priority", 3),
                    }
                    self._queue_for_later_execution(queued)
                    return
            except PlanningError as exc:
                logger.warning("Resource fallback decomposition failed: %s", exc)

        self._queue_for_later_execution(adjustment)

    def _handle_temporal_failure(self, e: TemporalError, adjustment: Dict[str, Any]) -> None:
        """Apply temporal fallback policies without mutating unsafe live state."""
        if "updates" in adjustment and "deadline" in adjustment.get("updates", {}):
            extended = copy.deepcopy(adjustment)
            original = float(extended["updates"]["deadline"])
            extended["updates"]["deadline"] = original + float(self.safety_margin_model.time_buffer) * 2.0
            extended["_retry_count"] = int(extended.get("_retry_count", 0)) + 1
            self._queue_for_later_execution(extended)
            return

        if self._reallocate_resources_for_priority(adjustment):
            return

        self._enable_partial_execution(adjustment)

    def _handle_generic_failure(self, e: Exception, adjustment: Dict[str, Any]) -> None:
        """Use bounded retry metadata and queue unsafe adjustments for later."""
        retries = int(adjustment.get("_retry_count", 0))
        max_retries = int(self.safety_config.get("max_adjustment_retries", 3))
        if retries < max_retries:
            retry = copy.deepcopy(adjustment)
            retry["_retry_count"] = retries + 1
            retry["not_before"] = time.time() + compute_backoff_delay(
                retry["_retry_count"],
                base_delay=float(self.safety_config.get("retry_base_delay", 0.25)),
                backoff_factor=float(self.safety_config.get("retry_backoff_factor", 2.0)),
                max_delay=float(self.safety_config.get("max_retry_delay", 5.0)),
            )
            self._queue_for_later_execution(retry)
            return
        self._escalate_to_human_operator(adjustment, str(e))

    def _attempt_resource_scaling(self, required: Any, available: Any) -> bool:
        """Hook for cluster autoscaling; returns False unless explicitly enabled."""
        if not self.safety_config.get("resource_scaling_enabled", False):
            return False
        logger.info(
            "Resource scaling requested: required=%s available=%s",
            truncate_for_logging(required),
            truncate_for_logging(available),
        )
        return False

    def _reallocate_resources_for_priority(self, adjustment: Dict[str, Any]) -> bool:
        """Release resources from lower-priority tasks when a higher-priority task needs them."""
        task_id = str(adjustment.get("task_id", ""))
        target = self._find_task(self.current_plan, task_id) if task_id else None
        if target is None:
            return False

        victims = self._find_resource_victims(target)
        if not victims:
            return False

        for victim in victims:
            self.resource_monitor.release_resources(victim.id)
            victim.update_status(TaskStatus.BLOCKED, reason="resources_reallocated")
        self._queue_for_later_execution(adjustment)
        return True

    def _queue_for_later_execution(self, adjustment: Dict[str, Any]) -> None:
        """Queue an adjustment with bounded priority and stable ordering."""
        priority = int(adjustment.get("priority", 3))
        record = (priority, float(adjustment.get("not_before", time.time())), uuid.uuid4().hex, copy.deepcopy(adjustment))
        try:
            self.adjustment_queue.put_nowait(record)
        except Exception as exc:
            raise AdjustmentError(
                "Adjustment queue is full",
                adjustment=adjustment,
                conflict_details={"queue_max_size": self.safety_config.get("queue_max_size")},
            ) from exc

    def _enable_partial_execution(self, adjustment: Dict[str, Any]) -> None:
        """Mark a failed adjustment for partial execution when policy allows it."""
        if not self.safety_config.get("partial_execution_enabled", True):
            self._escalate_to_human_operator(adjustment, "partial_execution_disabled")
            return
        partial = copy.deepcopy(adjustment)
        partial["execution_mode"] = "partial"
        partial["not_before"] = time.time()
        self._queue_for_later_execution(partial)

    def _escalate_to_human_operator(self, adjustment: Dict[str, Any], error_msg: str) -> None:
        """Create an escalation payload without assuming an external operator exists."""
        payload = {
            "timestamp": time.time(),
            "type": "human_escalation",
            "enabled": bool(self.safety_config.get("human_escalation_enabled", False)),
            "adjustment": copy.deepcopy(adjustment),
            "error": error_msg,
        }
        logger.warning("Safety escalation created: %s", truncate_for_logging(payload))
        self._publish_adjustment_event(payload)

    def _send_alert_notification(self, payload: Dict[str, Any]) -> None:
        """Send an optional webhook alert; failures are logged and non-fatal."""
        url = str(self.safety_config.get("alert_webhook_url", "") or "").strip()
        if not url:
            return
        try:
            requests.post(
                url,
                data=safe_json_dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=float(self.safety_config.get("alert_timeout_seconds", 2.0)),
            )
        except RequestException as exc:
            logger.warning("Safety alert notification failed: %s", exc)

    # ------------------------------------------------------------------
    # Distributed decomposition and allocation
    # ------------------------------------------------------------------
    def distributed_decomposition(self, task: Task) -> List[Task]:
        """Decompose a task using registered methods or the distributed orchestrator."""
        require_type(task, Task, "task")
        if task.task_type == TaskType.ABSTRACT and task.methods:
            method_index = int(getattr(task, "selected_method", 0) or 0)
            subtasks = task.get_subtasks(method_index=method_index)
        else:
            strategy = self.safety_config.get("distributed_decomposition", {}).get("default_strategy", "hybrid")
            subtasks = self.distributed_orchestrator.decompose_and_distribute(task, strategy=strategy)

        for subtask in subtasks:
            self._reset_temporal_attributes(subtask)
            subtask.validate()
        return subtasks

    def _reset_temporal_attributes(self, task: Task) -> None:
        """Reset runtime timing on a cloned subtask while preserving deadlines."""
        task.start_time = 0.0
        task.end_time = 0.0
        task.status = TaskStatus.PENDING
        task.actual_duration = 0.0
        task.progress = 0.0
        task.last_updated = time.time()

    def update_allocations(self, task: Task) -> bool:
        """Reserve resources for a task through ResourceMonitor."""
        require_type(task, Task, "task")
        try:
            result = self.resource_monitor.acquire_resources(task.resource_requirements, task_id=task.id)
            if result:
                task.update_status(TaskStatus.EXECUTING)
            return bool(result)
        except ResourceAcquisitionError:
            # Already a proper PlanningError – let it propagate
            raise
        except ResourceViolation as exc:
            raise ResourceAcquisitionError(
                f"Failed to allocate resources for {task.name}: {exc}",
                resource_type=exc.resource_type,
                requested=exc.requested,
                available=exc.available,
                task_id=task.id,
            ) from exc
        except Exception as e:
            raise ResourceAcquisitionError(
                f"Unexpected error allocating resources for {task.name}",
                resource_type="unknown",
                requested=task.resource_requirements.to_dict(),
                available={},
                task_id=task.id,
            ) from e

    # ------------------------------------------------------------------
    # Replanning and repair selection
    # ------------------------------------------------------------------
    def dynamic_replanning_pipeline(self, failed_task: Task) -> List[Task]:
        """Generate and apply the lowest-risk repair plan for a failed task."""
        require_type(failed_task, Task, "failed_task")
        with self.lock:
            failed_task.mark_failed(getattr(failed_task, "failure_reason", "") or "dynamic_replanning_requested")
            candidates = self._generate_repair_candidates(failed_task)
            self.last_repair_candidates = candidates
            if not candidates:
                raise ReplanningError(
                    "No repair candidates generated",
                    failed_task=failed_task,
                    candidates=[],
                    failure_reason="no_candidates",
                )

            best = self._select_optimal_repair(candidates)
            if best is None:
                raise ReplanningError(
                    "No safe repair candidate selected",
                    failed_task=failed_task,
                    candidates=candidates,
                    failure_reason="no_safe_candidate",
                )

            if not self.safety_check(best.repaired_plan):
                raise ReplanningError(
                    "Selected repair failed safety validation",
                    failed_task=failed_task,
                    candidates=candidates,
                    failure_reason="selected_candidate_unsafe",
                )

            self.current_plan = best.repaired_plan
            if self.safety_config.get("auto_checkpoint_on_repair", True):
                self._checkpoint(
                    f"repair_{best.strategy}",
                    {"failed_task": failed_task.id, "candidate": best.to_dict()},
                )
            return self.current_plan

    def _generate_repair_candidates(self, failed_task: Task) -> List[RepairCandidate]:
        """Build repair candidates using retry, prune, decomposition, and reallocation strategies."""
        base_plan = [self._copy_task(task) for task in self.current_plan]
        candidates: List[RepairCandidate] = []

        if failed_task.remaining_retries > 0:
            retry_plan = self._replace_task(
                base_plan,
                failed_task.id,
                self._retry_task(failed_task),
            )
            candidates.append(self._make_repair_candidate("retry_failed_task", retry_plan, failed_task))

        pruned = self._create_pruned_plan(base_plan, failed_task)
        candidates.append(self._make_repair_candidate("prune_failed_branch", pruned, failed_task))

        try:
            decomposition = self.distributed_decomposition(failed_task)
            if decomposition:
                repaired = self._create_repair_plan(failed_task, decomposition, self._get_dependent_tasks(base_plan, failed_task))
                candidates.append(self._make_repair_candidate("distributed_decomposition", repaired, failed_task))
        except PlanningError as exc:
            logger.debug("Repair decomposition skipped: %s", exc)

        victims = self._find_resource_victims(failed_task)
        if victims:
            repaired = self._create_repair_plan(failed_task, [self._retry_task(failed_task)], victims)
            candidates.append(self._make_repair_candidate("resource_reallocation", repaired, failed_task, victims=victims))

        return candidates

    def _create_pruned_plan(self, original_plan: List[Task], failed_task: Task) -> List[Task]:
        """Remove a failed task and its dependent branch from a plan."""
        remove_ids = {failed_task.id}
        remove_ids |= set(self._dependent_ids(original_plan, failed_task.id))
        return [task for task in original_plan if task.id not in remove_ids]

    def _get_dependent_tasks(self, plan: List[Task], task: Task) -> List[Task]:
        ids = set(self._dependent_ids(plan, task.id))
        return [candidate for candidate in plan if candidate.id in ids]

    def _create_repair_plan(self, failed_task: Task, replacement_tasks: List[Task], victims: Optional[List[Task]] = None) -> List[Task]:
        """Replace a failed task with repair tasks and optionally block resource victims."""
        victim_ids = {victim.id for victim in victims or []}
        repaired: List[Task] = []
        inserted = False
        for task in self.current_plan:
            if task.id in victim_ids:
                blocked = self._copy_task(task)
                blocked.update_status(TaskStatus.BLOCKED, reason="repair_resource_victim")
                repaired.append(blocked)
                continue
            if task.id == failed_task.id:
                repaired.extend([self._copy_task(subtask) for subtask in replacement_tasks])
                inserted = True
            else:
                repaired.append(self._copy_task(task))
        if not inserted:
            repaired.extend([self._copy_task(subtask) for subtask in replacement_tasks])
        return repaired

    def _select_optimal_repair(self, candidates: List[RepairCandidate]) -> Optional[RepairCandidate]:
        """Select the lowest weighted score among safe repair candidates."""
        if not candidates:
            return None
        weights = dict(self.safety_config.get("repair_weights", {}))
        best: Optional[RepairCandidate] = None
        best_score = float("inf")
        for candidate in candidates:
            if not candidate.repaired_plan:
                continue
            try:
                if not self.safety_check(candidate.repaired_plan):
                    continue
            except PlanningError:
                continue
            score = (
                float(weights.get("cost", 0.35)) * self._normalize_cost(candidate.estimated_cost)
                + float(weights.get("risk", 0.40)) * self._normalize_risk(candidate.risk_assessment)
                + float(weights.get("time", 0.15)) * (1.0 - self._estimate_time_efficiency(candidate.repaired_plan))
                + float(weights.get("resource", 0.10)) * (1.0 - self._assess_resource_efficiency(candidate.repaired_plan))
            )
            if score < best_score:
                best = candidate
                best_score = score
        return best

    def _make_repair_candidate(
        self,
        strategy: str,
        plan: List[Task],
        failed_task: Task,
        *,
        victims: Optional[List[Task]] = None,
    ) -> RepairCandidate:
        return RepairCandidate(
            strategy=strategy,
            repaired_plan=plan,
            estimated_cost=self._estimate_repair_cost(plan, victims or []),
            risk_assessment=self._assess_repair_risk(plan, strategy, victims or []),
        )

    def _estimate_repair_cost(self, tasks: List[Task], victims: Optional[List[Task]] = None) -> float:
        base_cost = self.calculations.calculate_plan_cost(tasks)
        victim_penalty = sum(float(getattr(victim, "cost", 1.0) or 1.0) for victim in (victims or []))
        return max(0.0, base_cost + victim_penalty)

    def _assess_repair_risk(
        self,
        tasks: List[Task],
        strategy: str,
        victims: Optional[List[Task]] = None,
    ) -> Dict[str, Any]:
        profile = self.calculations.calculate_plan_risk_profile(tasks)
        profile["strategy"] = strategy
        profile["victim_count"] = len(victims or [])
        profile["resource_efficiency"] = self._assess_resource_efficiency(tasks)
        profile["time_efficiency"] = self._estimate_time_efficiency(tasks)
        return profile

    def _can_reallocate_resources(self, task: Task) -> bool:
        return bool(self._find_resource_victims(task))

    def _find_resource_victims(self, task: Task) -> List[Task]:
        """Return lower-priority tasks that can be paused for resource recovery."""
        priority = int(getattr(task, "priority", 0) or 0)
        candidates = [
            candidate
            for candidate in self.current_plan
            if candidate.id != task.id
            and int(getattr(candidate, "priority", 0) or 0) < priority
            and getattr(candidate, "status", None) in {TaskStatus.EXECUTING, TaskStatus.PENDING}
        ]
        return sorted(candidates, key=lambda item: (int(getattr(item, "priority", 0) or 0), float(getattr(item, "risk_score", 0.0) or 0.0)))

    def _normalize_cost(self, cost: float) -> float:
        plan_cost = max(1.0, self.calculations.calculate_plan_cost(self.current_plan))
        return clamp(float(cost) / plan_cost, 0.0, 1.0)

    def _normalize_risk(self, risk_assessment: Dict[str, Any]) -> float:
        return clamp(float(risk_assessment.get("overall_risk", 0.0) or 0.0), 0.0, 1.0)

    def _estimate_time_efficiency(self, plan: List[Task]) -> float:
        if not plan:
            return 1.0
        duration = self.calculations.calculate_plan_duration(plan)
        deadline_tasks = [float(task.deadline) for task in plan if float(getattr(task, "deadline", 0.0) or 0.0) > 0.0]
        if not deadline_tasks:
            return 1.0
        available = max(1.0, min(deadline_tasks) - time.time())
        return compute_temporal_margin(duration, available, time_buffer=float(self.safety_margin_model.time_buffer))

    def _assess_resource_efficiency(self, plan: List[Task]) -> float:
        if not plan:
            return 1.0
        try:
            return self.calculations.calculate_resource_margin(plan, self.resource_monitor.get_available_resources())
        except ResourceViolation:
            return 0.0

    # ------------------------------------------------------------------
    # Logging, metrics, and compatibility helpers
    # ------------------------------------------------------------------
    def log_adjustment(self, adjustment: Dict[str, Any]) -> None:
        """Log and persist a compact adjustment event."""
        entry = {
            "timestamp": time.time(),
            "adjustment": copy.deepcopy(adjustment),
            "plan_size": len(self.current_plan),
            "resource_utilization": self._get_resource_utilization(),
            "performance": self._collect_performance_metrics(),
        }
        self.last_adjustment_log.append(entry)
        self._publish_adjustment_event(entry)

    def _get_resource_utilization(self) -> Dict[str, Any]:
        report = self.resource_monitor.get_resource_report()
        return {
            "available": report.get("available", {}),
            "allocations": report.get("allocations", {}),
            "last_update": report.get("last_update", 0.0),
        }

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        return {
            "system_load": self._get_system_load(),
            "network_latency": self._measure_network_latency(),
            "service_health": self._check_service_health(),
            "queued_adjustments": self.adjustment_queue.qsize(),
        }

    def _get_system_load(self) -> float:
        report = self.resource_monitor.get_resource_report()
        history = report.get("history", [])
        if not history:
            return 0.0
        latest = history[-1]
        cpu_values = list((latest.get("cpu_utilization") or {}).values())
        return clamp(sum(cpu_values) / max(len(cpu_values), 1), 0.0, 1.0)

    def _measure_network_latency(self) -> float:
        url = str(self.safety_config.get("network_probe_url", "") or "").strip()
        if not url:
            return -1.0
        started = time.time()
        try:
            requests.head(url, timeout=float(self.safety_config.get("network_probe_timeout", 1.0)))
            return (time.time() - started) * 1000.0
        except RequestException:
            return -1.0

    def _check_service_health(self) -> Dict[str, str]:
        report = self.resource_monitor.get_resource_report()
        return {
            "resource_monitor": "healthy" if not report.get("last_error") else "degraded",
            "memory": "healthy",
            "calculations": "healthy",
        }

    def _publish_adjustment_event(self, log_entry: Dict[str, Any]) -> None:
        logger.info("Safety adjustment event: %s", truncate_for_logging(safe_json_dumps(log_entry), 512))

    def _emergency_shutdown_procedure(self) -> None:
        """Release planner-owned resources and block current executing tasks."""
        with self.lock:
            for task in self.current_plan:
                if task.status == TaskStatus.EXECUTING:
                    task.update_status(TaskStatus.BLOCKED, reason="emergency_shutdown")
                    self.resource_monitor.release_resources(task.id)
            self.resource_monitor.release_all_resources()
            self._checkpoint("emergency_shutdown", {"reason": "safety_procedure"})

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        require_type(self.safety_config, dict, "safety_planning")
        require_positive(int(self.safety_config.get("max_adjustment_retries", 3)) + 1, "safety.max_adjustment_retries_plus_one")
        require_positive(int(self.safety_config.get("queue_max_size", 1000)), "safety.queue_max_size")
        require_positive(int(self.safety_config.get("violation_history_limit", 1000)), "safety.violation_history_limit")
        require_type(self.safety_config.get("allowed_update_fields", []), list, "safety.allowed_update_fields")
        thresholds = self.safety_config.get("min_margin_thresholds", {})
        require_type(thresholds, dict, "safety.min_margin_thresholds")
        for key, value in thresholds.items():
            validate_probability(float(value), f"safety.min_margin_thresholds.{key}")

    def _normalise_plan(self, plan: List[Task]) -> List[Task]:
        require_type(plan, list, "plan")
        normalised: List[Task] = []
        for item in plan:
            normalised.append(self._coerce_task(item, "plan item"))
        return normalised

    def _coerce_task(self, value: Any, name: str = "task") -> Task:
        if isinstance(value, Task):
            return value
        if isinstance(value, dict):
            return Task.from_dict(value)
        raise AdjustmentError(
            f"{name} must be a Task or task dictionary",
            adjustment={"value_type": type(value).__name__},
        )

    def _copy_task(self, task: Task) -> Task:
        return task.copy() if hasattr(task, "copy") else copy.deepcopy(task)

    def _coerce_resource_profile(self, value: Any) -> ResourceProfile:
        if isinstance(value, ResourceProfile):
            return value
        if isinstance(value, dict):
            return ResourceProfile.from_dict(value)
        raise ResourceViolation(
            "resource_requirements must be ResourceProfile or dict",
            resource_type="resource_profile",
            requested=type(value).__name__,
            available=["ResourceProfile", "dict"],
        )

    def _apply_task_updates(self, task: Task, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key in {"resource_requirements", "requirements"}:
                task.resource_requirements = self._coerce_resource_profile(value)
            elif key == "status":
                task.update_status(value)
            elif hasattr(task, key):
                setattr(task, key, value)
            else:
                raise AdjustmentError(
                    f"Task has no updateable field {key!r}",
                    adjustment={"type": "modify_task", "task_id": task.id, "updates": updates},
                )
        task._post_init()

    def _validate_deadline_value(self, deadline: float, adjustment: Dict[str, Any]) -> None:
        if deadline < time.time():
            raise TemporalViolation(
                "Adjusted deadline is in the past",
                violation_type="window",
                task_id=str(adjustment.get("task_id", "")),
                constraint_details={"deadline": deadline, "now": time.time()},
            )

    def _validate_dependencies_known(self, task: Task, plan: List[Task], *, allow_existing_only: bool) -> None:
        known = {candidate.id for candidate in plan}
        missing = [dep for dep in getattr(task, "dependencies", []) if dep not in known]
        if missing and allow_existing_only:
            raise AdjustmentError(
                f"Task {task.id} has unknown dependencies",
                adjustment={"type": "add_task", "task_id": task.id},
                conflict_details={"missing_dependencies": missing},
            )

    def _dependent_ids(self, plan: List[Task], task_id: str) -> List[str]:
        dep_map = build_dependency_map(plan)
        if task_id not in dep_map:
            # task names are also accepted by legacy call sites
            match = self._find_task(plan, task_id)
            if match is not None:
                task_id = match.id
        return sorted(get_all_successors(task_id, dep_map))

    def _find_task(self, plan: List[Task], task_id_or_name: str) -> Optional[Task]:
        for task in plan:
            if task.id == task_id_or_name or task.name == task_id_or_name:
                return task
        return None

    def _replace_task(self, plan: List[Task], task_id: str, replacement: Task) -> List[Task]:
        result: List[Task] = []
        replaced = False
        for task in plan:
            if task.id == task_id:
                result.append(replacement)
                replaced = True
            else:
                result.append(task)
        if not replaced:
            result.append(replacement)
        return result

    def _retry_task(self, task: Task) -> Task:
        retry = self._copy_task(task)
        retry.retry_count = int(getattr(retry, "retry_count", 0) or 0) + 1
        retry.status = TaskStatus.PENDING
        retry.progress = 0.0
        retry.failure_reason = ""
        retry.start_time = 0.0
        retry.end_time = 0.0
        return retry

    def _record_violation(
        self,
        violation_type: str,
        resource: str,
        measured_value: float,
        threshold: float,
        task_id: str,
        *,
        severity: str = "medium",
        corrective_action: str = "",
        impact_analysis: Optional[Dict[str, Any]] = None,
    ) -> SafetyViolation:
        violation = SafetyViolation(
            violation_type=violation_type,
            resource=resource,
            measured_value=float(measured_value),
            threshold=float(threshold),
            task_id=task_id,
            severity=severity,
            corrective_action=corrective_action,
            impact_analysis=dict(impact_analysis or {}),
        )
        self.current_violations.append(violation)
        return violation

    def _extend_violation_history(self, violations: Iterable[SafetyViolation]) -> None:
        self.violation_history.extend(list(violations))
        limit = int(self.safety_config.get("violation_history_limit", 1000))
        if len(self.violation_history) > limit:
            self.violation_history = self.violation_history[-limit:]

    def _checkpoint(self, label: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.memory.save_checkpoint(label=label, metadata=metadata or {})
        except Exception as exc:
            logger.warning("Safety checkpoint failed: %s", exc)

    @staticmethod
    def _cluster_to_dict(resources: ClusterResources) -> Dict[str, Any]:
        return resources.to_dict() if hasattr(resources, "to_dict") else {
            "gpu_total": getattr(resources, "gpu_total", 0.0),
            "ram_total": getattr(resources, "ram_total", 0.0),
            "specialized_hardware_available": list(getattr(resources, "specialized_hardware_available", []) or []),
            "current_allocations": dict(getattr(resources, "current_allocations", {}) or {}),
        }

    def __enter__(self) -> "SafetyPlanning":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Ensure resources are released even if an exception occurred."""
        try:
            if hasattr(self.resource_monitor, "stop_monitoring"):
                self.resource_monitor.stop_monitoring()
        except Exception as e:
            logger.error("Error stopping resource monitor during exit: %s", e)
        # Optionally run emergency shutdown if a critical error occurred
        if exc_type is not None and issubclass(exc_type, (ResourceViolation, TemporalError, PlanningError)):
            logger.warning("Emergency shutdown triggered due to %s", exc_type.__name__)
            self._emergency_shutdown_procedure()


if __name__ == "__main__":
    print("\n=== Running Safety Planning ===\n")
    printer.status("TEST", "Safety Planning initialized", "info")

    planner = SafetyPlanning()
    try:
        task = Task(
            id="safety_task",
            name="SafetySmokeTask",
            task_type=TaskType.PRIMITIVE,
            resource_requirements=ResourceProfile(gpu=0.0, ram=0.1),
            duration=60,
            deadline=3600,
            priority=5,
        )
        planner.current_plan = [task]
        assert planner.safety_check(planner.current_plan) is True

        update = {"type": "modify_task", "task_id": task.id, "updates": {"priority": 8}}
        planner.interactive_adjustment_handler(update)
        assert planner.current_plan[0].priority == 8

        subtasks = planner.distributed_decomposition(task)
        assert subtasks

        candidate = planner._generate_repair_candidates(task)
        assert candidate

        print("\n=== Test ran successfully ===\n")
    finally:
        if hasattr(planner.resource_monitor, "stop_monitoring"):
            planner.resource_monitor.stop_monitoring()
