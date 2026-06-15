"""
Task Scheduler – production-grade deadline-aware, risk-sensitive scheduling.

This module provides an abstract scheduler contract and a concrete
``DeadlineAwareScheduler`` used by the planning subsystem.  The scheduler is
responsible for capability matching, dependency-aware ordering, load-aware time
placement, risk-sensitive prioritisation, and mitigation of high-risk
assignments.  It deliberately delegates validation, graph helpers, retry math,
resource feasibility checks, and structured errors to the shared planning helper
and error modules instead of duplicating that logic.

Compatibility notes
-------------------
- ``TaskScheduler`` and ``DeadlineAwareScheduler.schedule(...)`` keep the same
  public shape as the existing implementation.
- The scheduler still accepts task dictionaries and also supports ``Task``
  instances from ``planning_types``.
- The returned schedule remains a mapping of ``task_id -> assignment``.
- Legacy private method names are retained where practical because other
  planning modules may call them during tests or diagnostics.
"""

from __future__ import annotations

import copy
import math
import threading
import time

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

from .utils.config_loader import load_global_config, get_config_section
from .utils.planning_errors import *
from .utils.planning_helpers import *
from .utils.planning_calculations import PlanningCalculations
from .planning_types import ResourceProfile, Task
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Task Scheduler")
printer = PrettyPrinter()

TaskLike = Union[Dict[str, Any], Task]
AgentLike = Union[Dict[str, Any], Any]
RiskAssessor = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class ScheduleDiagnostics:
    """Structured diagnostic summary for the latest scheduling pass."""

    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    task_count: int = 0
    agent_count: int = 0
    scheduled_count: int = 0
    unscheduled: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    dependency_order: List[str] = field(default_factory=list)
    high_risk_tasks: List[str] = field(default_factory=list)
    deadline_misses: List[str] = field(default_factory=list)

    def finish(self, schedule: Dict[str, Dict[str, Any]]) -> None:
        self.completed_at = time.time()
        self.scheduled_count = len(schedule)

    @property
    def duration_seconds(self) -> float:
        end = self.completed_at or time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "task_count": self.task_count,
            "agent_count": self.agent_count,
            "scheduled_count": self.scheduled_count,
            "unscheduled": list(self.unscheduled),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "dependency_order": list(self.dependency_order),
            "high_risk_tasks": list(self.high_risk_tasks),
            "deadline_misses": list(self.deadline_misses),
        }


class TaskScheduler(ABC):
    """Abstract base class for all task schedulers."""

    @abstractmethod
    def schedule(
        self,
        tasks: List[TaskLike],
        agents: Dict[str, Any],
        risk_assessor: Optional[Callable] = None,
        state: Optional[Dict] = None,
    ) -> Dict:
        """Schedule tasks to agents."""
        raise NotImplementedError


class DeadlineAwareScheduler(TaskScheduler):
    """
    Deadline-aware scheduler with capability matching and risk mitigation.

    Scheduling phases
    -----------------
    1. Normalise and validate task / agent input without mutating caller data.
    2. Assess risk and compute priority using deadline urgency and configured
       weights.
    3. Validate dependencies using the shared graph helpers.
    4. Match eligible agents by capabilities, load, historical performance,
       efficiency, and optional resource feasibility.
    5. Create assignments respecting predecessor completion times and agent
       availability.
    6. Apply mitigation for assignments whose risk exceeds the configured
       threshold.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        raw_cfg = get_config_section("task_scheduler", config=self.config, default={})
        self.task_config = deep_update(self._default_config(), raw_cfg)
        self._load_config_values()

        self.calculations = PlanningCalculations() if self.use_planning_calculations else None
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.state: Optional[Dict[str, Any]] = None
        self.task_history: Dict[str, Any] = defaultdict(list)
        self.last_schedule_report: Dict[str, Any] = {}
        self._last_tasks_by_id: Dict[str, Dict[str, Any]] = {}
        self._last_candidate_map: Dict[str, List[Tuple[str, float]]] = {}
        self._last_agent_available: Dict[str, float] = {}
        self._lock = threading.RLock()

        logger.info("DeadlineAwareScheduler successfully initialized")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        return {
            "strict_validation": True,
            "raise_on_validation_error": False,
            "risk_threshold": 0.70,
            "base_duration_per_requirement": 5.0,
            "default_task_duration": 30.0,
            "default_deadline_seconds": 300.0,
            "deadline_horizon_seconds": 3600.0,
            "max_priority": 10.0,
            "max_agent_load": 1.0,
            "load_time_scale_seconds": 60.0,
            "efficiency_attribute": "efficiency",
            "use_planning_calculations": False,
            "include_unassigned_diagnostics": True,
            "allow_deadline_miss": True,
            "allow_partial_schedule": True,
            "allow_decomposition_mitigation": True,
            "allowed_agent_statuses": ["idle", "available", "ready", "healthy", "online"],
            "retry_policy": {
                "max_retries": 3,
                "max_attempts": 3,
                "backoff_factor": 1.5,
                "delay": 10.0,
                "base_delay": 1.0,
                "max_delay": 60.0,
            },
            "priority_weights": {
                "base_priority": 0.45,
                "deadline_urgency": 0.35,
                "risk_urgency": 0.15,
                "dependency_pressure": 0.05,
            },
            "agent_score_weights": {
                "success_rate": 0.30,
                "efficiency": 0.25,
                "specialization": 0.20,
                "availability": 0.15,
                "load": 0.10,
            },
            "risk_mitigation": {
                "reassign_min_score": 0.05,
                "delay_factor": 1.0,
                "risk_reduction_on_delay": 0.05,
                "risk_reduction_on_decomposition": 0.25,
            },
        }

    def _load_config_values(self) -> None:
        cfg = self.task_config
        self.strict_validation = bool(cfg.get("strict_validation", True))
        self.raise_on_validation_error = bool(cfg.get("raise_on_validation_error", False))
        self.risk_threshold = float(cfg.get("risk_threshold", 0.70))
        validate_probability(self.risk_threshold, "task_scheduler.risk_threshold")

        self.base_duration_per_requirement = float(cfg.get("base_duration_per_requirement", 5.0))
        require_positive(self.base_duration_per_requirement, "task_scheduler.base_duration_per_requirement")

        self.default_task_duration = float(cfg.get("default_task_duration", 30.0))
        require_positive(self.default_task_duration, "task_scheduler.default_task_duration")

        self.default_deadline_seconds = float(cfg.get("default_deadline_seconds", 300.0))
        require_positive(self.default_deadline_seconds, "task_scheduler.default_deadline_seconds")

        self.deadline_horizon_seconds = float(cfg.get("deadline_horizon_seconds", 3600.0))
        require_positive(self.deadline_horizon_seconds, "task_scheduler.deadline_horizon_seconds")

        self.max_priority = float(cfg.get("max_priority", 10.0))
        require_positive(self.max_priority, "task_scheduler.max_priority")

        self.max_agent_load = float(cfg.get("max_agent_load", 1.0))
        require_positive(self.max_agent_load, "task_scheduler.max_agent_load")

        self.load_time_scale_seconds = float(cfg.get("load_time_scale_seconds", 60.0))
        require_non_negative(self.load_time_scale_seconds, "task_scheduler.load_time_scale_seconds")

        self.efficiency_attribute = str(cfg.get("efficiency_attribute", "efficiency"))
        require_non_empty(self.efficiency_attribute, "task_scheduler.efficiency_attribute")

        self.use_planning_calculations = bool(cfg.get("use_planning_calculations", False))
        self.include_unassigned_diagnostics = bool(cfg.get("include_unassigned_diagnostics", True))
        self.allow_deadline_miss = bool(cfg.get("allow_deadline_miss", True))
        self.allow_partial_schedule = bool(cfg.get("allow_partial_schedule", True))
        self.allow_decomposition_mitigation = bool(cfg.get("allow_decomposition_mitigation", True))
        self.allowed_agent_statuses = {str(s).lower() for s in cfg.get("allowed_agent_statuses", [])}

        self.retry_policy = dict(cfg.get("retry_policy", {}))
        self.priority_weights = self._normalise_weights(dict(cfg.get("priority_weights", {})), "priority_weights")
        self.agent_score_weights = self._normalise_weights(dict(cfg.get("agent_score_weights", {})), "agent_score_weights")
        self.risk_mitigation = dict(cfg.get("risk_mitigation", {}))

    @staticmethod
    def _normalise_weights(weights: Dict[str, Any], name: str) -> Dict[str, float]:
        require_type(weights, dict, f"task_scheduler.{name}")
        numeric = {str(k): max(0.0, float(v)) for k, v in weights.items()}
        total = sum(numeric.values())
        if total <= 0.0:
            raise PlanningConfigError(
                f"task_scheduler.{name} must contain at least one positive weight",
                config_key=name,
                config_section="task_scheduler",
                expected_type="mapping of positive numeric weights",
            )
        return {k: v / total for k, v in numeric.items()}

    # ------------------------------------------------------------------
    # Public scheduling entry point
    # ------------------------------------------------------------------
    def schedule(
        self,
        tasks: List[TaskLike],
        agents: Dict[str, AgentLike],
        risk_assessor: Optional[Callable] = None,
        state: Optional[Dict] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Produce a schedule mapping task IDs to assignment dictionaries."""
        diagnostics = ScheduleDiagnostics()
        with self._lock:
            try:
                normal_tasks = self._normalise_tasks(tasks)
                normal_agents = self._normalise_agents(agents)
                diagnostics.task_count = len(normal_tasks)
                diagnostics.agent_count = len(normal_agents)

                self.agents = normal_agents
                self.state = copy.deepcopy(state or {})
                self.task_history["current"] = copy.deepcopy(normal_tasks)
                self._last_tasks_by_id = {task["id"]: task for task in normal_tasks}

                errors, warnings = self._validate_normalized_inputs(normal_tasks, normal_agents)
                diagnostics.errors.extend(errors)
                diagnostics.warnings.extend(warnings)
                if errors:
                    self._handle_validation_errors(errors)
                    diagnostics.finish({})
                    self.last_schedule_report = diagnostics.to_dict()
                    return {}

                prioritized = self._prioritize_tasks(normal_tasks, risk_assessor)
                candidate_map = self._map_capabilities(prioritized, normal_agents)
                self._last_candidate_map = {k: list(v) for k, v in candidate_map.items()}

                schedule = self._create_schedule(candidate_map, normal_agents, self._schedule_state(normal_tasks, state))
                schedule = self._apply_risk_mitigation(schedule, risk_assessor)

                scheduled_ids = set(schedule)
                diagnostics.dependency_order = [a.get("task_id", tid) for tid, a in schedule.items()]
                diagnostics.unscheduled = [task["id"] for task in normal_tasks if task["id"] not in scheduled_ids]
                diagnostics.high_risk_tasks = [
                    tid for tid, a in schedule.items()
                    if float(a.get("risk_score", 0.0)) > self.risk_threshold
                ]
                diagnostics.deadline_misses = [
                    tid for tid, a in schedule.items()
                    if float(a.get("lateness_seconds", 0.0)) > 0.0
                ]
                diagnostics.finish(schedule)
                self.last_schedule_report = diagnostics.to_dict()

                if diagnostics.unscheduled and not self.allow_partial_schedule:
                    raise SchedulingConflictError(
                        "Unable to schedule all tasks and partial schedules are disabled",
                        conflicting_task_ids=diagnostics.unscheduled,
                        conflict_type="unscheduled_tasks",
                        context=self.last_schedule_report,
                    )

                logger.info(
                    "Scheduling complete: %s/%s tasks assigned in %.4fs",
                    len(schedule),
                    len(normal_tasks),
                    diagnostics.duration_seconds,
                )
                return schedule
            except PlanningError:
                raise
            except Exception as exc:
                logger.error("Unexpected scheduling failure: %s", exc, exc_info=True)
                if self.raise_on_validation_error:
                    raise SchedulingConflictError(
                        f"Unexpected scheduling failure: {exc}",
                        conflict_type="unexpected_error",
                        context={"error": str(exc)},
                    ) from exc
                diagnostics.errors.append(str(exc))
                diagnostics.finish({})
                self.last_schedule_report = diagnostics.to_dict()
                return {}

    # ------------------------------------------------------------------
    # Normalisation and validation
    # ------------------------------------------------------------------
    def _normalise_tasks(self, tasks: List[TaskLike]) -> List[Dict[str, Any]]:
        require_type(tasks, list, "tasks")
        now = time.time()
        normalised: List[Dict[str, Any]] = []
        for idx, raw in enumerate(tasks):
            if isinstance(raw, dict):
                task = copy.deepcopy(raw)
                task_id = str(task.get("id") or task.get("task_id") or f"task_{idx}")
                task_name = str(task.get("name") or task_id)
                requirements = self._as_string_list(
                    task.get("requirements", task.get("required_skills", task.get("capabilities_required", [])))
                )
                dependencies = self._as_string_list(task.get("dependencies", []))
                duration = float(task.get("duration", task.get("estimated_duration", 0.0)) or 0.0)
                resource_requirements = task.get("resource_requirements", task.get("resources", None))
            else:
                task_id = str(getattr(raw, "id", f"task_{idx}"))
                task_name = str(getattr(raw, "name", task_id))
                requirements = self._task_requirements_from_object(raw)
                dependencies = self._as_string_list(getattr(raw, "dependencies", []))
                duration = float(getattr(raw, "duration", getattr(raw, "estimated_duration", 0.0)) or 0.0)
                resource_requirements = getattr(raw, "resource_requirements", None)
                task = {
                    "id": task_id,
                    "name": task_name,
                    "priority": getattr(raw, "priority", 1),
                    "deadline": getattr(raw, "deadline", 0.0),
                    "duration": duration,
                    "estimated_duration": getattr(raw, "estimated_duration", duration),
                    "dependencies": dependencies,
                    "risk_score": getattr(raw, "risk_score", 0.0),
                    "cost": getattr(raw, "cost", 1.0),
                    "blacklisted_agents": getattr(raw, "blacklisted_agents", []),
                    "resource_requirements": resource_requirements,
                    "original_task": raw,
                }

            if duration <= 0.0:
                duration = self.default_task_duration

            deadline = self._normalise_deadline(task.get("deadline"), now)
            priority = clamp(float(task.get("priority", 1) or 1), 0.0, self.max_priority)
            risk_score = clamp(float(task.get("risk_score", 0.0) or 0.0), 0.0, 1.0)

            resource_profile = self._normalise_resource_profile(resource_requirements)
            normalised.append(
                {
                    **task,
                    "id": task_id,
                    "task_id": task_id,
                    "name": task_name,
                    "requirements": requirements,
                    "dependencies": dependencies,
                    "deadline": deadline,
                    "priority": priority,
                    "duration": duration,
                    "estimated_duration": float(task.get("estimated_duration", duration) or duration),
                    "risk_score": risk_score,
                    "cost": float(task.get("cost", 1.0) or 1.0),
                    "blacklisted_agents": self._as_string_list(task.get("blacklisted_agents", [])),
                    "resource_requirements": resource_profile,
                    "metadata": copy.deepcopy(task.get("metadata", {})) if isinstance(task.get("metadata", {}), dict) else {},
                }
            )
        return normalised

    def _normalise_agents(self, agents: Dict[str, AgentLike]) -> Dict[str, Dict[str, Any]]:
        require_type(agents, dict, "agents")
        now = time.time()
        normalised: Dict[str, Dict[str, Any]] = {}
        for raw_id, raw_details in agents.items():
            agent_id = str(raw_id)
            if isinstance(raw_details, dict):
                details = copy.deepcopy(raw_details)
            else:
                details = {
                    "capabilities": getattr(raw_details, "capabilities", []),
                    "current_load": getattr(raw_details, "current_load", 0.0),
                    "successes": getattr(raw_details, "successes", 1),
                    "failures": getattr(raw_details, "failures", 0),
                    self.efficiency_attribute: getattr(raw_details, self.efficiency_attribute, 1.0),
                    "status": getattr(raw_details, "status", "available"),
                    "available_from": getattr(raw_details, "available_from", now),
                    "available_resources": getattr(raw_details, "available_resources", None),
                    "original_agent": raw_details,
                }

            load = max(0.0, float(details.get("current_load", 0.0) or 0.0))
            available_from = self._normalise_available_from(details, load, now)
            efficiency = max(0.01, float(details.get(self.efficiency_attribute, 1.0) or 1.0))
            successes = max(0, int(details.get("successes", 1) or 0))
            failures = max(0, int(details.get("failures", 0) or 0))
            capabilities = self._as_string_list(details.get("capabilities", []))
            status = str(details.get("status", "available")).lower()

            normalised[agent_id] = {
                **details,
                "id": agent_id,
                "capabilities": capabilities,
                "current_load": load,
                "available_from": available_from,
                self.efficiency_attribute: efficiency,
                "successes": successes,
                "failures": failures,
                "status": status,
                "blacklisted": bool(details.get("blacklisted", False)),
                "available_resources": self._normalise_agent_resources(details.get("available_resources")),
            }
        return normalised

    def _validate_inputs(self, tasks: List[Dict], agents: Dict[str, Any]) -> bool:
        """Backward-compatible validation wrapper used by legacy callers."""
        try:
            normal_tasks = self._normalise_tasks(cast(List[TaskLike], tasks))
            normal_agents = self._normalise_agents(agents)
            errors, warnings = self._validate_normalized_inputs(normal_tasks, normal_agents)
            for warning in warnings:
                logger.warning(warning)
            for error in errors:
                logger.error(error)
            return not errors
        except PlanningError as exc:
            logger.error("Validation failed: %s", exc)
            return False

    def _validate_normalized_inputs(
        self,
        tasks: List[Dict[str, Any]],
        agents: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []

        if not tasks:
            errors.append("No tasks provided")
        if not agents:
            errors.append("No agents provided")

        seen: set[str] = set()
        declared = {task["id"] for task in tasks}
        dep_map: Dict[str, List[str]] = {}
        now = time.time()

        for task in tasks:
            task_id = task["id"]
            if not is_valid_task_id(task_id):
                errors.append(f"Task {task_id!r}: invalid task id")
            if task_id in seen:
                errors.append(f"Task {task_id!r}: duplicate task id")
            seen.add(task_id)

            if not isinstance(task.get("requirements", []), list):
                errors.append(f"Task {task_id}: requirements must be a list")
            if float(task.get("duration", 0.0)) <= 0.0:
                errors.append(f"Task {task_id}: duration must be positive")
            if float(task.get("deadline", 0.0)) <= now:
                warnings.append(f"Task {task_id}: deadline is in the past")

            deps = task.get("dependencies", [])
            dep_map[task_id] = list(deps)
            for dep in deps:
                if dep not in declared:
                    errors.append(f"Task {task_id}: dependency {dep!r} is not present in task list")

        if tasks:
            cycle = detect_cycles([task["id"] for task in tasks], dep_map)
            if cycle:
                errors.append(f"Cyclic task dependency detected: {cycle}")

        for agent_id, details in agents.items():
            if not agent_id.strip():
                errors.append("Agent id cannot be empty")
            if details.get("blacklisted", False):
                warnings.append(f"Agent {agent_id}: blacklisted")
            if self.allowed_agent_statuses and details.get("status") not in self.allowed_agent_statuses:
                warnings.append(f"Agent {agent_id}: status {details.get('status')!r} may be unavailable")
            if not isinstance(details.get("capabilities", []), list):
                errors.append(f"Agent {agent_id}: capabilities must be a list")
            if float(details.get("current_load", 0.0)) < 0.0:
                errors.append(f"Agent {agent_id}: current_load must be non-negative")
            if float(details.get(self.efficiency_attribute, 0.0)) <= 0.0:
                errors.append(f"Agent {agent_id}: {self.efficiency_attribute} must be positive")

        return errors, warnings

    def _handle_validation_errors(self, errors: List[str]) -> None:
        for error in errors:
            logger.error(error)
        if self.raise_on_validation_error:
            raise SchedulingConflictError(
                "Task scheduling validation failed",
                conflicting_task_ids=[],
                conflict_type="validation_error",
                context={"errors": errors},
            )

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------
    def _prioritize_tasks(self, tasks: List[Dict], risk_assessor: Optional[Callable]) -> List[Dict]:
        prioritized: List[Tuple[float, Dict[str, Any]]] = []
        dep_map = {task["id"]: list(task.get("dependencies", [])) for task in tasks}
        for task in tasks:
            assessment = self._assess_risk(task, risk_assessor)
            task["risk_assessment"] = assessment
            task["risk_score"] = clamp(float(assessment.get("risk_score", task.get("risk_score", 0.0))), 0.0, 1.0)
            priority = self._calculate_priority(
                task["deadline"],
                task.get("priority", 1),
                task["risk_score"],
                dependency_count=len(dep_map.get(task["id"], [])),
            )
            task["scheduler_priority"] = priority
            prioritized.append((priority, task))
        prioritized.sort(key=lambda item: (item[0], -float(item[1].get("deadline", 0.0))), reverse=True)
        return [copy.deepcopy(task) for _, task in prioritized]

    def _assess_risk(self, task: Dict[str, Any], risk_assessor: Optional[Callable]) -> Dict[str, Any]:
        baseline = {"risk_score": clamp(float(task.get("risk_score", 0.0) or 0.0), 0.0, 1.0)}
        if risk_assessor is None:
            return baseline
        try:
            result = risk_assessor(copy.deepcopy(task))
            if not isinstance(result, dict):
                logger.warning("Risk assessor returned non-dict for task %s", task.get("id"))
                return baseline
            merged = deep_update(baseline, result)
            merged["risk_score"] = clamp(float(merged.get("risk_score", baseline["risk_score"])), 0.0, 1.0)
            return merged
        except PlanningError:
            raise
        except Exception as exc:
            logger.error("Risk assessor failed for task %s: %s", task.get("id"), exc)
            return deep_update(baseline, {"risk_error": str(exc)})

    def _calculate_priority(
        self,
        deadline: float,
        base_priority: Union[int, float],
        risk_score: float,
        dependency_count: int = 0,
    ) -> float:
        remaining = seconds_until_deadline(float(deadline))
        urgency = 1.0 if remaining <= 0.0 else clamp(1.0 - (remaining / self.deadline_horizon_seconds), 0.0, 1.0)
        priority_norm = clamp(float(base_priority) / self.max_priority, 0.0, 1.0)
        risk_urgency = clamp(float(risk_score), 0.0, 1.0)
        dep_pressure = clamp(float(dependency_count) / 10.0, 0.0, 1.0)
        return (
            self.priority_weights.get("base_priority", 0.0) * priority_norm
            + self.priority_weights.get("deadline_urgency", 0.0) * urgency
            + self.priority_weights.get("risk_urgency", 0.0) * risk_urgency
            + self.priority_weights.get("dependency_pressure", 0.0) * dep_pressure
        )

    # ------------------------------------------------------------------
    # Capability mapping
    # ------------------------------------------------------------------
    def _map_capabilities(self, tasks: List[Dict], agents: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
        candidate_map: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for task in tasks:
            for agent_id, details in agents.items():
                if self._agent_is_eligible(agent_id, task, details):
                    score = self._calculate_agent_score(agent_id, task, details)
                    candidate_map[task["id"]].append((agent_id, score))
            candidate_map[task["id"]].sort(key=lambda item: item[1], reverse=True)
        return dict(candidate_map)

    def _agent_is_eligible(self, agent_id: str, task: Dict, details: Dict) -> bool:
        if details.get("blacklisted", False):
            return False
        if self.allowed_agent_statuses and details.get("status") not in self.allowed_agent_statuses:
            return False
        if agent_id in set(task.get("blacklisted_agents", [])):
            return False
        if float(details.get("current_load", 0.0)) >= self.max_agent_load:
            return False

        capabilities = set(details.get("capabilities", []))
        requirements = set(task.get("requirements", []))
        if not capabilities.issuperset(requirements):
            return False

        available_resources = details.get("available_resources")
        resource_profile = task.get("resource_requirements")
        if available_resources and isinstance(resource_profile, ResourceProfile):
            try:
                check_resource_feasibility(
                    {"gpu": float(resource_profile.gpu), "ram": float(resource_profile.ram)},
                    {
                        "gpu": float(available_resources.get("gpu", available_resources.get("gpu_total", 0.0)) or 0.0),
                        "ram": float(available_resources.get("ram", available_resources.get("ram_total", 0.0)) or 0.0),
                    },
                    task_id=task.get("id", ""),
                )
                required_hw = set(resource_profile.specialized_hardware)
                available_hw = set(available_resources.get("specialized_hardware", available_resources.get("specialized_hardware_available", [])) or [])
                if required_hw and not required_hw.issubset(available_hw):
                    return False
            except ResourceViolation:
                return False
        return True

    def _calculate_agent_score(self, agent_id: str, task: Dict, details: Dict) -> float:
        successes = max(0, int(details.get("successes", 1) or 0))
        failures = max(0, int(details.get("failures", 0) or 0))
        success_rate = successes / max(1, successes + failures)

        efficiency = clamp(float(details.get(self.efficiency_attribute, 1.0) or 1.0), 0.0, 2.0) / 2.0
        requirements = set(task.get("requirements", []))
        capabilities = set(details.get("capabilities", []))
        specialization = 1.0 if not requirements else len(capabilities & requirements) / len(requirements)

        wait = max(0.0, float(details.get("available_from", time.time())) - time.time())
        availability = clamp(1.0 - wait / self.deadline_horizon_seconds, 0.0, 1.0)
        load_score = clamp(1.0 - float(details.get("current_load", 0.0)) / self.max_agent_load, 0.0, 1.0)

        score = (
            self.agent_score_weights.get("success_rate", 0.0) * success_rate
            + self.agent_score_weights.get("efficiency", 0.0) * efficiency
            + self.agent_score_weights.get("specialization", 0.0) * specialization
            + self.agent_score_weights.get("availability", 0.0) * availability
            + self.agent_score_weights.get("load", 0.0) * load_score
        )
        agent_risk = self._calculate_agent_risk(task, details)
        return clamp(score * (1.0 - 0.25 * agent_risk), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Temporal scheduling
    # ------------------------------------------------------------------
    def _create_schedule(
        self,
        candidate_map: Dict[str, List[Tuple[str, float]]],
        agents: Dict[str, Any],
        state: Optional[Dict],
    ) -> Dict[str, Dict[str, Any]]:
        schedule: Dict[str, Dict[str, Any]] = {}
        now = time.time()
        task_map = {task["id"]: task for task in self.task_history.get("current", [])}
        dependency_map = self._build_dependency_graph(state)
        task_order = self._order_by_dependencies(candidate_map, dependency_map)

        agent_available = {
            agent_id: max(now, float(details.get("available_from", now)))
            for agent_id, details in agents.items()
        }
        self._last_agent_available = dict(agent_available)
        task_end_times: Dict[str, float] = {}

        for task_id in task_order:
            task = task_map.get(task_id)
            if task is None:
                continue
            candidates = candidate_map.get(task_id, [])
            if not candidates:
                logger.warning("No eligible agent for task %s", task_id)
                continue

            dependency_ready_time = max((task_end_times.get(dep, now) for dep in task.get("dependencies", [])), default=now)
            best_assignment: Optional[Dict[str, Any]] = None
            best_rank = -math.inf

            for agent_id, score in candidates:
                agent_details = agents[agent_id]
                proposed = self._create_assignment(
                    task_id,
                    agent_id,
                    agent_details,
                    max(agent_available.get(agent_id, now), dependency_ready_time),
                    state,
                )
                proposed["agent_score"] = score
                lateness = float(proposed.get("lateness_seconds", 0.0))
                if lateness > 0.0 and not self.allow_deadline_miss:
                    continue
                deadline_penalty = clamp(lateness / max(self.deadline_horizon_seconds, 1.0), 0.0, 1.0)
                rank = score - (0.5 * deadline_penalty)
                if rank > best_rank:
                    best_rank = rank
                    best_assignment = proposed

            if best_assignment is None:
                logger.warning("Unable to produce feasible assignment for task %s", task_id)
                continue

            schedule[task_id] = best_assignment
            agent_available[best_assignment["agent_id"]] = best_assignment["end_time"]
            task_end_times[task_id] = best_assignment["end_time"]

        return schedule

    def _build_dependency_graph(self, state: Optional[Dict]) -> Dict[str, List[str]]:
        """Return dependency map ``task_id -> prerequisite task_ids``."""
        state = state or {}
        if isinstance(state.get("dependency_map"), dict):
            return {str(k): self._as_string_list(v) for k, v in state["dependency_map"].items()}
        if isinstance(state.get("dependency_graph"), dict):
            raw = state["dependency_graph"]
            # Existing code sometimes used prerequisite -> dependents; detect and invert
            task_ids = {task["id"] for task in self.task_history.get("current", [])}
            keys_are_tasks = set(raw.keys()).issubset(task_ids)
            if keys_are_tasks:
                return {str(k): self._as_string_list(v) for k, v in raw.items()}
            inverted: Dict[str, List[str]] = {tid: [] for tid in task_ids}
            for dep, dependents in raw.items():
                for dependent in self._as_string_list(dependents):
                    if dependent in inverted:
                        inverted[dependent].append(str(dep))
            return inverted
        return {task["id"]: list(task.get("dependencies", [])) for task in self.task_history.get("current", [])}

    def _order_by_dependencies(
        self,
        candidate_map: Dict[str, List[Tuple[str, float]]],
        dependency_graph: Dict[str, List[str]],
    ) -> List[str]:
        task_ids = list(candidate_map.keys())
        full_graph = {tid: [dep for dep in dependency_graph.get(tid, []) if dep in task_ids] for tid in task_ids}
        cycle = detect_cycles(task_ids, full_graph)
        if cycle:
            raise SchedulingConflictError(
                f"Cyclic task dependency detected: {cycle}",
                conflicting_task_ids=cycle,
                conflict_type="cyclic_dependency",
            )
        order = topological_sort(task_ids, full_graph)
        priority = {task["id"]: float(task.get("scheduler_priority", 0.0)) for task in self.task_history.get("current", [])}
        return self._priority_stable_topological_order(order, full_graph, priority)

    def _priority_stable_topological_order(
        self,
        topo_order: List[str],
        dependency_map: Dict[str, List[str]],
        priority: Dict[str, float],
    ) -> List[str]:
        rank = {tid: idx for idx, tid in enumerate(topo_order)}
        indegree = {tid: 0 for tid in topo_order}
        successors: Dict[str, List[str]] = defaultdict(list)
        for tid, deps in dependency_map.items():
            for dep in deps:
                if dep in indegree:
                    indegree[tid] += 1
                    successors[dep].append(tid)

        ready = [tid for tid in topo_order if indegree[tid] == 0]
        ready.sort(key=lambda tid: (-priority.get(tid, 0.0), rank.get(tid, 0)))
        result: List[str] = []
        while ready:
            tid = ready.pop(0)
            result.append(tid)
            for succ in successors.get(tid, []):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
                    ready.sort(key=lambda x: (-priority.get(x, 0.0), rank.get(x, 0)))
        return result

    def _create_assignment(
        self,
        task_id: str,
        agent_id: str,
        agent_details: Dict,
        current_load: float,
        state: Optional[Dict],
    ) -> Dict[str, Any]:
        task = self._last_tasks_by_id.get(task_id)
        if task is None:
            task = next((t for t in self.task_history.get("current", []) if t.get("id") == task_id), None)
        if task is None:
            task = {"id": task_id, "name": task_id, "requirements": [], "risk_assessment": {"risk_score": 0.0}}

        duration = self._estimate_duration(task, agent_details)
        start_time = max(time.time(), float(current_load))
        end_time = estimate_end_time(start_time, duration)
        deadline = float(task.get("deadline", start_time + self.default_deadline_seconds))
        lateness = max(0.0, end_time - deadline)
        risk_score = clamp(float(task.get("risk_score", task.get("risk_assessment", {}).get("risk_score", 0.0))), 0.0, 1.0)
        status = "scheduled" if lateness <= 0.0 else "deadline_risk"

        return {
            "task_id": task_id,
            "task_name": task.get("name", task_id),
            "agent_id": agent_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "deadline": deadline,
            "lateness_seconds": lateness,
            "risk_score": risk_score,
            "priority": task.get("priority", 1),
            "scheduler_priority": task.get("scheduler_priority", 0.0),
            "requirements": list(task.get("requirements", [])),
            "dependencies": list(task.get("dependencies", [])),
            "status": status,
            "metadata": {
                "agent_efficiency": float(agent_details.get(self.efficiency_attribute, 1.0)),
                "agent_load": float(agent_details.get("current_load", 0.0)),
                "risk_assessment": copy.deepcopy(task.get("risk_assessment", {})),
            },
        }

    # ------------------------------------------------------------------
    # Risk mitigation
    # ------------------------------------------------------------------
    def _apply_risk_mitigation(self, schedule: Dict, risk_assessor: Optional[Callable]) -> Dict:
        mitigated: Dict[str, Dict[str, Any]] = {}
        for task_id, assignment in schedule.items():
            risk_score = float(assignment.get("risk_score", 0.0))
            if risk_score > self.risk_threshold:
                alt = self._find_alternative(task_id, assignment, schedule, risk_assessor)
                if alt:
                    mitigated[task_id] = alt
                    continue
                assignment = copy.deepcopy(assignment)
                assignment["mitigation_strategy"] = "monitored_execution"
                assignment["requires_monitoring"] = True
            mitigated[task_id] = assignment
        return mitigated

    def _find_alternative(
        self,
        task_id: str,
        assignment: Dict,
        schedule: Dict,
        risk_assessor: Optional[Callable],
    ) -> Optional[Dict]:
        task = self._last_tasks_by_id.get(task_id)
        if not task:
            return None

        original_agent = assignment.get("agent_id")
        alternatives = [
            (aid, score) for aid, score in self._last_candidate_map.get(task_id, [])
            if aid != original_agent and score >= float(self.risk_mitigation.get("reassign_min_score", 0.05))
        ]
        alternatives.sort(key=lambda item: item[1], reverse=True)

        for agent_id, score in alternatives:
            details = self.agents[agent_id]
            start = max(time.time(), float(details.get("available_from", time.time())))
            candidate = self._create_assignment(task_id, agent_id, details, start, self.state)
            candidate["agent_score"] = score
            candidate["risk_score"] = clamp(candidate["risk_score"] * (1.0 - 0.10 * score), 0.0, 1.0)
            candidate["mitigation_strategy"] = f"reassigned_from_{original_agent}"
            if candidate["risk_score"] < assignment.get("risk_score", 1.0):
                return candidate

        if self.allow_decomposition_mitigation:
            subtasks = self._decompose_task(task)
            if subtasks:
                candidate = copy.deepcopy(assignment)
                candidate["subtasks"] = subtasks
                candidate["risk_score"] = clamp(
                    float(candidate.get("risk_score", 0.0))
                    - float(self.risk_mitigation.get("risk_reduction_on_decomposition", 0.25)),
                    0.0,
                    1.0,
                )
                candidate["mitigation_strategy"] = "task_decomposition"
                return candidate

        delay = float(self.retry_policy.get("delay", 10.0)) * float(self.risk_mitigation.get("delay_factor", 1.0))
        if delay > 0.0:
            candidate = copy.deepcopy(assignment)
            candidate["start_time"] += delay
            candidate["end_time"] += delay
            candidate["lateness_seconds"] = max(0.0, candidate["end_time"] - candidate["deadline"])
            candidate["risk_score"] = clamp(
                float(candidate.get("risk_score", 0.0))
                - float(self.risk_mitigation.get("risk_reduction_on_delay", 0.05)),
                0.0,
                1.0,
            )
            candidate["mitigation_strategy"] = f"delayed_by_{delay:.1f}s"
            return candidate
        return None

    def _calculate_agent_risk(self, task: Dict, agent_details: Dict) -> float:
        requirements = set(task.get("requirements", []))
        capabilities = set(agent_details.get("capabilities", []))
        gap_risk = 0.0 if not requirements else 1.0 - (len(capabilities & requirements) / len(requirements))
        successes = max(0, int(agent_details.get("successes", 1) or 0))
        failures = max(0, int(agent_details.get("failures", 0) or 0))
        perf_risk = failures / max(1, successes + failures)
        load_risk = clamp(float(agent_details.get("current_load", 0.0)) / self.max_agent_load, 0.0, 1.0)
        return clamp(0.50 * gap_risk + 0.30 * perf_risk + 0.20 * load_risk, 0.0, 1.0)

    def _decompose_task(self, task: Dict) -> Optional[List[Dict]]:
        """Domain hook for subclasses.  Returns None by default."""
        return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _schedule_state(self, tasks: List[Dict[str, Any]], state: Optional[Dict]) -> Dict[str, Any]:
        merged = copy.deepcopy(state or {})
        merged["tasks"] = tasks
        merged["dependency_map"] = {task["id"]: list(task.get("dependencies", [])) for task in tasks}
        return merged

    def _estimate_duration(self, task: Dict[str, Any], agent_details: Dict[str, Any]) -> float:
        explicit = float(task.get("duration", task.get("estimated_duration", 0.0)) or 0.0)
        if explicit > 0.0:
            base_duration = explicit
        else:
            base_duration = self.base_duration_per_requirement * max(len(task.get("requirements", [])), 1)
        efficiency = max(0.01, float(agent_details.get(self.efficiency_attribute, 1.0) or 1.0))
        return max(0.001, base_duration / efficiency)

    def _normalise_deadline(self, raw_deadline: Any, now: float) -> float:
        if raw_deadline in (None, ""):
            return now + self.default_deadline_seconds
        deadline = float(raw_deadline)
        # Small positive values are treated as relative offsets, preserving the
        # style used by the wider planning stack.
        if 0.0 < deadline < now and deadline <= self.default_deadline_seconds * 10.0:
            return now + deadline
        if deadline <= 0.0:
            return now + self.default_deadline_seconds
        return deadline

    def _normalise_available_from(self, details: Dict[str, Any], load: float, now: float) -> float:
        for key in ("available_from", "available_at", "next_available_time"):
            if key in details and details[key] not in (None, ""):
                value = float(details[key])
                return now + value if 0.0 <= value < now and value <= self.default_deadline_seconds * 10.0 else value
        return now + min(load, self.max_agent_load) * self.load_time_scale_seconds

    @staticmethod
    def _as_string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (set, tuple, list)):
            return list(dict.fromkeys(str(v).strip() for v in value if str(v).strip()))
        return [str(value).strip()] if str(value).strip() else []

    def _task_requirements_from_object(self, task: Task) -> List[str]:
        reqs: List[str] = []
        for attr in ("requirements", "required_skills", "required_tools", "capability_requirements"):
            value = getattr(task, attr, None)
            if isinstance(value, ResourceProfile):
                continue
            reqs.extend(self._as_string_list(value))
        return list(dict.fromkeys(reqs))

    @staticmethod
    def _normalise_resource_profile(value: Any) -> ResourceProfile:
        if isinstance(value, ResourceProfile):
            return value
        if isinstance(value, dict):
            return ResourceProfile(
                gpu=float(value.get("gpu", 0.0) or 0.0),
                ram=float(value.get("ram", 0.0) or value.get("memory", 0.0) or 0.0),
                specialized_hardware=list(value.get("specialized_hardware", []) or []),
            )
        return ResourceProfile()

    @staticmethod
    def _normalise_agent_resources(value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, ResourceProfile):
            return {
                "gpu": value.gpu,
                "ram": value.ram,
                "specialized_hardware": list(value.specialized_hardware),
            }
        if isinstance(value, dict):
            return copy.deepcopy(value)
        return None

    def explain_last_schedule(self) -> Dict[str, Any]:
        """Return the diagnostics for the most recent scheduling pass."""
        return copy.deepcopy(self.last_schedule_report)


if __name__ == "__main__":
    print("\n=== Running Task Scheduler ===\n")
    printer.status("TEST", "Task Scheduler initialized", "info")

    scheduler = DeadlineAwareScheduler()
    now = time.time()
    tasks = [
        {
            "id": "prepare",
            "requirements": ["prep"],
            "deadline": now + 120,
            "duration": 20,
            "priority": 4,
        },
        {
            "id": "deliver",
            "requirements": ["navigation", "door_access"],
            "deadline": now + 300,
            "duration": 40,
            "priority": 8,
            "dependencies": ["prepare"],
            "risk_score": 0.25,
        },
    ]
    agents = {
        "agent_a": {
            "capabilities": ["prep", "navigation", "door_access"],
            "current_load": 0.1,
            "successes": 12,
            "failures": 1,
            "efficiency": 1.2,
            "status": "available",
        },
        "agent_b": {
            "capabilities": ["prep"],
            "current_load": 0.0,
            "successes": 5,
            "failures": 0,
            "efficiency": 1.0,
            "status": "available",
        },
    }
    plan = scheduler.schedule(tasks=tasks, agents=agents, state={"tasks": tasks}) # type: ignore
    assert "prepare" in plan and "deliver" in plan
    assert plan["deliver"]["start_time"] >= plan["prepare"]["end_time"]
    printer.pretty("Schedule", plan, "success")
    printer.pretty("Diagnostics", scheduler.explain_last_schedule(), "info")
    print("\n=== Test ran successfully ===\n")
