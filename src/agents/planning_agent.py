"""
Production-hardened Planning Agent entry point for SLAI.
"""

from __future__ import annotations

__version__ = "2.1.0"

import copy
import random
import time

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any as TypingAny, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .base.utils.main_config_loader import load_global_config, get_config_section
from .base_agent import BaseAgent

from .planning.planning_types import Any, ResourceProfile, Task, TaskStatus, TaskType, WorldState
from .planning.planning_metrics import PlanningMetrics
from .planning.planning_executor import PlanningExecutor
from .planning.heuristic_selector import HeuristicSelector
from .planning.task_scheduler import DeadlineAwareScheduler
from .planning.probabilistic_planner import ProbabilisticPlanner
from .planning.safety_planning import SafetyPlanning
from .planning.local_behavior_arbitrator import LocalBehaviorArbitrator, LocalPlanningContext
from .planning.utils.resource_monitor import ResourceMonitor
from .planning.utils.planning_errors import *
from .planning.utils.planning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Planning Agent")
printer = PrettyPrinter()

CostProfile = Tuple[float, float]
StateTuple = Tuple[Tuple[str, TypingAny], ...]


def _coerce_float(value: TypingAny, default: float = 0.0, *, minimum: Optional[float] = None,
                  maximum: Optional[float] = None) -> float:
    """Coerce numeric input while delegating bounds to the shared clamp helper."""
    try:
        result = default if value is None else float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def _coerce_int(value: TypingAny, default: int = 0, *, minimum: Optional[int] = None,
                maximum: Optional[int] = None) -> int:
    """Coerce integer input with optional bounds."""
    try:
        result = default if value is None else int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def _status_name(value: TypingAny) -> str:
    """Return a stable status name for enum and non-enum values."""
    if isinstance(value, TaskStatus):
        return value.name
    return str(value)


def _error_payload(error: BaseException) -> Dict[str, TypingAny]:
    """Return structured, JSON-safe error information."""
    if isinstance(error, PlanningError):
        return error.to_dict()
    return {
        "error_class": type(error).__name__,
        "message": str(error),
        "timestamp": time.time(),
    }


def _safe_error_message(error: BaseException, limit: int = 240) -> str:
    """Return a logging-safe error message using the shared truncation helper."""
    try:
        return truncate_for_logging(str(error), limit)
    except Exception:
        return str(error)[:limit]


class PlanningAgent(BaseAgent):
    """
    Agent-level coordinator for hierarchical, scheduled, safety-aware planning.

    This class intentionally does not own the planning subsystem configuration.
    Agent-level settings are loaded from the global agents configuration via
    ``get_config_section("planning_agent")``.  Subsystem objects such as
    DeadlineAwareScheduler, PlanningExecutor, PlanningMetrics, SafetyPlanning,
    PlanningMemory, HeuristicSelector, and ProbabilisticPlanner load their own
    planning subsystem settings internally.
    """

    def __init__(self, shared_memory, agent_factory, config: Optional[Dict[str, TypingAny]] = None, **kwargs):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
        )

        self.config = load_global_config()
        self.agent_config: Dict[str, TypingAny] = self._load_agent_config(config)
        # Backward-compatible alias.  This is agent config, not planning subsystem config.
        self.planning_config = self.agent_config

        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        history_window = _coerce_int(self.agent_config.get("execution_history_window"), 100)
        plan_history_window = _coerce_int(self.agent_config.get("plan_history_window"), 1000)

        self.current_goal: Optional[Task] = None
        self.current_plan: List[Task] = []
        self.task_library: Dict[str, Task] = {}
        self.world_state: Dict[str, TypingAny] = {}
        self.execution_history: Deque[Dict[str, TypingAny]] = deque(maxlen=history_window)
        self.plan_history: Deque[Dict[str, TypingAny]] = deque(maxlen=plan_history_window)
        self.method_stats: Dict[TypingAny, Dict[str, float]] = defaultdict(
            lambda: {"success": 0.0, "total": 0.0, "avg_cost": 0.0}
        )
        self.schedule_state: Dict[str, TypingAny] = {
            "agent_loads": defaultdict(float),
            "task_history": defaultdict(list),
        }
        self.memo_table: Dict[Tuple[Tuple[str, int], WorldState], List[Task]] = {}
        self.expected_state_projections: Dict[str, Dict[str, TypingAny]] = {}
        self.execution_interrupted = False
        self.adaptive_context: Dict[str, TypingAny] = {}
        self._node_cache: Dict[str, TypingAny] = {}
        self._planning_start_time: Optional[float] = None
        self._planning_end_time: Optional[float] = None
        self._last_replan_at: float = 0.0
        self.last_error: Optional[Dict[str, TypingAny]] = None
        self.last_diagnostics: Dict[str, TypingAny] = {}

        self.scheduler = DeadlineAwareScheduler()
        self.metrics = PlanningMetrics()
        self.heuristic_selector = HeuristicSelector()
        self.safety_planner = SafetyPlanning()
        self.resource_monitor = ResourceMonitor()
        self.executor = PlanningExecutor()
        self.executor.agent = self
        self.probabilistic_planner = ProbabilisticPlanner()
        self.local_arbitrator = LocalBehaviorArbitrator()

        # Wire shared resources where subsystem classes expose compatible fields.
        if hasattr(self.safety_planner, "resource_monitor"):
            self.safety_planner.resource_monitor = self.resource_monitor
        if hasattr(self.safety_planner, "base_state"):
            self.safety_planner.base_state = {"execution_history": self.execution_history}

        # self._configure_resource_limits()
        self._ensure_shared_memory_defaults()

        logger.info("PlanningAgent successfully initialized")

    # ------------------------------------------------------------------
    # Configuration and shared-memory helpers
    # ------------------------------------------------------------------
    def _load_agent_config(self, explicit_config: Optional[Dict[str, TypingAny]]) -> Dict[str, TypingAny]:
        """
        Load agent-level configuration from agents_config.yaml via main_config_loader.

        The planning subsystem still owns planning_config.yaml internally.  This
        method intentionally reads only agent-level settings.
        """
        defaults = {
            "execution_history_window": 100,
            "plan_history_window": 1000,
            "gpu_limit": 1.0,
            "ram_limit": 16.0,
            "fallback_agent_id": "planner",
            "fallback_agent_capabilities": ["gpu", "ram"],
            "fallback_task_duration": 30.0,
            "fallback_task_deadline_seconds": 3600.0,
            "heuristic_time_budget": 0.5,
            "default_start_offset_seconds": 1.0,
            "default_deadline_seconds": 3600.0,
            "max_policy_steps": 100,
            "replan_debounce_seconds": 1.0,
            "high_load_replan_threshold": 85.0,
            "low_score_replan_threshold": 0.60,
            "resource_safety_buffers": {"gpu": 0.15, "ram": 0.20},
        }

        if isinstance(explicit_config, dict):
            raw = explicit_config
        else:
            raw = get_config_section("planning_agent")
            if not isinstance(raw, dict):
                agents_section = get_config_section("agents")
                raw = {}
                if isinstance(agents_section, dict):
                    nested = agents_section.get("planning_agent") or agents_section.get("PlanningAgent")
                    if isinstance(nested, dict):
                        raw = nested

        config = deep_update(defaults, dict(raw or {}))

        try:
            config["execution_history_window"] = _coerce_int(config.get("execution_history_window"), 100, minimum=1)
            config["plan_history_window"] = _coerce_int(config.get("plan_history_window"), 1000, minimum=1)
            config["gpu_limit"] = _coerce_float(config.get("gpu_limit"), 1.0, minimum=0.0)
            config["ram_limit"] = _coerce_float(config.get("ram_limit"), 16.0, minimum=0.0)
            config["fallback_task_duration"] = _coerce_float(config.get("fallback_task_duration"), 30.0, minimum=1.0)
            config["fallback_task_deadline_seconds"] = _coerce_float(config.get("fallback_task_deadline_seconds"), 3600.0, minimum=1.0)
            config["heuristic_time_budget"] = _coerce_float(config.get("heuristic_time_budget"), 0.5, minimum=0.001)
            validate_probability(_coerce_float(config.get("low_score_replan_threshold"), 0.60, minimum=0.0, maximum=1.0), "planning_agent.low_score_replan_threshold")
            if not isinstance(config.get("fallback_agent_capabilities"), list):
                raise PlanningConfigError(
                    "planning_agent.fallback_agent_capabilities must be a list",
                    config_key="fallback_agent_capabilities",
                    config_section="planning_agent",
                    expected_type="list[str]",
                )
        except PlanningError:
            raise
        except Exception as exc:
            raise PlanningConfigError(
                f"Invalid planning_agent configuration: {exc}",
                config_section="planning_agent",
                context={"source": "agents_config.yaml"},
            ) from exc

        if not raw:
            logger.warning("No planning_agent section found in agents_config.yaml. Using safe agent defaults.")
        return config

    #def _configure_resource_limits(self) -> None:
    #    """Apply agent-level resource caps without overriding subsystem internals unnecessarily."""
    #    gpu_limit = _coerce_float(self.agent_config.get("gpu_limit"), 1.0, minimum=0.0)
    #    ram_limit = _coerce_float(self.agent_config.get("ram_limit"), 16.0, minimum=0.0)
    #    if hasattr(self.resource_monitor, "gpu_limit"):
    #        self.resource_monitor.gpu_limit = gpu_limit
    #    if hasattr(self.resource_monitor, "ram_limit"):
    #        self.resource_monitor.ram_limit = ram_limit

    def _ensure_shared_memory_defaults(self) -> None:
        base_state = getattr(self.shared_memory, "base_state", None)
        if isinstance(base_state, dict):
            base_state.setdefault("execution_history", [])
            base_state.setdefault("planning", {})
            base_state["planning"].setdefault("plan_history", [])

    def _sm_get(self, key: str, default: TypingAny = None) -> TypingAny:
        if self.shared_memory is None:
            return default
        getter = getattr(self.shared_memory, "get", None)
        if callable(getter):
            try:
                return getter(key, default=default)
            except TypeError:
                try:
                    value = getter(key)
                    return default if value is None else value
                except Exception:
                    return default
            except Exception:
                return default
        if isinstance(self.shared_memory, dict):
            return self.shared_memory.get(key, default)
        try:
            return self.shared_memory[key]
        except Exception:
            return default

    def _sm_set(self, key: str, value: TypingAny) -> None:
        if self.shared_memory is None:
            return
        setter = getattr(self.shared_memory, "set", None)
        if callable(setter):
            try:
                setter(key, value)
                return
            except Exception as exc:
                logger.debug("Shared memory set failed for key %s: %s", key, _safe_error_message(exc))
        if isinstance(self.shared_memory, dict):
            self.shared_memory[key] = value

    def _record_error(
        self,
        error: BaseException,
        *,
        stage: str,
        task: Optional[Task] = None,
        extra: Optional[Dict[str, TypingAny]] = None,
    ) -> Dict[str, TypingAny]:
        """Store the latest structured planning error and mirror it into shared memory."""
        payload = _error_payload(error)
        payload["stage"] = stage
        if task is not None:
            payload["task_name"] = getattr(task, "name", "")
            payload["task_id"] = getattr(task, "id", getattr(task, "name", ""))
        if extra:
            payload["extra"] = dict(extra)

        self.last_error = payload
        self.last_diagnostics.setdefault("errors", []).append(payload)
        self._sm_set("planning:last_error", payload)
        return payload

    # ------------------------------------------------------------------
    # State conversion
    # ------------------------------------------------------------------
    def get_current_state_tuple(self) -> WorldState:
        """Return an immutable, stable representation of the current world state."""
        items: List[Tuple[str, TypingAny]] = []
        for key, value in self.world_state.items():
            if isinstance(value, Any):          # type: ignore # custom class
                value = value.value          # type: ignore
            try:
                hash(value)
                hashable_value = value
            except Exception:
                hashable_value = safe_json_dumps(value)
            items.append((str(key), hashable_value))
        return tuple(sorted(items))

    def load_state_from_tuple(self, state_tuple: WorldState) -> None:
        """Restore world_state from an immutable tuple representation."""
        self.world_state = dict(state_tuple)
        logger.debug("World state restored from tuple")

    # ------------------------------------------------------------------
    # Task registration and safety
    # ------------------------------------------------------------------
    def register_task(self, task: Task) -> None:
        """Register a task template and its decomposition methods."""
        self._validate_task_identity(task)
        self._validate_task_safety(task)

        if task.name in self.task_library:
            logger.warning("Task %r already registered. Overwriting.", task.name)
        self.task_library[task.name] = task

        if getattr(task, "is_probabilistic", False):
            self._register_probabilistic_task_actions(task)

        if task.task_type == TaskType.ABSTRACT and not task.methods:
            logger.warning("Abstract task %r has no methods. Adding safe fallback.", task.name)
            task.methods = [[self._build_fallback_task(task)]]

    def _validate_task_identity(self, task: Task) -> None:
        """Validate task identity through the shared planning error model."""
        if not isinstance(task, Task):
            raise PlanningConfigError(
                "register_task expects a planning_types.Task instance",
                config_key="task",
                config_section="planning_agent",
                expected_type="Task",
                context={"actual_type": type(task).__name__},
            )
        if not getattr(task, "name", ""):
            raise AcademicPlanningError(
                "Task name cannot be empty",
                context={"task_id": getattr(task, "id", "")},
            )
        if not isinstance(getattr(task, "task_type", None), TaskType):
            raise AcademicPlanningError(
                "Invalid task_type for task",
                context={"task_name": task.name, "task_type": repr(getattr(task, "task_type", None))},
            )
        task_id = str(getattr(task, "id", ""))
        if task_id and not is_valid_task_id(task_id):
            raise AcademicPlanningError(
                f"Invalid task id for {task.name!r}: {task_id!r}",
                context={"task_name": task.name, "task_id": task_id},
            )

    def _validate_task_safety(self, task: Task) -> bool:
        """
        Validate resource safety through planning_helpers.check_resource_feasibility.

        This keeps resource math consistent with the rest of the planning subsystem.
        """
        profile = getattr(task, "resource_requirements", None)
        if profile is None:
            raise ResourceViolation(
                f"Task {task.name!r} is missing resource requirements",
                resource_type="resource_profile",
                requested=None,
                available={},
                task_id=getattr(task, "id", task.name),
            )

        requirements = {
            "gpu": _coerce_float(getattr(profile, "gpu", 0.0), 0.0, minimum=0.0),
            "ram": _coerce_float(getattr(profile, "ram", 0.0), 0.0, minimum=0.0),
        }
        available = {
            "gpu": max(_coerce_float(getattr(self.resource_monitor, "gpu_limit", self.agent_config.get("gpu_limit")), 1.0), 1e-9),
            "ram": max(_coerce_float(getattr(self.resource_monitor, "ram_limit", self.agent_config.get("ram_limit")), 16.0), 1e-9),
        }
        buffers = dict(self.agent_config.get("resource_safety_buffers", {}) or {})

        try:
            check_resource_feasibility(
                requirements,
                available,
                safety_buffers=buffers,
                task_id=getattr(task, "id", task.name),
            )
            return True
        except ResourceViolation as exc:
            self._record_error(exc, stage="task_safety", task=task)
            raise

    def _register_probabilistic_task_actions(self, task: Task) -> None:
        for item in getattr(task, "probabilistic_actions", []) or []:
            try:
                if isinstance(item, tuple) and len(item) == 2:
                    probability, effect = item
                    validate_probability(float(probability), f"{task.name}.probability")
                    self.probabilistic_planner.register_action(
                        {
                            "name": task.name,
                            "probability": probability,
                            "effect": effect,
                            "preconditions": lambda _state: True,
                        }
                    )
                else:
                    raise MethodSelectionError(
                        "Malformed probabilistic action",
                        task_name=task.name,
                        task_id=getattr(task, "id", task.name),
                        candidate_methods=[repr(item)],
                    )
            except PlanningError as exc:
                self._record_error(exc, stage="probabilistic_action_registration", task=task)
                logger.warning("Skipping probabilistic action for %s: %s", task.name, _safe_error_message(exc))
            except Exception as exc:
                wrapped = MethodSelectionError(
                    f"Failed to register probabilistic action for {task.name}: {exc}",
                    task_name=task.name,
                    task_id=getattr(task, "id", task.name),
                )
                self._record_error(wrapped, stage="probabilistic_action_registration", task=task)
                logger.warning("%s", _safe_error_message(wrapped))

    def _build_fallback_task(self, task: Task) -> Task:
        return Task(
            name=f"{task.name}_fallback",
            task_type=TaskType.PRIMITIVE,
            preconditions=[lambda _state: True],
            effects=[lambda state: state.update({"fallback_completed": True})],
            resource_requirements=ResourceProfile(gpu=0, ram=0, specialized_hardware=[]),
            duration=_coerce_float(self.agent_config.get("fallback_task_duration"), 30.0),
            deadline=time.time() + _coerce_float(self.agent_config.get("fallback_task_deadline_seconds"), 3600.0),
            priority=getattr(task, "priority", 1),
        )

    # ------------------------------------------------------------------
    # Decomposition and planning
    # ------------------------------------------------------------------
    def decompose_task(self, task_to_decompose: Task, current_state: Dict[str, TypingAny]) -> Optional[List[Task]]:
        """Recursively decompose an abstract task into primitive tasks."""
        try:
            self._validate_task_identity(task_to_decompose)
        except PlanningError as exc:
            self._record_error(exc, stage="decomposition_identity", task=task_to_decompose if isinstance(task_to_decompose, Task) else None)
            raise

        logger.debug("Decomposing task: %s", task_to_decompose.name)

        if task_to_decompose.task_type == TaskType.PRIMITIVE:
            try:
                check_preconditions(
                    current_state,
                    list(getattr(task_to_decompose, "preconditions", []) or []),
                    task_name=task_to_decompose.name,
                    task_id=getattr(task_to_decompose, "id", task_to_decompose.name),
                )
                self._ensure_task_timing(task_to_decompose)
                return [task_to_decompose]
            except PreconditionViolation as exc:
                task_to_decompose.status = TaskStatus.FAILED
                self._record_error(exc, stage="primitive_preconditions", task=task_to_decompose)
                logger.warning("%s", _safe_error_message(exc))
                return None
            except PlanningError:
                raise

        library_task = self.task_library.get(task_to_decompose.name, task_to_decompose)
        if not getattr(library_task, "methods", None):
            error = DecompositionError(
                f"Task {task_to_decompose.name!r} has no registered decomposition methods",
                task_name=task_to_decompose.name,
                task_id=getattr(task_to_decompose, "id", task_to_decompose.name),
                attempted_methods=[],
            )
            task_to_decompose.status = TaskStatus.FAILED
            self._record_error(error, stage="decomposition", task=task_to_decompose)
            logger.error("%s", _safe_error_message(error))
            return None

        attempted: List[str] = []
        for method_index, _score in self._rank_methods(task_to_decompose, library_task):
            attempted.append(str(method_index))
            memo_key = ((task_to_decompose.name, int(method_index)), self.get_current_state_tuple())
            if memo_key in self.memo_table:
                return [task.copy() for task in self.memo_table[memo_key]]

            candidate = self._try_decomposition_method(task_to_decompose, library_task, int(method_index), current_state)
            if candidate:
                self.memo_table[memo_key] = [task.copy() for task in candidate]
                return candidate

        error = DecompositionError(
            f"All decomposition methods failed for task {task_to_decompose.name!r}",
            task_name=task_to_decompose.name,
            task_id=getattr(task_to_decompose, "id", task_to_decompose.name),
            attempted_methods=attempted,
        )
        task_to_decompose.status = TaskStatus.FAILED
        self._record_error(error, stage="decomposition", task=task_to_decompose)
        return None

    def _rank_methods(self, task: Task, library_task: Task) -> List[Tuple[int, float]]:
        candidate_methods = [str(i) for i in range(len(library_task.methods))]
        scores: List[Tuple[int, float]] = []

        if not candidate_methods:
            raise MethodSelectionError(
                f"No candidate methods available for {task.name}",
                task_name=task.name,
                task_id=getattr(task, "id", task.name),
                candidate_methods=[],
            )

        for method_id in candidate_methods:
            try:
                probability = self.heuristic_selector.predict_success_prob(
                    task=self._task_feature_payload(task, int(method_id)),
                    world_state=self.world_state,
                    method_stats=self.method_stats,
                    method_id=method_id,
                    time_budget=_coerce_float(self.agent_config.get("heuristic_time_budget"), 0.5, minimum=0.001),
                )
                validate_probability(float(probability), f"{task.name}.method_{method_id}.probability")
            except PlanningError as exc:
                self._record_error(exc, stage="method_ranking", task=task)
                logger.debug("Heuristic probability fallback for %s method %s: %s", task.name, method_id, _safe_error_message(exc))
                probability = self._method_success_rate(task.name, method_id)
            except Exception as exc:
                wrapped = MethodSelectionError(
                    f"Heuristic failed for method {method_id}: {exc}",
                    task_name=task.name,
                    task_id=getattr(task, "id", task.name),
                    candidate_methods=candidate_methods,
                )
                self._record_error(wrapped, stage="method_ranking", task=task)
                probability = self._method_success_rate(task.name, method_id)

            scores.append((int(method_id), clamp(float(probability), 0.0, 1.0)))

        scores.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        return scores

    def _task_feature_payload(self, task: Task, selected_method: Optional[int] = None) -> Dict[str, TypingAny]:
        return {
            "id": getattr(task, "id", getattr(task, "name", "task")),
            "name": getattr(task, "name", "task"),
            "selected_method": selected_method if selected_method is not None else getattr(task, "selected_method", 0),
            "priority": _coerce_float(getattr(task, "priority", 1), 1.0),
            "goal_state": getattr(task, "goal_state", {}) or {},
            "parent": getattr(task, "parent", None),
            "creation_time": getattr(task, "created_at", None),
            "deadline": getattr(task, "deadline", None),
            "risk_score": _coerce_float(getattr(task, "risk_score", 0.0), 0.0),
            "task_type": getattr(getattr(task, "task_type", None), "name", str(getattr(task, "task_type", ""))),
        }

    def _method_success_rate(self, task_name: str, method_id: Union[str, int]) -> float:
        keys = [(task_name, str(method_id)), (task_name, int(method_id)) if str(method_id).isdigit() else None, str(method_id)]
        for key in keys:
            if key is None:
                continue
            stats = self.method_stats.get(key)
            if stats:
                total = _coerce_float(stats.get("total"), 0.0, minimum=0.0)
                if total > 0:
                    return clamp(_coerce_float(stats.get("success"), 0.0, minimum=0.0) / total, 0.0, 1.0)
        return 0.5

    def _try_decomposition_method(
        self,
        task_to_decompose: Task,
        library_task: Task,
        method_index: int,
        current_state: Dict[str, TypingAny],
    ) -> Optional[List[Task]]:
        if not (0 <= method_index < len(library_task.methods)):
            return None

        task_to_decompose.selected_method = method_index
        simulated_state = copy.deepcopy(current_state)
        result: List[Task] = []

        for subtask_template in library_task.get_subtasks(method_index):
            subtask = subtask_template.copy()
            subtask.parent = task_to_decompose
            subtask.parent_task = task_to_decompose
            try:
                self._ensure_task_timing(subtask)
                check_preconditions(
                    simulated_state,
                    list(getattr(subtask, "preconditions", []) or []),
                    task_name=subtask.name,
                    task_id=getattr(subtask, "id", subtask.name),
                )
            except PreconditionViolation as exc:
                self._record_error(exc, stage="method_preconditions", task=subtask)
                logger.debug(
                    "Method %s rejected for %s because subtask %s failed preconditions: %s",
                    method_index,
                    task_to_decompose.name,
                    subtask.name,
                    _safe_error_message(exc),
                )
                return None

            nested = self.decompose_task(subtask, simulated_state)
            if not nested:
                return None

            result.extend(nested)
            for primitive in nested:
                if primitive.task_type == TaskType.PRIMITIVE:
                    simulated_state = apply_state_effects(simulated_state, list(getattr(primitive, "effects", []) or []))

        return result or None

    def _ensure_task_timing(self, task: Task) -> None:
        """Normalise task timing using shared temporal helpers."""
        now = time.time()
        duration = _coerce_float(getattr(task, "duration", 0.0), 0.0, minimum=1.0)
        task.duration = duration

        start = getattr(task, "start_time", 0.0)
        if not isinstance(start, (int, float)) or start <= 0 or start < now:
            start = now + _coerce_float(self.agent_config.get("default_start_offset_seconds"), 1.0, minimum=0.0)
        task.start_time = float(start)

        deadline = getattr(task, "deadline", 0.0)
        if not isinstance(deadline, (int, float)) or deadline <= task.start_time:
            deadline = task.start_time + max(
                duration,
                _coerce_float(self.agent_config.get("default_deadline_seconds"), 3600.0, minimum=duration),
            )
        task.deadline = float(deadline)

        try:
            compute_schedule_window(task.start_time, task.deadline, duration)
        except TemporalViolation as exc:
            self._record_error(exc, stage="task_timing", task=task)
            raise

        task.end_time = estimate_end_time(task.start_time, duration)

    def _find_alternative_methods(self, task: Task) -> List[int]:
        library_task = self.task_library.get(task.name, task)
        if task.task_type != TaskType.ABSTRACT or not getattr(library_task, "methods", None):
            return []

        candidate_methods = [str(i) for i in range(len(library_task.methods))]
        current = str(getattr(task, "selected_method", 0))

        try:
            best_method, _confidence = self.heuristic_selector.select_best_method(
                task=self._task_feature_payload(task),
                world_state=self.world_state,
                candidate_methods=candidate_methods,
                method_stats=self.method_stats,
                time_budget=_coerce_float(self.agent_config.get("heuristic_time_budget"), 0.5),
            )
            ordered = [str(best_method)] + [m for m in candidate_methods if m != str(best_method)]
        except Exception:
            ordered = [str(idx) for idx, _score in self._rank_methods(task, library_task)]

        return [int(method_id) for method_id in ordered if method_id != current and str(method_id).isdigit()]

    def replan(self, failed_task: Optional[Task]) -> Optional[List[Task]]:
        """Replan after failure using alternative methods or full goal decomposition."""
        failed_task = failed_task or self.current_goal
        if failed_task is None:
            logger.error("Cannot replan: failed_task and current_goal are both None")
            return None

        logger.warning("Replanning triggered by failed task: %s", failed_task.name)
        self._update_scheduler_state(failed_task)

        for method_idx in self._find_alternative_methods(failed_task):
            candidate_root = failed_task.copy()
            candidate_root.selected_method = method_idx
            candidate_plan = self.decompose_task(candidate_root, self.world_state)
            if candidate_plan and self._validate_plan(candidate_plan):
                self.current_plan = candidate_plan
                return candidate_plan

        if self.current_goal and failed_task is not self.current_goal:
            fallback = self.decompose_task(self.current_goal.copy(), self.world_state)
            if fallback and self._validate_plan(fallback):
                self.current_plan = fallback
                return fallback

        logger.error("All replanning attempts failed")
        return None

    def generate_plan(self, goal_task: Task) -> Optional[List[Task]]:
        """Generate a safe, scheduled executable plan for a goal task."""
        try:
            self._validate_task_identity(goal_task)
        except PlanningError as exc:
            self._record_error(exc, stage="goal_validation", task=goal_task if isinstance(goal_task, Task) else None)
            raise

        self._planning_start_time = time.time()
        self.current_goal = goal_task
        self._refresh_task_tree_timing(goal_task)

        try:
            if getattr(goal_task, "is_probabilistic", False):
                probabilistic_task = self._plan_probabilistic_goal(goal_task)
                if probabilistic_task is None:
                    raise GoalUnreachableError(
                        f"Probabilistic planner could not reach goal for {goal_task.name}",
                        goal_state=getattr(goal_task, "goal_state", {}) or {},
                        reason="probabilistic_policy_unavailable",
                    )
                plan = [probabilistic_task]
            else:
                plan = self.decompose_task(goal_task, self.world_state)

            if isinstance(plan, Task):
                plan = [plan]
            if not plan:
                raise GoalUnreachableError(
                    f"Task decomposition produced no executable plan for {goal_task.name}",
                    goal_state=getattr(goal_task, "goal_state", {}) or {},
                    reason="decomposition_failed",
                )

            self._assert_plan_acyclic(plan)

            if not self._validate_plan(plan):
                raise AcademicPlanningError(
                    "Generated plan failed safety validation",
                    context={"goal": goal_task.name, "plan_length": len(plan)},
                )

            scheduled_payload = self._convert_to_schedule_format(plan)
            schedule = self.scheduler.schedule(
                tasks=scheduled_payload,
                agents=self._get_available_agents(),
                risk_assessor=self._sm_get("risk_assessor"),
                state={"tasks": scheduled_payload, "dependency_map": self._dependency_map_for_schedule(plan)},
            )

            if not schedule:
                raise SchedulingConflictError(
                    "Scheduling failed; no valid schedule was created",
                    conflicting_task_ids=[getattr(task, "id", task.name) for task in plan],
                    conflict_type="no_feasible_schedule",
                )

            self._planning_end_time = time.time()
            self.current_plan = self._convert_to_plan(schedule, source_plan=plan)
            self.expected_state_projections = self._generate_state_projections(self.current_plan)
            self._record_plan_generation_metrics(plan, schedule) # type: ignore
            self.plan_history.append(
                {
                    "goal": goal_task.name,
                    "plan_length": len(self.current_plan),
                    "created_at": self._planning_end_time,
                    "schedule": schedule,
                }
            )
            self.last_diagnostics["last_schedule_report"] = getattr(self.scheduler, "last_schedule_report", {})
            return self.current_plan
        except PlanningError as exc:
            self._record_error(exc, stage="generate_plan", task=goal_task)
            logger.error("%s", _safe_error_message(exc))
            return self._handle_safety_violation(goal_task, exc) if isinstance(exc, (SafetyMarginError, ResourceViolation, TemporalError)) else None
        except Exception as exc:
            wrapped = PlanningError(
                f"Unexpected plan generation failure: {exc}",
                context={"goal": getattr(goal_task, "name", "")},
            )
            self._record_error(wrapped, stage="generate_plan", task=goal_task)
            logger.error("%s", _safe_error_message(wrapped))
            return None

    def _plan_probabilistic_goal(self, goal_task: Task) -> Optional[Task]:
        goal_state = getattr(goal_task, "goal_state", {}) or {}
        if not isinstance(goal_state, dict):
            raise PlanningConfigError(
                "Probabilistic goal_state must be a dictionary",
                config_key="goal_state",
                config_section="planning_agent",
                expected_type="dict",
            )

        report_or_policy = self.probabilistic_planner.perform_task(
            {
                "initial_state": self._json_safe_state(self.world_state),
                "goal_state": goal_state,
                "success_threshold": clamp(_coerce_float(getattr(goal_task, "success_threshold", 0.9), 0.9), 0.0, 1.0),
            }
        )
        if not report_or_policy:
            logger.error("Probabilistic planner did not produce a policy for %s", goal_task.name)
            return None

        task = goal_task.copy()
        task.task_type = TaskType.PRIMITIVE
        task.type = TaskType.PRIMITIVE
        task.context = dict(getattr(task, "context", {}) or {})
        task.context["probabilistic_policy"] = report_or_policy
        task.preconditions = task.preconditions or [lambda _state: True]
        return task

    def _json_safe_state(self, state: Dict[str, TypingAny]) -> Dict[str, TypingAny]:
        """Convert world state to a JSON-safe dict using shared serialization helpers."""
        result: Dict[str, TypingAny] = {}
        for key, value in state.items():
            if isinstance(value, Any):
                value = value.value
            if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                result[str(key)] = value
            else:
                try:
                    result[str(key)] = safe_json_loads(safe_json_dumps(value, fallback_str=repr(value)))
                except Exception:
                    result[str(key)] = repr(value)
            return result

    def _refresh_task_tree_timing(self, task: Task) -> None:
        self._ensure_task_timing(task)
        for method in getattr(task, "methods", []) or []:
            for subtask in method:
                if isinstance(subtask, Task):
                    self._refresh_task_tree_timing(subtask)

    # ------------------------------------------------------------------
    # Scheduling conversion
    # ------------------------------------------------------------------
    def _convert_to_schedule_format(self, plan: Sequence[Task]) -> List[Dict[str, TypingAny]]:
        return [self._task_to_schedule_payload(task) for task in plan]

    def _task_to_schedule_payload(self, task: Task) -> Dict[str, TypingAny]:
        self._ensure_task_timing(task)
        profile = getattr(task, "resource_requirements", ResourceProfile())
        requirements: List[str] = list(getattr(task, "required_skills", []) or [])
        if _coerce_float(getattr(profile, "gpu", 0.0), 0.0) > 0:
            requirements.append("gpu")
        if _coerce_float(getattr(profile, "ram", 0.0), 0.0) > 0:
            requirements.append("ram")
        requirements.extend(list(getattr(profile, "specialized_hardware", []) or []))

        return {
            "id": getattr(task, "id", task.name),
            "name": task.name,
            "requirements": sorted(set(str(req) for req in requirements if str(req).strip())),
            "deadline": getattr(task, "deadline", 0.0),
            "duration": max(_coerce_float(getattr(task, "duration", 1.0), 1.0), 1.0),
            "estimated_duration": max(_coerce_float(getattr(task, "estimated_duration", getattr(task, "duration", 1.0)), 1.0), 1.0),
            "risk_score": clamp(_coerce_float(getattr(task, "risk_score", 0.0), 0.0), 0.0, 1.0),
            "priority": _coerce_float(getattr(task, "priority", 1), 1.0),
            "dependencies": self._dependency_ids(task),
            "resource_requirements": profile,
            "metadata": {"source_task_name": task.name},
        }

    def _dependency_ids(self, task: Task) -> List[str]:
        result: List[str] = []
        for dependency in getattr(task, "dependencies", []) or []:
            if isinstance(dependency, Task):
                result.append(getattr(dependency, "id", dependency.name))
            elif isinstance(dependency, str):
                result.append(dependency)
        return result

    def _dependency_map_for_schedule(self, plan: Sequence[Task]) -> Dict[str, List[str]]:
        return {getattr(task, "id", task.name): self._dependency_ids(task) for task in plan}

    def _assert_plan_acyclic(self, plan: Sequence[Task]) -> None:
        """Validate the task dependency graph through shared helper functions."""
        dependency_map = self._dependency_map_for_schedule(plan)
        task_ids = [getattr(task, "id", task.name) for task in plan]
        cycle = detect_cycles(task_ids, dependency_map)
        if cycle:
            raise CyclicDependencyError(
                "Generated plan contains a cyclic dependency",
                cycle_path=cycle,
                context={"dependency_map": dependency_map},
            )
        # topological_sort raises CyclicDependencyError as a second guard and
        # gives downstream code a deterministic execution order if needed.
        topological_sort(task_ids, dependency_map)

    def _create_task_from_assignment(
        self,
        assignment: Dict[str, TypingAny],
        source_task: Optional[Task] = None,
    ) -> Task:
        task = source_task.copy() if source_task is not None else Task(
            name=assignment.get("task_name") or assignment.get("task_id") or "scheduled_task",
            task_type=TaskType.PRIMITIVE,
        )
        task.id = str(assignment.get("task_id", getattr(task, "id", task.name)))
        task.start_time = _coerce_float(assignment.get("start_time"), getattr(task, "start_time", 0.0))
        task.end_time = _coerce_float(assignment.get("end_time"), getattr(task, "end_time", 0.0))
        task.deadline = _coerce_float(assignment.get("deadline"), getattr(task, "deadline", 0.0))
        task.duration = max(_coerce_float(assignment.get("duration"), getattr(task, "duration", 1.0)), 1.0)
        task.risk_score = _coerce_float(assignment.get("risk_score"), getattr(task, "risk_score", 0.0))
        task.context = dict(getattr(task, "context", {}) or {})
        task.context["assigned_agent"] = assignment.get("agent_id")
        task.context["assignment"] = copy.deepcopy(assignment)
        return task

    def _convert_to_plan(
        self,
        schedule: Dict[str, Dict[str, Any]],
        source_plan: Optional[Sequence[Task]] = None,
    ) -> List[Task]:
        source_by_id = {getattr(task, "id", task.name): task for task in (source_plan or [])}
        ordered_assignments = sorted(
            schedule.values(),
            key=lambda item: (_coerce_float(item.get("start_time")), str(item.get("task_id")))
        )
        result: List[Task] = []
        for assignment in ordered_assignments:
            task_id = str(assignment.get("task_id", ""))
            source = source_by_id.get(task_id)
            if source is None:
                # Fallback: create a minimal task if source not found (should not happen)
                source = Task(
                    name=assignment.get("task_name") or assignment.get("task_id") or "scheduled_task", # type: ignore
                    task_type=TaskType.PRIMITIVE,
                )
            # Make a proper deep copy of the source task (now preserves callables)
            task = source.copy()
            # Update timing and assignment metadata from the schedule
            task.start_time = _coerce_float(assignment.get("start_time"), getattr(task, "start_time", 0.0))
            task.end_time = _coerce_float(assignment.get("end_time"), getattr(task, "end_time", 0.0))
            task.deadline = _coerce_float(assignment.get("deadline"), getattr(task, "deadline", 0.0))
            task.duration = max(_coerce_float(assignment.get("duration"), getattr(task, "duration", 1.0)), 1.0)
            task.risk_score = _coerce_float(assignment.get("risk_score"), getattr(task, "risk_score", 0.0))
            task.context = dict(getattr(task, "context", {}) or {})
            task.context["assigned_agent"] = assignment.get("agent_id")
            task.context["assignment"] = copy.deepcopy(assignment)
            result.append(task)
        return result

    def _get_available_agents(self) -> Dict[str, Dict[str, TypingAny]]:
        registry = self._sm_get("agent_registry", default={})
        if isinstance(registry, dict) and registry:
            return registry

        fallback_agent_id = str(self.agent_config.get("fallback_agent_id", "planner"))
        return {
            fallback_agent_id: {
                "capabilities": list(self.agent_config.get("fallback_agent_capabilities", ["gpu", "ram"])),
                "current_load": 0.0,
                "successes": 1,
                "failures": 0,
                "efficiency": 1.0,
                "status": "available",
                "available_resources": {
                    "gpu": _coerce_float(self.agent_config.get("gpu_limit"), 1.0),
                    "ram": _coerce_float(self.agent_config.get("ram_limit"), 16.0),
                    "specialized_hardware": [],
                },
            }
        }

    # ------------------------------------------------------------------
    # Safety, state projections, and metrics
    # ------------------------------------------------------------------
    def _validate_plan(self, plan: Sequence[Task]) -> bool:
        if not plan:
            self._record_error(
                GoalUnreachableError("Plan is empty", reason="empty_plan"),
                stage="plan_validation",
            )
            return False

        try:
            self._assert_plan_acyclic(plan)
            self.safety_planner.current_plan = plan # type: ignore
            safe = bool(self.safety_planner.safety_check(plan)) # type: ignore
            if not safe:
                raise AcademicPlanningError(
                    "SafetyPlanning returned False",
                    context={"plan_length": len(plan)},
                )
            return True
        except PlanningError as exc:
            self._record_error(exc, stage="plan_validation")
            logger.warning("Plan validation failed: %s", _safe_error_message(exc))
            return False
        except Exception as exc:
            wrapped = PlanningError(
                f"Unexpected plan validation failure: {exc}",
                context={"plan_length": len(plan)},
            )
            self._record_error(wrapped, stage="plan_validation")
            logger.warning("%s", _safe_error_message(wrapped))
            return False

    def _generate_state_projections(self, plan: List[Task]) -> Dict[str, Dict[str, TypingAny]]:
        projections: Dict[str, Dict[str, TypingAny]] = {}
        sim_state = copy.deepcopy(self.world_state)
        for task in plan:
            if task.task_type == TaskType.PRIMITIVE:
                sim_state = apply_state_effects(sim_state, list(getattr(task, "effects", []) or []))
                projections[task.name] = copy.deepcopy(sim_state)
        return projections

    def _handle_safety_violation(self, task: Task, error: Exception) -> Optional[List[Task]]:
        logger.warning("Safety violation detected for %s: %s", task.name, _safe_error_message(error))
        self._record_error(error, stage="safety_violation", task=task)

        try:
            candidates = self.safety_planner.dynamic_replanning_pipeline(task)
        except PlanningError as exc:
            self._record_error(exc, stage="safety_replanning", task=task)
            logger.error("Safety replanning failed: %s", _safe_error_message(exc))
            return None
        except Exception as exc:
            wrapped = ReplanningError(
                f"Safety replanning failed: {exc}",
                failed_task=task,
                failure_reason="safety_replanning_exception",
            )
            self._record_error(wrapped, stage="safety_replanning", task=task)
            logger.error("%s", _safe_error_message(wrapped))
            return None

        for candidate in candidates or []:
            candidate_plan = candidate if isinstance(candidate, list) else [candidate]
            if self._validate_plan(candidate_plan):
                return candidate_plan

        no_candidate = ReplanningError(
            "No safety-compliant alternative found",
            failed_task=task,
            candidates=list(candidates or []),
            failure_reason="no_safe_candidate",
        )
        self._record_error(no_candidate, stage="safety_replanning", task=task)
        logger.error("%s", _safe_error_message(no_candidate))
        return None

    def _record_plan_generation_metrics(self, decomposed_plan: List[Task], schedule: Dict[str, Dict[str, TypingAny]]) -> None:
        duration = 0.0
        if self._planning_start_time is not None and self._planning_end_time is not None:
            duration = max(0.0, self._planning_end_time - self._planning_start_time)

        try:
            if hasattr(self.metrics, "record_planning_metrics"):
                self.metrics.record_planning_metrics(
                    plan_length=len(decomposed_plan),
                    planning_time=duration,
                    success_rate=1.0 if schedule else 0.0,
                )
        except Exception as exc:
            logger.debug("Planning metric recording failed: %s", exc)

        end_times = [_coerce_float(assignment.get("end_time")) for assignment in schedule.values()]
        self._sm_set(
            "planning:last_plan_metadata",
            {
                "goal": self.current_goal.name if self.current_goal else None,
                "estimated_completion": max(end_times) if end_times else 0.0,
                "schedule": schedule,
                "planning_time": duration,
            },
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_plan(self, plan: Iterable[Any], goal: Any = None) -> Dict[str, TypingAny]:
        # Convert plan to list of Task (assuming each element is a Task)
        effective_plan: List[Task] = []
        for item in plan:
            if isinstance(item, Task):
                effective_plan.append(item)
            else:
                raise PlanningError(f"Unsupported plan element type: {type(item)}")
        # Apply local arbitration (safety overrides)
        effective_plan = self._apply_local_arbitration(effective_plan)
        # Use goal as the goal_task (if it is a Task)
        goal_task = goal if isinstance(goal, Task) else None
        execution_metrics = {
            "success_count": 0,
            "failure_count": 0,
            "total_cost": 0.0,
            "resource_usage": defaultdict(float),
        }

        if not effective_plan:
            error = GoalUnreachableError("Cannot execute an empty plan", reason="empty_execution_plan")
            self._record_error(error, stage="execute_plan")
            return {"status": TaskStatus.FAILED.name, "world_state": self.world_state, "metrics": execution_metrics, "error": error.to_dict()}

        self.execution_interrupted = False
        plan_meta = None

        try:
            self._assert_plan_acyclic(effective_plan)
        except PlanningError as exc:
            self._record_error(exc, stage="execute_plan")
            return {"status": TaskStatus.FAILED.name, "world_state": self.world_state, "metrics": execution_metrics, "error": exc.to_dict()}

        try:
            self.executor.start_monitoring(effective_plan, self.expected_state_projections)
        except Exception as exc:
            wrapped = PlanningError(f"Execution monitor failed to start: {exc}")
            self._record_error(wrapped, stage="executor_start")
            logger.warning("%s", _safe_error_message(wrapped))

        try:
            try:
                plan_meta = self.metrics.track_plan_start(effective_plan)
            except Exception as exc:
                self._record_error(PlanningError(f"PlanningMetrics.track_plan_start failed: {exc}"), stage="metrics_start")
                plan_meta = {"start_time": time.time(), "task_count": len(effective_plan)}

            for task in effective_plan:
                if self.execution_interrupted:
                    logger.warning("Execution interrupted by monitor")
                    break

                start_time = time.time()
                task.status = TaskStatus.EXECUTING

                try:
                    self._validate_task_before_execution(task)
                    self._execute_action(task)
                    task.status = TaskStatus.SUCCESS
                    execution_metrics["success_count"] += 1
                    execution_metrics["total_cost"] += _coerce_float(getattr(task, "cost", 0.0), 0.0)
                    self._update_resource_metrics(execution_metrics, task)
                    self._record_execution_history(task, start_time, time.time(), "success")
                except PlanningError as exc:
                    task.status = TaskStatus.FAILED
                    execution_metrics["failure_count"] += 1
                    self._record_execution_history(task, start_time, time.time(), "failed", error=exc)
                    self._record_error(exc, stage="task_execution", task=task)
                    logger.warning("Execution failure for %s: %s", task.name, _safe_error_message(exc))

                    recovery_plan = None
                    try:
                        recovery_plan = self.safety_planner.dynamic_replanning_pipeline(task)
                    except Exception as recovery_exc:
                        self._record_error(
                            ReplanningError(
                                f"Recovery planning failed: {recovery_exc}",
                                failed_task=task,
                                failure_reason="recovery_exception",
                            ),
                            stage="task_recovery",
                            task=task,
                        )
                    if recovery_plan:
                        recovery = recovery_plan if isinstance(recovery_plan, list) else [recovery_plan]
                        recovery_result = self.execute_plan(recovery, goal_task)
                        if recovery_result.get("status") == TaskStatus.SUCCESS.name:
                            execution_metrics["success_count"] += 1
                        else:
                            execution_metrics["failure_count"] += 1
                except Exception as exc:
                    wrapped = PlanningError(
                        f"Unexpected task execution failure: {exc}",
                        context={"task_name": task.name, "task_id": getattr(task, "id", task.name)},
                    )
                    task.status = TaskStatus.FAILED
                    execution_metrics["failure_count"] += 1
                    self._record_execution_history(task, start_time, time.time(), "failed", error=wrapped)
                    self._record_error(wrapped, stage="task_execution", task=task)
                    logger.error("%s", _safe_error_message(wrapped))

            final_status = TaskStatus.SUCCESS if execution_metrics["failure_count"] == 0 else TaskStatus.FAILED

            try:
                self.metrics.track_plan_completion(plan_meta or {}, final_status)
            except Exception as exc:
                self._record_error(PlanningError(f"track_plan_completion failed: {exc}"), stage="metrics_completion")

            try:
                if hasattr(self.metrics, "record_execution_metrics"):
                    self.metrics.record_execution_metrics(
                        success_count=execution_metrics["success_count"],
                        failure_count=execution_metrics["failure_count"],
                        resource_usage=execution_metrics["resource_usage"],
                    ) # type: ignore
            except Exception as exc:
                self._record_error(PlanningError(f"Execution metric recording failed: {exc}"), stage="metrics_execution")

            summary = {
                "status": final_status.name,
                "world_state": self.world_state,
                "metrics": execution_metrics,
                "last_error": self.last_error,
            }
            self._log_performance(summary)
            return summary
        finally:
            try:
                self.executor.stop_monitoring()
            except Exception as exc:
                self._record_error(PlanningError(f"Execution monitor stop failed: {exc}"), stage="executor_stop")

    def _validate_task_before_execution(self, task: Task) -> None:
        check_preconditions(
            self.world_state,
            list(getattr(task, "preconditions", []) or []),
            task_name=task.name,
            task_id=getattr(task, "id", task.name),
        )
        self._check_temporal_constraints_for_task(task)
    
        profile = getattr(task, "resource_requirements", ResourceProfile())
        requirements = {
            "gpu": _coerce_float(getattr(profile, "gpu", 0.0), 0.0, minimum=0.0),
            "ram": _coerce_float(getattr(profile, "ram", 0.0), 0.0, minimum=0.0),
        }
        # Get actual available resources from the monitor
        available_resources = self.resource_monitor.get_available_resources()
        available = {
            "gpu": float(available_resources.gpu_total),
            "ram": float(available_resources.ram_total),
        }
        check_resource_feasibility(
            requirements,
            available,
            safety_buffers=dict(self.agent_config.get("resource_safety_buffers", {}) or {}),
            task_id=getattr(task, "id", task.name),
        )

    def _check_temporal_constraints_for_task(self, task: Task) -> bool:
        deadline = _coerce_float(getattr(task, "deadline", 0.0), 0.0)
        if deadline > 0 and is_past_deadline(deadline):
            raise DeadlineExceededError(
                f"Task {task.name!r} deadline has passed",
                task_name=task.name,
                task_id=getattr(task, "id", task.name),
                deadline=deadline,
                projected_completion=time.time() + _coerce_float(getattr(task, "duration", 0.0), 0.0),
            )
        return True

    def _execute_action(self, task: Task) -> None:
        if getattr(task, "is_probabilistic", False) and isinstance(getattr(task, "context", None), dict):
            policy = task.context.get("probabilistic_policy")
            if policy:
                self._execute_policy(policy, task)
                return

        profile = getattr(task, "resource_requirements", ResourceProfile())
        acquired = False
        if hasattr(self.resource_monitor, "acquire_resources"):
            try:
                try:
                    self.resource_monitor.acquire_resources(profile, task_id=getattr(task, "id", task.name))
                except TypeError:
                    self.resource_monitor.acquire_resources(profile)
                acquired = True
            except ResourceViolation:
                raise
            except Exception as exc:
                raise ResourceAcquisitionError(
                    f"Failed to acquire resources for {task.name}: {exc}",
                    resource_type="resource_profile",
                    requested=profile,
                    available={},
                    task_id=getattr(task, "id", task.name),
                ) from exc

        try:
            action_payload = (getattr(task, "context", {}) or {}).get("action") if isinstance(getattr(task, "context", {}), dict) else None
            if isinstance(action_payload, dict):
                self._sm_set("planning:last_action_command", {"task": task.name, **action_payload})
            self.world_state = apply_state_effects(self.world_state, list(getattr(task, "effects", []) or []))
        finally:
            if acquired and hasattr(self.resource_monitor, "release_resources"):
                try:
                    # Release by task ID only
                    self.resource_monitor.release_resources(task_id=getattr(task, "id", task.name))
                except TypeError:
                    # Fallback for older signature (if any)
                    self.resource_monitor.release_resources(getattr(task, "id", task.name))
                except Exception as exc:
                    self._record_error(ResourceAcquisitionError(
                        f"Failed to release resources for {task.name}: {exc}",
                        resource_type="resource_profile",
                        requested=profile,
                        available={},
                        task_id=getattr(task, "id", task.name),
                    ), stage="resource_release", task=task)

    def _execute_policy(self, policy: Dict[WorldState, TypingAny], goal_task: Task) -> Dict[str, TypingAny]:
        current_state = self.get_current_state_tuple()
        execution_path: List[Dict[str, TypingAny]] = []
        success = False

        for _ in range(_coerce_int(self.agent_config.get("max_policy_steps"), 100, minimum=1)):
            action = policy.get(current_state)
            if action is None:
                break

            state_dict = dict(current_state)
            if hasattr(action, "preconditions") and not action.preconditions(state_dict):
                raise PreconditionViolation(
                    f"Policy action {getattr(action, 'name', 'unknown')} preconditions failed",
                    task_name=getattr(action, "name", ""),
                    task_id=getattr(goal_task, "id", goal_task.name),
                    world_state_snapshot=state_dict,
                )

            outcome_roll = random.random()
            cumulative = 0.0
            selected = False
            for probability, effect in getattr(action, "outcomes", []):
                validate_probability(float(probability), f"{getattr(action, 'name', 'action')}.outcome_probability")
                cumulative += _coerce_float(probability, 0.0, minimum=0.0, maximum=1.0)
                if outcome_roll <= cumulative:
                    next_state = dict(current_state)
                    effect(next_state)
                    next_tuple = tuple(sorted(next_state.items()))
                    execution_path.append(
                        {
                            "action": getattr(action, "name", "unknown"),
                            "state": current_state,
                            "next_state": next_tuple,
                            "outcome_prob": probability,
                        }
                    )
                    self.world_state = next_state
                    current_state = next_tuple
                    selected = True
                    break

            if not selected:
                raise MethodSelectionError(
                    "No probabilistic outcome selected; outcome probabilities may not sum to 1",
                    task_name=goal_task.name,
                    task_id=getattr(goal_task, "id", goal_task.name),
                )

            goal_state = getattr(goal_task, "goal_state", {}) or {}
            if state_satisfies_goal(self.world_state, goal_state):
                success = True
                break

        return {
            "status": TaskStatus.SUCCESS.name if success else TaskStatus.FAILED.name,
            "execution_path": execution_path,
            "final_state": self.world_state,
            "goal_distance": compute_state_distance(self.world_state, getattr(goal_task, "goal_state", {}) or {}),
        }

    def _record_execution_history(
        self,
        task: Task,
        start_time: float,
        end_time: float,
        status: str,
        error: Optional[BaseException] = None,
    ) -> None:
        entry = {
            "task": task.name,
            "task_id": getattr(task, "id", task.name),
            "name": task.name,
            "start_time": start_time,
            "end_time": end_time,
            "status": status,
            "state_snapshot": copy.deepcopy(self.world_state),
        }
        if error:
            entry["error"] = _error_payload(error)

        self.execution_history.append(entry)

        base_state = getattr(self.shared_memory, "base_state", None)
        if isinstance(base_state, dict):
            base_state.setdefault("execution_history", []).append(entry)

    # ------------------------------------------------------------------
    # Execution-monitor callbacks
    # ------------------------------------------------------------------
    def replan_from_execution_failure(self, task: Optional[Task], reason: str) -> None:
        logger.warning("Replanning triggered due to: %s", reason)
        now = time.time()
        debounce_seconds = _coerce_float(self.agent_config.get("replan_debounce_seconds"), 1.0)
        if now - self._last_replan_at < debounce_seconds:
            logger.warning("Skipping replan due to debounce window")
            return
        self._last_replan_at = now
        self.execution_interrupted = True

        if reason == "precondition_violation" and task is not None:
            recovery_plan = self._create_recovery_plan(task)
        else:
            recovery_plan = self.replan(task or self.current_goal)

        if recovery_plan:
            logger.info("Recovery plan generated with %d tasks", len(recovery_plan))
            self.current_plan = recovery_plan
        else:
            logger.error("Recovery planning failed")

    def adjust_for_resource_violation(self, resource: str, usage: float, limit: float) -> None:
        logger.warning("Resource violation reported by executor: %s %.3f > %.3f", resource, usage, limit)
        self.execution_interrupted = True

    def adjust_for_temporal_violation(self, task: Task, time_delta: float) -> None:
        if self._accelerate_subsequent_tasks(task, time_delta):
            return
        if self._reallocate_time(task, time_delta):
            return
        self.replan_from_execution_failure(task, "temporal_violation")

    def _create_recovery_plan(self, failed_task: Task) -> Optional[List[Task]]:
        for method_idx in self._find_alternative_methods(failed_task):
            candidate = failed_task.copy()
            candidate.selected_method = method_idx
            recovery_plan = self.decompose_task(candidate, self.world_state)
            if recovery_plan and self._validate_plan(recovery_plan):
                return recovery_plan

        repair_plan = self._create_precondition_repair_plan(failed_task)
        if repair_plan is not None:
            return repair_plan + [failed_task.copy()]
        return None

    def _create_precondition_repair_plan(self, task: Task) -> Optional[List[Task]]:
        try:
            check_preconditions(
                self.world_state,
                list(getattr(task, "preconditions", []) or []),
                task_name=task.name,
                task_id=getattr(task, "id", task.name),
            )
            return []
        except PreconditionViolation as exc:
            self._record_error(exc, stage="precondition_repair", task=task)
            logger.warning("No automated repair available for failed preconditions: %s", exc.failed_conditions)
            return None

    def _accelerate_subsequent_tasks(self, task: Task, time_delta: float) -> bool:
        """Best-effort temporal mitigation by shrinking later task durations."""
        if time_delta <= 0 or not self.current_plan:
            return False
        seen = False
        changed = False
        for candidate in self.current_plan:
            if candidate == task:
                seen = True
                continue
            if not seen:
                continue
            duration = _coerce_float(getattr(candidate, "duration", 0.0), 0.0)
            if duration > 2.0:
                candidate.duration = max(1.0, duration - min(duration * 0.10, time_delta))
                changed = True
        return changed

    def _reallocate_time(self, task: Task, time_delta: float) -> bool:
        """Best-effort temporal mitigation by pushing low-priority future tasks."""
        if time_delta <= 0 or not self.current_plan:
            return False
        changed = False
        for candidate in self.current_plan:
            if candidate is task:
                continue
            if _coerce_float(getattr(candidate, "priority", 1.0), 1.0) < _coerce_float(getattr(task, "priority", 1.0), 1.0):
                candidate.start_time = _coerce_float(getattr(candidate, "start_time", time.time()), time.time()) + time_delta
                candidate.end_time = _coerce_float(getattr(candidate, "end_time", candidate.start_time), candidate.start_time) + time_delta
                changed = True
        return changed

    # ------------------------------------------------------------------
    # Local planning arbitration
    # ------------------------------------------------------------------
    def _read_local_planning_context(self) -> LocalPlanningContext:
        payload = self._sm_get("planning:local_context", default={}) or {}
        if not isinstance(payload, dict):
            payload = {}
        return LocalPlanningContext(
            obstacle_distance_m=payload.get("obstacle_distance_m"),
            obstacle_bearing_deg=payload.get("obstacle_bearing_deg"),
            clearance_left_m=payload.get("clearance_left_m"),
            clearance_right_m=payload.get("clearance_right_m"),
            reverse_clearance_m=payload.get("reverse_clearance_m"),
            current_speed_mps=_coerce_float(payload.get("current_speed_mps"), 0.0),
            desired_speed_mps=_coerce_float(payload.get("desired_speed_mps"), 0.0),
            horizon_seconds=_coerce_float(payload.get("horizon_seconds"), 2.0),
            metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
        )

    def _build_reactive_override_task(self, decision, context: LocalPlanningContext) -> Task:
        command = dict(getattr(decision, "command", {}) or {})
        command["reason"] = getattr(decision, "reason", "local override")
        return Task(
            name=self.local_arbitrator.build_reactive_task_name(decision),
            task_type=TaskType.PRIMITIVE,
            priority=100,
            risk_score=0.0,
            context={"action": command, "local_context": context.metadata},
            preconditions=[lambda _state: True],
            effects=[],
            duration=max(1.0, min(5.0, _coerce_float(getattr(context, "horizon_seconds", 2.0), 2.0))),
        )

    def _apply_local_arbitration(self, plan: List[Task]) -> List[Task]:
        context = self._read_local_planning_context()
        decision = self.local_arbitrator.decide(context)
        self._sm_set(
            "planning:last_local_decision",
            {
                "behavior": getattr(getattr(decision, "behavior", None), "value", str(getattr(decision, "behavior", "unknown"))),
                "reason": getattr(decision, "reason", ""),
                "priority": getattr(decision, "priority", 0),
            },
        )

        if not getattr(decision, "is_override", False):
            return plan

        reactive_task = self._build_reactive_override_task(decision, context)
        if self.local_arbitrator.should_trigger_short_horizon_replan(decision, context):
            logger.warning("Applying local safety override before nominal plan")
            return [reactive_task] + plan
        return plan

    # ------------------------------------------------------------------
    # Utility methods retained for compatibility
    # ------------------------------------------------------------------
    def _check_preconditions(self, task: Task, state: Dict[str, TypingAny]) -> bool:
        try:
            check_preconditions(
                state,
                list(getattr(task, "preconditions", []) or []),
                task_name=task.name,
                task_id=getattr(task, "id", task.name),
            )
            return True
        except PreconditionViolation as exc:
            self._record_error(exc, stage="precondition_check", task=task)
            logger.error("Precondition check failed for %s: %s", task.name, _safe_error_message(exc))
            return False

    def _check_temporal_constraints(self, previous_task: Task, current_task: Task, state: Dict[str, TypingAny]) -> bool:
        for constraint in getattr(current_task, "temporal_constraints", []) or []:
            if callable(constraint) and not constraint(state):
                self._record_error(
                    TemporalViolation(
                        f"Custom temporal constraint failed for {current_task.name}",
                        violation_type="custom",
                        task_name=current_task.name,
                        task_id=getattr(current_task, "id", current_task.name),
                    ),
                    stage="temporal_check",
                    task=current_task,
                )
                return False

        previous_end = _coerce_float(getattr(previous_task, "end_time", 0.0), 0.0)
        current_start = _coerce_float(getattr(current_task, "start_time", 0.0), 0.0)
        if previous_end and current_start and previous_end > current_start:
            self._record_error(
                TemporalViolation(
                    f"{current_task.name} starts before predecessor {previous_task.name} ends",
                    violation_type="ordering",
                    task_name=current_task.name,
                    task_id=getattr(current_task, "id", current_task.name),
                    constraint_details={
                        "previous_task": previous_task.name,
                        "previous_end": previous_end,
                        "current_start": current_start,
                    },
                    time_delta=previous_end - current_start,
                ),
                stage="temporal_check",
                task=current_task,
            )
            return False
        return True

    def _grid_search_alternatives(self, task: Task) -> List[Task]:
        library_task = self.task_library.get(task.name, task)
        if task.task_type != TaskType.ABSTRACT or not library_task.methods:
            return []
        current = int(getattr(task, "selected_method", 0))
        return self._create_alternatives(task, [(idx, 0.0) for idx in range(current + 1, len(library_task.methods))])

    def _bayesian_alternatives(self, task: Task) -> List[Task]:
        library_task = self.task_library.get(task.name, task)
        if task.task_type != TaskType.ABSTRACT or not library_task.methods:
            return []

        method_scores: List[Tuple[int, float]] = []
        for method_idx in range(len(library_task.methods)):
            stats = self.method_stats[(task.name, str(method_idx))]
            success = _coerce_float(stats.get("success"), 0.0) + 1.0
            total = _coerce_float(stats.get("total"), 0.0) + 2.0
            method_scores.append((method_idx, success / total))

        method_scores.sort(key=lambda item: item[1], reverse=True)
        return self._create_alternatives(task, [item for item in method_scores if item[0] != getattr(task, "selected_method", 0)][:2])

    def _create_alternatives(self, task: Task, method_scores: Iterable[Tuple[int, float]]) -> List[Task]:
        alternatives: List[Task] = []
        library_task = self.task_library.get(task.name, task)
        for method_idx, _score in method_scores:
            if 0 <= method_idx < len(getattr(library_task, "methods", []) or []):
                candidate = library_task.copy()
                candidate.selected_method = method_idx
                alternatives.append(candidate)
        return alternatives

    def _update_scheduler_state(self, task: Task) -> None:
        agent = getattr(task, "assigned_agent", None) or (getattr(task, "context", {}) or {}).get("assigned_agent")
        if agent:
            self.schedule_state["agent_loads"][agent] -= _coerce_float(getattr(task, "cost", 0.0), 0.0)
        self.schedule_state["task_history"][task.name].append({"status": "failed", "timestamp": time.time()})

    def _update_task_success(self, parent: Task, children: List[Task]) -> None:
        if parent.task_type != TaskType.ABSTRACT:
            return
        success = all(child.status == TaskStatus.SUCCESS for child in children)
        self._update_method_stats(parent, success)

    def _update_method_stats(self, task: Task, success: bool, cost: float = 0.0) -> None:
        method_idx = str(getattr(task, "selected_method", 0))
        key = (task.name, method_idx)
        stats = self.method_stats[key]
        previous_total = _coerce_float(stats.get("total"), 0.0, minimum=0.0)
        previous_avg = _coerce_float(stats.get("avg_cost"), 0.0, minimum=0.0)

        stats["total"] = previous_total + 1.0
        if success:
            stats["success"] = _coerce_float(stats.get("success"), 0.0, minimum=0.0) + 1.0
        stats["success"] = min(stats["success"], stats["total"])
        stats["avg_cost"] = ((previous_avg * previous_total) + _coerce_float(cost, 0.0, minimum=0.0)) / max(stats["total"], 1.0)

    def _update_resource_metrics(self, metrics: Dict[str, TypingAny], task: Task) -> None:
        totals = aggregate_resource_requirements([task])
        for resource, value in totals.items():
            metrics["resource_usage"][resource] += value
        profile = getattr(task, "resource_requirements", None)
        for hardware in getattr(profile, "specialized_hardware", []) or []:
            metrics["resource_usage"][hardware] += 1

    def _log_performance(self, result: Dict[str, TypingAny]) -> None:
        log_key = f"log:performance:{getattr(self, 'name', 'PlanningAgent')}"
        log_entry = {
            "timestamp": time.time(),
            "status": result.get("status"),
            "metrics": {
                "total_cost": result.get("metrics", {}).get("total_cost", 0),
                "plan_length": len(self.current_plan),
                "success_count": result.get("metrics", {}).get("success_count", 0),
                "failure_count": result.get("metrics", {}).get("failure_count", 0),
            },
        }

        performance_logs = self._sm_get(log_key, default=deque(maxlen=500))
        if not isinstance(performance_logs, deque):
            performance_logs = deque(list(performance_logs or []), maxlen=500)
        performance_logs.append(log_entry)
        self._sm_set(log_key, performance_logs)

    def needs_new_plan(self, synapse: Task) -> bool:
        load = _coerce_float(self.adaptive_context.get("system_load"), 0.0, minimum=0.0)
        feedback = self.adaptive_context.get("performance_feedback", {})
        score = feedback.get("score", 1.0) if isinstance(feedback, dict) else _coerce_float(feedback, 1.0)
        score = clamp(_coerce_float(score, 1.0), 0.0, 1.0)
        if load > _coerce_float(self.agent_config.get("high_load_replan_threshold"), 85.0):
            return True
        if score < _coerce_float(self.agent_config.get("low_score_replan_threshold"), 0.6, minimum=0.0, maximum=1.0):
            return True
        return not hasattr(synapse, "history") or not getattr(synapse, "history")

    def set_adaptive_context(self, context: dict) -> None:
        self.adaptive_context = context or {}

    def sync_with_shared_memory(self, shared_memory) -> None:
        self.shared_memory = shared_memory
        resource_usage = self._sm_get("resource_usage", default={}) or {}
        # Optionally update monitor metrics via its public API
        if hasattr(self.resource_monitor, "update_metrics"):
            self.resource_monitor.update_metrics({"gpu": resource_usage.get("gpu", 0.0),
                                                  "ram": resource_usage.get("ram", 0.0)})
        logger.info("PlanningAgent synchronized with shared memory")

    def update_shared_memory(self, shared_memory) -> None:
        if hasattr(shared_memory, "base_state") and isinstance(shared_memory.base_state, dict):
            shared_memory.base_state.update(self.world_state)
        setter = getattr(shared_memory, "set", None)
        if callable(setter):
            setter(
                "resource_usage",
                {
                    "gpu": getattr(self.resource_monitor, "gpu_usage", 0.0),
                    "ram": getattr(self.resource_monitor, "ram_usage", 0.0),
                },
            )

    def predict(self, state: Any = None) -> Dict[str, TypingAny]:
        return {
            "agent": "PlanningAgent",
            "status": "active",
            "current_goal": self.current_goal.name if self.current_goal else None,
            "plan_length": len(self.current_plan),
            "config_source": "agents_config.yaml",
        }


def run_planning_cycle(agent: PlanningAgent, goal_task: Task) -> Optional[Dict[str, TypingAny]]:
    """Run generation, safety validation, execution, and metrics for one goal."""
    try:
        plan = agent.generate_plan(goal_task)
        if not plan:
            raise GoalUnreachableError(
                "Plan generation failed",
                goal_state=getattr(goal_task, "goal_state", {}) or {},
                reason="generate_plan_returned_none",
            )

        if not agent._validate_plan(plan):
            raise AcademicPlanningError(
                "Final plan failed safety validation",
                context={"goal": getattr(goal_task, "name", "")},
            )

        result = agent.execute_plan(plan, goal_task)

        try:
            planning_start = agent._planning_start_time or 0.0
            planning_end = agent._planning_end_time or time.time()  # or 0.0
            metrics = agent.metrics.calculate_all_metrics(
                plan=plan,
                planning_start_time=planning_start,
                planning_end_time=planning_end,
                final_status=TaskStatus[result.get("status", TaskStatus.FAILED.name)],
            )
        except Exception:
            metrics = {
                "plan_length": len(plan),
                "status": result.get("status"),
                "planning_time": (
                    max(0.0, (agent._planning_end_time or time.time()) - (agent._planning_start_time or time.time()))
                ),
            }

        logger.info("Planning cycle completed: %s", metrics)
        return {"execution_result": result, "metrics": metrics, "last_error": agent.last_error}
    except PlanningError as exc:
        agent._record_error(exc, stage="planning_cycle", task=goal_task)
        logger.error("Planning cycle failed: %s", _safe_error_message(exc))
        return None
    except Exception as exc:
        wrapped = PlanningError(
            f"Unexpected planning cycle failure: {exc}",
            context={"goal": getattr(goal_task, "name", "")},
        )
        agent._record_error(wrapped, stage="planning_cycle", task=goal_task)
        logger.error("%s", _safe_error_message(wrapped))
        return None


class HTNPlanner(PlanningAgent):
    """Compatibility subclass for HTN-style planning workflows."""

    StateTuple = Tuple[Tuple[str, Any], ...]

    def _ordered_decomposition(self, task: Task) -> Optional[List[Task]]:
        return self.decompose_task(task, self.world_state)

    def _freeze_state(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(sorted(self.world_state.items()))

    def _apply_effects(self, state: StateTuple, task: Task) -> StateTuple:
        state_dict = dict(state)
        state_dict = apply_state_effects(state_dict, list(getattr(task, "effects", []) or []))
        return tuple(sorted(state_dict.items()))

    def _thompson_sampling_alternatives(self, task: Task) -> List[Task]:
        method_probs: List[Tuple[int, float]] = []
        for method_idx in range(len(getattr(task, "methods", []) or [])):
            stats = self.method_stats[(task.name, str(method_idx))]
            alpha = _coerce_float(stats.get("success"), 0.0) + 1.0
            beta = _coerce_float(stats.get("total"), 0.0) - _coerce_float(stats.get("success"), 0.0) + 1.0
            method_probs.append((method_idx, random.betavariate(max(alpha, 1e-9), max(beta, 1e-9))))
        method_probs.sort(key=lambda item: item[1], reverse=True)
        return self._create_alternatives(task, method_probs)

    def _validate_plan(self, plan: Sequence[Task]) -> bool:
        sim_state = copy.deepcopy(self.world_state)
        for task in plan:
            try:
                check_preconditions(
                    sim_state,
                    list(getattr(task, "preconditions", []) or []),
                    task_name=task.name,
                    task_id=getattr(task, "id", task.name),
                )
                sim_state = apply_state_effects(sim_state, list(getattr(task, "effects", []) or []))
            except PlanningError as exc:
                self._record_error(exc, stage="htn_plan_validation", task=task)
                return False
        return super()._validate_plan(plan)


class PartialOrderPlanner(PlanningAgent):
    """Compatibility subclass for temporal/causal-link planning extensions."""

    def __init__(self, shared_memory, agent_factory, config: Optional[Dict[str, TypingAny]] = None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config, **kwargs)
        self.temporal_constraints: Set[Tuple[Task, Task, str]] = set()
        self.causal_links: Set[Tuple[Task, Task, Callable]] = set()

    def _add_temporal_constraint(self, constraint: Tuple[Task, Task, str]) -> None:
        valid_relations = {"before", "after", "contains", "during", "meets"}
        if constraint[2] not in valid_relations:
            raise ValueError(f"Invalid temporal relation: {constraint[2]}")
        self.temporal_constraints.add(constraint)

    def _resolve_threats(self) -> None:
        for producer, consumer, condition in list(self.causal_links):
            for task in self.current_plan:
                if task is producer or task is consumer:
                    continue
                for effect in getattr(task, "effects", []) or []:
                    try:
                        threatens = not condition(effect)
                    except Exception:
                        threatens = False
                    if threatens:
                        self._add_temporal_constraint((task, producer, "before"))


class AStarPlanner(PlanningAgent):
    """Compatibility subclass with simple cost-based plan ordering support."""

    def _optimize_plan(self, plan: List[Task]) -> List[Task]:
        primitive_tasks = [task for task in plan if task.task_type == TaskType.PRIMITIVE]
        abstract_tasks = [task for task in plan if task.task_type == TaskType.ABSTRACT]
        primitive_tasks.sort(key=lambda task: (self._task_cost(task), getattr(task, "deadline", float("inf"))))
        return abstract_tasks + primitive_tasks

    def _task_cost(self, task: Task) -> float:
        profile = getattr(task, "resource_requirements", ResourceProfile())
        resource_cost = _coerce_float(getattr(profile, "gpu", 0.0), 0.0) * 10.0
        resource_cost += _coerce_float(getattr(profile, "ram", 0.0), 0.0)
        return _coerce_float(getattr(task, "cost", 1.0), 1.0) + resource_cost + _coerce_float(getattr(task, "risk_score", 0.0), 0.0)

    def _extract_optimal_plan(self, and_or_graph: Dict[Task, Dict[str, TypingAny]]) -> List[Task]:
        plan: List[Task] = []
        for task, payload in and_or_graph.items():
            if task.task_type == TaskType.PRIMITIVE:
                plan.append(task)
                continue
            methods = payload.get("methods", [])
            if methods:
                best_method, _cost = min(methods, key=lambda item: item[1])
                plan.extend([subtask for subtask in best_method if subtask.task_type == TaskType.PRIMITIVE])
        return self._optimize_plan(plan)


if __name__ == "__main__":
    print("\n=== PlanningAgent smoke test ===")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory

    cfg={"gpu_limit":1,"ram_limit":8,"fallback_agent_capabilities":["gpu","ram"],
         "resource_safety_buffers":{"gpu":0,"ram":0},"default_deadline_seconds":120}
    a=PlanningAgent(shared_memory=SharedMemory, agent_factory=AgentFactory, config=cfg)
    a.safety_planner.safety_check=lambda p: True # type: ignore
    a.executor.start_monitoring=lambda p,s: None # type: ignore
    a.executor.stop_monitoring=lambda: None # type: ignore
    a.resource_monitor.acquire_resources=lambda *x,**k: None # type: ignore
    a.resource_monitor.release_resources=lambda *x,**k: None # type: ignore

    def sched(tasks, agents, risk_assessor=None, state=None):
        ids=[t["id"] for t in tasks]; deps={t["id"]:t.get("dependencies",[]) for t in tasks}
        now=time.time(); out={}
        for i,tid in enumerate(topological_sort(ids,deps)):
            s=next(t for t in tasks if t["id"]==tid)
            out[tid]={"task_id":tid,"task_name":s["name"],"agent_id":"planner",
                      "start_time":now+i,"end_time":now+i+s["duration"],
                      "deadline":s["deadline"],"duration":s["duration"],
                      "risk_score":s.get("risk_score",0)}
        return out
    a.scheduler.schedule=sched

    def ok(n,c): print(f"[{'PASS' if c else 'FAIL'}] {n}"); assert c,n
    def fx(k,v): return lambda s: s.update({k:v})
    R=ResourceProfile

    t1=Task("collect",TaskType.PRIMITIVE,preconditions=[lambda s: True],
            effects=[fx("data",True)],resource_requirements=R(gpu=.1,ram=1),
            duration=1,deadline=time.time()+60); t1.id="collect"
    t2=Task("plan",TaskType.PRIMITIVE,preconditions=[lambda s:s.get("data") is True],
            effects=[fx("planned",True)],resource_requirements=R(gpu=.1,ram=1),
            duration=1,deadline=time.time()+60,dependencies=["collect"]); t2.id="plan"
    goal=Task("demo_goal",TaskType.ABSTRACT,methods=[[t1,t2]],
              resource_requirements=R(gpu=0,ram=0),duration=2,
              deadline=time.time()+90,goal_state={"planned":True}); goal.id="demo_goal"

    a.register_task(goal)
    plan=a.generate_plan(goal)
    ok("plan generation + scheduling", bool(plan) and len(plan)==2)
    ok("Task conversion", all(isinstance(t,Task) for t in plan)) # type: ignore
    ok("DAG helper", detect_cycles([t.id for t in plan], # type: ignore
       {t.id:getattr(t,"dependencies",[]) for t in plan}) is None) # type: ignore
    res=a.execute_plan(plan, goal) # type: ignore
    ok("execution", res["status"]==TaskStatus.SUCCESS.name)
    ok("goal reached", state_satisfies_goal(a.world_state,{"planned":True}))
    ok("history + diagnostics", len(a.execution_history)>=2 and isinstance(a.predict(),dict))

    bad=Task("bad_gpu",TaskType.PRIMITIVE,preconditions=[lambda s: True],
             effects=[],resource_requirements=R(gpu=2,ram=1),
             duration=1,deadline=time.time()+30); bad.id="bad_gpu"
    try: a.register_task(bad); raise AssertionError("bad resource task accepted")
    except ResourceViolation: print("[PASS] structured ResourceViolation")

    blocked=Task("blocked",TaskType.PRIMITIVE,preconditions=[lambda s: False],
                 effects=[],resource_requirements=R(gpu=0,ram=0),
                 duration=1,deadline=time.time()+30); blocked.id="blocked"
    ok("PreconditionViolation path",
       a.decompose_task(blocked,a.world_state) is None and a.last_error is not None)
    print("\n=== smoke test passed ===")
