from __future__ import annotations

__version__ = "2.2.0"

"""
SLAI Execution Agent:

This agent is the "doer" of the AI system. It takes high-level tasks from the TaskCoordinator,
and through a continuous loop of context-gathering, action-selection, and execution, it works
to complete those tasks. It maintains its own state (e.g., energy, position) and uses its
specialized components to make intelligent, moment-to-moment decisions.

Academic Foundations:
- Behavior Trees: The agent's execution loop, where it repeatedly selects and executes an action
  to progress towards a goal, is analogous to a simple behavior tree. The ActionSelector acts as
  a sophisticated selector node.
- Finite State Machines (FSM): The agent's internal state (e.g., `hand_empty`, `at_destination`)
  and the action postconditions that modify it are a form of FSM, where actions trigger state
  transitions.
- Utility-Based AI: The `utility` and `hybrid` strategies in the ActionSelector are direct
  implementations of utility theory, where the agent chooses the action that maximizes its
  expected utility based on the current context. (Reference: "Artificial Intelligence: A Modern
  Approach" by Russell and Norvig).

Real-World Application:
- Robotics/Automation:
    Navigation (move_to), object handling (pick_object), and rest/recharge behaviors (idle) directly map to mobile robots,
    warehouse automation, or domestic service bots.
- Gaming:
    NPCs that can navigate, interact with environments, and plan under uncertainty.
    Excellent for sandbox, open-world, or simulation-heavy games.
- Task Managment:
    The task coordination and execution loop can be applied in manufacturing or logistics,
    where tasks have dependencies, retries, and require stateful agents.
"""

import copy
import json
import time
import uuid
import numpy as np

from datetime import datetime, timedelta
from typing import Any, Dict, Mapping, Optional, Sequence, Type

from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section, load_global_config
from .execution.task_coordinator import TaskCoordinator, TaskState
from .execution.execution_validator import ExecutionValidator
from .execution.execution_recovery import ExecutionRecovery
from .execution.execution_memory import ExecutionMemory
from .execution.action_selector import ActionSelector
from .execution.actions.base_action import BaseAction
from .execution.actions.idle import IdleAction
from .execution.actions.move_to import MoveToAction
from .execution.actions.pick_object import PickObjectAction
from .execution.actions.place_object import PlaceObjectAction
from .execution.actions.robot_actions import *
from .execution.utils.execution_error import *
from .execution.modules.robot_interface import RobotInterface
from .planning.task_scheduler import DeadlineAwareScheduler
from .runtime_contracts import RuntimeLifecycle
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Execution Agent")
printer = PrettyPrinter()

class ExecutionAgent(BaseAgent):
    """
    Orchestrates the execution of tasks by managing agent state, selecting
    appropriate actions, and overseeing their lifecycle. It acts as the bridge

    between high-level planning and low-level action execution.
    """
    DEFAULT_THRESHOLDS = {"timeout": 300, "energy_alert": 2.0}
    DEFAULT_ACTIONS: Dict[str, Type[BaseAction]] = {
        "move_to": MoveToAction,
        "pick_object": PickObjectAction,
        "place_object": PlaceObjectAction,
        "idle": IdleAction,
    }
    ROBOT_ACTIONS: Dict[str, Type[BaseAction]] = {
        "motor": MotorAction,
        "ackermann": AckermannAction,
        "stop": StopAction,
        "spin": SpinAction,
        "navigate": NavigateAction,
        "follow_path": FollowPathAction,
        "gripper": GripperAction,
        "joint": JointAction,
        "sensor_read": SensorReadAction,
        "wait": WaitAction,
        "led": LedAction,
        "sequence": SequenceAction,
    }
    CHECKPOINTING_SUPPORTED = True


    def __init__(self, shared_memory, agent_factory, config: Optional[Mapping[str, Any]]=None,
                 execution_memory: Optional[ExecutionMemory] = None, task_coordinator: Optional[TaskCoordinator] = None,
                 action_selector: Optional[ActionSelector] = None, validator: Optional[ExecutionValidator] = None,
                 recovery: Optional[ExecutionRecovery] = None, scheduler: Optional[DeadlineAwareScheduler] = None,
                 robot: Optional[RobotInterface] = None, action_registry: Optional[Mapping[str, Type[BaseAction]]] = None,
                 *,
                 checkpoint_manager: Any = None,
                 ):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config, checkpoint_manager=checkpoint_manager)
        self.adaptive_agent = None
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.execution_agent_config = get_config_section("execution_agent") or {}

        # self.thresholds = self.shared_memory.get("execution_thresholds", self.DEFAULT_THRESHOLDS)
        # self.max_task_seconds = int(self.execution_agent_config.get("max_task_seconds", self.thresholds["timeout"]))
        self.thresholds = self.shared_memory.get("execution_thresholds", default=self.DEFAULT_THRESHOLDS)
        self.max_task_seconds = int(self.execution_agent_config.get("max_task_seconds", self.thresholds["timeout"]))
        self.task_lock_ttl_buffer_seconds = int(self.execution_agent_config.get("task_lock_ttl_buffer_seconds", 120))
        self.max_step_retries = int(self.execution_agent_config.get("max_step_retries", 2))
        self.default_grid_size = int(self.execution_agent_config.get("default_grid_size", 100))
        self.efficiency = float(self.execution_agent_config.get("efficiency", 1.0))
        if not np.isfinite(self.efficiency) or self.efficiency <= 0.0:
            raise ValueError("execution_agent.efficiency must be finite and greater than zero")

        # Core subsystem composition
        self.execution_memory = execution_memory if execution_memory is not None else ExecutionMemory()
        self._owns_execution_memory = execution_memory is None
        self.task_coordinator = (task_coordinator if task_coordinator is not None else TaskCoordinator(memory=self.execution_memory))
        self.action_selector = (action_selector if action_selector is not None else ActionSelector(memory=self.execution_memory))
        self.validator = validator if validator is not None else ExecutionValidator(memory=self.execution_memory)
        self.recovery = ExecutionRecovery(task_coordinator=self.task_coordinator)
        self._scheduler_initialization_error: Optional[Exception] = None
        self.scheduler = (scheduler
            if scheduler is not None
            else DeadlineAwareScheduler())
        self.robot = robot

        # Local runtime state
        self._current_plan_index = 0
        self._initial_task_distance: Optional[float] = None
        self.action_class_registry = self._build_action_registry(action_registry)
        self.state: Dict[str, Any] = self._initialize_state()
        self.current_task: Optional[Dict[str, Any]] = None
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

        self._register_actions()

        logger.info("ExecutionAgent initialized")

    def _require_scheduler(self) -> Any:
        if self.scheduler is None:
            self.scheduler
        if self.scheduler is None:
            raise ActionFailureError(
                "scheduler",
                "Planning scheduler is unavailable; inspect runtime degradation details",
            )
        self._mark_runtime_recovered("dependency", "execution.scheduler.initialize")
        return self.scheduler

    def _build_action_registry(
        self,
        injected_registry: Optional[Mapping[str, Type[BaseAction]]],
    ) -> Dict[str, Type[BaseAction]]:
        available = {**self.DEFAULT_ACTIONS, **self.ROBOT_ACTIONS}
        if injected_registry is not None:
            for name, action_class in injected_registry.items():
                if not isinstance(action_class, type) or not issubclass(action_class, BaseAction):
                    raise TypeError(f"Action registry entry {name!r} must be a BaseAction subclass")
                available[str(name)] = action_class

        configured_names = self.execution_agent_config.get(
            "enabled_actions",
            list(self.DEFAULT_ACTIONS),
        )
        if isinstance(configured_names, (str, bytes)) or not isinstance(configured_names, Sequence):
            raise TypeError("execution_agent.enabled_actions must be a sequence of action names")
        names = [str(name).strip() for name in configured_names if str(name).strip()]
        if not names:
            raise ValueError("execution_agent.enabled_actions cannot be empty")
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise ValueError(f"Unknown execution actions configured: {unknown}")
        if "idle" not in names:
            raise ValueError("execution_agent.enabled_actions must include the idle fallback action")
        return {name: available[name] for name in dict.fromkeys(names)}

    def _register_actions(self) -> None:
        for name, cls in self.action_class_registry.items():
            self.action_selector.register_action(name, cls.preconditions, cls.postconditions)
            self.validator.register_action_handler(name, cls)

    def register_action(self, name: str, action_class: Type[BaseAction]) -> None:
        """Register an action consistently with selection and validation."""
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("Action name cannot be empty")
        if not isinstance(action_class, type) or not issubclass(action_class, BaseAction):
            raise TypeError("action_class must be a BaseAction subclass")
        self.action_class_registry[normalized_name] = action_class
        self.action_selector.register_action(
            normalized_name,
            action_class.preconditions,
            action_class.postconditions,
        )
        self.validator.register_action_handler(normalized_name, action_class)

    def _initialize_state(self) -> Dict[str, Any]:
        printer.status("EXECUTION", "Initializing states...", "info")

        saved_state = self.shared_memory.get(f"agent_state:{self.name}")
        if isinstance(saved_state, dict) and saved_state:
            printer.status("STATE", "Loaded from shared memory", "success")
            return saved_state

        size = max(10, self.default_grid_size)
        state = {
            "energy": 10.0,
            "max_energy": 10.0,
            "current_position": (0.0, 0.0),
            "destination": None,
            "at_destination": False,
            "robot_ready": False,
            "hand_empty": True,
            "holding_object": False,
            "held_object": None,
            "carrying_items": 0,
            "inventory": {},
            "map_data": [[0] * size for _ in range(size)],
            "object_state": {},
            "required_field": True,
        }
        self.shared_memory.set(f"agent_state:{self.name}", state)
        return state

    def predict(self, state: Any = None) -> Dict[str, Any]:
        context = copy.deepcopy(state) if isinstance(state, dict) else self._gather_context()
        try:
            selected = self._select_action(context)
            return {
                "selected_action": selected.get("name", "idle"),
                "confidence": 1.0,
                "context": context,
                "task_progress": self._calculate_task_progress(context) if self.current_task else 0.0,
            }
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return {
                "selected_action": "idle",
                "confidence": 0.0,
                "context": context,
                "task_progress": 0.0,
                "error": str(exc),
            }

    def _generate_default_plan(self, task_data: Dict[str, Any]) -> list[Dict[str, Any]]:
        return []

    def _normalize_task_input(self, task_data: Any) -> Dict[str, Any]:
        if isinstance(task_data, str):
            try:
                task_data = json.loads(task_data)
            except json.JSONDecodeError as exc:
                raise InvalidContextError("perform_task", ["task_data(dict/json)"]) from exc

        if not isinstance(task_data, dict):
            raise InvalidContextError("perform_task", ["task_data(dict)"])

        normalized = copy.deepcopy(task_data)
        normalized.setdefault("name", f"task_{str(uuid.uuid4())[:8]}")
        normalized.setdefault("id", f"{normalized['name']}_{str(uuid.uuid4())[:8]}")
        normalized.setdefault("goal_type", normalized.get("task_type", "generic"))
        normalized.setdefault("requirements", [])
        normalized.setdefault("deadline", time.time() + self.max_task_seconds)
        normalized.setdefault("timeout", self.max_task_seconds)
        normalized.setdefault("priority", 0)
        normalized.setdefault("dependencies", [])
        normalized.setdefault("metadata", {})
        normalized.setdefault("action_sequence", self._generate_default_plan(normalized))
        return normalized

    def perform_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        printer.status("EXECUTION", "Task performer", "info")

        task = self._normalize_task_input(task_data)
        lock_key = f"task_in_progress:{task['name']}"

        if self.shared_memory.get(lock_key):
            return {"status": "failed", "reason": "Task already in progress"}

        self.shared_memory.set(
            lock_key,
            {"agent": self.name, "start_time": time.time(), "task_id": task["id"]},
            ttl=int(task["timeout"]) + self.task_lock_ttl_buffer_seconds,
        )

        try:
            self._preflight_task(task)
            self._start_task(task)
            self._run_task_loop(task)
            return self._finalize_task_result(task)
        finally:
            self.shared_memory.delete(lock_key)

    def _preflight_task(self, task: Dict[str, Any]) -> None:
        self.current_plan = list(task.get("action_sequence") or [])
        self._current_plan_index = 0
        self._initial_task_distance = None
        if task.get("action_sequence"):
            context = self._gather_context(task)
            is_valid, report = self.validator.validate_plan(task["action_sequence"], context)
            if not is_valid:
                summary = self.validator.generate_validation_summary(report)
                raise ActionFailureError("preflight", f"Plan validation failed: {summary}")
        schedule = self._require_scheduler().schedule(
            tasks=[task],
            agents={self.name: self._get_agent_capabilities()},
            state=self.state,
        )
        if task["id"] not in schedule:
            raise ActionFailureError("scheduler", f"Scheduling failed for task id={task['id']}")

    def _start_task(self, task: Dict[str, Any]) -> None:
        if not self.task_coordinator.assign_task(task):
            raise ActionFailureError("task_coordinator", f"Unable to assign task '{task['name']}'")

        next_task = self.task_coordinator.get_next_task()
        if not next_task or next_task.get("name") != task["name"]:
            raise ActionFailureError("task_coordinator", "Could not fetch assigned task")

        if not self.task_coordinator.start_task(task["name"], assignee=self.name):
            raise ActionFailureError("task_coordinator", "Failed to transition task to IN_PROGRESS")

        self.current_task = self.task_coordinator.get_task_snapshot(task["name"])
        self.shared_memory.publish("task_events", {"event": "task_started", "task": task, "agent": self.name})

    def _run_task_loop(self, task: Dict[str, Any]) -> None:
        start_time = time.time()
        timeout = int(task.get("timeout", self.max_task_seconds))

        self.recovery.create_recovery_checkpoint("pre_task", self.state, extra_tags=[task["name"]])

        while self.current_task and self.current_task.get("state") == TaskState.IN_PROGRESS.value:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.task_coordinator.fail_task(task["name"], "Execution timed out")
                raise TimeoutError(task["name"], timeout, elapsed)

            step_outcome = self._execution_step()
            if step_outcome.get("status") == "interrupted":
                self.task_coordinator.pause_task(task["name"])
                self.task_coordinator.resume_task(task["name"])
            elif step_outcome.get("status") == "failed":
                raise ActionFailureError(step_outcome.get("action", "unknown"), step_outcome.get("reason", "Failed"))

            self.current_task = self.task_coordinator.get_task_snapshot(task["name"])

    def _finalize_task_result(self, task: Dict[str, Any]) -> Dict[str, Any]:
        final_ok = self.task_coordinator.validate_task_completion(task["name"])
        if final_ok:
            self.shared_memory.publish("task_events", {"event": "task_completed", "task": task, "agent": self.name})
            return {"status": "success", "result": "Task completed.", "task": task["name"]}

        history_task = next((t for t in self.task_coordinator.task_history if t.get("name") == task["name"]), None)
        reason = history_task.get("failure_reason", "Unknown reason") if history_task else "Task not found in history"
        return {"status": "failed", "reason": reason, "task": task["name"]}

    def _execution_step(self) -> Dict[str, Any]:
        printer.status("EXECUTION", "Execution loop", "info")

        if not self.current_task:
            return {"status": "failed", "action": "none", "reason": "No active task"}

        context = self._gather_context()
        if self._is_task_complete(context):
            self.task_coordinator.complete_task(self.current_task["name"])
            return {"status": "completed", "action": "none"}

        self.recovery.create_recovery_checkpoint(
            f"pre_{self.current_task['name']}",
            self.state,
            extra_tags=["pre_action", self.current_task["name"]],
        )

        last_error: Optional[Exception] = None
        action_name = "unknown"
        working_context = context
        blocked_actions: set[str] = set()

        for attempt in range(self.max_step_retries + 1):
            try:
                if blocked_actions:
                    working_context["disallowed_actions"] = sorted(blocked_actions)
                selected_action = self._select_action(working_context)
                action_name = selected_action.get("name", "idle")
                action_context = self._context_for_action(selected_action, working_context)
                self._validate_action(action_name, action_context)
                self._execute_selected_action(action_name, action_context)

                if self.current_plan and self._current_plan_index < len(self.current_plan):
                    self._current_plan_index += 1

                refreshed_context = self._gather_context()
                progress = self._calculate_task_progress(refreshed_context)
                self.task_coordinator.update_task_progress(self.current_task["name"], progress)
                return {"status": "ok", "action": action_name, "progress": progress, "attempt": attempt}

            except ActionInterruptionError as exc:
                logger.warning("Action interrupted (%s): %s", action_name, exc)
                return {"status": "interrupted", "action": action_name, "reason": str(exc)}
            except (ActionFailureError, InvalidContextError) as exc:
                last_error = exc
                recovered, new_context = self.recovery.handle_failure(action_name, exc, working_context)
                blocked_actions.update(new_context.get("disallowed_actions", []))
                if not recovered:
                    break
                working_context = new_context
                self.state.update({k: v for k, v in new_context.items() if k in self.state})
            except (ExecutionError, StaleCheckpointError, DeadlockError, CookieMismatchError) as exc:
                raise exc
            except Exception as exc:
                last_error = ActionFailureError(action_name, f"Unexpected error: {exc}")
                recovered, new_context = self.recovery.handle_failure(action_name, last_error, working_context)
                blocked_actions.update(new_context.get("disallowed_actions", []))
                if not recovered:
                    break
                working_context = new_context

        reason = str(last_error) if last_error else "Unknown execution step failure"
        self.task_coordinator.fail_task(self.current_task["name"], reason)
        return {"status": "failed", "action": action_name, "reason": reason}

    def _build_potential_actions(self) -> list[Dict[str, Any]]:
        return [
            {"name": str(name), "priority": cls.priority, "preconditions": list(cls.preconditions)}
            for name, cls in self.action_class_registry.items()
        ]

    def _select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.current_plan and self._current_plan_index < len(self.current_plan):
            planned_step = self.current_plan[self._current_plan_index]
            if isinstance(planned_step, str):
                return {"name": planned_step}
            if isinstance(planned_step, Mapping):
                selected = dict(planned_step)
                selected.setdefault("name", "unknown")
                return selected
            raise InvalidContextError("action_sequence", [f"step[{self._current_plan_index}]"])
        return self.action_selector.select(self._build_potential_actions(), context)

    @staticmethod
    def _context_for_action(
        selected_action: Mapping[str, Any],
        base_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Merge one plan step into context without mutating task or agent state."""
        merged = copy.deepcopy(dict(base_context))
        for field in ("context", "params", "parameters"):
            payload = selected_action.get(field)
            if payload is None:
                continue
            if not isinstance(payload, Mapping):
                raise InvalidContextError(
                    str(selected_action.get("name", "action")),
                    [f"{field}(mapping)"],
                )
            merged.update(copy.deepcopy(dict(payload)))

        control_fields = {"name", "context", "params", "parameters", "priority"}
        merged.update(
            {
                str(key): copy.deepcopy(value)
                for key, value in selected_action.items()
                if key not in control_fields
            }
        )
        return merged

    def _validate_action(self, action_name: str, context: Dict[str, Any]) -> None:
        is_valid, report = self.validator.validate_plan(
            [{"name": action_name}],
            context,
            mode="continuous",
            level="strict",
        )
        if not is_valid:
            errors = report[0].get("errors", []) if report else ["validation failed"]
            raise ActionFailureError(action_name, f"Action validation failed: {errors}")

    def _execute_selected_action(self, action_name: str, context: Dict[str, Any]) -> None:
        action_class = self.action_class_registry.get(action_name)
        if not action_class:
            raise ActionFailureError(action_name, "Selected action is not registered")

        action_instance = action_class(context=copy.deepcopy(context))
        success = action_instance.execute()
        if not success:
            raise ActionFailureError(action_name, action_instance.failure_reason or "Execution returned False")

        self._update_state_from_action(action_instance)

    def _gather_context(
        self,
        task_override: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        printer.status("EXECUTION", "Context gatherer", "info")

        context = copy.deepcopy(self.state)
        context["current_time"] = time.time()
        context.setdefault("cancel_movement", False)
        context.setdefault("urgent_event", False)
        if self.robot is not None:
            context["robot"] = self.robot
            context["robot_ready"] = True

        task = task_override if task_override is not None else self.current_task
        if task:
            goal = task.get("goal_type", task.get("task_type", "generic"))
            destination = task.get("destination") or task.get("target_position")
            place_position = task.get("place_position") or task.get("destination")
            target_object = task.get("target_object") or task.get("object_id")

            context.update(
                {
                    "current_goal": goal,
                    "deadline": task.get("deadline"),
                    "destination": destination,
                    "place_position": place_position,
                    "target_position": destination,
                    "target_object": target_object,
                    "object_position": task.get("object_position", destination),
                    "object_properties": task.get("object_properties", context.get("object_properties", {})),
                    "robot": task.get("robot") or context.get("robot"),
                    "robot_ready": bool(task.get("robot") or context.get("robot_ready", False)),
                }
            )

            if destination is not None:
                context["has_destination"] = True
            if target_object:
                context["object_detected"] = True
                context["object_nearby"] = True

        pos = context.get("current_position")
        dest = context.get("destination")
        if isinstance(pos, tuple) and isinstance(dest, tuple) and len(pos) >= 2 and len(dest) >= 2:
            context["destination_distance"] = float(((pos[0] - dest[0]) ** 2 + (pos[1] - dest[1]) ** 2) ** 0.5)

        if "required_field" not in context:
            raise InvalidContextError("gather_context", ["required_field"])

        return context

    def _is_task_complete(self, context: Dict[str, Any]) -> bool:
        if not self.current_task:
            return False

        if self.current_plan and self._current_plan_index >= len(self.current_plan):
            return True

        goal = str(self.current_task.get("goal_type", "")).lower()
        if goal in {"navigate", "move", "travel"}:
            return bool(context.get("at_destination"))
        if goal in {"collect", "pick", "pickup"}:
            return bool(context.get("holding_object")) and context.get("held_object") == self.current_task.get("target_object")
        if goal in {"deposit", "place", "drop"}:
            return bool(context.get("object_placed"))
        if goal in {"rest", "recharge", "cooldown"}:
            return bool(context.get("has_rested"))

        return bool(self.current_task.get("success", False))

    def _update_state_from_action(self, action_instance: BaseAction) -> None:
        printer.status("EXECUTION", "Updating state...", "info")

        for key in list(self.state.keys()):
            if key in action_instance.context:
                self.state[key] = action_instance.context[key]

        self.shared_memory.set(f"agent_state:{self.name}", self.state)

    def _calculate_task_progress(self, context: Dict[str, Any]) -> float:
        if not self.current_task:
            return 0.0

        if self._is_task_complete(context):
            return 1.0

        goal = str(self.current_task.get("goal_type", "")).lower()
        if goal in {"navigate", "move", "travel"} and context.get("destination_distance") is not None:
            initial_distance = self._initial_task_distance
            if initial_distance is None:
                initial_distance = max(1.0, float(context.get("destination_distance", 0.0)))
                self._initial_task_distance = initial_distance
            remaining = max(0.0, float(context.get("destination_distance", initial_distance)))
            return max(0.0, min(0.99, 1.0 - (remaining / max(0.1, float(initial_distance)))))

        if goal in {"collect", "pick", "pickup"}:
            return 0.8 if context.get("object_nearby") else 0.2
        if goal in {"deposit", "place", "drop"}:
            return 0.9 if context.get("holding_object") else 0.4
        if goal in {"rest", "recharge", "cooldown"}:
            max_energy = max(1.0, float(context.get("max_energy", 10.0)))
            return max(0.0, min(0.99, float(context.get("energy", 0.0)) / max_energy))

        return float(self.current_task.get("progress", 0.0))

    def dispatch_task(self, task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        task_id = f"{task_type}_{task_data.get('id', task_data.get('order_id', str(uuid.uuid4())[:8]))}"
        task = {
            "task_id": task_id,
            "type": task_type,
            "state": {
                "current_position": task_data.get("pickup_location", task_data.get("current_position")),
                "status": "DISPATCHED",
            },
            "estimated_completion": datetime.utcnow() + timedelta(minutes=30),
            "progress_milestones": ["task_dispatched"],
            "metadata": task_data,
        }
        self.active_tasks[task_id] = task
        return task

    def get_active_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.active_tasks.get(task_id)

    def alternative_execute(self, task_data, original_error=None):
        printer.status("EXECUTION", "Alternative execution...", "info")
        logger.warning("Primary execution failed. Attempting idle recovery.")

        try:
            recovered_state = self.shared_memory.get(f"agent_state:{self.name}")
            if isinstance(recovered_state, dict):
                self.state = recovered_state

            idle_context = self._gather_context()
            idle_action = IdleAction(context=idle_context)
            success = idle_action.execute()
            if success:
                self._update_state_from_action(idle_action)
                return {"status": "recovered", "result": "Agent idled and partially recovered."}
        except Exception as exc:
            logger.error("Alternative execution failed: %s", exc)

        return super().alternative_execute(task_data, original_error)

    def _get_agent_capabilities(self) -> Dict[str, Any]:
        return {
            "capabilities": sorted(
                {"navigation", "object_manipulation", "task_execution", *self.action_class_registry}
            ),
            "current_load": 0.0,
            "efficiency": self.efficiency,
        }

    def sync_state(self, env_state: np.ndarray):
        self.state.update(
            {
                "current_position": (float(env_state[0]), float(env_state[1])),
                "energy": float(env_state[8]),
                "holding_object": bool(env_state[6]),
                "carrying_items": int(env_state[6]),
                "robot_ready": bool(env_state[7]) if len(env_state) > 7 else self.state.get("robot_ready", False),
            }
        )

    def attach_adaptive(self, adaptive_agent):
        self.adaptive_agent = adaptive_agent
        logger.info("Adaptive agent attached to ExecutionAgent")

    def attach_robot(self, robot: RobotInterface) -> None:
        """Attach a robot adapter used by enabled hardware actions."""
        if robot is None:
            raise TypeError("robot cannot be None")
        self.robot = robot

    def get_health_report(self) -> Dict[str, Any]:
        report = self.runtime_status()
        report["subsystems"] = {
            "execution_memory": self.execution_memory.summary(),
            "task_coordinator": self.task_coordinator.summary(),
            "validator": self.validator.get_validation_stats(),
            "recovery": self.recovery.get_recovery_report(),
        }
        report["enabled_actions"] = sorted(self.action_class_registry)
        report["robot_attached"] = self.robot is not None
        report["scheduler_ready"] = self.scheduler is not None
        return report

    def get_validation_report(self):
        return self.validator.get_validation_stats()

    def get_recovery_report(self):
        return self.recovery.get_recovery_report()

    def shutdown(self) -> None:
        """Persist execution state and release agent-owned storage handles."""
        lifecycle = self._runtime_status.lifecycle
        if lifecycle in {RuntimeLifecycle.STOPPING, RuntimeLifecycle.STOPPED}:
            return
        self._transition_runtime_lifecycle(RuntimeLifecycle.STOPPING)
        self.operational_state = "stopping"
        try:
            self.shared_memory.set(f"agent_state:{self.name}", self.state)
            if self._owns_execution_memory:
                self.execution_memory.close()
            else:
                self.execution_memory.flush()
        except Exception as exc:
            self._mark_runtime_degraded("persistence", "execution_memory.shutdown", exc)
            logger.warning("ExecutionAgent shutdown persistence is degraded: %s", exc)
        finally:
            self.operational_state = "stopped"
            self._transition_runtime_lifecycle(RuntimeLifecycle.STOPPED)


if __name__ == "__main__":
    print("\n=== Running Execution Agent ===\n")
    printer.status("TEST", "Execution Agent initialized", "info")
    import uuid

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    memory = SharedMemory()
    factory = AgentFactory()
    execution_config = get_config_section("execution_agent")

    agent = ExecutionAgent(shared_memory=memory, agent_factory=factory, config=execution_config)

    # --- Clear any stale tasks from previous runs ---
    for task in agent.task_coordinator.list_tasks():
        agent.task_coordinator.cancel_task(task["name"], cascade=True)
    # Force immediate save to persist the cleared state
    agent.task_coordinator._save_state()

    task_payload = {
        "name": f"deliver_test_package_{uuid.uuid4().hex[:8]}",
        "goal_type": "navigate",
        "destination": (5.0, 5.0),
        "timeout": 30,
        "requirements": [],
        "priority": 1,
    }

    task_result = agent.perform_task(task_payload)
    pre = agent.predict()
    print(pre)

    ttype="Loose man"
    data={}

    dispatch = agent.dispatch_task(task_data=data, task_type=ttype)
    printer.pretty("PERFORM", task_result, "success" if task_result.get("status") == "success" else "warning")