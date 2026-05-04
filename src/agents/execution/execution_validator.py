import copy
import json
import math

from typing import Any, Dict, List, Optional, Tuple, Type

from .utils.config_loader import load_global_config, get_config_section
from .utils.execution_error import (ActionFailureError, InvalidContextError,
                                                        InvalidTaskTransitionError, MissingActionHandlerError,
                                                        UnreachableTargetError,)
from .execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Validator")
printer = PrettyPrinter

class ExecutionValidator:
    VALIDATION_MODES = ["preflight", "continuous", "simulation"]
    VALIDATION_LEVELS = ["strict", "relaxed", "partial"]

    def __init__(self, memory: Optional[ExecutionMemory] = None):
        self.config = load_global_config()
        self.validator_config = get_config_section("execution_validator") or {}
        self.validation_mode = self.validator_config.get("default_mode", "preflight")
        self.validation_level = self.validator_config.get("default_level", "strict")
        self.max_object_distance = float(self.validator_config.get("max_object_distance", 5.0))
        self.max_navigation_distance = self.validator_config.get("max_navigation_distance")
        if self.max_navigation_distance is not None:
            self.max_navigation_distance = float(self.max_navigation_distance)
        self.min_energy_threshold = float(self.validator_config.get("min_energy_threshold", 2.0))
        self.position_tolerance = float(self.validator_config.get("position_tolerance", 0.5))
        self.validation_cache_ttl = int(self.validator_config.get("validation_cache_ttl", 60))
        self.max_consecutive_same_action = int(self.validator_config.get("max_consecutive_same_action", 3))

        self.memory = memory or ExecutionMemory()
        self.world_model: Optional[Any] = None
        self.action_registry: Dict[str, Type[Any]] = {}

        self.thresholds = {
            "max_distance": self.max_object_distance,
            "max_navigation_distance": self.max_navigation_distance,
            "min_energy": self.min_energy_threshold,
            "position_tolerance": self.position_tolerance,
        }

    def register_action_handler(self, name: str, action_class: Type[Any]) -> None:
        self.action_registry[name] = action_class

    def register_world_model(self, world_model: Any) -> None:
        self.world_model = world_model

    def validate_plan(
        self,
        plan: List[Any],
        context: Dict[str, Any],
        mode: Optional[str] = None,
        level: Optional[str] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a plan's feasibility with configurable strictness.

        Returns:
            (is_valid, validation_report)
        """
        validation_mode = self._normalize_mode(mode)
        validation_level = self._normalize_level(level)

        cache_key = self._validation_cache_key(plan, context, validation_mode, validation_level)
        cached = self.memory.get_cache(cache_key, namespace="execution_validator")
        if cached:
            return cached["is_valid"], cached["report"]

        simulated_context = copy.deepcopy(context or {})
        validation_report: List[Dict[str, Any]] = []
        overall_valid = True

        for index, task in enumerate(plan or []):
            task_name = self._get_task_name(task)
            report = self._new_task_report(task_name, index + 1)

            action_class = self.action_registry.get(task_name)
            if not action_class:
                error = MissingActionHandlerError(task_name)
                report["errors"].append(str(error))
                validation_report.append(report)
                overall_valid = False
                if validation_mode == "preflight":
                    break
                continue

            action = None
            try:
                action = action_class(copy.deepcopy(simulated_context))
                self._validate_action_state(action, task_name)
                report["preconditions_met"] = True
            except Exception as exc:
                report["errors"].append(str(exc))
                overall_valid = False

            env_report = self._check_environment_constraints(task_name, simulated_context, validation_level)
            report["warnings"].extend(env_report["warnings"])
            report["errors"].extend(env_report["errors"])
            report["environment_constraints"] = env_report["valid"]
            if not env_report["valid"]:
                overall_valid = False

            if index > 0:
                prev_name = self._get_task_name(plan[index - 1])
                transition_valid, transition_warning = self._check_action_transition(prev_name, task_name)
                report["logical_progression"] = transition_valid
                if transition_warning:
                    report["warnings"].append(transition_warning)
                if validation_level == "strict" and not transition_valid:
                    overall_valid = False

            if action is not None and (not report["errors"] or validation_level != "strict"):
                try:
                    simulated_context = self._simulate_postconditions(action, simulated_context)
                    report["postconditions_simulated"] = True
                    logical_ok = self._check_logical_progression(plan[: index + 1], simulated_context)
                    report["logical_progression"] = report["logical_progression"] and logical_ok
                    if validation_level == "strict" and not logical_ok:
                        report["errors"].append(
                            str(ActionFailureError(task_name, "Logical progression check failed"))
                        )
                        overall_valid = False
                except Exception as exc:
                    report["errors"].append(
                        str(ActionFailureError(task_name, f"Postcondition simulation failed: {exc}"))
                    )
                    overall_valid = False

            validation_report.append(report)
            if validation_mode == "preflight" and report["errors"]:
                break

        payload = {"is_valid": overall_valid, "report": validation_report}
        self.memory.set_cache(
            cache_key,
            payload,
            namespace="execution_validator",
            ttl=self.validation_cache_ttl,
        )
        return overall_valid, validation_report

    def _normalize_mode(self, mode: Optional[str]) -> str:
        mode = mode or self.validation_mode
        if mode not in self.VALIDATION_MODES:
            logger.warning("Invalid validation mode %s. Using default.", mode)
            return "preflight"
        return mode

    def _normalize_level(self, level: Optional[str]) -> str:
        level = level or self.validation_level
        if level not in self.VALIDATION_LEVELS:
            logger.warning("Invalid validation level %s. Using default.", level)
            return "strict"
        return level

    def _new_task_report(self, task_name: str, step: int) -> Dict[str, Any]:
        return {
            "task_name": task_name,
            "step": step,
            "preconditions_met": False,
            "postconditions_simulated": False,
            "environment_constraints": False,
            "logical_progression": True,
            "errors": [],
            "warnings": [],
        }

    def _get_task_name(self, task: Any) -> str:
        if hasattr(task, "name"):
            return str(task.name)
        if isinstance(task, dict):
            return str(task.get("name", "unknown"))
        return str(task)

    def _validate_action_state(self, action: Any, task_name: str) -> None:
        if hasattr(action, "validate_context"):
            action.validate_context()
        if hasattr(action, "check_preconditions"):
            action.check_preconditions()
        elif hasattr(action, "_validate_state"):
            preconditions_met = bool(action._validate_state())
            if not preconditions_met:
                raise ActionFailureError(task_name, "Preconditions not satisfied")
        else:
            logger.debug("Action %s exposes no explicit validation hook", task_name)

    def _check_environment_constraints(
        self,
        action_name: str,
        context: Dict[str, Any],
        validation_level: str,
    ) -> Dict[str, Any]:
        report = {"valid": True, "errors": [], "warnings": []}

        if action_name in {"move_to", "pick_object", "place_object"}:
            position_key = "destination" if action_name == "move_to" else "target_position"
            if position_key not in context:
                report["errors"].append(str(InvalidContextError(action_name, [position_key])))
                report["valid"] = False
            else:
                target_pos = context[position_key]
                current_pos = context.get("current_position", (0.0, 0.0))
                if not self._is_position(current_pos) or not self._is_position(target_pos):
                    report["errors"].append(str(ActionFailureError(action_name, "Invalid position format")))
                    report["valid"] = False
                else:
                    distance = self._calculate_distance(current_pos, target_pos)
                    if action_name == "move_to":
                        if self.max_navigation_distance is not None and distance > self.max_navigation_distance:
                            report["errors"].append(
                                str(UnreachableTargetError(action_name, target_pos, current_pos))
                            )
                            report["valid"] = False
                        if not self._is_position_in_map(target_pos, context.get("map_data")):
                            report["errors"].append(
                                str(UnreachableTargetError(action_name, target_pos, current_pos))
                            )
                            report["valid"] = False
                    elif distance > self.thresholds["max_distance"]:
                        report["errors"].append(
                            str(UnreachableTargetError(action_name, target_pos, current_pos))
                        )
                        report["valid"] = False

        if action_name != "idle" and "energy" in context:
            energy = float(context.get("energy", 0.0))
            if energy < self.thresholds["min_energy"]:
                warning = str(
                    ActionFailureError(
                        action_name,
                        f"Low energy ({energy:.2f} < {self.thresholds['min_energy']:.2f})",
                    )
                )
                report["warnings"].append(warning)
                if validation_level == "strict":
                    report["valid"] = False

        if action_name == "pick_object" and context.get("holding_object"):
            report["errors"].append(
                str(ActionFailureError(action_name, "Cannot pick object while already holding one"))
            )
            report["valid"] = False

        if action_name == "place_object" and not context.get("holding_object", False):
            report["errors"].append(
                str(ActionFailureError(action_name, "Cannot place object without holding one"))
            )
            report["valid"] = False

        if "holding_object" in context and context.get("holding_object") and "held_object" not in context:
            report["warnings"].append("holding_object is True but held_object is missing")

        if self.world_model is not None:
            current_pos = context.get("current_position")
            try:
                if current_pos is not None and hasattr(self.world_model, "validate_position"):
                    if not self.world_model.validate_position(current_pos):
                        report["errors"].append(str(ActionFailureError(action_name, "Invalid position in world model")))
                        report["valid"] = False
                if action_name == "move_to" and hasattr(self.world_model, "is_reachable"):
                    target_pos = context.get("destination")
                    if target_pos is not None and not self.world_model.is_reachable(current_pos, target_pos):
                        report["errors"].append(str(ActionFailureError(action_name, "Destination is unreachable in world model")))
                        report["valid"] = False
            except Exception as exc:
                report["warnings"].append(f"World model validation skipped: {exc}")

        return report

    def _is_position_in_map(self, position: Tuple[float, float], map_data: Any) -> bool:
        if map_data is None:
            return True
        if (
            not isinstance(map_data, list)
            or not map_data
            or not isinstance(map_data[0], list)
            or not map_data[0]
        ):
            return False
        rows = len(map_data)
        cols = len(map_data[0])
        x, y = position[0], position[1]
        return 0.0 <= float(x) < float(rows) and 0.0 <= float(y) < float(cols)

    def _simulate_postconditions(self, action: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        new_context = copy.deepcopy(context)

        for postcondition in getattr(action, "postconditions", []):
            if postcondition == "at_destination" and "destination" in new_context:
                new_context["current_position"] = new_context["destination"]
                new_context["at_destination"] = True
            elif postcondition == "holding_object":
                new_context["holding_object"] = True
                if "target_object" in new_context:
                    new_context["held_object"] = new_context["target_object"]
            elif postcondition == "object_placed":
                new_context["holding_object"] = False
                new_context["held_object"] = None
                new_context["object_placed"] = True
            else:
                new_context[postcondition] = True

        action_name = getattr(action, "name", None)
        if action_name == "move_to" and "destination" in new_context:
            new_context["current_position"] = new_context["destination"]
            new_context["at_destination"] = True

        energy_cost = float(getattr(action, "energy_cost", getattr(action, "cost", 0.0)))
        if "energy" in new_context:
            new_context["energy"] = max(0.0, float(new_context["energy"]) - max(0.0, energy_cost))

        return new_context

    def _check_action_transition(self, prev_action: str, next_action: str) -> Tuple[bool, Optional[str]]:
        illegal_transitions = {
            "place_object": {"pick_object"},
            "idle": {"idle"},
        }
        allowed_transitions = {
            "pick_object": {"move_to", "place_object", "idle"},
            "place_object": {"move_to", "idle"},
            "move_to": {"pick_object", "place_object", "idle"},
        }

        if next_action in illegal_transitions.get(prev_action, set()):
            warning = str(InvalidTaskTransitionError(prev_action, next_action))
            return False, warning

        if prev_action in allowed_transitions and next_action not in allowed_transitions[prev_action]:
            warning = str(InvalidTaskTransitionError(prev_action, next_action))
            return False, warning

        return True, None

    def _check_logical_progression(self, partial_plan: List[Any], context: Dict[str, Any]) -> bool:
        if len(partial_plan) >= self.max_consecutive_same_action:
            tail = [self._get_task_name(task) for task in partial_plan[-self.max_consecutive_same_action:]]
            if len(set(tail)) == 1:
                return False
    
        if context.get("holding_object") and "held_object" not in context:
            return False
    
        if not partial_plan:
            return True
    
        goal = context.get("current_goal")
        last_action = self._get_task_name(partial_plan[-1])
    
        if goal == "navigate":
            return last_action == "move_to" or bool(context.get("at_destination", False))
    
        if goal == "collect":
            return last_action in {"move_to", "pick_object"} and not bool(context.get("object_placed", False))
    
        if goal == "place":
            if last_action in {"move_to", "pick_object"}:
                return True
            if last_action == "place_object":
                return bool(context.get("object_placed", False) or not context.get("holding_object", False))
            return False
    
        return True

    def generate_validation_summary(self, validation_report: List[Dict[str, Any]]) -> str:
        if not validation_report:
            return "No validation report available"

        summary = ["Plan Validation Report:"]
        total_tasks = len(validation_report)
        passed_tasks = 0

        for task_report in validation_report:
            status = "PASSED" if not task_report["errors"] else "FAILED"
            if status == "PASSED":
                passed_tasks += 1

            summary.append(f"\n[{status}] Task {task_report['step']}: {task_report['task_name']}")
            if task_report["warnings"]:
                summary.append("  Warnings:")
                for warning in task_report["warnings"]:
                    summary.append(f"    - {warning}")
            if task_report["errors"]:
                summary.append("  Errors:")
                for error in task_report["errors"]:
                    summary.append(f"    - {error}")

        success_rate = (passed_tasks / total_tasks) * 100 if total_tasks else 0.0
        summary.append(f"\nValidation Result: {passed_tasks}/{total_tasks} tasks passed ({success_rate:.1f}%)")
        return "\n".join(summary)

    def _validation_cache_key(
        self,
        plan: List[Any],
        context: Dict[str, Any],
        mode: str,
        level: str,
    ) -> str:
        plan_repr = [self._get_task_name(task) for task in plan]
        payload = json.dumps(
            {
                "plan": plan_repr,
                "context": context,
                "mode": mode,
                "level": level,
            },
            sort_keys=True,
            default=str,
        )
        return payload

    @staticmethod
    def _is_position(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and all(isinstance(component, (int, float)) for component in value[:2])
        )

    @staticmethod
    def _calculate_distance(pos1: Any, pos2: Any) -> float:
        try:
            return math.hypot(float(pos1[0]) - float(pos2[0]), float(pos1[1]) - float(pos2[1]))
        except (TypeError, IndexError, ValueError):
            return float("inf")


if __name__ == "__main__":
    print("\n=== Running Execution Validator Tests ===\n")
    printer.status("TEST", "Starting Execution Validator tests", "info")

    validator = ExecutionValidator()

    class Action:
        name = "move_to"
        postconditions = ["at_destination"]
        energy_cost = 2.0

        def __init__(self, context):
            self.context = context

        def validate_context(self):
            return True

        def check_preconditions(self):
            return True

    validator.action_registry = {
        "move_to": Action,
        "pick_object": Action,
        "place_object": Action,
    }

    class Task:
        def __init__(self, name):
            self.name = name

    test_plan = [Task("move_to"), Task("pick_object"), Task("place_object")]
    validator.memory.clear_cache(namespace="execution_validator")
    
    test_context = {
        "current_position": (0, 0),
        "target_position": (2, 2),
        "destination": (2, 2),
        "energy": 10.0,
        "holding_object": False,
        "current_goal": "place",   # or remove this key
    }

    is_valid, report = validator.validate_plan(test_plan, test_context)
    printer.status("VALIDATOR", f"Plan Valid: {is_valid}", "info" if is_valid else "error")
    print(validator.generate_validation_summary(report))
    print("\n=== All validator tests completed ===")
