
import copy
import os

from typing import Dict, List, Optional, Tuple, Type

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.utils.execution_error import (
    InvalidContextError, ActionFailureError, 
    MissingActionHandlerError, InvalidTaskTransitionError,
    UnreachableTargetError
)
from src.agents.execution.execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Validator")
printer = PrettyPrinter

class ExecutionValidator:
    VALIDATION_MODES = ["preflight", "continuous", "simulation"]
    VALIDATION_LEVELS = ["strict", "relaxed", "partial"]

    def __init__(self):
        self.config = load_global_config()
        self.validator_config = get_config_section("execution_validator")
        self.validation_mode = self.validator_config.get("default_mode", "preflight")
        self.validation_level = self.validator_config.get("default_level", "strict")
        self.max_object_distance = self.validator_config.get("max_object_distance")
        self.min_energy_threshold = self.validator_config.get("min_energy_threshold")
        self.position_tolerance = self.validator_config.get("position_tolerance")

        self.memory = ExecutionMemory()
        self.world_model = None or []
        self.action_registry = []
        
        # Load validation thresholds
        self.thresholds = {
            "max_distance": self.max_object_distance,
            "min_energy": self.min_energy_threshold,
            "position_tolerance": self.position_tolerance
        }

    def validate_plan(self, plan, context, mode=None, level=None) -> Tuple[bool, List[Dict]]:
        """
        Validate a plan's feasibility with configurable strictness
        Returns tuple: (is_valid, validation_report)
        """
        validation_mode = mode or self.validation_mode
        validation_level = level or self.validation_level
        
        if validation_mode not in self.VALIDATION_MODES:
            logger.warning(f"Invalid validation mode {validation_mode}. Using default.")
            validation_mode = "preflight"
            
        if validation_level not in self.VALIDATION_LEVELS:
            logger.warning(f"Invalid validation level {validation_level}. Using default.")
            validation_level = "strict"
        
        simulated_context = copy.deepcopy(context)
        validation_report = []
        is_valid = True

        def get_task_name(task):
            if hasattr(task, 'name'):
                return task.name
            elif isinstance(task, dict):
                return task.get('name', 'unknown')
            else:
                return str(task)

        for i, task in enumerate(plan):
            task_name = get_task_name(task)
            task_report = {
                "task_name": task_name,
                "step": i+1,
                "preconditions_met": False,
                "postconditions_simulated": False,
                "environment_constraints": False,
                "logical_progression": False,
                "errors": [],
                "warnings": []
            }

            # 1. Action existence check
            action_class = self.action_registry.get(task_name)
            if not action_class:
                error = MissingActionHandlerError(task_name)
                task_report["errors"].append(str(error))
                logger.error(str(error))
                is_valid = False
                validation_report.append(task_report)
                continue

            # 2. Precondition validation
            try:
                action = action_class(simulated_context)
                preconditions_met = action._validate_state()
                
                if not preconditions_met:
                    error = ActionFailureError(task.name, "Preconditions not satisfied")
                    task_report["errors"].append(str(error))
                    is_valid = False
                else:
                    task_report["preconditions_met"] = True
            except KeyError as e:
                error = InvalidContextError(task_name, [str(e)])
            except Exception as e:
                error = ActionFailureError(task_name, f"Validation error: {str(e)}")

            # 3. Environment constraints check
            env_check = self._check_environment_constraints(task_name, context)
            if not env_check["valid"]:
                for error in env_check.get("errors", []):
                    task_report["errors"].append(error)
                    is_valid = False
            else:
                task_report["environment_constraints"] = True
                
            for warning in env_check.get("warnings", []):
                task_report["warnings"].append(warning)

            # 4. Logical progression check (only in strict mode)
            if validation_level == "strict" and i > 0:
                prev_action = get_task_name(plan[i-1])
                if not self._check_action_transition(prev_action, task.name):
                    warning = InvalidTaskTransitionError(
                        current_state=prev_action,
                        attempted_action=task.name
                    )
                    task_report["warnings"].append(str(warning))
                    if validation_level == "strict":
                        is_valid = False

            # 5. Simulate postconditions
            if is_valid or validation_level != "strict":
                try:
                    simulated_context = self._simulate_postconditions(action, simulated_context)
                    task_report["postconditions_simulated"] = True
                    
                    # Check logical progression after simulation
                    if i > 0:
                        task_report["logical_progression"] = self._check_logical_progression(
                            plan[:i+1], simulated_context
                        )
                except Exception as e:
                    error = ActionFailureError(task.name, f"Postcondition simulation failed: {str(e)}")
                    task_report["errors"].append(str(error))
                    is_valid = False

            validation_report.append(task_report)
            
            # Early termination for preflight mode
            if not is_valid and validation_mode == "preflight":
                break

        return is_valid, validation_report

    def _check_environment_constraints(self, action_name, context) -> Dict:
        """Check physical and environmental constraints"""
        report = {"valid": True, "errors": [], "warnings": []}
        
        # Position-based checks
        if action_name in ["pick_object", "place_object", "move_to"]:
            position_key = "destination" if action_name == "move_to" else "target_position"
            if position_key not in context:
                error = InvalidContextError(action_name, [position_key])
                report["errors"].append(str(error))
                report["valid"] = False
            else:
                target_pos = context[position_key]
                current_pos = context.get("current_position", (0, 0))
                distance = self._calculate_distance(current_pos, target_pos)
                if distance > self.thresholds["max_distance"]:
                    error = UnreachableTargetError(
                        action_name=action_name,
                        target=target_pos,
                        agent_pos=current_pos
                    )
                    report["errors"].append(str(error))
                    report["valid"] = False
        
        # Energy-based checks
        if action_name != "idle" and "energy" in context:
            if context["energy"] < self.thresholds["min_energy"]:
                warning = ActionFailureError(
                    action_name, 
                    f"Low energy ({context['energy']:.2f} < {self.thresholds['min_energy']})"
                )
                report["warnings"].append(str(warning))
                if self.validation_level == "strict":
                    report["valid"] = False
        
        # Object state checks
        if action_name == "pick_object" and "holding_object" in context:
            if context["holding_object"]:
                error = ActionFailureError(
                    action_name, 
                    "Cannot pick object while already holding one"
                )
                report["errors"].append(str(error))
                report["valid"] = False
        
        if action_name == "place_object" and "holding_object" in context:
            if not context["holding_object"]:
                error = ActionFailureError(
                    action_name, 
                    "Cannot place object without holding one"
                )
                report["errors"].append(str(error))
                report["valid"] = False
        
        # World model validation
        if self.world_model:
            if not self.world_model.validate_position(context.get("current_position")):
                error = ActionFailureError(
                    action_name, 
                    "Invalid position in world model"
                )
                report["errors"].append(str(error))
                report["valid"] = False
        
        return report

    def _simulate_postconditions(self, action, context) -> Dict:
        """Simulate state changes from action execution"""
        new_context = copy.deepcopy(context)
        
        # Apply registered postconditions
        for pc in getattr(action, 'postconditions', []):
            if pc == "at_destination":
                if "destination" in new_context:
                    new_context["current_position"] = new_context["destination"]
            elif pc == "holding_object":
                new_context["holding_object"] = True
                if "target_object" in new_context:
                    new_context["held_object"] = new_context["target_object"]
            elif pc == "object_placed":
                new_context["holding_object"] = False
                new_context["held_object"] = None
            else:
                new_context[pc] = True
        
        # Apply special state transitions
        if action.name == "move_to":
            if "destination" in new_context:
                new_context["current_position"] = new_context["destination"]
                new_context["at_destination"] = True
        
        # Energy consumption simulation
        energy_cost = getattr(action, 'energy_cost', 0)
        if "energy" in new_context:
            new_context["energy"] = max(0, new_context["energy"] - energy_cost)
        
        return new_context

    def _check_action_transition(self, prev_action, next_action) -> bool:
        """Validate logical progression between actions"""
        # Define illegal transitions
        illegal_transitions = {
            "place_object": ["pick_object"],  # Can't pick after placing without intermediate
            "idle": ["idle"]  # Can't idle consecutively
        }
        
        # Allow transitions that are explicitly allowed
        allowed_transitions = {
            "pick_object": ["move_to", "place_object"],
            "place_object": ["move_to"],
            "move_to": ["pick_object", "place_object", "idle"]
        }
        
        # Check for explicitly illegal transitions
        if next_action in illegal_transitions.get(prev_action, []):
            return False
        
        # Check if transition is in allowed set (if defined)
        if prev_action in allowed_transitions:
            return next_action in allowed_transitions[prev_action]
        
        return True

    def _check_logical_progression(self, partial_plan, context) -> bool:
        """Check if the partial plan makes logical sense"""
        # Check for repeated failures
        if len(partial_plan) > 3:
            last_three = [t.name for t in partial_plan[-3:]]
            if len(set(last_three)) == 1:  # Same action 3x in row
                return False
        
        # Check for impossible states
        if "holding_object" in context:
            if context["holding_object"] and "held_object" not in context:
                return False
        
        # Check goal progression
        if "current_goal" in context:
            goal = context["current_goal"]
            if goal == "navigate" and "at_destination" in context:
                return context["at_destination"]
            if goal == "collect" and "holding_object" in context:
                return context["holding_object"]
        
        return True

    def _calculate_distance(self, pos1, pos2) -> float:
        """Calculate Euclidean distance between two points"""
        try:
            return ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5
        except (TypeError, IndexError):
            return float('inf')

    def generate_validation_summary(self, validation_report) -> str:
        """Generate human-readable validation report"""
        if not validation_report:
            return "No validation report available"
        
        summary = ["Plan Validation Report:"]
        total_tasks = len(validation_report)
        passed_tasks = 0
        
        for task_report in validation_report:
            status = "PASSED" if not task_report["errors"] else "FAILED"
            if status == "PASSED":
                passed_tasks += 1
                
            summary.append(
                f"\n[{status}] Task {task_report['step']}: {task_report['task_name']}"
            )
            
            if task_report["warnings"]:
                summary.append("  Warnings:")
                for warning in task_report["warnings"]:
                    summary.append(f"    - {warning}")
            
            if task_report["errors"]:
                summary.append("  Errors:")
                for error in task_report["errors"]:
                    summary.append(f"    - {error}")
        
        success_rate = (passed_tasks / total_tasks) * 100
        summary.append(f"\nValidation Result: {passed_tasks}/{total_tasks} tasks passed ({success_rate:.1f}%)")
        
        return "\n".join(summary)
    
    def _find_task(self, task_name: str) -> Optional[Dict]:
        return {'name': task_name}  # Simplified for training

if __name__ == "__main__":
    print("\n=== Running Execution Validator Tests ===\n")
    printer.status("TEST", "Starting Execution Validator tests", "info")

    validator = ExecutionValidator()

    printer.pretty("Movement Validator", validator, "success" if validator else "error")

    # Register dummy action handlers for testing
    class Action:
        name = "move_to"
        postconditions = ["at_destination"]
        energy_cost = 2.0
        def __init__(self, context): self.context = context
        def _validate_state(self): return True

    validator.action_registry = {
        "move_to": Action,
        "pick_object": Action,
        "place_object": Action
    }

    # Define test plan (mock Task objects)
    class Task:  # lightweight mock
        def __init__(self, name): self.name = name

    test_plan = [Task("move_to"), Task("pick_object"), Task("place_object")]

    # Define mock context
    test_context = {
        "current_position": (0, 0),
        "target_position": (2, 2),
        "destination": (2, 2),
        "energy": 10.0,
        "holding_object": False,
        "current_goal": "collect"
    }

    # Run validator
    is_valid, report = validator.validate_plan(test_plan, test_context)

    # Display results
    printer.status("VALIDATOR", f"Plan Valid: {is_valid}", "info" if is_valid else "error")
    summary = validator.generate_validation_summary(report)
    print(summary)
    print("\n=== All validator tests completed ===")