__version__ = "1.9.0"

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

import json
import uuid
import torch
import copy
import time
import math
import numpy as np

from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, Union, Tuple, Optional, Any, Type

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.execution.utils.execution_error import (TimeoutError, InvalidContextError, ExecutionError,
    StaleCheckpointError, DeadlockError, ActionFailureError, CookieMismatchError, ActionInterruptionError)
from src.agents.execution.task_coordinator import TaskCoordinator, TaskState
from src.agents.execution.action_selector import ActionSelector
from src.agents.execution.actions.base_action import BaseAction
from src.agents.execution.actions.move_to import MoveToAction
from src.agents.execution.actions.pick_object import PickObjectAction
from src.agents.execution.actions.place_object import PlaceObjectAction
from src.agents.execution.actions.idle import IdleAction
from src.agents.execution.execution_recovery import ExecutionRecovery
from src.agents.execution.execution_validator import ExecutionValidator
from src.agents.planning.task_scheduler import DeadlineAwareScheduler
from src.agents.planning.planning_types import Task
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Execution Agent")
printer = PrettyPrinter

class ExecutionAgent(BaseAgent):
    """
    Orchestrates the execution of tasks by managing agent state, selecting
    appropriate actions, and overseeing their lifecycle. It acts as the bridge

    between high-level planning and low-level action execution.
    """
    def __init__(self,
                 shared_memory,
                 agent_factory, config=None):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config, 
        )
        self.adaptive_agent = None
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.execution_config = get_config_section('execution_agent')

        self.thresholds = self.shared_memory.get(
            "execution_thresholds", 
            {"timeout": 300, "energy_alert": 2.0}  # Defaults
        )

        # Core components for decision making and task management
        self.task_coordinator = TaskCoordinator()
        self.action_selector = ActionSelector()
        self.validator = ExecutionValidator()
        self.recovery = ExecutionRecovery()

        # Agent's internal state
        self.state: Dict[str, Any] = self._initialize_state()
        self.current_task: Optional[Dict] = None

        # Registry to map action names to their respective classes for dynamic instantiation
        self.action_class_registry: Dict[str, Type[BaseAction]] = {
            "move_to": MoveToAction,
            "pick_object": PickObjectAction,
            "place_object": PlaceObjectAction,
            "idle": IdleAction
        }
        self.active_tasks = {}

        # Register actions with the selector for precondition awareness
        for name, cls in self.action_class_registry.items():
            self.action_selector.register_action(name, cls.preconditions, cls.postconditions)

        self.validator.action_registry = self.action_class_registry
        self.recovery.task_coordinator = self.task_coordinator

        logger.info("Execution Agent initialized with core components and state.")

    def _initialize_state(self) -> Dict[str, Any]:
        """Initializes the agent's internal state."""
        printer.status("EXECUTION", "Initializig states...", "info")

        # Try loading from shared memory
        saved_state = self.shared_memory.get(f"agent_state:{self.name}")
        if saved_state:
            printer.status("STATE", "Loaded from shared memory", "success")
            return saved_state

        state = {
            "energy": 10.0,
            "max_energy": 10.0,
            "current_position": (0.0, 0.0),
            "hand_empty": True,
            "holding_object": False,
            "held_object": None,
            "carrying_items": 0,
            "inventory": {},
            "map_data": [[0] * 100 for _ in range(100)],  # 100x100 grid
            "object_state": {},
            "required_field": True
        }

        self.shared_memory.set(f"agent_state:{self.name}", state)
        return state
    
    def predict(self, state: Any = None) -> Dict[str, Any]:
        """
        Predicts the next action the agent would take given the current state.
        
        Args:
            state (Any, optional): The current state of the agent. If None, use the agent's current state.
            
        Returns:
            Dict[str, Any]: A structured prediction containing:
                - selected_action: The name of the action that would be taken
                - confidence: Confidence score for the prediction
                - context: Context used for decision-making
                - task_progress: Current progress of the active task
        """
        # If state is provided, use it; otherwise gather current context
        context = state if state is not None else self._gather_context()
        
        # Generate potential actions
        potential_actions = [
            {"name": str(name), "priority": cls.priority, "preconditions": cls.preconditions}
            for name, cls in self.action_class_registry.items()
        ]
        
        try:
            # Select the best action
            selected_action_dict = self.action_selector.select(potential_actions, context)
            action_name = selected_action_dict.get("name")
            
            return {
                "selected_action": action_name,
                "confidence": 1.0,  # Action selector always returns highest confidence
                "context": context,
                "task_progress": self._calculate_task_progress(context) if self.current_task else 0.0
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "selected_action": "idle",
                "confidence": 0.0,
                "context": context,
                "task_progress": 0.0
            }

    def _generate_default_plan(self, task_data):
        """Generate a default empty plan"""
        return []

    def perform_task(self, task_data: Dict) -> Dict:
        printer.status("EXECUTION", "Task Performer", "info")
        if 'action_sequence' not in task_data:
            task_data['action_sequence'] = self._generate_default_plan(task_data)

        # Validate plan only if it exists and is non-empty
        if 'action_sequence' in task_data and task_data['action_sequence']:
            plan = task_data['action_sequence']
            context = self._gather_context()
            
            # Validate plan before execution
            is_valid, report = self.validator.validate_plan(plan, context)
            if not is_valid:
                logger.error(f"Plan validation failed: {self.validator.generate_validation_summary(report)}")
                return {
                    "status": "failed",
                    "reason": "Plan validation failed",
                    "validation_report": report
                }
            self.current_plan = plan

        # Ensure task has all required fields for scheduler
        task_data.setdefault('id', f"{task_data.get('name', 'task')}_{str(uuid.uuid4())[:8]}")
        task_data.setdefault('requirements', [])
        task_data.setdefault('deadline', time.time() + 300)  # Default 5 min deadline

        if isinstance(task_data, str):
            try:
                # Attempt to convert string to dictionary
                task_data = json.loads(task_data)
            except json.JSONDecodeError:
                return {
                    "status": "failed",
                    "reason": f"Invalid task_data format: {task_data}"
                }
        elif not isinstance(task_data, dict):
            return {
                "status": "failed",
                "reason": f"Expected dict, got {type(task_data).__name__}"
            }
    
        # Check if task is already being handled
        if self.shared_memory.get(f"task_in_progress:{task_data['name']}"):
            return {"status": "failed", "reason": "Task already in progress"}
    
        # Get timeout value first
        timeout = task_data.get('timeout', 300)  # Default to 300 seconds
    
        # Claim task with proper TTL
        self.shared_memory.set(
            f"task_in_progress:{task_data['name']}", 
            {"agent": self.name, "start_time": time.time()},
            ttl=timeout + 120  # Now timeout is defined
        )
    
        self.shared_memory.publish(
            "task_events",
            {"event": "task_started", "task": task_data, "agent": self.name}
        )

        try:
            self.scheduler = DeadlineAwareScheduler()
            schedule = self.scheduler.schedule(
                tasks=[task_data],
                agents={self.name: self._get_agent_capabilities()},
                state=self.state
            )
            
            # Validate schedule using task ID instead of name
            if not schedule or task_data['id'] not in schedule:
                logger.error("Scheduler failed to create valid schedule")
                return {
                    "status": "failed", 
                    "reason": "Scheduling failed - task not scheduled"
                }
                
            # Get scheduled assignment using task ID
            assignment = schedule.get(task_data['id'])
            if not assignment:
                logger.error("Scheduler returned invalid assignment")
                return {
                    "status": "failed", 
                    "reason": "Scheduling failed - no valid assignment"
                }

            if not self.task_coordinator.assign_task(task_data):
                logger.error(f"Failed to assign task: {task_data.get('name')}. It may be a duplicate or invalid.")
                return {"status": "failed", "reason": "Task assignment failed."}

            self.current_task = self.task_coordinator.get_next_task()
            if not self.current_task or self.current_task['name'] != task_data['name']:
                return {"status": "failed", "reason": "Could not retrieve assigned task for execution."}

            self.task_coordinator.start_task(self.current_task['name'], assignee=self.name)
            logger.info(f"Starting execution loop for task: {self.current_task['name']}")

            start_time = time.time()
            timeout = self.current_task.get('timeout', 300)

            recovery_id = self.recovery.create_recovery_checkpoint("pre_task", self.state)
            logger.info(f"Created pre-task recovery checkpoint: {recovery_id}")

            while self.current_task and self.current_task['state'] == TaskState.IN_PROGRESS.value:
                if time.time() - start_time > timeout:
                    self.task_coordinator.fail_task(self.current_task['name'], "Execution timed out")
                    raise TimeoutError(self.current_task['name'], timeout, time.time() - start_time)
                
                # RESET FAILURE FLAGS BEFORE RETRY
                self.context = self._gather_context()
                self.context["cancel_movement"] = False
                self.context["urgent_event"] = False

                try:
                    self._execution_step()
                except ActionInterruptionError as e:
                    logger.warning(f"Action interrupted: {e}")
                    self.task_coordinator.pause_task(self.current_task['name'])
                except ActionFailureError as e:
                    # SPECIAL HANDLING FOR MOVEMENT FAILURES
                    if "move_to" in str(e):
                        logger.warning("Movement failure detected, resetting state")
                        self.state["cancel_movement"] = False
                        self.task_coordinator.retry_task(self.current_task['name'])
                        continue  # Retry immediately

                try:
                    self._execution_step()
                except (ExecutionError, InvalidContextError, StaleCheckpointError, 
                        DeadlockError, CookieMismatchError) as e:
                    self.task_coordinator.fail_task(self.current_task['name'], reason=str(e))
                    logger.error(f"Execution step failed for task '{self.current_task['name']}' with error: {e}")
                    raise
                except Exception as e:
                    # Catch unexpected errors
                    self.task_coordinator.fail_task(self.current_task['name'], reason=f"Unexpected error: {str(e)}")
                    raise ActionFailureError("unknown", f"Unexpected runtime error during execution step: {e}")

                # Check if task was completed or failed inside the step
                self.current_task = self.task_coordinator._find_task(self.current_task['name'])

            final_task_state = self.task_coordinator.validate_task_completion(task_data['name'])
            if final_task_state:
                logger.info(f"Task '{task_data['name']}' completed successfully.")
                return {"status": "success", "result": "Task completed."}
            else:
                history_task = next((t for t in self.task_coordinator.task_history if t['name'] == task_data['name']), None)
                reason = history_task.get('failure_reason', 'Unknown reason') if history_task else 'Task not found in history'
                logger.warning(f"Task '{task_data['name']}' did not complete successfully. Final state: {reason}")
                return {"status": "failed", "reason": reason}

        finally:
            # Release task lock
            self.shared_memory.delete(f"task_in_progress:{task_data['name']}")

    def _execution_step(self):
        """A single tick of the execution loop."""
        printer.status("EXECUTION", "Execution loop", "info")

        context = self._gather_context()
        if self._is_task_complete(context):
            self.task_coordinator.complete_task(self.current_task['name'])
            return

        # Validate current task exists
        if not self.current_task:
            logger.error("Execution step aborted: No active task")
            return
            
        # Create checkpoint before action execution
        checkpoint_id = self.recovery.create_recovery_checkpoint(
            f"pre_{self.current_task['name']}",
            self.state
        )
        logger.debug(f"Created pre-action checkpoint: {checkpoint_id}")

        try:
            action_name = selected_action_dict.get("name", "unknown")
            # Generate potential actions for the selector
            potential_actions = [
                {"name": str(name), "priority": cls.priority, "preconditions": cls.preconditions}
                for name, cls in self.action_class_registry.items()
            ]

            # Validate action before execution
            is_valid, report = self.validator.validate_plan(
                [Task(name=action_name)],
                context,
                mode="continuous",
                level="strict"
            )
            if not is_valid:
                raise ActionFailureError(
                    action_name,
                    f"Action validation failed: {report[0]['errors']}"
                )

            selected_action_dict = self.action_selector.select(potential_actions, context)
            action_name = selected_action_dict.get("name")
            action_class = self.action_class_registry.get(action_name)

            if not action_class:
                raise ActionFailureError(action_name, "Selected action is not registered in the agent.")

            # Instantiate and execute the chosen action
            action_instance = action_class(context=context)
            success = action_instance.execute()

            if not success:
                reason = action_instance.failure_reason or f"'{action_name}' execution returned false."
                raise ActionFailureError(action_name, reason)

            # Update agent's master state from the action's resulting context
            self._update_state_from_action(action_instance)

            # Update task progress (example: based on distance or sub-goals)
            progress = self._calculate_task_progress(context)
            self.task_coordinator.update_task_progress(self.current_task['name'], progress)
    
        except ActionInterruptionError as e:
            logger.warning(f"Action interrupted: {e}")
            self.task_coordinator.pause_task(self.current_task['name'])
            
        except (ActionFailureError, InvalidContextError) as e:
            logger.error(f"Action failed: {str(e)}")
            
            # Enhanced recovery with task validation
            if self.current_task:  # Ensure task still exists
                recovery_success, new_context = self.recovery.handle_failure(
                    action_name, e, context
                )
                
                if recovery_success:
                    self.state = {**self.state, **new_context}
                    self.shared_memory.set(f"agent_state:{self.name}", self.state)
                    logger.info("Retrying action after recovery")
                    return self._execution_step()
                else:
                    self.task_coordinator.fail_task(
                        self.current_task['name'],
                        f"Unrecoverable failure: {str(e)}"
                    )
                    
        except Exception as e:
            # Handle unexpected errors with task validation
            if self.current_task:
                error = ActionFailureError("unknown", f"Unexpected error: {str(e)}")
                recovery_success, new_context = self.recovery.handle_failure(
                    "unknown", error, context
                )
                
                if recovery_success:
                    self.state = {**self.state, **new_context}
                    self.shared_memory.set(f"agent_state:{self.name}", self.state)
                    return self._execution_step()
                else:
                    self.task_coordinator.fail_task(
                        self.current_task['name'],
                        f"Critical failure: {str(e)}"
                    )

    def _gather_context(self) -> Dict[str, Any]:
        """Merges agent state and task details into a comprehensive context object."""
        printer.status("EXECUTION", "Context gatherer", "info")

        context = copy.deepcopy(self.state)
        context['current_time'] = time.time()
        context['cancel_movement'] = False  # Ensure no false interruptions
        context['urgent_event'] = False
        
        if self.current_task:
            context['current_goal'] = self.current_task.get('goal_type')
            context['deadline'] = self.current_task.get('deadline')
            context['destination'] = self.current_task.get('destination')
            context['target_object'] = self.current_task.get('target_object')
            context['target_position'] = context['destination']
            
            # Context flags for action preconditions
            if context.get('destination'):
                context['has_destination'] = True
                
            if context.get('target_object'):
                # In a real scenario, this would come from a perception system
                context['object_detected'] = True 
                context['object_nearby'] = True # Assume for now

        # Add derived context
        if context.get('has_destination') and context.get('current_position'):
             pos, dest = context['current_position'], context['destination']
             context['destination_distance'] = ((pos[0]-dest[0])**2 + (pos[1]-dest[1])**2)**0.5

        if 'required_field' not in context:
            raise InvalidContextError("gather_context", ['required_field'])

        return context

    def _is_task_complete(self, context: Dict) -> bool:
        """Determines if the current task's goal has been achieved."""
        printer.status("EXECUTION", "Analyzing Task State", "info")

        if not self.current_task:
            return False

        # Example completion conditions based on task type
        goal = self.current_task.get("goal_type")
        if goal == "navigate" and context.get("at_destination"):
            return True
        if goal == "collect" and context.get("holding_object") and context.get("held_object") == self.current_task.get("target_object"):
            return True
        if goal == "deposit" and context.get("object_placed"):
            return True
        if goal == "rest" and context.get("has_rested"):
            return True

        return False

    def _update_state_from_action(self, action_instance: BaseAction):
        """Merges the mutated context from a successful action back into the agent's main state."""
        printer.status("EXECUTION", "Updating state...", "info")

        # The action instance's context was modified in place.
        # We only need to update the keys that are part of our master state.
        for key in self.state.keys():
            if key in action_instance.context:
                self.state[key] = action_instance.context[key]

        self.shared_memory.set(f"agent_state:{self.name}", self.state)
        logger.debug(f"State updated after action '{action_instance.name}'. New energy: {self.state['energy']:.2f}")

    def _calculate_task_progress(self, context: Dict) -> float:
        """Estimates task progress (0.0 to 1.0) based on context."""
        printer.status("EXECUTION", "Task progression...", "info")

        if not self.current_task or not self._is_task_complete(context):
            # Example progress calculation for a navigation task
            if self.current_task.get("goal_type") == "navigate" and context.get('destination_distance') is not None:
                start_dist = self.current_task.get('initial_distance', context['destination_distance'] + 1)
                return 1.0 - (context['destination_distance'] / start_dist)
            return self.current_task.get('progress', 0.0) if self.current_task else 0.0
        return 1.0
    
    def dispatch_task(self, task_type: str, task_data: dict):
        task_id = f"{task_type}_{task_data.get('order_id')}"
        task = {
            "task_id": task_id,
            "type": task_type,
            "state": {
                "current_position": task_data.get("pickup_location"),
                "status": "DISPATCHED"
            },
            "estimated_completion": datetime.utcnow() + timedelta(minutes=30),
            "progress_milestones": ["task_dispatched"]
        }
        self.active_tasks[task_id] = task
        return task

    def get_active_task(self, task_id: str):
        """
        Retrieve active task by ID (e.g. delivery_1234)
        Can be used across apps without assuming food delivery logic
        """
        return self.active_tasks.get(task_id, None)

    def alternative_execute(self, task_data, original_error=None):
        """Fallback logic: try to execute an 'idle' action to recover or wait."""
        printer.status("EXECUTION", "Alternative execution...", "info")

        logger.warning(f"Primary execution failed. Attempting alternative: resting via IdleAction.")
        try:
            # Try to reload state from shared memory
            recovered_state = self.shared_memory.get(f"agent_state:{self.name}")
            if recovered_state:
                self.state = recovered_state
                logger.info("Recovered state from shared memory")
            idle_context = self._gather_context()
            idle_action = IdleAction(context=idle_context)
            success = idle_action.execute()
            if success:
                self._update_state_from_action(idle_action)
                return {"status": "recovered", "result": "Agent rested to recover from error."}
        except Exception as e:
            logger.error(f"Alternative execution (idle) also failed: {e}")

        return super().alternative_execute(task_data, original_error)
    
    def _get_agent_capabilities(self) -> Dict:
        return {
            "capabilities": ["navigation", "object_manipulation"],
            "current_load": 0.0,  # Initialize to 0
            "efficiency": self.config.get("efficiency", 1.0)
        }
    
    def sync_state(self, env_state: np.ndarray):
        """Sync agent state with environment state"""
        self.state.update({
            'current_position': (env_state[0], env_state[1]),
            'energy': env_state[8],
            'holding_object': bool(env_state[6]),
            'carrying_items': int(env_state[6])
        })
    
    def attach_adaptive(self, adaptive_agent):
        """Connect AdaptiveAgent to ExecutionAgent"""
        self.adaptive_agent = adaptive_agent
        logger.info("Adaptive agent attached to Execution agent")

    def get_validation_report(self):
        """Get recent validation statistics"""
        return self.validator.get_validation_stats()

    def get_recovery_report(self):
        """Get recovery system status"""
        return self.recovery.get_recovery_report()

if __name__ == "__main__":
    print("\n=== Running Execution Task Coordinator ===\n")
    printer.status("TEST", "Starting Task Coordinator tests", "info")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory

    memory = SharedMemory()
    factory = AgentFactory()
    execution_config = get_config_section('execution_agent')
    agent_type="execution"

    agent = ExecutionAgent(
        shared_memory=memory,
        agent_factory=factory,
        config=execution_config
    )
    print(agent)
    print("\n* * * * * Phase 2 * * * * *\n")
    data = {
        "name": "deliver_test_package",
        "goal_type": "navigate",  # could also be "collect", "deposit", or "rest"
        "destination": (5, 5),
        "timeout": 30,
        "requirements": [],
        "priority": 1
    }

    p_task = agent.perform_task(task_data=data)
    e_step = agent._execution_step()

    printer.pretty("Perform", p_task, "success" if p_task else "error")
    printer.pretty("Execute", e_step, "success" if e_step else "error")
    print("\n=== All tests completed successfully! ===\n")
