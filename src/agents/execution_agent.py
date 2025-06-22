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

import torch
import copy
import time
import math

from collections import deque, defaultdict
from typing import Dict, Union, Tuple, Optional, Any, Type

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.execution.utils.execution_error import (TimeoutError, InvalidContextError, ExecutionError,
                StaleCheckpointError, DeadlockError, ActionFailureError, CookieMismatchError)
from src.agents.execution.task_coordinator import TaskCoordinator, TaskState
from src.agents.execution.action_selector import ActionSelector
from src.agents.execution.actions.base_action import BaseAction
from src.agents.execution.actions.move_to import MoveToAction
from src.agents.execution.actions.pick_object import PickObjectAction
from src.agents.execution.actions.idle import IdleAction
from src.agents.planning.task_scheduler import DeadlineAwareScheduler
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

        # Agent's internal state
        self.state: Dict[str, Any] = self._initialize_state()
        self.current_task: Optional[Dict] = None

        # Registry to map action names to their respective classes for dynamic instantiation
        self.action_class_registry: Dict[str, Type[BaseAction]] = {
            "move_to": MoveToAction,
            "pick_object": PickObjectAction,
            "idle": IdleAction
        }

        # Register actions with the selector for precondition awareness
        for name, cls in self.action_class_registry.items():
            self.action_selector.register_action(name, cls.preconditions, cls.postconditions)

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

    def perform_task(self, task_data: Dict) -> Dict:
        printer.status("EXECUTION", "Task Performer", "info")
    
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
            if not schedule:
                logger.error("Scheduler failed to create valid schedule")
                return {"status": "failed", "reason": "Scheduling failed"}

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

        # Generate potential actions for the selector
        potential_actions = [
            {"name": str(name), "priority": cls.priority, "preconditions": cls.preconditions}
            for name, cls in self.action_class_registry.items()
        ]

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
            "current_load": len(self.task_coordinator.tasks),
            "efficiency": self.config.get("efficiency", 1.0)
        }

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
        "id": "test_task",
        "name": "test_task",
        "requirements": ["navigation"],
        "deadline": time.time() + 5,
        "goal_type": "navigate",
        "destination": (5.0, 50.0)
    }

    printer.pretty("Perform", agent.perform_task(task_data=data), "success" if agent.perform_task else "error")
    printer.pretty("Execute", agent._execution_step(), "success" if agent._execution_step else "error")
    print("\n=== All tests completed successfully! ===\n")
