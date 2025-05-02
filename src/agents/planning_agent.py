"""
Planning Agent with Alternative Method Search Strategies
Implements grid search and Bayesian-inspired decomposition selection

References:
1. Nau, D., Au, T., Ilghami, O., et al. (2003). SHOP2: A HTN Planning System
2. Wilkins, D. (1988). SIPE-2: Systematic Initiative Planning Environment
3. Martelli, A., Montanari, U. (1973). Optimal Efficiency of AO* Algorithm
4. Bonet, B., Geffner, H. (2001). Heuristic Planning with HSP
5. Allen, J. (1983). Maintaining Knowledge about Temporal Intervals

Real-World Usage:
1. Robotics Task Planning: Domestic robots or warehouse robots need to break down complex goals like "prepare breakfast" or "pick 10 items" into primitive tasks: navigate, grip, open, etc.
2. Game AI / NPC Behavior: Strategy games (RTS) or open-world RPGs require AI to plan goals like "defend base" or "patrol zone" using adaptive strategies.
3. Workflow Automation in Enterprise Systems: A digital assistant receives a request like “schedule a customer onboarding session,” which it decomposes into tasks like:
    - Check calendar → Create Zoom link → Email invite → Log session
4. Military or Disaster Response Simulation: Simulated agents plan missions like “clear building,” which break into “scan,” “enter,” “check room,” etc.
5. Cognitive Assistants: Personalized assistants adapting through method statistics tracking
6. Healthcare Coordination: Adaptive treatment plan generation considering patient responses and resource availability
"""

import math
import time
import heapq
import random
import numpy as np

from enum import Enum
from typing import List, Dict, Optional, Callable, Tuple, Any, Set
from collections import defaultdict, deque

from src.agents.base_agent import BaseAgent
from src.agents.planning.planning_types import Task, TaskType, TaskStatus, WorldState, Any
from src.agents.planning.planning_metrics import PlanningMetrics
from src.agents.planning.decision_tree_heuristic import DecisionTreeHeuristic
from src.agents.planning.gradient_boosting_heuristic import GradientBoostingHeuristic
from src.agents.planning.task_scheduler import DeadlineAwareScheduler
from src.tuning.tuner import HyperparamTuner
from logs.logger import get_logger

logger = get_logger(__name__)


CostProfile = Tuple[float, float]
StateTuple = Tuple[Tuple[str, Any], ...]

class PlanningAgent(BaseAgent):
    """Enhanced planner with alternative search strategies"""
    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        base_config = {'defer_initialization': True}
        if config:
            base_config.update(config)
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=base_config
        )
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.decision_tree_heuristic = DecisionTreeHeuristic()
        self.gb_heuristic = GradientBoostingHeuristic()
        self.plan_history = deque(maxlen=1000)
        self.task_library: Dict[str, Task] = {}
        self.current_plan: List[Task] = []
        self.world_state: Dict[str, any] = {}
        self.execution_history = []
        self.method_stats = defaultdict(lambda: {'success': 0, 'total': 0})

        self.shared_memory = self.shared_memory[0] if isinstance(self.shared_memory, tuple) else self.shared_memory
        self.agent_factory = self.agent_factory[0] if isinstance(self.agent_factory, tuple) else self.agent_factory

        self.task_library: Dict[str, Task] = {} # Stores registered task templates
        self.current_plan: List[Task] = [] # The sequence of primitive tasks to execute
        self.world_state: Dict[str, Any] = {} # Current state of the world
        self.execution_history = deque(maxlen=100) # Keep track of recently executed tasks
        self.method_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'avg_cost': 0.0}) # For Bayesian method selection

        # Performance tracking
        self._planning_start_time: Optional[float] = None
        self._planning_end_time: Optional[float] = None

        config=config or {}
        self.scheduler = DeadlineAwareScheduler(risk_threshold=config.get('risk_threshold', 0.7))
        self.schedule_state = {'agent_loads': defaultdict(float), 'task_history': defaultdict(list)}

        logger.info(f"PlanningAgent initialized. World state: {self.world_state}")

    def log_plan_outcome(self, task, success):
        """Store plan features and outcome for training"""
        features = self.gb_heuristic.extract_features(
            task, 
            self.world_state,
            self.method_stats
        )
        self.plan_history.append((features, success))
        
    def _heuristic(self, task: Task) -> float:
        """Augmented heuristic combining learned models and original"""
        learned_prob = 0.7 * self.gb_heuristic.predict_success_prob(
            task, self.world_state, self.method_stats
        ) + 0.3 * self.decision_tree_heuristic.predict_success_prob(
            task, self.world_state, self.method_stats
        )
        
        original = len([t for t in self.task_library.values() 
                       if t.type == TaskType.ABSTRACT])
        
        return learned_prob * original
        
    def periodic_retraining(self):
        """Call this periodically (e.g., every 100 plans)"""
        if len(self.plan_history) > 100:
            X, y = zip(*self.plan_history)
            self.decision_tree_heuristic.train(np.array(X), np.array(y))
            self.gb_heuristic.train(np.array(X), np.array(y))
            self.plan_history.clear()

    def get_current_state_tuple(self) -> WorldState:
        """Returns an immutable representation of the world state for memoization/hashing."""
        # Convert complex objects in state to a hashable representation if needed
        items = []
        for k, v in self.world_state.items():
            try:
                # Attempt to make the value hashable, convert complex objects to str as fallback
                hashable_value = v if isinstance(v, (int, float, str, bool, tuple)) else str(v)
                items.append((k, hashable_value))
            except Exception:
                items.append((k, str(v))) # Fallback for unhashable items
        return tuple(sorted(items))

    def load_state_from_tuple(self, state_tuple: WorldState):
        """Restores the world state from an immutable tuple (used in backtracking)."""
        self.world_state = dict(state_tuple)
        logger.debug(f"World state restored to: {self.world_state}")

    def register_task(self, task: Task):
        """Register task with possible decomposition methods"""
        self.task_library[task.name] = task
        if task.name in self.task_library:
            logger.warning(f"Task '{task.name}' already registered. Overwriting.")
        self.task_library[task.name] = task
        logger.debug(f"Registered task: {task.name}")
        # Initialize stats for Bayesian method selection if it's an abstract task
        if task.type == TaskType.ABSTRACT:
            for i in range(len(task.methods)):
                key = (task.name, i)
                self.method_stats[key]

    def decompose_task(self, task_to_decompose: Task) -> Optional[List[Task]]:
        """Recursive decomposition with method selection tracking"""
        logger.debug(f"Decomposing task: {task_to_decompose.name}")

        memo_key = ((task_to_decompose.name, method_index_to_try), self.get_current_state_tuple())
        if memo_key in self.memo_table:
            logger.debug(f"Memo hit for {memo_key}")
            return self.memo_table[memo_key]

        # Base case: If the task is primitive, it's already decomposed.
        if task_to_decompose.type == TaskType.PRIMITIVE:
            if task_to_decompose.check_preconditions(self.world_state):
                if task_to_decompose.start_time is None:
                    task_to_decompose.start_time = time.time()
                return [task_to_decompose]
            else:
                logger.warning(f"Preconditions failed for primitive task: {task_to_decompose.name}")
                task_to_decompose.status = TaskStatus.FAILED
                return None

        # Retrieve the task template from the library
        library_task = self.task_library.get(task_to_decompose.name)
        if not library_task:
            logger.error(f"Task '{task_to_decompose.name}' not found in library.")
            task_to_decompose.status = TaskStatus.FAILED
            return None

        scores = []
        for i in range(len(library_task.methods)):
            task_to_decompose.selected_method = i
            score = self._heuristic(task_to_decompose)
            scores.append((i, score))
        
        # Select method with highest predicted success
        method_index_to_try = max(scores, key=lambda x: x[1])[0]
        task_to_decompose.selected_method = method_index_to_try

        if not (0 <= method_index_to_try < len(library_task.methods)):
             logger.error(f"Selected method index {method_index_to_try} out of range for task '{task_to_decompose.name}'")
             task_to_decompose.status = TaskStatus.FAILED
             return None

        # Get the list of subtasks for the selected method
        subtasks_template = library_task.get_subtasks(method_index_to_try)
        if not subtasks_template:
             logger.warning(f"Method {method_index_to_try} for task '{task_to_decompose.name}' has no subtasks or is invalid.")
             # Optionally, try another method here if implementing replanning/backtracking within decompose
             task_to_decompose.status = TaskStatus.FAILED
             return None

        logger.debug(f"Trying method {method_index_to_try} for task '{task_to_decompose.name}' with subtasks: {[st.name for st in subtasks_template]}")

        fully_decomposed_plan: List[Task] = []
        # Keep track of the world state simulation for this decomposition path
        simulated_world_state = self.world_state.copy()

        for subtask_template in subtasks_template:
             # Create a runtime instance of the subtask
             subtask_instance = subtask_template.copy()
             subtask_instance.parent = task_to_decompose # Link for hierarchy tracking

             # Apply effects of previously decomposed tasks in this branch to the simulated state
             # (This assumes effects are applied *before* checking next subtask's preconditions)
             # Note: This simple simulation might need refinement for complex interactions

             # Recursively decompose the subtask using the *simulated* state
             # We need a way to pass the simulated state down or restore it.
             # For simplicity here, we'll use the main world_state, but this
             # is a key area for refinement in more complex planners (backtracking needed).

             # Check subtask preconditions in the *current* (simulated) world state
             if not subtask_instance.check_preconditions(simulated_world_state):
                 logger.warning(f"Precondition failed for subtask '{subtask_instance.name}' during decomposition of '{task_to_decompose.name}'. Aborting method {method_index_to_try}.")
                 # Backtracking would happen here: try another method for task_to_decompose
                 return None # Abort this decomposition path

             decomposed_subplan = self.decompose_task(subtask_instance) # Recursive call

             if decomposed_subplan is None:
                 logger.warning(f"Failed to decompose subtask '{subtask_instance.name}'. Aborting method {method_index_to_try} for '{task_to_decompose.name}'.")
                 # Backtracking point: Try a different method for task_to_decompose if possible
                 return None # Indicate failure for this decomposition path

             fully_decomposed_plan.extend(decomposed_subplan)

             # Update the simulated world state by applying effects of the *primitive* tasks just added
             # This is crucial for the preconditions of subsequent subtasks in the *same* method.
             for primitive_task in decomposed_subplan:
                  if primitive_task.type == TaskType.PRIMITIVE:
                      primitive_task.apply_effects(simulated_world_state)

        # If loop completes, the decomposition for this method was successful
        logger.debug(f"Successfully decomposed '{task_to_decompose.name}' using method {method_index_to_try}.")
        return fully_decomposed_plan

    def _find_alternative_methods(self, task: Task) -> List[int]:
        """
        Finds alternative decomposition method indices for a failed abstract task.
        Uses a hybrid strategy: Bayesian optimization and grid search fallback.

        Args:
            task (Task): The abstract task that failed decomposition or execution.

        Returns:
            List[int]: A list of alternative method indices to try, ordered by
                       estimated likelihood of success.
        """
        library_task = self.task_library.get(task.name)
        if not library_task or task.type != TaskType.ABSTRACT or len(library_task.methods) <= 1:
            return [] # No alternatives if not abstract, not found, or only one method

        num_methods = len(library_task.methods)
        current_method_idx = task.selected_method

        # --- Bayesian Method Selection (using UCB1-like approach for exploration/exploitation) ---
        method_scores = []
        total_executions = sum(stats['total'] for stats in self.method_stats.values()) + 1 # Avoid div by zero

        for idx in range(num_methods):
            if idx == current_method_idx: # Skip the failed method initially
                 continue

            key = (task.name, idx)
            stats = self.method_stats[key]
            success_rate = (stats['success'] + 1) / (stats['total'] + 2) # Laplace smoothing

            # Exploration bonus (UCB1)
            exploration_term = math.sqrt(2 * math.log(total_executions) / (stats['total'] + 1)) if stats['total'] > 0 else float('inf')
            # Heuristic cost term (lower avg cost is better) - optional
            # cost_term = 1 / (stats['avg_cost'] + 0.1) if stats['avg_cost'] > 0 else 1.0

            score = success_rate + exploration_term # + cost_term * 0.1 # Add cost heuristic if desired
            method_scores.append((idx, score))

        # Sort alternatives by score (higher is better)
        bayesian_alternatives = sorted([idx for idx, score in method_scores], key=lambda idx: next(s for i, s in method_scores if i == idx), reverse=True)

        # --- Grid Search Fallback (systematically try next methods) ---
        grid_alternatives = []
        for i in range(1, num_methods):
            next_idx = (current_method_idx + i) % num_methods
            if next_idx not in bayesian_alternatives: # Avoid duplicates
                 grid_alternatives.append(next_idx)

        # Combine: Prioritize Bayesian suggestions, then systematic ones
        final_alternatives = bayesian_alternatives + grid_alternatives
        # Ensure the failed method isn't the first one tried again immediately
        if final_alternatives and final_alternatives[0] == current_method_idx:
             final_alternatives.pop(0)

        logger.debug(f"Alternatives for task '{task.name}' (failed method {current_method_idx}): {final_alternatives}")
        return final_alternatives

    def _update_method_stats(self, task: Task, success: bool, cost: float):
        """Updates statistics for the selected decomposition method."""
        if task.type != TaskType.ABSTRACT or task.parent is None:
             # Only update stats for methods chosen for a parent abstract task
             # Or potentially update based on overall plan success if task is the root goal.
             # Let's assume update happens when a parent's decomposition path is evaluated.
            return

        # Find the parent and the method index used to generate this task instance
        # This requires careful tracking during decomposition or execution feedback.
        # Assuming task.selected_method holds the relevant index for the *parent's* choice.
        parent_task = task.parent
        method_idx = parent_task.selected_method # Method index of the PARENT task that led to this `task`

        key = (parent_task.name, method_idx)
        stats = self.method_stats[key]
        stats['total'] += 1
        if success:
            stats['success'] += 1

        # Update average cost using Welford's online algorithm or simpler moving average
        # Simple moving average:
        current_total_cost = stats['avg_cost'] * (stats['total'] -1) # Get previous total cost
        new_avg_cost = (current_total_cost + cost) / stats['total']
        stats['avg_cost'] = new_avg_cost

        logger.debug(f"Updated stats for method {key}: Success={success}, Cost={cost:.2f}, New Avg Cost={new_avg_cost:.2f}")


    def replan(self, failed_task: Task, current_plan_segment: List[Task]) -> Optional[List[Task]]:
        """
        Attempts to replan starting from a failed task.

        Args:
            failed_task (Task): The task (primitive or abstract) that failed execution.
            current_plan_segment (List[Task]): The portion of the plan executed so far.

        Returns:
            Optional[List[Task]]: A new plan segment to replace the failed portion,
                                  or None if replanning fails.
        """
        logger.warning(f"Replanning triggered by failed task: {failed_task.name}")

        # 1. Identify the highest-level abstract task in the hierarchy that failed
        #    This requires navigating up the `parent` links from the `failed_task`.
        ancestor_to_replan = failed_task
        while ancestor_to_replan.parent is not None and ancestor_to_replan.parent.status != TaskStatus.FAILED:
             # Mark ancestors as potentially failed if a subtask failed
             if ancestor_to_replan.parent.status == TaskStatus.EXECUTING:
                  ancestor_to_replan.parent.status = TaskStatus.FAILED # Propagate failure up
             ancestor_to_replan = ancestor_to_replan.parent
        # `ancestor_to_replan` is now the highest task in the failed branch

        logger.info(f"Replanning from abstract task: {ancestor_to_replan.name}")

        # 2. Find alternative decomposition methods for this ancestor task
        alternative_method_indices = self._find_alternative_methods(ancestor_to_replan)

        if not alternative_method_indices:
            logger.error(f"No alternative methods found for task '{ancestor_to_replan.name}'. Replanning failed.")
            return None

        # 3. Try alternative methods one by one
        original_world_state_tuple = self.get_current_state_tuple() # Save state before trying alternatives

        for method_idx in alternative_method_indices:
            logger.info(f"Trying alternative method {method_idx} for task '{ancestor_to_replan.name}'")
            # Reset world state to before the failed branch started *executing*
            # This might require more sophisticated state logging or restoring from `current_plan_segment` start.
            # For simplicity, we restore to the state *before* attempting the replan.
            self.load_state_from_tuple(original_world_state_tuple)

            # Create a copy of the ancestor task and set the new method index
            task_copy = ancestor_to_replan.copy()
            task_copy.selected_method = method_idx
            task_copy.status = TaskStatus.PENDING # Reset status for decomposition attempt

            # Attempt to decompose using the alternative method
            new_plan_segment = self.decompose_task(task_copy)

            if new_plan_segment:
                 # 4. Validate the new plan segment (preconditions, potential conflicts)
                 if self._validate_plan(new_plan_segment):
                     logger.info(f"Replanning successful using method {method_idx} for task '{ancestor_to_replan.name}'.")
                     # We need to integrate this new segment into the overall plan.
                     # This function should return the *new segment* to be executed.
                     return new_plan_segment
                 else:
                     logger.warning(f"Validation failed for new plan segment from method {method_idx}.")
            else:
                 logger.warning(f"Decomposition failed for alternative method {method_idx}.")


        # 5. If all alternatives fail
        logger.error(f"All replanning attempts failed for task '{ancestor_to_replan.name}'.")
        self.load_state_from_tuple(original_world_state_tuple) # Restore original state
        return None

    def _validate_plan(self, plan_segment: List[Task]) -> bool:
        """
        Validates plan segment by fully simulating task effects and checking
        all preconditions at each step using Allen's temporal logic foundations.
        """
        if not plan_segment:
            return True  # Empty plan is trivially valid
    
        simulated_state = self.world_state.copy()
        logger.debug(f"Starting validation simulation with initial state: {simulated_state}")
    
        for idx, task in enumerate(plan_segment):
            # Check preconditions against simulated state
            if not self._check_preconditions(task, simulated_state):
                logger.warning(
                    f"Validation failed at step {idx+1}/{len(plan_segment)} "
                    f"({task.name}): Preconditions not met in simulated state"
                )
                return False
    
            try:
                # Apply task effects to simulated state
                for effect in task.effects:
                    effect(simulated_state)
                logger.debug(f"Applied effects of {task.name} | New state: {simulated_state}")
            except Exception as e:
                logger.error(
                    f"Effect application failed for {task.name} at step {idx+1}: {str(e)}",
                    exc_info=True
                )
                return False
    
            # Check temporal consistency with Allen's interval algebra
            if idx > 0:
                prev_task = plan_segment[idx-1]
                if not self._check_temporal_constraints(prev_task, task, simulated_state):
                    logger.warning(
                        f"Temporal constraint violation between {prev_task.name} "
                        f"and {task.name} at step {idx+1}"
                    )
                    return False
    
        logger.debug("Plan segment validation successful with final state: "
                    f"{simulated_state}")
        return True
    
    def _check_preconditions(self, task: Task, state: Dict[str, Any]) -> bool:
        """Formal verification of task preconditions against a state"""
        try:
            return all(precond(state) for precond in task.preconditions)
        except Exception as e:
            logger.error(
                f"Precondition check failed for {task.name}: {str(e)}",
                exc_info=True
            )
            return False
    
    def _check_temporal_constraints(self, 
                                   previous_task: Task,
                                   current_task: Task,
                                   state: Dict[str, Any]) -> bool:
        """
        Implements Allen's interval algebra checks for temporal consistency
        between consecutive tasks
        """
        # Get temporal constraints from task metadata
        constraints = getattr(current_task, 'temporal_constraints', [])
        
        # Check each constraint against simulated state
        for constraint in constraints:
            if not constraint(state):
                return False
        
        # Default temporal relationship checks
        if previous_task.end_time and current_task.start_time:
            if previous_task.end_time > current_task.start_time:
                logger.debug("Temporal overlap detected between "
                            f"{previous_task.name} and {current_task.name}")
                return False
        
        return True

    def generate_plan(self, goal_task: Task) -> Optional[List[Task]]:
        """Decomposes the goal task into a plan of primitive tasks."""
        raw_plan = super().generate_plan(goal_task)
        if not raw_plan:
            return None
        self._planning_start_time = time.time()
        plan = self.decompose_task(goal_task)
        self._planning_end_time = time.time()
        scheduled_tasks = self._convert_to_schedule_format(raw_plan)
        risk_assessor = self.shared_memory.get('risk_assessor')

        # Execute scheduling
        schedule = self.scheduler.schedule(
            tasks=scheduled_tasks,
            agents=self._get_available_agents(),
            risk_assessor=risk_assessor,
            state=self.world_state
        )

        if plan is None:
            goal_task.status = TaskStatus.FAILED
            logger.error("Failed to generate plan.")
            return None

        logger.info(f"Plan generated with {len(plan)} steps.")
        self.current_plan = plan
        return plan(schedule)

    def _convert_to_schedule_format(self, plan):
        """Map plan tasks to scheduler format"""
        return [{
            'id': task.name,
            'requirements': task.requirements,
            'deadline': task.deadline,
            'risk_score': task.risk_score,
            'dependencies': [t.name for t in task.dependencies]
        } for task in plan]

    def _convert_to_plan(self, schedule):
        """Convert scheduler output to executable plan"""
        return [self._create_task_from_assignment(a) for a in schedule.values()]

    def _get_available_agents(self):
        """Get agent capabilities from collaborative agent's registry"""
        return self.shared_memory.get('agent_registry', {})

    def execute_plan(self, plan: List[Task], goal_task: Optional[Task] = None) -> Dict[str, any]:
        """Executes a plan of primitive tasks. Tracks success, updates statistics, and handles failures."""
        self.current_plan = plan
        executed_successfully = True
        plan_idx = 0
    
        task_hierarchy = []
        current_parent = None
    
        while plan_idx < len(plan):
            task = plan[plan_idx]
    
            if task.parent != current_parent:
                if current_parent is not None:
                    self._update_task_success(current_parent, task_hierarchy)
                current_parent = task.parent
                task_hierarchy = []
    
            task.status = TaskStatus.EXECUTING
            self._execute_action(task)
            task_hierarchy.append(task)
            self.execution_history.append(task)
    
            if task.status == TaskStatus.FAILED:
                executed_successfully = False
                if goal_task:
                    goal_task.status = TaskStatus.FAILED
                break
    
            plan_idx += 1
    
        if current_parent is not None:
            self._update_task_success(current_parent, task_hierarchy)
    
        if executed_successfully and goal_task:
            goal_task.status = TaskStatus.SUCCESS
    
        return {
            "status": goal_task.status.name if goal_task else ("SUCCESS" if executed_successfully else "FAILED"),
            "world_state": self.world_state
        }

    def _grid_search_alternatives(self, task: Task) -> List[Task]:
        """Systematic exploration of decomposition methods"""
        library_task = self.task_library.get(task.name)
        if not library_task or task.type != TaskType.ABSTRACT:
            return []

        current_method = task.selected_method
        total_methods = len(library_task.methods)

        alternatives = []
        for method_idx in range(current_method + 1, total_methods):
            new_task = library_task.copy()
            new_task.selected_method = method_idx
            alternatives.append(new_task)

        return alternatives

    def _bayesian_alternatives(self, task: Task) -> List[Task]:
        """Bayesian optimization of decomposition methods"""
        library_task = self.task_library.get(task.name)
        if not library_task or task.type != TaskType.ABSTRACT:
            return []

        # Calculate success probabilities with Laplace smoothing
        method_scores = []
        for method_idx in range(len(library_task.methods)):
            key = (task.name, method_idx)
            stats = self.method_stats[key]
            success = stats['success'] + 1  # Laplace prior
            total = stats['total'] + 2
            method_scores.append((method_idx, success / total))

        # Sort by descending score, exclude current method
        sorted_methods = sorted(method_scores, key=lambda x: -x[1])
        current_method = task.selected_method

        alternatives = []
        for method_idx, score in sorted_methods:
            if method_idx != current_method:
                new_task = library_task.copy()
                new_task.selected_method = method_idx
                alternatives.append(new_task)

        return alternatives[:2]  # Return top 2 alternatives

    def _update_method_stats(self, task: Task, success: bool):
        """Update Bayesian statistics after execution"""
        if task.type != TaskType.ABSTRACT:
            return

        key = (task.name, task.selected_method)
        self.method_stats[key]['total'] += 1
        if success:
            self.method_stats[key]['success'] += 1

    def replan(self, failed_task: Task) -> Optional[List[Task]]:
        """Enhanced replanning with alternative method selection"""
        alternatives = self._find_alternative_methods(failed_task)
        if not alternatives:
            return None
        
        self._update_scheduler_state(failed_task)

        # Try alternatives in recommended order
        for alt_task in alternatives:
            new_plan = self.decompose_task(alt_task)
            if new_plan and self._validate_plan(new_plan):
                return new_plan
        return super().replan(failed_task)

    def _update_scheduler_state(self, task):
        """Maintain scheduler's view of world state"""
        self.schedule_state['agent_loads'][task.assigned_agent] -= task.cost
        self.schedule_state['task_history'][task.name].append({
            'status': 'failed',
            'timestamp': time.time()
        })

    def _validate_plan(self, plan: List[Task]) -> bool:
        """Validate that a plan is executable based on world state and task preconditions.
        This is a placeholder; you can implement more sophisticated logic here.
        """
        return True

    def _update_task_success(self, parent: Task, children: List[Task]):
        """Update method success statistics for abstract tasks"""
        if parent.type != TaskType.ABSTRACT:
            return

        success = all(t.status == TaskStatus.SUCCESS for t in children)
        self._update_method_stats(parent, success)

    def _execute_action(self, task: Task):
        """
        Execute a primitive task by checking preconditions and applying its effects.
        This is a basic implementation that you can expand based on your requirements.
        """
        # Check preconditions for the task
        for precondition in task.preconditions:
            if not precondition(self.world_state):
                print(f"Precondition failed for task: {task.name}")
                task.status = TaskStatus.FAILED
                return

        # Execute all effects associated with the task
        for effect in task.effects:
            effect(self.world_state)

        print(f"Executed task: {task.name}")
        task.status = TaskStatus.SUCCESS

    def evaluate_method_combination(self, param_dict: Dict[str, Any], fold: int = 0) -> float:
        for task_name, method_idx in param_dict.items():
            if task_name in self.task_library:
                self.task_library[task_name].selected_method = method_idx
    
        goal = self.shared_memory.get("planning_goal")
        if not goal:
            logger.error("No planning goal set.")
            return 0.0
    
        plan = self.generate_plan(goal)
        if not plan:
            return 0.0
    
        result = self.execute_plan(plan, goal_task=goal)
        metrics = PlanningMetrics.calculate_all_metrics(
            plan, self._planning_start_time, self._planning_end_time, goal.status
        )
    
        planning_time = metrics.get("planning_time", 1.0)
        plan_cost = metrics.get("total_cost", 1.0)
        goal_achieved = 1.0 if result.get("status") == "SUCCESS" else 0.0
    
        return (
            0.5 * goal_achieved +
            0.3 * (1.0 / (plan_cost + 1)) +
            0.2 * (1.0 / (planning_time + 0.01))
        )

    def optimize_methods(self, strategy: Optional[str] = None):
        """Choose tuning strategy based on task domain or past outcomes."""
        if not strategy:
            domain = self.shared_memory.get("task_domain") or "default"
            strategy = self._select_strategy_by_domain(domain)
            logger.info(f"Auto-selected tuning strategy: {strategy}")
    
        tuner = HyperparamTuner(
            config_path="configs/hyperparam_config.yaml",
            evaluation_function=self.evaluate_method_combination,
            strategy=strategy,
            n_calls=25,
            n_random_starts=5
        )

        best_params = tuner.run_tuning_pipeline()

        # Update selected methods with best parameters
        for task_name, method_idx in best_params.items():
            if task_name in self.task_library:
                self.task_library[task_name].selected_method = method_idx

        logger.info(f"Updated task methods using {strategy} search: {best_params}")

    # --- Placeholder for inherited/overridden methods from academic planners ---
    # Keep the specific planner implementations (HTN, A*, etc.) as separate classes
    # inheriting from PlanningAgent if they introduce significantly different algorithms
    # or state management. If they primarily override `decompose_task` or `replan`
    # with specific strategies, they can potentially stay as methods or inner classes
    # if the core PlanningAgent structure remains the same.

class HTNPlanner(PlanningAgent):

    StateTuple = Tuple[Tuple[str, Any], ...]

    """Implements Algorithm 1 from Nau et al. (JAIR 2003)"""
    def _ordered_decomposition(self, task: Task) -> Optional[List[Task]]:
        # Tuple-based state representation
        StateTuple = Tuple[Tuple[str, Any], ...]
        
        decomposition_stack: List[Tuple[Task, int, StateTuple]] = [
            (task, 0, self._freeze_state())
        ]
        current_plan = []
        backtrack_points = []

        while decomposition_stack:
            current_task, method_step, state = decomposition_stack.pop()
            
            if method_step >= len(current_task.methods[current_task.selected_method]):
                if backtrack_points:
                    # Backtrack to last decision point
                    current_plan, state = backtrack_points.pop()
                continue
                
            next_subtask = current_task.methods[current_task.selected_method][method_step]
            new_state = self._apply_effects(state, next_subtask)
            
            if not self._check_preconditions(state, next_subtask):
                continue
                
            if next_subtask.type == TaskType.ABSTRACT:
                # Record backtrack point (plan, state, method_step)
                backtrack_points.append((
                    current_plan.copy(),
                    state,
                    method_step + 1
                ))
                decomposition_stack.append((
                    next_subtask,
                    0,
                    new_state
                ))
            else:
                current_plan.append(next_subtask)
                
        return current_plan

    def _freeze_state(self) -> Tuple[Tuple[str, Any], ...]:
        """Immutable state representation for academic planning"""
        return tuple(sorted(self.world_state.items()))

    def _apply_effects(self, state: StateTuple, task: Task) -> StateTuple:
        """STRIPS-style effect application (Fikes & Nilsson 1971)"""
        state_dict = dict(state)
        for effect in task.effects:
            effect(state_dict)
        return tuple(sorted(state_dict.items()))

    def _partial_order_planning(self):
        """
        Implements partial-order planning based on:
        'SIPE: A Unified Theory of Planning' (Wilkins, 1988)
        """
        # Temporal constraint network using Allen's interval algebra
        temporal_network = {
            'relations': defaultdict(set),
            'intervals': {}
        }
        
        # Plan steps with causal links
        plan_steps = []
        open_conditions = []
        ordering_constraints = []
        
        # Initialize with start and goal
        start = Task("start", TaskType.PRIMITIVE)
        goal = self.current_goal.copy()
        plan_steps.extend([start, goal])
        temporal_network['intervals'] = {
            start: (0, 0),
            goal: (float('inf'), float('inf'))
        }
        
        while open_conditions:
            # Select next open condition using LCF strategy
            condition = min(open_conditions, key=lambda c: c[2])  # [step, precondition, criticality]
            
            # Find candidate providers using knowledge base
            candidates = self._find_candidate_steps(condition[1])
            
            for candidate in candidates:
                # Add causal link and temporal constraints
                new_constraints = self._add_causal_link(
                    candidate, condition[0], temporal_network
                )
                
                if not self._detect_temporal_inconsistencies(temporal_network):
                    # Resolve threats using promotion/demotion
                    self._resolve_threats(plan_steps, temporal_network)
                    break
                else:
                    # Remove failed constraints
                    self._remove_constraints(new_constraints, temporal_network)
            
            # Update open conditions
            open_conditions = self._identify_new_conditions(plan_steps, temporal_network)

    def _detect_temporal_inconsistencies(self, network):
        """Implements path consistency algorithm from Allen's temporal logic"""
        # Use Floyd-Warshall adaptation for temporal networks
        for k in network['intervals']:
            for i in network['intervals']:
                for j in network['intervals']:
                    intersection = network['relations'][(i,k)] & network['relations'][(k,j)]
                    if not intersection:
                        return True
                    network['relations'][(i,j)] |= intersection
        return False

    def _thompson_sampling_alternatives(self, task: Task) -> List[Task]:
        """Thompson sampling for decomposition method selection (Chapelle & Li 2011)"""
        # Maintain beta distributions for each method
        method_probs = []
        for method_idx in range(len(task.methods)):
            key = (task.name, method_idx)
            alpha = self.method_stats[key]['success'] + 1
            beta = self.method_stats[key]['total'] - self.method_stats[key]['success'] + 1
            sample = random.betavariate(alpha, beta)
            method_probs.append((method_idx, sample))
        
        sorted_methods = sorted(method_probs, key=lambda x: -x[1])
        return self._create_alternatives(task, sorted_methods)

    def _validate_plan(self, plan: List[Task]) -> bool:
        """Full STRIPS-style validation (Fikes & Nilsson 1971)"""
        sim_state = self.world_state.copy()
        for task in plan:
            if not all(precond(sim_state) for precond in task.preconditions):
                return False
            for effect in task.effects:
                effect(sim_state)
        return True

class PartialOrderPlanner(PlanningAgent):
    """Implements Wilkins' temporal constraint management"""
    def __init__(self):
        super().__init__()
        self.temporal_constraints: Set[Tuple[Task, Task, str]] = set()  # (A,B,relation)
        self.causal_links: Set[Tuple[Task, Task, Callable]] = set()  # (producer, consumer, condition)

    def _add_temporal_constraint(self, constraint: Tuple[Task, Task, str]):
        """Allen's interval algebra relations (before/after/contains)"""
        valid_relations = {'before', 'after', 'contains', 'during', 'meets'}
        if constraint[2] not in valid_relations:
            raise ValueError(f"Invalid temporal relation: {constraint[2]}")
        self.temporal_constraints.add(constraint)

    def _resolve_threats(self):
        """Threat resolution via promotion/demotion (Wilkins 1988)"""
        for link in self.causal_links:
            producer, consumer, condition = link
            for task in self.current_plan:
                if task.effects and any(not condition(eff) for eff in task.effects):
                    # Add ordering constraint: task < producer or task > consumer
                    if random.choice([True, False]):
                        self._add_temporal_constraint((task, producer, 'before'))
                    else:
                        self._add_temporal_constraint((consumer, task, 'before'))

class AStarPlanner(PlanningAgent):
    """Implements AO* cost propagation (Martelli & Montanari 1973)"""
    def _optimize_plan(self, plan: List[Task]) -> List[Task]:
        # Tuple-based cost representation (current, heuristic)
        for task in plan:
            if task.type == TaskType.ABSTRACT:
                and_or_graph[task] = {
                    'methods': [
                        (method, sum(self._task_cost(t) for t in method))
                        for method in task.methods
                    ],
                    'best_cost': (float('inf'), float('inf'))
                }
                
        and_or_graph = {
            task: {
                'methods': [
                    (method, sum(self._task_cost(t) for t in method))
                    for method in task.methods
                ],
                'best_cost': (float('inf'), float('inf'))
            }
            for task in plan if task.type == TaskType.ABSTRACT
        }
        
        # Initialize with primitive costs
        for task in plan:
            if task.type == TaskType.PRIMITIVE:
                and_or_graph[task] = {'best_cost': (1.0, 0.0)}  # (execution_cost, 0 heuristic)

        # Cost propagation from leaves to root
        changed = True
        while changed:
            changed = False
            for task in reversed(plan):
                if task.type != TaskType.ABSTRACT:
                    continue
                
                # Find minimal cost method
                min_method_cost = min(
                    (cost for _, cost in and_or_graph[task]['methods']),
                    default=(float('inf'), float('inf'))
                )
                
                # Update if better than current
                if min_method_cost < and_or_graph[task]['best_cost']:
                    and_or_graph[task]['best_cost'] = min_method_cost
                    changed = True

        return self._extract_optimal_plan(and_or_graph)

    def _task_cost(self, task: Task) -> CostProfile:
        """Academic cost model from HSP (Bonet & Geffner 2001)"""
        base = len(self.decompose_task(task))
        heuristic = self._hsp_heuristic(task)
        return (base, base + heuristic)

    def _heuristic(self, current_state: WorldState, goal_state: WorldState) -> float:
        return sum(1 for k, v in dict(goal_state).items() if dict(current_state).get(k) != v)

    def execute_plan(self, goal_task: Task) -> Dict[str, Any]:
        goal_state = goal_task.goal_state if hasattr(goal_task, "goal_state") else ()
        open_set = []
        start_state = self.get_current_state_tuple()
        heapq.heappush(open_set, (0, [], start_state))

        visited = set()

        while open_set:
            cost, path, state = heapq.heappop(open_set)
            if state in visited:
                continue
            visited.add(state)

            self.load_state_from_tuple(state)

            if self._goal_satisfied(goal_state):
                return super().execute_plan(path, goal_task=None)

            for task_name in self.task_library:
                task = self.task_library[task_name].copy()
                if task.type != TaskType.PRIMITIVE:
                    continue
                if not task.check_preconditions(self.world_state):
                    continue
                task.apply_effects(self.world_state)
                new_state = self.get_current_state_tuple()
                heuristic = self._heuristic(new_state, goal_state)
                new_cost = cost + task.cost
                heapq.heappush(open_set, (new_cost + heuristic, path + [task], new_state))

        return {"status": "FAILURE", "world_state": self.world_state}

    def _goal_satisfied(self, goal_state: WorldState) -> bool:
        current = dict(self.get_current_state_tuple())
        goal = dict(goal_state)
        return all(current.get(k) == v for k, v in goal.items())

class ExplanatoryPlanner(PlanningAgent):
    def generate_explanation(self, plan: List[Task]) -> Dict:
        """Produces human-understandable plan rationale"""
        return {
            'goal_satisfaction': self._explain_goal_achievement(plan),
            'method_choices': self._explain_method_selections(plan),
            'failure_points': self._identify_risk_points(plan)
        }
    
    def _optimize_plan(self, plan: List[Task]) -> List[Task]:
        """
        Implements AO* algorithm with cost propagation from:
        'Optimal Efficiency of the AO* Algorithm' (Martelli & Montanari, 1973)
        """
        # Build AND-OR graph representation
        and_or_graph = self._build_and_or_graph(plan)

        # Initialize heuristic estimates
        for node in reversed(math.topological_order(and_or_graph)):
            if node.is_and_node:
                node.cost = sum(child.cost for child in node.children)
            else:
                node.cost = min(child.cost for child in node.children)

        # Priority queue based on f(n) = g(n) + h(n)
        frontier = math.PriorityQueue()
        frontier.put((self._heuristic(plan[0]), plan[0]))

        while not frontier.empty():
            current = frontier.get()[1]

            if current.is_primitive:
                continue

            # Expand best partial plan
            best_method = min(current.methods, key=lambda m: m.cost)

            if best_method.cost < current.cost:
                current.cost = best_method.cost
                # Propagate cost changes upwards
                for parent in current.parents:
                    new_cost = parent.recalculate_cost()
                    if new_cost < parent.cost:
                        frontier.put((new_cost + self._heuristic(parent.task)), parent)

        return self._extract_optimal_plan(and_or_graph)

    def _heuristic(self, task: Task) -> float:
        """Academic admissible heuristic (HSP-style)"""
        if task.type == TaskType.PRIMITIVE:
            return 0
        # Count of remaining abstract tasks (Bonet & Geffner, 2001)
        return len([t for t in self.task_library.values() if t.type == TaskType.ABSTRACT])

    def _build_and_or_graph(self, plan):
        """Construct AND-OR graph with cost annotations"""
        graph = math.ANDORGraph()
        current_level = {plan[0]: graph.add_node(plan[0], is_and=False)}

        while current_level:
            next_level = {}
            for task, node in current_level.items():
                if task.type == TaskType.ABSTRACT:
                    # AND nodes for decomposition methods
                    for method in task.methods:
                        method_node = graph.add_node(method, is_and=True)
                        graph.add_edge(node, method_node)
                        # OR nodes for subtasks
                        for subtask in method:
                            subtask_node = graph.add_node(subtask, is_and=False)
                            graph.add_edge(method_node, subtask_node)
                            next_level[subtask] = subtask_node
            current_level = next_level

        return graph

    def _memoize_decompositions(self):
        """Memoization cache for common decompositions (Markovitch & Scott 1988)"""
        self.decomposition_cache = {}

    def decompose_task(self, task: Task) -> Optional[List[Task]]:
        """Memoized version of decomposition"""
        cache_key = (task.name, task.selected_method, frozenset(self.world_state.items()))
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]

        result = super().decompose_task(task)
        self.decomposition_cache[cache_key] = result
        return result
