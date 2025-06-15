__version__ = "1.9.0"

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

from email.policy import Policy
import math
import time
import heapq
import random

from typing import List, Dict, Optional, Callable, Tuple, Set
from collections import defaultdict, deque

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.planning.utils.planning_errors import (AdjustmentError, ReplanningError, TemporalViolation,
                                                SafetyMarginError, ResourceViolation, AcademicPlanningError)
from src.agents.base_agent import BaseAgent
from src.agents.planning.planning_metrics import PlanningMetrics
from src.agents.planning.planning_executor import PlanningExecutor
from src.agents.planning.heuristic_selector import HeuristicSelector
from src.agents.planning.task_scheduler import DeadlineAwareScheduler
from src.agents.planning.probabilistic_planner import ProbabilisticPlanner
from src.agents.planning.planning_types import Task, TaskType, TaskStatus, WorldState, Any
from src.agents.planning.safety_planning import SafetyPlanning, ResourceMonitor
from src.agents.perception.modules.transformer import ClassificationHead, RegressionHead, Seq2SeqHead
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Agent")
printer = PrettyPrinter

CostProfile = Tuple[float, float]
StateTuple = Tuple[Tuple[str, Any], ...]

class PlanningAgent(BaseAgent):
    """Enhanced planner with alternative search strategies"""
    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        self.config = load_global_config()
        self.planning_config = get_config_section('planning_agent')

        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        self.task_library: Dict[str, Task] = {} # Stores registered task templates
        self.current_plan: List[Task] = [] # The sequence of primitive tasks to execute
        self.world_state: Dict[str, Any] = {} # Current state of the world
        self.execution_history = deque(maxlen=100) # Keep track of recently executed tasks
        self.plan_history = deque(maxlen=1000)
        self.method_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'avg_cost': 0.0}) # For Bayesian method selection
        self.schedule_state = {'agent_loads': defaultdict(float), 'task_history': defaultdict(list)}

        # Performance tracking
        self._planning_start_time: Optional[float] = None
        self._planning_end_time: Optional[float] = None

        self.memo_table = {}
        self.scheduler = DeadlineAwareScheduler()
        self.metrics = PlanningMetrics()
        self.heuristic_selector = HeuristicSelector()
        self.safety_planner = SafetyPlanning()
        self.resource_monitor = ResourceMonitor()
        self.executor = PlanningExecutor()
        self.probabilistic_planner = ProbabilisticPlanner()
        self.safety_planner.resource_monitor = self.resource_monitor

        self.expected_state_projections = {}
        self.execution_interrupted = False
        self._node_cache = {}

        logger.info(f"PlanningAgent succesfully initialized")

        # Task head registry
        self.task_heads = {
            'classification': ClassificationHead,
            'regression': RegressionHead,
            'seq2seq': Seq2SeqHead
        }

    def configure_task_head(self, task_type: str, **kwargs):
        """Dynamically attach task heads based on current plan"""
        if task_type not in self.task_heads:
            raise ValueError(f"Unsupported task type: {task_type}")
            
        # Get transformer instance from text encoder
        transformer = self.shared_memory['text_encoder'].transformer
        return self.task_heads[task_type](transformer, **kwargs)

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

        # Initialize probabilistic actions if applicable
        if task.is_probabilistic:
            for action in task.probabilistic_actions:
                self.probabilistic_planner.register_action(action)

        self.task_library[task.name] = task
        logger.debug(f"Registered task: {task.name}")
        # Initialize stats for Bayesian method selection if it's an abstract task
        if task.type == TaskType.ABSTRACT:
            for i in range(len(task.methods)):
                key = (task.name, i)
                self.method_stats[key]

    def decompose_task(self, task_to_decompose: Task, current_state: Dict) -> Optional[List[Task]]:
        """Recursive decomposition with method selection tracking"""
        logger.debug(f"Decomposing task: {task_to_decompose.name}")

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
            # Use HeuristicSelector to get success probability
            prob = self.heuristic_selector.predict_success_prob(
                task={
                    "name": task_to_decompose.name,
                    "selected_method": i,
                    "priority": getattr(task_to_decompose, 'priority', 0.5),
                    "goal_state": getattr(task_to_decompose, 'goal_state', {}),
                    "parent": getattr(task_to_decompose, 'parent', None),
                    "creation_time": getattr(task_to_decompose, 'creation_time', None),
                    "deadline": getattr(task_to_decompose, 'deadline', None)
                },
                world_state=self.world_state,
                method_stats=self.method_stats,
                method_id=str(i)  # Convert method index to string
            )
            scores.append((i, prob))
        
        if not scores:
            logger.error(f"No methods available for task '{task_to_decompose.name}'.")
            task_to_decompose.status = TaskStatus.FAILED
            return None
        
        # Select method with highest predicted success
        method_index_to_try = max(scores, key=lambda x: x[1])[0]
        task_to_decompose.selected_method = method_index_to_try

        memo_key = ((task_to_decompose.name, method_index_to_try), self.get_current_state_tuple())
        if memo_key in self.memo_table:
            logger.debug(f"Memo hit for {memo_key}")
            return self.memo_table[memo_key]

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

             # Check subtask preconditions in the *current* (simulated) world state
             if not subtask_instance.check_preconditions(simulated_world_state):
                 logger.warning(f"Precondition failed for subtask '{subtask_instance.name}' during decomposition of '{task_to_decompose.name}'. Aborting method {method_index_to_try}.")
                 # Backtracking would happen here: try another method for task_to_decompose
                 return None # Abort this decomposition path

             decomposed_subplan = self.decompose_task(subtask_instance, simulated_world_state)

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
            return []
            
        # Use HeuristicSelector to get best methods
        candidate_methods = [str(i) for i in range(len(library_task.methods))]
        best_method, _ = self.heuristic_selector.select_best_method(
            task={
                "name": task.name,
                "priority": getattr(task, 'priority', 0.5),
                "goal_state": getattr(task, 'goal_state', {}),
                "creation_time": getattr(task, 'creation_time', None),
                "deadline": getattr(task, 'deadline', None)
            },
            world_state=self.world_state,
            candidate_methods=candidate_methods,
            method_stats=self.method_stats
        )
        
        # Convert string method IDs back to integers
        return [int(method_id) for method_id in candidate_methods 
                if method_id != str(task.selected_method)]

    def _update_method_stats(self, task: Task, success: bool, cost: float):
        """Updates statistics for the selected decomposition method."""
        if task.type != TaskType.ABSTRACT or task.parent is None:
             # Only update stats for methods chosen for a parent abstract task
             # Or potentially update based on overall plan success if task is the root goal.
            return

        # Find the parent and the method index used to generate this task instance
        # This requires careful tracking during decomposition or execution feedback.
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

    def replan(self, failed_task: Task) -> Optional[List[Task]]:
        """Enhanced replanning with alternative method selection"""
        logger.warning(f"Replanning triggered by failed task: {failed_task.name}")
        
        # Find alternative methods
        alternative_method_indices = self._find_alternative_methods(failed_task)
        if not alternative_method_indices:
            logger.error("No alternative methods found")
            return None
        
        # Update scheduler state with failure information
        self._update_scheduler_state(failed_task)
        
        # Try alternatives in recommended order
        for method_idx in alternative_method_indices:
            task_copy = failed_task.copy()
            task_copy.selected_method = method_idx
            new_plan = self.decompose_task(task_copy)
            
            # Validate plan and check safety
            if new_plan and self._validate_plan(new_plan) and self.safety_planner.safety_check(new_plan):
                return new_plan
        
        logger.error("All replanning attempts failed")
        return None

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

    def copy(self):
        new_task = Task(name=self.name, type=self.type)
        # Copy other attributes like preconditions, effects, etc.
        new_task.preconditions = self.preconditions.copy()
        new_task.effects = self.effects.copy()
        return new_task

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

    def generate_plan(self, goal_task: Task) -> Optional[List[Task]]:
        self._planning_start_time = time.time()

        # Use probabilistic planner for probabilistic tasks
        if goal_task.is_probabilistic:
            task_data = {
                'initial_state': self.world_state,
                'goal_state': goal_task.goal_state,
                'success_threshold': goal_task.success_threshold
            }
            policy = self.probabilistic_planner.perform_task(task_data)
            if policy:
                return policy
            return None
        plan = self.decompose_task(goal_task, self.world_state)
        
        # Integrated safety check
        try:
            if not self.safety_planner.safety_check(plan):
                raise AcademicPlanningError("Plan failed initial safety validation")
        except (ResourceViolation, TemporalViolation, SafetyMarginError) as e:
            logger.error(f"Safety violation: {str(e)}")
            return self._handle_safety_violation(goal_task, e)
        
        # Schedule with resource awareness
        scheduled_tasks = self._convert_to_schedule_format(plan)
        schedule = self.scheduler.schedule(
            tasks=scheduled_tasks,
            agents=self._get_available_agents(),
            risk_assessor=self.shared_memory.get('risk_assessor'),
            state=self.world_state
        )
        
        self._planning_end_time = time.time()
        self.current_plan = self._convert_to_plan(schedule)
        
        # Track planning metrics
        self.metrics.track_plan_start(self.current_plan)

        # Record planning metrics
        self.metrics.record_planning_metrics(
            plan_length=len(plan),
            planning_time=self._planning_end_time - self._planning_start_time,
            success_rate=1.0 if plan else 0.0
        )
        self.expected_state_projections = self._generate_state_projections(plan)
        return plan

    def _generate_state_projections(self, plan: List[Task]) -> Dict[str, Any]:
        """Generate expected state after each task execution"""
        projections = {}
        sim_state = self.world_state.copy()
        
        for task in plan:
            if task.type == TaskType.PRIMITIVE:
                # Apply task effects to simulated state
                for effect in task.effects:
                    effect(sim_state)
                projections[task.name] = sim_state.copy()
        
        return projections

    def _handle_safety_violation(self, task: Task, error: Exception) -> Optional[List[Task]]:
        """Handle safety violations during planning"""
        logger.warning(f"Safety violation detected: {str(error)}")
        candidates = self.safety_planner.dynamic_replanning_pipeline(task)
        
        for candidate in candidates:
            if self.safety_planner.safety_check(candidate):
                logger.info("Safety-compliant alternative found")
                return candidate
                
        logger.error("No safe alternatives available")
        return None

    def _execute_policy(self, policy: Policy, goal_task: Task) -> Dict[str, any]:
        """Execute probabilistic policy from PPDDL planner"""
        current_state = self.get_current_state_tuple()
        execution_path = []
        success = False
        
        for _ in range(100):  # Max 100 steps
            if current_state not in policy:
                logger.error(f"No policy defined for state: {current_state}")
                break
            
            action = policy[current_state]
            if not action.preconditions(dict(current_state)):
                logger.warning(f"Preconditions failed for {action.name}")
                break
            
            # Sample outcome
            r = random.random()
            cumulative_prob = 0
            for prob, effect in action.outcomes:
                cumulative_prob += prob
                if r <= cumulative_prob:
                    next_state = dict(current_state)
                    effect(next_state)  # Apply effect
                    next_state_tuple = tuple(sorted(next_state.items()))
                    
                    # Record execution step
                    execution_path.append({
                        'action': action.name,
                        'state': current_state,
                        'next_state': next_state_tuple,
                        'outcome_prob': prob
                    })
                    
                    # Update world state
                    self.world_state = next_state
                    current_state = next_state_tuple
                    break
            
            # Check goal satisfaction
            if all(self.world_state.get(k) == v 
                   for k, v in goal_task.goal_state.items()):
                success = True
                break
        
        return {
            'status': 'SUCCESS' if success else 'FAILURE',
            'execution_path': execution_path,
            'final_state': self.world_state
        }

    def execute_plan(self, plan: List[Task], goal_task: Optional[Task] = None) -> Dict[str, any]:
        execution_metrics = {
            'success_count': 0,
            'failure_count': 0,
            'total_cost': 0.0,
            'resource_usage': defaultdict(float)
        }


        # Start execution monitoring
        self.executor.start_monitoring(plan, self.expected_state_projections)
        self.execution_interrupted = False
        
        # Track plan start
        plan_meta = self.metrics.track_plan_start(plan)
        
        try:
            for task in plan:
                if self.execution_interrupted:
                    logger.warning("Execution interrupted by monitor")
                    break
                    
                try:
                    task.status = TaskStatus.EXECUTING    # Update task status
                    
                    # Pre-execution safety check
                    self.safety_planner._validate_equipment_constraints(task)
                    self.safety_planner._validate_temporal_constraints(task)
                    
                    # Execute task
                    start_time = time.time()
                    self._execute_action(task)
                    end_time = time.time()
                    
                    # Update execution times
                    task.start_time = start_time
                    task.end_time = end_time
                    task.status = TaskStatus.COMPLETED

                    self.safety_planner.update_allocations(task)    # Update resource allocations
                    
                    # Update metrics
                    execution_metrics['success_count'] += 1
                    execution_metrics['total_cost'] += task.cost
                    self._update_resource_metrics(execution_metrics, task)
                    
                    # Add to execution history
                    self.memory.base_state['execution_history'].append({
                        'task': task.name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'status': 'success',
                        'state_snapshot': self.world_state.copy()
                    })
                    
                except (SafetyMarginError, ResourceViolation, TemporalViolation) as e:
                    logger.warning(f"Execution safety violation: {str(e)}")
                    execution_metrics['failure_count'] += 1
                    task.status = TaskStatus.FAILED
                    
                    # Attempt recovery
                    recovery_plan = self.safety_planner.dynamic_replanning_pipeline(task)
                    if recovery_plan:
                        recovery_result = self.execute_plan(recovery_plan, goal_task)
                        if recovery_result.get('status') == 'SUCCESS':
                            execution_metrics['success_count'] += 1
                        else:
                            execution_metrics['failure_count'] += 1

                    self.memory.base_state['execution_history'].append({
                        'task': task.name,
                        'start_time': start_time if 'start_time' in locals() else None,
                        'end_time': end_time if 'end_time' in locals() else None,
                        'status': 'failed',
                        'error': str(e),
                        'state_snapshot': self.world_state.copy()
                    })
                    
        finally:
            self.executor.stop_monitoring()    # Stop monitoring regardless of outcome
            
        # Determine final status
        final_status = TaskStatus.SUCCESS if execution_metrics['failure_count'] == 0 else TaskStatus.FAILED

        self.metrics.track_plan_completion(plan_meta, final_status)    # Track plan completion
        
        # Record execution metrics
        self.metrics.record_execution_metrics(
            success_count=execution_metrics['success_count'],
            failure_count=execution_metrics['failure_count'],
            resource_usage=execution_metrics['resource_usage']
        )
        execution_summary = {
            "status": final_status.name,
            "world_state": self.world_state,
            "metrics": execution_metrics
        }
        self._log_performance(execution_summary)
        return execution_summary

    def replan_from_execution_failure(self, task: Optional[Task], reason: str):
        """Handle execution failures detected by monitor"""
        logger.warning(f"Replanning triggered due to: {reason}")
        self.execution_interrupted = True
        
        # Create recovery task based on failure type
        if reason == "precondition_violation" and task:
            recovery_plan = self._create_recovery_plan(task)
        else:
            # Full replan from current state
            recovery_plan = self.replan(self.current_goal)
        
        if recovery_plan:
            logger.info("Executing recovery plan")
            recovery_result = self.execute_plan(recovery_plan, self.current_goal)
            # Merge results with main execution
            # ... implementation specific ...
        else:
            logger.error("Recovery planning failed")

    def _create_recovery_plan(self, failed_task: Task) -> List[Task]:
        """Create targeted recovery plan for a specific task failure"""
        # 1. Attempt alternative methods
        alternatives = self._find_alternative_methods(failed_task)
        for method_idx in alternatives:
            new_task = failed_task.copy()
            new_task.selected_method = method_idx
            recovery_plan = self.decompose_task(new_task)
            if recovery_plan and self._validate_plan(recovery_plan):
                return recovery_plan
        
        # 2. Create precondition satisfaction subplan
        logger.info("Attempting to repair preconditions")
        repair_plan = self._create_precondition_repair_plan(failed_task)
        if repair_plan:
            repair_plan.append(failed_task.copy())
            return repair_plan
        
        return None

    def _create_precondition_repair_plan(self, task: Task) -> Optional[List[Task]]:
        """Generate plan to satisfy missing preconditions"""
        # ... implementation of precondition repair ...
        # This would use task's precondition gap analysis
        # and method from the task library to satisfy conditions
        
    def adjust_for_resource_violation(self, resource: str, usage: float, limit: float):
        """Adjust plan based on resource violation"""
        # 1. Check if we can redistribute tasks
        if self._redistribute_resource_load(resource):
            return
            
        # 2. Scale back resource-intensive tasks
        self._scale_back_resource_usage(resource, usage, limit)
        
        # 3. If still over, trigger replan
        if self._check_resource_overload(resource):
            self.replan_from_execution_failure(None, f"resource_violation_{resource}")

    def adjust_for_temporal_violation(self, task: Task, time_delta: float):
        """Adjust plan based on temporal violation"""
        # 1. Attempt to accelerate subsequent tasks
        if self._accelerate_subsequent_tasks(task, time_delta):
            return
            
        # 2. Reallocate time from less critical tasks
        if self._reallocate_time(task, time_delta):
            return
            
        # 3. If still behind, trigger replan
        self.replan_from_execution_failure(task, "temporal_violation")

    def _update_resource_metrics(self, metrics: dict, task: Task):
        """Track resource consumption metrics"""
        if hasattr(task, 'resource_requirements'):
            req = task.resource_requirements
            metrics['resource_usage']['gpu'] += req.gpu
            metrics['resource_usage']['ram'] += req.ram
            if req.specialized_hardware:
                for hw in req.specialized_hardware:
                    metrics['resource_usage'][hw] += 1

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

    def _update_scheduler_state(self, task):
        """Maintain scheduler's view of world state"""
        self.schedule_state['agent_loads'][task.assigned_agent] -= task.cost
        self.schedule_state['task_history'][task.name].append({
            'status': 'failed',
            'timestamp': time.time()
        })

    def _validate_plan(self, plan: List[Task]) -> bool:
        """Enhanced validation with safety constraints"""
        # Basic validation
        if not plan:
            return False
            
        # Safety validation
        try:
            self.safety_planner.safety_check(plan)
            return True
        except (ResourceViolation, TemporalViolation) as e:
            logger.warning(f"Plan validation failed: {str(e)}")
            return False

    def _update_task_success(self, parent: Task, children: List[Task]):
        """Update method success statistics for abstract tasks"""
        if parent.type != TaskType.ABSTRACT:
            return

        success = all(t.status == TaskStatus.SUCCESS for t in children)
        self._update_method_stats(parent, success)

    def _update_method_stats(self, task: Task, success: bool, cost: float):
        """Update Bayesian statistics after execution"""
        if task.type != TaskType.ABSTRACT or task.parent is None:
            return

        parent_task = task.parent
        method_idx = parent_task.selected_method
        
        # Use string keys for method_stats
        key = (parent_task.name, str(method_idx))
        stats = self.method_stats[key]
        self.method_stats[key]['total'] += 1
        if success:
            self.method_stats[key]['success'] += 1

    def _execute_action(self, task: Task):
        """Execute task with resource locking"""
        # Handle probabilistic actions differently
        if task.is_probabilistic:
            # Probabilistic effects handled in policy execution
            return

        # Acquire resources
        self.resource_monitor.acquire_resources(task.resource_requirements)
        
        try:
            # Execute task
            super()._execute_action(task)
        finally:
            # Release resources
            self.resource_monitor.release_resources(task.resource_requirements)

    def _log_performance(self, result: Dict[str, Any]):
        """Logs the result of a planning and execution cycle to shared memory for the MetaController."""
        # Use self.name to create a unique key for this agent's logs
        log_key = f"log:performance:{self.name}"
        
        log_entry = {
            'timestamp': time.time(),
            'status': result.get('status'),
            'metrics': {
                'total_cost': result.get('metrics', {}).get('total_cost', 0),
                'plan_length': len(self.current_plan),
                'success_count': result.get('metrics', {}).get('success_count', 0),
                'failure_count': result.get('metrics', {}).get('failure_count', 0),
            }
        }
        
        # Get the existing log (or a new deque) and append
        performance_logs = self.shared_memory.get(log_key, default=deque(maxlen=500))
        if not isinstance(performance_logs, deque):
             performance_logs = deque(list(performance_logs), maxlen=500)

        performance_logs.append(log_entry)
        self.shared_memory.set(log_key, performance_logs)
        logger.debug(f"Logged performance data to '{log_key}'.")

def run_planning_cycle(agent: PlanningAgent, goal_task: Task) -> Optional[Dict[str, Any]]:
    """Full planning-execution cycle with safety and metrics"""
    # Phase 1: Plan Generation
    plan = agent.generate_plan(goal_task)
    if not plan:
        logger.error("Plan generation failed")
        return None
    
    # Phase 2: Safety Validation
    try:
        if not agent.safety_planner.safety_check(plan):
            logger.error("Final plan failed safety validation")
            return None
    except (ResourceViolation, TemporalViolation, SafetyMarginError) as e:
        logger.error(f"Safety violation: {str(e)}")
        return None
    
    # Phase 3: Plan Execution
    result = agent.execute_plan(plan, goal_task)
    
    # Phase 4: Metrics Collection
    metrics = PlanningMetrics.calculate_all_metrics(
        plan=plan,
        planning_start_time=agent._planning_start_time,
        planning_end_time=agent._planning_end_time,
        final_status=goal_task.status
    )
    
    logger.info(f"Planning cycle completed: {metrics}")
    return {
        "execution_result": result,
        "metrics": metrics
    }

class HTNPlanner(PlanningAgent):

    StateTuple = Tuple[Tuple[str, Any], ...]

    """Implements Algorithm 1 from Nau et al. (JAIR 2003)"""
    def _ordered_decomposition(self, task: Task) -> Optional[List[Task]]:
        # Tuple-based state representation
        #StateTuple = Tuple[Tuple[str, Any], ...]
        
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

    def decompose_task(self, task_to_decompose: Task) -> Optional[List[Task]]:
        """Augmented decomposition with safety-aware distribution"""
        try:
            # Attempt distributed decomposition first
            return self.safety_planner.distributed_decomposition(task_to_decompose)
        except ResourceViolation:
            logger.info("Falling back to local decomposition")
            return self._local_decomposition(task_to_decompose)

    def _handle_safety_violation(self, task: Task, error: Exception) -> Optional[List[Task]]:
        """Integrated safety violation recovery"""
        logger.warning(f"Initial plan failed safety checks. Attempting repair...")
        
        # Get safety-aware repair candidates
        candidates = self.safety_planner.dynamic_replanning_pipeline(task)
        
        # Validate and select best candidate
        for candidate in candidates:
            if self.safety_planner.safety_check(candidate):
                logger.info("Found valid safety-compliant alternative plan")
                return candidate
                
        logger.error("No safe alternatives found")
        return None

    def interactive_adjustment(self, adjustment: dict):
        """Expose safety planner's adjustment interface"""
        self.safety_planner.interactive_adjustment_handler(adjustment)
        self.current_plan = self.safety_planner.current_plan

if __name__ == "__main__":
    print("\n=== Running AI Planning Agent Test ===\n")
    printer.status("Init", "Planning Agent initialized", "success")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory
    import datetime

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    planner = PlanningAgent(shared_memory, agent_factory, config=None)
    print(planner)
    print("\n* * * * * Phase 2 - Heuristic selector * * * * *\n")
    selector = HeuristicSelector()
    dummy_method_stats = {
        ("test_task", "0"): {"success": 3, "total": 5},
        ("test_task", "1"): {"success": 2, "total": 4},
    }

    print("\n=== Heuristic Selection Tests ===")

    # Test Case 1: RL (sequential task)
    rl_task = {
        "name": "test_task",
        "priority": 0.8,
        "goal_state": {"x": 1},
        "parent": {
            "name": "sub_task_3",
            "parent": {
                "name": "sub_task_2",
                "parent": {
                    "name": "sub_task_1",
                    "parent": None
                }
            }
        },
        "creation_time": datetime.datetime.now().isoformat(),
        "deadline": (datetime.datetime.now() + datetime.timedelta(hours=2)).isoformat(),
    }

    printer.status("1", "Testing RL Heuristic (sequential task):", "success")
    selector.predict_success_prob(rl_task, world_state={}, method_stats=dummy_method_stats, method_id="0")

    # Test Case 2: DT (deep hierarchy)
    dt_task = {
        "name": "test_task",
        "priority": 0.5,
        "goal_state": {"x": 1},
        "parent": {
            "parent": {
                "parent": {
                    "parent": {
                        "parent": {
                            "parent": {
                                "parent": {
                                    "parent": {
                                        "parent": {
                                            "parent": None  # Depth of 10
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "creation_time": datetime.datetime.now().isoformat(),
        "deadline": (datetime.datetime.now() + datetime.timedelta(hours=5)).isoformat(),
    }

    printer.status("2", "Testing Decision Tree Heuristic (deep task):", "success")
    selector.predict_success_prob(dt_task, world_state={}, method_stats=dummy_method_stats, method_id="1")

    # Test Case 3: GB (resource-constrained)
    gb_task = {
        "name": "test_task",
        "priority": 0.4,
        "goal_state": {"x": 1},
        "parent": None,
        "creation_time": datetime.datetime.now().isoformat(),
        "deadline": (datetime.datetime.now() + datetime.timedelta(hours=3)).isoformat(),
    }

    printer.status("3", "Testing Gradient Boosting Heuristic (low CPU):", "success")
    selector.predict_success_prob(
        gb_task,
        world_state={"cpu_available": 0.2, "memory_available": 0.3},
        method_stats=dummy_method_stats,
        method_id="0"
    )
    print("\n=== Planning Agent Demo Completed ===\n")
