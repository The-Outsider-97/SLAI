import heapq
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Callable

class PlanningAgent:
    def __init__(self):
        """General-purpose planning agent with learning and parallel task support."""
        self.methods = defaultdict(list)          # Task decomposition methods
        self.primitive_tasks = set()              # Executable tasks
        self.task_metadata = defaultdict(dict)    # Duration, resources, etc.
        self.plan_history = []                    # Past plans and outcomes
        self.concurrency_groups = defaultdict(int) # Parallel execution constraints
        self.task_dependencies = defaultdict(set) # Prerequisite relationships

    def register_task(
        self,
        task_name: str,
        is_primitive: bool = False,
        duration: float = 1.0,
        resources: Set[str] = None,
        concurrency_group: int = 0,
        success_prob: float = 1.0,
        preconditions: Optional[Callable] = None
    ):
        """
        Register a task with the agent.
        
        Args:
            task_name: Name of the task
            is_primitive: Whether the task is directly executable
            duration: Estimated time to complete
            resources: Required resources for execution
            concurrency_group: Tasks in same group can't run in parallel
            success_prob: Probability of successful execution
            preconditions: Conditions that must be true to attempt this task
        """
        if resources is None:
            resources = set()
            
        self.task_metadata[task_name] = {
            'duration': duration,
            'resources': resources,
            'success_prob': success_prob,
            'preconditions': preconditions or (lambda _: True)
        }
        
        self.concurrency_groups[task_name] = concurrency_group
        
        if is_primitive:
            self.primitive_tasks.add(task_name)

    def register_method(
        self,
        task_name: str,
        method: Callable,
        success_prob: float = 1.0,
        preconditions: Optional[Callable] = None
    ):
        """
        Register a decomposition method for a non-primitive task.
        
        Args:
            task_name: Task this method decomposes
            method: Function that returns subtasks (list or None)
            success_prob: Probability this decomposition will succeed
            preconditions: Conditions required for this method
        """
        self.methods[task_name].append({
            'function': method,
            'success_prob': success_prob,
            'preconditions': preconditions or (lambda _: True)
        })

    def add_dependency(self, task: str, depends_on: str):
        """Add prerequisite relationship between tasks."""
        self.task_dependencies[task].add(depends_on)

    def is_primitive(self, task: str) -> bool:
        """Check if a task is directly executable."""
        return task in self.primitive_tasks

    def decompose_task(
        self,
        task: str,
        state: Dict,
        depth: int = 0,
        max_depth: int = 10
    ) -> Optional[List[str]]:
        """
        Recursively decompose a task into primitive tasks.
        
        Args:
            task: Task to decompose
            state: Current environment state
            depth: Current recursion depth
            max_depth: Maximum allowed recursion depth
            
        Returns:
            List of primitive tasks or None if decomposition fails
        """
        if depth > max_depth:
            return None
            
        # Check if task is primitive
        if self.is_primitive(task):
            return [task]
            
        # Check preconditions
        if not self.task_metadata[task]['preconditions'](state):
            return None
            
        # Try all registered methods for this task
        best_plan = None
        best_score = -1
        
        for method in self.methods.get(task, []):
            # Check method-specific preconditions
            if not method['preconditions'](state):
                continue
                
            # Attempt decomposition
            subtasks = method['function'](state)
            if subtasks is None:
                continue
                
            # Recursively decompose subtasks
            valid = True
            plan = []
            
            for subtask in subtasks:
                subplan = self.decompose_task(subtask, state, depth+1, max_depth)
                if subplan is None:
                    valid = False
                    break
                plan.extend(subplan)
                
            if valid:
                # Score plan based on success probability and length
                score = method['success_prob'] * (1 / (len(plan) + 1))
                if score > best_score:
                    best_score = score
                    best_plan = plan
                    
        return best_plan

    def schedule_tasks(
        self,
        tasks: List[str],
        available_resources: Set[str],
        state: Dict,
        deadline: Optional[float] = None
    ) -> Dict:
        """
        Schedule tasks considering dependencies, resources, and time constraints.
        
        Args:
            tasks: List of primitive tasks to schedule
            available_resources: Set of available resource names
            state: Current environment state
            deadline: Optional time constraint
            
        Returns:
            Dictionary containing:
                - schedule: List of (start, end, task) tuples
                - metrics: Success probability, makespan, etc.
        """
        # Initialize scheduling structures
        schedule = []
        current_time = 0
        task_queue = []
        in_progress = {}  # concurrency_group: end_time
        resources_available = available_resources.copy()
        
        # Build dependency graph
        remaining_deps = defaultdict(set)
        for task in tasks:
            remaining_deps[task] = self.task_dependencies[task].intersection(tasks)
            
        # Initialize queue with tasks that have no dependencies
        ready_tasks = [t for t in tasks if not remaining_deps[t]]
        for task in ready_tasks:
            heapq.heappush(
                task_queue,
                (self._get_task_priority(task), current_time, task)
            )
            
        while task_queue:
            priority, suggested_time, task = heapq.heappop(task_queue)
            
            # Get task metadata
            meta = self.task_metadata[task]
            duration = meta['duration']
            resources = meta['resources']
            concurrency_group = self.concurrency_groups[task]
            
            # Determine actual start time considering constraints
            start_time = max(
                suggested_time,
                in_progress.get(concurrency_group, 0)
            )
            
            # Check resource availability
            if not resources.issubset(resources_available):
                # Reschedule for later
                heapq.heappush(
                    task_queue,
                    (priority, start_time + 0.1, task)
                )
                continue
                
            # Schedule the task
            end_time = start_time + duration
            schedule.append((start_time, end_time, task))
            
            # Update resource tracking
            resources_available -= resources
            in_progress[concurrency_group] = end_time
            
            # Mark task as completed and check dependents
            for dependent in self._get_dependents(task, tasks):
                remaining_deps[dependent].discard(task)
                if not remaining_deps[dependent]:
                    heapq.heappush(
                        task_queue,
                        (self._get_task_priority(dependent), end_time, dependent)
                    )
                    
            # Return resources when task completes
            def callback(t=task, r=resources):
                resources_available.update(r)
            self._schedule_callback(end_time, callback)
            
        # Calculate metrics
        metrics = self._calculate_metrics(schedule, deadline)
        
        return {
            'schedule': sorted(schedule, key=lambda x: x[0]),
            'metrics': metrics
        }

    def _get_task_priority(self, task: str) -> float:
        """Calculate priority for task scheduling (lower is higher priority)."""
        meta = self.task_metadata[task]
        return (1 - meta['success_prob']) * meta['duration']

    def _get_dependents(self, task: str, all_tasks: List[str]) -> List[str]:
        """Get tasks that depend on the given task."""
        return [t for t in all_tasks if task in self.task_dependencies[t]]

    def _schedule_callback(self, time: float, callback: Callable):
        """Schedule a callback for when a task completes (simulated)."""
        # In a real implementation, this would use an event loop
        pass

    def _calculate_metrics(self, schedule: List[Tuple], deadline: Optional[float]) -> Dict:
        """Calculate various plan metrics."""
        if not schedule:
            return {}
            
        makespan = max(end for _, end, _ in schedule)
        success_prob = 1.0
        
        for _, _, task in schedule:
            success_prob *= self.task_metadata[task]['success_prob']
            
        meets_deadline = (deadline is None) or (makespan <= deadline)
        
        return {
            'makespan': makespan,
            'success_probability': success_prob,
            'meets_deadline': meets_deadline,
            'task_count': len(schedule)
        }

    def create_plan(
        self,
        goals: List[str],
        state: Dict,
        available_resources: Set[str],
        deadline: Optional[float] = None,
        max_attempts: int = 3
    ) -> Dict:
        """
        Create an optimized plan from high-level goals.
        
        Args:
            goals: High-level tasks to accomplish
            state: Current environment state
            available_resources: Available resources for execution
            deadline: Optional time constraint
            max_attempts: Maximum planning attempts
            
        Returns:
            Dictionary containing plan and metrics
        """
        best_plan = None
        best_metrics = {'success_probability': 0}
        
        for _ in range(max_attempts):
            # Decompose all goals into primitive tasks
            primitive_tasks = []
            for goal in goals:
                tasks = self.decompose_task(goal, state)
                if tasks is None:
                    continue
                primitive_tasks.extend(tasks)
                
            if not primitive_tasks:
                continue
                
            # Schedule the primitive tasks
            scheduled = self.schedule_tasks(
                primitive_tasks,
                available_resources,
                state,
                deadline
            )
            
            # Track best plan found
            if scheduled['metrics']['success_probability'] > best_metrics['success_probability']:
                best_plan = scheduled
                best_metrics = scheduled['metrics']
                
        if best_plan is None:
            return {'error': 'No valid plan found'}
            
        # Record this plan in history
        self.plan_history.append((goals, state, best_plan))
        return best_plan

    def learn_from_execution(
        self,
        plan: Dict,
        success: bool,
        actual_durations: Optional[Dict[str, float]] = None
    ):
        """
        Update agent's knowledge based on plan execution results.
        
        Args:
            plan: Executed plan (from create_plan)
            success: Whether execution succeeded
            actual_durations: Measured task durations for adjustment
        """
        if 'schedule' not in plan:
            return
            
        # Update success probabilities
        learning_rate = 0.1
        for _, _, task in plan['schedule']:
            old_prob = self.task_metadata[task]['success_prob']
            if success:
                new_prob = min(1.0, old_prob + learning_rate * (1 - old_prob))
            else:
                new_prob = max(0.1, old_prob - learning_rate * old_prob)
            self.task_metadata[task]['success_prob'] = new_prob
            
        # Update duration estimates
        if actual_durations:
            for task, duration in actual_durations.items():
                if task in self.task_metadata:
                    # Exponential moving average
                    old_duration = self.task_metadata[task]['duration']
                    self.task_metadata[task]['duration'] = (
                        0.9 * old_duration + 0.1 * duration
                    )

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = PlanningAgent()
    
    # Register resources
    resources = {'worker', 'truck', 'crane'}
    
    # Register tasks
    agent.register_task('dig', True, 2.0, {'worker'})
    agent.register_task('pour_concrete', True, 1.0, {'worker', 'truck'}, success_prob=0.9)
    agent.register_task('build_walls', True, 3.0, {'worker', 'crane'}, concurrency_group=1)
    agent.register_task('install_roof', True, 2.0, {'worker', 'crane'}, concurrency_group=1)
    
    # Register decomposition methods
    def build_house(state):
        return ['foundation', 'walls', 'roof']
        
    def foundation(state):
        if state.get('ground_clear'):
            return ['dig', 'pour_concrete']
        return None
    
    agent.register_method('build_house', build_house)
    agent.register_method('foundation', foundation, 
                        preconditions=lambda s: s.get('ground_clear'))
    
    # Set initial state
    state = {'ground_clear': True}
    
    # Create plan
    plan = agent.create_plan(
        goals=['build_house'],
        state=state,
        available_resources=resources,
        deadline=8.0
    )
    
    # Display results
    print("Scheduled Plan:")
    for start, end, task in plan['schedule']:
        print(f"{start:.1f}-{end:.1f}: {task}")
    
    print("\nPlan Metrics:")
    for k, v in plan['metrics'].items():
        print(f"{k}: {v}")
