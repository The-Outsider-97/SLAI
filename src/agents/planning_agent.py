""
"""
Enhanced Planning Agent with Alternative Method Search Strategies
Implements grid search and Bayesian-inspired decomposition selection
"""

from enum import Enum
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict
import random

class TaskStatus(Enum):
    PENDING = 0
    EXECUTING = 1
    SUCCESS = 2
    FAILED = 3

class TaskType(Enum):
    PRIMITIVE = 0
    ABSTRACT = 1

class Task:
    """Enhanced Task class with multiple decomposition methods"""
    def __init__(self, name: str, task_type: TaskType,
                 methods: List[List['Task']] = None,
                 preconditions: List[Callable] = None,
                 effects: List[Callable] = None):
        self.name = name
        self.type = task_type
        self.methods = methods or [[]]  # Multiple decomposition methods
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.status = TaskStatus.PENDING
        self.parent = None
        self.selected_method = 0  # Track decomposition method index

    def copy(self) -> 'Task':
        """Create a copy with shared decomposition templates"""
        return Task(
            name=self.name,
            task_type=self.type,
            methods=self.methods,
            preconditions=self.preconditions,
            effects=self.effects
        )

    def get_subtasks(self, method_index: int = 0) -> List['Task']:
        """Get subtasks for specific decomposition method"""
        if 0 <= method_index < len(self.methods):
            return [t.copy() for t in self.methods[method_index]]
        return []

    def __repr__(self):
        return f"Task({self.name}, {self.type}, method:{self.selected_method})"

class PlanningAgent:
    """Enhanced planner with alternative search strategies"""
    def __init__(self):
        self.task_library: Dict[str, Task] = {}
        self.current_plan: List[Task] = []
        self.world_state: Dict[str, any] = {}
        self.execution_history = []
        self.method_stats = defaultdict(lambda: {'success': 0, 'total': 0})

    def register_task(self, task: Task):
        """Register task with possible decomposition methods"""
        self.task_library[task.name] = task
        # Initialize Bayesian priors
        for i in range(len(task.methods)):
            key = (task.name, i)
            self.method_stats[key]  # Initialize defaultdict entry

    def decompose_task(self, task: Task) -> Optional[List[Task]]:
        """Recursive decomposition with method selection tracking"""
        if task.type == TaskType.PRIMITIVE:
            return [task.copy()]

        library_task = self.task_library.get(task.name)
        if not library_task:
            return None

        # Create plan-specific task instance
        plan_task = library_task.copy()
        plan_task.selected_method = task.selected_method

        # Get selected decomposition method
        subtasks = plan_task.get_subtasks(plan_task.selected_method)
        if not subtasks:
            return None

        decomposed = []
        for subtask in subtasks:
            result = self.decompose_task(subtask)
            if result is None:
                return None
            decomposed.extend(result)

        return decomposed

    def _find_alternative_methods(self, task: Task) -> List[Task]:
        """Find alternative decompositions using hybrid strategy"""
        # Grid search fallback
        grid_alternatives = self._grid_search_alternatives(task)

        # Bayesian optimization
        bayesian_alternatives = self._bayesian_alternatives(task)

        # Combine and deduplicate
        alternatives = []
        seen = set()
        for alt in grid_alternatives + bayesian_alternatives:
            key = (alt.name, alt.selected_method)
            if key not in seen:
                seen.add(key)
                alternatives.append(alt)
        return alternatives

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

        # Try alternatives in recommended order
        for alt_task in alternatives:
            new_plan = self.decompose_task(alt_task)
            if new_plan and self._validate_plan(new_plan):
                return new_plan
        return None

    def _validate_plan(self, plan: List[Task]) -> bool:
        """Validate that a plan is executable based on world state and task preconditions.
        This is a placeholder; you can implement more sophisticated logic here.
        """
        return True

    def execute_plan(self, goal) -> Dict[str, any]:
        self.current_plan = self.decompose_task(goal)
        task_hierarchy = []
        current_parent = None
        for task in self.current_plan:
            if task.parent != current_parent:
                if current_parent is not None:
                    self._update_task_success(current_parent, task_hierarchy)
                current_parent = task.parent
                task_hierarchy = []

            task.status = TaskStatus.EXECUTING
            self._execute_action(task)
            task_hierarchy.append(task)
            self.execution_history.append(task)

        if current_parent is not None:
            self._update_task_success(current_parent, task_hierarchy)

        return self.world_state

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

# Example usage with multiple decomposition methods
if __name__ == "__main__":
    # Primitive actions
    search = Task("search", TaskType.PRIMITIVE,
                  effects=[lambda s: s.update({'sources': []})])

    filter_ = Task("filter", TaskType.PRIMITIVE,
                   preconditions=[lambda s: 'sources' in s],
                   effects=[lambda s: s.update({'filtered': s['sources'][:2]})])

    summarize = Task("summarize", TaskType.PRIMITIVE,
                     preconditions=[lambda s: 'filtered' in s],
                     effects=[lambda s: s.update({'summary': 'Sample'})])

    # Abstract tasks with multiple decomposition methods
    research_method1 = [search, filter_]
    research_method2 = [search.copy(), Task("evaluate", TaskType.PRIMITIVE), filter_]

    research = Task("research", TaskType.ABSTRACT,
                    methods=[research_method1, research_method2])

    summarize_task = Task("summarize_topic", TaskType.ABSTRACT,
                           methods=[[research, summarize]])

    planner = PlanningAgent()
    planner.register_task(research)
    planner.register_task(summarize_task)

    # Initial plan generation
    initial_goal = summarize_task.copy()
    plan = planner.decompose_task(initial_goal)
    planner.current_plan = plan
    planner.execute_plan(initial_goal)
    print("Initial plan:", planner.world_state)

    # Simulate execution failure
    if planner.current_plan:
        planner.current_plan[0].status = TaskStatus.FAILED  # Simulate a failure (e.g., search failure)
        new_plan = planner.replan(planner.current_plan[0])
        print("Replan result:", new_plan)
