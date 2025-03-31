"""
SLAI Planning Agent - Hierarchical Task Network (HTN) Planner
Inspired by: Erol, K., Hendler, J., & Nau, D. S. (1994). HTN planning: Complexity and expressivity. AAAI
Implements core HTN concepts with extensions for dynamic replanning
"""

from enum import Enum
from typing import List, Dict, Optional, Callable

class TaskStatus(Enum):
    PENDING = 0
    EXECUTING = 1
    SUCCESS = 2
    FAILED = 3

class TaskType(Enum):
    PRIMITIVE = 0
    ABSTRACT = 1

class Task:
    """Represents a task in hierarchical decomposition"""
    def __init__(self, name: str, task_type: TaskType,
                 decomposition: List['Task'] = None,
                 preconditions: List[Callable] = None,
                 effects: List[Callable] = None):
        self.name = name
        self.type = task_type
        self.decomposition = decomposition or []
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.status = TaskStatus.PENDING
        self.parent = None

    def add_subtask(self, subtask: 'Task'):
        self.decomposition.append(subtask)
        subtask.parent = self

    def is_executable(self) -> bool:
        return self.type == TaskType.PRIMITIVE

    def __repr__(self):
        return f"Task({self.name}, {self.type})"

class PlanningAgent:
    """Implements HTN-based planning with task decomposition and monitoring"""
    def __init__(self):
        self.task_library: Dict[str, Task] = {}
        self.current_plan: List[Task] = []
        self.world_state: Dict[str, any] = {}
        self.execution_history = []

    def register_task(self, task: Task):
        """Add task to the planning library"""
        self.task_library[task.name] = task

    def decompose_task(self, task: Task) -> Optional[List[Task]]:
        """Recursive task decomposition using HTN methods"""
        if task.type == TaskType.PRIMITIVE:
            return [task]

        if task.name not in self.task_library:
            return None

        decomposed = []
        for subtask in self.task_library[task.name].decomposition:
            result = self.decompose_task(subtask)
            if result is None:
                return None
            decomposed.extend(result)
        
        return decomposed

    def generate_plan(self, goal: Task) -> Optional[List[Task]]:
        """Generate executable plan through hierarchical decomposition"""
        if not self._validate_preconditions(goal):
            return None

        self.current_plan = self.decompose_task(goal)
        return self.current_plan if self._validate_plan() else None

    def execute_plan(self) -> Dict[str, any]:
        """Execute generated plan with state management"""
        for task in self.current_plan:
            if task.status != TaskStatus.PENDING:
                continue

            task.status = TaskStatus.EXECUTING
            self._execute_action(task)
            self._update_world_state(task)
            self.execution_history.append(task)

        return self.world_state

    def _validate_preconditions(self, task: Task) -> bool:
        """Check task preconditions against current world state"""
        return all(cond(self.world_state) for cond in task.preconditions)

    def _validate_plan(self) -> bool:
        """Verify plan consistency with current state"""
        simulated_state = self.world_state.copy()
        
        for task in self.current_plan:
            if not self._validate_preconditions(task):
                return False
            for effect in task.effects:
                effect(simulated_state)
                
        return True

    def _execute_action(self, task: Task):
        """Execute primitive task and handle outcomes"""
        try:
            if self._validate_preconditions(task):
                for effect in task.effects:
                    effect(self.world_state)
                task.status = TaskStatus.SUCCESS
            else:
                task.status = TaskStatus.FAILED
        except Exception as e:
            print(f"Execution failed for {task.name}: {str(e)}")
            task.status = TaskStatus.FAILED

    def monitor_progress(self) -> Dict:
        """Track plan execution progress"""
        return {
            'completed': len([t for t in self.current_plan if t.status == TaskStatus.SUCCESS]),
            'total': len(self.current_plan),
            'failed': len([t for t in self.current_plan if t.status == TaskStatus.FAILED]),
            'current_state': self.world_state.copy()
        }

    def replan(self, failed_task: Task) -> Optional[List[Task]]:
        """Handle plan failures through dynamic replanning"""
        alternative_methods = self._find_alternative_methods(failed_task)
        if not alternative_methods:
            return None
            
        # Implementation would involve modifying task decomposition
        # and regenerating plan from failure point
        # (Extended in practice with dependency analysis)
        return self.generate_plan(failed_task.parent)

    def _find_alternative_methods(self, task: Task) -> List[Task]:
        """Find alternative decompositions for failed task"""
        # Placeholder for actual alternative search logic
        return []

# Example usage
if __name__ == "__main__":
    # Define primitive tasks
    search_action = Task("search", TaskType.PRIMITIVE,
                        effects=[lambda s: s.update({'sources': []})])
    
    filter_action = Task("filter", TaskType.PRIMITIVE,
                       preconditions=[lambda s: 'sources' in s],
                       effects=[lambda s: s.update({'filtered': s['sources'][:2]})])
    
    summarize_action = Task("summarize", TaskType.PRIMITIVE,
                          preconditions=[lambda s: 'filtered' in s],
                          effects=[lambda s: s.update({'summary': 'Sample summary'})])

    # Define abstract tasks
    research_task = Task("research", TaskType.ABSTRACT)
    research_task.add_subtask(search_action)
    research_task.add_subtask(filter_action)

    summary_task = Task("summarize_topic", TaskType.ABSTRACT)
    summary_task.add_subtask(research_task)
    summary_task.add_subtask(summarize_action)

    # Initialize and configure planner
    planner = PlanningAgent()
    planner.register_task(research_task)
    planner.register_task(summary_task)
    
    # Generate and execute plan
    plan = planner.generate_plan(summary_task)
    if plan:
        print("Generated plan:", plan)
        result_state = planner.execute_plan()
        print("Execution result:", result_state)
        print("Progress:", planner.monitor_progress())
    else:
        print("Planning failed")
