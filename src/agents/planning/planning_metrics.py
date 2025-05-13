import yaml
import psutil
import statistics

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Planning Metrics")

CONFIG_PATH = "src/agents/planning/configs/planning_config.yaml"

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: Union[str, Dict], config_file_path: str):
    if isinstance(section, dict):
        return dict_to_namespace(section)
    
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])


# === Added TaskType Enum ===
class TaskType(Enum):
    """Differentiates between primitive actions and abstract goals."""
    PRIMITIVE = 0
    ABSTRACT = 1

class TaskStatus(Enum):
    """Enhanced Task Status enumeration matching planning_types.py"""
    PENDING = 0
    EXECUTING = 1
    SUCCESS = 2
    FAILED = 3

class Task:
    """Expanded Task class with planning-related properties"""
    def __init__(self, 
                 name: str,
                 task_type: TaskType = TaskType.PRIMITIVE,
                 status: TaskStatus = TaskStatus.PENDING,
                 cost: float = 1.0,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 parent: Optional['Task'] = None,
                 methods: List[List['Task']] = None,
                 preconditions: List[Callable] = None,
                 effects: List[Callable] = None):
        self.name = name
        self.task_type = task_type
        self.status = status
        self.cost = cost
        self.start_time = start_time
        self.end_time = end_time
        self.parent = parent
        self.methods = methods or []
        self.preconditions = preconditions or []
        self.effects = effects or []

    def copy(self) -> 'Task':
        return Task(
            name=self.name,
            task_type=self.task_type,
            status=self.status,
            cost=self.cost,
            start_time=self.start_time,
            end_time=self.end_time,
            parent=self.parent,
            methods=self.methods.copy(),
            preconditions=self.preconditions.copy(),
            effects=self.effects.copy()
        )


class PlanningMetrics(Task):
    """
    Calculates and tracks various metrics related to planning performance.
    Inspired by metrics used in the International Planning Competition (IPC).
    """
    def __init__(self, name, agent=None,
                 config_section_name: str = "planning_metrics",
                 config_file_path: str = CONFIG_PATH,
                 status=TaskStatus.SUCCESS,
                 cost = 1):
        super().__init__(name, status, cost)
        self.agent=agent
        self.config = get_config_section(config_section_name, config_file_path)

    @staticmethod
    def plan_length(plan: List[Task]) -> int:
        """
        Calculates the number of steps (primitive actions) in the plan.
        Metric: Plan Length (L)
        """
        return sum(1 for task in plan if task.task_type == TaskType.PRIMITIVE)

    def plan_makespan(self, plan: List[Task]) -> float:
        """
        Calculates total execution duration using task timestamps
        Uses config settings for timing enforcement and fallbacks
        """
        if not plan:
            return 0.0

        valid_tasks = [t for t in plan if t.start_time and t.end_time]
        
        if self.config.enable_timing and len(valid_tasks) != len(plan):
            logger.warning(f"Missing timing data for {len(plan)-len(valid_tasks)} tasks")

        if not valid_tasks:
            if self.config.use_length_fallback:
                return len(plan) * self.config.default_task_duration
            return 0.0

        start_times = [t.start_time for t in valid_tasks]
        end_times = [t.end_time for t in valid_tasks]

        return max(end_times) - min(start_times)

    @staticmethod
    def plan_cost(plan: List[Task], default_cost: float = 1.0) -> float:
        """
        Calculates the total cost of the plan, assuming each task has a 'cost'.
        Metric: Plan Cost (C)
        """
        return sum(task.cost if hasattr(task, 'cost') else default_cost for task in plan)

    @staticmethod
    def goal_achievement_rate(plan: List[Task], final_status: TaskStatus) -> float:
        """
        Checks if the plan achieved the goal successfully.
        Metric: Goal Achievement (G) - Binary (1 for success, 0 for failure)
        """
        # Assumes the final status of the overall goal task is passed
        return 1.0 if final_status == TaskStatus.SUCCESS else 0.0

    @staticmethod
    def planning_time(start_time: float, end_time: float) -> float:
        """
        Calculates the time taken for the planning process itself.
        Metric: Planning Time (Tp)
        """
        return end_time - start_time

    @staticmethod
    def cpu_usage() -> float:
        """
        Gets the current CPU utilization percentage.
        Metric: CPU Usage (%)
        """
        return psutil.cpu_percent()

    @staticmethod
    def memory_usage() -> float:
        """
        Gets the current process memory usage in MB.
        Metric: Memory Usage (Mem)
        """
        return psutil.Process().memory_info().rss / (1024 * 1024)

    @classmethod
    def calculate_all_metrics(cls,
                             plan: List[Task],
                             planning_start_time: float,
                             planning_end_time: float,
                             final_status: TaskStatus) -> Dict[str, Any]:
        """Comprehensive metrics calculation with config integration"""
        metrics = cls(name="DefaultMetrics")
        config = metrics.config

        return {
            "plan_length": cls.plan_length(plan),
            "plan_makespan": metrics.plan_makespan(plan),
            "plan_cost": cls.plan_cost(plan, config.default_task_cost),
            "planning_time": planning_end_time - planning_start_time,
            "success_rate": 1.0 if final_status == TaskStatus.SUCCESS else 0.0,
            "resource_usage": {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent
            },
            "efficiency_score": (
                config.metrics_weights.success * (1.0 if final_status == TaskStatus.SUCCESS else 0.0) +
                config.metrics_weights.cost * (1 / (cls.plan_cost(plan) + 1e-6)) +
                config.metrics_weights.time * (1 / (planning_end_time - planning_start_time + 1e-6))
            )
        }

if __name__ == "__main__":
    print("")
    print("\n=== Running Planning Metrics ===")
    print("")
    from unittest.mock import Mock
    tasks = [
        Task("Prim1", TaskType.PRIMITIVE, start_time=0, end_time=2),
        Task("Abstract1", TaskType.ABSTRACT),
        Task("Prim2", TaskType.PRIMITIVE, start_time=2, end_time=5)
    ]
    mock_agent = Mock()
    mock_agent.shared_memory = {}
    
    metrics = PlanningMetrics(name=None, agent=mock_agent)
    print(f"\n  Plan length: {metrics.plan_length(tasks)}")  # Should be 2
    print(f"Makespan: {metrics.plan_makespan(tasks)}")   # Should be 5-0=5
    print("")
    print("\n=== Successfully Ran Planning Metrics ===\n")

if __name__ == "__main__":
    print("\n=== Kitchen Planning Metrics Demo ===")

    # Initialize metrics calculator
    metrics = PlanningMetrics(name="KitchenMetrics")

    # Create realistic cooking tasks with timing and cost data
    tasks = [
        Task("GatherIngredients", TaskType.PRIMITIVE,
             cost=0.5, start_time=0.0, end_time=2.5,
             status=TaskStatus.SUCCESS),
        
        Task("ChopVegetables", TaskType.PRIMITIVE,
             cost=1.2, start_time=2.5, end_time=5.0,
             status=TaskStatus.SUCCESS),
        
        Task("PreheatOven", TaskType.PRIMITIVE,
             cost=0.8, start_time=5.0, end_time=8.0,
             status=TaskStatus.SUCCESS),
        
        Task("CookMainDish", TaskType.ABSTRACT,
             cost=2.5, methods=[[
                 Task("PrepareProtein", TaskType.PRIMITIVE, cost=1.0),
                 Task("BakeDish", TaskType.PRIMITIVE, cost=1.5)
             ]]),
        
        Task("SetTable", TaskType.PRIMITIVE,
             cost=0.7, start_time=8.0, end_time=9.5,  # Missing end time
             status=TaskStatus.FAILED)
    ]

    # Create failed plan version
    failed_tasks = [t.copy() for t in tasks]
    for t in failed_tasks:
        if t.name == "CookMainDish":
            t.status = TaskStatus.FAILED

    # Test successful plan
    print("\n=== Testing Successful Meal Preparation ===")
    success_metrics = metrics.calculate_all_metrics(
        plan=[t for t in tasks if t.task_type == TaskType.PRIMITIVE],
        planning_start_time=0.0,
        planning_end_time=2.3,
        final_status=TaskStatus.SUCCESS
    )
    
    # Test failed plan
    print("\n=== Testing Failed Meal Preparation ===")
    failure_metrics = metrics.calculate_all_metrics(
        plan=[t for t in failed_tasks if t.task_type == TaskType.PRIMITIVE],
        planning_start_time=0.0,
        planning_end_time=1.8,
        final_status=TaskStatus.FAILED
    )

    # Test plan with missing timing data
    print("\n=== Testing Partial Timing Data ===")
    partial_tasks = [t.copy() for t in tasks[:3]]  # First 3 tasks only
    partial_metrics = metrics.calculate_all_metrics(
        plan=partial_tasks,
        planning_start_time=0.0,
        planning_end_time=1.2,
        final_status=TaskStatus.SUCCESS
    )

    # Display results
    def print_metrics(label, metrics):
        print(f"\n{label} Metrics:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk}: {sv}")
            else:
                print(f"  {k}: {v}")

    print_metrics("Successful Meal", success_metrics)
    print_metrics("Failed Meal", failure_metrics)
    print_metrics("Partial Preparation", partial_metrics)

    print("\n=== Real-World Metric Observations ===")
    print("1. Successful plan shows full makespan calculation")
    print("2. Failed plan shows 0 success rate but includes partial costs")
    print("3. Partial plan demonstrates timing fallback mechanism")
    print("4. Efficiency score varies with success/cost/time factors")
    print("5. Resource usage reflects actual system performance")

    print("\n=== Metric Demo Complete ===")
