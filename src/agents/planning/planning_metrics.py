
import psutil

from typing import List, Dict, Any

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_types import Task, TaskType, TaskStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Metrics")
printer = PrettyPrinter

class PlanningMetrics(Task):
    """
    Calculates and tracks various metrics related to planning performance.
    Inspired by metrics used in the International Planning Competition (IPC).
    """
    def __init__(self):
        super().__init__(name="Metrics", status=TaskStatus.SUCCESS, cost=0.0)
        self.config = load_global_config()
        self.task_config = get_config_section('planning_metrics')
        self.enable_timing = self.task_config.get('enable_timing')
        self.default_task_cost = self.task_config.get('default_task_cost')
        self.default_task_duration = self.task_config.get('default_task_duration')
        self.use_length_fallback = self.task_config.get('use_length_fallback')
        self.metrics_weights = self.task_config.get('metrics_weights', {
            'success', 'cost', 'time'
        })
        self.agent={}

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
        
        if self.enable_timing and len(valid_tasks) != len(plan):
            logger.warning(f"Missing timing data for {len(plan)-len(valid_tasks)} tasks")

        if not valid_tasks:
            if self.use_length_fallback:
                return len(plan) * self.default_task_duration
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
        metrics = cls()
        config = metrics.config

        return {
            "plan_length": cls.plan_length(plan),
            "plan_makespan": metrics.plan_makespan(plan),
            "plan_cost": cls.plan_cost(plan),
            "planning_time": planning_end_time - planning_start_time,
            "success_rate": 1.0 if final_status == TaskStatus.SUCCESS else 0.0,
            "resource_usage": {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent
            },
            "efficiency_score": (
                metrics.metrics_weights.get("success", 1.0) * (1.0 if final_status == TaskStatus.SUCCESS else 0.0) +
                metrics.metrics_weights.get("cost", 1.0) * (1 / (cls.plan_cost(plan) + 1e-6)) +
                metrics.metrics_weights.get("time", 1.0) * (1 / (planning_end_time - planning_start_time + 1e-6))
            )
        }

if __name__ == "__main__":
    print("\n=== Running Planning Metrics Test ===\n")
    printer.status("Init", "Planning Metrics initialized", "success")

    tasks = [
        Task("Prim1", TaskType.PRIMITIVE, start_time=0, end_time=2),
        Task("Abstract1", TaskType.ABSTRACT),
        Task("Prim2", TaskType.PRIMITIVE, start_time=2, end_time=5)
    ]

    metrics = PlanningMetrics()
    print(f"\n  Plan length: {metrics.plan_length(tasks)}")  # Should be 2
    print(f"Makespan: {metrics.plan_makespan(tasks)}")   # Should be 5-0=5

    print("\n=== Kitchen Planning Metrics Demo ===\n")

    # Initialize metrics calculator
    name="KitchenMetrics"

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

    print("\n=== Successfully Ran Planning Metrics ===\n")
