import time
import psutil
import statistics
from typing import List, Dict, Any, Optional

# Assuming Task is defined elsewhere (e.g., in planning_types.py)
# from .planning_types import Task, TaskStatus

# Placeholder for Task and TaskStatus if not imported
class TaskStatus:
    SUCCESS = 2

class Task:
    def __init__(self, name: str, status=TaskStatus.SUCCESS, cost: float = 1.0):
        self.name = name
        self.status = status
        # Add start_time and end_time if available in your Task implementation
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.cost = cost # Example cost attribute


class PlanningMetrics:
    """
    Calculates and tracks various metrics related to planning performance.
    Inspired by metrics used in the International Planning Competition (IPC).
    """

    @staticmethod
    def plan_length(plan: List[Task]) -> int:
        """
        Calculates the number of steps (primitive actions) in the plan.
        Metric: Plan Length (L)
        """
        return len(plan)

    @staticmethod
    def plan_makespan(plan: List[Task]) -> float:
        """
        Calculates the total time duration of the plan execution.
        Requires tasks to have 'start_time' and 'end_time' attributes.
        Metric: Makespan (M)
        """
        if not plan:
            return 0.0

        start_times = [t.start_time for t in plan if t.start_time is not None]
        end_times = [t.end_time for t in plan if t.end_time is not None]

        if not start_times or not end_times:
            # Fallback if timing info is missing, could estimate based on plan length
            print("Warning: Task start/end times missing for makespan calculation.")
            return float(len(plan)) # Basic estimate

        min_start = min(start_times)
        max_end = max(end_times)
        return max_end - min_start

    @staticmethod
    def plan_cost(plan: List[Task]) -> float:
        """
        Calculates the total cost of the plan, assuming each task has a 'cost'.
        Metric: Plan Cost (C)
        """
        return sum(getattr(task, 'cost', 1.0) for task in plan) # Default cost 1 if missing

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
                              final_plan_status: TaskStatus) -> Dict[str, Any]:
        """Calculates a dictionary of all standard planning metrics."""
        metrics = {
            "plan_length": cls.plan_length(plan),
            "plan_makespan": cls.plan_makespan(plan),
            "plan_cost": cls.plan_cost(plan),
            "goal_achievement_rate": cls.goal_achievement_rate(plan, final_plan_status),
            "planning_time_sec": round(cls.planning_time(planning_start_time, planning_end_time), 4),
            "cpu_usage_percent": cls.cpu_usage(),
            "memory_usage_mb": round(cls.memory_usage(), 2),
        }
        return metrics
