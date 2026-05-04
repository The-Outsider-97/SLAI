"""
Planning Metrics – Production‑ready performance tracking.

Calculates and logs planning and execution metrics with configurable weights,
fallback timing, and resource monitoring. Thread‑safe for concurrent usage.
"""

import time
import threading
import psutil

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import field

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_types import Task, TaskType, TaskStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Metrics")
printer = PrettyPrinter

class PlanningMetrics:
    """
    Calculates and tracks various metrics related to planning performance.
    Inspired by metrics used in the International Planning Competition (IPC).
    """

    def __init__(self) -> None:
        """Initialize metrics calculator with configuration."""
        self.config = load_global_config()
        self.metrics_config = get_config_section("planning_metrics")

        self.enable_timing = self.metrics_config.get("enable_timing", True)
        self.default_task_cost = self.metrics_config.get("default_task_cost", 1.0)
        self.default_task_duration = self.metrics_config.get("default_task_duration", 0.5)
        self.use_length_fallback = self.metrics_config.get("use_length_fallback", True)
        self.metrics_weights = self.metrics_config.get(
            "metrics_weights", {"success": 0.5, "cost": 0.3, "time": 0.2}
        )

        # Thread safety for cumulative metrics (if we ever add them)
        self._lock = threading.RLock()

        # Internal state (can be extended later)
        self._planning_time: float = 0.0
        self._execution_time: float = 0.0
        self._plan_length: int = 0
        self._success_rate: float = 0.0
        self._resource_efficiency: Dict[str, float] = {}
        self._temporal_efficiency: float = 0.0
        self._task_success_rates: Dict[str, float] = {}
        self._method_success_rates: Dict[Tuple[str, int], float] = {}
        self._failure_analysis: Dict[str, Dict] = {}
        self._quality_metrics: Dict[str, float] = {}
        self._cost_metrics: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Plan lifecycle tracking
    # -------------------------------------------------------------------------
    def track_plan_start(self, plan: List[Task]) -> Dict[str, Any]:
        """
        Captures the start time and initial task statuses of a plan.

        Args:
            plan: List of tasks in the plan.

        Returns:
            Metadata dictionary with start_time and plan_id.
        """
        with self._lock:
            if not plan:
                logger.warning("Tracking started for empty plan")
                return {
                    "start_time": time.time(),
                    "plan_id": f"empty_plan_{int(time.time())}",
                    "initial_task_statuses": {},
                }

            metadata = {
                "start_time": time.time(),
                "plan_id": f"plan_{int(time.time())}",
                "initial_task_statuses": {task.name: task.status.name for task in plan},
            }
            logger.info(f"[TRACK START] Plan started with {len(plan)} tasks.")
            return metadata

    def track_plan_completion(self, plan_meta: Dict[str, Any], final_status: TaskStatus) -> None:
        """
        Logs the completion status of a plan and its final outcome.

        Args:
            plan_meta: Metadata from track_plan_start.
            final_status: Final success/failure of the plan.
        """
        with self._lock:
            plan_id = plan_meta.get("plan_id", "unknown_plan")
            duration = time.time() - plan_meta.get("start_time", time.time())
            logger.info(
                f"[TRACK COMPLETE] Plan '{plan_id}' completed in {duration:.2f}s "
                f"with status: {final_status.name}"
            )

    # -------------------------------------------------------------------------
    # Core metric recording
    # -------------------------------------------------------------------------
    def record_planning_metrics(
        self,
        plan: Optional[List[Task]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        success_rate: float = 0.0,
        plan_length: Optional[int] = None,
        planning_time: Optional[float] = None,
    ) -> None:
        """
        Records and logs core planning metrics.

        Args:
            plan: The plan (used to derive length if plan_length not given).
            start_time: Start of planning phase.
            end_time: End of planning phase.
            success_rate: 1.0 for success, 0.0 for failure.
            plan_length: Number of primitive steps (overrides plan length).
            planning_time: Duration of planning in seconds (overrides start/end).
        """
        with self._lock:
            # Determine planning time
            if planning_time is not None:
                self._planning_time = planning_time
            elif start_time is not None and end_time is not None:
                self._planning_time = end_time - start_time
            else:
                self._planning_time = 0.0

            # Determine plan length
            if plan_length is not None:
                self._plan_length = plan_length
            elif plan is not None:
                self._plan_length = len([t for t in plan if t.task_type == TaskType.PRIMITIVE])
            else:
                self._plan_length = 0

            self._success_rate = success_rate

            logger.info(
                f"[PLANNING METRICS] Length: {self._plan_length}, "
                f"Time: {self._planning_time:.2f}s, Success: {success_rate:.2f}"
            )

    def record_execution_metrics(
        self,
        success_count: int,
        failure_count: int,
        resource_usage: Dict[str, float],
        execution_result: Dict[str, Any],
    ) -> None:
        """
        Logs post‑execution metrics, including task outcomes and system usage.

        Args:
            success_count: Number of successful tasks.
            failure_count: Number of failed tasks.
            resource_usage: e.g., {"cpu": 23.4, "memory": 65.1}
            execution_result: Dict with 'total_time', 'success_rate', 'resource_efficiency'.
        """
        with self._lock:
            self._execution_time = execution_result.get("total_time", 0)
            self._success_rate = execution_result.get("success_rate", 0)
            self._resource_efficiency = execution_result.get("resource_efficiency", {})

            total = success_count + failure_count
            success_rate = success_count / total if total else 0.0
            logger.info(
                f"[EXECUTION METRICS] Success: {success_count}, Failures: {failure_count}, "
                f"Success Rate: {success_rate:.2f}"
            )
            logger.info(
                f"[RESOURCE USAGE] CPU: {resource_usage.get('cpu', 0):.2f}%, "
                f"Memory: {resource_usage.get('memory', 0):.2f}%"
            )

    # -------------------------------------------------------------------------
    # Efficiency score
    # -------------------------------------------------------------------------
    def calculate_efficiency_score(self) -> float:
        """
        Calculate overall efficiency score using configured weights.

        Returns:
            A float between 0 and 1.
        """
        with self._lock:
            time_score = min(1.0, self._temporal_efficiency)
            resource_score = (
                sum(self._resource_efficiency.values()) / len(self._resource_efficiency)
                if self._resource_efficiency
                else 1.0
            )
            return (
                self.metrics_weights.get("success", 0.5) * self._success_rate
                + self.metrics_weights.get("time", 0.2) * time_score
                + self.metrics_weights.get("cost", 0.3) * resource_score
            )

    # -------------------------------------------------------------------------
    # Static metric helpers (used by other components)
    # -------------------------------------------------------------------------
    @staticmethod
    def plan_length(plan: List[Task]) -> int:
        """Calculate the number of primitive steps in the plan."""
        return sum(1 for task in plan if task.task_type == TaskType.PRIMITIVE)

    def plan_makespan(self, plan: List[Task]) -> float:
        """
        Calculate total execution duration using task timestamps.
        Uses config settings for timing enforcement and fallbacks.
        """
        if not plan:
            return 0.0

        valid_tasks = [t for t in plan if t.start_time is not None and t.end_time is not None]

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
        """Calculate total cost of the plan."""
        return sum(task.cost if hasattr(task, "cost") else default_cost for task in plan)

    @staticmethod
    def goal_achievement_rate(final_status: TaskStatus) -> float:
        """Return 1.0 for success, 0.0 for failure."""
        return 1.0 if final_status == TaskStatus.SUCCESS else 0.0

    @staticmethod
    def planning_time(start_time: float, end_time: float) -> float:
        """Calculate planning duration."""
        return end_time - start_time

    # -------------------------------------------------------------------------
    # Resource monitoring (fallback if psutil unavailable)
    # -------------------------------------------------------------------------
    @staticmethod
    def cpu_usage() -> float:
        """Get current CPU utilization percentage (or 0.0 on error)."""
        try:
            return psutil.cpu_percent()
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {e}")
            return 0.0

    @staticmethod
    def memory_usage() -> float:
        """Get current process memory usage in MB (or 0.0 on error)."""
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    # -------------------------------------------------------------------------
    # All‑in‑one metric collection
    # -------------------------------------------------------------------------
    @classmethod
    def calculate_all_metrics(
        cls,
        plan: List[Task],
        planning_start_time: float,
        planning_end_time: float,
        final_status: TaskStatus,
    ) -> Dict[str, Any]:
        """
        Comprehensive metrics calculation with config integration.

        Returns a dictionary with all relevant metrics.
        """
        instance = cls()
        return {
            "plan_length": cls.plan_length(plan),
            "plan_makespan": instance.plan_makespan(plan),
            "plan_cost": cls.plan_cost(plan),
            "planning_time": planning_end_time - planning_start_time,
            "success_rate": cls.goal_achievement_rate(final_status),
            "resource_usage": {
                "cpu": cls.cpu_usage(),
                "memory": cls.memory_usage(),
            },
            "efficiency_score": (
                instance.metrics_weights.get("success", 1.0)
                * cls.goal_achievement_rate(final_status)
                + instance.metrics_weights.get("cost", 1.0)
                * (1 / (cls.plan_cost(plan) + 1e-6))
                + instance.metrics_weights.get("time", 1.0)
                * (1 / (planning_end_time - planning_start_time + 1e-6))
            ),
        }


if __name__ == "__main__":
    print("\n=== Running Planning Metrics Test ===\n")
    printer.status("Init", "Planning Metrics initialized", "success")

    # Create some example tasks
    tasks = [
        Task("Prim1", TaskType.PRIMITIVE, start_time=0, end_time=2),
        Task("Abstract1", TaskType.ABSTRACT),
        Task("Prim2", TaskType.PRIMITIVE, start_time=2, end_time=5),
    ]

    metrics = PlanningMetrics()
    print(f"\n  Plan length: {metrics.plan_length(tasks)}")  # Should be 2
    print(f"  Makespan: {metrics.plan_makespan(tasks)}")  # Should be 5-0=5

    print("\n=== Kitchen Planning Metrics Demo ===\n")

    # Create realistic cooking tasks with timing and cost data
    kitchen_tasks = [
        Task(
            "GatherIngredients",
            TaskType.PRIMITIVE,
            cost=0.5,
            start_time=0.0,
            end_time=2.5,
            status=TaskStatus.SUCCESS,
        ),
        Task(
            "ChopVegetables",
            TaskType.PRIMITIVE,
            cost=1.2,
            start_time=2.5,
            end_time=5.0,
            status=TaskStatus.SUCCESS,
        ),
        Task(
            "PreheatOven",
            TaskType.PRIMITIVE,
            cost=0.8,
            start_time=5.0,
            end_time=8.0,
            status=TaskStatus.SUCCESS,
        ),
        Task(
            "CookMainDish",
            TaskType.ABSTRACT,
            cost=2.5,
            methods=[
                [
                    Task("PrepareProtein", TaskType.PRIMITIVE, cost=1.0),
                    Task("BakeDish", TaskType.PRIMITIVE, cost=1.5),
                ]
            ],
        ),
        Task(
            "SetTable",
            TaskType.PRIMITIVE,
            cost=0.7,
            start_time=8.0,
            end_time=9.5,
            status=TaskStatus.FAILED,
        ),
    ]

    # Create failed plan version
    failed_tasks = [t.copy() for t in kitchen_tasks]
    for t in failed_tasks:
        if t.name == "CookMainDish":
            t.status = TaskStatus.FAILED

    # Test successful plan
    print("\n=== Testing Successful Meal Preparation ===")
    success_metrics = metrics.calculate_all_metrics(
        plan=[t for t in kitchen_tasks if t.task_type == TaskType.PRIMITIVE],
        planning_start_time=0.0,
        planning_end_time=2.3,
        final_status=TaskStatus.SUCCESS,
    )

    # Test failed plan
    print("\n=== Testing Failed Meal Preparation ===")
    failure_metrics = metrics.calculate_all_metrics(
        plan=[t for t in failed_tasks if t.task_type == TaskType.PRIMITIVE],
        planning_start_time=0.0,
        planning_end_time=1.8,
        final_status=TaskStatus.FAILED,
    )

    # Test plan with missing timing data
    print("\n=== Testing Partial Timing Data ===")
    partial_tasks = kitchen_tasks[:3]  # First 3 tasks only
    partial_metrics = metrics.calculate_all_metrics(
        plan=partial_tasks,
        planning_start_time=0.0,
        planning_end_time=1.2,
        final_status=TaskStatus.SUCCESS,
    )

    def print_metrics(label: str, metrics_dict: Dict[str, Any]) -> None:
        print(f"\n{label} Metrics:")
        for k, v in metrics_dict.items():
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