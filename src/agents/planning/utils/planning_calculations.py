"""
Planning Calculations – Centralized service for safety‑critical planning computations.

Provides methods to compute resource margins, temporal margins, dependency risk,
plan duration/cost, probability of success, and risk scores. All calculations
respect the global configuration and can be safely called from multiple threads.
"""

import time
import threading

from typing import Dict, List, Optional, Union, Any

from src.agents.planning.planning_types import Task, ResourceProfile, ClusterResources
from src.agents.planning.utils.config_loader import get_config_section, load_global_config
from src.agents.planning.utils.resource_monitor import ResourceMonitor
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Calculations")
printer = PrettyPrinter


class PlanningCalculations:
    """
    Central service for planning‑related calculations.
    All public methods are thread‑safe and use cached results where appropriate.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.safety_config = get_config_section("safety_margins")
        self.resource_buffers = self.safety_config.get("resource_buffers", {})
        self.temporal_config = self.safety_config.get("temporal", {})

        # Default buffers from config or hardcoded
        self.gpu_buffer = self.resource_buffers.get("gpu", 0.15)
        self.ram_buffer = self.resource_buffers.get("ram", 0.2)
        self.hw_buffer = 0.1  # default buffer for specialized hardware
        self.time_buffer = self.temporal_config.get("time_buffer", 120)
        self.min_task_duration = self.temporal_config.get("min_task_duration", 30)

        # Resource monitor (may be None if not initialised)
        self.resource_monitor: Optional[ResourceMonitor] = None

        # Simple caches with locks
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()

        logger.info("PlanningCalculations initialised")

    # -------------------------------------------------------------------------
    # Resource calculations
    # -------------------------------------------------------------------------
    def calculate_resource_margin(
        self, tasks: Union[Task, List[Task]], resource_state: Optional[ClusterResources] = None
    ) -> float:
        """
        Calculate resource margin for a task or list of tasks (0‑1 scale, 1 = best).

        Uses the current cluster resource state if resource_state is provided,
        otherwise tries to obtain it from the resource monitor.

        Args:
            tasks: Single task or list of tasks.
            resource_state: Optional current cluster resources (if not provided, uses monitor).

        Returns:
            Float between 0 and 1 representing the margin (1 = fully safe).
        """
        if not tasks:
            return 1.0

        if not isinstance(tasks, list):
            tasks = [tasks]

        # Get available resources
        if resource_state is None:
            if self.resource_monitor is None:
                logger.warning("No resource monitor available – returning default margin")
                return 0.7  # Conservative default
            available = self.resource_monitor.get_available_resources()
        else:
            available = resource_state

        # Aggregate requirements
        total_req = ResourceProfile()
        for task in tasks:
            req = task.resource_requirements
            total_req.gpu += req.gpu
            total_req.ram += req.ram
            total_req.specialized_hardware = list(
                set(total_req.specialized_hardware) | set(req.specialized_hardware)
            )

        # Calculate component margins
        gpu_margin = self._calculate_component_margin(total_req.gpu, available.gpu_total, "gpu")
        ram_margin = self._calculate_component_margin(total_req.ram, available.ram_total, "ram")
        hw_margin = self._calculate_hardware_margin(
            total_req.specialized_hardware, available.specialized_hardware_available
        )

        # Geometric mean for balanced view
        return (gpu_margin * ram_margin * hw_margin) ** (1 / 3)

    def _calculate_component_margin(self, required: float, available: float, resource_type: str) -> float:
        """Calculate margin for a single resource type."""
        if required <= 0:
            return 1.0
        if available <= 0:
            return 0.0

        utilization = required / available
        margin = 1 - utilization

        # Apply configured buffer
        if resource_type == "gpu":
            buffer = self.gpu_buffer
        elif resource_type == "ram":
            buffer = self.ram_buffer
        else:
            buffer = 0.0  # fallback

        return max(0.0, min(1.0, margin - buffer))

    def _calculate_hardware_margin(self, required: List[str], available: List[str]) -> float:
        """Calculate margin for specialised hardware."""
        if not required:
            return 1.0
        if not available:
            return 0.0

        coverage = len(set(required) & set(available)) / len(required)
        return max(0.0, min(1.0, coverage - self.hw_buffer))

    # -------------------------------------------------------------------------
    # Temporal calculations
    # -------------------------------------------------------------------------
    def calculate_temporal_margin(
        self, tasks: Union[Task, List[Task]], current_time: Optional[float] = None
    ) -> float:
        """
        Calculate temporal margin for a task or list of tasks (0‑1 scale, 1 = best).

        Uses the earliest deadline and total estimated duration, respecting dependencies
        if the tasks are part of a plan.

        Args:
            tasks: Single task or list of tasks.
            current_time: Current timestamp (default = time.time()).

        Returns:
            Float between 0 and 1 representing the margin.
        """
        if not tasks:
            return 1.0

        if not isinstance(tasks, list):
            tasks = [tasks]

        if current_time is None:
            current_time = time.time()

        # For a simple list, we sum durations and take the latest deadline.
        # More sophisticated methods (e.g., critical path) would require a plan structure.
        total_duration = 0.0
        max_deadline = 0.0

        for task in tasks:
            duration = getattr(task, "duration", 300.0)
            deadline = getattr(task, "deadline", current_time + 3600.0)

            total_duration += duration
            if deadline > max_deadline:
                max_deadline = deadline

        available_time = max(0.0, max_deadline - current_time)
        if available_time <= 0:
            return 0.0

        utilization = total_duration / available_time
        margin = 1 - utilization
        return max(0.0, min(1.0, margin - (self.time_buffer / max(available_time, 1e-6))))

    def calculate_plan_duration(self, plan: List[Task]) -> float:
        """
        Estimate total plan duration considering task durations and dependencies.

        This is a simplified estimate; a full critical‑path analysis would require
        a dependency graph.

        Args:
            plan: List of tasks in execution order.

        Returns:
            Total estimated duration in seconds.
        """
        if not plan:
            return 0.0
        # Simple sum (assumes sequential execution)
        return sum(task.duration for task in plan)

    def estimate_remaining_time(self, plan: List[Task], current_time: float) -> float:
        """
        Estimate remaining time based on currently executing task.

        Args:
            plan: List of tasks in order.
            current_time: Current timestamp.

        Returns:
            Estimated seconds remaining, or 0.0 if plan is empty.
        """
        if not plan:
            return 0.0

        # Find the index of the first non‑completed task
        remaining = 0.0
        for task in plan:
            if task.status.value < 2:  # not SUCCESS or FAILED
                # If it's currently executing, add remaining duration (estimated)
                if task.start_time and task.duration:
                    elapsed = current_time - task.start_time
                    remaining += max(0.0, task.duration - elapsed)
                else:
                    remaining += task.duration
            # else completed, skip
        return remaining

    # -------------------------------------------------------------------------
    # Risk and probability calculations
    # -------------------------------------------------------------------------
    def calculate_dependency_risk(self, tasks: Union[Task, List[Task]]) -> float:
        """
        Calculate risk associated with dependencies (0‑1 scale, 1 = best / no risk).

        Uses graph complexity metrics.

        Args:
            tasks: Single task or list of tasks.

        Returns:
            Float between 0 and 1 (1 = safe).
        """
        if not tasks:
            return 1.0

        if not isinstance(tasks, list):
            tasks = [tasks]

        # Build dependency graph (task.id -> list of dependency ids)
        graph: Dict[str, List[str]] = {}
        for task in tasks:
            deps = getattr(task, "dependencies", [])
            if deps:
                graph[task.id] = deps

        num_nodes = len(graph)
        if num_nodes == 0:
            return 1.0

        num_edges = sum(len(deps) for deps in graph.values())

        # Criticality (longest path)
        criticality = self._find_criticality(graph)

        # Normalise
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        normalized_criticality = criticality / num_nodes if num_nodes > 0 else 0

        risk = 0.6 * normalized_criticality + 0.4 * edge_density
        return max(0.0, min(1.0, 1.0 - risk))  # convert to margin

    def _find_criticality(self, graph: Dict[str, List[str]]) -> int:
        """Find longest path length in a DAG (if cycle exists, returns maximum)."""
        if not graph:
            return 0

        # Build in‑degree
        in_degree = {node: 0 for node in graph}
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Kahn's algorithm to compute longest distances
        dist = {node: 0 for node in graph}
        queue = [node for node in graph if in_degree.get(node, 0) == 0]

        while queue:
            node = queue.pop(0)
            for neighbor in graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if dist[neighbor] < dist[node] + 1:
                        dist[neighbor] = dist[node] + 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return max(dist.values()) if dist else 0

    def calculate_probability_of_success(self, task: Task) -> float:
        """
        Estimate probability of success for a task.

        Uses task's probabilistic actions if available, otherwise falls back
        to historical success rate from memory (if accessible).

        Args:
            task: The task to evaluate.

        Returns:
            Float between 0 and 1.
        """
        if task.is_probabilistic and task.probabilistic_actions:
            # For now, take the maximum success probability among actions
            # (A more sophisticated approach would combine them)
            max_prob = 0.0
            for action in task.probabilistic_actions:
                # Assume action has a 'success_probability' attribute or we compute from outcomes
                prob = getattr(action, "success_rate", 0.0)
                max_prob = max(max_prob, prob)
            if max_prob > 0:
                return max_prob
        # Fallback: use success threshold as a crude estimate
        return getattr(task, "success_threshold", 0.9)

    def estimate_risk_score(self, task: Task) -> float:
        """
        Estimate risk score for a task (0‑1, where 0 = safe, 1 = very risky).

        Combines probability of failure, uncertainty, and dependency impact.

        Args:
            task: The task to evaluate.

        Returns:
            Float between 0 and 1 (higher = riskier).
        """
        prob_failure = 1.0 - self.calculate_probability_of_success(task)
        # Uncertainty could be derived from variance of duration or cost; placeholder
        uncertainty = getattr(task, "risk_score", 0.0)  # if task has a pre‑computed risk
        return min(1.0, prob_failure + uncertainty * 0.5)

    # -------------------------------------------------------------------------
    # Safety margin checks
    # -------------------------------------------------------------------------
    def check_safety_margins(
        self, plan: List[Task], resources: Optional[ClusterResources] = None
    ) -> Dict[str, float]:
        """
        Check all safety margins (resource and temporal) for a plan.

        Returns a dictionary of margin names and their values.
        """
        margins = {
            "resource": self.calculate_resource_margin(plan, resources),
            "temporal": self.calculate_temporal_margin(plan),
        }
        if plan:
            margins["dependency"] = self.calculate_dependency_risk(plan)
        return margins

    # -------------------------------------------------------------------------
    # Cost calculations
    # -------------------------------------------------------------------------
    def calculate_plan_cost(self, plan: List[Task]) -> float:
        """
        Calculate total cost of a plan.
        """
        return sum(task.cost for task in plan)

    # -------------------------------------------------------------------------
    # Cache management (optional)
    # -------------------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear internal calculation cache."""
        with self._cache_lock:
            self._cache.clear()
            logger.debug("Calculation cache cleared")

    # -------------------------------------------------------------------------
    # Resource monitor integration
    # -------------------------------------------------------------------------
    def set_resource_monitor(self, monitor: ResourceMonitor) -> None:
        """Inject a resource monitor instance."""
        self.resource_monitor = monitor
        logger.info("Resource monitor set")


if __name__ == "__main__":
    print("\n=== Running Planning Calculations Test ===\n")
    printer.status("Init", "Planning Calculations initialized", "success")

    # Create some dummy tasks
    task1 = Task("A", preconditions=[], effects=[], duration=10.0, cost=5.0)
    task2 = Task("B", preconditions=[], effects=[], duration=20.0, cost=3.0, dependencies=[task1.id])

    calc = PlanningCalculations()
    print(f"Resource margin (single task): {calc.calculate_resource_margin(task1):.3f}")
    print(f"Temporal margin (two tasks): {calc.calculate_temporal_margin([task1, task2]):.3f}")
    print(f"Dependency risk: {calc.calculate_dependency_risk([task1, task2]):.3f}")
    print(f"Plan duration: {calc.calculate_plan_duration([task1, task2]):.1f} s")
    print(f"Plan cost: {calc.calculate_plan_cost([task1, task2]):.1f}")
    print(f"Probability of success (task1): {calc.calculate_probability_of_success(task1):.3f}")

    print("\n=== Successfully Ran Planning Calculations ===\n")