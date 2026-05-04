import psutil
import time

from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_types import Task, TaskType, TaskStatus
from src.agents.planning.planning_metrics import PlanningMetrics
from src.agents.planning.heuristics import DecisionTreeHeuristic, GradientBoostingHeuristic
from src.agents.planning.task_scheduler import DeadlineAwareScheduler
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Monitor")
printer = PrettyPrinter

class PlanningMonitor:
    """Monitors planner outcomes, method quality, and local resource pressure."""

    def __init__(self) -> None:
        self.config = load_global_config()
        self.monitor_config = get_config_section("planning_monitor")

        self.metrics_window = self.monitor_config.get("metrics_window", 50)
        self.method_analysis_depth = self.monitor_config.get("method_analysis_depth", 10)
        self.anomaly_thresholds = self.monitor_config.get(
            "anomaly_thresholds",
            {"success_rate": 0.5, "cpu_peak": 95.0, "memory_peak": 4096.0},
        )
        self.check_intervals = self.monitor_config.get(
            "check_intervals",
            {"plan_execution": 10, "resource_scan": 60},
        )

        self.agent: Optional[Any] = None
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        self.plan_history = deque(maxlen=self.metrics_window)
        self.method_performance = defaultdict(
            lambda: {"success": 0, "total": 0, "last_used": 0.0}
        )
        self.resource_history = deque(maxlen=3600)
        self.last_full_scan = time.time()

    def track_plan_start(self, plan: List[Task]) -> Dict[str, Any]:
        plan_meta = {
            "start_time": time.time(),
            "task_count": len(plan),
            "abstract_count": sum(1 for t in plan if t.task_type == TaskType.ABSTRACT),
            "status": None,
        }
        self.plan_history.append(plan_meta)
        return plan_meta

    def track_plan_completion(self, plan_meta: Dict[str, Any], final_status: TaskStatus) -> None:
        plan_meta["status"] = final_status
        plan_meta["duration"] = time.time() - plan_meta["start_time"]

        current_plan = getattr(self.agent, "current_plan", []) if self.agent else []
        for task in current_plan:
            if task.task_type == TaskType.PRIMITIVE:
                key = (task.name, "primitive")
                self.method_performance[key]["total"] += 1
                if task.status == TaskStatus.SUCCESS:
                    self.method_performance[key]["success"] += 1

        self._perform_interval_checks()

    def update_method_stats(self, method_key: Tuple[str, str], success: bool) -> None:
        self.method_performance[method_key]["total"] += 1
        self.method_performance[method_key]["last_used"] = time.time()
        if success:
            self.method_performance[method_key]["success"] += 1

    def _perform_interval_checks(self) -> None:
        self._check_resource_limits()

        plan_interval = int(self.check_intervals.get("plan_execution", 10))
        if plan_interval > 0 and len(self.plan_history) % plan_interval == 0:
            self._analyze_planning_trends()
            self._identify_method_anomalies()

    def _check_resource_limits(self) -> None:
        current_cpu = psutil.cpu_percent()
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024

        self.resource_history.append(
            {"timestamp": time.time(), "cpu": current_cpu, "memory": current_mem}
        )

        if current_cpu > self.anomaly_thresholds.get("cpu_peak", 95.0):
            logger.warning(f"CPU threshold breached: {current_cpu}%")

        if current_mem > self.anomaly_thresholds.get("memory_peak", 4096.0):
            logger.warning(f"Memory threshold breached: {current_mem:.1f}MB")

        if time.time() - self.last_full_scan > self.check_intervals.get("resource_scan", 60):
            self._perform_full_system_scan()
            self.last_full_scan = time.time()

    def _analyze_planning_trends(self) -> None:
        interval = int(self.check_intervals.get("plan_execution", 10))
        recent_plans = list(self.plan_history)[-interval:]
        success_count = sum(1 for p in recent_plans if p["status"] == TaskStatus.SUCCESS)
        success_rate = success_count / len(recent_plans) if recent_plans else 0

        if success_rate < self.anomaly_thresholds.get("success_rate", 0.5):
            logger.error(f"Success rate alert: {success_rate:.1%} below threshold")

    def _identify_method_anomalies(self) -> None:
        methods = sorted(
            self.method_performance.items(),
            key=lambda x: x[1]["total"],
            reverse=True,
        )[: self.method_analysis_depth]

        for (name, type_), stats in methods:
            if stats["total"] == 0:
                continue
            success_rate = stats["success"] / stats["total"]
            if success_rate < self.anomaly_thresholds.get("success_rate", 0.5):
                logger.warning(
                    f"Underperforming method: {name} ({type_}) Success rate: {success_rate:.1%}"
                )

    def _perform_full_system_scan(self) -> None:
        logger.info("Performing full planning system scan...")

        total_method_calls = sum(m["total"] for m in self.method_performance.values())
        if total_method_calls > 0:
            top_method = max(self.method_performance.items(), key=lambda x: x[1]["total"])
            logger.debug(f"Most used method: {top_method[0][0]} ({top_method[1]['total']} uses)")

        if self.resource_history:
            avg_cpu = sum(r["cpu"] for r in self.resource_history) / len(self.resource_history)
            avg_mem = sum(r["memory"] for r in self.resource_history) / len(self.resource_history)
            logger.info(f"Resource averages - CPU: {avg_cpu:.1f}%, Memory: {avg_mem:.1f}MB")

    def generate_diagnostics(self) -> Dict[str, Any]:
        return {
            "recent_success_rate": self._calculate_recent_success_rate(),
            "active_methods": len(self.method_performance),
            "resource_stats": self._current_resource_usage(),
            "pending_plans": len(getattr(self.agent, "current_plan", [])) if self.agent else 0,
        }

    def monitor_planning_metrics(
        self,
        plan: List[Task],
        final_status: TaskStatus,
        planning_start: float,
        planning_end: float,
    ) -> Dict[str, Any]:
        results = PlanningMetrics.calculate_all_metrics(
            plan,
            planning_start,
            planning_end,
            final_status,
        )

        logger.info(f"Plan metrics summary: {results}")
        return results

    def monitor_decision_tree_heuristic(self, dt_heuristic: DecisionTreeHeuristic) -> Dict[str, Any]:
        if not dt_heuristic.trained or dt_heuristic.model is None:
            logger.warning("DecisionTreeHeuristic is not trained.")
            return {"trained": False}

        importances = getattr(dt_heuristic.model, "feature_importances_", [])
        named = list(zip(dt_heuristic.feature_names, importances))
        top_features = sorted(named, key=lambda x: x[1], reverse=True)

        return {
            "trained": True,
            "top_features": top_features[:5],
            "full_feature_importances": named,
        }

    def monitor_gradient_boosting_heuristic(
        self, gb_heuristic: GradientBoostingHeuristic
    ) -> Dict[str, Any]:
        if not gb_heuristic.trained or gb_heuristic.model is None:
            logger.warning("GradientBoostingHeuristic is not trained.")
            return {"trained": False}

        importances = getattr(gb_heuristic.model, "feature_importances_", [])
        named = list(zip(gb_heuristic.feature_names, importances))
        top_features = sorted(named, key=lambda x: x[1], reverse=True)

        return {
            "trained": True,
            "top_features": top_features[:5],
            "full_feature_importances": named,
        }

    def monitor_deadline_scheduler(self, scheduler: DeadlineAwareScheduler) -> Dict[str, Any]:
        assignments = scheduler.task_history.get("current", [])
        total_assignments = len(assignments)

        return {
            "total_assignments": total_assignments,
            "tracked_tasks": [task.get("id") for task in assignments if isinstance(task, dict)],
        }

    def _calculate_recent_success_rate(self) -> float:
        recent = list(self.plan_history)[-self.metrics_window :]
        successes = sum(1 for p in recent if p.get("status") == TaskStatus.SUCCESS)
        return successes / len(recent) if recent else 0.0

    def _current_resource_usage(self) -> Dict[str, Any]:
        return {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.Process().memory_info().rss / 1024 / 1024,
            "timestamp": time.time(),
        }


if __name__ == "__main__":
    print("")
    print("\n=== Running Planning Monitor ===")
    print("")
    monitor = PlanningMonitor()

    # Example stubs:
    dt_heuristic = DecisionTreeHeuristic()
    gb_heuristic = GradientBoostingHeuristic()
    scheduler = DeadlineAwareScheduler()

    monitor.monitor_decision_tree_heuristic(dt_heuristic)
    monitor.monitor_gradient_boosting_heuristic(gb_heuristic)
    monitor.monitor_deadline_scheduler(scheduler)
    print("")
    print("\n=== Successfully Ran Planning Monitor ===\n")
