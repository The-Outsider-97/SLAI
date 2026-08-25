"""Lazy public API for the planning subsystem.

Planning implementations include optional scientific and ML dependencies. A
package import should expose the API without eagerly importing every backend.
"""

from __future__ import annotations

from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ArbitrationDecision": (".local_behavior_arbitrator", "ArbitrationDecision"),
    "BehaviorType": (".local_behavior_arbitrator", "BehaviorType"),
    "DeadlineAwareScheduler": (".task_scheduler", "DeadlineAwareScheduler"),
    "DistributedOrchestrator": (".safety_planning", "DistributedOrchestrator"),
    "HeuristicPerformance": (".heuristic_selector", "HeuristicPerformance"),
    "HeuristicSelector": (".heuristic_selector", "HeuristicSelector"),
    "LocalBehaviorArbitrator": (".local_behavior_arbitrator", "LocalBehaviorArbitrator"),
    "LocalPlanningContext": (".local_behavior_arbitrator", "LocalPlanningContext"),
    "MetricSnapshot": (".planning_metrics", "MetricSnapshot"),
    "PlanningExecutor": (".planning_executor", "PlanningExecutor"),
    "PlanningMemory": (".planning_memory", "PlanningMemory"),
    "PlanningMetrics": (".planning_metrics", "PlanningMetrics"),
    "PlanningMonitor": (".planning_monitor", "PlanningMonitor"),
    "ProbabilisticAction": (".probabilistic_planner", "ProbabilisticAction"),
    "ProbabilisticPlanner": (".probabilistic_planner", "ProbabilisticPlanner"),
    "SafetyPlanning": (".safety_planning", "SafetyPlanning"),
    "ScheduleDiagnostics": (".task_scheduler", "ScheduleDiagnostics"),
    "SelectorDecision": (".heuristic_selector", "SelectorDecision"),
    "TaskScheduler": (".task_scheduler", "TaskScheduler"),
}

__all__ = sorted(_EXPORTS) # pyright: ignore[reportUnsupportedDunderAll, reportMissingImports]


def __getattr__(name: str):
    """Resolve one exported planning symbol on first use."""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = __import__(f"{__name__}{module_name}", fromlist=[attribute_name])
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
