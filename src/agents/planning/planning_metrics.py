"""
Planning Metrics – production-grade performance tracking for the Planning Agent.

This module collects, normalises, stores, and reports planning/execution metrics
without duplicating lower-level scheduling, temporal, or resource calculations.
Numerical planning calculations are delegated to ``PlanningCalculations`` and
shared validation/serialisation utilities are delegated to ``planning_helpers``.

The public API remains compatible with the earlier implementation:
- ``track_plan_start``
- ``track_plan_completion``
- ``record_planning_metrics``
- ``record_execution_metrics``
- ``calculate_efficiency_score``
- ``plan_length``, ``plan_makespan``, ``plan_cost``
- ``goal_achievement_rate``, ``planning_time``
- ``cpu_usage``, ``memory_usage``
- ``calculate_all_metrics``
"""

from __future__ import annotations

import statistics
import threading
import time
import uuid
import psutil  # type: ignore

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.planning_errors import *
from .utils.planning_helpers import *
from .utils.planning_calculations import PlanningCalculations
from .planning_types import Task, TaskStatus, TaskType
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Planning Metrics")
printer = PrettyPrinter()


MetricValue = Union[int, float]
MetricDict = Dict[str, Any]


@dataclass
class MetricSnapshot:
    """Immutable-ish metric event stored in the bounded metrics history."""

    timestamp: float
    event_type: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "event_type": self.event_type,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


class PlanningMetrics:
    """
    Thread-safe metrics collector for planning and execution quality.

    The class intentionally separates metric collection from core calculations:
    - primitive plan cost/duration/risk calculations can be delegated to
      ``PlanningCalculations``;
    - validation, clamping, dependency maps, serialisation, and logging truncation
      are delegated to ``planning_helpers``;
    - this module owns aggregation, histories, score normalisation, and reports.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "enable_timing": True,
        "enable_system_metrics": True,
        "record_history": True,
        "history_window": 1000,
        "default_task_cost": 1.0,
        "default_task_duration": 0.5,
        "use_length_fallback": True,
        "use_critical_path": True,
        "slow_plan_threshold_seconds": 30.0,
        "memory_unit": "mb",
        "process_memory_only": True,
        "smoothing_alpha": 0.30,
        "metrics_weights": {
            "success": 0.50,
            "cost": 0.20,
            "time": 0.20,
            "resource": 0.10,
        },
        "normalizers": {
            "planning_time_seconds": 60.0,
            "execution_time_seconds": 600.0,
            "cost": 100.0,
            "makespan_seconds": 600.0,
            "plan_length": 100.0,
            "memory_mb": 4096.0,
            "cpu_percent": 100.0,
        },
        "success_statuses": ["SUCCESS"],
        "failure_statuses": ["FAILED", "CANCELLED"],
        "quality_thresholds": {
            "excellent": 0.85,
            "good": 0.70,
            "warning": 0.50,
        },
    }

    def __init__(self, calculations: Optional[PlanningCalculations] = None) -> None:
        """Initialise metrics tracking using ``planning_config.yaml``."""
        self.config = load_global_config()
        raw_cfg = get_config_section("planning_metrics")
        self.metrics_config = deep_update(self.DEFAULT_CONFIG, raw_cfg or {})

        self._lock = threading.RLock()
        self.calculations = calculations if calculations is not None else PlanningCalculations()

        self._load_config_values()

        # Backward-compatible scalar state.
        self._planning_time: float = 0.0
        self._execution_time: float = 0.0
        self._plan_length: int = 0
        self._success_rate: float = 0.0
        self._resource_efficiency: Dict[str, float] = {}
        self._temporal_efficiency: float = 0.0
        self._task_success_rates: Dict[str, float] = {}
        self._method_success_rates: Dict[Tuple[str, int], float] = {}
        self._failure_analysis: Dict[str, Dict[str, Any]] = {}
        self._quality_metrics: Dict[str, float] = {}
        self._cost_metrics: Dict[str, float] = {}

        # Production state.
        self._active_plans: Dict[str, Dict[str, Any]] = {}
        self._history: Deque[MetricSnapshot] = deque(maxlen=self.history_window)
        self._counters: Dict[str, int] = defaultdict(int)
        self._running_averages: Dict[str, float] = {}
        self._last_snapshot: Optional[MetricSnapshot] = None

        logger.info("Planning Metrics successfully initialized")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _load_config_values(self) -> None:
        cfg = self.metrics_config

        self.enable_timing = bool(cfg.get("enable_timing", True))
        self.enable_system_metrics = bool(cfg.get("enable_system_metrics", True))
        self.record_history = bool(cfg.get("record_history", True))

        self.history_window = int(cfg.get("history_window", 1000))
        require_positive(self.history_window, "planning_metrics.history_window")

        self.default_task_cost = float(cfg.get("default_task_cost", 1.0))
        require_non_negative(self.default_task_cost, "planning_metrics.default_task_cost")

        self.default_task_duration = float(cfg.get("default_task_duration", 0.5))
        require_non_negative(self.default_task_duration, "planning_metrics.default_task_duration")

        self.use_length_fallback = bool(cfg.get("use_length_fallback", True))
        self.use_critical_path = bool(cfg.get("use_critical_path", True))

        self.slow_plan_threshold_seconds = float(cfg.get("slow_plan_threshold_seconds", 30.0))
        require_non_negative(self.slow_plan_threshold_seconds, "planning_metrics.slow_plan_threshold_seconds")

        self.memory_unit = str(cfg.get("memory_unit", "mb")).lower()
        if self.memory_unit not in {"bytes", "kb", "mb", "gb"}:
            raise PlanningConfigError(
                "planning_metrics.memory_unit must be one of bytes, kb, mb, gb",
                config_key="memory_unit",
                config_section="planning_metrics",
                expected_type="'bytes' | 'kb' | 'mb' | 'gb'",
            )

        self.process_memory_only = bool(cfg.get("process_memory_only", True))
        self.smoothing_alpha = float(cfg.get("smoothing_alpha", 0.30))
        validate_probability(self.smoothing_alpha, "planning_metrics.smoothing_alpha")

        self.metrics_weights = self._normalise_weights(dict(cfg.get("metrics_weights", {})))
        self.normalizers: Dict[str, float] = {
            str(k): max(float(v), 1e-9)
            for k, v in dict(cfg.get("normalizers", {})).items()
        }

        self.success_statuses = {str(s).upper() for s in cfg.get("success_statuses", ["SUCCESS"])}
        self.failure_statuses = {str(s).upper() for s in cfg.get("failure_statuses", ["FAILED", "CANCELLED"])}
        self.quality_thresholds = dict(cfg.get("quality_thresholds", {}))

    @staticmethod
    def _normalise_weights(weights: Dict[str, Any]) -> Dict[str, float]:
        cleaned = {str(k): max(float(v), 0.0) for k, v in weights.items()}
        if not cleaned:
            cleaned = {"success": 1.0}
        total = sum(cleaned.values())
        if total <= 0.0:
            raise PlanningConfigError(
                "planning_metrics.metrics_weights must contain at least one positive weight",
                config_key="metrics_weights",
                config_section="planning_metrics",
                expected_type="mapping[str, positive number]",
            )
        return {k: v / total for k, v in cleaned.items()}

    # ------------------------------------------------------------------
    # Plan lifecycle tracking
    # ------------------------------------------------------------------
    def track_plan_start(self, plan: List[Task]) -> Dict[str, Any]:
        """
        Capture start metadata for a plan.

        Returns a mutable metadata dict expected by ``track_plan_completion``.
        """
        require_type(plan, list, "plan")
        now = time.time()
        task_ids = [str(getattr(task, "id", f"task_{idx}")) for idx, task in enumerate(plan)]
        plan_id = f"plan_{int(now * 1000)}_{uuid.uuid4().hex[:8]}"

        metadata: Dict[str, Any] = {
            "plan_id": plan_id,
            "start_time": now,
            "task_count": len(plan),
            "task_ids": task_ids,
            "initial_task_statuses": {
                str(getattr(task, "name", task_ids[idx])): self._status_name(getattr(task, "status", None))
                for idx, task in enumerate(plan)
            },
            "initial_resource_usage": self.resource_usage_snapshot() if self.enable_system_metrics else {},
        }

        with self._lock:
            self._active_plans[plan_id] = dict(metadata)
            self._record_snapshot("plan_start", metadata=metadata)

        logger.info("[TRACK START] Plan %s started with %d tasks.", plan_id, len(plan))
        return metadata

    def track_plan_completion(self, plan_meta: Dict[str, Any], final_status: TaskStatus) -> None:
        """Record plan completion and update lifecycle history."""
        require_type(plan_meta, dict, "plan_meta")
        final_status = self._coerce_status(final_status)
        now = time.time()

        with self._lock:
            plan_id = str(plan_meta.get("plan_id", "unknown_plan"))
            start = float(plan_meta.get("start_time", now) or now)
            duration = max(0.0, now - start)
            plan_meta["end_time"] = now
            plan_meta["duration"] = duration
            plan_meta["final_status"] = final_status.name
            plan_meta["success_rate"] = self.goal_achievement_rate(final_status)
            plan_meta["final_resource_usage"] = self.resource_usage_snapshot() if self.enable_system_metrics else {}

            self._execution_time = duration
            self._success_rate = plan_meta["success_rate"]
            self._active_plans.pop(plan_id, None)
            self._update_running_average("plan_duration", duration)
            self._record_snapshot(
                "plan_completion",
                metrics={"duration": duration, "success_rate": self._success_rate},
                metadata=plan_meta,
            )

        logger.info(
            "[TRACK COMPLETE] Plan '%s' completed in %.2fs with status: %s",
            plan_meta.get("plan_id", "unknown_plan"),
            duration,
            final_status.name,
        )

    # ------------------------------------------------------------------
    # Core metric recording
    # ------------------------------------------------------------------
    def record_planning_metrics(
        self,
        plan: Optional[List[Task]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        success_rate: float = 0.0,
        plan_length: Optional[int] = None,
        planning_time: Optional[float] = None,
    ) -> None:
        """Record planner-side metrics such as planning latency and plan length."""
        validate_probability(float(success_rate), "success_rate")

        with self._lock:
            if planning_time is not None:
                self._planning_time = max(0.0, float(planning_time))
            elif start_time is not None and end_time is not None:
                self._planning_time = self.planning_time(float(start_time), float(end_time))
            else:
                self._planning_time = 0.0

            if plan_length is not None:
                self._plan_length = max(0, int(plan_length))
            elif plan is not None:
                self._plan_length = self.plan_length(plan)
            else:
                self._plan_length = 0

            self._success_rate = float(success_rate)
            self._temporal_efficiency = self._time_efficiency(self._planning_time, "planning_time_seconds")
            self._update_running_average("planning_time", self._planning_time)
            self._update_running_average("plan_length", float(self._plan_length))

            metrics = {
                "plan_length": self._plan_length,
                "planning_time": self._planning_time,
                "success_rate": self._success_rate,
                "temporal_efficiency": self._temporal_efficiency,
            }
            self._record_snapshot("planning_metrics", metrics=metrics)

        logger.info(
            "[PLANNING METRICS] Length: %d, Time: %.2fs, Success: %.2f",
            self._plan_length,
            self._planning_time,
            self._success_rate,
        )

    def record_execution_metrics(
        self,
        success_count: int,
        failure_count: int,
        resource_usage: Dict[str, float],
        execution_result: Dict[str, Any],
    ) -> None:
        """Record post-execution metrics including task outcomes and resource use."""
        require_non_negative(success_count, "success_count")
        require_non_negative(failure_count, "failure_count")
        require_type(resource_usage, dict, "resource_usage")
        require_type(execution_result, dict, "execution_result")

        total = int(success_count) + int(failure_count)
        observed_success = (float(success_count) / total) if total else 0.0

        with self._lock:
            self._execution_time = max(0.0, float(execution_result.get("total_time", 0.0) or 0.0))
            self._success_rate = clamp(
                float(execution_result.get("success_rate", observed_success) or 0.0), 0.0, 1.0
            )
            self._resource_efficiency = self._normalise_resource_efficiency(
                execution_result.get("resource_efficiency", resource_usage)
            )
            self._temporal_efficiency = self._time_efficiency(
                self._execution_time, "execution_time_seconds"
            )

            self._counters["successful_tasks"] += int(success_count)
            self._counters["failed_tasks"] += int(failure_count)
            self._update_running_average("execution_time", self._execution_time)

            metrics = {
                "success_count": int(success_count),
                "failure_count": int(failure_count),
                "success_rate": self._success_rate,
                "execution_time": self._execution_time,
                "resource_efficiency": dict(self._resource_efficiency),
                "temporal_efficiency": self._temporal_efficiency,
            }
            self._record_snapshot("execution_metrics", metrics=metrics, metadata={"resource_usage": dict(resource_usage)})

        logger.info(
            "[EXECUTION METRICS] Success: %d, Failures: %d, Success Rate: %.2f",
            success_count,
            failure_count,
            observed_success,
        )

    # ------------------------------------------------------------------
    # Derived scores and reports
    # ------------------------------------------------------------------
    def calculate_efficiency_score(self) -> float:
        """Calculate an overall efficiency score in [0, 1]."""
        with self._lock:
            resource_score = self._average_or_default(self._resource_efficiency.values(), 1.0)
            cost_score = self._cost_efficiency(self._cost_metrics.get("last_plan_cost", 0.0))
            score = (
                self.metrics_weights.get("success", 0.0) * clamp(self._success_rate, 0.0, 1.0)
                + self.metrics_weights.get("time", 0.0) * clamp(self._temporal_efficiency, 0.0, 1.0)
                + self.metrics_weights.get("cost", 0.0) * cost_score
                + self.metrics_weights.get("resource", 0.0) * resource_score
            )
            return clamp(score, 0.0, 1.0)

    def calculate_plan_quality(self, plan: List[Task], final_status: Optional[TaskStatus] = None) -> Dict[str, float]:
        """
        Calculate a quality vector for a plan.

        This augments the legacy scalar efficiency score with interpretable
        sub-scores for success, cost, time, dependency safety, and resource margin.
        """
        require_type(plan, list, "plan")
        status = self._coerce_status(final_status) if final_status is not None else None

        cost = self.plan_cost(plan, self.default_task_cost)
        makespan = self.plan_makespan(plan)
        success = self.goal_achievement_rate(status) if status is not None else self._plan_success_rate(plan)

        try:
            dependency_margin = self.calculations.calculate_dependency_risk(plan)
        except PlanningError as exc:
            logger.warning("Dependency risk calculation failed: %s", exc)
            dependency_margin = 0.0

        try:
            resource_margin = self.calculations.calculate_resource_margin(plan)
        except PlanningError as exc:
            logger.debug("Resource margin unavailable for quality report: %s", exc)
            resource_margin = 0.0

        quality = {
            "success": clamp(success, 0.0, 1.0),
            "cost_efficiency": self._cost_efficiency(cost),
            "time_efficiency": self._time_efficiency(makespan, "makespan_seconds"),
            "dependency_margin": clamp(dependency_margin, 0.0, 1.0),
            "resource_margin": clamp(resource_margin, 0.0, 1.0),
        }
        quality["overall"] = clamp(
            0.40 * quality["success"]
            + 0.20 * quality["cost_efficiency"]
            + 0.20 * quality["time_efficiency"]
            + 0.10 * quality["dependency_margin"]
            + 0.10 * quality["resource_margin"],
            0.0,
            1.0,
        )

        with self._lock:
            self._quality_metrics = dict(quality)
            self._cost_metrics["last_plan_cost"] = cost
            self._record_snapshot("plan_quality", metrics=quality, metadata={"cost": cost, "makespan": makespan})

        return quality

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a thread-safe summary of current metric state."""
        with self._lock:
            return {
                "planning_time": self._planning_time,
                "execution_time": self._execution_time,
                "plan_length": self._plan_length,
                "success_rate": self._success_rate,
                "resource_efficiency": dict(self._resource_efficiency),
                "temporal_efficiency": self._temporal_efficiency,
                "efficiency_score": self.calculate_efficiency_score(),
                "task_success_rates": dict(self._task_success_rates),
                "method_success_rates": {str(k): v for k, v in self._method_success_rates.items()},
                "failure_analysis": dict(self._failure_analysis),
                "quality_metrics": dict(self._quality_metrics),
                "cost_metrics": dict(self._cost_metrics),
                "counters": dict(self._counters),
                "running_averages": dict(self._running_averages),
                "history_size": len(self._history),
                "active_plan_count": len(self._active_plans),
            }

    def export_history(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return recent metric snapshots as JSON-safe dictionaries."""
        with self._lock:
            items = list(self._history)
            if limit is not None:
                items = items[-max(0, int(limit)) :]
            return [snapshot.to_dict() for snapshot in items]

    def to_json(self, *, limit: Optional[int] = None) -> str:
        """Serialise the metrics summary and recent history."""
        payload = {
            "summary": self.get_metrics_summary(),
            "history": self.export_history(limit=limit),
        }
        return safe_json_dumps(payload, indent=2)

    # ------------------------------------------------------------------
    # Success, failure, task, and method statistics
    # ------------------------------------------------------------------
    def update_task_success_rate(self, task_name: str, success: bool) -> float:
        """Update and return an exponentially-smoothed task success rate."""
        require_non_empty(task_name, "task_name")
        return self._update_smoothed_rate(self._task_success_rates, str(task_name), bool(success))

    def update_method_success_rate(self, method_key: Tuple[str, int], success: bool) -> float:
        """Update and return an exponentially-smoothed method success rate."""
        require_type(method_key, tuple, "method_key")
        if len(method_key) != 2:
            raise PlanningConfigError(
                "method_key must be a tuple of (task_name, method_index)",
                config_key="method_key",
                config_section="planning_metrics",
                expected_type="Tuple[str, int]",
            )
        key = (str(method_key[0]), int(method_key[1]))
        return self._update_smoothed_rate(self._method_success_rates, key, bool(success))

    def record_failure(self, task: Optional[Task], reason: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        """Record a structured failure entry for later diagnostics."""
        require_non_empty(reason, "reason")
        task_name = str(getattr(task, "name", "unknown"))
        task_id = str(getattr(task, "id", "unknown"))
        entry = self._failure_analysis.setdefault(
            task_name,
            {"count": 0, "task_ids": [], "reasons": defaultdict(int), "last_context": {}},
        )
        entry["count"] += 1
        if task_id not in entry["task_ids"]:
            entry["task_ids"].append(task_id)
        entry["reasons"][reason] += 1
        entry["last_context"] = dict(context or {})

        with self._lock:
            self._counters["failures_recorded"] += 1
            self._record_snapshot(
                "failure",
                metrics={"count": entry["count"]},
                metadata={"task": task_name, "task_id": task_id, "reason": reason},
            )

    # ------------------------------------------------------------------
    # Static metric helpers (used by other components)
    # ------------------------------------------------------------------
    @staticmethod
    def plan_length(plan: List[Task]) -> int:
        """Calculate the number of primitive steps in the plan."""
        require_type(plan, list, "plan")
        return sum(1 for task in plan if getattr(task, "task_type", None) == TaskType.PRIMITIVE)

    def plan_makespan(self, plan: List[Task]) -> float:
        """Calculate total execution duration using timestamps or duration fallback."""
        require_type(plan, list, "plan")
        if not plan:
            return 0.0

        valid_intervals: List[Tuple[float, float]] = []
        for task in plan:
            start = getattr(task, "start_time", None)
            end = getattr(task, "end_time", None)
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start and end > 0:
                valid_intervals.append((float(start), float(end)))

        if valid_intervals:
            return max(end for _, end in valid_intervals) - min(start for start, _ in valid_intervals)

        if not self.use_length_fallback:
            return 0.0

        try:
            return float(self.calculations.calculate_plan_duration(plan, use_critical_path=self.use_critical_path))
        except PlanningError:
            return len(plan) * self.default_task_duration

    @staticmethod
    def plan_cost(plan: List[Task], default_cost: float = 1.0) -> float:
        """Calculate total non-negative plan cost."""
        require_type(plan, list, "plan")
        total = 0.0
        for task in plan:
            cost = getattr(task, "cost", default_cost)
            try:
                total += max(0.0, float(cost))
            except (TypeError, ValueError):
                total += max(0.0, float(default_cost))
        return total

    @staticmethod
    def goal_achievement_rate(final_status: TaskStatus) -> float:
        """Return 1.0 for success-like terminal status, otherwise 0.0."""
        if isinstance(final_status, TaskStatus):
            return 1.0 if final_status == TaskStatus.SUCCESS else 0.0
        return 1.0 if str(final_status).upper() == "SUCCESS" else 0.0

    @staticmethod
    def planning_time(start_time: float, end_time: float) -> float:
        """Calculate non-negative planning duration."""
        require_type(start_time, (int, float), "start_time")
        require_type(end_time, (int, float), "end_time")
        return max(0.0, float(end_time) - float(start_time))

    # ------------------------------------------------------------------
    # Resource monitoring
    # ------------------------------------------------------------------
    @staticmethod
    def cpu_usage() -> float:
        """Get current CPU utilisation percentage, or 0.0 on telemetry error."""
        try:
            return float(psutil.cpu_percent())
        except Exception as exc:
            logger.warning("Failed to get CPU usage: %s", exc)
            return 0.0

    @staticmethod
    def memory_usage(unit: str = "mb", *, process_only: bool = True) -> float:
        """Get current memory usage in the requested unit."""
        try:
            if process_only:
                bytes_used = float(psutil.Process().memory_info().rss)
            else:
                bytes_used = float(psutil.virtual_memory().used)
            return PlanningMetrics._convert_bytes(bytes_used, unit)
        except Exception as exc:
            logger.warning("Failed to get memory usage: %s", exc)
            return 0.0

    def resource_usage_snapshot(self) -> Dict[str, float]:
        """Capture CPU and memory telemetry using configured units."""
        return {
            "cpu": self.cpu_usage(),
            "memory": self.memory_usage(self.memory_unit, process_only=self.process_memory_only),
        }

    # ------------------------------------------------------------------
    # All-in-one metric collection
    # ------------------------------------------------------------------
    @classmethod
    def calculate_all_metrics(
        cls,
        plan: List[Task],
        planning_start_time: float,
        planning_end_time: float,
        final_status: TaskStatus,
    ) -> Dict[str, Any]:
        """Comprehensive metrics calculation with config integration."""
        instance = cls()
        final_status = instance._coerce_status(final_status)
        planning_duration = cls.planning_time(planning_start_time, planning_end_time)
        cost = cls.plan_cost(plan, instance.default_task_cost)
        makespan = instance.plan_makespan(plan)
        success = cls.goal_achievement_rate(final_status)
        resource_usage = instance.resource_usage_snapshot() if instance.enable_system_metrics else {}

        instance.record_planning_metrics(
            plan=plan,
            planning_time=planning_duration,
            success_rate=success,
        )
        instance._cost_metrics["last_plan_cost"] = cost
        quality = instance.calculate_plan_quality(plan, final_status)

        metrics = {
            "plan_length": cls.plan_length(plan),
            "plan_makespan": makespan,
            "plan_cost": cost,
            "planning_time": planning_duration,
            "success_rate": success,
            "resource_usage": resource_usage,
            "quality": quality,
            "efficiency_score": instance.calculate_efficiency_score(),
        }
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_snapshot(
        self,
        event_type: str,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.record_history:
            return
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            event_type=event_type,
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
        )
        self._history.append(snapshot)
        self._last_snapshot = snapshot
        self._counters[f"{event_type}_events"] += 1

    def _update_running_average(self, key: str, value: float) -> None:
        prev = self._running_averages.get(key)
        if prev is None:
            self._running_averages[key] = float(value)
        else:
            alpha = self.smoothing_alpha
            self._running_averages[key] = (1.0 - alpha) * prev + alpha * float(value)

    def _update_smoothed_rate(self, target: Dict[Any, float], key: Any, success: bool) -> float:
        with self._lock:
            previous = target.get(key, 0.5)
            observed = 1.0 if success else 0.0
            updated = clamp((1.0 - self.smoothing_alpha) * previous + self.smoothing_alpha * observed, 0.0, 1.0)
            target[key] = updated
            self._record_snapshot(
                "rate_update",
                metrics={"rate": updated, "success": observed},
                metadata={"key": str(key)},
            )
            return updated

    def _normalise_resource_efficiency(self, values: Any) -> Dict[str, float]:
        if not isinstance(values, dict):
            return {}
        result: Dict[str, float] = {}
        for key, value in values.items():
            try:
                raw = float(value)
            except (TypeError, ValueError):
                continue
            # If callers provide usage percentages, invert them into efficiency.
            if raw > 1.0:
                norm = 1.0 - clamp(raw / 100.0, 0.0, 1.0)
            else:
                norm = clamp(raw, 0.0, 1.0)
            result[str(key)] = norm
        return result

    def _time_efficiency(self, duration: float, normalizer_key: str) -> float:
        normalizer = self.normalizers.get(normalizer_key, 1.0)
        return clamp(1.0 - (max(0.0, float(duration)) / normalizer), 0.0, 1.0)

    def _cost_efficiency(self, cost: float) -> float:
        normalizer = self.normalizers.get("cost", 1.0)
        return clamp(1.0 - (max(0.0, float(cost)) / normalizer), 0.0, 1.0)

    @staticmethod
    def _average_or_default(values: Iterable[float], default: float) -> float:
        vals = [clamp(float(v), 0.0, 1.0) for v in values]
        return statistics.fmean(vals) if vals else default

    def _plan_success_rate(self, plan: List[Task]) -> float:
        if not plan:
            return 0.0
        completed = 0
        successes = 0
        for task in plan:
            status_name = self._status_name(getattr(task, "status", None))
            if status_name in self.success_statuses or status_name in self.failure_statuses:
                completed += 1
                if status_name in self.success_statuses:
                    successes += 1
        return successes / completed if completed else 0.0

    @staticmethod
    def _status_name(status: Any) -> str:
        if isinstance(status, TaskStatus):
            return status.name
        if hasattr(status, "name"):
            return str(status.name).upper()
        return str(status).upper() if status is not None else "UNKNOWN"

    @staticmethod
    def _coerce_status(status: Any) -> TaskStatus:
        if isinstance(status, TaskStatus):
            return status
        if isinstance(status, str):
            key = status.upper()
            if key in TaskStatus.__members__:
                return TaskStatus[key]
        raise PlanningConfigError(
            f"Invalid task status: {status!r}",
            config_key="final_status",
            config_section="planning_metrics",
            expected_type="TaskStatus",
        )

    @staticmethod
    def _convert_bytes(bytes_used: float, unit: str) -> float:
        unit = str(unit).lower()
        if unit == "bytes":
            return bytes_used
        if unit == "kb":
            return bytes_used / 1024.0
        if unit == "gb":
            return bytes_used / (1024.0 ** 3)
        return bytes_used / (1024.0 ** 2)


if __name__ == "__main__":
    print("\n=== Running Planning Metrics ===\n")
    printer.status("TEST", "Planning Metrics initialized", "info")

    now = time.time()
    tasks = [
        Task("CollectData", TaskType.PRIMITIVE, start_time=now, end_time=now + 2, cost=1.0, status=TaskStatus.SUCCESS),
        Task("AnalyseData", TaskType.PRIMITIVE, start_time=now + 2, end_time=now + 5, cost=2.0, status=TaskStatus.SUCCESS),
        Task("Report", TaskType.PRIMITIVE, start_time=now + 5, end_time=now + 6, cost=0.5, status=TaskStatus.FAILED),
    ]

    metrics = PlanningMetrics()
    meta = metrics.track_plan_start(tasks)
    metrics.record_planning_metrics(plan=tasks, planning_time=0.25, success_rate=0.8)
    metrics.record_execution_metrics(
        success_count=2,
        failure_count=1,
        resource_usage={"cpu": 25.0, "memory": 512.0},
        execution_result={"total_time": 6.0, "success_rate": 2 / 3, "resource_efficiency": {"cpu": 0.75, "memory": 0.85}},
    )
    metrics.track_plan_completion(meta, TaskStatus.SUCCESS)

    report = metrics.calculate_all_metrics(tasks, now, now + 0.25, TaskStatus.SUCCESS)
    printer.pretty("METRICS", report, "success")
    assert report["plan_length"] == 3
    assert report["plan_cost"] >= 3.5
    assert 0.0 <= report["efficiency_score"] <= 1.0

    print("\n=== Test ran successfully ===\n")
