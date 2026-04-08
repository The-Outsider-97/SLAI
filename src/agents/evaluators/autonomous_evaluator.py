"""
Production-ready evaluator for planning systems, robotics, and automation tasks.

This module expands the original autonomous evaluator into a structured,
configuration-driven subsystem that:
- validates and normalizes autonomous-task payloads
- evaluates single tasks and batches with detailed diagnostics
- computes planning, efficiency, collision, and graph-structure metrics
- integrates with evaluator memory and the reporting/visualization pipeline
- uses structured evaluation errors for predictable failure handling
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import uuid
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

from .utils.config_loader import load_global_config, get_config_section
from .utils.evaluators_calculations import EvaluatorsCalculations
from .utils.evaluation_errors import (ConfigLoadError, OperationalError, VisualizationError,
                                      MetricCalculationError, ReportGenerationError,
                                      ThresholdViolationError, ValidationFailureError)
from .modules.report import get_visualizer
from .evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Autonomous Evaluator")
printer = PrettyPrinter

MODULE_VERSION = "2.0.0"


@dataclass(slots=True)
class TaskMetrics:
    """Normalized metrics for a single autonomous task."""

    task_id: str
    task_type: str
    completion_time: float
    path_length: float
    optimal_path_length: float
    energy_consumed: float
    success: bool
    collisions: int
    deviation_from_optimal: float
    path_efficiency: float
    energy_efficiency: float
    waypoint_count: int
    graph_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskEvaluationRecord:
    """Detailed task-level evaluation artifact."""

    task_id: str
    task_type: str
    metrics: TaskMetrics
    composite_score: float
    status: str
    threshold_violations: List[Dict[str, Any]] = field(default_factory=list)
    plan_graph_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "metrics": self.metrics.to_dict(),
            "composite_score": self.composite_score,
            "status": self.status,
            "threshold_violations": list(self.threshold_violations),
            "plan_graph_metrics": dict(self.plan_graph_metrics),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AutonomousEvaluationSummary:
    """Aggregate autonomous evaluation output."""

    batch_id: str
    generated_at: str
    total_tasks: int
    evaluated_tasks: int
    failed_evaluations: int
    success_rate: float
    path_efficiency: float
    energy_efficiency: float
    collision_rate: float
    completion_time_mean: float
    deviation_mean: float
    composite_score: float
    threshold_status: str
    threshold_violations: List[Dict[str, Any]] = field(default_factory=list)
    graph_summary: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    task_results: List[Dict[str, Any]] = field(default_factory=list)
    task_failures: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutonomousEvaluator:
    """
    Evaluator for planning systems, robotics, and automation tasks.

    Responsibilities
    ----------------
    - Validate autonomous-task payloads and normalize geometry/metadata
    - Compute per-task and aggregate planning metrics
    - Analyze optional planning graphs and render graph visualizations
    - Store artifacts in evaluator memory for downstream audit/reporting
    - Forward summarized status into the reporting visualizer when enabled
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))
        self.eval_config = get_config_section("autonomous_evaluator")
        if not isinstance(self.eval_config, Mapping):
            raise ConfigLoadError(
                config_path=self.config_path,
                section="autonomous_evaluator",
                error_details="Section must be a mapping.",
            )

        self.store_results = bool(self.eval_config.get("store_results", self.config.get("store_results", True)))

        self.thresholds = self._load_thresholds(self.eval_config.get("thresholds", {}))
        self.metric_weights = self._load_metric_weights(self.eval_config.get("weights", {}))
        self.viz_config = self._load_visualization_config(self.eval_config.get("visualization", {}))

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()

        self.task_history: List[TaskEvaluationRecord] = []
        self.plan_graphs: Dict[str, Dict[str, Any]] = {}
        self.batch_history: List[AutonomousEvaluationSummary] = []
        self._visualizer = None

        logger.info(
            "AutonomousEvaluator initialized with thresholds=%s weights=%s",
            self.thresholds,
            self.metric_weights,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_task(
        self,
        task: Mapping[str, Any],
        *,
        store_result: Optional[bool] = None,
        update_visualizer: bool = False,
    ) -> TaskEvaluationRecord:
        """Evaluate a single planning/robotics task in detail."""
        normalized_task = self._normalize_task(task)
        graph_metrics = self._analyze_plan_graph(normalized_task.get("plan_graph"))
        path_length = self._calculate_path_length(normalized_task["path"])
        optimal_length = self._calculate_path_length(normalized_task["optimal_path"])
        deviation = self._calculate_deviation(path_length, optimal_length)
        path_efficiency = self._bounded_ratio(optimal_length, path_length)
        energy_efficiency = self._bounded_inverse_ratio(
            normalized_task["energy_consumed"],
            float(self.thresholds["max_energy_per_task"]),
        )
        success = self._resolve_success(normalized_task, deviation)

        metrics = TaskMetrics(
            task_id=normalized_task["id"],
            task_type=normalized_task["type"],
            completion_time=float(normalized_task["completion_time"]),
            path_length=path_length,
            optimal_path_length=optimal_length,
            energy_consumed=float(normalized_task["energy_consumed"]),
            success=success,
            collisions=int(normalized_task["collisions"]),
            deviation_from_optimal=deviation,
            path_efficiency=path_efficiency,
            energy_efficiency=energy_efficiency,
            waypoint_count=len(normalized_task["path"]),
            graph_available=bool(normalized_task.get("plan_graph")),
        )

        violations = self._evaluate_task_thresholds(metrics)
        composite_score = self._calculate_task_composite_score(metrics)
        recommendations = self._generate_task_recommendations(metrics, graph_metrics, violations)
        status = "PASSED" if not violations and metrics.success else "FAILED"

        record = TaskEvaluationRecord(
            task_id=metrics.task_id,
            task_type=metrics.task_type,
            metrics=metrics,
            composite_score=composite_score,
            status=status,
            threshold_violations=violations,
            plan_graph_metrics=graph_metrics,
            recommendations=recommendations,
            metadata={
                "evaluated_at": _utcnow().isoformat(),
                "module_version": MODULE_VERSION,
                "input_metadata": dict(normalized_task.get("metadata", {})),
            },
        )

        self.task_history.append(record)
        if normalized_task.get("plan_graph"):
            self.plan_graphs[metrics.task_id] = dict(normalized_task["plan_graph"])

        should_store = self.store_results if store_result is None else bool(store_result)
        if should_store:
            self._store_task_result(record)

        if update_visualizer:
            self._update_visualizer_for_record(record)

        return record

    def evaluate_task_set(
        self,
        tasks: Sequence[Mapping[str, Any]],
        *,
        batch_id: Optional[str] = None,
        store_result: Optional[bool] = None,
        update_visualizer: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a batch of planning/robotics tasks and return an aggregate summary."""
        validated_tasks = self._ensure_task_sequence(tasks)
        resolved_batch_id = batch_id or self._generate_batch_id()

        task_records: List[TaskEvaluationRecord] = []
        task_failures: List[Dict[str, Any]] = []

        for task in validated_tasks:
            task_id = str(task.get("id") or "unknown")
            try:
                task_records.append(
                    self.evaluate_task(
                        task,
                        store_result=False,
                        update_visualizer=False,
                    )
                )
            except (ValidationFailureError, MetricCalculationError, OperationalError) as exc:
                failure = self._serialize_failure(task_id, exc)
                task_failures.append(failure)
                self._store_error_if_possible(exc, task_id=task_id, batch_id=resolved_batch_id)
            except Exception as exc:
                wrapped = OperationalError(
                    message=f"Unexpected failure while evaluating autonomous task '{task_id}'.",
                    context={"task_id": task_id, "batch_id": resolved_batch_id, "error": str(exc)},
                    cause=exc,
                )
                failure = self._serialize_failure(task_id, wrapped)
                task_failures.append(failure)
                self._store_error_if_possible(wrapped, task_id=task_id, batch_id=resolved_batch_id)

        summary = self._build_summary(
            batch_id=resolved_batch_id,
            task_records=task_records,
            task_failures=task_failures,
        )

        should_store = self.store_results if store_result is None else bool(store_result)
        if should_store:
            self._store_batch_result(summary)

        if update_visualizer:
            self._update_visualizer_from_summary(summary)

        self.batch_history.append(summary)
        return summary.to_dict()

    def generate_report(
        self,
        results: Mapping[str, Any],
        *,
        format: str = "markdown",
        include_visualizations: bool = True,
        include_system_visualizer_report: bool = False,
    ) -> str | Dict[str, Any]:
        """Generate a detailed autonomous evaluation report in markdown, JSON, or YAML."""
        if not isinstance(results, Mapping):
            raise ReportGenerationError(
                report_type="autonomous_evaluation",
                template="autonomous_evaluator",
                error_details="results must be a mapping.",
            )

        normalized_format = str(format).strip().lower()
        try:
            structured_report = self._build_structured_report(
                results,
                include_visualizations=include_visualizations,
                include_system_visualizer_report=include_system_visualizer_report,
            )
            if normalized_format == "dict":
                return structured_report
            if normalized_format == "json":
                return json.dumps(structured_report, indent=2, sort_keys=False, default=str)
            if normalized_format == "yaml":
                return yaml.safe_dump(structured_report, default_flow_style=False, sort_keys=False)
            if normalized_format == "markdown":
                return self._build_markdown_report(structured_report)

            raise ReportGenerationError(
                report_type="autonomous_evaluation",
                template=normalized_format,
                error_details="Unsupported report format. Use 'markdown', 'json', 'yaml', or 'dict'.",
            )
        except ReportGenerationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                report_type="autonomous_evaluation",
                template=normalized_format,
                error_details=str(exc),
            ) from exc

    def export_report(
        self,
        results: Mapping[str, Any],
        *,
        destination_path: str,
        format: str = "json",
        include_visualizations: bool = True,
        include_system_visualizer_report: bool = False,
    ) -> str:
        """Generate and persist a report to disk."""
        serialized = self.generate_report(
            results,
            format=format,
            include_visualizations=include_visualizations,
            include_system_visualizer_report=include_system_visualizer_report,
        )
        if isinstance(serialized, dict):
            serialized = json.dumps(serialized, indent=2, sort_keys=False, default=str)
        try:
            with open(destination_path, "w", encoding="utf-8") as handle:
                handle.write(serialized)
            return destination_path
        except OSError as exc:
            raise ReportGenerationError(
                report_type="autonomous_evaluation",
                template=format,
                error_details=f"Failed to write report to '{destination_path}': {exc}",
            ) from exc

    def render_plan_graph(self, graph_data: Mapping[str, Any]) -> str:
        """Public wrapper for base64 graph rendering."""
        return self._render_plan_graph(graph_data)

    def disable_temporarily(self) -> None:
        """Temporarily clear volatile autonomous evaluation state."""
        self.task_history.clear()
        self.plan_graphs.clear()
        logger.warning("Autonomous Evaluator temporarily disabled and volatile state cleared.")

    def reset(self) -> None:
        """Fully reset local in-memory autonomous evaluation state."""
        self.task_history.clear()
        self.plan_graphs.clear()
        self.batch_history.clear()
        logger.info("Autonomous evaluator state reset")

    # ------------------------------------------------------------------
    # Core evaluation helpers
    # ------------------------------------------------------------------

    def _normalize_task(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(task, Mapping):
            raise ValidationFailureError("autonomous_task_mapping", type(task).__name__, "mapping")

        normalized: Dict[str, Any] = dict(task)
        normalized.setdefault("id", f"task_{uuid.uuid4().hex[:12]}")
        normalized.setdefault("type", "autonomous")
        normalized.setdefault("success", None)
        normalized.setdefault("metadata", {})

        required_keys = ["completion_time", "path", "optimal_path", "energy_consumed", "collisions"]
        missing = [key for key in required_keys if key not in normalized]
        if missing:
            raise ValidationFailureError(
                "autonomous_task_required_fields",
                data=missing,
                expected=required_keys,
            )

        normalized["id"] = self._normalize_non_empty_string(normalized["id"], "task.id")
        normalized["type"] = self._normalize_non_empty_string(normalized["type"], "task.type")
        normalized["completion_time"] = self._require_non_negative_float(
            normalized["completion_time"], "task.completion_time"
        )
        normalized["path"] = self._normalize_path(normalized["path"], "task.path")
        normalized["optimal_path"] = self._normalize_path(normalized["optimal_path"], "task.optimal_path")
        normalized["energy_consumed"] = self._require_non_negative_float(
            normalized["energy_consumed"], "task.energy_consumed"
        )
        normalized["collisions"] = self._require_non_negative_int(normalized["collisions"], "task.collisions")

        if normalized["success"] is not None and not isinstance(normalized["success"], bool):
            raise ValidationFailureError("task.success", type(normalized["success"]).__name__, "bool or None")

        if not isinstance(normalized.get("metadata", {}), Mapping):
            raise ValidationFailureError("task.metadata", type(normalized.get("metadata")).__name__, "mapping")
        normalized["metadata"] = dict(normalized.get("metadata", {}))

        if "plan_graph" in normalized and normalized["plan_graph"] is not None:
            normalized["plan_graph"] = self._normalize_plan_graph(normalized["plan_graph"])

        return normalized

    def _build_summary(
        self,
        *,
        batch_id: str,
        task_records: Sequence[TaskEvaluationRecord],
        task_failures: Sequence[Mapping[str, Any]],
    ) -> AutonomousEvaluationSummary:
        total_tasks = len(task_records) + len(task_failures)
        if total_tasks == 0:
            raise ValidationFailureError("autonomous_task_batch", 0, "at least one task")

        successful_tasks = sum(1 for record in task_records if record.metrics.success)
        total_collisions = sum(record.metrics.collisions for record in task_records)
        total_energy = sum(record.metrics.energy_consumed for record in task_records)
        total_path_length = sum(record.metrics.path_length for record in task_records)
        total_optimal_length = sum(record.metrics.optimal_path_length for record in task_records)
        completion_times = [record.metrics.completion_time for record in task_records]
        deviations = [record.metrics.deviation_from_optimal for record in task_records if math.isfinite(record.metrics.deviation_from_optimal)]

        success_rate = successful_tasks / total_tasks if total_tasks else 0.0
        path_efficiency = self._bounded_ratio(total_optimal_length, total_path_length)
        energy_efficiency = self._bounded_inverse_ratio(total_energy, self.thresholds["max_energy_per_task"] * max(total_tasks, 1))
        collision_rate = total_collisions / total_tasks if total_tasks else 0.0
        completion_time_mean = float(mean(completion_times)) if completion_times else 0.0
        deviation_mean = float(mean(deviations)) if deviations else float("inf")

        aggregate_metrics = {
            "success_rate": success_rate,
            "path_efficiency": path_efficiency,
            "energy_efficiency": energy_efficiency,
            "collision_rate": collision_rate,
        }
        composite_score = self._calculate_aggregate_composite_score(aggregate_metrics)
        threshold_violations = self._evaluate_batch_thresholds(
            success_rate=success_rate,
            deviation_mean=deviation_mean,
            energy_per_task=(total_energy / total_tasks) if total_tasks else 0.0,
            collision_rate=collision_rate,
        )
        graph_summary = self._summarize_graph_metrics(task_records)
        statistics = self._build_statistics(task_records)
        recommendations = self._generate_batch_recommendations(
            aggregate_metrics=aggregate_metrics,
            threshold_violations=threshold_violations,
            graph_summary=graph_summary,
            task_records=task_records,
        )

        return AutonomousEvaluationSummary(
            batch_id=batch_id,
            generated_at=_utcnow().isoformat(),
            total_tasks=total_tasks,
            evaluated_tasks=len(task_records),
            failed_evaluations=len(task_failures),
            success_rate=success_rate,
            path_efficiency=path_efficiency,
            energy_efficiency=energy_efficiency,
            collision_rate=collision_rate,
            completion_time_mean=completion_time_mean,
            deviation_mean=deviation_mean,
            composite_score=composite_score,
            threshold_status="PASS" if not threshold_violations else "ATTENTION",
            threshold_violations=threshold_violations,
            graph_summary=graph_summary,
            statistics=statistics,
            recommendations=recommendations,
            task_results=[record.to_dict() for record in task_records],
            task_failures=[dict(item) for item in task_failures],
            metadata={
                "module_version": MODULE_VERSION,
                "config_path": self.config_path,
                "evaluated_task_ids": [record.task_id for record in task_records],
            },
        )

    def _calculate_task_composite_score(self, metrics: TaskMetrics) -> float:
        success_component = 1.0 if metrics.success else 0.0
        collision_component = 1.0 - min(1.0, metrics.collisions / max(1.0, float(self.thresholds["max_collisions"] + 1)))
        weighted = (
            self.metric_weights["success_rate"] * success_component
            + self.metric_weights["path_efficiency"] * metrics.path_efficiency
            + self.metric_weights["energy_efficiency"] * metrics.energy_efficiency
            + self.metric_weights["collision_penalty"] * (1.0 - collision_component)
        )
        return float(max(0.0, min(1.0, weighted)))

    def _calculate_aggregate_composite_score(self, aggregate_metrics: Mapping[str, float]) -> float:
        weighted = (
            self.metric_weights["success_rate"] * float(aggregate_metrics["success_rate"])
            + self.metric_weights["path_efficiency"] * float(aggregate_metrics["path_efficiency"])
            + self.metric_weights["energy_efficiency"] * float(aggregate_metrics["energy_efficiency"])
            + self.metric_weights["collision_penalty"] * float(aggregate_metrics["collision_rate"])
        )
        return float(max(0.0, min(1.0, weighted)))

    def _evaluate_task_thresholds(self, metrics: TaskMetrics) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []

        if metrics.deviation_from_optimal > self.thresholds["max_path_deviation"]:
            violations.append(
                ThresholdViolationError(
                    metric="max_path_deviation",
                    value=float(metrics.deviation_from_optimal),
                    threshold=float(self.thresholds["max_path_deviation"]),
                    comparator="<=",
                ).to_audit_dict()
            )
        if metrics.energy_consumed > self.thresholds["max_energy_per_task"]:
            violations.append(
                ThresholdViolationError(
                    metric="max_energy_per_task",
                    value=float(metrics.energy_consumed),
                    threshold=float(self.thresholds["max_energy_per_task"]),
                    comparator="<=",
                ).to_audit_dict()
            )
        if metrics.collisions > self.thresholds["max_collisions"]:
            violations.append(
                ThresholdViolationError(
                    metric="max_collisions",
                    value=float(metrics.collisions),
                    threshold=float(self.thresholds["max_collisions"]),
                    comparator="<=",
                ).to_audit_dict()
            )
        return violations

    def _evaluate_batch_thresholds(
        self,
        *,
        success_rate: float,
        deviation_mean: float,
        energy_per_task: float,
        collision_rate: float,
    ) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        if success_rate < self.thresholds["min_success_rate"]:
            violations.append(
                ThresholdViolationError(
                    metric="min_success_rate",
                    value=float(success_rate),
                    threshold=float(self.thresholds["min_success_rate"]),
                    comparator=">=",
                ).to_audit_dict()
            )
        if deviation_mean > self.thresholds["max_path_deviation"]:
            violations.append(
                ThresholdViolationError(
                    metric="max_path_deviation_mean",
                    value=float(deviation_mean),
                    threshold=float(self.thresholds["max_path_deviation"]),
                    comparator="<=",
                ).to_audit_dict()
            )
        if energy_per_task > self.thresholds["max_energy_per_task"]:
            violations.append(
                ThresholdViolationError(
                    metric="max_energy_per_task_mean",
                    value=float(energy_per_task),
                    threshold=float(self.thresholds["max_energy_per_task"]),
                    comparator="<=",
                ).to_audit_dict()
            )
        if collision_rate > self.thresholds["max_collisions"]:
            violations.append(
                ThresholdViolationError(
                    metric="max_collision_rate",
                    value=float(collision_rate),
                    threshold=float(self.thresholds["max_collisions"]),
                    comparator="<=",
                ).to_audit_dict()
            )
        return violations

    def _build_statistics(self, task_records: Sequence[TaskEvaluationRecord]) -> Dict[str, Any]:
        datasets = {
            "completion_time": [record.metrics.completion_time for record in task_records],
            "energy_consumed": [record.metrics.energy_consumed for record in task_records],
            "path_length": [record.metrics.path_length for record in task_records],
            "deviation_from_optimal": [record.metrics.deviation_from_optimal for record in task_records if math.isfinite(record.metrics.deviation_from_optimal)],
        }
        non_empty = {key: values for key, values in datasets.items() if values}
        if not non_empty:
            return {}

        summary: Dict[str, Any] = {
            "basic": {
                name: {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "std_dev": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
                for name, values in non_empty.items()
            }
        }

        try:
            min_sample_size = int(getattr(self.calculations, "min_sample_size", 10))
            eligible = {name: values for name, values in non_empty.items() if len(values) >= min_sample_size}
            if eligible:
                summary["inferential"] = self.calculations.calculate_statistical_analysis(eligible)
        except Exception as exc:
            summary["inferential_error"] = str(exc)
            logger.warning("Autonomous statistical analysis skipped: %s", exc)

        return summary

    # ------------------------------------------------------------------
    # Report and visualization helpers
    # ------------------------------------------------------------------

    def _build_structured_report(
        self,
        results: Mapping[str, Any],
        *,
        include_visualizations: bool,
        include_system_visualizer_report: bool,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "metadata": {
                "generated_at": _utcnow().isoformat(),
                "module": self.__class__.__name__,
                "module_version": MODULE_VERSION,
                "config_path": self.config_path,
            },
            "summary": {
                "batch_id": results.get("batch_id"),
                "total_tasks": results.get("total_tasks", 0),
                "evaluated_tasks": results.get("evaluated_tasks", 0),
                "failed_evaluations": results.get("failed_evaluations", 0),
                "success_rate": results.get("success_rate", 0.0),
                "path_efficiency": results.get("path_efficiency", 0.0),
                "energy_efficiency": results.get("energy_efficiency", 0.0),
                "collision_rate": results.get("collision_rate", 0.0),
                "composite_score": results.get("composite_score", 0.0),
                "threshold_status": results.get("threshold_status", "UNKNOWN"),
            },
            "thresholds": {
                "configured": dict(self.thresholds),
                "violations": list(results.get("threshold_violations", [])),
            },
            "statistics": dict(results.get("statistics", {})),
            "graph_summary": dict(results.get("graph_summary", {})),
            "recommendations": list(results.get("recommendations", [])),
            "task_results": list(results.get("task_results", [])),
            "task_failures": list(results.get("task_failures", [])),
        }

        if include_visualizations:
            report["visualizations"] = self._generate_visualization_assets(results)

        if include_system_visualizer_report:
            try:
                visualizer = self._get_visualizer()
                report["system_visualizer_report"] = visualizer.generate_full_report(include_visualizations=False)
            except Exception as exc:
                report["system_visualizer_report_error"] = str(exc)
                logger.warning("Failed to embed system visualizer report: %s", exc)

        return report

    def _build_markdown_report(self, report: Mapping[str, Any]) -> str:
        summary = report.get("summary", {})
        lines = [
            "# Autonomous Evaluator Report",
            "",
            f"**Generated**: {report.get('metadata', {}).get('generated_at', _utcnow().isoformat())}",
            f"**Batch ID**: {summary.get('batch_id', '<unknown>')}",
            "",
            "## Executive Summary",
            f"- **Total Tasks**: {summary.get('total_tasks', 0)}",
            f"- **Evaluated Tasks**: {summary.get('evaluated_tasks', 0)}",
            f"- **Failed Evaluations**: {summary.get('failed_evaluations', 0)}",
            f"- **Success Rate**: {float(summary.get('success_rate', 0.0)):.2%}",
            f"- **Path Efficiency**: {float(summary.get('path_efficiency', 0.0)):.4f}",
            f"- **Energy Efficiency**: {float(summary.get('energy_efficiency', 0.0)):.4f}",
            f"- **Collision Rate**: {float(summary.get('collision_rate', 0.0)):.4f}",
            f"- **Composite Score**: {float(summary.get('composite_score', 0.0)):.4f}",
            f"- **Threshold Status**: {summary.get('threshold_status', 'UNKNOWN')}",
            "",
            "## Threshold Violations",
        ]

        violations = report.get("thresholds", {}).get("violations", [])
        if violations:
            for violation in violations:
                lines.append(f"- **{violation.get('message', 'Violation')}**")
        else:
            lines.append("- None")

        lines.extend(["", "## Recommendations"])
        recommendations = report.get("recommendations", [])
        if recommendations:
            for item in recommendations:
                lines.append(f"- {item}")
        else:
            lines.append("- No recommendations generated")

        lines.extend(["", "## Task Metrics", "", "| Task ID | Type | Success | Path Eff. | Energy Eff. | Collisions | Composite |", "|---|---|---:|---:|---:|---:|---:|"])
        for record in report.get("task_results", []):
            metrics = record.get("metrics", {})
            lines.append(
                f"| {record.get('task_id', '')} | {record.get('task_type', '')} | "
                f"{str(metrics.get('success', False))} | {float(metrics.get('path_efficiency', 0.0)):.4f} | "
                f"{float(metrics.get('energy_efficiency', 0.0)):.4f} | {int(metrics.get('collisions', 0))} | "
                f"{float(record.get('composite_score', 0.0)):.4f} |"
            )

        if report.get("graph_summary"):
            lines.extend(["", "## Planning Graph Summary"])
            for key, value in report["graph_summary"].items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        visuals = report.get("visualizations", {})
        if visuals.get("plan_graphs"):
            lines.extend(["", "## Graph Visualizations"])
            for graph in visuals["plan_graphs"]:
                lines.append(f"### {graph.get('task_id', 'graph')}")
                lines.append("")
                lines.append(f"![Plan Graph](data:image/png;base64,{graph.get('image', '')})")
                lines.append("")

        lines.extend(["", "---", f"*Report generated by {self.__class__.__name__} {MODULE_VERSION}*"])
        return "\n".join(lines)

    def _generate_visualization_assets(self, results: Mapping[str, Any]) -> Dict[str, Any]:
        assets: Dict[str, Any] = {"plan_graphs": []}
        task_results = results.get("task_results", [])
        rendered = 0
        for record in task_results:
            task_id = record.get("task_id")
            if not task_id or task_id not in self.plan_graphs:
                continue
            try:
                assets["plan_graphs"].append(
                    {
                        "task_id": task_id,
                        "encoding": "base64-png",
                        "image": self._render_plan_graph(self.plan_graphs[task_id]),
                    }
                )
                rendered += 1
                if rendered >= 3:
                    break
            except VisualizationError as exc:
                assets.setdefault("errors", []).append(exc.to_audit_dict())
        return assets

    def _render_plan_graph(self, graph_data: Mapping[str, Any]) -> str:
        try:
            normalized = self._normalize_plan_graph(graph_data)
            graph = nx.DiGraph()
            for node in normalized["nodes"]:
                graph.add_node(node["id"], **node.get("properties", {}))
            for edge in normalized["edges"]:
                graph.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))

            if graph.number_of_nodes() == 0:
                raise VisualizationError("plan_graph", graph_data, "Graph contains no nodes.")

            figure, axis = plt.subplots(figsize=(10, 7))
            positions = nx.spring_layout(graph, seed=42)

            node_colors = []
            for _, attrs in graph.nodes(data=True):
                if attrs.get("start", False):
                    node_colors.append("green")
                elif attrs.get("goal", False):
                    node_colors.append("red")
                else:
                    node_colors.append("skyblue")

            nx.draw_networkx_nodes(
                graph,
                positions,
                node_size=self.viz_config["node_size"],
                node_color=node_colors,
                ax=axis,
            )
            nx.draw_networkx_edges(graph, positions, arrowstyle="->", arrowsize=18, ax=axis)
            nx.draw_networkx_labels(
                graph,
                positions,
                font_size=self.viz_config["font_size"],
                ax=axis,
            )
            if self.viz_config["show_weights"]:
                edge_labels = nx.get_edge_attributes(graph, "weight")
                nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, ax=axis)

            axis.set_axis_off()
            figure.tight_layout()
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight")
            plt.close(figure)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except VisualizationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                chart_type="plan_graph",
                data=graph_data,
                error_details=f"Graph rendering failed: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Graph analysis
    # ------------------------------------------------------------------

    def _analyze_plan_graph(self, graph_data: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not graph_data:
            return {}

        normalized = self._normalize_plan_graph(graph_data)
        graph = nx.DiGraph()
        for node in normalized["nodes"]:
            graph.add_node(node["id"], **node.get("properties", {}))
        for edge in normalized["edges"]:
            graph.add_edge(edge["source"], edge["target"], weight=float(edge.get("weight", 1.0)))

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        is_dag = nx.is_directed_acyclic_graph(graph)
        start_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("start", False)]
        goal_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get("goal", False)]

        shortest_cost = None
        if start_nodes and goal_nodes:
            for start in start_nodes:
                for goal in goal_nodes:
                    try:
                        candidate = nx.shortest_path_length(graph, start, goal, weight="weight")
                        shortest_cost = candidate if shortest_cost is None else min(shortest_cost, candidate)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

        metrics: Dict[str, Any] = {
            "node_count": node_count,
            "edge_count": edge_count,
            "density": float(nx.density(graph)) if node_count > 1 else 0.0,
            "is_dag": bool(is_dag),
            "has_cycle": not is_dag,
            "average_out_degree": float(sum(dict(graph.out_degree()).values()) / node_count) if node_count else 0.0,
            "start_nodes": start_nodes,
            "goal_nodes": goal_nodes,
            "shortest_weighted_path": None if shortest_cost is None else float(shortest_cost),
        }

        if is_dag and node_count > 0:
            try:
                metrics["dag_longest_path"] = int(len(nx.dag_longest_path(graph)) - 1)
            except Exception:
                metrics["dag_longest_path"] = None
        else:
            metrics["dag_longest_path"] = None

        return metrics

    def _summarize_graph_metrics(self, task_records: Sequence[TaskEvaluationRecord]) -> Dict[str, Any]:
        graph_metrics = [record.plan_graph_metrics for record in task_records if record.plan_graph_metrics]
        if not graph_metrics:
            return {}

        densities = [float(item.get("density", 0.0)) for item in graph_metrics]
        edge_counts = [int(item.get("edge_count", 0)) for item in graph_metrics]
        cycle_count = sum(1 for item in graph_metrics if item.get("has_cycle", False))

        return {
            "graphs_analyzed": len(graph_metrics),
            "average_density": float(mean(densities)) if densities else 0.0,
            "average_edge_count": float(mean(edge_counts)) if edge_counts else 0.0,
            "cyclic_graphs": cycle_count,
            "dag_ratio": float(sum(1 for item in graph_metrics if item.get("is_dag", False)) / len(graph_metrics)),
        }

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def _store_task_result(self, record: TaskEvaluationRecord) -> None:
        try:
            payload = record.to_dict()
            if hasattr(self.memory, "add_evaluation_result"):
                self.memory.add_evaluation_result(
                    evaluator_name="AutonomousEvaluator",
                    result=payload,
                    tags=["autonomous", "task", record.task_type],
                    priority="high" if record.status == "FAILED" else "medium",
                    metadata={"task_id": record.task_id},
                )
            else:
                self.memory.add(
                    entry=payload,
                    tags=["autonomous", "task", record.task_type],
                    priority="high" if record.status == "FAILED" else "medium",
                )
        except Exception as exc:
            logger.error("Failed to store autonomous task result %s: %s", record.task_id, exc)

    def _store_batch_result(self, summary: AutonomousEvaluationSummary) -> None:
        try:
            payload = summary.to_dict()
            if hasattr(self.memory, "add_evaluation_result"):
                self.memory.add_evaluation_result(
                    evaluator_name="AutonomousEvaluator",
                    result=payload,
                    tags=["autonomous", "batch", summary.threshold_status.casefold()],
                    priority="high" if summary.threshold_violations else "medium",
                    metadata={"batch_id": summary.batch_id},
                )
            else:
                self.memory.add(
                    entry=payload,
                    tags=["autonomous", "batch", summary.threshold_status.casefold()],
                    priority="high" if summary.threshold_violations else "medium",
                )
        except Exception as exc:
            logger.error("Failed to store autonomous batch result %s: %s", summary.batch_id, exc)

    def _store_error_if_possible(self, error: Exception, **context: Any) -> None:
        try:
            if hasattr(self.memory, "add_error") and hasattr(error, "to_audit_dict"):
                self.memory.add_error(
                    error,
                    tags=["autonomous", "error"],
                    metadata=context,
                )
            else:
                self.memory.add(
                    entry={
                        "error_type": error.__class__.__name__,
                        "message": str(error),
                        "context": context,
                        "timestamp": _utcnow().isoformat(),
                    },
                    tags=["autonomous", "error"],
                    priority="high",
                )
        except Exception as exc:
            logger.error("Failed to persist autonomous evaluator error: %s", exc)

    def _get_visualizer(self):
        if self._visualizer is None:
            self._visualizer = get_visualizer()
        return self._visualizer

    def _update_visualizer_for_record(self, record: TaskEvaluationRecord) -> None:
        try:
            visualizer = self._get_visualizer()
            visualizer.update_metrics(
                {
                    "successes": int(record.metrics.success),
                    "failures": int(not record.metrics.success),
                    "operational_time": record.metrics.completion_time,
                    "hazards": {"system_failure": float(record.metrics.collisions > 0)},
                    "risk": float(record.metrics.collisions),
                    "pass_rate": 1.0 if record.metrics.success else 0.0,
                    "reward": record.composite_score,
                }
            )
        except Exception as exc:
            logger.warning("Failed to update shared visualizer from autonomous task: %s", exc)

    def _update_visualizer_from_summary(self, summary: AutonomousEvaluationSummary) -> None:
        try:
            visualizer = self._get_visualizer()
            visualizer.update_metrics(
                {
                    "successes": int(round(summary.success_rate * summary.total_tasks)),
                    "failures": summary.total_tasks - int(round(summary.success_rate * summary.total_tasks)),
                    "operational_time": summary.completion_time_mean,
                    "hazards": {"system_failure": float(summary.collision_rate)},
                    "risk": float(summary.collision_rate),
                    "pass_rate": float(summary.success_rate),
                    "reward": float(summary.composite_score),
                }
            )
        except Exception as exc:
            logger.warning("Failed to update shared visualizer from autonomous batch: %s", exc)

    # ------------------------------------------------------------------
    # Recommendations and formatting
    # ------------------------------------------------------------------

    def _generate_task_recommendations(
        self,
        metrics: TaskMetrics,
        graph_metrics: Mapping[str, Any],
        threshold_violations: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        recommendations: List[str] = []
        if metrics.collisions > 0:
            recommendations.append("Introduce stronger collision prediction and reactive avoidance policies.")
        if metrics.deviation_from_optimal > self.thresholds["max_path_deviation"]:
            recommendations.append("Review path-planning heuristics and replanning triggers to reduce route deviation.")
        if metrics.energy_consumed > self.thresholds["max_energy_per_task"]:
            recommendations.append("Adopt energy-aware trajectory optimization and actuator scheduling.")
        if graph_metrics.get("has_cycle", False):
            recommendations.append("Inspect planning graph cycles for deadlock or oscillatory decision patterns.")
        if not recommendations and not threshold_violations:
            recommendations.append("Task performance is within configured autonomous-operation thresholds.")
        return recommendations

    def _generate_batch_recommendations(
        self,
        *,
        aggregate_metrics: Mapping[str, float],
        threshold_violations: Sequence[Mapping[str, Any]],
        graph_summary: Mapping[str, Any],
        task_records: Sequence[TaskEvaluationRecord],
    ) -> List[str]:
        recommendations: List[str] = []
        if aggregate_metrics["success_rate"] < self.thresholds["min_success_rate"]:
            recommendations.append("Increase scenario coverage and failure-recovery robustness for autonomous execution.")
        if aggregate_metrics["path_efficiency"] < 0.9:
            recommendations.append("Tune planning heuristics or search strategies to improve path optimality.")
        if aggregate_metrics["energy_efficiency"] < 0.8:
            recommendations.append("Add energy-aware planning cost terms and actuator-control optimization.")
        if aggregate_metrics["collision_rate"] > 0.0:
            recommendations.append("Strengthen safety buffers, obstacle modeling, and contingency behaviors.")
        if graph_summary.get("cyclic_graphs", 0) > 0:
            recommendations.append("Investigate cyclic planning graphs for unstable policy loops or planner indecision.")

        failure_count = sum(1 for record in task_records if record.status == "FAILED")
        if failure_count > 0:
            recommendations.append("Inspect failed task records individually to isolate scenario-specific regressions.")

        if not recommendations and not threshold_violations:
            recommendations.append("Autonomous task set is operating within configured performance and safety thresholds.")
        return recommendations

    # ------------------------------------------------------------------
    # Validation and normalization helpers
    # ------------------------------------------------------------------

    def _load_thresholds(self, raw: Any) -> Dict[str, float]:
        defaults = {
            "max_path_deviation": 1.2,
            "max_energy_per_task": 500.0,
            "max_collisions": 0.0,
            "min_success_rate": 0.95,
        }
        payload = dict(defaults)
        if raw:
            if not isinstance(raw, Mapping):
                raise ConfigLoadError(self.config_path, "autonomous_evaluator.thresholds", "Thresholds must be a mapping.")
            payload.update(raw)

        payload["max_path_deviation"] = self._require_positive_float(payload["max_path_deviation"], "max_path_deviation")
        payload["max_energy_per_task"] = self._require_positive_float(payload["max_energy_per_task"], "max_energy_per_task")
        payload["max_collisions"] = float(self._require_non_negative_int(payload["max_collisions"], "max_collisions"))
        payload["min_success_rate"] = self._coerce_probability(payload["min_success_rate"], "min_success_rate")
        return payload

    def _load_metric_weights(self, raw: Any) -> Dict[str, float]:
        defaults = {
            "success_rate": 0.4,
            "path_efficiency": 0.3,
            "energy_efficiency": 0.2,
            "collision_penalty": -0.5,
        }
        payload = dict(defaults)
        if raw:
            if not isinstance(raw, Mapping):
                raise ConfigLoadError(self.config_path, "autonomous_evaluator.weights", "Weights must be a mapping.")
            payload.update(raw)

        normalized = {key: float(value) for key, value in payload.items()}
        required = {"success_rate", "path_efficiency", "energy_efficiency", "collision_penalty"}
        missing = sorted(required - set(normalized))
        if missing:
            raise ConfigLoadError(self.config_path, "autonomous_evaluator.weights", f"Missing weights: {missing}")
        return normalized

    def _load_visualization_config(self, raw: Any) -> Dict[str, Any]:
        defaults = {"node_size": 300, "font_size": 8, "show_weights": True}
        payload = dict(defaults)
        if raw:
            if not isinstance(raw, Mapping):
                raise ConfigLoadError(self.config_path, "autonomous_evaluator.visualization", "Visualization config must be a mapping.")
            payload.update(raw)
        payload["node_size"] = self._require_positive_int(payload["node_size"], "node_size")
        payload["font_size"] = self._require_positive_int(payload["font_size"], "font_size")
        payload["show_weights"] = bool(payload["show_weights"])
        return payload

    def _ensure_task_sequence(self, tasks: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
            raise ValidationFailureError("autonomous_task_batch", type(tasks).__name__, "sequence of task mappings")
        validated = list(tasks)
        if not validated:
            raise ValidationFailureError("autonomous_task_batch", 0, "at least one task")
        return validated

    def _normalize_path(self, path: Any, field_name: str) -> List[Tuple[float, float]]:
        if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
            raise ValidationFailureError(field_name, type(path).__name__, "sequence of 2D waypoints")
        normalized: List[Tuple[float, float]] = []
        for index, waypoint in enumerate(path):
            if not isinstance(waypoint, Sequence) or isinstance(waypoint, (str, bytes)) or len(waypoint) < 2:
                raise ValidationFailureError(f"{field_name}[{index}]", waypoint, "2D coordinate sequence")
            x = self._coerce_float(waypoint[0], f"{field_name}[{index}].x")
            y = self._coerce_float(waypoint[1], f"{field_name}[{index}].y")
            normalized.append((x, y))
        return normalized

    def _normalize_plan_graph(self, graph_data: Any) -> Dict[str, Any]:
        if not isinstance(graph_data, Mapping):
            raise ValidationFailureError("plan_graph", type(graph_data).__name__, "mapping")
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        if not isinstance(nodes, Sequence) or not isinstance(edges, Sequence):
            raise ValidationFailureError("plan_graph_structure", graph_data, "nodes/edges sequences")

        normalized_nodes = []
        node_ids = set()
        for index, node in enumerate(nodes):
            if not isinstance(node, Mapping):
                raise ValidationFailureError(f"plan_graph.nodes[{index}]", type(node).__name__, "mapping")
            node_id = self._normalize_non_empty_string(node.get("id"), f"plan_graph.nodes[{index}].id")
            node_ids.add(node_id)
            normalized_nodes.append({
                "id": node_id,
                "properties": dict(node.get("properties", {})) if isinstance(node.get("properties", {}), Mapping) else {},
            })

        normalized_edges = []
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping):
                raise ValidationFailureError(f"plan_graph.edges[{index}]", type(edge).__name__, "mapping")
            source = self._normalize_non_empty_string(edge.get("source"), f"plan_graph.edges[{index}].source")
            target = self._normalize_non_empty_string(edge.get("target"), f"plan_graph.edges[{index}].target")
            if source not in node_ids or target not in node_ids:
                raise ValidationFailureError(
                    f"plan_graph.edges[{index}]",
                    {"source": source, "target": target},
                    "edge endpoints must reference declared nodes",
                )
            normalized_edges.append({
                "source": source,
                "target": target,
                "weight": self._coerce_float(edge.get("weight", 1.0), f"plan_graph.edges[{index}].weight"),
            })

        return {"nodes": normalized_nodes, "edges": normalized_edges}

    def _resolve_success(self, task: Mapping[str, Any], deviation: float) -> bool:
        explicit = task.get("success")
        if explicit is not None:
            return bool(explicit)
        return bool(
            task["collisions"] <= self.thresholds["max_collisions"]
            and deviation <= self.thresholds["max_path_deviation"]
            and task["energy_consumed"] <= self.thresholds["max_energy_per_task"]
        )

    # ------------------------------------------------------------------
    # Primitive metric helpers
    # ------------------------------------------------------------------

    def _calculate_path_length(self, path: Sequence[Tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for index in range(1, len(path)):
            x1, y1 = path[index - 1]
            x2, y2 = path[index]
            total += math.hypot(x2 - x1, y2 - y1)
        return float(total)

    @staticmethod
    def _calculate_deviation(path_length: float, optimal_length: float) -> float:
        if optimal_length <= 0.0:
            return float("inf") if path_length > 0 else 1.0
        return float(path_length / optimal_length)

    @staticmethod
    def _bounded_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, numerator / denominator)))

    @staticmethod
    def _bounded_inverse_ratio(observed: float, threshold: float) -> float:
        if observed <= 0.0:
            return 1.0
        if threshold <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, threshold / observed)))

    # ------------------------------------------------------------------
    # Generic validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_non_empty_string(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValidationFailureError(field_name, value, "non-empty string")
        return value.strip()

    @staticmethod
    def _coerce_float(value: Any, field_name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "float") from exc
        if math.isnan(result) or math.isinf(result):
            raise ValidationFailureError(field_name, value, "finite float")
        return result

    def _require_non_negative_float(self, value: Any, field_name: str) -> float:
        result = self._coerce_float(value, field_name)
        if result < 0.0:
            raise ValidationFailureError(field_name, result, "non-negative float")
        return result

    def _require_positive_float(self, value: Any, field_name: str) -> float:
        result = self._coerce_float(value, field_name)
        if result <= 0.0:
            raise ValidationFailureError(field_name, result, "positive float")
        return result

    @staticmethod
    def _require_non_negative_int(value: Any, field_name: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "non-negative integer") from exc
        if result < 0:
            raise ValidationFailureError(field_name, result, "non-negative integer")
        return result

    @staticmethod
    def _require_positive_int(value: Any, field_name: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "positive integer") from exc
        if result <= 0:
            raise ValidationFailureError(field_name, result, "positive integer")
        return result

    @staticmethod
    def _coerce_probability(value: Any, field_name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "probability in [0,1]") from exc
        if result < 0.0 or result > 1.0:
            raise ValidationFailureError(field_name, result, "probability in [0,1]")
        return result

    @staticmethod
    def _generate_batch_id() -> str:
        return f"auto_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _serialize_failure(task_id: str, error: Exception) -> Dict[str, Any]:
        if hasattr(error, "to_audit_dict"):
            payload = error.to_audit_dict()
        else:
            payload = {
                "type": error.__class__.__name__,
                "message": str(error),
            }
        payload["task_id"] = task_id
        payload["recorded_at"] = _utcnow().isoformat()
        return payload


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "AutonomousEvaluator",
    "AutonomousEvaluationSummary",
    "TaskEvaluationRecord",
    "TaskMetrics",
]

if __name__ == "__main__":
    print("\n=== Running Autonomous Evaluator ===\n")

    tasks = [
        {
            'id': 'nav_001',
            'type': 'navigation',
            'path': [(0,0), (1,1), (2,2), (3,3)],
            'optimal_path': [(0,0), (3,3)],
            'completion_time': 12.5,
            'energy_consumed': 120,
            'collisions': 0,
            'success': True,
            'plan_graph': {
                'nodes': [
                    {'id': 'A', 'properties': {'start': True}},
                    {'id': 'B'},
                    {'id': 'C', 'properties': {'goal': True}}
                ],
                'edges': [
                    {'source': 'A', 'target': 'B', 'weight': 1.2},
                    {'source': 'B', 'target': 'C', 'weight': 0.8}
                ]
            }
        },
        {
            'id': 'manip_002',
            'type': 'manipulation',
            'path': [(0,0), (1,0), (1,1), (2,1)],
            'optimal_path': [(0,0), (2,1)],
            'completion_time': 18.2,
            'energy_consumed': 210,
            'collisions': 1,
            'success': False
        }
    ]
    evaluator = AutonomousEvaluator()
    print(evaluator)

    results = evaluator.evaluate_task_set(tasks)
    
    print(f"Results: {results}")
    print(f"\nReport:\n{evaluator.generate_report(results)}")

    print("\n=== Successfully Ran Autonomous Evaluator ===\n")