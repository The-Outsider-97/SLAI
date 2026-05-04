from __future__ import annotations

import json
import time
import psutil
import yaml

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .utils.config_loader import load_global_config, get_config_section
from .utils.evaluators_calculations import EvaluatorsCalculations
from .utils.evaluation_errors import (ConfigLoadError, EvaluationError, MetricCalculationError,
                                      MemoryAccessError, VisualizationError, ComparisonError,
                                      ReportGenerationError, ValidationFailureError)
from .modules.report import get_visualizer
from .evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Resource Utilization Evaluator")
printer = PrettyPrinter

MODULE_VERSION = "2.0.0"
_METRIC_ORDER = ("cpu", "memory", "disk", "network")


@dataclass(slots=True)
class ResourceSnapshot:
    """Single sampled system-resource snapshot."""

    sampled_at: str
    sample_window_seconds: float
    cpu: float
    memory: float
    disk: float
    network: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResourceEvaluationResult:
    """Structured output returned by the resource utilization evaluator."""

    metadata: Dict[str, Any]
    metrics: Dict[str, float]
    scores: Dict[str, float]
    health_status: Dict[str, str]
    weighted_score: float
    threshold_violations: Dict[str, float]
    violation_events: List[Dict[str, Any]]
    historical_summary: Dict[str, Any]
    baseline_comparison: Dict[str, Dict[str, Any]]
    diagnostics: Dict[str, Any]
    recommendations: List[str]
    memory_entry_id: Optional[str] = None
    report: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "metrics": dict(self.metrics),
            "scores": dict(self.scores),
            "health_status": dict(self.health_status),
            "weighted_score": self.weighted_score,
            "threshold_violations": dict(self.threshold_violations),
            "violation_events": list(self.violation_events),
            "historical_summary": dict(self.historical_summary),
            "baseline_comparison": {k: dict(v) for k, v in self.baseline_comparison.items()},
            "diagnostics": dict(self.diagnostics),
            "recommendations": list(self.recommendations),
            "memory_entry_id": self.memory_entry_id,
            "report": self.report,
        }


class ResourceUtilizationEvaluator:
    """
    Production-grade resource utilization evaluator.

    Responsibilities
    ----------------
    - Sample CPU, memory, disk, and network utilization safely
    - Calculate normalized efficiency scores using the shared calculations layer
    - Produce structured threshold violations and health states
    - Retain bounded historical snapshots for trend analysis
    - Persist results into evaluator memory and integrate with shared reporting
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))
        self.section = get_config_section("resource_utilization_evaluator")
        if not isinstance(self.section, Mapping):
            raise ConfigLoadError(
                config_path=self.config_path,
                section="resource_utilization_evaluator",
                error_details="Section must be a mapping.",
            )

        self.monitor_duration = self._require_positive_number(
            self.section.get("monitor_duration", 3),
            "resource_utilization_evaluator.monitor_duration",
        )
        self.enable_historical = bool(self.section.get("enable_historical", True))
        self.store_results = bool(self.section.get("store_results", True))
        self.thresholds = self._normalize_threshold_mapping(
            self.section.get("thresholds", {}),
            "resource_utilization_evaluator.thresholds",
        )
        self.weights = self._normalize_weight_mapping(
            self.section.get("weights", {}),
            "resource_utilization_evaluator.weights",
        )

        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()
        self.history: List[ResourceSnapshot] = []
        self.disabled = False

        logger.info(
            "Resource Utilization Evaluator initialized: duration=%s historical=%s store=%s",
            self.monitor_duration,
            self.enable_historical,
            self.store_results,
        )

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        report: bool = False,
        *,
        report_format: str = "markdown",
        baseline_metrics: Optional[Mapping[str, Any]] = None,
        sample_metadata: Optional[Mapping[str, Any]] = None,
        store_result: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if self.disabled:
            raise ValidationFailureError(
                rule_name="resource_evaluator_enabled",
                data=False,
                expected=True,
            )

        persist = self.store_results if store_result is None else bool(store_result)
        metadata = dict(sample_metadata or {})

        snapshot = self._gather_metrics(metadata=metadata)
        resource_analysis = self.calculations.calculate_resource_efficiency(snapshot.to_dict())
        baseline_comparison = self.compare_with_baseline(snapshot.to_dict(), baseline_metrics or {})
        health_status = self._build_health_status(snapshot.to_dict())
        historical_summary = self._build_historical_summary()
        diagnostics = self._build_diagnostics(snapshot, resource_analysis, baseline_comparison, health_status)
        recommendations = self._generate_recommendations(snapshot.to_dict(), resource_analysis, health_status)

        result = ResourceEvaluationResult(
            metadata={
                "evaluated_at": _utcnow().isoformat(),
                "module_version": MODULE_VERSION,
                "config_path": self.config_path,
                "historical_enabled": self.enable_historical,
                "store_result": persist,
                "sample_window_seconds": snapshot.sample_window_seconds,
                "caller_metadata": metadata,
            },
            metrics={key: float(snapshot.to_dict()[key]) for key in _METRIC_ORDER},
            scores=dict(resource_analysis["scores"]),
            health_status=health_status,
            weighted_score=float(resource_analysis["weighted_score"]),
            threshold_violations=dict(resource_analysis["threshold_violations"]),
            violation_events=list(resource_analysis["violation_events"]),
            historical_summary=historical_summary,
            baseline_comparison=baseline_comparison,
            diagnostics=diagnostics,
            recommendations=recommendations,
        )

        self._record_history(snapshot)
        self._update_visualizer(result)

        if persist:
            result.memory_entry_id = self._store_result(result)

        payload = result.to_dict()
        if report:
            payload["report"] = self.generate_report(
                payload,
                format=report_format,
            )
        return payload

    def compare_with_baseline(
        self,
        current_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        if not baseline_metrics:
            return {}
        if not isinstance(current_metrics, Mapping) or not isinstance(baseline_metrics, Mapping):
            raise ComparisonError(
                baseline="resource_baseline",
                current="resource_current",
                error_details="Current and baseline metrics must both be mappings.",
            )

        comparison: Dict[str, Dict[str, Any]] = {}
        for metric in _METRIC_ORDER:
            if metric not in current_metrics or metric not in baseline_metrics:
                continue
            current_value = self._require_non_negative_number(current_metrics[metric], f"current.{metric}")
            baseline_value = self._require_non_negative_number(baseline_metrics[metric], f"baseline.{metric}")
            difference = current_value - baseline_value
            comparison[metric] = {
                "current": current_value,
                "baseline": baseline_value,
                "absolute_difference": difference,
                "relative_change": (difference / baseline_value) if baseline_value > 0 else None,
                "improvement": current_value <= baseline_value,
                "direction": "lower_is_better",
            }
        return comparison

    def generate_report(
        self,
        results: Mapping[str, Any],
        *,
        format: str = "markdown",
    ) -> Any:
        if not isinstance(results, Mapping):
            raise ReportGenerationError(
                report_type="Resource Utilization",
                template="resource_utilization_report",
                error_details="Results must be a mapping.",
            )

        normalized = str(format).strip().lower()
        if normalized == "dict":
            return dict(results)
        if normalized == "json":
            return json.dumps(dict(results), indent=2, sort_keys=False, default=str)
        if normalized == "yaml":
            return yaml.safe_dump(dict(results), default_flow_style=False, sort_keys=False)
        if normalized != "markdown":
            raise ReportGenerationError(
                report_type="Resource Utilization",
                template="resource_utilization_report",
                error_details=f"Unsupported report format: {format}",
            )

        try:
            report: List[str] = [
                "# Resource Utilization Report",
                f"**Generated**: {results['metadata']['evaluated_at']}",
                "",
                "## Executive Summary",
                f"- **Composite Efficiency Score**: {results['weighted_score']:.3f}/1.0",
                f"- **Violating Metrics**: {len(results.get('threshold_violations', {}))}",
                f"- **Historical Samples Retained**: {results.get('historical_summary', {}).get('samples_retained', 0)}",
                "",
                "## Resource Metrics",
            ]

            metrics = results.get("metrics", {})
            scores = results.get("scores", {})
            health_status = results.get("health_status", {})
            for metric in _METRIC_ORDER:
                unit = "Mbps" if metric == "network" else "%"
                report.append(
                    f"- **{metric.upper()}**: {metrics.get(metric, 0):.2f}{unit} | "
                    f"Threshold: {self.thresholds.get(metric, 0):.2f}{unit} | "
                    f"Score: {scores.get(metric, 0):.3f} | "
                    f"Status: {health_status.get(metric, 'UNKNOWN')}"
                )

            if results.get("baseline_comparison"):
                report.extend(["", "## Baseline Comparison"])
                for metric, entry in results["baseline_comparison"].items():
                    arrow = "↓" if entry.get("improvement") else "↑"
                    change = entry.get("relative_change")
                    change_text = f"{change:.2%}" if isinstance(change, (int, float)) else "n/a"
                    report.append(
                        f"- **{metric.upper()}**: current={entry['current']:.2f}, baseline={entry['baseline']:.2f}, "
                        f"delta={entry['absolute_difference']:.2f}, relative={change_text} {arrow}"
                    )

            if results.get("threshold_violations"):
                report.extend(["", "## Threshold Violations"])
                for metric, value in results["threshold_violations"].items():
                    report.append(
                        f"- **{metric.upper()}** exceeded threshold: observed={value:.2f}, limit={self.thresholds.get(metric, 0):.2f}"
                    )

            diagnostics = results.get("diagnostics", {})
            report.extend([
                "",
                "## Diagnostics",
                f"- **Network Burst Indicator**: {diagnostics.get('network_burst_indicator', 0):.3f}",
                f"- **Max Utilization Ratio**: {diagnostics.get('max_utilization_ratio', 0):.3f}",
                f"- **Pressure Index**: {diagnostics.get('pressure_index', 0):.3f}",
            ])

            if results.get("recommendations"):
                report.extend(["", "## Recommendations"])
                for recommendation in results["recommendations"]:
                    report.append(f"- {recommendation}")

            report.extend(["", f"*Report generated by {self.__class__.__name__}*"])
            return "\n".join(report)
        except Exception as exc:
            raise ReportGenerationError(
                report_type="Resource Utilization",
                template="resource_utilization_report",
                error_details=str(exc),
            ) from exc

    def disable_temporarily(self) -> None:
        self.disabled = True
        logger.warning("Resource Utilization Evaluator temporarily disabled.")

    def enable(self) -> None:
        self.disabled = False
        logger.info("Resource Utilization Evaluator re-enabled.")

    # ------------------------------------------------------------------
    # Metric acquisition
    # ------------------------------------------------------------------

    def _gather_metrics(self, metadata: Optional[Mapping[str, Any]] = None) -> ResourceSnapshot:
        metadata_map = dict(metadata or {})
        sample_window = self.monitor_duration
        start_network = psutil.net_io_counters()
        cpu_percent = psutil.cpu_percent(interval=sample_window)
        end_network = psutil.net_io_counters()

        memory_percent = float(psutil.virtual_memory().percent)
        disk_percent = float(psutil.disk_usage("/").percent)
        network_mbps = self._calculate_network_mbps(start_network, end_network, sample_window)

        return ResourceSnapshot(
            sampled_at=_utcnow().isoformat(),
            sample_window_seconds=sample_window,
            cpu=round(float(cpu_percent), 4),
            memory=round(memory_percent, 4),
            disk=round(disk_percent, 4),
            network=round(network_mbps, 4),
            metadata=metadata_map,
        )

    def _calculate_network_mbps(self, start: Any, end: Any, duration: float) -> float:
        try:
            total_bytes = float((end.bytes_sent - start.bytes_sent) + (end.bytes_recv - start.bytes_recv))
            total_megabits = (total_bytes * 8.0) / 1_000_000.0
            return total_megabits / max(duration, 1e-9)
        except Exception as exc:
            raise MetricCalculationError(
                metric_name="network_utilization",
                inputs={"duration": duration},
                reason=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Diagnostics, state, and integration
    # ------------------------------------------------------------------

    def _build_health_status(self, metrics: Mapping[str, float]) -> Dict[str, str]:
        status: Dict[str, str] = {}
        for metric in _METRIC_ORDER:
            value = float(metrics.get(metric, 0.0))
            threshold = float(self.thresholds[metric])
            if value > threshold:
                status[metric] = "CRITICAL"
            elif value > threshold * 0.85:
                status[metric] = "WARNING"
            else:
                status[metric] = "NORMAL"
        return status

    def _record_history(self, snapshot: ResourceSnapshot) -> None:
        if not self.enable_historical:
            return
        self.history.append(snapshot)
        max_history = 200
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

    def _build_historical_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"samples_retained": 0, "averages": {}, "peaks": {}, "latest": None}

        averages = {
            metric: round(sum(getattr(record, metric) for record in self.history) / len(self.history), 6)
            for metric in _METRIC_ORDER
        }
        peaks = {
            metric: round(max(getattr(record, metric) for record in self.history), 6)
            for metric in _METRIC_ORDER
        }
        return {
            "samples_retained": len(self.history),
            "averages": averages,
            "peaks": peaks,
            "latest": self.history[-1].to_dict(),
        }

    def _build_diagnostics(
        self,
        snapshot: ResourceSnapshot,
        resource_analysis: Mapping[str, Any],
        baseline_comparison: Mapping[str, Any],
        health_status: Mapping[str, str],
    ) -> Dict[str, Any]:
        metrics = snapshot.to_dict()
        ratios = {
            metric: (float(metrics[metric]) / float(self.thresholds[metric])) if self.thresholds[metric] > 0 else 0.0
            for metric in _METRIC_ORDER
        }
        return {
            "max_utilization_ratio": max(ratios.values()) if ratios else 0.0,
            "pressure_index": round(sum(ratios.values()) / len(ratios), 6) if ratios else 0.0,
            "network_burst_indicator": ratios.get("network", 0.0),
            "critical_metrics": [metric for metric, status in health_status.items() if status == "CRITICAL"],
            "warning_metrics": [metric for metric, status in health_status.items() if status == "WARNING"],
            "violation_event_count": len(resource_analysis.get("violation_events", [])),
            "baseline_metrics_compared": sorted(baseline_comparison.keys()),
        }

    def _generate_recommendations(
        self,
        metrics: Mapping[str, float],
        resource_analysis: Mapping[str, Any],
        health_status: Mapping[str, str],
    ) -> List[str]:
        recommendations: List[str] = []
        for metric, status in health_status.items():
            if status == "CRITICAL":
                if metric == "cpu":
                    recommendations.append("Investigate CPU hotspots, thread saturation, and batching pressure.")
                elif metric == "memory":
                    recommendations.append("Reduce in-memory retention, checkpoint more aggressively, or profile leaks.")
                elif metric == "disk":
                    recommendations.append("Prune retained artifacts and review checkpoint/export growth on disk.")
                elif metric == "network":
                    recommendations.append("Throttle network-heavy telemetry or batch outbound transfers.")
            elif status == "WARNING":
                recommendations.append(f"Monitor {metric.upper()} closely; utilization is approaching the configured threshold.")

        if not recommendations:
            recommendations.append("Resource utilization is within configured bounds; continue routine monitoring.")

        if resource_analysis.get("weighted_score", 0.0) < 0.5:
            recommendations.append("Composite resource efficiency is low; consider scaling or reducing workload concurrency.")
        return recommendations

    def _store_result(self, result: ResourceEvaluationResult) -> str:
        try:
            priority = "high" if result.threshold_violations else "medium"
            return self.memory.add(
                entry=result.to_dict(),
                tags=["resource_analysis", "resource_utilization"],
                priority=priority,
                category="resource_utilization",
                source=self.__class__.__name__,
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise MemoryAccessError(
                operation="add",
                key="resource_utilization_result",
                error_details=str(exc),
            ) from exc

    def _update_visualizer(self, result: ResourceEvaluationResult) -> None:
        try:
            visualizer = get_visualizer()
            visualizer.update_metrics({
                "reward": result.weighted_score,
                "risk": max(float(result.metrics[metric]) / float(self.thresholds[metric]) for metric in _METRIC_ORDER),
                "operational_time": result.metadata["sample_window_seconds"],
                "pass_rate": max(0.0, min(1.0, result.weighted_score)),
            })
        except Exception as exc:
            logger.warning("Visualizer update skipped for resource evaluator: %s", exc)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _normalize_threshold_mapping(self, value: Any, field_name: str) -> Dict[str, float]:
        if not isinstance(value, Mapping):
            raise ConfigLoadError(
                config_path=self.config_path,
                section=field_name,
                error_details="Thresholds must be a mapping.",
            )
        normalized: Dict[str, float] = {}
        for metric in _METRIC_ORDER:
            if metric not in value:
                raise ConfigLoadError(
                    config_path=self.config_path,
                    section=field_name,
                    error_details=f"Missing threshold for '{metric}'.",
                )
            normalized[metric] = self._require_positive_number(value[metric], f"{field_name}.{metric}")
        return normalized

    def _normalize_weight_mapping(self, value: Any, field_name: str) -> Dict[str, float]:
        if not isinstance(value, Mapping):
            raise ConfigLoadError(
                config_path=self.config_path,
                section=field_name,
                error_details="Weights must be a mapping.",
            )
        normalized: Dict[str, float] = {}
        for metric in _METRIC_ORDER:
            normalized[metric] = self._require_non_negative_number(value.get(metric, 0.0), f"{field_name}.{metric}")
        total = sum(normalized.values())
        if total <= 0:
            return {"cpu": 0.25, "memory": 0.25, "disk": 0.25, "network": 0.25}
        return {key: val / total for key, val in normalized.items()}

    @staticmethod
    def _require_positive_number(value: Any, field_name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "positive number") from exc
        if number <= 0:
            raise ValidationFailureError(field_name, number, "positive number")
        return number

    @staticmethod
    def _require_non_negative_number(value: Any, field_name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailureError(field_name, value, "non-negative number") from exc
        if number < 0:
            raise ValidationFailureError(field_name, number, "non-negative number")
        return number



def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Resource Utilization Evaluator ===\n")
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    
    try:
        evaluator = ResourceUtilizationEvaluator()
        results = evaluator.evaluate(report=True)
        
        if 'report' in results:
            print(results['report'])
        elif 'error' in results:
            printer.pretty("Evaluation failed", results, "error")
            
    except Exception as e:
        printer.pretty("Fatal error during evaluation", str(e), "error")
    
    print("\n=== Resource Utilization Evaluation Complete ===\n")
