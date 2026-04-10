"""
- Throughput/latency histograms.
- Queue depth and backlog trend detection.
- Resource pressure indicators (CPU/memory/GPU where available).

This module is the runtime analysis layer for the Observability Agent's capacity
and performance responsibilities. It turns raw queue/resource telemetry and
completed execution traces into operational signals that higher-level incident
and planning components can consume.

Design goals:
- Keep capacity analysis and performance analysis separate, but compatible.
- Preserve lightweight ingestion while still producing rich summaries.
- Use the observability error model for invalid input, pipeline failures, and
  objective breaches rather than silently swallowing problems.
- Integrate with the waterfall analyzer for trace-level latency attribution.
- Degrade safely when optional memory capabilities are not available yet.
"""

from __future__ import annotations

import math
import time

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .utils import (load_global_config, get_config_section,
                    # Error handling
                    MetricCardinalityError, ObservabilityError, ObservabilityErrorType,
                    ObservabilitySeverity, SLOBreachError, normalize_observability_exception,
                    # Waterfall
                    WaterfallAnalyzer, summarize_waterfall)
from .observability_memory import ObservabilityMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Observability Capacity & Performance")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------
@dataclass
class CapacityAlert:
    scope: str
    subject: str
    level: str
    signal: str
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueueSample:
    queue_name: str
    depth: float
    timestamp_ms: float
    inflow_per_sec: Optional[float] = None
    outflow_per_sec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceSample:
    resource_name: str
    resource_type: str
    utilization_pct: float
    timestamp_ms: float
    used: Optional[float] = None
    capacity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatencySample:
    subject: str
    duration_ms: float
    timestamp_ms: float
    trace_id: Optional[str] = None
    status: str = "ok"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThroughputSample:
    subject: str
    count: int
    timestamp_ms: float
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TracePerformanceRecord:
    trace_id: str
    timestamp_ms: float
    total_duration_ms: float
    critical_path_ms: float
    bottleneck_count: int
    anomaly_count: int
    retry_chain_count: int
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionAssessment:
    subject: str
    metric_name: str
    baseline_value: float
    recent_value: float
    delta: float
    delta_ratio: float
    level: str
    direction: str
    enough_data: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _now_ms() -> float:
    return time.time() * 1000.0


def _safe_mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    return {str(key): payload for key, payload in value.items()}


def _coerce_non_empty_str(value: Any, *, field_name: str, operation: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ObservabilityError(
            message=f"{operation} requires a non-empty '{field_name}'",
            error_type=ObservabilityErrorType.METRIC_PIPELINE_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            context={"field_name": field_name, "operation": operation},
            remediation=f"Provide a valid {field_name} before invoking {operation}.",
        )
    return text


def _coerce_non_negative_float(value: Any, *, field_name: str, operation: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservabilityError(
            message=f"{operation} received non-numeric value for '{field_name}': {value!r}",
            error_type=ObservabilityErrorType.METRIC_PIPELINE_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            context={"field_name": field_name, "value": value, "operation": operation},
            remediation=f"Provide a numeric value for {field_name}.",
        ) from exc

    if number < 0:
        raise ObservabilityError(
            message=f"{operation} received negative value for '{field_name}': {number}",
            error_type=ObservabilityErrorType.METRIC_PIPELINE_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            context={"field_name": field_name, "value": number, "operation": operation},
            remediation=f"Provide a non-negative value for {field_name}.",
        )
    return number


def _coerce_utilization_pct(value: Any, *, field_name: str, operation: str) -> float:
    pct = _coerce_non_negative_float(value, field_name=field_name, operation=operation)
    if pct > 100.0:
        raise ObservabilityError(
            message=f"{operation} received utilization percentage above 100 for '{field_name}': {pct}",
            error_type=ObservabilityErrorType.METRIC_PIPELINE_FAILED,
            severity=ObservabilitySeverity.HIGH,
            retryable=False,
            context={"field_name": field_name, "value": pct, "operation": operation},
            remediation=f"Normalize {field_name} to a 0..100 percentage.",
        )
    return pct


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(float(value) for value in values)
    percentile = min(100.0, max(0.0, float(percentile)))
    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _bucketize(value: float, buckets: Sequence[float]) -> str:
    for bucket in buckets:
        if value <= bucket:
            return f"<= {bucket:g}ms"
    if not buckets:
        return "> unbounded"
    return f"> {float(buckets[-1]):g}ms"


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0 if numerator <= 0 else float("inf")
    return numerator / denominator


# ---------------------------------------------------------------------------
# Capacity analysis
# ---------------------------------------------------------------------------
class ObservabilityCapacity:
    def __init__(self, memory: Optional[ObservabilityMemory] = None) -> None:
        self.config = load_global_config()
        self.capacity_config = get_config_section("observability_capacity")
        self._lock = RLock()

        self.enabled = bool(self.capacity_config.get("enabled", True))
        self.enable_memory_integration = bool(self.capacity_config.get("enable_memory_integration", False))
        self.queue_history_limit = int(self.capacity_config.get("queue_history_limit", 512))
        self.resource_history_limit = int(self.capacity_config.get("resource_history_limit", 512))
        self.max_queue_series = int(self.capacity_config.get("max_queue_series", 128))
        self.max_resource_series = int(self.capacity_config.get("max_resource_series", 64))
        self.sustained_growth_min_points = int(self.capacity_config.get("sustained_growth_min_points", 3))
        self.stale_after_ms = float(self.capacity_config.get("stale_after_ms", 300_000.0))

        self.queue_depth_warning = float(self.capacity_config.get("queue_depth_warning", 50.0))
        self.queue_depth_critical = float(self.capacity_config.get("queue_depth_critical", 200.0))
        self.backlog_growth_ratio_warning = float(self.capacity_config.get("backlog_growth_ratio_warning", 1.25))
        self.backlog_growth_ratio_critical = float(self.capacity_config.get("backlog_growth_ratio_critical", 1.75))
        self.backlog_slope_warning_per_min = float(self.capacity_config.get("backlog_slope_warning_per_min", 10.0))
        self.backlog_slope_critical_per_min = float(self.capacity_config.get("backlog_slope_critical_per_min", 30.0))
        self.drain_ratio_warning = float(self.capacity_config.get("drain_ratio_warning", 0.9))
        self.drain_ratio_critical = float(self.capacity_config.get("drain_ratio_critical", 0.75))

        thresholds = self.capacity_config.get("resource_pressure_thresholds", {}) or {}
        self.resource_pressure_thresholds: Dict[str, Dict[str, float]] = {}
        for resource_type, resource_thresholds in thresholds.items():
            current = dict(resource_thresholds or {})
            self.resource_pressure_thresholds[str(resource_type).lower()] = {
                "warning": float(current.get("warning", 75.0)),
                "critical": float(current.get("critical", 90.0)),
            }

        self.memory = memory
        if self.memory is None and self.enable_memory_integration:
            self.memory = ObservabilityMemory()

        self._queue_samples: Dict[str, Deque[QueueSample]] = defaultdict(
            lambda: deque(maxlen=self.queue_history_limit)
        )
        self._resource_samples: Dict[str, Deque[ResourceSample]] = defaultdict(
            lambda: deque(maxlen=self.resource_history_limit)
        )

    def record_queue_sample(
        self,
        queue_name: str,
        *,
        depth: float,
        inflow_per_sec: Optional[float] = None,
        outflow_per_sec: Optional[float] = None,
        timestamp_ms: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="record_queue_sample")
            queue_name = _coerce_non_empty_str(queue_name, field_name="queue_name", operation="record_queue_sample")
            depth = _coerce_non_negative_float(depth, field_name="depth", operation="record_queue_sample")
            timestamp = float(timestamp_ms if timestamp_ms is not None else _now_ms())
            sample = QueueSample(
                queue_name=queue_name,
                depth=depth,
                timestamp_ms=timestamp,
                inflow_per_sec=(
                    _coerce_non_negative_float(
                        inflow_per_sec,
                        field_name="inflow_per_sec",
                        operation="record_queue_sample",
                    )
                    if inflow_per_sec is not None
                    else None
                ),
                outflow_per_sec=(
                    _coerce_non_negative_float(
                        outflow_per_sec,
                        field_name="outflow_per_sec",
                        operation="record_queue_sample",
                    )
                    if outflow_per_sec is not None
                    else None
                ),
                metadata=_safe_mapping(metadata),
            )

            with self._lock:
                if queue_name not in self._queue_samples and len(self._queue_samples) >= self.max_queue_series:
                    raise MetricCardinalityError(
                        metric_name="observability_capacity.queue_name",
                        cardinality=len(self._queue_samples) + 1,
                        limit=self.max_queue_series,
                        context={"queue_name": queue_name},
                    )
                self._queue_samples[queue_name].append(sample)
                report = self._build_queue_report_locked(queue_name)

            self._emit_capacity_alert_events(report)
            return report
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="capacity.queue",
                context={"queue_name": queue_name},
            ) from exc

    def record_resource_sample(
        self,
        resource_name: str,
        *,
        utilization_pct: float,
        resource_type: str,
        used: Optional[float] = None,
        capacity: Optional[float] = None,
        timestamp_ms: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="record_resource_sample")
            resource_name = _coerce_non_empty_str(
                resource_name,
                field_name="resource_name",
                operation="record_resource_sample",
            )
            resource_type = _coerce_non_empty_str(
                resource_type,
                field_name="resource_type",
                operation="record_resource_sample",
            ).lower()
            utilization_pct = _coerce_utilization_pct(
                utilization_pct,
                field_name="utilization_pct",
                operation="record_resource_sample",
            )
            timestamp = float(timestamp_ms if timestamp_ms is not None else _now_ms())
            sample = ResourceSample(
                resource_name=resource_name,
                resource_type=resource_type,
                utilization_pct=utilization_pct,
                timestamp_ms=timestamp,
                used=(
                    _coerce_non_negative_float(used, field_name="used", operation="record_resource_sample")
                    if used is not None
                    else None
                ),
                capacity=(
                    _coerce_non_negative_float(capacity, field_name="capacity", operation="record_resource_sample")
                    if capacity is not None
                    else None
                ),
                metadata=_safe_mapping(metadata),
            )

            with self._lock:
                if resource_name not in self._resource_samples and len(self._resource_samples) >= self.max_resource_series:
                    raise MetricCardinalityError(
                        metric_name="observability_capacity.resource_name",
                        cardinality=len(self._resource_samples) + 1,
                        limit=self.max_resource_series,
                        context={"resource_name": resource_name},
                    )
                self._resource_samples[resource_name].append(sample)
                report = self._build_resource_report_locked(resource_name)

            self._emit_capacity_alert_events(report)
            return report
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="capacity.resource",
                context={"resource_name": resource_name},
            ) from exc

    def get_queue_report(self, queue_name: str) -> Dict[str, Any]:
        try:
            queue_name = _coerce_non_empty_str(queue_name, field_name="queue_name", operation="get_queue_report")
            with self._lock:
                return self._build_queue_report_locked(queue_name)
        except Exception as exc:
            raise self._handle_exception(exc, stage="capacity.queue", context={"queue_name": queue_name}) from exc

    def get_resource_report(self, resource_name: str) -> Dict[str, Any]:
        try:
            resource_name = _coerce_non_empty_str(
                resource_name,
                field_name="resource_name",
                operation="get_resource_report",
            )
            with self._lock:
                return self._build_resource_report_locked(resource_name)
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="capacity.resource",
                context={"resource_name": resource_name},
            ) from exc

    def summarize_capacity(self) -> Dict[str, Any]:
        try:
            with self._lock:
                queue_reports = [self._build_queue_report_locked(name) for name in sorted(self._queue_samples)]
                resource_reports = [self._build_resource_report_locked(name) for name in sorted(self._resource_samples)]

            alerts: List[Dict[str, Any]] = []
            warning_count = 0
            critical_count = 0
            for report in queue_reports + resource_reports:
                for alert in report.get("alerts", []):
                    alerts.append(alert)
                    if alert["level"] == "critical":
                        critical_count += 1
                    elif alert["level"] == "warning":
                        warning_count += 1

            return {
                "queue_count": len(queue_reports),
                "resource_count": len(resource_reports),
                "warning_alert_count": warning_count,
                "critical_alert_count": critical_count,
                "pressure_queues": [report["queue_name"] for report in queue_reports if report["status"] != "ok"],
                "saturated_resources": [
                    report["resource_name"] for report in resource_reports if report["status"] != "ok"
                ],
                "queue_reports": queue_reports,
                "resource_reports": resource_reports,
                "alerts": alerts,
            }
        except Exception as exc:
            raise self._handle_exception(exc, stage="capacity.queue", context={"operation": "summarize_capacity"}) from exc

    def _build_queue_report_locked(self, queue_name: str) -> Dict[str, Any]:
        samples = list(self._queue_samples.get(queue_name, ()))
        if not samples:
            return {
                "queue_name": queue_name,
                "sample_count": 0,
                "status": "unknown",
                "alerts": [],
            }

        depths = [sample.depth for sample in samples]
        latest = samples[-1]
        first = samples[0]
        elapsed_ms = max(0.0, latest.timestamp_ms - first.timestamp_ms)
        elapsed_minutes = elapsed_ms / 60_000.0 if elapsed_ms > 0 else 0.0
        depth_delta = latest.depth - first.depth
        growth_ratio = _ratio(latest.depth, first.depth if first.depth > 0 else max(latest.depth, 1.0))
        backlog_slope_per_min = depth_delta / elapsed_minutes if elapsed_minutes > 0 else 0.0

        inflow_values = [sample.inflow_per_sec for sample in samples if sample.inflow_per_sec is not None]
        outflow_values = [sample.outflow_per_sec for sample in samples if sample.outflow_per_sec is not None]
        avg_inflow = sum(inflow_values) / len(inflow_values) if inflow_values else 0.0
        avg_outflow = sum(outflow_values) / len(outflow_values) if outflow_values else 0.0
        drain_ratio = _ratio(avg_outflow, avg_inflow) if avg_inflow > 0 else 1.0

        increasing_pairs = sum(
            1 for previous, current in zip(samples, samples[1:]) if current.depth >= previous.depth
        )
        sustained_growth = (
            len(samples) >= self.sustained_growth_min_points
            and increasing_pairs >= len(samples[1:])
        )
        stale = (_now_ms() - latest.timestamp_ms) > self.stale_after_ms

        alerts: List[CapacityAlert] = []
        if latest.depth >= self.queue_depth_critical:
            alerts.append(
                self._make_capacity_alert(
                    scope="queue",
                    subject=queue_name,
                    level="critical",
                    signal="queue_depth",
                    message=f"Queue '{queue_name}' depth is critically high ({latest.depth:.2f}).",
                    metrics={"depth": latest.depth, "threshold": self.queue_depth_critical},
                )
            )
        elif latest.depth >= self.queue_depth_warning:
            alerts.append(
                self._make_capacity_alert(
                    scope="queue",
                    subject=queue_name,
                    level="warning",
                    signal="queue_depth",
                    message=f"Queue '{queue_name}' depth is elevated ({latest.depth:.2f}).",
                    metrics={"depth": latest.depth, "threshold": self.queue_depth_warning},
                )
            )

        if sustained_growth and backlog_slope_per_min >= self.backlog_slope_critical_per_min:
            alerts.append(
                self._make_capacity_alert(
                    scope="queue",
                    subject=queue_name,
                    level="critical",
                    signal="backlog_growth",
                    message=(
                        f"Queue '{queue_name}' backlog is growing rapidly "
                        f"({backlog_slope_per_min:.2f} items/min)."
                    ),
                    metrics={
                        "backlog_slope_per_min": backlog_slope_per_min,
                        "depth_delta": depth_delta,
                        "growth_ratio": growth_ratio,
                    },
                )
            )
        elif sustained_growth and (
            backlog_slope_per_min >= self.backlog_slope_warning_per_min
            or growth_ratio >= self.backlog_growth_ratio_warning
        ):
            alerts.append(
                self._make_capacity_alert(
                    scope="queue",
                    subject=queue_name,
                    level="warning",
                    signal="backlog_growth",
                    message=(
                        f"Queue '{queue_name}' backlog is trending upward "
                        f"({backlog_slope_per_min:.2f} items/min)."
                    ),
                    metrics={
                        "backlog_slope_per_min": backlog_slope_per_min,
                        "depth_delta": depth_delta,
                        "growth_ratio": growth_ratio,
                    },
                )
            )

        if inflow_values and outflow_values:
            if drain_ratio <= self.drain_ratio_critical:
                alerts.append(
                    self._make_capacity_alert(
                        scope="queue",
                        subject=queue_name,
                        level="critical",
                        signal="drain_ratio",
                        message=(
                            f"Queue '{queue_name}' drain ratio is critically low "
                            f"({drain_ratio:.2f}); outflow is not keeping pace with inflow."
                        ),
                        metrics={
                            "avg_inflow_per_sec": avg_inflow,
                            "avg_outflow_per_sec": avg_outflow,
                            "drain_ratio": drain_ratio,
                        },
                    )
                )
            elif drain_ratio <= self.drain_ratio_warning:
                alerts.append(
                    self._make_capacity_alert(
                        scope="queue",
                        subject=queue_name,
                        level="warning",
                        signal="drain_ratio",
                        message=(
                            f"Queue '{queue_name}' drain ratio is below target ({drain_ratio:.2f}); "
                            "backlog risk is increasing."
                        ),
                        metrics={
                            "avg_inflow_per_sec": avg_inflow,
                            "avg_outflow_per_sec": avg_outflow,
                            "drain_ratio": drain_ratio,
                        },
                    )
                )

        if stale:
            alerts.append(
                self._make_capacity_alert(
                    scope="queue",
                    subject=queue_name,
                    level="warning",
                    signal="stale_telemetry",
                    message=(
                        f"Queue '{queue_name}' telemetry is stale; no recent sample within "
                        f"{self.stale_after_ms / 1000.0:.0f}s."
                    ),
                    metrics={"age_ms": _now_ms() - latest.timestamp_ms},
                )
            )

        status = "critical" if any(alert.level == "critical" for alert in alerts) else "warning" if alerts else "ok"
        return {
            "queue_name": queue_name,
            "sample_count": len(samples),
            "latest_depth": latest.depth,
            "avg_depth": sum(depths) / len(depths),
            "max_depth": max(depths),
            "depth_delta": depth_delta,
            "growth_ratio": growth_ratio,
            "backlog_slope_per_min": backlog_slope_per_min,
            "avg_inflow_per_sec": avg_inflow,
            "avg_outflow_per_sec": avg_outflow,
            "drain_ratio": drain_ratio,
            "sustained_growth": sustained_growth,
            "latest_timestamp_ms": latest.timestamp_ms,
            "stale": stale,
            "status": status,
            "alerts": [alert.to_dict() for alert in alerts],
        }

    def _build_resource_report_locked(self, resource_name: str) -> Dict[str, Any]:
        samples = list(self._resource_samples.get(resource_name, ()))
        if not samples:
            return {
                "resource_name": resource_name,
                "sample_count": 0,
                "status": "unknown",
                "alerts": [],
            }

        latest = samples[-1]
        values = [sample.utilization_pct for sample in samples]
        thresholds = self.resource_pressure_thresholds.get(
            latest.resource_type,
            {"warning": 75.0, "critical": 90.0},
        )
        latest_delta = latest.utilization_pct - samples[0].utilization_pct
        stale = (_now_ms() - latest.timestamp_ms) > self.stale_after_ms
        alerts: List[CapacityAlert] = []

        if latest.utilization_pct >= thresholds["critical"]:
            alerts.append(
                self._make_capacity_alert(
                    scope="resource",
                    subject=resource_name,
                    level="critical",
                    signal="resource_pressure",
                    message=(
                        f"{latest.resource_type.upper()} resource '{resource_name}' is saturated "
                        f"at {latest.utilization_pct:.2f}% utilization."
                    ),
                    metrics={
                        "utilization_pct": latest.utilization_pct,
                        "threshold": thresholds["critical"],
                    },
                    context={"resource_type": latest.resource_type},
                )
            )
        elif latest.utilization_pct >= thresholds["warning"]:
            alerts.append(
                self._make_capacity_alert(
                    scope="resource",
                    subject=resource_name,
                    level="warning",
                    signal="resource_pressure",
                    message=(
                        f"{latest.resource_type.upper()} resource '{resource_name}' is under pressure "
                        f"at {latest.utilization_pct:.2f}% utilization."
                    ),
                    metrics={
                        "utilization_pct": latest.utilization_pct,
                        "threshold": thresholds["warning"],
                    },
                    context={"resource_type": latest.resource_type},
                )
            )

        if stale:
            alerts.append(
                self._make_capacity_alert(
                    scope="resource",
                    subject=resource_name,
                    level="warning",
                    signal="stale_telemetry",
                    message=(
                        f"Resource telemetry for '{resource_name}' is stale; no recent sample within "
                        f"{self.stale_after_ms / 1000.0:.0f}s."
                    ),
                    metrics={"age_ms": _now_ms() - latest.timestamp_ms},
                    context={"resource_type": latest.resource_type},
                )
            )

        status = "critical" if any(alert.level == "critical" for alert in alerts) else "warning" if alerts else "ok"
        return {
            "resource_name": resource_name,
            "resource_type": latest.resource_type,
            "sample_count": len(samples),
            "latest_utilization_pct": latest.utilization_pct,
            "avg_utilization_pct": sum(values) / len(values),
            "max_utilization_pct": max(values),
            "p95_utilization_pct": _percentile(values, 95.0),
            "utilization_delta_pct": latest_delta,
            "latest_used": latest.used,
            "latest_capacity": latest.capacity,
            "latest_timestamp_ms": latest.timestamp_ms,
            "stale": stale,
            "status": status,
            "alerts": [alert.to_dict() for alert in alerts],
        }

    def _make_capacity_alert(self, *, scope: str, subject: str, level: str, signal: str, message: str,
                             metrics: Optional[Mapping[str, float]] = None, context: Optional[Mapping[str, Any]] = None) -> CapacityAlert:
        return CapacityAlert(
            scope=scope,
            subject=subject,
            level=level,
            signal=signal,
            message=message,
            metrics={str(key): float(value) for key, value in (metrics or {}).items()},
            context=_safe_mapping(context),
        )

    def _emit_capacity_alert_events(self, report: Mapping[str, Any]) -> None:
        if self.memory is None:
            return
        if not hasattr(self.memory, "append_timeline_event"):
            return

        for alert in report.get("alerts", []):
            try:
                self.memory.append_timeline_event(
                    trace_id=f"capacity:{alert['scope']}:{alert['subject']}",
                    event_type=f"capacity.{alert['signal']}",
                    severity=alert["level"],
                    message=alert["message"],
                    payload={
                        "metrics": alert.get("metrics", {}),
                        "context": alert.get("context", {}),
                    },
                )
            except Exception:
                logger.debug("Capacity alert event emission skipped for '%s'.", alert.get("subject"))

    def _ensure_enabled(self, *, operation: str) -> None:
        if not self.enabled:
            raise ObservabilityError(
                message=f"ObservabilityCapacity is disabled; cannot execute '{operation}'",
                error_type=ObservabilityErrorType.METRIC_PIPELINE_FAILED,
                severity=ObservabilitySeverity.MEDIUM,
                retryable=False,
                context={"operation": operation},
                remediation="Enable observability_capacity in observability_config.yaml.",
            )

    def _handle_exception(
        self,
        exc: Exception,
        *,
        stage: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ObservabilityError:
        if isinstance(exc, ObservabilityError):
            return exc
        return normalize_observability_exception(exc, stage=stage, context=context)


# ---------------------------------------------------------------------------
# Performance analysis
# ---------------------------------------------------------------------------
class ObservabilityPerformance:
    def __init__(self, memory: Optional[ObservabilityMemory] = None) -> None:
        self.config = load_global_config()
        self.performance_config = get_config_section("observability_performance")
        self._lock = RLock()

        self.enabled = bool(self.performance_config.get("enabled", True))
        self.enable_memory_integration = bool(self.performance_config.get("enable_memory_integration", True))
        self.max_trace_records = int(self.performance_config.get("max_trace_records", 2000))
        self.max_latency_samples_per_subject = int(
            self.performance_config.get("max_latency_samples_per_subject", 10000)
        )
        self.max_throughput_samples_per_subject = int(
            self.performance_config.get("max_throughput_samples_per_subject", 5000)
        )
        self.max_subjects = int(self.performance_config.get("max_subjects", 256))
        self.min_regression_samples = int(self.performance_config.get("min_regression_samples", 5))
        self.recent_sample_window = int(self.performance_config.get("recent_sample_window", 20))
        self.baseline_sample_window = int(self.performance_config.get("baseline_sample_window", 50))
        self.histogram_buckets_ms = [
            float(value) for value in self.performance_config.get(
                "histogram_buckets_ms",
                [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
            )
        ]
        self.percentiles = [int(value) for value in self.performance_config.get("percentiles", [50, 90, 95, 99])]
        self.latency_regression_warning_ratio = float(
            self.performance_config.get("latency_regression_warning_ratio", 1.20)
        )
        self.latency_regression_critical_ratio = float(
            self.performance_config.get("latency_regression_critical_ratio", 1.50)
        )
        self.throughput_regression_warning_ratio = float(
            self.performance_config.get("throughput_regression_warning_ratio", 0.85)
        )
        self.throughput_regression_critical_ratio = float(
            self.performance_config.get("throughput_regression_critical_ratio", 0.65)
        )
        self.default_latency_slo_ms = float(self.performance_config.get("default_latency_slo_ms", 1500.0))
        self.default_slo_percentile = int(self.performance_config.get("default_slo_percentile", 95))
        self.default_throughput_window_seconds = float(
            self.performance_config.get("default_throughput_window_seconds", 300.0)
        )
        self.archive_trace_reports = bool(self.performance_config.get("archive_trace_reports", True))
        self.record_timeline_events = bool(self.performance_config.get("record_timeline_events", True))

        self.memory = memory
        if self.memory is None and self.enable_memory_integration:
            self.memory = ObservabilityMemory()

        self.analyzer = WaterfallAnalyzer()

        self._trace_records: Deque[TracePerformanceRecord] = deque(maxlen=self.max_trace_records)
        self._latency_samples: MutableMapping[str, Deque[LatencySample]] = defaultdict(
            lambda: deque(maxlen=self.max_latency_samples_per_subject)
        )
        self._throughput_samples: MutableMapping[str, Deque[ThroughputSample]] = defaultdict(
            lambda: deque(maxlen=self.max_throughput_samples_per_subject)
        )

    def analyze_trace(self, trace_id: str, spans: Sequence[Mapping[str, Any]], *,
        incident_level: str = "info",
        metadata: Optional[Mapping[str, Any]] = None,
        persist_to_memory: bool = True,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="analyze_trace")
            trace_id = _coerce_non_empty_str(trace_id, field_name="trace_id", operation="analyze_trace")
            if not spans:
                raise ObservabilityError(
                    message=f"analyze_trace requires at least one span for trace '{trace_id}'",
                    error_type=ObservabilityErrorType.TRACE_CONTEXT_MISSING,
                    severity=ObservabilitySeverity.HIGH,
                    retryable=False,
                    context={"trace_id": trace_id},
                    remediation="Supply the completed trace spans before invoking performance analysis.",
                )

            report = self.analyzer.analyze(spans)
            summary = summarize_waterfall(report)
            report_payload = self._waterfall_report_payload(report, summary)
            timestamp_ms = _now_ms()
            record = TracePerformanceRecord(
                trace_id=trace_id,
                timestamp_ms=timestamp_ms,
                total_duration_ms=float(report.total_duration_ms),
                critical_path_ms=float(report.critical_path_ms),
                bottleneck_count=len(report.bottleneck_spans),
                anomaly_count=len(report.anomalies),
                retry_chain_count=len(report.retry_chains),
                summary=report_payload,
                metadata=_safe_mapping(metadata),
            )

            with self._lock:
                self._trace_records.append(record)
                self._record_latency_sample_locked(
                    subject="system",
                    duration_ms=float(report.total_duration_ms),
                    timestamp_ms=timestamp_ms,
                    trace_id=trace_id,
                    status="error" if report.anomalies else "ok",
                    metadata={"source": "trace_total", **_safe_mapping(metadata)},
                )
                self._record_throughput_sample_locked(
                    subject="system",
                    count=1,
                    timestamp_ms=timestamp_ms,
                    success_count=0 if report.anomalies else 1,
                    failure_count=1 if report.anomalies else 0,
                    metadata={"source": "trace_total", **_safe_mapping(metadata)},
                )
                for agent_name, duration_ms in report.per_agent_duration_ms.items():
                    self._record_latency_sample_locked(
                        subject=agent_name,
                        duration_ms=float(duration_ms),
                        timestamp_ms=timestamp_ms,
                        trace_id=trace_id,
                        status="ok",
                        metadata={"source": "waterfall_agent_duration", **_safe_mapping(metadata)},
                    )

            if persist_to_memory:
                self._archive_trace_if_supported(
                    trace_id=trace_id,
                    spans=spans,
                    report_payload=report_payload,
                    incident_level=incident_level,
                    metadata=metadata,
                )

            if self.record_timeline_events:
                self._append_trace_timeline_if_supported(trace_id=trace_id, report_payload=report_payload)

            return {
                "trace_id": trace_id,
                "incident_level": incident_level,
                "waterfall": report_payload,
                "latency_summary": self.latency_summary(subject="system"),
                "throughput_summary": self.throughput_summary(subject="system"),
            }
        except Exception as exc:
            raise self._handle_exception(exc, stage="metrics.pipeline", context={"trace_id": trace_id}) from exc

    def record_latency_sample(self, subject: str, *, duration_ms: float,
        timestamp_ms: Optional[float] = None,
        trace_id: Optional[str] = None,
        status: str = "ok",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="record_latency_sample")
            subject = _coerce_non_empty_str(subject, field_name="subject", operation="record_latency_sample")
            duration_ms = _coerce_non_negative_float(
                duration_ms,
                field_name="duration_ms",
                operation="record_latency_sample",
            )
            timestamp = float(timestamp_ms if timestamp_ms is not None else _now_ms())
            with self._lock:
                self._record_latency_sample_locked(
                    subject=subject,
                    duration_ms=duration_ms,
                    timestamp_ms=timestamp,
                    trace_id=trace_id,
                    status=str(status or "ok").lower(),
                    metadata=_safe_mapping(metadata),
                )
            return self.latency_summary(subject=subject)
        except Exception as exc:
            raise self._handle_exception(exc, stage="metrics.pipeline", context={"subject": subject}) from exc

    def record_throughput_sample(self, subject: str, *, count: int = 1,
        success_count: Optional[int] = None,
        failure_count: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._ensure_enabled(operation="record_throughput_sample")
            subject = _coerce_non_empty_str(subject, field_name="subject", operation="record_throughput_sample")
            count_value = int(_coerce_non_negative_float(count, field_name="count", operation="record_throughput_sample"))
            success_value = int(success_count if success_count is not None else count_value)
            failure_value = int(failure_count if failure_count is not None else max(0, count_value - success_value))
            timestamp = float(timestamp_ms if timestamp_ms is not None else _now_ms())
            with self._lock:
                self._record_throughput_sample_locked(
                    subject=subject,
                    count=count_value,
                    success_count=max(0, success_value),
                    failure_count=max(0, failure_value),
                    timestamp_ms=timestamp,
                    metadata=_safe_mapping(metadata),
                )
            return self.throughput_summary(subject=subject)
        except Exception as exc:
            raise self._handle_exception(exc, stage="metrics.pipeline", context={"subject": subject}) from exc

    def latency_summary(self, *, subject: str = "system") -> Dict[str, Any]:
        try:
            subject = _coerce_non_empty_str(subject, field_name="subject", operation="latency_summary")
            with self._lock:
                samples = list(self._latency_samples.get(subject, ()))
            durations = [sample.duration_ms for sample in samples]
            histogram = self._latency_histogram_from_durations(durations)
            summary = {
                "subject": subject,
                "sample_count": len(samples),
                "histogram": histogram,
                "status_counts": dict(Counter(sample.status for sample in samples)),
            }
            if not durations:
                summary.update(
                    {
                        "mean_ms": 0.0,
                        "min_ms": 0.0,
                        "max_ms": 0.0,
                        "percentiles_ms": {str(p): 0.0 for p in self.percentiles},
                    }
                )
                return summary

            summary.update(
                {
                    "mean_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "percentiles_ms": {str(p): _percentile(durations, p) for p in self.percentiles},
                }
            )
            return summary
        except Exception as exc:
            raise self._handle_exception(exc, stage="performance.latency", context={"subject": subject}) from exc

    def throughput_summary(self, *, subject: str = "system", window_seconds: Optional[float] = None) -> Dict[str, Any]:
        try:
            subject = _coerce_non_empty_str(subject, field_name="subject", operation="throughput_summary")
            window = float(window_seconds if window_seconds is not None else self.default_throughput_window_seconds)
            now_ms = _now_ms()
            with self._lock:
                samples = [
                    sample
                    for sample in self._throughput_samples.get(subject, ())
                    if (now_ms - sample.timestamp_ms) <= window * 1000.0
                ]

            total_count = sum(sample.count for sample in samples)
            success_count = sum(sample.success_count for sample in samples)
            failure_count = sum(sample.failure_count for sample in samples)
            rate_per_sec = total_count / window if window > 0 else 0.0
            return {
                "subject": subject,
                "window_seconds": window,
                "sample_count": len(samples),
                "total_count": total_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "rate_per_sec": rate_per_sec,
                "rate_per_min": rate_per_sec * 60.0,
                "error_rate": _ratio(failure_count, total_count),
            }
        except Exception as exc:
            raise self._handle_exception(exc, stage="performance.throughput", context={"subject": subject}) from exc

    def detect_latency_regression(self, *, subject: str = "system") -> Dict[str, Any]:
        try:
            subject = _coerce_non_empty_str(subject, field_name="subject", operation="detect_latency_regression")
            with self._lock:
                samples = list(self._latency_samples.get(subject, ()))
            return self._assess_latency_regression(subject, samples).to_dict()
        except Exception as exc:
            raise self._handle_exception(exc, stage="performance.latency", context={"subject": subject}) from exc

    def detect_throughput_regression(self, *, subject: str = "system") -> Dict[str, Any]:
        try:
            subject = _coerce_non_empty_str(subject, field_name="subject", operation="detect_throughput_regression")
            with self._lock:
                samples = list(self._throughput_samples.get(subject, ()))
            return self._assess_throughput_regression(subject, samples).to_dict()
        except Exception as exc:
            raise self._handle_exception(exc, stage="performance.throughput", context={"subject": subject}) from exc

    def evaluate_latency_slo(self, *, service: str = "system", slo_name: str = "p95_latency",
        percentile: Optional[int] = None,
        target_ms: Optional[float] = None,
        raise_on_breach: bool = False,
    ) -> Dict[str, Any]:
        try:
            service = _coerce_non_empty_str(service, field_name="service", operation="evaluate_latency_slo")
            percentile_value = int(percentile if percentile is not None else self.default_slo_percentile)
            target_value = float(target_ms if target_ms is not None else self.default_latency_slo_ms)
            summary = self.latency_summary(subject=service)
            observed = float(summary["percentiles_ms"].get(str(percentile_value), 0.0))
            status = "ok" if observed <= target_value else "breach"
            payload = {
                "service": service,
                "slo_name": slo_name,
                "percentile": percentile_value,
                "observed_ms": observed,
                "target_ms": target_value,
                "status": status,
                "sample_count": summary["sample_count"],
            }
            if status == "breach" and raise_on_breach:
                raise SLOBreachError(
                    service=service,
                    slo_name=f"{slo_name}_p{percentile_value}",
                    observed=observed,
                    target=target_value,
                    context={"sample_count": summary["sample_count"]},
                )
            return payload
        except Exception as exc:
            raise self._handle_exception(exc, stage="slo.evaluate", context={"service": service}) from exc

    def summarize_performance(self) -> Dict[str, Any]:
        try:
            with self._lock:
                subjects = sorted(self._latency_samples)
                trace_records = [record.to_dict() for record in self._trace_records]

            subject_summaries = {}
            latency_regressions = {}
            throughput_regressions = {}
            for subject in subjects:
                subject_summaries[subject] = {
                    "latency": self.latency_summary(subject=subject),
                    "throughput": self.throughput_summary(subject=subject),
                }
                latency_regressions[subject] = self.detect_latency_regression(subject=subject)
                throughput_regressions[subject] = self.detect_throughput_regression(subject=subject)

            return {
                "subject_count": len(subjects),
                "trace_record_count": len(trace_records),
                "subjects": subject_summaries,
                "latency_regressions": latency_regressions,
                "throughput_regressions": throughput_regressions,
                "recent_traces": trace_records[-10:],
            }
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="metrics.pipeline",
                context={"operation": "summarize_performance"},
            ) from exc

    def _record_latency_sample_locked(self, *, subject: str, duration_ms: float, timestamp_ms: float,
                                      trace_id: Optional[str], status: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self._ensure_subject_capacity(subject, metric_name="observability_performance.subject")
        self._latency_samples[subject].append(
            LatencySample(
                subject=subject,
                duration_ms=duration_ms,
                timestamp_ms=timestamp_ms,
                trace_id=trace_id,
                status=str(status or "ok").lower(),
                metadata=_safe_mapping(metadata),
            )
        )

    def _record_throughput_sample_locked(self, *, subject: str, count: int, timestamp_ms: float,
                                         success_count: int, failure_count: int,
                                         metadata: Optional[Mapping[str, Any]] = None) -> None:
        self._ensure_subject_capacity(subject, metric_name="observability_performance.subject")
        self._throughput_samples[subject].append(
            ThroughputSample(
                subject=subject,
                count=count,
                timestamp_ms=timestamp_ms,
                success_count=success_count,
                failure_count=failure_count,
                metadata=_safe_mapping(metadata),
            )
        )

    def _ensure_subject_capacity(self, subject: str, *, metric_name: str) -> None:
        unique_subjects = set(self._latency_samples) | set(self._throughput_samples)
        if subject not in unique_subjects and len(unique_subjects) >= self.max_subjects:
            raise MetricCardinalityError(
                metric_name=metric_name,
                cardinality=len(unique_subjects) + 1,
                limit=self.max_subjects,
                context={"subject": subject},
            )

    def _latency_histogram_from_durations(self, durations: Sequence[float]) -> Dict[str, int]:
        histogram: Dict[str, int] = {bucket: 0 for bucket in [_bucketize(0.0, [b]) for b in []]}
        histogram = {}
        for duration in durations:
            label = _bucketize(float(duration), self.histogram_buckets_ms)
            histogram[label] = histogram.get(label, 0) + 1
        ordered: Dict[str, int] = {}
        for bucket in self.histogram_buckets_ms:
            label = f"<= {float(bucket):g}ms"
            ordered[label] = histogram.get(label, 0)
        if self.histogram_buckets_ms:
            overflow_label = f"> {float(self.histogram_buckets_ms[-1]):g}ms"
            ordered[overflow_label] = histogram.get(overflow_label, 0)
        return ordered

    def _assess_latency_regression(self, subject: str, samples: Sequence[LatencySample]) -> RegressionAssessment:
        durations = [sample.duration_ms for sample in samples]
        enough_data = len(durations) >= (self.min_regression_samples * 2)
        if not enough_data:
            return RegressionAssessment(
                subject=subject,
                metric_name="latency_ms",
                baseline_value=0.0,
                recent_value=0.0,
                delta=0.0,
                delta_ratio=0.0,
                level="info",
                direction="flat",
                enough_data=False,
                notes=["Not enough latency samples to evaluate regression."],
            )

        baseline_values = durations[-(self.baseline_sample_window + self.recent_sample_window):-self.recent_sample_window]
        recent_values = durations[-self.recent_sample_window:]
        if not baseline_values:
            baseline_values = durations[:-self.recent_sample_window]
        baseline_value = _percentile(baseline_values, 95.0)
        recent_value = _percentile(recent_values, 95.0)
        delta = recent_value - baseline_value
        ratio = _ratio(recent_value, baseline_value) if baseline_value > 0 else 0.0

        if ratio >= self.latency_regression_critical_ratio:
            level = "critical"
        elif ratio >= self.latency_regression_warning_ratio:
            level = "warning"
        else:
            level = "ok"

        direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
        return RegressionAssessment(
            subject=subject,
            metric_name="latency_ms",
            baseline_value=baseline_value,
            recent_value=recent_value,
            delta=delta,
            delta_ratio=ratio,
            level=level,
            direction=direction,
            enough_data=True,
            notes=["Latency regression uses p95 comparison between baseline and recent windows."],
        )

    def _assess_throughput_regression(self, subject: str,
        samples: Sequence[ThroughputSample],
    ) -> RegressionAssessment:
        enough_data = len(samples) >= (self.min_regression_samples * 2)
        if not enough_data:
            return RegressionAssessment(
                subject=subject,
                metric_name="throughput_rate_per_sample",
                baseline_value=0.0,
                recent_value=0.0,
                delta=0.0,
                delta_ratio=0.0,
                level="info",
                direction="flat",
                enough_data=False,
                notes=["Not enough throughput samples to evaluate regression."],
            )

        sample_rates = [float(sample.count) for sample in samples]
        baseline_values = sample_rates[-(self.baseline_sample_window + self.recent_sample_window):-self.recent_sample_window]
        recent_values = sample_rates[-self.recent_sample_window:]
        if not baseline_values:
            baseline_values = sample_rates[:-self.recent_sample_window]

        baseline_value = sum(baseline_values) / len(baseline_values)
        recent_value = sum(recent_values) / len(recent_values)
        delta = recent_value - baseline_value
        ratio = _ratio(recent_value, baseline_value) if baseline_value > 0 else 0.0

        if baseline_value > 0 and ratio <= self.throughput_regression_critical_ratio:
            level = "critical"
        elif baseline_value > 0 and ratio <= self.throughput_regression_warning_ratio:
            level = "warning"
        else:
            level = "ok"

        direction = "down" if delta < 0 else "up" if delta > 0 else "flat"
        return RegressionAssessment(
            subject=subject,
            metric_name="throughput_rate_per_sample",
            baseline_value=baseline_value,
            recent_value=recent_value,
            delta=delta,
            delta_ratio=ratio,
            level=level,
            direction=direction,
            enough_data=True,
            notes=["Throughput regression compares recent average event count against baseline average."],
        )

    def _archive_trace_if_supported(self, *, trace_id: str,
        spans: Sequence[Mapping[str, Any]],
        report_payload: Mapping[str, Any],
        incident_level: str,
        metadata: Optional[Mapping[str, Any]],
    ) -> None:
        if not self.archive_trace_reports or self.memory is None:
            return
        if not hasattr(self.memory, "archive_trace"):
            return
        try:
            self.memory.archive_trace(
                trace_id=trace_id,
                agent_spans=spans,
                summary=report_payload,
                incident_level=incident_level,
                metadata=_safe_mapping(metadata),
            )
        except Exception:
            logger.debug("Trace archive skipped for '%s' because memory persistence is unavailable.", trace_id)

    def _append_trace_timeline_if_supported(self, *, trace_id: str, report_payload: Mapping[str, Any]) -> None:
        if self.memory is None or not hasattr(self.memory, "append_timeline_event"):
            return
        try:
            self.memory.append_timeline_event(
                trace_id=trace_id,
                event_type="performance.trace_analyzed",
                severity="warning" if report_payload.get("anomaly_count", 0) else "info",
                message=(
                    f"Trace '{trace_id}' analyzed with {report_payload.get('bottleneck_count', 0)} bottlenecks "
                    f"and {report_payload.get('anomaly_count', 0)} anomalies."
                ),
                payload={
                    "critical_path_ms": report_payload.get("critical_path_ms", 0.0),
                    "total_duration_ms": report_payload.get("total_duration_ms", 0.0),
                },
            )
        except Exception:
            logger.debug("Timeline event emission skipped for trace '%s'.", trace_id)

    def _waterfall_report_payload(self, report: Any, summary: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(summary)
        payload.update(
            {
                "total_duration_ms": float(getattr(report, "total_duration_ms", summary.get("total_duration_ms", 0.0))),
                "critical_path_ms": float(getattr(report, "critical_path_ms", summary.get("critical_path_ms", 0.0))),
                "critical_path_span_ids": list(
                    getattr(report, "critical_path_span_ids", summary.get("critical_path_span_ids", []))
                ),
                "per_agent_duration_ms": dict(
                    getattr(report, "per_agent_duration_ms", summary.get("per_agent_duration_ms", {}))
                ),
                "retry_chains": list(getattr(report, "retry_chains", [])),
                "bottleneck_spans": list(getattr(report, "bottleneck_spans", [])),
                "anomalies": list(getattr(report, "anomalies", [])),
                "retry_chain_count": len(getattr(report, "retry_chains", [])),
                "bottleneck_count": len(getattr(report, "bottleneck_spans", [])),
                "anomaly_count": len(getattr(report, "anomalies", [])),
            }
        )
        return payload

    def _ensure_enabled(self, *, operation: str) -> None:
        if not self.enabled:
            raise ObservabilityError(
                message=f"ObservabilityPerformance is disabled; cannot execute '{operation}'",
                error_type=ObservabilityErrorType.METRIC_PIPELINE_FAILED,
                severity=ObservabilitySeverity.MEDIUM,
                retryable=False,
                context={"operation": operation},
                remediation="Enable observability_performance in observability_config.yaml.",
            )

    def _handle_exception(self, exc: Exception, *, stage: str,
                          context: Optional[Dict[str, Any]] = None) -> ObservabilityError:
        if isinstance(exc, ObservabilityError):
            return exc
        return normalize_observability_exception(exc, stage=stage, context=context)


# ---------------------------------------------------------------------------
# Script test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Observability Capacity & Performance ===\n")
    printer.status("TEST", "Observability Capacity & Performance initialized", "info")

    capacity = ObservabilityCapacity(memory=None)
    performance = ObservabilityPerformance(memory=None)

    printer.status("TEST", "Recording queue telemetry", "info")
    capacity.record_queue_sample("planner_queue", depth=18, inflow_per_sec=6.0, outflow_per_sec=6.5)
    capacity.record_queue_sample("planner_queue", depth=42, inflow_per_sec=8.0, outflow_per_sec=6.5)
    capacity.record_queue_sample("planner_queue", depth=79, inflow_per_sec=10.0, outflow_per_sec=6.0)
    capacity.record_queue_sample("handler_queue", depth=7, inflow_per_sec=2.0, outflow_per_sec=2.2)

    printer.status("TEST", "Recording resource telemetry", "info")
    capacity.record_resource_sample("node-a-cpu", utilization_pct=68.0, resource_type="cpu", used=68, capacity=100)
    capacity.record_resource_sample("node-a-cpu", utilization_pct=84.0, resource_type="cpu", used=84, capacity=100)
    capacity.record_resource_sample("node-a-gpu", utilization_pct=91.0, resource_type="gpu", used=91, capacity=100)
    capacity.record_resource_sample("node-a-memory", utilization_pct=73.0, resource_type="memory", used=73, capacity=100)

    print("Queue report:")
    print(capacity.get_queue_report("planner_queue"))
    print("\nResource report:")
    print(capacity.get_resource_report("node-a-gpu"))
    print("\nCapacity summary:")
    print(capacity.summarize_capacity())

    printer.status("TEST", "Recording throughput and latency telemetry", "info")
    performance.record_latency_sample("system", duration_ms=120.0)
    performance.record_latency_sample("system", duration_ms=180.0)
    performance.record_latency_sample("system", duration_ms=240.0)
    performance.record_throughput_sample("system", count=5, success_count=5, failure_count=0)
    performance.record_throughput_sample("system", count=4, success_count=3, failure_count=1)

    printer.status("TEST", "Analyzing synthetic waterfall trace", "info")
    test_spans = [
        {
            "span_id": "trace-1-root",
            "agent_name": "PlanningAgent",
            "start_ms": 0,
            "end_ms": 80,
            "status": "ok",
        },
        {
            "span_id": "trace-1-eval",
            "agent_name": "EvaluationAgent",
            "start_ms": 5,
            "end_ms": 65,
            "status": "ok",
            "parent_span_id": "trace-1-root",
        },
        {
            "span_id": "trace-1-handler-retry-1",
            "agent_name": "HandlerAgent",
            "start_ms": 20,
            "end_ms": 55,
            "status": "retry",
            "parent_span_id": "trace-1-root",
        },
        {
            "span_id": "trace-1-handler-retry-2",
            "agent_name": "HandlerAgent",
            "start_ms": 56,
            "end_ms": 95,
            "status": "retry",
            "parent_span_id": "trace-1-root",
        },
        {
            "span_id": "trace-1-observability",
            "agent_name": "ObservabilityAgent",
            "start_ms": 96,
            "end_ms": 130,
            "status": "ok",
            "parent_span_id": "trace-1-root",
        },
    ]
    print("\nTrace analysis:")
    print(performance.analyze_trace("trace-1", test_spans))
    print("\nLatency summary:")
    print(performance.latency_summary(subject="system"))
    print("\nThroughput summary:")
    print(performance.throughput_summary(subject="system"))
    print("\nLatency regression:")
    print(performance.detect_latency_regression(subject="system"))
    print("\nThroughput regression:")
    print(performance.detect_throughput_regression(subject="system"))
    print("\nLatency SLO evaluation:")
    print(performance.evaluate_latency_slo(service="system"))
    print("\nPerformance summary:")
    print(performance.summarize_performance())

    print("\n=== Test ran successfully ===\n")