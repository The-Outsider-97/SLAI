"""
Structured metrics hooks (latency histograms, cache hit/miss, sync failures, rule failures).
Provides thread‑safe collection and Prometheus exposition.
"""

import time
import threading

from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from ..utils.knowledge_errors import MetricsCollectionError
from ..utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Runtime Metrics")
printer = PrettyPrinter


@dataclass
class Metric:
    """Base metric container."""
    name: str
    help_text: str
    typ: str  # counter, gauge, histogram


@dataclass
class Counter(Metric):
    """Counter metric (only increases)."""
    typ: str = "counter"
    _value: int = 0

    def inc(self, amount: int = 1) -> None:
        self._value += amount

    def value(self) -> int:
        return self._value


@dataclass
class Gauge(Metric):
    """Gauge metric (can go up and down)."""
    typ: str = "gauge"
    _value: float = 0.0

    def set(self, value: float) -> None:
        self._value = value

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        self._value -= amount

    def value(self) -> float:
        return self._value


@dataclass
class Histogram(Metric):
    """Histogram metric (bucketed latency/values)."""
    typ: str = "histogram"
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
    _count: int = 0
    _sum: float = 0.0
    _bucket_counts: Dict[float, int] = field(default_factory=dict)

    def __post_init__(self):
        self._bucket_counts = {b: 0 for b in self.buckets}

    def observe(self, value: float) -> None:
        self._count += 1
        self._sum += value
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1

    def count(self) -> int:
        return self._count

    def sum(self) -> float:
        return self._sum

    def bucket_counts(self) -> Dict[float, int]:
        return self._bucket_counts.copy()


class RTMetrics:
    """
    Runtime Metrics collector for knowledge agent.
    Thread‑safe, supports counters, gauges, histograms.
    Can export to Prometheus exposition format.
    """

    def __init__(self):
        self.config = load_global_config()
        self.metrics_config = get_config_section('runtime_metrics')
        self._lock = threading.RLock()
        self._metrics: Dict[str, Metric] = {}
        self._init_default_metrics()

    def _init_default_metrics(self) -> None:
        """Initialize standard metrics used across components."""
        # Counters
        self.register_counter(
            "knowledge_cache_hits_total",
            "Total number of cache hits"
        )
        self.register_counter(
            "knowledge_cache_misses_total",
            "Total number of cache misses"
        )
        self.register_counter(
            "knowledge_rule_success_total",
            "Total number of successful rule executions"
        )
        self.register_counter(
            "knowledge_rule_failures_total",
            "Total number of failed rule executions"
        )
        self.register_counter(
            "knowledge_rule_timeouts_total",
            "Total number of rule timeouts"
        )
        self.register_counter(
            "knowledge_sync_attempts_total",
            "Total number of sync attempts"
        )
        self.register_counter(
            "knowledge_sync_failures_total",
            "Total number of sync failures"
        )
        self.register_counter(
            "knowledge_action_executions_total",
            "Total number of action executions"
        )
        self.register_counter(
            "knowledge_action_failures_total",
            "Total number of action failures"
        )

        # Gauges
        self.register_gauge(
            "knowledge_memory_size",
            "Current number of entries in knowledge memory"
        )
        self.register_gauge(
            "knowledge_cache_size",
            "Current number of items in cache"
        )
        self.register_gauge(
            "knowledge_rule_count",
            "Number of loaded rules"
        )
        self.register_gauge(
            "knowledge_rule_failure_rate",
            "Rolling failure rate for rules (0-1)"
        )

        # Histograms
        self.register_histogram(
            "knowledge_retrieval_latency_seconds",
            "Latency of knowledge retrieval operations",
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]
        )
        self.register_histogram(
            "knowledge_rule_apply_latency_seconds",
            "Latency of rule engine apply operations",
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]
        )
        self.register_histogram(
            "knowledge_sync_latency_seconds",
            "Latency of sync operations",
            buckets=[0.1, 0.5, 1, 5, 10, 30]
        )
        self.register_histogram(
            "knowledge_action_latency_seconds",
            "Latency of action executions",
            buckets=[0.05, 0.1, 0.5, 1, 2, 5]
        )

    def register_counter(self, name: str, help_text: str) -> None:
        """Register a new counter metric."""
        with self._lock:
            if name in self._metrics:
                raise MetricsCollectionError(name, "registration", f"Metric {name} already exists")
            self._metrics[name] = Counter(name=name, help_text=help_text)

    def register_gauge(self, name: str, help_text: str) -> None:
        with self._lock:
            if name in self._metrics:
                raise MetricsCollectionError(name, "registration", f"Metric {name} already exists")
            self._metrics[name] = Gauge(name=name, help_text=help_text)

    def register_histogram(self, name: str, help_text: str, buckets: Optional[List[float]] = None) -> None:
        with self._lock:
            if name in self._metrics:
                raise MetricsCollectionError(name, "registration", f"Metric {name} already exists")
            self._metrics[name] = Histogram(name=name, help_text=help_text, buckets=buckets or [])

    def get_counter(self, name: str) -> Counter:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None or not isinstance(metric, Counter):
                raise MetricsCollectionError(name, "get", "Counter not found or wrong type")
            return metric

    def get_gauge(self, name: str) -> Gauge:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None or not isinstance(metric, Gauge):
                raise MetricsCollectionError(name, "get", "Gauge not found or wrong type")
            return metric

    def get_histogram(self, name: str) -> Histogram:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None or not isinstance(metric, Histogram):
                raise MetricsCollectionError(name, "get", "Histogram not found or wrong type")
            return metric

    # Convenience methods for common operations
    def inc_counter(self, name: str, amount: int = 1) -> None:
        try:
            self.get_counter(name).inc(amount)
        except MetricsCollectionError as e:
            logger.warning(f"Failed to increment counter {name}: {e}")

    def set_gauge(self, name: str, value: float) -> None:
        try:
            self.get_gauge(name).set(value)
        except MetricsCollectionError as e:
            logger.warning(f"Failed to set gauge {name}: {e}")

    def observe_histogram(self, name: str, value: float) -> None:
        try:
            self.get_histogram(name).observe(value)
        except MetricsCollectionError as e:
            logger.warning(f"Failed to observe histogram {name}: {e}")

    def export_prometheus(self) -> str:
        """
        Export all metrics in Prometheus exposition format.
        """
        lines = []
        with self._lock:
            for metric in self._metrics.values():
                lines.append(f"# HELP {metric.name} {metric.help_text}")
                lines.append(f"# TYPE {metric.name} {metric.typ}")
                if isinstance(metric, Counter):
                    lines.append(f"{metric.name} {metric.value()}")
                elif isinstance(metric, Gauge):
                    lines.append(f"{metric.name} {metric.value()}")
                elif isinstance(metric, Histogram):
                    # Buckets
                    for bucket, count in metric.bucket_counts().items():
                        lines.append(f"{metric.name}_bucket{{le=\"{bucket}\"}} {count}")
                    lines.append(f"{metric.name}_bucket{{le=\"+Inf\"}} {metric.count()}")
                    lines.append(f"{metric.name}_count {metric.count()}")
                    lines.append(f"{metric.name}_sum {metric.sum()}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            for metric in self._metrics.values():
                if isinstance(metric, Counter):
                    metric._value = 0
                elif isinstance(metric, Gauge):
                    metric._value = 0.0
                elif isinstance(metric, Histogram):
                    metric._count = 0
                    metric._sum = 0.0
                    metric._bucket_counts = {b: 0 for b in metric.buckets}


if __name__ == "__main__":
    print("\n=== Running Runtime Metrics ===\n")
    printer.status("Init", "Runtime Metrics initialized", "success")

    metrics = RTMetrics()
    print(f"{metrics}")

    print("\n=== Successfully ran the Runtime Metrics ===\n")