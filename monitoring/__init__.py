"""SLAI Monitoring Subsystem – production-ready, reusable monitoring layer."""

from .alert_manager import AlertManager, EmailAlertTransport, send_alert
from .drift_detection import DriftDetector, DriftResult, detect_data_drift
from .health_check import HealthChecker, HealthReport, service_health_check
from .metrics_collector import MetricsCollector, MetricSnapshot, collect_system_metrics

__all__ = [
    # Alert
    "AlertManager",
    "EmailAlertTransport",
    "send_alert",
    # Drift Detection
    "DriftDetector",
    "DriftResult",
    "detect_data_drift",
    # Health Check
    "HealthChecker",
    "HealthReport",
    "service_health_check",
    # Metric Collector
    "MetricsCollector",
    "MetricSnapshot",
    "collect_system_metrics",
]
