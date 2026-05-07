"""
monitoring
──────────
SLAI Monitoring Subsystem – public API surface.
"""

from .config_loader import (
    load_global_config,
    get_config_section,
)

from .resilience import (
    RetryPolicy,
    CircuitBreakerRegistry,
    CircuitBreakerOpen,
    TokenBucketLimiter,
    RateLimitExceeded,
)

from .metrics_collector import (
    MetricsCollector,
    MetricSnapshot,
    MetricAggregates,
    CPUStats,
    MemoryStats,
    DiskStats,
    NetworkStats,
    ProcessStats,
    collect_system_metrics,
)

from .drift_detection import (
    DriftDetector,
    DriftTest,
    DriftResult,
    WindowedDriftDetector,
    detect_data_drift,
)

from .health_check import (
    HealthChecker,
    HealthReport,
    ServiceHealthResult,
    service_health_check,
)

from .alert_manager import (
    Alert,
    AlertTransport,
    AlertManager,
    EmailTransport,
    SlackTransport,
    WebhookTransport,
    send_alert,
)

__all__ = [
    # Config
    "load_global_config", "get_config_section",
    # Resilience
    "RetryPolicy",
    "CircuitBreakerRegistry", "CircuitBreakerOpen",
    "TokenBucketLimiter", "RateLimitExceeded",
    # Metrics
    "MetricsCollector", "MetricSnapshot", "MetricAggregates",
    "CPUStats", "MemoryStats", "DiskStats", "NetworkStats", "ProcessStats",
    "collect_system_metrics",
    # Drift
    "DriftDetector", "DriftTest", "DriftResult",
    "WindowedDriftDetector", "detect_data_drift",
    # Health
    "HealthChecker", "HealthReport", "ServiceHealthResult",
    "service_health_check",
    # Alerts
    "Alert", "AlertTransport", "AlertManager",
    "EmailTransport", "SlackTransport", "WebhookTransport",
    "send_alert",
]