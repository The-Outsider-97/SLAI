# SLAI Monitoring Subsystem

A production-ready monitoring layer for the SLAI Hub, providing metrics
collection, statistical drift detection, service health checks, and
multi-transport alerting — with built-in resilience primitives.

---

## Architecture

```
monitoring/
├── __init__.py            ← Clean public API surface
├── config_loader.py       ← load_config() reads monitor_config.yaml
├── resilience.py          ← RetryPolicy, CircuitBreaker, TokenBucket, StructuredLogger
├── metrics_collector.py   ← System metrics + background loop + Prometheus export
├── drift_detection.py     ← KS, Chi-squared, PSI tests + windowed streaming
├── health_check.py        ← Parallel TCP/HTTP/TLS checks with retries
├── alert_manager.py       ← Evaluation, dedup, rate-limiting, transport dispatch
└── dashboard.py           ← Flask/SocketIO real-time dashboard
```

### Resilience layer (`resilience.py`)

All network operations in the subsystem are protected by three primitives:

| Primitive | What it does |
|---|---|
| `RetryPolicy` | Exponential back-off with full jitter; configurable exception whitelist |
| `CircuitBreakerRegistry` | Per-key CLOSED→OPEN→HALF-OPEN state machine; prevents thundering herd |
| `TokenBucketLimiter` | Token-bucket rate limiter on outbound alert dispatch; burst protection |

---

## Quick Start

```python
from monitoring import load_config, MetricsCollector, AlertManager

# 1. Load config from monitor_config.yaml (or env-var / defaults if absent)
cfg = load_config()

# 2. Collect metrics
collector = MetricsCollector(disk_path=cfg.collector.disk_path)
snapshot = collector.collect_snapshot()
print(snapshot.to_pretty_string())

# 3. Build alert manager (transports auto-enabled from config)
manager = AlertManager.from_config(cfg)

# 4. Evaluate and dispatch
alerts = manager.process_metrics(snapshot, cfg.thresholds.to_metrics_dict())
outcomes = manager.dispatch(alerts)
```

---

## Configuration

### `monitor_config.yaml`

Place this file in your working directory. `load_config()` reads it automatically,
or pass an explicit path: `load_config("path/to/monitor_config.yaml")`.

### Environment variables

All variables use the `SLAI_` prefix and are applied as fallback when a key is
absent from the YAML file:

```bash
# Thresholds
SLAI_THRESHOLD_CPU=85.0
SLAI_THRESHOLD_MEMORY=85.0
SLAI_THRESHOLD_DISK=85.0
SLAI_THRESHOLD_DRIFT_PVALUE=0.05

# Email transport
SLAI_ALERT_EMAIL_ENABLED=true
SLAI_SMTP_HOST=smtp.example.com
SLAI_SMTP_PORT=587
SLAI_SMTP_USER=user@example.com
SLAI_SMTP_PASSWORD=secret
SLAI_SMTP_FROM=alerts@example.com
SLAI_ALERT_EMAIL_RECIPIENT=ops@example.com
SLAI_SMTP_TLS=true

# Slack transport
SLAI_ALERT_SLACK_ENABLED=true
SLAI_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLAI_SLACK_CHANNEL=#alerts

# Webhook transport
SLAI_ALERT_WEBHOOK_ENABLED=true
SLAI_WEBHOOK_URL=https://example.com/hooks/slai
SLAI_WEBHOOK_SECRET_HEADER=X-SLAI-Signature
SLAI_WEBHOOK_SECRET_VALUE=my-hmac-secret

# Resilience
SLAI_RETRY_MAX=3
SLAI_CB_FAILURE_THRESHOLD=5
SLAI_CB_RECOVERY_TIMEOUT=60
SLAI_RATE_LIMIT_TOKENS=10
SLAI_RATE_LIMIT_REFILL=1.0
SLAI_ALERT_COOLDOWN=300
SLAI_DEDUP_STATE_PATH=/var/run/slai/dedup.json

# Logging
SLAI_LOG_LEVEL=INFO
SLAI_LOG_JSON=true        # structured JSON logs for log aggregators
SLAI_LOG_FILE=/var/log/slai/monitoring.log
```

### `monitor_config.yaml` — full example

```yaml
thresholds:
  cpu_percent: 85.0
  memory_percent: 85.0
  disk_percent: 90.0
  drift_p_value: 0.05

email:
  enabled: true
  smtp_host: smtp.example.com
  smtp_port: 587
  username: alerts@example.com
  password: "${SMTP_PASSWORD}"
  from_email: alerts@example.com
  recipient_email: ops@example.com

slack:
  enabled: true
  webhook_url: "https://hooks.slack.com/services/..."
  channel: "#alerts"

webhook:
  enabled: true
  url: "https://example.com/hooks/slai"
  secret_header: "X-SLAI-Signature"
  secret_value: "${WEBHOOK_SECRET}"

resilience:
  max_retries: 3
  cb_failure_threshold: 5
  cb_recovery_timeout: 60.0
  rate_limit_tokens: 10
  alert_cooldown_seconds: 300
```

### Module Reference

### `MetricsCollector`

```python
collector = MetricsCollector(disk_path="/", history_max_snapshots=240)

# One-shot
snapshot = collector.collect_snapshot()
print(snapshot.to_pretty_string())
print(snapshot.to_prometheus())     # Prometheus text format

# Background loop
collector.start(interval_seconds=15.0)
# ... later ...
history = collector.get_history()        # list[MetricSnapshot]
aggregates = collector.get_aggregates()  # MetricAggregates | None
collector.stop()
```

### `DriftDetector`

```python
from monitoring import DriftDetector, DriftTest, WindowedDriftDetector

det = DriftDetector()

# KS test (continuous)
result = det.detect(reference, current, threshold=0.05, metric_name="score")

# Chi-squared (categorical)
result = det.detect(ref_cats, cur_cats, test=DriftTest.CHI2, metric_name="labels")

# PSI (probability distributions)
result = det.detect(ref_probs, cur_probs, test=DriftTest.PSI, threshold=0.2,
                    metric_name="output_probs")

# Batch
results = det.detect_batch([
    {"reference_data": ref1, "new_data": cur1, "metric_name": "m1"},
    {"reference_data": ref2, "new_data": cur2, "metric_name": "m2", "test": "chi2"},
])

# Windowed streaming
wdet = WindowedDriftDetector(initial_reference, max_window_size=1000)
result = wdet.update(new_batch, metric_name="live_scores", advance_window=True)
```

### `HealthChecker`

```python
from monitoring import HealthChecker

checker = HealthChecker(max_workers=8)
report = checker.run_checks([
    # TCP
    {"name": "Redis", "type": "tcp", "host": "redis", "port": 6379, "timeout": 1.0},
    # HTTP with custom expected status codes
    {"name": "API", "type": "http", "url": "https://api.example.com/health",
     "expected_status": [200], "latency_warn_ms": 500},
    # TLS certificate expiry
    {"name": "API TLS", "type": "tls", "host": "api.example.com",
     "port": 443, "tls_warn_days": 30},
])
print(checker.format_report(report))
```

### `AlertManager`

```python
from monitoring import AlertManager, SlackTransport, WebhookTransport

manager = AlertManager(
    transports=[
        SlackTransport(webhook_url="https://hooks.slack.com/...", channel="#alerts"),
        WebhookTransport(url="https://example.com/hook",
                         secret_header="X-SLAI-Sig", secret_value="secret"),
    ],
    cooldown_seconds=300,
    rate_limit_tokens=10,
    cb_failure_threshold=5,
)

# Evaluate data sources
alerts  = manager.process_metrics(snapshot, thresholds)
alerts += manager.process_drift(drift_result)
alerts += manager.process_health(health_report)

# Dispatch (retries + circuit breaker + rate limiter applied automatically)
outcomes = manager.dispatch(alerts)

# Inspect circuit breaker states
print(manager.circuit_breaker_status())
# → {"slack": "closed", "webhook": "open"}
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/test_monitoring.py -v
```

---

## Dashboard

```python
from monitoring.dashboard import run_dashboard_thread, push_metrics_update

run_dashboard_thread(host="0.0.0.0", port=5000)

# From your metrics loop:
snapshot = collector.collect_snapshot()
push_metrics_update(snapshot.to_dict())
```

Routes:
- `GET /`                  → Real-time dashboard (requires `templates/dashboard.html`)
- `GET /metrics/prometheus` → Prometheus scrape endpoint
- `GET /health`            → JSON liveness probe