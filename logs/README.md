# Logs Directory

This directory now enforces structured observability standards across logging, metrics, and governance.

## Structured Logging

`logs/observability.py` defines a `StructuredLogger` that emits JSON records with a fixed schema:

- `timestamp`, `level`, `logger`, `event`, `message`
- `service`, `environment`, `component`
- `trace_id`, `span_id`
- `metadata` (object payload for context)

Use canonical Python log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) only.

## Service-level Metrics and Alerts

`ServiceMetrics` computes rolling-window SLO telemetry:

- Average and p95 latency (ms)
- Error rate
- Composite health score

Alert thresholds are configured in `MetricsAlertThresholds`:

- `min_health_score` (default: `0.95`)
- `max_p95_latency_ms` (default: `1000`)
- `max_error_rate` (default: `0.02`)

## Log Lifecycle and Access Controls

`LogGovernancePolicy` configures:

- Rotation policy (`rotation_bytes`, `rotation_backups`)
- Retention policy (`retention_days`) via `enforce_retention()`
- PII redaction for email/phone/SSN/credit-card patterns
- File and directory permission masks (`0640` files, `0750` directories)

This gives baseline controls for privacy and least-privilege log access.
