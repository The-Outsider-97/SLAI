# Logs Directory

This directory enforces a deterministic and standardized logging layout.

## Canonical Event Schema

All structured log events should use this shape:

- `timestamp` (UTC ISO-8601)
- `agent`
- `trace_id`
- `event`
- `severity`
- `payload` (object)

This schema is implemented in `logs/standards.py` via `StandardLogEvent` and `build_event()`.

## Deterministic Log Domains

`logs/standards.py` creates and manages three deterministic directories:

- `logs/runtime/` → runtime/service logs
- `logs/audit/` → compliance and deployment audit trails
- `logs/training/` → experiment/training logs

Use `default_log_path(domain, filename)` to prevent ad-hoc paths.

## Integration Points

- `logs/logger.py` writes application runtime logs to `logs/runtime/app.log`.
- `logs/observability.py` emits redacted JSON runtime events using the standard shape.
- `deployment/audit_logger.py` writes audit events to `logs/audit/deployment_audit.jsonl`.

## Governance

`logs/observability.py` still provides:

- Rotation controls
- Retention enforcement
- PII redaction
- Access permission settings
