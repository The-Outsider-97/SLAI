# SLAI Monitoring Subsystem

This folder now provides a cohesive, production-style monitoring layer used by both automation and the SLAI Hub UI.

## Architecture Overview

The subsystem is split into focused modules:

1. `metrics_collector.py`
   - Collects structured runtime snapshots.
   - Captures CPU (total/per-core/load), memory, disk, process, and network telemetry.
   - Returns typed snapshots (`MetricSnapshot`) with UTC timestamps.

2. `drift_detection.py`
   - Runs KS-test drift checks through `DriftDetector`.
   - Validates threshold and input data (including empty/invalid values).
   - Returns a structured `DriftResult` containing statistic, p-value, threshold, drift flag, sample counts, and notes.

3. `health_check.py`
   - Executes multi-service health probes with `HealthChecker`.
   - Supports both TCP and HTTP checks.
   - Handles failures and timeouts gracefully and returns aggregate `HealthReport` summaries.

4. `alert_manager.py`
   - Central alert decision layer (`AlertManager`) for metrics, drift, and health events.
   - Supports threshold-based triggering and deduplication cooldowns.
   - Separates transport logic via `EmailAlertTransport`.
   - Keeps legacy helper function compatibility where practical.

## How the SLAI Hub popup uses this layer

`main.py` includes a **System Monitoring** dialog in the hamburger menu. That popup:

- Calls `MetricsCollector.collect_snapshot()` for current host telemetry.
- Calls `HealthChecker.run_checks()` for local TCP/HTTP endpoint status.
- Calls `DriftDetector.detect()` to surface drift diagnostics in a consistent format.
- Renders the monitoring state in a SLAI-styled dialog, keeping the monitoring UX integrated into Hub.

## Configuration / Setup Notes

- `psutil` and `scipy` are required for full metrics + drift capabilities.
- Email delivery requires explicit SMTP configuration through `EmailAlertTransport` (host/port/auth/from-address).
- No credentials are hardcoded; secure values should be injected from environment-specific config.

## Compatibility Notes

- Legacy helper functions were preserved:
  - `collect_system_metrics()`
  - `detect_data_drift()`
  - `service_health_check()`
  - `send_alert()`
- New class-based APIs are recommended for all new integrations.
