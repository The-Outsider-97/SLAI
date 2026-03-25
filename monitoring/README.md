# SLAI Monitoring Subsystem

This folder provides a cohesive, production‑ready monitoring layer used by both automation scripts and the SLAI Hub UI.

## Architecture

The subsystem is split into four focused modules:

1. **`metrics_collector.py`**  
   Collects structured system metrics (CPU, memory, disk, network, processes) with fallback when `psutil` is missing.  
   Returns a `MetricSnapshot` dataclass.

2. **`drift_detection.py`**  
   Implements a Kolmogorov‑Smirnov test for data drift.  
   Returns a `DriftResult` dataclass with statistic, p‑value, drift flag, and notes.

3. **`health_check.py`**  
   Performs TCP and HTTP health checks on a list of services.  
   Returns a `HealthReport` containing per‑service results and overall status.

4. **`alert_manager.py`**  
   Evaluates metrics, drift, and health results against thresholds, applies deduplication cooldowns, and dispatches alerts via email (or other transports).  
   Separates alert logic from delivery.

## Public API

All public classes and functions are exported in `__init__.py`.

### Metrics Collection
```python
collector = MetricsCollector(disk_path="/")
snapshot = collector.collect_snapshot()
print(snapshot.to_pretty_string())
