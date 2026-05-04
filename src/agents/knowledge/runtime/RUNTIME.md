# Knowledge Runtime Operations Guide

This document describes operational behavior for the runtime package:

- `RTHealth` (`runtime/health.py`)
- `RTMetrics` (`runtime/metrics.py`)

It is intended as a practical runbook for production operators and developers.

---

## 1) Health Model (`RTHealth`)

`RTHealth` maintains component-level health states:

- `healthy`
- `degraded`
- `unhealthy`

### Built-in component probes

By default, the following checks are registered:

- `memory`
- `cache`
- `rule_engine`
- `governor`
- `sync`
- `action_executor`
- `ontology`

### Probe semantics

- **Liveness**: requires critical components (`memory`, `cache`, `rule_engine`) to be non-unhealthy.
- **Readiness**: requires core serving path (`memory`, `cache`, `rule_engine`, `governor`) to be non-unhealthy.
- **Periodic checks**: optional background loop controlled by config.

### Runtime lifecycle

- Start: automatic on `RTHealth` initialization when `enable_periodic_checks: true`.
- Stop: call `shutdown()` during process teardown to avoid orphan checker threads.

---

## 2) Metrics Model (`RTMetrics`)

`RTMetrics` is a thread-safe in-process collector with:

- `Counter`
- `Gauge`
- `Histogram`

and Prometheus exposition support via `export_prometheus()`.

### Default counters

- `knowledge_cache_hits_total`
- `knowledge_cache_misses_total`
- `knowledge_rule_success_total`
- `knowledge_rule_failures_total`
- `knowledge_rule_timeouts_total`
- `knowledge_sync_attempts_total`
- `knowledge_sync_failures_total`
- `knowledge_action_executions_total`
- `knowledge_action_failures_total`

### Default gauges

- `knowledge_memory_size`
- `knowledge_cache_size`
- `knowledge_rule_count`
- `knowledge_rule_failure_rate`

### Default histograms

- `knowledge_retrieval_latency_seconds`
- `knowledge_rule_apply_latency_seconds`
- `knowledge_sync_latency_seconds`
- `knowledge_action_latency_seconds`

---

## 3) Configuration Keys

These runtime components read configuration from:

- `runtime_health`
- `runtime_metrics`

Recommended defaults are provided in `configs/knowledge_config.yaml`.

---

## 4) Production Recommendations

1. Keep periodic health checks enabled unless the host process already runs centralized probes.
2. Set `check_interval_seconds` to match your alerting cadence (e.g., 30-60s).
3. Export metrics on a predictable endpoint and scrape at a stable interval.
4. Track long-tail retrieval latency (`p95`, `p99`) and rising rule failure trends.
5. Tie readiness failures to automated traffic draining in orchestration layers.

---

## 5) Suggested Alert Baselines (starting point)

- Readiness false for > 2 consecutive intervals.
- `knowledge_sync_failures_total` growth burst (rate-based alert).
- `knowledge_rule_failure_rate > 0.2` sustained for 5+ minutes.
- `knowledge_retrieval_latency_seconds` p95 above SLO for 10+ minutes.

Tune thresholds to workload, data shape, and deployment scale.
