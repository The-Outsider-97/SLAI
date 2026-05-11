# Handler Agent Utilities

## Overview
The `src/agents/handler/` package provides lightweight operational guardrails for agent orchestration:

- **`HandlerMemory`** for checkpointing and telemetry buffering.
- **`HandlerPolicy`** for retries and circuit-breaker control.
- **`AdaptiveRetryPolicy`** for fingerprint-aware retry budgets.
- **`ProbabilisticStrategySelector`** for data-informed strategy routing.
- **`SLARecoveryPolicy`** for budget-aware retry constraints.
- **`EscalationManager`** for typed handoff payloads.
- **`FailureIntelligence`** for deterministic failure signatures, category hints, and bounded confidence recommendations.
- **`HandlerError`** for structured, severity-aware error reporting.
- **YAML-driven configuration** loaded through `utils/config_loader.py`.

This module is intentionally minimal and can be used by higher-level agents as a resilience layer around task execution.

---

## Directory Layout

```text
src/agents/handler/
├── __init__.py
├── handler_memory.py
├── handler_policy.py
├── adaptive_retry_policy.py
├── strategy_selector.py
├── sla_policy.py
├── escalation_manager.py
├── failure_intelligence.py
├── configs/
│   └── handler_config.yaml
└── utils/
    ├── __init__.py
    ├── config_loader.py
    └── handler_error.py
```

---

## Components

### `HandlerMemory`
File: `handler_memory.py`

`HandlerMemory` stores short-lived execution artifacts with bounded memory use.

#### Responsibilities
- Keeps a bounded deque of **state checkpoints** (`max_checkpoints`).
- Keeps a bounded deque of **telemetry events** (`max_telemetry_events`).
- Provides restore/search APIs for checkpoint recovery and incident triage.

#### Public API
- `save_checkpoint(label, state, metadata=None) -> checkpoint_id`
  - Saves a deep-copied state snapshot.
  - Checkpoint IDs are generated as `<label>:<epoch_ms>`.
- `find_checkpoints(label=None, max_age=None) -> list[dict]`
  - Returns recent checkpoints, optionally filtered by label and age.
- `restore_checkpoint(checkpoint_id) -> Optional[dict]`
  - Returns a deep copy of the stored state (or `None` if not found).
- `append_telemetry(event) -> None`
  - Appends arbitrary telemetry payloads.
- `recent_telemetry(limit=100) -> list[dict]`
  - Returns the most recent telemetry entries.

#### Notes
- Checkpoint payloads are deep-copied on save and restore to prevent aliasing.
- Storage is in-memory only; restarts clear all data.

### `HandlerPolicy`
File: `handler_policy.py`

`HandlerPolicy` manages failure behavior for named agents.

#### Responsibilities
- Retry gating via `max_retries`.
- Circuit-breaker state per `agent_name`.
- Failure-window tracking for budget-aware blocking.

#### Public API
- `can_attempt(agent_name) -> bool`
  - `False` while breaker cooldown is active.
- `retries_allowed(attempted_retries) -> bool`
  - Compares retries to `max_retries`.
- `record_failure(agent_name) -> None`
  - Increments failure counters and may open breaker.
- `record_success(agent_name) -> None`
  - Resets failure counters and breaker state.
- `breaker_status(agent_name) -> dict`
  - Returns `is_open`, `open_until`, and `seconds_remaining`.

#### Circuit-Breaker Behavior
1. Failures are tracked by `agent_name`.
2. If failures exceed `circuit_breaker_threshold`, breaker opens for `cooldown_seconds`.
3. Calls to `can_attempt` return `False` until cooldown expires.
4. `record_success` immediately clears breaker and counters.

### `HandlerError`
File: `utils/handler_error.py`

`HandlerError` is a dataclass exception with normalized metadata:

- `message`
- `error_type` (default: `HandlerError`)
- `severity` (`low | medium | high | critical`)
- `retryable` flag
- `context` dictionary for diagnostic payloads

Use `to_dict()` when serializing errors to logs, APIs, or telemetry streams.

---

## Configuration
File: `configs/handler_config.yaml`

The package reads config through `utils/config_loader.py`, which caches a global parsed YAML document.

### Key Sections
- `handler_agent`: top-level runtime defaults.
- `memory`: buffer sizes used by `HandlerMemory`.
- `policy`: retry/circuit-breaker values used by `HandlerPolicy`.

### Default Values (high-level)
- Memory:
  - `max_checkpoints: 100`
  - `max_telemetry_events: 1000`
- Policy:
  - `max_retries: 2`
  - `circuit_breaker_threshold: 5`
  - `cooldown_seconds: 30`
  - `failure_budget_window_seconds: 300`

---

## Basic Usage

```python
from src.agents.handler import HandlerMemory, HandlerPolicy
from src.agents.handler.utils.handler_error import HandlerError, FailureSeverity

memory = HandlerMemory()
policy = HandlerPolicy()

agent_name = "planner_agent"

if not policy.can_attempt(agent_name):
    raise HandlerError(
        message="Circuit breaker open",
        severity=FailureSeverity.HIGH,
        retryable=True,
        context=policy.breaker_status(agent_name),
    )

checkpoint_id = memory.save_checkpoint(
    label="before_plan",
    state={"task_id": "t-123", "step": "dispatch"},
    metadata={"agent": agent_name},
)

# ... perform operation ...

policy.record_success(agent_name)
memory.append_telemetry({"event": "task_success", "checkpoint_id": checkpoint_id})
```

---

## Operational Guidance
- Use checkpoint labels consistently (e.g., `before_x`, `after_x`) to simplify recovery workflows.
- Include compact but meaningful metadata in checkpoints for auditability.
- Feed breaker status into observability dashboards to detect degraded agent loops.
- Convert all surfaced exceptions to `HandlerError.to_dict()` before external transport.
