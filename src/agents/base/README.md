# Base Runtime Utilities (`src/agents/base`)

This package provides the shared runtime support used by `BaseAgent` (`src/agents/base_agent.py`) and, by extension, all role-specific SLAI agents.

The recent `BaseAgent` update introduced a stronger execution envelope (retries, recovery, auditing, lifecycle events, execution history, and compatibility dispatch), and this folder contains the helper modules that power those behaviors.

---

## What this package is responsible for

`src/agents/base/` is focused on **reusable runtime primitives**, not domain logic.

It supports:

- lazy component initialization for expensive resources
- lightweight metric tracking and value recording
- issue recovery orchestration and known-issue handlers
- centralized configuration loading and validation utilities (`utils/`)
- reusable domain-agnostic helper modules for sanitization/constraints/encoding (`modules/`)
- foundational memory abstractions (`base_memory.py`)

---

## Directory structure

```text
base/
├── __init__.py
├── README.md
├── base_memory.py
├── issue_handler.py
├── lazy_agent.py
├── light_metric_store.py
├── configs/
│   ├── agents_config.yaml
│   └── base_config.yaml
├── modules/
│   ├── __init__.py
│   ├── activation_engine.py
│   ├── base_tokenizer.py
│   ├── base_transformer.py
│   ├── biology_constraints.py
│   ├── chemistry_constraints.py
│   ├── input_sanitizer.py
│   ├── math_science.py
│   ├── numpy_encoder.py
│   └── physics_constraints.py
└── utils/
    ├── __init__.py
    ├── base_errors.py
    ├── base_helpers.py
    ├── config_loader.py
    └── main_config_loader.py
```

> Note: `BaseAgent` itself lives in `src/agents/base_agent.py` and imports key modules from this directory (for example: `LazyAgent`, `LightMetricStore`, `IssueHandler`, and config/error helpers).

---

## How `BaseAgent` uses these components

### 1) `LazyAgent` (`lazy_agent.py`)

`BaseAgent` initializes a lazy wrapper in `_init_core_components()`:

- defers expensive component creation until first use
- keeps startup fast and memory-friendly
- allows optional features to be loaded only when needed

`BaseAgent` also provides `register_lazy_component(...)` and `lazy_property(...)` so subclasses can add their own deferred components safely.

### 2) `LightMetricStore` (`light_metric_store.py`)

`BaseAgent` uses `LightMetricStore` for runtime observability:

- tracks timing for `execute` envelope start/stop
- records per-metric values when performance data is emitted by task results
- supports lightweight analytics without heavy external dependencies

### 3) `IssueHandler` (`issue_handler.py`)

Error recovery is now layered:

1. main task execution (with retry/backoff)
2. centralized issue-handler attempt (`IssueHandler.handle_issue(...)`)
3. local registered known-issue handlers (pattern-based)
4. alternative fallback execution path (`alternative_execute`)

This layered approach improves resilience while keeping subclass implementations simple.

### 4) Config + support utilities (`utils/main_config_loader.py`, etc.)

`BaseAgent` reads `base_agent` config values through shared loaders and validates thresholds/ranges centrally.

This includes runtime controls such as:

- retry counts and backoff windows
- shared-memory audit toggles and key prefixes
- execution-history limits
- similarity thresholds and plan monitoring thresholds
- metric buffer sizes by memory profile (`low` / `medium` / `high`)

---

## `BaseAgent` runtime capabilities reflected by this package

The updated runtime now provides:

- **Execution records** via structured `ExecutionRecord` entries
- **Lifecycle events** (initialized, retry, execution_recorded, etc.) written to shared memory
- **Error audit logs** with similarity detection for repeated failures
- **Recovery path controls** (`enable_known_issue_recovery`, `enable_alternative_execute`)
- **Capability dispatch** that can call compatible `predict`, `get_action`, or `act` methods
- **Plan execution helpers** (`execute_plan`, `execute_step`, `recover_step`, `compile_results`)
- **Evaluation/retraining hooks** via metric thresholds and retraining flags
- **Optional torch helpers** (`create_lightweight_network`, `update_projection`) with lazy import guards

All of these rely on utilities and support modules in this directory.

---

## Usage guidance for agent authors

When implementing a new agent that inherits `BaseAgent`:

1. implement at least one callable capability (`predict`, `get_action`, or `act`) **or** override `perform_task`
2. return metric fields (for example `accuracy`, `latency_ms`, `loss`) when available to feed performance tracking
3. register lazy components for expensive state instead of eager-loading everything in `__init__`
4. add custom known-issue handlers when your domain has predictable recoverable failures
5. keep shared-memory payloads compact and serializable for reliable audit/event persistence

---

## Minimal example

```python
from typing import Any, Dict

from src.agents.base_agent import BaseAgent
from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory


class DemoAgent(BaseAgent):
    def predict(self, state: Any, context: Any = None) -> Dict[str, Any]:
        return {
            "status": "success",
            "prediction": state,
            "accuracy": 0.99,
            "latency_ms": 12.4,
            "context": {"source": "demo"},
        }


shared_memory = SharedMemory()
agent = DemoAgent(shared_memory=shared_memory, agent_factory=AgentFactory())
result = agent.execute({"operation": "predict", "input_data": {"text": "hello"}})
print(result)
```

---

## Summary

`src/agents/base/` is the **support layer** for the production `BaseAgent` runtime. As `BaseAgent` evolves, this package should remain the single, consistent place for shared initialization, metrics, issue handling, and config-driven runtime utilities.
