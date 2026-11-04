# Collaborative Runtime (`src/agents/collaborative`)

This package contains SLAI's **collaborative subsystem internals**: registry, routing, policy/contract enforcement, reliability, and shared-memory primitives.

As of the latest `CollaborativeAgent` update, this folder is now best understood as:

1. **Subsystem layer** (this directory): reusable collaboration building blocks.
2. **Agent-facing orchestration layer** (`src/agents/collaborative_agent.py`): a BaseAgent-compatible coordinator that composes these building blocks and exposes operational modes such as `assess`, `coordinate`, `delegate`, `health`, `metrics`, and shared-memory operations.

---

## What lives in this directory

- `collaboration_manager.py` — top-level subsystem facade used by `CollaborativeAgent` for registration/delegation and manager telemetry.
- `registry.py` — dynamic in-memory registry/discovery of agent instances and capabilities.
- `task_router.py` — capability-aware routing and execution selection.
- `task_contracts.py` — declarative task schemas/validators used before assignment/delegation.
- `policy_engine.py` — prioritized allow/deny/review policy rules.
- `router_strategy.py` — ranking strategies (e.g., weighted and least-loaded).
- `reliability.py` — retry/backoff and circuit-breaker protections.
- `shared_memory.py` — thread/process-safe shared state with TTL/versioning/pub-sub/CAS/metrics.
- `utils/collaboration_error.py` — typed collaboration exceptions + audit/report utilities.
- `utils/collaborative_helpers.py` — normalization, result shaping, redaction, memory wrappers, metrics/audit helpers.
- `configs/collaborative_config.yaml` — collaborative runtime tuning.

> Note: `src/agents/collaborative_agent.py` intentionally **does not own** low-level router/registry/reliability internals. It composes this subsystem through `CollaborationManager` and local policy/contract/risk orchestration.

---

## Updated architecture (Agent facade + collaborative subsystem)

```mermaid
flowchart TD
    A[Caller / Planner / Runtime] --> B[CollaborativeAgent.perform_task]

    B --> C{Mode}
    C --> C1[assess / assess_task]
    C --> C2[coordinate]
    C --> C3[delegate]
    C --> C4[register_agent]
    C --> C5[snapshot / health / metrics]
    C --> C6[shared_get/set/update]

    C1 --> R1[BayesianRiskModel + contextual thresholds]
    C2 --> V1[TaskContractRegistry.validate]
    C2 --> P1[PolicyEngine evaluate]
    C2 --> R1
    C2 --> M1{delegate_during_coordination?}
    M1 -->|yes| D1[CollaborationManager.run_task]
    M1 -->|no / fallback| L1[Local assignment scoring]

    C3 --> D1
    C4 --> D2[CollaborationManager.register_agent]

    D1 --> S1[TaskRouter + Strategy + Reliability]
    S1 --> G1[AgentRegistry]

    B <--> SH[(SharedMemory)]
    B --> AU[Audit events + metrics + histories]

    D1 <--> SH
    D2 <--> SH
```

---

## Responsibilities split (important)

### `CollaborativeAgent` (`src/agents/collaborative_agent.py`)
- BaseAgent-compatible external interface.
- Mode-based dispatch (`assess`, `coordinate`, `delegate`, `register_agent`, `explain`, `snapshot`, `health`, `metrics`, `shared_*`).
- Adaptive Bayesian risk model + contextual thresholding.
- Coordination lifecycle + bounded histories (assessments, coordinations, delegations, metric events).
- Shared memory publication for status, metrics, health, snapshots, and audit events.
- State serialization/deserialization (`serialize_state`, `save_state`, `load_state`).

### Collaborative subsystem (`src/agents/collaborative/*`)
- Manager facade for run/register/list/stats/health/snapshot/export operations.
- Router + strategy + reliability execution mechanics.
- Registry discovery/capability filtering.
- Policy and contract rule execution.
- Shared-memory transport primitives and consistency controls.
- Typed errors and helper utilities.

---

## End-to-end runtime flow

```mermaid
sequenceDiagram
    participant Caller
    participant Agent as CollaborativeAgent
    participant Manager as CollaborationManager
    participant Router as TaskRouter
    participant Registry as AgentRegistry
    participant Target as Selected Agent
    participant Memory as SharedMemory

    Caller->>Agent: perform_task({mode, ...})
    Agent->>Agent: Normalize payload + mode dispatch

    alt mode=coordinate
        Agent->>Agent: Validate contract + evaluate policy + assess risk
        opt delegate during coordination
            Agent->>Manager: run_task(task_type, task_data)
            Manager->>Router: route(...)
            Router->>Registry: get_agents_by_task(...)
            Router->>Target: execute(...)
            Target-->>Router: result/error
            Router-->>Manager: normalized outcome
            Manager-->>Agent: delegation result
        end
        Agent->>Memory: write coordination/status/metrics/audit
        Agent-->>Caller: CoordinationResult payload
    else mode=assess|health|metrics|snapshot|shared_*
        Agent->>Memory: read/write as needed
        Agent-->>Caller: operation result
    end
```

---

## Coordination pipeline details

For each task in a coordination run, `CollaborativeAgent` follows:

1. Normalize task payload and infer `task_type`.
2. Validate task contract (if contract registry is available).
3. Optionally delegate through `CollaborationManager` (config-driven).
4. If not delegated, compute local best-agent score from capability/load/risk/priority weights.
5. Evaluate policy decision (`allow` / `deny` / `require_review`).
6. Perform safety assessment via adaptive thresholding.
7. Emit assignment status:
   - `delegated`
   - `assigned`
   - `assigned_with_review`
   - `rejected_policy`
   - `rejected_high_risk`
   - `rejected_invalid_contract`
   - `unassigned`
8. Publish bounded artifacts (coordination result, histories, metrics, status, audit event) to shared memory when enabled.

---

## Risk model behavior (new in agent-facing layer)

`CollaborativeAgent` uses a Beta-Binomial-style adaptive model:

- Maintains posterior safety observations per key:
  - `task:<task_type>`
  - `agent:<agent_name>`
- Computes dynamic thresholds from:
  - base configured threshold
  - task posterior threshold
  - agent posterior threshold
  - contextual threshold adjustments (e.g., privileged actions, sensitive data)
- Maps risk to levels/actions:
  - `low` → proceed
  - `moderate` → proceed with guardrails
  - `high` → human review
  - `critical` → halt and escalate

This enables risk tolerance to adapt over time while staying bounded by configured min/max thresholds.

---

## Configuration map

Primary knobs are split across agent-level and subsystem-level config sections:

- **Agent-facing (`collaborative_agent`)**
  - Enablement flags for manager/policy/contracts.
  - Coordination toggles (delegate during coordination, local fallback).
  - Risk + Bayesian parameters.
  - Scoring weights (capability/load/risk/priority).
  - History limits and shared-memory keys.
  - Audit/status publication + redaction controls.

- **Subsystem (`collaboration`, `collaboration_manager`, `task_routing`, `reliability`, `shared_memory`)**
  - Manager concurrency/load/threading.
  - Router strategy and retry behavior.
  - Reliability and circuit-breaker behavior.
  - Shared-memory capacity/TTL/versioning/perf settings.

---

## Minimal usage examples

### 1) Use `CollaborativeAgent` (recommended entry point)

```python
from src.agents.collaborative_agent import CollaborativeAgent
from src.agents.collaborative.shared_memory import SharedMemory

memory = SharedMemory()
agent = CollaborativeAgent(shared_memory=memory)

result = agent.perform_task(
    {
        "mode": "coordinate",
        "tasks": [{"id": "t1", "type": "echo_task", "payload": {"message": "hello"}}],
        "available_agents": {
            "EchoAgent": {
                "capabilities": ["echo_task"],
                "current_load": 0,
                "priority": 5,
            }
        },
        "constraints": {"external_user": False},
    }
)

print(result)
```

### 2) Use `CollaborationManager` directly (subsystem-level)

```python
from src.agents.collaborative.collaboration_manager import CollaborationManager
from src.agents.collaborative.shared_memory import SharedMemory

class EchoAgent:
    def execute(self, data):
        return {"echo": data}

memory = SharedMemory()
manager = CollaborationManager(shared_memory=memory)
manager.register_agent("Echo", EchoAgent(), capabilities=["echo_task"])

result = manager.run_task("echo_task", {"message": "hello"}, retries=2)
print(result)
print(manager.get_agent_stats())
```

---

## Operational notes

- Prefer `CollaborativeAgent` when integrating with SLAI's agent runtime and BaseAgent interface.
- Use `CollaborationManager` directly for subsystem-only orchestration/testing.
- Keep policy and contract semantics in their dedicated modules; avoid duplicating logic in callers.
- Shared-memory payloads and audit events are designed to be JSON-safe and optionally redacted for operational safety.
- Manager and agent snapshots/health endpoints are intended for observability, not just debugging.

---

## Durable inter-agent messages and handoffs (v2.4)

`ProjectCoordinationStore` now includes application-independent, project-scoped
message inboxes and exclusive handoff records in its existing SQLite database.
`CollaborativeAgent` exposes thin wrappers around that store; it optionally
publishes opaque message/handoff identifiers through existing `SharedMemory`
pub/sub as **best-effort wake-up hints**. It never publishes message bodies,
handoff details, or authorization contexts to the unauthenticated pub/sub bus.

The host must issue `AuthorizationContext` objects from authenticated identities
and policy, register active agent participants using `join`, and grant the
`collaboration:message` and/or `collaboration:handoff` scope. An agent-provided
payload cannot grant itself these privileges.

```python
from src.agents.collaborative.project_coordination import ProjectCoordinationStore
from src.agents.collaborative_agent import CollaborativeAgent
from src.agents.planning.workflow_runtime import AuthorizationContext

# Example for trusted host code; auth is issued by the host, not agent output.
store = ProjectCoordinationStore("data/collaboration/coordination.sqlite3")
scopes = frozenset({"collaboration:join", "collaboration:message", "collaboration:handoff"})
a = AuthorizationContext("agent-a", "project-1", scopes)
b = AuthorizationContext("agent-b", "project-1", scopes)
store.join(auth=a, kind="agent")
store.join(auth=b, kind="agent")

agent = CollaborativeAgent(project_store=store, shared_memory=None, agent_factory=None)
message = agent.send_agent_message("agent-b", "REQUEST", {"operation": "analyze"},
                                   auth=a, task_id="task-1", idempotency_key="request-1")
received = agent.agent_message_inbox(auth=b)
agent.acknowledge_agent_message(message["message_id"], auth=b)

handoff = agent.prepare_agent_handoff("agent-b", "task-1", "verify", "Verify result",
                                      auth=a, context_refs=["artifact:revision-1"],
                                      idempotency_key="handoff-1")
pending = agent.agent_handoff_inbox(auth=b)  # Durable discovery after missed hints/restart.
accepted = agent.transition_agent_handoff(handoff["handoff_id"], "accept", auth=b)
completed = agent.transition_agent_handoff(handoff["handoff_id"], "complete", auth=b,
                                            result_refs={"verification": "artifact:revision-2"})
agent.shutdown()
store.close()
```

Messages have versioned envelopes, trusted sender IDs, correlation/causation
references, TTLs, bounded JSON payloads, deduplication keys, and explicit
recipient acknowledgments. Delivery acknowledgments **never** imply execution
completion. Handoffs transition `pending -> accepted -> completed/failed`, or
`pending -> rejected/cancelled/expired`. Before acceptance the sender owns the
handoff; afterward the recipient owns it. SQLite transactions and an active-task
uniqueness constraint prevent two simultaneous exclusive handoffs in the same
project. Repeated terminal transitions are rejected.

Important boundaries and current limitations:

- A handoff record is not a substitute for `RevisionWorkflowStore` task claims,
  external authorization, revision checks, or host-side commit fencing. The
  workflow/host remains authoritative for actual side effects.
- Context references identify host-managed artifacts or memory records. The
  exchange does not automatically resolve or persist their content, and
  `SharedMemory` itself remains process-local/manager-backed, not a secure
  durable cross-host context database.
- Published hints are optional, non-durable, and carry only opaque identifiers.
  Consumers must call the authorized store to retrieve records. Polling the
  inbox is the durable recovery path.
- Accepted handoffs presently require an explicit completion/failure decision;
  automatic reassignment of a crashed accepted recipient, exactly-once effects,
  and distributed transport are not implemented.
- Authorization is enforced by the store against the supplied trusted host
  context and participant table; the store cannot authenticate an arbitrary
  caller who is permitted to forge `AuthorizationContext` objects.

`CollaborativeAgent` provides the agent-facing methods directly when a host
injects `project_store`; `ProjectCoordinationStore` exposes the same durable
primitives for other authorized callers. Neither creates or closes a competing
store, preserving existing factory and workflow responsibilities.
