# Technical documentation

* Package: `src/agents/governance/` (stdlib only); agent: `src/agents/governance_agent.py`; config: `governance_agent:` in `agents_config.yaml`.
* State: `data/governance/{processing,oversight,grants,evidence,incidents,decisions,privacy}.db` (SQLite WAL).
* Keys: `SLAI_GOVERNANCE_MASTER_KEY` or `_FILE` (>= 32 chars); domain-separated derivation (identity, grants, evidence, tombstone). `SLAI_SHARED_MEMORY_KEY` for shared-memory persistence.
* Decision states: ALLOW, ALLOW_WITH_CONTROLS, REDACT, RESTRICT, REVIEW_REQUIRED, QUARANTINE, BLOCK; most-restrictive wins; rule errors fail closed.
* `enabled: false` for the governance agent is ignored by design.
* Tests: `tests/compliance/` (stdlib `unittest`).

## Central gate, registration and modes

* **Agent gate** (`governance/agent_gate.py`): `BaseAgent.execute()` (and the out-of-process proxy's `execute`) call `gate_before` / `gate_after`. Installed by `GovernanceAgent`; `AgentFactory.create()` starts the Governance Agent first (`governance_agent.auto_start`). Until installed, the gate is a no-op and `health()` says so.
* **Modes** (`governance_agent.agent_gate.mode`): `off` | `observe` (default: evaluate + durably record, never block) | `enforce` (non-permitting decision raises `GovernanceDenied` before the agent runs). Explicit `governance_context` in the input and `enforce_agents` are always enforced. Without an explicit context the gate evaluates UNCLASSIFIED data, which the policy engine does not permit - so **enforce is only usable once callers supply contexts**.
* **Coverage**: only `execute()`. Direct calls to `perform_task` or other public methods are not gated. Static check: all 31 factory agents inherit `BaseAgent` and none overrides `execute` (`integration_report()["summary"]["covered_by_central_gate"]`).
* **Delegation**: `enforce_capability_registry` is `audit` by default (counts and logs what would be denied; `health()["delegation_audit"]`); set `true` to deny unregistered `agent.action` delegations. `execute` and `perform_task` are registered for every agent (inherited).
* **Registration** (`governance/registrations.py`, run at start-up): erasure adapters + access providers for SharedMemory, SecureMemory, every live PrivacyMemory instance (needs `privacy_memory_tenant_id`), configured `file_store_roots`; incident bridge attached to SharedMemory (`*:latest_incident`). `health()["registration"]` lists what is registered and what is **not covered**, with reasons.
* **Art. 50 marking**: Mapping results of `agent_gate.marking_agents` (default `language`) get `ai_content_marking`; string results cannot carry it and are counted as `marking_unattached`.
* **Learning**: executions of `learning_agents` in a learning mode are evaluated as `LEARN`; in enforce mode every `dataset_ids` entry must be registered for the tenant.
