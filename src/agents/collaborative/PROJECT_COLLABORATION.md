# SLAI v2.4 — Typed workflow and project coordination integration

This describes the implemented, application-independent extension to the
existing SLAI planning and collaborative runtimes. It is deliberately **not**
a new agent factory, resource editor, revision database, or commit authority.

## Ownership and actual runtime paths

| Responsibility | Existing / extended implementation |
| --- | --- |
| Agent construction and registration | `AgentFactory` and its registry (unchanged) |
| Computational capability dispatch | `planning.workflow_runtime.OrchestratorCapabilityAdapter` (extended) |
| Durable task states, dependencies, input revisions | `planning.revision_workflow.RevisionWorkflowStore` (extended contract) |
| External mutation and uncertain-outcome reconciliation | `planning.project_adapter` and `planning.commit_reconciliation` (unchanged) |
| Shared agent task routing and policies | `collaborative.collaboration_manager` (unchanged) |
| Durable project participant sessions, advisory leases, event cursors | `collaborative.project_coordination.ProjectCoordinationStore` (new) |
| Agent-facing project coordination | `CollaborativeAgent` with injected `project_store` (extended) |
| Authoritative resource data, revisions, permissions, and commits | Host application / injected callbacks |

The existing Orchestrator still supports its legacy delegation behavior,
including dynamic specialist creation. The strict workflow adapter **does not**
use that fallback: it dispatches only an explicitly approved, already
registered specialty. Applications are not considered integrated merely
because they appear in a design brief.

## Typed computational capabilities

```python
from src.agents.planning.workflow_runtime import (
    AuthorizationContext, CapabilityBinding, OrchestratorCapabilityAdapter,
    RevisionWorkflowRunner,
)
from src.agents.planning.revision_workflow import RevisionTask, RevisionWorkflowStore

binding = CapabilityBinding(
    specialty="stem", version="1.0", required_scope="workflow:execute",
    input_schema={"type": "object", "properties": {"value": {"type": "number"}},
                  "required": ["value"], "additionalProperties": False},
    result_schema={"type": "object", "required": ["status", "output_refs"]},
)
adapter = OrchestratorCapabilityAdapter(
    orchestrator, {"calculate": binding}, registered_specialties={"stem"},
)
auth = AuthorizationContext("user-from-host", "project-from-host", frozenset({"workflow:execute"}))
capabilities = adapter.discover(auth)
workflows = RevisionWorkflowStore("data/planning/revision_workflows.sqlite3",
                                  revision_provider=host.current_revisions)
plan = workflows.create_plan("project-from-host", [
    RevisionTask("compute", capability="calculate", capability_version="1.0",
                 reads={"input": "revision-from-host"}),
], {"input": "revision-from-host"})
outcome = RevisionWorkflowRunner(workflows, adapter).run_until_blocked(
    plan["plan_id"], {"compute": {"value": 3}}, auth=auth,
)
```

`orchestrator` is an existing `OrchestratorAgent` with an existing factory;
`host` is the real host revision provider. These values must be wired by the
application. The schemas use JSON Schema Draft 2020-12; malformed schemas are
rejected at binding construction. Inputs are checked before dispatch and
results after delegation. Schema mismatches become explicit failed task
attempts; an incompatible task capability version remains blocked and unclaimed.
The adapter rejects mismatched project/actor IDs and supplies the trusted
principal identity to the delegated request; domain adapters must also check
authorization for any resource IDs embedded in application-specific input.

`CapabilityBinding.side_effects` currently permits only `"none"`. This runner
is for computations and proposals; it never asserts that a host mutation has
committed. Use `HostCommitRequest` and `CommitReconciliationStore` for external
commits, with an application adapter enforcing an **atomic compare-and-swap**
of the expected revision and a **durable idempotency key**.

## Project participation and leases

```python
from src.agents.collaborative.project_coordination import ProjectCoordinationStore

projects = ProjectCoordinationStore("data/collaboration/projects.sqlite3")
identity = AuthorizationContext(
    "host-verified-user", "project-from-host",
    frozenset({"collaboration:join", "collaboration:read", "collaboration:lock"}),
)
projects.join(auth=identity, kind="human")
lease = projects.acquire("resource-id-from-host", auth=identity, ttl=30)
# Host edit/commit occurs elsewhere; the host must validate its own revisions
# and any required fencing token AT its authoritative mutation boundary.
projects.release("resource-id-from-host", lease["lease_id"],
                 lease["fencing_token"], auth=identity)
snapshot = projects.snapshot(auth=identity)
events = projects.events(auth=identity, after=snapshot["event_cursor"])
```

A host may inject this store into `CollaborativeAgent(..., project_store=projects)`
and call `join_project`, `acquire_resource_lease`, `release_resource_lease`,
`project_snapshot`, and `project_events` on the agent. The store lifecycle belongs to the host, not
to the agent. These privileged methods are **not** exposed as modes of the
general `perform_task` payload because arbitrary agent input must not supply
trusted identity or grant project access.

Participant sessions and leases persist in SQLite. A short `BEGIN IMMEDIATE`
transaction ensures one successful acquisition per `(project_id, resource_id)`
in that database; unrelated resources can be worked on without holding a
global transaction. Lease tokens increase after release, expiry, and restart.
Ownership is checked on renewal and release. A transactionally consistent
snapshot provides a cursor plus active members/leases. Per-project event
cursors then provide bounded replay of subsequent metadata changes. The
`active` participant flag indicates registered membership, not proof of a
live network connection.

**Limitations:** These leases are coordination metadata, not an authoritative
model lock or distributed transaction. A host must check the fencing token
inside its own mutation transaction if lease enforcement is required. The
store uses wall-clock lease expiry and therefore assumes reasonably
synchronized clocks among writers; it is not a consensus-backed service.
Event replay covers collaboration metadata only, not application edits or
committed revision notifications. Offline merge, shared-read locks, and
multi-resource atomic edits are not implemented here.

## Authorization, revocation, and mutation safety

The host must construct `AuthorizationContext` from authenticated principal
and project policy. Passing client- or agent-generated scopes is unsafe.
If the host can check current permissions, it should provide
`live_scope_check(actor_id, project_id, scope) -> bool`. Every `require()` call
then re-checks current permissions (including after specialist execution and
during lease renewal); explicit expiry is also enforced. If no callback is
provided, the context is a time-bounded authorization snapshot, **not** live
revocation protection. For true commit-time authorization, the host must
validate permissions atomically or under its own correct consistency policy.

Existing workflow state records task run IDs, input revisions, result refs,
invalidations and uncertain outcomes. A computational success is not a
confirmed edit. The existing commit outbox preserves `outcome_unknown` after
interrupted external calls and uses authoritative host lookup before
reconciliation; do not retry non-idempotent writes blindly.

## Integration checklist for SLAI applications

1. Expose real host resource/revision lookup (do not synthesize revision IDs).
2. Register real agents through the existing `AgentFactory`.
3. Define trusted typed `CapabilityBinding` records with documented versions,
   required scopes and input/result schemas.
4. Submit `RevisionTask` records through `RevisionWorkflowStore`; include exact
   source revisions and capability version when a task depends on it.
5. Issue `AuthorizationContext` from the host and perform live policy checks
   where required; never trust agent-supplied scopes.
6. Inject a durable project coordination store if sessions, leases and event
   cursors are required; choose resources and granularity in the host.
7. Validate proposed changes in the host, including domain semantics,
   revisions, permissions and (if applicable) fencing tokens.
8. Commit through the existing host adapter and reconciliation outbox. Handle
   `confirmed`, `rejected`, and `outcome_unknown` separately.
9. Publish verified host revisions to provenance and application subscribers.
   Project coordination events alone do not publish committed document edits.

## Verification and outstanding work

Focused tests live in `tests/agents/test_workflow_runtime.py`,
`test_revision_workflow.py`, and `test_project_coordination.py`. They cover
typed schemas, version compatibility, participant isolation, concurrent
independent leases, ownership, lease expiry, fencing increments, restarts,
event replay, and live permission revocation through the shared context.

This is an incremental slice, not a claim of full P0–P6 completion. Remaining
work includes real application adapters and end-to-end UI integration,
specialization profiles/selection beyond approved specialty mappings,
distributed lease enforcement at actual host commits, cancellation semantics,
offline reconciliation, broader adversarial tests, and measured performance.
