## Integration update — 29 October 2026

**Partial enforcement implemented on the draft branch** (not production-ready).

### New changes
- `ExecutionAgent.perform_task(..., request_context=...)` requires an injected trusted `CommandGateway`, a separately supplied request context and a stable task ID; missing values are rejected.
- `ExecutionBoundary` binds task-scoped commands to registered action types, durable idempotency keys and a non-retrying gateway dispatch.
- `ExecutionAgent._execute_selected_action` delegates all actions to that boundary; gateway denials, ambiguous outcomes and duplicate results are not passed into ordinary action recovery.
- `BaseAction.execute` and `BaseAction.actuate` require a scoped, matching command. Direct calls outside that scope are rejected.
- `ExecutionAgent.register_action` denies runtime mutations; vetted action types can be supplied only at bootstrap.
- `alternative_execute` is fail-closed and no longer starts an implicit idle action after failures.
- Robot adapters supplied at bootstrap or through `attach_robot` are wrapped by `GuardedRobotAdapter`; untrusted task data cannot override the robot reference in the action's effective execution context.
- New targeted regression coverage is included.

### Local evidence
`python -m unittest discover -s /mnt/data/slai_execution_patch/tests -p 'test_*.py' -q` resulted in **32/32 isolated tests passing** (25 gateway/guard + 7 boundary), with an incidental sqlite ResourceWarning. The repository-wide tests and full integration tests were **not run** in a complete SLAI environment.

### Remaining high-severity boundaries
- This is not a Python sandbox: privileged Python extensions or callers with raw adapter references can bypass process-local guards. True hardware/OS isolation requires out-of-process adapter isolation and host-controlled capability access.
- The wrapper is a cooperative defense-in-depth facade; hostile reflection or direct access to a raw robot object must be handled by trust isolation, not this Python wrapper.
- Calls to private `_execute` methods or untrusted action class code are not secured by an OS boundary.
- The legacy `SequenceAction` and `FollowPathAction` invoke nested actions directly. Such actions fail closed under name-matched scopes, rather than running. Supported nested actions need separate authorized command dispatch.
- No centralized production authorization provider, config migration, full cancellation semantics, safety-policy integration, durable result auditing or end-to-end integration test has been added.
- A host-provided trusted request_context is mandatory and breaks legacy direct `perform_task(payload)` calls intentionally. Host callsites must be migrated.
- Gateway capabilities are registered at init; supplied action classes must be vetted and trust-bound. `ExecutionBoundary` remains an internal API, not model-accessible.
- An internal exception after an action's side effect is conservatively treated as indeterminate and cannot be blindly retried; durable reconciliation is still required.
- Journal key retention, sensitive-data/OS permissions, concurrency across multiple agent instances and real hardware emergency-stop policy need full validation.

**Do not merge as a complete security production hardening fix.**