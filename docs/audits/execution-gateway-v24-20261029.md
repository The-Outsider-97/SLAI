# SLAI v2.4 — Execution security gateway (incremental hardening)

## Audited findings
- **Critical:** `src/agents/execution_agent.py::_execute_selected_action` invokes `action_instance.execute()` after feasibility validation, without a trusted authorization boundary. **Not yet repaired in legacy path.**
- **High:** `_execution_step` retries via `ExecutionRecovery.handle_failure`; external side effects may be duplicated if the previous outcome is uncertain. **Not yet repaired in legacy path.**
- **High:** `ExecutionMemory` uses pickle/shelve for checkpoints/cache and is not a durable command transaction journal. **Not replaced.**
- **Medium:** `ExecutionAgent.__version__` is 2.2.0 on the v2.4 branch; configuration lives in `src/agents/base/configs/agents_config.yaml`. **No version migration performed.**
- **Medium:** README mentions `ExecutionManager`, but the expected `execution_manager.py` was not found at the inspected path.

## Added components
- `src/agents/execution/command_gateway.py`: isolated, general-purpose opt-in command gateway with trusted injected authorization, registered capabilities, JSON-only parameters, basic deadline/version checks, durable SQLite operation claims, idempotency scope, fail-closed unknown commands, explicit uncertain-outcome reporting, and an opt-in SQLite-only transactional handler.
- `tests/test_command_gateway.py`: 18 isolated unittest tests (including denial, revocation, replay suppression, restart, concurrency, ambiguous acknowledgments and SQLite rollback).

## Important boundaries / limitations
1. This is **not integrated** into `ExecutionAgent.perform_task`, `BaseAction.execute`, robot interfaces, or the direct legacy action paths. Those remain unsafe for untrusted effectful requests. No blanket claim that the agent is production-ready.
2. `Authority` must be a trusted host-provided provider. No authenticator or production RBAC provider is supplied.
3. External operations support durable *at-most-one-dispatch intention*, **not exactly once** or universal atomicity. Interrupted IN_PROGRESS requires reconciliation.
4. The SQLite 'transactional' capability only provides the claimed atomicity if the registered handler exclusively uses the same supplied connection, does not commit independently, and makes no external effects. These contracts require host enforcement.
5. The reconciliation API requires trusted external policy and evidence; its authority callback must verify an authenticated caller in a production integration.
6. The journal does not persist results or audited authorizations; sensitive payloads are not written, but the journal needs OS-level access controls and lifecycle/retention policies.
7. The isolated tests ran locally; repository-wide regression, security and factory tests have **not** run.

## Verified commands
```sh
python -m unittest discover -s tests -p 'test_command_gateway.py' -v
```
In an isolated harness, **18/18 tests passed** using Python 3's standard library. This is a narrow test result, not a production gate.

## Required next stage
Wire trusted authorization and all side-effect adapters into enforced execution boundaries, protect all legacy and fallback paths, implement policy/configuration integration under the existing agent configuration loader, add cancellation/recovery, independent cross-process tests, and run the full SLAI regression suite in the actual repository environment.
