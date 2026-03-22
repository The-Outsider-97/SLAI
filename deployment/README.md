# SLAI Deployment System

This module provides a robust, modular, and rollback-first deployment architecture for SLAI (`v1.6.2+`). It supports version tagging, model rollback, Git history management, CI/CD triggering, and environment-aware deployments (e.g., `dev`, `staging`, `prod`).

It integrates with SLAI's core language architecture and is designed to ensure **traceability**, **recoverability**, and **structured release management** during rapid development and experimentation.

---

## Core Features

- **Rollback-First**: Full rollback support for model files and Git history (tags/commits).
- **Audit Logging**: Structured JSONL event logging with system metrics and sensitive field scrubbing.
- **Versioning**: SemVer-compliant Git tag creation and automated version bumping.
- **Branch Operations**: Safe creation, merging, and deletion of Git branches.
- **CI/CD Triggering**: Single authoritative trigger path via `trigger_ci()` + provider connector adapters.
- **Release Invariants**: Automatic lockfile generation/verification and environment invariant verification in pre-deploy checks.
- **Multi-Environment Support**: Isolate and manage deployments per environment (`dev`, `prod`, etc.).

---

## Directory Structure

```bash
deployment/
├── audit_logger.py
├── ci_trigger.py              # Authoritative CI trigger path + provider connectors + retry/auth validation
├── deployment_manager.py      # High-level orchestrator: deploy/rollback across environments
├── release_invariants.py      # Lockfile + environment invariant generation and verification
├── rollback/
│   ├── __init__.py
│   ├── model_rollback.py
│   └── code_rollback.py
├── git_ops/
│   ├── __init__.py
│   ├── branch_ops.py
│   ├── commit_ops.py
│   └── version_ops.py
└── tests/
    ├── test_ci_trigger_contract.py
    └── test_release_invariants.py
```

---

## CI Triggering Contract

`deployment/ci_trigger.py` enforces one dispatch flow:

1. `trigger_ci(env, branch)`
2. `get_connector(env)`
3. provider connector `.trigger(branch)`

For GitHub this includes:
- branch format validation,
- environment token validation,
- API dispatch to workflow `dispatches` endpoint,
- transient retry behavior for HTTP/network failures,
- explicit auth and non-retryable error handling.

---

## Release Invariants

`deployment/release_invariants.py` provides reproducibility artifacts integrated into release/deploy checks:

- `requirements.lock`: deterministic lock metadata generated from normalized `requirements.txt`.
- `deployment/environment_invariants.json`: runtime invariants (python version, platform, machine, deploy environment).

`DeploymentManager._pre_deployment_check()` now calls:

```python
ensure_release_invariants(strict=True)
```

So release flow fails fast when lockfile/environment contracts are violated.

---

## Contract Tests

`deployment/tests/` now includes contract tests for:
- mock CI API success dispatch,
- auth-token failure path,
- retry behavior on transient HTTP/network failures,
- fallback/failure branch after retry exhaustion,
- branch input validation,
- release invariant generation and verification integration.
