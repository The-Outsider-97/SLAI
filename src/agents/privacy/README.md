# Privacy Agent Subsystem

## Overview

The Privacy Agent is SLAI’s **trust/compliance multiplier**. It enforces privacy-by-design controls at runtime across:

- data identification and classification,
- data minimization/redaction,
- consent and purpose validation,
- retention/deletion governance, and
- auditability/evidence generation.

In practical terms, this subsystem decides whether a request should be **allowed**, **modified** (sanitized), or **blocked/escalated**, and it emits the artifacts needed for traceability and policy assurance.

---

## Mission

The Privacy Agent operationalizes personal-data policy into runtime decisions for collection, processing, storage, sharing, and deletion. It turns compliance requirements into executable controls instead of post-hoc checks.

---

## Why SLAI needs this now

SLAI processes user prompts, uploads, extracted browser/reader content, memory artifacts, and downstream tool payloads. As those capabilities and integrations expand, privacy risk expands with them.

A dedicated Privacy Agent helps prevent:

- accidental PII/PHI leakage,
- unauthorized cross-context data sharing,
- retention/deletion policy drift,
- missing audit evidence during reviews/incidents, and
- trust erosion from inconsistent data handling.

---

## Non-overlap with adjacent agents

- **SafetyAgent**: broad risk/harm mitigation.
- **AlignmentAgent**: behavioral/value alignment.
- **Privacy Agent**: lifecycle governance of personal/sensitive data (consent, purpose, minimization, retention, deletion, audit trace).

This separation keeps ownership boundaries clear while allowing coordinated gating in high-risk workflows.

---

## Directory layout

```text
src/agents/privacy/
├── __init__.py
├── data_id.py
├── data_minimization.py
├── data_consent.py
├── data_retention.py
├── privacy_auditability.py
├── privacy_memory.py
├── configs/
│   └── privacy_config.yaml
├── templates/
│   └── privacy_defaults.json
└── utils/
    ├── __init__.py
    ├── config_loader.py
    └── privacy_error.py
```

---

## Core capabilities

### A) Data identification and classification (`data_id.py`)

- Detects PII/PHI/entity signals in structured/semi-structured payloads.
- Tags sensitive attributes and derives contextual sensitivity score.
- Supports regex-, field-, and keyword-based detection with weighted scoring.
- Publishes shared contract artifacts (when enabled) for downstream consumers.

### B) Data minimization and redaction (`data_minimization.py`)

- Enforces least-data-required policy before tool calls or storage.
- Applies field strategies: keep/drop/mask/partial_mask/last4/tokenize/hash.
- Tracks redaction lineage and transformation metadata in `privacy_memory`.
- Supports allowlist + required-field validation to balance utility and risk.

### C) Retention and deletion governance (`data_retention.py`)

- Creates and evaluates retention obligations for records.
- Handles due/expired/legal-hold status transitions.
- Schedules and tracks deletion workflows + SLA compliance.
- Supports right-to-be-forgotten hooks and tombstone-aware lifecycle.

### D) Consent and purpose binding (`data_consent.py`)

- Registers and validates consent artifacts per subject + purpose.
- Enforces purpose limitation and source/destination context constraints.
- Validates allowed processors/actions and legal-basis requirements.
- Fails closed (configurable) when required artifacts/bindings are missing.

### E) Auditability (`privacy_auditability.py`)

- Records decision checkpoints and privacy events.
- Generates evidence bundles with policy metadata + fingerprints.
- Maintains request/record/subject event indexes for investigations.
- Supports sinks for propagating structured audit events.

### F) `privacy_memory` specialized module (`privacy_memory.py`)

`PrivacyMemory` provides stateful, auditable privacy memory with indexed retrieval.

It stores:

- consent artifacts and purpose bindings,
- decision traces and runtime policy outcomes,
- retention/deletion obligations and status transitions,
- redaction metadata and lineage references,
- immutable audit references attached to events.

Key retrieval APIs:

- `consent_status(subject_id, purpose)`
- `retention_obligation(record_id)`
- `privacy_decision_trace(request_id)`

---

## Runtime orchestration (`src/agents/privacy_agent.py`)

`PrivacyAgent.evaluate_privacy(...)` orchestrates the full pipeline:

1. **Identification** (`DataID.identify_entities`)
2. **Consent/Purpose check** (`DataConsent.evaluate_request`)
3. **Minimization** (`DataMinimization.minimize_payload`)
4. **Retention obligation + enforcement** (`DataRetention.create_retention_obligation` + `enforce_retention`)
5. **Audit checkpoint + optional evidence bundle** (`PrivacyAuditability`)
6. **Publish result/error events to shared memory channels**

Final decision behavior:

- `allow`: no blocking constraints, no required mutation
- `modify`: privacy-safe mutation applied (e.g., masking/tokenization)
- `block`/`escalate`: forbidden transfer, missing consent/purpose, retention violations, or fail-closed subsystem errors

---

## Shared memory contract

Suggested integration keys:

- `privacy.sensitivity_score`
- `privacy.detected_entities`
- `privacy.redaction_actions`
- `privacy.retention_policy_id`
- `privacy.consent_status`
- `privacy.audit_trail_ref`

The top-level Privacy Agent also publishes request-scoped summaries/results/errors using a configurable prefix (`privacy_agent.*` by default).

---

## Inputs, outputs, and dependencies

### Inputs

- Raw user prompts and uploads
- Reader/browser extracted content
- Memory reads/writes
- External tool invocation payloads

### Outputs

- allow/modify/block (or escalate) decision
- sanitized payloads
- retention/deletion tasks
- audit event records and evidence bundles

### Integrations

- **SafetyAgent**: joint high-risk gating
- **KnowledgeAgent**: privacy-aware persistence
- **ExecutionAgent**: tool-call payload filtering
- **Compliance layer**: policy synchronization/version alignment

---

## Decision policy (reference)

- **allow**: data is non-sensitive or processing purpose is authorized with valid consent/binding.
- **modify**: sensitive data exists but can be made compliant via minimization/redaction.
- **block**: prohibited transfer, missing/invalid consent, invalid purpose/context/action, or hard policy violation.
- **escalate**: operator/compliance review required (e.g., legal hold or unresolved policy uncertainty).

---

## Configuration model

Primary config file: `src/agents/privacy/configs/privacy_config.yaml`

Main sections:

- `data_id`
- `data_minimization`
- `data_retention`
- `data_consent`
- `privacy_auditability`
- `privacy_memory`

The detection defaults template (`templates/privacy_defaults.json`) supplies runtime defaults and policy metadata for identification/classification behavior.

---

## KPIs

Recommended operational KPIs:

- PII leakage incident rate
- redaction precision/recall
- policy violation prevention count
- deletion SLA compliance
- audit completeness score

---

## Failure modes and mitigations

- **Over-redaction harming utility**
  - Mitigation: context-aware exceptions, tiered masking strategies, required-field safeguards.
- **Under-redaction/privacy leakage**
  - Mitigation: ensemble detection signals + conservative defaults + fail-closed options.
- **Policy drift**
  - Mitigation: versioned policy metadata, template governance, periodic validation and audit bundle sampling.

---

## Minimal usage sketch

```python
from src.agents.privacy_agent import PrivacyAgent
from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory

shared_memory = SharedMemory()
factory = AgentFactory()
privacy_agent = PrivacyAgent(shared_memory=shared_memory, agent_factory=factory)

payload = {
    "email": "user@example.com",
    "message": "Please process my request",
    "diagnosis": "hypertension",
}

result = privacy_agent.evaluate_privacy(
    payload,
    subject_id="subject-123",
    purpose="support_request",
    source_context="chat",
    destination_context="ticketing",
    action="process",
    allowed_fields=["message", "email"],
)

print(result["decision"])           # allow | modify | block | escalate
print(result["sanitized_payload"])  # policy-constrained payload
```

---

## Notes for implementers

- Prefer strict/fail-closed behavior in production unless explicitly justified.
- Treat policy IDs and versions as first-class metadata in every stage.
- Keep privacy decisions deterministic and auditable; avoid silent fallbacks.
- Use `privacy_memory` retrieval APIs as the canonical source when debugging privacy outcomes.

