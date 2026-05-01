# Safety Modules Pipeline & Contracts

This document provides a detailed module-level companion to `src/agents/safety/README.md`.

## Orchestration contract summary

The Safety Agent expects each module to provide **structured, machine-readable** output so that risk can be aggregated consistently.

Minimum expected shape (conceptually):

- `risk_score` (0.0 to 1.0 where applicable)
- findings/details payloads
- optional recommendation/decision fields

---

## Module responsibilities

| Module | Primary responsibility | Typical contribution to aggregation |
|---|---|---|
| `SafetyGuard` | Early sanitization + immediate unsafe-content interception | Can force early blocking path, contributes guard-level risk context |
| `CyberSafetyModule` | Threat pattern/signature/context analysis | High-signal cyber risk score and findings |
| `AdaptiveSecurity` | URL/email phishing and transport-surface checks | Additional targeted cyber indicators |
| `RewardModel` | Policy/ethics-alignment scoring | Composite reward/safety score -> mapped risk |
| `AttentionMonitor` (optional) | Attention anomaly/alignment signals | Optional risk bump when anomalous |
| `ComplianceChecker` | Compliance posture checks | May introduce warnings/blockers depending on status |
| `SecureSTPA` | Unsafe control action and system-hazard reasoning | Action-validation risk and governance evidence |

---

## Assessment pipeline details (`perform_task`)

```mermaid
sequenceDiagram
    participant U as Upstream Caller
    participant SA as SafetyAgent
    participant SG as SafetyGuard
    participant CS as CyberSafetyModule
    participant AS as AdaptiveSecurity
    participant RM as RewardModel
    participant AM as AttentionMonitor
    participant CC as ComplianceChecker

    U->>SA: perform_task(data, context)
    SA->>SG: analyze_input(sanitized_text)
    SG-->>SA: guard_report

    alt guard blocks
        SA-->>U: decision=block (bounded response)
    else continue
        SA->>CS: analyze_input(...)
        CS-->>SA: cyber risk/findings
        SA->>AS: analyze_email/analyze_url (when applicable)
        AS-->>SA: phishing/security report
        SA->>RM: evaluate(...)
        RM-->>SA: composite score
        opt attention data available
            SA->>AM: analyze_attention(...)
            AM-->>SA: attention risk
        end
        SA->>CC: check_compliance(...)
        CC-->>SA: status/findings
        SA->>SA: constitutional checks + weighted aggregation
        SA-->>U: SafetyAssessment {allow|review|block}
    end
```

---

## Action-validation pipeline details (`validate_action`)

```mermaid
sequenceDiagram
    participant C as Caller
    participant SA as SafetyAgent
    participant CS as CyberSafetyModule
    participant RM as RewardModel
    participant STPA as SecureSTPA

    C->>SA: validate_action(action_params, action_context)
    SA->>CS: analyze_input(action payload)
    CS-->>SA: cyber risk
    SA->>RM: evaluate(action payload)
    RM-->>SA: reward/composite score
    SA->>STPA: action/system hazard checks
    STPA-->>SA: unsafe control findings
    SA->>SA: constitutional checks + corrections + decision
    SA-->>C: ActionValidationResult
```

---

## Notes for maintainers

1. Keep module outputs stable; if a field changes, update aggregation logic and docs together.
2. Keep threshold names/config alignment synchronized with code.
3. Preserve fail-closed behavior for critical module errors unless deliberately reconfigured.
4. When adding a new module, update:
   - top-level safety README,
   - this module pipeline doc,
   - config defaults and tests.

