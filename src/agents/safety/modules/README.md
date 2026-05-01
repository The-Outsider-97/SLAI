# Safety Modules Reference

This directory-level reference documents the modules orchestrated by `SafetyAgent`.

## Module inventory

| Module | File | Role in pipeline |
|---|---|---|
| SafetyGuard | `src/agents/safety/safety_guard.py` | Sanitizes and may block unsafe content before deeper analysis. |
| CyberSafetyModule | `src/agents/safety/cyber_safety.py` | Detects cyber risks via rules, signatures, heuristics, and event-anomaly logic. |
| AdaptiveSecurity | `src/agents/safety/adaptive_security.py` | Specialized checks for phishing and URL/email-oriented attack surfaces. |
| RewardModel | `src/agents/safety/reward_model.py` | Produces ethical/safety scores used in final gating decisions. |
| AttentionMonitor (optional) | `src/agents/safety/attention_monitor.py` | Analyzes attention matrices for anomaly or alignment-related concerns. |
| SecureSTPA | `src/agents/safety/utils/secure_stpa.py` | Performs STPA-style unsafe control action analysis for action validation. |
| ComplianceChecker | `src/agents/safety/compliance_checker.py` | Policy/compliance verification module for governance-style checks. |
| SecureMemory | `src/agents/safety/secure_memory.py` | Stores security-relevant events and supports traceability/auditing. |

---

## CyberSafetyModule deep dive

`CyberSafetyModule` contributes both **content-level** and **event-level** defenses.

### 1) Content-level analysis (`analyze_input`)

- Converts input to string/JSON-safe representation.
- Runs configurable rule pattern matching:
  - Sensitive keyword leakage patterns
  - Weak credential/password patterns
  - Injection-like patterns
- Runs vulnerability signature matching (CVE-style signatures).
- Applies context heuristics (`code_review`, `config_file`, `api_request`).
- Returns:
  - `risk_score`
  - `findings`
  - `recommendations`

### 2) Event-level analysis (`analyze_event_stream`)

- Maintains rolling event history.
- Calculates statistical anomaly signals (z-score style).
- Tracks sequence rarity patterns per entity/user/IP.
- Supports optional QNN-inspired vector scoring path.
- Produces an anomaly score and thresholded decision.

---

## Integration expectations

When extending safety modules:

1. Keep module outputs structured and machine-readable.
2. Add explicit threshold semantics in config and docs.
3. Preserve graceful degradation for optional dependencies.
4. Update top-level `src/agents/safety/README.md` orchestration diagrams.
5. Ensure template/config files remain consistent with module assumptions.
