# Final report - SLAI regulatory governance (AVG/GDPR, UAVG, EU AI Act)

Generated 2026-11-24 from the repository state described below. **This document is not legal advice and not a certification.** No statement here claims that SLAI is compliant with GDPR, UAVG or the EU AI Act; it states what the software enforces, what has been tested, and what remains open.

## 1. Verified repository state
- Remote: `https://github.com/The-Outsider-97/SLAI.git` (not contacted from the sandbox; remote contents not verified)
- Branch: `claude/exact-box-split`; HEAD `df7f3b10d3be690bc3a8cc39781a4dfcc48658a8` (2026-11-22 16:51:59 +0000)
- **Nothing is committed.** Working tree: 301 changed paths in total; 79 new files and 12 modified files belong to this work (lists in section 8). The remaining 283 paths (data files, `applications/tools`, `.env`, a `.joblib`, language artifacts, a spatial test, ...) were already present (modified or untracked) before this work and were not changed by it. **`.env` is a tracked, modified file: do not commit it with this work.**
- Test environment: Linux sandbox, Python 3.10.12, `cryptography` 50.0.2 present, **`torch` absent** (tests shim `torch` only when it is missing). Nothing was run on the Windows venv at `G:\GAIA\AI\SLAI`, and the CI workflow has never run.

## 2. Regulatory sources and dates
| Item | Status |
|---|---|
| Regulation (EU) 2026/1744 (Digital Omnibus on AI), OJ 24.7.2026, in force 27.7.2026 | Existence and status confirmed on EUR-Lex (2026-11-24) |
| Art. 50 general application 2026-08-02; Art. 50(2) marking grace for systems on the market before that date until **2026-12-02** | From several law-firm summaries; consolidated text **not read** |
| New Art. 5 prohibitions (non-consensual intimate material, CSAM) from **2026-12-02** | Same; modelled as tag `csam_ncii_generation`, applied immediately (conservative) |
| Annex III high-risk 2027-12-02; Annex I 2028-08-02; GPAI 2025-08-02 (legacy 2027-08-02) | Same |
| GDPR articles, UAVG (child age 16, BSN rules), Commission adequacy list | Not re-verified in this work; counsel must confirm |

Two dates fall within days of this report: Art. 50(2) marking for systems already on the market (2026-12-02) and the new Art. 5 prohibitions (2026-12-02).

## 3. Audit findings 4.1 - 4.10
| # | Finding (as referenced in code/tests) | Status | Evidence |
|---|---|---|---|
| 4.1 | Disabled Privacy Agent must fail closed | Implemented but not fully verified | `test_privacy_agent_fail_closed.py` runs against the real source; the full agent was not run (torch) |
| 4.2 | Durable authoritative consent/restriction state | Implemented but not fully verified | `privacy_ledger.py`; `test_consent.py` |
| 4.3 | (no code or test in this work references finding 4.3) | **Cannot confirm** - I do not have the audit text for 4.3 in my records | Please supply it; it is not claimed as addressed |
| 4.4 | Verified distributed deletion | Partially implemented | Adapters for 5 store types tested end to end; external recipients, backups, model weights unreachable |
| 4.5 | Erasure request: reason not mandatory, identity verification required | Implemented but not fully verified | `test_retention.py`; also found and fixed a legal-hold bypass |
| 4.6 | Delegation only to public, authorised actions | Partially implemented | Structural guard always on; registry mode `audit` by default, `true` available and tested |
| 4.7 | SecureMemory tenant isolation / fail closed | Implemented but not fully verified | `test_tenant_isolation.py` |
| 4.8 | Authenticated, safe, versioned shared-memory persistence | Implemented but not fully verified | `test_secure_persistence.py`; AES-GCM only when `cryptography` is installed |
| 4.9 | No PII/secrets in logs | Partially implemented | Pattern-based filter (best effort); `test_diagnostics_sanitization.py` |
| 4.10 | Operational score must not imply legal compliance | Partially implemented | `legal_compliance: not_determined`; no separate compliance auditor module exists in the repo |

## 4. Architecture (as built)
- `GovernanceAgent` (`src/agents/governance_agent.py`) owns a `GovernanceCore` (`src/agents/governance/core.py`): deterministic policy engine (most restrictive wins, rule errors -> BLOCK), processing registry (Art. 30), lawful-basis validation, human oversight, signed grants/identity/evidence (HMAC, domain-separated keys), incident registry, privacy ledger, learning registry, subject-rights workflows, privacy dashboard API, transparency marker.
- **Central gate** (`agent_gate.py`): `BaseAgent.execute()` and the out-of-process proxy call it. Modes `off | observe | enforce`; **default `observe`: decisions are recorded, nothing is blocked.** Explicit `governance_context` and `enforce_agents` are always enforced.
- **Registration** (`registrations.py`): at start-up erasure adapters + access providers + incident bridge are registered for the reachable stores, and the report lists what is not covered.
- `AgentFactory.create()` starts the Governance Agent before any other agent (`auto_start`); failure is logged CRITICAL, recorded, and fatal only if `require_for_agent_creation` is true.

## 5. Agent-by-agent integration (static analysis + runtime tests)
Basis: static analysis of source; evidence of integration, not proof of runtime behaviour. Capabilities are registered for all 31 agents (487 actions incl. inherited `execute`/`perform_task`). **No agent calls the governance or privacy gate in its own code** (0 / 0); coverage is the central gate in `BaseAgent.execute()` only. Direct calls to `perform_task` or other public methods are not gated. No agent contains its own GDPR/AI-Act engine (0 found).

| Agent | Class | Capabilities | Covered by central gate (`execute`) | Own GDPR/AI Act engine |
|---|---|---|---|---|
| adaptive | AdaptiveAgent | 18 | yes | no |
| alignment | AlignmentAgent | 17 | yes | no |
| animation | AnimationAgent | 6 | yes | no |
| browser | BrowserAgent | 33 | yes | no |
| collaborative | CollaborativeAgent | 39 | yes | no |
| evaluation | EvaluationAgent | 16 | yes | no |
| execution | ExecutionAgent | 14 | yes | no |
| financial | FinancialAgent | 16 | yes | no |
| governance | GovernanceAgent | 9 | yes | no |
| handler | HandlerAgent | 8 | yes | no |
| knowledge | KnowledgeAgent | 27 | yes | no |
| language | LanguageAgent | 16 | yes | no |
| learning | LearningAgent | 9 | yes | no |
| network | NetworkAgent | 7 | yes | no |
| observability | ObservabilityAgent | 5 | yes | no |
| optimization | OptimizationAgent | 10 | yes | no |
| orchestrator | OrchestratorAgent | 4 | yes | no |
| perception | PerceptionAgent | 25 | yes | no |
| planning | PlanningAgent | 21 | yes | no |
| privacy | PrivacyAgent | 6 | yes | no |
| provenance | ProvenanceAgent | 34 | yes | no |
| qnn | QNNAgent | 34 | yes | no |
| quality | QualityAgent | 10 | yes | no |
| reader | ReaderAgent | 8 | yes | no |
| reasoning | ReasoningAgent | 32 | yes | no |
| safety | SafetyAgent | 18 | yes | no |
| simulation | SimulationAgent | 12 | yes | no |
| software | SoftwareAgent | 8 | yes | no |
| spatial | SpatialAgent | 7 | yes | no |
| stem | STEMAgent | 10 | yes | no |
| verification | VerificationAgent | 8 | yes | no |

Gate is enforced for none of them by default. `marking_agents` default: `language` only. Learning-mode evaluation applies to `learning`. Agents that process personal data are not yet declared in `personal_data_agents` (that is a business/legal decision); all capabilities are therefore limited to non-personal/anonymised classes.

## 6. Requirement classes
Class counts below use the 27 rows of `configs/traceability.json`. **Zero rows are classed 'Implemented and verified'**: the system's own rule requires independent, signed evidence from a producer other than the author (`verification_agent` / `ci_pipeline`), and none has been recorded. The reporter's computed status differs from my class where linked controls have no evidence yet - both are shown.

| Class | Rows |
|---|---|
| Implemented and verified | 0 |
| Implemented but not fully verified | 8 |
| Partially implemented | 13 |
| Missing or failed | 0 |
| Not applicable | 0 |
| Dependent on deployment | 0 |
| Dependent on organisational or legal action | 6 |

## 7. Traceability matrix
| Article | Requirement | Declared | Reporter (computed from evidence) | Class | Note |
|---|---|---|---|---|---|
| GDPR Art. 5(1)(a)-(b) | Lawfulness, purpose limitation | partial | missing | Partially implemented | Engine enforces purpose/activity/lawful-basis rules where the gate is used; default gate mode is observe, so nothing is blocked by default. |
| GDPR Art. 6 | All six lawful bases validated with evidence | implemented | implemented_not_fully_verified | Implemented but not fully verified | Unit-tested; no independent signed evidence. |
| GDPR Art. 7-8 | Consent validity, withdrawal, children | partial | missing | Partially implemented | Ledger + rule tested; consent collection UI and child-age rule (16) are application/legal decisions. |
| GDPR Art. 9-10 | Special-category and criminal-offence conditions | implemented | implemented_not_fully_verified | Implemented but not fully verified | Unit-tested. |
| GDPR Art. 12-14 | Privacy notices | organisational | dependent_on_organisational_action | Dependent on organisational or legal action | Content and delivery are organisational. |
| GDPR Art. 15-16 | Access and rectification workflows | partial | partial | Partially implemented | Access works for registered stores; rectification needs per-store apply callbacks - none registered. |
| GDPR Art. 17-19 | Erasure with verified completion, restriction, recipient notification | partial | missing | Partially implemented | Verified for SharedMemory, SecureMemory, PrivacyMemory (all live instances), files, learning registry. Not reachable: external recipients, offline backups (tombstones only), model weights. |
| GDPR Art. 20-21 | Portability and objection | partial | partial | Partially implemented | Objection enforced; portability exports only stores declared portable (legal decision, none by default). |
| GDPR Art. 22 | Automated decisions: exception check and human intervention | implemented | implemented_not_fully_verified | Implemented but not fully verified | Unit-tested; depends on callers declaring automated_decision. |
| GDPR Art. 24-25 | Accountability, privacy by design/default | partial | missing | Partially implemented | Fail-closed design, sanitised logs, evidence chain; org accountability documents outstanding. |
| GDPR Art. 26-29 | Controller/processor roles and recipient registry | partial | partial | Partially implemented | Roles in context/registry; no contract management. |
| GDPR Art. 30 | Records of Processing Activities | implemented | dependent_on_organisational_action | Dependent on organisational or legal action | Register mechanism exists; its content is the organisation's. |
| GDPR Art. 32 | Security of processing | partial | missing | Partially implemented | Sealed persistence, tenant isolation, log sanitising, key derivation. No penetration test; keys must be provisioned. |
| GDPR Art. 33-34 | Breach lifecycle and deadlines | implemented | implemented_not_fully_verified | Implemented but not fully verified | 72h/Art. 34 tracking, drafts only; submission is a human act. |
| GDPR Art. 35-36 | DPIA and prior consultation | organisational | dependent_on_organisational_action | Dependent on organisational or legal action | No DPIA tooling. |
| GDPR Art. 37-39 | DPO | organisational | dependent_on_organisational_action | Dependent on organisational or legal action | Roles are configurable; appointment is organisational. |
| GDPR Art. 44-49 | International transfers | implemented | implemented_not_fully_verified | Implemented but not fully verified | Policy implemented; adequacy list and SCC/TIA not verified or provided. |
| AI Act Art. 5 | Prohibited practices | implemented | implemented_not_fully_verified | Implemented but not fully verified | Includes the Omnibus NCII/CSAM prohibition, applied immediately (not date-gated). |
| AI Act Art. 6, Annex I/III | Risk classification | implemented | implemented_not_fully_verified | Implemented but not fully verified | Dates confirmed only via secondary sources. |
| AI Act Art. 9-15, 17 | High-risk requirements | partial | partial | Partially implemented | Oversight, logging, risk classification; QMS, technical file, conformity are organisational. |
| AI Act Art. 14 | Human oversight | implemented | implemented_not_fully_verified | Implemented but not fully verified | Human-only approvals, emergency stop, contestation tested incl. concurrency. |
| AI Act Art. 26-27 | Deployer duties, FRIA | organisational | dependent_on_organisational_action | Dependent on organisational or legal action | FRIA is an organisational document. |
| AI Act Art. 43, 49 | Conformity assessment and registration | organisational | dependent_on_organisational_action | Dependent on organisational or legal action | Not software. |
| AI Act Art. 50 | Transparency, marking, deepfake disclosure | partial | missing | Partially implemented | Module tested; applied to Mapping results of configured agents (default: language). String results unmarked; metadata is strippable; not C2PA. |
| AI Act Art. 53-55 | GPAI obligations | partial | partial | Partially implemented | Completeness check + systemic-risk presumption; documents themselves are organisational. |
| AI Act Art. 72-73 | Post-market monitoring, serious incidents | partial | missing | Partially implemented | Bridge verified against the real SharedMemory; works only for incidents Observability/Handler publish on *:latest_incident. |
| GDPR Art. 5(1)(b), 25; AI Act Art. 10 | Governed learning, lineage and erasure impact | partial | partial | Partially implemented | Registry + gate tested; LearningAgent does not itself register datasets/runs. |

Reporter summary: `software_release_allowed = False`; control status counts {'missing': 15, 'not_applicable': 8, 'dependent_on_organisational_action': 3}. The reporter withholds a release verdict until independent evidence exists, which is the correct outcome today.

## 8. Files
### Created (79)
- `.github/workflows/compliance.yml`
- `docs/compliance/README.md`
- `docs/compliance/benchmarks.md`
- `docs/compliance/data_subject_rights.md`
- `docs/compliance/deployment_compliance.md`
- `docs/compliance/eu_ai_act.md`
- `docs/compliance/gdpr.md`
- `docs/compliance/incident_response.md`
- `docs/compliance/learning_governance.md`
- `docs/compliance/processing_activities.md`
- `docs/compliance/regulatory_traceability.md`
- `docs/compliance/risk_register.md`
- `docs/compliance/technical_documentation.md`
- `docs/compliance/uavg.md`
- `src/agents/collaborative/utils/secure_persistence.py`
- `src/agents/governance/__init__.py`
- `src/agents/governance/agent_gate.py`
- `src/agents/governance/agent_manifest.py`
- `src/agents/governance/compliance_engine.py`
- `src/agents/governance/configs/traceability.json`
- `src/agents/governance/core.py`
- `src/agents/governance/delegation_guard.py`
- `src/agents/governance/deletion_adapters.py`
- `src/agents/governance/human_oversight.py`
- `src/agents/governance/incident_bridge.py`
- `src/agents/governance/incident_registry.py`
- `src/agents/governance/lawful_basis.py`
- `src/agents/governance/learning_governance.py`
- `src/agents/governance/modules/__init__.py`
- `src/agents/governance/policy_engine.py`
- `src/agents/governance/policy_registry.py`
- `src/agents/governance/privacy_dashboard.py`
- `src/agents/governance/privacy_ledger.py`
- `src/agents/governance/processing_context.py`
- `src/agents/governance/processing_registry.py`
- `src/agents/governance/registrations.py`
- `src/agents/governance/regulatory_context.py`
- `src/agents/governance/regulatory_evidence.py`
- `src/agents/governance/regulatory_reporting.py`
- `src/agents/governance/rights_providers.py`
- `src/agents/governance/risk_classification.py`
- `src/agents/governance/subject_rights.py`
- `src/agents/governance/transparency.py`
- `src/agents/governance/utils/__init__.py`
- `src/agents/governance/utils/diagnostics.py`
- `src/agents/governance/utils/identity.py`
- `src/agents/governance/utils/store.py`
- `src/agents/governance_agent.py`
- `tests/compliance/__init__.py`
- `tests/compliance/_fixtures.py`
- `tests/compliance/benchmarks/bench_governance.py`
- `tests/compliance/test_agent_gate.py`
- `tests/compliance/test_agent_integration.py`
- `tests/compliance/test_agent_isolation.py`
- `tests/compliance/test_concurrency_adversarial.py`
- `tests/compliance/test_consent.py`
- `tests/compliance/test_deletion.py`
- `tests/compliance/test_deletion_adapters.py`
- `tests/compliance/test_diagnostics_sanitization.py`
- `tests/compliance/test_external_transfers.py`
- `tests/compliance/test_governance_runtime.py`
- `tests/compliance/test_human_oversight.py`
- `tests/compliance/test_incident_bridge.py`
- `tests/compliance/test_incident_management.py`
- `tests/compliance/test_lawful_basis.py`
- `tests/compliance/test_learning_governance.py`
- `tests/compliance/test_model_governance.py`
- `tests/compliance/test_privacy_agent_fail_closed.py`
- `tests/compliance/test_processing_authority.py`
- `tests/compliance/test_prohibited_practices.py`
- `tests/compliance/test_registrations.py`
- `tests/compliance/test_regulatory_release_gates.py`
- `tests/compliance/test_retention.py`
- `tests/compliance/test_rights_workflows.py`
- `tests/compliance/test_risk_classification.py`
- `tests/compliance/test_secure_persistence.py`
- `tests/compliance/test_subject_rights.py`
- `tests/compliance/test_tenant_isolation.py`
- `tests/compliance/test_transparency.py`

### Modified by this work (12)
- `src/agents/agent_factory.py`
- `src/agents/base/configs/agents_config.yaml`
- `src/agents/base_agent.py`
- `src/agents/collaborative/shared_memory.py`
- `src/agents/factory/out_of_process_agent.py`
- `src/agents/orchestrator_agent.py`
- `src/agents/privacy/configs/privacy_config.yaml`
- `src/agents/privacy/data_retention.py`
- `src/agents/privacy/privacy_memory.py`
- `src/agents/privacy_agent.py`
- `src/agents/safety/compliance_checker.py`
- `src/agents/safety/secure_memory.py`

Also changed this round: `privacy/privacy_memory.py` (live-instance registry, `export_subject`), `factory/out_of_process_agent.py` (gate), `agent_factory.py` (governance bootstrap), `base_agent.py` (`execute` -> gate -> `_execute_governed` -> marking), `base/configs/agents_config.yaml` (gate, registration, `audit` registry mode).

## 9. Configuration and migration
- Provision `SLAI_GOVERNANCE_MASTER_KEY` (or `_FILE`, >= 32 bytes) and `SLAI_SHARED_MEMORY_KEY`. Keys are never generated or stored by the code. A key shorter than 32 bytes is treated as 'no key' and the core reports `degraded_enforcing`.
- Set `governance_agent.privacy_memory_tenant_id`, `file_store_roots`, `portable_stores`, `shared_memory_persist_path` as applicable (see `technical_documentation.md`).
- Data written by producers **must** be tagged with `subject_tag(tenant, subject)` to be erasable/exportable; untagged data cannot be attributed.
- SharedMemory files written before this change are unsealed raw pickles and are refused by default. Migrate once with `load_from_file(path, migrate_legacy=True)` (restricted decoder, no arbitrary objects) in a trusted environment, then re-save in the sealed format.
- State databases are created under `data/governance/` (relative to the working directory).

## 10. Tests - real commands and results
```
python3 -m unittest discover -s tests/compliance -t .
Ran 294 tests in ~31s   OK   (0 failures, 0 errors, 0 skipped)   - Linux sandbox, Python 3.10.12, torch shimmed
python -m tests.compliance.benchmarks.bench_governance     (benchmarks; not a test)
```
| Test file | Tests |
|---|---|
| test_agent_gate.py | 23 |
| test_agent_integration.py | 9 |
| test_agent_isolation.py | 7 |
| test_concurrency_adversarial.py | 12 |
| test_consent.py | 11 |
| test_deletion.py | 10 |
| test_deletion_adapters.py | 9 |
| test_diagnostics_sanitization.py | 5 |
| test_external_transfers.py | 8 |
| test_governance_runtime.py | 14 |
| test_human_oversight.py | 15 |
| test_incident_bridge.py | 8 |
| test_incident_management.py | 9 |
| test_lawful_basis.py | 10 |
| test_learning_governance.py | 7 |
| test_model_governance.py | 8 |
| test_privacy_agent_fail_closed.py | 9 |
| test_processing_authority.py | 21 |
| test_prohibited_practices.py | 7 |
| test_registrations.py | 10 |
| test_regulatory_release_gates.py | 15 |
| test_retention.py | 9 |
| test_rights_workflows.py | 12 |
| test_risk_classification.py | 11 |
| test_secure_persistence.py | 10 |
| test_subject_rights.py | 6 |
| test_tenant_isolation.py | 8 |
| test_transparency.py | 11 |

Defects found by tests and fixed during this work: legal hold silently released by an erasure request; incident bridge never matched the real `SharedMemory` event shape (events carry a repr'd key and no value, delivered on threads) although it passed against a fake; a 40-character alphanumeric master key was silently rejected as 'no key' (valid base64 decoding to 30 bytes); `PrivacyMemory` is instantiated once per privacy component, so a single-instance adapter would have missed most of the data (adapters now cover all live instances); `execute` was not registered as a capability for any agent (would have broken delegation under enforcement); several docs said 'verified' where only local testing exists (corrected).

Not tested: the full runtime on Windows; real `torch` environment; the CI workflow; multi-process SharedMemory managers; load beyond the benchmark; any UI.

## 11. Benchmarks
See `benchmarks.md` (sandbox, not production hardware; baseline = the same call without governance). Policy evaluation about 0.08 ms; a durable decision record about 0.7 ms (fsync); the gate adds that cost to every `execute()` in observe and enforce modes.

## 12. Evidence
The evidence store holds **no** independently produced, signed evidence. The only evidence is this local test run, which is the author's own and therefore classed as 'not fully verified'. To change that, CI (`.github/workflows/compliance.yml`) or a verification agent must record signed results.

## 13. Limitations
- Gate default is `observe`: no execution is blocked unless the deployment switches to `enforce` or callers pass `governance_context`. Enforcing now would block essentially every execution, because no caller supplies a context and the default context is UNCLASSIFIED.
- Only `BaseAgent.execute()` (and the out-of-process proxy's `execute`) is gated.
- `enforce_capability_registry` defaults to `audit`; the audit list has not been reviewed against real orchestrator traffic.
- Erasure cannot reach external recipients, offline backups (tombstone check on restore only), model weights (impact tracking only), or stores not registered. Physical erasure on SSD/journalling filesystems is not guaranteed.
- Rectification has no store-level apply callbacks registered. Portability is off for every store until declared.
- Art. 50 marking: Mapping results only; string results unmarked; metadata marking is removable; not C2PA or a robust watermark.
- LearningAgent does not itself register datasets, runs or models; unlearning itself is not implemented (only tracked).
- Incident bridge depends on Observability/Handler publishing on `*:latest_incident` with recognisable content; classification is marker-based.
- Log sanitising is pattern-based and best effort.
- Not built: external provider/transfer governance beyond the policy rule, Secure Folder authorisation levels, Phase 5 integrations (Writer, SPADRA, SReader, SPlanner, SciCal, Documaster, CADCon, GangUP, LANTRA, plug-ins), key-generation helper, README section.
- Finding 4.3 cannot be confirmed (see section 3). No separate compliance auditor module (`modules/compliance_auditor.py`) exists in the repo, so none was changed.

## 14. Organisational and legal obligations (not software)
DPO appointment; DPIA and prior consultation; privacy notices; consent UX; processor/joint-controller contracts; SCCs and transfer impact assessments; FRIA; AI Act conformity assessment, CE marking, EU database registration where applicable; QMS and technical documentation for high-risk systems; staffing of oversight and incident roles; submission of any regulator notification (the software prepares drafts only); counsel review of all dates; penetration test and independent audit.

## 15. Deployment considerations
Provision keys and rotate them; set the configuration in section 9; run the full suite on the Windows venv and in CI first; deploy in `observe`, review `would_block` counts and `delegation_audit.would_deny`, make callers supply `governance_context`, then move chosen agents to `enforce` via `enforce_agents` before flipping the global mode; verify the AI Act dates (Art. 50(2) grace and new prohibitions: 2026-12-02); decide which stores are portable; schedule backup-restore tombstone checks.

## 16. Readiness assessment
The governance layer is built, wired into `BaseAgent.execute()` and the factory, registered against the reachable stores, and passes its own 294 tests in one environment. It is **not** ready to be described as enforcing GDPR/UAVG/AI Act compliance: enforcement is off by default, verification is not independent, key regulatory dates rest on secondary sources, several stores and workflows are unreachable or unregistered, and the organisational obligations in section 14 are outstanding. No certification or full-compliance claim is made or implied.
