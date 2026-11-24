# SLAI compliance documentation

Technical documentation for the SLAI regulatory-governance layer (`src/agents/governance_agent.py`, `src/agents/governance/`).
These documents describe what the **software** enforces and evidences. They are not legal advice and **do not
constitute certification** of GDPR, UAVG or EU AI Act compliance. Organisational duties (DPO, DPIA, FRIA, notices,
contracts, conformity assessment, registrations) remain with the deploying organisation - see `deployment_compliance.md`.

| File | Content |
|---|---|
| gdpr.md | GDPR controls and where they are enforced |
| uavg.md | Dutch UAVG specifics |
| eu_ai_act.md | EU AI Act controls, dates and open verification items |
| regulatory_traceability.md | Article -> component -> test matrix with honest status |
| processing_activities.md | Art. 30 register and lawful-basis evidence model |
| data_subject_rights.md | Rights workflows: implemented vs open |
| incident_response.md | Unified incident / breach lifecycle |
| learning_governance.md | Governed learning registry, lineage, erasure impact (implemented; call sites in LearningAgent pending) |
| final_report.md | Final report: verified state, status per requirement, evidence, limitations |
| risk_register.md | Known risks and open items |
| technical_documentation.md | Architecture, state, keys, operations |
| deployment_compliance.md | What must be done outside the code |

Verification: `python -m unittest discover -s tests/compliance -t .` (CI: `.github/workflows/compliance.yml`).
