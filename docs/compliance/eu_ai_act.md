# EU AI Act controls

Ruleset version: `RULESET_VERSION` in `governance/risk_classification.py`.

**Date verification status:** EUR-Lex confirms Regulation (EU) 2026/1744 (OJ 24.7.2026, in force 27.7.2026). The application dates below come from several law-firm summaries and were **not** read from the consolidated text: Art. 50 general application 2026-08-02, Art. 50(2) grace for systems placed on the market before that date until 2026-12-02, new Art. 5 prohibitions (non-consensual intimate material, CSAM) from 2026-12-02, Annex III high-risk 2027-12-02, Annex I 2028-08-02, GPAI 2025-08-02 (legacy models 2027-08-02). They are configuration (`RISK_DATES`) and counsel must confirm them. The new prohibition is modelled as the tag `csam_ncii_generation` and is applied immediately (conservative), not date-gated.

| Requirement | Component | Status |
|---|---|---|
| Art. 5 prohibited practices (9 tags, exception evidence required) | `risk_classification.py`, GOV-AI5-03 | Implemented; locally tested, not independently verified |
| Risk classification (Annex III, Art. 6(3) derogation, Annex I, Art. 50, GPAI) | `RiskClassifier` | Implemented; locally tested, not independently verified; open questions surfaced, never silently resolved |
| Human oversight (Art. 14): human-only, authenticated, scoped, expiring, revocable approvals; emergency stop; contestation | `human_oversight.py` | Implemented; locally tested, not independently verified |
| Art. 22 GDPR assessed separately from AI Act oversight | GOV-ADM-10 | Implemented; locally tested, not independently verified |
| Incidents (Art. 73 serious incidents, 15/2/10 days) | `incident_registry.py` | Implemented; submission is a human act |
| Art. 50 runtime disclosure/marking | `governance/transparency.py` (signed manifests, PNG embedding, disclosures, deepfake/public-interest labels, fail-closed `release`) | Implemented; locally tested in isolation; **not yet called by generating agents**; metadata marking is removable, not a robust watermark, not C2PA |
| High-risk technical documentation, QMS, conformity assessment, registration | - | Organisational / not implemented |
| GPAI documentation / systemic-risk check (incl. LANTRA) | `learning_governance.py::gpai_assessment` | Partial: checks completeness of referenced documents; producing them is organisational |
