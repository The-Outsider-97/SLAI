# GDPR controls

| Control | Enforcement | Status |
|---|---|---|
| Lawful basis, all six Art. 6 bases, evidence validated per basis | `governance/lawful_basis.py`, `processing_registry.py` (Art. 30 register is the anchor) | Implemented; locally tested, not independently verified (unit tests) |
| Art. 9 / 10 conditions | `lawful_basis.py`, rule GOV-SPC-06 | Implemented; locally tested, not independently verified |
| Consent (Art. 7/8): per purpose, withdrawal, explicit, child | `privacy_ledger.py` + rule GOV-CON-07 | Implemented; locally tested, not independently verified; consent *collection UI* is application work |
| Fail-closed enforcement | `policy_engine.py` (rule errors => BLOCK GOV-ERR-99) | Implemented; locally tested, not independently verified |
| Privacy Agent cannot be bypassed by `enabled: false` | `privacy_agent.py::_disabled_privacy_decision` | Implemented; tested against real source, full agent not run (needs torch env) |
| Erasure (Art. 17): no mandatory reason, identity verification, legal holds, per-store verified deletion, recipient notices, backup tombstones | `privacy_ledger.py`, `privacy/data_retention.py` | Implemented; locally tested, not independently verified for registered adapters. Adapters for SharedMemory, SecureMemory and PrivacyMemory in `governance/deletion_adapters.py`; data is attributable only if written with `subject_tag(tenant, subject)`; external recipients and backups are not covered |
| Restriction / objection (Art. 18/21) enforced at processing time | `privacy_ledger.py` + GOV-RTS-08 | Implemented; locally tested, not independently verified |
| Tenant isolation | `safety/secure_memory.py` | Implemented; locally tested, not independently verified |
| Integrity/confidentiality of persisted shared memory | `collaborative/utils/secure_persistence.py` | Implemented; locally tested, not independently verified (AES-256-GCM only if `cryptography` installed, else integrity-only) |
| No PII in diagnostics | `governance/utils/diagnostics.py` (central log filter) | Implemented; best-effort pattern detection |
| Transfers (Chapter V) | `policy_engine.py::TransferPolicy` | Implemented; adequacy list must be verified against the current Commission list |
| Breach lifecycle (Art. 33/34) | `incident_registry.py` | Implemented; **never auto-submits** to authorities |

Access/rectification/portability workflows exist (`data_subject_rights.md`) but need stores to be registered as providers (done at start-up for SharedMemory, SecureMemory, PrivacyMemory when `privacy_memory_tenant_id` is set, and configured file stores). Not implemented: DPIA tooling, privacy notices, consent-collection UI.
