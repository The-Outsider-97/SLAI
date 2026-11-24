# Data subject rights

| Right | Status |
|---|---|
| Access (Art. 15), portability (Art. 20), rectification (Art. 16) | Workflow implemented in `governance/subject_rights.py`: bound identity verification, one-month deadline with Art. 12(3) extension rules, tenant/subject isolation, all-or-nothing release, bundle not retained. **Stores must register a `RightsProvider`; none are registered yet.** |
| Erasure (Art. 17) | `privacy_ledger.py` + `deletion_adapters.py` (SharedMemory, SecureMemory, PrivacyMemory, files, learning pipeline); no reason required |
| Restriction / objection (Art. 18/21) | Implemented and enforced at processing time |
| Consent withdrawal (Art. 7(3)) | Implemented |
| Art. 22 human intervention | `human_oversight.py` contestation |
| Dashboard API | `governance/privacy_dashboard.py`: transport-agnostic methods plus an HTTP-style `handle()` router; self-service for the verified subject, staff only inside their tenant |

The dashboard has no UI and no web server: applications (SLAI Writer, SReader, ...) embed or expose it.
