# Risk register (open items)

| ID | Risk | Status |
|---|---|---|
| R1 | Regulatory dates unverified against EUR-Lex | Open - counsel |
| R2 | Adequacy list not verified against current Commission list | Open |
| R3 | Deletion adapters exist for SharedMemory (+ sealed state file), SecureMemory (+ checkpoints) and PrivacyMemory (`governance/deletion_adapters.py`). Still without adapters: external providers/recipients, model weights, other persisted files, offline backups (tombstone check on restore only). Items are only erasable by subject if producers tag them with `subject_tag()` | Partially open |
| R4 | All 31 factory agents get governed capabilities (static discovery) and share the delegation guard, but none calls the governance/privacy gate itself, and `enforce_capability_registry` is off by default | Open |
| R5 | Governance master key must be provisioned; without it the core reports `degraded_enforcing` and signed artefacts are unavailable | Deployment |
| R6 | Full agent runtime (torch) tests not executed in this environment | Open |
| R7 | Learning registry, Art. 50 marking, rights workflows and dashboard exist but are not yet called by LearningAgent / generating agents / any store (no RightsProvider registered) | Open |
| R8 | Legal-hold bypass in `DataRetention.process_forget_request` (status overwrite) was found by `test_retention.py` and fixed | Closed |
