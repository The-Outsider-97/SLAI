# What the deploying organisation must do

Not provided by software: appoint/assess a DPO; DPIA (and prior consultation); privacy notices and consent UX; processor/joint-controller
contracts; transfer impact assessments and SCCs; FRIA for high-risk deployers; AI Act conformity assessment, CE marking and EU database registration
for high-risk systems; staffing of oversight and incident roles with qualified humans; regulator notification; verifying the regulatory
dates/adequacy list; provisioning and rotating keys; backups policy (tombstone check on restore); penetration testing and independent audit.

Configuration the deployment must set (`governance_agent:` in `agents_config.yaml`): master key via environment; `privacy_memory_tenant_id`; `file_store_roots`; `portable_stores` (Art. 20 eligibility is a legal decision); `shared_memory_persist_path`; `agent_gate.mode` (move from `observe` to `enforce` only after callers supply `governance_context` and the decision log shows no unexpected `would_block`); `enforce_capability_registry` (`audit` -> `true` once `delegation_audit.would_deny` is empty or understood). Re-verify the AI Act dates against the consolidated text; the Art. 50(2) grace for systems already on the market ends 2026-12-02.
