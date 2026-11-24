# Regulatory traceability

The authoritative matrix is machine-readable: `src/agents/governance/configs/traceability.json`. Statuses shown in reports are
**computed from evidence** by `RegulatoryReporter` - a row is only VERIFIED when independent, signed evidence exists; the
`declared` status in the JSON is the author's claim and is never copied into reports.

Generate: `GovernanceAgent.compliance_report()` (JSON) / `RegulatoryReporter.to_markdown()`.
Status vocabulary: Implemented and verified / Implemented but not fully verified / Partially implemented / Missing or failed /
Not applicable / Dependent on deployment / Dependent on organisational or legal action.
