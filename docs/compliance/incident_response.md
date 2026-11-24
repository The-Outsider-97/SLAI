# Incident response

`IncidentRegistry` enforces a lifecycle state machine without skipping states, tracks GDPR Art. 33 (72h from awareness) and Art. 34,
and EU AI Act serious-incident deadlines, prepares **draft** notifications only, and requires an authorised human role plus an external
reference to record a submission. Corrective actions need a verifier different from the owner.

`IncidentIntake` (`governance/incident_bridge.py`) turns Observability/Handler incident records into registry incidents: documented marker-based
classification (or an explicit `governance_kind`), sanitised descriptions, honest awareness time (a future timestamp falls back to now),
persistent de-duplication, and a SharedMemory subscription (`attach`) that contains its own failures so publishers are never broken.
It never notifies an authority. Limit: Observability/Handler do not call it on their own; deployment must call `core.incident_intake.attach(shared_memory)`.
