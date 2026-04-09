# Data Quality Agent (Stability Multiplier)

The `src/agents/quality/` package is the dedicated quality gate for SLAI data pipelines. It sits between raw inputs and downstream consumers (knowledge ingestion, training loops, inference context assembly, and memory updates), ensuring only fit-for-use data proceeds.

At runtime, this subsystem combines four execution domains:

- **Structural quality** (schema, types, completeness, constraints)
- **Statistical quality** (drift, missingness, duplicates, outliers)
- **Semantic quality** (leakage, consistency, provenance trust)
- **Workflow control** (quarantine, routing, remediation plans)

State and historical decisions are persisted through a specialized memory module (`QualityMemory`) that supports baseline management, conflict resolution, and quality-history retrieval.

---

## Mission

The Data Quality Agent is SLAI's **data integrity gatekeeper**. Its objective is to continuously measure and enforce:

- **Integrity**: records adhere to expected structure and constraints.
- **Consistency**: cross-field and cross-batch behavior remains coherent.
- **Reliability**: provenance and source trust are explicit and auditable.
- **Operational safety**: suspect records are quarantined before they can contaminate downstream systems.

This is a prevention-first subsystem: it blocks silent data failures that might otherwise surface as misleading model, planner, or retrieval issues.

---

## Why SLAI Needs This Layer

SLAI already has strong planning, reasoning, and learning capabilities. However, many production incidents begin earlier in the pipeline and are fundamentally data-origin problems, such as:

- malformed or drifted schemas,
- distribution shift against stale baselines,
- label leakage and semantic contradictions,
- duplicated noisy records from volatile sources,
- low-trust provenance entering training or inference context.

Without a dedicated quality owner, these defects cascade and appear as downstream agent regressions. The Data Quality Agent isolates and contains those failures upstream.

---

## Non-Overlap with Existing Agents

- **KnowledgeAgent**: retrieves and organizes knowledge.
- **PerceptionAgent**: encodes multimodal inputs.
- **LearningAgent**: optimizes policy/model behavior from data.
- **Data Quality Agent**: validates input fitness and enforces quarantine/remediation policy **before** downstream usage.

This boundary keeps responsibilities clear: quality validation is centralized and policy-driven, not duplicated across consumer agents.

---

## Directory Layout

```text
src/agents/quality/
├── __init__.py
├── structural_quality.py
├── statistical_quality.py
├── semantic_quality.py
├── workflow_control.py
├── quality_memory.py
├── README.md
├── configs/
│   └── quality_config.yaml
└── utils/
    ├── __init__.py
    ├── config_loader.py
    └── quality_error.py
```

---

## Core Components

### 1) Structural Quality (`structural_quality.py`)

Primary concern: **record well-formedness**.

Capabilities include:

- schema and type validation,
- required field completeness checks,
- range/domain/regex constraints,
- unknown-field policy handling,
- optional type coercion with thresholded guardrails,
- schema version registration into quality memory.

Output includes findings, flags, remediation suggestions, score, verdict, and optional quality snapshot persistence.

---

### 2) Statistical Quality (`statistical_quality.py`)

Primary concern: **batch-level statistical fitness**.

Capabilities include:

- distribution shift detection against baselines,
- missingness profiling,
- duplicate concentration scoring,
- numeric outlier burden scoring,
- baseline bootstrap/registration when configured.

Statistical checks generate per-domain findings and a weighted batch score, then derive a verdict with associated flags, shift metrics, and remediation actions.

---

### 3) Semantic Quality (`semantic_quality.py`)

Primary concern: **semantic trustworthiness and label safety**.

Capabilities include:

- label leakage detection,
- cross-field consistency rules,
- provenance and source trust assessment,
- source reliability recording,
- conflict reconciliation via memory-backed consensus logic.

Semantic quality protects downstream training/evaluation integrity by reducing contamination risks (e.g., leaked targets, contradictory labels, low-trust sources).

---

### 4) Workflow Control (`workflow_control.py`)

Primary concern: **operational execution of quality decisions**.

Capabilities include:

- quarantine entry creation (record and batch scope),
- routing/escalation to Handler and Safety bridges,
- remediation plan synthesis,
- downstream action decisioning from findings + policy,
- optional event publication into shared memory.

`WorkflowControl` is intentionally bridge-driven; it should receive runtime dependencies rather than importing other agents directly.

---

### 5) Quality Memory (`quality_memory.py`)

Primary concern: **durable quality state and historical intelligence**.

`QualityMemory` persists and retrieves:

- canonical quality snapshots (`source`, `batch`, `window`),
- drift baselines and drift observations,
- schema versions and field contracts,
- threshold decisions and governance rationale,
- remediation outcomes and measured uplift,
- conflict resolution history,
- source reliability events.

It also exposes retrieval-style APIs for operational queries:

- `latest_quality_state(source_id)`
- `historical_drift(source_id, metric, window)`
- `remediation_effectiveness(rule_id)`

---

## Agent-Level Orchestration (`src/agents/quality_agent.py`)

`QualityAgent` is the parent orchestrator that wires and executes subsystem checks in policy-aware order. It:

1. normalizes incoming task/batch payloads,
2. runs structural/statistical/semantic subsystems,
3. aggregates subsystem scores/verdicts,
4. delegates operational routing/quarantine to `WorkflowControl`,
5. publishes decision artifacts to shared memory,
6. records result history by batch and source.

This orchestrator-heavy model keeps component logic modular while preserving a stable quality decision contract for sibling agents.

---

## Interfaces and Dependencies

### Inputs

- Reader outputs (documents/files)
- Browser outputs (web content)
- Knowledge ingestion streams
- Training/replay samples

### Outputs

- Quality verdict: `pass`, `warn`, or `block`
- Record-level flags with confidence metadata
- Quarantine queue entries
- Remediation suggestions
- Shared-memory quality keys and event payloads

### Integrations

- **PlanningAgent**: consumes quality-aware context for safer planning.
- **LearningAgent**: can skip/down-weight low-confidence samples.
- **EvaluationAgent**: tracks quality KPIs and trend stability.
- **HandlerAgent**: receives severe event escalations/fallback requests.
- **Safety systems**: receive critical privacy/leakage conflict escalations.

---

## Decision Policy (Default Shape)

Reference policy:

- **pass**: `score >= 0.90` and no critical flags.
- **warn**: `0.75 <= score < 0.90` or moderate findings.
- **block**: `score < 0.75` or any critical leakage/privacy/safety conflict.

These thresholds are configurable in `configs/quality_config.yaml` and can vary per subsystem and workflow phase.

---

## Shared Memory Contract

Recommended integration keys:

- `data_quality.batch_score`
- `data_quality.flags`
- `data_quality.quarantine_count`
- `data_quality.shift_metrics`
- `data_quality.remediation_actions`
- `data_quality.source_reliability`

These keys provide a stable, cross-agent surface for downstream quality awareness.

---

## KPIs

Track subsystem performance using:

- bad-record escape rate,
- quarantine precision/recall,
- drift detection latency,
- post-quality-gate incident reduction,
- training stability variance.

Recommended practice: monitor KPI trends per source and dataset, not only global averages.

---

## Failure Modes and Mitigations

- **Overblocking**
  - Mitigation: adaptive thresholds, weighted consensus, and controlled manual override channel.

- **Under-detection**
  - Mitigation: combine deterministic rule checks with statistical detectors and provenance scoring.

- **Source volatility**
  - Mitigation: source-specific reliability scoring, baseline windowing, and source cooldown logic.

- **Cross-check disagreement**
  - Mitigation: memory-backed conflict reconciliation and auditable rationale tracking.

---

## Configuration Notes

Main config file: `src/agents/quality/configs/quality_config.yaml`.

Key sections:

- `structural_quality`
- `statistical_quality`
- `semantic_quality`
- `workflow_control`
- `quality_memory`
- `quality_agent` (orchestrator-level behavior)

When tuning policy:

1. adjust warning/blocking thresholds conservatively,
2. preserve required key names expected by runtime loaders,
3. calibrate per-source before applying global hard thresholds,
4. track KPI impact after each threshold revision.

---

## Operational Guidance

When introducing new quality checks:

1. Add the check to the relevant subsystem (structural/statistical/semantic).
2. Attach clear findings metadata (severity, confidence, flags, metrics, remediation).
3. Register memory persistence behavior if historical tracking is needed.
4. Ensure workflow behavior for warn/block outcomes is explicitly defined.
5. Update this README and config defaults to keep operator behavior predictable.

This documentation should remain the canonical subsystem overview for maintainers integrating data quality with planning, learning, and safety workflows.
