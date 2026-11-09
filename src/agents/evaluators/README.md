# Evaluators Subsystem

## Overview
The `src/agents/evaluators/` package provides autonomous assessment, risk estimation, statistical analysis, and validation/report tooling for SLAI behaviors and task outcomes.

It includes:

- **Task and autonomy evaluators** (`autonomous_evaluator.py`, `performance_evaluator.py`)
- **Efficiency and utilization evaluators** (`efficiency_evaluator.py`, `resource_utilization_evaluator.py`)
- **Safety/risk and behavior checks** (`safety_evaluator.py`, `adaptive_risk.py`, `behavioral_validator.py`)
- **Statistical analysis** (`statistical_evaluator.py`)
- **Persistence + infrastructure** (`evaluators_memory.py`, `base_infra.py`)
- **Validation/report/certification utilities** (`utils/`)

---

## Directory Structure

The main implementation groups in the current repository are:

- Evaluator modules: autonomous, behavioral, efficiency, performance,
  performance-budget, resource utilization, safety, statistical, adaptive risk.
- modules/: evaluation_transformer.py, evaluators_calculations.py,
  static_analyzer.py, validation_protocol.py, test_generator.py.
- configs/: evaluator_config.yaml.
- data/: issue_db.py.
- documentation/: report.py, documentation.py, certification_framework.py,
  test_generator.md.
- utils/: config_loader.py, evaluation_errors.py, evaluation_helpers.py,
  plugin_loader.py.
- templates/: structured audit, certification, and recommendation schemas.

### Experimental TestGenerator

The Evaluation Agent can optionally generate **preview-only** candidate Python
tests using a separately trainable, versioned byte-level model. Generation
is disabled by default; no generated test is executed or integrated into code.
See documentation/test_generator.md for dataset contracts, PowerShell commands,
review gates, and the outstanding production-readiness blockers.

---

## Evaluation Architecture

```mermaid
flowchart LR
    A[Task Runs / Agent Outputs] --> B[Domain Evaluators]
    B --> B1[Autonomous]
    B --> B2[Efficiency]
    B --> B3[Performance]
    B --> B4[Safety + Behavioral]
    B --> B5[Statistical]
    B --> B6[Adaptive Risk]

    B1 --> C[Validation Protocol]
    B2 --> C
    B3 --> C
    B4 --> C
    B5 --> C
    B6 --> C

    C --> D[Report + Visualization]
    D --> E[Certification Framework]
    D --> F[Documentation / Audit Trail]
    D --> G[EvaluatorsMemory]
```

---

## Core Modules

### Evaluators
- `autonomous_evaluator.py`: evaluates robotics/planning-style tasks (completion, path quality, collisions, energy, success rates).
- `efficiency_evaluator.py`: analyzes operational efficiency metrics and optimization opportunities.
- `performance_evaluator.py`: tracks broader runtime/quality performance signals.
- `resource_utilization_evaluator.py`: evaluates compute/resource usage and allocation patterns.
- `safety_evaluator.py`: applies safety-centric constraints and checks.
- `adaptive_risk.py`: estimates risk dynamically from observed conditions.
- `statistical_evaluator.py`: computes aggregate/distributional metrics and analytical summaries.
- `behavioral_validator.py`: validates behavioral policy adherence and consistency.

### Utility Layer (`utils/`)
- `validation_protocol.py`: shared protocol/hooks for validation staging.
- `report.py`: report assembly and visualization support.
- `documentation.py`: audit-ready documentation components (including chain/block style artifacts).
- `certification_framework.py`: certification-oriented policies and decisions.
- `evaluation_transformer.py`: data transformation for evaluation pipelines.
- `evaluation_errors.py`: typed evaluator exceptions.
- `evaluators_calculations.py`: reusable metric computations.

---

## End-to-End Flow

```mermaid
sequenceDiagram
    participant R as Runtime System
    participant E as Evaluator Modules
    participant V as Validation Protocol
    participant P as Report/Docs
    participant M as EvaluatorsMemory

    R->>E: Submit tasks, traces, metrics
    E->>E: Compute evaluator-specific scores
    E->>V: Run validation checks
    V-->>E: Validation status + findings
    E->>P: Generate reports/audit artifacts
    E->>M: Persist metrics and outcomes
    P-->>R: Deliver summarized evaluation package
```

---

## Notes

- Global evaluator behavior is configured via `configs/evaluator_config.yaml` and `utils/config_loader.py`.
- Templates under `templates/` support schema-driven audit and certification workflows.
- The subsystem is designed to support both online evaluation and post-run audit/report generation.
