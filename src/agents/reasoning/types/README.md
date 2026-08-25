# Reasoning Types Subsystem

## Overview
The `src/agents/reasoning/types/` package defines SLAI’s **typed reasoning modes** and the base contract they share.

It exists to make reasoning strategies:
- modular (each reasoning family is isolated),
- composable (single mode or combined multi-mode chains),
- testable (consistent input/output expectations), and
- extensible (new reasoning modes can be added without rewriting the core engine).

The reasoning types layer is consumed by the broader reasoning subsystem (`rule_engine.py`, `validation.py`, and related orchestration layers), and should be treated as the canonical home for reasoning-style semantics.

---

## Package Contents

```text
src/agents/reasoning/types/
├── README.md
├── __init__.py
├── base_reasoning.py
├── reasoning_abduction.py
├── reasoning_analogical.py
├── reasoning_cause_effect.py
├── reasoning_decompositional.py
├── reasoning_deductive.py
└── reasoning_inductive.py
```

### Files and Responsibilities

- `base_reasoning.py`  
  Defines the shared base abstraction/interface for all reasoning types.

- `reasoning_abduction.py`  
  Implements abductive reasoning (best explanation from observations).

- `reasoning_deductive.py`  
  Implements deductive reasoning (conclusion from premises/rules).

- `reasoning_inductive.py`  
  Implements inductive reasoning (pattern/regularity extraction from examples).

- `reasoning_analogical.py`  
  Implements analogical transfer (mapping source structures to a target problem).

- `reasoning_decompositional.py`  
  Implements decomposition of complex systems/problems into subcomponents.

- `reasoning_cause_effect.py`  
  Implements causal reasoning around causes, effects, and influence pathways.

---

## Supported Reasoning Families

### 1) Abductive Reasoning
**Goal:** infer the most plausible explanation for observed evidence.

Typical use:
- explaining anomalies,
- generating candidate causes,
- triage/investigative workflows.

### 2) Deductive Reasoning
**Goal:** derive conclusions that logically follow from known premises/rules.

Typical use:
- policy/rule compliance,
- deterministic entailment,
- formal consistency checks.

### 3) Inductive Reasoning
**Goal:** generalize from examples or observations to broader hypotheses.

Typical use:
- discovering regularities,
- deriving probable rules from repeated evidence,
- trend-driven hypothesis generation.

### 4) Analogical Reasoning
**Goal:** map structure from a known domain to a new domain.

Typical use:
- transfer learning in symbolic form,
- solution bootstrapping for novel tasks,
- cross-domain explanation generation.

### 5) Decompositional Reasoning
**Goal:** break a large problem into tractable subproblems.

Typical use:
- planning,
- systems diagnosis,
- hierarchical analysis.

### 6) Cause-and-Effect Reasoning
**Goal:** model and evaluate causal relationships.

Typical use:
- intervention analysis,
- dependency tracing,
- outcome forecasting under causal assumptions.

---

## Common Interface Contract

All reasoning types are expected to conform to the base contract defined in `base_reasoning.py`.

### Conceptual Contract
- Accept task input + optional context.
- Produce structured output (not opaque free-form text only).
- Preserve enough metadata for downstream validation/audit.
- Fail with explicit, typed errors whenever possible.

### Recommended Output Shape
While each mode can enrich output differently, keep these minimum fields stable where practical:

- `reasoning_type`: mode identifier (e.g., `"abduction"`).
- `result` or `output`: primary reasoning conclusion.
- `confidence`: normalized confidence in `[0.0, 1.0]` where meaningful.
- `trace`: concise explanation of steps/criteria used.
- `metadata`: optional diagnostics (timing, matches, heuristics, etc.).

---

## Composition Model

Reasoning types are designed to work in **single-mode** and **multi-mode** chains.

### Single-Mode Execution
One reasoning type receives input + context and returns its own structured result.

### Multi-Mode Chaining
Multiple reasoning types can execute sequentially (left-to-right), where each step can read:
- original input,
- initial shared context,
- outputs from prior steps.

This allows pipelines such as:
- `abduction -> deduction` (hypothesis generation then logical filtering),
- `decomposition -> induction -> deduction`,
- `analogical -> cause_effect`.

### Chaining Guidance
- Keep step outputs compact and structured.
- Pass only necessary state forward.
- Normalize confidence before downstream comparison.
- Prefer deterministic merge rules for combined outputs.

---

## Suggested Dataflow

```mermaid
flowchart LR
    A[Task Input] --> B[Reasoning Type 1]
    B --> C[Intermediate Structured Result]
    C --> D[Reasoning Type 2]
    D --> E[Combined/Final Result]
    E --> F[Validation + Memory]
```

---

## Integration with the Parent Reasoning Subsystem

The `types/` layer should integrate cleanly with:

- **Rule Engine (`rule_engine.py`)** for symbolic rule-aware operations,
- **Validation Engine (`validation.py`)** for consistency/soundness checks,
- **Reasoning Memory (`reasoning_memory.py`)** for experience tagging and replay,
- **Reasoning Cache (`reasoning_cache.py`)** for repeated-query optimization.

### Integration Principles
- Keep type modules stateless where possible.
- If a type needs shared state, expose explicit lifecycle hooks.
- Avoid hidden global mutation that bypasses validation/memory layers.

---

## Quality and Robustness Standards

### Input Validation
Each reasoning type should defensively validate:
- input type/shape,
- required fields,
- domain assumptions,
- confidence bounds.

### Determinism and Reproducibility
For the same input/context:
- deterministic paths should be stable,
- stochastic paths should expose seed/config controls,
- emitted traces should explain any non-deterministic decisions.

### Error Semantics
Prefer typed exceptions from the parent reasoning error hierarchy.
Include actionable context (which phase failed, and why).

### Confidence Handling
- Clamp/normalize confidence where appropriate.
- Do not silently default over explicit caller-provided values.
- Use explicit `None` checks for override semantics.

---

## Extension Guide: Adding a New Reasoning Type

1. **Create module** under `types/` (e.g., `reasoning_counterfactual.py`).
2. **Implement base contract** from `base_reasoning.py`.
3. **Add export/wiring** via `types/__init__.py` and any type registry/factory used by the parent package.
4. **Document behavior** in this README (purpose, inputs, outputs, limits).
5. **Add tests** for:
   - nominal paths,
   - malformed inputs,
   - boundary confidence values (`0`, `1`, `0.0`),
   - composition with at least one existing type.

### Backward Compatibility Expectations
- Keep output keys stable unless versioning/migration is introduced.
- Avoid breaking call signatures without coordinated subsystem updates.

---

## Practical Usage Patterns

### Pattern A: Explain-then-Verify
1. Run `abduction` to generate hypotheses.
2. Run `deduction` to verify hypotheses against rule premises.
3. Send final claims to validation for contradiction/redundancy checks.

### Pattern B: Decompose-then-Induce
1. Run `decompositional` to split a complex scenario.
2. Run `inductive` on each subcomponent.
3. Aggregate with explicit confidence weighting.

### Pattern C: Analogical Bootstrap
1. Run `analogical` using a trusted source domain.
2. Validate mapped assumptions with `cause_effect` reasoning.
3. Persist accepted mappings as reusable reasoning artifacts.
