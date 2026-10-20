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

### Result contract (`ReasoningTypes.reason()`)
`reason()` returns the `slai.reasoning.result.v1` envelope around the strategy's native result
(`execute()` still returns the native result unchanged). Fields that matter when interpreting it:

| Field | Meaning |
|---|---|
| `outcome` | `supported`, `unsupported`, `disproven`, `indeterminate`, `partial`, `failed`, or `completed` (decompositional only) |
| `confidence` | `{value, source, calibrated, calibration_method}`. **Never calibrated**: a heuristic score from the strategy, not a probability |
| `errors` | per-step failures (always present, empty for a clean single run) |
| `contradictions` | internal premise conflicts and, for abduction, hypotheses contradicted by evidence |
| `stop_reason` | why the strategy stopped (for example `step_budget_exhausted`, `inconsistent_premises`, `hypothesis_refuted`, `no_grounded_hypothesis`, `association_only_no_mechanism`, `not_decomposable`) |

Outcome rules per strategy:
- **deduction** - `supported` only when the hypothesis is derived at or above `certainty_threshold`; `disproven` when the premises entail its negation; otherwise `indeterminate`. Native result also carries `verdict` (`entailed`, `entailed_low_confidence`, `refuted`, `inconsistent`, `undetermined`) and `truncated`.
- **abduction** - `unsupported` when the best hypothesis is only a generic template (`origin == "template"`); `partial` when it was chosen under relaxed criteria; `indeterminate` with no admissible hypothesis.
- **induction** - `supported` only if validation passes; `evidence_level` states whether that was `in_sample` or `holdout`.
- **analogical** - `partial` when the best transfer fails its own validation.
- **cause_effect** - `supported` needs an explicit link, a declared mechanism or a mechanism score >= 0.5; association/temporal precedence alone is `partial`.
- **decompositional** - `completed`, or `indeterminate` when the input was not actually decomposable.

---

## Composition Model

Reasoning types are designed to work in **single-mode** and **multi-mode** chains.

### Single-Mode Execution
One reasoning type receives input + context and returns its own structured result.

### Multi-Mode Chaining
Strategies can be combined with `+` (for example `abduction+deduction`) and run left-to-right,
at most `reasoning_types.max_combined_types` (default 3) per expression.

What a later step actually receives:
- the original problem and the shared context,
- a **deep copy** of the immediately preceding step's native result under
  `context["prev_step_result"]` (and `context["prev_<name>_result"]`), so a step cannot corrupt an
  earlier one.

What it does **not** do: the six built-in strategies do not read `prev_step_result`. A combined
run of built-ins is therefore *independent analyses of the same input, merged*, not a data
pipeline; output of step 1 is not an input of step 2 unless you pass it through `context`
yourself (for example `context["premises"]`). Custom strategies registered with `register()`
may consume `prev_step_result`.

Failure semantics: each step runs in its own `try/except`. A failing step is recorded in
`errors` / `step_errors` and the run continues (set `context["combined_fail_fast"] = True` to stop at
the first failure). The combined status is `success`, `partial` (some step failed) or `failed`
(every step failed, which raises `InferenceExecutionError`). Aggregate outcome rules:
`supported` + `disproven` -> `indeterminate`; any step error -> `partial`; a final `supported` after
an earlier `unsupported`/`indeterminate`/`partial`/`failed`/`disproven` step -> `partial`; otherwise the
last step's outcome.

### Names and aliases
The canonical names are `abduction`, `deduction`, `induction`, `analogical`, `decompositional`,
`cause_effect` (the learned selector's label space is exactly these six and is unchanged).
`ReasoningTypes.resolve_strategy_name()` maps aliases (`abductive`, `deductive`, `inductive`,
`analogy`, `decomposition`, `causal`, `cause_and_effect`, and `<name>_reasoning`) to them, in single
and combined expressions. `register()` is instance-scoped: registering a strategy on one
`ReasoningTypes` never affects another instance or the class defaults.

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
- Strategy instances hold per-run state; `ReasoningTypes` creates a fresh instance per request and does not cache instances. Do not share one instance across threads.
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

## Known limitations (read before relying on a result)

- **Deduction** is a small forward-chaining engine over textual patterns: modus ponens/tollens, categorical syllogism, hypothetical and (n-ary, sound) disjunctive syllogism. Negation is recognised only in the forms `not P` and `P is not Q`; "it does not rain" is not matched to "it rains". Possession (`has`) is deliberately not class membership. Rule-confidence decay is off by default (`apply_rule_decay: false`) because deduction is truth-preserving; certainty is the weakest premise's confidence.
- **Induction** reports in-sample fit unless `holdout_data` is supplied; numeric trends need a significant slope (`trend_significance_alpha`, default 0.05); temporal/periodic patterns need explicit timestamps. Extrapolations are labelled `unvalidated_extrapolation`.
- **Abduction** has no knowledge base of its own. Without `seed_hypotheses` (optionally with `prior` and `explains`) or domain knowledge it can only produce templates, and the outcome is `unsupported`.
- **Cause/effect** scores are heuristics (temporal order, co-variation, a two-rule keyword counterfactual). Without a `counterfactual_simulator` each relationship's `counterfactual_basis` is `builtin_rule` or `default`. `intervention_model: "confidence_scaling"` is **not** Pearl's `do()` operator; there is no structural causal model.
- **Analogical** similarity is lexical (`difflib`), not semantic.
- **Decompositional** is structural; free text is returned as `decomposed: false`.
- Inputs nested deeper than `reasoning_types.max_input_depth` (default 64) are rejected with `ReasoningValidationError`.

### Not implemented here (by design)
Temporal, constraint, spatial, mathematical and probabilistic reasoning are delegated to the planning, verification, spatial and stem agents and to `probabilistic_models`; counterfactual reasoning extends `cause_effect`; diagnostic reasoning is the composition `abduction+cause_effect(+deduction)`. Defeasible and argumentation reasoning are not provided. No new registered strategy was added, so the learned selector's label space is unchanged. The selector checkpoint is synthetic-only and not production-eligible; keep the keyword policy authoritative.

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
