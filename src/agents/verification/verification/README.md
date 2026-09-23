# SLAI v2.3 Verification Subsystem

`src/agents/verification/` is the formal-methods substrate for a future `VerificationAgent`. It verifies explicit formal objects; it does **not** score generic output quality, make runtime safety decisions, execute actions, repair source code, or replace SLAI's Reasoning, Safety, Evaluation, Quality, or Execution agents.

## Scope

The package provides six tightly bounded capabilities:

1. **Formal contracts and obligations** — named state predicates, Hoare-style pre/postconditions, invariants, and concrete proof-obligation representation.
2. **Abstract verification support** — a minimal semilattice protocol and bounded ascending post-fixpoint computation, with optional caller-supplied widening.
3. **SAT/SMT verification** — a backend-neutral Boolean/integer/real term language, capability-aware solver interface, deterministic backend registry, SAT/UNSAT/UNKNOWN preservation, and optional Z3 integration.
4. **Finite-state model checking** — explicit transition systems, deterministic breadth-first reachability/safety checking, inductive-invariant checking, resource/depth limits, and structured witness/counterexample traces.
5. **Proof/certificate evidence** — backend- or caller-supplied proof artifact metadata is represented without inventing or upgrading proof claims.
6. **Transient verification memory** — bounded, thread-safe result history for the future agent facade; durable persistence remains with BaseAgent/checkpointing.

The subsystem intentionally does **not** implement a general theorem prover, custom SAT/SMT solver, generic validation framework, or a full LTL/CTL/CTL* model checker. Those would either duplicate other SLAI responsibilities or exceed what can be implemented compactly without overstating formal guarantees.

## Result semantics

`VerificationStatus` deliberately distinguishes `VERIFIED`, `REFUTED`, `SATISFIABLE`, `UNSATISFIABLE`, `UNKNOWN`, and `BOUNDED`. `BOUNDED` means the declared semantic depth was exhausted without a decisive witness/counterexample. `UNKNOWN` is used for solver uncertainty, unsupported/incomplete quantified reasoning, or non-semantic resource exhaustion. Neither status is treated as proof.

A complete explicit-state traversal may return `VERIFIED` or `UNSATISFIABLE` even when configured resource ceilings exist, but only when the reachable finite state space was actually exhausted without hitting those ceilings. A discovered witness or counterexample is decisive even if found before the search bound, because the trace itself establishes the existential fact or violation.

## Configuration

No configuration loader is defined here. Use SLAI's existing loader and pass the resulting mapping into `VerificationSettings`:

```python
from src.agents.base.utils.config_loader import load_global_config
from src.agents.verification import VerificationSettings

raw = load_global_config("src/agents/verification/configs/verification_config.yaml")
settings = VerificationSettings.from_mapping(raw)
```

`verification_config.yaml` contains only declarative defaults for solver preference, model-checking resource bounds, abstract-interpretation iteration bounds, and transient verification-memory capacity.

## Optional solver backend

Z3 is optional and is not imported when the package is imported. If `z3-solver` is absent, the subsystem remains importable; selecting Z3 raises `SolverUnavailableError`. Other solvers can be integrated by implementing the narrow `SolverBackend` protocol and registering a factory in a `BackendRegistry`. No mutable global backend registry or solver context is created.

## Academic design basis

The implementation uses the supplied sources conservatively:

| Source | Implementation consequence |
| --- | --- |
| C. A. R. Hoare, *An Axiomatic Basis for Computer Programming* | `HoareContract`, preconditions/postconditions, consequence/composition/iteration-oriented `ProofObligationKind`, and explicit partial-correctness wording without a termination guarantee. |
| Cousot & Cousot, *Abstract Interpretation* | `AbstractDomain`, semilattice join/order, bounded least/post-fixpoint-style ascending iteration, optional widening, and explicit non-convergence rather than fabricated certainty. |
| Baier/Katoen, *Model Checking — Regular Properties* chapter | Safety violations represented by finite bad-prefix-style counterexample traces; regular-safety reasoning motivates invariant/reachability reduction rather than generic output checking. |
| Clarke et al., *Model Checking and the State Explosion Problem* | Finite transition-system representation, exhaustive reachable-state search, counterexample paths, bounded search semantics, visited-state tracking, and hard state/transition/time controls. |
| *A History of Satisfiability* | SAT/UNSAT are distinct semantic outcomes; `prove` establishes validity by checking the unsatisfiability of assumptions conjoined with the negated property. |
| Mann et al., *Smt-Switch* | Backend-neutral `SolverBackend`, common terms/sorts/capabilities, deterministic factories/registry, and exact SAT/UNSAT/UNKNOWN translation. |
| Niemetz et al., *Syntax-Guided Quantifier Instantiation* | Quantified formulas are capability-marked; incomplete or unsupported quantifier handling remains `UNKNOWN`, never a proof by heuristic exhaustion. |
| *Model Checking Security Protocols* | Counterexamples are preserved as concrete structured state/transition traces; bounded search is reported with its scope rather than promoted to unbounded verification. |

## Architecture and dependency direction

```text
verification_types.py ───────────────┐
utils/verification_errors.py ────────┼──> verification_proof.py ──> verification_result.py
                                     │
formal/ ─────────────────────────────┤
solving/solver.py ─> solving/backends.py ─> solving/satisfiability.py
model/transition_system.py ───────────────> model/model_checker.py
verification_result.py ─> verification_helpers.py ─> verification_memory.py
                                     │
                                     └──> package __init__.py
```

Functional folders are exactly `formal/`, `solving/`, and `model/`. `utils/` contains only Verification-specific exceptions, while the three root modules `verification_helpers.py`, `verification_memory.py`, and `verification_proof.py` provide cross-cutting formal utilities/evidence without adding another functional folder. `configs/` remains declarative configuration only.

## SLAI reuse decisions

- `src/agents/base/utils/base_errors.py`: reused directly; every Verification exception ultimately derives from SLAI `BaseError` and follows its code/context/cause/retryability contract.
- `src/agents/base/utils/config_loader.py`: reused by callers; not wrapped, copied, or modified.
- `logs/logger.py`: not duplicated. The subsystem is deliberately stateless and library-like, so it leaves operational logging to the future agent façade rather than emitting hidden solver-global logs.
- `src/tuning/utils/tuning_helpers.py`: reused directly for stable fingerprints and UTC timestamp serialization, avoiding duplicate hashing/serialization helpers. No tuning policy or tuner behavior is imported.
- `checkpointing/` and `BaseMemory`: not reimplemented. `VerificationMemory` is explicitly transient and bounded; `VerificationResult.to_dict()` remains the hand-off representation for future BaseAgent/checkpoint persistence.
- `src/utils/`: no current helper has verification semantics matching solver terms, fixpoints, or explicit-state model checking, so unrelated statistics/parallel/config helpers are not repurposed.

## Cross-cutting modules

- `verification_helpers.py` contains Verification-specific deterministic term/result fingerprints and structural term statistics while delegating generic hashing/serialization to SLAI tuning helpers.
- `verification_memory.py` provides a bounded `RLock`-protected FIFO history of immutable `VerificationResult` records with status/method/backend/property/tag filters. It performs no filesystem persistence.
- `verification_proof.py` owns `ProofArtifact` and `ProofArtifactSet`, validating only the existence/shape of supplied evidence. It never fabricates proof content or treats an artifact as independently checked.

## Minimal usage

```python
from src.agents.verification import (
    Invariant,
    ModelChecker,
    Transition,
    TransitionSystem,
)

system = TransitionSystem(
    states=("idle", "running", "error"),
    initial_states=("idle",),
    transitions=(
        Transition("idle", "running", "start"),
        Transition("running", "error", "fail"),
    ),
)

result = ModelChecker().check_invariant(
    system,
    Invariant("never_error", lambda state: state != "error"),
)

assert result.status.value == "refuted"
assert result.trace is not None
```

## Testing

The repository test file is `tests/test_verification_subsystem.py`. Core tests do not require any optional SMT package; solver behavior is faked only at the external `SolverBackend` boundary. A real-Z3 test is automatically skipped when `z3-solver` is absent.
