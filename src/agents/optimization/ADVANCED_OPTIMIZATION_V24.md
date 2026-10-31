# SLAI v2.4 — Advanced Optimization Runtime and Formulations

**Implementation branch:** `opt-v24-temporal-stochastic-runtime-20261031`  
**Scope:** incremental general-purpose changes to the existing canonical Optimization architecture.  
**Release status:** *Integration tests and benchmark measurements pending*.

## 1. Architecture and ownership

The existing canonical `OptimizationProblem`, `Variable`, `Objective`,
`Constraint`, `Solution`, solver registry, `OptimizationSolver`, and
`OptimizationMemory` remain authoritative. New functions construct those
types or support the existing `BlackBoxSolver`. There are **no** separate
SPADRA, CAD, robot or animation optimizers.

- `domain_problems.py` owns generic problem formulations and sampled motion
  utilities. It does **not** own a scene graph, collision engine or playback.
- `derivative_checks.py` adapts existing STEM `NumericalMethods` scalar
  finite differences to the canonical objective-gradient interface.
- `evaluation_runtime.py` owns candidate-evaluation checkpoints, **not**
  optimization algorithms or global SLAI memory.
- `BlackBoxSolver` continues to own pattern search and SciPy differential
  evolution. Checkpoints and cancellation remain solver options.
- Orientation metrics delegate matrix validation and SO(3) logarithms to
  existing Spatial transformation utilities.

## 2. Time-dependent constraints

`trajectory_problem(..., sampled_state_constraints=[(name, evaluator, sense, rhs)])`

Here `evaluator(time_seconds, position_tuple)` returns a numeric
left-hand side evaluated at **every discretized waypoint including endpoints**.
`sense` is an existing `ConstraintSense`. Each sample becomes a canonical
`Constraint` with a stable identifier and sample-time metadata.

Time-varying restrictions, moving-obstacle surrogates, zone boundaries or
application-specific limits can use the same callback contract.

**Guarantee limitation:** all constraints are checked at sample times only.
No swept-volume, continuous collision or intersample feasibility certificate
is implied. These callbacks must be trusted and side-effect free.

`sampled_trajectory_derivatives(positions, duration, order=1|2|3)` provides
uniform-grid finite differences for velocity, acceleration and jerk.
Uniform spacing and sample ordering are explicit. Higher derivatives amplify
noise; discrete values are not continuous-time derivatives between samples.

## 3. Orientation handling

`orientation_geodesic_distance(first, second)` computes the shortest
relative SO(3) rotation angle using `Spatial.modules.transform`.

`orientation_path_metrics(rotations, duration)` reports discrete step angles,
angular speed magnitudes, and total rotational travel.

This does **not** claim quaternion interpolation, angular acceleration,
rotational jerk, orientation constraint satisfaction or continuous path
certification. Applications may supply an external orientation evaluator as
a canonical objective/constraint callback.

## 4. Scenario-based stochastic / risk-aware formulations

The existing `robust_scenario_problem` implements finite-scenario worst-case
loss minimization. Two additional builders reuse the canonical model:

- `expected_scenario_problem` minimizes normalized finite-distribution
  weighted mean scenario loss.
- `cvar_scenario_problem` minimizes finite-distribution conditional
  value-at-risk (CVaR), with a confidence level in `(0, 1)`,
  a VaR threshold variable `cvar_eta`, and one nonnegative excess variable
  per scenario.

Both require explicitly supplied fixed, trusted scenario evaluators.
Probability masses are validated and normalized with overflow-resistant
scaling. The CVaR formulation follows Rockafellar and Uryasev (2000).

These formulations are **not** general chance-constrained solvers, random
sample generators, stochastic-gradient methods, distributionally robust
optimization or unconditional probabilistic certificates. They are exact
reformulations for the provided discrete scenarios, subject to solver
accuracy and feasibility.

## 5. Mixed-variable optimization

The existing `BlackBoxSolver` already accepts bounded continuous, integer,
binary and finite-domain variables. This work adds regression coverage
instead of introducing duplicate solver backends.

Domain compatibility remains solver-specific. In particular, unbounded
black-box domains, arbitrary categorical encodings and mixed-integer
nonlinear global guarantees are unsupported or unproven.

## 6. Derivative accuracy diagnostics

`verify_objective_gradient(problem, point)` requires a scalar deterministic
objective with an analytical `Objective.gradient` callback and real scalar
decision variables.

It uses existing STEM central differences in the interior and one-sided
differences near box boundaries. Results include per-variable analytical
and numerical derivatives, step sizes, error tolerances and pass/fail
diagnostics. It does not silently alter gradients or affect solver behavior.
It is not a Hessian verifier or proof of differentiability.

## 7. Evaluation caching, interruption and resume

`BlackBoxSolver.solve` now accepts the following optional keyword arguments:

| Option | Purpose |
| --- | --- |
| `cancel_event` | Object exposing `is_set()`, e.g. `threading.Event`. Checked cooperatively before evaluations and at pattern-search iterations. |
| `checkpoint_path` | File path for persistent candidate evaluation evidence. |
| `evaluator_revision` | Required explicit stable identifier for the exact caller-owned evaluator and model state. |
| `resume` | Boolean, default false. **Only pattern_search** supports algorithm-state resume. |
| `checkpoint_max_bytes` | Maximum serialized checkpoint file size (default 32 MiB). |
| `max_evaluations` | **Total unique** evaluation budget, counting restored checkpoint evaluations. |

**Cache validity and failure model:**

- Only problems declaring deterministic evaluations can persist candidate
  cache entries. These declarations are caller responsibilities.
- Checkpoint identity includes the existing `OptimizationMemory` problem
  fingerprint, explicit evaluator revision, method and relevant solver
  configuration. A mismatch is rejected.
- JSON checkpoints are atomically replaced and SHA-256 checksummed to catch
  accidental corruption. Checksums are *not* authenticity signatures.
- Failed evaluations can be cached but can never become feasible incumbents.
- The same parameter vector is evaluated at most once per matching cache
  identity except for interrupted in-flight calls. A crash between actual
  external work and durable evidence may re-execute that work.
- **Pattern-search** resumes from its last durable completed iteration with
  cached evaluations reused, and may receive a larger total budget.
- SciPy differential evolution may persist candidate cache records, but
  cannot resume its exact population/RNG state: `resume=True` is rejected.
- Cancellation is cooperative; neither SciPy backends nor arbitrary blocked
  external callbacks are forcibly terminated. Hard resource/process isolation
  remains an external execution-layer responsibility.
- Checkpoints are single-writer local files, not a distributed cache,
  transactional application state, or untrusted-file sandbox.

A deterministic external evaluator MUST be pure and its revision must change
when any material input, geometry/model revision, solver objective semantics,
simulation binary, random seed, or environment setting changes. Reuse cannot
be safe when external state changes behind an unchanged identity.

## 8. Benchmarks and reproducibility

Run from the SLAI repository root:

```powershell
py -m compileall -q src/agents/optimization
py -m pytest -v tests/agents/test_optimization_advanced_v24.py
py -m pytest -q
py -m benchmarks.optimization_advanced_v24 --repeat 3 --json benchmark.json
```

The synthetic benchmarks measure wall-clock duration, solution status,
feasibility, objective value and evaluation count for simple scalar,
mixed-variable and finite-distribution cases. Report measured hardware and
software versions. There are no invented speedup figures and benchmark
completion is not proof of optimality.

## 9. Explicit remaining gaps

- End-to-end validation under full SLAI runtime has not yet been executed
  for this implementation.
- General nonlinear optimal control, nonuniform-grid dynamic solvers and
  continuous-time constraint certification are not provided.
- Accurate angular acceleration/jerk require common-frame transport and
  independently validated dynamics; not included here.
- Advanced Bayesian optimization, surrogates and global MINLP methods have
  not been added; available established solvers should be reused where feasible.
- Out-of-process evaluator isolation, strict CPU/memory quota enforcement,
  multi-writer checkpoint locking and distributed resume remain unimplemented.
- Differential evolution full-state deterministic restart remains unsupported.
- End-to-end Spatial, STEM, Simulation, Handler and Provenance execution
  bindings still require dedicated integration tests.
- Solver performance improvements must be substantiated through measured
  repeated benchmarks before any speedup is claimed.

**Production certification is withheld pending real repository tests,
regression checks, numerical accuracy evidence and independent benchmarks.**
