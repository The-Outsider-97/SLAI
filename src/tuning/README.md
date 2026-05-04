# Tuning Module (`src/tuning/`)

This package provides the unified tuning pipeline for SLAI models, including:

- strategy orchestration (`HyperparamTuner`),
- Bayesian optimization (`BayesianSearch`),
- deterministic grid search (`GridSearch`),
- built-in neural network evaluators for uncertainty-aware and deterministic models,
- configuration loading/validation and persisted reports.

## Directory map

- `tuner.py`  
  Top-level orchestrator that selects a strategy from config (`tuning.strategy`) and executes the appropriate search engine.

- `bayesian_search.py`  
  Production-style Bayesian optimization wrapper around `skopt.gp_minimize` with:
  - explicit config validation,
  - search space validation,
  - optional built-in BNN evaluation,
  - JSON summaries + optional progress plots.

- `grid_search.py`  
  Exhaustive combinatorial search with:
  - k-fold CV evaluation,
  - optional built-in GNN evaluation,
  - robust candidate validation,
  - JSON summaries + optional performance plots.

- `networks/`  
  Native NumPy model implementations used by the tuning stack:
  - `bayesian_neural_network.py` (variational BNN),
  - `grid_neural_network.py` (deterministic dense NN for grid tuning).

- `utils/`  
  Shared infrastructure:
  - `config_loader.py` for loading global YAML config,
  - `evaluator.py` for consistent BNN/GNN candidate scoring,
  - `tuning_error.py` for structured error taxonomy.

- `configs/hyperparam.yaml`  
  Single source of truth for runtime defaults and model search spaces.

## Configuration contract

Primary config path: `src/tuning/configs/hyperparam.yaml`.

Expected top-level sections:

- `model`, `run`, `training` (global metadata/runtime defaults)
- `tuning` (strategy selector + generation behavior)
- `bayesian_search` (Bayesian-specific runtime settings)
- `grid_search` (grid-specific runtime settings)
- `bnn` (Bayesian neural network defaults)
- `gnn` (deterministic neural network defaults)
- `hyperparameters` (model-keyed search spaces)
- `configs` (generated JSON config output/input references)

### Hyperparameter spec rules

Each model entry under `hyperparameters` is a list of parameter specs. Each parameter must include:

- `name` (unique within model scope)
- `type` (`integer`, `real`, or `categorical`; aliases `int`/`float` are normalized)
- either:
  - `values` (categorical/discrete candidates), or
  - `min` + `max` (for numeric ranges)
- optional `prior` (for Bayesian numeric dimensions, e.g. `log-uniform`)

## Runtime behavior

### Strategy selection

`HyperparamTuner` reads `tuning.strategy` and dispatches to:

- `BayesianSearch` when `strategy: bayesian`
- `GridSearch` when `strategy: grid`

### Built-in evaluators

- Bayesian search can directly evaluate built-in `BayesianNeuralNetwork` candidates when no external evaluation function is provided.
- Grid search can directly evaluate built-in `GridNeuralNetwork` candidates when no external evaluation function is provided.

### Reporting and artifacts

Both search strategies write timestamped JSON summaries to configured output paths and can render plots when enabled.

## Consistency expectations

When updating tuning code/config, keep these invariants:

1. **No duplicated config authority**
   - Keep strategy parameters in strategy sections (`bayesian_search`, `grid_search`) rather than repeating them in `tuning`.

2. **Model defaults stay in model sections**
   - BNN defaults belong in `bnn`.
   - GNN defaults belong in `gnn`.

3. **Search spaces stay model-keyed**
   - Every model’s hyperparameters remain isolated and free of duplicate `name` entries.

4. **Generated config references are accurate**
   - Paths under `configs` should point to real `src/tuning/configs/*.json` locations.

## Recommended maintenance workflow

1. Update `hyperparam.yaml` first.
2. Validate that all required keys are present for any strategy/model you run.
3. Run a smoke tuning job for both strategies (or unit tests around configuration loading/validation).
4. Verify JSON/plot outputs are produced under the configured report directories.
