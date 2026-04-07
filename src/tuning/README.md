This module provides the project’s hyperparameter-tuning surface and uncertainty-aware utility models.

## Scope

- **Search orchestration:** `tuner.py` chooses and executes the configured strategy.
- **Search strategies:**
  - `bayesian_search.py` for Bayesian optimization.
  - `grid_search.py` for exhaustive discrete search.
- **Config management:** `configs/hyperparam.yaml` with strategy- and model-specific parameter spaces.
- **Utilities:** `utils/` contains helper components (including the Bayesian neural network implementation).

## Design principles

The tuning stack should be treated as production infrastructure, not experimental notebooks:

1. **Deterministic where practical**
   - Explicit random seeds for optimization/model sampling.
   - Timestamped run artifacts for traceability.

2. **Strict validation and fail-fast behavior**
   - Validate strategy settings (fold counts, call budgets, etc.) at startup.
   - Validate search spaces (duplicate names, invalid bounds, empty candidate sets).

3. **Failure isolation**
   - Individual trial/fold failures should not crash entire runs.
   - Failures are logged and reflected in result artifacts.

4. **Operational observability**
   - Structured logs for start/stop, best-so-far updates, and failures.
   - Persisted JSON summaries and progress plots for post-run analysis.

5. **Consistent interfaces**
   - Shared conventions across search strategies for initialization, execution, and outputs.

## Configuration contract

Primary config file: `src/tuning/configs/hyperparam.yaml`.

Expected top-level sections:

- `hyperparameters`: model-keyed search-space definitions.
- `tuning`: global strategy selection.
- `bayesian_search`: Bayesian runtime settings.
- `grid_search`: grid runtime settings.

### Hyperparameter specification

Each parameter spec should provide:

- `name` (unique per model)
- `type` (`integer`, `real`, or `categorical` depending on strategy)
- either explicit `values` or numeric bounds (`min`/`max`) where supported.

## Usage patterns

### Bayesian search

- Provide an objective function `f(params: dict) -> float`.
- Choose objective semantics (`minimize` or `maximize`).
- Run `run_search()` and consume:
  - best parameters,
  - best score,
  - optimizer state.

### Grid search

- Provide evaluation function
  `f(params, X_train, y_train, X_val, y_val) -> float`.
- Run `run_search(X, y)` with data arrays.
- Consume best parameters and persisted result summary.

## Bayesian Neural Network utility

`utils/bayesian_neural_network.py` provides a lightweight variational BNN that can be used by planning/selection components for uncertainty-aware scoring.

Key behavior:

- Gaussian variational posterior over weights/biases.
- Reparameterization trick for Monte Carlo estimates.
- ELBO-based updates and predictive uncertainty from MC sampling.
- JSON save/load for reproducible deployment snapshots.

## Operational recommendations

- Keep search spaces bounded and auditable; avoid unreviewed auto-generated large spaces.
- Track dependency versions (`numpy`, `scikit-learn`, `scikit-optimize`, `joblib`, `matplotlib`) in lock files.
- Prefer small smoke tests in CI and larger tuning jobs in scheduled/batch environments.
- Version artifact schemas if external tooling depends on JSON field names.
