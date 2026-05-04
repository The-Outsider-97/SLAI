# Tuning Networks (`src/tuning/networks/`)

This directory contains the two native neural-network implementations used by the tuning subsystem.

## Modules

## 1) `bayesian_neural_network.py`
Variational Bayesian neural network (NumPy implementation) for uncertainty-aware regression.

### Key characteristics
- Diagonal Gaussian variational posterior over weights/biases.
- ELBO-oriented training with Monte Carlo sampling.
- Predictive uncertainty estimation via repeated stochastic forward passes.
- Structured persistence (JSON payload with format metadata).
- Config-driven defaults loaded from `bnn` section in `hyperparam.yaml`.

### Important config inputs (`bnn`)
- Core: `learning_rate`, `prior_mu`, `prior_logvar`, `random_state`
- Numerical stability: `min_variance`, `stability_epsilon`, `logvar_clip_range`
- Optimization controls: `gradient_clip_norm`, `weight_init_scale`
- Activations/noise: `hidden_activation`, `leaky_relu_slope`, `likelihood_std`
- Subsections:
  - `training`: epochs, batch size, early stopping settings
  - `prediction`: Monte Carlo sampling defaults and interval quantiles
  - `persistence` / `monitoring`: output formatting + diagnostics toggles

## 2) `grid_neural_network.py`
Deterministic dense neural network (NumPy implementation) intended for exhaustive/grid-based tuning workflows.

> Note: despite the filename, this is not a graph neural network.

### Key characteristics
- Supports regression and classification task modes.
- Configurable hidden/output activations with task compatibility checks.
- Supports Adam or SGD optimization behavior.
- Supports optional dropout, L2 regularization, gradient clipping, and early stopping.
- Structured persistence (JSON payload with format metadata).
- Config-driven defaults loaded from `gnn` section in `hyperparam.yaml`.

### Important config inputs (`gnn`)
- Task/optimizer: `task_type`, `optimizer`
- Core training controls: `learning_rate`, `batch_size`, `epochs`, `shuffle`
- Regularization/stability: `dropout_rate`, `l2_lambda`, `stability_epsilon`, `gradient_clip_norm`
- Adam controls: `beta1`, `beta2`, `adam_epsilon`
- Activation/output behavior: `hidden_activation`, `output_activation`, `leaky_relu_slope`
- Inference/classification: `prediction_threshold`
- Subsections:
  - `training`: epoch + early stopping behavior
  - `grid_defaults`: default scoring metric used by evaluator
  - `persistence` / `monitoring`: serialization + diagnostic toggles

## Integration points

- Both models are re-exported via `src/tuning/networks/__init__.py`.
- Candidate evaluation is mediated through `src/tuning/utils/evaluator.py`.
- Hyperparameter candidate spaces are defined in `src/tuning/configs/hyperparam.yaml` under:
  - `hyperparameters.BayesianNeuralNetwork`
  - `hyperparameters.GridNeuralNetwork`

## Practical guidance

- Keep constructor defaults in sync with YAML defaults.
- Add/rename constructor-fit parameters together with evaluator allowlists.
- Preserve stable JSON persistence fields when possible to avoid backward compatibility breaks in saved artifacts.
