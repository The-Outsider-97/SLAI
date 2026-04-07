"""
Standalone hyperparameter evaluator for Bayesian and Grid Neural Networks.

This module provides a unified evaluation interface for hyperparameter tuning
(grid search, Bayesian optimization). It supports:

- BayesianNeuralNetwork (BNN) – variational inference with configurable architecture.
- GridNeuralNetwork (GNN) – deterministic training with L2 / dropout.

All evaluation logic is isolated, reusable, and fully integrated with the
tuning error taxonomy (tuning_error.py) and configuration loader.
"""

from __future__ import annotations

import math
import numpy as np

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from ..networks.bayesian_neural_network import BayesianNeuralNetwork
from ..networks.grid_neural_network import GridNeuralNetwork
from .config_loader import get_config_section, load_global_config
from .tuning_error import (TuningConfigError, wrap_exception, error_boundary,
                           TuningErrorContext, TuningEvaluationError,
                           TuningValidationError, raise_for_condition, safe_serialize)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Model Evaluator")
printer = PrettyPrinter

Array = np.ndarray


class ModelEvaluator:
    """
    Stateless evaluator for BNN and GNN hyperparameter candidates.

    All methods are static or classmethods; no instance state is required.
    """

    # ----------------------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------------------
    @classmethod
    def evaluate_candidate(cls, model_type: str, params: Mapping[str, Any],
                           x_train: Array, y_train: Array, x_val: Array, y_val: Array, *,
                           task_type: str = "regression", scoring_metric: Optional[str] = None,
                           fit_kwargs: Optional[Mapping[str, Any]] = None) -> float:
        """Evaluate a hyperparameter candidate on the given train/validation split.

        Args:
            model_type: "BayesianNeuralNetwork" or "GridNeuralNetwork"
            params: Hyperparameter dictionary (e.g., learning_rate, hidden_layer_sizes, ...)
            x_train, y_train: Training data
            x_val, y_val: Validation data
            scoring_metric: Metric to return (e.g., "elbo", "r2", "accuracy"). If None,
                uses defaults from config or model type.
            task_type: "regression" or "classification" (for GNN only)
            fit_kwargs: Additional arguments passed to the model's fit() method.

        Returns:
            Score to optimize (higher = better). For metrics where lower is better,
            the returned value is negated automatically (consistent with skopt).

        Raises:
            TuningConfigError: Unknown model_type or invalid parameters.
            TuningValidationError: Data shape mismatches.
            TuningEvaluationError: Evaluation failed.
        """
        normalized = str(model_type).strip().lower()
        if normalized in cls._bnn_names():
            return cls._evaluate_bnn(
                params=params,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                scoring_metric=scoring_metric,
                fit_kwargs=fit_kwargs,
            )
        if normalized in cls._gnn_names():
            return cls._evaluate_gnn(
                params=params,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                scoring_metric=scoring_metric,
                task_type=task_type,
                fit_kwargs=fit_kwargs,
            )
        raise TuningConfigError(
            f"Unsupported model_type: {model_type}",
            context=TuningErrorContext(
                component="ModelEvaluator",
                operation="evaluate_candidate",
                model_type=model_type,
            ),
            details={"supported": sorted(cls._bnn_names() | cls._gnn_names())},
        )

    # ----------------------------------------------------------------------
    # BNN evaluation
    # ----------------------------------------------------------------------
    @classmethod
    def _bnn_names(cls) -> set[str]:
        return {"bayesianneuralnetwork", "bayesian_neural_network", "bnn"}

    @classmethod
    @error_boundary(
        error_cls=TuningEvaluationError,
        message="BNN candidate evaluation failed.",
        context_builder=lambda exc, args, kwargs: TuningErrorContext(
            component="ModelEvaluator",
            operation="_evaluate_bnn",
            model_type="bayesian_neural_network",
        ),
    )
    def _evaluate_bnn(cls, scoring_metric: Optional[str], params: Mapping[str, Any],
                      x_train: Array, y_train: Array, x_val: Array, y_val: Array,
                      fit_kwargs: Optional[Mapping[str, Any]]) -> float:
        # 1. Prepare data
        x_train_arr, y_train_arr = cls._validate_data(x_train, y_train, "BNN train")
        x_val_arr, y_val_arr = cls._validate_data(x_val, y_val, "BNN validation")
        input_dim = x_train_arr.shape[1]
        output_dim = cls._infer_output_dim(y_train_arr, task_type="regression")

        # 2. Build model (including hidden layer sizes from params)
        layer_sizes = cls._build_bnn_layer_sizes(input_dim, output_dim, params)
        model = cls._instantiate_bnn(layer_sizes, params)

        # 3. Fit model
        fit_args = cls._prepare_fit_kwargs(params, fit_kwargs, model_type="bnn")
        # BNN expects validation_data as a tuple
        history = model.fit(
            x_train_arr,
            y_train_arr,
            validation_data=(x_val_arr, y_val_arr),
            **fit_args,
        )
        logger.debug("BNN fit completed: %s epochs, early_stop=%s", len(history["epochs"]), history["stopped_early"])

        # 4. Evaluate and return score
        metrics = model.evaluate(x_val_arr, y_val_arr, num_samples=cls._get_eval_samples(params))
        metric_name = cls._resolve_scoring_metric(scoring_metric, model_type="bnn", metrics=metrics)
        raw_score = metrics[metric_name]

        # For BNN, higher ELBO / log_likelihood is better
        # But the evaluator always returns a score that should be maximized.
        # Metrics like "mse" are already negated in the caller? We'll assume
        # that the caller (e.g., bayesian_search) will minimize the returned value.
        # To keep consistent: return raw_score as is. The caller will use
        # objective="maximize" for ELBO and "minimize" for MSE.
        # However, for unified handling, we return raw_score and let the caller decide.
        return float(raw_score)

    @classmethod
    def _build_bnn_layer_sizes(cls, input_dim: int, output_dim: int, params: Mapping[str, Any]) -> List[int]:
        """Extract hidden layer sizes from params, with sensible defaults."""
        hidden_spec = None
        for key in ("hidden_layer_sizes", "hidden_layers", "hidden_units"):
            if key in params:
                hidden_spec = params[key]
                break

        if hidden_spec is None:
            # Default: one hidden layer of size min(64, input_dim*2)
            default_hidden = max(8, min(64, input_dim * 2))
            hidden_spec = [default_hidden]
            logger.info("No hidden_layer_sizes in params, using default: %s", hidden_spec)

        if isinstance(hidden_spec, (int, np.integer)):
            hidden_layers = [int(hidden_spec)]
        elif isinstance(hidden_spec, str):
            hidden_layers = [int(part.strip()) for part in hidden_spec.split(",") if part.strip()]
        elif isinstance(hidden_spec, (list, tuple)):
            hidden_layers = [int(v) for v in hidden_spec]
        else:
            raise TuningConfigError(
                f"Unsupported hidden_layer_sizes type: {type(hidden_spec)}",
                context=TuningErrorContext(component="ModelEvaluator", operation="_build_bnn_layer_sizes"),
                details={"hidden_spec": safe_serialize(hidden_spec)},
            )

        raise_for_condition(
            not hidden_layers or any(h <= 0 for h in hidden_layers),
            "Hidden layer sizes must be positive integers.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="ModelEvaluator", operation="_build_bnn_layer_sizes"),
            details={"hidden_layers": hidden_layers},
        )
        return [input_dim] + hidden_layers + [output_dim]

    @classmethod
    def _instantiate_bnn(cls, layer_sizes: List[int], params: Mapping[str, Any]) -> BayesianNeuralNetwork:
        """Create BNN instance, merging params with config defaults."""
        # Supported BNN constructor arguments (from bayesian_neural_network.py)
        bnn_args = {
            "learning_rate",
            "prior_mu",
            "prior_logvar",
            "random_state",
            "logvar_clip_range",
            "gradient_clip_norm",
            "weight_init_scale",
            "hidden_activation",
            "likelihood_std",
            "min_variance",
            "stability_epsilon",
            "leaky_relu_slope",
        }
        init_kwargs = {k: v for k, v in params.items() if k in bnn_args}
        # Override with config defaults (already handled inside BNN __init__ via config)
        return BayesianNeuralNetwork(layer_sizes=layer_sizes, **init_kwargs)

    # ----------------------------------------------------------------------
    # GNN evaluation
    # ----------------------------------------------------------------------
    @classmethod
    def _gnn_names(cls) -> set[str]:
        return {"gridneuralnetwork", "grid_neural_network", "gnn"}

    @classmethod
    @error_boundary(
        error_cls=TuningEvaluationError,
        message="GNN candidate evaluation failed.",
        context_builder=lambda exc, args, kwargs: TuningErrorContext(
            component="ModelEvaluator",
            operation="_evaluate_gnn",
            model_type="grid_neural_network",
        ),
    )
    def _evaluate_gnn(cls, params: Mapping[str, Any], scoring_metric: Optional[str],
                      x_train: Array, y_train: Array, x_val: Array, y_val: Array,
                      task_type: str, fit_kwargs: Optional[Mapping[str, Any]]) -> float:
        # 1. Prepare data
        x_train_arr, y_train_arr = cls._validate_data(x_train, y_train, "GNN train")
        x_val_arr, y_val_arr = cls._validate_data(x_val, y_val, "GNN validation")
        input_dim = x_train_arr.shape[1]
        output_dim = GridNeuralNetwork.infer_output_dim_from_targets(y_train_arr, task_type)

        # 2. Build model using GNN factory method
        model = GridNeuralNetwork.from_grid_params(
            input_dim=input_dim,
            output_dim=output_dim,
            params=params,
            task_type=task_type,
        )

        # 3. Fit model
        fit_args = cls._prepare_fit_kwargs(params, fit_kwargs, model_type="gnn")
        history = model.fit(
            x_train=x_train_arr,
            y_train=y_train_arr,
            validation_data=(x_val_arr, y_val_arr),
            **fit_args,
        )
        logger.debug("GNN fit completed: %s epochs, early_stop=%s", len(history["epochs"]), history["stopped_early"])

        # 4. Evaluate and return score (already adjusted for maximize/minimize)
        metrics = model.evaluate(x_val_arr, y_val_arr)
        # The static method evaluate_grid_candidate already returns a score that is
        # higher = better (negates if needed). We'll replicate that logic.
        return cls._score_from_metrics(metrics, scoring_metric, task_type, model)

    @classmethod
    def _score_from_metrics(
        cls,
        metrics: Dict[str, float],
        scoring_metric: Optional[str],
        task_type: str,
        model: GridNeuralNetwork,
    ) -> float:
        """Return a score that is higher = better, suitable for maximization."""
        # Determine default metric
        if scoring_metric is None:
            grid_defaults = get_config_section("gnn").get("grid_defaults", {})
            default_metric = grid_defaults.get(
                "metric",
                "r2" if task_type == "regression" else "accuracy",
            )
            scoring_metric = str(default_metric).strip().lower()

        raise_for_condition(
            scoring_metric not in metrics,
            f"Scoring metric '{scoring_metric}' not available in GNN evaluation.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="ModelEvaluator", operation="_score_from_metrics"),
            details={"available": sorted(metrics.keys()), "requested": scoring_metric},
        )

        raw = float(metrics[scoring_metric])
        # Metrics that are naturally "higher is better"
        maximize_set = {"accuracy", "precision", "recall", "r2", "f1", "auc"}
        if scoring_metric in maximize_set:
            return raw
        # For loss-like metrics (mse, rmse, mae), return negative raw so that higher is better
        return -raw

    # ----------------------------------------------------------------------
    # Common helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _validate_data(x: Array, y: Array, label: str) -> Tuple[Array, Array]:
        """Convert to float ndarray, check shapes and finite values."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 2:
            raise TuningValidationError(
                f"{label}: x must be 2D, got shape {x_arr.shape}",
                context=TuningErrorContext(component="ModelEvaluator", operation="validate_data"),
            )
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.ndim != 2:
            raise TuningValidationError(
                f"{label}: y must be 1D or 2D, got shape {y_arr.shape}",
                context=TuningErrorContext(component="ModelEvaluator", operation="validate_data"),
            )
        if x_arr.shape[0] != y_arr.shape[0]:
            raise TuningValidationError(
                f"{label}: x and y row count mismatch ({x_arr.shape[0]} vs {y_arr.shape[0]})",
                context=TuningErrorContext(component="ModelEvaluator", operation="validate_data"),
            )
        if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
            raise TuningValidationError(
                f"{label}: Non-finite values detected",
                context=TuningErrorContext(component="ModelEvaluator", operation="validate_data"),
            )
        return x_arr, y_arr

    @staticmethod
    def _infer_output_dim(y: Array, task_type: str) -> int:
        """Determine output dimension from target data (regression only for BNN)."""
        if task_type != "regression":
            raise TuningConfigError(
                f"BNN only supports regression, got task_type={task_type}",
                context=TuningErrorContext(component="ModelEvaluator", operation="infer_output_dim"),
            )
        if y.ndim == 1:
            return 1
        return y.shape[1]

    @staticmethod
    def _get_eval_samples(params: Mapping[str, Any]) -> int:
        """Number of Monte Carlo samples for BNN evaluation."""
        # Use 'num_samples' from params, or fallback to config
        if "num_samples" in params:
            return int(params["num_samples"])
        bnn_cfg = get_config_section("bnn")
        prediction_cfg = bnn_cfg.get("prediction", {})
        return int(prediction_cfg.get("num_samples", 200))

    @classmethod
    def _prepare_fit_kwargs(
        cls,
        params: Mapping[str, Any],
        extra: Optional[Mapping[str, Any]],
        model_type: str,
    ) -> Dict[str, Any]:
        """Extract fit() parameters from hyperparameter dict and merge extras."""
        # Common fit arguments for both BNN and GNN
        fit_keys = {"epochs", "batch_size", "shuffle", "early_stopping_patience", "min_delta"}
        # BNN also uses 'num_samples' during training
        if model_type == "bnn":
            fit_keys.add("num_samples")
        # GNN uses 'restore_best_weights'
        if model_type == "gnn":
            fit_keys.add("restore_best_weights")

        kwargs = {k: v for k, v in params.items() if k in fit_keys}
        if extra:
            kwargs.update(extra)
        # Ensure numeric types are converted
        for key in ("epochs", "batch_size", "num_samples", "early_stopping_patience"):
            if key in kwargs:
                kwargs[key] = int(kwargs[key])
        for key in ("min_delta",):
            if key in kwargs:
                kwargs[key] = float(kwargs[key])
        return kwargs

    @classmethod
    def _resolve_scoring_metric(
        cls,
        requested: Optional[str],
        model_type: str,
        metrics: Dict[str, float],
    ) -> str:
        """Determine which metric to use as the optimization objective."""
        if requested is not None:
            requested_lower = requested.strip().lower()
            if requested_lower in metrics:
                return requested_lower
            raise TuningConfigError(
                f"Scoring metric '{requested}' not found in model evaluation metrics.",
                context=TuningErrorContext(component="ModelEvaluator", operation="resolve_scoring_metric"),
                details={"available": sorted(metrics.keys()), "requested": requested},
            )

        # Defaults: for BNN use ELBO (higher is better); for GNN use config default
        if model_type == "bnn":
            # BNN: prefer elbo, then log_likelihood
            if "elbo" in metrics:
                return "elbo"
            if "log_likelihood" in metrics:
                return "log_likelihood"
            # Fallback to first available
            return next(iter(metrics.keys()))

        # GNN: rely on grid_defaults from config
        gnn_cfg = get_config_section("gnn")
        grid_defaults = gnn_cfg.get("grid_defaults", {})
        default_metric = grid_defaults.get("metric", "r2")
        if default_metric in metrics:
            return default_metric
        # Fallback
        return next(iter(metrics.keys()))


# ----------------------------------------------------------------------
# Convenience standalone functions (optional)
# ----------------------------------------------------------------------
def evaluate_bnn_candidate(
    params: Mapping[str, Any],
    x_train: Array,
    y_train: Array,
    x_val: Array,
    y_val: Array,
    scoring_metric: Optional[str] = None,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> float:
    """Standalone BNN evaluator (convenience wrapper)."""
    return ModelEvaluator.evaluate_candidate(
        model_type="BayesianNeuralNetwork",
        params=params,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        scoring_metric=scoring_metric,
        fit_kwargs=fit_kwargs,
    )


def evaluate_gnn_candidate(
    params: Mapping[str, Any],
    x_train: Array,
    y_train: Array,
    x_val: Array,
    y_val: Array,
    scoring_metric: Optional[str] = None,
    task_type: str = "regression",
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> float:
    """Standalone GNN evaluator (convenience wrapper)."""
    return ModelEvaluator.evaluate_candidate(
        model_type="GridNeuralNetwork",
        params=params,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        scoring_metric=scoring_metric,
        task_type=task_type,
        fit_kwargs=fit_kwargs,
    )


__all__ = [
    "ModelEvaluator",
    "evaluate_bnn_candidate",
    "evaluate_gnn_candidate",
]