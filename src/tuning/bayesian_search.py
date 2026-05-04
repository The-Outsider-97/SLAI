"""Production-ready Bayesian hyperparameter search with optional native BNN integration."""

from __future__ import annotations

import inspect
import json
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import OptimizeResult, use_named_args

from .utils.evaluator import evaluate_bnn_candidate
from .utils.config_loader import get_config_section, load_global_config
from .utils.tuning_error import (TuningConfigError, safe_serialize, wrap_exception,
                                 TuningOptimizationError, raise_for_condition,
                                 TuningErrorContext, TuningStrategyError,
                                 TuningEvaluationError, TuningPersistenceError,
                                 TuningReportingError, TuningSearchSpaceError,
                                 TuningValidationError)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Bayesian Search")
printer = PrettyPrinter

Dimension = Union[Integer, Real, Categorical]


@dataclass(frozen=True)
class BayesianSearchSettings:
    """Validated runtime settings for Bayesian search."""

    n_calls: int
    n_initial_points: int
    random_state: Optional[int]
    output_dir: Path
    model_type: str
    objective: str
    scoring_metric: Optional[str]
    validation_fraction: float
    shuffle_validation: bool
    plot_results: bool


class BayesianSearch:
    """Bayesian optimization wrapper with validation, logging, persistence, and optional native BNN support."""

    BUILTIN_BNN_MODEL_NAMES = frozenset({"bayesianneuralnetwork", "bayesian_neural_network", "bnn"})
    SUPPORTED_OBJECTIVES = frozenset({"auto", "minimize", "maximize"})
    BNN_MAXIMIZE_METRICS = frozenset({"elbo", "log_likelihood"})
    BNN_MINIMIZE_METRICS = frozenset({"mse", "rmse", "kl_divergence", "mean_predictive_std"})

    def __init__(
        self,
        evaluation_function: Optional[Callable[..., float]] = None,
        model_type: Optional[str] = None,
        objective: str = "auto",
    ) -> None:
        self.config: Dict[str, Any] = load_global_config() or {}
        self.settings = self._load_settings(model_type=model_type, objective=objective)
        self.search_space_config, self.dimensions = self._load_search_space(self.settings.model_type)

        raise_for_condition(
            not self.dimensions,
            "Bayesian search requires at least one hyperparameter dimension.",
            error_cls=TuningSearchSpaceError,
            context=self._context("__init__"),
            details={"model_type": self.settings.model_type},
        )

        self.param_names = [dimension.name for dimension in self.dimensions]
        self.evaluation_function = evaluation_function
        self.use_builtin_bnn_evaluator = self._is_builtin_bnn_model(self.settings.model_type) and evaluation_function is None

        raise_for_condition(
            not self.use_builtin_bnn_evaluator and evaluation_function is None,
            "evaluation_function is required unless model_type is a built-in BayesianNeuralNetwork.",
            error_cls=TuningStrategyError,
            context=self._context("__init__"),
            details={"model_type": self.settings.model_type},
        )

        self.optimization_history: List[Dict[str, Any]] = []
        self.best_score_so_far = np.inf if self.settings.objective == "minimize" else -np.inf
        self.best_params_so_far: Optional[Dict[str, Any]] = None
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "BayesianSearch initialized with model=%s, objective=%s, n_calls=%s, builtin_bnn=%s",
            self.settings.model_type,
            self.settings.objective,
            self.settings.n_calls,
            self.use_builtin_bnn_evaluator,
        )

    @classmethod
    def _normalize_model_type(cls, value: Optional[str]) -> str:
        return str(value or "").strip().replace(" ", "").replace("-", "_").lower()

    @classmethod
    def _is_builtin_bnn_model(cls, model_type: str) -> bool:
        normalized = cls._normalize_model_type(model_type)
        return normalized in cls.BUILTIN_BNN_MODEL_NAMES

    def _context(self, operation: str, **kwargs: Any) -> TuningErrorContext:
        return TuningErrorContext(
            component="BayesianSearch",
            operation=operation,
            strategy="bayesian_search",
            model_type=self.settings.model_type if hasattr(self, "settings") else None,
            random_state=self.settings.random_state if hasattr(self, "settings") else None,
            config_path=str(self.config.get("__config_path__", "")) or None,
            output_path=str(self.settings.output_dir) if hasattr(self, "settings") else None,
            parameters={key: value for key, value in kwargs.items() if value is not None},
        )

    def requires_dataset(self) -> bool:
        return self.use_builtin_bnn_evaluator

    def _resolve_objective(self, requested_objective: str, *, model_type: str, scoring_metric: Optional[str]) -> str:
        normalized = str(requested_objective or "auto").strip().lower()
        raise_for_condition(
            normalized not in self.SUPPORTED_OBJECTIVES,
            "objective must be one of: auto, minimize, maximize.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(
                component="BayesianSearch",
                operation="resolve_objective",
                strategy="bayesian_search",
                model_type=model_type,
            ),
            details={"objective": requested_objective},
        )
        if normalized in {"minimize", "maximize"}:
            return normalized

        metric = str(scoring_metric or "").strip().lower()
        if self._is_builtin_bnn_model(model_type):
            if metric in self.BNN_MINIMIZE_METRICS:
                return "minimize"
            return "maximize"
        return "minimize"

    def _load_settings(self, model_type: Optional[str], objective: str) -> BayesianSearchSettings:
        section = get_config_section("bayesian_search") or {}
        configured_model = model_type or section.get("model_type") or "GradientBoosting"

        n_calls = int(section.get("n_calls", 20))
        n_initial_points = int(section.get("n_initial_points", 5))
        random_state = section.get("random_state")
        output_dir = Path(section.get("output_dir", "src/tuning/reports/bayesian_search"))
        scoring_metric = section.get("scoring_metric")
        validation_fraction = float(section.get("validation_fraction", 0.2))
        shuffle_validation = bool(section.get("shuffle_validation", True))
        plot_results = bool(section.get("plot_results", True))

        resolved_objective = self._resolve_objective(
            objective if objective != "auto" else str(section.get("objective", "auto")),
            model_type=configured_model,
            scoring_metric=str(scoring_metric) if scoring_metric is not None else None,
        )

        raise_for_condition(
            n_calls < 2,
            "n_calls must be >= 2.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="BayesianSearch", operation="load_settings", strategy="bayesian_search", model_type=configured_model),
            details={"n_calls": n_calls},
        )
        raise_for_condition(
            n_initial_points < 1,
            "n_initial_points must be >= 1.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="BayesianSearch", operation="load_settings", strategy="bayesian_search", model_type=configured_model),
            details={"n_initial_points": n_initial_points},
        )
        raise_for_condition(
            n_initial_points >= n_calls,
            "n_initial_points must be less than n_calls.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="BayesianSearch", operation="load_settings", strategy="bayesian_search", model_type=configured_model),
            details={"n_initial_points": n_initial_points, "n_calls": n_calls},
        )
        raise_for_condition(
            not 0.0 < validation_fraction < 0.5,
            "validation_fraction must be in the range (0.0, 0.5).",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="BayesianSearch", operation="load_settings", strategy="bayesian_search", model_type=configured_model),
            details={"validation_fraction": validation_fraction},
        )

        return BayesianSearchSettings(
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            output_dir=output_dir,
            model_type=configured_model,
            objective=resolved_objective,
            scoring_metric=None if scoring_metric is None else str(scoring_metric).strip().lower(),
            validation_fraction=validation_fraction,
            shuffle_validation=shuffle_validation,
            plot_results=plot_results,
        )

    def _load_search_space(self, model_type: str) -> Tuple[List[Dict[str, Any]], List[Dimension]]:
        hp_root = self.config.get("hyperparameters")
        raise_for_condition(
            not isinstance(hp_root, dict),
            "Expected 'hyperparameters' to be a model-keyed dictionary.",
            error_cls=TuningSearchSpaceError,
            context=self._context("load_search_space"),
            details={"hyperparameters_type": type(hp_root).__name__},
        )

        model_specs = next((value for key, value in hp_root.items() if str(key).lower() == str(model_type).lower()), None)
        raise_for_condition(
            not isinstance(model_specs, list),
            f"No hyperparameter list found for model_type='{model_type}'.",
            error_cls=TuningSearchSpaceError,
            context=self._context("load_search_space"),
            details={"model_type": model_type},
        )

        dimensions: List[Dimension] = []
        seen_names: set[str] = set()

        for spec in model_specs:
            raise_for_condition(
                not isinstance(spec, dict),
                "Invalid hyperparameter specification.",
                error_cls=TuningSearchSpaceError,
                context=self._context("load_search_space"),
                details={"spec": safe_serialize(spec)},
            )
            name = str(spec.get("name", "")).strip()
            raise_for_condition(
                not name,
                "Missing hyperparameter name in specification.",
                error_cls=TuningSearchSpaceError,
                context=self._context("load_search_space"),
                details={"spec": safe_serialize(spec)},
            )
            raise_for_condition(
                name in seen_names,
                f"Duplicate hyperparameter name detected: {name}",
                error_cls=TuningSearchSpaceError,
                context=self._context("load_search_space"),
                details={"name": name},
            )
            seen_names.add(name)

            raw_type = str(spec.get("type", "")).strip().lower()
            param_type = {"int": "integer", "float": "real"}.get(raw_type, raw_type)
            dimensions.append(self._build_dimension(name=name, param_type=param_type, spec=spec))

        return model_specs, dimensions

    def _build_dimension(self, name: str, param_type: str, spec: Dict[str, Any]) -> Dimension:
        if param_type == "integer":
            low, high = self._resolve_bounds(spec, cast_type=int)
            return Integer(low=low, high=high, name=name)

        if param_type == "real":
            low, high = self._resolve_bounds(spec, cast_type=float)
            prior = str(spec.get("prior", "uniform")).lower()
            raise_for_condition(
                prior not in {"uniform", "log-uniform"},
                f"Unsupported prior '{prior}' for real parameter '{name}'.",
                error_cls=TuningSearchSpaceError,
                context=self._context("build_dimension"),
                details={"name": name, "prior": prior},
            )
            raise_for_condition(
                prior == "log-uniform" and (low <= 0 or high <= 0),
                f"log-uniform prior requires positive bounds for '{name}'.",
                error_cls=TuningSearchSpaceError,
                context=self._context("build_dimension"),
                details={"name": name, "low": low, "high": high},
            )
            return Real(low=low, high=high, prior=prior, name=name)

        if param_type == "categorical":
            values = spec.get("values")
            raise_for_condition(
                not isinstance(values, list) or not values,
                f"Categorical parameter '{name}' requires a non-empty list of values.",
                error_cls=TuningSearchSpaceError,
                context=self._context("build_dimension"),
                details={"name": name, "values": safe_serialize(values)},
            )
            # Convert unhashable lists to tuples
            def make_hashable(v):
                if isinstance(v, list):
                    return tuple(v)
                return v
            hashable_values = [make_hashable(v) for v in values]
            return Categorical(categories=hashable_values, name=name)

        raise TuningSearchSpaceError(
            f"Unsupported parameter type '{param_type}' for '{name}'.",
            context=self._context("build_dimension"),
            details={"name": name, "param_type": param_type},
        )

    @staticmethod
    def _resolve_bounds(spec: Dict[str, Any], cast_type: Callable[[Any], Any]) -> Tuple[Any, Any]:
        if "min" in spec and "max" in spec:
            low, high = cast_type(spec["min"]), cast_type(spec["max"])
        elif "values" in spec:
            values = [value for value in spec["values"] if value is not None]
            if not values:
                raise ValueError("Parameter 'values' cannot be empty.")
            casted = [cast_type(value) for value in values]
            low, high = min(casted), max(casted)
        else:
            raise ValueError("Numeric parameter requires either min/max or values.")
        if low > high:
            raise ValueError(f"Invalid bounds: min ({low}) > max ({high}).")
        return low, high

    def _score_to_minimize(self, raw_score: float) -> float:
        return raw_score if self.settings.objective == "minimize" else -raw_score

    def _validate_dataset(self, x_data: Any, y_data: Any, *, operation: str) -> Tuple[np.ndarray, np.ndarray]:
        x_array = np.asarray(x_data, dtype=float)
        y_array = np.asarray(y_data, dtype=float)

        raise_for_condition(
            x_array.ndim != 2,
            "X_data must be a 2D array.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"x_shape": list(x_array.shape)},
        )
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        raise_for_condition(
            y_array.ndim != 2,
            "y_data must be a 1D or 2D array.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"y_shape": list(y_array.shape)},
        )
        raise_for_condition(
            x_array.shape[0] != y_array.shape[0],
            "X_data and y_data must have matching sample counts.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"x_rows": int(x_array.shape[0]), "y_rows": int(y_array.shape[0])},
        )
        raise_for_condition(
            x_array.shape[0] < 3,
            "Bayesian search requires at least 3 samples when using a validation split.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"num_rows": int(x_array.shape[0])},
        )
        raise_for_condition(
            not np.isfinite(x_array).all() or not np.isfinite(y_array).all(),
            "Input data contains non-finite values.",
            error_cls=TuningValidationError,
            context=self._context(operation),
            details={"x_shape": list(x_array.shape), "y_shape": list(y_array.shape)},
        )
        return x_array, y_array

    def _split_train_validation(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_rows = x_data.shape[0]
        validation_count = max(1, min(num_rows - 1, int(round(num_rows * self.settings.validation_fraction))))
        indices = np.arange(num_rows)

        if self.settings.shuffle_validation:
            rng = np.random.default_rng(self.settings.random_state)
            rng.shuffle(indices)

        validation_indices = indices[:validation_count]
        train_indices = indices[validation_count:]

        raise_for_condition(
            train_indices.size == 0 or validation_indices.size == 0,
            "Validation split produced an empty training or validation partition.",
            error_cls=TuningValidationError,
            context=self._context("split_train_validation"),
            details={
                "num_rows": num_rows,
                "validation_fraction": self.settings.validation_fraction,
                "validation_count": int(validation_count),
            },
        )
        return (
            x_data[train_indices],
            y_data[train_indices],
            x_data[validation_indices],
            y_data[validation_indices],
        )

    def _invoke_external_evaluation(
        self,
        params: Dict[str, Any],
        *,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> float:
        if self.evaluation_function is None:
            raise TuningStrategyError(
                "No external evaluation function is configured.",
                context=self._context("invoke_external_evaluation"),
            )

        signature = inspect.signature(self.evaluation_function)
        parameters = list(signature.parameters.values())
        accepts_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
        positional_capacity = sum(
            parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            for parameter in parameters
        )

        if x_train is not None and y_train is not None and x_val is not None and y_val is not None:
            if accepts_varargs or positional_capacity >= 5:
                return float(self.evaluation_function(params, x_train, y_train, x_val, y_val))

        raise_for_condition(
            positional_capacity < 1 and not accepts_varargs,
            "evaluation_function must accept at least the params argument.",
            error_cls=TuningStrategyError,
            context=self._context("invoke_external_evaluation"),
            details={"signature": str(signature)},
        )
        return float(self.evaluation_function(params))

    def _evaluate_candidate(
        self,
        params: Dict[str, Any],
        *,
        x_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> float:
        if self.use_builtin_bnn_evaluator:
            raise_for_condition(
                x_train is None or y_train is None or x_val is None or y_val is None,
                "Built-in BayesianNeuralNetwork evaluation requires train and validation data.",
                error_cls=TuningValidationError,
                context=self._context("evaluate_candidate"),
            )
            metric_name = self.settings.scoring_metric or "elbo"
            return evaluate_bnn_candidate(
                params=params,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                scoring_metric=metric_name,
                fit_kwargs=None,   # optional, could be passed from config
            )

        return self._invoke_external_evaluation(
            params,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

    def run_search(
        self,
        X_data: Optional[Any] = None,
        y_data: Optional[Any] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float, Optional[OptimizeResult]]:
        logger.info("Starting Bayesian search with %s dimensions.", len(self.dimensions))

        x_train: Optional[np.ndarray] = None
        y_train: Optional[np.ndarray] = None
        x_val: Optional[np.ndarray] = None
        y_val: Optional[np.ndarray] = None

        if self.use_builtin_bnn_evaluator or (X_data is not None and y_data is not None):
            x_array, y_array = self._validate_dataset(X_data, y_data, operation="run_search_dataset")
            x_train, y_train, x_val, y_val = self._split_train_validation(x_array, y_array)

        @use_named_args(self.dimensions)
        def objective_function(**params: Any) -> float:
            normalized_params = dict(params)
            try:
                raw_score = self._evaluate_candidate(
                    normalized_params,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                )
                raise_for_condition(
                    not np.isfinite(raw_score),
                    "Evaluation function returned a non-finite score.",
                    error_cls=TuningEvaluationError,
                    context=self._context("objective_function", parameters=normalized_params),
                    details={"raw_score": raw_score},
                )
                objective_value = self._score_to_minimize(float(raw_score))
                self._record_iteration(
                    params=normalized_params,
                    raw_score=float(raw_score),
                    objective_value=float(objective_value),
                    failed=False,
                )
                return float(objective_value)
            except Exception as exc:  # noqa: BLE001
                wrapped = wrap_exception(
                    exc,
                    message="Bayesian search evaluation failed for a candidate parameter set.",
                    error_cls=TuningEvaluationError,
                    context=self._context("objective_function", parameters=normalized_params),
                    details={"params": safe_serialize(normalized_params)},
                )
                logger.error("Evaluation failed for params=%s: %s", normalized_params, wrapped, exc_info=True)
                self._record_iteration(
                    params=normalized_params,
                    raw_score=float("nan"),
                    objective_value=float("inf"),
                    failed=True,
                    error=wrapped.to_log_record(include_traceback=False),
                )
                return float("inf")

        result: Optional[OptimizeResult] = None
        try:
            result = gp_minimize(
                func=objective_function,
                dimensions=self.dimensions,
                n_calls=self.settings.n_calls,
                n_initial_points=self.settings.n_initial_points,
                random_state=self.settings.random_state,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            wrapped = wrap_exception(
                exc,
                message="Bayesian optimization run failed.",
                error_cls=TuningOptimizationError,
                context=self._context("run_search"),
            )
            logger.error("%s", wrapped, exc_info=True)

        if result is not None:
            best_params = dict(zip(self.param_names, result.x))
            best_score = float(result.fun if self.settings.objective == "minimize" else -result.fun)
        else:
            best_params = self.best_params_so_far
            best_score = float(self.best_score_so_far) if self.best_params_so_far is not None else float("nan")

        self._persist_results(best_params=best_params, best_score=best_score, result=result, x_train=x_train, x_val=x_val)
        if self.settings.plot_results:
            self._plot_optimization_progress()
        return best_params, best_score, result

    def _record_iteration(
        self,
        *,
        params: Dict[str, Any],
        raw_score: float,
        objective_value: float,
        failed: bool = False,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        iteration = len(self.optimization_history) + 1
        entry: Dict[str, Any] = {
            "iteration": iteration,
            "parameters": safe_serialize(params),
            "raw_score": raw_score,
            "objective_value": objective_value,
            "failed": failed,
        }
        if error is not None:
            entry["error"] = safe_serialize(error)
        self.optimization_history.append(entry)

        if failed or not np.isfinite(raw_score):
            return

        is_better = (
            raw_score < self.best_score_so_far
            if self.settings.objective == "minimize"
            else raw_score > self.best_score_so_far
        )
        if is_better:
            self.best_score_so_far = raw_score
            self.best_params_so_far = dict(params)

    def _persist_results(
        self,
        *,
        best_params: Optional[Dict[str, Any]],
        best_score: float,
        result: Optional[OptimizeResult],
        x_train: Optional[np.ndarray],
        x_val: Optional[np.ndarray],
    ) -> None:
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            summary: Dict[str, Any] = {
                "created_at_utc": timestamp,
                "search_configuration": {
                    "model_type": self.settings.model_type,
                    "objective": self.settings.objective,
                    "scoring_metric": self.settings.scoring_metric,
                    "n_calls": self.settings.n_calls,
                    "n_initial_points": self.settings.n_initial_points,
                    "random_state": self.settings.random_state,
                    "validation_fraction": self.settings.validation_fraction,
                    "shuffle_validation": self.settings.shuffle_validation,
                    "uses_builtin_bnn_evaluator": self.use_builtin_bnn_evaluator,
                    "search_space_definition": self.search_space_config,
                },
                "dataset_summary": {
                    "train_shape": None if x_train is None else list(x_train.shape),
                    "validation_shape": None if x_val is None else list(x_val.shape),
                },
                "best_result": {
                    "parameters": safe_serialize(best_params),
                    "score": best_score,
                },
                "optimization_history": safe_serialize(self.optimization_history),
            }

            if result is not None:
                summary["optimizer_state"] = {
                    "x_iters": safe_serialize(result.x_iters),
                    "func_vals": safe_serialize(result.func_vals.tolist()),
                    "space": str(result.space),
                }

            output_path = self.settings.output_dir / f"bayesian_search_summary_{self.settings.model_type}_{timestamp}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
            logger.info("Saved Bayesian search summary to %s", output_path)
        except Exception as exc:  # noqa: BLE001
            wrapped = wrap_exception(
                exc,
                message="Failed to persist Bayesian search results.",
                error_cls=TuningPersistenceError,
                context=self._context("persist_results"),
            )
            logger.error("%s", wrapped, exc_info=True)

    def _plot_optimization_progress(self) -> None:
        try:
            if not self.optimization_history:
                return

            iterations = [item["iteration"] for item in self.optimization_history]
            scores = np.array([item["raw_score"] for item in self.optimization_history], dtype=float)
            finite_mask = np.isfinite(scores)
            if not finite_mask.any():
                logger.warning("No finite scores available for plotting.")
                return

            x_values = np.array(iterations)[finite_mask]
            y_values = scores[finite_mask]
            best_trace = (
                np.minimum.accumulate(y_values)
                if self.settings.objective == "minimize"
                else np.maximum.accumulate(y_values)
            )

            figure = plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, marker="o", linewidth=1.2, label="Score")
            plt.plot(x_values, best_trace, linestyle="--", label="Best so far")
            plt.title(f"Bayesian Optimization Progress ({self.settings.model_type})")
            plt.xlabel("Iteration")
            plt.ylabel("Score")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            plt.tight_layout()

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            plot_path = self.settings.output_dir / f"bayesian_progress_{self.settings.model_type}_{timestamp}.png"
            figure.savefig(plot_path)
            plt.close(figure)
            logger.info("Saved Bayesian optimization plot to %s", plot_path)
        except Exception as exc:  # noqa: BLE001
            wrapped = wrap_exception(
                exc,
                message="Failed to generate Bayesian optimization plot.",
                error_cls=TuningReportingError,
                context=self._context("plot_optimization_progress"),
            )
            logger.error("%s", wrapped, exc_info=True)


if __name__ == "__main__":
    rng = np.random.default_rng(17)

    x_demo = rng.normal(size=(48, 4))
    y_demo = (x_demo[:, :1] * 0.7 + rng.normal(0.0, 0.1, size=(48, 1))).astype(float)

    search = BayesianSearch(evaluation_function=evaluate_bnn_candidate, model_type="BayesianNeuralNetwork", objective="auto")
    params, score, _ = search.run_search(x_demo, y_demo)
    print(f"best params={params}, score={score}")
