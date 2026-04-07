"""Production-ready exhaustive grid search with optional native GNN evaluator integration."""

from __future__ import annotations

import inspect
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, StratifiedKFold

from .utils.config_loader import get_config_section, load_global_config
from .utils.evaluator import evaluate_gnn_candidate
from .utils.tuning_error import (TuningConfigError, TuningError, TuningErrorContext,
                                 TuningEvaluationError, TuningOptimizationError, wrap_exception,
                                 TuningPersistenceError, TuningReportingError, safe_serialize,
                                 TuningSearchSpaceError, TuningStrategyError, raise_for_condition,
                                 TuningValidationError)
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Grid Search")
printer = PrettyPrinter

SUPPORTED_TASK_TYPES = frozenset(
    {"auto", "regression", "classification", "binary_classification", "multiclass_classification"}
)


@dataclass(frozen=True)
class GridSearchSettings:
    """Validated runtime settings for grid search."""

    n_jobs: int
    cross_val_folds: int
    random_state: int
    output_dir: Path
    model_type: str
    scoring_metric: Optional[str]
    task_type: str
    plot_results: bool


class GridSearch:
    """Deterministic grid search with validation, reporting, and optional native GNN support."""

    BUILTIN_GNN_MODEL_NAMES = frozenset({"gridneuralnetwork", "grid_neural_network", "gnn"})

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        evaluation_function: Optional[Callable[..., float]] = None,
        model_type: Optional[str] = None,
    ) -> None:
        self.config: Dict[str, Any] = config or load_global_config() or {}
        self.settings = self._load_settings(model_type=model_type)
        self.param_names, self.hyperparam_space = self._load_search_space(self.settings.model_type)
        self._validate_search_space()

        self.evaluation_function = evaluation_function
        self.use_builtin_gnn_evaluator = self._is_builtin_gnn_model(self.settings.model_type) and evaluation_function is None
        raise_for_condition(
            not self.use_builtin_gnn_evaluator and evaluation_function is None,
            "evaluation_function is required unless model_type is a built-in GridNeuralNetwork.",
            error_cls=TuningStrategyError,
            context=self._context("__init__"),
            details={"model_type": self.settings.model_type},
        )

        self.results: List[Dict[str, Any]] = []
        self.best_score = -np.inf
        self.best_score_std = np.nan
        self.best_params: Optional[Dict[str, Any]] = None

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "GridSearch initialized with model=%s, combinations=%s, builtin_gnn=%s",
            self.settings.model_type,
            self._combination_count(),
            self.use_builtin_gnn_evaluator,
        )

    @classmethod
    def _normalize_model_type(cls, value: Optional[str]) -> str:
        return str(value or "").strip().replace(" ", "").replace("-", "_").lower()

    @classmethod
    def _is_builtin_gnn_model(cls, model_type: str) -> bool:
        return cls._normalize_model_type(model_type) in cls.BUILTIN_GNN_MODEL_NAMES

    def _context(self, operation: str, **kwargs: Any) -> TuningErrorContext:
        return TuningErrorContext(
            component="GridSearch",
            operation=operation,
            strategy="grid_search",
            model_type=self.settings.model_type if hasattr(self, "settings") else None,
            random_state=self.settings.random_state if hasattr(self, "settings") else None,
            config_path=str(self.config.get("__config_path__", "")) or None,
            output_path=str(self.settings.output_dir) if hasattr(self, "settings") else None,
            parameters={key: value for key, value in kwargs.items() if value is not None},
        )

    def requires_dataset(self) -> bool:
        return True

    def _load_settings(self, model_type: Optional[str]) -> GridSearchSettings:
        section = get_config_section("grid_search") or {}
        gnn_section = get_config_section("gnn") or {}
        configured_model = model_type or section.get("model_type") or "GradientBoosting"

        n_jobs = int(section.get("n_jobs", -1))
        folds = int(section.get("cross_val_folds", 5))
        random_state = int(section.get("random_state", 42))
        output_dir = Path(section.get("output_dir", "src/tuning/reports/grid_search"))
        scoring_metric = section.get("scoring_metric")
        task_type = str(section.get("task_type", gnn_section.get("task_type", "regression"))).strip().lower()
        plot_results = bool(section.get("plot_results", True))

        raise_for_condition(
            n_jobs == 0,
            "n_jobs cannot be 0.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="GridSearch", operation="load_settings", strategy="grid_search", model_type=configured_model),
            details={"n_jobs": n_jobs},
        )
        raise_for_condition(
            folds < 2,
            "cross_val_folds must be >= 2.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="GridSearch", operation="load_settings", strategy="grid_search", model_type=configured_model),
            details={"cross_val_folds": folds},
        )
        raise_for_condition(
            task_type not in SUPPORTED_TASK_TYPES,
            "task_type must be one of: auto, regression, classification, binary_classification, multiclass_classification.",
            error_cls=TuningConfigError,
            context=TuningErrorContext(component="GridSearch", operation="load_settings", strategy="grid_search", model_type=configured_model),
            details={"task_type": task_type},
        )

        normalized_metric = None if scoring_metric is None else str(scoring_metric).strip().lower()
        return GridSearchSettings(
            n_jobs=n_jobs,
            cross_val_folds=folds,
            random_state=random_state,
            output_dir=output_dir,
            model_type=configured_model,
            scoring_metric=normalized_metric,
            task_type=task_type,
            plot_results=plot_results,
        )

    def _load_search_space(self, model_type: str) -> Tuple[List[str], List[List[Any]]]:
        hp_root = self.config.get("hyperparameters")

        if isinstance(hp_root, dict):
            param_specs = next((value for key, value in hp_root.items() if str(key).lower() == str(model_type).lower()), None)
            raise_for_condition(
                param_specs is None,
                f"No hyperparameter list found for model_type='{model_type}'.",
                error_cls=TuningSearchSpaceError,
                context=self._context("load_search_space"),
                details={"model_type": model_type},
            )
        elif isinstance(hp_root, list):
            param_specs = hp_root
        else:
            raise TuningSearchSpaceError(
                "'hyperparameters' must be a list or model-keyed dictionary.",
                context=self._context("load_search_space"),
                details={"hyperparameters_type": type(hp_root).__name__},
            )

        raise_for_condition(
            not isinstance(param_specs, list),
            "Hyperparameter configuration must be a list of parameter specifications.",
            error_cls=TuningSearchSpaceError,
            context=self._context("load_search_space"),
            details={"param_specs_type": type(param_specs).__name__},
        )

        param_names: List[str] = []
        space: List[List[Any]] = []
        for spec in param_specs:
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
                name in param_names,
                f"Duplicate hyperparameter name detected: {name}",
                error_cls=TuningSearchSpaceError,
                context=self._context("load_search_space"),
                details={"name": name},
            )

            values = self._normalize_values(spec)
            raise_for_condition(
                not values,
                f"Hyperparameter '{name}' has no candidate values.",
                error_cls=TuningSearchSpaceError,
                context=self._context("load_search_space"),
                details={"spec": safe_serialize(spec)},
            )
            param_names.append(name)
            space.append(values)

        return param_names, space

    def _normalize_values(self, spec: Dict[str, Any]) -> List[Any]:
        if "values" in spec:
            values = [v for v in spec["values"] if v is not None]
            return list(values)

        raw_type = str(spec.get("type", "")).strip().lower()
        # Handle integer ranges
        if raw_type in {"integer", "int"} and "min" in spec and "max" in spec:
            low, high = int(spec["min"]), int(spec["max"])
            raise_for_condition(
                low > high,
                "Invalid integer bounds in specification.",
                error_cls=TuningSearchSpaceError,
                context=self._context("normalize_values"),
                details={"spec": safe_serialize(spec), "low": low, "high": high},
            )
            return list(range(low, high + 1))

        # Handle real (float) ranges – generate a discrete grid
        if raw_type in {"real", "float"} and "min" in spec and "max" in spec:
            prior = str(spec.get("prior", "uniform")).lower()
            if prior == "log-uniform":
                raise TuningSearchSpaceError(
                    "Grid search does not support log-uniform prior for real parameters. "
                    "Provide explicit 'values' or use uniform prior.",
                    context=self._context("normalize_values"),
                    details={"spec": safe_serialize(spec)},
                )
            low, high = float(spec["min"]), float(spec["max"])
            # Default number of grid points – can be made configurable later
            n_points = 5
            step = (high - low) / (n_points - 1)
            values = [low + i * step for i in range(n_points)]
            # Ensure the high value is included exactly
            if not np.isclose(values[-1], high):
                values[-1] = high
            return values

        raise TuningSearchSpaceError(
            "Grid search requires 'values' or min/max bounds for integer/real parameters.",
            context=self._context("normalize_values"),
            details={"spec": safe_serialize(spec)},
        )

    def _validate_search_space(self) -> None:
        combinations = self._combination_count()
        raise_for_condition(
            combinations <= 0,
            "Search space produced zero combinations.",
            error_cls=TuningSearchSpaceError,
            context=self._context("validate_search_space"),
        )
        if combinations > 100_000:
            logger.warning("Large search space detected (%s combinations).", combinations)

    def _combination_count(self) -> int:
        if not self.hyperparam_space:
            return 1
        return int(np.prod([len(values) for values in self.hyperparam_space]))

    def _validate_dataset(self, X_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X_data)
        y = np.asarray(y_data)

        raise_for_condition(
            X.ndim != 2,
            "X_data must be a 2D array.",
            error_cls=TuningValidationError,
            context=self._context("validate_dataset"),
            details={"x_shape": list(X.shape)},
        )
        raise_for_condition(
            X.shape[0] != y.shape[0],
            "X_data and y_data must have the same sample count.",
            error_cls=TuningValidationError,
            context=self._context("validate_dataset"),
            details={"x_rows": int(X.shape[0]), "y_rows": int(y.shape[0])},
        )
        raise_for_condition(
            X.shape[0] < self.settings.cross_val_folds,
            "Number of samples must be >= cross_val_folds.",
            error_cls=TuningValidationError,
            context=self._context("validate_dataset"),
            details={"num_samples": int(X.shape[0]), "cross_val_folds": self.settings.cross_val_folds},
        )
        raise_for_condition(
            not np.isfinite(np.asarray(X, dtype=float)).all(),
            "X_data contains non-finite values.",
            error_cls=TuningValidationError,
            context=self._context("validate_dataset"),
        )
        if y.dtype.kind in {"f", "i", "u", "b"}:
            raise_for_condition(
                not np.isfinite(np.asarray(y, dtype=float)).all(),
                "y_data contains non-finite values.",
                error_cls=TuningValidationError,
                context=self._context("validate_dataset"),
            )
        return X, y

    def _resolve_runtime_task_type(self, y_data: np.ndarray) -> str:
        configured = self.settings.task_type
        if configured in {"regression", "binary_classification", "multiclass_classification"}:
            return configured
        flattened = np.asarray(y_data).reshape(-1)
        unique_count = int(np.unique(flattened).size)
        if configured == "classification":
            return "binary_classification" if unique_count <= 2 else "multiclass_classification"
        if configured == "auto":
            if flattened.dtype.kind in {"b", "i", "u"} and unique_count <= max(20, self.settings.cross_val_folds):
                return "binary_classification" if unique_count <= 2 else "multiclass_classification"
            return "regression"
        return configured

    def _build_cv_splitter(self, y_data: np.ndarray):
        runtime_task_type = self._resolve_runtime_task_type(y_data)
        if runtime_task_type == "regression":
            return KFold(
                n_splits=self.settings.cross_val_folds,
                shuffle=True,
                random_state=self.settings.random_state,
            )

        flattened = np.asarray(y_data).reshape(-1)
        unique, counts = np.unique(flattened, return_counts=True)
        if unique.size >= 2 and int(np.min(counts)) >= self.settings.cross_val_folds:
            return StratifiedKFold(
                n_splits=self.settings.cross_val_folds,
                shuffle=True,
                random_state=self.settings.random_state,
            )

        logger.warning(
            "Falling back to KFold because stratification is not feasible for model=%s with class counts=%s.",
            self.settings.model_type,
            dict(zip(unique.tolist(), counts.tolist())),
        )
        return KFold(
            n_splits=self.settings.cross_val_folds,
            shuffle=True,
            random_state=self.settings.random_state,
        )

    def _invoke_external_evaluation(self, params: Dict[str, Any], *,
        x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
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

    def _evaluate_candidate(self, params: Dict[str, Any], *,
                            x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                            ) -> float:
        if self.use_builtin_gnn_evaluator:
            metric_name = self.settings.scoring_metric
            runtime_task_type = self._resolve_runtime_task_type(y_train)
            return evaluate_gnn_candidate(
                params=params,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                scoring_metric=metric_name,
                task_type=runtime_task_type,
                fit_kwargs=None,
            )

        return self._invoke_external_evaluation(
            params,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

    def run_search(self, X_data: np.ndarray, y_data: np.ndarray) -> Optional[Dict[str, Any]]:
        X, y = self._validate_dataset(X_data, y_data)

        self.results = []
        self.best_score = -np.inf
        self.best_score_std = np.nan
        self.best_params = None

        combinations = list(itertools.product(*self.hyperparam_space)) if self.hyperparam_space else [()]
        logger.info(
            "Starting grid search across %s combinations with %s-fold CV.",
            len(combinations),
            self.settings.cross_val_folds,
        )

        try:
            evaluated = Parallel(n_jobs=self.settings.n_jobs, prefer="processes")(
                delayed(self._evaluate_combination)(combo, X, y) for combo in combinations
            )
        except Exception as exc:  # noqa: BLE001
            wrapped = wrap_exception(
                exc,
                message="Grid search execution failed while evaluating parameter combinations.",
                error_cls=TuningOptimizationError,
                context=self._context("run_search"),
            )
            logger.error("%s", wrapped, exc_info=True)
            self._persist_results()
            if self.settings.plot_results:
                self.plot_search_performance()
            return None

        for index, result in enumerate(evaluated):
            result["id"] = index
            self.results.append(result)
            mean_score = result["scores"]["mean"]
            if np.isfinite(mean_score) and mean_score > self.best_score:
                self.best_score = mean_score
                self.best_score_std = result["scores"]["std"]
                self.best_params = result["params"]

        self._attach_effect_sizes()
        self._persist_results()
        if self.settings.plot_results:
            self.plot_search_performance()

        if self.best_params is None:
            logger.warning("Grid search completed but all configurations failed.")
        else:
            logger.info("Grid search best params=%s, mean_score=%.6f", self.best_params, self.best_score)

        return self.best_params

    def _evaluate_combination(self, combo: Tuple[Any, ...], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        params = dict(zip(self.param_names, combo))
        scores = self._cross_validate(params=params, X_data=X, y_data=y)
        return {"params": params, "scores": scores}

    def _cross_validate(self, params: Dict[str, Any], X_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        splitter = self._build_cv_splitter(y_data)
        fold_scores: List[float] = []
        fold_errors: List[Dict[str, Any]] = []

        iterator: Iterable[Tuple[np.ndarray, np.ndarray]]
        if isinstance(splitter, StratifiedKFold):
            iterator = splitter.split(X_data, np.asarray(y_data).reshape(-1))
        else:
            iterator = splitter.split(X_data, y_data)

        for fold_idx, (train_idx, val_idx) in enumerate(iterator, start=1):
            X_train, X_val = X_data[train_idx], X_data[val_idx]
            y_train, y_val = y_data[train_idx], y_data[val_idx]

            try:
                score = float(
                    self._evaluate_candidate(
                        params=dict(params),
                        x_train=X_train,
                        y_train=y_train,
                        x_val=X_val,
                        y_val=y_val,
                    )
                )
                raise_for_condition(
                    not np.isfinite(score),
                    "Evaluation function returned a non-finite score.",
                    error_cls=TuningEvaluationError,
                    context=self._context("cross_validate", fold_index=fold_idx, parameters=params),
                    details={"score": score},
                )
                fold_scores.append(score)
            except Exception as exc:  # noqa: BLE001
                wrapped = wrap_exception(
                    exc,
                    message="Fold evaluation failed during grid search.",
                    error_cls=TuningEvaluationError,
                    context=self._context("cross_validate", fold_index=fold_idx, parameters=params),
                    details={"params": safe_serialize(params), "fold_index": fold_idx},
                )
                logger.error("Fold evaluation failed for params=%s, fold=%s: %s", params, fold_idx, wrapped, exc_info=True)
                fold_scores.append(float("-inf"))
                fold_errors.append(self._serialize_error(wrapped))

        finite_scores = np.array([score for score in fold_scores if np.isfinite(score)], dtype=float)
        if finite_scores.size == 0:
            return {
                "mean": float("-inf"),
                "std": float("inf"),
                "ci95": [float("-inf"), float("-inf")],
                "raw_scores": fold_scores,
                "failed_folds": len(fold_errors),
                "errors": fold_errors,
            }

        mean_score = float(np.mean(finite_scores))
        std_score = float(np.std(finite_scores))
        margin = 1.96 * std_score / np.sqrt(finite_scores.size) if finite_scores.size > 1 else 0.0
        return {
            "mean": mean_score,
            "std": std_score,
            "ci95": [mean_score - margin, mean_score + margin],
            "raw_scores": fold_scores,
            "failed_folds": len(fold_errors),
            "errors": fold_errors,
        }

    def _attach_effect_sizes(self) -> None:
        if not self.results:
            return

        running_best_mean: Optional[float] = None
        running_best_std: Optional[float] = None

        for item in self.results:
            mean = item["scores"]["mean"]
            std = item["scores"]["std"]

            if running_best_mean is None or running_best_std is None:
                item["effect_size_vs_best_so_far"] = 0.0
                running_best_mean, running_best_std = mean, std
                continue

            pooled = np.sqrt((std**2 + running_best_std**2) / 2.0)
            if pooled > 1e-12 and np.isfinite(pooled) and np.isfinite(mean) and np.isfinite(running_best_mean):
                item["effect_size_vs_best_so_far"] = float((mean - running_best_mean) / pooled)
            elif np.isclose(mean, running_best_mean, equal_nan=False):
                item["effect_size_vs_best_so_far"] = 0.0
            else:
                item["effect_size_vs_best_so_far"] = float("nan")

            if np.isfinite(mean) and mean > running_best_mean:
                running_best_mean, running_best_std = mean, std

    def _persist_results(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        payload = {
            "created_at_utc": timestamp,
            "search_configuration": {
                "model_type": self.settings.model_type,
                "builtin_gnn_evaluator": self.use_builtin_gnn_evaluator,
                "n_jobs": self.settings.n_jobs,
                "cross_val_folds": self.settings.cross_val_folds,
                "random_state": self.settings.random_state,
                "scoring_metric": self.settings.scoring_metric,
                "task_type": self.settings.task_type,
                "parameter_names": self.param_names,
                "parameter_value_options": self.hyperparam_space,
            },
            "best_result": {
                "parameters": self.best_params,
                "mean_score": self.best_score,
                "score_std_dev": self.best_score_std,
            },
            "all_evaluated_combinations": self.results,
        }

        output_path = self.settings.output_dir / f"grid_search_results_{self.settings.model_type}_{timestamp}.json"
        try:
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(self._to_json_safe(payload), handle, indent=2)
            logger.info("Saved grid search results to %s", output_path)
        except Exception as exc:  # noqa: BLE001
            wrapped = wrap_exception(
                exc,
                message="Failed to persist grid search results.",
                error_cls=TuningPersistenceError,
                context=self._context("persist_results"),
            )
            logger.error("%s", wrapped, exc_info=True)

    def plot_search_performance(self) -> None:
        try:
            if not self.results:
                return

            indices = np.arange(len(self.results))
            means = np.array([item["scores"]["mean"] for item in self.results], dtype=float)
            ci_low = np.array([item["scores"]["ci95"][0] for item in self.results], dtype=float)
            ci_high = np.array([item["scores"]["ci95"][1] for item in self.results], dtype=float)
            effects = np.array([item.get("effect_size_vs_best_so_far", np.nan) for item in self.results], dtype=float)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
            finite_mean = np.isfinite(means)
            finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)

            ax1.plot(indices[finite_mean], means[finite_mean], marker="o", linewidth=1.1, label="Mean CV score")
            ax1.fill_between(indices[finite_ci], ci_low[finite_ci], ci_high[finite_ci], alpha=0.2, label="95% CI")

            if self.best_params is not None:
                best_idx = next(index for index, item in enumerate(self.results) if item["params"] == self.best_params)
                ax1.scatter([best_idx], [self.best_score], s=120, marker="*", color="gold", edgecolor="black", label="Best")

            ax1.set_ylabel("Score")
            ax1.set_title(f"Grid Search CV Performance ({self.settings.model_type})")
            ax1.grid(True, linestyle="--", alpha=0.4)
            ax1.legend()

            finite_effect = np.isfinite(effects)
            ax2.plot(indices[finite_effect], effects[finite_effect], marker=".", linestyle="-", label="Effect size")
            ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
            ax2.set_xlabel("Combination index")
            ax2.set_ylabel("Cohen's d")
            ax2.grid(True, linestyle="--", alpha=0.4)
            ax2.legend()

            fig.tight_layout()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            plot_path = self.settings.output_dir / f"grid_search_performance_{self.settings.model_type}_{timestamp}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info("Saved grid search performance plot to %s", plot_path)
        except Exception as exc:  # noqa: BLE001
            wrapped = wrap_exception(
                exc,
                message="Failed to generate grid search performance plot.",
                error_cls=TuningReportingError,
                context=self._context("plot_search_performance"),
            )
            logger.error("%s", wrapped, exc_info=True)

    def _serialize_error(self, error: BaseException) -> Dict[str, Any]:
        if hasattr(error, "to_log_record"):
            try:
                return safe_serialize(error.to_log_record())
            except Exception:  # noqa: BLE001
                pass
        if hasattr(error, "to_dict"):
            try:
                return safe_serialize(error.to_dict())
            except Exception:  # noqa: BLE001
                pass
        return {
            "name": error.__class__.__name__,
            "message": str(error),
        }

    def _to_json_safe(self, data: Any) -> Any:
        try:
            return safe_serialize(data)
        except Exception:  # noqa: BLE001
            if isinstance(data, dict):
                return {key: self._to_json_safe(value) for key, value in data.items()}
            if isinstance(data, list):
                return [self._to_json_safe(value) for value in data]
            if isinstance(data, tuple):
                return [self._to_json_safe(value) for value in data]
            if isinstance(data, np.integer):
                return int(data)
            if isinstance(data, np.floating):
                return float(data)
            if isinstance(data, np.ndarray):
                return data.tolist()
            if isinstance(data, np.bool_):
                return bool(data)
            return data


if __name__ == "__main__":
    rng = np.random.default_rng(5)

    X_demo = rng.normal(size=(100, 5))
    y_demo = rng.integers(0, 2, size=100)

    search = GridSearch(evaluation_function=evaluate_gnn_candidate, model_type="GridNeuralNetwork")
    print(search.run_search(X_demo, y_demo))
