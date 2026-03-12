"""Production-ready exhaustive grid search with cross-validation."""

from __future__ import annotations

import itertools
import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from joblib import Parallel, delayed
from datetime import datetime, timezone
from sklearn.model_selection import KFold
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.tuning.utils.config_loader import get_config_section, load_global_config
from logs.logger import get_logger

logger = get_logger("GridSearch")


@dataclass(frozen=True)
class GridSearchSettings:
    """Validated runtime settings for grid search."""

    n_jobs: int
    cross_val_folds: int
    random_state: int
    output_dir: Path
    model_type: str


class GridSearch:
    """Deterministic grid search with robust validation, tracking, and reporting."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        evaluation_function: Optional[
            Callable[[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
        ] = None,
        model_type: Optional[str] = None,
    ) -> None:
        if evaluation_function is None:
            raise ValueError("evaluation_function is required.")

        self.config: Dict[str, Any] = config or load_global_config() or {}
        self.settings = self._load_settings(model_type=model_type)
        self.param_names, self.hyperparam_space = self._load_search_space(self.settings.model_type)
        self._validate_search_space()

        self.evaluation_function = evaluation_function
        self.results: List[Dict[str, Any]] = []
        self.best_score = -np.inf
        self.best_score_std = np.nan
        self.best_params: Optional[Dict[str, Any]] = None

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "GridSearch initialized with model=%s, combinations=%s",
            self.settings.model_type,
            self._combination_count(),
        )

    def _load_settings(self, model_type: Optional[str]) -> GridSearchSettings:
        section = get_config_section("grid_search") or {}
        configured_model = model_type or section.get("model_type") or "GradientBoosting"

        n_jobs = int(section.get("n_jobs", -1))
        folds = int(section.get("cross_val_folds", 5))
        random_state = int(section.get("random_state", 42))
        output_dir = Path(section.get("output_dir", "src/tuning/reports/grid_search"))

        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0.")
        if folds < 2:
            raise ValueError("cross_val_folds must be >= 2.")

        return GridSearchSettings(
            n_jobs=n_jobs,
            cross_val_folds=folds,
            random_state=random_state,
            output_dir=output_dir,
            model_type=configured_model,
        )

    def _load_search_space(self, model_type: str) -> Tuple[List[str], List[List[Any]]]:
        hp_root = self.config.get("hyperparameters")

        if isinstance(hp_root, dict):
            param_specs = next((v for k, v in hp_root.items() if k.lower() == model_type.lower()), None)
            if param_specs is None:
                raise ValueError(f"No hyperparameter list found for model_type='{model_type}'.")
        elif isinstance(hp_root, list):
            param_specs = hp_root
        else:
            raise ValueError("'hyperparameters' must be a list or model-keyed dictionary.")

        if not isinstance(param_specs, list):
            raise ValueError("Hyperparameter configuration must be a list of parameter specs.")

        param_names: List[str] = []
        space: List[List[Any]] = []
        for spec in param_specs:
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid parameter spec: {spec!r}")

            name = str(spec.get("name", "")).strip()
            if not name:
                raise ValueError(f"Missing hyperparameter name in spec: {spec!r}")
            if name in param_names:
                raise ValueError(f"Duplicate hyperparameter name detected: {name}")

            values = self._normalize_values(spec)
            if not values:
                raise ValueError(f"Hyperparameter '{name}' has no candidate values.")

            param_names.append(name)
            space.append(values)

        return param_names, space

    def _normalize_values(self, spec: Dict[str, Any]) -> List[Any]:
        if "values" in spec:
            values = [v for v in spec["values"] if v is not None]
            return values

        raw_type = str(spec.get("type", "")).lower()
        if "min" in spec and "max" in spec and raw_type in {"integer", "int"}:
            low, high = int(spec["min"]), int(spec["max"])
            if low > high:
                raise ValueError(f"Invalid integer bounds in spec: {spec}")
            return list(range(low, high + 1))

        raise ValueError("Grid search requires 'values' or integer min/max bounds.")

    def _validate_search_space(self) -> None:
        combinations = self._combination_count()
        if combinations <= 0:
            raise ValueError("Search space produced zero combinations.")
        if combinations > 100_000:
            logger.warning("Large search space detected (%s combinations).", combinations)

    def _combination_count(self) -> int:
        if not self.hyperparam_space:
            return 1
        return int(np.prod([len(v) for v in self.hyperparam_space]))

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

        evaluated = Parallel(n_jobs=self.settings.n_jobs, prefer="processes")(
            delayed(self._evaluate_combination)(combo, X, y) for combo in combinations
        )

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
        self.plot_search_performance()

        if self.best_params is None:
            logger.warning("Grid search completed but all configurations failed.")
        else:
            logger.info("Grid search best params=%s, mean_score=%.6f", self.best_params, self.best_score)

        return self.best_params

    def _validate_dataset(self, X_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X_data)
        y = np.asarray(y_data)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same sample count: {X.shape[0]} != {y.shape[0]}")
        if X.shape[0] < self.settings.cross_val_folds:
            raise ValueError("Number of samples must be >= cross_val_folds.")

        return X, y

    def _evaluate_combination(self, combo: Tuple[Any, ...], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        params = dict(zip(self.param_names, combo))
        scores = self._cross_validate(params=params, X_data=X, y_data=y)
        return {"params": params, "scores": scores}

    def _cross_validate(self, params: Dict[str, Any], X_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
        kfold = KFold(
            n_splits=self.settings.cross_val_folds,
            shuffle=True,
            random_state=self.settings.random_state,
        )
        fold_scores: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_data, y_data), start=1):
            X_train, X_val = X_data[train_idx], X_data[val_idx]
            y_train, y_val = y_data[train_idx], y_data[val_idx]

            try:
                score = float(self.evaluation_function(params, X_train, y_train, X_val, y_val))
                if not np.isfinite(score):
                    raise ValueError("non-finite score")
                fold_scores.append(score)
            except Exception as exc:
                logger.error(
                    "Fold evaluation failed for params=%s, fold=%s: %s",
                    params,
                    fold_idx,
                    exc,
                    exc_info=True,
                )
                fold_scores.append(float("-inf"))

        finite_scores = np.array([s for s in fold_scores if np.isfinite(s)], dtype=float)
        if finite_scores.size == 0:
            return {"mean": float("-inf"), "std": float("inf"), "ci95": [float("-inf"), float("-inf")], "raw_scores": fold_scores}

        mean_score = float(np.mean(finite_scores))
        std_score = float(np.std(finite_scores))
        margin = 1.96 * std_score / np.sqrt(finite_scores.size) if finite_scores.size > 1 else 0.0
        return {
            "mean": mean_score,
            "std": std_score,
            "ci95": [mean_score - margin, mean_score + margin],
            "raw_scores": fold_scores,
        }

    def _attach_effect_sizes(self) -> None:
        if not self.results:
            return

        running_best_mean = None
        running_best_std = None

        for item in self.results:
            mean = item["scores"]["mean"]
            std = item["scores"]["std"]

            if running_best_mean is None:
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
                "n_jobs": self.settings.n_jobs,
                "cross_val_folds": self.settings.cross_val_folds,
                "random_state": self.settings.random_state,
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
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self._to_json_safe(payload), handle, indent=2)
        logger.info("Saved grid search results to %s", output_path)

    def plot_search_performance(self) -> None:
        if not self.results:
            return

        indices = np.arange(len(self.results))
        means = np.array([r["scores"]["mean"] for r in self.results], dtype=float)
        ci_low = np.array([r["scores"]["ci95"][0] for r in self.results], dtype=float)
        ci_high = np.array([r["scores"]["ci95"][1] for r in self.results], dtype=float)
        effects = np.array([r.get("effect_size_vs_best_so_far", np.nan) for r in self.results], dtype=float)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        finite_mean = np.isfinite(means)
        finite_ci = np.isfinite(ci_low) & np.isfinite(ci_high)

        ax1.plot(indices[finite_mean], means[finite_mean], marker="o", linewidth=1.1, label="Mean CV score")
        ax1.fill_between(indices[finite_ci], ci_low[finite_ci], ci_high[finite_ci], alpha=0.2, label="95% CI")

        if self.best_params is not None:
            best_idx = next(i for i, r in enumerate(self.results) if r["params"] == self.best_params)
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

    def _to_json_safe(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._to_json_safe(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_json_safe(v) for v in data]
        if isinstance(data, tuple):
            return [self._to_json_safe(v) for v in data]
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

    def _dummy_eval(params: Dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
        _ = (x_train, y_train, x_val, y_val)
        base = float(params.get("n_estimators", 0)) * 1e-4
        return float(rng.random()) + base

    X_demo = rng.normal(size=(100, 5))
    y_demo = rng.integers(0, 2, size=100)

    search = GridSearch(evaluation_function=_dummy_eval, model_type="GradientBoosting")
    print(search.run_search(X_demo, y_demo))
