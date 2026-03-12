"""Production-ready Bayesian hyperparameter search."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import json

from pathlib import Path
from skopt import gp_minimize
from dataclasses import dataclass
from datetime import datetime, timezone
from skopt.space import Categorical, Integer, Real
from skopt.utils import OptimizeResult, use_named_args
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.tuning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("BayesianSearch")
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


class BayesianSearch:
    """Bayesian optimization wrapper with validation, logging, and persistence."""

    def __init__(
        self,
        evaluation_function: Callable[[Dict[str, Any]], float],
        model_type: Optional[str] = None,
        objective: str = "minimize",
    ) -> None:
        if evaluation_function is None:
            raise ValueError("evaluation_function is required.")

        self.evaluation_function = evaluation_function
        self.config: Dict[str, Any] = load_global_config() or {}
        self.settings = self._load_settings(model_type=model_type, objective=objective)
        self.search_space_config, self.dimensions = self._load_search_space(self.settings.model_type)

        if not self.dimensions:
            raise ValueError("Bayesian search requires at least one hyperparameter dimension.")

        self.param_names = [dim.name for dim in self.dimensions]
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_score_so_far = np.inf if self.settings.objective == "minimize" else -np.inf
        self.best_params_so_far: Optional[Dict[str, Any]] = None

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "BayesianSearch initialized with model=%s, objective=%s, n_calls=%s",
            self.settings.model_type,
            self.settings.objective,
            self.settings.n_calls,
        )

    def _load_settings(self, model_type: Optional[str], objective: str) -> BayesianSearchSettings:
        section = get_config_section("bayesian_search") or {}
        configured_model = model_type or section.get("model_type") or "GradientBoosting"

        n_calls = int(section.get("n_calls", 20))
        n_initial_points = int(section.get("n_initial_points", 5))
        random_state = section.get("random_state")
        output_dir = Path(section.get("output_dir", "src/tuning/reports/bayesian_search"))

        if n_calls < 2:
            raise ValueError("n_calls must be >= 2.")
        if n_initial_points < 1:
            raise ValueError("n_initial_points must be >= 1.")
        if n_initial_points >= n_calls:
            raise ValueError("n_initial_points must be less than n_calls.")

        normalized_objective = objective.strip().lower()
        if normalized_objective not in {"minimize", "maximize"}:
            raise ValueError("objective must be either 'minimize' or 'maximize'.")

        return BayesianSearchSettings(
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            output_dir=output_dir,
            model_type=configured_model,
            objective=normalized_objective,
        )

    def _load_search_space(self, model_type: str) -> Tuple[List[Dict[str, Any]], List[Dimension]]:
        hp_root = self.config.get("hyperparameters")
        if not isinstance(hp_root, dict):
            raise ValueError("Expected 'hyperparameters' to be a model-keyed dictionary.")

        model_specs = next((v for k, v in hp_root.items() if k.lower() == model_type.lower()), None)
        if not isinstance(model_specs, list):
            raise ValueError(f"No hyperparameter list found for model_type='{model_type}'.")

        dimensions: List[Dimension] = []
        seen_names: set[str] = set()

        for spec in model_specs:
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid hyperparameter specification: {spec!r}")

            name = str(spec.get("name", "")).strip()
            if not name:
                raise ValueError(f"Missing hyperparameter name in spec: {spec!r}")
            if name in seen_names:
                raise ValueError(f"Duplicate hyperparameter name detected: {name}")
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
            if prior not in {"uniform", "log-uniform"}:
                raise ValueError(f"Unsupported prior '{prior}' for real parameter '{name}'.")
            if prior == "log-uniform" and (low <= 0 or high <= 0):
                raise ValueError(f"log-uniform prior requires positive bounds for '{name}'.")
            return Real(low=low, high=high, prior=prior, name=name)

        if param_type == "categorical":
            values = spec.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"Categorical parameter '{name}' requires a non-empty list of values.")
            return Categorical(categories=values, name=name)

        raise ValueError(f"Unsupported parameter type '{param_type}' for '{name}'.")

    @staticmethod
    def _resolve_bounds(spec: Dict[str, Any], cast_type: Callable[[Any], Any]) -> Tuple[Any, Any]:
        if "min" in spec and "max" in spec:
            low, high = cast_type(spec["min"]), cast_type(spec["max"])
        elif "values" in spec:
            values = [v for v in spec["values"] if v is not None]
            if not values:
                raise ValueError("Parameter 'values' cannot be empty.")
            casted = [cast_type(v) for v in values]
            low, high = min(casted), max(casted)
        else:
            raise ValueError("Numeric parameter requires either min/max or values.")

        if low > high:
            raise ValueError(f"Invalid bounds: min ({low}) > max ({high}).")
        return low, high

    def run_search(self) -> Tuple[Optional[Dict[str, Any]], float, Optional[OptimizeResult]]:
        logger.info("Starting Bayesian search with %s dimensions.", len(self.dimensions))

        def score_to_minimize(raw_score: float) -> float:
            if self.settings.objective == "minimize":
                return raw_score
            return -raw_score

        @use_named_args(self.dimensions)
        def objective_function(**params: Any) -> float:
            try:
                raw_score = float(self.evaluation_function(params))
                if not np.isfinite(raw_score):
                    raise ValueError("non-finite score returned")
                transformed = score_to_minimize(raw_score)
                self._record_iteration(params=params, raw_score=raw_score)
                return transformed
            except Exception as exc:
                logger.error("Evaluation failed for params=%s: %s", params, exc, exc_info=True)
                self._record_iteration(params=params, raw_score=np.nan, failed=True)
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
        except Exception as exc:
            logger.error("Bayesian optimization run failed: %s", exc, exc_info=True)

        if result is not None:
            best_params = dict(zip(self.param_names, result.x))
            best_score = float(result.fun if self.settings.objective == "minimize" else -result.fun)
        else:
            best_params = self.best_params_so_far
            best_score = float(self.best_score_so_far) if self.best_params_so_far is not None else float("nan")

        self._persist_results(best_params=best_params, best_score=best_score, result=result)
        self._plot_optimization_progress()
        return best_params, best_score, result

    def _record_iteration(self, params: Dict[str, Any], raw_score: float, failed: bool = False) -> None:
        iteration = len(self.optimization_history) + 1
        entry = {
            "iteration": iteration,
            "parameters": params,
            "score": raw_score,
            "failed": failed,
        }
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
            self.best_params_so_far = params

    def _persist_results(
        self,
        best_params: Optional[Dict[str, Any]],
        best_score: float,
        result: Optional[OptimizeResult],
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        summary = {
            "created_at_utc": timestamp,
            "search_configuration": {
                "model_type": self.settings.model_type,
                "objective": self.settings.objective,
                "n_calls": self.settings.n_calls,
                "n_initial_points": self.settings.n_initial_points,
                "random_state": self.settings.random_state,
                "search_space_definition": self.search_space_config,
            },
            "best_result": {
                "parameters": best_params,
                "score": best_score,
            },
            "optimization_history": self.optimization_history,
        }

        if result is not None:
            summary["optimizer_state"] = {
                "x_iters": result.x_iters,
                "func_vals": result.func_vals.tolist(),
                "space": str(result.space),
            }

        output_path = self.settings.output_dir / f"bayesian_search_summary_{self.settings.model_type}_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self._to_json_safe(summary), handle, indent=2)
        logger.info("Saved Bayesian search summary to %s", output_path)

    def _plot_optimization_progress(self) -> None:
        if not self.optimization_history:
            return

        iterations = [item["iteration"] for item in self.optimization_history]
        scores = np.array([item["score"] for item in self.optimization_history], dtype=float)
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            logger.warning("No finite scores available for plotting.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(np.array(iterations)[finite_mask], scores[finite_mask], marker="o", linewidth=1.2, label="Score")

        best_trace = np.minimum.accumulate(scores[finite_mask]) if self.settings.objective == "minimize" else np.maximum.accumulate(scores[finite_mask])
        plt.plot(np.array(iterations)[finite_mask], best_trace, linestyle="--", label="Best so far")

        plt.title(f"Bayesian Optimization Progress ({self.settings.model_type})")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        plot_path = self.settings.output_dir / f"bayesian_progress_{self.settings.model_type}_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info("Saved Bayesian optimization plot to %s", plot_path)

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
    def _dummy_eval(params: Dict[str, Any]) -> float:
        return float(np.random.default_rng(7).random()) + float(params.get("n_estimators", 0)) * 1e-5

    search = BayesianSearch(evaluation_function=_dummy_eval, model_type="GradientBoosting")
    params, score, _ = search.run_search()
    print(f"best params={params}, score={score}")
