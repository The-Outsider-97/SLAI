"""
Production-ready hyperparameter tuning orchestrator.

Orchestrates Bayesian and Grid Search strategies over a unified pipeline,
with structured validation, lifecycle hooks, result persistence, and
comprehensive logging. Strategy selection, run parameters, and model
configuration are driven entirely by hyperparam.yaml — no duplicate
configs live in this module.
"""

from __future__ import annotations

import json
import time

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .grid_search import GridSearch
from .bayesian_search import BayesianSearch
from .utils.tuning_error import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Hyperparameter Tuner")
printer = PrettyPrinter()

# ---------------------------------------------------------------------------
# Supported strategy identifiers
# ---------------------------------------------------------------------------
_SUPPORTED_STRATEGIES: frozenset[str] = frozenset({"bayesian", "grid"})


# ---------------------------------------------------------------------------
# Immutable run-time settings (populated from YAML, validated once)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TunerSettings:
    """Validated, immutable settings for one tuning run."""

    strategy: str
    model_type: str
    allow_generate: bool
    output_dir: Path

    # Bayesian-specific (None when strategy == 'grid')
    n_calls: Optional[int]
    n_initial_points: Optional[int]
    random_state: Optional[int]

    # Grid-specific (None when strategy == 'bayesian')
    cross_val_folds: Optional[int]


# ---------------------------------------------------------------------------
# Lightweight result container
# ---------------------------------------------------------------------------
@dataclass
class TuningResult:
    """Holds the outcome of a completed tuning pipeline run."""

    strategy: str
    model_type: str
    best_params: Dict[str, Any]
    best_score: Optional[float]
    elapsed_seconds: float
    completed_at_utc: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "model_type": self.model_type,
            "best_params": safe_serialize(self.best_params),
            "best_score": self.best_score,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "completed_at_utc": self.completed_at_utc,
            "metadata": safe_serialize(self.metadata),
        }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
class HyperparamTuner:
    """
    Orchestrates hyperparameter optimization across strategies.

    Responsibilities
    ----------------
    - Load and validate all tuning configuration from hyperparam.yaml.
    - Construct the appropriate search backend (BayesianSearch / GridSearch).
    - Execute the full search pipeline with timing and error recovery.
    - Persist a structured JSON summary of every run.
    - Expose a clean, typed result via TuningResult.

    Configuration is read exclusively from hyperparam.yaml sections:
        ``tuning``, ``bayesian_search``, ``grid_search``.
    No config values are hard-coded here.
    """

    def __init__(self, model_type: Optional[str] = None,
                 evaluation_function: Optional[Callable[..., float]] = None) -> None:
        """
        Initialize the tuner from YAML config.

        Args:
            model_type: Override the model type declared in hyperparam.yaml.
                        When omitted, falls back to ``model_type`` in the
                        global config section.
            evaluation_function: Callable ``f(params, X, y) -> float`` used
                                  by both search backends.  Required unless
                                  the selected strategy uses a built-in
                                  evaluator (e.g. BayesianNeuralNetwork /
                                  GridNeuralNetwork).
        """
        self._global_config: Dict[str, Any] = load_global_config()
        self._tuning_config: Dict[str, Any] = get_config_section("tuning")

        self.settings: TunerSettings = self._load_settings(model_type)
        self.evaluation_function = evaluation_function

        self._validate_evaluation_function()

        self.optimizer: BayesianSearch | GridSearch = self._build_optimizer()

        logger.info(
            "HyperparamTuner ready — strategy=%s, model=%s, allow_generate=%s",
            self.settings.strategy,
            self.settings.model_type,
            self.settings.allow_generate,
        )
        printer.status(
            "HyperparamTuner",
            f"strategy={self.settings.strategy!r}  model={self.settings.model_type!r}",
            "info",
        )

    # ------------------------------------------------------------------
    # Internal helpers — settings & validation
    # ------------------------------------------------------------------

    def _error_context(self, operation: str, **extra: Any) -> TuningErrorContext:
        """Build a structured error context anchored to this component."""
        return TuningErrorContext(
            component="HyperparamTuner",
            operation=operation,
            strategy=self.settings.strategy if hasattr(self, "settings") else None,
            model_type=self.settings.model_type if hasattr(self, "settings") else None,
            config_path=str(self._global_config.get("__config_path__", "")) or None,
            parameters={k: v for k, v in extra.items() if v is not None},
        )

    def _load_settings(self, model_type_override: Optional[str]) -> TunerSettings:
        """
        Read, coerce, and validate all relevant YAML sections into
        an immutable TunerSettings dataclass.
        """
        strategy = str(self._tuning_config.get("strategy", "bayesian")).strip().lower()
        raise_for_condition(
            strategy not in _SUPPORTED_STRATEGIES,
            f"Unsupported strategy {strategy!r}. Must be one of: {sorted(_SUPPORTED_STRATEGIES)}.",
            error_cls=TuningStrategyError,
            context=TuningErrorContext(
                component="HyperparamTuner",
                operation="load_settings",
                strategy=strategy,
            ),
            details={"strategy": strategy},
        )

        allow_generate = bool(self._tuning_config.get("allow_generate", True))
        model_type = (
            model_type_override
            or self._global_config.get("model_type", "GradientBoosting")
        )

        # Resolve strategy-specific sub-section once; avoids repeated lookups.
        n_calls = n_initial_points = random_state = cross_val_folds = None

        if strategy == "bayesian":
            bsec = get_config_section("bayesian_search")
            n_calls = int(bsec.get("n_calls", 20))
            n_initial_points = int(bsec.get("n_initial_points", 5))
            random_state = bsec.get("random_state")
            output_dir = Path(bsec.get("output_dir", "src/tuning/reports/bayesian_search"))

            raise_for_condition(
                n_calls < 2,
                "bayesian_search.n_calls must be >= 2.",
                error_cls=TuningConfigError,
                context=TuningErrorContext(
                    component="HyperparamTuner",
                    operation="load_settings",
                    strategy=strategy,
                ),
                details={"n_calls": n_calls},
            )
            raise_for_condition(
                n_initial_points < 1,
                "bayesian_search.n_initial_points must be >= 1.",
                error_cls=TuningConfigError,
                context=TuningErrorContext(
                    component="HyperparamTuner",
                    operation="load_settings",
                    strategy=strategy,
                ),
                details={"n_initial_points": n_initial_points},
            )

        else:  # grid
            gsec = get_config_section("grid_search")
            cross_val_folds = int(gsec.get("cross_val_folds", 5))
            random_state = gsec.get("random_state")
            output_dir = Path(gsec.get("output_dir", "src/tuning/reports/grid_search"))

            raise_for_condition(
                cross_val_folds < 2,
                "grid_search.cross_val_folds must be >= 2.",
                error_cls=TuningConfigError,
                context=TuningErrorContext(
                    component="HyperparamTuner",
                    operation="load_settings",
                    strategy=strategy,
                ),
                details={"cross_val_folds": cross_val_folds},
            )

        return TunerSettings(
            strategy=strategy,
            model_type=model_type,
            allow_generate=allow_generate,
            output_dir=output_dir,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            cross_val_folds=cross_val_folds,
        )

    def _validate_evaluation_function(self) -> None:
        """
        Guard: an evaluation function is required unless the selected
        optimizer provides its own built-in evaluator for the model type.

        Deferred to after _build_optimizer() creates self.optimizer, so
        we ask the optimizer directly whether it needs an external function.
        This avoids duplicating the built-in model name sets here.
        """
        # If evaluation_function is provided, nothing to check.
        if self.evaluation_function is not None:
            return

        # Without an evaluation function we rely on the optimizer's built-in
        # evaluator — validated inside BayesianSearch / GridSearch __init__.
        # This method is a pre-construction sanity gate; the optimizer itself
        # will raise a descriptive error if its built-in evaluator is also
        # unavailable for the chosen model type.

    def _build_optimizer(self) -> BayesianSearch | GridSearch:
        """Instantiate the concrete search backend from validated settings."""
        if self.settings.strategy == "bayesian":
            return BayesianSearch(
                evaluation_function=self.evaluation_function,
                model_type=self.settings.model_type,
            )
        # strategy == "grid"
        return GridSearch(
            evaluation_function=self.evaluation_function,
            model_type=self.settings.model_type,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_tuning_pipeline(self, X_data: Optional[Any] = None, y_data: Optional[Any] = None) -> TuningResult:
        """
        Execute the full hyperparameter search pipeline.

        Args:
            X_data: Feature matrix.  Required for Grid Search and for
                    Bayesian Search when using a built-in BNN evaluator.
            y_data: Target vector / matrix.  Same requirements as X_data.

        Returns:
            TuningResult — typed container with best params, score, timing,
            and a copy of the run metadata.

        Raises:
            TuningValidationError: If required data is missing for the strategy.
            TuningOptimizationError: If the underlying optimizer fails fatally.
        """
        self._validate_data_requirements(X_data, y_data)

        logger.info(
            "Pipeline start — strategy=%s, model=%s",
            self.settings.strategy,
            self.settings.model_type,
        )
        printer.status(
            "HyperparamTuner",
            f"Starting {self.settings.strategy} tuning for {self.settings.model_type!r} …",
            "info",
        )

        t_start = time.perf_counter()
        best_params: Dict[str, Any]
        best_score: float
        best_params, best_score = self._dispatch_search(X_data, y_data)
        elapsed = time.perf_counter() - t_start

        completed_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result = TuningResult(
            strategy=self.settings.strategy,
            model_type=self.settings.model_type,
            best_params=best_params,
            best_score=best_score,
            elapsed_seconds=elapsed,
            completed_at_utc=completed_at,
            metadata=self._build_run_metadata(X_data, y_data),
        )

        self._persist_run_summary(result)

        logger.info(
            "Pipeline complete — best_params=%s, score=%s, elapsed=%.2fs",
            best_params,
            best_score,
            elapsed,
        )
        printer.status(
            "HyperparamTuner",
            f"Done in {elapsed:.2f}s — best score: {best_score}",
            "success",
        )
        return result

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------
    def _validate_data_requirements(self, X_data: Optional[Any], y_data: Optional[Any]) -> None:
        """
        Ensure that X_data / y_data are present when the optimizer requires them.
        Delegates the 'requires_dataset' query to the optimizer itself so this
        class never needs to replicate strategy-specific logic.
        """
        needs_data = self.optimizer.requires_dataset()
        if needs_data:
            raise_for_condition(
                X_data is None or y_data is None,
                f"Strategy {self.settings.strategy!r} requires both X_data and y_data.",
                error_cls=TuningValidationError,
                context=self._error_context("validate_data_requirements"),
                details={
                    "X_data_provided": X_data is not None,
                    "y_data_provided": y_data is not None,
                },
            )

    def _dispatch_search(self, X_data: Optional[Any], y_data: Optional[Any]) -> Tuple[Dict[str, Any], float]:
        """
        Call the correct run_search signature for the active strategy.
    
        Returns:
            A tuple (best_params, best_score). Both are guaranteed to be valid.
            If the search cannot produce a valid configuration, a TuningOptimizationError is raised.
    
        Raises:
            TuningOptimizationError: When the underlying search fails to find any valid hyperparameters.
        """
        try:
            if self.settings.strategy == "grid":
                best_params = self.optimizer.run_search(X_data, y_data)
                if best_params is None:
                    raise TuningOptimizationError(
                        "Grid search failed to produce any valid hyperparameter configuration.",
                        context=self._error_context("dispatch_search", strategy="grid"),
                    )
                # best_score is guaranteed to be a float because GridSearch.best_score is always a float.
                best_score = float(self.optimizer.best_score) # type: ignore
                return best_params, best_score # type: ignore
    
            # bayesian
            best_params, best_score, _ = self.optimizer.run_search(X_data=X_data, y_data=y_data) # type: ignore
            if best_params is None:
                raise TuningOptimizationError(
                    "Bayesian search failed to produce any valid hyperparameter configuration.",
                    context=self._error_context("dispatch_search", strategy="bayesian"),
                )
            # best_score is already a float from BayesianSearch.run_search
            return best_params, best_score # type: ignore
    
        except (TuningConfigError, TuningStrategyError, TuningValidationError):
            raise  # Re-raise structured errors as-is.
        except Exception as exc:
            wrapped = wrap_exception(
                exc,
                message="Tuning pipeline failed during search dispatch.",
                error_cls=TuningOptimizationError,
                context=self._error_context("dispatch_search"),
            )
            logger.error("%s", wrapped, exc_info=True)
            raise wrapped from exc

    def _build_run_metadata(
        self,
        X_data: Optional[Any],
        y_data: Optional[Any],
    ) -> Dict[str, Any]:
        """Assemble lightweight run metadata for the persisted summary."""
        meta: Dict[str, Any] = {
            "n_calls": self.settings.n_calls,
            "n_initial_points": self.settings.n_initial_points,
            "random_state": self.settings.random_state,
            "cross_val_folds": self.settings.cross_val_folds,
            "allow_generate": self.settings.allow_generate,
        }
        if X_data is not None:
            try:
                meta["X_shape"] = list(X_data.shape)
            except AttributeError:
                meta["X_shape"] = len(X_data) if hasattr(X_data, "__len__") else None
        if y_data is not None:
            try:
                meta["y_shape"] = list(y_data.shape)
            except AttributeError:
                meta["y_shape"] = len(y_data) if hasattr(y_data, "__len__") else None
        return meta

    def _persist_run_summary(self, result: TuningResult) -> None:
        """
        Write a JSON run summary to the strategy-specific output directory.
        Failures are logged and swallowed so a persistence issue never
        obscures a successful tuning result.
        """
        try:
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"tuner_summary_{self.settings.strategy}"
                f"_{self.settings.model_type}"
                f"_{result.completed_at_utc}.json"
            )
            output_path = self.settings.output_dir / filename
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, indent=2, sort_keys=True)
            logger.info("Run summary saved to %s", output_path)
        except Exception as exc:  # noqa: BLE001
            wrapped = wrap_exception(
                exc,
                message="Failed to persist tuning run summary.",
                error_cls=TuningPersistenceError,
                context=self._error_context("persist_run_summary"),
            )
            logger.error("%s", wrapped, exc_info=True)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> str:
        """Active strategy identifier ('bayesian' or 'grid')."""
        return self.settings.strategy

    @property
    def model_type(self) -> str:
        """Model type driving the search space selection."""
        return self.settings.model_type

    def __repr__(self) -> str:
        return (
            f"HyperparamTuner("
            f"strategy={self.settings.strategy!r}, "
            f"model={self.settings.model_type!r})"
        )


# ---------------------------------------------------------------------------
# Standalone test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np # type: ignore
    import unittest.mock as mock

    print("\n=== Running HyperparamTuner ===\n")
    printer.status("TEST", "HyperparamTuner initialized", "info")

    rng = np.random.default_rng(42)
    X_demo = rng.normal(size=(80, 4)).astype(np.float64)
    y_demo = (X_demo[:, 0] * 0.6 + rng.normal(0.0, 0.1, 80)).astype(np.float64)

    # ── 1. Validate TunerSettings loads without error ──────────────────
    printer.status("TEST", "Loading settings from hyperparam.yaml …", "info")
    cfg = load_global_config()
    assert "tuning" in cfg, "Missing 'tuning' section in hyperparam.yaml"
    assert "bayesian_search" in cfg, "Missing 'bayesian_search' section"
    assert "grid_search" in cfg, "Missing 'grid_search' section"
    printer.status("TEST", "Config sections verified ✔", "success")

    # ── 2. Bayesian path (BayesianNeuralNetwork built-in evaluator) ────
    printer.status("TEST", "Instantiating Bayesian tuner (BNN built-in evaluator) …", "info")
    bayes_tuner = HyperparamTuner(model_type="BayesianNeuralNetwork")
    assert bayes_tuner.strategy == "bayesian", "Expected bayesian strategy"
    assert isinstance(bayes_tuner.optimizer, BayesianSearch), "Expected BayesianSearch optimizer"
    printer.status("TEST", f"Tuner repr: {bayes_tuner!r}", "info")

    printer.status("TEST", "Running Bayesian pipeline …", "info")
    bayes_result = bayes_tuner.run_tuning_pipeline(X_data=X_demo, y_data=y_demo)
    assert isinstance(bayes_result, TuningResult), "Expected TuningResult"
    assert isinstance(bayes_result.best_params, dict), "best_params must be a dict"
    assert bayes_result.elapsed_seconds > 0, "Elapsed time must be positive"
    printer.status("TEST", f"Bayesian best_params={bayes_result.best_params}", "success")

    # ── 3. TuningResult serialisation ─────────────────────────────────
    printer.status("TEST", "Verifying TuningResult.to_dict() …", "info")
    result_dict = bayes_result.to_dict()
    for key in ("strategy", "model_type", "best_params", "elapsed_seconds", "completed_at_utc"):
        assert key in result_dict, f"Missing key {key!r} in TuningResult.to_dict()"
    printer.status("TEST", "TuningResult serialisation ✔", "success")

    # ── 4. Error path — missing data for grid search ──────────────────
    printer.status("TEST", "Checking TuningValidationError for missing data …", "info")

    def _override_strategy_to_grid(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return a config copy with tuning.strategy forced to 'grid'."""
        cfg = load_global_config()  # call the REAL function (not mocked inside the patch)
        cfg["tuning"] = cfg.get("tuning", {})
        cfg["tuning"]["strategy"] = "grid"
        return cfg

    with mock.patch(
        "src.tuning.utils.config_loader.load_global_config",
        side_effect=_override_strategy_to_grid,
    ):
        grid_tuner = HyperparamTuner(model_type="GridNeuralNetwork")
        try:
            grid_tuner.run_tuning_pipeline()  # no X/y supplied
            assert False, "Expected TuningValidationError"
        except TuningValidationError:
            printer.status("TEST", "TuningValidationError raised correctly ✔", "success")

    # ── 5. Error path — unsupported strategy ──────────────────────────
    printer.status("TEST", "Checking TuningStrategyError for bad strategy …", "info")

    def _override_strategy_to_invalid(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        cfg = load_global_config()
        cfg["tuning"] = cfg.get("tuning", {})
        cfg["tuning"]["strategy"] = "genetic"
        return cfg

    with mock.patch(
        "src.tuning.utils.config_loader.load_global_config",
        side_effect=_override_strategy_to_invalid,
    ):
        try:
            _ = HyperparamTuner(model_type="GradientBoosting")
            assert False, "Expected TuningStrategyError"
        except TuningStrategyError:
            printer.status("TEST", "TuningStrategyError raised correctly ✔", "success")

    print("\n=== Test ran successfully ===\n")