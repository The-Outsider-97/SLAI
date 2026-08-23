"""Leakage-resistant supervised candidate evaluation for SLAI tuning.

Every built-in split has three disjoint partitions:

* training data fit model parameters;
* validation data may drive early stopping or model selection; and
* test data produce the score recorded by tuning.

Failed splits are never removed from an average.  Any model, prediction, or
metric failure produces a failed trial record so optimization cannot benefit
from selective fold survival.
"""

from __future__ import annotations

import math
import time
import numpy as np

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any

from ..tuning_contracts import *
from ..tuning_types import *
from ..tuning_validation import *
from ..utils.tuning_errors import *
from ..utils.tuning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Supervised Evaluator")
printer = PrettyPrinter()


class SupervisedSplitStrategy(str, Enum):
    HOLDOUT = "holdout"
    NESTED_K_FOLD = "nested_k_fold"
    TIME_SERIES = "time_series"


@dataclass(frozen=True, slots=True)
class SupervisedEvaluationConfig:
    objective: MetricSpec
    metrics: Sequence[str]
    seeds: Sequence[int]
    split_strategy: SupervisedSplitStrategy
    validation_fraction: float
    test_fraction: float = 0.2
    n_splits: int = 5
    shuffle: bool = True
    constraints: Sequence[MetricConstraint] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.objective, MetricSpec):
            raise TypeError("objective must be a MetricSpec")
        metrics = tuple(str(item).strip() for item in self.metrics)
        if not metrics or any(not item for item in metrics):
            raise ValueError("metrics must contain non-empty names")
        if len(set(metrics)) != len(metrics):
            raise ValueError("metrics must not contain duplicates")
        if self.objective.name not in metrics:
            raise ValueError("metrics must include the objective name")
        object.__setattr__(self, "metrics", metrics)
        seeds = tuple(self.seeds)
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise TypeError("seeds must contain integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", seeds)
        if not isinstance(self.split_strategy, SupervisedSplitStrategy):
            object.__setattr__(
                self,
                "split_strategy",
                SupervisedSplitStrategy(self.split_strategy),
            )
        if not 0.0 < float(self.validation_fraction) < 0.5:
            raise ValueError("validation_fraction must be within (0, 0.5)")
        if not 0.0 < float(self.test_fraction) < 1.0:
            raise ValueError("test_fraction must be within (0, 1)")
        if (
            self.split_strategy is SupervisedSplitStrategy.HOLDOUT
            and self.validation_fraction + self.test_fraction >= 1.0
        ):
            raise ValueError("validation_fraction + test_fraction must be less than 1")
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, int):
            raise TypeError("n_splits must be an integer")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not isinstance(self.shuffle, bool):
            raise TypeError("shuffle must be a bool")
        if self.split_strategy is SupervisedSplitStrategy.TIME_SERIES and self.shuffle:
            raise ValueError("Time-series evaluation cannot shuffle observations")
        constraints = tuple(self.constraints)
        if any(not isinstance(item, MetricConstraint) for item in constraints):
            raise TypeError("constraints must contain MetricConstraint objects")
        object.__setattr__(self, "constraints", constraints)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "SupervisedEvaluationConfig":
        report = validate_supervised_config(config)
        report.raise_if_invalid(
            message="Invalid supervised evaluation configuration.",
            context=TuningErrorContext(
                component="SupervisedEvaluator", operation="load_config"
            ),
        )
        try:
            return cls(
                objective=parse_metric_spec(
                    config["objective"], path="supervised_evaluation.objective"
                ),
                metrics=tuple(str(item) for item in config["metrics"]),
                seeds=tuple(config.get("seeds", ())),
                split_strategy=SupervisedSplitStrategy(
                    str(config["split_strategy"]).strip().casefold()
                ),
                validation_fraction=float(config["validation_fraction"]),
                test_fraction=float(config.get("test_fraction", 0.2)),
                n_splits=int(config.get("n_splits", 5)),
                shuffle=coerce_bool(config.get("shuffle", True), name="shuffle"),
                constraints=parse_metric_constraints(
                    config.get("constraints"),
                    path="supervised_evaluation.constraints",
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TuningValidationError(
                "Unable to construct supervised evaluation configuration.",
                context=TuningErrorContext(
                    component="SupervisedEvaluator", operation="load_config"
                ),
                details={"validation_error": str(exc)},
                cause=exc,
            ) from exc


@dataclass(frozen=True, slots=True)
class SupervisedEvaluationContext:
    x: Any
    y: Any


@dataclass(frozen=True, slots=True)
class CallableSupervisedAdapter:
    """Convenience adapter around three exact callables.

    This is explicit composition, not signature inspection.  The fitter must
    accept model, train X/y, and validation X/y in that order.
    """

    builder: Callable[[Mapping[str, Any], int], Any]
    fitter: Callable[[Any, Any, Any, Any, Any], Mapping[str, Any] | None]
    predictor: Callable[[Any, Any], Any]

    def __post_init__(self) -> None:
        for name in ("builder", "fitter", "predictor"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

    def build(self, parameters: Mapping[str, Any], seed: int) -> Any:
        return self.builder(parameters, seed)

    def fit(
        self,
        model: Any,
        x_train: Any,
        y_train: Any,
        x_validation: Any,
        y_validation: Any,
    ) -> Mapping[str, Any] | None:
        return self.fitter(model, x_train, y_train, x_validation, y_validation)

    def predict(self, model: Any, x_test: Any) -> Any:
        return self.predictor(model, x_test)


class SupervisedEvaluator:
    """Evaluate one candidate over all configured seeds and data splits."""

    def __init__(
        self,
        adapter: SupervisedAdapter,
        metric_functions: Mapping[str, MetricFunction],
        config: SupervisedEvaluationConfig | Mapping[str, Any],
        *,
        split_provider: SplitProvider | None = None,
    ) -> None:
        if not isinstance(adapter, SupervisedAdapter):
            raise TuningContractError(
                "adapter must implement SupervisedAdapter.",
                context=TuningErrorContext(
                    component=self.__class__.__name__, operation="initialize"
                ),
            )
        self.adapter = adapter
        self.config = (
            config
            if isinstance(config, SupervisedEvaluationConfig)
            else SupervisedEvaluationConfig.from_mapping(config)
        )
        if not isinstance(metric_functions, Mapping):
            raise TuningContractError("metric_functions must be a mapping.")
        missing = set(self.config.metrics) - set(metric_functions)
        if missing:
            raise TuningContractError(
                "Metric functions are missing for configured metrics.",
                details={"missing_metrics": sorted(missing)},
            )
        invalid = [name for name, function in metric_functions.items() if not callable(function)]
        if invalid:
            raise TuningContractError(
                "Every metric function must be callable.",
                details={"invalid_metrics": sorted(invalid)},
            )
        self.metric_functions = {
            name: metric_functions[name] for name in self.config.metrics
        }
        if split_provider is not None and not isinstance(split_provider, SplitProvider):
            raise TuningContractError("split_provider must implement SplitProvider.")
        self.split_provider = split_provider

    def evaluate(
        self,
        request: TuningRunRequest,
        trial_id: str,
        parameters: Mapping[str, Any],
        context: SupervisedEvaluationContext,
    ) -> TrialRecord:
        """Return a terminal TrialRecord; runtime failures are retained in it."""

        if not isinstance(request, TuningRunRequest):
            raise TuningContractError("request must be a TuningRunRequest.")
        if not isinstance(context, SupervisedEvaluationContext):
            raise TuningContractError(
                "context must be a SupervisedEvaluationContext."
            )
        validation = validate_parameters_against_space(parameters, request.search_space)
        validation.raise_if_invalid(
            message="Candidate parameters violate the search space.",
            error_cls=TuningSearchSpaceError,
            context=TuningErrorContext(
                run_id=request.run_id,
                trial_id=trial_id,
                component=self.__class__.__name__,
                operation="validate_candidate",
            ),
        )
        if request.objective is not None and request.objective != self.config.objective:
            raise TuningValidationError(
                "Evaluator objective disagrees with the tuning request.",
                context=TuningErrorContext(
                    run_id=request.run_id,
                    trial_id=trial_id,
                    component=self.__class__.__name__,
                    operation="validate_objective",
                ),
            )
        sample_count = self._validate_data(context.x, context.y)
        seeds = tuple(request.seeds) or tuple(self.config.seeds)
        if not seeds:
            raise TuningValidationError("At least one evaluation seed is required.")

        started_at = utc_now()
        trial_started = time.perf_counter()
        evaluations: list[EvaluationSlice] = []
        try:
            for seed in seeds:
                splits = self._splits(sample_count, seed)
                for split in splits:
                    evaluations.append(
                        self._evaluate_split(
                            parameters,
                            context,
                            split,
                            seed,
                            run_id=request.run_id,
                            trial_id=trial_id,
                        )
                    )
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, TuningError)
                else wrap_exception(
                    exc,
                    message="Supervised candidate evaluation failed.",
                    error_cls=TuningEvaluationError,
                    context=TuningErrorContext(
                        run_id=request.run_id,
                        trial_id=trial_id,
                        component=self.__class__.__name__,
                        operation="evaluate",
                    ),
                )
            )
            failed_slice = getattr(error, "evaluation_slice", None)
            if isinstance(failed_slice, EvaluationSlice):
                evaluations.append(failed_slice)
            elif not evaluations or evaluations[-1].status is not TrialStatus.FAILED:
                evaluations.append(
                    EvaluationSlice(
                        status=TrialStatus.FAILED,
                        error=ErrorRecord.from_exception(error),
                        metadata={"stage": "candidate_evaluation"},
                    )
                )
            return TrialRecord(
                trial_id=trial_id,
                run_id=request.run_id,
                status=TrialStatus.FAILED,
                parameters=dict(parameters),
                started_at=started_at,
                completed_at=utc_now(),
                evaluations=tuple(evaluations),
                resources=ResourceUsage(
                    wall_time_seconds=time.perf_counter() - trial_started,
                    sample_count=len(evaluations),
                ),
                error=ErrorRecord.from_exception(error),
            )

        aggregate = self._aggregate(evaluations)
        constraints = tuple(
            constraint.evaluate(aggregate) for constraint in self.config.constraints
        )
        status = (
            TrialStatus.SUCCEEDED
            if all(item.passed for item in constraints)
            else TrialStatus.REJECTED
        )
        latencies = [
            item.resources.wall_time_seconds
            for item in evaluations
            if item.resources is not None and item.resources.wall_time_seconds is not None
        ]
        return TrialRecord(
            trial_id=trial_id,
            run_id=request.run_id,
            status=status,
            parameters=dict(parameters),
            started_at=started_at,
            completed_at=utc_now(),
            metrics=aggregate,
            objective_value=aggregate[self.config.objective.name],
            evaluations=tuple(evaluations),
            constraints=constraints,
            resources=ResourceUsage(
                wall_time_seconds=time.perf_counter() - trial_started,
                latency_quantiles_seconds=self._latency_quantiles(latencies),
                sample_count=len(evaluations),
            ),
            metadata={
                "split_strategy": self.config.split_strategy.value,
                "seed_count": len(seeds),
                "evaluation_count": len(evaluations),
            },
        )

    def _evaluate_split(
        self,
        parameters: Mapping[str, Any],
        context: SupervisedEvaluationContext,
        split: SupervisedSplit,
        seed: int,
        *,
        run_id: str,
        trial_id: str,
    ) -> EvaluationSlice:
        started = time.perf_counter()
        try:
            model = self.adapter.build(parameters, seed)
            if model is None:
                raise TuningContractError("SupervisedAdapter.build returned None.")
            fit_metadata = self.adapter.fit(
                model,
                self._take(context.x, split.train_indices),
                self._take(context.y, split.train_indices),
                self._take(context.x, split.validation_indices),
                self._take(context.y, split.validation_indices),
            )
            if fit_metadata is not None and not isinstance(fit_metadata, Mapping):
                raise TuningContractError(
                    "SupervisedAdapter.fit must return a mapping or None."
                )
            predictions = self.adapter.predict(
                model, self._take(context.x, split.test_indices)
            )
            y_test = self._take(context.y, split.test_indices)
            self._validate_prediction_count(predictions, len(split.test_indices))
            metrics: dict[str, float] = {}
            for name, function in self.metric_functions.items():
                raw_value = function(y_test, predictions)
                if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                    raise TuningContractError(
                        f"Metric {name!r} must return a real scalar."
                    )
                value = float(raw_value)
                if not math.isfinite(value):
                    raise TuningEvaluationError(
                        f"Metric {name!r} returned a non-finite value."
                    )
                metrics[name] = value
            elapsed = time.perf_counter() - started
            return EvaluationSlice(
                status=TrialStatus.SUCCEEDED,
                scenario_id=split.split_id,
                seed=seed,
                metrics=metrics,
                resources=ResourceUsage(
                    wall_time_seconds=elapsed,
                    latency_quantiles_seconds={"p100": elapsed},
                    sample_count=len(split.test_indices),
                ),
                metadata={
                    "train_size": len(split.train_indices),
                    "validation_size": len(split.validation_indices),
                    "test_size": len(split.test_indices),
                    "fit": to_json_safe(fit_metadata or {}),
                    **dict(split.metadata),
                },
            )
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, TuningError)
                else wrap_exception(
                    exc,
                    message=f"Supervised split {split.split_id!r} failed.",
                    error_cls=TuningEvaluationError,
                    context=TuningErrorContext(
                        run_id=run_id,
                        trial_id=trial_id,
                        component=self.__class__.__name__,
                        operation="evaluate_split",
                        scenario_id=split.split_id,
                        seed=seed,
                    ),
                )
            )
            return_slice = EvaluationSlice(
                status=TrialStatus.FAILED,
                scenario_id=split.split_id,
                seed=seed,
                resources=ResourceUsage(
                    wall_time_seconds=time.perf_counter() - started,
                    sample_count=len(split.test_indices),
                ),
                error=ErrorRecord.from_exception(error),
            )
            # Returning then raising would lose the structured slice.  Attach it
            # to the error so the outer boundary can retain it deterministically.
            setattr(error, "evaluation_slice", return_slice)
            raise error

    def _splits(self, sample_count: int, seed: int) -> tuple[SupervisedSplit, ...]:
        if self.split_provider is not None:
            splits = tuple(self.split_provider(sample_count, seed))
            if not splits:
                raise TuningValidationError("split_provider returned no splits.")
            if any(not isinstance(item, SupervisedSplit) for item in splits):
                raise TuningContractError(
                    "split_provider must return SupervisedSplit objects."
                )
            self._validate_split_bounds(splits, sample_count)
            return splits
        if self.config.split_strategy is SupervisedSplitStrategy.HOLDOUT:
            splits = (self._holdout_split(sample_count, seed),)
        elif self.config.split_strategy is SupervisedSplitStrategy.NESTED_K_FOLD:
            splits = self._nested_k_fold_splits(sample_count, seed)
        else:
            splits = self._time_series_splits(sample_count)
        self._validate_split_bounds(splits, sample_count)
        return splits

    def _holdout_split(self, sample_count: int, seed: int) -> SupervisedSplit:
        indices = np.arange(sample_count)
        if self.config.shuffle:
            indices = np.random.default_rng(seed).permutation(indices)
        test_count = max(1, int(round(sample_count * self.config.test_fraction)))
        validation_count = max(
            1, int(round(sample_count * self.config.validation_fraction))
        )
        train_count = sample_count - validation_count - test_count
        if train_count < 1:
            raise TuningValidationError(
                "Dataset is too small for the configured holdout fractions."
            )
        return SupervisedSplit(
            split_id="holdout",
            train_indices=tuple(int(item) for item in indices[:train_count]),
            validation_indices=tuple(
                int(item) for item in indices[train_count : train_count + validation_count]
            ),
            test_indices=tuple(int(item) for item in indices[train_count + validation_count :]),
        )

    def _nested_k_fold_splits(
        self, sample_count: int, seed: int
    ) -> tuple[SupervisedSplit, ...]:
        if sample_count < self.config.n_splits * 3:
            raise TuningValidationError(
                "Nested k-fold evaluation requires at least three observations "
                "per outer fold."
            )
        indices = np.arange(sample_count)
        if self.config.shuffle:
            indices = np.random.default_rng(seed).permutation(indices)
        folds = [
            np.asarray(fold, dtype=int)
            for fold in np.array_split(indices, self.config.n_splits)
        ]
        splits: list[SupervisedSplit] = []
        for fold_index, test in enumerate(folds):
            outer_train = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_index]
            )
            validation_count = max(
                1, int(round(len(outer_train) * self.config.validation_fraction))
            )
            if len(outer_train) - validation_count < 1:
                raise TuningValidationError("Outer fold leaves no training observations.")
            validation = outer_train[:validation_count]
            train = outer_train[validation_count:]
            splits.append(
                SupervisedSplit(
                    split_id=f"outer-fold-{fold_index}",
                    train_indices=tuple(int(item) for item in train),
                    validation_indices=tuple(int(item) for item in validation),
                    test_indices=tuple(int(item) for item in test),
                    metadata={"outer_fold": fold_index},
                )
            )
        return tuple(splits)

    def _time_series_splits(self, sample_count: int) -> tuple[SupervisedSplit, ...]:
        test_count = sample_count // (self.config.n_splits + 1)
        if test_count < 1:
            raise TuningValidationError("Dataset is too small for time-series splits.")
        initial_train = sample_count - self.config.n_splits * test_count
        splits: list[SupervisedSplit] = []
        for split_index in range(self.config.n_splits):
            test_start = initial_train + split_index * test_count
            test_stop = (
                sample_count
                if split_index == self.config.n_splits - 1
                else test_start + test_count
            )
            outer_train = np.arange(test_start)
            validation_count = max(
                1, int(round(len(outer_train) * self.config.validation_fraction))
            )
            if len(outer_train) - validation_count < 1:
                raise TuningValidationError(
                    "Initial time-series window leaves no training observations."
                )
            splits.append(
                SupervisedSplit(
                    split_id=f"time-split-{split_index}",
                    train_indices=tuple(int(item) for item in outer_train[:-validation_count]),
                    validation_indices=tuple(
                        int(item) for item in outer_train[-validation_count:]
                    ),
                    test_indices=tuple(range(test_start, test_stop)),
                    metadata={"temporal_order_preserved": True},
                )
            )
        return tuple(splits)

    @staticmethod
    def _validate_data(x: Any, y: Any) -> int:
        try:
            x_count = len(x)
            y_count = len(y)
        except TypeError as exc:
            raise TuningValidationError("x and y must be sized collections.") from exc
        if x_count != y_count:
            raise TuningValidationError(
                f"x and y have different sample counts ({x_count} vs {y_count})."
            )
        if x_count < 3:
            raise TuningValidationError(
                "At least three observations are required for disjoint partitions."
            )
        return x_count

    @staticmethod
    def _validate_split_bounds(
        splits: Sequence[SupervisedSplit], sample_count: int
    ) -> None:
        for split in splits:
            all_indices = (
                *split.train_indices,
                *split.validation_indices,
                *split.test_indices,
            )
            if max(all_indices) >= sample_count:
                raise TuningValidationError(
                    f"Split {split.split_id!r} contains an out-of-range index."
                )

    @staticmethod
    def _take(values: Any, indices: Sequence[int]) -> Any:
        if hasattr(values, "iloc"):
            return values.iloc[list(indices)]
        index_array = np.asarray(indices, dtype=int)
        try:
            return values[index_array]
        except (TypeError, IndexError):
            return [values[index] for index in indices]

    @staticmethod
    def _validate_prediction_count(predictions: Any, expected: int) -> None:
        try:
            observed = len(predictions)
        except TypeError as exc:
            raise TuningContractError(
                "SupervisedAdapter.predict must return one prediction per test sample."
            ) from exc
        if observed != expected:
            raise TuningContractError(
                "Prediction count differs from the test partition size.",
                details={"expected": expected, "observed": observed},
            )

    @staticmethod
    def _aggregate(evaluations: Sequence[EvaluationSlice]) -> dict[str, float]:
        if not evaluations or any(item.status is not TrialStatus.SUCCEEDED for item in evaluations):
            raise TuningEvaluationError(
                "All supervised evaluations must succeed before aggregation."
            )
        metric_names = set(evaluations[0].metrics)
        if any(set(item.metrics) != metric_names for item in evaluations):
            raise TuningEvaluationError("Metric sets differ across supervised evaluations.")
        aggregate: dict[str, float] = {}
        for name in sorted(metric_names):
            values = np.asarray([item.metrics[name] for item in evaluations], dtype=float)
            aggregate[name] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values, ddof=0))
            seeds = sorted(
                {item.seed for item in evaluations},
                key=lambda seed: (seed is None, seed if seed is not None else 0),
            )
            seed_means = np.asarray(
                [
                    np.mean(
                        [
                            item.metrics[name]
                            for item in evaluations
                            if item.seed == seed
                        ]
                    )
                    for seed in seeds
                ],
                dtype=float,
            )
            aggregate[f"{name}_seed_std"] = float(np.std(seed_means, ddof=0))
            aggregate[f"{name}_seed_sem"] = float(
                np.std(seed_means, ddof=0) / math.sqrt(len(seed_means))
            )
        aggregate["evaluation_count"] = float(len(evaluations))
        return aggregate

    @staticmethod
    def _latency_quantiles(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=float)
        return {
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95)),
            "p100": float(np.max(array)),
        }


__all__ = [
    "CallableSupervisedAdapter",
    "SupervisedEvaluationConfig",
    "SupervisedEvaluationContext",
    "SupervisedEvaluator",
    "SupervisedSplitStrategy",
]

if __name__ == "__main__":
    print("\n=== Running Supervised Evaluator Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting supervised evaluator tests", "info")

    _failures: list[str] = []

    def _check(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def _run_test(name: str, test: Callable[[], None]) -> None:
        try:
            test()
            printer.status("TEST", name, "success")
        except Exception as exc:
            _failures.append(f"{name}: {type(exc).__name__}: {exc}")
            printer.status("TEST", _failures[-1], "error")

    _objective = MetricSpec("rmse", ObjectiveDirection.MINIMIZE)
    _config = SupervisedEvaluationConfig.from_mapping(
        {
            "objective": _objective.to_dict(),
            "metrics": ["rmse"],
            "seeds": [3, 5],
            "split_strategy": "holdout",
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "n_splits": 2,
            "shuffle": True,
            "constraints": [],
        }
    )
    _request = TuningRunRequest(
        run_id="supervised-self-test",
        settings=TunerSettings(
            strategy=TuningStrategy.GRID,
            model_type="SelfTestModel",
            allow_generate=False,
        ),
        config={},
        strategy_config={"fail_fast": False},
        search_space=(
            {"name": "scale", "type": "real", "values": [1.0]},
        ),
        config_fingerprint="supervised-self-test",
        objective=_objective,
        seeds=(3, 5),
    )
    _x = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    _y = 2.5 * _x[:, 0] - 0.75

    def _builder(parameters: Mapping[str, Any], seed: int) -> dict[str, Any]:
        return {"seed": seed, "scale": parameters["scale"]}

    def _fitter(
        model: dict[str, Any],
        x_train: Any,
        y_train: Any,
        x_validation: Any,
        y_validation: Any,
    ) -> Mapping[str, Any]:
        design = np.column_stack(
            (np.asarray(x_train, dtype=float).reshape(-1), np.ones(len(x_train)))
        )
        coefficients, *_ = np.linalg.lstsq(
            design, np.asarray(y_train, dtype=float), rcond=None
        )
        model["slope"] = float(coefficients[0])
        model["intercept"] = float(coefficients[1])
        return {"validation_samples": len(x_validation)}

    def _predictor(model: Mapping[str, Any], x_test: Any) -> np.ndarray:
        values = np.asarray(x_test, dtype=float).reshape(-1)
        return model["slope"] * values + model["intercept"]

    def _rmse(y_true: Any, y_pred: Any) -> float:
        difference = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean(difference**2)))

    def _test_multiseed_evaluation() -> None:
        adapter = CallableSupervisedAdapter(_builder, _fitter, _predictor)
        evaluator = SupervisedEvaluator(adapter, {"rmse": _rmse}, _config)
        trial = evaluator.evaluate(
            _request,
            "supervised-trial",
            {"scale": 1.0},
            SupervisedEvaluationContext(_x, _y),
        )
        _check(trial.status is TrialStatus.SUCCEEDED, "evaluation did not succeed")
        _check(len(trial.evaluations) == 2, "not every configured seed was evaluated")
        _check(trial.metrics["evaluation_count"] == 2.0, "wrong evaluation count")
        _check(trial.metrics["rmse"] < 1.0e-10, "linear fit is unexpectedly inaccurate")
        _check(trial.objective_value == trial.metrics["rmse"], "objective mismatch")

    def _test_prediction_contract_failure() -> None:
        adapter = CallableSupervisedAdapter(
            _builder,
            _fitter,
            lambda model, x_test: np.zeros(max(0, len(x_test) - 1)),
        )
        evaluator = SupervisedEvaluator(adapter, {"rmse": _rmse}, _config)
        trial = evaluator.evaluate(
            _request,
            "supervised-failure",
            {"scale": 1.0},
            SupervisedEvaluationContext(_x, _y),
        )
        _check(trial.status is TrialStatus.FAILED, "bad prediction count was accepted")
        _check(trial.error is not None, "failed trial has no error evidence")

    def _test_time_series_shuffle_rejected() -> None:
        invalid = {
            "objective": _objective.to_dict(),
            "metrics": ["rmse"],
            "seeds": [3],
            "split_strategy": "time_series",
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "n_splits": 3,
            "shuffle": True,
            "constraints": [],
        }
        try:
            SupervisedEvaluationConfig.from_mapping(invalid)
        except TuningValidationError:
            return
        raise AssertionError("time-series shuffling was accepted")

    _run_test("multi-seed leakage-resistant evaluation", _test_multiseed_evaluation)
    _run_test("prediction contract failure retention", _test_prediction_contract_failure)
    _run_test("time-series ordering invariant", _test_time_series_shuffle_rejected)

    _all_passed = not _failures
    printer.status(
        "",
        f"{3 - len(_failures)}/3 supervised evaluator tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All supervised evaluator tests passed ===\n")