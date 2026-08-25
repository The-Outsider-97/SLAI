"""Sequential Bayesian optimization for SLAI tuning trials.

The strategy owns candidate selection only.  It never constructs models,
splits datasets, mutates agents, promotes candidates, writes reports, or
interprets task-specific metrics.  Those responsibilities remain with the
evaluator, promotion policy, and artifact writer.

``evaluation_context`` must be one of:

* a callable accepting ``(request, trial_id, parameters)``;
* an object exposing an exact ``evaluate_candidate`` method with that
  signature; or
* a mapping containing such a callable under ``candidate_evaluator``.

The callable returns a terminal :class:`TrialRecord`.  Scalar-only evaluator
callbacks are intentionally unsupported because they cannot preserve failure,
constraint, resource, or agent-state evidence.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import math
import warnings
import numpy as np

from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ..tuning_types import *
from ..tuning_validation import *
from ..utils.tuning_errors import *
from ..utils.tuning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Bayesian Search")
printer = PrettyPrinter()


CandidateEvaluator = Callable[
    [TuningRunRequest, str, Mapping[str, Any]],
    TrialRecord | Awaitable[TrialRecord],
]


@dataclass(frozen=True, slots=True)
class BayesianSearchSettings:
    """Validated strategy settings with no evaluation or persistence policy."""

    n_trials: int
    n_initial_points: int
    random_state: int | None
    candidate_pool_size: int = 1024
    exploration: float = 0.01
    normalize_y: bool = True
    noise: float = 1.0e-6
    matern_nu: float = 2.5
    optimizer_restarts: int = 2
    fail_fast: bool = False

    def __post_init__(self) -> None:
        for name in (
            "n_trials",
            "n_initial_points",
            "candidate_pool_size",
            "optimizer_restarts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if not 1 <= self.n_initial_points <= self.n_trials:
            raise ValueError("n_initial_points must be within [1, n_trials]")
        if self.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1")
        if self.optimizer_restarts < 0:
            raise ValueError("optimizer_restarts must be non-negative")
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
        ):
            raise TypeError("random_state must be an integer or None")
        for name in ("exploration", "noise", "matern_nu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.exploration < 0:
            raise ValueError("exploration must be non-negative")
        if self.noise <= 0:
            raise ValueError("noise must be positive")
        if self.matern_nu not in {0.5, 1.5, 2.5, math.inf}:
            raise ValueError("matern_nu must be 0.5, 1.5, 2.5, or infinity")
        if not isinstance(self.normalize_y, bool):
            raise TypeError("normalize_y must be a bool")
        if not isinstance(self.fail_fast, bool):
            raise TypeError("fail_fast must be a bool")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "BayesianSearchSettings":
        if not isinstance(config, Mapping):
            raise TuningValidationError("bayesian_search must be a mapping.")
        acquisition = config.get("acquisition", {})
        surrogate = config.get("surrogate", {})
        if not isinstance(acquisition, Mapping):
            raise TuningValidationError("bayesian_search.acquisition must be a mapping.")
        if not isinstance(surrogate, Mapping):
            raise TuningValidationError("bayesian_search.surrogate must be a mapping.")
        name = str(acquisition.get("name", "expected_improvement")).casefold()
        if name != "expected_improvement":
            raise TuningValidationError(
                "Only expected_improvement is currently implemented.",
                details={"configured_acquisition": name},
            )
        kernel = str(surrogate.get("kernel", "matern")).casefold()
        if kernel != "matern":
            raise TuningValidationError(
                "Only the Matern surrogate kernel is currently implemented.",
                details={"configured_kernel": kernel},
            )
        try:
            random_state = config.get("random_state")
            return cls(
                n_trials=config["n_trials"],
                n_initial_points=config["n_initial_points"],
                random_state=random_state,
                candidate_pool_size=config.get("candidate_pool_size", 1024),
                exploration=acquisition.get("exploration", 0.01),
                normalize_y=coerce_bool(
                    surrogate.get("normalize_y", True), name="normalize_y"
                ),
                noise=surrogate.get("noise", 1.0e-6),
                matern_nu=surrogate.get("matern_nu", 2.5),
                optimizer_restarts=surrogate.get("optimizer_restarts", 2),
                fail_fast=coerce_bool(
                    config.get("fail_fast", False), name="fail_fast"
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TuningValidationError(
                "Invalid Bayesian search configuration.",
                details={"validation_error": str(exc)},
                cause=exc,
            ) from exc


@dataclass(frozen=True, slots=True)
class _Dimension:
    name: str
    kind: str
    values: tuple[Any, ...] | None
    lower: float | int | None
    upper: float | int | None
    prior: str

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "_Dimension":
        name = str(spec["name"]).strip()
        kind = str(spec["type"]).strip().casefold()
        values = tuple(spec["values"]) if "values" in spec else None
        return cls(
            name=name,
            kind=kind,
            values=values,
            lower=None if values is not None else spec.get("min"),
            upper=None if values is not None else spec.get("max"),
            prior=str(spec.get("prior", "uniform")).strip().casefold(),
        )

    def sample(self, rng: np.random.Generator) -> Any:
        if self.values is not None:
            return self.values[int(rng.integers(0, len(self.values)))]
        assert self.lower is not None and self.upper is not None
        if self.kind == "integer":
            return int(rng.integers(int(self.lower), int(self.upper) + 1))
        if self.prior == "log-uniform":
            low = math.log(float(self.lower))
            high = math.log(float(self.upper))
            return float(math.exp(rng.uniform(low, high)))
        return float(rng.uniform(float(self.lower), float(self.upper)))

    @property
    def encoded_width(self) -> int:
        if self.kind == "categorical":
            assert self.values is not None
            return len(self.values)
        return 1

    def encode(self, value: Any) -> list[float]:
        if self.kind == "categorical":
            assert self.values is not None
            observed = stable_fingerprint(value)
            return [
                1.0 if stable_fingerprint(item) == observed else 0.0
                for item in self.values
            ]
        if self.values is not None:
            fingerprints = [stable_fingerprint(item) for item in self.values]
            index = fingerprints.index(stable_fingerprint(value))
            denominator = max(1, len(self.values) - 1)
            return [index / denominator]
        assert self.lower is not None and self.upper is not None
        lower = float(self.lower)
        upper = float(self.upper)
        numeric = float(value)
        if self.prior == "log-uniform":
            lower, upper, numeric = math.log(lower), math.log(upper), math.log(numeric)
        denominator = upper - lower
        return [0.0 if denominator == 0 else (numeric - lower) / denominator]


class BayesianSearch:
    """Gaussian-process search with expected-improvement acquisition."""

    def __init__(
        self,
        request: TuningRunRequest,
        evaluation_context: Any,
    ) -> None:
        if not isinstance(request, TuningRunRequest):
            raise TuningContractError("request must be a TuningRunRequest.")
        if request.settings.strategy is not TuningStrategy.BAYESIAN:
            raise TuningContractError("BayesianSearch received a non-Bayesian request.")
        report = validate_search_space(request.search_space, TuningStrategy.BAYESIAN)
        report.raise_if_invalid(
            message="Invalid Bayesian search space.",
            error_cls=TuningSearchSpaceError,
            context=self._context(request, "validate_search_space"),
        )
        self.request = request
        self.settings = BayesianSearchSettings.from_mapping(request.strategy_config)
        self.objective = self._resolve_objective(request)
        self.evaluator = self._resolve_evaluator(evaluation_context)
        self.dimensions = tuple(_Dimension.from_spec(item) for item in request.search_space)
        self.rng = np.random.default_rng(self.settings.random_state)
        self._seen: set[str] = set()
        self._surrogate_warning_count = 0

    def run(self) -> SearchResult:
        started_at = utc_now()
        trials: list[TrialRecord] = []
        search_warnings: list[str] = []

        for index in range(self.settings.n_trials):
            parameters = self._select_candidate(trials, index)
            if parameters is None:
                search_warnings.append(
                    "Search space was exhausted before n_trials was reached."
                )
                break
            trial_id = f"bayesian-{index + 1:05d}"
            trial = self._evaluate(parameters, trial_id)
            trials.append(trial)
            if self.settings.fail_fast and trial.status is TrialStatus.FAILED:
                search_warnings.append("Bayesian search stopped after a failed trial.")
                break

        best = self._best_trial(trials)
        status = self._status(trials, best)
        if self._surrogate_warning_count:
            search_warnings.append(
                "The Gaussian-process optimizer emitted convergence warnings; "
                "candidate evidence remains valid."
            )
        return SearchResult(
            run_id=self.request.run_id,
            strategy=TuningStrategy.BAYESIAN,
            status=status,
            objective=self.objective,
            trials=tuple(trials),
            started_at=started_at,
            completed_at=utc_now(),
            best_trial_id=None if best is None else best.trial_id,
            warnings=tuple(search_warnings),
            metadata={
                "requested_trials": self.settings.n_trials,
                "evaluated_trials": len(trials),
                "initial_random_trials": min(
                    self.settings.n_initial_points, len(trials)
                ),
                "acquisition": "expected_improvement",
                "surrogate": "gaussian_process_matern",
                "surrogate_warning_count": self._surrogate_warning_count,
            },
        )

    def _select_candidate(
        self, trials: Sequence[TrialRecord], index: int
    ) -> dict[str, Any] | None:
        eligible = [
            trial
            for trial in trials
            if trial.eligible_for_promotion and trial.objective_value is not None
        ]
        if index < self.settings.n_initial_points or len(eligible) < 2:
            return self._unseen_random_candidate()

        pool = self._candidate_pool()
        if not pool:
            return None
        x_train = np.asarray(
            [self._encode(trial.parameters) for trial in eligible], dtype=float
        )
        raw_objectives = np.asarray(
            [
                float(trial.objective_value)
                for trial in eligible
                if trial.objective_value is not None
            ],
            dtype=float,
        )
        y_train = (
            raw_objectives
            if self.objective.direction is ObjectiveDirection.MINIMIZE
            else -raw_objectives
        )
        model = self._fit_surrogate(x_train, y_train)
        x_pool = np.asarray([self._encode(item) for item in pool], dtype=float)
        mean, std = model.predict(x_pool, return_std=True)
        acquisition = self._expected_improvement(
            mean=np.asarray(mean, dtype=float),
            std=np.asarray(std, dtype=float),
            incumbent=float(np.min(y_train)),
            exploration=self.settings.exploration,
        )
        selected = pool[int(np.argmax(acquisition))]
        self._seen.add(stable_fingerprint(selected))
        return selected

    def _unseen_random_candidate(self) -> dict[str, Any] | None:
        for _ in range(max(100, self.settings.candidate_pool_size * 4)):
            candidate = {
                dimension.name: dimension.sample(self.rng)
                for dimension in self.dimensions
            }
            fingerprint = stable_fingerprint(candidate)
            if fingerprint not in self._seen:
                self._seen.add(fingerprint)
                return candidate
        # Random retries are not evidence that a finite space is exhausted.
        # Fall back to its canonical Cartesian order to make exhaustion exact.
        for candidate in self._finite_candidates():
            fingerprint = stable_fingerprint(candidate)
            if fingerprint not in self._seen:
                self._seen.add(fingerprint)
                return candidate
        return None

    def _candidate_pool(self) -> list[dict[str, Any]]:
        pool: list[dict[str, Any]] = []
        local_seen: set[str] = set()
        attempts = max(100, self.settings.candidate_pool_size * 8)
        for _ in range(attempts):
            candidate = {
                dimension.name: dimension.sample(self.rng)
                for dimension in self.dimensions
            }
            fingerprint = stable_fingerprint(candidate)
            if fingerprint in self._seen or fingerprint in local_seen:
                continue
            local_seen.add(fingerprint)
            pool.append(candidate)
            if len(pool) >= self.settings.candidate_pool_size:
                break
        if len(pool) < self.settings.candidate_pool_size:
            for candidate in self._finite_candidates():
                fingerprint = stable_fingerprint(candidate)
                if fingerprint in self._seen or fingerprint in local_seen:
                    continue
                local_seen.add(fingerprint)
                pool.append(candidate)
                if len(pool) >= self.settings.candidate_pool_size:
                    break
        return pool

    def _finite_candidates(self) -> Iterator[dict[str, Any]]:
        """Yield a discrete space exactly; yield nothing for bounded continua."""

        value_sets: list[tuple[Any, ...]] = []
        for dimension in self.dimensions:
            if dimension.values is None:
                return
            value_sets.append(dimension.values)
        for values in itertools.product(*value_sets):
            yield {
                dimension.name: value
                for dimension, value in zip(
                    self.dimensions, values, strict=True
                )
            }

    def _fit_surrogate(self, x: np.ndarray, y: np.ndarray) -> Any:
        try:
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        except ImportError as exc:
            raise TuningDependencyError(
                "Bayesian search requires scikit-learn.",
                context=self._context(self.request, "load_surrogate"),
                cause=exc,
            ) from exc

        kernel = ConstantKernel(1.0, (1.0e-3, 1.0e3)) * Matern(
            length_scale=np.ones(x.shape[1], dtype=float),
            length_scale_bounds=(1.0e-3, 1.0e3),
            nu=self.settings.matern_nu,
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.settings.noise,
            normalize_y=self.settings.normalize_y,
            n_restarts_optimizer=self.settings.optimizer_restarts,
            random_state=self.settings.random_state,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                model.fit(x, y)
            except Exception as exc:
                raise TuningOptimizationError(
                    "Gaussian-process surrogate fitting failed.",
                    context=self._context(self.request, "fit_surrogate"),
                    details={
                        "observations": int(x.shape[0]),
                        "encoded_dimensions": int(x.shape[1]),
                    },
                    cause=exc,
                ) from exc
        self._surrogate_warning_count += sum(
            issubclass(item.category, ConvergenceWarning) for item in captured
        )
        return model

    @staticmethod
    def _expected_improvement(
        *,
        mean: np.ndarray,
        std: np.ndarray,
        incumbent: float,
        exploration: float,
    ) -> np.ndarray:
        improvement = incumbent - mean - exploration
        safe_std = np.maximum(std, np.finfo(float).eps)
        z = improvement / safe_std
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        density = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        expected = improvement * cdf + safe_std * density
        return np.where(std > np.finfo(float).eps, expected, 0.0)

    def _encode(self, parameters: Mapping[str, Any]) -> list[float]:
        encoded: list[float] = []
        for dimension in self.dimensions:
            encoded.extend(dimension.encode(parameters[dimension.name]))
        return encoded

    def _evaluate(
        self, parameters: Mapping[str, Any], trial_id: str
    ) -> TrialRecord:
        started_at = utc_now()
        try:
            value = self.evaluator(self.request, trial_id, dict(parameters))
            if inspect.isawaitable(value):
                async def _await_value() -> TrialRecord:
                    return await cast(Awaitable[TrialRecord], value)

                value = asyncio.run(_await_value())
            if not isinstance(value, TrialRecord):
                raise TuningContractError(
                    "Candidate evaluator must return TrialRecord.",
                    details={"actual_type": type(value).__name__},
                )
            self._validate_evaluator_record(value, trial_id, parameters)
            return value
        except Exception as exc:
            error = wrap_exception(
                exc,
                message="Bayesian candidate evaluation failed.",
                error_cls=TuningEvaluationError,
                context=self._context(
                    self.request,
                    "evaluate_candidate",
                    trial_id=trial_id,
                    parameters=dict(parameters),
                ),
            )
            return TrialRecord(
                trial_id=trial_id,
                run_id=self.request.run_id,
                status=TrialStatus.FAILED,
                parameters=dict(parameters),
                started_at=started_at,
                completed_at=utc_now(),
                error=ErrorRecord.from_exception(error),
                metadata={"search_strategy": "bayesian"},
            )

    def _validate_evaluator_record(
        self,
        trial: TrialRecord,
        trial_id: str,
        parameters: Mapping[str, Any],
    ) -> None:
        if trial.run_id != self.request.run_id or trial.trial_id != trial_id:
            raise TuningContractError(
                "Candidate evaluator returned mismatched run or trial identity."
            )
        if stable_fingerprint(trial.parameters) != stable_fingerprint(parameters):
            raise TuningContractError(
                "Candidate evaluator changed the candidate parameters."
            )
        report = validate_trial_record(trial, self.objective)
        report.raise_if_invalid(
            message="Candidate evaluator returned an invalid trial record.",
            error_cls=TuningContractError,
            context=self._context(
                self.request, "validate_trial", trial_id=trial_id
            ),
        )

    def _best_trial(self, trials: Sequence[TrialRecord]) -> TrialRecord | None:
        eligible = [trial for trial in trials if trial.eligible_for_promotion]
        if not eligible:
            return None
        def objective_value(trial: TrialRecord) -> float:
            assert trial.objective_value is not None
            return float(trial.objective_value)

        return (
            min(eligible, key=objective_value)
            if self.objective.direction is ObjectiveDirection.MINIMIZE
            else max(eligible, key=objective_value)
        )

    @staticmethod
    def _status(
        trials: Sequence[TrialRecord], best: TrialRecord | None
    ) -> RunStatus:
        if best is None:
            return RunStatus.FAILED
        return (
            RunStatus.DEGRADED
            if any(trial.status is TrialStatus.FAILED for trial in trials)
            else RunStatus.SUCCEEDED
        )

    @staticmethod
    def _resolve_evaluator(evaluation_context: Any) -> CandidateEvaluator:
        if callable(evaluation_context):
            return cast(CandidateEvaluator, evaluation_context)
        method = getattr(evaluation_context, "evaluate_candidate", None)
        if callable(method):
            return cast(CandidateEvaluator, method)
        if isinstance(evaluation_context, Mapping):
            candidate = evaluation_context.get("candidate_evaluator")
            if callable(candidate):
                return cast(CandidateEvaluator, candidate)
        raise TuningContractError(
            "evaluation_context must expose an exact candidate evaluator contract."
        )

    @staticmethod
    def _resolve_objective(request: TuningRunRequest) -> MetricSpec:
        if request.objective is not None:
            return request.objective
        raw = request.strategy_config.get("objective")
        if raw is None:
            raise TuningValidationError(
                "Bayesian search requires an explicit objective mapping."
            )
        return parse_metric_spec(raw, path="bayesian_search.objective")

    @staticmethod
    def _context(
        request: TuningRunRequest,
        operation: str,
        *,
        trial_id: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> TuningErrorContext:
        return TuningErrorContext(
            run_id=request.run_id,
            trial_id=trial_id,
            component="BayesianSearch",
            operation=operation,
            strategy=TuningStrategy.BAYESIAN.value,
            model_type=request.settings.model_type,
            parameters=dict(parameters or {}),
        )


def run_search(
    request: TuningRunRequest, evaluation_context: Any
) -> SearchResult:
    """Default tuner entry point."""

    return BayesianSearch(request, evaluation_context).run()


__all__ = ["BayesianSearch", "BayesianSearchSettings", "run_search"]

if __name__ == "__main__":
    print("\n=== Running Bayesian Search Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting Bayesian search tests", "info")

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

    _objective = MetricSpec("loss", ObjectiveDirection.MINIMIZE)
    _request = TuningRunRequest(
        run_id="bayesian-self-test",
        settings=TunerSettings(
            strategy=TuningStrategy.BAYESIAN,
            model_type="SelfTestModel",
            allow_generate=False,
        ),
        config={},
        strategy_config={
            "n_trials": 3,
            "n_initial_points": 2,
            "random_state": 17,
            "candidate_pool_size": 8,
            "fail_fast": False,
            "acquisition": {
                "name": "expected_improvement",
                "exploration": 0.01,
            },
            "surrogate": {
                "kernel": "matern",
                "matern_nu": 2.5,
                "normalize_y": True,
                "noise": 1.0e-6,
                "optimizer_restarts": 0,
            },
        },
        search_space=(
            {"name": "x", "type": "integer", "values": [0, 1, 2]},
        ),
        config_fingerprint="bayesian-self-test",
        objective=_objective,
    )

    def _evaluate(
        request: TuningRunRequest,
        trial_id: str,
        parameters: Mapping[str, Any],
    ) -> TrialRecord:
        started = utc_now()
        loss = float((int(parameters["x"]) - 1) ** 2)
        return TrialRecord(
            trial_id=trial_id,
            run_id=request.run_id,
            status=TrialStatus.SUCCEEDED,
            parameters=dict(parameters),
            started_at=started,
            completed_at=utc_now(),
            metrics={"loss": loss},
            objective_value=loss,
        )

    def _test_finite_space_search() -> None:
        result = BayesianSearch(_request, _evaluate).run()
        _check(result.status is RunStatus.SUCCEEDED, "search did not succeed")
        _check(len(result.trials) == 3, "finite space was not evaluated exactly")
        _check(
            {trial.parameters["x"] for trial in result.trials} == {0, 1, 2},
            "candidate de-duplication or exhaustion is incorrect",
        )
        _check(result.best_trial is not None, "best trial is missing")
        assert result.best_trial is not None
        _check(result.best_trial.parameters["x"] == 1, "wrong best candidate")

    def _test_expected_improvement() -> None:
        expected = BayesianSearch._expected_improvement(
            mean=np.asarray([0.5, 1.0]),
            std=np.asarray([0.2, 0.0]),
            incumbent=0.75,
            exploration=0.01,
        )
        _check(bool(np.isfinite(expected).all()), "acquisition produced non-finite values")
        _check(bool((expected >= 0.0).all()), "expected improvement became negative")
        _check(expected[1] == 0.0, "zero-variance acquisition must be zero")

    def _test_failed_evaluation_record() -> None:
        def _failing_evaluator(
            request: TuningRunRequest,
            trial_id: str,
            parameters: Mapping[str, Any],
        ) -> TrialRecord:
            raise RuntimeError("intentional self-test failure")

        search = BayesianSearch(_request, _failing_evaluator)
        trial = search._evaluate({"x": 0}, "bayesian-failure")
        _check(trial.status is TrialStatus.FAILED, "failure evidence was discarded")
        _check(trial.error is not None, "failed trial has no error record")

    _run_test("finite-space Gaussian-process search", _test_finite_space_search)
    _run_test("expected-improvement numerics", _test_expected_improvement)
    _run_test("candidate failure retention", _test_failed_evaluation_record)

    _all_passed = not _failures
    printer.status(
        "",
        f"{3 - len(_failures)}/3 Bayesian search tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All Bayesian search tests passed ===\n")