"""Deterministic exhaustive grid search for SLAI tuning trials.

Grid search enumerates the validated Cartesian product in configuration order.
It delegates all model and agent behavior to the same exact candidate-evaluator
contract used by the Bayesian strategy and returns the common ``SearchResult``
representation.  Candidate failures remain first-class trial records and are
never removed from aggregate evidence.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import math

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from ..tuning_types import *
from ..tuning_validation import *
from ..utils.tuning_errors import *
from ..utils.tuning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Grid Search")
printer = PrettyPrinter()


CandidateEvaluator = Callable[
    [TuningRunRequest, str, Mapping[str, Any]],
    TrialRecord | Awaitable[TrialRecord],
]


@dataclass(frozen=True, slots=True)
class GridSearchSettings:
    """Runtime behavior that does not alter grid membership or ordering."""

    fail_fast: bool = False
    max_combinations: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fail_fast, bool):
            raise TypeError("fail_fast must be a bool")
        if self.max_combinations is not None:
            if isinstance(self.max_combinations, bool) or not isinstance(
                self.max_combinations, int
            ):
                raise TypeError("max_combinations must be an integer or None")
            if self.max_combinations < 1:
                raise ValueError("max_combinations must be positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "GridSearchSettings":
        if not isinstance(config, Mapping):
            raise TuningValidationError("grid_search must be a mapping.")
        try:
            maximum = config.get("max_combinations")
            return cls(
                fail_fast=coerce_bool(
                    config.get("fail_fast", False), name="fail_fast"
                ),
                max_combinations=maximum,
            )
        except (TypeError, ValueError) as exc:
            raise TuningValidationError(
                "Invalid grid search configuration.",
                details={"validation_error": str(exc)},
                cause=exc,
            ) from exc


class GridSearch:
    """Enumerate and evaluate every declared grid point exactly once."""

    def __init__(
        self,
        request: TuningRunRequest,
        evaluation_context: Any,
    ) -> None:
        if not isinstance(request, TuningRunRequest):
            raise TuningContractError("request must be a TuningRunRequest.")
        if request.settings.strategy is not TuningStrategy.GRID:
            raise TuningContractError("GridSearch received a non-grid request.")
        report = validate_search_space(request.search_space, TuningStrategy.GRID)
        report.raise_if_invalid(
            message="Invalid grid search space.",
            error_cls=TuningSearchSpaceError,
            context=self._context(request, "validate_search_space"),
        )
        self.request = request
        self.settings = GridSearchSettings.from_mapping(request.strategy_config)
        self.objective = self._resolve_objective(request)
        self.evaluator = self._resolve_evaluator(evaluation_context)
        self.parameter_names, self.parameter_values = self._compile_grid(
            request.search_space
        )
        self.total_combinations = math.prod(
            len(values) for values in self.parameter_values
        )
        if (
            self.settings.max_combinations is not None
            and self.total_combinations > self.settings.max_combinations
        ):
            raise TuningSearchSpaceError(
                "Grid cardinality exceeds max_combinations.",
                context=self._context(request, "validate_cardinality"),
                details={
                    "total_combinations": self.total_combinations,
                    "max_combinations": self.settings.max_combinations,
                },
            )

    def run(self) -> SearchResult:
        started_at = utc_now()
        trials: list[TrialRecord] = []
        search_warnings: list[str] = []

        combinations = itertools.product(*self.parameter_values)
        for index, values in enumerate(combinations, start=1):
            parameters = dict(zip(self.parameter_names, values, strict=True))
            trial_id = f"grid-{index:05d}"
            trial = self._evaluate(parameters, trial_id)
            trials.append(trial)
            if self.settings.fail_fast and trial.status is TrialStatus.FAILED:
                search_warnings.append("Grid search stopped after a failed trial.")
                break

        best = self._best_trial(trials)
        completed_exhaustively = len(trials) == self.total_combinations
        return SearchResult(
            run_id=self.request.run_id,
            strategy=TuningStrategy.GRID,
            status=self._status(trials, best),
            objective=self.objective,
            trials=tuple(trials),
            started_at=started_at,
            completed_at=utc_now(),
            best_trial_id=None if best is None else best.trial_id,
            warnings=tuple(search_warnings),
            metadata={
                "total_combinations": self.total_combinations,
                "evaluated_combinations": len(trials),
                "completed_exhaustively": completed_exhaustively,
                "parameter_order": list(self.parameter_names),
            },
        )

    @classmethod
    def _compile_grid(
        cls, search_space: Sequence[Mapping[str, Any]]
    ) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
        names: list[str] = []
        value_sets: list[tuple[Any, ...]] = []
        for spec in search_space:
            name = str(spec["name"]).strip()
            kind = str(spec["type"]).strip().casefold()
            if "values" in spec:
                values = tuple(spec["values"])
            elif kind == "integer":
                step = spec["step"]
                if isinstance(step, bool) or not isinstance(step, int):
                    raise TuningSearchSpaceError(
                        f"Integer grid step for {name!r} must be an integer."
                    )
                values = tuple(
                    range(int(spec["min"]), int(spec["max"]) + 1, step)
                )
            else:
                if str(spec.get("prior", "uniform")).casefold() != "uniform":
                    raise TuningSearchSpaceError(
                        f"Real grid parameter {name!r} with bounded steps must use "
                        "a uniform prior; use explicit values for a logarithmic grid."
                    )
                values = cls._decimal_range(
                    spec["min"], spec["max"], spec["step"]
                )
            if not values:
                raise TuningSearchSpaceError(
                    f"Grid parameter {name!r} produced no values."
                )
            names.append(name)
            value_sets.append(values)
        return tuple(names), tuple(value_sets)

    @staticmethod
    def _decimal_range(lower: Any, upper: Any, step: Any) -> tuple[float, ...]:
        low = Decimal(str(lower))
        high = Decimal(str(upper))
        increment = Decimal(str(step))
        if increment <= 0:
            raise TuningSearchSpaceError("Real grid step must be positive.")
        values: list[float] = []
        current = low
        while current <= high:
            values.append(float(current))
            current += increment
        return tuple(values)

    def _evaluate(
        self, parameters: Mapping[str, Any], trial_id: str
    ) -> TrialRecord:
        started_at = utc_now()
        try:
            value = self.evaluator(self.request, trial_id, dict(parameters))
            if inspect.isawaitable(value):
                async def resolve(awaitable: Awaitable[TrialRecord]) -> TrialRecord:
                    return await awaitable

                value = asyncio.run(resolve(value))
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
                message="Grid candidate evaluation failed.",
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
                metadata={"search_strategy": "grid"},
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
                "Grid search requires an explicit objective mapping."
            )
        return parse_metric_spec(raw, path="grid_search.objective")

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
            component="GridSearch",
            operation=operation,
            strategy=TuningStrategy.GRID.value,
            model_type=request.settings.model_type,
            parameters=dict(parameters or {}),
        )


def run_search(
    request: TuningRunRequest, evaluation_context: Any
) -> SearchResult:
    """Default tuner entry point."""

    return GridSearch(request, evaluation_context).run()


__all__ = ["GridSearch", "GridSearchSettings", "run_search"]

if __name__ == "__main__":
    print("\n=== Running Grid Search Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting grid search tests", "info")

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

    def _request(
        search_space: Sequence[Mapping[str, Any]],
        *,
        run_id: str = "grid-self-test",
    ) -> TuningRunRequest:
        return TuningRunRequest(
            run_id=run_id,
            settings=TunerSettings(
                strategy=TuningStrategy.GRID,
                model_type="SelfTestModel",
                allow_generate=False,
            ),
            config={},
            strategy_config={"fail_fast": False, "max_combinations": None},
            search_space=search_space,
            config_fingerprint="grid-self-test",
            objective=_objective,
        )

    def _evaluate(
        request: TuningRunRequest,
        trial_id: str,
        parameters: Mapping[str, Any],
    ) -> TrialRecord:
        started = utc_now()
        loss = float(abs(int(parameters["x"]) - 1))
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

    def _test_order_and_cardinality() -> None:
        request = _request(
            (
                {"name": "x", "type": "integer", "values": [2, 1]},
                {"name": "mode", "type": "categorical", "values": ["b", "a"]},
            )
        )
        result = GridSearch(request, _evaluate).run()
        observed = [
            (trial.parameters["x"], trial.parameters["mode"])
            for trial in result.trials
        ]
        _check(
            observed == [(2, "b"), (2, "a"), (1, "b"), (1, "a")],
            "Cartesian configuration order was not preserved",
        )
        _check(result.metadata["total_combinations"] == 4, "wrong cardinality")
        _check(result.best_trial is not None, "best trial is missing")
        assert result.best_trial is not None
        _check(result.best_trial.parameters["x"] == 1, "wrong best candidate")

    def _test_decimal_grid() -> None:
        values = GridSearch._decimal_range(0.0, 0.3, 0.1)
        _check(values == (0.0, 0.1, 0.2, 0.3), "decimal expansion drifted")

    def _test_failure_retention() -> None:
        request = _request(
            ({"name": "x", "type": "integer", "values": [0, 1]},),
            run_id="grid-failure-self-test",
        )

        def _partly_failing(
            run_request: TuningRunRequest,
            trial_id: str,
            parameters: Mapping[str, Any],
        ) -> TrialRecord:
            if parameters["x"] == 0:
                raise RuntimeError("intentional self-test failure")
            return _evaluate(run_request, trial_id, parameters)

        result = GridSearch(request, _partly_failing).run()
        _check(len(result.trials) == 2, "failed candidate was removed")
        _check(result.trials[0].status is TrialStatus.FAILED, "failure not retained")
        _check(result.status is RunStatus.DEGRADED, "partial failure not surfaced")

    _run_test("deterministic Cartesian order", _test_order_and_cardinality)
    _run_test("drift-free decimal expansion", _test_decimal_grid)
    _run_test("candidate failure retention", _test_failure_retention)

    _all_passed = not _failures
    printer.status(
        "",
        f"{3 - len(_failures)}/3 grid search tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All grid search tests passed ===\n")