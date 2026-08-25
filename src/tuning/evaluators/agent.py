"""Transactional, multi-seed evaluation for mutable SLAI agents.

Each seed receives an independent transaction created from the configured
checkpoint or fresh state.  Candidate state is reset before every scenario,
then restored or discarded after the seed.  A trial cannot succeed unless all
transactions prove isolation and all scenario outcomes are structurally valid.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
import numpy as np

from collections.abc import Mapping, Sequence, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, cast

from ..tuning_contracts import *
from ..tuning_types import *
from ..tuning_validation import *
from ..utils.tuning_errors import *
from ..utils.tuning_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Agent Evaluation")
printer = PrettyPrinter()


class AgentCleanupMode(str, Enum):
    AUTO = "auto"
    RESTORE = "restore"
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class AgentEvaluationConfig:
    objective: MetricSpec
    seeds: Sequence[int]
    state_source: AgentStateSource
    checkpoint_id: str | None = None
    cleanup: AgentCleanupMode = AgentCleanupMode.AUTO
    calibration_bins: int = 10
    require_calibration: bool = False
    fail_fast: bool = True
    constraints: Sequence[MetricConstraint] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.objective, MetricSpec):
            raise TypeError("objective must be a MetricSpec")
        seeds = tuple(self.seeds)
        if not seeds:
            raise ValueError("At least one agent evaluation seed is required")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise TypeError("seeds must contain integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", seeds)
        if not isinstance(self.state_source, AgentStateSource):
            object.__setattr__(
                self, "state_source", AgentStateSource(self.state_source)
            )
        if self.state_source is AgentStateSource.CHECKPOINT:
            if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():
                raise ValueError("checkpoint source requires checkpoint_id")
            object.__setattr__(self, "checkpoint_id", self.checkpoint_id.strip())
        elif self.checkpoint_id not in {None, ""}:
            raise ValueError("fresh state source cannot declare checkpoint_id")
        else:
            object.__setattr__(self, "checkpoint_id", None)
        if not isinstance(self.cleanup, AgentCleanupMode):
            object.__setattr__(self, "cleanup", AgentCleanupMode(self.cleanup))
        if isinstance(self.calibration_bins, bool) or not isinstance(
            self.calibration_bins, int
        ):
            raise TypeError("calibration_bins must be an integer")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least 2")
        if not isinstance(self.require_calibration, bool):
            raise TypeError("require_calibration must be a bool")
        if not isinstance(self.fail_fast, bool):
            raise TypeError("fail_fast must be a bool")
        constraints = tuple(self.constraints)
        if any(not isinstance(item, MetricConstraint) for item in constraints):
            raise TypeError("constraints must contain MetricConstraint objects")
        object.__setattr__(self, "constraints", constraints)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AgentEvaluationConfig":
        report = validate_agent_config(config)
        report.raise_if_invalid(
            message="Invalid agent evaluation configuration.",
            context=TuningErrorContext(
                component="AgentEvaluator", operation="load_config"
            ),
        )
        state = config["state"]
        assert isinstance(state, Mapping)
        try:
            return cls(
                objective=parse_metric_spec(
                    config["objective"], path="agent_evaluation.objective"
                ),
                seeds=tuple(config["seeds"]),
                state_source=AgentStateSource(
                    str(state["source"]).strip().casefold()
                ),
                checkpoint_id=state.get("checkpoint_id"),
                cleanup=AgentCleanupMode(
                    str(state.get("cleanup", "auto")).strip().casefold()
                ),
                calibration_bins=int(config.get("calibration_bins", 10)),
                require_calibration=coerce_bool(
                    config.get("require_calibration", False),
                    name="require_calibration",
                ),
                fail_fast=coerce_bool(
                    config.get("fail_fast", True), name="fail_fast"
                ),
                constraints=parse_metric_constraints(
                    config.get("constraints"), path="agent_evaluation.constraints"
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TuningValidationError(
                "Unable to construct agent evaluation configuration.",
                context=TuningErrorContext(
                    component="AgentEvaluator", operation="load_config"
                ),
                details={"validation_error": str(exc)},
                cause=exc,
            ) from exc


class AgentEvaluator:
    """Run an isolated scenario suite and return one terminal trial record."""

    def __init__(
        self,
        transaction_factory: AgentTransactionFactory,
        scenario_runner: AgentScenarioRunner,
        scenarios: Sequence[AgentScenario],
        config: AgentEvaluationConfig | Mapping[str, Any],
    ) -> None:
        if not callable(transaction_factory):
            raise TuningContractError("transaction_factory must be callable.")
        if not callable(scenario_runner):
            raise TuningContractError("scenario_runner must be callable.")
        scenario_tuple = tuple(scenarios)
        if not scenario_tuple:
            raise TuningValidationError("At least one agent scenario is required.")
        if any(not isinstance(item, AgentScenario) for item in scenario_tuple):
            raise TuningContractError("scenarios must contain AgentScenario objects.")
        identifiers = [item.scenario_id for item in scenario_tuple]
        if len(set(identifiers)) != len(identifiers):
            raise TuningValidationError("Agent scenario identifiers must be unique.")
        self.transaction_factory = transaction_factory
        self.scenario_runner = scenario_runner
        self.scenarios = scenario_tuple
        self.config = (
            config
            if isinstance(config, AgentEvaluationConfig)
            else AgentEvaluationConfig.from_mapping(config)
        )

    def evaluate(
        self,
        request: TuningRunRequest,
        trial_id: str,
        parameters: Mapping[str, Any],
    ) -> TrialRecord:
        """Synchronous entry point; async adapters must use ``evaluate_async``."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_async(request, trial_id, parameters))
        raise TuningContractError(
            "evaluate() cannot run inside an active event loop; use evaluate_async()."
        )

    async def evaluate_async(
        self,
        request: TuningRunRequest,
        trial_id: str,
        parameters: Mapping[str, Any],
    ) -> TrialRecord:
        if not isinstance(request, TuningRunRequest):
            raise TuningContractError("request must be a TuningRunRequest.")
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
                "Evaluator objective disagrees with the tuning request."
            )
        scenarios = self._selected_scenarios(request.scenario_ids)
        seeds = tuple(request.seeds) or tuple(self.config.seeds)
        if not seeds:
            raise TuningValidationError("At least one agent evaluation seed is required.")

        started_at = utc_now()
        trial_started = time.perf_counter()
        evaluations: list[EvaluationSlice] = []
        outcomes: list[tuple[int, AgentScenario, AgentScenarioOutcome]] = []
        state_records: list[AgentStateRecord] = []
        reset_count = 0
        primary_error: TuningError | None = None

        for seed in seeds:
            transaction: AgentTransaction | None = None
            transaction_valid = False
            application_started = False
            disposition = CandidateStateDisposition.NOT_APPLIED
            seed_error: TuningError | None = None
            seed_reset_count = 0
            try:
                transaction_value = self.transaction_factory(
                    self.config.state_source, self.config.checkpoint_id, seed
                )
                transaction = await self._await_value(transaction_value)
                self._validate_transaction(transaction)
                if transaction is None:
                    raise TuningContractError("transaction_factory must return an AgentTransaction.")
                transaction_valid = True
                self._validate_transaction_source(transaction)

                application_started = True
                await self._await_value(transaction.apply_candidate(parameters))
                for scenario_index, scenario in enumerate(scenarios):
                    if scenario_index > 0:
                        await self._await_value(transaction.reset_candidate())
                        seed_reset_count += 1
                        reset_count += 1
                    wall_started = time.perf_counter()
                    try:
                        raw_outcome = self.scenario_runner(
                            transaction.agent, scenario, seed
                        )
                        outcome = await self._await_value(raw_outcome)
                        if not isinstance(outcome, AgentScenarioOutcome):
                            raise TuningContractError(
                                "scenario_runner must return AgentScenarioOutcome.",
                                details={"actual_type": type(outcome).__name__},
                            )
                        wall_elapsed = time.perf_counter() - wall_started
                        evaluations.append(
                            EvaluationSlice(
                                status=TrialStatus.SUCCEEDED,
                                scenario_id=scenario.scenario_id,
                                seed=seed,
                                metrics=outcome.metric_values(),
                                resources=ResourceUsage(
                                    wall_time_seconds=wall_elapsed,
                                    peak_memory_bytes=outcome.peak_memory_bytes,
                                    latency_quantiles_seconds={
                                        "reported": outcome.latency_seconds,
                                        "wall": wall_elapsed,
                                    },
                                    sample_count=1,
                                ),
                                metadata={
                                    "success": outcome.success,
                                    "safety_violations": list(
                                        outcome.safety_violations
                                    ),
                                    "confidence": outcome.confidence,
                                    "correct": outcome.correct,
                                    "scenario": dict(scenario.metadata),
                                    "outcome": dict(outcome.metadata),
                                },
                            )
                        )
                        outcomes.append((seed, scenario, outcome))
                    except Exception as exc:
                        seed_error = self._as_evaluation_error(
                            exc, request.run_id, trial_id, scenario.scenario_id, seed
                        )
                        evaluations.append(
                            EvaluationSlice(
                                status=TrialStatus.FAILED,
                                scenario_id=scenario.scenario_id,
                                seed=seed,
                                resources=ResourceUsage(
                                    wall_time_seconds=time.perf_counter() - wall_started,
                                    sample_count=1,
                                ),
                                error=ErrorRecord.from_exception(seed_error),
                            )
                        )
                        primary_error = primary_error or seed_error
                        if self.config.fail_fast:
                            break
            except Exception as exc:
                seed_error = self._as_lifecycle_error(
                    exc, request.run_id, trial_id, seed
                )
                primary_error = primary_error or seed_error
                evaluations.append(
                    EvaluationSlice(
                        status=TrialStatus.FAILED,
                        seed=seed,
                        error=ErrorRecord.from_exception(seed_error),
                        metadata={"stage": "transaction_or_candidate_application"},
                    )
                )
            finally:
                if transaction is not None and transaction_valid:
                    cleanup_error: TuningError | None = None
                    try:
                        disposition = await self._cleanup(transaction)
                        if not application_started:
                            disposition = CandidateStateDisposition.NOT_APPLIED
                    except Exception as exc:
                        cleanup_error = self._as_cleanup_error(
                            exc, request.run_id, trial_id, seed
                        )
                        primary_error = cleanup_error
                        disposition = self._failed_cleanup_disposition(
                            transaction.source
                        )
                        evaluations.append(
                            EvaluationSlice(
                                status=TrialStatus.FAILED,
                                seed=seed,
                                error=ErrorRecord.from_exception(cleanup_error),
                                metadata={"stage": "candidate_cleanup"},
                            )
                        )
                    state_records.append(
                        AgentStateRecord(
                            source=transaction.source,
                            transaction_id=str(transaction.transaction_id),
                            baseline_checkpoint_id=transaction.baseline_checkpoint_id,
                            candidate_applied=application_started,
                            disposition=disposition,
                            metadata={
                                "seed": seed,
                                "reset_count": seed_reset_count,
                                "cleanup_error": (
                                    None
                                    if cleanup_error is None
                                    else cleanup_error.to_dict(include_traceback=False)
                                ),
                            },
                        )
                    )
            if primary_error is not None and self.config.fail_fast:
                break

        audit = (
            None
            if not state_records
            else AgentStateAudit(
                transactions=tuple(state_records),
                reset_count=reset_count,
                metadata={
                    "configured_seed_count": len(seeds),
                    "completed_transaction_count": len(state_records),
                },
            )
        )
        if primary_error is not None or any(
            item.status is TrialStatus.FAILED for item in evaluations
        ):
            error = primary_error or TuningEvaluationError(
                "One or more agent scenario evaluations failed."
            )
            return self._failed_trial(
                request,
                trial_id,
                parameters,
                started_at,
                trial_started,
                evaluations,
                audit,
                error,
            )

        try:
            aggregate = self._aggregate(outcomes, seeds)
            constraints = tuple(
                constraint.evaluate(aggregate) for constraint in self.config.constraints
            )
            if self.config.objective.name not in aggregate:
                raise TuningEvaluationError(
                    f"Objective metric {self.config.objective.name!r} was not produced."
                )
            status = (
                TrialStatus.SUCCEEDED
                if all(item.passed for item in constraints)
                else TrialStatus.REJECTED
            )
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
                resources=self._aggregate_resources(
                    outcomes, time.perf_counter() - trial_started
                ),
                agent_state=audit,
                metadata={
                    "scenario_count": len(scenarios),
                    "seed_count": len(seeds),
                    "evaluation_count": len(outcomes),
                    "calibration_required": self.config.require_calibration,
                },
            )
        except Exception as exc:
            error = self._as_evaluation_error(
                exc, request.run_id, trial_id, None, None
            )
            return self._failed_trial(
                request,
                trial_id,
                parameters,
                started_at,
                trial_started,
                evaluations,
                audit,
                error,
            )

    def _selected_scenarios(
        self, requested_ids: Sequence[str]
    ) -> tuple[AgentScenario, ...]:
        if not requested_ids:
            return self.scenarios
        by_id = {item.scenario_id: item for item in self.scenarios}
        missing = [item for item in requested_ids if item not in by_id]
        if missing:
            raise TuningValidationError(
                "Tuning request references unknown scenarios.",
                details={"missing_scenario_ids": missing},
            )
        return tuple(by_id[item] for item in requested_ids)

    @staticmethod
    async def _await_value(value: Any) -> Any:
        return await value if inspect.isawaitable(value) else value

    @staticmethod
    def _validate_transaction(transaction: Any) -> None:
        required_attributes = (
            "transaction_id",
            "source",
            "baseline_checkpoint_id",
            "agent",
        )
        required_methods = (
            "apply_candidate",
            "reset_candidate",
            "restore_baseline",
            "discard_candidate",
        )
        missing = [name for name in required_attributes if not hasattr(transaction, name)]
        missing.extend(
            name for name in required_methods if not callable(getattr(transaction, name, None))
        )
        if missing:
            raise TuningContractError(
                "Agent transaction does not satisfy the required contract.",
                details={"missing_members": sorted(set(missing))},
            )
        if not isinstance(transaction.source, AgentStateSource):
            raise TuningContractError("transaction.source must be AgentStateSource.")
        if not isinstance(
            transaction.transaction_id, str
        ) or not transaction.transaction_id.strip():
            raise TuningContractError("transaction_id must be a non-empty string.")

    def _validate_transaction_source(self, transaction: AgentTransaction) -> None:
        if transaction.source is not self.config.state_source:
            raise TuningCheckpointError(
                "Transaction state source differs from evaluator configuration."
            )
        if self.config.state_source is AgentStateSource.CHECKPOINT and (
            transaction.baseline_checkpoint_id != self.config.checkpoint_id
        ):
            raise TuningCheckpointError(
                "Transaction restored a different checkpoint than requested.",
                details={
                    "requested": self.config.checkpoint_id,
                    "actual": transaction.baseline_checkpoint_id,
                },
            )

    async def _cleanup(
        self, transaction: AgentTransaction
    ) -> CandidateStateDisposition:
        cleanup = self.config.cleanup
        if cleanup is AgentCleanupMode.AUTO:
            cleanup = (
                AgentCleanupMode.RESTORE
                if transaction.source is AgentStateSource.CHECKPOINT
                else AgentCleanupMode.DISCARD
            )
        if cleanup is AgentCleanupMode.RESTORE:
            await self._await_value(transaction.restore_baseline())
            return CandidateStateDisposition.RESTORED
        await self._await_value(transaction.discard_candidate())
        return CandidateStateDisposition.DISCARDED

    def _failed_cleanup_disposition(
        self, source: AgentStateSource
    ) -> CandidateStateDisposition:
        cleanup = self.config.cleanup
        if cleanup is AgentCleanupMode.AUTO:
            cleanup = (
                AgentCleanupMode.RESTORE
                if source is AgentStateSource.CHECKPOINT
                else AgentCleanupMode.DISCARD
            )
        return (
            CandidateStateDisposition.RESTORE_FAILED
            if cleanup is AgentCleanupMode.RESTORE
            else CandidateStateDisposition.DISCARD_FAILED
        )

    def _aggregate(
        self,
        outcomes: Sequence[tuple[int, AgentScenario, AgentScenarioOutcome]],
        seeds: Sequence[int],
    ) -> dict[str, float]:
        expected_count = len(seeds) * len(
            {scenario.scenario_id for _, scenario, _ in outcomes}
        )
        if not outcomes or len(outcomes) != expected_count:
            raise TuningEvaluationError(
                "Agent aggregation requires a complete seed/scenario matrix."
            )
        utilities = np.asarray(
            [outcome.task_utility for _, _, outcome in outcomes], dtype=float
        )
        successes = np.asarray(
            [1.0 if outcome.success else 0.0 for _, _, outcome in outcomes],
            dtype=float,
        )
        violations = np.asarray(
            [len(outcome.safety_violations) for _, _, outcome in outcomes],
            dtype=float,
        )
        latencies = np.asarray(
            [outcome.latency_seconds for _, _, outcome in outcomes], dtype=float
        )
        aggregate: dict[str, float] = {
            "task_utility": float(np.mean(utilities)),
            "success_rate": float(np.mean(successes)),
            "safety_violation_count": float(np.sum(violations)),
            "safety_violation_rate": float(np.mean(violations > 0)),
            "latency_mean_seconds": float(np.mean(latencies)),
            "latency_p95_seconds": float(np.quantile(latencies, 0.95)),
        }
        utility_by_seed = np.asarray(
            [
                np.mean(
                    [
                        outcome.task_utility
                        for observed_seed, _, outcome in outcomes
                        if observed_seed == seed
                    ]
                )
                for seed in seeds
            ],
            dtype=float,
        )
        aggregate["utility_seed_std"] = float(np.std(utility_by_seed, ddof=0))
        aggregate["utility_seed_sem"] = float(
            np.std(utility_by_seed, ddof=0) / math.sqrt(len(utility_by_seed))
        )

        memory_values = [
            outcome.peak_memory_bytes for _, _, outcome in outcomes
        ]
        if any(value is not None for value in memory_values):
            if any(value is None for value in memory_values):
                raise TuningEvaluationError(
                    "Peak-memory instrumentation must cover every scenario or none."
                )
            aggregate["peak_memory_bytes"] = float(
                max(int(value) for value in memory_values if value is not None)
            )

        calibration = [
            (outcome.confidence, outcome.correct)
            for _, _, outcome in outcomes
            if outcome.confidence is not None
        ]
        coverage = len(calibration) / len(outcomes)
        aggregate["calibration_coverage"] = float(coverage)
        if self.config.require_calibration and len(calibration) != len(outcomes):
            raise TuningEvaluationError(
                "Calibration is required but not every outcome supplied confidence/correct."
            )
        if calibration:
            confidences = np.asarray([float(item[0]) for item in calibration])
            correctness = np.asarray([1.0 if item[1] else 0.0 for item in calibration])
            aggregate["calibration_ece"] = self._expected_calibration_error(
                confidences, correctness, self.config.calibration_bins
            )
            aggregate["calibration_brier"] = float(
                np.mean((confidences - correctness) ** 2)
            )

        extra_names = [set(outcome.metrics) for _, _, outcome in outcomes]
        if extra_names and any(names != extra_names[0] for names in extra_names):
            raise TuningEvaluationError(
                "Additional metric sets must be identical across all outcomes."
            )
        for name in sorted(extra_names[0] if extra_names else ()):
            values = np.asarray(
                [outcome.metrics[name] for _, _, outcome in outcomes], dtype=float
            )
            aggregate[name] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values, ddof=0))
        aggregate["evaluation_count"] = float(len(outcomes))
        return aggregate

    @staticmethod
    def _expected_calibration_error(
        confidence: np.ndarray, correctness: np.ndarray, bins: int
    ) -> float:
        indices = np.minimum((confidence * bins).astype(int), bins - 1)
        total = len(confidence)
        ece = 0.0
        for bin_index in range(bins):
            mask = indices == bin_index
            count = int(np.sum(mask))
            if count:
                accuracy = float(np.mean(correctness[mask]))
                mean_confidence = float(np.mean(confidence[mask]))
                ece += (count / total) * abs(accuracy - mean_confidence)
        return float(ece)

    @staticmethod
    def _aggregate_resources(
        outcomes: Sequence[tuple[int, AgentScenario, AgentScenarioOutcome]],
        wall_time_seconds: float,
    ) -> ResourceUsage:
        latencies = np.asarray(
            [outcome.latency_seconds for _, _, outcome in outcomes], dtype=float
        )
        memory = [
            outcome.peak_memory_bytes
            for _, _, outcome in outcomes
            if outcome.peak_memory_bytes is not None
        ]
        return ResourceUsage(
            wall_time_seconds=wall_time_seconds,
            peak_memory_bytes=max(memory) if memory else None,
            latency_quantiles_seconds={
                "p50": float(np.quantile(latencies, 0.50)),
                "p95": float(np.quantile(latencies, 0.95)),
                "p100": float(np.max(latencies)),
            },
            sample_count=len(outcomes),
        )

    @staticmethod
    def _as_evaluation_error(
        error: BaseException,
        run_id: str,
        trial_id: str,
        scenario_id: str | None,
        seed: int | None,
    ) -> TuningError:
        return wrap_exception(
            error,
            message="Agent scenario evaluation failed.",
            error_cls=TuningEvaluationError,
            context=TuningErrorContext(
                run_id=run_id,
                trial_id=trial_id,
                component="AgentEvaluator",
                operation="run_scenario",
                scenario_id=scenario_id,
                seed=seed,
            ),
        )

    @staticmethod
    def _as_lifecycle_error(
        error: BaseException, run_id: str, trial_id: str, seed: int
    ) -> TuningError:
        return wrap_exception(
            error,
            message="Agent transaction or candidate application failed.",
            error_cls=TuningLifecycleError,
            context=TuningErrorContext(
                run_id=run_id,
                trial_id=trial_id,
                component="AgentEvaluator",
                operation="apply_candidate",
                seed=seed,
            ),
        )

    @staticmethod
    def _as_cleanup_error(
        error: BaseException, run_id: str, trial_id: str, seed: int
    ) -> TuningError:
        return wrap_exception(
            error,
            message="Candidate state cleanup failed.",
            error_cls=TuningCheckpointError,
            context=TuningErrorContext(
                run_id=run_id,
                trial_id=trial_id,
                component="AgentEvaluator",
                operation="cleanup_candidate",
                seed=seed,
            ),
            details={"live_state_may_be_mutated": True},
        )

    @staticmethod
    def _failed_trial(
        request: TuningRunRequest,
        trial_id: str,
        parameters: Mapping[str, Any],
        started_at: datetime,
        trial_started: float,
        evaluations: Sequence[EvaluationSlice],
        audit: AgentStateAudit | None,
        error: TuningError,
    ) -> TrialRecord:
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
            agent_state=audit,
            error=ErrorRecord.from_exception(error),
            metadata={"failure": to_json_safe(error.to_dict(include_traceback=False))},
        )


__all__ = [
    "AgentCleanupMode",
    "AgentEvaluationConfig",
    "AgentEvaluator",
]

if __name__ == "__main__":
    print("\n=== Running Agent Evaluator Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting agent evaluator tests", "info")

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

    _objective = MetricSpec("task_utility", ObjectiveDirection.MAXIMIZE)
    _config = AgentEvaluationConfig.from_mapping(
        {
            "objective": _objective.to_dict(),
            "seeds": [11, 13],
            "state": {
                "source": "fresh",
                "checkpoint_id": None,
                "cleanup": "auto",
            },
            "calibration_bins": 5,
            "require_calibration": True,
            "fail_fast": True,
            "constraints": [],
        }
    )
    _request = TuningRunRequest(
        run_id="agent-self-test",
        settings=TunerSettings(
            strategy=TuningStrategy.GRID,
            model_type="SelfTestAgent",
            allow_generate=False,
        ),
        config={},
        strategy_config={"fail_fast": False},
        search_space=(
            {"name": "temperature", "type": "real", "values": [0.5]},
        ),
        config_fingerprint="agent-self-test",
        objective=_objective,
        seeds=(11, 13),
    )
    _scenarios = (
        AgentScenario("scenario-a", {"utility": 0.8}),
        AgentScenario("scenario-b", {"utility": 0.6}),
    )

    class _Transaction:
        transaction_id: str
        source: AgentStateSource
        baseline_checkpoint_id: str | None
        agent: Any

        def __init__(self, seed: int, *, fail_cleanup: bool = False) -> None:
            self.transaction_id = f"transaction-{seed}"
            self.source = AgentStateSource.FRESH
            self.baseline_checkpoint_id = None
            self.agent = self
            self.seed = seed
            self.fail_cleanup = fail_cleanup
            self.parameters: dict[str, Any] = {}
            self.reset_count = 0
            self.discarded = False

        def apply_candidate(self, parameters: Mapping[str, Any]) -> None:
            self.parameters = dict(parameters)

        def reset_candidate(self) -> None:
            self.reset_count += 1

        def restore_baseline(self) -> None:
            raise AssertionError("fresh state must not be restored in AUTO mode")

        def discard_candidate(self) -> None:
            if self.fail_cleanup:
                raise RuntimeError("intentional cleanup failure")
            self.discarded = True

    def _scenario_runner(agent: _Transaction, scenario: AgentScenario, seed: int) -> AgentScenarioOutcome:
        _check(agent.parameters == {"temperature": 0.5}, "candidate not applied")
        return AgentScenarioOutcome(
            task_utility=float(scenario.payload["utility"]),
            success=True,
            latency_seconds=0.01,
            safety_violations=(),
            peak_memory_bytes=2048,
            confidence=0.9,
            correct=True,
            metrics={"stability": 1.0},
        )

    def _test_transactional_matrix() -> None:
        transactions: list[_Transaction] = []

        def _factory(source: AgentStateSource, checkpoint_id: str | None, seed: int) -> _Transaction:
            _check(source is AgentStateSource.FRESH, "wrong state source")
            _check(checkpoint_id is None, "fresh state received checkpoint id")
            transaction = _Transaction(seed)
            transactions.append(transaction)
            return transaction

        evaluator = AgentEvaluator(_factory, _scenario_runner, _scenarios, _config)
        trial = evaluator.evaluate(
            _request, "agent-trial", {"temperature": 0.5}
        )
        _check(trial.status is TrialStatus.SUCCEEDED, "agent trial did not succeed")
        _check(len(trial.evaluations) == 4, "seed/scenario matrix is incomplete")
        _check(trial.metrics["evaluation_count"] == 4.0, "wrong evaluation count")
        _check(
            math.isclose(trial.metrics["task_utility"], 0.7),
            "utility aggregation failed",
        )
        _check(trial.metrics["calibration_coverage"] == 1.0, "calibration lost")
        _check(trial.agent_state is not None, "state audit is missing")
        assert trial.agent_state is not None
        state_audit = cast(AgentStateAudit, trial.agent_state)
        _check(
            all(
                item.candidate_applied
                and item.disposition is CandidateStateDisposition.DISCARDED
                for item in state_audit.transactions
            ),
            "candidate state was not isolated",
        )
        _check(state_audit.reset_count == 2, "scenario reset count is wrong")
        _check(all(item.discarded for item in transactions), "fresh state not discarded")

    def _test_cleanup_failure() -> None:
        def _factory(
            source: AgentStateSource, checkpoint_id: str | None, seed: int
        ) -> _Transaction:
            return _Transaction(seed, fail_cleanup=True)

        evaluator = AgentEvaluator(_factory, _scenario_runner, _scenarios, _config)
        trial = evaluator.evaluate(
            _request, "agent-cleanup-failure", {"temperature": 0.5}
        )
        _check(trial.status is TrialStatus.FAILED, "cleanup failure was concealed")
        _check(trial.error is not None, "cleanup failure has no error evidence")
        _check(trial.agent_state is not None, "cleanup audit is missing")
        assert trial.agent_state is not None
        state_audit = cast(AgentStateAudit, trial.agent_state)
        dispositions = {
            item.disposition for item in state_audit.transactions
        }
        _check(
            CandidateStateDisposition.DISCARD_FAILED in dispositions,
            "failed discard disposition was not recorded",
        )
        _check(
            all(
                item.disposition is not CandidateStateDisposition.DISCARDED
                for item in state_audit.transactions
            ),
            "failed cleanup reported isolation",
        )

    def _test_unknown_scenario_rejected() -> None:
        evaluator = AgentEvaluator(
            lambda source, checkpoint_id, seed: _Transaction(seed),
            _scenario_runner,
            _scenarios,
            _config,
        )
        try:
            evaluator._selected_scenarios(("missing-scenario",))
        except TuningValidationError:
            return
        raise AssertionError("unknown scenario identifier was accepted")

    _run_test("transactional seed/scenario matrix", _test_transactional_matrix)
    _run_test("cleanup failure evidence", _test_cleanup_failure)
    _run_test("scenario selection validation", _test_unknown_scenario_rejected)

    _all_passed = not _failures
    printer.status(
        "",
        f"{3 - len(_failures)}/3 agent evaluator tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All agent evaluator tests passed ===\n")