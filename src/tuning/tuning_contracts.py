"""Structural contracts shared by SLAI tuning components.

The contracts are deliberately adapter-oriented.  They do not assume that an
agent exposes ``state_dict()``, that a scikit-learn estimator accepts
``validation_data``, or that checkpoint identifiers are filesystem paths.
Concrete SLAI integrations translate their native APIs into these small,
explicit boundaries.
"""

from __future__ import annotations

import math

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

from .tuning_types import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Tuning Contracts")
printer = PrettyPrinter()


T = TypeVar("T")
MaybeAwaitable: TypeAlias = T | Awaitable[T]
MetricFunction: TypeAlias = Callable[[Any, Any], float]


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class SupervisedSplit:
    """Disjoint train, early-stopping validation, and final-test indices."""

    split_id: str
    train_indices: Sequence[int]
    validation_indices: Sequence[int]
    test_indices: Sequence[int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_id", _non_empty(self.split_id, "split_id"))
        normalized: dict[str, tuple[int, ...]] = {}
        for name in ("train_indices", "validation_indices", "test_indices"):
            values = tuple(getattr(self, name))
            if not values:
                raise ValueError(f"{name} must not be empty")
            if any(isinstance(index, bool) or not isinstance(index, int) for index in values):
                raise TypeError(f"{name} must contain integers")
            if any(index < 0 for index in values):
                raise ValueError(f"{name} must contain non-negative indices")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate indices")
            normalized[name] = values
            object.__setattr__(self, name, values)
        train = set(normalized["train_indices"])
        validation = set(normalized["validation_indices"])
        test = set(normalized["test_indices"])
        if train & validation or train & test or validation & test:
            raise ValueError("train, validation, and test indices must be disjoint")
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class SupervisedAdapter(Protocol):
    """Translate a model family into the supervised evaluation lifecycle."""

    def build(self, parameters: Mapping[str, Any], seed: int) -> Any:
        """Construct an unfitted model for one independent evaluation."""

    def fit(
        self,
        model: Any,
        x_train: Any,
        y_train: Any,
        x_validation: Any,
        y_validation: Any,
    ) -> Mapping[str, Any] | None:
        """Fit without observing the final-test partition."""

    def predict(self, model: Any, x_test: Any) -> Any:
        """Return predictions used by registered metric functions."""


@runtime_checkable
class SplitProvider(Protocol):
    def __call__(self, sample_count: int, seed: int) -> Sequence[SupervisedSplit]:
        """Return validated split definitions for a seed."""
        ...


@dataclass(frozen=True, slots=True)
class AgentScenario:
    """One immutable task/scenario presented to an agent evaluator."""

    scenario_id: str
    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_id", _non_empty(self.scenario_id, "scenario_id")
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class AgentScenarioOutcome:
    """Minimum evidence returned by one scenario execution.

    ``confidence`` and ``correct`` form one calibration observation.  They are
    optional together because some agent tasks do not produce probabilistic
    correctness judgments.  Calibration is never inferred from unrelated
    scores.
    """

    task_utility: float
    success: bool
    latency_seconds: float
    safety_violations: Sequence[str] = field(default_factory=tuple)
    peak_memory_bytes: int | None = None
    confidence: float | None = None
    correct: bool | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _RESERVED_METRICS = frozenset(
        {
            "task_utility",
            "success",
            "safety_violation_count",
            "latency_seconds",
            "peak_memory_bytes",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_utility", _finite(self.task_utility, "task_utility")
        )
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        latency = _finite(self.latency_seconds, "latency_seconds")
        if latency < 0:
            raise ValueError("latency_seconds must be non-negative")
        object.__setattr__(self, "latency_seconds", latency)
        violations = tuple(
            _non_empty(str(item), "safety violation") for item in self.safety_violations
        )
        object.__setattr__(self, "safety_violations", violations)
        if self.peak_memory_bytes is not None:
            if isinstance(self.peak_memory_bytes, bool) or not isinstance(
                self.peak_memory_bytes, int
            ):
                raise TypeError("peak_memory_bytes must be an integer")
            if self.peak_memory_bytes < 0:
                raise ValueError("peak_memory_bytes must be non-negative")
        if (self.confidence is None) != (self.correct is None):
            raise ValueError("confidence and correct must either both be set or both be null")
        if self.confidence is not None:
            confidence = _finite(self.confidence, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be within [0, 1]")
            object.__setattr__(self, "confidence", confidence)
            if not isinstance(self.correct, bool):
                raise TypeError("correct must be a bool")
        copied_metrics: dict[str, float] = {}
        for raw_name, raw_value in self.metrics.items():
            name = _non_empty(str(raw_name), "metric name")
            if name in self._RESERVED_METRICS:
                raise ValueError(f"Additional metric {name!r} uses a reserved name")
            copied_metrics[name] = _finite(raw_value, f"metrics.{name}")
        object.__setattr__(self, "metrics", copied_metrics)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def metric_values(self) -> dict[str, float]:
        values = {
            "task_utility": self.task_utility,
            "success": 1.0 if self.success else 0.0,
            "safety_violation_count": float(len(self.safety_violations)),
            "latency_seconds": self.latency_seconds,
            **dict(self.metrics),
        }
        if self.peak_memory_bytes is not None:
            values["peak_memory_bytes"] = float(self.peak_memory_bytes)
        return values

    def resource_usage(self) -> ResourceUsage:
        return ResourceUsage(
            wall_time_seconds=self.latency_seconds,
            peak_memory_bytes=self.peak_memory_bytes,
            latency_quantiles_seconds={"p100": self.latency_seconds},
            sample_count=1,
        )


@dataclass(frozen=True, slots=True)
class MetricConstraint:
    """Declarative eligibility constraint evaluated after metric aggregation."""

    name: str
    metric_name: str
    operator: ConstraintOperator
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "constraint name"))
        object.__setattr__(
            self, "metric_name", _non_empty(self.metric_name, "constraint metric_name")
        )
        if not isinstance(self.operator, ConstraintOperator):
            object.__setattr__(self, "operator", ConstraintOperator(self.operator))
        object.__setattr__(
            self, "threshold", _finite(self.threshold, "constraint threshold")
        )

    def evaluate(self, metrics: Mapping[str, float]) -> ConstraintEvaluation:
        if self.metric_name not in metrics:
            raise KeyError(
                f"Constraint {self.name!r} requires missing metric {self.metric_name!r}"
            )
        observed = _finite(metrics[self.metric_name], self.metric_name)
        passed = self.operator.evaluate(observed, self.threshold)
        return ConstraintEvaluation(
            name=self.name,
            metric_name=self.metric_name,
            operator=self.operator,
            threshold=self.threshold,
            observed=observed,
            passed=passed,
            reason=None if passed else "Observed metric violates the declared threshold.",
        )


@runtime_checkable
class AgentTransaction(Protocol):
    """One isolated baseline/candidate-state transaction.

    The adapter must capture candidate state after ``apply_candidate`` so
    ``reset_candidate`` can make scenario/seed evaluations independent.
    """

    transaction_id: str
    source: AgentStateSource
    baseline_checkpoint_id: str | None
    agent: Any

    def apply_candidate(self, parameters: Mapping[str, Any]) -> MaybeAwaitable[None]:
        ...

    def reset_candidate(self) -> MaybeAwaitable[None]:
        ...

    def restore_baseline(self) -> MaybeAwaitable[None]:
        ...

    def discard_candidate(self) -> MaybeAwaitable[None]:
        ...


@runtime_checkable
class AgentTransactionFactory(Protocol):
    def __call__(
        self,
        source: AgentStateSource,
        checkpoint_id: str | None,
        seed: int,
    ) -> MaybeAwaitable[AgentTransaction]:
        """Create a transaction without applying a candidate configuration."""
        ...


@runtime_checkable
class AgentScenarioRunner(Protocol):
    def __call__(
        self,
        agent: Any,
        scenario: AgentScenario,
        seed: int,
    ) -> MaybeAwaitable[AgentScenarioOutcome]:
        ...


@runtime_checkable
class SearchRunnerProtocol(Protocol):
    def __call__(
        self, request: TuningRunRequest, evaluation_context: Any
    ) -> MaybeAwaitable[SearchResult]:
        ...


@runtime_checkable
class PromotionPolicyProtocol(Protocol):
    def __call__(
        self,
        request: TuningRunRequest,
        result: SearchResult,
        evaluation_context: Any,
    ) -> MaybeAwaitable[PromotionRecord | None]:
        ...


@runtime_checkable
class ArtifactWriterProtocol(Protocol):
    def __call__(self, result: TuningResult) -> MaybeAwaitable[Sequence[ArtifactRecord]]:
        ...


__all__ = [
    "AgentScenario",
    "AgentScenarioOutcome",
    "AgentScenarioRunner",
    "AgentTransaction",
    "AgentTransactionFactory",
    "ArtifactWriterProtocol",
    "MaybeAwaitable",
    "MetricConstraint",
    "MetricFunction",
    "PromotionPolicyProtocol",
    "SearchRunnerProtocol",
    "SplitProvider",
    "SupervisedAdapter",
    "SupervisedSplit",
]

if __name__ == "__main__":
    print("\n=== Running Tuning Contracts Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting tuning contract tests", "info")

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

    def _test_supervised_split_invariants() -> None:
        split = SupervisedSplit(
            split_id="fold-1",
            train_indices=(0, 1),
            validation_indices=(2,),
            test_indices=(3,),
        )
        _check(split.train_indices == (0, 1), "indices were not normalized")
        try:
            SupervisedSplit(
                split_id="leaking",
                train_indices=(0, 1),
                validation_indices=(1, 2),
                test_indices=(3,),
            )
        except ValueError:
            return
        raise AssertionError("overlapping supervised partitions were accepted")

    def _test_agent_outcome_evidence() -> None:
        outcome = AgentScenarioOutcome(
            task_utility=0.8,
            success=True,
            latency_seconds=0.02,
            safety_violations=("policy-warning",),
            peak_memory_bytes=1024,
            confidence=0.75,
            correct=True,
            metrics={"stability": 0.9},
        )
        values = outcome.metric_values()
        _check(values["success"] == 1.0, "success was aggregated incorrectly")
        _check(
            values["safety_violation_count"] == 1.0,
            "safety violations were aggregated incorrectly",
        )
        _check(
            outcome.resource_usage().peak_memory_bytes == 1024,
            "resource evidence was lost",
        )
        try:
            AgentScenarioOutcome(
                task_utility=0.0,
                success=False,
                latency_seconds=0.0,
                confidence=0.5,
            )
        except ValueError:
            return
        raise AssertionError("partial calibration evidence was accepted")

    def _test_metric_constraint() -> None:
        constraint = MetricConstraint(
            name="latency-budget",
            metric_name="latency_seconds",
            operator=ConstraintOperator.LESS_THAN_OR_EQUAL,
            threshold=0.1,
        )
        passed = constraint.evaluate({"latency_seconds": 0.05})
        failed = constraint.evaluate({"latency_seconds": 0.2})
        _check(passed.passed, "passing constraint was rejected")
        _check(not failed.passed, "violating constraint was accepted")

    def _test_runtime_protocols() -> None:
        class _Adapter:
            def build(self, parameters: Mapping[str, Any], seed: int) -> Any:
                return {"parameters": dict(parameters), "seed": seed}

            def fit(
                self,
                model: Any,
                x_train: Any,
                y_train: Any,
                x_validation: Any,
                y_validation: Any,
            ) -> Mapping[str, Any] | None:
                return {"fitted": True}

            def predict(self, model: Any, x_test: Any) -> Any:
                return x_test

        _check(isinstance(_Adapter(), SupervisedAdapter), "adapter protocol failed")

        def _split_provider(sample_count: int, seed: int) -> Sequence[SupervisedSplit]:
            return (
                SupervisedSplit("provided", (0,), (1,), (2,)),
            )

        _check(
            isinstance(_split_provider, SplitProvider),
            "split-provider protocol failed",
        )

    _run_test("disjoint supervised partitions", _test_supervised_split_invariants)
    _run_test("agent outcome evidence", _test_agent_outcome_evidence)
    _run_test("metric constraint evaluation", _test_metric_constraint)
    _run_test("runtime-checkable adapter protocols", _test_runtime_protocols)

    _all_passed = not _failures
    printer.status(
        "",
        f"{4 - len(_failures)}/4 tuning contract tests passed",
        "success" if _all_passed else "error",
    )
    if not _all_passed:
        raise SystemExit(1)
    print("\n=== All tuning contract tests passed ===\n")