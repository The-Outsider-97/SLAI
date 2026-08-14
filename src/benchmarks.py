"""Enforceable foundation benchmark budgets for SLAI.

This module owns build-time regression benchmarks. Runtime agent SLO reporting
remains in ``PerformanceBudgetEvaluator``; keeping these concerns separate
prevents synthetic CI measurements from being mixed with production telemetry.

Run the configured gate with::

    python -m src.benchmarks
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
import tracemalloc

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from src.agents.runtime_contracts import AgentRuntimeIdentity, RuntimeLifecycle, RuntimeStatus, build_runtime_scope_id
from src.utils.configuration import bind_config


_CONFIG = bind_config(Path(__file__).parent / "agents/base/configs/agents_config.yaml")
load_global_config = _CONFIG.load
get_config_section = _CONFIG.section


class BenchmarkConfigurationError(ValueError):
    """Raised when the configured benchmark contract is incomplete or invalid."""


class BenchmarkBudgetExceeded(AssertionError):
    """Raised when one or more enforced foundation budgets are exceeded."""

    def __init__(self, report: "BenchmarkReport") -> None:
        failures = "; ".join(
            f"{result.name}: {', '.join(result.violations)}"
            for result in report.results
            if not result.passed
        )
        super().__init__(f"Foundation benchmark budget exceeded: {failures}")
        self.report = report


@dataclass(frozen=True, slots=True)
class BenchmarkBudget:
    """Hard limits for one deterministic, dependency-light operation."""

    name: str
    max_p95_latency_ms: float
    max_peak_memory_mb: float
    warmup_iterations: int
    measurement_iterations: int

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        default_warmups: int,
        default_iterations: int,
    ) -> "BenchmarkBudget":
        name = str(payload.get("name", "")).strip()
        if not name:
            raise BenchmarkConfigurationError("benchmark_budgets.contracts[].name must be non-empty")
        try:
            budget = cls(
                name=name,
                max_p95_latency_ms=float(payload["max_p95_latency_ms"]),
                max_peak_memory_mb=float(payload["max_peak_memory_mb"]),
                warmup_iterations=int(payload.get("warmup_iterations", default_warmups)),
                measurement_iterations=int(payload.get("measurement_iterations", default_iterations)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkConfigurationError(f"Invalid benchmark contract for {name!r}: {exc}") from exc
        budget.validate()
        return budget

    def validate(self) -> None:
        if self.max_p95_latency_ms <= 0:
            raise BenchmarkConfigurationError(f"{self.name}.max_p95_latency_ms must be positive")
        if self.max_peak_memory_mb <= 0:
            raise BenchmarkConfigurationError(f"{self.name}.max_peak_memory_mb must be positive")
        if self.warmup_iterations < 0:
            raise BenchmarkConfigurationError(f"{self.name}.warmup_iterations must be non-negative")
        if self.measurement_iterations < 5:
            raise BenchmarkConfigurationError(f"{self.name}.measurement_iterations must be at least 5")


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    """Observed latency distribution and peak traced allocation for one operation."""

    median_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    peak_memory_mb: float
    iterations: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One measured operation evaluated against its hard budget."""

    name: str
    budget: BenchmarkBudget
    measurement: Optional[BenchmarkMeasurement]
    passed: bool
    violations: tuple[str, ...] = ()
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = list(self.violations)
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Serializable result of the complete configured foundation gate."""

    enabled: bool
    enforced: bool
    passed: bool
    results: tuple[BenchmarkResult, ...] = field(default_factory=tuple)
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "enforced": self.enforced,
            "passed": self.passed,
            "reason": self.reason,
            "summary": {
                "benchmarks": len(self.results),
                "passed": sum(1 for result in self.results if result.passed),
                "failed": sum(1 for result in self.results if not result.passed),
            },
            "results": [result.to_dict() for result in self.results],
        }


def evaluate_measurement(budget: BenchmarkBudget, measurement: BenchmarkMeasurement) -> BenchmarkResult:
    """Compare one observation with one contract using hard, non-relative limits."""

    violations = []
    if measurement.p95_latency_ms > budget.max_p95_latency_ms:
        violations.append(
            f"p95 latency {measurement.p95_latency_ms:.4f}ms > {budget.max_p95_latency_ms:.4f}ms"
        )
    if measurement.peak_memory_mb > budget.max_peak_memory_mb:
        violations.append(
            f"peak memory {measurement.peak_memory_mb:.4f}MB > {budget.max_peak_memory_mb:.4f}MB"
        )
    return BenchmarkResult(
        name=budget.name,
        budget=budget,
        measurement=measurement,
        passed=not violations,
        violations=tuple(violations),
    )


class BenchmarkSuite:
    """Measure a closed set of operations and enforce their configured budgets."""

    def __init__(
        self,
        budgets: Mapping[str, BenchmarkBudget],
        operations: Mapping[str, Callable[[], Any]],
    ) -> None:
        self.budgets = dict(budgets)
        self.operations = dict(operations)
        missing_operations = sorted(set(self.budgets) - set(self.operations))
        unbudgeted_operations = sorted(set(self.operations) - set(self.budgets))
        if missing_operations or unbudgeted_operations:
            raise BenchmarkConfigurationError(
                "Benchmark registry and configured contracts must match exactly "
                f"(missing_operations={missing_operations}, unbudgeted_operations={unbudgeted_operations})"
            )

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
        return ordered[index]

    @classmethod
    def _measure(cls, operation: Callable[[], Any], budget: BenchmarkBudget) -> BenchmarkMeasurement:
        for _ in range(budget.warmup_iterations):
            operation()

        gc.collect()
        tracemalloc.start()
        starting_memory, _ = tracemalloc.get_traced_memory()
        latencies_ms: list[float] = []
        try:
            for _ in range(budget.measurement_iterations):
                started_ns = time.perf_counter_ns()
                operation()
                latencies_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
            _, peak_memory = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        return BenchmarkMeasurement(
            median_latency_ms=statistics.median(latencies_ms),
            p95_latency_ms=cls._percentile(latencies_ms, 0.95),
            max_latency_ms=max(latencies_ms),
            peak_memory_mb=max(0, peak_memory - starting_memory) / (1024.0 * 1024.0),
            iterations=len(latencies_ms),
        )

    def run(self, *, enforce: bool = True) -> BenchmarkReport:
        results = []
        for name in sorted(self.budgets):
            budget = self.budgets[name]
            try:
                measurement = self._measure(self.operations[name], budget)
                result = evaluate_measurement(budget, measurement)
            except Exception as exc:
                result = BenchmarkResult(
                    name=name,
                    budget=budget,
                    measurement=None,
                    passed=False,
                    violations=("benchmark operation raised an exception",),
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)

        report = BenchmarkReport(
            enabled=True,
            enforced=enforce,
            passed=all(result.passed for result in results),
            results=tuple(results),
        )
        if enforce and not report.passed:
            raise BenchmarkBudgetExceeded(report)
        return report


def load_benchmark_settings(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load and validate the canonical foundation benchmark configuration."""

    source = config if config is not None else load_global_config()
    section = get_config_section("benchmark_budgets", config=source, default={})
    if not section:
        raise BenchmarkConfigurationError("Missing benchmark_budgets section in agents_config.yaml")
    contracts = section.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        raise BenchmarkConfigurationError("benchmark_budgets.contracts must be a non-empty list")

    default_warmups = int(section.get("warmup_iterations", 5))
    default_iterations = int(section.get("measurement_iterations", 40))
    budgets: Dict[str, BenchmarkBudget] = {}
    for payload in contracts:
        if not isinstance(payload, Mapping):
            raise BenchmarkConfigurationError("Every benchmark contract must be a mapping")
        budget = BenchmarkBudget.from_mapping(
            payload,
            default_warmups=default_warmups,
            default_iterations=default_iterations,
        )
        if budget.name in budgets:
            raise BenchmarkConfigurationError(f"Duplicate benchmark contract: {budget.name}")
        budgets[budget.name] = budget
    return {
        "enabled": bool(section.get("enabled", True)),
        "enforce": bool(section.get("enforce", True)),
        "budgets": budgets,
    }


def foundation_operations() -> Dict[str, Callable[[], Any]]:
    """Return dependency-light operations that represent hardened foundation paths."""

    runtime_status = RuntimeStatus()
    runtime_status.transition(RuntimeLifecycle.ACTIVE)
    identity = AgentRuntimeIdentity(
        agent_type="benchmark",
        version="2.2.0",
        instance_id="benchmark:foundation",
        scope_id="foundation",
    )
    scope_memory = object()
    scope_config = {"profile": "foundation", "enabled": True}

    return {
        "config.cached_load": load_global_config,
        "runtime.scope_identity": lambda: build_runtime_scope_id(
            shared_memory=scope_memory,
            config=scope_config,
            constructor_kwargs={"mode": "benchmark"},
        ),
        "runtime.status_snapshot": lambda: runtime_status.snapshot(identity),
    }


def run_foundation_benchmarks(
    *,
    config: Optional[Mapping[str, Any]] = None,
    enforce: Optional[bool] = None,
) -> BenchmarkReport:
    """Run the configured foundation baseline and optionally fail on violations."""

    settings = load_benchmark_settings(config)
    resolved_enforce = settings["enforce"] if enforce is None else bool(enforce)
    if not settings["enabled"]:
        return BenchmarkReport(
            enabled=False,
            enforced=resolved_enforce,
            passed=True,
            reason="benchmark_budgets_disabled",
        )
    suite = BenchmarkSuite(settings["budgets"], foundation_operations())
    return suite.run(enforce=resolved_enforce)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run SLAI foundation benchmark budgets.")
    parser.add_argument("--no-enforce", action="store_true", help="Report violations without returning a failing exit code.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON report.")
    args = parser.parse_args(argv)

    try:
        report = run_foundation_benchmarks(enforce=not args.no_enforce)
    except BenchmarkBudgetExceeded as exc:
        report = exc.report
        print(json.dumps(report.to_dict(), indent=2 if args.pretty else None, sort_keys=True))
        return 1
    except BenchmarkConfigurationError as exc:
        print(json.dumps({"passed": False, "configuration_error": str(exc)}, sort_keys=True))
        return 2

    print(json.dumps(report.to_dict(), indent=2 if args.pretty else None, sort_keys=True))
    return 0 if report.passed or args.no_enforce else 1


if __name__ == "__main__":
    raise SystemExit(main())
