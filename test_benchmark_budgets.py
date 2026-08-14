from __future__ import annotations

import pytest

from src.benchmarks import (
    BenchmarkBudget,
    BenchmarkConfigurationError,
    BenchmarkMeasurement,
    BenchmarkSuite,
    evaluate_measurement,
    foundation_operations,
    load_benchmark_settings,
    run_foundation_benchmarks,
)


def test_configured_foundation_benchmarks_are_complete_and_within_budget() -> None:
    settings = load_benchmark_settings()

    assert settings["enabled"] is True
    assert settings["enforce"] is True
    assert set(settings["budgets"]) == set(foundation_operations())

    report = run_foundation_benchmarks()

    assert report.enforced is True
    assert report.passed is True
    assert all(result.measurement is not None for result in report.results)


def test_budget_evaluation_reports_each_hard_limit_without_hiding_it() -> None:
    budget = BenchmarkBudget(
        name="contract.probe",
        max_p95_latency_ms=2.0,
        max_peak_memory_mb=1.0,
        warmup_iterations=0,
        measurement_iterations=5,
    )
    measurement = BenchmarkMeasurement(
        median_latency_ms=1.0,
        p95_latency_ms=2.5,
        max_latency_ms=3.0,
        peak_memory_mb=1.25,
        iterations=5,
    )

    result = evaluate_measurement(budget, measurement)

    assert result.passed is False
    assert len(result.violations) == 2
    assert "p95 latency" in result.violations[0]
    assert "peak memory" in result.violations[1]


def test_benchmark_registry_rejects_missing_or_unbudgeted_operations() -> None:
    budget = BenchmarkBudget(
        name="configured",
        max_p95_latency_ms=2.0,
        max_peak_memory_mb=1.0,
        warmup_iterations=0,
        measurement_iterations=5,
    )

    with pytest.raises(BenchmarkConfigurationError, match="must match exactly"):
        BenchmarkSuite({"configured": budget}, {"different": lambda: None})


def test_benchmark_config_rejects_duplicate_contract_names() -> None:
    contract = {
        "name": "duplicate",
        "max_p95_latency_ms": 1.0,
        "max_peak_memory_mb": 1.0,
    }
    config = {
        "benchmark_budgets": {
            "enabled": True,
            "enforce": True,
            "warmup_iterations": 0,
            "measurement_iterations": 5,
            "contracts": [contract, contract],
        }
    }

    with pytest.raises(BenchmarkConfigurationError, match="Duplicate benchmark contract"):
        load_benchmark_settings(config)
