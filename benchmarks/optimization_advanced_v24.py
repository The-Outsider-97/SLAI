"""Deterministic baseline measurements for SLAI v2.4 Optimization.

Run from repository root:
    py -m benchmarks.optimization_advanced_v24 --repeat 3 --json benchmark.json

Results are measurements, not implied performance improvements or global proofs.
Do not run this script on production models; it uses synthetic test objectives.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from src.agents.optimization.domain_problems import expected_scenario_problem
from src.agents.optimization.optimization_math import with_classification
from src.agents.optimization.optimization_types import (
    Objective, OptimizationProblem, ProblemClass, Variable, VariableDomain,
)
from src.agents.optimization.solvers.black_box_solver import BlackBoxSolver
from src.agents.optimization.optimization_solver import OptimizationSolver


def cases():
    scalar = OptimizationProblem(
        "benchmark_derivative_free",
        (Variable("x", VariableDomain.real(-5, 5), initial=0.),),
        (Objective("loss", function=lambda p: (p["x"] - 2.5)**2),),
        problem_class=ProblemClass.BLACK_BOX,
    )
    mixed = OptimizationProblem(
        "benchmark_mixed",
        (Variable("x", VariableDomain.real(-5, 5), initial=0.),
         Variable("n", VariableDomain.integer(0, 4), initial=0)),
        (Objective("loss", function=lambda p:
                   (p["x"] - 2.5)**2 + (p["n"] - 3)**2),),
        problem_class=ProblemClass.BLACK_BOX,
    )
    scenario = expected_scenario_problem(
        "benchmark_expectation",
        variables=(Variable("x", VariableDomain.real(-3, 3), initial=0.),),
        scenario_losses=(lambda p: (p["x"] - 1)**2,
                         lambda p: (p["x"] + 1)**2),
    )
    return (
        ("scalar_pattern", scalar, "pattern"),
        ("mixed_pattern", mixed, "pattern"),
        ("scenario_nonlinear", scenario, "nonlinear"),
    )


def measure(repeats: int) -> dict:
    records = []
    for label, problem, kind in cases():
        times = []
        details = []
        for _ in range(repeats):
            started = time.perf_counter()
            if kind == "pattern":
                result = BlackBoxSolver({
                    "method": "pattern_search",
                    "max_evaluations": 200,
                }).solve(problem, method="pattern_search", max_evaluations=200)
            else:
                output = OptimizationSolver(config={
                    "use_memory_warm_start": False,
                }).solve_single_detailed(
                    with_classification(problem),
                    solver="scipy_nonlinear",
                    options={"max_iterations": 150},
                )
                if output.solution is None:
                    raise RuntimeError("solver did not produce a normalized solution")
                result = output.solution
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            details.append({
                "status": result.status.value,
                "feasibility": result.feasibility.value,
                "optimality": result.optimality.value,
                "evaluations": result.function_evaluation_count,
                "objective": result.objective_value,
                "point": dict(result.primal_solution) if result.primal_solution else None,
            })
        records.append({
            "case": label, "repetitions": repeats,
            "wall_seconds_min": min(times),
            "wall_seconds_median": statistics.median(times),
            "wall_seconds_max": max(times),
            "runs": details,
        })
    return {
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "cases": records,
        "warnings": [
            "Synthetic benchmark only.",
            "Wall-clock results are environment-dependent.",
            "Search completion is not a certificate of global optimality.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 100:
        parser.error("--repeat must be between 1 and 100")
    payload = measure(args.repeat)
    output = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
