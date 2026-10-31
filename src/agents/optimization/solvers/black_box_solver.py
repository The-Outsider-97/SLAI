"""Generic derivative-free / black-box mathematical optimization.

This module is intentionally independent of ``src.tuning``.  Decision variables
are arbitrary optimization variables and callbacks are caller-supplied
mathematical evaluations.  The implementation is informed by Conn, Scheinberg
& Vicente (2009), Audet & Dennis (2006), and Jones, Schonlau & Welch (1998).

Two concrete methods are provided:
- ``differential_evolution``: bounded population-based global search;
- ``pattern_search``: deterministic mesh-refining direct search suitable for
  local derivative-free refinement.

Neither method fabricates a global optimality certificate.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import time
import numpy as np # type: ignore

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from scipy.optimize import differential_evolution # type: ignore

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from ..optimization_math import *
from ..optimization_memory import OptimizationMemory
from ..evaluation_runtime import EvaluationJournal
from ..optimization_types import *
from .base_solver import BaseSolver
from logs.logger import configure_logging, get_logger,  PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Black-Box Optimization Solver")
printer = PrettyPrinter()


class _BudgetStop(RuntimeError):
    def __init__(self, reason: TerminationReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass
class _EvaluationState:
    evaluations: int = 0
    cache_hits: int = 0
    failures: int = 0
    best_feasible_point: dict[str, Any] | None = None
    best_feasible_signed: float = float("inf")
    best_violation_point: dict[str, Any] | None = None
    best_violation: float = float("inf")


class BlackBoxSolver(BaseSolver):
    solver_name = "black_box_derivative_free"
    backend_package = "scipy"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        global_config = load_global_config()
        section = dict(get_config_section("black_box_solver", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.black_box_config = section
        self.method = str(section.get("method", "differential_evolution"))
        if self.method not in {"differential_evolution", "pattern_search"}:
            raise ValueError("black_box_solver.method must be differential_evolution or pattern_search")
        self.max_evaluations = int(section.get("max_evaluations", 5000))
        self.seed = int(section.get("seed", 0))
        self.popsize = int(section.get("popsize", 12))
        self.tolerance = float(section.get("tolerance", 1e-7))
        self.constraint_penalty = float(section.get("constraint_penalty", 1e6))
        self.failure_penalty = float(section.get("failure_penalty", 1e30))
        self.polish = bool(section.get("polish", False))
        self.pattern_initial_mesh = float(section.get("pattern_initial_mesh", 0.25))
        self.pattern_mesh_reduction = float(section.get("pattern_mesh_reduction", 0.5))
        self.pattern_min_mesh = float(section.get("pattern_min_mesh", 1e-6))
        if self.max_evaluations <= 0 or self.popsize <= 0:
            raise ValueError("black-box evaluation/population limits must be positive")
        if self.constraint_penalty <= 0 or self.failure_penalty <= 0:
            raise ValueError("black-box penalties must be positive")
        if not (0 < self.pattern_mesh_reduction < 1):
            raise ValueError("pattern_mesh_reduction must be between 0 and 1")

    @property
    def capability(self) -> SolverCapability:
        return SolverCapability(
            name=self.solver_name,
            problem_classes=frozenset({ProblemClass.BLACK_BOX}),
            supports_continuous=True,
            supports_integer=True,
            supports_binary=True,
            supports_finite_domain=True,
            supports_callable_objective=True,
            supports_quadratic_objective=True,
            supports_nonlinear_constraints=True,
            supports_black_box=True,
            supports_warm_start=True,
            requires_finite_bounds=True,
            deterministic=True,
            backend="native pattern search / scipy.optimize.differential_evolution",
        )

    def solve(self, problem: OptimizationProblem, **options: Any):
        self.ensure_usable(problem)
        if len(problem.objectives) != 1:
            raise UnsupportedProblemError("BlackBoxSolver requires one objective")
        if any(variable.shape for variable in problem.variables):
            raise UnsupportedProblemError("BlackBoxSolver supports scalar variables only")
        method = str(options.get("method", self.method))
        if method not in {"differential_evolution", "pattern_search"}:
            raise UnsupportedProblemError("unknown black-box method", context={"method": method})

        bounds: list[tuple[float, float]] = []
        integrality: list[bool] = []
        for variable in problem.variables:
            domain = variable.domain
            if domain.kind is DomainKind.FINITE:
                bounds.append((0.0, float(len(domain.values) - 1)))
                integrality.append(True)
            elif domain.kind is DomainKind.BINARY:
                bounds.append((0.0, 1.0))
                integrality.append(True)
            elif domain.kind is DomainKind.INTEGER:
                if domain.lower is None or domain.upper is None:
                    raise UnsupportedProblemError("black-box integer variable requires finite bounds", context={"variable": variable.name})
                bounds.append((domain.lower, domain.upper)); integrality.append(True)
            elif domain.kind is DomainKind.REAL:
                if domain.lower is None or domain.upper is None:
                    raise UnsupportedProblemError("black-box real variable requires finite bounds", context={"variable": variable.name})
                bounds.append((domain.lower, domain.upper)); integrality.append(False)
            else:
                raise UnsupportedProblemError("black-box solver does not support this domain", context={"variable": variable.name, "domain": domain.kind.value})

        budget = int(options.get("max_evaluations", self.max_evaluations))
        if budget <= 0:
            raise ValueError("max_evaluations must be positive")
        deadline = Deadline.from_seconds(self.effective_time_limit(options))
        cancel_event = options.get("cancel_event")
        if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
            raise OptimizationValidationError("cancel_event must expose is_set()")
        resume = options.get("resume", False)
        if not isinstance(resume, bool):
            raise OptimizationValidationError("resume must be boolean")
        checkpoint_path = options.get("checkpoint_path")
        if resume and (checkpoint_path is None or method != "pattern_search"):
            raise UnsupportedProblemError("resume requires checkpoint_path and pattern_search")
        if checkpoint_path is not None and not problem.deterministic:
            raise UnsupportedProblemError("checkpoint caching requires deterministic evaluator semantics")
        journal = None
        if checkpoint_path is not None:
            if not isinstance(checkpoint_path, (str, __import__("os").PathLike)):
                raise OptimizationValidationError("checkpoint_path must be a filesystem path")
            revision = options.get("evaluator_revision")
            if not isinstance(revision, str) or not revision.strip():
                raise OptimizationValidationError("checkpoint requires explicit evaluator_revision")
            from ..utils.optimization_helpers import stable_fingerprint
            identity = stable_fingerprint({
                "problem": OptimizationMemory.problem_fingerprint(problem),
                "revision": revision,
                "method": method,
                "solver_version": self.solver_version(),
                "seed": options.get("seed", self.seed),
                "tolerance": options.get("tolerance", self.tolerance),
                "constraint_penalty": self.constraint_penalty,
                "failure_penalty": self.failure_penalty,
                "pattern_initial_mesh": options.get("initial_mesh", self.pattern_initial_mesh),
                "pattern_mesh_reduction": options.get("mesh_reduction", self.pattern_mesh_reduction),
                "pattern_min_mesh": options.get("min_mesh", self.pattern_min_mesh),
            })
            journal = EvaluationJournal(
                checkpoint_path, identity=identity, resume=resume,
                maximum_bytes=int(options.get("checkpoint_max_bytes", 32 * 1024 * 1024)),
            )
        state = _EvaluationState()
        cache: dict[str, tuple[float, float, dict[str, Any]]] = {}
        objective = problem.objectives[0]
        if journal is not None and resume:
            for cache_key, entry in journal.entries.items():
                point = dict(entry["point"])
                cache[cache_key] = (entry["signed"], entry["violation"], point)
                state.evaluations += 1
                if entry["failed"]:
                    state.failures += 1
                else:
                    if entry["violation"] < state.best_violation:
                        state.best_violation = entry["violation"]
                        state.best_violation_point = dict(point)
                    if is_feasible(problem, point, tolerance=self.feasibility_tolerance) and entry["signed"] < state.best_feasible_signed:
                        state.best_feasible_signed = entry["signed"]
                        state.best_feasible_point = dict(point)

        def evaluate_vector(x: Sequence[float]) -> tuple[float, float, dict[str, Any]]:
            if cancel_event is not None and cancel_event.is_set():
                raise _BudgetStop(TerminationReason.INTERRUPTED)
            if deadline.expired():
                raise _BudgetStop(TerminationReason.TIME_LIMIT)
            point = point_from_vector(problem, x)
            key = stable_fingerprint(tuple((name, point[name]) for name in problem.variable_names))
            if key in cache:
                state.cache_hits += 1
                return cache[key]
            if state.evaluations >= budget:
                raise _BudgetStop(TerminationReason.EVALUATION_LIMIT)
            state.evaluations += 1
            failed = False
            try:
                raw = evaluate_objective(objective, point)
                signed = raw if objective.sense is ObjectiveSense.MINIMIZE else -raw
                violation = max_constraint_violation(problem, point)
                if violation is None or not math.isfinite(violation):
                    raise OptimizationEvaluationError("black-box constraint evaluation produced invalid violation")
            except OptimizationEvaluationError:
                state.failures += 1
                failed = True
                signed, violation = self.failure_penalty, self.failure_penalty
            if violation < state.best_violation:
                state.best_violation = violation
                state.best_violation_point = dict(point)
            if not failed and is_feasible(problem, point, tolerance=self.feasibility_tolerance) and signed < state.best_feasible_signed:
                state.best_feasible_signed = signed
                state.best_feasible_point = dict(point)
            result = (signed, violation, dict(point))
            cache[key] = result
            if journal is not None:
                journal.record(key, point=point, signed=signed, violation=violation, failed=failed)
            return result

        def merit(x: Sequence[float]) -> float:
            signed, violation, _ = evaluate_vector(x)
            if not math.isfinite(signed) or not math.isfinite(violation):
                return self.failure_penalty
            return signed + self.constraint_penalty * max(0.0, violation)

        started = time.perf_counter()
        limit_reason: TerminationReason | None = None
        backend_message = ""
        iterations: int | None = None

        try:
            if method == "differential_evolution":
                maxiter = max(1, int(math.ceil(budget / max(1, self.popsize * len(bounds)))) - 1)
                result = differential_evolution(
                    merit,
                    bounds,
                    strategy=str(options.get("strategy", "best1bin")),
                    maxiter=maxiter,
                    popsize=int(options.get("popsize", self.popsize)),
                    tol=float(options.get("tolerance", self.tolerance)),
                    atol=float(options.get("absolute_tolerance", 0.0)),
                    rng=np.random.default_rng(int(options.get("seed", self.seed))),
                    polish=bool(options.get("polish", self.polish)),
                    updating="immediate",
                    workers=1,
                    integrality=np.asarray(integrality, dtype=bool),
                    x0=np.asarray(options["x0"], dtype=float) if "x0" in options else None,
                )
                backend_message = str(result.message)
                iterations = int(getattr(result, "nit", 0))
            else:
                iterations = self._pattern_search(
                    problem, bounds, integrality, merit, evaluate_vector, options,
                    journal=journal, cancel_event=cancel_event,
                )
                backend_message = "pattern mesh converged"
        except _BudgetStop as stop:
            limit_reason = stop.reason
            backend_message = stop.reason.value

        runtime = self.timed(started)
        point = state.best_feasible_point
        if point is not None:
            primal = evaluate_objective(objective, point)
        else:
            primal = None

        if limit_reason is not None:
            status = (
                SolverStatus.TIMEOUT if limit_reason is TerminationReason.TIME_LIMIT
                else SolverStatus.INTERRUPTED if limit_reason is TerminationReason.INTERRUPTED
                else SolverStatus.LIMIT_REACHED
            )
            return self._solution(
                problem,
                status=status,
                termination=limit_reason,
                success=point is not None,
                point=point,
                optimality=OptimalityStatus.NONE,
                runtime_seconds=runtime,
                primal_bound=primal,
                iterations=iterations,
                evaluations=state.evaluations,
                diagnostics=self._diagnostics(method, state, cache, backend_message, budget),
            )

        if point is None:
            # Failure to discover feasibility is not a proof of infeasibility in
            # black-box search.
            return self._solution(
                problem,
                status=SolverStatus.LIMIT_REACHED,
                termination=TerminationReason.EVALUATION_LIMIT if state.evaluations >= budget else TerminationReason.CONVERGED,
                success=False,
                point=None,
                optimality=OptimalityStatus.NONE,
                runtime_seconds=runtime,
                iterations=iterations,
                evaluations=state.evaluations,
                diagnostics=self._diagnostics(method, state, cache, backend_message, budget),
                warnings=("No feasible black-box point was found; infeasibility is not certified.",),
            )

        # Derivative-free heuristics return a feasible incumbent, never an
        # unjustified global certificate.
        optimality = OptimalityStatus.LOCAL if method == "pattern_search" else OptimalityStatus.NONE
        return self._solution(
            problem,
            status=SolverStatus.FEASIBLE,
            termination=TerminationReason.CONVERGED,
            success=True,
            point=point,
            optimality=optimality,
            runtime_seconds=runtime,
            primal_bound=primal,
            iterations=iterations,
            evaluations=state.evaluations,
            diagnostics=self._diagnostics(method, state, cache, backend_message, budget),
        )

    def _pattern_search(
        self,
        problem: OptimizationProblem,
        bounds: Sequence[tuple[float, float]],
        integrality: Sequence[bool],
        merit: Any,
        evaluate_vector: Any,
        options: Mapping[str, Any],
        *,
        journal: EvaluationJournal | None = None,
        cancel_event: Any = None,
    ) -> int:
        x = np.asarray(options.get("x0", initial_vector(problem)), dtype=float)
        lower = np.asarray([bound[0] for bound in bounds], dtype=float)
        upper = np.asarray([bound[1] for bound in bounds], dtype=float)
        x = np.minimum(np.maximum(x, lower), upper)
        span = np.maximum(upper - lower, 1.0)
        mesh = float(options.get("initial_mesh", self.pattern_initial_mesh))
        reduction = float(options.get("mesh_reduction", self.pattern_mesh_reduction))
        minimum = float(options.get("min_mesh", self.pattern_min_mesh))
        current = merit(x)
        iterations = 0
        if journal is not None and journal.progress is not None:
            previous = journal.progress
            try:
                x = np.asarray(previous["x"], dtype=float)
                mesh = float(previous["mesh"])
                iterations = int(previous["iterations"])
                current = float(previous["current"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise OptimizationValidationError("corrupt pattern-search progress checkpoint") from exc
            if len(x) != len(bounds) or not np.all(np.isfinite(x)) or not math.isfinite(mesh) or not math.isfinite(current) or mesh <= 0 or iterations < 0:
                raise OptimizationValidationError("invalid pattern-search progress checkpoint")
        elif journal is not None:
            journal.set_progress({"x": x.tolist(), "mesh": mesh, "current": current, "iterations": iterations})
        while mesh > minimum:
            if cancel_event is not None and cancel_event.is_set():
                raise _BudgetStop(TerminationReason.INTERRUPTED)
            iterations += 1
            improved = False
            for axis in range(len(x)):
                step = mesh * span[axis]
                if integrality[axis]:
                    step = max(1.0, round(step))
                for direction in (-1.0, 1.0):
                    candidate = x.copy()
                    candidate[axis] += direction * step
                    candidate = np.minimum(np.maximum(candidate, lower), upper)
                    if integrality[axis]:
                        candidate[axis] = round(candidate[axis])
                    candidate_merit = merit(candidate)
                    if candidate_merit < current - self.tolerance:
                        x, current = candidate, candidate_merit
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                mesh *= reduction
            if journal is not None:
                journal.set_progress({"x": x.tolist(), "mesh": mesh, "current": current, "iterations": iterations})
        evaluate_vector(x)
        return iterations

    @staticmethod
    def _diagnostics(
        method: str,
        state: _EvaluationState,
        cache: Mapping[str, Any],
        backend_message: str,
        budget: int,
    ) -> dict[str, Any]:
        return {
            "method": method,
            "message": backend_message,
            "evaluation_budget": budget,
            "unique_evaluations": state.evaluations,
            "cache_hits": state.cache_hits,
            "evaluation_failures": state.failures,
            "cached_points": len(cache),
            "best_violation": None if math.isinf(state.best_violation) else state.best_violation,
        }


__all__ = ["BlackBoxSolver"]


if __name__ == "__main__":
    def _smoke_test() -> None:
        configure_logging()
        smoke_logger = get_logger("BlackBoxSolverSmoke")
        problem = OptimizationProblem(
            "black_box_smoke",
            (Variable("x", VariableDomain.real(-2, 2), initial=0.0),),
            (Objective("f", function=lambda p: (p["x"] - 0.5) ** 2),),
            problem_class=ProblemClass.BLACK_BOX,
        )
        result = BlackBoxSolver({"method": "pattern_search", "max_evaluations": 250}).solve(problem)
        assert result.success and result.primal_solution is not None
        assert abs(result.primal_solution["x"] - 0.5) <= 1e-3
        smoke_logger.info("BlackBoxSolver smoke test passed | objective=%s", result.objective_value)

    _smoke_test()
