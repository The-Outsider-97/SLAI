"""Production orchestration for SLAI mathematical optimization solvers.

The orchestrator performs classification, capability-based solver selection,
deterministic fallback, memory-backed warm starts and evidence recording, and
multi-objective transformations.  It does not own planning, simulation, or
SLAI/model hyperparameter tuning.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Mapping, Sequence
from dataclasses import replace
from time import monotonic
from typing import Any

from .utils.config_loader import get_config_section, load_global_config
from .utils.optimization_errors import *
from .modules.multiobjective import Multiobjective
from .optimization_math import vector_from_point, with_classification
from .optimization_memory import OptimizationMemory
from .optimization_registry import OptimizationRegistries
from .optimization_types import *
from logs.logger import PrettyPrinter, configure_logging, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Solver")
printer = PrettyPrinter()


class OptimizationSolver:
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        memory: OptimizationMemory | None = None,
        registries: OptimizationRegistries | None = None,
    ) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("optimization_solver", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.solver_config = section
        self.memory = memory or OptimizationMemory()
        self.registries = registries or OptimizationRegistries(memory=self.memory)
        self.multiobjective = Multiobjective()
        self.allow_fallback = bool(section.get("allow_fallback", True))
        self.use_memory_warm_start = bool(section.get("use_memory_warm_start", True))
        raw_statuses = section.get(
            "fallback_statuses",
            [
                SolverStatus.NUMERICAL_FAILURE.value,
                SolverStatus.ERROR.value,
                SolverStatus.SOLVER_UNAVAILABLE.value,
                SolverStatus.UNSUPPORTED.value,
            ],
        )
        self.fallback_statuses = {SolverStatus(str(status)) for status in raw_statuses}

    @staticmethod
    def _classified(problem: OptimizationProblem) -> OptimizationProblem:
        if problem.is_multi_objective:
            return problem
        if problem.problem_class is ProblemClass.UNKNOWN:
            return with_classification(problem)
        # Validate that the declared class remains structurally consistent.
        with_classification(problem)
        return problem

    def _solver_options(
        self,
        problem: OptimizationProblem,
        solver_name: str,
        options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(options or {})
        solver = self.registries.solvers.get(solver_name)
        if self.use_memory_warm_start and solver.capability.supports_warm_start and "x0" not in merged:
            warm = self.memory.warm_start(problem)
            if warm is not None:
                try:
                    merged["x0"] = vector_from_point(problem, warm)
                    merged["warm_start_source"] = "optimization_memory"
                except OptimizationError:
                    logger.warning("Stored optimization warm start rejected | problem=%s | solver=%s", problem.name, solver_name)
        return merged

    def solve_single_detailed(
        self,
        problem: OptimizationProblem,
        *,
        solver: str | None = None,
        options: Mapping[str, Any] | None = None,
        allow_fallback: bool | None = None,
    ) -> OptimizationResult:
        model = self._classified(problem)
        if model.is_multi_objective or model.problem_class is ProblemClass.MULTIOBJECTIVE:
            raise UnsupportedProblemError("solve_single_detailed cannot solve a multi-objective problem directly")
        fallback = self.allow_fallback if allow_fallback is None else bool(allow_fallback)

        inspection = list(self.registries.solvers.inspect(model))
        if solver is not None:
            candidates = (self.registries.select(model, requested=solver),)
        else:
            candidates = tuple(item.solver for item in self.registries.solvers.ranked(model))
            if not candidates:
                # Raises a precise unavailable/incompatible error with evidence.
                self.registries.select(model)
                raise SolverExecutionError("solver registry returned no candidates without an explicit selection error")

        attempt_updates: dict[str, SolverAttempt] = {attempt.solver_name: attempt for attempt in inspection}
        last_solution: Solution | None = None
        errors: list[dict[str, str]] = []

        for index, adapter in enumerate(candidates):
            solver_options = self._solver_options(model, adapter.solver_name, options)
            logger.info(
                "Optimization solve attempt | problem=%s | class=%s | solver=%s | attempt=%d/%d",
                model.name,
                model.problem_class.value,
                adapter.solver_name,
                index + 1,
                len(candidates),
            )
            attempt_started = monotonic()
            try:
                result = adapter.solve(model, **solver_options)
            except (SolverUnavailableError, IncompatibleSolverError) as exc:
                errors.append({"solver": adapter.solver_name, "error": str(exc)})
                previous = attempt_updates.get(adapter.solver_name)
                attempt_updates[adapter.solver_name] = SolverAttempt(
                    solver_name=adapter.solver_name,
                    available=not isinstance(exc, SolverUnavailableError),
                    compatible=not isinstance(exc, IncompatibleSolverError),
                    preference_score=None if previous is None else previous.preference_score,
                    incompatibilities=() if previous is None else previous.incompatibilities,
                    status=SolverStatus.SOLVER_UNAVAILABLE if isinstance(exc, SolverUnavailableError) else SolverStatus.UNSUPPORTED,
                    termination_reason=TerminationReason.SOLVER_UNAVAILABLE if isinstance(exc, SolverUnavailableError) else TerminationReason.UNSUPPORTED,
                    runtime_seconds=monotonic() - attempt_started,
                    error=str(exc),
                )
                if not fallback:
                    raise
                continue
            except OptimizationError as exc:
                errors.append({"solver": adapter.solver_name, "error": str(exc)})
                previous = attempt_updates.get(adapter.solver_name)
                attempt_updates[adapter.solver_name] = SolverAttempt(
                    solver_name=adapter.solver_name,
                    available=True if previous is None else previous.available,
                    compatible=True if previous is None else previous.compatible,
                    preference_score=None if previous is None else previous.preference_score,
                    incompatibilities=() if previous is None else previous.incompatibilities,
                    status=SolverStatus.ERROR,
                    termination_reason=TerminationReason.ERROR,
                    runtime_seconds=monotonic() - attempt_started,
                    error=str(exc),
                )
                if not fallback:
                    raise
                continue
            except (ArithmeticError, ValueError, TypeError, RuntimeError) as exc:
                wrapped = SolverExecutionError(
                    "unexpected solver backend failure",
                    context={"solver": adapter.solver_name, "problem": model.name},
                    cause=exc,
                )
                errors.append({"solver": adapter.solver_name, "error": str(wrapped)})
                previous = attempt_updates.get(adapter.solver_name)
                attempt_updates[adapter.solver_name] = SolverAttempt(
                    solver_name=adapter.solver_name,
                    available=True if previous is None else previous.available,
                    compatible=True if previous is None else previous.compatible,
                    preference_score=None if previous is None else previous.preference_score,
                    incompatibilities=() if previous is None else previous.incompatibilities,
                    status=SolverStatus.ERROR,
                    termination_reason=TerminationReason.ERROR,
                    runtime_seconds=monotonic() - attempt_started,
                    error=str(wrapped),
                )
                if not fallback:
                    raise wrapped from exc
                continue

            last_solution = result
            try:
                self.memory.record(
                    model,
                    result,
                    options=solver_options,
                    metadata={"orchestrator_attempt": index + 1},
                )
            except (OptimizationError, OSError) as exc:
                # Solver evidence persistence is operational bookkeeping, not a
                # mathematical precondition for accepting a normalized solver
                # result. Preserve the solution and surface the failed local
                # evidence write explicitly for the agent/runtime layer.
                logger.warning(
                    "Optimization result valid but evidence recording failed | problem=%s | solver=%s | error=%s",
                    model.name,
                    adapter.solver_name,
                    exc,
                )
                result = replace(
                    result,
                    warnings=tuple(result.warnings)
                    + (f"optimization_memory_persistence_failed: {type(exc).__name__}: {exc}",),
                    metadata={
                        **dict(result.metadata),
                        "optimization_memory_persistence_error": f"{type(exc).__name__}: {exc}",
                    },
                )
                last_solution = result
            previous = attempt_updates.get(adapter.solver_name)
            attempt_updates[adapter.solver_name] = SolverAttempt(
                solver_name=adapter.solver_name,
                available=True,
                compatible=True,
                preference_score=None if previous is None else previous.preference_score,
                incompatibilities=(),
                status=result.status,
                termination_reason=result.termination_reason,
                runtime_seconds=result.runtime_seconds,
            )
            if result.success:
                selection = SolverSelection(
                    selected_solver=adapter.solver_name,
                    attempts=tuple(attempt_updates[name] for name in sorted(attempt_updates)),
                    requested_solver=solver,
                )
                return OptimizationResult(solution=result, selection=selection, metadata={"fallback_count": index})
            if not fallback or result.status not in self.fallback_statuses:
                selection = SolverSelection(
                    selected_solver=adapter.solver_name,
                    attempts=tuple(attempt_updates[name] for name in sorted(attempt_updates)),
                    requested_solver=solver,
                )
                return OptimizationResult(solution=result, selection=selection, metadata={"fallback_count": index})

        if last_solution is not None:
            selection = SolverSelection(
                selected_solver=last_solution.solver_name,
                attempts=tuple(attempt_updates[name] for name in sorted(attempt_updates)),
                requested_solver=solver,
            )
            return OptimizationResult(solution=last_solution, selection=selection, metadata={"errors": errors})
        raise SolverExecutionError(
            "all eligible optimization solvers failed before producing a normalized result",
            context={"problem": model.name, "errors": errors},
        )

    def solve_single(
        self,
        problem: OptimizationProblem,
        *,
        solver: str | None = None,
        options: Mapping[str, Any] | None = None,
        allow_fallback: bool | None = None,
    ) -> Solution:
        result = self.solve_single_detailed(problem, solver=solver, options=options, allow_fallback=allow_fallback)
        if result.solution is None:
            raise SolverExecutionError("single-objective orchestration returned no solution")
        return result.solution

    def solve_multiobjective(
        self,
        problem: OptimizationProblem,
        *,
        method: str = "weighted",
        weight_vectors: Sequence[Mapping[str, float]] | None = None,
        epsilon_grid: Sequence[Mapping[str, float]] | None = None,
        primary: str | None = None,
        solver: str | None = None,
        options: Mapping[str, Any] | None = None,
        allow_fallback: bool | None = None,
    ) -> OptimizationResult:
        if not problem.is_multi_objective:
            raise UnsupportedProblemError("multi-objective solve requires at least two objectives")

        persistence_errors: list[str] = []

        def solve_scalar(subproblem: OptimizationProblem) -> Solution:
            solution = self.solve_single(
                subproblem,
                solver=solver,
                options=options,
                allow_fallback=allow_fallback,
            )
            error = solution.metadata.get("optimization_memory_persistence_error")
            if error:
                persistence_errors.append(str(error))
            return solution

        def metadata(method_name: str) -> dict[str, Any]:
            payload: dict[str, Any] = {"method": method_name}
            if persistence_errors:
                # Multi-objective transformations may solve several scalar
                # subproblems. Preserve local-evidence failures without
                # altering the mathematical Pareto/lexicographic result.
                payload["optimization_memory_persistence_errors"] = tuple(persistence_errors)
            return payload

        normalized = method.strip().lower()
        if normalized in {"weighted", "weighted_sum", "scalarization"}:
            front = self.multiobjective.solve_weighted_front(problem, solve_scalar, weight_vectors=weight_vectors)
            return OptimizationResult(solution=None, pareto_front=front, metadata=metadata("weighted_scalarization"))
        if normalized in {"epsilon", "epsilon_constraint"}:
            if primary is None or epsilon_grid is None:
                raise UnsupportedProblemError("epsilon-constraint solve requires primary objective and epsilon_grid")
            front = self.multiobjective.solve_epsilon_front(problem, solve_scalar, primary=primary, epsilon_grid=epsilon_grid)
            return OptimizationResult(solution=None, pareto_front=front, metadata=metadata("epsilon_constraint"))
        if normalized in {"lexicographic", "lex"}:
            solution = self.multiobjective.lexicographic(problem, solve_scalar)
            return OptimizationResult(solution=solution, metadata=metadata("lexicographic"))
        raise UnsupportedProblemError("unsupported multi-objective orchestration method", context={"method": method})

    def solve(self, problem: OptimizationProblem, **kwargs: Any) -> Solution | OptimizationResult:
        if problem.is_multi_objective:
            return self.solve_multiobjective(problem, **kwargs)
        return self.solve_single(
            problem,
            solver=kwargs.get("solver"),
            options=kwargs.get("options"),
            allow_fallback=kwargs.get("allow_fallback"),
        )


__all__ = ["OptimizationSolver"]


if __name__ == "__main__":
    def _smoke_test() -> None:
        configure_logging()
        smoke_logger = get_logger("OptimizationSolverSmoke")
        problem = OptimizationProblem(
            "orchestrator_smoke",
            (Variable("x", VariableDomain.real(0, 3)),),
            (Objective("f", ObjectiveSense.MAXIMIZE, linear={"x": 1}),),
            problem_class=ProblemClass.LP,
        )
        result = OptimizationSolver().solve_single_detailed(problem)
        assert result.solution is not None and result.solution.status is SolverStatus.OPTIMAL
        assert result.selection is not None and result.selection.selected_solver == "scipy_highs_linear"
        smoke_logger.info("OptimizationSolver smoke test passed | solver=%s", result.selection.selected_solver)


    _smoke_test()
