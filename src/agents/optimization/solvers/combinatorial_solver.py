"""Exact finite-domain constraint/combinatorial optimizer.

The implementation follows the search/propagation separation described by
Rossi, van Beek & Walsh (2006), with branch-and-bound reasoning rooted in Land
& Doig (1960).  It is intentionally an exact finite-domain solver, not a
planning heuristic: it enumerates mathematical decision domains, propagates
linear feasibility, bounds linear objectives, and certifies global optimality
only when the search tree is exhausted.
"""
from __future__ import annotations

__version__ = "2.3.0"

import time

from collections.abc import Mapping, Sequence
from typing import Any

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import UnsupportedProblemError
from ..utils.optimization_helpers import Deadline
from ..optimization_math import *
from ..optimization_types import *
from .base_solver import BaseSolver
from logs.logger import configure_logging, PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Combinatorial Optimization Solver")
printer = PrettyPrinter()


class _SearchLimit(RuntimeError):
    def __init__(self, reason: TerminationReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class CombinatorialSolver(BaseSolver):
    solver_name = "finite_domain_exact"
    backend_package = None

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        global_config = load_global_config()
        section = dict(get_config_section("combinatorial_solver", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.combinatorial_config = section
        self.max_nodes = int(section.get("max_nodes", 1_000_000))
        self.max_evaluations = int(section.get("max_evaluations", 1_000_000))
        self.max_domain_cardinality = int(section.get("max_domain_cardinality", 100_000))
        self.propagation = bool(section.get("propagation", True))
        self.objective_bounding = bool(section.get("objective_bounding", True))
        self.variable_ordering = str(section.get("variable_ordering", "smallest_domain"))
        if self.variable_ordering not in {"smallest_domain", "declared"}:
            raise ValueError("combinatorial_solver.variable_ordering must be smallest_domain or declared")
        if min(self.max_nodes, self.max_evaluations, self.max_domain_cardinality) <= 0:
            raise ValueError("combinatorial search limits must be positive")

    @property
    def capability(self) -> SolverCapability:
        return SolverCapability(
            name=self.solver_name,
            problem_classes=frozenset({ProblemClass.CP, ProblemClass.COMBINATORIAL}),
            supports_continuous=False,
            supports_integer=True,
            supports_binary=True,
            supports_finite_domain=True,
            supports_semi_integer=True,
            supports_callable_objective=True,
            supports_quadratic_objective=True,
            supports_nonlinear_constraints=True,
            supports_warm_start=True,
            deterministic=True,
            backend="native exact finite-domain branch-and-bound",
        )

    def solve(self, problem: OptimizationProblem, **options: Any):
        self.ensure_usable(problem)
        if len(problem.objectives) != 1:
            raise UnsupportedProblemError("CombinatorialSolver requires one objective; use multi-objective orchestration for several")
        if any(variable.shape for variable in problem.variables):
            raise UnsupportedProblemError("CombinatorialSolver supports scalar variables only")
        if any(variable.domain.kind in {DomainKind.REAL, DomainKind.SEMI_CONTINUOUS, DomainKind.CUSTOM} for variable in problem.variables):
            raise UnsupportedProblemError("exact finite-domain search cannot enumerate continuous/custom domains")

        max_nodes = int(options.get("max_nodes", self.max_nodes))
        max_evaluations = int(options.get("max_evaluations", self.max_evaluations))
        max_domain_cardinality = int(options.get("max_domain_cardinality", self.max_domain_cardinality))
        if min(max_nodes, max_evaluations, max_domain_cardinality) <= 0:
            raise ValueError("search limits must be positive")
        deadline = Deadline.from_seconds(self.effective_time_limit(options))

        domains: dict[str, tuple[Any, ...]] = {
            variable.name: enumerable_domain(variable, max_cardinality=max_domain_cardinality)
            for variable in problem.variables
        }
        if any(not values for values in domains.values()):
            raise UnsupportedProblemError("empty finite domain encountered")

        objective = problem.objectives[0]
        best_point: dict[str, Any] | None = None
        best_signed = float("inf")
        nodes = 0
        evaluations = 0
        pruned_constraint = 0
        pruned_bound = 0
        propagation_removals = 0
        started = time.perf_counter()

        def check_limits() -> None:
            if deadline.expired():
                raise _SearchLimit(TerminationReason.TIME_LIMIT)
            if nodes >= max_nodes:
                raise _SearchLimit(TerminationReason.NODE_LIMIT)
            if evaluations >= max_evaluations:
                raise _SearchLimit(TerminationReason.EVALUATION_LIMIT)

        def propagate(assignment: Mapping[str, Any], current_domains: Mapping[str, tuple[Any, ...]]) -> dict[str, tuple[Any, ...]] | None:
            nonlocal propagation_removals
            if not self.propagation:
                return dict(current_domains)
            filtered = {name: tuple(values) for name, values in current_domains.items()}
            changed = True
            while changed:
                changed = False
                for name, values in list(filtered.items()):
                    if name in assignment:
                        continue
                    kept: list[Any] = []
                    for value in values:
                        trial = dict(assignment)
                        trial[name] = value
                        possible = True
                        for constraint in problem.constraints:
                            if not partial_linear_constraint_possible(constraint, trial, filtered):
                                possible = False
                                break
                        if possible:
                            kept.append(value)
                    if not kept:
                        return None
                    if len(kept) != len(values):
                        propagation_removals += len(values) - len(kept)
                        filtered[name] = tuple(kept)
                        changed = True
            return filtered

        def choose_variable(assignment: Mapping[str, Any], current_domains: Mapping[str, tuple[Any, ...]]) -> str:
            remaining = [variable.name for variable in problem.variables if variable.name not in assignment]
            if self.variable_ordering == "declared":
                return remaining[0]
            return min(remaining, key=lambda name: (len(current_domains[name]), remaining.index(name)))

        def search(assignment: dict[str, Any], current_domains: dict[str, tuple[Any, ...]]) -> None:
            nonlocal nodes, evaluations, best_point, best_signed, pruned_constraint, pruned_bound
            check_limits()
            nodes += 1

            propagated = propagate(assignment, current_domains)
            if propagated is None:
                pruned_constraint += 1
                return
            for constraint in problem.constraints:
                if not partial_linear_constraint_possible(constraint, assignment, propagated):
                    pruned_constraint += 1
                    return

            if self.objective_bounding and best_point is not None:
                optimistic = optimistic_linear_objective_bound(objective, assignment, propagated)
                if optimistic is not None and optimistic >= best_signed - self.optimality_tolerance:
                    pruned_bound += 1
                    return

            if len(assignment) == len(problem.variables):
                check_limits()
                evaluations += 1
                if not is_feasible(problem, assignment, tolerance=self.feasibility_tolerance):
                    return
                value = signed_objective_value(objective, assignment)
                if best_point is None or value < best_signed - self.optimality_tolerance:
                    best_signed = value
                    best_point = dict(assignment)
                return

            name = choose_variable(assignment, propagated)
            for value in propagated[name]:
                child = dict(assignment)
                child[name] = value
                search(child, propagated)

        limit_reason: TerminationReason | None = None
        try:
            search({}, domains)
        except _SearchLimit as stop:
            limit_reason = stop.reason
        runtime = self.timed(started)

        if limit_reason is None:
            if best_point is None:
                return self._solution(
                    problem,
                    status=SolverStatus.INFEASIBLE,
                    termination=TerminationReason.INFEASIBLE,
                    success=False,
                    point=None,
                    optimality=OptimalityStatus.NONE,
                    runtime_seconds=runtime,
                    nodes=nodes,
                    evaluations=evaluations,
                    diagnostics={
                        "search_exhausted": True,
                        "constraint_prunes": pruned_constraint,
                        "objective_bound_prunes": pruned_bound,
                        "propagation_removals": propagation_removals,
                    },
                )
            original_value = -best_signed if objective.sense.value == "maximize" else best_signed
            return self._solution(
                problem,
                status=SolverStatus.OPTIMAL,
                termination=TerminationReason.OPTIMAL,
                success=True,
                point=best_point,
                optimality=OptimalityStatus.GLOBAL,
                runtime_seconds=runtime,
                primal_bound=original_value,
                dual_bound=original_value,
                absolute_gap=0.0,
                relative_gap=0.0,
                nodes=nodes,
                evaluations=evaluations,
                diagnostics={
                    "search_exhausted": True,
                    "constraint_prunes": pruned_constraint,
                    "objective_bound_prunes": pruned_bound,
                    "propagation_removals": propagation_removals,
                    "variable_ordering": self.variable_ordering,
                },
            )

        status = SolverStatus.TIMEOUT if limit_reason is TerminationReason.TIME_LIMIT else SolverStatus.LIMIT_REACHED
        original_value = None
        if best_point is not None:
            original_value = -best_signed if objective.sense.value == "maximize" else best_signed
        return self._solution(
            problem,
            status=status,
            termination=limit_reason,
            success=best_point is not None,
            point=best_point,
            optimality=OptimalityStatus.NONE,
            runtime_seconds=runtime,
            primal_bound=original_value,
            nodes=nodes,
            evaluations=evaluations,
            diagnostics={
                "search_exhausted": False,
                "constraint_prunes": pruned_constraint,
                "objective_bound_prunes": pruned_bound,
                "propagation_removals": propagation_removals,
                "variable_ordering": self.variable_ordering,
            },
        )


__all__ = ["CombinatorialSolver"]


def _smoke_test() -> None:
    configure_logging()
    smoke_logger = get_logger("CombinatorialSolverSmoke")
    problem = OptimizationProblem(
        "combinatorial_smoke",
        (Variable("x", VariableDomain.finite((1, 2, 3))),),
        (Objective("f", function=lambda p: float((p["x"] - 2) ** 2)),),
        problem_class=ProblemClass.CP,
    )
    result = CombinatorialSolver().solve(problem)
    assert result.status is SolverStatus.OPTIMAL
    assert result.primal_solution == {"x": 2}
    smoke_logger.info("CombinatorialSolver smoke test passed | objective=%s", result.objective_value)


if __name__ == "__main__":
    _smoke_test()
