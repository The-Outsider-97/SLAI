"""Algorithm-independent multi-objective optimization transformations.

Miettinen (1998) is the primary foundation for Pareto optimality,
scalarization, epsilon-constraint and lexicographic approaches.  Deb et al.
(2002) motivates evolutionary Pareto search, but no NSGA-II implementation is
claimed here; the subsystem keeps Pareto semantics independent of any one
search algorithm.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import MultiObjectiveError, OptimizationValidationError
from ..utils.optimization_helpers import require_finite
from ..registries.scalarization_registry import ScalarizationRegistry
from ..optimization_math import evaluate_objective, objective_values, with_classification
from ..optimization_types import *
from .pareto_efficiency import Candidate, ParetoEfficiency
from logs.logger import configure_logging, get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Multiobjective")
printer = PrettyPrinter()

SingleSolve = Callable[[OptimizationProblem], Solution]


class Multiobjective:
    def __init__(self, config: Mapping[str, Any] | None = None, template: Mapping[str, Any] | None = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("multiobjective", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.config = section
        self.tolerance = require_finite(section.get("dominance_tolerance", 1e-9), "dominance_tolerance")
        self.default_weight_samples = int(section.get("weight_samples", 11))
        self.epsilon_samples = int(section.get("epsilon_samples", 9))
        self.lexicographic_tolerance = require_finite(section.get("lexicographic_tolerance", 1e-7), "lexicographic_tolerance")
        self.seed = int(section.get("seed", 0))
        if self.default_weight_samples < 2 or self.epsilon_samples < 2:
            raise OptimizationValidationError("multi-objective sample counts must be >= 2")
        self.pareto = ParetoEfficiency(section)
        self.scalarizations = ScalarizationRegistry(register_defaults=True)

        logger.info(f"Initializing Multiobjective with {self.default_weight_samples}")

    def weighted_scalarization(self, problem: OptimizationProblem, weights: Mapping[str, float]) -> OptimizationProblem:
        """Delegate weighted-sum reformulation to the scalarization registry.

        The registry owns the mathematical transform; this module owns the
        higher-level multi-objective orchestration that decides when and how a
        scalarized subproblem should be solved.
        """
        return self.scalarizations.transform("weighted_sum", problem, weights=weights)

    def epsilon_constraint(
        self,
        problem: OptimizationProblem,
        *,
        primary: str,
        epsilons: Mapping[str, float],
    ) -> OptimizationProblem:
        """Delegate epsilon-constraint reformulation to the scalarization registry."""
        return self.scalarizations.transform(
            "epsilon_constraint",
            problem,
            primary=primary,
            epsilons=epsilons,
            tolerance=self.tolerance,
        )

    def lexicographic(
        self,
        problem: OptimizationProblem,
        solve_single: SingleSolve,
        *,
        order: Sequence[str] | None = None,
        tolerances: Mapping[str, float] | None = None,
    ) -> Solution:
        if not problem.is_multi_objective:
            raise MultiObjectiveError("lexicographic optimization requires multiple objectives")
        objective_map = {objective.name: objective for objective in problem.objectives}
        if order is None:
            ordered = sorted(problem.objectives, key=lambda o: (o.priority, problem.objectives.index(o)))
        else:
            if set(order) != set(objective_map) or len(order) != len(objective_map):
                raise OptimizationValidationError("lexicographic order must contain every objective exactly once")
            ordered = [objective_map[name] for name in order]

        constraints = list(problem.constraints)
        last_solution: Solution | None = None
        for index, objective in enumerate(ordered):
            subproblem = OptimizationProblem(
                name=f"{problem.name}:lex:{index}:{objective.name}",
                variables=problem.variables,
                objectives=(objective,),
                constraints=tuple(constraints),
                problem_class=ProblemClass.UNKNOWN,
                known_convex=problem.known_convex,
                known_smooth=problem.known_smooth,
                derivatives_available=problem.derivatives_available,
                deterministic=problem.deterministic,
                expected_evaluation_cost=problem.expected_evaluation_cost,
                metadata={**dict(problem.metadata), "multiobjective_transform": "lexicographic", "rank": index},
            )
            last_solution = solve_single(with_classification(subproblem))
            if not last_solution.success or last_solution.primal_solution is None:
                return last_solution
            optimum = evaluate_objective(objective, last_solution.primal_solution)
            tolerance = require_finite(
                (tolerances or {}).get(objective.name, self.lexicographic_tolerance),
                f"lexicographic_tolerance[{objective.name}]",
            )
            if tolerance < 0:
                raise OptimizationValidationError("lexicographic tolerance must be non-negative")
            if objective.sense is ObjectiveSense.MINIMIZE:
                bound = optimum + tolerance
                sense = ConstraintSense.LE
            else:
                bound = optimum - tolerance
                sense = ConstraintSense.GE
            # Preserve algebraic structure whenever the optimized objective is
            # linear.  Converting a linear lexicographic stage to a callback
            # would unnecessarily reclassify the next stage as NLP and lose
            # access to LP/MILP certificates and bounds.
            if objective.function is None and not objective.quadratic:
                constraints.append(
                    Constraint(
                        name=f"lex_fix:{objective.name}",
                        sense=sense,
                        rhs=bound - objective.constant,
                        linear=objective.linear,
                        tolerance=tolerance,
                        metadata={
                            "source_objective": objective.name,
                            "lexicographic_optimum": optimum,
                            "tolerance": tolerance,
                        },
                    )
                )
            else:
                constraints.append(
                    Constraint(
                        name=f"lex_fix:{objective.name}",
                        sense=sense,
                        rhs=bound,
                        function=lambda point, obj=objective: evaluate_objective(obj, point),
                        tolerance=tolerance,
                        metadata={
                            "source_objective": objective.name,
                            "lexicographic_optimum": optimum,
                            "tolerance": tolerance,
                        },
                    )
                )
        if last_solution is None:
            raise MultiObjectiveError("lexicographic optimization produced no solver stage")
        return last_solution

    def _default_weight_vectors(self, problem: OptimizationProblem, samples: int) -> tuple[dict[str, float], ...]:
        names = [objective.name for objective in problem.objectives]
        if len(names) == 2:
            grid = np.linspace(0.0, 1.0, samples)
            return tuple({names[0]: float(w), names[1]: float(1.0 - w)} for w in grid)
        rng = np.random.default_rng(self.seed)
        vectors: list[dict[str, float]] = []
        # Anchor each extreme objective explicitly.
        for i, name in enumerate(names):
            vectors.append({other: 1.0 if other == name else 0.0 for other in names})
        while len(vectors) < samples:
            raw = rng.dirichlet(np.ones(len(names)))
            vectors.append({name: float(value) for name, value in zip(names, raw)})
        return tuple(vectors[:samples])

    def solve_weighted_front(
        self,
        problem: OptimizationProblem,
        solve_single: SingleSolve,
        *,
        weight_vectors: Sequence[Mapping[str, float]] | None = None,
        samples: int | None = None,
    ) -> ParetoFront:
        if not problem.is_multi_objective:
            raise MultiObjectiveError("weighted front construction requires multiple objectives")
        count = self.default_weight_samples if samples is None else int(samples)
        if count < 2:
            raise OptimizationValidationError("weighted front requires at least two samples")
        weights = tuple(weight_vectors) if weight_vectors is not None else self._default_weight_vectors(problem, count)
        candidates: list[Candidate] = []
        for weight in weights:
            scalarized = self.weighted_scalarization(problem, weight)
            solution = solve_single(scalarized)
            if not solution.success or solution.primal_solution is None:
                continue
            values = objective_values(problem, solution.primal_solution)
            candidates.append(Candidate(solution.primal_solution, values, True, solution.max_constraint_violation or 0.0, {"weights": dict(weight), "solver": solution.solver_name}))
        if not candidates:
            raise MultiObjectiveError("weighted scalarization produced no feasible solutions")
        result = self.pareto.analyze(candidates, problem.objectives, approximate=True)
        return result.to_front(algorithm="weighted_scalarization")

    def solve_epsilon_front(
        self,
        problem: OptimizationProblem,
        solve_single: SingleSolve,
        *,
        primary: str,
        epsilon_grid: Sequence[Mapping[str, float]],
    ) -> ParetoFront:
        if not epsilon_grid:
            raise OptimizationValidationError("epsilon_grid must not be empty")
        candidates: list[Candidate] = []
        for epsilons in epsilon_grid:
            subproblem = self.epsilon_constraint(problem, primary=primary, epsilons=epsilons)
            solution = solve_single(subproblem)
            if not solution.success or solution.primal_solution is None:
                continue
            candidates.append(
                Candidate(
                    solution.primal_solution,
                    objective_values(problem, solution.primal_solution),
                    True,
                    solution.max_constraint_violation or 0.0,
                    {"epsilons": dict(epsilons), "solver": solution.solver_name},
                )
            )
        if not candidates:
            raise MultiObjectiveError("epsilon-constraint sweep produced no feasible solutions")
        return self.pareto.analyze(candidates, problem.objectives, approximate=True).to_front(algorithm="epsilon_constraint")

    def filter_points(self, problem: OptimizationProblem, points: Sequence[ParetoPoint]) -> ParetoFront:
        result = self.pareto.analyze(points, problem.objectives, approximate=False)
        return result.to_front(algorithm="non_dominated_filter")


__all__ = ["Multiobjective", "SingleSolve"]
