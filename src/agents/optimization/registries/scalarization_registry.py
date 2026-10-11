"""Registry of mathematically defined multi-objective scalarization transforms.

The registry owns *reformulation strategies*, not solver algorithms.  It turns
one multi-objective ``OptimizationProblem`` into a solver-independent
single-objective problem that can be passed back through normal solver
selection.  Search, ranking, and Pareto archive management remain outside this
module.

Academic basis
--------------
Miettinen (1998), *Nonlinear Multiobjective Optimization*, provides the primary
foundation for weighted scalarization and epsilon-constraint methods.  Weighted
sums are intentionally marked as conditionally complete: they can miss
non-supported portions of a non-convex Pareto front.  Epsilon-constraint is
kept as a separate strategy because it can expose efficient points that a
weighted sum cannot recover.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..optimization_math import evaluate_objective, with_classification
from ..optimization_types import Constraint, ConstraintSense, Objective, ObjectiveSense, OptimizationProblem, ProblemClass
from ..utils.optimization_errors import MultiObjectiveError, RegistryError
from ..utils.optimization_helpers import require_finite
from .base_registry import RegistryStore


ScalarizationTransform = Callable[..., OptimizationProblem]


@dataclass(frozen=True, slots=True)
class ScalarizationStrategy:
    """Named scalarization/reformulation with explicit mathematical semantics."""

    name: str
    transform: ScalarizationTransform
    description: str
    academic_basis: str
    pareto_coverage: str
    preserves_algebraic_structure: bool
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise RegistryError("scalarization strategy name must be non-empty")
        if not callable(self.transform):
            raise RegistryError("scalarization transform must be callable", context={"strategy": self.name})
        if not isinstance(self.description, str) or not self.description.strip():
            raise RegistryError("scalarization strategy description must be non-empty", context={"strategy": self.name})
        if not isinstance(self.academic_basis, str) or not self.academic_basis.strip():
            raise RegistryError("scalarization academic_basis must be non-empty", context={"strategy": self.name})
        if not isinstance(self.pareto_coverage, str) or not self.pareto_coverage.strip():
            raise RegistryError("scalarization pareto_coverage must be non-empty", context={"strategy": self.name})
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _require_multiobjective(problem: OptimizationProblem) -> None:
    if not isinstance(problem, OptimizationProblem):
        raise MultiObjectiveError(
            "scalarization requires OptimizationProblem",
            context={"type": type(problem).__name__},
        )
    if not problem.is_multi_objective:
        raise MultiObjectiveError(
            "scalarization requires multiple objectives",
            context={"problem": problem.name, "objective_count": len(problem.objectives)},
        )


def _normalize_weights(problem: OptimizationProblem, weights: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(weights, Mapping):
        raise MultiObjectiveError("weights must be a mapping of objective name to non-negative weight")
    known = {objective.name for objective in problem.objectives}
    unknown = set(weights) - known
    if unknown:
        raise MultiObjectiveError("weights contain unknown objectives", context={"unknown": sorted(unknown)})

    normalized: dict[str, float] = {}
    total = 0.0
    for objective in problem.objectives:
        value = require_finite(weights.get(objective.name, objective.weight), f"weight[{objective.name}]")
        if value < 0:
            raise MultiObjectiveError(
                "multi-objective weights must be non-negative",
                context={"objective": objective.name, "weight": value},
            )
        normalized[objective.name] = value
        total += value
    if total <= 0:
        raise MultiObjectiveError("at least one multi-objective weight must be positive")
    return {name: value / total for name, value in normalized.items()}


def weighted_sum_transform(
    problem: OptimizationProblem,
    *,
    weights: Mapping[str, float],
) -> OptimizationProblem:
    """Create a weighted-sum scalarization in minimization space.

    Maximize objectives are sign-inverted before aggregation.  Explicit
    algebraic objectives remain algebraic so LP/QP structure can survive the
    reformulation and still reach certificate-producing solvers.
    """
    _require_multiobjective(problem)
    normalized = _normalize_weights(problem, weights)

    algebraic = all(objective.function is None for objective in problem.objectives)
    if algebraic:
        linear: dict[str, float] = {}
        quadratic: dict[tuple[str, str], float] = {}
        constant = 0.0
        for objective in problem.objectives:
            direction = 1.0 if objective.sense is ObjectiveSense.MINIMIZE else -1.0
            scale = normalized[objective.name] * direction
            constant += scale * objective.constant
            for name, coefficient in objective.linear.items():
                linear[name] = linear.get(name, 0.0) + scale * coefficient
            for key, coefficient in objective.quadratic.items():
                quadratic[key] = quadratic.get(key, 0.0) + scale * coefficient

        # Drop exact algebraic cancellations.  A constant-only aggregate is a
        # legitimate scalar objective, but the canonical Objective contract
        # intentionally requires an executable representation, so represent
        # that rare case as a deterministic callable.
        linear = {name: value for name, value in linear.items() if value != 0.0}
        quadratic = {key: value for key, value in quadratic.items() if value != 0.0}
        common_metadata = {
            "weights": normalized,
            "source_objectives": tuple(objective.name for objective in problem.objectives),
            "scalarization": "weighted_sum",
        }
        if linear or quadratic:
            scalar = Objective(
                name="weighted_scalarization",
                sense=ObjectiveSense.MINIMIZE,
                linear=linear,
                quadratic=quadratic,
                constant=constant,
                metadata=common_metadata,
            )
        else:
            scalar = Objective(
                name="weighted_scalarization",
                sense=ObjectiveSense.MINIMIZE,
                function=lambda _point, value=constant: value,
                metadata={**common_metadata, "constant_only": True},
            )
    else:
        def scalar_function(point: Mapping[str, Any]) -> float:
            total = 0.0
            for objective in problem.objectives:
                value = evaluate_objective(objective, point)
                direction = 1.0 if objective.sense is ObjectiveSense.MINIMIZE else -1.0
                total += normalized[objective.name] * direction * value
            return total

        scalar = Objective(
            name="weighted_scalarization",
            sense=ObjectiveSense.MINIMIZE,
            function=scalar_function,
            metadata={
                "weights": normalized,
                "source_objectives": tuple(objective.name for objective in problem.objectives),
                "scalarization": "weighted_sum",
            },
        )

    transformed = OptimizationProblem(
        name=f"{problem.name}:weighted",
        variables=problem.variables,
        objectives=(scalar,),
        constraints=problem.constraints,
        problem_class=ProblemClass.UNKNOWN,
        known_convex=problem.known_convex,
        known_smooth=problem.known_smooth,
        derivatives_available=problem.derivatives_available,
        deterministic=problem.deterministic,
        expected_evaluation_cost=problem.expected_evaluation_cost,
        metadata={
            **dict(problem.metadata),
            "multiobjective_transform": "weighted_sum",
            "weights": normalized,
            "source_problem": problem.name,
        },
    )
    return with_classification(transformed)


def epsilon_constraint_transform(
    problem: OptimizationProblem,
    *,
    primary: str,
    epsilons: Mapping[str, float],
    tolerance: float = 1e-9,
) -> OptimizationProblem:
    """Create an epsilon-constraint reformulation.

    The primary objective remains the objective.  Every supplied secondary
    objective becomes a mathematical constraint in its original direction.
    Linear objectives are converted to linear constraints without introducing
    callbacks, preserving LP/MILP structure where possible.
    """
    _require_multiobjective(problem)
    if not isinstance(primary, str) or not primary:
        raise MultiObjectiveError("primary objective must be a non-empty string")
    if not isinstance(epsilons, Mapping):
        raise MultiObjectiveError("epsilons must be a mapping")
    epsilon_tolerance = require_finite(tolerance, "epsilon_constraint_tolerance")
    if epsilon_tolerance < 0:
        raise MultiObjectiveError("epsilon-constraint tolerance must be non-negative")

    objective_map = {objective.name: objective for objective in problem.objectives}
    if primary not in objective_map:
        raise MultiObjectiveError("unknown primary objective", context={"primary": primary})
    unknown = set(epsilons) - set(objective_map)
    if unknown:
        raise MultiObjectiveError("epsilon supplied for unknown objective", context={"unknown": sorted(unknown)})

    constraints = list(problem.constraints)
    applied: dict[str, float] = {}
    for name, epsilon_raw in epsilons.items():
        if name == primary:
            continue
        epsilon = require_finite(epsilon_raw, f"epsilon[{name}]")
        applied[name] = epsilon
        objective = objective_map[name]
        sense = ConstraintSense.LE if objective.sense is ObjectiveSense.MINIMIZE else ConstraintSense.GE
        if objective.function is None and not objective.quadratic:
            constraints.append(
                Constraint(
                    name=f"epsilon:{name}",
                    sense=sense,
                    rhs=epsilon - objective.constant,
                    linear=objective.linear,
                    tolerance=epsilon_tolerance,
                    metadata={"source_objective": name, "epsilon": epsilon, "scalarization": "epsilon_constraint"},
                )
            )
        else:
            constraints.append(
                Constraint(
                    name=f"epsilon:{name}",
                    sense=sense,
                    rhs=epsilon,
                    function=lambda point, obj=objective: evaluate_objective(obj, point),
                    tolerance=epsilon_tolerance,
                    metadata={"source_objective": name, "epsilon": epsilon, "scalarization": "epsilon_constraint"},
                )
            )

    if not applied:
        raise MultiObjectiveError(
            "epsilon-constraint requires at least one secondary objective epsilon",
            context={"primary": primary},
        )

    transformed = OptimizationProblem(
        name=f"{problem.name}:epsilon:{primary}",
        variables=problem.variables,
        objectives=(objective_map[primary],),
        constraints=tuple(constraints),
        problem_class=ProblemClass.UNKNOWN,
        known_convex=problem.known_convex,
        known_smooth=problem.known_smooth,
        derivatives_available=problem.derivatives_available,
        deterministic=problem.deterministic,
        expected_evaluation_cost=problem.expected_evaluation_cost,
        metadata={
            **dict(problem.metadata),
            "multiobjective_transform": "epsilon_constraint",
            "primary": primary,
            "epsilons": applied,
            "source_problem": problem.name,
        },
    )
    return with_classification(transformed)


class ScalarizationRegistry:
    """Thread-safe registry of multi-objective problem reformulations."""

    def __init__(self, *, register_defaults: bool = True) -> None:
        self._store: RegistryStore[ScalarizationStrategy] = RegistryStore()
        if register_defaults:
            self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            ScalarizationStrategy(
                name="weighted_sum",
                transform=weighted_sum_transform,
                description="Normalized weighted aggregation of objectives in minimization space.",
                academic_basis="Miettinen (1998), Nonlinear Multiobjective Optimization.",
                pareto_coverage=(
                    "Finds supported efficient points under standard regularity assumptions; "
                    "may miss non-convex portions of the Pareto front."
                ),
                preserves_algebraic_structure=True,
                metadata={"family": "scalarization", "requires": ("weights",)},
            ),
            replace=True,
        )
        self.register(
            ScalarizationStrategy(
                name="epsilon_constraint",
                transform=epsilon_constraint_transform,
                description="Optimizes one primary objective while bounding secondary objectives.",
                academic_basis="Miettinen (1998), Nonlinear Multiobjective Optimization.",
                pareto_coverage=(
                    "Can expose non-supported/non-convex efficient regions; practical coverage depends "
                    "on the chosen epsilon sweep."
                ),
                preserves_algebraic_structure=True,
                metadata={"family": "constraint_reformulation", "requires": ("primary", "epsilons")},
            ),
            replace=True,
        )

    def register(self, strategy: ScalarizationStrategy, *, replace: bool = False) -> ScalarizationStrategy:
        if not isinstance(strategy, ScalarizationStrategy):
            raise RegistryError(
                "scalarization registry accepts ScalarizationStrategy instances only",
                context={"type": type(strategy).__name__},
            )
        return self._store.register(strategy.name, strategy, replace=replace)

    def unregister(self, name: str) -> bool:
        return self._store.unregister(name)

    def get(self, name: str) -> ScalarizationStrategy:
        try:
            return self._store.get(name)
        except RegistryError as exc:
            raise MultiObjectiveError("unknown scalarization strategy", context={"strategy": name}) from exc

    def names(self, *, include_disabled: bool = False) -> tuple[str, ...]:
        if include_disabled:
            return self._store.names()
        return tuple(strategy.name for strategy in self.strategies() if strategy.enabled)

    def strategies(self) -> tuple[ScalarizationStrategy, ...]:
        return self._store.values()

    def set_enabled(self, name: str, enabled: bool) -> None:
        try:
            self._store.update(
                name,
                lambda strategy: ScalarizationStrategy(
                    name=strategy.name,
                    transform=strategy.transform,
                    description=strategy.description,
                    academic_basis=strategy.academic_basis,
                    pareto_coverage=strategy.pareto_coverage,
                    preserves_algebraic_structure=strategy.preserves_algebraic_structure,
                    enabled=enabled,
                    metadata=strategy.metadata,
                ),
            )
        except RegistryError as exc:
            raise MultiObjectiveError("unknown scalarization strategy", context={"strategy": name}) from exc

    def transform(self, name: str, problem: OptimizationProblem, **parameters: Any) -> OptimizationProblem:
        strategy = self.get(name)
        if not strategy.enabled:
            raise MultiObjectiveError("scalarization strategy is disabled", context={"strategy": name})
        try:
            transformed = strategy.transform(problem, **parameters)
        except MultiObjectiveError:
            raise
        except (TypeError, ValueError) as exc:
            raise MultiObjectiveError(
                "invalid scalarization parameters",
                context={"strategy": name, "parameters": tuple(sorted(parameters))},
                cause=exc,
            ) from exc
        if not isinstance(transformed, OptimizationProblem):
            raise MultiObjectiveError(
                "scalarization strategy returned invalid result type",
                context={"strategy": name, "type": type(transformed).__name__},
            )
        if transformed.is_multi_objective:
            raise MultiObjectiveError(
                "scalarization strategy did not reduce the problem to one objective",
                context={"strategy": name, "objective_count": len(transformed.objectives)},
            )
        return transformed

    def describe(self, name: str) -> dict[str, Any]:
        strategy = self.get(name)
        return {
            "name": strategy.name,
            "enabled": strategy.enabled,
            "description": strategy.description,
            "academic_basis": strategy.academic_basis,
            "pareto_coverage": strategy.pareto_coverage,
            "preserves_algebraic_structure": strategy.preserves_algebraic_structure,
            "metadata": dict(strategy.metadata),
        }


__all__ = [
    "ScalarizationRegistry",
    "ScalarizationStrategy",
    "ScalarizationTransform",
    "epsilon_constraint_transform",
    "weighted_sum_transform",
]
