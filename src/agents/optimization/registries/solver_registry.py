"""Capability-, availability-, and evidence-aware solver registry.

Rice (1976) and Kerschke et al. (2019) motivate the separation between problem
features, compatible algorithms, and algorithm selection.  Historical evidence
(Dolan & Moré, 2002) influences ranking only after hard compatibility and
availability checks have been satisfied.

Hard mathematical compatibility is delegated to ``CapabilityRegistry``.  This
keeps solver selection from becoming the authority for both *what a solver can
solve* and *which compatible solver should be preferred*.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..optimization_math import problem_features
from ..optimization_memory import OptimizationMemory
from ..optimization_types import OptimizationProblem, SolverAttempt, SolverSelection
from ..solvers import BlackBoxSolver, CombinatorialSolver, LinearSolver, NonLinearSolver, QuadraticSolver
from ..solvers.base_solver import BaseSolver
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import IncompatibleSolverError, RegistryError, SolverUnavailableError
from .base_registry import BaseRegistry, RegistryEntry
from .capability_registry import CapabilityRegistry


@dataclass(frozen=True, slots=True)
class RankedSolver:
    solver: BaseSolver
    score: float
    priority: int
    evidence_score: float
    reasons: tuple[str, ...] = ()


class SolverRegistry(BaseRegistry):
    """Rank compatible solver adapters after availability and capability gates."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        memory: OptimizationMemory | None = None,
        capability_registry: CapabilityRegistry | None = None,
        register_defaults: bool = True,
    ) -> None:
        super().__init__()
        global_config = load_global_config()
        section = dict(get_config_section("optimization_registries", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.registry_config = section
        self.memory = memory
        self.capabilities = capability_registry or CapabilityRegistry()
        self.priority = {str(k): int(v) for k, v in dict(section.get("priority", {})).items()}
        self.priority_weight = float(section.get("priority_weight", 1.0))
        self.evidence_weight = float(section.get("evidence_weight", 25.0))
        self.determinism_bonus = float(section.get("determinism_bonus", 2.0))
        self.warm_start_bonus = float(section.get("warm_start_bonus", 1.0))
        if register_defaults:
            self._register_defaults()

    def register(
        self,
        solver: BaseSolver,
        *,
        priority: int = 100,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> RegistryEntry:
        capability = solver.capability
        if capability.name != solver.solver_name:
            raise RegistryError(
                "solver capability name must match solver_name",
                context={"solver": solver.solver_name, "capability_name": capability.name},
            )
        entry = super().register(
            solver,
            priority=priority,
            enabled=enabled,
            metadata=metadata,
            replace=replace,
        )
        # SolverCapability is validated when constructed by the solver.  Keep
        # the capability catalogue synchronized with the concrete adapter.
        self.capabilities.register(
            solver.solver_name,
            capability,
            enabled=entry.enabled,
            metadata={
                "priority": entry.priority,
                "backend_package": solver.backend_package,
                **dict(entry.metadata or {}),
            },
            replace=True,
        )
        return entry

    def unregister(self, solver_name: str) -> bool:
        removed = super().unregister(solver_name)
        if removed:
            self.capabilities.unregister(solver_name)
        return removed

    def set_enabled(self, solver_name: str, enabled: bool) -> None:
        super().set_enabled(solver_name, enabled)
        self.capabilities.set_enabled(solver_name, enabled)

    def _register_defaults(self) -> None:
        for solver in (
            LinearSolver(),
            QuadraticSolver(),
            NonLinearSolver(),
            CombinatorialSolver(),
            BlackBoxSolver(),
        ):
            self.register(solver, priority=self.priority.get(solver.solver_name, 100), replace=True)

    def inspect(self, problem: OptimizationProblem) -> tuple[SolverAttempt, ...]:
        attempts: list[SolverAttempt] = []
        for entry in self.entries():
            solver = entry.solver
            available = entry.enabled and solver.is_available()
            assessment = self.capabilities.assess(solver.solver_name, problem)
            attempts.append(
                SolverAttempt(
                    solver_name=solver.solver_name,
                    available=available,
                    compatible=assessment.compatible,
                    preference_score=self._score(entry, problem) if available and assessment.compatible else None,
                    incompatibilities=assessment.incompatibilities,
                )
            )
        return tuple(attempts)

    def _score(self, entry: RegistryEntry, problem: OptimizationProblem) -> float:
        # Lower configured priority is preferred. Convert it to a larger score.
        features = problem_features(problem)
        priority_component = -self.priority_weight * float(entry.priority)
        evidence = 0.0
        if self.memory is not None:
            evidence = self.memory.historical_score(
                entry.solver.solver_name,
                problem_class=features.problem_class.value,
            )
        score = priority_component + self.evidence_weight * evidence
        if entry.solver.capability.deterministic and problem.deterministic:
            score += self.determinism_bonus
        warm_start_available = any(variable.initial is not None for variable in problem.variables)
        if not warm_start_available and self.memory is not None:
            warm_start_available = self.memory.warm_start(problem) is not None
        if entry.solver.capability.supports_warm_start and warm_start_available:
            score += self.warm_start_bonus
        return score

    def ranked(self, problem: OptimizationProblem) -> tuple[RankedSolver, ...]:
        features = problem_features(problem)
        ranked: list[RankedSolver] = []
        for entry in self.entries():
            if not entry.enabled or not entry.solver.is_available():
                continue
            assessment = self.capabilities.assess(entry.solver.solver_name, features)
            if not assessment.compatible:
                continue
            evidence = (
                0.0
                if self.memory is None
                else self.memory.historical_score(
                    entry.solver.solver_name,
                    problem_class=features.problem_class.value,
                )
            )
            ranked.append(
                RankedSolver(
                    entry.solver,
                    self._score(entry, problem),
                    entry.priority,
                    evidence,
                    assessment.incompatibilities,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.priority, item.solver.solver_name))
        return tuple(ranked)

    def select(self, problem: OptimizationProblem, *, requested: str | None = None) -> BaseSolver:
        if requested is not None:
            entry = self.get_entry(requested)
            if not entry.enabled or not entry.solver.is_available():
                raise SolverUnavailableError("requested solver is unavailable", context={"solver": requested})
            assessment = self.capabilities.assess(requested, problem)
            if not assessment.compatible:
                raise IncompatibleSolverError(
                    "requested solver is incompatible",
                    context={"solver": requested, "reasons": assessment.incompatibilities},
                )
            return entry.solver

        ranked = self.ranked(problem)
        if ranked:
            return ranked[0].solver

        inspection = self.inspect(problem)
        compatible_but_unavailable = [item.solver_name for item in inspection if item.compatible and not item.available]
        if compatible_but_unavailable:
            raise SolverUnavailableError(
                "all compatible solvers are unavailable",
                context={"solvers": compatible_but_unavailable, "problem_class": problem.problem_class.value},
            )
        raise IncompatibleSolverError(
            "no registered solver is compatible with this problem",
            context={
                "problem_class": problem.problem_class.value,
                "inspection": [
                    {
                        "solver": item.solver_name,
                        "available": item.available,
                        "compatible": item.compatible,
                        "reasons": item.incompatibilities,
                    }
                    for item in inspection
                ],
            },
        )

    def selection(self, problem: OptimizationProblem, *, requested: str | None = None) -> SolverSelection:
        attempts = self.inspect(problem)
        selected = self.select(problem, requested=requested)
        return SolverSelection(selected_solver=selected.solver_name, attempts=attempts, requested_solver=requested)


__all__ = ["RankedSolver", "SolverRegistry"]
