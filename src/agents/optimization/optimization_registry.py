"""Optimization registry façade.

The façade exposes three non-overlapping registries:

- ``CapabilityRegistry``: hard mathematical compatibility and coverage;
- ``SolverRegistry``: availability-, preference-, and evidence-aware solver selection;
- ``ScalarizationRegistry``: named multi-objective reformulations.

It remains intentionally separate from the future ``OptimizationAgent`` façade.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Mapping
from typing import Any

from .optimization_memory import OptimizationMemory
from .optimization_types import OptimizationProblem, ProblemClass, SolverSelection
from .registries.capability_registry import CapabilityAudit, CapabilityRegistry
from .registries.scalarization_registry import ScalarizationRegistry
from .registries.solver_registry import SolverRegistry
from .solvers.base_solver import BaseSolver
from .utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Registries")
printer = PrettyPrinter()


class OptimizationRegistries:
    """Composition façade for the Optimization subsystem's registry layer."""

    def __init__(self, config: Mapping[str, Any] | None = None, memory: OptimizationMemory | None = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("optimization_registries", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.registry_config = section
        self.memory = memory

        # One shared capability registry ensures dynamic solver
        # registration/enablement is visible consistently across the façade.
        self.capabilities = CapabilityRegistry()
        self.solvers = SolverRegistry(
            section,
            memory=memory,
            capability_registry=self.capabilities,
            register_defaults=True,
        )
        self.scalarizations = ScalarizationRegistry(register_defaults=True)

    def register_solver(self, solver: BaseSolver, *, priority: int = 100, enabled: bool = True, replace: bool = False) -> None:
        self.solvers.register(solver, priority=priority, enabled=enabled, replace=replace)
        logger.info(
            "Optimization solver registered | solver=%s | priority=%d | enabled=%s",
            solver.solver_name,
            priority,
            enabled,
        )

    def unregister_solver(self, solver_name: str) -> bool:
        removed = self.solvers.unregister(solver_name)
        if removed:
            logger.info("Optimization solver unregistered | solver=%s", solver_name)
        return removed

    def select(self, problem: OptimizationProblem, *, requested: str | None = None) -> BaseSolver:
        return self.solvers.select(problem, requested=requested)

    def selection(self, problem: OptimizationProblem, *, requested: str | None = None) -> SolverSelection:
        return self.solvers.selection(problem, requested=requested)

    def capability_audit(self, required_classes: tuple[ProblemClass, ...] | None = None) -> CapabilityAudit:
        return self.capabilities.audit(required_classes)

    def scalarize(self, strategy: str, problem: OptimizationProblem, **parameters: Any) -> OptimizationProblem:
        return self.scalarizations.transform(strategy, problem, **parameters)

    def summary(self, problem: OptimizationProblem) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "solver": attempt.solver_name,
                "available": attempt.available,
                "compatible": attempt.compatible,
                "preference_score": attempt.preference_score,
                "incompatibilities": attempt.incompatibilities,
            }
            for attempt in self.solvers.inspect(problem)
        )

    def registry_summary(self) -> dict[str, Any]:
        """Return an introspection snapshot without executing a solve."""
        audit = self.capabilities.audit()
        return {
            "solvers": self.solvers.names(),
            "capabilities": self.capabilities.support_matrix(),
            "capability_coverage_complete": audit.complete,
            "uncovered_problem_classes": tuple(problem.value for problem in audit.uncovered),
            "scalarizations": self.scalarizations.names(),
        }


__all__ = ["OptimizationRegistries"]
