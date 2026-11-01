"""Common production solver-adapter contract for SLAI optimization."""
from __future__ import annotations

__version__ = "2.3.0"

import importlib.metadata
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from ..optimization_math import is_feasible, max_constraint_violation, objective_values, problem_features
from ..optimization_types import (
    FeasibilityStatus,
    OptimizationProblem,
    OptimalityStatus,
    Solution,
    SolverCapability,
    SolverStatus,
    TerminationReason,
)
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import (
    IncompatibleSolverError,
    InvalidSolverResultError,
    SolverUnavailableError,
)
from ..utils.optimization_helpers import absolute_relative_gap, require_finite
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Base Optimization Solver")
printer = PrettyPrinter()


class BaseSolver(ABC):
    """Stable contract shared by all concrete solver adapters.

    Public ``solve`` implementations should call ``ensure_usable`` before
    entering their backend and construct results via ``_solution``.  The base
    class then guarantees common availability, compatibility and result
    validation semantics across heterogeneous backends.
    """

    solver_name = "base"
    backend_package: str | None = None

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("base_solver", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.base_config = section
        self.feasibility_tolerance = require_finite(section.get("feasibility_tolerance", 1e-8), "feasibility_tolerance")
        self.optimality_tolerance = require_finite(section.get("optimality_tolerance", 1e-8), "optimality_tolerance")
        if self.feasibility_tolerance < 0 or self.optimality_tolerance < 0:
            raise ValueError("optimization tolerances must be non-negative")
        raw_time_limit = section.get("time_limit_seconds")
        self.time_limit_seconds = None if raw_time_limit is None else require_finite(raw_time_limit, "time_limit_seconds")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive when configured")
        self.validate_results = bool(section.get("validate_results", True))
        self.strict_success_feasibility = bool(section.get("strict_success_feasibility", True))

    @property
    @abstractmethod
    def capability(self) -> SolverCapability:
        raise IncompatibleSolverError("BaseSolver has no concrete capability contract")

    def is_available(self) -> bool:
        if self.backend_package is None:
            return True
        try:
            importlib.metadata.version(self.backend_package)
        except importlib.metadata.PackageNotFoundError:
            return False
        return True

    def solver_version(self) -> str | None:
        if self.backend_package is None:
            return __version__
        try:
            return importlib.metadata.version(self.backend_package)
        except importlib.metadata.PackageNotFoundError:
            return None

    def compatibility_reasons(self, problem: OptimizationProblem) -> tuple[str, ...]:
        return self.capability.incompatibilities(problem_features(problem))

    def compatible(self, problem: OptimizationProblem) -> bool:
        return not self.compatibility_reasons(problem)

    def ensure_usable(self, problem: OptimizationProblem) -> None:
        if not self.is_available():
            raise SolverUnavailableError(
                "solver backend is not installed",
                context={"solver": self.solver_name, "backend": self.backend_package},
            )
        reasons = self.compatibility_reasons(problem)
        if reasons:
            raise IncompatibleSolverError(
                "solver is incompatible with mathematical problem features",
                context={
                    "solver": self.solver_name,
                    "problem_class": problem.problem_class.value,
                    "reasons": reasons,
                },
            )

    @abstractmethod
    def solve(self, problem: OptimizationProblem, **options: Any) -> Solution:
        raise IncompatibleSolverError("BaseSolver cannot execute a problem directly")

    def effective_time_limit(self, options: Mapping[str, Any]) -> float | None:
        raw = options.get("time_limit_seconds", self.time_limit_seconds)
        if raw is None:
            return None
        value = require_finite(raw, "time_limit_seconds")
        if value <= 0:
            raise ValueError("time_limit_seconds must be positive")
        return value

    def _solution(
        self,
        problem: OptimizationProblem,
        *,
        status: SolverStatus,
        termination: TerminationReason,
        success: bool,
        point: Mapping[str, Any] | None = None,
        optimality: OptimalityStatus = OptimalityStatus.UNKNOWN,
        runtime_seconds: float | None = None,
        primal_bound: float | None = None,
        dual_bound: float | None = None,
        absolute_gap: float | None = None,
        relative_gap: float | None = None,
        iterations: int | None = None,
        nodes: int | None = None,
        evaluations: int | None = None,
        warnings: tuple[str, ...] = (),
        diagnostics: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Solution:
        values = objective_values(problem, point) if point is not None else {}
        violation = max_constraint_violation(problem, point) if point is not None else None
        if point is None or violation is None or not isinstance(violation, (int, float)):
            feasibility = FeasibilityStatus.UNKNOWN
        elif is_feasible(problem, point, tolerance=self.feasibility_tolerance):
            feasibility = FeasibilityStatus.FEASIBLE
        else:
            feasibility = FeasibilityStatus.INFEASIBLE

        if (absolute_gap is None or relative_gap is None) and primal_bound is not None and dual_bound is not None:
            computed_abs, computed_rel = absolute_relative_gap(primal_bound, dual_bound)
            absolute_gap = computed_abs if absolute_gap is None else absolute_gap
            relative_gap = computed_rel if relative_gap is None else relative_gap

        result = Solution(
            solver_name=self.solver_name,
            solver_version=self.solver_version(),
            problem_class=problem.problem_class,
            status=status,
            termination_reason=termination,
            success=success,
            primal_solution=dict(point) if point is not None else None,
            objective_values=values,
            feasibility=feasibility,
            optimality=optimality,
            max_constraint_violation=violation,
            primal_bound=primal_bound,
            dual_bound=dual_bound,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap,
            iteration_count=iterations,
            node_count=nodes,
            function_evaluation_count=evaluations,
            runtime_seconds=runtime_seconds,
            warnings=warnings,
            diagnostics=dict(diagnostics or {}),
            metadata=dict(metadata or {}),
        )
        return self.validate_result(problem, result) if self.validate_results else result

    def validate_result(self, problem: OptimizationProblem, result: Solution) -> Solution:
        """Enforce solver-independent invariants after backend normalization."""
        if result.solver_name != self.solver_name:
            raise InvalidSolverResultError(
                "result solver_name does not match adapter",
                context={"expected": self.solver_name, "actual": result.solver_name},
            )
        if result.problem_class != problem.problem_class:
            raise InvalidSolverResultError(
                "result problem_class does not match solved problem",
                context={"expected": problem.problem_class.value, "actual": result.problem_class.value},
            )
        if result.success and result.primal_solution is None:
            raise InvalidSolverResultError("successful result is missing a primal solution")
        if result.success and self.strict_success_feasibility and result.feasibility is not FeasibilityStatus.FEASIBLE:
            raise InvalidSolverResultError(
                "backend reported success for an infeasible/unknown point",
                context={
                    "solver": self.solver_name,
                    "feasibility": result.feasibility.value,
                    "max_violation": result.max_constraint_violation,
                },
            )
        if result.status in {SolverStatus.INFEASIBLE, SolverStatus.UNBOUNDED, SolverStatus.SOLVER_UNAVAILABLE, SolverStatus.UNSUPPORTED} and result.success:
            raise InvalidSolverResultError("terminal failure status cannot be marked successful", context={"status": result.status.value})
        if result.status is SolverStatus.OPTIMAL and result.optimality not in {OptimalityStatus.GLOBAL, OptimalityStatus.LOCAL}:
            raise InvalidSolverResultError("OPTIMAL status requires explicit local/global optimality semantics")
        return result

    @staticmethod
    def timed(started: float) -> float:
        return max(0.0, time.perf_counter() - started)


__all__ = ["BaseSolver"]
