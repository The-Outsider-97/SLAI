"""LP/IP/MILP adapter backed by SciPy HiGHS.

Academic foundations
--------------------
Dantzig (1963) provides the classical LP/simplex foundation; Karmarkar (1984)
provides the polynomial-time interior-point foundation.  Land & Doig (1960),
Gomory (1958), and Achterberg (2009) provide the branch-and-bound/cut and
modern mixed-integer architecture context.

This adapter does not reimplement HiGHS.  It translates SLAI's solver-neutral
model into SciPy/HiGHS, validates compatibility, and normalizes backend status,
bounds, gaps, counts and diagnostics into the common ``Solution`` contract.
"""
from __future__ import annotations

__version__ = "2.3.0"

import time
import numpy as np # type: ignore

from collections.abc import Mapping
from typing import Any
from scipy.optimize import Bounds, LinearConstraint, linprog, milp # type: ignore

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import UnsupportedProblemError
from ..utils.optimization_helpers import absolute_relative_gap, require_finite
from ..optimization_math import *
from ..optimization_types import *
from .base_solver import BaseSolver
from logs.logger import configure_logging, get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Linear Optimization Solver")
printer = PrettyPrinter()


class LinearSolver(BaseSolver):
    solver_name = "scipy_highs_linear"
    backend_package = "scipy"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        global_config = load_global_config()
        section = dict(get_config_section("linear_solver", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.linear_config = section
        self.presolve = bool(section.get("presolve", True))
        self.lp_method = str(section.get("lp_method", "highs"))
        if self.lp_method not in {"highs", "highs-ds", "highs-ipm"}:
            raise ValueError("linear_solver.lp_method must be highs/highs-ds/highs-ipm")
        self.mip_rel_gap = require_finite(section.get("mip_rel_gap", 1e-6), "mip_rel_gap")
        self.node_limit = section.get("node_limit")
        if self.node_limit is not None:
            if isinstance(self.node_limit, bool) or not isinstance(self.node_limit, int) or self.node_limit <= 0:
                raise ValueError("linear_solver.node_limit must be a positive integer")

    @property
    def capability(self) -> SolverCapability:
        return SolverCapability(
            name=self.solver_name,
            problem_classes=frozenset({ProblemClass.LP, ProblemClass.IP, ProblemClass.MILP}),
            supports_continuous=True,
            supports_integer=True,
            supports_binary=True,
            supports_semi_continuous=True,
            supports_semi_integer=True,
            supports_quadratic_objective=False,
            supports_callable_objective=False,
            supports_nonlinear_constraints=False,
            supports_warm_start=False,
            deterministic=True,
            backend="scipy.optimize.linprog/milp (HiGHS)",
        )

    def solve(self, problem: OptimizationProblem, **options: Any):
        self.ensure_usable(problem)
        if len(problem.objectives) != 1:
            raise UnsupportedProblemError("LinearSolver requires one objective")
        if any(variable.shape for variable in problem.variables):
            raise UnsupportedProblemError("LinearSolver supports scalar variables only")
        c, constant, sign = linear_objective_vector(problem)
        discrete = any(
            variable.domain.kind
            in {DomainKind.INTEGER, DomainKind.BINARY, DomainKind.SEMI_CONTINUOUS, DomainKind.SEMI_INTEGER}
            for variable in problem.variables
        )
        return self._solve_milp(problem, c, constant, sign, options) if discrete else self._solve_lp(problem, c, constant, sign, options)

    def _solve_lp(
        self,
        problem: OptimizationProblem,
        c: np.ndarray,
        constant: float,
        sign: float,
        options: Mapping[str, Any],
    ):
        a_ub, b_ub, a_eq, b_eq = linear_constraint_matrices(problem)
        bounds: list[tuple[float | None, float | None]] = []
        for variable in problem.variables:
            domain = variable.domain
            if domain.kind is not DomainKind.REAL:
                raise UnsupportedProblemError("continuous LP backend received non-real variable", context={"variable": variable.name})
            if domain.strict_lower or domain.strict_upper:
                raise UnsupportedProblemError("strict variable bounds are not representable by LP backends", context={"variable": variable.name})
            bounds.append((domain.lower, domain.upper))

        backend_options: dict[str, Any] = {
            "presolve": self.presolve,
            "primal_feasibility_tolerance": self.feasibility_tolerance,
            "dual_feasibility_tolerance": self.optimality_tolerance,
        }
        time_limit = self.effective_time_limit(options)
        if time_limit is not None:
            backend_options["time_limit"] = time_limit
        backend_options.update(dict(options.get("scipy_options", {})))

        started = time.perf_counter()
        result = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=self.lp_method,
            options=backend_options,
        )
        runtime = self.timed(started)
        point = point_from_vector(problem, result.x, project=False) if getattr(result, "x", None) is not None else None

        status_code = int(result.status)
        message = str(result.message)
        if status_code == 0:
            status, termination = SolverStatus.OPTIMAL, TerminationReason.OPTIMAL
            optimality = OptimalityStatus.GLOBAL
        elif status_code == 1:
            status = SolverStatus.TIMEOUT if "time" in message.lower() else SolverStatus.LIMIT_REACHED
            termination = TerminationReason.TIME_LIMIT if "time" in message.lower() else TerminationReason.ITERATION_LIMIT
            optimality = OptimalityStatus.NONE
        elif status_code == 2:
            status, termination, optimality = SolverStatus.INFEASIBLE, TerminationReason.INFEASIBLE, OptimalityStatus.NONE
        elif status_code == 3:
            status, termination, optimality = SolverStatus.UNBOUNDED, TerminationReason.UNBOUNDED, OptimalityStatus.NONE
        else:
            status, termination, optimality = SolverStatus.NUMERICAL_FAILURE, TerminationReason.NUMERICAL_FAILURE, OptimalityStatus.NONE

        feasible_point = point is not None and is_feasible(problem, point, tolerance=self.feasibility_tolerance)
        success = status is SolverStatus.OPTIMAL or (status in {SolverStatus.LIMIT_REACHED, SolverStatus.TIMEOUT} and feasible_point)
        transformed_fun = getattr(result, "fun", None)
        primal = None if transformed_fun is None else sign * float(transformed_fun) + constant
        dual = primal if status is SolverStatus.OPTIMAL else None
        abs_gap, rel_gap = absolute_relative_gap(primal, dual)

        diagnostics = {
            "backend": "HiGHS",
            "method": self.lp_method,
            "scipy_status": status_code,
            "message": message,
            "lower_residual": getattr(getattr(result, "lower", None), "residual", None),
            "upper_residual": getattr(getattr(result, "upper", None), "residual", None),
            "inequality_residual": getattr(getattr(result, "ineqlin", None), "residual", None),
            "equality_residual": getattr(getattr(result, "eqlin", None), "residual", None),
        }
        return self._solution(
            problem,
            status=status,
            termination=termination,
            success=success,
            point=point,
            optimality=optimality,
            runtime_seconds=runtime,
            primal_bound=primal,
            dual_bound=dual,
            absolute_gap=abs_gap,
            relative_gap=rel_gap,
            iterations=int(result.nit) if getattr(result, "nit", None) is not None else None,
            diagnostics=diagnostics,
            metadata={"backend_success": bool(getattr(result, "success", False))},
        )

    def _solve_milp(
        self,
        problem: OptimizationProblem,
        c: np.ndarray,
        constant: float,
        sign: float,
        options: Mapping[str, Any],
    ):
        n = len(problem.variables)
        lower = np.empty(n, dtype=float)
        upper = np.empty(n, dtype=float)
        integrality = np.zeros(n, dtype=np.uint8)

        for index, variable in enumerate(problem.variables):
            domain = variable.domain
            if domain.strict_lower or domain.strict_upper:
                raise UnsupportedProblemError("strict variable bounds are not representable by MILP backends", context={"variable": variable.name})
            lower[index] = -np.inf if domain.lower is None else domain.lower
            upper[index] = np.inf if domain.upper is None else domain.upper
            if domain.kind is DomainKind.REAL:
                integrality[index] = 0
            elif domain.kind in {DomainKind.INTEGER, DomainKind.BINARY}:
                integrality[index] = 1
            elif domain.kind is DomainKind.SEMI_CONTINUOUS:
                integrality[index] = 2
            elif domain.kind is DomainKind.SEMI_INTEGER:
                integrality[index] = 3
            else:
                raise UnsupportedProblemError("HiGHS MILP does not support this domain", context={"variable": variable.name, "domain": domain.kind.value})

        rows: list[list[float]] = []
        lb: list[float] = []
        ub: list[float] = []
        name_to_index = {variable.name: i for i, variable in enumerate(problem.variables)}
        for constraint in problem.constraints:
            if constraint.function is not None:
                raise UnsupportedProblemError("MILP backend requires linear constraints", context={"constraint": constraint.name})
            row = [0.0] * n
            for name, coefficient in constraint.linear.items():
                row[name_to_index[name]] = coefficient
            rows.append(row)
            if constraint.sense.value == "<=":
                lb.append(-np.inf); ub.append(constraint.rhs)
            elif constraint.sense.value == ">=":
                lb.append(constraint.rhs); ub.append(np.inf)
            else:
                lb.append(constraint.rhs); ub.append(constraint.rhs)

        linear_constraints = None
        if rows:
            linear_constraints = LinearConstraint(np.asarray(rows, dtype=float), np.asarray(lb), np.asarray(ub))

        backend_options: dict[str, Any] = {
            "presolve": self.presolve,
            "mip_rel_gap": self.mip_rel_gap,
        }
        time_limit = self.effective_time_limit(options)
        if time_limit is not None:
            backend_options["time_limit"] = time_limit
        node_limit = options.get("node_limit", self.node_limit)
        if node_limit is not None:
            backend_options["node_limit"] = int(node_limit)
        backend_options.update(dict(options.get("scipy_options", {})))

        started = time.perf_counter()
        result = milp(
            c,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=linear_constraints,
            options=backend_options,
        )
        runtime = self.timed(started)
        point = point_from_vector(problem, result.x, project=False) if getattr(result, "x", None) is not None else None
        status_code = int(result.status)
        message = str(result.message)
        feasible_point = point is not None and is_feasible(problem, point, tolerance=self.feasibility_tolerance)

        if status_code == 0:
            status, termination, optimality = SolverStatus.OPTIMAL, TerminationReason.OPTIMAL, OptimalityStatus.GLOBAL
        elif status_code == 1:
            status = SolverStatus.TIMEOUT if "time" in message.lower() else SolverStatus.LIMIT_REACHED
            if "time" in message.lower():
                termination = TerminationReason.TIME_LIMIT
            elif "node" in message.lower():
                termination = TerminationReason.NODE_LIMIT
            else:
                termination = TerminationReason.ITERATION_LIMIT
            optimality = OptimalityStatus.NONE
        elif status_code == 2:
            status, termination, optimality = SolverStatus.INFEASIBLE, TerminationReason.INFEASIBLE, OptimalityStatus.NONE
        elif status_code == 3:
            status, termination, optimality = SolverStatus.UNBOUNDED, TerminationReason.UNBOUNDED, OptimalityStatus.NONE
        else:
            status, termination, optimality = SolverStatus.NUMERICAL_FAILURE, TerminationReason.NUMERICAL_FAILURE, OptimalityStatus.NONE

        success = status is SolverStatus.OPTIMAL or (status in {SolverStatus.LIMIT_REACHED, SolverStatus.TIMEOUT} and feasible_point)
        transformed_fun = getattr(result, "fun", None)
        raw_dual = getattr(result, "mip_dual_bound", None)
        primal = None if transformed_fun is None else sign * float(transformed_fun) + constant
        dual = None if raw_dual is None or not np.isfinite(raw_dual) else sign * float(raw_dual) + constant
        abs_gap, rel_gap = absolute_relative_gap(primal, dual)
        backend_gap = getattr(result, "mip_gap", None)
        if backend_gap is not None and np.isfinite(backend_gap):
            rel_gap = float(backend_gap)

        return self._solution(
            problem,
            status=status,
            termination=termination,
            success=success,
            point=point,
            optimality=optimality,
            runtime_seconds=runtime,
            primal_bound=primal,
            dual_bound=dual,
            absolute_gap=abs_gap,
            relative_gap=rel_gap,
            nodes=int(result.mip_node_count) if getattr(result, "mip_node_count", None) is not None else None,
            diagnostics={
                "backend": "HiGHS",
                "scipy_status": status_code,
                "message": message,
                "mip_gap_backend": None if backend_gap is None else float(backend_gap),
            },
            metadata={"backend_success": bool(getattr(result, "success", False))},
        )


__all__ = ["LinearSolver"]


if __name__ == "__main__":
    def _smoke_test() -> None:
        configure_logging()
        smoke_logger = get_logger("LinearSolverSmoke")
        problem = OptimizationProblem(
            "linear_smoke",
            (Variable("x", VariableDomain.real(0, 2)),),
            (Objective("f", ObjectiveSense.MAXIMIZE, linear={"x": 1}),),
            problem_class=ProblemClass.LP,
        )
        result = LinearSolver().solve(problem)
        assert result.status is SolverStatus.OPTIMAL
        assert result.primal_solution is not None and abs(result.primal_solution["x"] - 2.0) <= 1e-8
        smoke_logger.info("LinearSolver smoke test passed | objective=%s", result.objective_value)

    _smoke_test()
