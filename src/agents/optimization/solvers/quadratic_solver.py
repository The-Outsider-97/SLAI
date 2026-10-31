"""Constrained quadratic-programming adapter.

Convex-QP semantics follow Boyd & Vandenberghe (2004); numerical stopping and
local/global distinctions follow Nocedal & Wright (2006).  The adapter uses an
explicit quadratic Hessian and linear constraints rather than treating a QP as
an opaque black-box NLP.
"""
from __future__ import annotations

__version__ = "2.3.0"

import time
import numpy as np # type: ignore

from collections.abc import Mapping
from typing import Any
from scipy.optimize import Bounds, LinearConstraint, minimize # type: ignore

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import *
from ..optimization_math import *
from ..optimization_types import *
from .base_solver import BaseSolver
from logs.logger import configure_logging, get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Quadratic Optimization Solver")
printer = PrettyPrinter()


class QuadraticSolver(BaseSolver):
    solver_name = "scipy_quadratic"
    backend_package = "scipy"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        global_config = load_global_config()
        section = dict(get_config_section("quadratic_solver", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.quadratic_config = section
        self.method = str(section.get("method", "trust-constr"))
        if self.method not in {"trust-constr", "SLSQP"}:
            raise ValueError("quadratic_solver.method must be trust-constr or SLSQP")
        self.max_iterations = int(section.get("max_iterations", 1000))
        if self.max_iterations <= 0:
            raise ValueError("quadratic_solver.max_iterations must be positive")
        self.psd_tolerance = require_finite(section.get("psd_tolerance", 1e-10), "psd_tolerance")

        logger.info(f"Initializing quadratic solver with {self.max_iterations}")

    @property
    def capability(self) -> SolverCapability:
        return SolverCapability(
            name=self.solver_name,
            problem_classes=frozenset({ProblemClass.QP}),
            supports_continuous=True,
            supports_quadratic_objective=True,
            supports_callable_objective=False,
            supports_nonlinear_constraints=False,
            supports_warm_start=True,
            deterministic=True,
            backend=f"scipy.optimize.{self.method}",
        )

    def solve(self, problem: OptimizationProblem, **options: Any):
        self.ensure_usable(problem)
        if len(problem.objectives) != 1 or not problem.objectives[0].quadratic:
            raise UnsupportedProblemError("QuadraticSolver requires exactly one explicit quadratic objective")
        if any(variable.domain.kind is not DomainKind.REAL for variable in problem.variables):
            raise UnsupportedProblemError("QuadraticSolver supports continuous real variables only")
        if any(variable.shape for variable in problem.variables):
            raise UnsupportedProblemError("QuadraticSolver supports scalar variables only")
        if any(constraint.function is not None for constraint in problem.constraints):
            raise UnsupportedProblemError("QuadraticSolver supports linear constraints only")
        if any(variable.domain.strict_lower or variable.domain.strict_upper for variable in problem.variables):
            raise UnsupportedProblemError("strict bounds are not representable by this QP backend")

        q, c, transformed_constant, sign = quadratic_objective_matrices(problem)
        convex_transformed = is_positive_semidefinite(q, tolerance=self.psd_tolerance)
        if problem.known_convex is True and not convex_transformed:
            raise UnsupportedProblemError("problem declares convex QP but transformed Hessian is not positive semidefinite")

        name_to_index = {variable.name: i for i, variable in enumerate(problem.variables)}
        rows: list[list[float]] = []
        lb: list[float] = []
        ub: list[float] = []
        for constraint in problem.constraints:
            row = [0.0] * len(problem.variables)
            for name, coefficient in constraint.linear.items():
                row[name_to_index[name]] = coefficient
            rows.append(row)
            if constraint.sense is ConstraintSense.LE:
                lb.append(-np.inf); ub.append(constraint.rhs)
            elif constraint.sense is ConstraintSense.GE:
                lb.append(constraint.rhs); ub.append(np.inf)
            else:
                lb.append(constraint.rhs); ub.append(constraint.rhs)

        scipy_constraints: list[Any] = []
        if rows:
            scipy_constraints.append(LinearConstraint(np.asarray(rows, dtype=float), np.asarray(lb), np.asarray(ub)))
        lower = np.asarray([-np.inf if v.domain.lower is None else v.domain.lower for v in problem.variables], dtype=float)
        upper = np.asarray([np.inf if v.domain.upper is None else v.domain.upper for v in problem.variables], dtype=float)
        bounds = Bounds(lower, upper)

        x0 = np.asarray(options.get("x0", initial_vector(problem)), dtype=float)
        if x0.shape != (len(problem.variables),) or not np.all(np.isfinite(x0)):
            raise ValueError("QP warm start x0 has invalid dimension or non-finite values")
        x0 = np.minimum(np.maximum(x0, np.where(np.isfinite(lower), lower, x0)), np.where(np.isfinite(upper), upper, x0))

        deadline = Deadline.from_seconds(self.effective_time_limit(options))
        best_feasible_x: np.ndarray | None = None
        best_transformed = float("inf")
        evaluations = 0

        def observe(x: np.ndarray, value: float | None = None) -> None:
            nonlocal best_feasible_x, best_transformed
            candidate = point_from_vector(problem, x, project=False)
            if not is_feasible(problem, candidate, tolerance=self.feasibility_tolerance):
                return
            transformed = (
                float(0.5 * x @ q @ x + c @ x + sign * transformed_constant)
                if value is None
                else float(value)
            )
            if best_feasible_x is None or transformed < best_transformed:
                best_feasible_x = np.asarray(x, dtype=float).copy()
                best_transformed = transformed

        def check_time() -> None:
            if deadline.expired():
                raise StopForTime

        def fun(x: np.ndarray) -> float:
            nonlocal evaluations
            check_time()
            value = float(0.5 * x @ q @ x + c @ x + sign * transformed_constant)
            evaluations += 1
            observe(x, value)
            return value

        def jac(x: np.ndarray) -> np.ndarray:
            check_time()
            return q @ x + c

        def hess(_: np.ndarray) -> np.ndarray:
            check_time()
            return q

        def callback(xk: np.ndarray, *_: Any) -> None:
            check_time()
            observe(np.asarray(xk, dtype=float))

        solver_options: dict[str, Any]
        if self.method == "trust-constr":
            solver_options = {
                "maxiter": self.max_iterations,
                "gtol": self.optimality_tolerance,
                "xtol": self.optimality_tolerance,
                "barrier_tol": self.feasibility_tolerance,
                "verbose": 0,
            }
        else:
            solver_options = {
                "maxiter": self.max_iterations,
                "ftol": self.optimality_tolerance,
                "disp": False,
            }
        solver_options.update(dict(options.get("scipy_options", {})))

        started = time.perf_counter()
        result = None
        try:
            result = minimize(
                fun,
                x0,
                method=self.method,
                jac=jac,
                hess=hess if self.method == "trust-constr" else None,
                bounds=bounds,
                constraints=scipy_constraints,
                callback=callback,
                options=solver_options,
            )
        except StopForTime:
            runtime = self.timed(started)
            point = (
                point_from_vector(problem, best_feasible_x, project=False)
                if best_feasible_x is not None
                else None
            )
            return self._solution(
                problem,
                status=SolverStatus.TIMEOUT,
                termination=TerminationReason.TIME_LIMIT,
                success=point is not None,
                point=point,
                optimality=OptimalityStatus.NONE,
                runtime_seconds=runtime,
                evaluations=evaluations,
                diagnostics={
                    "backend": self.method,
                    "stopped_by": "time_limit",
                    "convex_transformed_hessian": convex_transformed,
                },
            )

        runtime = self.timed(started)
        if result is None:
            raise RuntimeError("quadratic backend returned without a result")
        point = point_from_vector(problem, result.x, project=False) if getattr(result, "x", None) is not None else None
        feasible = point is not None and is_feasible(problem, point, tolerance=self.feasibility_tolerance)
        backend_success = bool(getattr(result, "success", False))
        message = str(getattr(result, "message", ""))

        if backend_success and feasible:
            status, termination = SolverStatus.OPTIMAL, TerminationReason.CONVERGED
            optimality = OptimalityStatus.GLOBAL if convex_transformed else OptimalityStatus.LOCAL
            success = True
        else:
            lowered = message.lower()
            if "maximum" in lowered or "iteration" in lowered or int(getattr(result, "status", -1)) == 0:
                status, termination = SolverStatus.LIMIT_REACHED, TerminationReason.ITERATION_LIMIT
            elif "nan" in lowered or "singular" in lowered or "precision" in lowered:
                status, termination = SolverStatus.NUMERICAL_FAILURE, TerminationReason.NUMERICAL_FAILURE
            else:
                status, termination = SolverStatus.NUMERICAL_FAILURE, TerminationReason.NUMERICAL_FAILURE
            optimality = OptimalityStatus.NONE
            success = feasible

        primal = None
        if point is not None:
            # _solution evaluates the objective in original sense; the bound is
            # populated from the same scalar below without relying on backend
            # sign conventions.
            from ..optimization_math import evaluate_objective
            primal = evaluate_objective(problem.objectives[0], point)
        dual = primal if status is SolverStatus.OPTIMAL and optimality is OptimalityStatus.GLOBAL else None

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
            iterations=int(result.nit) if getattr(result, "nit", None) is not None else None,
            evaluations=evaluations if evaluations else (int(result.nfev) if getattr(result, "nfev", None) is not None else None),
            diagnostics={
                "backend": self.method,
                "message": message,
                "scipy_status": int(getattr(result, "status", -1)),
                "convex_transformed_hessian": convex_transformed,
                "optimality_measure": None if getattr(result, "optimality", None) is None else float(result.optimality),
                "constraint_violation_backend": None if getattr(result, "constr_violation", None) is None else float(result.constr_violation),
            },
            metadata={"backend_success": backend_success},
        )


__all__ = ["QuadraticSolver"]


if __name__ == "__main__":
    def _smoke_test() -> None:
        configure_logging()
        smoke_logger = get_logger("QuadraticSolverSmoke")
        problem = OptimizationProblem(
            "quadratic_smoke",
            (Variable("x", VariableDomain.real(-5, 5)),),
            (Objective("f", linear={"x": -4}, quadratic={("x", "x"): 1}, constant=4),),
            problem_class=ProblemClass.QP,
            known_convex=True,
        )
        result = QuadraticSolver().solve(problem)
        assert result.success and result.primal_solution is not None
        assert abs(result.primal_solution["x"] - 2.0) <= 1e-4
        smoke_logger.info("QuadraticSolver smoke test passed | objective=%s", result.objective_value)


    _smoke_test()
