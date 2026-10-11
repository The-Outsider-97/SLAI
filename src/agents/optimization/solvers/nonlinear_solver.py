"""Constrained nonlinear-programming adapter using SciPy.

Nocedal & Wright (2006) are the principal numerical foundation.  Successful
non-convex solves are reported as local optima/converged points; they are never
silently promoted to global optima without a convexity certificate supplied by
or derivable from the problem.
"""
from __future__ import annotations

__version__ = "2.3.0"

import time
import numpy as np # type: ignore

from collections.abc import Mapping, Sequence
from typing import Any
from scipy.optimize import minimize # type: ignore

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.optimization_errors import *
from ..utils.optimization_helpers import Deadline
from ..optimization_math import *
from ..optimization_types import *
from .base_solver import BaseSolver
from logs.logger import configure_logging, get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Non-Linear Optimization Solver")
printer = PrettyPrinter()


class NonLinearSolver(BaseSolver):
    solver_name = "scipy_nonlinear"
    backend_package = "scipy"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        global_config = load_global_config()
        section = dict(get_config_section("non_linear_solver", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.nls_config = section
        self.method = str(section.get("method", "SLSQP"))
        if self.method not in {"SLSQP", "trust-constr"}:
            raise ValueError("non_linear_solver.method must be SLSQP or trust-constr")
        self.max_iterations = int(section.get("max_iterations", 1000))
        self.max_evaluations = section.get("max_evaluations")
        if self.max_iterations <= 0:
            raise ValueError("non_linear_solver.max_iterations must be positive")
        if self.max_evaluations is not None and (isinstance(self.max_evaluations, bool) or int(self.max_evaluations) <= 0):
            raise ValueError("non_linear_solver.max_evaluations must be positive when set")

    @property
    def capability(self) -> SolverCapability:
        return SolverCapability(
            name=self.solver_name,
            problem_classes=frozenset({ProblemClass.NLP}),
            supports_continuous=True,
            supports_callable_objective=True,
            supports_quadratic_objective=True,
            supports_nonlinear_constraints=True,
            supports_warm_start=True,
            deterministic=True,
            backend=f"scipy.optimize.{self.method}",
        )

    def solve(self, problem: OptimizationProblem, **options: Any):
        self.ensure_usable(problem)
        if len(problem.objectives) != 1:
            raise UnsupportedProblemError("NonLinearSolver requires exactly one objective")
        if any(variable.domain.kind is not DomainKind.REAL for variable in problem.variables):
            raise UnsupportedProblemError("NonLinearSolver supports continuous real variables only")
        if any(variable.shape for variable in problem.variables):
            raise UnsupportedProblemError("NonLinearSolver supports scalar variables only")
        if any(variable.domain.strict_lower or variable.domain.strict_upper for variable in problem.variables):
            raise UnsupportedProblemError("strict variable bounds are not supported by SciPy constrained minimizers")

        objective = problem.objectives[0]
        x0 = np.asarray(options.get("x0", initial_vector(problem)), dtype=float)
        if x0.shape != (len(problem.variables),) or not np.all(np.isfinite(x0)):
            raise ValueError("NLP warm start x0 has invalid dimension or non-finite values")
        bounds = scipy_bounds(problem)
        time_limit = self.effective_time_limit(options)
        deadline = Deadline.from_seconds(time_limit)
        max_evaluations_raw = options.get("max_evaluations", self.max_evaluations)
        max_evaluations = None if max_evaluations_raw is None else int(max_evaluations_raw)
        evaluations = 0
        best_feasible_point: dict[str, Any] | None = None
        best_signed = float("inf")

        def point(x: Sequence[float]) -> dict[str, Any]:
            # Do not clip backend iterates: a solver result outside declared
            # bounds must remain visible to common feasibility validation.
            return point_from_vector(problem, x, project=False)

        def check_limits() -> None:
            if deadline.expired():
                raise StopForTime
            if max_evaluations is not None and evaluations >= max_evaluations:
                raise StopForEvaluations

        def fun(x: np.ndarray) -> float:
            nonlocal evaluations, best_feasible_point, best_signed
            check_limits()
            p = point(x)
            try:
                value = signed_objective_value(objective, p)
            except OptimizationEvaluationError:
                raise
            evaluations += 1
            if max_constraint_violation(problem, p) is not None and is_feasible(problem, p, tolerance=self.feasibility_tolerance):
                if value < best_signed:
                    best_signed = value
                    best_feasible_point = dict(p)
            return value

        names = [variable.name for variable in problem.variables]
        sign = 1.0 if objective.sense is ObjectiveSense.MINIMIZE else -1.0

        def derivative_vector(raw: Mapping[str, float] | Sequence[float], *, label: str) -> np.ndarray:
            if isinstance(raw, Mapping):
                try:
                    values = [raw[name] for name in names]
                except KeyError as exc:
                    raise OptimizationEvaluationError(
                        f"{label} mapping is missing a variable",
                        context={"variable": str(exc)},
                    ) from exc
            else:
                values = list(raw)
            if len(values) != len(names):
                raise OptimizationEvaluationError(
                    f"{label} dimension mismatch",
                    context={"expected": len(names), "actual": len(values)},
                )
            vector = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(vector)):
                raise OptimizationEvaluationError(f"{label} returned non-finite values")
            return vector

        jac: Any = None
        objective_gradient = objective.gradient
        if objective_gradient is not None:
            def objective_jacobian(x: np.ndarray) -> np.ndarray:
                check_limits()
                return sign * derivative_vector(objective_gradient(point(x)), label="objective gradient")

            jac = objective_jacobian

        hess: Any = None
        objective_hessian_function = objective.hessian
        if objective_hessian_function is not None and self.method == "trust-constr":
            def objective_hessian(x: np.ndarray) -> np.ndarray:
                check_limits()
                matrix = np.asarray(objective_hessian_function(point(x)), dtype=float)
                expected = (len(names), len(names))
                if matrix.shape != expected:
                    raise OptimizationEvaluationError(
                        "objective Hessian dimension mismatch",
                        context={"expected": expected, "actual": matrix.shape},
                    )
                if not np.all(np.isfinite(matrix)):
                    raise OptimizationEvaluationError("objective Hessian returned non-finite values")
                return sign * matrix

            hess = objective_hessian

        constraints: list[dict[str, Any]] = []
        for constraint in problem.constraints:
            if constraint.sense is ConstraintSense.LE:
                direction = -1.0
                constraint_fun = lambda x, c=constraint: c.rhs - evaluate_constraint_value(c, point(x))
            elif constraint.sense is ConstraintSense.GE:
                direction = 1.0
                constraint_fun = lambda x, c=constraint: evaluate_constraint_value(c, point(x)) - c.rhs
            else:
                direction = 1.0
                constraint_fun = lambda x, c=constraint: evaluate_constraint_value(c, point(x)) - c.rhs

            item: dict[str, Any] = {
                "type": "eq" if constraint.sense is ConstraintSense.EQ else "ineq",
                "fun": constraint_fun,
            }
            if constraint.jacobian is not None:
                def constraint_jac(
                    x: np.ndarray,
                    c=constraint,
                    factor=direction,
                    derivative: Any = constraint.jacobian,
                ) -> np.ndarray:
                    check_limits()
                    return factor * derivative_vector(derivative(point(x)), label=f"constraint Jacobian [{c.name}]")
                item["jac"] = constraint_jac
            elif constraint.function is None and constraint.linear:
                exact = np.asarray([constraint.linear.get(name, 0.0) for name in names], dtype=float) * direction
                item["jac"] = lambda _x, values=exact: values
            constraints.append(item)

        solver_options: dict[str, Any] = {"maxiter": self.max_iterations}
        if self.method == "SLSQP":
            solver_options.update({"ftol": self.optimality_tolerance, "disp": False})
        else:
            solver_options.update({
                "gtol": self.optimality_tolerance,
                "xtol": self.optimality_tolerance,
                "barrier_tol": self.feasibility_tolerance,
                "verbose": 0,
            })
        solver_options.update(dict(options.get("scipy_options", {})))

        started = time.perf_counter()
        stopped_for: str | None = None
        result = None
        try:
            result = minimize(
                fun,
                x0,
                method=self.method,
                jac=jac,
                hess=hess if self.method == "trust-constr" else None,
                bounds=bounds,
                constraints=constraints,
                options=solver_options,
                callback=(lambda *_: check_limits()),
            )
        except StopForTime:
            stopped_for = "time"
        except StopForEvaluations:
            stopped_for = "evaluations"
        runtime = self.timed(started)

        if stopped_for is not None:
            point_result = best_feasible_point
            if stopped_for == "time":
                status, termination = SolverStatus.TIMEOUT, TerminationReason.TIME_LIMIT
            else:
                status, termination = SolverStatus.LIMIT_REACHED, TerminationReason.EVALUATION_LIMIT
            return self._solution(
                problem,
                status=status,
                termination=termination,
                success=point_result is not None,
                point=point_result,
                optimality=OptimalityStatus.NONE,
                runtime_seconds=runtime,
                evaluations=evaluations,
                diagnostics={"backend": self.method, "stopped_by": stopped_for},
            )

        if result is None:
            raise RuntimeError("nonlinear backend returned without a result")
        point_result = point(result.x) if getattr(result, "x", None) is not None else best_feasible_point
        feasible = point_result is not None and is_feasible(problem, point_result, tolerance=self.feasibility_tolerance)
        backend_success = bool(getattr(result, "success", False))
        message = str(getattr(result, "message", ""))
        if backend_success and feasible:
            status, termination = SolverStatus.OPTIMAL, TerminationReason.CONVERGED
            optimality = OptimalityStatus.GLOBAL if problem.known_convex is True else OptimalityStatus.LOCAL
            success = True
        else:
            lowered = message.lower()
            if "iteration" in lowered or "maximum" in lowered:
                status, termination = SolverStatus.LIMIT_REACHED, TerminationReason.ITERATION_LIMIT
            elif "incompatible" in lowered or "inequality constraints incompatible" in lowered:
                status, termination = SolverStatus.INFEASIBLE, TerminationReason.INFEASIBLE
            else:
                status, termination = SolverStatus.NUMERICAL_FAILURE, TerminationReason.NUMERICAL_FAILURE
            optimality = OptimalityStatus.NONE
            success = feasible and status not in {SolverStatus.INFEASIBLE, SolverStatus.UNBOUNDED}

        primal = evaluate_objective(objective, point_result) if point_result is not None else None
        dual = primal if success and problem.known_convex is True and status is SolverStatus.OPTIMAL else None
        return self._solution(
            problem,
            status=status,
            termination=termination,
            success=success,
            point=point_result,
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
                "backend_nfev": int(result.nfev) if getattr(result, "nfev", None) is not None else None,
                "backend_njev": int(result.njev) if getattr(result, "njev", None) is not None else None,
            },
            metadata={"backend_success": backend_success},
        )


__all__ = ["NonLinearSolver"]


def _smoke_test() -> None:
    configure_logging()
    smoke_logger = get_logger("NonLinearSolverSmoke")
    problem = OptimizationProblem(
        "nonlinear_smoke",
        (Variable("x", VariableDomain.real(-5, 5), initial=2.0),),
        (Objective("f", function=lambda p: p["x"] ** 2),),
        (Constraint("lower", ConstraintSense.GE, 1.0, function=lambda p: p["x"]),),
        problem_class=ProblemClass.NLP,
        known_convex=True,
    )
    result = NonLinearSolver().solve(problem)
    assert result.success and result.primal_solution is not None
    assert abs(result.primal_solution["x"] - 1.0) <= 1e-4
    smoke_logger.info("NonLinearSolver smoke test passed | objective=%s", result.objective_value)


if __name__ == "__main__":
    _smoke_test()
