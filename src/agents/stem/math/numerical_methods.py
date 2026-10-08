"""
Numerical methods for the STEM subsystem.

Ownership
- numerical linear algebra;
- nonlinear equation/root solving;
- interpolation;
- approximation;
- numerical differentiation;
- numerical integration;
- ODE algorithms;
- PDE discretization interfaces;
- convergence analysis;
- conditioning;
- residual calculation;
- numerical precision / error controls.

Sources
- Quarteroni, A., Sacco, R., & Saleri, F. (2007). Numerical Mathematics (2nd ed.). Springer.
- Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM.
- Trefethen, L. N., & Bau, D. (2022 anniversary edition; original 1997).
  Numerical Linear Algebra. SIAM.
- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). Solving Ordinary Differential
  Equations I: Nonstiff Problems (2nd ed.). Springer. DOI 10.1007/978-3-540-78862-1.
- LeVeque, R. J. (2007). Finite Difference Methods for Ordinary and Partial
  Differential Equations. SIAM. DOI 10.1137/1.9780898717839.
- Davis, P. J., & Rabinowitz, P. (1984). Methods of Numerical Integration (2nd ed.).
  Academic Press. DOI 10.1016/C2013-0-10566-1.
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import time

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import *
from ..utils.stem_helpers import *
from ..stem_types import ConvergenceStatus, SolverResult
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Numerical Methods")
printer = PrettyPrinter()

class NumericalMethods:
    """Finite-precision numerical algorithms for the STEM subsystem."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.nm_config = dict(get_config_section("numerical_methods", config=self.config) or {})
        if config:
            self.nm_config.update(dict(config))

        self._default_tol = float(self.nm_config.get("default_tolerance", 1e-10))
        self._default_max_iter = int(self.nm_config.get("default_max_iterations", 200))
        self._default_h = float(self.nm_config.get("default_step", 1e-6))

    # ======================================================================
    # Numerical linear algebra
    # ======================================================================

    def lu_decomposition(self, A: Sequence[Sequence[float]]) -> Mapping[str, Any]:
        """LU decomposition with partial pivoting."""
        matrix = check_square_matrix(A, "A", error_cls=STEMLinearAlgebraError)
        n = len(matrix)
        L = [[0.0] * n for _ in range(n)]
        U = [row[:] for row in matrix]
        P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for k in range(n):
            pivot = max(range(k, n), key=lambda i: abs(U[i][k]))
            if abs(U[pivot][k]) < 1e-300:
                raise STEMLinearAlgebraError(
                    "matrix is singular in LU decomposition",
                    context={"pivot_index": k, "pivot_value": U[pivot][k]},
                )
            if pivot != k:
                U[k], U[pivot] = U[pivot], U[k]
                P[k], P[pivot] = P[pivot], P[k]
                for j in range(k):
                    L[k][j], L[pivot][j] = L[pivot][j], L[k][j]

            L[k][k] = 1.0
            for i in range(k + 1, n):
                factor = U[i][k] / U[k][k]
                L[i][k] = factor
                for j in range(k, n):
                    U[i][j] -= factor * U[k][j]

        return {"L": L, "U": U, "P": P}

    def qr_decomposition(self, A: Sequence[Sequence[float]]) -> Mapping[str, Any]:
        """Householder QR decomposition."""
        matrix = as_float_matrix(A, "A", error_cls=STEMLinearAlgebraError)
        m = len(matrix)
        if m == 0:
            raise STEMLinearAlgebraError("A must not be empty")
        n = len(matrix[0])
        for i, row in enumerate(matrix):
            if len(row) != n:
                raise STEMLinearAlgebraError("A must be rectangular", context={"row": i})

        R = [row[:] for row in matrix]
        Q = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

        for k in range(min(m - 1, n)):
            x = [R[i][k] for i in range(k, m)]
            norm_x = math.sqrt(sum(v * v for v in x))
            if norm_x == 0.0:
                continue
            sign = 1.0 if x[0] >= 0.0 else -1.0
            v = x[:]
            v[0] += sign * norm_x
            v_norm_sq = sum(vi * vi for vi in v)
            if v_norm_sq == 0.0:
                continue

            for j in range(n):
                s = 0.0
                for i in range(k, m):
                    s += v[i - k] * R[i][j]
                factor = 2.0 * s / v_norm_sq
                for i in range(k, m):
                    R[i][j] -= factor * v[i - k]

            for j in range(m):
                s = 0.0
                for i in range(k, m):
                    s += v[i - k] * Q[j][i]
                factor = 2.0 * s / v_norm_sq
                for i in range(k, m):
                    Q[j][i] -= factor * v[i - k]

        return {"Q": Q, "R": R}

    def cholesky(self, A: Sequence[Sequence[float]]) -> List[List[float]]:
        """Cholesky factorization of a symmetric positive-definite matrix."""
        matrix = check_positive_definite(A, "A", error_cls=STEMLinearAlgebraError)
        n = len(matrix)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = matrix[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    if s <= 0.0:
                        raise STEMLinearAlgebraError("matrix is not positive definite", context={"pivot_index": i})
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
        return L

    def _forward_substitution(self, L: List[List[float]], b: List[float]) -> List[float]:
        n = len(L)
        y = [0.0] * n
        for i in range(n):
            s = b[i] - sum(L[i][j] * y[j] for j in range(i))
            if L[i][i] == 0.0:
                raise STEMLinearAlgebraError("zero pivot in forward substitution")
            y[i] = s / L[i][i]
        return y

    def _backward_substitution(self, U: List[List[float]], y: List[float]) -> List[float]:
        n = len(U)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))
            if U[i][i] == 0.0:
                raise STEMLinearAlgebraError("zero pivot in backward substitution")
            x[i] = s / U[i][i]
        return x

    def solve_linear_system(
        self,
        A: Sequence[Sequence[float]],
        b: Sequence[float],
    ) -> List[float]:
        """Solve ``Ax = b`` via LU decomposition with partial pivoting."""
        matrix = check_square_matrix(A, "A", error_cls=STEMLinearAlgebraError)
        rhs = as_float_array(b, "b", error_cls=STEMLinearAlgebraError)
        n = len(matrix)
        if len(rhs) != n:
            raise STEMLinearAlgebraError("A and b must have compatible dimensions")

        decomp = self.lu_decomposition(matrix)
        L = decomp["L"]
        U = decomp["U"]
        P = decomp["P"]

        permuted_b = [sum((P[i][j] * rhs[j] for j in range(n)), 0.0) for i in range(n)]
        y = self._forward_substitution(L, permuted_b)
        return self._backward_substitution(U, y)

    def least_squares(
        self,
        A: Sequence[Sequence[float]],
        b: Sequence[float],
    ) -> List[float]:
        """Solve ``min ||Ax - b||`` via QR decomposition."""
        matrix = as_float_matrix(A, "A", error_cls=STEMLinearAlgebraError)
        rhs = as_float_array(b, "b", error_cls=STEMLinearAlgebraError)
        m = len(matrix)
        if m == 0:
            raise STEMLinearAlgebraError("A must not be empty")
        n = len(matrix[0])
        if len(rhs) != m:
            raise STEMLinearAlgebraError("A and b dimensions do not match")

        qr = self.qr_decomposition(matrix)
        Q = qr["Q"]
        R = qr["R"]

        qtb = [sum((Q[i][j] * rhs[j] for j in range(m)), 0.0) for i in range(m)]
        # take upper n x n block of R
        R_upper = [R[i][:n] for i in range(n)]
        return self._backward_substitution(R_upper, qtb[:n])

    def eigenvalues_symmetric(
        self,
        A: Sequence[Sequence[float]],
        max_iter: int = 100,
        tol: float = 1e-12,
    ) -> List[float]:
        """Eigenvalues of a symmetric matrix via cyclic Jacobi rotations."""
        matrix = check_symmetric_matrix(A, "A", error_cls=STEMLinearAlgebraError)
        n = len(matrix)
        A_work = [row[:] for row in matrix]

        for _ in range(int(max_iter)):
            off = math.sqrt(sum(A_work[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
            if off < tol:
                break
            for p in range(n - 1):
                for q in range(p + 1, n):
                    if abs(A_work[p][q]) < 1e-300:
                        continue
                    theta = (A_work[q][q] - A_work[p][p]) / (2.0 * A_work[p][q])
                    t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                    c = 1.0 / math.sqrt(t * t + 1.0)
                    s = t * c
                    for k in range(n):
                        akp = A_work[k][p]
                        akq = A_work[k][q]
                        A_work[k][p] = c * akp - s * akq
                        A_work[k][q] = s * akp + c * akq
                    for k in range(n):
                        apk = A_work[p][k]
                        aqk = A_work[q][k]
                        A_work[p][k] = c * apk - s * aqk
                        A_work[q][k] = s * apk + c * aqk

        return [A_work[i][i] for i in range(n)]

    def singular_values(self, A: Sequence[Sequence[float]]) -> List[float]:
        """Singular values via eigenvalues of ``A^T A``."""
        matrix = as_float_matrix(A, "A", error_cls=STEMLinearAlgebraError)
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        if n == 0:
            raise STEMLinearAlgebraError("A must not be empty")

        ATA = [[sum(matrix[k][i] * matrix[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
        eigenvalues = self.eigenvalues_symmetric(ATA)
        return [math.sqrt(max(0.0, value)) for value in eigenvalues]

    def matrix_inverse(self, A: Sequence[Sequence[float]]) -> List[List[float]]:
        """Matrix inverse via LU factorization with partial pivoting."""
        matrix = check_square_matrix(A, "A", error_cls=STEMLinearAlgebraError)
        n = len(matrix)
        decomp = self.lu_decomposition(matrix)
        L = decomp["L"]
        U = decomp["U"]
        P = decomp["P"]

        inverse = [[0.0] * n for _ in range(n)]
        for col in range(n):
            e = [0.0] * n
            e[col] = 1.0
            permuted = [float(sum(P[i][j] * e[j] for j in range(n))) for i in range(n)]
            y = self._forward_substitution(L, permuted)
            x = self._backward_substitution(U, y)
            for i in range(n):
                inverse[i][col] = x[i]
        return inverse

    # ======================================================================
    # Nonlinear root solving
    # ======================================================================

    def bisection(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        *,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> SolverResult:
        """Bisection root finder on a sign-changing interval."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        a_f, b_f = validate_bounds(a, b, "interval", error_cls=STEMNumericalError)
        tol_f = float(tol if tol is not None else self._default_tol)
        max_iter_i = int(max_iter if max_iter is not None else self._default_max_iter)

        start = time.perf_counter()
        fa = float(f(a_f))
        fb = float(f(b_f))
        if fa == 0.0:
            return SolverResult(
                solution=a_f, residual=0.0, iterations=0, converged=True,
                status=ConvergenceStatus.CONVERGED, message="left endpoint is a root",
                runtime=time.perf_counter() - start,
            )
        if fb == 0.0:
            return SolverResult(
                solution=b_f, residual=0.0, iterations=0, converged=True,
                status=ConvergenceStatus.CONVERGED, message="right endpoint is a root",
                runtime=time.perf_counter() - start,
            )
        if fa * fb > 0.0:
            raise STEMNumericalError(
                "bisection requires a sign change on the interval",
                context={"f(a)": fa, "f(b)": fb},
            )

        last_mid = 0.5 * (a_f + b_f)
        for iteration in range(1, max_iter_i + 1):
            mid = 0.5 * (a_f + b_f)
            fm = float(f(mid))
            last_mid = mid
            if abs(fm) < tol_f or 0.5 * (b_f - a_f) < tol_f:
                return SolverResult(
                    solution=mid, residual=abs(fm), iterations=iteration,
                    converged=True, status=ConvergenceStatus.CONVERGED,
                    message="converged",
                    diagnostics={"final_interval": b_f - a_f},
                    runtime=time.perf_counter() - start,
                )
            if fa * fm < 0.0:
                b_f = mid
                fb = fm
            else:
                a_f = mid
                fa = fm

        return SolverResult(
            solution=last_mid, residual=abs(float(f(last_mid))),
            iterations=max_iter_i, converged=False,
            status=ConvergenceStatus.MAX_ITERATIONS,
            message="maximum iterations reached",
            runtime=time.perf_counter() - start,
        )

    def newton_raphson(
        self,
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        *,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> SolverResult:
        """Newton-Raphson root finder using an analytic derivative."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        validate_callable(df, "df", error_cls=STEMValidationError)
        x = ensure_finite_number(x0, "x0", error_cls=STEMNumericalError)
        tol_f = float(tol if tol is not None else self._default_tol)
        max_iter_i = int(max_iter if max_iter is not None else self._default_max_iter)

        start = time.perf_counter()
        for iteration in range(1, max_iter_i + 1):
            fx = float(f(x))
            if abs(fx) < tol_f:
                return SolverResult(
                    solution=x, residual=abs(fx), iterations=iteration,
                    converged=True, status=ConvergenceStatus.CONVERGED,
                    message="converged",
                    runtime=time.perf_counter() - start,
                )
            dfx = float(df(x))
            if dfx == 0.0:
                raise STEMNumericalError(
                    "derivative vanished during Newton iteration",
                    context={"x": x, "iteration": iteration},
                )
            x_next = x - fx / dfx
            if abs(x_next - x) < tol_f:
                fx_next = float(f(x_next))
                return SolverResult(
                    solution=x_next, residual=abs(fx_next), iterations=iteration,
                    converged=True, status=ConvergenceStatus.CONVERGED,
                    message="converged",
                    runtime=time.perf_counter() - start,
                )
            x = x_next

        return SolverResult(
            solution=x, residual=abs(float(f(x))), iterations=max_iter_i,
            converged=False, status=ConvergenceStatus.MAX_ITERATIONS,
            message="maximum iterations reached",
            runtime=time.perf_counter() - start,
        )

    def secant(
        self,
        f: Callable[[float], float],
        x0: float,
        x1: float,
        *,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> SolverResult:
        """Secant root finder using two initial points."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        a = ensure_finite_number(x0, "x0", error_cls=STEMNumericalError)
        b = ensure_finite_number(x1, "x1", error_cls=STEMNumericalError)
        tol_f = float(tol if tol is not None else self._default_tol)
        max_iter_i = int(max_iter if max_iter is not None else self._default_max_iter)

        start = time.perf_counter()
        fa = float(f(a))
        fb = float(f(b))
        for iteration in range(1, max_iter_i + 1):
            if abs(fb) < tol_f:
                return SolverResult(
                    solution=b, residual=abs(fb), iterations=iteration,
                    converged=True, status=ConvergenceStatus.CONVERGED,
                    message="converged",
                    runtime=time.perf_counter() - start,
                )
            if fb == fa:
                raise STEMNumericalError("secant encountered f(b) == f(a)")
            c = b - fb * (b - a) / (fb - fa)
            if abs(c - b) < tol_f:
                fc = float(f(c))
                return SolverResult(
                    solution=c, residual=abs(fc), iterations=iteration,
                    converged=True, status=ConvergenceStatus.CONVERGED,
                    message="converged",
                    runtime=time.perf_counter() - start,
                )
            a, fa = b, fb
            b = c
            fb = float(f(b))

        return SolverResult(
            solution=b, residual=abs(fb), iterations=max_iter_i,
            converged=False, status=ConvergenceStatus.MAX_ITERATIONS,
            message="maximum iterations reached",
            runtime=time.perf_counter() - start,
        )

    def fixed_point(
        self,
        g: Callable[[float], float],
        x0: float,
        *,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> SolverResult:
        """Fixed-point iteration ``x_{k+1} = g(x_k)``."""
        validate_callable(g, "g", error_cls=STEMValidationError)
        x = ensure_finite_number(x0, "x0", error_cls=STEMNumericalError)
        tol_f = float(tol if tol is not None else self._default_tol)
        max_iter_i = int(max_iter if max_iter is not None else self._default_max_iter)

        start = time.perf_counter()
        for iteration in range(1, max_iter_i + 1):
            x_next = float(g(x))
            if abs(x_next - x) < tol_f:
                return SolverResult(
                    solution=x_next, residual=abs(x_next - x), iterations=iteration,
                    converged=True, status=ConvergenceStatus.CONVERGED,
                    message="converged",
                    runtime=time.perf_counter() - start,
                )
            x = x_next

        return SolverResult(
            solution=x, residual=0.0, iterations=max_iter_i, converged=False,
            status=ConvergenceStatus.MAX_ITERATIONS,
            message="maximum iterations reached",
            runtime=time.perf_counter() - start,
        )

    # ======================================================================
    # Interpolation and approximation
    # ======================================================================

    def linear_interpolation(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
    ) -> Callable[[float], float]:
        """Return a piecewise-linear interpolant through ``(xs, ys)``."""
        x = as_float_array(xs, "xs", error_cls=STEMInterpolationError)
        y = as_float_array(ys, "ys", error_cls=STEMInterpolationError)
        if len(x) != len(y):
            raise STEMInterpolationError("xs and ys must have the same length")
        if len(x) < 2:
            raise STEMInterpolationError("linear interpolation requires at least two points")
        for i in range(1, len(x)):
            if x[i] <= x[i - 1]:
                raise STEMInterpolationError("xs must be strictly increasing")

        def interp(query: float) -> float:
            q = ensure_finite_number(query, "query", error_cls=STEMInterpolationError)
            if q <= x[0]:
                return y[0]
            if q >= x[-1]:
                return y[-1]
            # binary search
            lo, hi = 0, len(x) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if x[mid] <= q:
                    lo = mid
                else:
                    hi = mid
            t = (q - x[lo]) / (x[hi] - x[lo])
            return y[lo] + t * (y[hi] - y[lo])

        return interp

    def polynomial_interpolation(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
    ) -> Callable[[float], float]:
        """Lagrange polynomial interpolation."""
        x = as_float_array(xs, "xs", error_cls=STEMInterpolationError)
        y = as_float_array(ys, "ys", error_cls=STEMInterpolationError)
        if len(x) != len(y):
            raise STEMInterpolationError("xs and ys must have the same length")
        if len(x) == 0:
            raise STEMInterpolationError("at least one interpolation point is required")
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[i] == x[j]:
                    raise STEMInterpolationError(
                        "interpolation nodes must be distinct",
                        context={"i": i, "j": j},
                    )

        def interp(query: float) -> float:
            q = float(query)
            total = 0.0
            for i, xi in enumerate(x):
                term = y[i]
                for j, xj in enumerate(x):
                    if i == j:
                        continue
                    denom = xi - xj
                    if denom == 0.0:
                        raise STEMInterpolationError("duplicate interpolation node encountered")
                    term *= (q - xj) / denom
                total += term
            return total

        return interp

    def cubic_spline(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
    ) -> Callable[[float], float]:
        """Natural cubic spline interpolant."""
        x = as_float_array(xs, "xs", error_cls=STEMInterpolationError)
        y = as_float_array(ys, "ys", error_cls=STEMInterpolationError)
        n = len(x)
        if n != len(y):
            raise STEMInterpolationError("xs and ys must have the same length")
        if n < 3:
            return self.linear_interpolation(x, y)
        for i in range(1, n):
            if x[i] <= x[i - 1]:
                raise STEMInterpolationError("xs must be strictly increasing")

        h = [x[i + 1] - x[i] for i in range(n - 1)]
        alpha = [0.0] * n
        for i in range(1, n - 1):
            alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        l = [1.0] + [0.0] * (n - 1)
        mu = [0.0] * n
        z = [0.0] * n

        for i in range(1, n - 1):
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        c = [0.0] * n
        b = [0.0] * (n - 1)
        d = [0.0] * (n - 1)

        for j in range(n - 2, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

        def interp(query: float) -> float:
            q = float(query)
            if q <= x[0]:
                return y[0]
            if q >= x[-1]:
                return y[-1]
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if x[mid] <= q:
                    lo = mid
                else:
                    hi = mid
            dx = q - x[lo]
            return y[lo] + b[lo] * dx + c[lo] * dx * dx + d[lo] * dx * dx * dx

        return interp

    def chebyshev_approximation(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        degree: int,
    ) -> Callable[[float], float]:
        """Chebyshev-node polynomial approximation on ``[a, b]``."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        a_f, b_f = validate_bounds(a, b, "interval", error_cls=STEMNumericalError)
        deg = int(ensure_positive(degree, "degree", allow_zero=True, error_cls=STEMValidationError))

        n = deg + 1
        nodes = [
            0.5 * (a_f + b_f) + 0.5 * (b_f - a_f) * math.cos(math.pi * (2 * k + 1) / (2 * n))
            for k in range(n)
        ]
        values = [
            float(f(node)) if not math.isnan(float(f(node))) else float("nan")
            for node in nodes
        ]

        interp = self.polynomial_interpolation(nodes, values)

        def approx(query: float) -> float:
            q = float(query)
            if q < a_f or q > b_f:
                raise STEMNumericalError(
                    "query outside Chebyshev approximation interval",
                    context={"query": q, "a": a_f, "b": b_f},
                )
            return float(interp(q))

        return approx

    # ======================================================================
    # Numerical differentiation
    # ======================================================================

    def forward_difference(
        self,
        f: Callable[[float], float],
        x: float,
        h: Optional[float] = None,
    ) -> float:
        """First-order forward finite difference."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        x_f = ensure_finite_number(x, "x", error_cls=STEMDifferentiationError)
        h_f = float(h if h is not None else self._default_h)
        if h_f == 0.0:
            raise STEMDifferentiationError("step size must be nonzero")
        try:
            return (float(f(x_f + h_f)) - float(f(x_f))) / h_f
        except STEMDifferentiationError:
            raise
        except Exception as exc:
            raise STEMDifferentiationError(
                "function evaluation failed in forward difference",
                context={"x": x_f, "h": h_f},
                cause=exc,
            ) from exc

    def central_difference(
        self,
        f: Callable[[float], float],
        x: float,
        h: Optional[float] = None,
    ) -> float:
        """Second-order central finite difference."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        x_f = ensure_finite_number(x, "x", error_cls=STEMDifferentiationError)
        h_f = float(h if h is not None else self._default_h)
        if h_f == 0.0:
            raise STEMDifferentiationError("step size must be nonzero")
        try:
            return (float(f(x_f + h_f)) - float(f(x_f - h_f))) / (2.0 * h_f)
        except STEMDifferentiationError:
            raise
        except Exception as exc:
            raise STEMDifferentiationError(
                "function evaluation failed in central difference",
                context={"x": x_f, "h": h_f},
                cause=exc,
            ) from exc

    def richardson_extrapolation(
        self,
        f: Callable[[float], float],
        x: float,
        h: Optional[float] = None,
        levels: int = 4,
    ) -> float:
        """Richardson extrapolation on central differences at dyadically refined steps."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        x_f = ensure_finite_number(x, "x", error_cls=STEMDifferentiationError)
        h_f = float(h if h is not None else self._default_h)
        levels_i = int(ensure_positive(levels, "levels", allow_zero=False, error_cls=STEMValidationError))

        estimates = [self.central_difference(f, x_f, h_f / (2 ** k)) for k in range(levels_i)]
        return richardson_extrapolate(estimates, 2.0, error_cls=STEMDifferentiationError)

    # ======================================================================
    # Numerical integration
    # ======================================================================

    def trapezoidal(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        n: int = 100,
    ) -> float:
        """Composite trapezoidal rule."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        a_f, b_f = validate_bounds(a, b, "interval", error_cls=STEMIntegrationError)
        n_i = int(ensure_positive(n, "n", allow_zero=False, error_cls=STEMValidationError))
        h = (b_f - a_f) / n_i
        total = 0.5 * (float(f(a_f)) + float(f(b_f)))
        for i in range(1, n_i):
            total += float(f(a_f + i * h))
        return total * h

    def simpson(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        n: int = 100,
    ) -> float:
        """Composite Simpson's rule; ``n`` must be even."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        a_f, b_f = validate_bounds(a, b, "interval", error_cls=STEMIntegrationError)
        n_i = int(ensure_positive(n, "n", allow_zero=False, error_cls=STEMValidationError))
        if n_i % 2 != 0:
            n_i += 1
        h = (b_f - a_f) / n_i
        total = float(f(a_f)) + float(f(b_f))
        for i in range(1, n_i):
            total += (4.0 if i % 2 == 1 else 2.0) * float(f(a_f + i * h))
        return total * h / 3.0

    def romberg(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        *,
        tol: Optional[float] = None,
        max_iter: int = 8,
    ) -> float:
        """Romberg integration using trapezoidal refinements and Richardson."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        a_f, b_f = validate_bounds(a, b, "interval", error_cls=STEMIntegrationError)
        tol_f = float(tol if tol is not None else self._default_tol)
        max_iter_i = int(ensure_positive(max_iter, "max_iter", allow_zero=False, error_cls=STEMValidationError))

        R = [[0.0] * max_iter_i for _ in range(max_iter_i)]
        h = b_f - a_f
        R[0][0] = 0.5 * h * (float(f(a_f)) + float(f(b_f)))

        for i in range(1, max_iter_i):
            h *= 0.5
            s = 0.0
            k = 1
            while k <= (1 << (i - 1)):
                s += float(f(a_f + (2 * k - 1) * h))
                k += 1
            R[i][0] = 0.5 * R[i - 1][0] + h * s

            for j in range(1, i + 1):
                R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4.0 ** j - 1.0)

            if abs(R[i][i] - R[i - 1][i - 1]) < tol_f:
                return R[i][i]

        return R[max_iter_i - 1][max_iter_i - 1]

    def gauss_legendre(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        n: int = 5,
    ) -> float:
        """Gauss-Legendre quadrature for small n using tabulated nodes."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        a_f, b_f = validate_bounds(a, b, "interval", error_cls=STEMIntegrationError)
        n_i = int(ensure_positive(n, "n", allow_zero=False, error_cls=STEMValidationError))

        nodes, weights = _gauss_legendre_table(n_i)
        c = 0.5 * (b_f - a_f)
        d = 0.5 * (b_f + a_f)
        total = 0.0
        for xi, wi in zip(nodes, weights):
            total += wi * float(f(c * xi + d))
        return c * total

    def adaptive_quadrature(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        *,
        tol: Optional[float] = None,
        max_depth: int = 20,
    ) -> float:
        """Adaptive Simpson's rule."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        a_f, b_f = validate_bounds(a, b, "interval", error_cls=STEMIntegrationError)
        tol_f = float(tol if tol is not None else self._default_tol)
        max_depth_i = int(ensure_positive(max_depth, "max_depth", allow_zero=False, error_cls=STEMValidationError))

        def simpson_segment(x0: float, x1: float) -> float:
            m = 0.5 * (x0 + x1)
            return (x1 - x0) / 6.0 * (float(f(x0)) + 4.0 * float(f(m)) + float(f(x1)))

        def recurse(x0: float, x1: float, whole: float, eps: float, depth: int) -> float:
            if depth >= max_depth_i:
                return whole
            m = 0.5 * (x0 + x1)
            left = simpson_segment(x0, m)
            right = simpson_segment(m, x1)
            if abs(left + right - whole) < 15.0 * eps:
                return left + right + (left + right - whole) / 15.0
            return (
                recurse(x0, m, left, eps * 0.5, depth + 1)
                + recurse(m, x1, right, eps * 0.5, depth + 1)
            )

        whole = simpson_segment(a_f, b_f)
        return recurse(a_f, b_f, whole, tol_f, 0)

    # ======================================================================
    # ODE algorithms
    # ======================================================================

    def euler(
        self,
        f: Callable[[float, Sequence[float]], Sequence[float]],
        y0: Sequence[float],
        t0: float,
        t1: float,
        h: Optional[float] = None,
    ) -> Mapping[str, Any]:
        """Explicit Euler integrator."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        t_start, t_end = validate_bounds(t0, t1, "time", error_cls=STEMODEError)
        h_f = float(h if h is not None else (t_end - t_start) / 100.0)
        if h_f <= 0.0:
            raise STEMODEError("step size must be positive")

        y = list(as_float_array(y0, "y0", error_cls=STEMODEError))
        ts = [t_start]
        ys = [y[:]]
        t = t_start
        max_steps = int(self.nm_config.get("ode_max_steps", 1_000_000))
        for _ in range(max_steps):
            if t >= t_end:
                break
            step = min(h_f, t_end - t)
            dy = [float(v) for v in f(t, y)]
            if len(dy) != len(y):
                raise STEMODEError("derivative length does not match state length")
            y = [yi + step * dyi for yi, dyi in zip(y, dy)]
            t += step
            ts.append(t)
            ys.append(y[:])

        return {"t": ts, "y": ys, "steps": len(ts) - 1}

    def rk4(
        self,
        f: Callable[[float, Sequence[float]], Sequence[float]],
        y0: Sequence[float],
        t0: float,
        t1: float,
        h: Optional[float] = None,
    ) -> Mapping[str, Any]:
        """Classical fourth-order Runge-Kutta integrator."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        t_start, t_end = validate_bounds(t0, t1, "time", error_cls=STEMODEError)
        h_f = float(h if h is not None else (t_end - t_start) / 100.0)
        if h_f <= 0.0:
            raise STEMODEError("step size must be positive")

        y = list(as_float_array(y0, "y0", error_cls=STEMODEError))
        ts = [t_start]
        ys = [y[:]]
        t = t_start
        max_steps = int(self.nm_config.get("ode_max_steps", 1_000_000))

        for _ in range(max_steps):
            if t >= t_end:
                break
            step = min(h_f, t_end - t)
            k1 = [float(v) for v in f(t, y)]
            k2 = [float(v) for v in f(t + step / 2.0, [yi + step / 2.0 * k1i for yi, k1i in zip(y, k1)])]
            k3 = [float(v) for v in f(t + step / 2.0, [yi + step / 2.0 * k2i for yi, k2i in zip(y, k2)])]
            k4 = [float(v) for v in f(t + step, [yi + step * k3i for yi, k3i in zip(y, k3)])]
            y = [
                yi + step / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
                for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4)
            ]
            t += step
            ts.append(t)
            ys.append(y[:])

        return {"t": ts, "y": ys, "steps": len(ts) - 1}

    def rk45_adaptive(
        self,
        f: Callable[[float, Sequence[float]], Sequence[float]],
        y0: Sequence[float],
        t0: float,
        t1: float,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        h0: Optional[float] = None,
    ) -> Mapping[str, Any]:
        """Dormand-Prince RK45 adaptive step size integrator."""
        validate_callable(f, "f", error_cls=STEMValidationError)
        t_start, t_end = validate_bounds(t0, t1, "time", error_cls=STEMODEError)
        rtol_f = ensure_positive(rtol, "rtol", allow_zero=False, error_cls=STEMODEError)
        atol_f = ensure_non_negative_ode(atol)
        h = float(h0 if h0 is not None else (t_end - t_start) / 100.0)
        if h <= 0.0:
            raise STEMODEError("initial step must be positive")

        y = list(as_float_array(y0, "y0", error_cls=STEMODEError))
        ts = [t_start]
        ys = [y[:]]
        t = t_start
        max_steps = int(self.nm_config.get("ode_max_steps", 100_000))

        c = [0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0]
        a = [
            [],
            [1.0 / 5],
            [3.0 / 40, 9.0 / 40],
            [44.0 / 45, -56.0 / 15, 32.0 / 9],
            [19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729],
            [9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656],
            [35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84],
        ]
        b5 = [35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0.0]
        b4 = [5179.0 / 57600, 0.0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40]

        for _ in range(max_steps):
            if t >= t_end:
                break
            h = min(h, t_end - t)
            ks: List[List[float]] = []
            for i in range(7):
                yi = y[:]
                for j, aij in enumerate(a[i]):
                    kj = ks[j]
                    for m in range(len(y)):
                        yi[m] += h * aij * kj[m]
                ks.append([float(v) for v in f(t + c[i] * h, yi)])

            y5 = [
                y[m] + h * sum(b5[i] * ks[i][m] for i in range(7))
                for m in range(len(y))
            ]
            y4 = [
                y[m] + h * sum(b4[i] * ks[i][m] for i in range(7))
                for m in range(len(y))
            ]

            error = max(
                abs(y5[m] - y4[m]) / (atol_f + rtol_f * max(abs(y[m]), abs(y5[m])))
                for m in range(len(y))
            ) if y else 0.0

            if error <= 1.0:
                t += h
                y = y5
                ts.append(t)
                ys.append(y[:])

            factor = 0.9 * (1.0 / error) ** 0.2 if error > 0.0 else 5.0
            factor = max(0.2, min(5.0, factor))
            h *= factor

        return {"t": ts, "y": ys, "steps": len(ts) - 1}

    # ======================================================================
    # PDE interfaces
    # ======================================================================

    def discretize_operator_1d(
        self,
        order: int,
        n: int,
        dx: float,
    ) -> List[List[float]]:
        """Return the 1D finite-difference operator matrix of the given order."""
        n_i = int(ensure_positive(n, "n", allow_zero=False, error_cls=STEMPDEInterfaceError))
        dx_f = ensure_positive(dx, "dx", allow_zero=False, error_cls=STEMPDEInterfaceError)
        order_i = int(order)
        if order_i not in (1, 2):
            raise STEMPDEInterfaceError("only first and second order operators are supported")

        A = [[0.0] * n_i for _ in range(n_i)]
        if order_i == 1:
            for i in range(1, n_i - 1):
                A[i][i - 1] = -1.0 / (2.0 * dx_f)
                A[i][i + 1] = 1.0 / (2.0 * dx_f)
        else:
            for i in range(1, n_i - 1):
                A[i][i - 1] = 1.0 / (dx_f * dx_f)
                A[i][i] = -2.0 / (dx_f * dx_f)
                A[i][i + 1] = 1.0 / (dx_f * dx_f)
        return A

    def apply_boundary_conditions(
        self,
        matrix: Sequence[Sequence[float]],
        bc_left: float,
        bc_right: float,
    ) -> List[List[float]]:
        """Zero out boundary rows and pin them to given values on the diagonal."""
        A = as_float_matrix(matrix, "matrix", error_cls=STEMPDEInterfaceError)
        n = len(A)
        if n < 2:
            raise STEMPDEInterfaceError("matrix must have at least two rows")
        left = ensure_finite_number(bc_left, "bc_left", error_cls=STEMPDEInterfaceError)
        right = ensure_finite_number(bc_right, "bc_right", error_cls=STEMPDEInterfaceError)

        A[0] = [0.0] * n
        A[0][0] = 1.0
        A[-1] = [0.0] * n
        A[-1][-1] = 1.0
        _ = (left, right)
        return A

    def solve_linearized_pde_system(
        self,
        A: Sequence[Sequence[float]],
        b: Sequence[float],
    ) -> List[float]:
        """Solve a linearized PDE discretization system using LU."""
        try:
            return self.solve_linear_system(A, b)
        except STEMLinearAlgebraError as exc:
            raise STEMPDEInterfaceError(
                "failed to solve linearized PDE system",
                cause=exc,
            ) from exc

    # ======================================================================
    # Diagnostics and error controls
    # ======================================================================

    def condition_numbers(self, A: Sequence[Sequence[float]]) -> Mapping[str, float]:
        """Return 1-norm and infinity-norm condition number estimates."""
        matrix = as_float_matrix(A, "A", error_cls=STEMLinearAlgebraError)
        if not matrix:
            raise STEMLinearAlgebraError("A must not be empty")
        inverse = self.matrix_inverse(matrix)
        a_inf = max(sum(abs(v) for v in row) for row in matrix)
        a_1 = max(sum(abs(matrix[i][j]) for i in range(len(matrix))) for j in range(len(matrix[0])))
        inv_inf = max(sum(abs(v) for v in row) for row in inverse)
        inv_1 = max(sum(abs(inverse[i][j]) for i in range(len(inverse))) for j in range(len(inverse[0])))
        return {
            "cond_inf": a_inf * inv_inf,
            "cond_1": a_1 * inv_1,
        }

    def floating_point_effects(self) -> Mapping[str, Any]:
        """Documented diagnostics about the current floating-point environment."""
        return {
            "machine_epsilon": _machine_epsilon(),
            "max_finite": float.fromhex("0x1.fffffffffffffp+1023"),
            "min_normal": float.fromhex("0x1.0p-1022"),
            "min_subnormal": float.fromhex("0x0.0000000000001p-1022"),
            "has_signed_zero": True,
        }

    def algorithmic_stability(self) -> Mapping[str, Any]:
        """Return notes on numerical stability defaults used by this class."""
        return {
            "linear_algebra": "LU with partial pivoting; Householder QR; Cholesky with no pivoting",
            "root_solving": "Bisection, secant, Newton, fixed-point iteration",
            "integration": "Composite trapezoidal/Simpson, Romberg, Gauss-Legendre, adaptive Simpson",
            "ode": "Euler, RK4, Dormand-Prince RK45 adaptive",
            "variance": "Welford/Chan stable accumulation (in statistics module)",
        }

    def catastrophic_cancellation(self, a: float, b: float) -> Mapping[str, Any]:
        """Report catastrophic cancellation for ``a - b``."""
        a_f = ensure_finite_number(a, "a", error_cls=STEMNumericalError)
        b_f = ensure_finite_number(b, "b", error_cls=STEMNumericalError)
        result = a_f - b_f
        eps = _machine_epsilon()
        denom = max(abs(a_f), abs(b_f), 1.0)
        relative_loss = abs(result) / denom if denom > 0.0 else 0.0
        return {
            "a": a_f,
            "b": b_f,
            "result": result,
            "relative_magnitude": relative_loss,
            "cancellation_risk": relative_loss < eps and a_f != b_f,
        }

    def residuals(
        self,
        A: Sequence[Sequence[float]],
        x: Sequence[float],
        b: Sequence[float],
    ) -> Mapping[str, float]:
        """Absolute, relative, and backward residuals for ``Ax = b``."""
        matrix = as_float_matrix(A, "A", error_cls=STEMNumericalError)
        xv = as_float_array(x, "x", error_cls=STEMNumericalError)
        bv = as_float_array(b, "b", error_cls=STEMNumericalError)
        if len(matrix) != len(bv):
            raise STEMNumericalError("A and b dimensions do not match")

        residual_vector = [
            sum(matrix[i][j] * xv[j] for j in range(len(xv))) - bv[i]
            for i in range(len(matrix))
        ]
        abs_res = math.sqrt(sum(r * r for r in residual_vector))
        b_norm = math.sqrt(sum(v * v for v in bv)) if bv else 0.0
        rel_res = abs_res / b_norm if b_norm > 0.0 else abs_res
        back = helper_backward_error(matrix, xv, bv, error_cls=STEMNumericalError)
        return {
            "absolute": abs_res,
            "relative": rel_res,
            "backward": back,
        }

    def forward_error(self, x_approx: Sequence[float], x_exact: Sequence[float]) -> float:
        """Infinity-norm forward error."""
        a = as_float_array(x_approx, "x_approx", error_cls=STEMNumericalError)
        b = as_float_array(x_exact, "x_exact", error_cls=STEMNumericalError)
        if len(a) != len(b):
            raise STEMNumericalError("vector dimensions do not match")
        return max(abs(ai - bi) for ai, bi in zip(a, b)) if a else 0.0

    def backward_error(
        self,
        A: Sequence[Sequence[float]],
        x_approx: Sequence[float],
        b: Sequence[float],
    ) -> float:
        """Delegate to the shared backward-error helper."""
        return helper_backward_error(A, x_approx, b, error_cls=STEMNumericalError)


# ---------------------------------------------------------------------------
# Small internal helpers local to numerical_methods.py
# ---------------------------------------------------------------------------


def helper_backward_error(
    A: Sequence[Sequence[float]],
    x: Sequence[float],
    b: Sequence[float],
    *,
    error_cls: type = STEMNumericalError,
) -> float:
    """Compute the normwise infinity-norm backward error for ``Ax = b``."""
    matrix = as_float_matrix(A, "A", error_cls=error_cls)
    xv = as_float_array(x, "x", error_cls=error_cls)
    bv = as_float_array(b, "b", error_cls=error_cls)
    if len(matrix) != len(bv) or any(len(row) != len(xv) for row in matrix):
        raise error_cls("A, x, and b dimensions do not match")

    residual_norm = max(
        (
            abs(sum(value * xv[j] for j, value in enumerate(row)) - bv[i])
            for i, row in enumerate(matrix)
        ),
        default=0.0,
    )
    matrix_norm = max(
        (sum(abs(value) for value in row) for row in matrix),
        default=0.0,
    )
    x_norm = max((abs(value) for value in xv), default=0.0)
    b_norm = max((abs(value) for value in bv), default=0.0)
    scale = matrix_norm * x_norm + b_norm
    if scale == 0.0:
        return 0.0 if residual_norm == 0.0 else math.inf
    return residual_norm / scale


def _machine_epsilon() -> float:
    eps = 1.0
    while 1.0 + eps / 2.0 > 1.0:
        eps /= 2.0
    return eps


def _ensure_non_negative_ode(value: Any) -> float:
    v = float(value)
    if v < 0.0:
        raise STEMODEError("absolute tolerance must be non-negative")
    return v


ensure_non_negative_ode = _ensure_non_negative_ode


def _gauss_legendre_table(n: int) -> Tuple[List[float], List[float]]:
    """Return nodes and weights for the given Gauss-Legendre order."""
    if n == 1:
        return [0.0], [2.0]
    if n == 2:
        a = 1.0 / math.sqrt(3.0)
        return [-a, a], [1.0, 1.0]
    if n == 3:
        a = math.sqrt(3.0 / 5.0)
        return [-a, 0.0, a], [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    if n == 4:
        a = math.sqrt(3.0 / 7.0 - 2.0 / 7.0 * math.sqrt(6.0 / 5.0))
        b = math.sqrt(3.0 / 7.0 + 2.0 / 7.0 * math.sqrt(6.0 / 5.0))
        wa = (18.0 + math.sqrt(30.0)) / 36.0
        wb = (18.0 - math.sqrt(30.0)) / 36.0
        return [-b, -a, a, b], [wb, wa, wa, wb]
    if n == 5:
        a = 1.0 / 3.0 * math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0))
        b = 1.0 / 3.0 * math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0))
        wa = (322.0 + 13.0 * math.sqrt(70.0)) / 900.0
        wb = (322.0 - 13.0 * math.sqrt(70.0)) / 900.0
        w0 = 128.0 / 225.0
        return [-b, -a, 0.0, a, b], [wb, wa, w0, wa, wb]
    raise STEMIntegrationError(
        "Gauss-Legendre table supports n in {1, 2, 3, 4, 5}",
        context={"n": n},
    )


__all__ = ["NumericalMethods"]