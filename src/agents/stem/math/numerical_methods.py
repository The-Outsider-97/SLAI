"""Floating-point numerical methods for SLAI STEM.

Grounding: Higham; Trefethen & Bau; Quarteroni, Sacco & Saleri; Hairer,
Nørsett & Wanner; LeVeque; Davis & Rabinowitz. The module owns numerical
kernels, not scenario simulation or generic optimization.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import sys
import time
import numpy as np # type: ignore

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..stem_types import ConvergenceStatus, SolverResult
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import *
from ..utils.stem_helpers import (
    as_float_array, as_float_matrix, backward_error as _backward_error_core,
    check_positive_definite, check_square_matrix, check_symmetric_matrix,
    condition_number_estimate, ensure_finite_number, ensure_non_negative,
    ensure_positive, validate_bounds, validate_callable,
)
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Numerical Methods")
printer = PrettyPrinter()


class NumericalMethods:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.num_config = dict(get_config_section("numerical_methods", config=self.config) or {})
        if config:
            self.num_config.update(dict(config))
        self.default_abs_tol = float(self.num_config.get("absolute_tolerance", 1e-12))
        self.default_rel_tol = float(self.num_config.get("relative_tolerance", 1e-9))
        self.default_max_iter = int(self.num_config.get("max_iterations", 200))
        self.condition_warning = float(self.num_config.get("condition_warning", 1e12))
        self.ode_step = float(self.num_config.get("ode_step", 0.01))
        ensure_non_negative(self.default_abs_tol, "absolute_tolerance")
        ensure_non_negative(self.default_rel_tol, "relative_tolerance")
        if self.default_max_iter < 1:
            raise STEMValidationError("max_iterations must be >= 1")

    # ------------------------------------------------------------------
    # Numerical linear algebra
    # ------------------------------------------------------------------
    def lu_decomposition(self, A: Sequence[Sequence[float]]) -> Mapping[str, Any]:
        matrix = check_square_matrix(A)
        n = len(matrix)
        U = [row[:] for row in matrix]
        L = [[0.0] * n for _ in range(n)]
        P = list(range(n))
        for k in range(n):
            pivot = max(range(k, n), key=lambda i: abs(U[i][k]))
            if abs(U[pivot][k]) <= self.default_abs_tol:
                raise STEMSingularSystemError("Matrix is singular to working precision", context={"pivot": k})
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
        return {"L": L, "U": U, "permutation": P, "condition_estimate": condition_number_estimate(matrix)}

    def qr_decomposition(self, A: Sequence[Sequence[float]]) -> Mapping[str, Any]:
        matrix = np.asarray(as_float_matrix(A, "A", error_cls=STEMLinearAlgebraError), dtype=float)
        try:
            Q, R = np.linalg.qr(matrix, mode="reduced")
        except np.linalg.LinAlgError as exc:
            raise STEMLinearAlgebraError("QR decomposition failed", cause=exc) from exc
        residual = float(np.linalg.norm(matrix - Q @ R, ord=np.inf))
        return {"Q": Q.tolist(), "R": R.tolist(), "residual": residual}

    def cholesky(self, A: Sequence[Sequence[float]]) -> List[List[float]]:
        matrix = np.asarray(check_positive_definite(A), dtype=float)
        try:
            return np.linalg.cholesky(matrix).tolist()
        except np.linalg.LinAlgError as exc:
            raise STEMLinearAlgebraError("Cholesky decomposition failed", cause=exc) from exc

    def _forward_substitution(self, L: List[List[float]], b: List[float]) -> List[float]:
        n = len(L); y = [0.0] * n
        for i in range(n):
            diag = L[i][i]
            if abs(diag) <= self.default_abs_tol:
                raise STEMSingularSystemError("Zero diagonal in forward substitution", context={"row": i})
            y[i] = (b[i] - math.fsum(L[i][j] * y[j] for j in range(i))) / diag
        return y

    def _backward_substitution(self, U: List[List[float]], y: List[float]) -> List[float]:
        n = len(U); x = [0.0] * n
        for i in range(n - 1, -1, -1):
            diag = U[i][i]
            if abs(diag) <= self.default_abs_tol:
                raise STEMSingularSystemError("Zero diagonal in backward substitution", context={"row": i})
            x[i] = (y[i] - math.fsum(U[i][j] * x[j] for j in range(i + 1, n))) / diag
        return x

    def solve_linear_system(self, A: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
        matrix = np.asarray(check_square_matrix(A), dtype=float)
        rhs = np.asarray(as_float_array(b, "b", error_cls=STEMLinearAlgebraError), dtype=float)
        if rhs.shape != (matrix.shape[0],):
            raise STEMLinearAlgebraError("Right-hand side dimension mismatch")
        cond = float(np.linalg.cond(matrix))
        if not math.isfinite(cond):
            raise STEMSingularSystemError("Matrix is singular")
        try:
            x = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise STEMSingularSystemError("Linear solve failed", cause=exc) from exc
        residual = float(np.linalg.norm(matrix @ x - rhs, ord=np.inf))
        if cond >= self.condition_warning:
            logger.warning("Ill-conditioned linear system | cond=%g | residual=%g", cond, residual)
        return [float(v) for v in x]

    def least_squares(self, A: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
        matrix = np.asarray(as_float_matrix(A, "A", error_cls=STEMLinearAlgebraError), dtype=float)
        rhs = np.asarray(as_float_array(b, "b", error_cls=STEMLinearAlgebraError), dtype=float)
        if matrix.shape[0] != rhs.shape[0]:
            raise STEMLinearAlgebraError("Least-squares row count must match b")
        x, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        if rank < min(matrix.shape):
            logger.warning("Rank-deficient least-squares system | rank=%d", rank)
        return [float(v) for v in x]

    def eigenvalues_symmetric(self, A: Sequence[Sequence[float]], max_iter: int = 100, tol: float = 1e-12) -> List[float]:
        del max_iter, tol
        matrix = np.asarray(check_symmetric_matrix(A), dtype=float)
        try:
            vals = np.linalg.eigvalsh(matrix)
        except np.linalg.LinAlgError as exc:
            raise STEMLinearAlgebraError("Symmetric eigenvalue computation failed", cause=exc) from exc
        return [float(v) for v in vals]

    def singular_values(self, A: Sequence[Sequence[float]]) -> List[float]:
        matrix = np.asarray(as_float_matrix(A, "A", error_cls=STEMLinearAlgebraError), dtype=float)
        try:
            vals = np.linalg.svd(matrix, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            raise STEMLinearAlgebraError("SVD failed", cause=exc) from exc
        return [float(v) for v in vals]

    def matrix_inverse(self, A: Sequence[Sequence[float]]) -> List[List[float]]:
        matrix = np.asarray(check_square_matrix(A), dtype=float)
        cond = float(np.linalg.cond(matrix))
        if not math.isfinite(cond):
            raise STEMSingularSystemError("Matrix is singular")
        try:
            inv = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            raise STEMSingularSystemError("Matrix inversion failed", cause=exc) from exc
        if cond >= self.condition_warning:
            logger.warning("Inverse of ill-conditioned matrix requested | cond=%g", cond)
        return inv.tolist()

    # ------------------------------------------------------------------
    # Root finding
    # ------------------------------------------------------------------
    def _root_result(self, solution: float, residual: float, iterations: int, converged: bool, status: ConvergenceStatus, started: float, method: str) -> SolverResult:
        return SolverResult(solution, abs(residual), iterations, converged, status, method, {"method": method}, time.perf_counter() - started)

    def bisection(self, f: Callable[[float], float], a: float, b: float, *, tol: Optional[float] = None, max_iter: Optional[int] = None) -> SolverResult:
        validate_callable(f, "f"); left, right = validate_bounds(a, b); epsilon = self.default_abs_tol if tol is None else ensure_positive(tol, "tol")
        limit = self.default_max_iter if max_iter is None else int(max_iter); start = time.perf_counter()
        fl, fr = ensure_finite_number(f(left), "f(a)"), ensure_finite_number(f(right), "f(b)")
        if fl == 0.0: return self._root_result(left, fl, 0, True, ConvergenceStatus.CONVERGED, start, "bisection")
        if fr == 0.0: return self._root_result(right, fr, 0, True, ConvergenceStatus.CONVERGED, start, "bisection")
        if fl * fr > 0.0: raise STEMNumericalError("Bisection requires a sign-changing bracket")
        mid = left
        for iteration in range(1, limit + 1):
            mid = left + 0.5 * (right - left); fm = ensure_finite_number(f(mid), "f(mid)")
            if abs(fm) <= epsilon or 0.5 * abs(right - left) <= epsilon:
                return self._root_result(mid, fm, iteration, True, ConvergenceStatus.CONVERGED, start, "bisection")
            if fl * fm <= 0.0: right, fr = mid, fm
            else: left, fl = mid, fm
        fm = ensure_finite_number(f(mid), "f(mid)")
        return self._root_result(mid, fm, limit, False, ConvergenceStatus.MAX_ITERATIONS, start, "bisection")

    def newton_raphson(self, f: Callable[[float], float], df: Callable[[float], float], x0: float, *, tol: Optional[float] = None, max_iter: Optional[int] = None) -> SolverResult:
        validate_callable(f, "f"); validate_callable(df, "df"); x = ensure_finite_number(x0, "x0"); epsilon = self.default_abs_tol if tol is None else ensure_positive(tol, "tol")
        limit = self.default_max_iter if max_iter is None else int(max_iter); start = time.perf_counter()
        for iteration in range(1, limit + 1):
            fx = ensure_finite_number(f(x), "f(x)"); dfx = ensure_finite_number(df(x), "df(x)")
            if abs(fx) <= epsilon: return self._root_result(x, fx, iteration - 1, True, ConvergenceStatus.CONVERGED, start, "newton_raphson")
            if abs(dfx) <= max(epsilon, sys.float_info.epsilon):
                return self._root_result(x, fx, iteration, False, ConvergenceStatus.STALLED, start, "newton_raphson")
            step = fx / dfx; candidate = x - step
            if not math.isfinite(candidate): raise STEMNumericalError("Newton step produced a non-finite iterate")
            x = candidate
            if abs(step) <= epsilon * max(1.0, abs(x)):
                fx = ensure_finite_number(f(x), "f(x)")
                return self._root_result(x, fx, iteration, True, ConvergenceStatus.CONVERGED, start, "newton_raphson")
        fx = ensure_finite_number(f(x), "f(x)")
        return self._root_result(x, fx, limit, False, ConvergenceStatus.MAX_ITERATIONS, start, "newton_raphson")

    def secant(self, f: Callable[[float], float], x0: float, x1: float, *, tol: Optional[float] = None, max_iter: Optional[int] = None) -> SolverResult:
        validate_callable(f, "f"); a, b = ensure_finite_number(x0, "x0"), ensure_finite_number(x1, "x1")
        epsilon = self.default_abs_tol if tol is None else ensure_positive(tol, "tol"); limit = self.default_max_iter if max_iter is None else int(max_iter); start=time.perf_counter()
        fa, fb = ensure_finite_number(f(a), "f(x0)"), ensure_finite_number(f(b), "f(x1)")
        for iteration in range(1, limit + 1):
            denom = fb - fa
            if abs(denom) <= sys.float_info.epsilon:
                return self._root_result(b, fb, iteration, False, ConvergenceStatus.STALLED, start, "secant")
            c = b - fb * (b - a) / denom
            fc = ensure_finite_number(f(c), "f(x)")
            if abs(fc) <= epsilon or abs(c - b) <= epsilon * max(1.0, abs(c)):
                return self._root_result(c, fc, iteration, True, ConvergenceStatus.CONVERGED, start, "secant")
            a, fa, b, fb = b, fb, c, fc
        return self._root_result(b, fb, limit, False, ConvergenceStatus.MAX_ITERATIONS, start, "secant")

    def fixed_point(self, g: Callable[[float], float], x0: float, *, tol: Optional[float] = None, max_iter: Optional[int] = None) -> SolverResult:
        validate_callable(g, "g"); x=ensure_finite_number(x0,"x0"); epsilon=self.default_abs_tol if tol is None else ensure_positive(tol,"tol"); limit=self.default_max_iter if max_iter is None else int(max_iter); start=time.perf_counter()
        for iteration in range(1,limit+1):
            candidate=ensure_finite_number(g(x),"g(x)")
            residual=candidate-x
            if abs(residual)<=epsilon*max(1.0,abs(candidate)):
                return self._root_result(candidate,residual,iteration,True,ConvergenceStatus.CONVERGED,start,"fixed_point")
            x=candidate
        return self._root_result(x,ensure_finite_number(g(x),"g(x)")-x,limit,False,ConvergenceStatus.MAX_ITERATIONS,start,"fixed_point")

    # ------------------------------------------------------------------
    # Interpolation and approximation
    # ------------------------------------------------------------------
    def _xy(self, xs: Sequence[float], ys: Sequence[float]) -> Tuple[List[float], List[float]]:
        x, y = as_float_array(xs, "xs", error_cls=STEMInterpolationError), as_float_array(ys, "ys", error_cls=STEMInterpolationError)
        if len(x) != len(y) or len(x) < 2: raise STEMInterpolationError("xs and ys must have equal length >= 2")
        if len(set(x)) != len(x): raise STEMInterpolationError("Interpolation abscissae must be unique")
        pairs=sorted(zip(x,y)); return [p[0] for p in pairs],[p[1] for p in pairs]

    def linear_interpolation(self, xs: Sequence[float], ys: Sequence[float]) -> Callable[[float], float]:
        x,y=self._xy(xs,ys)
        def interp(query: float)->float:
            q=ensure_finite_number(query,"query",error_cls=STEMInterpolationError)
            if q < x[0] or q > x[-1]: raise STEMInterpolationError("Query outside interpolation domain")
            i=max(0,min(len(x)-2,int(np.searchsorted(x,q)-1)))
            t=(q-x[i])/(x[i+1]-x[i]); return y[i]+t*(y[i+1]-y[i])
        return interp

    def polynomial_interpolation(self, xs: Sequence[float], ys: Sequence[float]) -> Callable[[float], float]:
        x,y=self._xy(xs,ys); n=len(x)
        weights=[]
        for j in range(n):
            denom=1.0
            for k in range(n):
                if k!=j: denom*=x[j]-x[k]
            weights.append(1.0/denom)
        def interp(query: float)->float:
            q=ensure_finite_number(query,"query",error_cls=STEMInterpolationError)
            for xi,yi in zip(x,y):
                if q==xi:return yi
            num=math.fsum(w*yi/(q-xi) for xi,yi,w in zip(x,y,weights)); den=math.fsum(w/(q-xi) for xi,w in zip(x,weights))
            return num/den
        return interp

    def cubic_spline(self, xs: Sequence[float], ys: Sequence[float]) -> Callable[[float], float]:
        x,y=self._xy(xs,ys); n=len(x)
        h=[x[i+1]-x[i] for i in range(n-1)]
        A=np.zeros((n,n),dtype=float); rhs=np.zeros(n,dtype=float); A[0,0]=A[-1,-1]=1.0
        for i in range(1,n-1):
            A[i,i-1]=h[i-1]; A[i,i]=2*(h[i-1]+h[i]); A[i,i+1]=h[i]
            rhs[i]=6*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
        m=np.linalg.solve(A,rhs)
        def interp(query: float)->float:
            q=ensure_finite_number(query,"query",error_cls=STEMInterpolationError)
            if q < x[0] or q > x[-1]: raise STEMInterpolationError("Query outside spline domain")
            i=max(0,min(n-2,int(np.searchsorted(x,q)-1))); hi=h[i]
            a=(x[i+1]-q)/hi; b=(q-x[i])/hi
            return float(a*y[i]+b*y[i+1]+((a**3-a)*m[i]+(b**3-b)*m[i+1])*(hi**2)/6.0)
        return interp

    def chebyshev_approximation(self, f: Callable[[float], float], a: float, b: float, degree: int) -> Callable[[float], float]:
        validate_callable(f,"f"); left,right=validate_bounds(a,b,error_cls=STEMInterpolationError)
        if degree < 0: raise STEMInterpolationError("degree must be non-negative")
        nodes=np.cos((2*np.arange(degree+1)+1)*math.pi/(2*(degree+1)))
        mapped=0.5*(left+right)+0.5*(right-left)*nodes
        values=np.asarray([ensure_finite_number(f(float(z)),"f(x)",error_cls=STEMInterpolationError) for z in mapped])
        coeff=np.polynomial.chebyshev.chebfit(nodes,values,degree)
        def approx(query: float)->float:
            q=ensure_finite_number(query,"query",error_cls=STEMInterpolationError); z=(2*q-left-right)/(right-left)
            return float(np.polynomial.chebyshev.chebval(z,coeff))
        return approx

    # ------------------------------------------------------------------
    # Numerical differentiation
    # ------------------------------------------------------------------
    def _step(self, x: float, h: Optional[float]) -> float:
        if h is not None:return ensure_positive(h,"h",error_cls=STEMDifferentiationError)
        return math.sqrt(sys.float_info.epsilon)*max(1.0,abs(x))

    def forward_difference(self, f: Callable[[float], float], x: float, h: Optional[float] = None) -> float:
        validate_callable(f,"f"); x0=ensure_finite_number(x,"x"); step=self._step(x0,h)
        return (ensure_finite_number(f(x0+step),"f(x+h)")-ensure_finite_number(f(x0),"f(x)"))/step

    def central_difference(self, f: Callable[[float], float], x: float, h: Optional[float] = None) -> float:
        validate_callable(f,"f"); x0=ensure_finite_number(x,"x"); step=(sys.float_info.epsilon**(1/3))*max(1.0,abs(x0)) if h is None else ensure_positive(h,"h",error_cls=STEMDifferentiationError)
        return (ensure_finite_number(f(x0+step),"f(x+h)")-ensure_finite_number(f(x0-step),"f(x-h)"))/(2*step)

    def richardson_extrapolation(self, f: Callable[[float], float], x: float, h: Optional[float] = None, levels: int = 4) -> float:
        if levels < 1: raise STEMDifferentiationError("levels must be >= 1")
        x0=ensure_finite_number(x,"x"); step=1e-2*max(1.0,abs(x0)) if h is None else ensure_positive(h,"h",error_cls=STEMDifferentiationError)
        table=[]
        for k in range(levels):
            hk=step/(2**k); table.append([(f(x0+hk)-f(x0-hk))/(2*hk)])
            for j in range(1,k+1): table[k].append(table[k][j-1]+(table[k][j-1]-table[k-1][j-1])/(4**j-1))
        result=float(table[-1][-1])
        if not math.isfinite(result): raise STEMDifferentiationError("Richardson extrapolation produced non-finite result")
        return result

    # ------------------------------------------------------------------
    # Quadrature
    # ------------------------------------------------------------------
    def trapezoidal(self, f: Callable[[float], float], a: float, b: float, n: int = 100) -> float:
        validate_callable(f,"f"); left,right=validate_bounds(a,b,error_cls=STEMIntegrationError)
        if n < 1: raise STEMIntegrationError("n must be >= 1")
        h=(right-left)/n; values=[ensure_finite_number(f(left+i*h),"f(x)",error_cls=STEMIntegrationError) for i in range(n+1)]
        return h*(0.5*values[0]+math.fsum(values[1:-1])+0.5*values[-1])

    def simpson(self, f: Callable[[float], float], a: float, b: float, n: int = 100) -> float:
        validate_callable(f,"f"); left,right=validate_bounds(a,b,error_cls=STEMIntegrationError)
        if n < 2: raise STEMIntegrationError("n must be >= 2")
        if n % 2:n += 1
        h=(right-left)/n
        values=[ensure_finite_number(f(left+i*h),"f(x)",error_cls=STEMIntegrationError) for i in range(n+1)]
        return h/3*(values[0]+values[-1]+4*math.fsum(values[1:-1:2])+2*math.fsum(values[2:-1:2]))

    def romberg(self, f: Callable[[float], float], a: float, b: float, *, tol: Optional[float] = None, max_iter: int = 8) -> float:
        epsilon=self.default_abs_tol if tol is None else ensure_positive(tol,"tol",error_cls=STEMIntegrationError)
        if max_iter < 1: raise STEMIntegrationError("max_iter must be >= 1")
        left,right=validate_bounds(a,b,error_cls=STEMIntegrationError); R=[[0.5*(right-left)*(f(left)+f(right))]]
        for k in range(1,max_iter):
            h=(right-left)/(2**k); subtotal=math.fsum(f(left+(2*i-1)*h) for i in range(1,2**(k-1)+1))
            row=[0.5*R[k-1][0]+h*subtotal]
            for j in range(1,k+1):row.append(row[j-1]+(row[j-1]-R[k-1][j-1])/(4**j-1))
            R.append(row)
            if abs(R[k][k]-R[k-1][k-1])<=epsilon:return float(R[k][k])
        return float(R[-1][-1])

    def gauss_legendre(self, f: Callable[[float], float], a: float, b: float, n: int = 5) -> float:
        validate_callable(f,"f"); left,right=validate_bounds(a,b,error_cls=STEMIntegrationError)
        if n < 1 or n > 64: raise STEMIntegrationError("n must be in [1, 64]")
        nodes,weights=np.polynomial.legendre.leggauss(n); mid=0.5*(left+right); half=0.5*(right-left)
        return float(half*math.fsum(float(w)*ensure_finite_number(f(mid+half*float(x)),"f(x)",error_cls=STEMIntegrationError) for x,w in zip(nodes,weights)))

    def adaptive_quadrature(self, f: Callable[[float], float], a: float, b: float, *, tol: Optional[float] = None, max_depth: int = 20) -> float:
        validate_callable(f,"f"); left,right=validate_bounds(a,b,error_cls=STEMIntegrationError); epsilon=self.default_abs_tol if tol is None else ensure_positive(tol,"tol",error_cls=STEMIntegrationError)
        def simp(x0:float,x1:float)->float:
            m=0.5*(x0+x1); return (x1-x0)*(f(x0)+4*f(m)+f(x1))/6
        def rec(x0:float,x1:float,whole:float,eps:float,depth:int)->float:
            m=0.5*(x0+x1); l=simp(x0,m); r=simp(m,x1); delta=l+r-whole
            if depth<=0:return l+r+delta/15
            if abs(delta)<=15*eps:return l+r+delta/15
            return rec(x0,m,l,eps/2,depth-1)+rec(m,x1,r,eps/2,depth-1)
        result=float(rec(left,right,simp(left,right),epsilon,max_depth))
        if not math.isfinite(result):raise STEMIntegrationError("Adaptive quadrature produced non-finite result")
        return result

    # ------------------------------------------------------------------
    # ODE solver kernels
    # ------------------------------------------------------------------
    def _ode_setup(self, f: Callable[[float, Sequence[float]], Sequence[float]], y0: Sequence[float], t0: float, t1: float, h: Optional[float]) -> Tuple[List[float],float,float,float]:
        validate_callable(f,"f"); y=as_float_array(y0,"y0",error_cls=STEMODEError); start=ensure_finite_number(t0,"t0",error_cls=STEMODEError); end=ensure_finite_number(t1,"t1",error_cls=STEMODEError)
        if end <= start: raise STEMODEError("t1 must be greater than t0")
        step=self.ode_step if h is None else ensure_positive(h,"h",error_cls=STEMODEError)
        return y,start,end,step

    @staticmethod
    def _ode_eval(f: Callable[[float, Sequence[float]], Sequence[float]], t: float, y: Sequence[float], n: int) -> List[float]:
        out=as_float_array(f(t,list(y)),"f(t,y)",error_cls=STEMODEError)
        if len(out)!=n: raise STEMODEError("ODE derivative dimension mismatch")
        return out

    def euler(self, f: Callable[[float, Sequence[float]], Sequence[float]], y0: Sequence[float], t0: float, t1: float, h: Optional[float] = None) -> Mapping[str, Any]:
        y,t,end,step=self._ode_setup(f,y0,t0,t1,h); ts=[t]; ys=[y[:]]; n=len(y)
        while t < end:
            dt=min(step,end-t); k=self._ode_eval(f,t,y,n); y=[yi+dt*ki for yi,ki in zip(y,k)]; t+=dt; ts.append(t); ys.append(y[:])
        return {"t":ts,"y":ys,"method":"euler","steps":len(ts)-1,"converged":True}

    def rk4(self, f: Callable[[float, Sequence[float]], Sequence[float]], y0: Sequence[float], t0: float, t1: float, h: Optional[float] = None) -> Mapping[str, Any]:
        y,t,end,step=self._ode_setup(f,y0,t0,t1,h); ts=[t]; ys=[y[:]]; n=len(y)
        while t < end:
            dt=min(step,end-t); k1=self._ode_eval(f,t,y,n)
            y2=[yi+dt*k/2 for yi,k in zip(y,k1)]; k2=self._ode_eval(f,t+dt/2,y2,n)
            y3=[yi+dt*k/2 for yi,k in zip(y,k2)]; k3=self._ode_eval(f,t+dt/2,y3,n)
            y4=[yi+dt*k for yi,k in zip(y,k3)]; k4=self._ode_eval(f,t+dt,y4,n)
            y=[yi+dt*(a+2*b+2*c+d)/6 for yi,a,b,c,d in zip(y,k1,k2,k3,k4)]; t+=dt; ts.append(t); ys.append(y[:])
        return {"t":ts,"y":ys,"method":"rk4","steps":len(ts)-1,"converged":True}

    def rk45_adaptive(self, f: Callable[[float, Sequence[float]], Sequence[float]], y0: Sequence[float], t0: float, t1: float, *, rtol: float = 1e-6, atol: float = 1e-9, h0: Optional[float] = None) -> Mapping[str, Any]:
        y,t,end,h=self._ode_setup(f,y0,t0,t1,h0 or min(self.ode_step,(t1-t0)/10)); n=len(y); ts=[t]; ys=[y[:]]; accepted=0; rejected=0; attempts=0
        ensure_positive(rtol,"rtol",error_cls=STEMODEError); ensure_positive(atol,"atol",error_cls=STEMODEError)
        max_steps=int(self.num_config.get("ode_max_steps",100000))
        while t < end:
            attempts += 1
            if attempts > max_steps: raise STEMConvergenceError("RK45 exceeded maximum step attempts")
            h=min(h,end-t)
            k1=self._ode_eval(f,t,y,n)
            def combine(coeffs:Sequence[Tuple[float,List[float]]])->List[float]: return [y[i]+h*math.fsum(c*k[i] for c,k in coeffs) for i in range(n)]
            k2=self._ode_eval(f,t+h/5,combine([(1/5,k1)]),n)
            k3=self._ode_eval(f,t+3*h/10,combine([(3/40,k1),(9/40,k2)]),n)
            k4=self._ode_eval(f,t+4*h/5,combine([(44/45,k1),(-56/15,k2),(32/9,k3)]),n)
            k5=self._ode_eval(f,t+8*h/9,combine([(19372/6561,k1),(-25360/2187,k2),(64448/6561,k3),(-212/729,k4)]),n)
            k6=self._ode_eval(f,t+h,combine([(9017/3168,k1),(-355/33,k2),(46732/5247,k3),(49/176,k4),(-5103/18656,k5)]),n)
            y5=combine([(35/384,k1),(500/1113,k3),(125/192,k4),(-2187/6784,k5),(11/84,k6)])
            k7=self._ode_eval(f,t+h,y5,n)
            y4=combine([(5179/57600,k1),(7571/16695,k3),(393/640,k4),(-92097/339200,k5),(187/2100,k6),(1/40,k7)])
            err=max(abs(a-b)/(atol+rtol*max(abs(yi),abs(a))) for yi,a,b in zip(y,y5,y4))
            if err <= 1.0:
                t+=h; y=y5; ts.append(t); ys.append(y[:]); accepted+=1
            else: rejected+=1
            factor=5.0 if err==0.0 else min(5.0,max(0.2,0.9*err**(-0.2))); h*=factor
            if h <= sys.float_info.epsilon*max(1.0,abs(t)): raise STEMConvergenceError("RK45 step size underflow")
        return {"t":ts,"y":ys,"method":"rk45_dormand_prince","steps":accepted,"rejected_steps":rejected,"converged":True,"rtol":rtol,"atol":atol}

    # ------------------------------------------------------------------
    # PDE numerical interfaces
    # ------------------------------------------------------------------
    def discretize_operator_1d(self, order: int, n: int, dx: float) -> List[List[float]]:
        if n < 3: raise STEMPDEInterfaceError("n must be >= 3")
        spacing=ensure_positive(dx,"dx",error_cls=STEMPDEInterfaceError)
        A=np.zeros((n,n),dtype=float)
        if order==1:
            c=1/(2*spacing)
            for i in range(1,n-1): A[i,i-1],A[i,i+1]=-c,c
        elif order==2:
            c=1/(spacing*spacing)
            for i in range(1,n-1): A[i,i-1],A[i,i],A[i,i+1]=c,-2*c,c
        else: raise STEMPDEInterfaceError("Only first- and second-derivative 1D operators are supported")
        return A.tolist()

    def apply_boundary_conditions(self, matrix: Sequence[Sequence[float]], bc_left: float, bc_right: float) -> List[List[float]]:
        A=np.asarray(check_square_matrix(matrix,error_cls=STEMPDEInterfaceError),dtype=float).copy()
        ensure_finite_number(bc_left,"bc_left",error_cls=STEMPDEInterfaceError); ensure_finite_number(bc_right,"bc_right",error_cls=STEMPDEInterfaceError)
        A[0,:]=0.0;A[0,0]=1.0;A[-1,:]=0.0;A[-1,-1]=1.0
        return A.tolist()

    def solve_linearized_pde_system(self, A: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
        try:return self.solve_linear_system(A,b)
        except STEMLinearAlgebraError as exc: raise STEMPDEInterfaceError("PDE linearized system solve failed",cause=exc) from exc

    # ------------------------------------------------------------------
    # Numerical diagnostics
    # ------------------------------------------------------------------
    def condition_numbers(self, A: Sequence[Sequence[float]]) -> Mapping[str, float]:
        matrix=np.asarray(check_square_matrix(A),dtype=float)
        return {"2_norm":float(np.linalg.cond(matrix,2)),"1_norm":float(np.linalg.cond(matrix,1)),"inf_norm":float(np.linalg.cond(matrix,np.inf))}

    def floating_point_effects(self) -> Mapping[str, Any]:
        info=np.finfo(float)
        return {"epsilon":float(info.eps),"tiny":float(info.tiny),"max":float(info.max),"radix":2,"mantissa_bits":int(info.nmant+1),"ieee754_binary64":True}

    def algorithmic_stability(self) -> Mapping[str, Any]:
        return {"principle":"Prefer backward-stable formulations and report conditioning separately.","condition_warning":self.condition_warning,"absolute_tolerance":self.default_abs_tol,"relative_tolerance":self.default_rel_tol}

    def catastrophic_cancellation(self, a: float, b: float) -> Mapping[str, Any]:
        x,y=ensure_finite_number(a,"a"),ensure_finite_number(b,"b"); diff=x-b
        scale=max(abs(x),abs(y),sys.float_info.min); ratio=abs(diff)/scale
        return {"difference":diff,"relative_separation":ratio,"risk":"high" if ratio < math.sqrt(sys.float_info.epsilon) else "low"}

    def residuals(self, A: Sequence[Sequence[float]], x: Sequence[float], b: Sequence[float]) -> Mapping[str, float]:
        matrix=np.asarray(as_float_matrix(A,"A"),dtype=float); xv=np.asarray(as_float_array(x,"x"),dtype=float); rhs=np.asarray(as_float_array(b,"b"),dtype=float)
        if matrix.shape[1]!=xv.size or matrix.shape[0]!=rhs.size: raise STEMLinearAlgebraError("Residual dimension mismatch")
        r=matrix@xv-rhs
        return {"l1":float(np.linalg.norm(r,1)),"l2":float(np.linalg.norm(r,2)),"linf":float(np.linalg.norm(r,np.inf))}

    def forward_error(self, x_approx: Sequence[float], x_exact: Sequence[float]) -> float:
        a=np.asarray(as_float_array(x_approx,"x_approx"),dtype=float); e=np.asarray(as_float_array(x_exact,"x_exact"),dtype=float)
        if a.shape!=e.shape: raise STEMNumericalError("Forward-error vector shape mismatch")
        return float(np.linalg.norm(a-e,np.inf))

    def backward_error(self, A: Sequence[Sequence[float]], x_approx: Sequence[float], b: Sequence[float]) -> float:
        return helper_backward_error(A,x_approx,b)


def helper_backward_error(A: Sequence[Sequence[float]], x: Sequence[float], b: Sequence[float], *, error_cls: type = STEMNumericalError) -> float:
    return _backward_error_core(A, x, b, error_cls=error_cls)


def _machine_epsilon() -> float:
    return sys.float_info.epsilon


def _ensure_non_negative_ode(value: Any) -> float:
    return ensure_non_negative(value,"value",error_cls=STEMODEError)

ensure_non_negative_ode = _ensure_non_negative_ode


def _gauss_legendre_table(n: int) -> Tuple[List[float], List[float]]:
    if n < 1 or n > 64: raise STEMIntegrationError("n must be in [1, 64]")
    nodes,weights=np.polynomial.legendre.leggauss(n); return nodes.tolist(),weights.tolist()


__all__ = ["NumericalMethods", "helper_backward_error", "ensure_non_negative_ode"]
