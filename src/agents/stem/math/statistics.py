"""Numerically stable deterministic statistics for SLAI STEM.

Grounding: Monahan, *Numerical Methods of Statistics*; Chan, Golub & LeVeque
(1983) for stable sample variance. Statistical belief/inference is outside this
module; methods here compute defined numerical statistics.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import random
import numpy as np # type: ignore

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from scipy.special import betainc, gammainc, gammaln # type: ignore

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import STEMStatisticsError
from ..utils.stem_helpers import as_float_array, ensure_finite_number, ensure_non_negative, ensure_positive, stable_variance, validate_callable
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Statistics")
printer = PrettyPrinter()


@dataclass
class OnlineMoments:
    n: int = 0
    mean_value: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, x: float) -> None:
        value = ensure_finite_number(x, "x", error_cls=STEMStatisticsError)
        self.n += 1
        delta = value - self.mean_value
        self.mean_value += delta / self.n
        self.m2 += delta * (value - self.mean_value)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def variance(self) -> float:
        if self.n < 2:
            raise STEMStatisticsError("Sample variance requires at least two observations")
        return self.m2 / (self.n - 1)

    def population_variance(self) -> float:
        if self.n < 1:
            raise STEMStatisticsError("Population variance requires at least one observation")
        return self.m2 / self.n

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "mean": self.mean_value, "sample_variance": None if self.n < 2 else self.variance(), "population_variance": None if self.n < 1 else self.population_variance(), "minimum": None if self.n == 0 else self.minimum, "maximum": None if self.n == 0 else self.maximum}


@dataclass
class OnlineCovariance:
    n: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    c: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0

    def update(self, x: float, y: float) -> None:
        xv = ensure_finite_number(x, "x", error_cls=STEMStatisticsError)
        yv = ensure_finite_number(y, "y", error_cls=STEMStatisticsError)
        self.n += 1
        dx = xv - self.mean_x
        self.mean_x += dx / self.n
        dy = yv - self.mean_y
        self.mean_y += dy / self.n
        self.c += dx * (yv - self.mean_y)
        self.m2_x += dx * (xv - self.mean_x)
        self.m2_y += dy * (yv - self.mean_y)

    def covariance(self) -> float:
        if self.n < 2:
            raise STEMStatisticsError("Covariance requires at least two observations")
        return self.c / (self.n - 1)

    def correlation(self) -> float:
        if self.n < 2 or self.m2_x <= 0.0 or self.m2_y <= 0.0:
            raise STEMStatisticsError("Correlation requires non-zero variance")
        return self.c / math.sqrt(self.m2_x * self.m2_y)

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "mean_x": self.mean_x, "mean_y": self.mean_y, "covariance": None if self.n < 2 else self.covariance(), "correlation": None if self.n < 2 or self.m2_x <= 0 or self.m2_y <= 0 else self.correlation()}


class Statistics:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.statistics_config = dict(get_config_section("statistics", config=self.config) or {})
        if config:
            self.statistics_config.update(dict(config))
        self.default_ddof = int(self.statistics_config.get("ddof", 1))
        self.random_seed = int(self.statistics_config.get("random_seed", 0))

    def _values(self, xs: Sequence[float], minimum: int = 1) -> List[float]:
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        if len(values) < minimum:
            raise STEMStatisticsError(f"At least {minimum} observations are required")
        return values

    def mean(self, xs: Sequence[float]) -> float:
        values = self._values(xs)
        return math.fsum(values) / len(values)

    def variance(self, xs: Sequence[float], ddof: Optional[int] = None) -> float:
        return stable_variance(self._values(xs), self.default_ddof if ddof is None else int(ddof), error_cls=STEMStatisticsError)

    def standard_deviation(self, xs: Sequence[float], ddof: Optional[int] = None) -> float:
        return math.sqrt(self.variance(xs, ddof))

    def median(self, xs: Sequence[float]) -> float:
        values = sorted(self._values(xs)); n = len(values); m = n // 2
        return values[m] if n % 2 else 0.5 * (values[m - 1] + values[m])

    def quantile(self, xs: Sequence[float], q: float, method: str = "linear") -> float:
        values = sorted(self._values(xs)); qv = ensure_finite_number(q, "q", error_cls=STEMStatisticsError)
        if not 0.0 <= qv <= 1.0:
            raise STEMStatisticsError("q must lie in [0, 1]")
        if method not in {"linear", "lower", "higher", "nearest", "midpoint"}:
            raise STEMStatisticsError("Unsupported quantile method")
        pos = qv * (len(values) - 1); lo, hi = math.floor(pos), math.ceil(pos)
        if method == "lower": return values[lo]
        if method == "higher": return values[hi]
        if method == "nearest": return values[int(round(pos))]
        if method == "midpoint": return 0.5 * (values[lo] + values[hi])
        return values[lo] + (pos - lo) * (values[hi] - values[lo])

    def mode(self, xs: Sequence[float]) -> Any:
        values = self._values(xs); counts: Dict[float, int] = {}
        for value in values: counts[value] = counts.get(value, 0) + 1
        max_count = max(counts.values()); modes = sorted(k for k, v in counts.items() if v == max_count)
        return modes[0] if len(modes) == 1 else modes

    def raw_moment(self, xs: Sequence[float], k: int) -> float:
        if k < 0: raise STEMStatisticsError("moment order must be non-negative")
        values = self._values(xs); return math.fsum(v ** k for v in values) / len(values)

    def central_moment(self, xs: Sequence[float], k: int) -> float:
        if k < 0: raise STEMStatisticsError("moment order must be non-negative")
        values = self._values(xs); mean = math.fsum(values) / len(values)
        return math.fsum((v - mean) ** k for v in values) / len(values)

    def standardized_moment(self, xs: Sequence[float], k: int) -> float:
        sd = math.sqrt(self.central_moment(xs, 2))
        if sd == 0.0: raise STEMStatisticsError("Standardized moment undefined for zero variance")
        return self.central_moment(xs, k) / (sd ** k)

    def skewness(self, xs: Sequence[float]) -> float:
        return self.standardized_moment(xs, 3)

    def kurtosis(self, xs: Sequence[float], excess: bool = True) -> float:
        value = self.standardized_moment(xs, 4)
        return value - 3.0 if excess else value

    def covariance(self, xs: Sequence[float], ys: Sequence[float], ddof: Optional[int] = None) -> float:
        x, y = self._values(xs, 2), self._values(ys, 2)
        if len(x) != len(y): raise STEMStatisticsError("xs and ys length mismatch")
        correction = self.default_ddof if ddof is None else int(ddof)
        if correction < 0 or len(x) <= correction: raise STEMStatisticsError("Invalid ddof")
        mx, my = math.fsum(x)/len(x), math.fsum(y)/len(y)
        return math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/(len(x)-correction)

    def _pearson(self, xs: Sequence[float], ys: Sequence[float]) -> float:
        x, y = self._values(xs, 2), self._values(ys, 2)
        if len(x) != len(y): raise STEMStatisticsError("xs and ys length mismatch")
        sx, sy = self.standard_deviation(x), self.standard_deviation(y)
        if sx == 0.0 or sy == 0.0: raise STEMStatisticsError("Correlation undefined for zero variance")
        return self.covariance(x, y) / (sx * sy)

    def correlation(self, xs: Sequence[float], ys: Sequence[float], method: str = "pearson") -> float:
        method = method.lower()
        if method == "pearson": return self._pearson(xs, ys)
        if method == "spearman": return self.spearman_correlation(xs, ys)
        if method == "kendall": return self.kendall_tau(xs, ys)
        raise STEMStatisticsError("Unsupported correlation method")

    def spearman_correlation(self, xs: Sequence[float], ys: Sequence[float]) -> float:
        return self._pearson(_ranks(xs), _ranks(ys))

    def kendall_tau(self, xs: Sequence[float], ys: Sequence[float]) -> float:
        x, y = self._values(xs, 2), self._values(ys, 2)
        if len(x) != len(y): raise STEMStatisticsError("xs and ys length mismatch")
        concordant = discordant = ties_x = ties_y = 0
        for i in range(len(x)-1):
            for j in range(i+1, len(x)):
                dx, dy = x[j]-x[i], y[j]-y[i]
                if dx == 0 and dy == 0: continue
                if dx == 0: ties_x += 1
                elif dy == 0: ties_y += 1
                elif dx*dy > 0: concordant += 1
                else: discordant += 1
        denom = math.sqrt((concordant+discordant+ties_x)*(concordant+discordant+ties_y))
        if denom == 0.0: raise STEMStatisticsError("Kendall tau undefined")
        return (concordant-discordant)/denom

    def covariance_matrix(self, rows: Sequence[Sequence[float]], ddof: Optional[int] = None) -> List[List[float]]:
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2: raise STEMStatisticsError("rows must be a 2D observation matrix")
        if not np.isfinite(matrix).all(): raise STEMStatisticsError("rows contain non-finite values")
        correction = self.default_ddof if ddof is None else int(ddof)
        return np.cov(matrix, rowvar=False, ddof=correction).tolist()

    def correlation_matrix(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2: raise STEMStatisticsError("rows must be a 2D observation matrix")
        if not np.isfinite(matrix).all(): raise STEMStatisticsError("rows contain non-finite values")
        result = np.corrcoef(matrix, rowvar=False)
        if not np.isfinite(result).all(): raise STEMStatisticsError("Correlation matrix undefined due to zero variance")
        return result.tolist()

    def linear_regression(self, xs: Sequence[float], ys: Sequence[float]) -> Mapping[str, Any]:
        x, y = self._values(xs, 2), self._values(ys, 2)
        if len(x) != len(y): raise STEMStatisticsError("xs and ys length mismatch")
        X = np.column_stack([np.ones(len(x)), np.asarray(x)])
        beta, residuals, rank, _ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
        fitted = X @ beta; residual = np.asarray(y) - fitted
        ss_res = float(residual @ residual); mean_y = math.fsum(y)/len(y); ss_tot = math.fsum((v-mean_y)**2 for v in y)
        return {"intercept": float(beta[0]), "slope": float(beta[1]), "r_squared": 1.0 - ss_res/ss_tot if ss_tot else 1.0, "residual_sum_squares": ss_res, "rank": int(rank)}

    def polynomial_regression(self, xs: Sequence[float], ys: Sequence[float], degree: int) -> Mapping[str, Any]:
        x, y = self._values(xs, 2), self._values(ys, 2)
        if len(x) != len(y) or degree < 0 or degree >= len(x): raise STEMStatisticsError("Invalid polynomial regression problem")
        coeff = np.polynomial.polynomial.polyfit(x, y, degree); fitted = np.polynomial.polynomial.polyval(x, coeff)
        rss = float(np.sum((np.asarray(y)-fitted)**2))
        return {"coefficients": [float(v) for v in coeff], "degree": degree, "residual_sum_squares": rss}

    def multiple_linear_regression(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> Mapping[str, Any]:
        matrix = np.asarray(X, dtype=float); rhs = np.asarray(self._values(y, 2), dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != rhs.size or not np.isfinite(matrix).all(): raise STEMStatisticsError("Invalid regression design matrix")
        design = np.column_stack([np.ones(matrix.shape[0]), matrix]); beta, _, rank, singular = np.linalg.lstsq(design, rhs, rcond=None)
        residual = rhs - design @ beta
        return {"intercept": float(beta[0]), "coefficients": [float(v) for v in beta[1:]], "residual_sum_squares": float(residual @ residual), "rank": int(rank), "singular_values": [float(v) for v in singular]}

    def weighted_least_squares(self, X: Sequence[Sequence[float]], y: Sequence[float], w: Sequence[float]) -> Mapping[str, Any]:
        matrix = np.asarray(X, dtype=float); rhs = np.asarray(self._values(y, 2)); weights = np.asarray(as_float_array(w, "w", error_cls=STEMStatisticsError))
        if matrix.ndim != 2 or matrix.shape[0] != rhs.size or weights.size != rhs.size or np.any(weights <= 0): raise STEMStatisticsError("Invalid WLS inputs")
        design = np.column_stack([np.ones(matrix.shape[0]), matrix]); root = np.sqrt(weights); beta, _, rank, _ = np.linalg.lstsq(design*root[:,None], rhs*root, rcond=None)
        return {"intercept": float(beta[0]), "coefficients": [float(v) for v in beta[1:]], "rank": int(rank)}

    def normal_pdf(self, x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        s = ensure_positive(sigma, "sigma", error_cls=STEMStatisticsError); z=(float(x)-float(mu))/s
        return math.exp(-0.5*z*z)/(s*math.sqrt(2*math.pi))

    def normal_cdf(self, x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        s=ensure_positive(sigma,"sigma",error_cls=STEMStatisticsError); return NormalDist(mu=float(mu), sigma=s).cdf(float(x))

    def normal_ppf(self, p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        pv=float(p); s=ensure_positive(sigma,"sigma",error_cls=STEMStatisticsError)
        if not 0.0 < pv < 1.0: raise STEMStatisticsError("p must lie strictly in (0, 1)")
        return NormalDist(mu=float(mu), sigma=s).inv_cdf(pv)

    def t_pdf(self, x: float, df: float) -> float:
        nu=ensure_positive(df,"df",error_cls=STEMStatisticsError); xv=float(x)
        return math.exp(float(gammaln((nu+1)/2)-gammaln(nu/2)))/(math.sqrt(nu*math.pi))*((1+xv*xv/nu)**(-(nu+1)/2))

    def t_cdf(self, x: float, df: float) -> float:
        nu=ensure_positive(df,"df",error_cls=STEMStatisticsError); xv=float(x); z=nu/(nu+xv*xv); ib=float(betainc(nu/2,0.5,z))
        return 1-0.5*ib if xv >= 0 else 0.5*ib

    def chi_square_cdf(self, x: float, df: float) -> float:
        xv=ensure_non_negative(x,"x",error_cls=STEMStatisticsError); nu=ensure_positive(df,"df",error_cls=STEMStatisticsError); return float(gammainc(nu/2,xv/2))

    def f_cdf(self, x: float, df1: float, df2: float) -> float:
        xv=ensure_non_negative(x,"x",error_cls=STEMStatisticsError); a=ensure_positive(df1,"df1",error_cls=STEMStatisticsError); b=ensure_positive(df2,"df2",error_cls=STEMStatisticsError)
        return float(betainc(a/2,b/2,(a*xv)/(a*xv+b)))

    def binomial_pmf(self, k: int, n: int, p: float) -> float:
        if not isinstance(k,int) or not isinstance(n,int) or n < 0 or k < 0 or k > n: raise STEMStatisticsError("Invalid binomial k/n")
        pv=float(p)
        if not 0 <= pv <= 1: raise STEMStatisticsError("p must lie in [0,1]")
        return math.comb(n,k)*(pv**k)*((1-pv)**(n-k))

    def log_likelihood_normal(self, xs: Sequence[float], mu: float, sigma: float) -> float:
        values=self._values(xs); s=ensure_positive(sigma,"sigma",error_cls=STEMStatisticsError); m=float(mu)
        return -len(values)*math.log(s*math.sqrt(2*math.pi))-math.fsum((v-m)**2 for v in values)/(2*s*s)

    def log_likelihood_bernoulli(self, xs: Sequence[float], p: float) -> float:
        values=self._values(xs); pv=float(p)
        if not 0 < pv < 1 or any(v not in (0.0,1.0) for v in values): raise STEMStatisticsError("Bernoulli data must be 0/1 and p in (0,1)")
        return math.fsum(v*math.log(pv)+(1-v)*math.log1p(-pv) for v in values)

    def monte_carlo_integrate(self, f: Callable[[float], float], a: float, b: float, n: int, rng: Optional[random.Random] = None) -> Mapping[str, float]:
        validate_callable(f,"f"); left,right=float(a),float(b)
        if n < 1 or not left < right: raise STEMStatisticsError("Invalid Monte Carlo integration domain/sample count")
        generator=rng or random.Random(self.random_seed); samples=[ensure_finite_number(f(generator.uniform(left,right)),"f(x)",error_cls=STEMStatisticsError) for _ in range(n)]
        mean=math.fsum(samples)/n; variance=0.0 if n==1 else stable_variance(samples,1,error_cls=STEMStatisticsError); estimate=(right-left)*mean; se=(right-left)*math.sqrt(variance/n) if n>1 else 0.0
        return {"estimate":estimate,"standard_error":se,"samples":float(n)}

    def bootstrap_ci(self, xs: Sequence[float], statistic: Callable[[Sequence[float]], float], n_resamples: int, confidence: float = 0.95, rng: Optional[random.Random] = None) -> Mapping[str, float]:
        values=self._values(xs); validate_callable(statistic,"statistic")
        if n_resamples < 1 or not 0 < confidence < 1: raise STEMStatisticsError("Invalid bootstrap settings")
        generator=rng or random.Random(self.random_seed); estimates=[]
        for _ in range(n_resamples):
            sample=[values[generator.randrange(len(values))] for _ in values]; estimates.append(ensure_finite_number(statistic(sample),"bootstrap statistic",error_cls=STEMStatisticsError))
        alpha=1-confidence
        return {"estimate":ensure_finite_number(statistic(values),"statistic",error_cls=STEMStatisticsError),"lower":self.quantile(estimates,alpha/2),"upper":self.quantile(estimates,1-alpha/2),"confidence":confidence}


def _ranks(values: Sequence[float]) -> List[float]:
    vals=as_float_array(values,"values",error_cls=STEMStatisticsError); order=sorted(range(len(vals)),key=vals.__getitem__); ranks=[0.0]*len(vals); i=0
    while i < len(order):
        j=i+1
        while j < len(order) and vals[order[j]] == vals[order[i]]: j += 1
        rank=0.5*(i+1+j)
        for k in range(i,j): ranks[order[k]]=rank
        i=j
    return ranks


def _as_column_matrix(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    matrix=np.asarray(rows,dtype=float)
    if matrix.ndim != 2:return []
    return matrix.tolist()


def _solve_square(A: List[List[float]], b: List[float]) -> List[float]:
    try:return np.linalg.solve(np.asarray(A,dtype=float),np.asarray(b,dtype=float)).tolist()
    except np.linalg.LinAlgError as exc: raise STEMStatisticsError("Statistical linear system is singular",cause=exc) from exc


__all__ = ["Statistics", "OnlineMoments", "OnlineCovariance"]
