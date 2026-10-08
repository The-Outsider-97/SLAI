"""
Deterministic descriptive and numerical statistics for the STEM subsystem.

Ownership
- mean/variance, moments;
- covariance and correlation;
- deterministic descriptive statistics;
- regression computation;
- distribution-function evaluation;
- quantiles;
- deterministic estimator calculation;
- numerical likelihood evaluation;
- stable online statistics;
- numerical statistical algorithms.

This module does NOT own probabilistic reasoning, hypothesis inference, or
Bayesian model selection. Those concerns belong elsewhere.

Sources
- Monahan, J. F. (2011). Numerical Methods of Statistics (2nd ed.).
  Cambridge University Press. DOI 10.1017/CBO9780511977176.
- Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for
  Computing the Sample Variance: Analysis and Recommendations."
  The American Statistician, 37(3), 242–247.
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import random

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import *
from ..utils.stem_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Statistics")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Special functions used by distribution CDFs
# ---------------------------------------------------------------------------


def _log_gamma(x: float) -> float:
    if x <= 0.0:
        raise STEMStatisticsError("log_gamma requires positive argument", context={"x": x})
    return math.lgamma(x)


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _log_gamma(a + b) - _log_gamma(a) - _log_gamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _regularized_gamma_p(a: float, x: float, max_iter: int = 500) -> float:
    """Regularized lower incomplete gamma P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise STEMStatisticsError("gamma arguments out of range")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(max_iter):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 3e-16:
                break
        return total * math.exp(-x + a * math.log(x) - _log_gamma(a))
    # continued fraction
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - _log_gamma(a)) * h


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0.0:
        raise STEMStatisticsError("sigma must be positive")
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0.0:
        raise STEMStatisticsError("sigma must be positive")
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _normal_ppf(p: float, mu: float, sigma: float) -> float:
    if not 0.0 < p < 1.0:
        raise STEMStatisticsError("p must lie in (0, 1)")
    if sigma <= 0.0:
        raise STEMStatisticsError("sigma must be positive")
    # Wichura's algorithm (AS 241)
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    # one refinement step (Halley)
    e = _normal_cdf(z, 0.0, 1.0) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * z * z)
    z = z - u / (1.0 + 0.5 * z * u)
    return mu + sigma * z


# ---------------------------------------------------------------------------
# Online accumulators
# ---------------------------------------------------------------------------
@dataclass
class OnlineMoments:
    """Welford's online mean and variance accumulator."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        """Incorporate a new observation."""
        x_f = ensure_finite_number(x, "x", error_cls=STEMStatisticsError)
        self.n += 1
        delta = x_f - self.mean
        self.mean += delta / self.n
        delta2 = x_f - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Sample variance (ddof=1)."""
        if self.n < 2:
            raise STEMStatisticsError("variance requires at least two observations")
        return self.m2 / (self.n - 1)

    @property
    def population_variance(self) -> float:
        """Population variance (ddof=0)."""
        if self.n < 1:
            raise STEMStatisticsError("population variance requires at least one observation")
        return self.m2 / self.n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "m2": self.m2,
            "variance": self.variance if self.n >= 2 else None,
            "population_variance": self.population_variance if self.n >= 1 else None,
        }


@dataclass
class OnlineCovariance:
    """Online covariance and correlation accumulator (Chan et al. 1983)."""

    n: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0
    co_moment: float = 0.0

    def update(self, x: float, y: float) -> None:
        """Incorporate a new (x, y) observation."""
        x_f = ensure_finite_number(x, "x", error_cls=STEMStatisticsError)
        y_f = ensure_finite_number(y, "y", error_cls=STEMStatisticsError)
        self.n += 1
        dx = x_f - self.mean_x
        dy = y_f - self.mean_y
        self.mean_x += dx / self.n
        self.mean_y += dy / self.n
        self.m2_x += dx * (x_f - self.mean_x)
        self.m2_y += dy * (y_f - self.mean_y)
        self.co_moment += dx * (y_f - self.mean_y)

    @property
    def covariance(self) -> float:
        """Sample covariance (ddof=1)."""
        if self.n < 2:
            raise STEMStatisticsError("covariance requires at least two observations")
        return self.co_moment / (self.n - 1)

    @property
    def correlation(self) -> float:
        """Pearson correlation."""
        if self.n < 2:
            raise STEMStatisticsError("correlation requires at least two observations")
        denom = math.sqrt(self.m2_x * self.m2_y)
        return self.co_moment / denom if denom > 0.0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "covariance": self.covariance if self.n >= 2 else None,
            "correlation": self.correlation if self.n >= 2 else None,
        }


# ---------------------------------------------------------------------------
# Statistics class
# ---------------------------------------------------------------------------
class Statistics:
    """Deterministic descriptive and numerical statistics."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.statistics_config = dict(get_config_section("statistics", config=self.config) or {})
        if config:
            self.statistics_config.update(dict(config))

        self._default_ddof = int(self.statistics_config.get("default_ddof", 1))

    # -- descriptive -------------------------------------------------------

    def mean(self, xs: Sequence[float]) -> float:
        """Arithmetic mean."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        if not values:
            raise STEMStatisticsError("mean requires at least one observation")
        return sum(values) / len(values)

    def variance(self, xs: Sequence[float], ddof: Optional[int] = None) -> float:
        """Stable sample variance (Welford)."""
        ddof_i = int(ddof if ddof is not None else self._default_ddof)
        return stable_variance(xs, ddof=ddof_i, error_cls=STEMStatisticsError)

    def standard_deviation(self, xs: Sequence[float], ddof: Optional[int] = None) -> float:
        """Standard deviation (square root of the sample variance)."""
        return math.sqrt(self.variance(xs, ddof=ddof))

    def median(self, xs: Sequence[float]) -> float:
        """Median of a sample."""
        values = sorted(as_float_array(xs, "xs", error_cls=STEMStatisticsError))
        if not values:
            raise STEMStatisticsError("median requires at least one observation")
        n = len(values)
        if n % 2 == 1:
            return values[n // 2]
        return 0.5 * (values[n // 2 - 1] + values[n // 2])

    def quantile(self, xs: Sequence[float], q: float, method: str = "linear") -> float:
        """Quantile via the given interpolation method (``linear`` by default)."""
        values = sorted(as_float_array(xs, "xs", error_cls=STEMStatisticsError))
        if not values:
            raise STEMStatisticsError("quantile requires at least one observation")
        q_f = ensure_finite_number(q, "q", error_cls=STEMStatisticsError)
        if not 0.0 <= q_f <= 1.0:
            raise STEMStatisticsError("q must lie in [0, 1]", context={"q": q_f})
        if method != "linear":
            raise STEMStatisticsError(
                "only 'linear' quantile method is currently supported",
                context={"method": method},
            )

        position = (len(values) - 1) * q_f
        lo = int(math.floor(position))
        hi = int(math.ceil(position))
        if lo == hi:
            return values[lo]
        return values[lo] + (position - lo) * (values[hi] - values[lo])

    def mode(self, xs: Sequence[float]) -> Any:
        """Most common value, breaking ties by smallest value."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        if not values:
            raise STEMStatisticsError("mode requires at least one observation")
        counts: Dict[float, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return min(
            (v for v, c in counts.items() if c == max(counts.values()))
        )

    def skewness(self, xs: Sequence[float]) -> float:
        """Sample skewness using the third standardized central moment."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        n = len(values)
        if n < 3:
            raise STEMStatisticsError("skewness requires at least three observations")
        m = sum(values) / n
        m2 = sum((v - m) ** 2 for v in values) / n
        m3 = sum((v - m) ** 3 for v in values) / n
        if m2 == 0.0:
            return 0.0
        return m3 / (m2 ** 1.5)

    def kurtosis(self, xs: Sequence[float], excess: bool = True) -> float:
        """Sample kurtosis, optionally excess (subtracting 3)."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        n = len(values)
        if n < 4:
            raise STEMStatisticsError("kurtosis requires at least four observations")
        m = sum(values) / n
        m2 = sum((v - m) ** 2 for v in values) / n
        m4 = sum((v - m) ** 4 for v in values) / n
        if m2 == 0.0:
            return 0.0
        k = m4 / (m2 * m2)
        return k - 3.0 if excess else k

    # -- moments -----------------------------------------------------------

    def raw_moment(self, xs: Sequence[float], k: int) -> float:
        """Raw moment of order ``k``."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        if not values:
            raise STEMStatisticsError("raw_moment requires at least one observation")
        k_i = int(ensure_positive(k, "k", allow_zero=True, error_cls=STEMStatisticsError))
        return sum(v ** k_i for v in values) / len(values)

    def central_moment(self, xs: Sequence[float], k: int) -> float:
        """Central moment of order ``k`` about the sample mean."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        if not values:
            raise STEMStatisticsError("central_moment requires at least one observation")
        k_i = int(ensure_positive(k, "k", allow_zero=True, error_cls=STEMStatisticsError))
        m = sum(values) / len(values)
        return sum((v - m) ** k_i for v in values) / len(values)

    def standardized_moment(self, xs: Sequence[float], k: int) -> float:
        """Standardized moment of order ``k``."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        m = sum(values) / len(values) if values else 0.0
        sigma2 = sum((v - m) ** 2 for v in values) / len(values) if values else 0.0
        if sigma2 == 0.0:
            return 0.0
        sigma = math.sqrt(sigma2)
        k_i = int(k)
        return sum(((v - m) / sigma) ** k_i for v in values) / len(values)

    # -- covariance and correlation ---------------------------------------

    def covariance(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        ddof: Optional[int] = None,
    ) -> float:
        """Sample covariance (Chan et al. stable update)."""
        x = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        y = as_float_array(ys, "ys", error_cls=STEMStatisticsError)
        if len(x) != len(y):
            raise STEMStatisticsError("xs and ys must have equal length")
        n = len(x)
        if n < 2:
            raise STEMStatisticsError("covariance requires at least two observations")
        ddof_i = int(ddof if ddof is not None else self._default_ddof)
        if n - ddof_i <= 0:
            raise STEMStatisticsError("ddof too large for sample size")

        accumulator = OnlineCovariance()
        for xi, yi in zip(x, y):
            accumulator.update(xi, yi)
        return accumulator.co_moment / (n - ddof_i)

    def correlation(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        method: str = "pearson",
    ) -> float:
        """Correlation coefficient for the requested method."""
        if method == "pearson":
            return self._pearson(xs, ys)
        if method == "spearman":
            return self.spearman_correlation(xs, ys)
        raise STEMStatisticsError(
            "unsupported correlation method",
            context={"method": method},
        )

    def _pearson(self, xs: Sequence[float], ys: Sequence[float]) -> float:
        x = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        y = as_float_array(ys, "ys", error_cls=STEMStatisticsError)
        if len(x) != len(y):
            raise STEMStatisticsError("xs and ys must have equal length")
        acc = OnlineCovariance()
        for xi, yi in zip(x, y):
            acc.update(xi, yi)
        return acc.correlation

    def spearman_correlation(self, xs: Sequence[float], ys: Sequence[float]) -> float:
        """Spearman rank correlation."""
        x = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        y = as_float_array(ys, "ys", error_cls=STEMStatisticsError)
        if len(x) != len(y):
            raise STEMStatisticsError("xs and ys must have equal length")
        rx = _ranks(x)
        ry = _ranks(y)
        return self._pearson(rx, ry)

    def kendall_tau(self, xs: Sequence[float], ys: Sequence[float]) -> float:
        """Kendall's tau-a."""
        x = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        y = as_float_array(ys, "ys", error_cls=STEMStatisticsError)
        if len(x) != len(y):
            raise STEMStatisticsError("xs and ys must have equal length")
        n = len(x)
        if n < 2:
            raise STEMStatisticsError("Kendall's tau requires at least two observations")
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = (x[i] - x[j]) * (y[i] - y[j])
                if s > 0:
                    concordant += 1
                elif s < 0:
                    discordant += 1
        return (concordant - discordant) / (0.5 * n * (n - 1))

    def covariance_matrix(
        self,
        rows: Sequence[Sequence[float]],
        ddof: Optional[int] = None,
    ) -> List[List[float]]:
        """Covariance matrix of column-wise observations."""
        matrix = _as_column_matrix(rows)
        p = len(matrix)
        result = [[0.0] * p for _ in range(p)]
        for i in range(p):
            for j in range(i, p):
                c = self.covariance(matrix[i], matrix[j], ddof=ddof)
                result[i][j] = c
                result[j][i] = c
        return result

    def correlation_matrix(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        """Pearson correlation matrix of column-wise observations."""
        matrix = _as_column_matrix(rows)
        p = len(matrix)
        result = [[0.0] * p for _ in range(p)]
        for i in range(p):
            result[i][i] = 1.0
            for j in range(i + 1, p):
                c = self._pearson(matrix[i], matrix[j])
                result[i][j] = c
                result[j][i] = c
        return result

    # -- regression -------------------------------------------------------

    def linear_regression(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
    ) -> Mapping[str, Any]:
        """Ordinary least-squares linear regression ``y = a + b x``."""
        x = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        y = as_float_array(ys, "ys", error_cls=STEMStatisticsError)
        if len(x) != len(y):
            raise STEMStatisticsError("xs and ys must have equal length")
        n = len(x)
        if n < 2:
            raise STEMStatisticsError("linear regression requires at least two points")

        acc = OnlineCovariance()
        for xi, yi in zip(x, y):
            acc.update(xi, yi)

        var_x = acc.m2_x / (n - 1)
        cov_xy = acc.co_moment / (n - 1)
        if var_x == 0.0:
            raise STEMStatisticsError("independent variable has zero variance")

        slope = cov_xy / var_x
        intercept = acc.mean_y - slope * acc.mean_x

        residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
        ss_res = sum(r * r for r in residuals)
        ss_tot = sum((yi - acc.mean_y) ** 2 for yi in y)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

        mse = ss_res / max(1, n - 2)
        se_slope = math.sqrt(mse / (acc.m2_x / (n - 1))) if acc.m2_x > 0 else 0.0
        se_intercept = math.sqrt(mse * (1.0 / n + (acc.mean_x ** 2) / acc.m2_x)) if acc.m2_x > 0 else 0.0

        return {
            "slope": slope,
            "intercept": intercept,
            "residuals": residuals,
            "r_squared": r_squared,
            "standard_error_slope": se_slope,
            "standard_error_intercept": se_intercept,
        }

    def polynomial_regression(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        degree: int,
    ) -> Mapping[str, Any]:
        """Polynomial regression via normal equations."""
        x = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        y = as_float_array(ys, "ys", error_cls=STEMStatisticsError)
        if len(x) != len(y):
            raise STEMStatisticsError("xs and ys must have equal length")
        deg = int(ensure_positive(degree, "degree", allow_zero=True, error_cls=STEMStatisticsError))
        n = len(x)
        if n < deg + 1:
            raise STEMStatisticsError("polynomial regression requires n >= degree + 1 points")

        # Build Vandermonde and solve normal equations
        X = [[xi ** k for k in range(deg + 1)] for xi in x]
        XT_X = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(deg + 1)] for a in range(deg + 1)]
        XT_y = [sum(X[i][a] * y[i] for i in range(n)) for a in range(deg + 1)]

        coefficients = _solve_square(XT_X, XT_y)
        predictions = [sum(coefficients[k] * (xi ** k) for k in range(deg + 1)) for xi in x]
        residuals = [yi - pi for yi, pi in zip(y, predictions)]
        ss_res = sum(r * r for r in residuals)
        mean_y = sum(y) / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        return {
            "coefficients": coefficients,
            "predictions": predictions,
            "residuals": residuals,
            "r_squared": r_squared,
        }

    def multiple_linear_regression(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
    ) -> Mapping[str, Any]:
        """Multiple linear regression via normal equations."""
        matrix = [as_float_array(row, f"X[{i}]", error_cls=STEMStatisticsError) for i, row in enumerate(X)]
        target = as_float_array(y, "y", error_cls=STEMStatisticsError)
        n = len(matrix)
        if n != len(target):
            raise STEMStatisticsError("X rows and y length must match")
        if n == 0:
            raise STEMStatisticsError("X must not be empty")
        p = len(matrix[0])

        XT_X = [[sum(matrix[i][a] * matrix[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
        XT_y = [sum(matrix[i][a] * target[i] for i in range(n)) for a in range(p)]
        coefficients = _solve_square(XT_X, XT_y)
        predictions = [sum(matrix[i][k] * coefficients[k] for k in range(p)) for i in range(n)]
        residuals = [target[i] - predictions[i] for i in range(n)]
        ss_res = sum(r * r for r in residuals)
        mean_y = sum(target) / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in target)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        return {
            "coefficients": coefficients,
            "predictions": predictions,
            "residuals": residuals,
            "r_squared": r_squared,
        }

    def weighted_least_squares(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        w: Sequence[float],
    ) -> Mapping[str, Any]:
        """Weighted least squares with non-negative weights."""
        matrix = [as_float_array(row, f"X[{i}]", error_cls=STEMStatisticsError) for i, row in enumerate(X)]
        target = as_float_array(y, "y", error_cls=STEMStatisticsError)
        weights = as_float_array(w, "w", error_cls=STEMStatisticsError)
        n = len(matrix)
        if n != len(target) or n != len(weights):
            raise STEMStatisticsError("X, y, and w must have consistent lengths")
        p = len(matrix[0])
        for weight in weights:
            if weight <= 0.0:
                raise STEMStatisticsError("weights must be strictly positive")

        XTWX = [[sum(weights[i] * matrix[i][a] * matrix[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
        XTWy = [sum(weights[i] * matrix[i][a] * target[i] for i in range(n)) for a in range(p)]
        coefficients = _solve_square(XTWX, XTWy)
        predictions = [sum(matrix[i][k] * coefficients[k] for k in range(p)) for i in range(n)]
        residuals = [target[i] - predictions[i] for i in range(n)]
        return {
            "coefficients": coefficients,
            "predictions": predictions,
            "residuals": residuals,
        }

    # -- distributions -----------------------------------------------------

    def normal_pdf(self, x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Normal probability density."""
        return _normal_pdf(ensure_finite_number(x, "x", error_cls=STEMStatisticsError),
                           ensure_finite_number(mu, "mu", error_cls=STEMStatisticsError),
                           ensure_positive(sigma, "sigma", allow_zero=False, error_cls=STEMStatisticsError))

    def normal_cdf(self, x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Normal cumulative distribution."""
        return _normal_cdf(ensure_finite_number(x, "x", error_cls=STEMStatisticsError),
                           ensure_finite_number(mu, "mu", error_cls=STEMStatisticsError),
                           ensure_positive(sigma, "sigma", allow_zero=False, error_cls=STEMStatisticsError))

    def normal_ppf(self, p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Normal quantile function."""
        return _normal_ppf(p, mu, sigma)

    def t_pdf(self, x: float, df: float) -> float:
        """Student's t probability density."""
        x_f = ensure_finite_number(x, "x", error_cls=STEMStatisticsError)
        df_f = ensure_positive(df, "df", allow_zero=False, error_cls=STEMStatisticsError)
        c = math.exp(_log_gamma((df_f + 1.0) / 2.0) - _log_gamma(df_f / 2.0)) / math.sqrt(df_f * math.pi)
        return c * (1.0 + x_f * x_f / df_f) ** (-(df_f + 1.0) / 2.0)

    def t_cdf(self, x: float, df: float) -> float:
        """Student's t cumulative distribution."""
        x_f = ensure_finite_number(x, "x", error_cls=STEMStatisticsError)
        df_f = ensure_positive(df, "df", allow_zero=False, error_cls=STEMStatisticsError)
        if x_f == 0.0:
            return 0.5
        t = df_f / (df_f + x_f * x_f)
        ib = 0.5 * _regularized_incomplete_beta(df_f / 2.0, 0.5, t)
        return 1.0 - ib if x_f > 0 else ib

    def chi_square_cdf(self, x: float, df: float) -> float:
        """Chi-square cumulative distribution."""
        x_f = ensure_finite_number(x, "x", error_cls=STEMStatisticsError)
        df_f = ensure_positive(df, "df", allow_zero=False, error_cls=STEMStatisticsError)
        if x_f <= 0.0:
            return 0.0
        return _regularized_gamma_p(df_f / 2.0, x_f / 2.0)

    def f_cdf(self, x: float, df1: float, df2: float) -> float:
        """F cumulative distribution."""
        x_f = ensure_finite_number(x, "x", error_cls=STEMStatisticsError)
        d1 = ensure_positive(df1, "df1", allow_zero=False, error_cls=STEMStatisticsError)
        d2 = ensure_positive(df2, "df2", allow_zero=False, error_cls=STEMStatisticsError)
        if x_f <= 0.0:
            return 0.0
        z = d1 * x_f / (d1 * x_f + d2)
        return _regularized_incomplete_beta(d1 / 2.0, d2 / 2.0, z)

    def binomial_pmf(self, k: int, n: int, p: float) -> float:
        """Binomial probability mass function."""
        k_i = int(ensure_non_negative(k, "k", allow_zero=True, error_cls=STEMStatisticsError))
        n_i = int(ensure_positive(n, "n", allow_zero=True, error_cls=STEMStatisticsError))
        p_f = ensure_finite_number(p, "p", error_cls=STEMStatisticsError)
        if not 0.0 <= p_f <= 1.0:
            raise STEMStatisticsError("p must lie in [0, 1]")
        if k_i < 0 or k_i > n_i:
            return 0.0
        return math.comb(n_i, k_i) * (p_f ** k_i) * ((1.0 - p_f) ** (n_i - k_i))

    # -- likelihood --------------------------------------------------------

    def log_likelihood_normal(
        self,
        xs: Sequence[float],
        mu: float,
        sigma: float,
    ) -> float:
        """Log likelihood of i.i.d. normal observations."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        mu_f = ensure_finite_number(mu, "mu", error_cls=STEMStatisticsError)
        sigma_f = ensure_positive(sigma, "sigma", allow_zero=False, error_cls=STEMStatisticsError)
        n = len(values)
        if n == 0:
            raise STEMStatisticsError("log_likelihood_normal requires at least one observation")
        ss = sum((v - mu_f) ** 2 for v in values)
        return -0.5 * n * math.log(2.0 * math.pi) - n * math.log(sigma_f) - ss / (2.0 * sigma_f * sigma_f)

    def log_likelihood_bernoulli(self, xs: Sequence[float], p: float) -> float:
        """Log likelihood of i.i.d. Bernoulli observations."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        p_f = ensure_finite_number(p, "p", error_cls=STEMStatisticsError)
        if not 0.0 < p_f < 1.0:
            raise STEMStatisticsError("p must lie strictly in (0, 1)")
        total = 0.0
        for value in values:
            if value == 1.0:
                total += math.log(p_f)
            elif value == 0.0:
                total += math.log(1.0 - p_f)
            else:
                raise STEMStatisticsError("Bernoulli observations must be 0 or 1")
        return total

    # -- monte carlo -------------------------------------------------------

    def monte_carlo_integrate(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        n: int,
        rng: Optional[random.Random] = None,
    ) -> Mapping[str, float]:
        """
        Monte Carlo integration of ``f`` over ``[a, b]``.

        If ``rng`` is ``None``, a fresh ``random.Random()`` is created. Provide
        a seeded generator for deterministic results.
        """
        validate_callable(f, "f", error_cls=STEMValidationError)
        a_f = ensure_finite_number(a, "a", error_cls=STEMStatisticsError)
        b_f = ensure_finite_number(b, "b", error_cls=STEMStatisticsError)
        if b_f < a_f:
            raise STEMStatisticsError("a must be <= b")
        n_i = int(ensure_positive(n, "n", allow_zero=False, error_cls=STEMStatisticsError))
        generator = rng if rng is not None else random.Random()

        total = 0.0
        total_sq = 0.0
        for _ in range(n_i):
            x = generator.uniform(a_f, b_f)
            y = float(f(x))
            total += y
            total_sq += y * y

        mean_y = total / n_i
        variance = (total_sq / n_i - mean_y * mean_y) if n_i > 1 else 0.0
        std_error = math.sqrt(variance / n_i) if n_i > 1 else 0.0
        integral = (b_f - a_f) * mean_y
        return {
            "integral": integral,
            "standard_error": (b_f - a_f) * std_error,
            "samples": float(n_i),
        }

    def bootstrap_ci(
        self,
        xs: Sequence[float],
        statistic: Callable[[Sequence[float]], float],
        n_resamples: int,
        confidence: float = 0.95,
        rng: Optional[random.Random] = None,
    ) -> Mapping[str, float]:
        """Percentile bootstrap confidence interval for a scalar statistic."""
        values = as_float_array(xs, "xs", error_cls=STEMStatisticsError)
        if not values:
            raise STEMStatisticsError("bootstrap requires at least one observation")
        validate_callable(statistic, "statistic", error_cls=STEMValidationError)
        n_resamples_i = int(ensure_positive(n_resamples, "n_resamples", allow_zero=False, error_cls=STEMStatisticsError))
        conf = ensure_finite_number(confidence, "confidence", error_cls=STEMStatisticsError)
        if not 0.0 < conf < 1.0:
            raise STEMStatisticsError("confidence must lie strictly in (0, 1)")
        generator = rng if rng is not None else random.Random()

        n = len(values)
        samples: List[float] = []
        for _ in range(n_resamples_i):
            resample = [values[generator.randrange(n)] for _ in range(n)]
            samples.append(float(statistic(resample)))
        samples.sort()

        alpha = (1.0 - conf) / 2.0
        lo_idx = max(0, int(math.floor(alpha * len(samples))) - 1)
        hi_idx = min(len(samples) - 1, int(math.ceil((1.0 - alpha) * len(samples))) - 1)
        return {
            "lower": samples[lo_idx],
            "upper": samples[hi_idx],
            "confidence": conf,
            "resamples": float(n_resamples_i),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> List[float]:
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _as_column_matrix(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    sequences = ensure_sequence(rows, "rows", error_cls=STEMValidationError)
    if not sequences:
        raise STEMStatisticsError("rows must not be empty")
    first = ensure_sequence(sequences[0], "rows[0]", error_cls=STEMValidationError)
    p = len(first)
    columns: List[List[float]] = [[] for _ in range(p)]
    for i, row in enumerate(sequences):
        row_values = as_float_array(row, f"rows[{i}]", error_cls=STEMStatisticsError)
        if len(row_values) != p:
            raise STEMStatisticsError("all rows must have the same length")
        for j, value in enumerate(row_values):
            columns[j].append(value)
    return columns


def _solve_square(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    # Gaussian elimination with partial pivoting
    aug = [A[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(aug[i][k]))
        if abs(aug[pivot][k]) < 1e-300:
            raise STEMStatisticsError("singular system in regression")
        if pivot != k:
            aug[k], aug[pivot] = aug[pivot], aug[k]
        for i in range(k + 1, n):
            factor = aug[i][k] / aug[k][k]
            for j in range(k, n + 1):
                aug[i][j] -= factor * aug[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / aug[i][i]
    return x


__all__ = ["Statistics", "OnlineMoments", "OnlineCovariance"]