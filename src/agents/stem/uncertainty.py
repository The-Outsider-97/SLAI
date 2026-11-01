"""Measurement-uncertainty propagation for SLAI STEM.

Grounding: JCGM 100 (GUM), JCGM GUM-6, and JCGM 101. This module handles
measurement/numerical uncertainty, not epistemic belief or probabilistic
reasoning.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import random
import numpy as np # pyright: ignore[reportMissingImports]

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .stem_memory import STEMMemory
from .stem_types import Distribution, NumericResult, Uncertainty as UncertaintyValue, UncertaintyBudget, UncertaintyType
from .utils.config_loader import get_config_section, load_global_config
from .utils.stem_errors import STEMUncertaintyError
from .utils.stem_helpers import as_float_array, ensure_finite_number, ensure_non_negative, ensure_positive, validate_callable
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Uncertainty")
printer = PrettyPrinter()


def standard_uncertainty(value: float, *, distribution: str | Distribution = Distribution.NORMAL, half_width: bool = False) -> float:
    """Convert a stated uncertainty/half-width to standard uncertainty.

    Normal values are assumed already standard unless ``half_width=True``;
    rectangular and triangular half-widths use sqrt(3) and sqrt(6),
    respectively, per GUM Type-B conventions.
    """
    magnitude = ensure_non_negative(value, "value", error_cls=STEMUncertaintyError)
    dist = distribution if isinstance(distribution, Distribution) else Distribution(distribution)
    if not half_width:
        return magnitude
    if dist == Distribution.RECTANGULAR:
        return magnitude / math.sqrt(3.0)
    if dist == Distribution.TRIANGULAR:
        return magnitude / math.sqrt(6.0)
    if dist == Distribution.U_SHAPED:
        return magnitude / math.sqrt(2.0)
    return magnitude


def combined_standard_uncertainty(components: Sequence[float], *, sensitivity: Optional[Sequence[float]] = None, covariance: Optional[Sequence[Sequence[float]]] = None) -> float:
    u = np.asarray(as_float_array(components, "components", error_cls=STEMUncertaintyError), dtype=float)
    if u.size == 0:
        raise STEMUncertaintyError("At least one uncertainty component is required")
    c = np.ones_like(u) if sensitivity is None else np.asarray(as_float_array(sensitivity, "sensitivity", error_cls=STEMUncertaintyError), dtype=float)
    if c.shape != u.shape:
        raise STEMUncertaintyError("sensitivity length must equal components length")
    if covariance is None:
        return float(math.sqrt(float(np.sum((c * u) ** 2))))
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (u.size, u.size) or not np.isfinite(cov).all():
        raise STEMUncertaintyError("covariance matrix has invalid shape or non-finite values")
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-15):
        raise STEMUncertaintyError("covariance matrix must be symmetric")
    variance = float(c @ cov @ c)
    if variance < -1e-15:
        raise STEMUncertaintyError("propagated variance is negative", context={"variance": variance})
    return math.sqrt(max(0.0, variance))


def expanded_uncertainty(standard: float, coverage_factor: float = 2.0) -> float:
    return ensure_non_negative(standard, "standard", error_cls=STEMUncertaintyError) * ensure_positive(coverage_factor, "coverage_factor", error_cls=STEMUncertaintyError)


def covariance_propagation(jacobian: Sequence[Sequence[float]] | Sequence[float], covariance: Sequence[Sequence[float]]) -> Any:
    J = np.asarray(jacobian, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if J.ndim == 1:
        J = J.reshape(1, -1)
    if J.ndim != 2 or cov.ndim != 2 or cov.shape[0] != cov.shape[1] or J.shape[1] != cov.shape[0]:
        raise STEMUncertaintyError("Jacobian/covariance dimensions are incompatible")
    if not np.isfinite(J).all() or not np.isfinite(cov).all():
        raise STEMUncertaintyError("Jacobian/covariance must contain finite values")
    result = J @ cov @ J.T
    return float(result[0, 0]) if result.shape == (1, 1) else result.tolist()


def jacobian_propagation(jacobian: Sequence[float], standard_uncertainties: Sequence[float], correlations: Optional[Sequence[Sequence[float]]] = None) -> float:
    j = np.asarray(as_float_array(jacobian, "jacobian", error_cls=STEMUncertaintyError), dtype=float)
    u = np.asarray(as_float_array(standard_uncertainties, "standard_uncertainties", error_cls=STEMUncertaintyError), dtype=float)
    if j.shape != u.shape:
        raise STEMUncertaintyError("Jacobian and uncertainty vectors must match")
    if correlations is None:
        cov = np.diag(u ** 2)
    else:
        corr = np.asarray(correlations, dtype=float)
        if corr.shape != (u.size, u.size):
            raise STEMUncertaintyError("correlation matrix shape mismatch")
        if np.any(np.abs(corr) > 1.0 + 1e-12):
            raise STEMUncertaintyError("correlations must lie in [-1, 1]")
        cov = np.diag(u) @ corr @ np.diag(u)
    variance = covariance_propagation(j, cov)
    return math.sqrt(max(0.0, float(variance)))


def sensitivity_coefficients(model: Callable[..., float], values: Sequence[float], *, steps: Optional[Sequence[float]] = None) -> list[float]:
    validate_callable(model, "model")
    x = as_float_array(values, "values", error_cls=STEMUncertaintyError)
    if steps is not None and len(steps) != len(x):
        raise STEMUncertaintyError("steps length mismatch")
    coefficients: list[float] = []
    for i, value in enumerate(x):
        h = abs(float(steps[i])) if steps is not None else (np.finfo(float).eps ** (1.0 / 3.0)) * max(1.0, abs(value))
        if h == 0.0:
            h = 1e-8
        xp, xm = x[:], x[:]
        xp[i] += h; xm[i] -= h
        fp = ensure_finite_number(model(*xp), "model(+h)", error_cls=STEMUncertaintyError)
        fm = ensure_finite_number(model(*xm), "model(-h)", error_cls=STEMUncertaintyError)
        coefficients.append((fp - fm) / (2.0 * h))
    return coefficients


def coverage_interval(value: float, standard: float, coverage_factor: float = 2.0) -> tuple[float, float]:
    center = ensure_finite_number(value, "value", error_cls=STEMUncertaintyError)
    expanded = expanded_uncertainty(standard, coverage_factor)
    return center - expanded, center + expanded


def monte_carlo_propagation(model: Callable[..., float], means: Sequence[float], standard_uncertainties: Sequence[float], *, distributions: Optional[Sequence[str | Distribution]] = None, samples: int = 10000, seed: int) -> Mapping[str, Any]:
    """JCGM-101-style reproducible distribution propagation.

    A seed is mandatory by design; stochastic numerical propagation must be
    reproducible and must not become implicit probabilistic reasoning.
    """
    validate_callable(model, "model")
    mu = as_float_array(means, "means", error_cls=STEMUncertaintyError)
    u = as_float_array(standard_uncertainties, "standard_uncertainties", error_cls=STEMUncertaintyError)
    if len(mu) != len(u) or samples < 2:
        raise STEMUncertaintyError("Invalid Monte Carlo dimensions/sample count")
    dists = list(distributions or [Distribution.NORMAL] * len(mu))
    if len(dists) != len(mu):
        raise STEMUncertaintyError("distributions length mismatch")
    rng = random.Random(int(seed))
    outputs: list[float] = []
    for _ in range(samples):
        inputs: list[float] = []
        for mean, std, dist_value in zip(mu, u, dists):
            std = ensure_non_negative(std, "standard_uncertainty", error_cls=STEMUncertaintyError)
            dist = dist_value if isinstance(dist_value, Distribution) else Distribution(dist_value)
            if dist == Distribution.NORMAL:
                draw = rng.gauss(mean, std)
            elif dist == Distribution.RECTANGULAR:
                half = std * math.sqrt(3.0); draw = rng.uniform(mean - half, mean + half)
            elif dist == Distribution.TRIANGULAR:
                half = std * math.sqrt(6.0); draw = rng.triangular(mean - half, mean + half, mean)
            elif dist == Distribution.U_SHAPED:
                angle = rng.uniform(0.0, 2.0 * math.pi); draw = mean + std * math.sqrt(2.0) * math.sin(angle)
            else:
                raise STEMUncertaintyError("Unsupported distribution")
            inputs.append(draw)
        outputs.append(ensure_finite_number(model(*inputs), "model output", error_cls=STEMUncertaintyError))
    outputs.sort()
    mean_out = math.fsum(outputs) / len(outputs)
    variance = math.fsum((v - mean_out) ** 2 for v in outputs) / (len(outputs) - 1)
    lower = outputs[int(0.025 * (len(outputs) - 1))]
    upper = outputs[int(0.975 * (len(outputs) - 1))]
    return {"mean": mean_out, "standard_uncertainty": math.sqrt(variance), "coverage_interval_95": (lower, upper), "samples": samples, "seed": int(seed), "method": "JCGM101_monte_carlo"}


def numerical_error_budget(*, truncation: float = 0.0, roundoff: float = 0.0, discretization: float = 0.0, iteration: float = 0.0, other: Optional[Mapping[str, float]] = None) -> Mapping[str, Any]:
    components = {
        "truncation": ensure_non_negative(truncation, "truncation", error_cls=STEMUncertaintyError),
        "roundoff": ensure_non_negative(roundoff, "roundoff", error_cls=STEMUncertaintyError),
        "discretization": ensure_non_negative(discretization, "discretization", error_cls=STEMUncertaintyError),
        "iteration": ensure_non_negative(iteration, "iteration", error_cls=STEMUncertaintyError),
    }
    for name, value in (other or {}).items():
        components[str(name)] = ensure_non_negative(value, str(name), error_cls=STEMUncertaintyError)
    rss = math.sqrt(math.fsum(value * value for value in components.values()))
    worst_case = math.fsum(components.values())
    return {"components": components, "rss": rss, "worst_case": worst_case, "method": "independent_rss_plus_worst_case"}


class Uncertainty:
    """Configuration-aware façade for measurement-uncertainty calculations."""

    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.uncertainty_config = dict(get_config_section("stem_uncertainty", config=self.config) or {})
        if config:
            self.uncertainty_config.update(dict(config))
        self.memory = memory

    standard_uncertainty = staticmethod(standard_uncertainty)
    combined_standard_uncertainty = staticmethod(combined_standard_uncertainty)
    expanded_uncertainty = staticmethod(expanded_uncertainty)
    covariance_propagation = staticmethod(covariance_propagation)
    jacobian_propagation = staticmethod(jacobian_propagation)
    sensitivity_coefficients = staticmethod(sensitivity_coefficients)
    coverage_interval = staticmethod(coverage_interval)
    monte_carlo_propagation = staticmethod(monte_carlo_propagation)
    numerical_error_budget = staticmethod(numerical_error_budget)


__all__ = [
    "Uncertainty", "UncertaintyValue", "UncertaintyBudget", "UncertaintyType", "Distribution",
    "standard_uncertainty", "combined_standard_uncertainty", "expanded_uncertainty",
    "covariance_propagation", "jacobian_propagation", "sensitivity_coefficients",
    "coverage_interval", "monte_carlo_propagation", "numerical_error_budget",
]
