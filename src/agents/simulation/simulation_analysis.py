"""Descriptive analysis of simulation output.

This module summarizes generated trajectories and stochastic samples without
judging which action, plan, or outcome is preferable.  Statistical output
analysis follows Law (1983, 2015) and Heidelberger & Welch (1983).  Sensitivity
methods implement the definitions of Morris (1991), Sobol (2001), Saltelli et
al. (2008), and Jansen (1999); malformed experimental designs are rejected rather than being
silently approximated.
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import numpy as np

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any, Optional, cast

from .simulation_types import *
from .utils.config_loader import get_config_section, load_global_config
from .utils.simulation_errors import *
from .utils.simulation_helpers import numeric_state_vector
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Simulation Analysis")


class SimulationAnalysis:
    """Numerically defensive descriptive analysis for simulation results."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("simulation_analysis", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.confidence_level = float(section.get("confidence_level", 0.95))
        self.convergence_tolerance = float(section.get("convergence_tolerance", 0.01))
        self.convergence_min_samples = int(section.get("convergence_min_samples", 30))
        self.convergence_window = int(section.get("convergence_window", 20))
        if not 0.0 < self.confidence_level < 1.0:
            raise SimulationValidationError("confidence_level must be between 0 and 1")
        if self.convergence_tolerance <= 0.0:
            raise SimulationValidationError("convergence_tolerance must be > 0")
        if self.convergence_min_samples < 2 or self.convergence_window < 2:
            raise SimulationValidationError("convergence sample settings must be >= 2")

    @staticmethod
    def _finite_1d(values: Sequence[float] | np.ndarray, *, name: str = "values") -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise SimulationValidationError(f"{name} must be a non-empty one-dimensional numeric sequence")
        if not bool(np.all(np.isfinite(array))):
            raise SimulationNumericalError(f"{name} contains NaN or infinity")
        return array

    def summarize_distribution(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        confidence_level: Optional[float] = None,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    ) -> DistributionSummary:
        array = self._finite_1d(values)
        confidence = self.confidence_level if confidence_level is None else float(confidence_level)
        if not 0.0 < confidence < 1.0:
            raise SimulationValidationError("confidence_level must be between 0 and 1")
        q_values: dict[str, float] = {}
        for q in quantiles:
            probability = float(q)
            if not 0.0 <= probability <= 1.0:
                raise SimulationValidationError("quantiles must be between 0 and 1")
            q_values[f"q{probability:g}"] = float(np.quantile(array, probability))
        count = int(array.size)
        mean = float(np.mean(array, dtype=np.float64))
        variance = float(np.var(array, ddof=1 if count > 1 else 0, dtype=np.float64))
        std = math.sqrt(max(0.0, variance))
        if count > 1:
            standard_error = std / math.sqrt(count)
            z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
            interval = (mean - z * standard_error, mean + z * standard_error)
        else:
            interval = (mean, mean)
        return DistributionSummary(
            count=count,
            mean=mean,
            variance=variance,
            standard_deviation=std,
            minimum=float(np.min(array)),
            maximum=float(np.max(array)),
            quantiles=q_values,
            confidence_interval=(float(interval[0]), float(interval[1])),
        )

    def convergence_diagnostics(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        tolerance: Optional[float] = None,
        minimum_samples: Optional[int] = None,
        window: Optional[int] = None,
    ) -> ConvergenceDiagnostics:
        array = self._finite_1d(values)
        tol = self.convergence_tolerance if tolerance is None else float(tolerance)
        minimum = self.convergence_min_samples if minimum_samples is None else int(minimum_samples)
        recent_window = self.convergence_window if window is None else int(window)
        if tol <= 0.0 or minimum < 2 or recent_window < 2:
            raise SimulationValidationError("invalid convergence settings")
        n = int(array.size)
        mean = float(np.mean(array))
        std = float(np.std(array, ddof=1)) if n > 1 else 0.0
        standard_error = std / math.sqrt(n) if n else math.inf
        relative_error = None if abs(mean) <= np.finfo(np.float64).eps else abs(standard_error / mean)
        shift: Optional[float] = None
        if n >= 2 * recent_window:
            previous = float(np.mean(array[-2 * recent_window : -recent_window]))
            recent = float(np.mean(array[-recent_window:]))
            scale = max(abs(mean), np.finfo(np.float64).eps)
            shift = abs(recent - previous) / scale
        converged = (
            n >= minimum
            and relative_error is not None
            and relative_error <= tol
            and (shift is None or shift <= tol)
        )
        return ConvergenceDiagnostics(
            sample_count=n,
            cumulative_mean=mean,
            standard_error=float(standard_error),
            relative_standard_error=None if relative_error is None else float(relative_error),
            recent_mean_shift=None if shift is None else float(shift),
            converged=bool(converged),
            tolerance=tol,
        )

    def summarize_monte_carlo(self, batch: MonteCarloResult, extractors: Mapping[str, Any]) -> MonteCarloResult:
        """Attach descriptive summaries for caller-defined scalar observables.

        ``extractors`` map labels to callables accepting ``SimulationResult``.
        The function never interprets a scalar as reward, preference, or utility.
        """
        if not isinstance(batch, MonteCarloResult):
            raise SimulationValidationError("batch must be a MonteCarloResult")
        summaries: dict[str, DistributionSummary] = {}
        convergence: dict[str, ConvergenceDiagnostics] = {}
        for name, extractor in extractors.items():
            if not callable(extractor):
                raise SimulationValidationError("Monte Carlo extractor must be callable", context={"name": name})
            values = [float(cast(Any, extractor(result))) for result in batch.results]
            if not values:
                continue
            summaries[str(name)] = self.summarize_distribution(values)
            convergence[str(name)] = self.convergence_diagnostics(values)
        return MonteCarloResult(
            batch_id=batch.batch_id,
            root_seed=batch.root_seed,
            sample_seeds=batch.sample_seeds,
            results=batch.results,
            failures=batch.failures,
            summaries=summaries,
            convergence=convergence,
            metadata=batch.metadata,
        )

    @staticmethod
    def branch_frequencies(scenario_ids: Sequence[str]) -> dict[str, int]:
        return dict(Counter(str(item) for item in scenario_ids))

    @staticmethod
    def trajectory_divergence(left: Trajectory, right: Trajectory) -> dict[str, float | int]:
        if not isinstance(left, Trajectory) or not isinstance(right, Trajectory):
            raise SimulationValidationError("trajectory_divergence requires two Trajectory instances")
        paired = min(len(left.steps), len(right.steps))
        squared: list[float] = []
        absolute: list[float] = []
        compared = 0
        for index in range(paired):
            a = numeric_state_vector(left.steps[index].state)
            b = numeric_state_vector(right.steps[index].state)
            if a.size != b.size:
                raise SimulationValidationError(
                    "numeric trajectory state dimensions differ",
                    context={"step": index, "left": int(a.size), "right": int(b.size)},
                )
            if a.size == 0:
                continue
            delta = a - b
            squared.extend(float(value * value) for value in delta)
            absolute.extend(float(abs(value)) for value in delta)
            compared += int(a.size)
        if compared == 0:
            return {"compared_values": 0, "rmse": 0.0, "mean_absolute_difference": 0.0, "max_absolute_difference": 0.0}
        return {
            "compared_values": compared,
            "rmse": math.sqrt(sum(squared) / compared),
            "mean_absolute_difference": sum(absolute) / compared,
            "max_absolute_difference": max(absolute),
        }

    def oat_effects(
        self,
        baseline_output: float,
        perturbed_outputs: Mapping[str, float],
        parameter_deltas: Mapping[str, float],
    ) -> SensitivityEstimate:
        """Compute one-at-a-time finite effects without selecting parameters."""
        baseline = float(baseline_output)
        if not math.isfinite(baseline):
            raise SimulationNumericalError("baseline_output must be finite")
        effects: dict[str, float] = {}
        for name, output in perturbed_outputs.items():
            if name not in parameter_deltas:
                raise SimulationValidationError("missing perturbation delta", context={"parameter": name})
            delta = float(parameter_deltas[name])
            value = float(output)
            if not math.isfinite(value) or not math.isfinite(delta) or delta == 0.0:
                raise SimulationValidationError("OAT outputs must be finite and deltas non-zero", context={"parameter": name})
            effects[str(name)] = (value - baseline) / delta
        return SensitivityEstimate(method="one_at_a_time", elementary_mean=effects, metadata={"baseline": baseline})

    def morris_elementary_effects(
        self,
        parameter_names: Sequence[str],
        input_paths: Sequence[Sequence[Sequence[float]]] | np.ndarray,
        output_paths: Sequence[Sequence[float]] | np.ndarray,
        *,
        change_tolerance: float = 1.0e-12,
    ) -> SensitivityEstimate:
        """Compute Morris elementary effects from externally generated trajectories.

        Each adjacent point must change exactly one input factor.  This validates
        the Morris design instead of synthesizing unrelated approximations.
        """
        names = tuple(str(name) for name in parameter_names)
        if not names or len(set(names)) != len(names):
            raise SimulationValidationError("parameter_names must be unique and non-empty")
        x = np.asarray(input_paths, dtype=np.float64)
        y = np.asarray(output_paths, dtype=np.float64)
        if x.ndim != 3 or x.shape[2] != len(names) or y.ndim != 2 or y.shape != x.shape[:2]:
            raise SimulationValidationError(
                "Morris design must have inputs shape (trajectories, points, parameters) and matching outputs",
                context={"input_shape": tuple(x.shape), "output_shape": tuple(y.shape)},
            )
        if x.shape[1] < 2 or not bool(np.all(np.isfinite(x))) or not bool(np.all(np.isfinite(y))):
            raise SimulationValidationError("Morris design must contain finite trajectories with at least two points")
        effects: dict[str, list[float]] = defaultdict(list)
        for trajectory_index in range(x.shape[0]):
            for step in range(x.shape[1] - 1):
                delta = x[trajectory_index, step + 1] - x[trajectory_index, step]
                changed = np.flatnonzero(np.abs(delta) > change_tolerance)
                if changed.size != 1:
                    raise SimulationValidationError(
                        "each Morris path step must change exactly one parameter",
                        context={"trajectory": trajectory_index, "step": step, "changed_parameters": changed.tolist()},
                    )
                parameter_index = int(changed[0])
                dx = float(delta[parameter_index])
                dy = float(y[trajectory_index, step + 1] - y[trajectory_index, step])
                effects[names[parameter_index]].append(dy / dx)
        missing = [name for name in names if not effects[name]]
        if missing:
            raise SimulationValidationError("Morris design did not perturb every parameter", context={"missing": missing})
        mu = {name: float(np.mean(effects[name])) for name in names}
        mu_star = {name: float(np.mean(np.abs(effects[name]))) for name in names}
        sigma = {
            name: float(np.std(effects[name], ddof=1 if len(effects[name]) > 1 else 0))
            for name in names
        }
        return SensitivityEstimate(
            method="morris",
            elementary_mean=mu,
            elementary_mean_abs=mu_star,
            elementary_std=sigma,
            metadata={"trajectory_count": int(x.shape[0])},
        )

    def sobol_indices(
        self,
        y_a: Sequence[float] | np.ndarray,
        y_b: Sequence[float] | np.ndarray,
        y_ab: Mapping[str, Sequence[float] | np.ndarray],
    ) -> SensitivityEstimate:
        """Estimate first- and total-order Sobol indices.

        ``y_ab[p]`` must be model output for matrix A with parameter ``p``
        replaced by its column from B.  First-order estimation uses the Saltelli
        covariance form; total-order estimation uses the Jansen squared-difference
        estimator.  No ranking or parameter selection is performed.
        """
        a = self._finite_1d(y_a, name="y_a")
        b = self._finite_1d(y_b, name="y_b")
        if a.shape != b.shape or a.size < 2:
            raise SimulationValidationError("Sobol A/B outputs must have equal length >= 2")
        variance = float(np.var(np.concatenate((a, b)), ddof=1))
        if not math.isfinite(variance) or variance <= np.finfo(np.float64).eps:
            raise SimulationNumericalError("Sobol output variance is zero or numerically degenerate")
        first: dict[str, float] = {}
        total: dict[str, float] = {}
        for name, values in y_ab.items():
            ab = self._finite_1d(values, name=f"y_ab[{name}]")
            if ab.shape != a.shape:
                raise SimulationValidationError("Sobol hybrid outputs must match A/B length", context={"parameter": name})
            first[str(name)] = float(np.mean(b * (ab - a)) / variance)
            total[str(name)] = float(0.5 * np.mean((a - ab) ** 2) / variance)
        return SensitivityEstimate(
            method="sobol_saltelli_jansen",
            first_order=first,
            total_order=total,
            metadata={"sample_count": int(a.size), "variance_estimate": variance},
        )


__all__ = ["SimulationAnalysis"]
