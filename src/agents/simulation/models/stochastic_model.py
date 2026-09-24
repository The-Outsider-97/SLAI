"""Stochastic environment primitives for SLAI Simulation.

The module represents stochastic transition kernels, parameter distributions,
noise, and exact event-time sampling.  Puterman (1994) motivates transition
kernels; Gillespie (1977) motivates rate-based exact next-event sampling; and
Robert & Casella (2004) provide the Monte Carlo sampling foundation.  Output
uncertainty is summarized elsewhere rather than evaluated here.
"""

from __future__ import annotations

__version__ = "2.3.0"

import math
import numpy as np

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..simulation_types import TransitionOutcome
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Stochastic Model")


class DistributionKind(str, Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    BERNOULLI = "bernoulli"
    CATEGORICAL = "categorical"


@dataclass(frozen=True, slots=True)
class DistributionSpec:
    kind: DistributionKind
    parameters: Mapping[str, Any] = field(default_factory=dict)
    values: tuple[Any, ...] = ()
    probabilities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DistributionKind(self.kind))
        if self.kind is DistributionKind.CATEGORICAL:
            if not self.values or len(self.values) != len(self.probabilities):
                raise SimulationValidationError("categorical distribution requires equally sized values and probabilities")
            validate_probabilities(self.probabilities)


@dataclass(frozen=True, slots=True)
class StochasticEvent:
    event: str
    waiting_time: float
    total_rate: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionDistribution:
    """Finite categorical representation of a stochastic transition kernel."""

    states: tuple[Any, ...]
    probabilities: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.states or len(self.states) != len(self.probabilities):
            raise SimulationValidationError("transition distribution requires equally sized states and probabilities")
        validate_probabilities(self.probabilities)

    def sample(self, rng: np.random.Generator) -> Any:
        probabilities = validate_probabilities(self.probabilities)
        index = int(rng.choice(len(self.states), p=probabilities))
        return clone_state(self.states[index])


class StochasticModel:
    """Reusable stochastic kernel and distribution sampler."""

    model_id = "stochastic-model"
    version = __version__
    stochastic = True
    thread_safe = False

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        transition_kernel: Optional[Callable[..., Any]] = None,
    ) -> None:
        global_config = load_global_config()
        section = dict(get_config_section("stochastic_model", config=global_config) or {})
        if config:
            section.update(dict(config))
        self.probability_tolerance = float(section.get("probability_tolerance", 1.0e-9))
        if self.probability_tolerance <= 0.0:
            raise SimulationValidationError("probability_tolerance must be > 0")
        self.transition_kernel = transition_kernel
        logger.info("Stochastic model initialized")

    def sample_distribution(self, spec: DistributionSpec, rng: np.random.Generator, *, size: Any = None) -> Any:
        if not isinstance(rng, np.random.Generator):
            raise SimulationValidationError("rng must be numpy.random.Generator")
        params = dict(spec.parameters)
        kind = DistributionKind(spec.kind)
        try:
            if kind is DistributionKind.NORMAL:
                scale = float(params.get("scale", params.get("std", 1.0)))
                if scale < 0.0:
                    raise SimulationValidationError("normal scale must be >= 0")
                return rng.normal(float(params.get("loc", params.get("mean", 0.0))), scale, size=size)
            if kind is DistributionKind.UNIFORM:
                low, high = float(params.get("low", 0.0)), float(params.get("high", 1.0))
                if not high > low:
                    raise SimulationValidationError("uniform high must be greater than low")
                return rng.uniform(low, high, size=size)
            if kind is DistributionKind.LOGNORMAL:
                sigma = float(params.get("sigma", 1.0))
                if sigma < 0.0:
                    raise SimulationValidationError("lognormal sigma must be >= 0")
                return rng.lognormal(float(params.get("mean", 0.0)), sigma, size=size)
            if kind is DistributionKind.EXPONENTIAL:
                scale = float(params.get("scale", 1.0))
                if scale <= 0.0:
                    raise SimulationValidationError("exponential scale must be > 0")
                return rng.exponential(scale, size=size)
            if kind is DistributionKind.BERNOULLI:
                p = float(params.get("p", 0.5))
                if not 0.0 <= p <= 1.0:
                    raise SimulationProbabilityError("Bernoulli p must be in [0, 1]")
                return rng.binomial(1, p, size=size)
            if kind is DistributionKind.CATEGORICAL:
                p = validate_probabilities(spec.probabilities, tolerance=self.probability_tolerance)
                if size is None:
                    return clone_state(spec.values[int(rng.choice(len(spec.values), p=p))])
                indices = rng.choice(len(spec.values), size=size, p=p)
                return np.asarray([spec.values[int(index)] for index in np.asarray(indices).ravel()], dtype=object).reshape(np.shape(indices))
        except (SimulationValidationError, SimulationProbabilityError):
            raise
        except (ValueError, FloatingPointError, OverflowError) as exc:
            raise SimulationNumericalError("stochastic sampling failed", cause=exc, context={"kind": kind.value}) from exc
        raise SimulationValidationError("unsupported distribution kind", context={"kind": kind.value})

    def sample_noise(
        self,
        shape: int | tuple[int, ...],
        *,
        scale: float = 1.0,
        mean: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if scale < 0.0 or not math.isfinite(float(scale)) or not math.isfinite(float(mean)):
            raise SimulationValidationError("noise mean/scale must be finite and scale >= 0")
        if rng is not None and seed is not None:
            raise SimulationValidationError("provide either rng or seed, not both")
        generator = rng if rng is not None else make_rng(seed)[0]
        return np.asarray(generator.normal(float(mean), float(scale), size=shape), dtype=np.float64)

    def sample_transition(
        self,
        *,
        state: Any,
        action: Any,
        interventions: tuple[Any, ...] = (),
        parameters: Optional[Mapping[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
        step: int = 0,
        simulation_time: float = 0.0,
        dt: float = 1.0,
    ) -> TransitionOutcome:
        if self.transition_kernel is None:
            raise SimulationModelError("no stochastic transition kernel is configured")
        generator = rng if rng is not None else make_rng()[0]
        raw = invoke_with_supported_kwargs(
            self.transition_kernel,
            state=state,
            action=action,
            interventions=interventions,
            parameters=dict(parameters or {}),
            rng=generator,
            step=step,
            simulation_time=simulation_time,
            dt=dt,
        )
        if isinstance(raw, TransitionOutcome):
            return raw
        if isinstance(raw, TransitionDistribution):
            return TransitionOutcome(state=raw.sample(generator), metadata=dict(raw.metadata))
        return TransitionOutcome(state=raw)


    def transition(
        self,
        *,
        state: Any,
        action: Any,
        interventions: tuple[Any, ...],
        parameters: Mapping[str, Any],
        rng: np.random.Generator,
        step: int,
        simulation_time: float,
        dt: float,
    ) -> TransitionOutcome:
        """SimulationModel-compatible stochastic transition entry point."""
        return self.sample_transition(
            state=state,
            action=action,
            interventions=interventions,
            parameters=parameters,
            rng=rng,
            step=step,
            simulation_time=simulation_time,
            dt=dt,
        )

    def initialize(self, *, initial_state: Any, parameters: Mapping[str, Any], rng: np.random.Generator) -> Any:
        return clone_state(initial_state)

    def observe(self, *, state: Any) -> Any:
        return clone_state(state)

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "stochastic": True,
            "thread_safe": self.thread_safe,
            "has_transition_kernel": self.transition_kernel is not None,
        }

    def next_event(
        self,
        rates: Mapping[str, float],
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> StochasticEvent:
        """Sample the next event from non-negative propensities (Gillespie direct method)."""
        if not rates:
            raise SimulationValidationError("rates must be non-empty")
        names = tuple(str(name) for name in rates)
        values = np.asarray([float(rates[name]) for name in rates], dtype=np.float64)
        if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0.0)):
            raise SimulationValidationError("event rates must be finite and non-negative")
        total = float(np.sum(values))
        if total <= 0.0:
            raise SimulationValidationError("at least one event rate must be positive")
        if rng is not None and seed is not None:
            raise SimulationValidationError("provide either rng or seed, not both")
        generator = rng if rng is not None else make_rng(seed)[0]
        waiting_time = float(generator.exponential(1.0 / total))
        event_probabilities = normalize_weights(values)
        index = int(generator.choice(len(names), p=event_probabilities))
        return StochasticEvent(
            event=names[index],
            waiting_time=waiting_time,
            total_rate=total,
            metadata=dict(metadata or {}),
        )


__all__ = [
    "DistributionKind",
    "DistributionSpec",
    "StochasticEvent",
    "StochasticModel",
    "TransitionDistribution",
]
