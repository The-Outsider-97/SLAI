"""
Deterministic quantitative biology models for SLAI STEM.

This module evaluates mathematical biology; interpretation belongs to
Reasoning and temporal world evolution belongs to Simulation.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..base.modules.biology_constraints import BiologyEngine
from .utils.config_loader import get_config_section, load_global_config
from .utils.stem_errors import STEMDomainError
from .utils.stem_helpers import *
from .stem_memory import STEMMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("SLAI Biology")
printer = PrettyPrinter()


class Biology(BiologyEngine):
    """Computational biology façade extending the existing Base BiologyEngine."""

    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        # Base environmental constraint configuration remains owned by Base.
        super().__init__(None)
        self.stem_config: Dict[str, Any] = load_global_config()
        self.biology_config = dict(get_config_section("stem_biology", config=self.stem_config) or {})
        if config:
            self.biology_config.update(dict(config))
        self.memory = memory

    @staticmethod
    def logistic_growth_rate(population: float, growth_rate: float, carrying_capacity: float) -> float:
        n = ensure_non_negative(population, "population", error_cls=STEMDomainError)
        r = ensure_finite_number(growth_rate, "growth_rate", error_cls=STEMDomainError)
        k = ensure_positive(carrying_capacity, "carrying_capacity", error_cls=STEMDomainError)
        return r * n * (1.0 - n / k)

    @staticmethod
    def logistic_population(initial: float, growth_rate: float, carrying_capacity: float, time: float) -> float:
        n0 = ensure_positive(initial, "initial", error_cls=STEMDomainError)
        r = ensure_finite_number(growth_rate, "growth_rate", error_cls=STEMDomainError)
        k = ensure_positive(carrying_capacity, "carrying_capacity", error_cls=STEMDomainError)
        t = ensure_non_negative(time, "time", error_cls=STEMDomainError)
        return k / (1.0 + ((k - n0) / n0) * math.exp(-r * t))

    @staticmethod
    def michaelis_menten(substrate: float, vmax: float, km: float) -> float:
        s = ensure_non_negative(substrate, "substrate", error_cls=STEMDomainError)
        v = ensure_non_negative(vmax, "vmax", error_cls=STEMDomainError)
        k = ensure_positive(km, "km", error_cls=STEMDomainError)
        return v * s / (k + s)

    @staticmethod
    def first_order_decay(initial: float, rate_constant: float, time: float) -> float:
        c0 = ensure_non_negative(initial, "initial", error_cls=STEMDomainError)
        k = ensure_non_negative(rate_constant, "rate_constant", error_cls=STEMDomainError)
        t = ensure_non_negative(time, "time", error_cls=STEMDomainError)
        return c0 * math.exp(-k * t)

    @staticmethod
    def lotka_volterra_rates(prey: float, predator: float, alpha: float, beta: float, delta: float, gamma: float) -> Tuple[float, float]:
        x = ensure_non_negative(prey, "prey", error_cls=STEMDomainError)
        y = ensure_non_negative(predator, "predator", error_cls=STEMDomainError)
        a = ensure_non_negative(alpha, "alpha", error_cls=STEMDomainError)
        b = ensure_non_negative(beta, "beta", error_cls=STEMDomainError)
        d = ensure_non_negative(delta, "delta", error_cls=STEMDomainError)
        g = ensure_non_negative(gamma, "gamma", error_cls=STEMDomainError)
        return a * x - b * x * y, d * x * y - g * y

    @staticmethod
    def compartment_balance(inflows: Sequence[float], outflows: Sequence[float], generation: float = 0.0, consumption: float = 0.0) -> float:
        ins = [ensure_finite_number(v, "inflow", error_cls=STEMDomainError) for v in inflows]
        outs = [ensure_finite_number(v, "outflow", error_cls=STEMDomainError) for v in outflows]
        gen = ensure_non_negative(generation, "generation", error_cls=STEMDomainError)
        con = ensure_non_negative(consumption, "consumption", error_cls=STEMDomainError)
        return math.fsum(ins) - math.fsum(outs) + gen - con


__all__ = ["Biology"]
    """"Mathematical Models in Biology""""


__all__ = ["Biology"]