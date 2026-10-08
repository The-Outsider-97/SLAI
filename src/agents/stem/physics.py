"""Deterministic physical equations and constants for SLAI STEM.

The class extends SLAI's existing PhysicsEngine. It does not own scenario/world
rollout; SimulationAgent may consume these equations and constants.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
from typing import Any, Dict, Mapping, Optional

from ..base.modules.physics_constraints import PhysicsEngine
from .utils.config_loader import get_config_section, load_global_config
from .utils.stem_errors import STEMDomainError
from .utils.stem_helpers import *
from .stem_memory import STEMMemory
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Physics")
printer = PrettyPrinter()


class Physics(PhysicsEngine):
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        super().__init__(None)
        self.stem_config: Dict[str, Any] = load_global_config()
        self.physics_config_stem = dict(get_config_section("stem_physics", config=self.stem_config) or {})
        if config:
            self.physics_config_stem.update(dict(config))
        self.memory = memory

    @staticmethod
    def kinetic_energy(mass: float, velocity: float) -> float:
        m = ensure_non_negative(mass, "mass", error_cls=STEMDomainError); v = ensure_finite_number(velocity, "velocity", error_cls=STEMDomainError)
        return 0.5 * m * v * v

    @staticmethod
    def momentum(mass: float, velocity: float) -> float:
        return ensure_non_negative(mass, "mass", error_cls=STEMDomainError) * ensure_finite_number(velocity, "velocity", error_cls=STEMDomainError)

    def gravitational_force(self, mass_a: float, mass_b: float, distance: float) -> float:
        m1 = ensure_non_negative(mass_a, "mass_a", error_cls=STEMDomainError); m2 = ensure_non_negative(mass_b, "mass_b", error_cls=STEMDomainError); r = ensure_positive(distance, "distance", error_cls=STEMDomainError)
        return float(self.CONSTANTS["G"]) * m1 * m2 / (r * r)

    def ideal_gas_pressure(self, moles: float, temperature: float, volume: float) -> float:
        n = ensure_non_negative(moles, "moles", error_cls=STEMDomainError); t = ensure_positive(temperature, "temperature", error_cls=STEMDomainError); v = ensure_positive(volume, "volume", error_cls=STEMDomainError)
        return n * float(self.CONSTANTS["R"]) * t / v

    @staticmethod
    def relativistic_gamma(velocity: float, c: float = 299792458.0) -> float:
        v = abs(ensure_finite_number(velocity, "velocity", error_cls=STEMDomainError)); speed = ensure_positive(c, "c", error_cls=STEMDomainError)
        if v >= speed: raise STEMDomainError("velocity magnitude must be less than c")
        return 1.0 / math.sqrt(1.0 - (v / speed) ** 2)


__all__ = ["Physics"]
