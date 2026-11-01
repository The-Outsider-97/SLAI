"""
Deterministic engineering computations for SLAI STEM.

The class extends the current Base EngineeringEngine rather than duplicating
its civil/mechanical/electronics/computer/medical engineering implementations.
"""
from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..base.modules.base_engineering import EngineeringEngine
from .utils.config_loader import get_config_section, load_global_config
from .utils.stem_errors import STEMDomainError
from .utils.stem_helpers import ensure_finite_number, ensure_positive
from .stem_memory import STEMMemory
from .stem_types import Quantity, Unit
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Engineering")
printer = PrettyPrinter()


@dataclass(frozen=True)
class EngineeringQuantity:
    name: str
    quantity: Quantity
    method: str


class Engineering(EngineeringEngine):
    def __init__(self, config: Mapping[str, Any] | None = None, memory: Optional[STEMMemory] = None) -> None:
        # Base engineering configuration remains owned by BaseEngine.
        super().__init__(None)
        self.stem_config: Dict[str, Any] = load_global_config()
        self.engineering_config_stem = dict(get_config_section("stem_engineering", config=self.stem_config) or {})
        if config:
            self.engineering_config_stem.update(dict(config))
        self.memory = memory

    @staticmethod
    def axial_stress(force: float, area: float) -> float:
        return ensure_finite_number(force, "force", error_cls=STEMDomainError) / ensure_positive(area, "area", error_cls=STEMDomainError)

    @staticmethod
    def heat_conduction_1d(conductivity: float, area: float, delta_temperature: float, length: float) -> float:
        k = ensure_positive(conductivity, "conductivity", error_cls=STEMDomainError)
        a = ensure_positive(area, "area", error_cls=STEMDomainError)
        dt = ensure_finite_number(delta_temperature, "delta_temperature", error_cls=STEMDomainError)
        l = ensure_positive(length, "length", error_cls=STEMDomainError)
        return k * a * dt / l

    @staticmethod
    def electrical_power(voltage: float, current: float) -> float:
        return ensure_finite_number(voltage, "voltage", error_cls=STEMDomainError) * ensure_finite_number(current, "current", error_cls=STEMDomainError)

    @staticmethod
    def reynolds_number(density: float, velocity: float, characteristic_length: float, dynamic_viscosity: float) -> float:
        rho = ensure_positive(density, "density", error_cls=STEMDomainError)
        v = ensure_finite_number(velocity, "velocity", error_cls=STEMDomainError)
        length = ensure_positive(characteristic_length, "characteristic_length", error_cls=STEMDomainError)
        mu = ensure_positive(dynamic_viscosity, "dynamic_viscosity", error_cls=STEMDomainError)
        return rho * abs(v) * length / mu


__all__ = ["Engineering", "EngineeringQuantity"]
