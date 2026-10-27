"""Explicit composition boundary for optional SLAI spatial-format services.

Nothing initializes an agent merely because ``spatial_formats`` was imported.
Applications opt in to individual integrations by calling a factory. The
Spatial and Provenance integrations can be added here independently later.
"""

from __future__ import annotations

from .slai_stem import STEMQuantityAdapter, STEMUnitService


def build_stem_adapter(unit_system: STEMUnitService | None = None) -> STEMQuantityAdapter:
    """Connect the native spatial formats to SLAI's real STEM ``UnitSystem``.

    A caller may provide its configured unit system, avoiding extra registry
    instances. Without one, this explicit factory constructs the actual STEM
    service using its existing configuration loader. Importing the package or
    the adapter does not trigger that scientific stack.
    """

    if unit_system is None:
        from src.agents.stem.units.unit_system import UnitSystem

        unit_system = UnitSystem()
    return STEMQuantityAdapter(unit_system)


__all__ = ["build_stem_adapter"]
