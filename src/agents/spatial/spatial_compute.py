"""
This nodule is an orchestration layer around reusable deterministic spatial mathematics.

sources:
- de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). Computational Geometry: Algorithms and Applications, 3rd ed. Springer.
- Burago, D., Burago, Y., & Ivanov, S. (2001). A Course in Metric Geometry. AMS.

Spatial Compute itself should not contain all those algorithms;
it should coordinate Calculations, geometry, transforms and other deterministic operations.
"""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Optional

from .modules.calculations import Calculations
from .modules.transform import *
from .world.geometry import *
from .utils.config_loader import load_global_config, get_config_section
from .utils.spatial_errors import *
from .utils.spatial_helpers import *
from .spatial_memory import *
from .spatial_types import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Compute")
printer = PrettyPrinter()


class SpatialCompute:
    def __init__(self, calculations: Optional[Calculations]=None, memory: Optional[SpatialMemory]=None) -> None:
        self.config = load_global_config()
        self.relations_config = get_config_section("spatial_calculations", config=self.config, default={})
        self.calculator = calculations
        self.memory = memory

        logger.info(f"Spatial calculatons successfully initialized")


__all__ = ["SpatialCompute"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Spatial Compute ===\n")
    printer.status("TEST", "Spatial Compute initialized", "info")