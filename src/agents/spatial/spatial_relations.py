"""

"""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Optional

from .utils.config_loader import load_global_config, get_config_section
from .utils.spatial_errors import *
from .utils.spatial_helpers import *
from .spatial_memory import *
from .spatial_types import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Relations")
printer = PrettyPrinter()


class SpatialRelations:
    def __init__(self, memory: Optional[SpatialMemory]=None) -> None:
        self.config = load_global_config()
        self.relations_config = get_config_section("spatial_relations", config=self.config, default={})
        self.memory = memory

        logger.info(f"Spatial relations successfully initialized")


__all__ = ["SpatialRelations"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Spatial Relations ===\n")
    printer.status("TEST", "Spatial Relations initialized", "info")