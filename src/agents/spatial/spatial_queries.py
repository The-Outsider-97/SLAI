"""

"""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Optional

from .utils.config_loader import *
from .utils.spatial_errors import *
from .utils.spatial_helpers import *
from .spatial_memory import *
from .spatial_types import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Queries")
printer = PrettyPrinter()


class SpatialQueries:
    def __init__(self, memory: Optional[SpatialMemory]=None) -> None:
        self.config = load_global_config()
        self.queries_config = get_config_section("spatial_queries", config=self.config, default={})
        self.memory = memory

        logger.info(f"Spatial queries successfully initialized")


__all__ = ["SpatialQueries"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Spatial Queries ===\n")
    printer.status("TEST", "Spatial Queries initialized", "info")