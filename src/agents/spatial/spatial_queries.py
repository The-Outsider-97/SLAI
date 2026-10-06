"""
This module should combine predicates and indexing without becoming a spatial database implementation.

sources:
- Guttman, A. (1984). “R-trees: A Dynamic Index Structure for Spatial Searching.” SIGMOD.
- Roussopoulos, N., Kelley, S., & Vincent, F. (1995). “Nearest Neighbor Queries.” SIGMOD.
- Chávez, E., Navarro, G., Baeza-Yates, R., & Marroquín, J. L. (2001). “Searching in Metric Spaces.” ACM Computing Surveys, 33(3), 273–321.
- Samet, H. (2006). Foundations of Multidimensional and Metric Data Structures.

This module could eventually expose:

nearest(...)
within_radius(...)
within_bounds(...)
intersects(...)
contained_by(...)
visible_from(...)
query_region(...)
query_relation(...)
while obtaining candidate sets from SpatialIndex through spatial memory NOT directly. avoiding circular import.
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