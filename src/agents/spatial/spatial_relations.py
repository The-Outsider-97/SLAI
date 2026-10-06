"""
Sources:
- Egenhofer & Franzosa (1991)
- Randell, D. A., Cui, Z., & Cohn, A. G. (1992). “A Spatial Logic Based on Regions and Connection.” KR'92, pp. 165–176.
- Clementini, E., Di Felice, P., & van Oosterom, P. (1993). “A Small Set of Formal Topological Relationships Suitable for End-User Interaction.”
- Cohn & Renz (2008) should provide the broader qualitative-spatial-reasoning framework.

point-set/topological predicates:
disjoint
touches
inside
contains
overlap
equal

the module handles relationships such as:
relate(a, b)
contains(a, b)
inside(a, b)
touches(a, b)
overlaps(a, b)
disconnected(a, b)
near(a, b)
far(a, b)
left_of(a, b)
right_of(a, b)
above(a, b)
below(a, b)

but the low-level geometric intersection computation should come from geometry.py/topology.py
"""
from __future__ import annotations

__version__ = "2.3.0"

from typing import Optional

from .utils.config_loader import load_global_config, get_config_section
from .utils.spatial_errors import *
from .utils.spatial_helpers import *
from .world.geometry import *
from .world.topology import *
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