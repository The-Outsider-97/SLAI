"""
sources:
- Bentley, J. L. (1975). “Multidimensional Binary Search Trees Used for Associative Searching.” Communications of the ACM, 18(9), 509–517. This is the foundational k-d tree paper.
- Guttman (1984) for R-trees.
- Samet (2006) for quadtrees, octrees, R-trees, k-d trees, high-dimensional indexing and metric structures. Its scope maps remarkably well to what SpatialIndex should eventually provide.
- Chávez et al. (2001) for metric-space indexing.

It allows several backend indexes:

SpatialIndex
├── KDTree
├── RTree
├── Octree
├── UniformGrid
└── MetricIndex
but expose a common SLAI-facing interface.

Selection of the index should depend on representation/query type, rather than having one supposedly universal structure.
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

logger = get_logger("Spatial Index")
printer = PrettyPrinter()


class SpatialIndex:
    def __init__(self, memory: Optional[SpatialMemory]=None) -> None:
        self.config = load_global_config()
        self.index_config = get_config_section("spatial_index", config=self.config, default={})
        self.memory = memory

        logger.info(f"Spatial index successfully initialized")


__all__ = ["SpatialIndex"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Spatial Index ===\n")
    printer.status("TEST", "Spatial Index initialized", "info")