"""
Subsystem memory module for caching, cotext-awere, and checkpointing. It store spatial state and derived spatial artifacts, like:
entity poses
reference frames
geometry handles
occupancy representations
map versions
spatial relations
cached query results
index snapshots
transform timestamps

sources:
- The Spatial Semantic Hierarchy by Benjamin Kuipers
- Hornung, A., Wurm, K.M., Bennewitz, M. et al. OctoMap: an efficient probabilistic 3D mapping framework based on octrees. Auton Robot 34, 189–206 (2013). https://doi.org/10.1007/s10514-012-9321-0

Spatial memory is the SLAI architectural concern, not a research algorithm.
"""
from __future__ import annotations

__version__ = "2.3.0"


from .utils.config_loader import *
from .utils.spatial_errors import *
from .utils.spatial_helpers import *
from .spatial_types import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Memory")
printer = PrettyPrinter()


class SpatialMemory:
    def __init__(self) -> None:
        self.config = load_global_config()
        self.memory_config = get_config_section("spatial_memory", config=self.config, default={})

        logger.info(f"Spatial Memory successfully initialized")


__all__ = ["SpatialMemory"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Spatial Memory ===\n")
    printer.status("TEST", "Spatial Memory initialized", "info")