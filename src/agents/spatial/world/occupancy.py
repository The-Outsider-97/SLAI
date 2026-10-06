"""
sources:
- Moravec, H., & Elfes, A. (1985). “High Resolution Maps from Wide Angle Sonar.” ICRA.
- Elfes, A. (1989). “Using Occupancy Grids for Mobile Robot Perception and Navigation.” Computer, 22(6), 46–57.
- Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., & Burgard, W. (2013). “OctoMap: an efficient probabilistic 3D mapping framework based on octrees.” Autonomous Robots, 34, 189–206.
- Oleynikova et al. (2017). “Voxblox: Incremental 3D Euclidean Signed Distance Fields for on-board MAV planning.” IROS.

occupancy exposes:
OccupancyGrid2D
OccupancyGrid3D
VoxelGrid
OctreeOccupancy
TSDFVolume
ESDFVolume

while sensor interpretation remains Perception's responsibility.
"""
from __future__ import annotations

__version__ = "2.3.0"


from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("(Occupancy)")
printer = PrettyPrinter()


__all__ = []


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Occupancy ===\n")
    printer.status("TEST", "Occupancy initialized", "info")