"""
Spatial Types defines the canonical data contracts exchanged throughout the subsystem.

sources:
- ISO 19107:2019 — Geographic information — Spatial schema.
- OGC Simple Feature Access / ISO 19125.
- Egenhofer, M. J., & Franzosa, R. D. (1991). “Point-set topological spatial relations.” International Journal of Geographical Information Systems, 5(2), 161–174.
"""
from __future__ import annotations

__version__ = "2.3.0"


from .utils.spatial_errors import *
from .utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Types")
printer = PrettyPrinter()


class SpatialEntity:
    """
    SpatialEntity
    ├── PointEntity
    ├── LinearEntity
    ├── SurfaceEntity
    ├── VolumeEntity
    └── CompositeEntity
    """


class SpatialActivity:
    pass


class SpatialRecords:
    pass


class SpatialRelationship:
    """
    SpatialRelationship
    ├── MetricRelationship
    ├── DirectionalRelationship
    └── TopologicalRelationship
    """


class SpatialBundle:
    pass


class SpatialReference:
    """
    SpatialReference
    ├── CRSReference
    └── LocalFrameReference
    """


class SpatialTypes:
    pass


__all__ = [
    "SpatialEntity",
    "SpatialActivity",
    "SpatialRecords",
    "SpatialRelationship",
    "SpatialBundle",
    "SpatialTypes",
]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Spatial Types ===\n")
    printer.status("TEST", "Spatial Types initialized", "info")