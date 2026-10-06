"""

"""
from __future__ import annotations

__version__ = "2.3.0"


from .utils.spatial_errors import *
from .utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Spatial Types")
printer = PrettyPrinter()


class SpatialEntity:
    pass


class SpatialActivity:
    pass


class SpatialRecords:
    pass


class SpatialRelationship:
    pass


class SpatialBundle:
    pass


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