"""
This a thin spatial mathematical layer over the existing SLAI base math/science facilities.

Sources:
- de Berg et al. (2008) for computational geometry.
- Burago et al. (2001) for metric spaces and distance.
- Ericson, C. (2005). Real-Time Collision Detection. Morgan Kaufmann.

Calculations provides deterministic primitives such as:

euclidean_distance
squared_distance
dot/cross
projection
closest_point
signed_distance
point_segment_distance
point_plane_distance
aabb_distance
orientation tests
barycentric coordinates

It avoids becoming a duplicate STEM Agent.

General algebra/numerical mathematics belongs downstream in STEM; this module should remain spatially specialized.
"""

from __future__ import annotations


from ...base.modules.math_science import *
from ...base.modules.physics_constraints import *
from ..utils.config_loader import (
    load_global_config as load_config,
    get_config_section as get_section
    )
from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Calculations")
printer = PrettyPrinter()


class Calculations:
    def __init__(self) -> None:
        self.config = load_config()
        self.calc_config = get_section("calculations", config=self.config, default={})


__all__ = ["Calculations"]


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Calculations ===\n")
    printer.status("TEST", "Calculations initialized", "info")