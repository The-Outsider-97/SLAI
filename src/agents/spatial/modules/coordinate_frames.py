"""
This module is capable of Coordinate reference systems and Coordinate-frame transforms

sources:
- Lynch & Park's Modern Robotics, particularly its rigid-body-motion treatment
- Barfoot, T. D. (2024). State Estimation for Robotics, 2nd ed. Cambridge University Press.
- Solà, J., Deray, J., & Atchuthan, D. (2018). “A micro Lie theory for state estimation in robotics.”
- Foote, T. (2013). “tf: The transform library.” IEEE TePRA.
- ISO 19111:2019 
- Snyder, J. P. (1987). Map Projections—A Working Manual. USGS Professional Paper 1395.

the rigid-body-motion treatment is excellent for:
SO(3)
SE(3)
rotation matrices
homogeneous transforms
rigid transformations
frame composition
frame inversion

the Lie theory for state estimation in robotics is excellent for:
SO(3)
SE(3)
Lie algebra
exp/log maps
Jacobians
perturbations
frame transformations

The coordinate frames capable of supporting both:

robotics:
camera → body → robot → map → world

GIS/BIM:
local model → projected CRS → geodetic CRS
without mixing either domain into Perception.
"""

from __future__ import annotations

from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Coordinate Frames")
printer = PrettyPrinter()


__all__ = []


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Coordinate Frames ===\n")
    printer.status("TEST", "Coordinate Frames initialized", "info")