"""
Mapping manages spatial map representations and alignment, not decide where the robot should travel and not detect scene objects.

sources:
- Besl, P. J., & McKay, N. D. (1992). “A Method for Registration of 3-D Shapes.” IEEE TPAMI, 14(2), 239–256. DOI 10.1109/34.121791.
- Rusinkiewicz, S., & Levoy, M. (2001). “Efficient Variants of the ICP Algorithm.”
- Horn, B. K. P. (1987). “Closed-form solution of absolute orientation using unit quaternions.” JOSA A, 4(4), 629–642.
- Curless & Levoy (1996)

Mapping owns:
map representation lifecycle
map merge
map alignment
rigid registration
frame-normalized integration
representation conversion

not

feature detection
sensor interpretation
pose estimation from raw camera frames
loop-closure policy
path planning
exploration strategy

Those cross into Perception, estimation, Reasoning or Planning.
"""

from __future__ import annotations

from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Mapping")
printer = PrettyPrinter()


__all__ = []


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Mapping ===\n")
    printer.status("TEST", "Mapping initialized", "info")