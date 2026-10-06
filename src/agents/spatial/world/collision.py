"""
Collision owns geometric collision/interference determination, not collision avoidance behaviour.

sources:
- Gottschalk, S., Lin, M. C., & Manocha, D. (1996). “OBBTree: A Hierarchical Structure for Rapid Interference Detection.” SIGGRAPH, 171–180.
- Ericson (2005), Real-Time Collision Detection.
- de Berg et al. (2008) should supplement this with computational-geometry fundamentals.

Architecture:
SpatialAgent / collision.py:
AABB(A) intersects AABB(B)
mesh(A) intersects mesh(B)
distance(A,B) = 0.17 m

PlanningAgent:
therefore choose another path

SafetyAgent:
therefore block action if safety threshold violated
That prevents one of the most tempting ownership violations.
"""
from __future__ import annotations

__version__ = "2.3.0"


from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Collision")
printer = PrettyPrinter()


__all__ = []


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Collision ===\n")
    printer.status("TEST", "Collision initialized", "info")