"""
sources:
- de Berg et al. (2008)
- Rusu, R. B., & Cousins, S. (2011). “3D is here: Point Cloud Library (PCL).” ICRA.
- Hoppe, H., DeRose, T., Duchamp, T., McDonald, J., & Stuetzle, W. (1992). “Surface Reconstruction from Unorganized Points.” SIGGRAPH.
- Curless & Levoy (1996).

Geometry owns:
Point2 / Point3
Vector
Ray
Line
Segment
Plane
Triangle
Polygon
Polyhedron
Sphere
AABB / OBB primitives
Mesh
PointCloud
"""
from __future__ import annotations

__version__ = "2.3.0"


from ..utils.spatial_errors import *
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Geometry")
printer = PrettyPrinter()


__all__ = []


if __name__ == "__main__":
    configure_logging()
    print("\n=== Running Geometry ===\n")
    printer.status("TEST", "Geometry initialized", "info")