"""Coordinate-light topological relations for supported region geometries.

Semantics are inspired by the 9-intersection family (Egenhofer & Franzosa),
RCC-style region connection (Randell, Cui & Cohn), and Simple Features.
Where exact boundary/interior semantics are unavailable for a geometry type,
the module raises rather than silently substituting a bounding-box relation.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from enum import Enum
from typing import Any

from .geometry import AABB, Polygon, Segment, Sphere, aabb_intersects, point_in_aabb, point_in_polygon_2d, segment_intersection_2d, sphere_intersects
from ..utils.spatial_errors import SpatialTopologyError, SpatialValidationError
from ..utils.spatial_helpers import DEFAULT_ABS_TOL, all_close, finite_vector
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Topology")
printer = PrettyPrinter()


class TopologicalRelation(str, Enum):
    DISJOINT = "disjoint"
    TOUCHES = "touches"
    OVERLAPS = "overlaps"
    CONTAINS = "contains"
    INSIDE = "inside"
    EQUALS = "equals"
    CROSSES = "crosses"
    CONNECTED = "connected"


def _aabb_contains(a: AABB, b: AABB, *, tolerance: float, strict: bool = False) -> bool:
    if a.dimension != b.dimension:
        return False
    if strict:
        return bool(np.all(b.minimum > a.minimum + tolerance) and np.all(b.maximum < a.maximum - tolerance))
    return bool(np.all(b.minimum >= a.minimum - tolerance) and np.all(b.maximum <= a.maximum + tolerance))


def _sphere_contains(a: Sphere, b: Sphere, *, tolerance: float, strict: bool = False) -> bool:
    distance = float(np.linalg.norm(a.center - b.center)) + b.radius
    return distance < a.radius - tolerance if strict else distance <= a.radius + tolerance


def equals(a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, AABB):
        return all_close(a.minimum, b.minimum, abs_tol=tolerance, rel_tol=0.0) and all_close(a.maximum, b.maximum, abs_tol=tolerance, rel_tol=0.0)
    if isinstance(a, Sphere):
        return all_close(a.center, b.center, abs_tol=tolerance, rel_tol=0.0) and abs(a.radius - b.radius) <= tolerance
    if isinstance(a, Polygon):
        return a.vertices.shape == b.vertices.shape and all_close(a.vertices, b.vertices, abs_tol=tolerance, rel_tol=0.0)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return all_close(a, b, abs_tol=tolerance, rel_tol=0.0)
        except (TypeError, ValueError, SpatialValidationError):
            return False
    return a == b


def contains(container: Any, item: Any, *, tolerance: float = DEFAULT_ABS_TOL, strict: bool = False) -> bool:
    if isinstance(container, AABB):
        if isinstance(item, AABB):
            return _aabb_contains(container, item, tolerance=tolerance, strict=strict)
        if isinstance(item, Sphere):
            lower = item.center - item.radius
            upper = item.center + item.radius
            return _aabb_contains(container, AABB(lower, upper), tolerance=tolerance, strict=strict)
        point = np.asarray(item, dtype=float)
        if point.ndim == 1:
            return point_in_aabb(point, container, tolerance=tolerance, strict=strict)
    if isinstance(container, Sphere):
        if isinstance(item, Sphere):
            return _sphere_contains(container, item, tolerance=tolerance, strict=strict)
        point = np.asarray(item, dtype=float)
        if point.ndim == 1 and point.size == 3:
            distance = float(np.linalg.norm(point - container.center))
            return distance < container.radius - tolerance if strict else distance <= container.radius + tolerance
    if isinstance(container, Polygon):
        point = np.asarray(item, dtype=float)
        if point.ndim == 1 and point.size == 2 and container.dimension == 2:
            return point_in_polygon_2d(point, container, tolerance=tolerance)
    raise SpatialTopologyError(
        "contains semantics are unsupported for this geometry pair",
        context={"container": type(container).__name__, "item": type(item).__name__},
    )


def inside(item: Any, container: Any, *, tolerance: float = DEFAULT_ABS_TOL, strict: bool = False) -> bool:
    return contains(container, item, tolerance=tolerance, strict=strict)


def disjoint(a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    if isinstance(a, AABB) and isinstance(b, AABB):
        return not aabb_intersects(a, b, tolerance=tolerance)
    if isinstance(a, Sphere) and isinstance(b, Sphere):
        return not sphere_intersects(a, b, tolerance=tolerance)
    if isinstance(a, Segment) and isinstance(b, Segment) and a.start.size == b.start.size == 2:
        return not segment_intersection_2d(a, b, tolerance=tolerance)
    raise SpatialTopologyError("disjoint semantics are unsupported for this geometry pair")


def connected(a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    return not disjoint(a, b, tolerance=tolerance)


def touches(a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    if isinstance(a, AABB) and isinstance(b, AABB):
        if not aabb_intersects(a, b, tolerance=tolerance):
            return False
        overlap = np.minimum(a.maximum, b.maximum) - np.maximum(a.minimum, b.minimum)
        return bool(np.any(np.abs(overlap) <= tolerance) and np.all(overlap >= -tolerance))
    if isinstance(a, Sphere) and isinstance(b, Sphere):
        distance = float(np.linalg.norm(a.center - b.center))
        return abs(distance - (a.radius + b.radius)) <= tolerance or abs(distance - abs(a.radius - b.radius)) <= tolerance
    raise SpatialTopologyError("touches semantics are unsupported for this geometry pair")


def overlaps(a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    if equals(a, b, tolerance=tolerance):
        return False
    if isinstance(a, AABB) and isinstance(b, AABB):
        if not aabb_intersects(a, b, tolerance=tolerance):
            return False
        if _aabb_contains(a, b, tolerance=tolerance) or _aabb_contains(b, a, tolerance=tolerance):
            return False
        overlap = np.minimum(a.maximum, b.maximum) - np.maximum(a.minimum, b.minimum)
        return bool(np.all(overlap > tolerance))
    if isinstance(a, Sphere) and isinstance(b, Sphere):
        distance = float(np.linalg.norm(a.center - b.center))
        return abs(a.radius - b.radius) + tolerance < distance < a.radius + b.radius - tolerance
    raise SpatialTopologyError("overlaps semantics are unsupported for this geometry pair")


def crosses(a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    if isinstance(a, Segment) and isinstance(b, Segment) and a.start.size == b.start.size == 2:
        if not segment_intersection_2d(a, b, tolerance=tolerance):
            return False
        # Crossing excludes merely sharing an endpoint.
        endpoints = (a.start, a.end)
        others = (b.start, b.end)
        return not any(all_close(x, y, abs_tol=tolerance, rel_tol=0.0) for x in endpoints for y in others)
    raise SpatialTopologyError("crosses semantics are unsupported for this geometry pair")


def boundary(geometry: Any) -> Any:
    """Return a lightweight boundary representation for supported regions."""
    if isinstance(geometry, AABB):
        return {"minimum": geometry.minimum.copy(), "maximum": geometry.maximum.copy()}
    if isinstance(geometry, Sphere):
        return {"center": geometry.center.copy(), "radius": geometry.radius}
    if isinstance(geometry, Polygon):
        return tuple(Segment(geometry.vertices[i], geometry.vertices[(i + 1) % len(geometry.vertices)]) for i in range(len(geometry.vertices)))
    raise SpatialTopologyError("boundary representation is unsupported for this geometry type")


def interior(geometry: Any, point: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    return contains(geometry, point, tolerance=tolerance, strict=True)


class Topology:
    """Facade for topological predicates with stable method names."""

    connected = staticmethod(connected)
    disconnected = staticmethod(disjoint)
    disjoint = staticmethod(disjoint)
    boundary = staticmethod(boundary)
    interior = staticmethod(interior)
    contains = staticmethod(contains)
    inside = staticmethod(inside)
    touches = staticmethod(touches)
    overlaps = staticmethod(overlaps)
    crosses = staticmethod(crosses)
    equal = staticmethod(equals)
    equals = staticmethod(equals)

    @staticmethod
    def relate(a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> set[TopologicalRelation]:
        result: set[TopologicalRelation] = set()
        if equals(a, b, tolerance=tolerance):
            result.add(TopologicalRelation.EQUALS)
        try:
            if contains(a, b, tolerance=tolerance):
                result.add(TopologicalRelation.CONTAINS)
            if inside(a, b, tolerance=tolerance):
                result.add(TopologicalRelation.INSIDE)
        except SpatialTopologyError:
            logger.debug("Topological containment relation not defined for %s/%s", type(a).__name__, type(b).__name__)
        try:
            if disjoint(a, b, tolerance=tolerance):
                result.add(TopologicalRelation.DISJOINT)
                return result
            result.add(TopologicalRelation.CONNECTED)
        except SpatialTopologyError:
            logger.debug("Topological connectivity relation not defined for %s/%s", type(a).__name__, type(b).__name__)
        for relation, function in ((TopologicalRelation.TOUCHES, touches), (TopologicalRelation.OVERLAPS, overlaps), (TopologicalRelation.CROSSES, crosses)):
            try:
                if function(a, b, tolerance=tolerance):
                    result.add(relation)
            except SpatialTopologyError:
                continue
        return result


__all__ = [
    "TopologicalRelation",
    "connected",
    "disjoint",
    "boundary",
    "interior",
    "contains",
    "inside",
    "touches",
    "overlaps",
    "crosses",
    "equals",
    "Topology",
]


if __name__ == "__main__":
    configure_logging()
    a = AABB(np.zeros(2), np.ones(2))
    b = AABB(np.array([1.0, 0.0]), np.array([2.0, 1.0]))
    printer.status("TEST", f"touches={touches(a, b)}", "info")
