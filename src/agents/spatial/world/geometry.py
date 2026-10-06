"""Fundamental geometric representations and predicates.

The module owns coordinate geometry, not perception. It accepts already
interpreted points/meshes and exposes tolerance-aware deterministic operations.
Primary references: de Berg et al.; Ericson; Rusu & Cousins; Hoppe et al.;
Curless & Levoy; Botsch et al.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..utils.spatial_errors import SpatialGeometryError, SpatialValidationError
from ..utils.spatial_helpers import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Geometry")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", finite_vector(self.origin, name="origin", dimension=3).copy())
        object.__setattr__(self, "direction", unit_vector(self.direction, name="direction").copy())


@dataclass(frozen=True, slots=True)
class Segment:
    start: np.ndarray
    end: np.ndarray

    def __post_init__(self) -> None:
        start = finite_vector(self.start, name="start")
        end = finite_vector(self.end, name="end", dimension=start.size)
        object.__setattr__(self, "start", start.copy())
        object.__setattr__(self, "end", end.copy())

    @property
    def direction(self) -> np.ndarray:
        return self.end - self.start

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.direction))


@dataclass(frozen=True, slots=True)
class Plane:
    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", finite_vector(self.point, name="point", dimension=3).copy())
        object.__setattr__(self, "normal", unit_vector(self.normal, name="normal").copy())


@dataclass(frozen=True, slots=True)
class Triangle:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        a = finite_vector(self.a, name="a", dimension=3)
        b = finite_vector(self.b, name="b", dimension=3)
        c = finite_vector(self.c, name="c", dimension=3)
        if np.linalg.norm(np.cross(b - a, c - a)) <= DEFAULT_ABS_TOL:
            raise SpatialGeometryError("triangle vertices are degenerate")
        object.__setattr__(self, "a", a.copy())
        object.__setattr__(self, "b", b.copy())
        object.__setattr__(self, "c", c.copy())

    @property
    def normal(self) -> np.ndarray:
        return unit_vector(np.cross(self.b - self.a, self.c - self.a), name="triangle_normal")

    @property
    def vertices(self) -> np.ndarray:
        return np.vstack((self.a, self.b, self.c))


@dataclass(frozen=True, slots=True)
class Polygon:
    vertices: np.ndarray

    def __post_init__(self) -> None:
        vertices = finite_array(self.vertices, name="vertices", ndim=2)
        if vertices.shape[0] < 3 or vertices.shape[1] not in (2, 3):
            raise SpatialGeometryError("polygon vertices must have shape (N>=3, 2|3)")
        object.__setattr__(self, "vertices", vertices.copy())

    @property
    def dimension(self) -> int:
        return int(self.vertices.shape[1])


@dataclass(frozen=True, slots=True)
class Sphere:
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", finite_vector(self.center, name="center", dimension=3).copy())
        object.__setattr__(self, "radius", validate_non_negative(self.radius, name="radius"))


@dataclass(frozen=True, slots=True)
class AABB:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = finite_vector(self.minimum, name="minimum")
        maximum = finite_vector(self.maximum, name="maximum", dimension=minimum.size)
        if np.any(minimum > maximum):
            raise SpatialGeometryError("AABB minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum.copy())
        object.__setattr__(self, "maximum", maximum.copy())

    @property
    def dimension(self) -> int:
        return int(self.minimum.size)

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) * 0.5

    @property
    def extents(self) -> np.ndarray:
        return (self.maximum - self.minimum) * 0.5

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum

    def expanded(self, amount: float) -> "AABB":
        margin = validate_non_negative(amount, name="amount")
        return AABB(self.minimum - margin, self.maximum + margin)


@dataclass(frozen=True, slots=True)
class OBB:
    center: np.ndarray
    half_extents: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        center = finite_vector(self.center, name="center", dimension=3)
        half = finite_vector(self.half_extents, name="half_extents", dimension=3)
        rotation = finite_array(self.rotation, name="rotation", shape=(3, 3))
        if np.any(half < 0.0):
            raise SpatialGeometryError("OBB half_extents must be non-negative")
        identity = rotation.T @ rotation
        if not np.allclose(identity, np.eye(3), atol=1.0e-8, rtol=0.0) or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-8:
            raise SpatialGeometryError("OBB rotation must be a proper orthonormal matrix")
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "half_extents", half.copy())
        object.__setattr__(self, "rotation", rotation.copy())

    @property
    def corners(self) -> np.ndarray:
        signs = np.array(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=float,
        )
        local = signs * self.half_extents
        return local @ self.rotation.T + self.center


@dataclass(frozen=True, slots=True)
class PointCloud:
    points: np.ndarray
    frame_id: str = "world"

    def __post_init__(self) -> None:
        points = finite_array(self.points, name="points", ndim=2)
        if points.shape[1] != 3:
            raise SpatialGeometryError("PointCloud points must have shape (N, 3)")
        object.__setattr__(self, "points", points.copy())

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def bounds(self) -> AABB | None:
        return aabb_from_points(self.points) if len(self) else None


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    frame_id: str = "world"

    def __post_init__(self) -> None:
        vertices = finite_array(self.vertices, name="vertices", ndim=2)
        if vertices.shape[1] != 3:
            raise SpatialGeometryError("Mesh vertices must have shape (N, 3)")
        faces = np.asarray(self.faces, dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise SpatialGeometryError("Mesh faces must have shape (M, 3)")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise SpatialGeometryError("Mesh face index is outside the vertex array")
        object.__setattr__(self, "vertices", vertices.copy())
        object.__setattr__(self, "faces", faces.copy())

    @property
    def bounds(self) -> AABB | None:
        return aabb_from_points(self.vertices) if len(self.vertices) else None

    def triangles(self) -> Iterable[Triangle]:
        for i, j, k in self.faces:
            try:
                yield Triangle(self.vertices[i], self.vertices[j], self.vertices[k])
            except SpatialGeometryError:
                continue


def aabb_from_points(points: Any) -> AABB | None:
    array = finite_array(points, name="points", ndim=2)
    if array.shape[0] == 0:
        return None
    return AABB(array.min(axis=0), array.max(axis=0))


def geometry_bounds(geometry: Any) -> AABB:
    if isinstance(geometry, AABB):
        return geometry
    if isinstance(geometry, OBB):
        bounds = aabb_from_points(geometry.corners)
        assert bounds is not None
        return bounds
    if isinstance(geometry, Sphere):
        radius = np.full(3, geometry.radius, dtype=float)
        return AABB(geometry.center - radius, geometry.center + radius)
    if isinstance(geometry, Triangle):
        bounds = aabb_from_points(geometry.vertices)
        assert bounds is not None
        return bounds
    if isinstance(geometry, Polygon):
        bounds = aabb_from_points(geometry.vertices)
        assert bounds is not None
        return bounds
    if isinstance(geometry, Mesh):
        if geometry.bounds is None:
            raise SpatialGeometryError("empty mesh has no bounds")
        return geometry.bounds
    if isinstance(geometry, PointCloud):
        if geometry.bounds is None:
            raise SpatialGeometryError("empty point cloud has no bounds")
        return geometry.bounds
    array = np.asarray(geometry, dtype=float)
    if array.ndim == 1:
        point = finite_vector(array, name="geometry")
        return AABB(point, point)
    raise SpatialGeometryError("unsupported geometry for bounds", context={"type": type(geometry).__name__})


def point_in_aabb(point: Any, box: AABB, *, tolerance: float = DEFAULT_ABS_TOL, strict: bool = False) -> bool:
    p = finite_vector(point, name="point", dimension=box.dimension)
    if strict:
        return bool(np.all(p > box.minimum + tolerance) and np.all(p < box.maximum - tolerance))
    return bool(np.all(p >= box.minimum - tolerance) and np.all(p <= box.maximum + tolerance))


def aabb_intersects(a: AABB, b: AABB, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    if a.dimension != b.dimension:
        raise SpatialGeometryError("AABB dimensions must match")
    return bool(np.all(a.maximum + tolerance >= b.minimum) and np.all(b.maximum + tolerance >= a.minimum))


def aabb_overlap_extent(a: AABB, b: AABB) -> np.ndarray:
    if a.dimension != b.dimension:
        raise SpatialGeometryError("AABB dimensions must match")
    return np.maximum(np.minimum(a.maximum, b.maximum) - np.maximum(a.minimum, b.minimum), 0.0)


def sphere_intersects(a: Sphere, b: Sphere, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    return float(np.linalg.norm(a.center - b.center)) <= a.radius + b.radius + tolerance


def closest_point_on_aabb(point: Any, box: AABB) -> np.ndarray:
    p = finite_vector(point, name="point", dimension=box.dimension)
    return np.minimum(np.maximum(p, box.minimum), box.maximum)


def ray_triangle_intersection(
    ray: Ray,
    triangle: Triangle,
    *,
    tolerance: float = DEFAULT_ABS_TOL,
    max_distance: float | None = None,
) -> float | None:
    """Möller-Trumbore ray/triangle intersection distance."""
    edge1 = triangle.b - triangle.a
    edge2 = triangle.c - triangle.a
    h = np.cross(ray.direction, edge2)
    determinant = float(edge1 @ h)
    if abs(determinant) <= tolerance:
        return None
    inverse = 1.0 / determinant
    s = ray.origin - triangle.a
    u = inverse * float(s @ h)
    if u < -tolerance or u > 1.0 + tolerance:
        return None
    q = np.cross(s, edge1)
    v = inverse * float(ray.direction @ q)
    if v < -tolerance or u + v > 1.0 + tolerance:
        return None
    distance = inverse * float(edge2 @ q)
    if distance < tolerance:
        return None
    if max_distance is not None and distance > max_distance + tolerance:
        return None
    return distance


def ray_aabb_intersection(
    ray: Ray,
    box: AABB,
    *,
    max_distance: float | None = None,
    tolerance: float = DEFAULT_ABS_TOL,
) -> float | None:
    if box.dimension != 3:
        raise SpatialGeometryError("ray/AABB intersection requires a 3D AABB")
    t_min = 0.0
    t_max = float("inf") if max_distance is None else float(max_distance)
    for axis in range(3):
        direction = float(ray.direction[axis])
        origin = float(ray.origin[axis])
        if abs(direction) <= tolerance:
            if origin < box.minimum[axis] - tolerance or origin > box.maximum[axis] + tolerance:
                return None
            continue
        inv = 1.0 / direction
        t1 = (box.minimum[axis] - origin) * inv
        t2 = (box.maximum[axis] - origin) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, float(t1))
        t_max = min(t_max, float(t2))
        if t_min > t_max + tolerance:
            return None
    return max(0.0, t_min)


def ray_sphere_intersection(
    ray: Ray,
    sphere: Sphere,
    *,
    max_distance: float | None = None,
    tolerance: float = DEFAULT_ABS_TOL,
) -> float | None:
    oc = ray.origin - sphere.center
    b = float(oc @ ray.direction)
    c = float(oc @ oc) - sphere.radius * sphere.radius
    discriminant = b * b - c
    if discriminant < -tolerance:
        return None
    root = np.sqrt(max(discriminant, 0.0))
    candidates = [-b - root, -b + root]
    valid = [distance for distance in candidates if distance >= -tolerance]
    if not valid:
        return None
    distance = max(0.0, min(valid))
    if max_distance is not None and distance > max_distance + tolerance:
        return None
    return float(distance)


def point_in_polygon_2d(point: Any, polygon: Polygon, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    if polygon.dimension != 2:
        raise SpatialGeometryError("point_in_polygon_2d requires a 2D polygon")
    p = finite_vector(point, name="point", dimension=2)
    vertices = polygon.vertices
    inside = False
    count = len(vertices)
    for i in range(count):
        a = vertices[i]
        b = vertices[(i + 1) % count]
        ab = b - a
        ap = p - a
        cross = float(ab[0] * ap[1] - ab[1] * ap[0])
        if abs(cross) <= tolerance:
            dot = float(ap @ ab)
            if -tolerance <= dot <= float(ab @ ab) + tolerance:
                return True
        if (a[1] > p[1]) != (b[1] > p[1]):
            x_cross = (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
            if p[0] < x_cross:
                inside = not inside
    return inside


def segment_intersection_2d(a: Segment, b: Segment, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    if a.start.size != 2 or b.start.size != 2:
        raise SpatialGeometryError("segment_intersection_2d requires 2D segments")

    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return bool(
            min(p[0], r[0]) - tolerance <= q[0] <= max(p[0], r[0]) + tolerance
            and min(p[1], r[1]) - tolerance <= q[1] <= max(p[1], r[1]) + tolerance
        )

    o1 = orientation(a.start, a.end, b.start)
    o2 = orientation(a.start, a.end, b.end)
    o3 = orientation(b.start, b.end, a.start)
    o4 = orientation(b.start, b.end, a.end)
    if (o1 > tolerance and o2 < -tolerance or o1 < -tolerance and o2 > tolerance) and (
        o3 > tolerance and o4 < -tolerance or o3 < -tolerance and o4 > tolerance
    ):
        return True
    return bool(
        (abs(o1) <= tolerance and on_segment(a.start, b.start, a.end))
        or (abs(o2) <= tolerance and on_segment(a.start, b.end, a.end))
        or (abs(o3) <= tolerance and on_segment(b.start, a.start, b.end))
        or (abs(o4) <= tolerance and on_segment(b.start, a.end, b.end))
    )


def intersects(a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    """Exact predicates for supported primitive pairs; bounds fallback is explicit."""
    if isinstance(a, AABB) and isinstance(b, AABB):
        return aabb_intersects(a, b, tolerance=tolerance)
    if isinstance(a, Sphere) and isinstance(b, Sphere):
        return sphere_intersects(a, b, tolerance=tolerance)
    if isinstance(a, Sphere) and isinstance(b, AABB):
        closest = closest_point_on_aabb(a.center, b)
        return float(np.linalg.norm(a.center - closest)) <= a.radius + tolerance
    if isinstance(a, AABB) and isinstance(b, Sphere):
        return intersects(b, a, tolerance=tolerance)
    if isinstance(a, Segment) and isinstance(b, Segment) and a.start.size == b.start.size == 2:
        return segment_intersection_2d(a, b, tolerance=tolerance)
    if isinstance(a, np.ndarray):
        a_arr = np.asarray(a)
        if a_arr.ndim == 1 and isinstance(b, AABB):
            return point_in_aabb(a_arr, b, tolerance=tolerance)
    if isinstance(b, np.ndarray):
        b_arr = np.asarray(b)
        if b_arr.ndim == 1 and isinstance(a, AABB):
            return point_in_aabb(b_arr, a, tolerance=tolerance)
    raise SpatialGeometryError(
        "exact intersection predicate is not implemented for this geometry pair",
        context={"a": type(a).__name__, "b": type(b).__name__},
    )


__all__ = [
    "Ray",
    "Segment",
    "Plane",
    "Triangle",
    "Polygon",
    "Sphere",
    "AABB",
    "OBB",
    "PointCloud",
    "Mesh",
    "aabb_from_points",
    "geometry_bounds",
    "point_in_aabb",
    "aabb_intersects",
    "aabb_overlap_extent",
    "sphere_intersects",
    "closest_point_on_aabb",
    "ray_triangle_intersection",
    "ray_aabb_intersection",
    "ray_sphere_intersection",
    "point_in_polygon_2d",
    "segment_intersection_2d",
    "intersects",
]


if __name__ == "__main__":
    configure_logging()
    box = AABB(np.zeros(3), np.ones(3))
    printer.status("TEST", f"contains_origin={point_in_aabb((0, 0, 0), box)}", "info")
