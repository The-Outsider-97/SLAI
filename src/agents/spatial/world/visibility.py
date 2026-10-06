"""Geometric visibility and line-of-sight primitives.

Visibility is reported as spatial state only; this module never plans a path or
chooses an action. Ray tests are based on computational-geometry primitives.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from dataclasses import dataclass
from math import radians, tan
from typing import Any, Iterable

from .geometry import AABB, Mesh, OBB, Ray, Sphere, Triangle, ray_aabb_intersection, ray_sphere_intersection, ray_triangle_intersection
from ..utils.spatial_errors import SpatialGeometryError
from ..utils.spatial_helpers import DEFAULT_ABS_TOL, finite_vector, unit_vector
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Visibility")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class VisibilityResult:
    visible: bool
    distance: float
    blocker_index: int | None = None
    blocker_distance: float | None = None


@dataclass(frozen=True, slots=True)
class Frustum:
    origin: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    vertical_fov_degrees: float
    aspect_ratio: float
    near: float = 0.0
    far: float = float("inf")

    def __post_init__(self) -> None:
        origin = finite_vector(self.origin, name="origin", dimension=3)
        forward = unit_vector(self.forward, name="forward")
        up = unit_vector(self.up, name="up")
        right = np.cross(forward, up)
        if np.linalg.norm(right) <= DEFAULT_ABS_TOL:
            raise SpatialGeometryError("frustum forward and up vectors must not be parallel")
        up = unit_vector(np.cross(unit_vector(right), forward), name="orthogonal_up")
        if not 0.0 < self.vertical_fov_degrees < 180.0:
            raise SpatialGeometryError("vertical_fov_degrees must lie in (0, 180)")
        if self.aspect_ratio <= 0.0 or self.near < 0.0 or self.far <= self.near:
            raise SpatialGeometryError("invalid frustum aspect or near/far values")
        object.__setattr__(self, "origin", origin.copy())
        object.__setattr__(self, "forward", forward.copy())
        object.__setattr__(self, "up", up.copy())

    def contains(self, point: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
        p = finite_vector(point, name="point", dimension=3)
        relative = p - self.origin
        depth = float(relative @ self.forward)
        if depth < self.near - tolerance or depth > self.far + tolerance:
            return False
        right = unit_vector(np.cross(self.forward, self.up), name="right")
        half_height = depth * tan(radians(self.vertical_fov_degrees) * 0.5)
        half_width = half_height * self.aspect_ratio
        vertical = float(relative @ self.up)
        horizontal = float(relative @ right)
        return abs(vertical) <= half_height + tolerance and abs(horizontal) <= half_width + tolerance


def ray_intersection_distance(ray: Ray, obstacle: Any, *, max_distance: float | None = None, tolerance: float = DEFAULT_ABS_TOL) -> float | None:
    if isinstance(obstacle, AABB):
        return ray_aabb_intersection(ray, obstacle, max_distance=max_distance, tolerance=tolerance)
    if isinstance(obstacle, Sphere):
        return ray_sphere_intersection(ray, obstacle, max_distance=max_distance, tolerance=tolerance)
    if isinstance(obstacle, Triangle):
        return ray_triangle_intersection(ray, obstacle, max_distance=max_distance, tolerance=tolerance)
    if isinstance(obstacle, OBB):
        # Transform ray into the OBB's local coordinates and test against local AABB.
        local_origin = obstacle.rotation.T @ (ray.origin - obstacle.center)
        local_direction = obstacle.rotation.T @ ray.direction
        local_ray = Ray(local_origin, local_direction)
        local_box = AABB(-obstacle.half_extents, obstacle.half_extents)
        return ray_aabb_intersection(local_ray, local_box, max_distance=max_distance, tolerance=tolerance)
    if isinstance(obstacle, Mesh):
        nearest: float | None = None
        bounds = obstacle.bounds
        if bounds is None or ray_aabb_intersection(ray, bounds, max_distance=max_distance, tolerance=tolerance) is None:
            return None
        for triangle in obstacle.triangles():
            hit = ray_triangle_intersection(ray, triangle, max_distance=max_distance, tolerance=tolerance)
            if hit is not None and (nearest is None or hit < nearest):
                nearest = hit
        return nearest
    raise SpatialGeometryError("unsupported visibility obstacle", context={"type": type(obstacle).__name__})


def visible_from(origin: Any, target: Any, obstacles: Iterable[Any] = (), *, tolerance: float = DEFAULT_ABS_TOL) -> VisibilityResult:
    start = finite_vector(origin, name="origin", dimension=3)
    end = finite_vector(target, name="target", dimension=3)
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance <= tolerance:
        return VisibilityResult(True, 0.0)
    ray = Ray(start, delta / distance)
    nearest_index: int | None = None
    nearest_distance: float | None = None
    for index, obstacle in enumerate(obstacles):
        hit = ray_intersection_distance(ray, obstacle, max_distance=distance, tolerance=tolerance)
        if hit is None:
            continue
        # Hits at the target itself are not considered occlusion.
        if hit < distance - tolerance and (nearest_distance is None or hit < nearest_distance):
            nearest_index = index
            nearest_distance = hit
    return VisibilityResult(nearest_index is None, distance, nearest_index, nearest_distance)


__all__ = [
    "VisibilityResult",
    "Frustum",
    "ray_intersection_distance",
    "visible_from",
]


if __name__ == "__main__":
    configure_logging()
    result = visible_from((0, 0, 0), (2, 0, 0), [AABB(np.array([0.9, -0.1, -0.1]), np.array([1.1, 0.1, 0.1]))])
    printer.status("TEST", f"visible={result.visible}", "info")
