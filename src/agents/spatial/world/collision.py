"""Geometric collision/interference computation.

This module reports collision state and separation geometry only. It never
selects an avoidance manoeuvre or path. Broad-phase tests use bounding
volumes; exact narrow-phase support is provided for the primitive pairs where
the subsystem can guarantee correctness.
"""
from __future__ import annotations

__version__ = "2.3.0"

import numpy as np # type: ignore

from dataclasses import dataclass
from typing import Any, Iterable

from .geometry import (
    AABB,
    OBB,
    Mesh,
    Sphere,
    Triangle,
    aabb_from_points,
    aabb_intersects,
    closest_point_on_aabb,
    geometry_bounds,
    sphere_intersects,
)
from ..utils.spatial_errors import SpatialGeometryError
from ..utils.spatial_helpers import DEFAULT_ABS_TOL
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Collision")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class CollisionResult:
    collides: bool
    distance: float
    broad_phase_only: bool = False
    details: dict[str, Any] | None = None


def aabb_separation(a: AABB, b: AABB) -> float:
    if a.dimension != b.dimension:
        raise SpatialGeometryError("AABB dimensions must match")
    gap = np.maximum(np.maximum(a.minimum - b.maximum, b.minimum - a.maximum), 0.0)
    return float(np.linalg.norm(gap))


def sphere_separation(a: Sphere, b: Sphere) -> float:
    return max(float(np.linalg.norm(a.center - b.center)) - a.radius - b.radius, 0.0)


def sphere_aabb_collision(sphere: Sphere, box: AABB, *, tolerance: float = DEFAULT_ABS_TOL) -> CollisionResult:
    closest = closest_point_on_aabb(sphere.center, box)
    center_distance = float(np.linalg.norm(sphere.center - closest))
    separation = max(center_distance - sphere.radius, 0.0)
    return CollisionResult(center_distance <= sphere.radius + tolerance, separation)


def obb_intersects(a: OBB, b: OBB, *, tolerance: float = DEFAULT_ABS_TOL) -> bool:
    """15-axis separating-axis test from Gottschalk/Ericson."""
    r = a.rotation.T @ b.rotation
    translation_world = b.center - a.center
    t = a.rotation.T @ translation_world
    abs_r = np.abs(r) + tolerance

    # A's local axes.
    for i in range(3):
        ra = a.half_extents[i]
        rb = float(b.half_extents @ abs_r[i, :])
        if abs(t[i]) > ra + rb:
            return False

    # B's local axes.
    for j in range(3):
        ra = float(a.half_extents @ abs_r[:, j])
        rb = b.half_extents[j]
        if abs(float(t @ r[:, j])) > ra + rb:
            return False

    # Cross products Ai x Bj.
    for i in range(3):
        for j in range(3):
            ra = a.half_extents[(i + 1) % 3] * abs_r[(i + 2) % 3, j] + a.half_extents[(i + 2) % 3] * abs_r[(i + 1) % 3, j]
            rb = b.half_extents[(j + 1) % 3] * abs_r[i, (j + 2) % 3] + b.half_extents[(j + 2) % 3] * abs_r[i, (j + 1) % 3]
            separation = abs(t[(i + 2) % 3] * r[(i + 1) % 3, j] - t[(i + 1) % 3] * r[(i + 2) % 3, j])
            if separation > ra + rb:
                return False
    return True


@dataclass(slots=True)
class BVHNode:
    bounds: AABB
    left: "BVHNode | None" = None
    right: "BVHNode | None" = None
    triangle_indices: tuple[int, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class MeshBVH:
    """Median-split BVH over triangle bounds for conservative broad phase."""

    def __init__(self, mesh: Mesh, *, leaf_size: int = 8) -> None:
        if leaf_size < 1:
            raise SpatialGeometryError("leaf_size must be >= 1")
        self.mesh = mesh
        self.leaf_size = int(leaf_size)
        self._triangles = list(mesh.triangles())
        self._bounds = [geometry_bounds(triangle) for triangle in self._triangles]
        self.root = self._build(np.arange(len(self._triangles), dtype=int)) if self._triangles else None

    def _build(self, indices: np.ndarray) -> BVHNode:
        points = np.vstack([np.vstack((self._bounds[i].minimum, self._bounds[i].maximum)) for i in indices])
        bounds = aabb_from_points(points)
        assert bounds is not None
        if len(indices) <= self.leaf_size:
            return BVHNode(bounds, triangle_indices=tuple(int(i) for i in indices))
        centers = np.vstack([self._bounds[i].center for i in indices])
        axis = int(np.argmax(np.ptp(centers, axis=0)))
        ordered = indices[np.argsort(centers[:, axis], kind="stable")]
        midpoint = len(ordered) // 2
        return BVHNode(bounds, left=self._build(ordered[:midpoint]), right=self._build(ordered[midpoint:]))

    def query_bounds(self, bounds: AABB) -> tuple[int, ...]:
        if self.root is None:
            return ()
        result: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not aabb_intersects(node.bounds, bounds):
                continue
            if node.is_leaf:
                result.extend(node.triangle_indices)
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)
        return tuple(result)


def mesh_broad_phase_pairs(a: Mesh, b: Mesh) -> tuple[tuple[int, int], ...]:
    """Return triangle pairs whose AABBs overlap; this is conservative only."""
    bvh = MeshBVH(b)
    pairs: list[tuple[int, int]] = []
    for i, triangle in enumerate(a.triangles()):
        query = geometry_bounds(triangle)
        for j in bvh.query_bounds(query):
            if aabb_intersects(query, bvh._bounds[j]):
                pairs.append((i, j))
    return tuple(pairs)


class CollisionEngine:
    def test(self, a: Any, b: Any, *, tolerance: float = DEFAULT_ABS_TOL) -> CollisionResult:
        if isinstance(a, AABB) and isinstance(b, AABB):
            distance = aabb_separation(a, b)
            return CollisionResult(distance <= tolerance, distance)
        if isinstance(a, Sphere) and isinstance(b, Sphere):
            distance = sphere_separation(a, b)
            return CollisionResult(sphere_intersects(a, b, tolerance=tolerance), distance)
        if isinstance(a, Sphere) and isinstance(b, AABB):
            return sphere_aabb_collision(a, b, tolerance=tolerance)
        if isinstance(a, AABB) and isinstance(b, Sphere):
            return sphere_aabb_collision(b, a, tolerance=tolerance)
        if isinstance(a, OBB) and isinstance(b, OBB):
            return CollisionResult(obb_intersects(a, b, tolerance=tolerance), 0.0 if obb_intersects(a, b, tolerance=tolerance) else float("nan"))
        if isinstance(a, Mesh) and isinstance(b, Mesh):
            if a.bounds is None or b.bounds is None:
                return CollisionResult(False, float("inf"), broad_phase_only=True)
            if not aabb_intersects(a.bounds, b.bounds, tolerance=tolerance):
                return CollisionResult(False, aabb_separation(a.bounds, b.bounds), broad_phase_only=True)
            pairs = mesh_broad_phase_pairs(a, b)
            return CollisionResult(bool(pairs), 0.0 if pairs else aabb_separation(a.bounds, b.bounds), broad_phase_only=True, details={"candidate_triangle_pairs": pairs})
        raise SpatialGeometryError(
            "collision pair is unsupported",
            context={"a": type(a).__name__, "b": type(b).__name__},
        )


__all__ = [
    "CollisionResult",
    "BVHNode",
    "MeshBVH",
    "aabb_separation",
    "sphere_separation",
    "sphere_aabb_collision",
    "obb_intersects",
    "mesh_broad_phase_pairs",
    "CollisionEngine",
]


if __name__ == "__main__":
    configure_logging()
    engine = CollisionEngine()
    result = engine.test(AABB(np.zeros(3), np.ones(3)), AABB(np.ones(3), np.ones(3) * 2.0))
    printer.status("TEST", f"collision={result.collides}", "info")
