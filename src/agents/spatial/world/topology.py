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
from typing import Any, Mapping, Sequence

from .geometry import AABB, Polygon, Segment, Sphere, aabb_intersects, point_in_aabb, point_in_polygon_2d, segment_intersection_2d, sphere_intersects
from ..utils.spatial_errors import SpatialTopologyError, SpatialValidationError
from ..utils.spatial_helpers import DEFAULT_ABS_TOL, all_close
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
    @staticmethod
    def validate_constraints(entities: Mapping[str, Any], constraints: Sequence[Mapping[str, Any]],
                             *, tolerance: float = DEFAULT_ABS_TOL) -> dict[str, Any]:
        """Evaluate point/vector constraints without moving caller-owned entities.

        Supported kinds: coincident, distance, minimum_clearance, parallel,
        perpendicular, collinear.  Does not claim solver completeness.
        """
        if not isinstance(entities, Mapping) or not isinstance(constraints, (list, tuple)) or len(constraints) > 5000:
            raise SpatialValidationError("entities must be mapping and constraints a bounded sequence")
        if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
            raise SpatialValidationError("tolerance must be finite and non-negative")
        points: dict[str, np.ndarray] = {}
        frames: dict[str, str] = {}
        for key, value in entities.items():
            if not isinstance(key, str) or not key:
                raise SpatialValidationError("entity identifiers must be non-empty strings")
            raw = value.position if hasattr(value, "position") else value
            vec = np.asarray(raw, dtype=float)
            if vec.ndim != 1 or vec.size not in (2, 3) or not np.all(np.isfinite(vec)):
                raise SpatialValidationError("constraint entities require finite 2D or 3D points")
            points[key] = vec.copy()
            frames[key] = str(getattr(value, "frame_id", "world"))
        results: list[dict[str, Any]] = []
        for index, specification in enumerate(constraints):
            if not isinstance(specification, Mapping):
                raise SpatialValidationError("constraints must be mappings")
            kind, refs = specification.get("kind"), specification.get("entities")
            expected_size = {"coincident": 2, "distance": 2, "minimum_clearance": 2,
                             "parallel": 4, "perpendicular": 4, "collinear": 3}
            if kind not in expected_size or not isinstance(refs, (list, tuple)) or len(refs) != expected_size[kind] or any(not isinstance(x, str) or not x for x in refs):
                raise SpatialValidationError("unsupported constraint kind or invalid entity references")
            missing = [ref for ref in refs if ref not in points]
            if missing:
                results.append({"index": index, "kind": kind, "status": "unresolved", "missing": missing})
                continue
            if len({frames[ref] for ref in refs}) > 1 or len({points[ref].size for ref in refs}) > 1:
                results.append({"index": index, "kind": kind, "status": "unresolved", "reason": "incompatible coordinate frames or dimensions"})
                continue
            coords = [points[ref] for ref in refs]
            residual: float
            if kind == "coincident":
                residual = float(np.linalg.norm(coords[0] - coords[1]))
            elif kind in ("distance", "minimum_clearance"):
                target = specification.get("target")
                if isinstance(target, bool) or not isinstance(target, (int, float)) or not np.isfinite(target) or target < 0:
                    raise SpatialValidationError("distance constraints need a finite non-negative target")
                measured = float(np.linalg.norm(coords[0] - coords[1]))
                residual = abs(measured - target) if kind == "distance" else max(0.0, target - measured)
            elif kind == "collinear":
                u, v = coords[1] - coords[0], coords[2] - coords[0]
                length = float(np.linalg.norm(u))
                residual = float(np.linalg.norm(v - np.dot(u, v) / (length * length) * u)) if length > tolerance else float("inf")
            else:
                u, v = coords[1] - coords[0], coords[3] - coords[2]
                lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
                if min(lu, lv) <= tolerance:
                    residual = float("inf")
                else:
                    cosine = float(np.clip(np.dot(u, v) / (lu * lv), -1, 1))
                    residual = abs(abs(cosine) - 1) if kind == "parallel" else abs(cosine)
            results.append({"index": index, "kind": kind,
                            "status": "unresolved" if not np.isfinite(residual) else ("satisfied" if residual <= tolerance else "violated"),
                            "residual": residual if np.isfinite(residual) else None,
                            "tolerance": tolerance})
        # Detect only contradictions provable from direct constraints on the
        # same referenced point pair. General nonlinear system consistency,
        # redundancy, and degrees of freedom require a separate solver.
        pair_constraints: dict[tuple[str, str], list[tuple[int, str, float]]] = {}
        for index, specification in enumerate(constraints):
            kind = specification["kind"]
            if kind not in {"coincident", "distance", "minimum_clearance"}:
                continue
            refs = specification["entities"]
            if results[index]["status"] == "unresolved":
                continue
            target = 0.0 if kind == "coincident" else float(specification["target"])
            pair_constraints.setdefault(tuple(sorted(refs)), []).append((index, kind, target))
        conflicts: list[dict[str, Any]] = []
        for pair, checks in pair_constraints.items():
            for left_index, left_kind, left_target in checks:
                for right_index, right_kind, right_target in checks:
                    if right_index <= left_index:
                        continue
                    left_exact = left_kind != "minimum_clearance"
                    right_exact = right_kind != "minimum_clearance"
                    if left_exact and right_exact:
                        contradictory = abs(left_target - right_target) > tolerance
                    elif left_exact:
                        contradictory = left_target + tolerance < right_target
                    else:
                        contradictory = right_exact and right_target + tolerance < left_target
                    if contradictory:
                        conflicts.append({"entities": list(pair), "constraint_indices": [left_index, right_index],
                                          "kind": "provable_pairwise_conflict"})
        return {"constraints": results, "satisfied": all(item["status"] == "satisfied" for item in results) and not conflicts,
                "unresolved": sum(item["status"] == "unresolved" for item in results),
                "consistency": "inconsistent" if conflicts else "not_proven",
                "conflicts": conflicts, "scope": "geometric-validation-only"}

    @staticmethod
    def validate_polygon(polygon: Polygon, *, tolerance: float = DEFAULT_ABS_TOL) -> dict[str, Any]:
        """Report duplicate/degenerate edges and crossings in a planar polygon.

        This is a read-only topology diagnostic, not a mesh repair operation.
        """
        if not isinstance(polygon, Polygon) or polygon.dimension != 2:
            raise SpatialTopologyError("validate_polygon requires a 2D Polygon")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise SpatialTopologyError("tolerance must be finite and non-negative")
        points = polygon.vertices
        n = len(points)
        findings: list[dict[str, Any]] = []
        for i in range(n):
            if float(np.linalg.norm(points[i] - points[(i + 1) % n])) <= tolerance:
                findings.append({"kind": "degenerate_edge", "edges": [i]})
        twice_area = float(sum(points[i, 0] * points[(i + 1) % n, 1] - points[(i + 1) % n, 0] * points[i, 1] for i in range(n)))
        if abs(twice_area) <= tolerance * tolerance:
            findings.append({"kind": "zero_signed_area"})
        for i in range(n):
            a, b = points[i], points[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue  # adjacent edges share a permitted endpoint
                c, d = points[j], points[(j + 1) % n]
                if segment_intersection_2d(Segment(a, b), Segment(c, d), tolerance=tolerance):
                    findings.append({"kind": "self_intersection", "edges": [i, j]})
        return {"valid": not findings, "findings": findings, "signed_area": twice_area / 2.0}

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
