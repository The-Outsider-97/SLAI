"""Deterministic rectilinear Boolean solids for two axis-aligned boxes.

This intentionally restricted algorithm partitions the input along their six
axis-aligned boundary planes and extracts the occupied cells' *outer* faces.
The resulting triangle mesh represents the exact set-theoretic polyhedron;
it does not provide general B-rep Boolean operations, curved surfaces, or
persistent face naming. No floating-point tolerance is silently introduced.
"""
from __future__ import annotations

from ..common.errors import GeometryValidationError, UnsupportedFeatureError, ValidationError
from .analysis import analyze_mesh
from .geometry import validate_geometry


def boolean_boxes(first: dict, second: dict, operation: str) -> dict | None:
    """Union, intersection or subtract closed SGEO axis-aligned boxes.

    Returns a validated indexed triangle mesh, or ``None`` for an empty set.
    A mesh is an exact planar boundary *representation* here, not a claim that
    an arbitrary mesh modeling kernel is available.
    """
    if operation not in {"union", "intersection", "difference"}:
        raise ValidationError("box Boolean operation must be union, intersection or difference")
    for geometry in (first, second):
        validate_geometry(geometry)
        if geometry["kind"] != "box":
            raise UnsupportedFeatureError("rectilinear Boolean backend accepts only axis-aligned boxes")

    def limits(box):
        return tuple((float(box["center"][i] - box["size"][i] / 2),
                      float(box["center"][i] + box["size"][i] / 2)) for i in range(3))

    a, b = limits(first), limits(second)
    planes = [sorted({a[i][0], a[i][1], b[i][0], b[i][1]}) for i in range(3)]
    occupied: set[tuple[int, int, int]] = set()
    for i in range(len(planes[0]) - 1):
        for j in range(len(planes[1]) - 1):
            for k in range(len(planes[2]) - 1):
                center = tuple((planes[axis][index] + planes[axis][index + 1]) / 2
                               for axis, index in enumerate((i, j, k)))
                inside_a = all(a[axis][0] < center[axis] < a[axis][1] for axis in range(3))
                inside_b = all(b[axis][0] < center[axis] < b[axis][1] for axis in range(3))
                keep = (inside_a or inside_b if operation == "union" else
                        inside_a and inside_b if operation == "intersection" else
                        inside_a and not inside_b)
                if keep:
                    occupied.add((i, j, k))
    if not occupied:
        return None

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    indices: dict[tuple[float, float, float], int] = {}

    def add_vertex(position):
        key = tuple(position)
        if key not in indices:
            indices[key] = len(vertices)
            vertices.append(list(key))
        return indices[key]

    for i, j, k in sorted(occupied):
        x0, x1 = planes[0][i:i + 2]
        y0, y1 = planes[1][j:j + 2]
        z0, z1 = planes[2][k:k + 2]
        # Winding is outward. Coplanar faces are subdivided on the *same*
        # grid, avoiding T-junctions across cell boundaries.
        sides = (
            ((i - 1, j, k), [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], False),
            ((i + 1, j, k), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], True),
            ((i, j - 1, k), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], True),
            ((i, j + 1, k), [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], False),
            ((i, j, k - 1), [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], False),
            ((i, j, k + 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], True),
        )
        for neighbor, corners, forward in sides:
            if neighbor in occupied:
                continue
            quad = [add_vertex(corner) for corner in corners]
            if not forward:
                quad.reverse()
            faces.extend(([quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]))
    if not faces:
        raise GeometryValidationError("Boolean result has no boundary")
    result = validate_geometry({"kind": "mesh", "vertices": vertices, "faces": faces})
    diagnostics = analyze_mesh(result)
    if not diagnostics.is_closed_oriented_surface:
        raise UnsupportedFeatureError(
            "rectilinear Boolean result has non-manifold contact (edge or corner touching solids)")
    return result


__all__ = ["boolean_boxes"]
