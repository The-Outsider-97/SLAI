"""Exact split of an axis-aligned SGEO box by an arbitrary plane.

A box is convex, so the part on the kept side of a plane is the convex polyhedron obtained by clipping each of its six faces against the half-space
(Sutherland-Hodgman) and closing the cut with one polygon. Every new vertex lies on a box edge and is computed once per edge from that edge's two end
distances, so the faces on both sides of the edge share one vertex exactly and the result is closed and consistently oriented by construction; it
is then verified (every edge used once in each direction, positive signed volume) and refused with the measured defect if it is not.

No numerical mesh library is needed, so the result does not depend on trimesh, shapely, scipy or a triangulation engine, and it is not an approximation.
The only tolerance is the classification of a vertex as lying *on* the plane: ``tolerance`` (default 1e-9 times the box's largest extent). A plane that
touches the box, or lies within that distance of a face, therefore leaves it unchanged or removes it entirely instead of producing a sliver.

The kept side is the positive side of ``normal`` (the same convention as the axis-aligned analytic split).
"""
from __future__ import annotations

from math import atan2, sqrt
from typing import Sequence

from ..common.errors import GeometryValidationError
from .geometry import validate_geometry

DEFAULT_RELATIVE_TOLERANCE = 1e-9


def _corners(box: dict) -> list[list[float]]:
    c, s = box["center"], box["size"]
    return [[c[0] + (s[0] / 2 if i & 1 else -s[0] / 2), c[1] + (s[1] / 2 if i & 2 else -s[1] / 2), c[2] + (s[2] / 2 if i & 4 else -s[2] / 2)] for i in range(8)]


def _faces() -> list[list[int]]:
    """The six faces as corner-index loops, each wound counter-clockwise seen from outside the box."""
    faces = []
    for axis in range(3):
        for side in (0, 1):
            others = [a for a in range(3) if a != axis]
            loop = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                bits = (side << axis) | (u << others[0]) | (v << others[1])
                loop.append(bits)
            # corner bits: x -> 1, y -> 2, z -> 4; the loop above is counter-clockwise from +axis only when (others[0], others[1], axis) is a right-handed frame
            handed = (others[0], others[1], axis) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            if (side == 1) != handed:
                loop.reverse()
            faces.append(loop)
    return faces


def split_box(box: dict, normal: Sequence[float], origin: Sequence[float], *, tolerance: float | None = None) -> tuple[dict | None, str, dict]:
    """``(geometry, outcome, info)``: ``geometry`` is the closed triangle mesh of the part on the positive side, the box itself (``unchanged``), or ``None`` (``empty``)."""
    length = sqrt(sum(v * v for v in normal))
    if not length:
        raise GeometryValidationError("split plane normal must be nonzero")
    n = [v / length for v in normal]
    corners = _corners(box)
    extent = max(box["size"])
    eps = DEFAULT_RELATIVE_TOLERANCE * extent if tolerance is None else float(tolerance)
    raw = [sum(n[i] * (p[i] - origin[i]) for i in range(3)) for p in corners]
    dist = [0.0 if abs(d) <= eps else d for d in raw]
    info = {"tolerance": eps, "distances": raw}
    if not any(d < 0 for d in dist):
        return box, "unchanged", info
    if not any(d > 0 for d in dist):
        return None, "empty", info

    points: list[list[float]] = []
    index: dict = {}

    def vertex(key):
        found = index.get(key)
        if found is None:
            if isinstance(key, int):
                p = corners[key]
            else:                                                          # a new vertex on the box edge (a, b), a < b, from that edge's end distances only
                a, b = key
                t = dist[a] / (dist[a] - dist[b])
                p = [corners[a][i] + t * (corners[b][i] - corners[a][i]) for i in range(3)]
            found = index[key] = len(points)
            points.append(p)
        return found

    triangles: list[list[int]] = []
    cut: set = set()
    for loop in _faces():
        clipped = []
        for k, a in enumerate(loop):
            b = loop[(k + 1) % 4]
            if dist[a] >= 0:
                clipped.append(a)
            if (dist[a] > 0 > dist[b]) or (dist[a] < 0 < dist[b]):
                clipped.append((min(a, b), max(a, b)))
        if len(clipped) < 3 or all(isinstance(k, int) and dist[k] == 0 for k in clipped):
            continue                                                       # removed, or lying in the plane (replaced by the cut polygon)
        ids = [vertex(k) for k in clipped]
        for k in clipped:
            if (isinstance(k, tuple)) or dist[k] == 0:
                cut.add(k)
        triangles += [[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)]

    # the cut polygon, in the plane, wound so that its outward normal is -n (the kept material is on the +n side)
    cut_ids = [vertex(k) for k in cut]
    helper = [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = [helper[1] * n[2] - helper[2] * n[1], helper[2] * n[0] - helper[0] * n[2], helper[0] * n[1] - helper[1] * n[0]]
    ul = sqrt(sum(x * x for x in u))
    u = [x / ul for x in u]
    v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]]
    centre = [sum(points[i][c] for i in cut_ids) / len(cut_ids) for c in range(3)]
    ordered = sorted(cut_ids, key=lambda i: -atan2(sum((points[i][c] - centre[c]) * v[c] for c in range(3)), sum((points[i][c] - centre[c]) * u[c] for c in range(3))))
    hub = len(points)
    points.append(centre)
    triangles += [[hub, ordered[k], ordered[(k + 1) % len(ordered)]] for k in range(len(ordered))]

    result = {"kind": "mesh", "vertices": points, "faces": triangles}
    _verify(result, info)
    return validate_geometry(result), "cut", info


def _verify(mesh: dict, info: dict) -> None:
    uses: dict = {}
    volume = 0.0
    ref = mesh["vertices"][0]                                              # relative to a vertex: from the world origin a distant box loses all precision
    for a, b, c in mesh["faces"]:
        for p, q in ((a, b), (b, c), (c, a)):
            uses.setdefault((min(p, q), max(p, q)), []).append((p, q))
        pa, pb, pc = ([mesh["vertices"][i][k] - ref[k] for k in range(3)] for i in (a, b, c))
        volume += (pa[0] * (pb[1] * pc[2] - pb[2] * pc[1]) + pa[1] * (pb[2] * pc[0] - pb[0] * pc[2]) + pa[2] * (pb[0] * pc[1] - pb[1] * pc[0])) / 6.0
    bad = [e for e, d in uses.items() if len(d) != 2 or d[0] == d[1]]
    if bad or volume <= 0:
        raise GeometryValidationError("box split produced an invalid solid", context={"bad_edges": len(bad), "volume": volume, "tolerance": info["tolerance"]})
    info["volume"] = volume
