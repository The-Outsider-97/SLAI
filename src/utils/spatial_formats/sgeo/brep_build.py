"""Builders for ready-made B-reps: box, cylinder / cone frustum, sphere, torus, planar extrusion, NURBS sheet.

Each builder returns a plain ``brep`` geometry dict that passes :func:`~.brep.validate_brep`; closed shapes
have outward facing faces, so :func:`~.brep_mesh.analyze_brep` reports a positive volume. They are the
reference for how periodic surfaces are bounded (a seam edge plus, for the sphere, degenerate pole edges).
"""
from __future__ import annotations

from math import atan, hypot, pi, sqrt

from ..common.errors import GeometryValidationError
from ..common.helpers import finite, vector
from . import nurbs
from .brep import validate_brep

__all__ = ["brep_box", "brep_cylinder", "brep_sphere", "brep_torus", "brep_extrusion", "brep_from_nurbs_surface"]

_X, _Y, _Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
_MAX_PROFILE_POINTS = 2048


def _fail(message):
    raise GeometryValidationError(message)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(a):
    n = sqrt(_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


class _Builder:
    def __init__(self):
        self.vertices, self.curves, self.surfaces, self.edges, self.faces = [], [], [], [], []
        self._edge_lookup = {}

    def vertex(self, p):
        self.vertices.append([float(c) for c in p])
        return len(self.vertices) - 1

    def curve(self, record):
        self.curves.append(record)
        return len(self.curves) - 1

    def surface(self, record):
        self.surfaces.append(record)
        return len(self.surfaces) - 1

    def line_edge(self, a, b):
        """The straight edge between vertices ``a`` and ``b`` (created once); returns ``(edge, forward)`` for a->b."""
        key = (min(a, b), max(a, b))
        if key not in self._edge_lookup:
            lo, hi = key
            p, q = self.vertices[lo], self.vertices[hi]
            direction = _unit(_sub(q, p))
            curve = self.curve({"type": "line", "origin": list(p), "direction": list(direction)})
            length = hypot(hypot(q[0] - p[0], q[1] - p[1]), q[2] - p[2])
            self.edges.append({"curve": curve, "start": lo, "end": hi, "range": [0.0, length]})
            self._edge_lookup[key] = len(self.edges) - 1
        return self._edge_lookup[key], a < b

    def planar_face(self, loops, normal, ref=None):
        """A face on a plane whose outward normal is ``normal``; ``loops`` are vertex cycles (outer first),
        the outer one counter-clockwise about ``normal`` and holes clockwise."""
        origin = self.vertices[loops[0][0]]
        if ref is None:
            ref = _sub(self.vertices[loops[0][1]], origin)
        ref = _sub(ref, tuple(c * _dot(ref, normal) for c in normal))
        plane = self.surface({"type": "plane", "origin": list(origin), "axis": list(_unit(normal)),
                              "ref": list(_unit(ref))})
        coloops = []
        for cycle in loops:
            coloops.append([{"edge": e, "forward": f} for e, f in
                            (self.line_edge(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle)))])
        self.faces.append({"surface": plane, "reversed": False, "loops": coloops})

    def build(self):
        brep = {"kind": "brep", "vertices": self.vertices, "curves": self.curves, "surfaces": self.surfaces,
                "edges": self.edges, "faces": self.faces}
        validate_brep(brep)
        return brep


def _centre(center):
    return [float(c) for c in vector(center, 3, "center")]


def brep_box(size, center=(0.0, 0.0, 0.0)) -> dict:
    """Axis-aligned box: 8 vertices, 12 line edges, 6 planar faces."""
    sx, sy, sz = (finite(c, "size") / 2 for c in vector(size, 3, "size"))
    if min(sx, sy, sz) <= 0:
        _fail("box dimensions must be positive")
    cx, cy, cz = _centre(center)
    b = _Builder()
    corner = {}
    for ix in (-1, 1):
        for iy in (-1, 1):
            for iz in (-1, 1):
                corner[(ix, iy, iz)] = b.vertex((cx + ix * sx, cy + iy * sy, cz + iz * sz))
    for axis in range(3):
        for sign in (-1, 1):
            others = [i for i in range(3) if i != axis]
            cycle = []
            for da, db in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                key = [0, 0, 0]
                key[axis], key[others[0]], key[others[1]] = sign, da, db
                cycle.append(corner[tuple(key)])
            normal = [0.0, 0.0, 0.0]
            normal[axis] = float(sign)
            p = [b.vertices[i] for i in cycle[:3]]
            if _dot(_cross3(_sub(p[1], p[0]), _sub(p[2], p[1])), normal) < 0:
                cycle.reverse()
            b.planar_face([cycle], normal)
    return b.build()


def brep_cylinder(radius, height, center=(0.0, 0.0, 0.0), radius_top=None) -> dict:
    """Right circular cylinder along +Z (a cone frustum when ``radius_top`` differs from ``radius``)."""
    r = finite(radius, "radius")
    h = finite(height, "height")
    rt = r if radius_top is None else finite(radius_top, "radius_top")
    if r <= 0 or h <= 0 or rt <= 0:
        _fail("radius, radius_top and height must be positive")
    cx, cy, cz = _centre(center)
    z0, z1 = cz - h / 2, cz + h / 2
    b = _Builder()
    vb, vt = b.vertex((cx + r, cy, z0)), b.vertex((cx + rt, cy, z1))
    c_bottom = b.curve({"type": "circle", "origin": [cx, cy, z0], "axis": list(_Z), "ref": list(_X), "radius": r})
    c_top = b.curve({"type": "circle", "origin": [cx, cy, z1], "axis": list(_Z), "ref": list(_X), "radius": rt})
    seam_dir = _unit((rt - r, 0.0, h))
    c_seam = b.curve({"type": "line", "origin": [cx + r, cy, z0], "direction": list(seam_dir)})
    e_bottom = len(b.edges)
    b.edges.append({"curve": c_bottom, "start": vb, "end": vb, "range": [0.0, 2 * pi]})
    b.edges.append({"curve": c_top, "start": vt, "end": vt, "range": [0.0, 2 * pi]})
    b.edges.append({"curve": c_seam, "start": vb, "end": vt, "range": [0.0, hypot(rt - r, h)]})
    e_top, e_seam = e_bottom + 1, e_bottom + 2
    if abs(rt - r) <= 1e-12 * max(r, rt):
        side = b.surface({"type": "cylinder", "origin": [cx, cy, z0], "axis": list(_Z), "ref": list(_X), "radius": r})
    else:
        side = b.surface({"type": "cone", "origin": [cx, cy, z0], "axis": list(_Z), "ref": list(_X), "radius": r,
                          "half_angle": atan((rt - r) / h)})
    bottom = b.surface({"type": "plane", "origin": [cx, cy, z0], "axis": [0.0, 0.0, -1.0], "ref": list(_X)})
    top = b.surface({"type": "plane", "origin": [cx, cy, z1], "axis": list(_Z), "ref": list(_X)})
    tau = 2 * pi
    b.faces.append({"surface": side, "reversed": False, "loops": [[
        {"edge": e_bottom, "forward": True, "pcurve": [[0.0, 0.0], [tau, 0.0]]},
        {"edge": e_seam, "forward": True, "pcurve": [[tau, 0.0], [tau, h]]},
        {"edge": e_top, "forward": False, "pcurve": [[tau, h], [0.0, h]]},
        {"edge": e_seam, "forward": False, "pcurve": [[0.0, h], [0.0, 0.0]]}]]})
    b.faces.append({"surface": bottom, "reversed": False, "loops": [[{"edge": e_bottom, "forward": False}]]})
    b.faces.append({"surface": top, "reversed": False, "loops": [[{"edge": e_top, "forward": True}]]})
    return b.build()


def brep_sphere(radius, center=(0.0, 0.0, 0.0)) -> dict:
    """Sphere as one face: a seam meridian plus a degenerate edge at each pole."""
    r = finite(radius, "radius")
    if r <= 0:
        _fail("radius must be positive")
    cx, cy, cz = _centre(center)
    b = _Builder()
    south, north = b.vertex((cx, cy, cz - r)), b.vertex((cx, cy, cz + r))
    meridian = b.curve({"type": "circle", "origin": [cx, cy, cz], "axis": [0.0, -1.0, 0.0], "ref": list(_X),
                        "radius": r})
    b.edges.append({"curve": meridian, "start": south, "end": north, "range": [-pi / 2, pi / 2]})
    b.edges.append({"curve": None, "start": south, "end": south})
    b.edges.append({"curve": None, "start": north, "end": north})
    sphere = b.surface({"type": "sphere", "origin": [cx, cy, cz], "axis": list(_Z), "ref": list(_X), "radius": r})
    tau, half = 2 * pi, pi / 2
    b.faces.append({"surface": sphere, "reversed": False, "loops": [[
        {"edge": 1, "forward": True, "pcurve": [[0.0, -half], [tau, -half]]},
        {"edge": 0, "forward": True, "pcurve": [[tau, -half], [tau, half]]},
        {"edge": 2, "forward": True, "pcurve": [[tau, half], [0.0, half]]},
        {"edge": 0, "forward": False, "pcurve": [[0.0, half], [0.0, -half]]}]]})
    return b.build()


def brep_torus(major_radius, minor_radius, center=(0.0, 0.0, 0.0)) -> dict:
    """Torus about +Z as one face bounded by its two seam circles."""
    big, small = finite(major_radius, "major_radius"), finite(minor_radius, "minor_radius")
    if small <= 0 or big <= small:
        _fail("torus radii must satisfy 0 < minor_radius < major_radius")
    cx, cy, cz = _centre(center)
    b = _Builder()
    v0 = b.vertex((cx + big + small, cy, cz))
    c_u = b.curve({"type": "circle", "origin": [cx, cy, cz], "axis": list(_Z), "ref": list(_X), "radius": big + small})
    c_v = b.curve({"type": "circle", "origin": [cx + big, cy, cz], "axis": [0.0, -1.0, 0.0], "ref": list(_X),
                   "radius": small})
    b.edges.append({"curve": c_u, "start": v0, "end": v0, "range": [0.0, 2 * pi]})
    b.edges.append({"curve": c_v, "start": v0, "end": v0, "range": [0.0, 2 * pi]})
    torus = b.surface({"type": "torus", "origin": [cx, cy, cz], "axis": list(_Z), "ref": list(_X),
                       "major_radius": big, "minor_radius": small})
    tau = 2 * pi
    b.faces.append({"surface": torus, "reversed": False, "loops": [[
        {"edge": 0, "forward": True, "pcurve": [[0.0, 0.0], [tau, 0.0]]},
        {"edge": 1, "forward": True, "pcurve": [[tau, 0.0], [tau, tau]]},
        {"edge": 0, "forward": False, "pcurve": [[tau, tau], [0.0, tau]]},
        {"edge": 1, "forward": False, "pcurve": [[0.0, tau], [0.0, 0.0]]}]]})
    return b.build()


def _orient(points, ccw):
    area = sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1]
               for i in range(len(points)))
    if area == 0:
        _fail("profile loop encloses no area")
    return points if (area > 0) == ccw else points[::-1]


def _segments_cross(p, q, r, s):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = orient(p, q, r), orient(p, q, s), orient(r, s, p), orient(r, s, q)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


def _check_simple(loops):
    segments = []
    for li, loop in enumerate(loops):
        for i in range(len(loop)):
            segments.append((li, i, loop[i], loop[(i + 1) % len(loop)]))
    for a in range(len(segments)):
        la, ia, p, q = segments[a]
        for bi in range(a + 1, len(segments)):
            lb, ib, r, s = segments[bi]
            adjacent = la == lb and (ib == ia + 1 or (ia == 0 and ib == len(loops[la]) - 1))
            if adjacent:
                continue
            if _segments_cross(p, q, r, s):
                _fail("profile edges cross each other")
    # a hole must lie inside the outer loop and holes must not nest or overlap
    def inside(point, loop):
        x, y = point
        hit = False
        for i in range(len(loop)):
            (x1, y1), (x2, y2) = loop[i], loop[(i + 1) % len(loop)]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                hit = not hit
        return hit
    for hole in loops[1:]:
        if not inside(hole[0], loops[0]):
            _fail("a hole lies outside the profile")
        for other in loops[1:]:
            if other is not hole and inside(hole[0], other):
                _fail("holes must not be nested")


def brep_extrusion(profile, depth, holes=None) -> dict:
    """Prism: a simple planar XY ``profile`` (may be concave) with optional ``holes``, extruded from z=0 to ``depth``."""
    d = finite(depth, "depth")
    if d <= 0:
        _fail("extrusion depth must be positive")

    def loop_points(points, label):
        if not isinstance(points, (list, tuple)) or not 3 <= len(points) <= _MAX_PROFILE_POINTS:
            _fail("%s needs 3..%d points" % (label, _MAX_PROFILE_POINTS))
        pts = [tuple(float(c) for c in vector(p, 2, label)) for p in points]
        if pts[0] == pts[-1]:
            pts.pop()
        if len(pts) < 3 or len(set(pts)) != len(pts):
            _fail(label + " has repeated points")
        return pts

    outer = _orient(loop_points(profile, "profile"), True)
    rings = [outer] + [_orient(loop_points(h, "hole"), False) for h in (holes or [])]
    if sum(len(r) for r in rings) > _MAX_PROFILE_POINTS:
        _fail("profile and holes may have at most %d points together" % _MAX_PROFILE_POINTS)
    _check_simple(rings)
    b = _Builder()
    bottoms, tops = [], []
    for ring in rings:
        bottoms.append([b.vertex((x, y, 0.0)) for x, y in ring])
        tops.append([b.vertex((x, y, d)) for x, y in ring])
    b.planar_face([list(reversed(r)) for r in bottoms], (0.0, 0.0, -1.0), _X)
    b.planar_face(tops, _Z, _X)
    for bottom, top in zip(bottoms, tops):
        n = len(bottom)
        for i in range(n):
            j = (i + 1) % n
            a, c = b.vertices[bottom[i]], b.vertices[bottom[j]]
            normal = (c[1] - a[1], -(c[0] - a[0]), 0.0)
            b.planar_face([[bottom[i], bottom[j], top[j], top[i]]], normal)
    return b.build()


def brep_from_nurbs_surface(surface) -> dict:
    """A one-face open sheet on a NURBS surface bounded by its four boundary curves."""
    record = {k: v for k, v in surface.items() if k not in ("kind", "type")}
    nurbs.validate_surface(record)
    net, weights = record["control_points"], record.get("weights")
    nu, nv = len(net), len(net[0])
    (u0, u1), (v0, v1) = nurbs.surface_domain(record)
    b = _Builder()
    corners = [net[0][0], net[nu - 1][0], net[nu - 1][nv - 1], net[0][nv - 1]]
    c = [b.vertex(p) for p in corners]

    def edge_curve(points, degree, knots, w):
        data = {"type": "nurbs", "degree": degree, "knots": list(knots), "control_points": [list(p) for p in points]}
        if w is not None:
            data["weights"] = list(w)
        return b.curve(data)

    pu, pv, ku, kv = record["degree_u"], record["degree_v"], record["knots_u"], record["knots_v"]
    wcol = (lambda j: [weights[i][j] for i in range(nu)]) if weights else (lambda j: None)
    wrow = (lambda i: list(weights[i])) if weights else (lambda i: None)
    cb = edge_curve([net[i][0] for i in range(nu)], pu, ku, wcol(0))
    cr = edge_curve(list(net[nu - 1]), pv, kv, wrow(nu - 1))
    ct = edge_curve([net[i][nv - 1] for i in range(nu)], pu, ku, wcol(nv - 1))
    cl = edge_curve(list(net[0]), pv, kv, wrow(0))
    b.edges.extend([{"curve": cb, "start": c[0], "end": c[1], "range": [u0, u1]},
                    {"curve": cr, "start": c[1], "end": c[2], "range": [v0, v1]},
                    {"curve": ct, "start": c[3], "end": c[2], "range": [u0, u1]},
                    {"curve": cl, "start": c[0], "end": c[3], "range": [v0, v1]}])
    sid = b.surface({"type": "nurbs", **record})
    b.faces.append({"surface": sid, "reversed": False, "loops": [[
        {"edge": 0, "forward": True, "pcurve": [[u0, v0], [u1, v0]]},
        {"edge": 1, "forward": True, "pcurve": [[u1, v0], [u1, v1]]},
        {"edge": 2, "forward": False, "pcurve": [[u1, v1], [u0, v1]]},
        {"edge": 3, "forward": False, "pcurve": [[u0, v1], [u0, v0]]}]]})
    return b.build()
