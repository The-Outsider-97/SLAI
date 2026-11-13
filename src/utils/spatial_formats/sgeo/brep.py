"""SGEO 0.2 boundary representation (B-rep): topology records and their structural validation.

A ``brep`` geometry is a self-contained set of bounded faces::

    {"kind": "brep",
     "vertices": [[x, y, z], ...],
     "curves":   [curve, ...],          # see surfaces.py: line, circle, nurbs
     "surfaces": [surface, ...],        # plane, cylinder, cone, sphere, torus, nurbs
     "edges":    [{"curve": i | null, "start": v, "end": v, "range": [t0, t1]}, ...],
     "faces":    [{"surface": s, "reversed": false, "loops": [[coedge, ...], ...]}, ...]}

    coedge = {"edge": e, "forward": true, "pcurve": [[u0, v0], [u1, v1]]}      # pcurve optional

* An **edge** runs from ``start`` (curve parameter ``t0``) to ``end`` (``t1``) along its curve. A *degenerate*
  edge has ``"curve": null`` and ``start == end``: it collapses to a point (the pole of a sphere).
* A **face** lies on a surface. Its first loop is the outer boundary, the others are holes. Walking a loop in
  the listed order, with the coedge ``forward`` flags applied, the face's material is on the left when seen
  from outside: outer loops run counter-clockwise about the outward normal, holes clockwise. The outward
  normal is the surface's natural normal, or its opposite when ``reversed`` is true.
* A **coedge** is one use of an edge by a loop. ``pcurve`` optionally gives the straight segment the coedge
  follows in the surface's ``(u, v)`` plane, *in traversal order* (from the coedge's start to its end).
  It is required for degenerate edges and for edges on periodic surfaces whose path is ambiguous without it;
  the tessellator projects edge points onto the surface when it is absent.

Consistency rules checked here (cheap, topological and endpoint geometry): every edge is used by one or two
coedges, twice only with opposite senses; consecutive coedges of a loop meet at a vertex and every loop
closes; edge ends coincide with their vertices; nothing is unreferenced. A brep whose non-degenerate edges
are all used twice is **closed** (a valid solid boundary when its faces face outward); otherwise it is an
open sheet. Faces on periodic surfaces (cylinder, cone, sphere, torus) must close in the ``(u, v)`` plane, so
they include their seam edge, exactly as STEP does. Orientation of loops and watertightness of the meshed
result are verified by :func:`~.brep_mesh.analyze_brep`.

There is no modelling kernel: no booleans, fillets or offsets, and no STEP geometry is read here.
"""
from __future__ import annotations

from math import hypot

from ..common.errors import GeometryValidationError, ResourceLimitError
from ..common.helpers import finite, vector
from . import surfaces as S

__all__ = ["BREP_KEYS", "validate_brep", "scale_brep", "brep_info", "BrepInfo", "MAX_BREP_VERTICES",
           "MAX_BREP_EDGES", "MAX_BREP_FACES", "MAX_BREP_COEDGES"]

BREP_KEYS = frozenset({"kind", "vertices", "curves", "surfaces", "edges", "faces"})
MAX_BREP_VERTICES = 100_000
MAX_BREP_EDGES = 200_000
MAX_BREP_FACES = 100_000
MAX_BREP_COEDGES = 400_000
MAX_BREP_CURVES = 200_000
MAX_BREP_SURFACES = 100_000
MAX_BREP_CONTROL_POINTS = 100_000
MAX_LOOPS_PER_FACE = 256
_REL_TOL = 1e-6


def _fail(message, **context):
    raise GeometryValidationError(message, context=context or None)


def _index(value, count, label):
    if type(value) is not int or not 0 <= value < count:
        _fail("%s must be an index in [0, %d)" % (label, count))
    return value


def _list(value, label, limit, *, minimum=0):
    if not isinstance(value, list):
        _fail(label + " must be a list")
    if len(value) > limit:
        raise ResourceLimitError("B-rep has too many %s (limit %d)" % (label, limit))
    if len(value) < minimum:
        _fail("B-rep needs at least %d %s" % (minimum, label))
    return value


class BrepInfo(dict):
    """Counts and flags from validation: vertices, edges, faces, coedges, closed, euler, tolerance."""


def _tolerance(vertices) -> float:
    lo = [min(v[i] for v in vertices) for i in range(3)]
    hi = [max(v[i] for v in vertices) for i in range(3)]
    diag = hypot(hypot(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2])
    magnitude = max(abs(c) for v in vertices for c in v)
    return _REL_TOL * max(diag, magnitude, 1e-3)


def _near(a, b, tol):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol and abs(a[2] - b[2]) <= tol


def validate_brep(b) -> BrepInfo:
    """Validate the structure of a brep geometry and return its :class:`BrepInfo`."""
    if not isinstance(b, dict) or b.get("kind") != "brep":
        _fail("not a brep geometry")
    if set(b) != BREP_KEYS:
        _fail("a brep needs exactly: " + ", ".join(sorted(BREP_KEYS - {"kind"})))
    vertices = _list(b["vertices"], "vertices", MAX_BREP_VERTICES, minimum=1)
    for v in vertices:
        vector(v, 3, "vertex")
    curves = _list(b["curves"], "curves", MAX_BREP_CURVES)
    surfaces = _list(b["surfaces"], "surfaces", MAX_BREP_SURFACES, minimum=1)
    edges = _list(b["edges"], "edges", MAX_BREP_EDGES)
    faces = _list(b["faces"], "faces", MAX_BREP_FACES, minimum=1)
    control_points = 0
    for curve in curves:
        S.validate_curve_def(curve)
        control_points += S.curve_point_budget(curve)
    for surface in surfaces:
        S.validate_surface_def(surface)
        control_points += S.surface_point_budget(surface)
    if control_points > MAX_BREP_CONTROL_POINTS:
        raise ResourceLimitError("B-rep NURBS data exceeds %d control points" % MAX_BREP_CONTROL_POINTS)
    tol = _tolerance(vertices)

    # vertices must be distinct points (merge coincident ones so faces share them)
    seen = {}
    cell = max(tol * 4, 1e-12)
    for i, v in enumerate(vertices):
        key = tuple(round(c / cell) for c in v)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = seen.get((key[0] + dx, key[1] + dy, key[2] + dz))
                    if other is not None and _near(vertices[other], v, tol):
                        _fail("vertices %d and %d coincide; merge them so faces share one vertex" % (other, i))
        seen[key] = i

    used_vertices, used_curves, used_surfaces = set(), set(), set()
    degenerate = []
    for position, edge in enumerate(edges):
        if not isinstance(edge, dict) or set(edge) - {"curve", "start", "end", "range"} or \
                not {"curve", "start", "end"} <= set(edge):
            _fail("edge %d needs curve, start, end (and range for a curve edge)" % position)
        start = _index(edge["start"], len(vertices), "edge start")
        end = _index(edge["end"], len(vertices), "edge end")
        used_vertices.update((start, end))
        if edge["curve"] is None:
            if "range" in edge or start != end:
                _fail("degenerate edge %d must have a single vertex and no range" % position)
            degenerate.append(True)
            continue
        degenerate.append(False)
        ci = _index(edge["curve"], len(curves), "edge curve")
        used_curves.add(ci)
        window = edge.get("range")
        if not isinstance(window, list) or len(window) != 2:
            _fail("edge %d needs range [t0, t1]" % position)
        t0, t1 = finite(window[0], "range"), finite(window[1], "range")
        if not t0 < t1:
            _fail("edge %d range must satisfy t0 < t1" % position)
        curve = curves[ci]
        if curve["type"] == "circle" and t1 - t0 > S.TWO_PI + 1e-9:
            _fail("circle edge %d spans more than a full turn" % position)
        if curve["type"] == "nurbs":
            lo, hi = S.nurbs.curve_domain(curve)
            if t0 < lo - 1e-9 or t1 > hi + 1e-9:
                _fail("edge %d range lies outside its NURBS curve domain" % position)
        if not _near(S.curve_eval(curve, t0), vertices[start], tol) or \
                not _near(S.curve_eval(curve, t1), vertices[end], tol):
            _fail("edge %d does not run between its vertices" % position)
        if start != end and _near(vertices[start], vertices[end], tol):
            _fail("edge %d has zero length" % position)

    uses = [[] for _ in edges]
    coedges = 0
    for fi, face in enumerate(faces):
        if not isinstance(face, dict) or set(face) - {"surface", "reversed", "loops"} or \
                not {"surface", "loops"} <= set(face):
            _fail("face %d needs surface and loops (and optionally reversed)" % fi)
        used_surfaces.add(_index(face["surface"], len(surfaces), "face surface"))
        if "reversed" in face and not isinstance(face["reversed"], bool):
            _fail("face %d reversed must be a boolean" % fi)
        loops = _list(face["loops"], "loops of face %d" % fi, MAX_LOOPS_PER_FACE, minimum=1)
        for li, loop in enumerate(loops):
            loop = _list(loop, "coedges of face %d loop %d" % (fi, li), MAX_BREP_COEDGES, minimum=1)
            coedges += len(loop)
            if coedges > MAX_BREP_COEDGES:
                raise ResourceLimitError("B-rep has too many coedges (limit %d)" % MAX_BREP_COEDGES)
            ends = []
            for co in loop:
                if not isinstance(co, dict) or set(co) - {"edge", "forward", "pcurve"} or \
                        not {"edge", "forward"} <= set(co):
                    _fail("coedge in face %d loop %d needs edge and forward (and optionally pcurve)" % (fi, li))
                e = _index(co["edge"], len(edges), "coedge edge")
                if not isinstance(co["forward"], bool):
                    _fail("coedge forward must be a boolean")
                if "pcurve" in co:
                    pc = co["pcurve"]
                    if not isinstance(pc, list) or len(pc) != 2:
                        _fail("pcurve must be [[u0, v0], [u1, v1]]")
                    for point in pc:
                        vector(point, 2, "pcurve point")
                elif degenerate[e]:
                    _fail("degenerate edge %d needs a pcurve on every coedge that uses it" % e)
                uses[e].append(co["forward"])
                edge = edges[e]
                ends.append((edge["start"], edge["end"]) if co["forward"] else (edge["end"], edge["start"]))
            for k, (_, stop) in enumerate(ends):
                if stop != ends[(k + 1) % len(ends)][0]:
                    _fail("loop %d of face %d is not connected at coedge %d" % (li, fi, k))

    closed = True
    for e, flags in enumerate(uses):
        if not flags:
            _fail("edge %d is not used by any face" % e)
        if degenerate[e]:
            continue
        if len(flags) > 2:
            _fail("edge %d is shared by more than two faces (non-manifold)" % e)
        if len(flags) == 2 and flags[0] == flags[1]:
            _fail("edge %d is traversed twice in the same direction (inconsistent orientation)" % e)
        if len(flags) == 1:
            closed = False
    if len(used_vertices) != len(vertices):
        _fail("B-rep has vertices that no edge uses")
    if len(used_curves) != len(curves):
        _fail("B-rep has curves that no edge uses")
    if len(used_surfaces) != len(surfaces):
        _fail("B-rep has surfaces that no face uses")
    real_edges = sum(1 for d in degenerate if not d)
    return BrepInfo(vertices=len(vertices), edges=real_edges, degenerate_edges=len(edges) - real_edges,
                    faces=len(faces), coedges=coedges, closed=closed,
                    euler=len(vertices) - real_edges + len(faces), tolerance=tol)


def brep_info(b) -> BrepInfo:
    """Structural facts of a brep (validates first)."""
    return validate_brep(b)


def scale_brep(b, f) -> dict:
    """The brep with every length multiplied by ``f`` (edge and ``pcurve`` parameters that are lengths included)."""
    out = {"kind": "brep", "vertices": [[c * f for c in v] for v in b["vertices"]],
           "curves": [S.scale_curve_def(c, f) for c in b["curves"]],
           "surfaces": [S.scale_surface_def(s, f) for s in b["surfaces"]]}
    edges = []
    for edge in b["edges"]:
        e = dict(edge)
        if edge["curve"] is not None and b["curves"][edge["curve"]]["type"] == "line":
            e["range"] = [edge["range"][0] * f, edge["range"][1] * f]
        edges.append(e)
    out["edges"] = edges
    faces = []
    for face in b["faces"]:
        lengths = S.surface_pcurve_lengths(b["surfaces"][face["surface"]])
        loops = []
        for loop in face["loops"]:
            new = []
            for co in loop:
                c = dict(co)
                if "pcurve" in co:
                    c["pcurve"] = [[p[0] * f if lengths[0] else p[0], p[1] * f if lengths[1] else p[1]]
                                   for p in co["pcurve"]]
                new.append(c)
            loops.append(new)
        faces.append({**face, "loops": loops})
    out["faces"] = faces
    return out
