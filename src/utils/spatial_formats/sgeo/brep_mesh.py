"""Deterministic tessellation, bounds and mass properties of B-reps (no kernel, pure Python).

Pipeline for :func:`tessellate_brep`
------------------------------------
1. **Edges** are sampled once, at the requested angular resolution, so the faces that share an edge use the
   very same mesh vertices: the result is watertight whenever the brep is closed. Straight edges next to a
   curved face are split to roughly the same length as that face's circle segments.
2. Each **face loop** is walked in the surface's ``(u, v)`` plane. Points take their ``(u, v)`` from the
   coedge's ``pcurve`` when it has one and are projected onto the surface otherwise; periodic parameters are
   unwrapped along the loop, and poles (where the angle is undefined) borrow the angle of their neighbours.
   A loop must close in the ``(u, v)`` plane and the outer loop must run counter-clockwise about the face's
   outward normal.
3. The trimmed domain (outer loop minus holes) is triangulated by hole bridging and **ear clipping**. On curved
   surfaces interior points are then seeded on a staggered grid whose spacing matches the boundary
   segments, inserted one by one (walking point location) and every interior edge is **flipped** towards the
   largest minimum 3D angle. A last **refinement** bisects interior edges wherever the chordal error at
   triangle centroids and edge midpoints still exceeds ``bbox diagonal * (1 - cos(pi / radial_segments))``.
   Boundary edges are never split, which keeps neighbouring faces conforming.
4. Triangle winding follows the face's outward normal; triangles that collapse (poles) are dropped.

The mesh approximates the exact surfaces: areas and volumes converge as ``radial_segments`` grows, and
planar-faced solids are exact.
"""
from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from hashlib import sha256
from json import dumps
from math import atan2, ceil, cos, floor, hypot, pi, sqrt, tan

from ..common.errors import GeometryValidationError, ResourceLimitError
from . import surfaces as S
from .brep import validate_brep

__all__ = ["tessellate_brep", "analyze_brep", "brep_bounds"]

_MAX_FACE_POINTS = 4096          # boundary points of one face
_MAX_FACE_TRIANGLES = 60_000
_MAX_SEEDS = 30_000
_EPS = 1e-14
_MIN_EDGE = 1e-6                  # refinement stops below this normalised edge length


def _fail(message, **context):
    raise GeometryValidationError(message, context=context or None)


# ----------------------------------------------------------------------------------------- sampling
def _target_length(surface, n):
    """Segment length a face on ``surface`` wants along its boundary (``None``: flat, any length will do)."""
    kind = surface["type"]
    if kind in ("cylinder", "cone", "sphere"):
        return S.TWO_PI * surface["radius"] / n
    if kind == "torus":
        return S.TWO_PI * surface["minor_radius"] / n
    return None


def _edge_targets(brep, n):
    """Per edge, the shortest segment length requested by an adjacent curved face (``None`` if none)."""
    targets = [None] * len(brep["edges"])
    for face in brep["faces"]:
        h = _target_length(brep["surfaces"][face["surface"]], n)
        if h is None:
            continue
        for loop in face["loops"]:
            for co in loop:
                e = co["edge"]
                targets[e] = h if targets[e] is None else min(targets[e], h)
    return targets


class _Edge:
    __slots__ = ("params", "points", "ids", "t0", "t1")


def _sample_edges(brep, n, positions):
    """Per edge: sample parameters, 3D points and global vertex ids (shared by every face using the edge)."""
    vertices, curves = brep["vertices"], brep["curves"]
    targets = _edge_targets(brep, n)
    vertex_ids = []
    for v in vertices:
        vertex_ids.append(len(positions))
        positions.append(tuple(v))
    samples = []
    for index, edge in enumerate(brep["edges"]):
        e = _Edge()
        if edge["curve"] is None:
            e.params = e.points = e.ids = None
            e.t0, e.t1 = 0.0, 1.0
            samples.append(e)
            continue
        curve = curves[edge["curve"]]
        t0, t1 = edge["range"]
        if curve["type"] == "line" and targets[index] is not None:
            count = max(1, ceil((t1 - t0) / targets[index] - 1e-9))
            params = [t0 + (t1 - t0) * i / count for i in range(count)] + [t1]
        else:
            params = S.curve_sample_params(curve, t0, t1, n)
        points = [S.curve_eval(curve, t) for t in params]
        points[0], points[-1] = tuple(vertices[edge["start"]]), tuple(vertices[edge["end"]])
        ids = [vertex_ids[edge["start"]]]
        for point in points[1:-1]:
            ids.append(len(positions))
            positions.append(point)
        ids.append(vertex_ids[edge["end"]])
        e.params, e.points, e.ids, e.t0, e.t1 = params, points, ids, t0, t1
        samples.append(e)
    return samples, vertex_ids


# ------------------------------------------------------------------------------------- face loops
def _loop_points(brep, face, samples, vertex_ids, n):
    """Boundary points of every loop of ``face``: lists of ``(gid, xyz or None, (u, v))`` (closed, unwrapped)."""
    surface = brep["surfaces"][face["surface"]]
    period_u, period_v = S.periods(surface)
    edges = brep["edges"]
    result = []
    for loop in face["loops"]:
        raw = []                                   # per coedge: [(gid, xyz or None, uv or None), ...]
        for co in loop:
            edge, sample = edges[co["edge"]], samples[co["edge"]]
            forward = co["forward"]
            pcurve = co.get("pcurve")
            if sample.points is None:              # degenerate edge: a point with a straight uv path
                (u0, v0), (u1, v1) = pcurve
                count = max(1, ceil(abs(u1 - u0) / S.TWO_PI * n - 1e-9))
                gid = vertex_ids[edge["start"]]
                entries = [(gid, None, (u0 + (u1 - u0) * i / count, v0 + (v1 - v0) * i / count))
                           for i in range(count + 1)]
            else:
                order = range(len(sample.points)) if forward else range(len(sample.points) - 1, -1, -1)
                span = sample.t1 - sample.t0
                entries = []
                for i in order:
                    uv = None
                    if pcurve is not None:
                        frac = (sample.params[i] - sample.t0) / span
                        if not forward:
                            frac = 1.0 - frac
                        uv = (pcurve[0][0] + (pcurve[1][0] - pcurve[0][0]) * frac,
                              pcurve[0][1] + (pcurve[1][1] - pcurve[0][1]) * frac)
                    entries.append((sample.ids[i], sample.points[i], uv))
            raw.append(entries)
        # each coedge contributes all but its last entry; the loop's final entry closes it
        flat = []
        for entries in raw:
            flat.extend(entries[:-1])
        flat.append(raw[-1][-1])
        points = [[gid, xyz, uv] for gid, xyz, uv in flat]
        previous = None
        for item in points:
            if item[2] is None:
                item[2] = S.project_point(surface, item[1], guess=previous)
            if item[2][0] is not None:
                previous = item[2]
        # angles at singularities (poles) are undefined: borrow them from the neighbours
        us = [it[2][0] for it in points]
        last = None
        for i, u in enumerate(us):
            if u is None:
                us[i] = last
            else:
                last = u
        nxt = None
        for i in range(len(us) - 1, -1, -1):
            if us[i] is None:
                us[i] = nxt
            else:
                nxt = us[i]
        us = [0.0 if u is None else u for u in us]
        vs = [it[2][1] for it in points]
        for i in range(1, len(points)):
            if period_u:
                us[i] += round((us[i - 1] - us[i]) / period_u) * period_u
            if period_v:
                vs[i] += round((vs[i - 1] - vs[i]) / period_v) * period_v
        scale = max(1.0, max(abs(x) for x in us), max(abs(x) for x in vs))
        if abs(us[-1] - us[0]) > 1e-6 * scale or abs(vs[-1] - vs[0]) > 1e-6 * scale:
            _fail("a loop of a face on a %s surface does not close in the surface's parameter plane "
                  "(include the seam edge and degenerate pole edges)" % surface["type"])
        result.append([(points[i][0], points[i][1], (us[i], vs[i])) for i in range(len(points) - 1)])
    return result


# --------------------------------------------------------------------------------- triangulation
def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _area(xy, loop):
    total = 0.0
    for i in range(len(loop)):
        x1, y1 = xy[loop[i]]
        x2, y2 = xy[loop[(i + 1) % len(loop)]]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _bridge_holes(xy, outer, holes):
    """Merge CW ``holes`` into the CCW ``outer`` loop by zero-width bridges (Eberly)."""
    poly = list(outer)
    for hole in sorted(holes, key=lambda h: -max(xy[i][0] for i in h)):
        mi = max(range(len(hole)), key=lambda k: (xy[hole[k]][0], -xy[hole[k]][1]))
        m = hole[mi]
        mx, my = xy[m]
        best = None
        count = len(poly)
        for k in range(count):
            a, b = xy[poly[k]], xy[poly[(k + 1) % count]]
            if (a[1] <= my <= b[1] or b[1] <= my <= a[1]) and a[1] != b[1]:
                x = a[0] + (my - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if x >= mx - _EPS and (best is None or x < best[0]):
                    best = (x, k)
        if best is None:
            _fail("a hole lies outside its face's outer loop")
        x, k = best
        a, b = xy[poly[k]], xy[poly[(k + 1) % count]]
        p = k if a[0] > b[0] else (k + 1) % count
        px, py = xy[poly[p]]
        chosen, chosen_key = p, None
        for j in range(count):
            r = poly[j]
            if j == p or r == poly[p]:
                continue
            rx, ry = xy[r]
            if _cross((mx, my), (x, my), (rx, ry)) >= -_EPS and _cross((x, my), (px, py), (rx, ry)) >= -_EPS \
                    and _cross((px, py), (mx, my), (rx, ry)) >= -_EPS:
                before, after = xy[poly[j - 1]], xy[poly[(j + 1) % count]]
                if _cross(before, (rx, ry), after) <= _EPS:     # reflex or flat: a possible blocker
                    key = (abs(ry - my) / max(abs(rx - mx), 1e-300), (rx - mx) ** 2 + (ry - my) ** 2)
                    if chosen_key is None or key < chosen_key:
                        chosen, chosen_key = j, key
        ring = hole[mi:] + hole[:mi] + [hole[mi]]
        poly = poly[:chosen + 1] + ring + [poly[chosen]] + poly[chosen + 1:]
    return poly


def _inside(a, b, c, q):
    """``q`` inside or on the boundary of counter-clockwise triangle ``abc`` (within ``_EPS``)."""
    return _cross(a, b, q) >= -_EPS and _cross(b, c, q) >= -_EPS and _cross(c, a, q) >= -_EPS


def _tri_quality(a, b, c, sx, sy):
    """Smallest angle (radians) of triangle ``abc`` after scaling the plane by ``(sx, sy)``."""
    pts = ((a[0] * sx, a[1] * sy), (b[0] * sx, b[1] * sy), (c[0] * sx, c[1] * sy))
    best = pi
    for i in range(3):
        p, q, r = pts[i], pts[(i + 1) % 3], pts[(i + 2) % 3]
        ux, uy, vx, vy = q[0] - p[0], q[1] - p[1], r[0] - p[0], r[1] - p[1]
        best = min(best, atan2(abs(ux * vy - uy * vx), ux * vx + uy * vy))
    return best


def _ear_clip(xy, poly, scale=(1.0, 1.0)):
    """Triangulate the simple (possibly bridged) polygon ``poly`` by clipping the best ear first.

    "Best" is the ear whose smallest angle, measured with the plane scaled by ``scale``, is largest; this keeps
    long runs of collinear boundary points from degenerating into fans.
    """
    count = len(poly)
    sx, sy = scale
    prev = [(i - 1) % count for i in range(count)]
    nxt = [(i + 1) % count for i in range(count)]
    stamp = [0] * count
    dead = [False] * count
    alive = count
    triangles = []

    def corner(i):
        return _cross(xy[poly[prev[i]]], xy[poly[i]], xy[poly[nxt[i]]])

    blockers = {i for i in range(count) if corner(i) <= _EPS}
    heap = []

    def push(i):
        if corner(i) > _EPS:
            heappush(heap, (-_tri_quality(xy[poly[prev[i]]], xy[poly[i]], xy[poly[nxt[i]]], sx, sy), i, stamp[i]))

    def refill():
        del heap[:]
        for i in range(count):
            if not dead[i]:
                push(i)

    def unlink(i):
        a, c = prev[i], nxt[i]
        nxt[a], prev[c] = c, a
        dead[i] = True
        blockers.discard(i)
        for j in (a, c):
            stamp[j] += 1
            if corner(j) <= _EPS:
                blockers.add(j)
            else:
                blockers.discard(j)
            push(j)

    refill()
    refilled = False
    while alive > 3:
        if not heap:
            if not refilled:
                refill()
                refilled = True
                if heap:
                    continue
            # no ear at all: drop a flat vertex if there is one, else the loops are not a simple polygon
            flat = next((j for j in sorted(blockers) if not dead[j] and abs(corner(j)) <= _EPS), None)
            if flat is None:
                _fail("a face's boundary loops cross or touch each other and cannot be triangulated")
            alive -= 1
            unlink(flat)
            refilled = False
            continue
        _, i, st = heappop(heap)
        if dead[i] or st != stamp[i]:
            continue
        a, c = prev[i], nxt[i]
        pa, pi_, pc = xy[poly[a]], xy[poly[i]], xy[poly[c]]
        ok = True
        for j in blockers:
            if j == a or j == i or j == c:
                continue
            q = xy[poly[j]]
            if q == pa or q == pi_ or q == pc:
                continue
            if _inside(pa, pi_, pc, q):
                ok = False
                break
        if not ok:
            continue
        triangles.append((poly[a], poly[i], poly[c]))
        alive -= 1
        unlink(i)
        refilled = False
    if alive == 3:
        i = next(k for k in range(count) if not dead[k])
        triangles.append((poly[prev[i]], poly[i], poly[nxt[i]]))
    return [t for t in triangles if _cross(xy[t[0]], xy[t[1]], xy[t[2]]) > _EPS * 1e-3]


# ------------------------------------------------------------------ face mesh: flips, seeds, refinement
def _metric(surface, centre):
    """``v -> (|dS/du|, |dS/dv|)``: the local scale of the parameter plane, analytic where possible."""
    kind = surface["type"]
    if kind == "plane":
        return lambda v: (1.0, 1.0)
    if kind == "cylinder":
        r = surface["radius"]
        return lambda v: (r, 1.0)
    if kind == "cone":
        r, t = surface["radius"], tan(surface["half_angle"])
        sv = 1.0 / cos(surface["half_angle"])
        floor_ = 1e-3 * r
        return lambda v: (max(abs(r + v * t), floor_), sv)
    if kind == "sphere":
        r = surface["radius"]
        return lambda v: (max(r * cos(v), 1e-3 * r), r)
    if kind == "torus":
        big, small = surface["major_radius"], surface["minor_radius"]
        return lambda v: (big + small * cos(v), small)
    fixed = (max(centre[0], 1e-9), max(centre[1], 1e-9))
    return lambda v: fixed


def _uv_min_angle(a, b, c, xs, sv):
    """Smallest angle of the parameter-plane triangle after scaling by ``(xs, sv)`` (the surface metric)."""
    best = pi
    pts = ((a[0] * xs, a[1] * sv), (b[0] * xs, b[1] * sv), (c[0] * xs, c[1] * sv))
    for i in range(3):
        p, q, r = pts[i], pts[(i + 1) % 3], pts[(i + 2) % 3]
        ux, uy, vx, vy = q[0] - p[0], q[1] - p[1], r[0] - p[0], r[1] - p[1]
        best = min(best, atan2(abs(ux * vy - uy * vx), ux * vx + uy * vy))
    return best


class _FaceMesh:
    """Triangles of one face in the ``(u, v)`` plane (counter-clockwise), with the 3D point of every vertex."""

    def __init__(self, xy, uv, p3, gid, metric):
        self.xy, self.uv, self.p3, self.gid = xy, uv, p3, gid
        self.metric = metric
        self.vtri = {}
        self.grid = None
        self.tris = []
        self.edges = {}
        self.fixed = set()
        self.hint = 0

    @staticmethod
    def key(a, b):
        return (a, b) if a < b else (b, a)

    def add(self, a, b, c):
        t = len(self.tris)
        self.tris.append((a, b, c))
        for x, y in ((a, b), (b, c), (c, a)):
            self.edges.setdefault(self.key(x, y), []).append(t)
        self.vtri[a] = self.vtri[b] = self.vtri[c] = t
        self.hint = t
        return t

    def remove(self, t):
        a, b, c = self.tris[t]
        for x, y in ((a, b), (b, c), (c, a)):
            self.edges[self.key(x, y)].remove(t)
        self.tris[t] = None

    def new_vertex(self, uv, xy, p3):
        self.uv.append(uv)
        self.xy.append(xy)
        self.p3.append(p3)
        self.gid.append(None)
        v = len(self.uv) - 1
        if self.grid is not None:
            self._grid_add(v)
        return v

    def set_grid(self, nx, ny):
        """Bucket the vertices on an ``nx`` x ``ny`` grid so point location can start near its target."""
        self.grid = (nx, ny, {})
        for v in range(len(self.xy)):
            self._grid_add(v)

    def _cell(self, x, y):
        nx, ny, _ = self.grid
        return min(nx - 1, max(0, int(x * nx))), min(ny - 1, max(0, int(y * ny)))

    def _grid_add(self, v):
        self.grid[2].setdefault(self._cell(*self.xy[v]), []).append(v)

    def _start(self, x, y):
        """A live triangle at the vertex nearest ``(x, y)``, or the last touched triangle."""
        if self.grid is None:
            return self.hint
        nx, ny, cells = self.grid
        cx, cy = self._cell(x, y)
        for r in range(max(nx, ny)):
            best, best_d = None, None
            for i in range(cx - r, cx + r + 1):
                for j in range(cy - r, cy + r + 1):
                    if max(abs(i - cx), abs(j - cy)) != r:
                        continue
                    for v in cells.get((i, j), ()):
                        d = ((self.xy[v][0] - x) * nx) ** 2 + ((self.xy[v][1] - y) * ny) ** 2
                        if best_d is None or d < best_d:
                            best, best_d = v, d
            if best is not None:
                t = self.vtri.get(best)
                if t is not None and self.tris[t] is not None:
                    return t
                break
        return self.hint

    def live(self):
        return [t for t in self.tris if t is not None]

    def flip(self, key):
        """Flip the interior edge ``key`` if that raises the smaller of the two triangles' minimum 3D angle.
        Returns the four surrounding edge keys when it flipped, else ``None``."""
        ts = self.edges.get(key)
        if not ts or len(ts) != 2 or key in self.fixed:
            return None
        t1, t2 = ts
        p, q = key
        tri = self.tris[t1]
        if not any((tri[i], tri[(i + 1) % 3]) == (p, q) for i in range(3)):
            t1, t2 = t2, t1                                  # t1 now runs p -> q
        c = next(v for v in self.tris[t1] if v != p and v != q)
        d = next(v for v in self.tris[t2] if v != p and v != q)
        xy = self.xy
        if _cross(xy[p], xy[d], xy[c]) <= _EPS or _cross(xy[d], xy[q], xy[c]) <= _EPS:
            return None                                      # the flipped quad is not strictly convex
        uv = self.uv
        xs, sv = self.metric((uv[p][1] + uv[q][1] + uv[c][1] + uv[d][1]) / 4.0)
        old = min(_uv_min_angle(uv[p], uv[q], uv[c], xs, sv), _uv_min_angle(uv[q], uv[p], uv[d], xs, sv))
        new = min(_uv_min_angle(uv[p], uv[d], uv[c], xs, sv), _uv_min_angle(uv[d], uv[q], uv[c], xs, sv))
        if new <= old + 1e-12:
            return None
        self.remove(t1)
        self.remove(t2)
        self.add(p, d, c)
        self.add(d, q, c)
        return [self.key(p, d), self.key(d, q), self.key(q, c), self.key(c, p)]

    def legalize(self, keys, budget):
        stack = list(keys)
        while stack and budget > 0:
            budget -= 1
            again = self.flip(stack.pop())
            if again:
                stack.extend(again)
        return budget

    def flip_pass(self, factor=40):
        queue = sorted(k for k, ts in self.edges.items() if len(ts) == 2 and k not in self.fixed)
        self.legalize(queue, factor * max(1, len(queue)) + 100)

    def locate(self, x, y):
        """A triangle containing ``(x, y)``: a visibility walk from the last touched triangle that rotates the
        edge it tries first (so it cannot cycle on a non-Delaunay mesh), then a scan; ``None`` if outside."""
        t = self._start(x, y)
        for count in range(2000):
            if t >= len(self.tris) or self.tris[t] is None:
                break
            tri = self.tris[t]
            step = None
            for k in range(3):
                i = (count + t + k) % 3
                u, v = tri[i], tri[(i + 1) % 3]
                if _cross(self.xy[u], self.xy[v], (x, y)) < -_EPS:
                    nb = [o for o in self.edges[self.key(u, v)] if o != t]
                    step = nb[0] if nb else -1
                    break
            if step is None:
                return t
            if step == -1:
                break
            t = step
        for t, tri in enumerate(self.tris):
            if tri is not None and _inside(self.xy[tri[0]], self.xy[tri[1]], self.xy[tri[2]], (x, y)):
                return t
        return None

    def insert(self, uv, xy, p3):
        """Split the triangle around a point strictly inside the domain (legalise afterwards with ``flip_pass``);
        returns the new vertex index, or ``None`` if the point lies on an existing edge."""
        t = self.locate(xy[0], xy[1])
        if t is None:
            return None
        a, b, c = self.tris[t]
        for u, v in ((a, b), (b, c), (c, a)):
            if abs(_cross(self.xy[u], self.xy[v], xy)) <= 1e-12:
                return None                                  # on an existing edge: skip (never degenerate)
        m = self.new_vertex(uv, xy, p3)
        self.remove(t)
        self.add(a, b, m)
        self.add(b, c, m)
        self.add(c, a, m)
        return m

    def refine(self, surface, tol, max_triangles):
        """Bisect interior edges (and isolated ears at their centroid) until the chordal error is within ``tol``."""
        work = deque(t for t, tri in enumerate(self.tris) if tri is not None)

        def evaluate(uv):
            return S.surface_eval(surface, uv[0], uv[1])

        live = len(self.live())
        while work and live < max_triangles:
            t = work.popleft()
            tri = self.tris[t]
            if tri is None:
                continue
            a, b, c = tri
            interior = []
            samples = [(((self.uv[a][0] + self.uv[b][0] + self.uv[c][0]) / 3, (self.uv[a][1] + self.uv[b][1] + self.uv[c][1]) / 3),
                        tuple((self.p3[a][i] + self.p3[b][i] + self.p3[c][i]) / 3 for i in range(3)))]
            for x, y in ((a, b), (b, c), (c, a)):
                if self.key(x, y) not in self.fixed:
                    samples.append((((self.uv[x][0] + self.uv[y][0]) / 2, (self.uv[x][1] + self.uv[y][1]) / 2),
                                    tuple((self.p3[x][i] + self.p3[y][i]) / 2 for i in range(3))))
                    interior.append((sum((self.p3[x][i] - self.p3[y][i]) ** 2 for i in range(3)), x, y))
            worst = 0.0
            for uv, expected in samples:
                actual = evaluate(uv)
                worst = max(worst, sqrt(sum((actual[i] - expected[i]) ** 2 for i in range(3))))
            if worst <= tol:
                continue
            if interior:
                _, x, y = max(interior)
                if hypot(self.xy[x][0] - self.xy[y][0], self.xy[x][1] - self.xy[y][1]) < _MIN_EDGE:
                    continue
                adjacent = list(self.edges[self.key(x, y)])
                uv = ((self.uv[x][0] + self.uv[y][0]) / 2, (self.uv[x][1] + self.uv[y][1]) / 2)
                m = self.new_vertex(uv, ((self.xy[x][0] + self.xy[y][0]) / 2, (self.xy[x][1] + self.xy[y][1]) / 2),
                                    evaluate(uv))
                created = []
                for tt in adjacent:
                    ta = self.tris[tt]
                    i = next(k for k in range(3) if {ta[k], ta[(k + 1) % 3]} == {x, y})
                    p, q, r = ta[i], ta[(i + 1) % 3], ta[(i + 2) % 3]
                    self.remove(tt)
                    created.append(self.add(p, m, r))
                    created.append(self.add(m, q, r))
                live += len(adjacent)
                work.extend(created)
            else:
                if max(abs(self.xy[a][0] - self.xy[b][0]), abs(self.xy[a][1] - self.xy[b][1])) < _MIN_EDGE:
                    continue
                uv = samples[0][0]
                m = self.new_vertex(uv, tuple(sum(self.xy[v][i] for v in tri) / 3 for i in range(2)), evaluate(uv))
                self.remove(t)
                work.extend((self.add(a, b, m), self.add(b, c, m), self.add(c, a, m)))
                live += 2


def _speeds(surface, u, v, du, dv):
    """``(|dS/du|, |dS/dv|)`` by central differences."""
    hu, hv = max(du * 1e-6, 1e-12), max(dv * 1e-6, 1e-12)
    a, b = S.surface_eval(surface, u - hu, v), S.surface_eval(surface, u + hu, v)
    c, d = S.surface_eval(surface, u, v - hv), S.surface_eval(surface, u, v + hv)
    return (sqrt(sum((b[i] - a[i]) ** 2 for i in range(3))) / (2 * hu),
            sqrt(sum((d[i] - c[i]) ** 2 for i in range(3))) / (2 * hv))


def _seed_points(surface, mesh, groups, h, bbox, n):
    """Interior seeds ``[(u, v, x, y), ...]``: a staggered grid with 3D spacing ``h``, kept only strictly inside
    the trimmed domain and clear of its boundary; ordered along serpentine rows for cheap point location."""
    (u_min, v_min), (eu, ev), _ = bbox
    period_u = S.periods(surface)[0]
    uc, vc = u_min + eu / 2, v_min + ev / 2
    su_c, sv_c = _speeds(surface, uc, vc, eu, ev)
    if sv_c <= 0 or su_c <= 0:
        su_c = max(su_c, sv_c, 1e-12)
        sv_c = max(sv_c, su_c)
    rows = min(400, max(1, ceil(ev * sv_c / h)))
    # scaled space: one unit is about h of 3D length
    scale_x = (su_c * eu) / h            # normalised x (0..1) -> scaled units
    scale_y = (sv_c * ev) / h
    segments = []
    for g in groups:
        for i in range(len(g)):
            a, b = mesh.xy[g[i]], mesh.xy[g[(i + 1) % len(g)]]
            segments.append((a[0] * scale_x, a[1] * scale_y, b[0] * scale_x, b[1] * scale_y, a, b))
    bins = {}
    for k, (ax, ay, bx, by, _, _) in enumerate(segments):
        for ci in range(floor(min(ax, bx)), floor(max(ax, bx)) + 1):
            for cj in range(floor(min(ay, by)), floor(max(ay, by)) + 1):
                bins.setdefault((ci, cj), []).append(k)
    seeds = []
    for j in range(rows):
        v = v_min + ev * (j + 0.5) / rows
        y = (v - v_min) / ev
        if period_u:                                     # one column per angular step, whatever the radius
            cols = max(1, ceil(eu / S.TWO_PI * n - 1e-9))
        else:
            su, _ = _speeds(surface, uc, v, eu, ev)
            if su <= 1e-12:
                continue
            cols = max(1, ceil(eu * su / h))
        crossings = []
        for _, _, _, _, a, b in segments:
            if (a[1] <= y < b[1]) or (b[1] <= y < a[1]):
                crossings.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        crossings.sort()
        offset = 0.25 if j % 2 == 0 else 0.75
        row = []
        for i in range(cols):
            x = (i + offset) / cols
            inside = sum(1 for cx in crossings if cx < x) % 2 == 1
            if not inside:
                continue
            sx, sy = x * scale_x, y * scale_y
            ci, cj = floor(sx), floor(sy)
            clear = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for k in bins.get((ci + di, cj + dj), ()):
                        ax, ay, bx, by, _, _ = segments[k]
                        dx, dy = bx - ax, by - ay
                        length2 = dx * dx + dy * dy
                        t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((sx - ax) * dx + (sy - ay) * dy) / length2))
                        if (sx - ax - t * dx) ** 2 + (sy - ay - t * dy) ** 2 < 0.16:
                            clear = False
                            break
                    if not clear:
                        break
                if not clear:
                    break
            if clear:
                row.append((u_min + eu * x, v, x, y))
        seeds.extend(row if j % 2 == 0 else row[::-1])
        if len(seeds) > _MAX_SEEDS:
            raise ResourceLimitError("a face needs more than %d interior points; lower the resolution" % _MAX_SEEDS)
    return seeds


def _tessellate_face(brep, fi, face, samples, vertex_ids, n, tol, positions, uv_mode, out_vertices, out_uvs,
                     out_faces):
    surface = brep["surfaces"][face["surface"]]
    loops = _loop_points(brep, face, samples, vertex_ids, n)
    if sum(len(lp) for lp in loops) > _MAX_FACE_POINTS:
        raise ResourceLimitError("face %d has more than %d boundary points; lower the resolution"
                                 % (fi, _MAX_FACE_POINTS))
    flat = [pt for lp in loops for pt in lp]
    us = [pt[2][0] for pt in flat]
    vs = [pt[2][1] for pt in flat]
    u0, v0 = min(us), min(vs)
    eu, ev = max(max(us) - u0, 1e-12), max(max(vs) - v0, 1e-12)
    uv = [pt[2] for pt in flat]
    xy = [((u - u0) / eu, (v - v0) / ev) for u, v in uv]
    p3 = [positions[pt[0]] if pt[1] is None else pt[1] for pt in flat]
    gid = [pt[0] for pt in flat]
    index, groups = 0, []
    for lp in loops:
        groups.append(list(range(index, index + len(lp))))
        index += len(lp)
    reversed_face = bool(face.get("reversed", False))
    areas = [_area(xy, g) for g in groups]
    outer_sign = -1.0 if reversed_face else 1.0
    if abs(areas[0]) <= _EPS:
        _fail("the outer loop of face %d encloses no area" % fi)
    if areas[0] * outer_sign < 0:
        _fail("the outer loop of face %d does not run counter-clockwise about the face's outward normal "
              "(check the loop direction or the 'reversed' flag)" % fi)
    for k in range(1, len(groups)):
        if areas[k] * outer_sign > 0:
            _fail("hole %d of face %d does not run clockwise about the face's outward normal" % (k, fi))
    if reversed_face:                                   # triangulate with a counter-clockwise outer loop
        groups = [list(reversed(g)) for g in groups]
    outer, holes = groups[0], groups[1:]
    poly = _bridge_holes(xy, outer, holes) if holes else outer
    centre = _speeds(surface, u0 + eu / 2, v0 + ev / 2, eu, ev) if surface["type"] == "nurbs" else (1.0, 1.0)
    metric = _metric(surface, centre)
    xs_c, sv_c = metric(v0 + ev / 2)
    scale = (xs_c * eu, sv_c * ev)
    mesh = _FaceMesh(list(xy), list(uv), list(p3), list(gid), metric)
    for lp in groups:
        for i in range(len(lp)):
            mesh.fixed.add(mesh.key(lp[i], lp[(i + 1) % len(lp)]))
    for t in _ear_clip(xy, poly, scale):
        mesh.add(*t)
    mesh.flip_pass()
    if surface["type"] != "plane":
        h = _target_length(surface, n)
        if h is None:                                   # NURBS: use the mean boundary segment
            lengths = []
            for g in groups:
                for i in range(len(g)):
                    a, b = mesh.p3[g[i]], mesh.p3[g[(i + 1) % len(g)]]
                    d = sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))
                    if d > 0:
                        lengths.append(d)
            h = sum(lengths) / len(lengths) if lengths else None
        if h:
            seeds = _seed_points(surface, mesh, groups, h, ((u0, v0), (eu, ev), (0.0, 0.0)), n)
            mesh.set_grid(max(1, min(256, ceil(scale[0] / (2 * h)))), max(1, min(256, ceil(scale[1] / (2 * h)))))
            for u, v, x, y in seeds:
                point = S.surface_eval(surface, u, v)
                mesh.insert((u, v), (x, y), point)
            mesh.flip_pass()
        mesh.refine(surface, tol, _MAX_FACE_TRIANGLES)
        mesh.flip_pass()
    base = {}
    for t in mesh.tris:
        if t is None:
            continue
        ids = []
        for local in t:
            if uv_mode or mesh.gid[local] is None:
                if local not in base:
                    base[local] = len(out_vertices)
                    out_vertices.append(mesh.p3[local])
                    if uv_mode:
                        out_uvs.append(((mesh.uv[local][0] - u0) / eu, (mesh.uv[local][1] - v0) / ev))
                ids.append(base[local])
            else:
                ids.append(mesh.gid[local])
        a, b, c = ids
        if reversed_face:
            b, c = c, b
        out_faces.append((a, b, c))


# --------------------------------------------------------------------------------------- public API
def _check_resolution(radial_segments):
    if type(radial_segments) is not int or not 8 <= radial_segments <= 128:
        _fail("radial_segments must be an integer in [8, 128]")


def _mesh_from(brep, radial_segments, uv_mode):
    info = validate_brep(brep)
    n = radial_segments
    positions = []
    samples, vertex_ids = _sample_edges(brep, n, positions)
    lo = [min(p[i] for p in positions) for i in range(3)]
    hi = [max(p[i] for p in positions) for i in range(3)]
    diag = sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    tol = max(diag * (1.0 - cos(pi / n)), 1e-12)
    out_uvs, out_faces = [], []
    if uv_mode:
        out_vertices, shared = [], positions
    else:
        out_vertices = list(positions)
        shared = out_vertices
    for fi, face in enumerate(brep["faces"]):
        _tessellate_face(brep, fi, face, samples, vertex_ids, n, tol, shared, uv_mode, out_vertices, out_uvs,
                         out_faces)
    return info, out_vertices, out_uvs, out_faces, tol


def _compact(vertices, uvs, faces):
    """Drop zero-area triangles and unreferenced vertices; return a mesh dict."""
    from .geometry import MAX_TRIANGLES, MAX_VERTICES
    kept = []
    for a, b, c in faces:
        if a == b or b == c or a == c:
            continue
        p, q, r = vertices[a], vertices[b], vertices[c]
        ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        vx, vy, vz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
        cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        if cx * cx + cy * cy + cz * cz <= 1e-30:
            continue
        kept.append((a, b, c))
    if not kept:
        _fail("the brep produced no triangles")
    used = sorted({i for f in kept for i in f})
    remap = {old: new for new, old in enumerate(used)}
    mesh = {"kind": "mesh", "vertices": [list(vertices[i]) for i in used],
            "faces": [[remap[i] for i in f] for f in kept]}
    if uvs:
        mesh["uvs"] = [list(uvs[i]) for i in used]
    if len(mesh["vertices"]) > MAX_VERTICES or len(mesh["faces"]) > MAX_TRIANGLES:
        raise ResourceLimitError("tessellation resource budget exceeded")
    return mesh


def tessellate_brep(brep, *, radial_segments: int = 32, uvs: bool = False) -> dict:
    """Triangle mesh of a brep. ``uvs=True`` adds per-face normalised texture coordinates (vertices are then
    not shared between faces, though neighbouring faces still meet exactly)."""
    _check_resolution(radial_segments)
    _, vertices, out_uvs, faces, _ = _mesh_from(brep, radial_segments, uvs)
    return _compact(vertices, out_uvs if uvs else None, faces)


def analyze_brep(brep, *, radial_segments: int = 64) -> dict:
    """Topology and mass properties from a tessellation at ``radial_segments``.

    ``volume`` and ``area`` are *approximate* for curved shapes. ``volume`` is the signed enclosed volume of
    the outward oriented mesh and is ``None`` for an open brep; a negative value means the faces point inward.
    """
    from .analysis import analyze_mesh
    _check_resolution(radial_segments)
    info, vertices, _, faces, tol = _mesh_from(brep, radial_segments, False)
    mesh = _compact(vertices, None, faces)
    diag = analyze_mesh(mesh)
    volume = diag.signed_volume if info["closed"] and diag.is_closed_oriented_surface else None
    return {"vertices": info["vertices"], "edges": info["edges"], "faces": info["faces"], "closed": info["closed"],
            "euler_characteristic": info["euler"], "mesh_triangles": len(mesh["faces"]),
            "watertight": diag.is_closed_oriented_surface, "area": diag.surface_area, "volume": volume,
            "outward": None if volume is None else volume > 0, "deviation_tolerance": tol}


_BOUNDS_CACHE: dict = {}


def brep_bounds(brep, *, radial_segments: int = 24) -> tuple:
    """Broad-phase bounds: those of a coarse tessellation, grown by the chordal tolerance (exact when flat-faced)."""
    key = sha256(dumps(brep, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cached = _BOUNDS_CACHE.get(key)
    if cached is not None:
        return cached
    _, vertices, _, faces, tol = _mesh_from(brep, radial_segments, False)
    used = [vertices[i] for f in faces for i in f]
    if not used:
        _fail("the brep produced no triangles")
    if all(s["type"] == "plane" for s in brep["surfaces"]) and all(c["type"] == "line" for c in brep["curves"]):
        tol = 0.0                                          # flat-faced: the tessellation is the exact shape
    lo = tuple(min(p[i] for p in used) - tol for i in range(3))
    hi = tuple(max(p[i] for p in used) + tol for i in range(3))
    if len(_BOUNDS_CACHE) >= 64:
        _BOUNDS_CACHE.pop(next(iter(_BOUNDS_CACHE)))
    _BOUNDS_CACHE[key] = (lo, hi)
    return lo, hi
