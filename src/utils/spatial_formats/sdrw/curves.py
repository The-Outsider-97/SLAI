"""Ellipse and spline entities (feature ``sdrw-curves-v1``): validation and exact analytic helpers.

``ellipse``  ``{"center": [x, y], "rx": a, "ry": b, "rotation": θ}``. ``rx`` is the semi-axis along direction ``rotation`` (radians, counter-clockwise,
             default 0), ``ry`` the semi-axis at +90° from it; either may be the larger. Always a full ellipse.
``spline``   ``{"degree": p, "control_points": [[x, y], ...], "knots": [...]}``. A non-rational, **clamped** B-spline: ``len(knots) == len(control_points) + p + 1``,
             the first and last ``p + 1`` knots are equal, knots never decrease, no knot is repeated more than ``p`` times inside the domain, and the domain
             ``[knots[p], knots[n]]`` is not empty. It starts at the first and ends at the last control point. Piecewise cubic Béziers and Hermite chains
             are expressible exactly (cubic with every interior knot repeated three times).

Bounds and Bézier extraction are analytic (bounds of splines above degree 3 are the control polygon's, which contains the curve but can be loose), lengths are
integrated to about 1e-12 relative, and ``flatten_*`` states the maximum deviation it guarantees.
"""
from __future__ import annotations

from math import cos, hypot, pi, sin, sqrt

from ..common import finite, vector
from ..common.errors import GeometryValidationError

ELLIPSE_KEYS = frozenset({"center", "rx", "ry", "rotation"})
SPLINE_KEYS = frozenset({"degree", "control_points", "knots"})
MAX_DEGREE = 9
MAX_SPLINE_POINTS = 100_000
_TAU = 2.0 * pi


# ------------------------------------------------------------------------------------------------------------------ validation
def validate_ellipse(g: dict) -> None:
    vector(g.get("center"), 2)
    rx, ry = finite(g.get("rx"), "rx"), finite(g.get("ry"), "ry")
    if rx <= 0 or ry <= 0:
        raise GeometryValidationError("ellipse semi-axes must be positive")
    if "rotation" in g:
        finite(g["rotation"], "rotation", limit=1e8)


def validate_spline(g: dict) -> None:
    degree = g.get("degree")
    if type(degree) is not int or not 1 <= degree <= MAX_DEGREE:
        raise GeometryValidationError("spline degree must be an integer from 1 to %d" % MAX_DEGREE)
    pts = g.get("control_points")
    if not isinstance(pts, list) or not degree + 1 <= len(pts) <= MAX_SPLINE_POINTS:
        raise GeometryValidationError("a degree-%d spline needs %d..%d control points" % (degree, degree + 1, MAX_SPLINE_POINTS))
    points = [vector(p, 2) for p in pts]
    if all(p == points[0] for p in points):
        raise GeometryValidationError("degenerate spline: all control points coincide")
    knots = g.get("knots")
    n = len(points)
    if not isinstance(knots, list) or len(knots) != n + degree + 1:
        raise GeometryValidationError("spline needs %d knots (control points + degree + 1)" % (n + degree + 1))
    ks = [finite(k, "knot", limit=1e12) for k in knots]
    if any(b < a for a, b in zip(ks, ks[1:])):
        raise GeometryValidationError("spline knots must not decrease")
    if len(set(ks[:degree + 1])) != 1 or len(set(ks[-degree - 1:])) != 1:
        raise GeometryValidationError("spline knots must be clamped: the first and last %d knots are equal" % (degree + 1))
    if not ks[degree] < ks[n]:
        raise GeometryValidationError("spline domain is empty")
    if sum(1 for k in ks if k == ks[0]) > degree + 1 or sum(1 for k in ks if k == ks[-1]) > degree + 1:
        raise GeometryValidationError("a clamped end knot cannot be repeated more than degree + 1 times")
    run = 1
    for a, b in zip(ks[degree + 1:n], ks[degree + 2:n + 1]):
        run = run + 1 if a == b else 1
        if run > degree:
            raise GeometryValidationError("an interior spline knot is repeated more than the degree allows")


def clamped_uniform_knots(n_points: int, degree: int) -> list[float]:
    """The usual knot vector of a clamped spline whose interior knots are evenly spaced."""
    return [0.0] * (degree + 1) + [float(i) for i in range(1, n_points - degree)] + [float(max(1, n_points - degree))] * (degree + 1)


# ------------------------------------------------------------------------------------------------------------------ evaluation
def _find_span(knots, p: int, n: int, u: float) -> int:
    if u >= knots[n]:
        return n - 1
    lo, hi = p, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if u < knots[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def spline_domain(g: dict) -> tuple[float, float]:
    return g["knots"][g["degree"]], g["knots"][len(g["control_points"])]


def spline_point(g: dict, u: float) -> tuple[float, float]:
    """Point of the spline at parameter ``u`` (de Boor)."""
    p, ctrl, knots = g["degree"], g["control_points"], g["knots"]
    n = len(ctrl)
    u = min(max(u, knots[p]), knots[n])
    k = _find_span(knots, p, n, u)
    d = [list(ctrl[j + k - p]) for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            den = knots[j + 1 + k - r] - knots[j + k - p]
            a = 0.0 if den == 0 else (u - knots[j + k - p]) / den
            d[j][0] = (1 - a) * d[j - 1][0] + a * d[j][0]
            d[j][1] = (1 - a) * d[j - 1][1] + a * d[j][1]
    return (d[p][0], d[p][1])


def bezier_segments(g: dict) -> list[list[tuple[float, float]]]:
    """The spline as Bézier segments of its own degree (each ``degree + 1`` control points), exactly, by Boehm knot insertion."""
    p, n = g["degree"], len(g["control_points"])
    ctrl = [tuple(map(float, c)) for c in g["control_points"]]
    knots = [float(k) for k in g["knots"]]
    lo, hi = knots[p], knots[n]
    for u in sorted({k for k in knots if lo < k < hi}):
        while sum(1 for k in knots if k == u) < p:
            k = max(i for i in range(len(knots) - 1) if knots[i] <= u < knots[i + 1])
            new = list(ctrl[: k - p + 1])
            for i in range(k - p + 1, k + 1):
                a = (u - knots[i]) / (knots[i + p] - knots[i])
                new.append(((1 - a) * ctrl[i - 1][0] + a * ctrl[i][0], (1 - a) * ctrl[i - 1][1] + a * ctrl[i][1]))
            new.extend(ctrl[k:])
            ctrl = new
            knots.insert(k + 1, u)
    return [ctrl[j * p: j * p + p + 1] for j in range((len(ctrl) - 1) // p)]


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def elevate_to_cubic(seg) -> tuple | None:
    """A Bézier segment of degree 1..3 as a cubic (exactly); ``None`` above degree 3."""
    seg = list(seg)
    if len(seg) == 4:
        return tuple(seg)
    if len(seg) == 2:
        a, b = seg
        return (a, _lerp(a, b, 1 / 3), _lerp(a, b, 2 / 3), b)
    if len(seg) == 3:
        a, q, b = seg
        return (a, _lerp(a, q, 2 / 3), _lerp(b, q, 2 / 3), b)
    return None


def cubic_segments(g: dict) -> list[tuple] | None:
    """Exact piecewise-cubic Bézier form (``None`` when the degree is above 3)."""
    if g["degree"] > 3:
        return None
    return [elevate_to_cubic(s) for s in bezier_segments(g)]


def _bernstein_point(seg, t):
    pts = list(seg)
    while len(pts) > 1:
        pts = [_lerp(pts[i], pts[i + 1], t) for i in range(len(pts) - 1)]
    return pts[0]


def _bernstein_derivative_norm(seg, t):
    p = len(seg) - 1
    d = [((seg[i + 1][0] - seg[i][0]) * p, (seg[i + 1][1] - seg[i][1]) * p) for i in range(p)]
    q = _bernstein_point(d, t) if d else (0.0, 0.0)
    return hypot(q[0], q[1])


_GL: tuple | None = None


def _gauss_legendre(n: int = 12):
    global _GL
    if _GL is None:
        xs, ws = [], []                      # nodes/weights from Newton iteration on Legendre polynomials (no tables to mistype)
        for i in range(1, n + 1):
            x = cos(pi * (i - 0.25) / (n + 0.5))
            for _ in range(100):
                p0, p1 = 1.0, x
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                dp = n * (x * p1 - p0) / (x * x - 1)
                dx = p1 / dp
                x -= dx
                if abs(dx) < 1e-16:
                    break
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            xs.append(x)
            ws.append(2.0 / ((1 - x * x) * dp * dp))
        _GL = (xs, ws)
    return _GL


def _segment_length(seg, a=0.0, b=1.0, whole=None, depth=0):
    xs, ws = _gauss_legendre()

    def gl(lo, hi):
        half, mid = (hi - lo) / 2, (hi + lo) / 2
        return half * sum(w * _bernstein_derivative_norm(seg, mid + half * x) for x, w in zip(xs, ws))

    whole = gl(a, b) if whole is None else whole
    mid = (a + b) / 2
    left, right = gl(a, mid), gl(mid, b)
    if depth >= 18 or abs(left + right - whole) <= 1e-13 * max(1.0, whole):
        return left + right
    return _segment_length(seg, a, mid, left, depth + 1) + _segment_length(seg, mid, b, right, depth + 1)


def spline_length(g: dict) -> float:
    """Arc length by adaptive Gauss-Legendre integration of each Bézier segment (relative error about 1e-12)."""
    return sum(_segment_length(s) for s in bezier_segments(g))


def spline_bounds(g: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis-aligned bounds. Exact (extrema of the cubic pieces) up to degree 3; above that the control polygon's box, which contains the curve but can be loose."""
    cubic = cubic_segments(g)
    xs, ys = [], []
    if cubic is None:
        for seg in bezier_segments(g):
            xs += [q[0] for q in seg]
            ys += [q[1] for q in seg]
        return (min(xs), min(ys)), (max(xs), max(ys))
    for p0, p1, p2, p3 in cubic:
        for axis, out in ((0, xs), (1, ys)):
            a, b, c, d = p0[axis], p1[axis], p2[axis], p3[axis]
            out += [a, d]
            A, B, C = -a + 3 * b - 3 * c + d, 2 * (a - 2 * b + c), b - a        # B'(t) / 3 = A t^2 + B t + C
            roots = []
            if abs(A) < 1e-14 * max(1.0, abs(B), abs(C)):
                if abs(B) > 1e-300:
                    roots.append(-C / B)
            else:
                disc = B * B - 4 * A * C
                if disc >= 0:
                    s = sqrt(disc)
                    roots += [(-B + s) / (2 * A), (-B - s) / (2 * A)]
            for t in roots:
                if 0 < t < 1:
                    u = 1 - t
                    out.append(u ** 3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t ** 3 * d)
    return (min(xs), min(ys)), (max(xs), max(ys))


# ------------------------------------------------------------------------------------------------------------------ ellipse
def ellipse_point(g: dict, t: float) -> tuple[float, float]:
    cx, cy = g["center"]
    rot = g.get("rotation", 0.0)
    x, y = g["rx"] * cos(t), g["ry"] * sin(t)
    return (cx + x * cos(rot) - y * sin(rot), cy + x * sin(rot) + y * cos(rot))


def ellipse_bounds(g: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    cx, cy = g["center"]
    rot, rx, ry = g.get("rotation", 0.0), g["rx"], g["ry"]
    hx = sqrt((rx * cos(rot)) ** 2 + (ry * sin(rot)) ** 2)
    hy = sqrt((rx * sin(rot)) ** 2 + (ry * cos(rot)) ** 2)
    return (cx - hx, cy - hy), (cx + hx, cy + hy)


def ellipse_length(rx: float, ry: float) -> float:
    """Circumference by the arithmetic-geometric mean (complete elliptic integral of the second kind): accurate to machine precision."""
    a, b = max(rx, ry), min(rx, ry)
    if a == b:
        return _TAU * a
    k2 = 1.0 - (b / a) ** 2
    an, bn, cn = 1.0, sqrt(1.0 - k2), sqrt(k2)
    total, power = 0.5 * cn * cn, 0.5
    for _ in range(40):
        an, bn, cn = (an + bn) / 2, sqrt(an * bn), (an - bn) / 2
        power *= 2
        total += power * cn * cn
        if abs(cn) < 1e-17:
            break
    return 4 * a * (pi / (2 * an)) * (1 - total)


def ellipse_area(rx: float, ry: float) -> float:
    return pi * rx * ry


# ------------------------------------------------------------------------------------------------------------------ flattening
def flatten_ellipse(g: dict, tol: float) -> list[tuple[float, float]]:
    """Closed polygon (first point not repeated) whose chords stay within ``tol`` of the ellipse. The sagitta over a parameter step ``d`` is at most
    ``rmax^3 d^2 / (8 rmin^2)`` (radius of curvature is at most rmax^2/rmin and the chord at most rmax*d)."""
    rmax, rmin = max(g["rx"], g["ry"]), min(g["rx"], g["ry"])
    step = sqrt(8.0 * max(tol, 1e-12) * rmin * rmin / (rmax ** 3))
    n = min(max(int(_TAU / step) + 1, 16), 20000)
    return [ellipse_point(g, _TAU * i / n) for i in range(n)]


def _flatten_cubic(seg, tol, depth=0):
    p0, c1, c2, p3 = seg
    cx, cy = p3[0] - p0[0], p3[1] - p0[1]
    chord = hypot(cx, cy)
    if chord > 1e-12:
        ux, uy = cx / chord, cy / chord
        dev = max(abs(ux * (c[1] - p0[1]) - uy * (c[0] - p0[0])) for c in (c1, c2))
        inside = all(-1e-9 <= ux * (c[0] - p0[0]) + uy * (c[1] - p0[1]) <= chord + 1e-9 for c in (c1, c2))
    else:
        dev = max(hypot(c[0] - p0[0], c[1] - p0[1]) for c in (c1, c2))
        inside = False
    if (0.75 * dev <= tol and inside) or depth >= 18 or (chord <= 1e-12 and dev <= tol):
        return [p0, p3]
    a, b, c = _lerp(p0, c1, .5), _lerp(c1, c2, .5), _lerp(c2, p3, .5)
    ab, bc = _lerp(a, b, .5), _lerp(b, c, .5)
    m = _lerp(ab, bc, .5)
    return _flatten_cubic((p0, a, ab, m), tol, depth + 1)[:-1] + _flatten_cubic((m, bc, c, p3), tol, depth + 1)


def _flatten_bezier(seg, tol, depth=0):
    """Polyline for one Bézier piece of any degree whose distance from the piece is at most ``tol``, *guaranteed* by the convex-hull property: the piece lies in the
    convex hull of its control points, so if every control point is within ``tol`` of the chord (and projects between its ends) the chord is within ``tol``."""
    p0, pn = seg[0], seg[-1]
    cx, cy = pn[0] - p0[0], pn[1] - p0[1]
    chord = hypot(cx, cy)
    if chord > 1e-12:
        ux, uy = cx / chord, cy / chord
        dev = max(abs(ux * (c[1] - p0[1]) - uy * (c[0] - p0[0])) for c in seg[1:-1]) if len(seg) > 2 else 0.0
        inside = all(-1e-12 <= ux * (c[0] - p0[0]) + uy * (c[1] - p0[1]) <= chord + 1e-12 for c in seg[1:-1])
    else:
        dev = max(hypot(c[0] - p0[0], c[1] - p0[1]) for c in seg[1:-1]) if len(seg) > 2 else 0.0
        inside = dev <= tol
    if (dev <= tol and inside) or depth >= 24:
        return [p0, pn]
    left, right, row = [seg[0]], [seg[-1]], list(seg)
    while len(row) > 1:
        row = [_lerp(row[i], row[i + 1], 0.5) for i in range(len(row) - 1)]
        left.append(row[0])
        right.append(row[-1])
    return _flatten_bezier(left, tol, depth + 1)[:-1] + _flatten_bezier(right[::-1], tol, depth + 1)


def flatten_spline(g: dict, tol: float) -> list[tuple[float, float]]:
    """Polyline from the first to the last control point that stays within ``tol`` of the spline. The bound is guaranteed for every degree: up to degree 3 by the
    cubic's distance from its chord (three quarters of its control points' distance), above by the convex-hull property of each Bézier piece."""
    cubic = cubic_segments(g)
    out: list[tuple[float, float]] = []
    if cubic is not None:
        for seg in cubic:
            pts = _flatten_cubic(seg, tol)
            out.extend(pts if not out else pts[1:])
        return out
    for seg in bezier_segments(g):
        pts = _flatten_bezier(seg, tol)
        out.extend(pts if not out else pts[1:])
    return out


def svg_spline_tolerance(g: dict) -> float:
    """The largest distance an SVG display path of this spline may stray from it: a millionth of the size of its control polygon. Exact up to degree 3 (cubic
    Béziers); above, SVG has no primitive for the curve and a polyline within this bound is drawn."""
    (x0, y0), (x1, y1) = spline_bounds(g)
    return max(1e-9, 1e-6 * hypot(x1 - x0, y1 - y0))


def spline_ends(g: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """First and last point of the spline (a clamped spline starts and ends on its end control points)."""
    pts = g["control_points"]
    return (float(pts[0][0]), float(pts[0][1])), (float(pts[-1][0]), float(pts[-1][1]))


def spline_is_closed(g: dict, tol: float | None = None) -> bool:
    """True when the spline's end points coincide, so it bounds a region."""
    (a, b), (c, d) = spline_ends(g)
    scale = max(1.0, *(abs(v) for p in g["control_points"] for v in p))
    return hypot(a - c, b - d) <= (1e-9 * scale if tol is None else tol)


def spline_green(g: dict) -> float:
    """``1/2 * integral(x dy - y dx)`` along the spline, from its start to its end: the signed area swept from the origin. Exact up to rounding, because the
    integrand is a polynomial on each Bézier piece and Gauss-Legendre with 12 nodes integrates polynomials up to degree 23 exactly (``2*degree - 1 <= 17``)."""
    xs, ws = _gauss_legendre()
    total = 0.0
    for seg in bezier_segments(g):
        p = len(seg) - 1
        d = [((seg[i + 1][0] - seg[i][0]) * p, (seg[i + 1][1] - seg[i][1]) * p) for i in range(p)]
        for x, w in zip(xs, ws):
            t = (x + 1) / 2
            px, py = _bernstein_point(seg, t)
            dx, dy = _bernstein_point(d, t)
            total += 0.5 * (w / 2) * (px * dy - py * dx)
    return total
