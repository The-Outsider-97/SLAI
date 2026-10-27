"""Non-uniform rational B-splines (NURBS): validation, evaluation and projection. Pure Python, no kernel.

Definitions are plain JSON records (``kind`` is added by the caller when they stand alone):

``nurbs`` curve::

    {"degree": p, "knots": [...], "control_points": [[x, y, z], ...], "weights": [...]?}

``nurbs`` surface (control net rows run along ``u``, columns along ``v``)::

    {"degree_u": p, "degree_v": q, "knots_u": [...], "knots_v": [...],
     "control_points": [[[x, y, z], ...v] ...u], "weights": [[...v] ...u]?}

Rules (checked by :func:`validate_curve` / :func:`validate_surface`): degrees 1..:data:`MAX_DEGREE`; knot
vectors non-decreasing and *clamped* (the first and last ``degree + 1`` knots are equal, so the curve starts
and ends at its end control points); interior knot multiplicity at most ``degree``; ``len(knots) =
len(control_points) + degree + 1``; weights positive and finite. Weights omitted means a plain B-spline.
Evaluation uses the Cox-de Boor recurrence (Piegl and Tiller, *The NURBS Book*, A2.1/A2.2/A4.1/A4.3).
"""
from __future__ import annotations

from ..common.errors import GeometryValidationError, ResourceLimitError
from ..common.helpers import finite, vector

__all__ = ["MAX_DEGREE", "MAX_CURVE_POINTS", "MAX_SURFACE_POINTS", "validate_curve", "validate_surface",
           "curve_domain", "surface_domain", "curve_point", "surface_point", "curve_params", "curve_bounds",
           "surface_bounds", "project_to_surface", "scale_curve", "scale_surface", "clamped_uniform_knots",
           "distinct_knots", "curve_point_count", "surface_point_count"]

MAX_DEGREE = 12
MAX_CURVE_POINTS = 65_536
MAX_SURFACE_POINTS = 65_536
MAX_SURFACE_SIDE = 1024
_KNOT_TOL = 1e-12


def _fail(message):
    raise GeometryValidationError(message)


def _knot_vector(knots, degree, count, label):
    if not isinstance(knots, (list, tuple)) or len(knots) != count + degree + 1:
        _fail("%s must have control points + degree + 1 = %d values" % (label, count + degree + 1))
    values = [finite(k, label, limit=1e9) for k in knots]
    for a, b in zip(values, values[1:]):
        if b < a:
            _fail(label + " must be non-decreasing")
    if values[0] >= values[-1]:
        _fail(label + " must span a positive parameter range")
    if len(set(values[:degree + 1])) != 1 or len(set(values[-(degree + 1):])) != 1:
        _fail(label + " must be clamped (first and last degree + 1 knots equal)")
    run, previous = 0, None
    for value in values[degree + 1:-(degree + 1)]:
        run = run + 1 if value == previous else 1
        previous = value
        if run > degree:
            _fail("interior knot multiplicity in %s must not exceed the degree" % label)
    return values


def _degree(value, label):
    if type(value) is not int or not 1 <= value <= MAX_DEGREE:
        _fail("%s must be an integer in [1, %d]" % (label, MAX_DEGREE))
    return value


def _weights(weights, shape, label):
    """Validate weights against ``shape`` (an int for a curve, ``(nu, nv)`` for a net)."""
    if weights is None:
        return
    if isinstance(shape, int):
        if not isinstance(weights, (list, tuple)) or len(weights) != shape:
            _fail("%s must match the control points" % label)
        rows = [weights]
    else:
        if not isinstance(weights, (list, tuple)) or len(weights) != shape[0] or any(
                not isinstance(r, (list, tuple)) or len(r) != shape[1] for r in weights):
            _fail("%s must match the control net" % label)
        rows = weights
    for row in rows:
        for w in row:
            if not 1e-6 <= finite(w, label, limit=1e6):
                _fail(label + " must be positive (at least 1e-6)")


def curve_point_count(curve) -> int:
    return len(curve["control_points"])


def surface_point_count(surface) -> int:
    return len(surface["control_points"]) * len(surface["control_points"][0])


def validate_curve(curve) -> dict:
    allowed = {"degree", "knots", "control_points", "weights"}
    if not isinstance(curve, dict) or set(curve) - allowed - {"kind", "type"}:
        _fail("invalid NURBS curve fields")
    p = _degree(curve.get("degree"), "curve degree")
    points = curve.get("control_points")
    if not isinstance(points, (list, tuple)) or len(points) < p + 1:
        _fail("a NURBS curve needs at least degree + 1 control points")
    if len(points) > MAX_CURVE_POINTS:
        raise ResourceLimitError("a NURBS curve may have at most %d control points" % MAX_CURVE_POINTS)
    for point in points:
        vector(point, 3, "control point")
    _knot_vector(curve.get("knots"), p, len(points), "knots")
    _weights(curve.get("weights"), len(points), "weights")
    return curve


def validate_surface(surface) -> dict:
    allowed = {"degree_u", "degree_v", "knots_u", "knots_v", "control_points", "weights"}
    if not isinstance(surface, dict) or set(surface) - allowed - {"kind", "type"}:
        _fail("invalid NURBS surface fields")
    p = _degree(surface.get("degree_u"), "degree_u")
    q = _degree(surface.get("degree_v"), "degree_v")
    net = surface.get("control_points")
    if not isinstance(net, (list, tuple)) or not p + 1 <= len(net) <= MAX_SURFACE_SIDE:
        _fail("a NURBS surface needs degree_u + 1 .. %d control point rows" % MAX_SURFACE_SIDE)
    width = len(net[0]) if isinstance(net[0], (list, tuple)) else -1
    if not q + 1 <= width <= MAX_SURFACE_SIDE or any(not isinstance(r, (list, tuple)) or len(r) != width for r in net):
        _fail("every control point row needs the same number of points (degree_v + 1 .. %d)" % MAX_SURFACE_SIDE)
    if len(net) * width > MAX_SURFACE_POINTS:
        raise ResourceLimitError("a NURBS surface may have at most %d control points" % MAX_SURFACE_POINTS)
    for row in net:
        for point in row:
            vector(point, 3, "control point")
    _knot_vector(surface.get("knots_u"), p, len(net), "knots_u")
    _knot_vector(surface.get("knots_v"), q, width, "knots_v")
    _weights(surface.get("weights"), (len(net), width), "weights")
    return surface


def clamped_uniform_knots(count: int, degree: int, lo: float = 0.0, hi: float = 1.0) -> list:
    """Clamped knot vector with evenly spaced interior knots for ``count`` control points."""
    spans = count - degree
    if spans < 1:
        _fail("not enough control points for the degree")
    return [lo] * (degree + 1) + [lo + (hi - lo) * i / spans for i in range(1, spans)] + [hi] * (degree + 1)


def distinct_knots(knots, degree) -> list:
    """The distinct knot values spanning the parameter domain, in order."""
    out = []
    for k in knots[degree:len(knots) - degree]:
        if not out or k > out[-1]:
            out.append(k)
    return out


def curve_domain(curve) -> tuple:
    p = curve["degree"]
    return curve["knots"][p], curve["knots"][len(curve["control_points"])]


def surface_domain(surface) -> tuple:
    p, q = surface["degree_u"], surface["degree_v"]
    ku, kv = surface["knots_u"], surface["knots_v"]
    return (ku[p], ku[len(surface["control_points"])]), (kv[q], kv[len(surface["control_points"][0])])


# ---------------------------------------------------------------------------------------- evaluation
def _span(count, degree, u, knots):
    """Knot span index containing ``u`` (A2.1); ``u`` is clamped into the domain."""
    n = count - 1
    if u >= knots[n + 1]:
        return n
    if u <= knots[degree]:
        return degree
    lo, hi = degree, n + 1
    mid = (lo + hi) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def _basis(span, u, degree, knots):
    """The ``degree + 1`` non-vanishing B-spline basis functions at ``u`` (A2.2)."""
    values = [0.0] * (degree + 1)
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)
    values[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            denominator = right[r + 1] + left[j - r]
            temp = values[r] / denominator if denominator else 0.0
            values[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        values[j] = saved
    return values


def curve_point(curve, t) -> tuple:
    p, knots, points = curve["degree"], curve["knots"], curve["control_points"]
    weights = curve.get("weights")
    span = _span(len(points), p, t, knots)
    basis = _basis(span, t, p, knots)
    x = y = z = w = 0.0
    for i in range(p + 1):
        index = span - p + i
        px, py, pz = points[index]
        b = basis[i] * (weights[index] if weights else 1.0)
        x += b * px
        y += b * py
        z += b * pz
        w += b
    return (x / w, y / w, z / w)


def surface_point(surface, u, v) -> tuple:
    p, q = surface["degree_u"], surface["degree_v"]
    net, weights = surface["control_points"], surface.get("weights")
    su = _span(len(net), p, u, surface["knots_u"])
    sv = _span(len(net[0]), q, v, surface["knots_v"])
    bu = _basis(su, u, p, surface["knots_u"])
    bv = _basis(sv, v, q, surface["knots_v"])
    x = y = z = w = 0.0
    for i in range(p + 1):
        row = su - p + i
        points = net[row]
        for j in range(q + 1):
            col = sv - q + j
            b = bu[i] * bv[j] * (weights[row][col] if weights else 1.0)
            px, py, pz = points[col]
            x += b * px
            y += b * py
            z += b * pz
            w += b
    return (x / w, y / w, z / w)


def curve_params(curve, t0, t1, per_span: int) -> list:
    """Parameters from ``t0`` to ``t1`` (inclusive): every knot span in range split into ``per_span`` parts."""
    breaks = [t0] + [k for k in distinct_knots(curve["knots"], curve["degree"]) if t0 < k < t1] + [t1]
    out = [breaks[0]]
    for a, b in zip(breaks, breaks[1:]):
        for i in range(1, per_span + 1):
            out.append(a + (b - a) * i / per_span)
        out[-1] = b
    return out


def curve_bounds(curve) -> tuple:
    """Bounds of the control polygon: contains the curve (convex hull property; weights are positive)."""
    pts = curve["control_points"]
    return (tuple(min(p[i] for p in pts) for i in range(3)), tuple(max(p[i] for p in pts) for i in range(3)))


def surface_bounds(surface) -> tuple:
    pts = [p for row in surface["control_points"] for p in row]
    return (tuple(min(p[i] for p in pts) for i in range(3)), tuple(max(p[i] for p in pts) for i in range(3)))


def project_to_surface(surface, point, guess=None, *, iterations: int = 40) -> tuple:
    """Parameters ``(u, v)`` of the surface point nearest ``point`` (Gauss-Newton with a grid start).

    ``guess`` seeds the search (used when walking along an edge); otherwise a coarse grid over the domain
    picks the start. The result is clamped to the domain. Accuracy is about 1e-9 of the domain size.
    """
    (u0, u1), (v0, v1) = surface_domain(surface)
    du, dv = u1 - u0, v1 - v0
    px, py, pz = point

    def distance2(u, v):
        x, y, z = surface_point(surface, u, v)
        return (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2

    if guess is None:
        steps = 12
        best = None
        for i in range(steps + 1):
            for j in range(steps + 1):
                u, v = u0 + du * i / steps, v0 + dv * j / steps
                d = distance2(u, v)
                if best is None or d < best[0]:
                    best = (d, u, v)
        _, u, v = best
    else:
        u, v = min(max(guess[0], u0), u1), min(max(guess[1], v0), v1)
    h = 1e-6
    for _ in range(iterations):
        f = surface_point(surface, u, v)
        # one-sided differences keep the evaluation inside the domain
        uh = h * du if u + h * du <= u1 else -h * du
        vh = h * dv if v + h * dv <= v1 else -h * dv
        fu = surface_point(surface, u + uh, v)
        fv = surface_point(surface, u, v + vh)
        su = [(fu[k] - f[k]) / uh for k in range(3)]
        sv = [(fv[k] - f[k]) / vh for k in range(3)]
        r = [f[0] - px, f[1] - py, f[2] - pz]
        a = sum(s * s for s in su)
        b = sum(s * t for s, t in zip(su, sv))
        c = sum(t * t for t in sv)
        ru = sum(s * x for s, x in zip(su, r))
        rv = sum(t * x for t, x in zip(sv, r))
        det = a * c - b * b
        if abs(det) < 1e-30:
            break
        step_u = -(c * ru - b * rv) / det
        step_v = -(a * rv - b * ru) / det
        nu = min(max(u + step_u, u0), u1)
        nv = min(max(v + step_v, v0), v1)
        moved = max(abs(nu - u) / du, abs(nv - v) / dv)
        u, v = nu, nv
        if moved < 1e-13:
            break
    return u, v


def scale_curve(curve, factor) -> dict:
    out = dict(curve)
    out["control_points"] = [[c * factor for c in p] for p in curve["control_points"]]
    return out


def scale_surface(surface, factor) -> dict:
    out = dict(surface)
    out["control_points"] = [[[c * factor for c in p] for p in row] for row in surface["control_points"]]
    return out

