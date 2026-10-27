"""Curve and surface carriers of the SGEO B-rep: lines, circles, planes, quadrics, tori and NURBS.

Every carrier is a plain JSON record with a ``type``. Frame based types share one convention: ``origin``,
a unit ``axis`` (the plane normal, the cylinder / cone / torus axis, the sphere pole axis) and a unit
``ref`` perpendicular to it; the frame is ``(x, y, z) = (ref, axis x ref, axis)``, right handed.

Curves (parameter ``t``)::

    line    {"type": "line",   "origin", "direction"}                      P = origin + t * direction (unit)
    circle  {"type": "circle", "origin", "axis", "ref", "radius"}          P = origin + r (cos t x + sin t y)
    nurbs   {"type": "nurbs", "degree", "knots", "control_points", "weights"?}

Surfaces (parameters ``u, v``); the *natural normal* is dP/du x dP/dv and always points out of the shape
for a face whose ``reversed`` flag is false::

    plane     P = origin + u x + v y
    cylinder  P = origin + r (cos u x + sin u y) + v z                       (radius)
    cone      P = origin + (r + v tan a)(cos u x + sin u y) + v z            (radius at v = 0, half_angle a)
    sphere    P = origin + r (cos v (cos u x + sin u y) + sin v z)           (radius; v is latitude)
    torus     P = origin + (R + r cos v)(cos u x + sin u y) + r sin v z      (major_radius R, minor_radius r)
    nurbs     see :mod:`.nurbs`

``u`` is an angle (period 2 pi) on cylinder, cone, sphere and torus; ``v`` is an angle on sphere and torus.
"""
from __future__ import annotations

from math import atan2, ceil, cos, floor, hypot, pi, sin, sqrt, tan

from ..common.errors import GeometryValidationError
from ..common.helpers import finite, vector
from . import nurbs

__all__ = ["CURVE_TYPES", "SURFACE_TYPES", "validate_curve_def", "validate_surface_def", "curve_eval",
           "surface_eval", "project_point", "periods", "curve_sample_params", "curve_bounds_estimate",
           "surface_pcurve_lengths", "scale_curve_def", "scale_surface_def", "frame", "curve_point_budget",
           "surface_point_budget", "TWO_PI"]

TWO_PI = 2.0 * pi
CURVE_TYPES = ("line", "circle", "nurbs")
SURFACE_TYPES = ("plane", "cylinder", "cone", "sphere", "torus", "nurbs")
_FRAME_TOL = 1e-9

_FIELDS = {
    "line": {"type", "origin", "direction"},
    "circle": {"type", "origin", "axis", "ref", "radius"},
    "plane": {"type", "origin", "axis", "ref"},
    "cylinder": {"type", "origin", "axis", "ref", "radius"},
    "cone": {"type", "origin", "axis", "ref", "radius", "half_angle"},
    "sphere": {"type", "origin", "axis", "ref", "radius"},
    "torus": {"type", "origin", "axis", "ref", "major_radius", "minor_radius"},
}


def _fail(message):
    raise GeometryValidationError(message)


def _unit(value, label):
    v = vector(value, 3, label)
    if abs(sqrt(sum(c * c for c in v)) - 1.0) > _FRAME_TOL:
        _fail(label + " must be a unit vector")
    return v


def _positive(value, label):
    if finite(value, label) <= 0:
        _fail(label + " must be positive")
    return float(value)


def _check_frame(record):
    vector(record.get("origin"), 3, "origin")
    axis, ref = _unit(record.get("axis"), "axis"), _unit(record.get("ref"), "ref")
    if abs(sum(a * b for a, b in zip(axis, ref))) > _FRAME_TOL:
        _fail("ref must be perpendicular to axis")


def validate_curve_def(curve) -> dict:
    if not isinstance(curve, dict) or not isinstance(curve.get("type"), str) or curve["type"] not in CURVE_TYPES:
        _fail("invalid or unsupported B-rep curve type")
    kind = curve["type"]
    if kind == "nurbs":
        nurbs.validate_curve({k: v for k, v in curve.items() if k != "type"})
        return curve
    if set(curve) != _FIELDS[kind]:
        _fail("invalid fields for a %s curve" % kind)
    if kind == "line":
        vector(curve["origin"], 3, "origin")
        _unit(curve["direction"], "direction")
    else:
        _check_frame(curve)
        _positive(curve["radius"], "radius")
    return curve


def validate_surface_def(surface) -> dict:
    if not isinstance(surface, dict) or not isinstance(surface.get("type"), str) or \
            surface["type"] not in SURFACE_TYPES:
        _fail("invalid or unsupported B-rep surface type")
    kind = surface["type"]
    if kind == "nurbs":
        nurbs.validate_surface({k: v for k, v in surface.items() if k != "type"})
        return surface
    if set(surface) != _FIELDS[kind]:
        _fail("invalid fields for a %s surface" % kind)
    _check_frame(surface)
    if kind in ("cylinder", "sphere"):
        _positive(surface["radius"], "radius")
    elif kind == "cone":
        _positive(surface["radius"], "radius")
        angle = finite(surface["half_angle"], "half_angle")
        if not 1e-6 <= abs(angle) <= pi / 2 - 1e-6:
            _fail("cone half_angle must satisfy 1e-6 <= |angle| <= pi/2 - 1e-6 radians")
    elif kind == "torus":
        major, minor = _positive(surface["major_radius"], "major_radius"), _positive(surface["minor_radius"], "minor_radius")
        if minor >= major:
            _fail("torus minor_radius must be smaller than major_radius")
    return surface


def curve_point_budget(curve) -> int:
    return nurbs.curve_point_count(curve) if curve["type"] == "nurbs" else 0


def surface_point_budget(surface) -> int:
    return nurbs.surface_point_count(surface) if surface["type"] == "nurbs" else 0


def frame(record) -> tuple:
    """``(origin, x, y, z)`` of a frame based record."""
    z = tuple(record["axis"])
    x = tuple(record["ref"])
    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
    return tuple(record["origin"]), x, y, z


def _combine(o, terms):
    return (o[0] + sum(c * d[0] for c, d in terms),
            o[1] + sum(c * d[1] for c, d in terms),
            o[2] + sum(c * d[2] for c, d in terms))


# --------------------------------------------------------------------------------------- evaluation
def curve_eval(curve, t) -> tuple:
    kind = curve["type"]
    if kind == "line":
        return _combine(curve["origin"], [(t, curve["direction"])])
    if kind == "circle":
        o, x, y, _ = frame(curve)
        r = curve["radius"]
        return _combine(o, [(r * cos(t), x), (r * sin(t), y)])
    return nurbs.curve_point(curve, t)


def surface_eval(surface, u, v) -> tuple:
    kind = surface["type"]
    if kind == "nurbs":
        return nurbs.surface_point(surface, u, v)
    o, x, y, z = frame(surface)
    if kind == "plane":
        return _combine(o, [(u, x), (v, y)])
    cu, su = cos(u), sin(u)
    if kind == "cylinder":
        r = surface["radius"]
        return _combine(o, [(r * cu, x), (r * su, y), (v, z)])
    if kind == "cone":
        rho = surface["radius"] + v * tan(surface["half_angle"])
        return _combine(o, [(rho * cu, x), (rho * su, y), (v, z)])
    if kind == "sphere":
        r, cv = surface["radius"], cos(v)
        return _combine(o, [(r * cv * cu, x), (r * cv * su, y), (r * sin(v), z)])
    big, small = surface["major_radius"], surface["minor_radius"]
    rho = big + small * cos(v)
    return _combine(o, [(rho * cu, x), (rho * su, y), (small * sin(v), z)])


def periods(surface) -> tuple:
    """``(period_u, period_v)``; ``None`` for a non-periodic direction."""
    kind = surface["type"]
    if kind in ("cylinder", "cone", "sphere"):
        return TWO_PI, None
    if kind == "torus":
        return TWO_PI, TWO_PI
    return None, None


def surface_pcurve_lengths(surface) -> tuple:
    """Whether ``u`` / ``v`` are lengths (and so scale with the unit) rather than angles or free parameters."""
    kind = surface["type"]
    if kind == "plane":
        return True, True
    if kind in ("cylinder", "cone"):
        return False, True
    return False, False


def project_point(surface, point, guess=None) -> tuple:
    """``(u, v)`` of the surface point nearest ``point``; ``u`` is ``None`` at a singularity (pole, apex).

    Closed form for the analytic types, Gauss-Newton (seeded by ``guess``) for NURBS. Periodic parameters are
    returned in ``[-pi, pi]``; the caller unwraps them along a loop.
    """
    kind = surface["type"]
    if kind == "nurbs":
        return nurbs.project_to_surface(surface, point, guess)
    o, x, y, z = frame(surface)
    d = (point[0] - o[0], point[1] - o[1], point[2] - o[2])
    px = d[0] * x[0] + d[1] * x[1] + d[2] * x[2]
    py = d[0] * y[0] + d[1] * y[1] + d[2] * y[2]
    pz = d[0] * z[0] + d[1] * z[1] + d[2] * z[2]
    if kind == "plane":
        return px, py
    rho = hypot(px, py)
    scale = surface.get("radius", surface.get("major_radius", 1.0))
    u = atan2(py, px) if rho > 1e-9 * scale else None
    if kind in ("cylinder", "cone"):
        return u, pz
    if kind == "sphere":
        return u, atan2(pz, rho)
    return u, atan2(pz, rho - surface["major_radius"])


# ----------------------------------------------------------------------------------------- sampling
def curve_sample_params(curve, t0, t1, radial_segments: int) -> list:
    """Parameters from ``t0`` to ``t1`` that polyline-approximate the curve at the requested resolution."""
    kind = curve["type"]
    if kind == "line":
        return [t0, t1]
    if kind == "circle":
        n = max(1, ceil((t1 - t0) / TWO_PI * radial_segments - 1e-9))
        return [t0 + (t1 - t0) * i / n for i in range(n)] + [t1]
    per_span = 1 if curve["degree"] == 1 else max(2, -(-radial_segments // 4))
    return nurbs.curve_params(curve, t0, t1, per_span)


def curve_bounds_estimate(curve, t0, t1) -> tuple:
    """Bounds that contain the edge's curve between ``t0`` and ``t1`` (exact for lines and circle arcs)."""
    kind = curve["type"]
    if kind == "nurbs":
        return nurbs.curve_bounds(curve)
    if kind == "line":
        a, b = curve_eval(curve, t0), curve_eval(curve, t1)
        return tuple(min(a[i], b[i]) for i in range(3)), tuple(max(a[i], b[i]) for i in range(3))
    o, x, y, _ = frame(curve)
    r = curve["radius"]
    ts = [t0, t1]
    for i in range(3):
        # extremes of r (cos t x_i + sin t y_i) lie at atan2(y_i, x_i) + k pi
        base = atan2(y[i], x[i])
        for k in range(ceil((t0 - base) / pi), floor((t1 - base) / pi) + 1):
            if t0 < base + k * pi < t1:
                ts.append(base + k * pi)
    pts = [curve_eval(curve, t) for t in ts]
    return tuple(min(p[i] for p in pts) for i in range(3)), tuple(max(p[i] for p in pts) for i in range(3))


# ------------------------------------------------------------------------------------------ scaling
def scale_curve_def(curve, f) -> dict:
    out = dict(curve)
    kind = curve["type"]
    if kind == "nurbs":
        return {**nurbs.scale_curve(curve, f), "type": "nurbs"}
    out["origin"] = [c * f for c in curve["origin"]]
    if kind == "circle":
        out["radius"] = curve["radius"] * f
    return out


def scale_surface_def(surface, f) -> dict:
    kind = surface["type"]
    if kind == "nurbs":
        return {**nurbs.scale_surface(surface, f), "type": "nurbs"}
    out = dict(surface)
    out["origin"] = [c * f for c in surface["origin"]]
    for key in ("radius", "major_radius", "minor_radius"):
        if key in out:
            out[key] = out[key] * f
    return out
