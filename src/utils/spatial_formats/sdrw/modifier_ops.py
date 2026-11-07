"""Non-destructive 2D region and drafting modifier evaluators for SDRW.

Results remain native SDRW records; multi-ring and multi-part regions use the
existing solid-hatch representation with even-odd boundary semantics. GEOS is
a numerical planar backend and never advertises exact arithmetic guarantees.
"""
from __future__ import annotations

from copy import deepcopy
from math import acos, ceil, cos, hypot, pi, sin

from ..common.errors import (DependencyUnavailableError, GeometryValidationError,
                             ResourceLimitError, UnsupportedFeatureError)
from ..common.helpers import finite
from .entities import validate_entity


def _shapely():
    try:
        from shapely.geometry import LineString, MultiPolygon, Polygon, Point
    except ImportError as exc:
        raise DependencyUnavailableError("SDRW region operations require shapely>=2") from exc
    return LineString, MultiPolygon, Polygon, Point


def _policy(modifier, allowed):
    mode = modifier["evaluation_policy"]["mode"]
    if mode not in allowed:
        raise UnsupportedFeatureError("operation cannot meet the requested evaluation policy",
                                      context={"mode": mode, "supported": sorted(allowed)})


def _sampling_step(radius, policy):
    if policy["mode"] == "display_only":
        return pi / 128
    tolerance = max(policy.get("absolute", 0.0), radius * policy.get("relative", 0.0))
    if tolerance <= 0:
        raise UnsupportedFeatureError("curved region requires a positive chord tolerance")
    angular_step = 2 * acos(max(-1.0, 1 - tolerance / radius)) if tolerance < radius else pi / 2
    if angular_step < pi / 2048:
        raise ResourceLimitError("requested curve chord tolerance would require too many segments")
    return min(pi / 2, angular_step)


def _region(record, policy):
    _, _, Polygon, Point = _shapely()
    if record is None:
        raise GeometryValidationError("empty region cannot be used as an operand")
    kind, g = record["kind"], record["geometry"]
    approximate = False
    if kind == "polyline" and g.get("closed"):
        if len(g["points"]) > 50_000:
            raise ResourceLimitError("region Boolean input has too many vertices")
        area = Polygon(g["points"])
    elif kind == "circle":
        step = _sampling_step(g["radius"], policy)
        area = Point(g["center"]).buffer(g["radius"], quad_segs=max(4, int(ceil(pi / (2 * step)))))
        approximate = True
    elif kind == "hatch":
        from .annotations import loop_polygon
        area = None
        total_points = 0
        for loop in g["boundary"]:
            radii = ([loop["radius"]] if loop["kind"] == "circle" else
                     [seg["radius"] for seg in loop.get("segments", ()) if seg["kind"] == "arc"])
            step = _sampling_step(max(radii), policy) if radii else pi / 128
            points = loop_polygon(loop, step=step)
            total_points += len(points)
            if total_points > 50_000:
                raise ResourceLimitError("region Boolean input has too many sampled boundary vertices")
            poly = Polygon(points)
            if not poly.is_valid:
                raise GeometryValidationError("hatch loop is not a valid simple region")
            area = poly if area is None else area.symmetric_difference(poly)
            approximate |= loop["kind"] != "polygon"
    else:
        raise UnsupportedFeatureError("region modifier requires a closed polyline, circle or hatch")
    if not area.is_valid or area.is_empty or area.area <= 0:
        raise GeometryValidationError("invalid, empty or self-intersecting input region")
    return area, approximate


def _as_hatch(source, region):
    _, MultiPolygon, Polygon, _ = _shapely()
    if region.is_empty:
        return None
    if isinstance(region, Polygon):
        polygons = [region]
    elif isinstance(region, MultiPolygon):
        polygons = list(region.geoms)
    else:
        raise UnsupportedFeatureError("operation produced lines or points, not a 2D region")
    boundary = []
    output_points = 0
    for poly in polygons:
        if not poly.is_valid or poly.area <= 0:
            raise GeometryValidationError("region backend produced invalid polygon topology")
        for ring in (poly.exterior, *poly.interiors):
            vertices = [[float(x), float(y)] for x, y in list(ring.coords)[:-1]]
            output_points += len(vertices)
            if output_points > 100_000:
                raise ResourceLimitError("region result has too many boundary vertices")
            if len(vertices) < 3:
                raise GeometryValidationError("region boundary collapsed to fewer than three points")
            boundary.append({"kind": "polygon", "points": vertices})
    if len(boundary) > 512:
        raise ResourceLimitError("region result has too many boundary loops")
    output = deepcopy(source)
    output["kind"] = "hatch"
    output["geometry"] = {"boundary": boundary, "pattern": "solid"}
    return validate_entity(output)


def _boolean(source, operands, modifier):
    _policy(modifier, {"tolerance_controlled", "display_only"})
    policy = modifier["evaluation_policy"]
    target, target_approx = _region(source, policy)
    operand, operand_approx = _region(operands[0], policy)
    operation = modifier["parameters"]["operation"]
    try:
        region = ({"union": target.union, "intersect": target.intersection,
                   "subtract": target.difference}[operation])(operand)
    except (ValueError, RuntimeError) as exc:
        raise GeometryValidationError("planar Boolean backend failed") from exc
    result = _as_hatch(source, region)
    return result, {"backend": "geos", "approximate": target_approx or operand_approx,
                    "error_bound": None, "empty": result is None,
                    "area": float(region.area)}


def _line_intersection(source, operand, endpoint, *, extend):
    if source["kind"] != "line" or operand["kind"] != "line":
        raise UnsupportedFeatureError("trim/extend currently support straight line entities")
    a, b = source["geometry"]["start"], source["geometry"]["end"]
    c, d = operand["geometry"]["start"], operand["geometry"]["end"]
    u, v = (b[0]-a[0], b[1]-a[1]), (d[0]-c[0], d[1]-c[1])
    denominator = u[0]*v[1] - u[1]*v[0]
    if abs(denominator) <= 1e-12 * max(1.0, hypot(*u) * hypot(*v)):
        raise GeometryValidationError("trim/extend lines are parallel or nearly parallel")
    w = (c[0]-a[0], c[1]-a[1])
    t = (w[0]*v[1]-w[1]*v[0]) / denominator
    t_other = (w[0]*u[1]-w[1]*u[0]) / denominator
    eps = 1e-10
    if not -eps <= t_other <= 1 + eps:
        raise GeometryValidationError("boundary line does not cross its finite segment")
    if extend and not (t > 1 + eps if endpoint == "end" else t < -eps):
        raise GeometryValidationError("intersection does not extend the selected endpoint")
    if not extend and not (eps < t < 1-eps):
        raise GeometryValidationError("intersection does not trim the source line interior")
    point = [finite(a[i] + t*u[i], "line intersection") for i in range(2)]
    out = deepcopy(source)
    out["geometry"][endpoint] = point
    return validate_entity(out), {"backend": "analytic-line-intersection", "approximate": False,
                                  "parameter": t, "intersection": point}


def _offset(source, modifier):
    distance = modifier["parameters"]["distance"]
    kind = source["kind"]
    if kind == "circle":
        _policy(modifier, {"exact", "tolerance_controlled", "display_only"})
        radius = source["geometry"]["radius"] + distance
        if radius <= 0:
            raise GeometryValidationError("circle offset collapses the radius")
        out = deepcopy(source)
        out["geometry"]["radius"] = radius
        return validate_entity(out), {"backend": "analytic-circle-offset", "approximate": False}
    _policy(modifier, {"tolerance_controlled", "display_only"})
    LineString, _, _, _ = _shapely()
    if kind in {"line", "polyline"} and not source["geometry"].get("closed", False):
        g = source["geometry"]
        points = [g["start"], g["end"]] if kind == "line" else g["points"]
        offset = LineString(points).offset_curve(distance, join_style="mitre")
        if offset.is_empty or offset.geom_type != "LineString":
            raise GeometryValidationError("offset produced empty or disconnected linework")
        coords = [[float(x), float(y)] for x, y in offset.coords]
        out = deepcopy(source)
        out["kind"] = "line" if len(coords) == 2 else "polyline"
        out["geometry"] = ({"start": coords[0], "end": coords[-1]} if len(coords) == 2 else
                           {"points": coords, "closed": False})
        return validate_entity(out), {"backend": "geos-offset-curve", "approximate": False,
                                      "error_bound": None}
    area, approximate = _region(source, modifier["evaluation_policy"])
    region = area.buffer(distance, join_style="mitre")
    result = _as_hatch(source, region)
    return result, {"backend": "geos-region-buffer", "approximate": approximate,
                    "error_bound": None, "empty": result is None}


def _bend(source, modifier):
    _policy(modifier, {"display_only"})
    params = modifier["parameters"]
    if source["kind"] not in {"line", "polyline"} or source["geometry"].get("closed", False):
        raise UnsupportedFeatureError("2D bend supports open lines and polylines")
    g = source["geometry"]
    points = [g["start"], g["end"]] if source["kind"] == "line" else g["points"]
    span = params["end"]-params["start"]
    axis = 0 if params["axis"] == "x" else 1
    other = 1-axis
    step = span / params["segments"]
    sampled = [list(points[0])]
    for a,b in zip(points,points[1:]):
        count = max(1, int(ceil(abs(b[axis]-a[axis])/step)))
        if count > 1024:
            raise ResourceLimitError("bend requires too many polyline subdivisions")
        sampled.extend([[a[i]+(b[i]-a[i])*j/count for i in range(2)] for j in range(1,count+1)])
    radius = span/params["angle"]
    offsets = [radius+p[other] for p in sampled]
    if min(offsets) <= 1e-8 <= max(offsets) or min(abs(v) for v in offsets) <= 1e-8:
        raise GeometryValidationError("2D bend intersects its center of curvature")
    result = []
    for p in sampled:
        s, t = p[axis], p[other]
        if s <= params["start"]:
            result.append(p)
            continue
        theta = params["angle"]*(min(s,params["end"])-params["start"])/span
        extra = max(0.0,s-params["end"])
        q = list(p)
        q[axis] = params["start"] + (radius+t)*sin(theta) + extra*cos(theta)
        q[other] = (radius+t)*cos(theta)-radius-extra*sin(theta)
        result.append([finite(v,"bent coordinate") for v in q])
    out = deepcopy(source)
    out["kind"] = "polyline"
    out["geometry"] = {"points": result, "closed": bool(g.get("closed", False))}
    return validate_entity(out), {"backend": "sampled-2d-bend", "approximate": True,
                                  "error_bound": None, "segments": len(result)-1}


def _chamfer(source, modifier):
    _policy(modifier, {"exact", "tolerance_controlled", "display_only"})
    if source["kind"] != "polyline":
        raise UnsupportedFeatureError("2D chamfer currently requires a polyline")
    amount, depth, segments = (modifier["parameters"][key] for key in ("amount", "depth", "segments"))
    g = source["geometry"]
    points = g["points"]
    closed = g.get("closed", False)
    if len(points) < 3:
        raise GeometryValidationError("chamfer requires at least one corner")
    LineString, _, Polygon, _ = _shapely()
    if closed and not Polygon(points).is_valid:
        raise GeometryValidationError("chamfer source polygon is self-intersecting")
    chamfered = []
    count = len(points)
    for i,p in enumerate(points):
        if not closed and i in {0, count-1}:
            chamfered.append(list(p))
            continue
        before, after = points[(i-1)%count], points[(i+1)%count]
        dl, dr = hypot(before[0]-p[0], before[1]-p[1]), hypot(after[0]-p[0], after[1]-p[1])
        if dl <= 2*amount or dr <= 2*depth:
            raise GeometryValidationError("chamfer amount/depth exceed adjacent edge clearance")
        incoming = [p[j] + (before[j]-p[j])*amount/dl for j in range(2)]
        outgoing = [p[j] + (after[j]-p[j])*depth/dr for j in range(2)]
        chamfered.append(incoming)
        # One segment is a straight chamfer. Additional segments approximate a
        # quadratic rounded corner controlled by the original vertex; merely
        # splitting the straight chamfer would add vertices without changing
        # the shape or honoring the user's segment count.
        chamfered.extend([[(1-t)**2*incoming[j] + 2*(1-t)*t*p[j] + t*t*outgoing[j]
                           for j in range(2)]
                          for k in range(1, segments+1) for t in (k/segments,)])
    out = deepcopy(source)
    out["geometry"] = {"points": chamfered, "closed": closed}
    validate_entity(out)
    if closed and not Polygon(chamfered).is_valid:
        raise GeometryValidationError("chamfer produced self-intersecting region")
    if not closed and not LineString(chamfered).is_simple:
        raise GeometryValidationError("chamfer produced self-intersecting open linework")
    return out, {"backend": "analytic-polyline-chamfer", "approximate": segments > 1,
                 "error_bound": None if segments > 1 else 0.0,
                 "segments": segments, "vertices": len(chamfered)}


def evaluate(source, operands, modifier):
    """Return one detached result and an operation diagnostic; never edit the drawing."""
    operation = modifier["kind"].split(".", 1)[1]
    if operation == "boolean":
        return _boolean(source, operands, modifier)
    if operation in {"trim", "extend"}:
        _policy(modifier, {"exact", "tolerance_controlled", "display_only"})
        return _line_intersection(source, operands[0], modifier["parameters"]["endpoint"],
                                  extend=operation == "extend")
    if operation == "offset":
        return _offset(source, modifier)
    if operation == "bend":
        return _bend(source, modifier)
    if operation == "chamfer":
        return _chamfer(source, modifier)
    raise UnsupportedFeatureError("unimplemented SDRW modifier kind")
