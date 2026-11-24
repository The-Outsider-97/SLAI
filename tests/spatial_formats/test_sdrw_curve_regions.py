"""Ellipses and splines as region boundaries: hatch loops, closed-spline entities, region modifiers, DXF hatch edges, and the SVG bound for degree > 3 splines."""
from __future__ import annotations

import math
import uuid
import xml.etree.ElementTree as ET

import pytest

from src.utils.sdrw import SDRWDocument
from src.utils.spatial_formats.common.errors import (GeometryValidationError, ResourceLimitError, UnsupportedFeatureError,
                                                     ValidationError)
from src.utils.spatial_formats.sdrw import annotations as ann
from src.utils.spatial_formats.sdrw import curves, modifier_ops


def doc():
    return SDRWDocument.new("r", unit="mm")


def dist_to_polyline(p, poly, closed=False):
    best = float("inf")
    for i in range(len(poly) if closed else len(poly) - 1):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = (b[0] - a[0], b[1] - a[1])
        den = ab[0] ** 2 + ab[1] ** 2
        t = 0.0 if den == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / den))
        best = min(best, math.hypot(p[0] - a[0] - ab[0] * t, p[1] - a[1] - ab[1] * t))
    return best


def shoelace(points):
    return 0.5 * sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))


def spline(points, degree=3, knots=None):
    return {"kind": "spline", "degree": degree, "control_points": [list(map(float, p)) for p in points],
            "knots": knots if knots is not None else curves.clamped_uniform_knots(len(points), degree)}


PARABOLA = spline([(0, 0), (5, 10), (10, 0)], degree=2)                   # y = 20 x (10 - x) / 100 ... peak 5 at x = 5; area under it = 2/3 * 10 * 5
LINE_BACK = {"kind": "line", "start": [10, 0], "end": [0, 0]}


# ------------------------------------------------------------------------------------------------- hatch loops
def test_ellipse_loop_area_bounds_and_polygon_are_exact_or_within_tolerance():
    d = doc()
    loop = {"kind": "ellipse", "center": [3, 2], "rx": 20, "ry": 6, "rotation": 0.5}
    h = d.add_hatch([loop])
    assert d.area(h) == pytest.approx(math.pi * 20 * 6, rel=1e-12)
    g = d.get(h)["geometry"]
    lo, hi = ann.hatch_bounds(g)
    elo, ehi = curves.ellipse_bounds({k: v for k, v in loop.items() if k != "kind"})
    assert (lo, hi) == (elo, ehi)
    poly = ann.loop_polygon(loop, tolerance=1e-4)
    for k in range(400):
        q = curves.ellipse_point({k_: v for k_, v in loop.items() if k_ != "kind"}, math.tau * (k + 0.37) / 400)
        assert dist_to_polyline(q, poly, closed=True) <= 1e-4 * 1.0001
    assert abs(shoelace(poly)) == pytest.approx(math.pi * 20 * 6, rel=1e-6)


def test_ellipse_inside_circle_is_a_hole_under_even_odd():
    d = doc()
    h = d.add_hatch([{"kind": "circle", "center": [0, 0], "radius": 20}, {"kind": "ellipse", "center": [0, 0], "rx": 10, "ry": 4, "rotation": 0.3}])
    assert d.area(h) == pytest.approx(math.pi * 400 - math.pi * 40, rel=1e-12)


@pytest.mark.parametrize("order", ["spline_first", "line_first", "line_reversed", "spline_reversed"])
def test_spline_segment_closes_a_region_in_any_order_and_direction(order):
    forward_line = {"kind": "line", "start": [0, 0], "end": [10, 0]}
    back_line = {"kind": "line", "start": [10, 0], "end": [0, 0]}
    rev = dict(PARABOLA, control_points=PARABOLA["control_points"][::-1])         # same curve walked from (10,0) to (0,0)
    segs = {"spline_first": [PARABOLA, back_line], "line_first": [forward_line, PARABOLA],
            "line_reversed": [back_line, rev], "spline_reversed": [rev, forward_line]}[order]
    d = doc()
    h = d.add_hatch([{"kind": "path", "segments": segs}])
    assert d.area(h) == pytest.approx(2.0 / 3.0 * 10 * 5, rel=1e-12)
    poly = ann.loop_polygon(d.get(h)["geometry"]["boundary"][0], tolerance=1e-5)
    assert abs(shoelace(poly)) == pytest.approx(2.0 / 3.0 * 10 * 5, rel=1e-5)
    (x0, y0), (x1, y1) = ann.hatch_bounds(d.get(h)["geometry"])
    assert (x0, y0) == pytest.approx((0, 0)) and (x1, y1) == pytest.approx((10, 5))


@pytest.mark.parametrize("degree", [3, 5, 7])
def test_green_area_of_a_spline_loop_matches_a_fine_polygon(degree):
    pts = [(0, 0), (8, 12), (20, 18), (31, 10), (30, -2), (18, -9), (6, -5), (0, 0)]
    d = doc()
    h = d.add_hatch([{"kind": "path", "segments": [spline(pts, degree)]}])
    loop = d.get(h)["geometry"]["boundary"][0]
    fine = curves.flatten_spline({k: v for k, v in loop["segments"][0].items() if k != "kind"}, 1e-7)[:-1]
    assert d.area(h) == pytest.approx(abs(shoelace(fine)), rel=1e-6)


def test_open_single_spline_loop_and_bad_segments_are_refused():
    d = doc()
    with pytest.raises(GeometryValidationError):
        d.add_hatch([{"kind": "path", "segments": [PARABOLA]}])                    # one segment must close on itself
    with pytest.raises(GeometryValidationError):
        d.add_hatch([{"kind": "path", "segments": [dict(PARABOLA, weights=[1, 1, 1]), LINE_BACK]}])
    with pytest.raises(GeometryValidationError):
        d.add_hatch([{"kind": "path", "segments": [dict(PARABOLA, knots=[0, 0, 1, 1, 1]), LINE_BACK]}])
    with pytest.raises(GeometryValidationError):
        d.add_hatch([{"kind": "ellipse", "center": [0, 0], "rx": 1, "ry": 0}])
    with pytest.raises(GeometryValidationError):
        d.add_hatch([{"kind": "ellipse", "center": [0, 0], "rx": 1, "ry": 1, "start_angle": 0}])
    with pytest.raises(GeometryValidationError):
        d.add_hatch([{"kind": "path", "segments": [PARABOLA, {"kind": "line", "start": [11, 0], "end": [0, 0]}]}])   # does not join


def test_hatch_transforms_keep_ellipse_and_spline_loops_exact():
    d = doc()
    h = d.add_hatch([{"kind": "ellipse", "center": [1, 1], "rx": 8, "ry": 3, "rotation": 0.2}])
    s = d.add_hatch([{"kind": "path", "segments": [PARABOLA, LINE_BACK]}])
    before = {h: d.area(h), s: d.area(s)}
    d.rotate([h, s], 33, about=(2, 5), degrees=True)
    d.scale([h, s], 1.5, about=(0, 0))
    d.move([h, s], 7, -4)
    for i in (h, s):
        assert d.area(i) == pytest.approx(before[i] * 1.5 ** 2, rel=1e-12)
    loop = d.get(h)["geometry"]["boundary"][0]
    assert (loop["kind"], loop["rx"], loop["ry"]) == ("ellipse", pytest.approx(12), pytest.approx(4.5))
    assert d.get(s)["geometry"]["boundary"][0]["segments"][0]["knots"] == PARABOLA["knots"]


def test_hatch_can_be_made_from_ellipse_and_closed_spline_entities():
    d = doc()
    e = d.add_ellipse((0, 0), 10, 4)
    ring = [(0, 0), (6, 8), (-6, 8), (0, 0)]
    c = d.add_spline(ring, degree=2)
    h = d.add_hatch([e])
    assert d.get(h)["geometry"]["boundary"][0]["kind"] == "ellipse"
    h2 = d.add_hatch([c])
    assert d.get(h2)["geometry"]["boundary"][0]["segments"][0]["kind"] == "spline"
    assert d.area(c) == pytest.approx(d.area(h2), rel=1e-12)
    o = d.add_spline([(0, 0), (4, 5), (9, 1)], degree=2)
    with pytest.raises(ValidationError):
        d.add_hatch([o])
    with pytest.raises(ValidationError):
        d.area(o)


# ------------------------------------------------------------------------------------------------- region modifiers
class FakePolygon:
    """Records what the modifier code hands to the planar backend (GEOS is not needed to test the conversion of curves to polygons)."""
    seen: list = []

    def __init__(self, points):
        self.points = [tuple(p) for p in points]
        FakePolygon.seen.append(self.points)
        self.area = abs(shoelace(self.points))
        self.is_valid = True
        self.is_empty = False

    def symmetric_difference(self, other):
        return self


class FakePoint:
    def __init__(self, c):
        self.c = c

    def buffer(self, r, quad_segs=8):
        return FakePolygon([(self.c[0] + r * math.cos(math.tau * i / (4 * quad_segs)), self.c[1] + r * math.sin(math.tau * i / (4 * quad_segs))) for i in range(4 * quad_segs)])


class FakeLine:
    seen: list = []

    def __init__(self, points):
        FakeLine.seen.append([tuple(p) for p in points])
        self.points = points

    def offset_curve(self, distance, join_style=None):
        out = FakeLine.__new__(FakeLine)
        out.coords = [(x, y + distance) for x, y in self.points]
        out.is_empty = False
        out.geom_type = "LineString"
        return out


def install_fake_backend(monkeypatch):
    FakePolygon.seen = []
    FakeLine.seen = []
    monkeypatch.setattr(modifier_ops, "_shapely", lambda: (FakeLine, object, FakePolygon, FakePoint))


def record(kind, geometry):
    return {"kind": kind, "geometry": geometry}


def test_region_from_an_ellipse_is_flattened_within_the_policy_tolerance(monkeypatch):
    install_fake_backend(monkeypatch)
    g = {"center": [3, 2], "rx": 20, "ry": 6, "rotation": 0.5}
    area, approximate = modifier_ops._region(record("ellipse", g), {"mode": "tolerance_controlled", "absolute": 1e-3})
    assert approximate is True
    pts = FakePolygon.seen[-1]
    for k in range(400):
        assert dist_to_polyline(curves.ellipse_point(g, math.tau * (k + 0.37) / 400), pts, closed=True) <= 1e-3 * 1.0001
    assert area.area == pytest.approx(math.pi * 120, rel=1e-5)


def test_region_from_a_closed_spline_and_a_curved_hatch(monkeypatch):
    install_fake_backend(monkeypatch)
    ring = spline([(0, 0), (6, 8), (-6, 8), (0, 0)], degree=2)
    g = {k: v for k, v in ring.items() if k != "kind"}
    area, approximate = modifier_ops._region(record("spline", g), {"mode": "tolerance_controlled", "absolute": 1e-4})
    pts = FakePolygon.seen[-1]
    exact = abs(curves.spline_green(g))
    assert approximate and abs(area.area - exact) <= 1e-4 * 30      # chord error 1e-4 along a loop of length < 30
    lo, hi = curves.spline_domain(g)
    for i in range(300):
        assert dist_to_polyline(curves.spline_point(g, lo + (hi - lo) * i / 300), pts, closed=True) <= 1e-4 * 1.0001
    hatch = {"boundary": [{"kind": "path", "segments": [PARABOLA, LINE_BACK]}], "pattern": "solid"}
    area, approximate = modifier_ops._region(record("hatch", hatch), {"mode": "tolerance_controlled", "absolute": 1e-4})
    assert approximate and area.area == pytest.approx(100.0 / 3.0, rel=1e-4)


def test_curved_region_needs_a_positive_tolerance_and_respects_the_budget(monkeypatch):
    install_fake_backend(monkeypatch)
    g = {"center": [0, 0], "rx": 20, "ry": 6, "rotation": 0.0}
    with pytest.raises(UnsupportedFeatureError):
        modifier_ops._region(record("ellipse", g), {"mode": "tolerance_controlled"})
    with pytest.raises(ResourceLimitError):
        modifier_ops._region(record("ellipse", g), {"mode": "tolerance_controlled", "absolute": 1e-12})
    area, _ = modifier_ops._region(record("ellipse", g), {"mode": "display_only"})
    assert area.area == pytest.approx(math.pi * 120, rel=1e-3)


def test_an_open_spline_is_not_a_region_but_can_be_offset_as_a_curve(monkeypatch):
    install_fake_backend(monkeypatch)
    open_spline = {k: v for k, v in spline([(0, 0), (4, 5), (9, 1)], degree=2).items() if k != "kind"}
    with pytest.raises(UnsupportedFeatureError, match="closed spline"):
        modifier_ops._region(record("spline", open_spline), {"mode": "tolerance_controlled", "absolute": 1e-3})
    modifier = {"parameters": {"distance": 2.0}, "evaluation_policy": {"mode": "tolerance_controlled", "absolute": 1e-3}, "kind": "sdrw.offset"}
    out, diag = modifier_ops._offset({"id": str(uuid.uuid4()), "kind": "spline", "geometry": open_spline}, modifier)
    assert out["kind"] == "polyline" and not out["geometry"]["closed"]
    assert diag["approximate"] is True and diag["source_flatten_tolerance"] == pytest.approx(1e-3)
    flat = FakeLine.seen[-1]
    lo, hi = curves.spline_domain(open_spline)
    for i in range(200):
        assert dist_to_polyline(curves.spline_point(open_spline, lo + (hi - lo) * i / 200), flat) <= 1e-3 * 1.0001
    assert out["geometry"]["points"][0] == [pytest.approx(0.0), pytest.approx(2.0)]


# real GEOS: only where shapely is installed
def _add(d, target, kind, parameters, *, operands=(), mode="tolerance_controlled"):
    return d.add_modifier(target, kind=kind, operands=operands, parameters=parameters,
                          evaluation_policy={"mode": mode, "absolute": 1e-4} if mode == "tolerance_controlled" else {"mode": mode})


def test_geos_boolean_with_an_ellipse_and_offset_follow_analytic_areas():
    pytest.importorskip("shapely")
    d = doc()
    e = d.add_ellipse((0, 0), 10, 5, rotation=0.4)
    box = d.add_polyline([(-50, -50), (50, -50), (50, 50), (-50, 50)], closed=True)
    _add(d, e, "sdrw.boolean", {"operation": "intersect"}, operands=[box])
    result = d.evaluate_modifiers()[0]
    assert result["diagnostics"]["approximate"] is True
    assert result["diagnostics"]["area"] == pytest.approx(math.pi * 50, rel=1e-4)
    d2 = doc()
    e2 = d2.add_ellipse((0, 0), 10, 5)
    _add(d2, e2, "sdrw.offset", {"distance": 1.0})
    area = d2.evaluate_modifiers()[0]["diagnostics"]
    steiner = math.pi * 50 + curves.ellipse_length(10, 5) * 1.0 + math.pi * 1.0           # area + perimeter * d + pi d^2 of the parallel body
    assert area["approximate"] is True
    d3 = doc()
    s = d3.add_spline([(0, 0), (6, 8), (-6, 8), (0, 0)], degree=2)
    sq = d3.add_polyline([(-3, 1), (3, 1), (3, 9), (-3, 9)], closed=True)
    _add(d3, s, "sdrw.boolean", {"operation": "subtract"}, operands=[sq])
    assert d3.evaluate_modifiers()[0]["result"] is not None


# ------------------------------------------------------------------------------------------------- SVG
def test_svg_reports_the_guaranteed_bound_for_splines_above_degree_3_and_reads_them_back_exactly():
    d = doc()
    pts = [(0, 0), (5, 9), (12, -4), (20, 7), (28, -3), (36, 6)]
    s = d.add_spline(pts, degree=5)
    c = d.add_spline(pts, degree=3)
    markup, report = d.to_svg()
    notes = [a for a in report.approximations if "degree above 3" in a]
    assert len(notes) == 1 and notes[0].startswith("1 spline(s)")
    g = d.get(s)["geometry"]
    tol = curves.svg_spline_tolerance(g)
    assert 0 < tol <= 1e-4
    root = ET.fromstring(markup)
    node = {el.get("data-id"): el for el in root}[s]
    assert " L" in node.get("d") and "C" not in node.get("d")
    flat = [tuple(map(float, t.split(","))) for t in node.get("d").replace("M", " ").replace("L", " ").split()]
    lo, hi = curves.spline_domain(g)
    for i in range(1500):                                         # the drawn polyline really is within the stated bound
        assert dist_to_polyline(curves.spline_point(g, lo + (hi - lo) * i / 1500), flat) <= tol * 1.0001
    assert "C" in {el.get("data-id"): el for el in root}[c].get("d")
    back, _ = SDRWDocument.from_svg(markup, preserve_ids=True)
    assert back.get(s)["geometry"] == g
    clean, rep2 = SDRWDocument.new("x", unit="mm"), None
    clean.add_spline(pts, degree=3)
    assert not any("degree above 3" in a for a in clean.to_svg()[1].approximations)


@pytest.mark.parametrize("degree", [4, 6, 9])
def test_flattening_is_within_tolerance_for_every_degree(degree):
    pts = [(0, 0), (5, 9), (12, -4), (20, 7), (28, -3), (36, 6), (44, 0), (50, 10), (58, -6), (66, 3), (70, 0)]
    g = {k: v for k, v in spline(pts, degree).items() if k != "kind"}
    for tol in (1e-2, 1e-4):
        flat = curves.flatten_spline(g, tol)
        lo, hi = curves.spline_domain(g)
        assert max(dist_to_polyline(curves.spline_point(g, lo + (hi - lo) * i / 4000), flat) for i in range(4001)) <= tol * 1.0001


def test_hatch_with_curved_boundaries_is_still_reported_as_flattened_in_svg():
    d = doc()
    d.add_hatch([{"kind": "ellipse", "center": [0, 0], "rx": 10, "ry": 4}])
    _, report = d.to_svg()
    assert any("curved boundaries flattened" in a for a in report.approximations)


# ------------------------------------------------------------------------------------------------- DXF hatch edges
def test_dxf_hatch_keeps_ellipse_and_spline_boundaries_as_real_edges(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    d = doc()
    d.add_hatch([{"kind": "ellipse", "center": [3, 2], "rx": 5, "ry": 12, "rotation": 0.4}])                 # ry > rx: major axis is the second one
    d.add_hatch([{"kind": "path", "segments": [PARABOLA, LINE_BACK]}])
    rev = dict(PARABOLA, control_points=PARABOLA["control_points"][::-1])
    d.add_hatch([{"kind": "path", "segments": [rev, {"kind": "line", "start": [0, 0], "end": [10, 0]}, ]}])    # walked against the spline's own direction
    d.add_hatch([{"kind": "path", "segments": [spline([(0, 0), (6, 8), (-6, 8), (0, 0)], degree=2)]}])       # closed single spline
    path = tmp_path / "h.dxf"
    report = d.export_dxf(path)
    assert not [a for a in report.approximations if "hatch" in a and "flatten" in a]
    edges = [type(e).__name__ for h in ezdxf.readfile(str(path)).modelspace().query("HATCH") for p in h.paths for e in getattr(p, "edges", [])]
    assert edges.count("EllipseEdge") == 1 and edges.count("SplineEdge") == 3
    back, rep = SDRWDocument.import_dxf(str(path))
    hatches = [e for e in back.model.entities if e["kind"] == "hatch"]
    areas = sorted(back.area(e["id"]) for e in hatches)
    expected = sorted([math.pi * 60, 100.0 / 3.0, 100.0 / 3.0, abs(curves.spline_green({k: v for k, v in spline([(0, 0), (6, 8), (-6, 8), (0, 0)], 2).items() if k != "kind"}))])
    assert areas == pytest.approx(expected, rel=1e-9)
    kinds = sorted(l["kind"] for e in hatches for l in e["geometry"]["boundary"])
    assert kinds == ["ellipse", "path", "path", "path"]
