"""Native SDRW ellipse and spline entities (feature ``sdrw-curves-v1``): validation, exact measurements, transforms, persistence, SVG and DXF."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from src.utils.sdrw import SDRWDocument
from src.utils.spatial_formats.common.errors import GeometryValidationError, UnsupportedFeatureError, ValidationError
from src.utils.spatial_formats.container.versions import FEATURES
from src.utils.spatial_formats.sdrw import curves

CTRL = [(0, 0), (10, 14), (22, -6), (35, 9), (48, 0)]


def doc():
    return SDRWDocument.new("c", unit="mm")


def dist_to_polyline(p, poly, closed=False):
    best = float("inf")
    for i in range(len(poly) if closed else len(poly) - 1):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = (b[0] - a[0], b[1] - a[1])
        den = ab[0] ** 2 + ab[1] ** 2
        t = 0.0 if den == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / den))
        best = min(best, math.hypot(p[0] - a[0] - ab[0] * t, p[1] - a[1] - ab[1] * t))
    return best


def spline_geometry(points=CTRL, degree=3):
    return {"degree": degree, "control_points": [list(map(float, p)) for p in points], "knots": curves.clamped_uniform_knots(len(points), degree)}


# ---------------------------------------------------------------------------------------------------- validation
@pytest.mark.parametrize("geometry", [
    {"center": [0, 0], "rx": 0, "ry": 1},
    {"center": [0, 0], "rx": 1, "ry": -1},
    {"center": [0, 0], "rx": float("nan"), "ry": 1},
    {"center": [0], "rx": 1, "ry": 1},
    {"center": [0, 0], "rx": 1, "ry": 1, "start_angle": 0.0},          # no elliptical arcs
])
def test_bad_ellipses_are_refused(geometry):
    d = doc()
    with pytest.raises((GeometryValidationError, ValidationError)):
        d.apply([{"op": "add", "kind": "ellipse", "geometry": geometry}])


def _spline(**changes):
    g = spline_geometry()
    g.update(changes)
    return g


@pytest.mark.parametrize("geometry", [
    _spline(degree=0),
    _spline(degree=True),
    _spline(degree=10),
    _spline(control_points=[[0, 0], [1, 1], [2, 2]]),                  # fewer points than degree + 1
    _spline(control_points=[[0, 0]] * 5),                              # all coincide
    _spline(knots=[0, 0, 0, 0, 1, 1, 1, 1]),                           # one knot short
    _spline(knots=[0, 0, 0, 0, 2, 1, 3, 3, 3]),                        # decreasing
    _spline(knots=[0, 0, 0, 1, 1, 2, 3, 3, 3]),                        # not clamped at the start
    _spline(knots=[0, 0, 0, 0, 1, 2, 2, 2, 3]),                        # not clamped at the end
    _spline(knots=[0, 0, 0, 0, 0, 2, 2, 2, 2]),                        # a clamped end repeated more than degree + 1 times
    _spline(weights=[1, 1, 1, 1, 1]),                                  # rational splines are not part of the format
])
def test_bad_splines_are_refused(geometry):
    d = doc()
    with pytest.raises((GeometryValidationError, ValidationError)):
        d.apply([{"op": "add", "kind": "spline", "geometry": geometry}])


def test_an_interior_knot_may_not_repeat_more_than_the_degree():
    g = spline_geometry([(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1), (6, 0)], 2)
    g["knots"] = [0, 0, 0, 1, 1, 1, 2, 3, 3, 3]
    with pytest.raises(GeometryValidationError):
        curves.validate_spline(g)
    g["knots"] = [0, 0, 0, 1, 1, 2, 3, 4, 4, 4]
    curves.validate_spline(g)                                           # a double knot with degree 2 is fine


def test_add_spline_defaults_to_a_clamped_uniform_knot_vector_and_bezier_chains_must_join():
    d = doc()
    s = d.add_spline(CTRL)
    assert d.get(s)["geometry"]["knots"] == [0, 0, 0, 0, 1, 2, 2, 2, 2]
    with pytest.raises(ValidationError, match="join"):
        d.add_bezier_spline([((0, 0), (1, 1), (2, 1), (3, 0)), ((4, 0), (5, 1), (6, 1), (7, 0))])
    with pytest.raises(ValidationError):
        d.add_spline(CTRL, degree=3.0)


# ---------------------------------------------------------------------------------------------------- measurement
def test_ellipse_circumference_area_and_bounds_are_exact():
    g = {"center": [3.0, 2.0], "rx": 20.0, "ry": 6.0, "rotation": 0.5}
    n = 200_000                                                          # independent: a fine inscribed polygon
    pts = [curves.ellipse_point(g, math.tau * i / n) for i in range(n)]
    perimeter = sum(math.dist(pts[i], pts[(i + 1) % n]) for i in range(n))
    assert curves.ellipse_length(20.0, 6.0) == pytest.approx(perimeter, rel=1e-9)
    assert curves.ellipse_length(5.0, 5.0) == pytest.approx(math.tau * 5.0, rel=1e-14)
    d = doc()
    e = d.add_ellipse((3, 2), 20, 6, rotation=0.5)
    m = d.measure(e)
    assert m["length"] == pytest.approx(perimeter, rel=1e-9)
    assert m["area"] == pytest.approx(math.pi * 120, rel=1e-14)
    (x0, y0), (x1, y1) = m["bounds"]
    assert x0 == pytest.approx(min(p[0] for p in pts), abs=1e-6) and x1 == pytest.approx(max(p[0] for p in pts), abs=1e-6)
    assert y0 == pytest.approx(min(p[1] for p in pts), abs=1e-6) and y1 == pytest.approx(max(p[1] for p in pts), abs=1e-6)


@pytest.mark.parametrize("degree,count", [(1, 4), (2, 5), (3, 5), (3, 9)])
def test_spline_bezier_form_length_and_bounds(degree, count):
    pts = [(i * 6.0, 9.0 * math.sin(i * 1.3) + (i % 2) * 3) for i in range(count)]
    g = spline_geometry(pts, degree)
    lo, hi = curves.spline_domain(g)
    n = 20_000
    dense = [curves.spline_point(g, lo + (hi - lo) * i / n) for i in range(n + 1)]
    assert dense[0] == pytest.approx(pts[0]) and dense[-1] == pytest.approx(pts[-1])
    # the Bezier segments are the same curve: every Bezier point lies on the de Boor curve
    for seg in curves.bezier_segments(g):
        for k in range(0, 11):
            q = curves._bernstein_point(seg, k / 10)
            assert dist_to_polyline(q, dense) < 1e-3
    if degree == 1:                                                      # a polyline: the exact length is the control polygon's
        assert curves.spline_length(g) == pytest.approx(sum(math.dist(pts[i], pts[i + 1]) for i in range(count - 1)), rel=1e-13)
    else:
        chord = sum(math.dist(dense[i], dense[i + 1]) for i in range(n))
        assert curves.spline_length(g) == pytest.approx(chord, rel=1e-6)
    (x0, y0), (x1, y1) = curves.spline_bounds(g)
    assert x0 <= min(p[0] for p in dense) + 1e-9 and x1 >= max(p[0] for p in dense) - 1e-9
    assert y0 <= min(p[1] for p in dense) + 1e-9 and y1 >= max(p[1] for p in dense) - 1e-9
    if degree <= 3:                                                      # tight, not just containing
        assert y1 == pytest.approx(max(p[1] for p in dense), abs=1e-3)
        assert y0 == pytest.approx(min(p[1] for p in dense), abs=1e-3)


def test_flattening_honours_its_stated_deviation():
    for degree in (2, 3, 5):
        g = spline_geometry([(0, 0), (5, 9), (12, -4), (20, 7), (28, -3), (36, 6)], degree)
        lo, hi = curves.spline_domain(g)
        tol = 1e-3
        poly = curves.flatten_spline(g, tol)
        assert math.dist(poly[0], (0, 0)) < 1e-12 and math.dist(poly[-1], (36, 6)) < 1e-12
        worst = max(dist_to_polyline(curves.spline_point(g, lo + (hi - lo) * i / 3000), poly) for i in range(3001))
        assert worst <= tol * 1.0001, (degree, worst)
    e = {"center": [1.0, 2.0], "rx": 30.0, "ry": 4.0, "rotation": 0.3}
    poly = curves.flatten_ellipse(e, 1e-3)
    worst = max(dist_to_polyline(curves.ellipse_point(e, math.tau * (i + 0.37) / 700), poly, closed=True) for i in range(700))
    assert worst <= 1e-3 * 1.0001


# ---------------------------------------------------------------------------------------------------- transforms
def test_rotate_scale_and_move_keep_the_curves_exact():
    d = doc()
    e = d.add_ellipse((3, 2), 20, 6, rotation=0.5)
    s = d.add_spline(CTRL)
    before_e, before_s = d.get(e)["geometry"], d.get(s)["geometry"]
    d.rotate([e, s], 90, about=(1, 1), degrees=True)
    d.scale([e, s], 2.0, about=(0, 0))
    d.move([e, s], 5, -3)
    ge, gs = d.get(e)["geometry"], d.get(s)["geometry"]
    assert (ge["rx"], ge["ry"]) == pytest.approx((40, 12))
    assert ge["rotation"] == pytest.approx(0.5 + math.pi / 2)

    def moved(p):                                                        # the same three steps applied to a point
        x, y = p[0] - 1, p[1] - 1
        x, y = -y + 1, x + 1
        return (2 * x + 5, 2 * y - 3)

    for t in (0.0, 1.0, 2.5, 4.0):
        assert curves.ellipse_point(ge, t) == pytest.approx(moved(curves.ellipse_point(before_e, t)), abs=1e-9)
    lo, hi = curves.spline_domain(before_s)
    for k in range(11):
        u = lo + (hi - lo) * k / 10
        assert curves.spline_point(gs, u) == pytest.approx(moved(curves.spline_point(before_s, u)), abs=1e-9)
    assert gs["knots"] == before_s["knots"]
    with pytest.raises(ValidationError):                                  # reflections stay refused, as for circles and arcs
        d.transform([e], [-1, 0, 0, 0, 1, 0, 0, 0, 1])


def test_blocks_expand_and_units_convert_with_the_curves():
    d = doc()
    d.define_block("b", [{"kind": "ellipse", "geometry": {"center": [1, 0], "rx": 2, "ry": 1}},
                         {"kind": "spline", "geometry": spline_geometry([(0, 0), (1, 2), (2, 0), (3, 2)])}])
    ins = d.add_insert("b", (10, 0), rotation=90, scale=2, degrees=True)
    from src.utils.spatial_formats.sdrw.operations import expand_insert
    parts = expand_insert(d.get(ins), {b["name"]: b for b in d._model.blocks})
    kinds = sorted(p["kind"] for p in parts)
    assert kinds == ["ellipse", "spline"]
    ell = next(p for p in parts if p["kind"] == "ellipse")["geometry"]
    assert ell["center"] == pytest.approx([10, 2]) and (ell["rx"], ell["ry"]) == pytest.approx((4, 2)) and ell["rotation"] == pytest.approx(math.pi / 2)
    spl = next(p for p in parts if p["kind"] == "spline")["geometry"]
    assert spl["control_points"][1] == pytest.approx([10 - 4, 2])               # (1, 2) turned 90 degrees, doubled, moved to (10, 0)
    f = SDRWDocument.new("u", unit="mm")
    e = f.add_ellipse((10, 20), 30, 10)
    s = f.add_spline([(0, 0), (10, 20), (20, 0), (30, 20)])
    f.convert_unit("cm")
    g = f.get(e)["geometry"]
    assert (g["center"], g["rx"], g["ry"]) == pytest.approx(([1, 2], 3, 1))
    assert f.get(s)["geometry"]["control_points"][-1] == pytest.approx([3, 2])
    assert f.get(s)["geometry"]["knots"] == [0, 0, 0, 0, 1, 1, 1, 1]


# ---------------------------------------------------------------------------------------------------- persistence
def test_files_declare_the_feature_only_when_curves_are_present_and_old_readers_refuse_it(monkeypatch):
    plain = doc()
    plain.add_line((0, 0), (1, 1))
    assert "sdrw-curves-v1" not in plain.manifest()["requires"]                 # older readers keep reading such files
    d = doc()
    e = d.add_ellipse((3, 2), 20, 6, rotation=0.5)
    s = d.add_spline(CTRL)
    assert "sdrw-curves-v1" in d.manifest()["requires"]
    blob = d.dumps()
    again = SDRWDocument.from_bytes(blob)
    assert again.get(e)["geometry"] == d.get(e)["geometry"] and again.get(s)["geometry"] == d.get(s)["geometry"]
    assert again.content_digest() == d.content_digest()
    with monkeypatch.context() as patch:                                         # a reader without the feature must refuse, not guess
        patch.setitem(FEATURES, "0.3.0", FEATURES["0.3.0"] - {"sdrw-curves-v1"})
        with pytest.raises(UnsupportedFeatureError, match="sdrw-curves-v1"):
            SDRWDocument.from_bytes(blob)
    # a curve inside a block also needs the feature
    b = doc()
    b.define_block("b", [{"kind": "ellipse", "geometry": {"center": [0, 0], "rx": 2, "ry": 1}}])
    assert "sdrw-curves-v1" in b.manifest()["requires"]


def test_legacy_versions_cannot_contain_the_new_kinds():
    from src.utils.spatial_formats.common.errors import MigrationError
    from src.utils.spatial_formats.container import migration
    with pytest.raises(MigrationError):
        migration._check_legacy_sdrw({"entities": [{"kind": "ellipse"}]})
    assert "ellipse" not in migration.LEGACY_SDRW_KINDS and "spline" not in migration.LEGACY_SDRW_KINDS


def test_the_json_schema_lists_the_new_kinds():
    import json
    from pathlib import Path
    root = Path(__import__("src.utils.spatial_formats", fromlist=["x"]).__file__).parent / "schemas"
    current = json.loads((root / "sdrw.json").read_text())["$defs"]["entity"]["properties"]["kind"]["enum"]
    assert {"ellipse", "spline"} <= set(current)
    for frozen in ("sdrw_0_1.json", "sdrw_0_2.json"):                           # earlier versions stay exactly as they were
        text = (root / frozen).read_text()
        assert '"ellipse"' not in text and '"spline"' not in text


# ---------------------------------------------------------------------------------------------------- SVG
SVG_NS = "{http://www.w3.org/2000/svg}"


def exported(d, *, high_degree=0):
    """SVG of ``d``. Only splines of degree above 3 may be reported as approximated (SVG has no primitive for them); everything else is exact."""
    markup, report = d.to_svg()
    assert [a for a in report.approximations if "degree above 3" not in a] == []
    assert len([a for a in report.approximations if "degree above 3" in a]) == (1 if high_degree else 0)
    return markup, ET.fromstring(markup)


def test_svg_writes_real_curves_in_the_same_axes_as_every_other_entity():
    d = doc()
    ln = d.add_line((0, 0), (5, 7))
    e = d.add_ellipse((3, 2), 20, 6, rotation=0.5)
    s = d.add_spline(CTRL)
    markup, root = exported(d)
    by_id = {el.get("data-id"): el for el in root}
    assert by_id[e].tag == SVG_NS + "ellipse" and by_id[s].tag == SVG_NS + "path"
    el = by_id[e]
    assert (float(el.get("cx")), float(el.get("cy")), float(el.get("rx")), float(el.get("ry"))) == (3.0, 2.0, 20.0, 6.0)   # y is not flipped for one entity only
    assert float(by_id[ln].get("y2")) == 7.0
    assert el.get("transform").startswith("rotate(%r" % math.degrees(0.5))
    nums = [float(v) for v in by_id[s].get("d").replace("M", " ").replace("C", " ").replace(",", " ").split()]
    cubic = curves.cubic_segments(d.get(s)["geometry"])
    want = list(cubic[0][0]) + [c for seg in cubic for q in seg[1:] for c in q]
    assert nums == pytest.approx(want, abs=1e-12)
    assert by_id[s].get("fill") == "none"


def test_svg_round_trip_is_exact_for_every_degree_and_tampering_is_caught():
    d = doc()
    ids = [d.add_ellipse((3, 2), 20, 6, rotation=0.5), d.add_ellipse((0, 0), 4, 9),
           d.add_spline(CTRL), d.add_spline([(0, 0), (5, 9), (12, -4), (20, 7), (28, -3), (36, 6)], degree=5),
           d.add_bezier_spline([((0, 30), (5, 40), (10, 40), (15, 30)), ((15, 30), (20, 20), (25, 20), (30, 30))])]
    markup, _ = exported(d, high_degree=1)
    back, _ = SDRWDocument.from_svg(markup, preserve_ids=True)
    for i in ids:
        assert back.get(i)["kind"] == d.get(i)["kind"]
        assert back.get(i)["geometry"] == d.get(i)["geometry"], i
    with pytest.raises(ValidationError, match="transform disagrees"):
        SDRWDocument.from_svg(markup.replace('rotate(28.6', 'rotate(30.6'))
    with pytest.raises(ValidationError, match="path disagrees"):
        SDRWDocument.from_svg(markup.replace("C 10.0,14.0", "C 11.0,14.0", 1))
    with pytest.raises(UnsupportedFeatureError):
        SDRWDocument.from_svg('<svg xmlns="http://www.w3.org/2000/svg" version="1.1"><path d="M 0,0 C 1,1 2,2 3,3"/></svg>')
    with pytest.raises(UnsupportedFeatureError):                                 # a transform is accepted on an ellipse only
        SDRWDocument.from_svg('<svg xmlns="http://www.w3.org/2000/svg" version="1.1"><circle cx="0" cy="0" r="1" transform="rotate(5)"/></svg>')
    plain, _ = SDRWDocument.from_svg('<svg xmlns="http://www.w3.org/2000/svg" version="1.1"><ellipse cx="1" cy="2" rx="3" ry="2"/></svg>')
    only = plain.get(plain.query_ids()[0])
    assert only["kind"] == "ellipse" and only["geometry"] == {"center": [1.0, 2.0], "rx": 3.0, "ry": 2.0, "rotation": 0.0}


# ---------------------------------------------------------------------------------------------------- DXF (needs ezdxf)
def _norm(kind, g):
    if kind == "ellipse":
        rx, ry, rot = g["rx"], g["ry"], g.get("rotation", 0.0)
        if ry > rx:
            rx, ry, rot = ry, rx, rot + math.pi / 2
        return ("E", round(g["center"][0], 9), round(g["center"][1], 9), round(rx, 9), round(ry, 9), round(rot % math.pi, 9))
    return ("S", g["degree"], tuple(round(c, 9) for p in g["control_points"] for c in p), tuple(round(k, 9) for k in g["knots"]))


def test_dxf_round_trip_of_ellipses_and_splines(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    d = doc()
    ids = [d.add_ellipse((3, 2), 20, 6, rotation=0.5), d.add_ellipse((-3, 1), 4, 9, rotation=-0.3),     # the second has the longer axis at +90 degrees
           d.add_spline(CTRL), d.add_bezier_spline([((0, 30), (5, 40), (10, 40), (15, 30)), ((15, 30), (20, 20), (25, 20), (30, 30))])]
    report = d.export_dxf(tmp_path / "c.dxf")
    assert not report.approximations
    kinds = sorted(e.dxftype() for e in ezdxf.readfile(str(tmp_path / "c.dxf")).modelspace())
    assert kinds == ["ELLIPSE", "ELLIPSE", "SPLINE", "SPLINE"]
    back, rep = SDRWDocument.import_dxf(tmp_path / "c.dxf")
    got = sorted(_norm(e["kind"], e["geometry"]) for e in (back.get(i) for i in back.query_ids()))
    assert got == sorted(_norm(d.get(i)["kind"], d.get(i)["geometry"]) for i in ids)


def test_dxf_import_takes_over_clean_splines_and_refuses_the_rest(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    doc_ = ezdxf.new("R2010")
    msp = doc_.modelspace()
    msp.add_open_spline([(0, 0, 0), (1, 2, 0), (3, 3, 0), (5, 0, 0)], degree=3)
    rational = msp.add_open_spline([(0, 0, 0), (1, 2, 0), (3, 3, 0), (5, 0, 0)], degree=3)
    rational.weights = [1.0, 2.0, 1.0, 1.0]
    elevated = msp.add_open_spline([(0, 0, 4), (1, 2, 4), (3, 3, 4), (5, 0, 4)], degree=3)
    doc_.saveas(str(tmp_path / "f.dxf"))
    with pytest.raises(UnsupportedFeatureError, match="rational"):
        SDRWDocument.import_dxf(tmp_path / "f.dxf")
    d, report = SDRWDocument.import_dxf(tmp_path / "f.dxf", skip_unsupported=True)
    assert d.count() == 1 and d.get(d.query_ids()[0])["kind"] == "spline"
    assert sum("SPLINE" in line for line in report.dropped) == 2
