"""SDRW 0.2 model (text, dimensions, hatches, blocks, linetypes) and the 0.1.0 -> 0.2.0 migration."""
from __future__ import annotations

import json
import math
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from src.utils.sdrw import SDRWDocument
from src.utils.sgeo import SGEODocument
from src.utils.spatial_formats import inspect_package
from src.utils.spatial_formats.common.errors import (
    GeometryValidationError, MigrationError, ReferenceError, ResourceLimitError, SpatialFormatError,
    UnsupportedFeatureError, ValidationError)
from src.utils.spatial_formats.common.helpers import canonical_json, digest
from src.utils.spatial_formats.sdrw import annotations as ann


def fresh(unit="mm") -> SDRWDocument:
    return SDRWDocument.new("t", unit=unit)


def square(document, size=10.0, origin=(0.0, 0.0)):
    x, y = origin
    return document.add_polyline([(x, y), (x + size, y), (x + size, y + size), (x, y + size)], closed=True)


# --------------------------------------------------------------------------- linetypes
def test_linetype_definition_rules():
    d = fresh()
    d.define_linetype("dashed", [6, -3])
    assert d.linetypes() == [{"name": "dashed", "pattern": [6.0, -3.0]}]
    for bad in ("continuous", "ByLayer", "BYBLOCK"):
        with pytest.raises(ValidationError):
            d.define_linetype(bad, [1, -1])
    with pytest.raises(ValidationError):
        d.define_linetype("gaps", [-1, -2])                      # nothing is drawn
    with pytest.raises(ValidationError):
        d.define_linetype("zero", [0, -0.0])                     # no length
    with pytest.raises(ValidationError):
        d.define_linetype("dashed", [1, -1])                     # exists
    assert d.define_linetype("dashed", [1, -1], replace=True) == "dashed"
    assert d.linetypes()[0]["pattern"] == [1.0, -1.0]


def test_linetype_references_are_checked_and_protect_the_registry():
    d = fresh()
    with pytest.raises(ReferenceError):
        d.add_layer("A", linetype="nope")
    with pytest.raises(ValidationError):
        d.add_line((0, 0), (1, 0), style={"linetype": "byblock"})
    d.define_linetype("dashed", [6, -3])
    d.add_layer("A", linetype="dashed")
    line = d.add_line((0, 0), (1, 0), style={"linetype": "dashed", "linetype_scale": 2.0})
    with pytest.raises(ValidationError):
        d.remove_linetype("dashed")                              # used by a layer and an entity
    d.update(line, style={})
    d.update_layer("A", linetype=None)
    d.remove_linetype("dashed")
    assert d.linetypes() == []
    revision = d.revision
    with pytest.raises(ValidationError):
        d.add_line((0, 0), (1, 0), style={"linetype_scale": 0})
    assert d.revision == revision


def test_layer_style_is_listed_and_updatable():
    d = fresh()
    d.add_layer("A", stroke="#ff0000", stroke_width=0.5)
    assert next(l for l in d.layers() if l["name"] == "A")["stroke"] == "#ff0000"
    d.update_layer("A", visible=False, stroke=None, stroke_width=0.25)
    layer = next(l for l in d.layers() if l["name"] == "A")
    assert layer["visible"] is False and "stroke" not in layer and layer["stroke_width"] == 0.25
    with pytest.raises(ValidationError):
        d.update_layer("A")
    with pytest.raises(ValidationError):
        d.add_layer("B", stroke_width=-1)


# --------------------------------------------------------------------------- text
def test_text_validation_and_editing():
    d = fresh()
    t = d.add_text((1, 2), "Hello", 2.5, rotation=90, degrees=True, halign="center", valign="middle")
    assert d.get(t)["geometry"]["rotation"] == pytest.approx(math.pi / 2)
    for bad in (dict(text="", height=1), dict(text="x", height=0), dict(text="a\x00b", height=1),
                dict(text="x", height=1, halign="middle"), dict(text="x", height=1, valign="center"),
                dict(text="x", height=1, width=-1), dict(text="x", height=1, line_spacing=9)):
        with pytest.raises(GeometryValidationError):
            d.add_text((0, 0), bad.pop("text"), bad.pop("height"), **bad)
    d.set_text(t, "World")
    assert d.get(t)["geometry"]["text"] == "World"
    with pytest.raises(ValidationError):
        d.set_text(d.add_line((0, 0), (1, 0)), "nope")


def test_text_extents_are_estimated_and_follow_alignment_and_rotation():
    d = fresh()
    left = d.add_text((0, 0), "abcd", 10)                         # 4 chars * 0.6 * 10 wide, baseline at y=0
    assert d.measure(left)["bounds"] == ((0.0, 0.0), (24.0, 10.0))
    centred = d.add_text((0, 0), "abcd", 10, halign="center", valign="top")
    assert d.measure(centred)["bounds"] == ((-12.0, -10.0), (12.0, 0.0))
    turned = d.add_text((0, 0), "abcd", 10, rotation=90, degrees=True)
    (x0, y0), (x1, y1) = d.measure(turned)["bounds"]
    assert (x0, y0, x1, y1) == pytest.approx((-10.0, 0.0, 0.0, 24.0))
    wrapped = d.add_text((0, 0), "aaaa bbbb cccc", 10, width=60)  # 10 chars per line -> 2 lines
    assert ann.text_layout(d.get(wrapped)["geometry"])["lines"] == ["aaaa bbbb", "cccc"]


def test_text_moves_rotates_and_scales_with_geometry():
    d = fresh()
    t = d.add_text((1, 0), "A", 2)
    [t] = d.scale([t], 3, about=(0, 0))
    g = d.get(t)["geometry"]
    assert g["position"] == pytest.approx([3.0, 0.0]) and g["height"] == pytest.approx(6.0)
    [t] = d.rotate([t], 90, about=(0, 0), degrees=True)
    g = d.get(t)["geometry"]
    assert g["position"] == pytest.approx([0.0, 3.0], abs=1e-12) and g["rotation"] == pytest.approx(math.pi / 2)


# --------------------------------------------------------------------------- hatch
def test_hatch_area_is_even_odd_and_exact():
    d = fresh()
    outer, inner = square(d, 10), d.add_circle((5, 5), 2)
    h = d.add_hatch([outer, inner])
    assert d.area(h) == pytest.approx(100 - math.pi * 4)
    nested = d.add_hatch([square(d, 10), d.add_circle((5, 5), 4), d.add_circle((5, 5), 2)])
    assert d.area(nested) == pytest.approx(100 - math.pi * 16 + math.pi * 4)       # island inside the hole is filled again
    path = d.add_hatch([{"kind": "path", "segments": [
        {"kind": "line", "start": [0, 0], "end": [4, 0]},
        {"kind": "arc", "center": [4, 2], "radius": 2, "start_angle": -math.pi / 2, "end_angle": math.pi / 2},
        {"kind": "line", "start": [4, 4], "end": [0, 4]},
        {"kind": "line", "start": [0, 4], "end": [0, 0]}]}])
    assert d.area(path) == pytest.approx(16 + math.pi * 2)
    assert d.measure(path)["length"] is None and d.measure(path)["area"] == pytest.approx(16 + math.pi * 2)


def test_hatch_validation():
    d = fresh()
    open_line = d.add_polyline([(0, 0), (1, 0), (1, 1)])
    with pytest.raises(ValidationError):
        d.add_hatch([open_line])                                  # not a closed polyline
    big, crossing = square(d, 10), square(d, 10, (5, 5))
    with pytest.raises(GeometryValidationError):
        d.add_hatch([big, crossing])                              # loops intersect
    with pytest.raises(GeometryValidationError):
        d.add_hatch([big], pattern="lines")                       # no spacing
    with pytest.raises(ValidationError):
        d.add_hatch([big], pattern="solid", spacing=1)
    with pytest.raises(GeometryValidationError):
        d.add_hatch([big], pattern="lines", spacing=-1)
    with pytest.raises(GeometryValidationError):
        d.add_hatch([{"kind": "path", "segments": [{"kind": "line", "start": [0, 0], "end": [1, 0]},
                                                    {"kind": "line", "start": [5, 5], "end": [6, 5]}]}])
    with pytest.raises(GeometryValidationError):
        d.add_hatch([[(0, 0), (1, 1), (2, 2)]])                   # zero area


def test_hatch_pattern_lines_stay_inside_the_boundary():
    d = fresh()
    h = d.add_hatch([square(d, 10), d.add_circle((5, 5), 2)], pattern="lines", spacing=1.0, angle=0.0)
    lines = ann.hatch_lines(d.get(h)["geometry"])
    assert lines
    for (ax, ay), (bx, by) in lines:
        assert ay == pytest.approx(by) and -1e-9 <= min(ax, bx) and max(ax, bx) <= 10 + 1e-9
        mid = ((ax + bx) / 2, ay)
        assert math.hypot(mid[0] - 5, mid[1] - 5) >= 2 - 1e-6 or not (0 < mid[0] < 10)
    assert sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in lines) == pytest.approx(100 - math.pi * 4, rel=0.15)
    with pytest.raises(ResourceLimitError):
        ann.hatch_lines({**d.get(h)["geometry"], "spacing": 1e-6})


# --------------------------------------------------------------------------- dimensions
def test_dimension_values_and_text():
    d = fresh()
    lin = d.add_dimension_linear((0, 0), (30, 40), (0, -5))
    aligned = d.add_dimension_aligned((0, 0), (30, 40), (-6, 4))
    vertical = d.add_dimension_linear((0, 0), (30, 40), (50, 0), angle=90, degrees=True)
    ang = d.add_dimension_angular((0, 0), (10, 0), (0, 10), (5, 5))
    rad = d.add_dimension_radial((0, 0), (3, 4))
    dia = d.add_dimension_diameter((0, 0), (3, 4), suffix="mm", precision=1)
    assert d.dimension_value(lin) == pytest.approx(30.0)
    assert d.dimension_value(aligned) == pytest.approx(50.0)
    assert d.dimension_value(vertical) == pytest.approx(40.0)
    assert d.dimension_value(ang, degrees=True) == pytest.approx(90.0)
    assert [d.dimension_text(i) for i in (lin, aligned, ang, rad, dia)] == [
        "30.00", "50.00", "90.00\N{DEGREE SIGN}", "R5.00", "\N{DIAMETER SIGN}10.0mm"]
    over = d.add_dimension_aligned((0, 0), (3, 4), (0, 1), text="L=<> (typ.)", precision=0)
    blank = d.add_dimension_aligned((0, 0), (3, 4), (0, 1), text="")
    assert d.dimension_text(over) == "L=5 (typ.)" and d.dimension_text(blank) == ""
    reflex = d.add_dimension_angular((0, 0), (0, 10), (10, 0), (5, 5))                # ccw from p1 to p2 is 270 degrees
    assert d.dimension_value(reflex, degrees=True) == pytest.approx(270.0)
    assert d.measure(rad)["value"] == pytest.approx(5.0) and d.measure(rad)["text"] == "R5.00"


def test_dimension_validation():
    d = fresh()
    with pytest.raises(GeometryValidationError):
        d.add_dimension_linear((0, 0), (0, 0), (0, 1))
    with pytest.raises(GeometryValidationError):
        d.add_dimension_linear((0, 0), (10, 0), (0, 1), angle=90, degrees=True)           # measures zero along the line
    with pytest.raises(GeometryValidationError):
        d.add_dimension_radial((1, 1), (1, 1))
    with pytest.raises(GeometryValidationError):
        d.add_dimension_angular((0, 0), (1, 0), (2, 0), (1, 1))                           # coincident rays
    with pytest.raises(GeometryValidationError):
        d.add_dimension_aligned((0, 0), (1, 0), (0, 1), precision=9)
    with pytest.raises(GeometryValidationError):
        d.add_dimension_aligned((0, 0), (1, 0), (0, 1), arrow_size=0)
    with pytest.raises(GeometryValidationError):
        d._add("dimension", {"dim_type": "aligned", "p1": [0, 0], "p2": [1, 0], "dimline_point": [0, 1],
                              "center": [0, 0]}, {})                                      # field of another type


def test_dimension_values_follow_their_points_and_explode_into_primitives():
    d = fresh()
    dim = d.add_dimension_linear((0, 0), (10, 0), (0, -3))
    [dim] = d.scale([dim], 2, about=(0, 0))
    assert d.dimension_value(dim) == pytest.approx(20.0)                                  # derived, never stale
    kinds = sorted(p["kind"] for p in d.dimension_primitives(dim))
    assert kinds.count("text") == 1 and kinds.count("line") >= 3
    new = d.explode([dim])
    assert len(new) == len(kinds) and d.count(kind="dimension") == 0
    assert {d.get(i)["kind"] for i in new} <= {"line", "text", "arc"}


# --------------------------------------------------------------------------- blocks and inserts
def test_insert_bounds_apply_base_point_rotation_and_scale():
    d = fresh()
    d.define_block("b", [{"kind": "line", "geometry": {"start": [0, 0], "end": [4, 0]}}], base_point=(1, 0))
    plain = d.add_insert("b", (10, 10))
    assert d.measure(plain)["bounds"] == ((9.0, 10.0), (13.0, 10.0))                      # base point (1,0) lands on (10,10)
    big = d.add_insert("b", (10, 10), scale=2)
    assert d.measure(big)["bounds"] == ((8.0, 10.0), (16.0, 10.0))
    turned = d.add_insert("b", (0, 0), rotation=90, degrees=True)
    (x0, y0), (x1, y1) = d.measure(turned)["bounds"]
    assert (x0, y0, x1, y1) == pytest.approx((0.0, -1.0, 0.0, 3.0))
    assert [v for corner in d.bounds() for v in corner] == pytest.approx([0.0, -1.0, 16.0, 10.0])   # union of all three
    assert d.query_ids(kind="insert", within=((8, 9), (17, 11))) == [plain, big]


def test_block_rules_cycles_depth_references_and_in_use():
    d = fresh()
    d.define_block("a", [{"kind": "line", "geometry": {"start": [0, 0], "end": [1, 0]}}])
    d.define_block("b", [{"kind": "insert", "geometry": {"block": "a", "position": [0, 0]}}])
    revision = d.revision
    with pytest.raises(ValidationError):                                                   # a -> b -> a
        d.define_block("a", [{"kind": "insert", "geometry": {"block": "b", "position": [0, 0]}}], replace=True)
    with pytest.raises(ValidationError):
        d.define_block("self", [{"kind": "insert", "geometry": {"block": "self", "position": [0, 0]}}])
    assert d.revision == revision and d.get_block("a")["entities"][0]["kind"] == "line"
    with pytest.raises(ReferenceError):
        d.add_insert("missing", (0, 0))
    with pytest.raises(ReferenceError):
        d.define_block("c", [{"kind": "insert", "geometry": {"block": "ghost", "position": [0, 0]}}])
    for bad in (0, -1):
        with pytest.raises(GeometryValidationError):
            d.add_insert("a", (0, 0), scale=bad)
    with pytest.raises(ValidationError):
        d.define_block("a", [{"kind": "point", "geometry": {"position": [0, 0]}}])        # exists
    with pytest.raises(ValidationError):
        d.define_block("*anon", [{"kind": "point", "geometry": {"position": [0, 0]}}])
    with pytest.raises(ValidationError):
        d.define_block("empty", [])
    d.add_insert("b", (0, 0))
    with pytest.raises(ValidationError):
        d.remove_block("b")                                                                # inserted
    with pytest.raises(ValidationError):
        d.remove_block("a")                                                                # used by b
    # depth limit: a chain of 9 nested levels is refused
    d2 = fresh()
    d2.define_block("l0", [{"kind": "point", "geometry": {"position": [0, 0]}}])
    for i in range(1, 8):
        d2.define_block("l%d" % i, [{"kind": "insert", "geometry": {"block": "l%d" % (i - 1), "position": [0, 0]}}])
    with pytest.raises(SpatialFormatError):
        d2.define_block("l8", [{"kind": "insert", "geometry": {"block": "l7", "position": [0, 0]}}])
        d2.define_block("l9", [{"kind": "insert", "geometry": {"block": "l8", "position": [0, 0]}}])


def test_rename_redefine_and_remove_blocks():
    d = fresh()
    d.define_block("b", [{"kind": "line", "geometry": {"start": [0, 0], "end": [2, 0]}}])
    ins = d.add_insert("b", (0, 0))
    assert d.rename_block("b", "box") == "box"
    assert d.get(ins)["geometry"]["block"] == "box" and [x["name"] for x in d.blocks()] == ["box"]
    assert d.measure(ins)["bounds"] == ((0.0, 0.0), (2.0, 0.0))
    d.define_block("box", [{"kind": "line", "geometry": {"start": [0, 0], "end": [5, 0]}}], replace=True)
    assert d.measure(ins)["bounds"] == ((0.0, 0.0), (5.0, 0.0))                            # every instance follows
    assert d.blocks()[0]["used_by"] == 1
    d.remove([ins])
    d.remove_block("box")
    assert d.blocks() == []
    with pytest.raises(ReferenceError):
        d.get_block("box")


def test_make_block_and_explode_round_trip_the_drawing():
    d = fresh()
    a = d.add_line((2, 2), (6, 2))
    b = d.add_circle((4, 4), 1)
    before = sorted(d.measure(i)["bounds"] for i in (a, b))
    info = d.make_block("pair", [a, b])
    assert d.count() == 1 and d.get_block("pair")["base_point"] == [2.0, 2.0]
    insert = info["insert"]
    assert d.measure(insert)["bounds"] == ((2.0, 2.0), (6.0, 5.0))                        # the union of the originals
    new = d.explode([insert])
    assert d.count() == 2 and sorted(d.measure(i)["bounds"] for i in new) == before
    assert d.blocks()[0]["used_by"] == 0                                                   # the definition stays
    again = d.make_block("pair2", new, replace_with_insert=False, base_point=(0, 0))
    assert again == {"block": "pair2", "insert": None} and d.count() == 2


def test_explode_nested_and_layer_zero_inheritance():
    d = fresh()
    d.add_layer("walls")
    d.define_block("leaf", [{"kind": "line", "geometry": {"start": [0, 0], "end": [1, 0]}},
                            {"kind": "line", "geometry": {"start": [0, 1], "end": [1, 1]}}])
    d.add_layer("fixed")
    d.define_block("branch", [{"kind": "insert", "geometry": {"block": "leaf", "position": [5, 0]}},
                              {"kind": "line", "geometry": {"start": [0, 0], "end": [0, 1]}, "layer": "fixed"}])
    ins = d.add_insert("branch", (0, 0), layer="walls")
    shallow = d.explode([ins])
    assert sorted(d.get(i)["kind"] for i in shallow) == ["insert", "line"]
    assert {d.get(i)["layer"] for i in shallow} == {"walls", "fixed"}                      # layer "0" floats, others stay
    deep = d.explode([i for i in shallow if d.get(i)["kind"] == "insert"])
    assert len(deep) == 2 and {d.get(i)["layer"] for i in deep} == {"walls"}
    with pytest.raises(ValidationError):
        d.explode([deep[0]])                                                               # a plain line
    with pytest.raises(ReferenceError):
        d.explode(["00000000-0000-4000-8000-000000000000"])


def test_batches_are_atomic_for_the_new_operations():
    d = fresh()
    d.add_line((0, 0), (1, 1))
    revision, count = d.revision, d.count()
    with pytest.raises(SpatialFormatError):
        d.apply([{"op": "define_linetype", "name": "x", "pattern": [1, -1]},
                 {"op": "define_block", "name": "b", "entities": [{"kind": "point", "geometry": {"position": [0, 0]}}]},
                 {"op": "add", "kind": "insert", "geometry": {"block": "missing", "position": [0, 0]}}])
    assert (d.revision, d.count(), d.linetypes(), d.blocks()) == (revision, count, [], [])
    d.apply([{"op": "define_linetype", "name": "x", "pattern": [1, -1]},
             {"op": "define_block", "name": "b", "entities": [{"kind": "point", "geometry": {"position": [0, 0]}}]},
             {"op": "add", "kind": "insert", "geometry": {"block": "b", "position": [3, 3]}}])
    assert d.revision == revision + 1 and d.count() == count + 1                           # one revision for the batch


def test_convert_unit_scales_every_length_in_the_new_content():
    d = fresh("mm")
    d.define_linetype("dashed", [6, -3])
    d.define_block("b", [{"kind": "line", "geometry": {"start": [0, 0], "end": [100, 0]}}], base_point=(50, 0))
    d.add_insert("b", (1000, 0), scale=2.0)
    d.add_text((500, 500), "T", 20)
    h = d.add_hatch([square(d, 100)], pattern="lines", spacing=10)
    dim = d.add_dimension_aligned((0, 0), (1000, 0), (0, -50), text_height=20, precision=3)
    d.convert_unit("m")
    assert d.unit == "m"
    block = d.get_block("b")
    assert block["base_point"] == pytest.approx([0.05, 0.0]) and block["entities"][0]["geometry"]["end"] == pytest.approx([0.1, 0.0])
    assert d.linetypes()[0]["pattern"] == pytest.approx([0.006, -0.003])
    insert = d.get(d.query_ids(kind="insert")[0])["geometry"]
    assert insert["position"] == pytest.approx([1.0, 0.0]) and insert["scale"] == 2.0     # scale is dimensionless
    assert d.get(d.query_ids(kind="text")[0])["geometry"]["height"] == pytest.approx(0.02)
    assert d.get(h)["geometry"]["spacing"] == pytest.approx(0.01)
    assert d.dimension_value(dim) == pytest.approx(1.0) and d.get(dim)["geometry"]["text_height"] == pytest.approx(0.02)
    assert d.validate().ok


def test_new_content_survives_save_and_open_with_declared_features(tmp_path):
    d = fresh()
    d.define_linetype("dashed", [6, -3])
    d.add_text((0, 0), "x", 1)
    d.define_block("b", [{"kind": "point", "geometry": {"position": [0, 0]}}])
    d.add_insert("b", (1, 1))
    d.save(tmp_path / "a.sdrw")
    info = inspect_package(tmp_path / "a.sdrw")
    assert info.format_version == "0.2.0" and info.schema == "slai.sdrw/0.2"
    assert {"annotations", "blocks", "linetypes", "drawing-primitives"} <= set(info.requires)
    with SDRWDocument.open(tmp_path / "a.sdrw") as back:
        assert back.source_version == "0.2.0"
        assert back.count() == 2 and back.linetypes() == d.linetypes() and back.blocks() == d.blocks()
    plain = fresh()
    plain.add_line((0, 0), (1, 1))
    plain.save(tmp_path / "plain.sdrw")
    assert inspect_package(tmp_path / "plain.sdrw").requires == ("drawing-primitives",)   # no new features claimed


# --------------------------------------------------------------------------- migration 0.1.0 -> 0.2.0
def legacy_package(path, format_id, document, units, assets=None):
    """Write a genuine 0.1.0 package, byte-for-byte the way the 0.1 writer did."""
    payload = "drawing.json" if format_id == "slai.sdrw" else "scene.json"
    members = {payload: canonical_json(document)}
    members.update(assets or {})
    manifest = {"format_id": format_id, "format_version": "0.1.0", "schema": format_id + "/0.1",
                "spatial_dimension": 2 if format_id == "slai.sdrw" else 3,
                "document_kind": "drawing" if format_id == "slai.sdrw" else "model",
                "document_id": document["document_id"], "revision": document["revision"], "units": units,
                "requires": (["drawing-primitives"] if format_id == "slai.sdrw"
                             else ["scene-graph", "analytic-primitives", "triangle-mesh"]),
                "parts": [{"path": n, "size": len(r), "sha256": digest(r)} for n, r in sorted(members.items())]}
    if format_id == "slai.sgeo":
        manifest.update({"axis_up": "+Z", "handedness": "right"})
    mime = b"application/x-slai-sdrw+zip" if format_id == "slai.sdrw" else b"application/x-slai-sgeo+zip"
    with ZipFile(path, "w", allowZip64=False) as z:
        for name, raw, comp in [("mimetype", mime, ZIP_STORED), ("manifest.json", canonical_json(manifest), ZIP_DEFLATED)] + \
                [(n, r, ZIP_DEFLATED) for n, r in sorted(members.items())]:
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = comp
            info.external_attr = 0o100644 << 16
            z.writestr(info, raw)
    return manifest


def legacy_drawing_dict():
    d = fresh()
    d.add_layer("walls")
    d.add_line((0, 0), (10, 0), layer="walls", style={"stroke": "#333333", "stroke_width": 0.5})
    d.add_circle((5, 5), 2)
    body = d.model.as_dict()
    body.pop("linetypes"), body.pop("blocks")                     # exactly what a 0.1 writer produced
    return body


def test_legacy_drawing_opens_upgrades_and_saves_as_current(tmp_path):
    body = legacy_drawing_dict()
    legacy_package(tmp_path / "old.sdrw", "slai.sdrw", body, "mm")
    info = inspect_package(tmp_path / "old.sdrw")
    assert (info.format_version, info.schema) == ("0.1.0", "slai.sdrw/0.1")
    with SDRWDocument.open(tmp_path / "old.sdrw") as old:
        assert old.source_version == "0.1.0" and old.count() == 2 and old.linetypes() == [] and old.blocks() == []
        assert old.validate().ok
        assert old.get(old.query_ids(kind="line")[0])["style"] == {"stroke": "#333333", "stroke_width": 0.5}
    with SDRWDocument.open(tmp_path / "old.sdrw", mode="write") as editable:
        editable.define_linetype("dashed", [2, -1])
        editable.add_text((1, 1), "migrated", 1)
        editable.save(tmp_path / "new.sdrw")
    with SDRWDocument.open(tmp_path / "new.sdrw") as new:
        assert new.source_version == "0.2.0" and new.count() == 3 and new.linetypes()[0]["name"] == "dashed"
    with ZipFile(tmp_path / "new.sdrw") as z:
        assert json.loads(z.read("manifest.json"))["format_version"] == "0.2.0"
    with ZipFile(tmp_path / "old.sdrw") as z:                      # the original file is never rewritten
        assert json.loads(z.read("manifest.json"))["format_version"] == "0.1.0"


def test_legacy_scene_opens_and_upgrades(tmp_path):
    s = SGEODocument.new("S", unit="m")
    s.add_box(width=1, depth=2, height=3, name="B")
    body = s.model.as_dict()
    body.pop("materials"), body.pop("images")
    legacy_package(tmp_path / "old.sgeo", "slai.sgeo", body, "m")
    with SGEODocument.open(tmp_path / "old.sgeo") as old:
        assert old.source_version == "0.1.0" and old.validate().ok
        assert old.count(type_name="box") == 1
    with SGEODocument.open(tmp_path / "old.sgeo", mode="write") as editable:
        editable.save(tmp_path / "new.sgeo")
    with SGEODocument.open(tmp_path / "new.sgeo") as new:
        assert new.source_version == "0.2.0"


def test_a_0_1_file_may_not_contain_newer_content(tmp_path):
    body = legacy_drawing_dict()
    body["entities"].append({"id": "11111111-1111-4111-8111-111111111111", "kind": "text", "layer": "0",
                             "geometry": {"position": [0, 0], "text": "sneaky", "height": 1}})
    legacy_package(tmp_path / "bad.sdrw", "slai.sdrw", body, "mm")
    with pytest.raises(SpatialFormatError):
        SDRWDocument.open(tmp_path / "bad.sdrw").close()
    styled = legacy_drawing_dict()
    styled["entities"][0]["style"] = {"linetype": "dashed"}
    legacy_package(tmp_path / "bad2.sdrw", "slai.sdrw", styled, "mm")
    with pytest.raises(SpatialFormatError):
        SDRWDocument.open(tmp_path / "bad2.sdrw").close()


def test_unknown_versions_and_features_are_refused(tmp_path):
    d = fresh()
    d.add_line((0, 0), (1, 1))
    d.save(tmp_path / "ok.sdrw")
    for label, edit in (("future", lambda m: m.update(format_version="0.9.0", schema="slai.sdrw/0.9")),
                        ("feature", lambda m: m["requires"].append("quantum-geometry"))):
        with ZipFile(tmp_path / "ok.sdrw") as src:
            members = {n: src.read(n) for n in src.namelist()}
        manifest = json.loads(members["manifest.json"])
        edit(manifest)
        members["manifest.json"] = canonical_json(manifest)
        with ZipFile(tmp_path / (label + ".sdrw"), "w") as z:
            for name in ("mimetype", "manifest.json", "drawing.json"):
                info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_STORED if name == "mimetype" else ZIP_DEFLATED
                z.writestr(info, members[name])
        with pytest.raises((UnsupportedFeatureError, MigrationError, ValidationError, SpatialFormatError)):
            SDRWDocument.open(tmp_path / (label + ".sdrw")).close()


# --------------------------------------------------------------------------- advisory report
def test_validation_report_covers_linetypes_blocks_and_references():
    d = fresh()
    d.define_linetype("dashed", [6, -3])
    d.define_linetype("unused", [1, -1])
    d.define_block("b", [{"kind": "line", "geometry": {"start": [0, 0], "end": [1, 0]}}])
    d.define_block("spare", [{"kind": "point", "geometry": {"position": [0, 0]}}])
    d.add_layer("A", linetype="dashed")
    d.add_insert("b", (0, 0))
    report = d.validate()
    assert report.ok
    advisories = {(i.code, i.context.get("block") or i.context.get("linetype")) for i in report.issues
                  if i.severity == "info"}
    assert ("block.unused", "spare") in advisories and ("linetype.unused", "unused") in advisories
    assert {"linetypes", "blocks"} <= set(report.checks)
    # break the model behind the API's back: the report lists every problem instead of raising
    model = d.model
    model.layers[1]["linetype"] = "gone"
    model.entities.append({"id": "22222222-2222-4222-8222-222222222222", "kind": "insert", "layer": "0",
                           "geometry": {"block": "ghost", "position": [0, 0]}})
    model.blocks[0]["entities"].append({"id": model.entities[0]["id"], "kind": "point", "layer": "0",
                                        "geometry": {"position": [0, 0]}})
    from src.utils.spatial_formats import check_drawing
    broken = check_drawing(model)
    assert not broken.ok
    messages = " | ".join(i.message for i in broken.issues if i.severity == "error")
    assert "unknown linetype" in messages and "unknown block" in messages and "duplicate object identifier" in messages
