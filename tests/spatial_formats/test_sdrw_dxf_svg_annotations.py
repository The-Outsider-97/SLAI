"""DXF and SVG interchange for SDRW 0.2 content: text, dimensions, hatches, blocks and linetypes."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from src.utils.sdrw import SDRWDocument
from src.utils.spatial_formats.common.errors import UnsupportedFeatureError, ValidationError
from src.utils.spatial_formats.sdrw import annotations as ann

SVG_NS = "{http://www.w3.org/2000/svg}"


def sample() -> SDRWDocument:
    d = SDRWDocument.new("sample", unit="mm")
    d.define_linetype("dashed", [6, -3])
    d.define_linetype("dashdot", [6, -2, 0, -2], description="dash dot")
    d.add_layer("L", stroke="#ff0000", stroke_width=0.5, linetype="dashed")
    d.add_layer("M")
    d.add_line((0, 0), (10, 0), layer="L", style={"stroke": "#00ff00", "linetype": "dashdot", "linetype_scale": 2.0})
    d.add_text((5, 5), "Hello", 3, rotation=0.5, halign="center", valign="middle", font="Arial")
    d.add_text((5, 15), "Two\nlines", 3, halign="right", valign="baseline", line_spacing=1.5)
    d.add_text((0, 30), "wrap me please this is long", 2, width=12, valign="top")
    sq = d.add_polyline([(0, 0), (20, 0), (20, 20), (0, 20)], closed=True)
    circle = d.add_circle((10, 10), 4)
    d.add_hatch([sq, circle], pattern="lines", spacing=3, angle=0.3, style={"stroke": "#0000ff"})
    d.add_hatch([{"kind": "path", "segments": [
        {"kind": "line", "start": [30, 0], "end": [40, 0]},
        {"kind": "arc", "center": [40, 5], "radius": 5, "start_angle": -math.pi / 2, "end_angle": math.pi / 2},
        {"kind": "line", "start": [40, 10], "end": [30, 10]},
        {"kind": "line", "start": [30, 10], "end": [30, 0]}]}], style={"fill": "#00ff00"})
    d.add_hatch([[(50, 0), (60, 0), (55, 8)]], pattern="cross", spacing=2)
    d.add_dimension_linear((0, 0), (20, 0), (0, -5), text="<> mm?", suffix="mm", precision=1, text_position=(10, -8))
    d.add_dimension_aligned((0, 0), (30, 40), (-6, 4))
    d.add_dimension_angular((0, 0), (10, 0), (0, 10), (5, 5))
    d.add_dimension_radial((10, 10), (14, 10))
    d.add_dimension_diameter((10, 10), (14, 10))
    d.add_dimension_linear((0, 0), (20, 10), (0, -5), angle=math.pi / 2, text="")
    d.define_block("sym", [{"kind": "line", "geometry": {"start": [0, 0], "end": [1, 1]}},
                           {"kind": "text", "geometry": {"position": [0, 2], "text": "T", "height": 1}}],
                   base_point=(0.5, 0.5))
    d.add_insert("sym", (100, 0), rotation=math.pi / 2, scale=2.0, layer="M")
    return d


def by_kind(document, kind):
    return [document.get(i) for i in document.query_ids(kind=kind)]


# ------------------------------------------------------------------------------------------- DXF
def test_dxf_round_trip_of_every_new_kind(tmp_path):
    pytest.importorskip("ezdxf")
    original = sample()
    report = original.export_dxf(tmp_path / "s.dxf")
    assert not [x for x in report.dropped if x != "document revision"] and not report.approximations
    back, incoming = SDRWDocument.import_dxf(tmp_path / "s.dxf")
    assert back.unit == "mm" and not incoming.approximations      # arrow blocks / dimension styles are not user content
    assert back.inspect()["entity_kinds"] == original.inspect()["entity_kinds"]

    # linetypes and layer defaults
    assert back.linetypes() == original.linetypes()
    layer = next(l for l in back.layers() if l["name"] == "L")
    assert (layer["stroke"], layer["stroke_width"], layer["linetype"]) == ("#ff0000", 0.5, "dashed")
    line = by_kind(back, "line")[0]
    assert line["layer"] == "L" and line["style"] == {"stroke": "#00ff00", "linetype": "dashdot", "linetype_scale": 2.0}

    # text: single line is exact, multi-line baseline text keeps its place (boxes agree)
    texts = {t["geometry"]["text"]: t["geometry"] for t in by_kind(back, "text")}
    single = texts["Hello"]
    assert single == {"position": [5.0, 5.0], "text": "Hello", "height": 3.0, "rotation": 0.5, "halign": "center",
                      "valign": "middle", "font": "Arial"}
    wrapped = texts["wrap me please this is long"]
    assert wrapped["width"] == 12.0 and wrapped["valign"] == "top" and wrapped["line_spacing"] == 1.0
    multi = texts["Two\nlines"]
    source = next(t for t in by_kind(original, "text") if t["geometry"]["text"] == "Two\nlines")["geometry"]
    for a, b in zip(ann.text_corners(multi), ann.text_corners(source)):
        assert a == pytest.approx(b)
    assert multi["line_spacing"] == 1.5

    # hatches: loops, pattern, spacing, angle, colours
    hatches = by_kind(back, "hatch")
    assert len(hatches) == 3
    lined = next(h for h in hatches if h["geometry"]["pattern"] == "lines")
    assert lined["style"] == {"stroke": "#0000ff"}
    assert lined["geometry"]["spacing"] == pytest.approx(3.0) and lined["geometry"]["angle"] == pytest.approx(0.3)
    assert [loop["kind"] for loop in lined["geometry"]["boundary"]] == ["polygon", "circle"]
    assert ann.hatch_area(lined["geometry"]) == pytest.approx(400 - math.pi * 16)
    solid = next(h for h in hatches if h["geometry"]["pattern"] == "solid")
    assert solid["style"] == {"fill": "#00ff00"}
    source_solid = next(h for h in by_kind(original, "hatch") if h["geometry"]["pattern"] == "solid")
    assert ann.hatch_area(solid["geometry"]) == pytest.approx(ann.hatch_area(source_solid["geometry"]))
    cross = next(h for h in hatches if h["geometry"]["pattern"] == "cross")
    assert cross["geometry"]["spacing"] == pytest.approx(2.0)

    # dimensions keep type, measured value and shown text
    def shown(document):
        return sorted((e["geometry"]["dim_type"] == "angular", round(ann.dimension_value(e["geometry"]), 9),
                       ann.dimension_text(e["geometry"])) for e in by_kind(document, "dimension"))
    assert len(by_kind(back, "dimension")) == 6
    original_shown = shown(original)
    assert shown(back) == original_shown
    types = sorted(e["geometry"]["dim_type"] for e in by_kind(back, "dimension"))
    assert types == ["aligned", "aligned", "angular", "diameter", "linear", "radial"]   # a parallel 'linear' is 'aligned'
    placed = [e["geometry"] for e in by_kind(back, "dimension") if "text_position" in e["geometry"]]
    assert len(placed) == 1 and placed[0]["text_position"] == [10.0, -8.0]

    # blocks and inserts
    assert [b["name"] for b in back.blocks()] == ["sym"]
    block = back.get_block("sym")
    assert block["base_point"] == [0.5, 0.5] and [e["kind"] for e in block["entities"]] == ["line", "text"]
    insert = by_kind(back, "insert")[0]
    assert insert["layer"] == "M"
    assert insert["geometry"] == {"block": "sym", "position": [100.0, 0.0], "rotation": pytest.approx(math.pi / 2),
                                  "scale": 2.0}
    assert back.validate().ok


def test_dxf_dimension_text_equals_sdrw_text(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    document = sample()
    document.export_dxf(tmp_path / "s.dxf")
    doc = ezdxf.readfile(str(tmp_path / "s.dxf"))
    rendered = []
    for dim in doc.modelspace().query("DIMENSION"):
        block = doc.blocks.get(dim.dxf.geometry)
        rendered.append("".join(e.plain_text() for e in block if e.dxftype() == "MTEXT").strip())
    expected = sorted(ann.dimension_text(e["geometry"]).strip() for e in by_kind(document, "dimension")
                      if ann.dimension_text(e["geometry"]))

    def norm(text):
        return text.replace("\N{LATIN CAPITAL LETTER O WITH STROKE}", "\N{DIAMETER SIGN}").replace("%%c", "\N{DIAMETER SIGN}")
    assert sorted(norm(t) for t in rendered if t.strip()) == expected, (rendered, expected)


def test_dxf_hatch_edge_chain_is_continuous_for_reversed_arcs(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    document = SDRWDocument.new("h", unit="mm")
    # segments listed so the second arc must be walked backwards
    document.add_hatch([{"kind": "path", "segments": [
        {"kind": "arc", "center": [0, 0], "radius": 5, "start_angle": 0.0, "end_angle": math.pi},
        {"kind": "line", "start": [5, 0], "end": [-5, 0]}]}])
    document.export_dxf(tmp_path / "h.dxf")
    doc = ezdxf.readfile(str(tmp_path / "h.dxf"))
    hatch = next(iter(doc.modelspace().query("HATCH")))
    edges = hatch.paths[0].edges
    points = []
    for edge in edges:
        a = edge.real_start_point if type(edge).__name__ == "ArcEdge" else edge.start
        b = edge.real_end_point if type(edge).__name__ == "ArcEdge" else edge.end
        points.append((tuple(a), tuple(b)))
    for (a0, a1), (b0, b1) in zip(points, points[1:] + points[:1]):
        assert math.hypot(a1[0] - b0[0], a1[1] - b0[1]) < 1e-9, points
    back, _ = SDRWDocument.import_dxf(tmp_path / "h.dxf")
    assert ann.hatch_area(by_kind(back, "hatch")[0]["geometry"]) == pytest.approx(math.pi * 25 / 2)


def test_dxf_names_are_made_legal_and_reported(tmp_path):
    pytest.importorskip("ezdxf")
    document = SDRWDocument.new("n", unit="mm")
    document.add_layer("A/B")
    document.add_layer("a:b")
    document.define_linetype("my line?", [2, -1])
    document.add_line((0, 0), (1, 1), layer="A/B")
    report = document.export_dxf(tmp_path / "n.dxf")
    assert any("'A/B' written as 'A_B'" in line for line in report.approximations)
    assert any("'my line?'" in line for line in report.approximations)
    back, _ = SDRWDocument.import_dxf(tmp_path / "n.dxf")
    # DXF names are case-insensitive: 'a:b' becomes 'a_b' which collides with 'A_B', so it gets a suffix
    assert {l["name"] for l in back.layers()} == {"0", "A_B", "a_b_2"}
    assert [x["name"] for x in back.linetypes()] == ["my line_"]
    with pytest.raises(UnsupportedFeatureError):
        document.export_dxf(tmp_path / "strict.dxf", policy="strict")
    assert not (tmp_path / "strict.dxf").exists()


def test_dxf_export_policy_sees_lineweight_snapping(tmp_path):
    pytest.importorskip("ezdxf")
    document = SDRWDocument.new("w", unit="mm")
    document.add_line((0, 0), (1, 0), style={"stroke_width": 0.5})
    document.export_dxf(tmp_path / "ok.dxf", policy="allow_approximation")      # 0.5 mm is a real DXF lineweight
    document.add_line((0, 1), (1, 1), style={"stroke_width": 0.33})
    with pytest.raises(UnsupportedFeatureError):
        document.export_dxf(tmp_path / "bad.dxf", policy="strict")
    report = document.export_dxf(tmp_path / "ok2.dxf")
    assert any("lineweight" in line for line in report.approximations)


def foreign(tmp_path, build):
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new("R2010", setup=["styles", "dimstyles"])
    doc.units = 4
    build(doc, doc.modelspace(), ezdxf)
    path = tmp_path / "foreign.dxf"
    doc.saveas(str(path))
    return path


def test_dxf_unsupported_entities_fail_closed_or_are_skipped_with_a_report(tmp_path):
    def build(doc, msp, ezdxf):
        msp.add_line((0, 0), (1, 1))
        msp.add_spline([(0, 0), (1, 2), (3, 3), (5, 0)])
        msp.add_ellipse((0, 0), major_axis=(2, 0), ratio=0.5)
    path = foreign(tmp_path, build)
    with pytest.raises(UnsupportedFeatureError, match="unsupported DXF entity: SPLINE"):
        SDRWDocument.import_dxf(path)
    document, report = SDRWDocument.import_dxf(path, skip_unsupported=True)
    assert document.count() == 1
    assert any("SPLINE" in line for line in report.dropped) and any("ELLIPSE" in line for line in report.dropped)
    with pytest.raises(UnsupportedFeatureError):                         # losses violate a strict policy
        SDRWDocument.import_dxf(path, skip_unsupported=True, policy="strict")


def test_dxf_foreign_text_colours_lineweights_and_inserts(tmp_path):
    def build(doc, msp, ezdxf):
        from ezdxf.enums import TextEntityAlignment
        t = msp.add_text("45%%d  %%c10", height=2.5, dxfattribs={"color": 1, "lineweight": 35})
        t.set_placement((1, 2), align=TextEntityAlignment.TOP_RIGHT)
        msp.add_mtext("{\\fArial|b1;Bold} plain\\Pline two", dxfattribs={"char_height": 2, "insert": (0, 10),
                                                                       "attachment_point": 5})
        b = doc.blocks.new("box", base_point=(0, 0))
        b.add_line((0, 0), (1, 0))
        b.add_line((1, 0), (1, 1))
        msp.add_blockref("box", (5, 5), dxfattribs={"xscale": 2, "yscale": 2, "rotation": 30})
        msp.add_blockref("box", (9, 9), dxfattribs={"xscale": -1, "yscale": -1})              # a half turn
        msp.add_blockref("box", (20, 0), dxfattribs={"xscale": 2, "yscale": 1})              # non-uniform -> exploded
        msp.add_blockref("box", (30, 0), dxfattribs={"column_count": 3, "row_count": 1, "column_spacing": 4})
        msp.add_line((0, 0), (1, 1), dxfattribs={"true_color": 0x336699, "linetype": "BYBLOCK"})
    path = foreign(tmp_path, build)
    document, report = SDRWDocument.import_dxf(path)
    texts = {t["geometry"]["text"]: t for t in by_kind(document, "text")}
    degree = texts["45\N{DEGREE SIGN}  \N{DIAMETER SIGN}10"]
    assert degree["geometry"]["halign"] == "right" and degree["geometry"]["valign"] == "top"
    assert degree["style"] == {"stroke": "#ff0000", "stroke_width": 0.35}
    assert "Bold plain\nline two" in texts and texts["Bold plain\nline two"]["geometry"]["valign"] == "middle"
    assert any("inline MTEXT formatting" in line for line in report.approximations)
    inserts = by_kind(document, "insert")
    assert len(inserts) == 2 + 3                                         # two plain, three array cells
    assert sorted(i["geometry"]["scale"] for i in inserts if i["geometry"]["position"][1] != 0)[:2] == [1.0, 2.0]
    half_turn = next(i for i in inserts if i["geometry"]["position"] == [9.0, 9.0])
    assert half_turn["geometry"]["rotation"] == pytest.approx(math.pi) and half_turn["geometry"]["scale"] == 1.0
    assert sorted(i["geometry"]["position"][0] for i in inserts if i["geometry"]["position"][1] == 0) == [30.0, 34.0, 38.0]
    assert len(by_kind(document, "line")) == 3                           # two lines of the exploded insert + the stray one
    assert any("exploded into plain entities" in line for line in report.approximations)
    assert any("MINSERT" in line for line in report.approximations)
    assert any("BYBLOCK linetypes" in line for line in report.approximations)
    stray = next(e for e in by_kind(document, "line") if e.get("style", {}).get("stroke") == "#336699")
    assert "linetype" not in stray["style"]
    with pytest.raises(UnsupportedFeatureError):
        SDRWDocument.import_dxf(path, policy="strict")


def test_dxf_foreign_hatches(tmp_path):
    def build(doc, msp, ezdxf):
        h = msp.add_hatch(color=3)
        h.paths.add_polyline_path([(0, 0, 0), (10, 0, 1), (10, 10, 0), (0, 10, 0)], is_closed=True)   # one bulged edge
        h.set_solid_fill(color=3, style=0)
        e = msp.add_hatch()
        e.paths.add_edge_path().add_ellipse((0, 20), (4, 0), 0.5)
        e.set_solid_fill(style=0)
        a = msp.add_hatch()
        a.paths.add_polyline_path([(0, 40), (10, 40), (10, 50), (0, 50)], is_closed=True)
        a.set_pattern_fill("ANSI31", scale=2.0, angle=10)
        g = msp.add_hatch()
        g.paths.add_polyline_path([(30, 40), (40, 40), (40, 50)], is_closed=True)
        g.set_gradient((255, 0, 0), (0, 0, 255))
    path = foreign(tmp_path, build)
    document, report = SDRWDocument.import_dxf(path)
    hatches = by_kind(document, "hatch")
    assert len(hatches) == 4
    bulged = hatches[0]["geometry"]
    assert bulged["boundary"][0]["kind"] == "path" and hatches[0]["style"] == {"fill": "#00ff00"}
    assert ann.hatch_area(bulged) == pytest.approx(100 + math.pi * 25 / 2)
    ellipse = hatches[1]["geometry"]
    assert ellipse["boundary"][0]["kind"] == "polygon"
    assert ann.hatch_area(ellipse) == pytest.approx(math.pi * 4 * 2, rel=1e-2)
    ansi = hatches[2]["geometry"]
    assert ansi["pattern"] == "lines" and ansi["spacing"] == pytest.approx(3.175 * 2, rel=1e-3)
    assert ansi["angle"] == pytest.approx(math.radians(55), abs=1e-6)
    assert any("flattened" in line for line in report.approximations)
    assert any("gradient" in line for line in report.approximations)


def test_dxf_foreign_dimensions_and_unsupported_types(tmp_path):
    def build(doc, msp, ezdxf):
        msp.add_linear_dim(base=(0, -3), p1=(0, 0), p2=(10, 5), angle=0).render()               # genuinely linear
        msp.add_radius_dim(center=(0, 0), mpoint=(4, 0)).render()
        msp.add_angular_dim_3p(base=(2, 2), center=(0, 0), p1=(5, 0), p2=(0, 5)).render()
        msp.add_angular_dim_2l(base=(2, 2), line1=((0, 0), (5, 0)), line2=((0, 0), (0, 5))).render()
    path = foreign(tmp_path, build)
    document, report = SDRWDocument.import_dxf(path)
    dims = by_kind(document, "dimension")
    assert sorted(d["geometry"]["dim_type"] for d in dims) == ["angular", "linear", "radial"]
    linear = next(d for d in dims if d["geometry"]["dim_type"] == "linear")["geometry"]
    assert ann.dimension_value(linear) == pytest.approx(10.0) and linear["angle"] == 0.0
    # EZDXF's stock style multiplies by DIMLFAC=100; the value is the geometry, and the report says so
    assert any("DIMLFAC" in line for line in report.approximations)
    assert any("DIMENSION (type 2) exploded" in line for line in report.approximations)
    assert len(by_kind(document, "line")) > 0 and len(by_kind(document, "text")) > 0   # the exploded 2-line dimension


def test_dxf_capabilities_list_the_new_entities():
    from src.utils.spatial_formats.adapters import dxf
    entities = dxf.capabilities().entities
    assert {"TEXT", "MTEXT", "DIMENSION", "HATCH", "INSERT"} <= set(entities)


# ------------------------------------------------------------------------------------------- SVG
def test_svg_export_of_new_kinds_is_well_formed_and_reported():
    document = sample()
    markup, report = document.to_svg()
    root = ET.fromstring(markup)
    tags = [child.tag.replace(SVG_NS, "") for child in root]
    assert tags.count("text") == 3 + 5 + 1          # texts, dimension labels (one is blank) and the block's text
    assert any(child.get("data-insert") for child in root)
    assert all(child.get("data-id") for child in root if not child.get("data-insert") and not child.get("data-dimension"))
    hatch = next(c for c in root if c.get("data-kind") == "hatch" and c.get("fill-rule") == "evenodd")
    assert hatch.get("fill") == "#00ff00" and hatch.get("d").startswith("M ")
    dashed = [c for c in root if c.get("stroke-dasharray")]
    assert dashed and all(float(x) > 0 for c in dashed for x in c.get("stroke-dasharray").split())
    assert any("dimension(s) flattened" in line for line in report.dropped)
    assert any("block structure" in line for line in report.dropped)
    assert any("export-only" in line for line in report.warnings)
    with pytest.raises(UnsupportedFeatureError):
        document.to_svg(policy="strict")


def test_svg_text_transform_places_the_anchor_in_the_drawing_frame():
    document = SDRWDocument.new("t", unit="mm")
    document.add_text((10, 20), "A", 2, rotation=math.pi / 2)
    root = ET.fromstring(document.to_svg()[0])
    node = next(iter(root))
    a, b, c, d, e, f = (float(v) for v in node.get("transform")[len("matrix("):-1].split())
    assert (e, f) == (10.0, 20.0)
    assert (a, b) == pytest.approx((0.0, 1.0), abs=1e-12) and (c, d) == pytest.approx((1.0, 0.0), abs=1e-12)
    assert node.get("font-size") == "2.0" and node.text == "A"


def test_svg_plain_export_is_unchanged_and_round_trips():
    document = SDRWDocument.new("p", unit="mm")
    document.add_line((0, 0), (1, 1))
    markup, report = document.to_svg()
    assert "stroke" not in markup and "<text" not in markup
    back, _ = SDRWDocument.from_svg(markup)
    assert back.count() == 1


def test_svg_with_text_hatch_dimension_or_insert_is_refused_on_reimport():
    markup, _ = sample().to_svg()
    with pytest.raises(UnsupportedFeatureError):
        SDRWDocument.from_svg(markup)


def test_svg_styles_are_written_for_styled_entities():
    document = SDRWDocument.new("s", unit="mm")
    document.define_linetype("d", [4, -2])
    document.add_layer("L", stroke="#123456", linetype="d")
    document.add_line((0, 0), (5, 0), layer="L", style={"stroke_width": 0.4, "linetype_scale": 2.0})
    node = next(iter(ET.fromstring(document.to_svg()[0])))
    assert node.get("stroke") == "#123456" and node.get("stroke-width") == "0.4"
    assert node.get("stroke-dasharray") == "8.0 4.0"


def test_svg_expands_nested_inserts_with_their_transform():
    document = SDRWDocument.new("e", unit="mm")
    document.define_block("leaf", [{"kind": "line", "geometry": {"start": [0, 0], "end": [1, 0]}}])
    document.define_block("branch", [{"kind": "insert", "geometry": {"block": "leaf", "position": [2, 0],
                                                                      "rotation": 0.0, "scale": 1.0}}])
    document.add_insert("branch", (10, 10), scale=3.0)
    node = next(iter(ET.fromstring(document.to_svg()[0])))
    assert node.tag == SVG_NS + "line"
    assert (float(node.get("x1")), float(node.get("y1")), float(node.get("x2")), float(node.get("y2"))) == \
        pytest.approx((16.0, 10.0, 19.0, 10.0))


def test_invalid_hatch_loops_are_reported_not_imported(tmp_path):
    def build(doc, msp, ezdxf):
        h = msp.add_hatch()
        h.paths.add_polyline_path([(0, 0), (1, 0)], is_closed=True)          # two points: no area
        h.set_solid_fill(style=0)
    path = foreign(tmp_path, build)
    document, report = SDRWDocument.import_dxf(path)
    assert document.count() == 0
    assert any("hatches without a usable boundary" in line for line in report.dropped)
