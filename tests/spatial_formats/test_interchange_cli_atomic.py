"""Interchange (SVG/DXF/GLB/STL), meshing, atomic writes, the CLI and performance guards."""
import json
import os
import struct
import time
from math import pi

import pytest

from src.utils.sdrw import SDRWDocument
from src.utils.sgeo import SGEODocument
from src.utils.spatial_formats import cli
from src.utils.spatial_formats.common import ConflictError, ValidationError
from src.utils.spatial_formats.common.atomic import atomic_output, file_sha256, write_bytes_atomic
from src.utils.spatial_formats.common.errors import GeometryValidationError, UnsupportedFeatureError
from src.utils.spatial_formats.sgeo.meshing import extract_meshes, flatten_scene, merge_meshes, transform_mesh, weld_vertices


def drawing():
    d = SDRWDocument.new("x", unit="mm")
    d.add_line((0, 0), (10, 0))
    d.add_polyline([(0, 0), (5, 5), (10, 0)], closed=False)
    d.add_circle((3, 3), 2)
    d.add_arc((0, 0), 5, 0.25, 1.25)
    d.add_point((1, 1))
    return d


def _floats(value):
    if isinstance(value, dict):
        return {k: _floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_floats(v) for v in value]
    return float(value) if isinstance(value, (int, float)) else value


def geometry_of(document):
    return sorted((e["kind"], json.dumps(_floats(e["geometry"]), sort_keys=True)) for e in document.as_dict()["entities"])


# --------------------------------------------------------------------------- atomic writes
def test_atomic_write_never_clobbers_and_leaves_no_temporaries(tmp_path):
    target = tmp_path / "f.bin"
    digest = write_bytes_atomic(target, b"one")
    assert digest == file_sha256(target)
    with pytest.raises(ConflictError):
        write_bytes_atomic(target, b"two")
    assert target.read_bytes() == b"one"
    write_bytes_atomic(target, b"two", overwrite=True)
    assert target.read_bytes() == b"two"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.bin"]


def test_failed_atomic_write_preserves_previous_content(tmp_path, monkeypatch):
    target = tmp_path / "f.bin"
    write_bytes_atomic(target, b"keep")
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        write_bytes_atomic(target, b"new", overwrite=True)
    assert target.read_bytes() == b"keep" and [p.name for p in tmp_path.iterdir()] == ["f.bin"]


def test_atomic_output_context_discards_on_error(tmp_path):
    target = tmp_path / "g.bin"
    with pytest.raises(RuntimeError):
        with atomic_output(target, overwrite=False) as temporary:
            with open(temporary, "wb") as handle:
                handle.write(b"half")
            raise RuntimeError("abort")
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- SVG
def test_svg_round_trip_preserves_geometry_and_optionally_ids(tmp_path):
    d = drawing()
    markup, report = d.to_svg()
    assert report.accuracy_class == "approximate" and report.adapter == "svg-subset"
    again, _ = SDRWDocument.from_svg(markup)
    assert geometry_of(again) == geometry_of(d)
    assert set(e["id"] for e in again.as_dict()["entities"]).isdisjoint(e["id"] for e in d.as_dict()["entities"])
    kept, _ = SDRWDocument.from_svg(markup, preserve_ids=True)
    assert {e["id"] for e in kept.as_dict()["entities"]} == {e["id"] for e in d.as_dict()["entities"]}


def test_svg_full_sweep_arc_becomes_a_circle_and_policy_is_enforced(tmp_path):
    d = SDRWDocument.new("s")
    d.add_arc((0, 0), 2, 0, 2 * pi)
    markup, _ = d.to_svg()
    assert "<circle" in markup
    with pytest.raises(UnsupportedFeatureError):
        d.to_svg(policy="strict")                          # revision / identity cannot survive SVG
    d.export_svg(tmp_path / "a.svg")
    with pytest.raises(ConflictError):
        d.export_svg(tmp_path / "a.svg")
    d.export_svg(tmp_path / "a.svg", overwrite=True)
    imported, _ = SDRWDocument.import_svg(tmp_path / "a.svg")
    assert imported.count(kind="circle") == 1


@pytest.mark.parametrize("markup", [
    '<svg xmlns="http://www.w3.org/2000/svg"><script>x</script></svg>',
    '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="0" cy="0" r="1" style="fill:red"/></svg>',
    '<!DOCTYPE svg [<!ENTITY a "b">]><svg xmlns="http://www.w3.org/2000/svg"/>',
    '<svg xmlns="http://www.w3.org/2000/svg"><path d="M 0 0 C 1 1 2 2 3 3"/></svg>',
])
def test_svg_import_fails_closed(markup):
    with pytest.raises((UnsupportedFeatureError, ValidationError)):
        SDRWDocument.from_svg(markup)


# --------------------------------------------------------------------------- DXF (ezdxf)
def test_dxf_round_trip_arcs_circles_and_policies(tmp_path):
    pytest.importorskip("ezdxf")
    d = drawing()
    report = d.export_dxf(tmp_path / "d.dxf")
    assert report.adapter == "dxf-ezdxf"
    again, back = SDRWDocument.import_dxf(tmp_path / "d.dxf")
    assert again.unit == "mm" and geometry_of(again) == geometry_of(d)
    with pytest.raises(ConflictError):
        d.export_dxf(tmp_path / "d.dxf")
    full = SDRWDocument.new("f")
    full.add_arc((0, 0), 1, 0, 2 * pi)
    full.export_dxf(tmp_path / "full.dxf")
    assert SDRWDocument.import_dxf(tmp_path / "full.dxf")[0].count(kind="circle") == 1
    with pytest.raises(GeometryValidationError):                 # zero-sweep arcs cannot exist, so DXF never sees one
        SDRWDocument.new("f").add_arc((0, 0), 1, 1.0, 1.0)


# --------------------------------------------------------------------------- STL / meshing / GLB
def scene():
    s = SGEODocument.new("S", unit="m")
    s.add_box(width=1, depth=2, height=3, name="B", position=(0, 0, 1.5))
    s.add_sphere(0.5, center=(3, 0, 0), name="Ball")
    return s


def test_meshing_world_frame_weld_and_winding():
    s = scene()
    parts = extract_meshes(s.snapshot(), radial_segments=12, latitude_bands=6)
    assert len(parts.meshes) == 2 and set(parts.tessellated.values()) == {"box", "sphere"}
    meshes = list(parts.meshes.values())
    merged = merge_meshes(meshes)
    welded, dropped = weld_vertices(merged)
    assert dropped >= 0 and len(welded["vertices"]) <= len(merged["vertices"])
    mirrored = transform_mesh(meshes[0], [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    area = lambda m: sum(  # signed volume must stay positive after a mirror: winding was fixed
        sum(a * b for a, b in zip(m["vertices"][f[0]], _cross(m["vertices"][f[1]], m["vertices"][f[2]]))) / 6 for f in m["faces"])
    assert area(mirrored) > 0
    flat, summary = flatten_scene(s.snapshot())
    assert all(o["geometry"]["kind"] == "mesh" for o in flat.as_dict()["objects"] if o["geometry"])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def test_stl_export_import_and_policy(tmp_path):
    s = scene()
    data, report = s.to_stl()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + 50 * count and report.accuracy_class == "approximate"
    with pytest.raises(UnsupportedFeatureError):
        s.to_stl(policy="strict")
    s.export_stl(tmp_path / "s.stl")
    back, _ = SGEODocument.import_stl(tmp_path / "s.stl", unit="m")
    assert back.count() == 1 and back.unit == "m" and back.validate().ok
    only_box, _ = s.to_stl(object_ids=[s.query_ids(name="B")[0]])
    assert struct.unpack_from("<I", only_box, 80)[0] == 12
    with pytest.raises(ValidationError):
        SGEODocument.from_stl(b"short")


def test_glb_round_trip_keeps_hierarchy_names_and_up_axis(tmp_path):
    pytest.importorskip("trimesh")
    s = scene()
    group = s.add_group("Level 1", position=(0, 0, 10))
    box = s.query_ids(name="B")[0]
    s.reparent(box, group)
    s.export_glb(tmp_path / "s.glb")
    back, report = SGEODocument.import_glb(tmp_path / "s.glb")
    assert back.unit == "m" and back.validate().ok
    names = {back.get(i)["name"] for i in back.query_ids()}
    assert {"B", "Ball", "Level 1"} <= names
    (b2,) = back.query_ids(name="B")
    lo, hi = back.world_bounds(b2)
    assert lo[2] == pytest.approx(10.0, abs=1e-4) and hi[2] == pytest.approx(13.0, abs=1e-4)
    assert report.assumptions


# --------------------------------------------------------------------------- CLI
def run(capsys, *argv):
    code = cli.main(list(map(str, argv)))
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip().startswith("{") else out)


def test_cli_inspect_validate_and_capabilities(tmp_path, capsys):
    d = drawing()
    d.save(tmp_path / "d.sdrw")
    scene().save(tmp_path / "s.sgeo")
    code, info = run(capsys, "inspect", tmp_path / "d.sdrw", "--verify", "digests")
    assert code == 0 and info["format_id"] == "slai.sdrw" and info["summary"]["entity_count"] == 5
    code, report = run(capsys, "validate", tmp_path / "s.sgeo")
    assert code == 0 and report["ok"] and report["document_kind"] == "scene"
    code, caps = run(capsys, "capabilities")
    assert code == 0 and {"svg", "stl", "dxf", "glb", "ifc", "step", "dwg"} <= set(caps["formats"])


def test_cli_exit_codes_for_invalid_missing_and_usage(tmp_path, capsys):
    (tmp_path / "bad.sdrw").write_bytes(b"PK\x03\x04 nope")
    code, report = run(capsys, "validate", tmp_path / "bad.sdrw")
    assert code == 1 and not report["ok"] and report["issues"][0]["code"].startswith("spatial.")
    code, failure = run(capsys, "inspect", tmp_path / "missing.sdrw")
    assert code == 1 and failure["code"] == "os.error"
    assert cli.main(["nonsense"]) == 2
    assert cli.main(["validate"]) == 2


def test_cli_convert_both_directions_and_refusals(tmp_path, capsys):
    drawing().save(tmp_path / "d.sdrw")
    scene().save(tmp_path / "s.sgeo")
    code, out = run(capsys, "convert", tmp_path / "d.sdrw", tmp_path / "d.svg")
    assert code == 0 and out["report"]["adapter"] == "svg-subset"
    code, out = run(capsys, "convert", tmp_path / "d.svg", tmp_path / "back.sdrw")
    assert code == 0 and SDRWDocument.open(tmp_path / "back.sdrw").count() == 5
    code, out = run(capsys, "convert", tmp_path / "d.sdrw", tmp_path / "d.svg")
    assert code == 1 and out["code"] == "spatial.conflict"
    code, out = run(capsys, "convert", tmp_path / "d.sdrw", tmp_path / "d.svg", "--overwrite")
    assert code == 0
    code, out = run(capsys, "convert", tmp_path / "s.sgeo", tmp_path / "s.stl")
    assert code == 0
    code, out = run(capsys, "convert", tmp_path / "s.sgeo", tmp_path / "s.dxf")
    assert code == 1 and out["code"] == "spatial.validation"
    code, out = run(capsys, "convert", tmp_path / "d.sdrw", tmp_path / "d.dwg")
    assert code == 1 and "dwg-executable" in out["error"]
    code, out = run(capsys, "convert", tmp_path / "d.sdrw", tmp_path / "x.svg", "--policy", "strict")
    assert code == 1 and out["code"] == "spatial.unsupported"


# --------------------------------------------------------------------------- performance guards
def test_incremental_adds_and_batches_are_not_quadratic():
    d = SDRWDocument.new("perf")
    start = time.perf_counter()
    for i in range(3000):
        d.add_point((i, i))
    assert time.perf_counter() - start < 15
    batch = SDRWDocument.new("perf2")
    start = time.perf_counter()
    for chunk in range(5):                                  # one batch is capped at 1000 operations
        batch.apply([{"op": "add", "kind": "point", "geometry": {"position": [i, i]}}
                     for i in range(chunk * 1000, (chunk + 1) * 1000)])
    assert time.perf_counter() - start < 10 and batch.count() == 5000
    with pytest.raises(ValidationError):
        batch.apply([{"op": "add", "kind": "point", "geometry": {"position": [0, 0]}}] * 1001)


def test_large_scene_saves_and_reloads(tmp_path):
    s = SGEODocument.new("big")
    for chunk in range(3):
        s.apply([{"op": "add", "geometry": {"kind": "point", "position": [i, 0, 0]}}
                 for i in range(chunk * 1000, (chunk + 1) * 1000)])
    s.save(tmp_path / "big.sgeo")
    assert SGEODocument.open(tmp_path / "big.sgeo").count() == 3000
