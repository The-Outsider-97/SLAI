"""SGEO 0.2: the native GLB codec (materials, textures, UVs, ids), 0.1.0 -> 0.2.0 migration, CLI and reports."""
from __future__ import annotations

import json
import struct
import zlib
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from src.utils.sgeo import SGEODocument
from src.utils.spatial_formats import cli, inspect_package
from src.utils.spatial_formats.adapters import exchange
from src.utils.spatial_formats.common.errors import (
    ResourceLimitError, SpatialFormatError, UnsupportedFeatureError, ValidationError)
from src.utils.spatial_formats.common.helpers import canonical_json, digest

TRIANGLE = ([(0, 0, 0), (10, 0, 0), (0, 10, 0)], [[0, 1, 2]])


def png(width=4, height=4, color=(200, 30, 30)) -> bytes:
    raw = b"".join(b"\x00" + bytes(color) * width for _ in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def rich_document() -> SGEODocument:
    d = SGEODocument.new("glb", unit="mm")
    image = d.add_image(png(), name="red.png")
    mat = d.add_material(
        "red", base_color=[1, 0.2, 0.2, 1], roughness=0.3, metallic=0.1, emissive=[0.1, 0, 0],
        alpha_mode="mask", alpha_cutoff=0.4, double_sided=True,
        textures={"base_color": {"image": image, "wrap_u": "clamp", "wrap_v": "mirror", "filter": "nearest"},
                  "normal": {"image": image, "scale": 0.5}, "occlusion": {"image": image, "strength": 0.7}})
    group = d.add_group("group", position=(1000, 0, 0))
    d.add_brep_box(width=100, depth=200, height=300, material=mat, name="bbox", parent=group)
    d.add_sphere(50, position=(0, 500, 0), material=mat, name="ball")
    d.add_mesh(*TRIANGLE, uvs=[(0, 0), (1, 0), (0, 1)], name="tri")
    return d


# --------------------------------------------------------------------------- GLB parsing helpers
def split_glb(data):
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length]), data[20 + length:]


def join_glb(model, rest):
    raw = json.dumps(model, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 4)
    body = struct.pack("<I4s", len(raw), b"JSON") + raw + rest
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


# --------------------------------------------------------------------------- GLB round trip
def test_glb_round_trip_keeps_materials_textures_uvs_and_ids():
    d = rich_document()
    data, report = d.to_glb()
    assert data[:4] == b"glTF" and report.accuracy_class == "approximate"
    assert any("tessellated" in note for note in report.approximations)
    back, back_report = SGEODocument.from_glb(data)
    assert back_report.dropped == []
    assert back.inspect()["material_count"] == 1 and back.inspect()["image_count"] == 1
    assert sorted(back.query_ids()) == sorted(d.query_ids())                 # identity travels in extras
    material = back.materials()[0]
    original = d.materials()[0]
    for key in ("base_color", "metallic", "roughness", "emissive"):
        assert material[key] == pytest.approx(original[key], rel=1e-6)
    assert (material["alpha_mode"], material["double_sided"]) == ("mask", True)
    assert material["alpha_cutoff"] == pytest.approx(0.4)
    slots = material["textures"]
    assert slots["base_color"]["wrap_u"] == "clamp" and slots["base_color"]["wrap_v"] == "mirror"
    assert slots["base_color"]["filter"] == "nearest"
    assert slots["normal"]["scale"] == pytest.approx(0.5) and slots["occlusion"]["strength"] == pytest.approx(0.7)
    assert back.image_bytes(back.images()[0]["id"]) == png()
    assert back.images()[0]["use_count"] == 3
    assert {o["name"] for o in back.find() if back.material_of(o["id"]) is not None} == {"bbox", "ball"}
    assert back.validate().ok


def test_glb_export_is_deterministic():
    d = rich_document()
    assert d.to_glb()[0] == d.to_glb()[0]


def test_glb_converts_units_flips_v_and_changes_up_axis():
    d = rich_document()
    data, _ = d.to_glb()
    model, _ = split_glb(data)
    assert model["asset"]["version"] == "2.0"
    back, _ = SGEODocument.from_glb(data)
    tri = back.get(back.query_ids(name="tri")[0])["geometry"]
    assert tri["uvs"] == [[0, 0], [1, 0], [0, 1]]                              # glTF v-down is undone on import
    # the document is in mm, GLB is in metres: the 100 x 200 x 300 mm box becomes 0.1 x 0.2 x 0.3 m
    box = back.query_ids(name="bbox")[0]
    lo, hi = back.world_bounds(box)
    size = [h - l for l, h in zip(lo, hi)]
    assert size == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    assert lo[0] == pytest.approx(1.0 - 0.05, abs=1e-6)                        # the group sits at x = 1000 mm = 1 m
    assert back.model.unit == "m"


def test_glb_without_uvs_or_materials_stays_minimal():
    d = SGEODocument.new("plain", unit="m")
    d.add_box(width=1, depth=1, height=1, name="b")
    data, _ = d.to_glb()
    model, _ = split_glb(data)
    assert "materials" not in model and "images" not in model and "textures" not in model
    back, _ = SGEODocument.from_glb(data)
    assert back.count() == 1 and back.materials() == []


def test_a_foreign_glb_without_slai_ids_gets_fresh_ids():
    d = SGEODocument.new("x", unit="m")
    d.add_box(width=1, depth=1, height=1, name="b")
    data, _ = d.to_glb()
    model, rest = split_glb(data)
    for node in model["nodes"]:
        node.pop("extras", None)
    back, _ = SGEODocument.from_glb(join_glb(model, rest))
    assert back.count() == 1 and back.query_ids()[0] != d.query_ids()[0]


@pytest.mark.parametrize("label, edit, error", [
    ("required extension", lambda m: m.__setitem__("extensionsRequired", ["KHR_draco_mesh_compression"]), UnsupportedFeatureError),
    ("optional extension", lambda m: m.__setitem__("extensionsUsed", ["KHR_materials_unlit"]), UnsupportedFeatureError),
    ("animation", lambda m: m.__setitem__("animations", [{"channels": [], "samplers": []}]), UnsupportedFeatureError),
    ("skin", lambda m: m.__setitem__("skins", [{"joints": [0]}]), UnsupportedFeatureError),
    ("external buffer", lambda m: m["buffers"][0].__setitem__("uri", "http://example.invalid/x.bin"), UnsupportedFeatureError),
    ("line primitives", lambda m: m["meshes"][0]["primitives"][0].__setitem__("mode", 1), UnsupportedFeatureError),
    ("second uv set", lambda m: m["meshes"][0]["primitives"][0]["attributes"].__setitem__(
        "TEXCOORD_1", m["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]), UnsupportedFeatureError),
    ("sheared node", lambda m: m["nodes"][0].__setitem__("matrix", [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), UnsupportedFeatureError),
    ("huge accessor", lambda m: m["accessors"][0].__setitem__("count", 10 ** 6), ResourceLimitError),
    ("wrong version", lambda m: m["asset"].__setitem__("version", "1.0"), ValidationError),
    ("dangling material", lambda m: m["meshes"][0]["primitives"][0].__setitem__("material", 9), ValidationError),
    ("dangling texture", lambda m: m["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"].__setitem__("index", 9), ValidationError),
])
def test_glb_features_the_codec_cannot_honour_are_refused(label, edit, error):
    d = SGEODocument.new("x", unit="mm")
    image = d.add_image(png())
    mat = d.add_material("m", textures={"base_color": {"image": image}})
    d.add_mesh(*TRIANGLE, uvs=[(0, 0), (1, 0), (0, 1)], material=mat, name="tri")
    model, rest = split_glb(d.to_glb()[0])
    edit(model)
    with pytest.raises(error):
        SGEODocument.from_glb(join_glb(model, rest))


@pytest.mark.parametrize("cut", [0, 4, 30, -4])
def test_truncated_glb_is_rejected(cut):
    data, _ = rich_document().to_glb()
    with pytest.raises(SpatialFormatError):
        SGEODocument.from_glb(data[:cut] if cut >= 0 else data[:cut])


def test_corrupt_embedded_texture_is_rejected():
    data, _ = rich_document().to_glb()
    index = data.rindex(b"\x89PNG")
    broken = bytearray(data)
    broken[index + 20] ^= 0xFF                                                   # inside the IHDR checksum area
    with pytest.raises(SpatialFormatError):
        SGEODocument.from_glb(bytes(broken))


def test_glb_export_via_file_helpers_and_cli(tmp_path, capsys):
    d = rich_document()
    d.save(tmp_path / "s.sgeo")
    assert cli.main(["convert", str(tmp_path / "s.sgeo"), str(tmp_path / "s.glb")]) == 0
    capsys.readouterr()
    assert cli.main(["convert", str(tmp_path / "s.glb"), str(tmp_path / "back.sgeo")]) == 0
    capsys.readouterr()
    with SGEODocument.open(tmp_path / "back.sgeo") as back:
        assert back.inspect()["material_count"] == 1 and back.inspect()["image_count"] == 1
    assert cli.main(["inspect", str(tmp_path / "s.sgeo")]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["format_version"] == "0.2.0"
    assert {"materials", "textures", "brep"} <= set(info["requires"])
    assert info["summary"]["material_count"] == 1
    assert cli.main(["capabilities"]) == 0
    caps = json.loads(capsys.readouterr().out)
    assert "glb" in caps["formats"]


# --------------------------------------------------------------------------- migration 0.1.0 -> 0.2.0
def legacy_package(path, document, assets=None):
    """A genuine 0.1.0 SGEO package, assembled the way the 0.1 writer did."""
    members = {"scene.json": canonical_json(document)}
    members.update(assets or {})
    manifest = {"format_id": "slai.sgeo", "format_version": "0.1.0", "schema": "slai.sgeo/0.1",
                "spatial_dimension": 3, "document_kind": "model", "document_id": document["document_id"],
                "revision": document["revision"], "units": "m", "axis_up": "+Z", "handedness": "right",
                "requires": ["scene-graph", "analytic-primitives", "triangle-mesh"],
                "parts": [{"path": n, "size": len(r), "sha256": digest(r)} for n, r in sorted(members.items())]}
    with ZipFile(path, "w", allowZip64=False) as z:
        for name, raw, comp in [("mimetype", b"application/x-slai-sgeo+zip", ZIP_STORED),
                                ("manifest.json", canonical_json(manifest), ZIP_DEFLATED)] + \
                [(n, r, ZIP_DEFLATED) for n, r in sorted(members.items())]:
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = comp
            info.external_attr = 0o100644 << 16
            z.writestr(info, raw)
    return manifest


def legacy_body():
    d = SGEODocument.new("old", unit="m")
    d.add_box(width=1, depth=2, height=3, name="box")
    d.add_sphere(1, position=(5, 0, 0), name="ball")
    body = d.model.as_dict()
    body.pop("materials"), body.pop("images")                                  # a 0.1 writer knew neither
    return body


def test_legacy_scene_upgrades_in_memory_and_saves_as_current(tmp_path):
    legacy_package(tmp_path / "old.sgeo", legacy_body())
    info = inspect_package(tmp_path / "old.sgeo")
    assert (info.format_version, info.schema) == ("0.1.0", "slai.sgeo/0.1")
    with SGEODocument.open(tmp_path / "old.sgeo") as old:
        assert old.source_version == "0.1.0" and old.validate().ok
        assert old.materials() == [] and old.images() == [] and old.count() == 2
    with SGEODocument.open(tmp_path / "old.sgeo", mode="write") as editable:
        image = editable.add_image(png())
        mat = editable.add_material("m", textures={"base_color": {"image": image}})
        editable.add_brep_box(width=1, depth=1, height=1, material=mat)
        editable.save(tmp_path / "new.sgeo")
    with SGEODocument.open(tmp_path / "new.sgeo") as new:
        assert new.source_version == "0.2.0" and new.count() == 3 and new.validate().ok
    with ZipFile(tmp_path / "new.sgeo") as z:
        assert json.loads(z.read("manifest.json"))["format_version"] == "0.2.0"
    with ZipFile(tmp_path / "old.sgeo") as z:                                   # the original is never rewritten
        assert json.loads(z.read("manifest.json"))["format_version"] == "0.1.0"


def test_legacy_scene_with_binary_meshes_upgrades(tmp_path):
    from src.utils.spatial_formats.container.scene_buffers import externalize_scene
    d = SGEODocument.new("old", unit="m")
    d.add_mesh(*TRIANGLE, name="tri")
    body = d.model.as_dict()
    body.pop("materials"), body.pop("images")
    external, assets = externalize_scene(body)
    legacy_package(tmp_path / "old.sgeo", external, assets)
    with SGEODocument.open(tmp_path / "old.sgeo") as old:
        mesh = old.get(old.query_ids(name="tri")[0])["geometry"]
        assert mesh["kind"] == "mesh" and "uvs" not in mesh and len(mesh["vertices"]) == 3


def object_record(**fields):
    return {"id": "11111111-1111-4111-8111-111111111111", "name": "x", "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            **fields}


@pytest.mark.parametrize("label, edit", [
    ("brep", lambda b: b["objects"].append(object_record(geometry={"kind": "brep"}))),
    ("nurbs curve", lambda b: b["objects"].append(object_record(geometry={
        "kind": "nurbs_curve", "control_points": [[0, 0, 0], [1, 1, 0], [2, 0, 0]], "degree": 2,
        "knots": [0, 0, 0, 1, 1, 1]}))),
    ("mesh uvs", lambda b: b["objects"].append(object_record(geometry={
        "kind": "mesh", "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 2]],
        "uvs": [[0, 0], [1, 0], [0, 1]]}))),
    ("material assignment", lambda b: b["objects"][0].__setitem__("material", "22222222-2222-4222-8222-222222222222")),
    ("material registry", lambda b: b.__setitem__("materials", [{"id": "22222222-2222-4222-8222-222222222222"}])),
])
def test_a_0_1_file_may_not_contain_newer_content(tmp_path, label, edit):
    body = legacy_body()
    edit(body)
    legacy_package(tmp_path / "bad.sgeo", body)
    with pytest.raises(SpatialFormatError):
        SGEODocument.open(tmp_path / "bad.sgeo").close()


def test_current_files_reject_unknown_features_and_versions(tmp_path):
    d = SGEODocument.new("x")
    d.add_box(width=1, depth=1, height=1)
    d.save(tmp_path / "ok.sgeo")
    with ZipFile(tmp_path / "ok.sgeo") as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]

    def variant(name, edit):
        with ZipFile(tmp_path / name, "w") as z:
            for info, raw in members:
                if info.filename == "manifest.json":
                    manifest = json.loads(raw)
                    edit(manifest)
                    raw = canonical_json(manifest)
                z.writestr(info, raw)
    variant("future.sgeo", lambda m: m.update(format_version="0.9.0", schema="slai.sgeo/0.9"))
    variant("feature.sgeo", lambda m: m["requires"].append("quantum-geometry"))
    for name in ("future.sgeo", "feature.sgeo"):
        with pytest.raises(SpatialFormatError):
            SGEODocument.open(tmp_path / name).close()


def test_requirements_follow_the_content():
    d = SGEODocument.new("x")
    d.add_box(width=1, depth=1, height=1)
    assert d.manifest()["requires"] == ["scene-graph", "analytic-primitives", "triangle-mesh"]
    d.add_nurbs_curve([(0, 0, 0), (1, 1, 0), (2, 0, 0)], degree=2)
    assert "nurbs" in d.manifest()["requires"] and "brep" not in d.manifest()["requires"]
    d.add_brep_sphere(1)
    assert "brep" in d.manifest()["requires"] and "materials" not in d.manifest()["requires"]
    d.add_material("m")
    assert "materials" in d.manifest()["requires"] and "textures" not in d.manifest()["requires"]
    d.add_image(png())
    assert "textures" in d.manifest()["requires"]


# --------------------------------------------------------------------------- limits and hostile input
def test_scene_binary_resources_are_bounded():
    d = SGEODocument.new("x")
    with pytest.raises(ResourceLimitError):
        for i in range(1001):
            d.add_image(png(color=(i % 256, i // 256, 7)))                      # more than the 1000 image cap


def test_exchange_functions_accept_scenes_directly():
    d = rich_document()
    data, report = exchange.scene_to_glb(d.snapshot())
    scene, back_report = exchange.glb_to_scene(data)
    assert len(scene.materials) == 1 and len(scene.images) == 1 and report.source_digest
    stl, stl_report = exchange.scene_to_stl(d.snapshot())
    assert len(stl) > 84 and stl[:5] != b"solid"                               # binary STL
    assert any("material" in item.lower() or "texture" in item.lower() for item in stl_report.dropped)
