"""SGEO 0.2 materials, embedded textures and UV coordinates: model, editing rules, container round trip."""
from __future__ import annotations

import json
import struct
import zlib
from zipfile import ZipFile

import pytest

from src.utils.sgeo import SGEODocument
from src.utils.spatial_formats import inspect_package
from src.utils.spatial_formats.common.errors import (
    GeometryValidationError, ReferenceError, ResourceLimitError, SpatialFormatError, ValidationError)
from src.utils.spatial_formats.sgeo.geometry import validate_geometry
from src.utils.spatial_formats.sgeo.materials import DEFAULT_MATERIAL, resolve_material

MISSING = "11111111-1111-4111-8111-111111111111"
TRIANGLE = {"vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "faces": [[0, 1, 2]]}


def png(width=4, height=4, color=(200, 30, 30), color_type=2) -> bytes:
    """A structurally valid, tiny PNG (RGB by default)."""
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = b"".join(b"\x00" + bytes(color[:channels] if channels <= 3 else color + (255,)) * width
                   for _ in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def jpeg(width=8, height=8) -> bytes:
    """The marker skeleton of a baseline JPEG: enough for structural inspection (it is never decoded)."""
    def segment(marker, body):
        return b"\xff" + bytes([marker]) + struct.pack(">H", len(body) + 2) + body

    frame = struct.pack(">BHHB", 8, height, width, 1) + bytes([1, 0x11, 0])
    scan = struct.pack(">B", 1) + bytes([1, 0x00]) + bytes([0, 63, 0])
    return b"\xff\xd8" + segment(0xC0, frame) + segment(0xDA, scan) + b"\x00\x00" + b"\xff\xd9"


def fresh(unit="m") -> SGEODocument:
    return SGEODocument.new("t", unit=unit)


# --------------------------------------------------------------------------- UV coordinates
def test_mesh_uvs_are_validated_and_survive_copies():
    ok = validate_geometry({"kind": "mesh", **TRIANGLE, "uvs": [[0, 0], [1, 0], [0, 2]]})
    assert ok["uvs"][2] == [0, 2]                                               # tiling beyond [0, 1] is legal
    for bad in ([[0, 0]], [[0, 0], [1, 0], [float("inf"), 1]], [[0, 0], [1, 0], [0, 1, 1]], "uvs",
                [[0, 0], [1, 0], [True, 1]]):
        with pytest.raises(SpatialFormatError):
            validate_geometry({"kind": "mesh", **TRIANGLE, "uvs": bad})


def test_primitives_tessellate_with_uvs_in_the_unit_square():
    d = fresh()
    for oid in (d.add_box(width=1, depth=2, height=3), d.add_cylinder(1, 2), d.add_sphere(1)):
        plain = d.tessellate(oid)
        mapped = d.tessellate(oid, uvs=True)
        assert "uvs" not in plain
        assert len(mapped["uvs"]) == len(mapped["vertices"])
        flat = [c for uv in mapped["uvs"] for c in uv]
        assert 0.0 <= min(flat) and max(flat) <= 1.0
    box = d.tessellate(d.query_ids(type_name="box")[0], uvs=True)
    assert len(box["vertices"]) == 24 and len(box["faces"]) == 12              # every face is its own unit square


def test_tessellation_with_uvs_keeps_the_same_surface():
    d = fresh()
    sphere = d.add_sphere(2)
    a, b = d.tessellate(sphere), d.tessellate(sphere, uvs=True)
    from src.utils.spatial_formats.sgeo.analysis import analyze_mesh
    area_a, area_b = analyze_mesh(a).surface_area, analyze_mesh(b).surface_area
    assert area_a == pytest.approx(area_b, rel=1e-9)


# --------------------------------------------------------------------------- images
def test_images_are_structurally_validated_deduplicated_and_bounded():
    d = fresh()
    first = d.add_image(png(), name="red.png")
    assert d.add_image(png()) == first and len(d.images()) == 1             # identical bytes are stored once
    jpg = d.add_image(jpeg(16, 8), name="photo.jpg")
    record = {r["id"]: r for r in d.images()}
    assert (record[first]["mime"], record[first]["width"], record[first]["height"]) == ("image/png", 4, 4)
    assert (record[jpg]["mime"], record[jpg]["width"], record[jpg]["height"]) == ("image/jpeg", 16, 8)
    assert d.image_bytes(first) == png()
    corrupt = bytearray(png())
    corrupt[-5] ^= 1
    for bad in (b"hello", b"\x89PNG\r\n\x1a\n\x00\x00", bytes(corrupt), b"", b"\xff\xd8\xff"):
        with pytest.raises(ValidationError):
            d.add_image(bad)
    with pytest.raises(ResourceLimitError):
        d.add_image(png(20000, 1))                                          # beyond 16384 pixels per side
    with pytest.raises(ResourceLimitError):
        d.add_image(png(10000, 10000))                                      # beyond the pixel budget
    with pytest.raises(ValidationError):
        d.add_image("not bytes")
    assert len(d.images()) == 2                                             # failures changed nothing


def test_image_file_helpers_round_trip(tmp_path):
    (tmp_path / "t.png").write_bytes(png())
    d = fresh()
    ident = d.add_image_file(tmp_path / "t.png")
    assert d.images()[0]["name"] == "t.png"
    digest = d.export_image(ident, tmp_path / "out.png")
    assert (tmp_path / "out.png").read_bytes() == png() and len(digest) == 64
    with pytest.raises(SpatialFormatError):
        d.export_image(ident, tmp_path / "out.png")                         # no silent overwrite
    d.export_image(ident, tmp_path / "out.png", overwrite=True)


# --------------------------------------------------------------------------- materials
def test_material_defaults_and_resolution():
    d = fresh()
    plain = d.add_material("plain")
    stored = d.get_material(plain)
    assert stored["base_color"] == [0.8, 0.8, 0.8, 1.0] and stored["alpha_mode"] == "opaque"
    assert "alpha_cutoff" not in stored or stored["alpha_cutoff"] == DEFAULT_MATERIAL["alpha_cutoff"]
    sparse = {"id": plain, "name": "x"}
    full = resolve_material(sparse)
    for key in DEFAULT_MATERIAL:
        assert key in full


@pytest.mark.parametrize("fields", [
    {"roughness": 2}, {"metallic": -0.1}, {"metallic": True}, {"base_color": [1, 1, 1]},
    {"base_color": [1, 1, 1, float("nan")]}, {"emissive": [0, 0]}, {"alpha_mode": "weird"},
    {"alpha_cutoff": 0.3}, {"double_sided": "yes"}, {"alpha_mode": "mask", "alpha_cutoff": 1.5},
])
def test_invalid_material_fields_are_refused(fields):
    d = fresh()
    with pytest.raises(SpatialFormatError):
        d.add_material("bad", **fields)
    assert d.materials() == []


def test_texture_slot_rules():
    d = fresh()
    image = d.add_image(png())
    good = {"base_color": {"image": image, "wrap_u": "clamp", "wrap_v": "mirror", "filter": "nearest"},
            "normal": {"image": image, "scale": 0.5}, "occlusion": {"image": image, "strength": 0.25}}
    mat = d.add_material("ok", textures=good)
    assert set(d.get_material(mat)["textures"]) == {"base_color", "normal", "occlusion"}
    bad_cases = [
        ({"bump": {"image": image}}, ValidationError),
        ({"base_color": {"image": image, "scale": 2}}, ValidationError),                 # scale is for normal only
        ({"normal": {"image": image, "strength": 2}}, ValidationError),
        ({"base_color": {"image": image, "wrap_u": "spin"}}, ValidationError),
        ({"base_color": {"image": image, "filter": "cubic"}}, ValidationError),
        ({"base_color": {"image": MISSING}}, ReferenceError),
        ({"base_color": {}}, SpatialFormatError),
    ]
    for textures, error in bad_cases:
        with pytest.raises(error):
            d.add_material("bad", textures=textures)
    d.set_texture(mat, "emissive", image)
    assert "emissive" in d.get_material(mat)["textures"]
    d.clear_texture(mat, "emissive")
    with pytest.raises(ReferenceError):
        d.clear_texture(mat, "emissive")
    with pytest.raises(ValidationError):
        d.set_texture(mat, "bump", image)


def test_material_assignment_rules():
    d = fresh()
    mat = d.add_material("m")
    group = d.add_group("g")
    box = d.add_box(width=1, depth=1, height=1)
    with pytest.raises(ValidationError):
        d.set_material(group, mat)                                           # no geometry, nothing to shade
    with pytest.raises(ReferenceError):
        d.set_material(box, MISSING)
    d.set_material(box, mat)
    assert d.material_of(box)["id"] == mat and d.materials()[0]["use_count"] == 1
    child = d.add_box(width=1, depth=1, height=1, parent=box)
    assert d.material_of(child) is None                                      # no inheritance
    with pytest.raises(ValidationError):
        d.remove_material(mat)                                               # still in use
    d.set_material(box, None)
    d.remove_material(mat)
    assert d.materials() == []


def test_image_in_use_cannot_be_removed():
    d = fresh()
    image = d.add_image(png())
    mat = d.add_material("m", textures={"base_color": {"image": image}})
    assert d.images()[0]["use_count"] == 1
    with pytest.raises(ValidationError):
        d.remove_image(image)
    d.clear_texture(mat, "base_color")
    d.remove_image(image)
    assert d.images() == []
    with pytest.raises(ReferenceError):
        d.image_bytes(image)


def test_material_edits_are_atomic_and_revisioned():
    d = fresh()
    mat = d.add_material("m", roughness=0.1)
    revision = d.revision
    d.update_material(mat, roughness=0.9, name="renamed", alpha_mode="mask")
    assert d.revision == revision + 1
    got = d.get_material(mat)
    assert (got["roughness"], got["name"], got["alpha_mode"]) == (0.9, "renamed", "mask")
    assert got["alpha_cutoff"] == 0.5
    revision = d.revision
    with pytest.raises(SpatialFormatError):
        d.update_material(mat, roughness=5)
    with pytest.raises(ValidationError):
        d.update_material(mat, shininess=1)
    assert d.revision == revision and d.get_material(mat)["roughness"] == 0.9
    d.update_material(mat, alpha_mode="opaque")
    assert "alpha_cutoff" not in d.get_material(mat)


def test_batched_operations_roll_back_together():
    d = fresh()
    mat = d.add_material("m")
    box = d.add_box(width=1, depth=1, height=1)
    before = d.content_digest()
    with pytest.raises(SpatialFormatError):
        d.apply([{"op": "set_material", "id": box, "value": mat},
                 {"op": "remove_material", "id": mat}])                      # second step is refused
    assert d.content_digest() == before and d.material_of(box) is None


# --------------------------------------------------------------------------- container round trip
def textured_document():
    d = fresh()
    image = d.add_image(png(), name="red.png")
    mat = d.add_material("brick", base_color=[1, 0.2, 0.2, 1], roughness=0.3, metallic=0.1,
                         textures={"base_color": {"image": image, "wrap_u": "clamp"}})
    tri = d.add_mesh(**TRIANGLE, uvs=[(0, 0), (1, 0), (0, 1)], material=mat, name="tri")
    box = d.add_box(width=1, depth=1, height=1, material=mat, name="box")
    return d, image, mat, tri, box


def test_textures_uvs_and_materials_survive_save_and_open(tmp_path):
    d, image, mat, tri, box = textured_document()
    d.save(tmp_path / "m.sgeo")
    with ZipFile(tmp_path / "m.sgeo") as z:
        names = z.namelist()
        texture_members = [n for n in names if n.startswith("textures/")]
        assert names[:2] == ["mimetype", "manifest.json"] and "scene.json" in names
        assert len(texture_members) == 1 and texture_members[0].endswith(".png")
        assert z.read(texture_members[0]) == png()                           # stored verbatim
        scene = json.loads(z.read("scene.json"))
        assert [rec["path"] for rec in scene["images"]] == texture_members      # records point at their member
        manifest = json.loads(z.read("manifest.json"))
    assert {"materials", "textures"} <= set(manifest["requires"])
    with SGEODocument.open(tmp_path / "m.sgeo") as back:
        assert back.image_bytes(image) == png()
        assert back.material_of(tri)["base_color"] == [1.0, 0.2, 0.2, 1.0]
        assert back.get(tri)["geometry"]["uvs"] == [[0, 0], [1, 0], [0, 1]]
        assert all("path" not in record for record in back.model.images)       # transport detail, not model state
        assert back.material_of(box)["textures"]["base_color"]["wrap_u"] == "clamp"
        assert back.validate().ok
        assert back.content_digest() == d.content_digest()


def test_saving_is_deterministic_and_inspect_reports_counts(tmp_path):
    d, *_ = textured_document()
    first = d.dumps()
    assert d.dumps() == first
    d.save(tmp_path / "a.sgeo")
    info = inspect_package(tmp_path / "a.sgeo")
    assert info.format_version == "0.2.0"
    summary = d.inspect()
    assert summary["material_count"] == 1 and summary["image_count"] == 1


def test_identical_textures_are_stored_once(tmp_path):
    d = fresh()
    one = d.add_image(png(color=(1, 2, 3)))
    other = d.add_image(png(color=(9, 9, 9)))
    d.add_material("a", textures={"base_color": {"image": one}, "emissive": {"image": one}})
    d.add_material("b", textures={"base_color": {"image": other}})
    d.save(tmp_path / "t.sgeo")
    with ZipFile(tmp_path / "t.sgeo") as z:
        assert len([n for n in z.namelist() if n.startswith("textures/")]) == 2


def test_a_tampered_texture_is_rejected_on_open(tmp_path):
    d, *_ = textured_document()
    d.save(tmp_path / "m.sgeo")
    with ZipFile(tmp_path / "m.sgeo") as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]
    with ZipFile(tmp_path / "bad.sgeo", "w") as z:
        for info, raw in members:
            if info.filename.startswith("textures/"):
                raw = raw[:-1] + bytes([raw[-1] ^ 1])
            z.writestr(info, raw)
    with pytest.raises(SpatialFormatError):
        SGEODocument.open(tmp_path / "bad.sgeo").close()


def test_a_missing_or_extra_texture_member_is_rejected(tmp_path):
    d, *_ = textured_document()
    d.save(tmp_path / "m.sgeo")
    with ZipFile(tmp_path / "m.sgeo") as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]
    with ZipFile(tmp_path / "missing.sgeo", "w") as z:
        for info, raw in members:
            if not info.filename.startswith("textures/"):
                z.writestr(info, raw)
    with pytest.raises(SpatialFormatError):
        SGEODocument.open(tmp_path / "missing.sgeo").close()
    with ZipFile(tmp_path / "extra.sgeo", "w") as z:
        for info, raw in members:
            z.writestr(info, raw)
        z.writestr("textures/sha256-" + "0" * 64 + ".png", png())
    with pytest.raises(SpatialFormatError):
        SGEODocument.open(tmp_path / "extra.sgeo").close()


def test_material_report_advisories():
    d = fresh()
    image = d.add_image(png())
    mat = d.add_material("textured", textures={"base_color": {"image": image}})
    d.add_material("idle")
    d.add_mesh(**TRIANGLE, material=mat, name="no-uvs")
    report = d.validate()
    assert report.ok
    codes = {i.code for i in report.issues}
    assert {"material.unused", "material.no_uvs"} <= codes
    d.add_image(png(color=(5, 5, 5)))
    assert "image.unused" in {i.code for i in d.validate().issues}


def test_geometry_kind_checks_reject_material_on_nothing():
    d = fresh()
    mat = d.add_material("m")
    with pytest.raises(ValidationError):
        d.add_group("g", material=mat)
    with pytest.raises(ReferenceError):
        d.add_box(width=1, depth=1, height=1, material=MISSING)
    assert len(d) == 0
    with pytest.raises(GeometryValidationError):
        d.add_mesh(**TRIANGLE, uvs=[(0, 0)])
