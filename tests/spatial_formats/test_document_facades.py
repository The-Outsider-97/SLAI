"""Lifecycle, concurrency, validation and agent-facing contracts of ``src.utils.sdrw`` / ``src.utils.sgeo``."""
import hashlib
import json
import re
from math import pi
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.utils import sdrw, sgeo
from src.utils.sdrw import SDRWDocument
from src.utils.sgeo import SGEODocument

PKG = Path(sdrw.__file__).parent / "spatial_formats"


# --------------------------------------------------------------------------- constants / layering
def test_facade_constants_match_the_container_contract():
    assert (sdrw.FORMAT_VERSION, sgeo.FORMAT_VERSION) == ("0.2.0", "0.2.0")
    assert (sdrw.SCHEMA, sgeo.SCHEMA) == ("slai.sdrw/0.2", "slai.sgeo/0.2")
    assert sdrw.MIMETYPE == "application/x-slai-sdrw+zip" and sgeo.MIMETYPE == "application/x-slai-sgeo+zip"
    assert (sdrw.FILE_EXTENSION, sgeo.FILE_EXTENSION) == (".sdrw", ".sgeo")
    assert sgeo.AXIS_UP == "+Z" and sgeo.HANDEDNESS == "right"
    for module in (sdrw, sgeo):
        assert module.__version__ == "0.2.0"
        for name in module.__all__:
            assert hasattr(module, name), name


def test_empty_documents_construct_like_the_original_stubs():
    assert len(SDRWDocument()) == 0 and SDRWDocument().validate().ok
    assert SGEODocument().count() == 0 and SGEODocument().validate().ok


def test_package_never_imports_the_facades():
    offenders = [str(p) for p in PKG.rglob("*.py") if re.search(r"from\s+(src\.utils|\.\.\.)\s*\.?(sdrw|sgeo)\s+import|import\s+src\.utils\.(sdrw|sgeo)\b", p.read_text())]
    assert not offenders, offenders


# --------------------------------------------------------------------------- SDRW lifecycle
def make_plan():
    plan = SDRWDocument.new("Plan A", unit="mm")
    wall = plan.add_line((0, 0), (1500, 0), layer="walls", create_layer=True)
    plan.attach_semantics(wall, {"category": "wall"}, namespace="aec")
    return plan, wall


def test_sdrw_save_open_round_trip_is_deterministic(tmp_path):
    plan, wall = make_plan()
    assert plan.dirty
    result = plan.save(tmp_path / "plan.sdrw")
    assert not plan.dirty and result.size == (tmp_path / "plan.sdrw").stat().st_size
    assert result.sha256 == hashlib.sha256((tmp_path / "plan.sdrw").read_bytes()).hexdigest()
    with SDRWDocument.open(tmp_path / "plan.sdrw") as opened:
        assert opened.read_only and not opened.dirty
        assert opened.query_ids(kind="line", layer="walls") == [wall]
        assert opened.get(wall)["metadata"]["aec:category"] == "wall"
        assert opened.content_digest() == plan.content_digest()
    assert opened.closed
    plan.save(tmp_path / "again.sdrw")
    assert (tmp_path / "plan.sdrw").read_bytes() == (tmp_path / "again.sdrw").read_bytes()
    with ZipFile(tmp_path / "plan.sdrw") as archive:
        assert archive.namelist() == ["mimetype", "manifest.json", "drawing.json"]
        assert archive.read("mimetype") == b"application/x-slai-sdrw+zip"


def test_read_only_documents_refuse_every_edit(tmp_path):
    plan, wall = make_plan()
    plan.save(tmp_path / "p.sdrw")
    reader = SDRWDocument.open(tmp_path / "p.sdrw")
    for attempt in (lambda: reader.add_point((1, 1)), lambda: reader.remove(wall), lambda: reader.rename("x"),
                    lambda: reader.apply([]), lambda: reader.move([wall], 1, 1)):
        with pytest.raises(sdrw.ReadOnlyDocumentError) as caught:
            attempt()
        assert caught.value.code == "spatial.read_only" and isinstance(caught.value, sdrw.ConflictError)
    assert reader.revision == SDRWDocument.open(tmp_path / "p.sdrw").revision
    reader.close()
    assert reader.closed
    with pytest.raises(sdrw.ReadOnlyDocumentError):
        reader.validate()
    writer = SDRWDocument.open(tmp_path / "p.sdrw", mode="write")
    writer.close()
    with pytest.raises(sdrw.ReadOnlyDocumentError):
        writer.add_point((0, 0))                 # a closed document accepts no edits ...
    with pytest.raises(sdrw.ReadOnlyDocumentError):
        writer.save(tmp_path / "other.sdrw")     # ... and cannot be saved


def test_overwrite_and_compare_and_swap_protect_against_lost_updates(tmp_path):
    plan, wall = make_plan()
    target = tmp_path / "p.sdrw"
    plan.save(target)
    with pytest.raises(sdrw.ConflictError):
        SDRWDocument.new("other").save(target)                       # never clobbers an existing file by default
    first = SDRWDocument.open(target, mode="write")
    second = SDRWDocument.open(target, mode="write")
    first.add_point((1, 2))
    first.save()                                                     # saves back to its own source
    second.add_point((3, 4))
    with pytest.raises(sdrw.ConflictError):
        second.save()                                                # file changed since second was opened
    second.save(expected_sha256=None, force=True)                    # explicit, visible override
    assert SDRWDocument.open(target).count(kind="point") == 1


def test_save_refuses_to_replace_an_unrelated_document_without_force(tmp_path):
    a, b = SDRWDocument.new("a"), SDRWDocument.new("b")
    a.add_point((0, 0))
    a.save(tmp_path / "x.sdrw")
    with pytest.raises(sdrw.ConflictError):
        b.save(tmp_path / "x.sdrw", overwrite=True)
    b.save(tmp_path / "x.sdrw", overwrite=True, force=True)
    (tmp_path / "junk.sdrw").write_bytes(b"not a zip")
    with pytest.raises(sdrw.ConflictError):
        b.save(tmp_path / "junk.sdrw", overwrite=True)


def test_expected_revision_rejects_stale_batches_atomically():
    plan, wall = make_plan()
    stale = plan.revision
    plan.add_point((1, 1))
    before = plan.content_digest()
    with pytest.raises(sdrw.ConflictError):
        plan.apply([{"op": "add", "kind": "point", "geometry": {"position": [9, 9]}}], expected_revision=stale)
    assert plan.content_digest() == before


def test_batch_is_one_revision_and_all_or_nothing():
    plan = SDRWDocument.new("b")
    start = plan.revision
    ids = plan.apply([{"op": "add", "kind": "point", "geometry": {"position": [i, i]}} for i in range(5)])
    assert len(ids) == 5 and plan.revision == start + 1
    digest = plan.content_digest()
    with pytest.raises(sdrw.SpatialFormatError):
        plan.apply([{"op": "add", "kind": "point", "geometry": {"position": [0, 0]}},
                    {"op": "remove", "id": "00000000-0000-4000-8000-000000000000"}])
    assert plan.content_digest() == digest and plan.revision == start + 1


def test_unknown_references_and_geometry_keys_are_rejected():
    plan = SDRWDocument.new("g")
    with pytest.raises(sdrw.SpatialReferenceError) as caught:
        plan.remove("00000000-0000-4000-8000-000000000000")
    assert caught.value.code.startswith("spatial.")
    with pytest.raises(sdrw.SpatialGeometryValidationError):
        plan.apply([{"op": "add", "kind": "line", "geometry": {"start": [0, 0], "end": [1, 1], "bogus": 1}}])
    with pytest.raises(sdrw.SpatialFormatError):
        plan.add_circle((0, 0), -1)
    with pytest.raises(sdrw.SpatialFormatError):
        plan.add_line((0, 0), (float("nan"), 1))
    assert len(plan) == 0


def test_metadata_keys_are_namespaced_and_aec_dot_is_hinted():
    plan, wall = make_plan()
    with pytest.raises(sdrw.ValidationError) as caught:
        plan.attach_semantics(wall, {"aec.category": "wall"})
    assert "aec:category" in str(caught.value)
    with pytest.raises(sdrw.ValidationError):
        plan.attach_semantics(wall, {"nonamespace": 1})
    plan.add_claim(wall, "is_load_bearing", evidence="survey-12", confidence=0.7)
    assert plan.get(wall)["metadata"]["slai.claims"][0]["predicate"] == "is_load_bearing"


def test_geometry_queries_and_measurement():
    plan = SDRWDocument.new("m", unit="mm")
    rect = plan.add_rectangle((0, 0), 4000, 3000)
    circle = plan.add_circle((0, 0), 500)
    assert plan.area(rect) == pytest.approx(12_000_000) and plan.area(circle) == pytest.approx(pi * 500 ** 2)
    assert plan.area(rect, unit="m") == pytest.approx(12.0)
    assert plan.bounds() == ((-500, -500), (4000, 3000))
    assert plan.convert_unit("m") == pytest.approx(0.001) and plan.unit == "m"
    assert plan.area(rect) == pytest.approx(12.0)
    assert plan.validate().ok


def test_transforms_and_duplicate_keep_ids_unique():
    plan = SDRWDocument.new("t")
    line = plan.add_line((0, 0), (1, 0))
    plan.rotate([line], 90, degrees=True)
    assert plan.get(line)["geometry"]["end"] == pytest.approx([0, 1], abs=1e-12)
    (copy_id,) = plan.duplicate([line], dx=5)
    assert copy_id != line and plan.count() == 2
    plan.move([copy_id], 1, 1)
    plan.scale([copy_id], 2)
    assert plan.validate().ok
    assert plan.copy().document_id != plan.document_id and plan.copy(fresh_id=False).document_id == plan.document_id


def test_layer_management_protects_content():
    plan = SDRWDocument.new("l")
    plan.add_layer("walls")
    line = plan.add_line((0, 0), (1, 1), layer="walls")
    with pytest.raises(sdrw.SpatialFormatError):
        plan.remove_layer("walls")          # in use
    plan.rename_layer("walls", "partitions")
    assert plan.get(line)["layer"] == "partitions"
    plan.set_layer_visibility("partitions", False)
    codes = {i.code for i in plan.validate().warnings}
    assert "layer.hidden_content" in codes
    plan.remove([line])
    plan.remove_layer("partitions")


def test_documents_can_be_opened_from_bytes_and_inspected(tmp_path):
    plan, wall = make_plan()
    again = SDRWDocument.from_bytes(plan.dumps())
    assert again.content_digest() == plan.content_digest()
    plan.save(tmp_path / "p.sdrw")
    info = sdrw.inspect(tmp_path / "p.sdrw")
    assert info.format_id == "slai.sdrw" and info.summary["entity_count"] == 1
    assert info.summary["layers"] == ["0", "walls"]
    assert sdrw.inspect(tmp_path / "p.sdrw", verify="digests").verified
    package = sdrw.read_package(tmp_path / "p.sdrw")
    assert package.sha256 == hashlib.sha256((tmp_path / "p.sdrw").read_bytes()).hexdigest()
    assert package.to_document(mode="write").content_digest() == plan.content_digest()


def test_facades_reject_the_other_format(tmp_path):
    SGEODocument.new("s").save(tmp_path / "s.sgeo")
    SDRWDocument.new("d").save(tmp_path / "d.sdrw")
    with pytest.raises(sdrw.SpatialFormatError):
        sdrw.inspect(tmp_path / "s.sgeo")
    with pytest.raises(sgeo.SpatialFormatError):
        sgeo.read_package(tmp_path / "d.sdrw")
    report = sgeo.validate(tmp_path / "d.sdrw")
    assert not report.ok and report.errors[0].code.startswith("spatial.")


def test_validate_reports_corruption_instead_of_raising(tmp_path):
    SDRWDocument.new("d").save(tmp_path / "d.sdrw")
    raw = bytearray((tmp_path / "d.sdrw").read_bytes())
    raw[-30] ^= 0xFF
    (tmp_path / "bad.sdrw").write_bytes(bytes(raw))
    report = sdrw.validate(tmp_path / "bad.sdrw")
    assert not report.ok
    with pytest.raises(sdrw.SpatialFormatError):
        report.raise_for_errors()
    with pytest.raises(FileNotFoundError):
        sdrw.validate(tmp_path / "missing.sdrw")


# --------------------------------------------------------------------------- SGEO lifecycle
def make_room():
    room = SGEODocument.new("Room", unit="m")
    wall = room.add_box(width=3.0, depth=0.2, height=2.7, name="Wall A", position=(0, 0, 1.35))
    room.attach_semantics(wall, {"category": "wall", "fire_rating": "unknown"}, namespace="aec")
    return room, wall


def test_sgeo_model_round_trip_with_binary_meshes(tmp_path):
    room, wall = make_room()
    mesh = room.add_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
                         name="Tetra")
    room.save(tmp_path / "r.sgeo")
    with ZipFile(tmp_path / "r.sgeo") as archive:
        names = archive.namelist()
        assert names[:2] == ["mimetype", "manifest.json"] and "scene.json" in names
        assert any(n.startswith("geometry/sha256-") for n in names)
    with SGEODocument.open(tmp_path / "r.sgeo") as opened:
        assert opened.get(mesh)["geometry"]["kind"] == "mesh"
        assert opened.query_ids(type_name="box") == [wall]
        assert opened.volume(mesh) == pytest.approx(1 / 6)
        assert opened.volume(wall) == pytest.approx(3.0 * 0.2 * 2.7)
        assert opened.content_digest() == room.content_digest()
        assert opened.inspect()["object_count"] == 2


def test_sgeo_hierarchy_queries_and_world_frames():
    room, wall = make_room()
    group = room.add_group("Level 1", position=(10, 0, 0))
    room.reparent(wall, group)                             # local frame kept: the wall moves with its new parent
    assert room.children(group) == [wall] and room.ancestors(wall) == [group]
    center = room.world_bounds(wall)
    assert center[0][0] == pytest.approx(10 - 1.5) and center[1][0] == pytest.approx(10 + 1.5)
    with pytest.raises(sgeo.SpatialFormatError):
        room.reparent(group, wall)                         # cycles are impossible
    mapping = room.duplicate(group)
    assert len(mapping) == 2 and set(mapping).isdisjoint(mapping.values())
    assert room.query_ids(name="Wall A", limit=10) == [wall] or len(room.query_ids(name="Wall A")) == 2
    assert room.validate().ok


def test_sgeo_families_instances_and_deterministic_regeneration(tmp_path):
    room = SGEODocument.new("F", unit="m")
    family = room.define_box_family("Column", width=0.4, depth=0.4, height=3.0)
    a = room.instantiate(family, position=(0, 0, 1.5))
    b = room.instantiate(family, {"height": 4.0}, position=(5, 0, 2.0))
    assert room.volume(a) == pytest.approx(0.4 * 0.4 * 3.0) and room.volume(b) == pytest.approx(0.4 * 0.4 * 4.0)
    room.update_parameters(a, width=0.5)
    assert room.parameter_values(a)["width"] == 0.5
    room.save(tmp_path / "f.sgeo")
    again = SGEODocument.open(tmp_path / "f.sgeo", mode="write")
    assert again.volume(a) == pytest.approx(0.5 * 0.4 * 3.0)
    with pytest.raises(sgeo.SpatialFormatError):
        again.convert_unit("mm")                           # families keep their own units: refuse, never guess
    with pytest.raises(sgeo.SpatialFormatError):
        again.remove_family(family)                        # still instantiated


def test_caller_chosen_object_ids_and_metadata_operation():
    room = SGEODocument.new("ids")
    chosen = "11111111-1111-4111-8111-111111111111"
    (got,) = room.apply([{"op": "add", "id": chosen, "geometry": {"kind": "point", "position": [0, 0, 0]}}])
    assert got == chosen
    with pytest.raises(sgeo.SpatialFormatError):
        room.apply([{"op": "add", "id": chosen, "geometry": {"kind": "point", "position": [1, 1, 1]}}])
    room.apply([{"op": "metadata", "id": chosen, "value": {"survey:station": "S1"}}])
    assert room.get(chosen)["metadata"]["survey:station"] == "S1"


def test_agent_records_reference_and_units(tmp_path):
    room, wall = make_room()
    (record,) = room.spatial_records([wall], unit="mm")
    assert record["unit"] == "mm"
    assert record["entity_id"] == wall and record["bounds"]["upper"][0] == pytest.approx(1500.0)
    assert record["metadata"]["slai.source_unit"] == "m" and record["metadata"]["slai.record_unit"] == "mm"
    assert room.reference()["sha256"] is None                      # never saved: no digest to cite
    room.save(tmp_path / "r.sgeo")
    ref = room.reference(uri="file:///room.sgeo", object_ids=[wall])
    assert ref["object_ids"] == [wall] and ref["sha256"] == hashlib.sha256((tmp_path / "r.sgeo").read_bytes()).hexdigest()


def test_scene_analysis_and_tessellation_budgets():
    room = SGEODocument.new("a")
    sphere = room.add_sphere(1.0)
    assert room.volume(sphere) == pytest.approx(4 / 3 * pi)
    assert room.surface_area(sphere) == pytest.approx(4 * pi)
    with pytest.raises(sgeo.ValidationError):
        room.analyze(sphere)                               # mesh diagnostics need a triangle mesh object
    solid = room.add_object(room.to_mesh(radial_segments=16, latitude_bands=8))
    diagnostics = room.analyze(solid)
    assert diagnostics["is_closed_oriented_surface"] and diagnostics["signed_volume"] > 0
    mesh = room.tessellate(sphere, radial_segments=16, latitude_bands=8)
    assert len(mesh["faces"]) > 0
    with pytest.raises(sgeo.SpatialFormatError):
        room.tessellate(sphere, radial_segments=10_000_000)


# --------------------------------------------------------------------------- reports and errors
def test_validation_report_is_structured_and_round_trips():
    plan = SDRWDocument.new("r")
    plan.add_layer("empty")
    report = plan.validate()
    assert report.ok and {i.code for i in report.infos} >= {"layer.unused", "drawing.empty"}
    assert type(report).from_dict(json.loads(json.dumps(report.as_dict()))).as_dict() == report.as_dict()
    basic = plan.validate("basic")
    assert basic.mode == "basic" and "round-trip" not in basic.checks


def test_validation_report_collects_every_error_and_raises_the_typed_one():
    plan = SDRWDocument.new("bad")
    a = plan.add_point((0, 0))
    b = plan.add_point((1, 1))
    for entity in plan.model.entities:
        entity["layer"] = "ghost"
    report = plan.validate(raise_on_error=False)
    assert not report.ok and len(report.errors) >= 2
    with pytest.raises(sdrw.SpatialFormatError) as caught:
        plan.validate(raise_on_error=True)
    assert caught.value.context["error_count"] >= 2


def test_error_codes_are_stable_and_round_trip():
    for cls in (sdrw.ValidationError, sdrw.SchemaError, sdrw.IntegrityError, sdrw.ConflictError,
                sdrw.ReadOnlyDocumentError, sdrw.UnsupportedFeatureError, sdrw.SecurityLimitError,
                sdrw.ResourceLimitError, sdrw.SpatialReferenceError, sdrw.GeometryValidationError):
        assert cls.code.startswith("spatial.") and sdrw.error_for_code(cls.code) is cls
    assert sdrw.SpatialSchemaError is sdrw.SchemaError
