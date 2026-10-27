"""Additional contracts for shared units, provenance, migrations and adapters.

These tests target previously uncovered behavioral branches, rather than
artificially importing unsupported converters just to inflate coverage.
"""
from __future__ import annotations

from math import pi
import pytest

from src.utils.spatial_formats import Drawing, Scene, load_package, write_package
from src.utils.spatial_formats.adapters.slai_provenance import make_observation, claim_digest
from src.utils.spatial_formats.adapters.slai_spatial import spatial_records
from src.utils.spatial_formats.adapters.slai_stem import quantity, convert_quantity
from src.utils.spatial_formats.adapters.svg import export_svg, import_svg
from src.utils.spatial_formats.common.errors import (
    ValidationError, UnsupportedFeatureError, MigrationError,
)
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.metadata import require_metadata, require_claim
from src.utils.spatial_formats.common.units import (
    convert_length, convert_area, convert_volume, require_unit,
)
from src.utils.spatial_formats.container.migration import inspect_migration, migrate_document
from src.utils.spatial_formats.sdrw.entities import make_entity


@pytest.mark.parametrize(("value", "source", "target", "expected"), [
    (1000, "mm", "m", 1.0),
    (1, "ft", "in", 12.0),
    (100, "cm", "m", 1.0),
])
def test_length_conversions(value, source, target, expected):
    assert convert_length(value, source, target) == pytest.approx(expected)


def test_area_volume_dimensions():
    assert convert_area(10000, "cm", "m") == pytest.approx(1)
    assert convert_volume(1_000_000, "cm", "m") == pytest.approx(1)
    assert convert_area(1, "m", "cm") == pytest.approx(10000)
    assert convert_volume(1, "m", "cm") == pytest.approx(1_000_000)


@pytest.mark.parametrize("unit", ["degrees", "M", "", None, 42])
def test_invalid_length_units_rejected(unit):
    with pytest.raises(ValidationError):
        require_unit(unit)


def test_stem_adapter_preserves_dimensions():
    assert convert_quantity(quantity(10, "cm"), "mm")["value"] == pytest.approx(100)
    assert convert_quantity(quantity(1, "m", "area"), "cm")["value"] == pytest.approx(10000)
    assert convert_quantity(quantity(1, "m", "volume"), "cm")["value"] == pytest.approx(1_000_000)
    with pytest.raises(ValidationError):
        quantity(1, "m", "temperature")
    with pytest.raises(ValidationError):
        convert_quantity({"dimension": "unsupported"}, "m")


def test_namespaced_metadata_and_claims():
    assert require_metadata({"org.example:tag": 2, "slai.note": "hello"})["slai.note"] == "hello"
    with pytest.raises(ValidationError):
        require_metadata({"tag": 3})
    with pytest.raises(ValidationError):
        require_metadata([])
    with pytest.raises(ValidationError):
        require_claim({"subject": new_id(), "predicate": "seen", "confidence": -0.5})
    with pytest.raises(ValidationError):
        require_claim({"subject": new_id(), "predicate": "seen", "evidence": ""})


def test_provenance_digests_are_canonical():
    oid = new_id()
    claim = make_observation(oid, "observed", evidence="document:42", agent="Spatial", confidence=0.75)
    assert claim["subject"] == oid
    assert claim["confidence"] == pytest.approx(0.75)
    assert len(claim_digest(claim)) == 64
    assert claim_digest(claim) == claim_digest(dict(reversed(list(claim.items()))))
    with pytest.raises(ValidationError):
        make_observation(oid, "", evidence="test")
    with pytest.raises(ValidationError):
        make_observation(oid, "seen", evidence="test", agent=42)


def test_version_negotiate_and_copy(tmp_path):
    scene = Scene()
    path = tmp_path / "source.sgeo"
    write_package(path, scene.manifest(), scene.as_dict())
    package = load_package(path)
    status = inspect_migration(package.manifest)
    assert status["compatible"] and not status["required"]
    manifest, document, result = migrate_document(package.manifest, package.document)
    assert document == package.document
    document["name"] = "changed"
    assert package.document["name"] != "changed"
    assert result == status
    with pytest.raises(UnsupportedFeatureError):
        migrate_document(package.manifest, package.document, target_version="9.9.9")


def test_version_rejects_unsupported_inputs():
    with pytest.raises(MigrationError):
        inspect_migration([])
    assert not inspect_migration({"format_version": "99.0"})["compatible"]
    with pytest.raises(UnsupportedFeatureError):
        migrate_document({"format_version": "99.0"}, {})


def test_spatial_records_use_scene_world_bounds():
    scene = Scene()
    root_id = scene.add_object(
        geometry={"kind": "box", "center": [0, 0, 0], "size": [2, 2, 2]},
        transform=[1, 0, 0, 10, 0, 1, 0, 20, 0, 0, 1, 30, 0, 0, 0, 1],
    )
    records = spatial_records(scene)
    assert records[0]["entity_id"] == root_id
    assert records[0]["position"] == pytest.approx([10, 20, 30])
    assert records[0]["bounds"]["lower"] == pytest.approx([9, 19, 29])
    assert records[0]["metadata"]["slai.source_document"] == scene.document_id


@pytest.mark.parametrize(("kind","geometry"), [
    ("line", {"start": [0, 0], "end": [3, 4]}),
    ("circle", {"center": [1, 2], "radius": 4}),
    ("polyline", {"points": [[0, 0], [2, 0], [2, 2]], "closed": True}),
])
def test_svg_roundtrip_supported_shapes(kind, geometry):
    drawing = Drawing()
    drawing.add(kind, geometry)
    imported = import_svg(export_svg(drawing))
    assert imported.entities[0]["kind"] == kind
    assert imported.entities[0]["geometry"] == geometry


def test_svg_rejects_external_dtd_and_script():
    with pytest.raises(UnsupportedFeatureError):
        import_svg('<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg" />')
    with pytest.raises(UnsupportedFeatureError):
        import_svg('<svg xmlns="http://www.w3.org/2000/svg"><script /></svg>')
    with pytest.raises(UnsupportedFeatureError):
        import_svg('<svg xmlns="http://www.w3.org/2000/svg"><line x1="0" y1="0" x2="1" y2="1" onload="alert(1)"/></svg>')


def test_svg_arc_export_round_trips_and_foreign_arc_paths_are_rejected():
    drawing = Drawing()
    drawing.add("arc", {"center": [0, 0], "radius": 1, "start_angle": 0, "end_angle": pi / 2})
    xml = export_svg(drawing)
    imported = import_svg(xml)
    geometry = imported.as_dict()["entities"][0]["geometry"]
    assert imported.as_dict()["entities"][0]["kind"] == "arc"
    assert geometry["center"] == [0, 0] and geometry["radius"] == 1
    assert geometry["start_angle"] == 0 and abs(geometry["end_angle"] - pi / 2) < 1e-12
    # A generic SVG path (no exact SLAI arc parameters) is still refused.
    with pytest.raises(UnsupportedFeatureError):
        import_svg('<svg xmlns="http://www.w3.org/2000/svg"><path d="M 1 0 A 1 1 0 0 1 0 1"/></svg>')
    # Parameters that disagree with the path data are refused rather than trusted.
    forged = xml.replace('data-r="1"', 'data-r="2"')
    assert forged != xml
    with pytest.raises(Exception) as caught:
        import_svg(forged)
    assert getattr(caught.value, "code", "").startswith("spatial.")
