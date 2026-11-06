"""Phase 0 native modifier contracts: identities, transactions, DAGs and migration."""
from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile

import pytest

from src.utils.sgeo import SGEODocument
from src.utils.sdrw import SDRWDocument
from src.utils.spatial_formats.common.errors import (
    ConflictError, ReadOnlyDocumentError, ReferenceError, ResourceLimitError, UnsupportedFeatureError, ValidationError,
)
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.helpers import canonical_json, digest
from src.utils.spatial_formats.common.modifiers import reference_status
from src.utils.spatial_formats.container.migration import migrate_document
from src.utils.spatial_formats.container.reader import MIMES, PAYLOADS


def _scene():
    doc = SGEODocument.new("Modifier contract")
    first = doc.add_box(width=2, depth=3, height=4)
    second = doc.add_box(width=1, depth=1, height=1)
    return doc, first, second


def _drawing():
    doc = SDRWDocument.new("Modifier contract")
    first = doc.add_line((0, 0), (10, 0))
    second = doc.add_line((0, 1), (10, 1))
    return doc, first, second


@pytest.mark.parametrize("factory,cls", [(_scene, SGEODocument), (_drawing, SDRWDocument)])
def test_identity_modifier_roundtrip_and_incremental_cache(factory, cls):
    doc, first, second = factory()
    mid = doc.add_modifier(first)
    assert doc.manifest()["requires"][-1] == "modifier-graph-v1"
    assert doc.modifiers()[0]["id"] == mid
    original = deepcopy(doc.get(first))
    first_pass = doc.evaluate_modifiers()
    assert len(first_pass) == 1 and not first_pass[0]["cache_hit"]
    assert first_pass[0]["result"] == original
    first_pass[0]["result"]["geometry"] = None
    assert doc.get(first) == original
    assert doc.evaluate_modifiers()[0]["cache_hit"]
    assert doc.modifier_graph().affected(item_ids=[second]) == ()
    assert doc.modifier_graph().affected(item_ids=[first]) == (mid,)

    # An unrelated edit changes document revision, not the target fingerprint.
    if cls is SGEODocument:
        doc.add_box(width=3, depth=2, height=1)
    else:
        doc.add_line((0, 2), (10, 2))
    assert doc.evaluate_modifiers()[0]["cache_hit"]

    blob = doc.dumps()
    reopened = cls.from_bytes(blob)
    assert reopened.source_version == "0.3.0"
    assert reopened.modifiers() == doc.modifiers()
    assert reopened.validate().ok
    assert reopened.get(first) == original
    assert not reopened.evaluate_modifiers()[0]["cache_hit"]
    assert reopened.evaluate_modifiers()[0]["cache_hit"]
    assert reopened.dumps() == blob


@pytest.mark.parametrize("factory", [_scene, _drawing])
def test_modifier_changes_atomic_revisioned_and_reorderable(factory):
    doc, first, _ = factory()
    one, two = doc.add_modifier(first), doc.add_modifier(first)
    assert doc.modifier_graph().dependencies[two] == frozenset({one})
    assert doc.modifier_graph().affected(modifier_ids=[one]) == (one, two)

    revision = doc.revision
    doc.reorder_modifiers([two, one])
    assert doc.revision == revision + 1
    assert doc.modifier_graph().dependencies[one] == frozenset({two})
    doc.update_modifier(one, enabled=False)
    assert [x["status"] for x in doc.evaluate_modifiers()] == ["ok", "suppressed"]
    doc.remove_modifier(one)
    assert [m["id"] for m in doc.modifiers()] == [two]

    before, revision = doc.dumps(), doc.revision
    with pytest.raises((ValidationError, ReferenceError)):
        doc.apply([{"op": "remove_modifier", "id": two},
                   {"op": "update_modifier", "id": two, "value": {"enabled": False}}])
    assert doc.revision == revision and doc.dumps() == before
    with pytest.raises(ConflictError):
        doc.apply([{"op": "remove_modifier", "id": two}], expected_revision=revision - 1)
    assert doc.dumps() == before


@pytest.mark.parametrize("factory", [_scene, _drawing])
def test_cycles_missing_references_and_unknown_operations_are_rejected(factory):
    doc, first, second = factory()
    a, b = new_id(), new_id()
    template = {"id": a, "kind": doc.FORMAT_ID.split(".")[-1] + ".identity",
                "version": 1, "target": {"document_id": doc.document_id, "item_id": first},
                "operands": [], "parameters": {}, "enabled": True,
                "depends_on": [b], "evaluation_policy": {"mode": "exact"}}
    other = {**deepcopy(template), "id": b, "target": {"document_id": doc.document_id, "item_id": second},
             "depends_on": [a]}
    baseline = doc.dumps()
    with pytest.raises(ValidationError, match="cyclic"):
        doc.apply([{"op": "add_modifier", "value": template}, {"op": "add_modifier", "value": other}])
    assert doc.dumps() == baseline

    broken = {**deepcopy(template), "depends_on": [new_id()]}
    with pytest.raises(ReferenceError):
        doc.apply([{"op": "add_modifier", "value": broken}])
    with pytest.raises(UnsupportedFeatureError):
        doc.add_modifier(first, kind=doc.FORMAT_ID.split(".")[-1] + ".boolean")
    assert doc.dumps() == baseline


@pytest.mark.parametrize("factory", [_scene, _drawing])
def test_references_copy_and_target_deletion(factory):
    doc, first, _ = factory()
    mid = doc.add_modifier(first)
    reference = deepcopy(doc.modifiers()[0]["target"])
    items = {first: doc.get(first)}
    assert reference_status(reference, document_id=doc.document_id,
                            revision=doc.revision, items=items)["status"] == "resolved"
    assert reference_status({**reference, "revision": doc.revision - 1},
                            document_id=doc.document_id, revision=doc.revision,
                            items=items)["status"] == "stale"
    assert reference_status({**reference, "document_id": new_id()},
                            document_id=doc.document_id, revision=doc.revision,
                            items=items)["status"] == "external"
    assert reference_status({**reference, "subelement": {"kind": "edge", "id": new_id()}},
                            document_id=doc.document_id, revision=doc.revision,
                            items=items)["status"] == "unsupported"
    assert reference_status(reference, document_id=doc.document_id,
                            revision=doc.revision, items={})["status"] == "missing"

    copy = doc.copy()
    assert copy.document_id != doc.document_id
    assert copy.modifiers()[0]["target"]["document_id"] == copy.document_id
    assert copy.evaluate_modifiers()[0]["status"] == "ok"
    assert copy.modifiers()[0]["id"] == mid

    old_revision = doc.revision
    with pytest.raises(ValidationError):
        if isinstance(doc, SGEODocument):
            doc.remove(first)
        else:
            doc.remove(first)
    assert doc.revision == old_revision


@pytest.mark.parametrize("factory", [_scene, _drawing])
def test_v02_migration_is_lossless_and_old_schema_is_frozen(factory):
    doc, first, _ = factory()
    current_manifest, current_doc = doc.manifest(), doc.as_dict()
    old_manifest = {**current_manifest, "format_version": "0.2.0",
                    "schema": doc.FORMAT_ID + "/0.2",
                    "parts": [{"path": "scene.json" if doc.FORMAT_ID == "slai.sgeo" else "drawing.json",
                               "sha256": "0" * 64, "size": 0}]}
    old_doc = deepcopy(current_doc)
    del old_doc["modifiers"]
    migrated_manifest, migrated_doc, report = migrate_document(old_manifest, old_doc)
    assert report["source_version"] == "0.2.0" and report["losses"] == []
    assert migrated_manifest["format_version"] == "0.3.0"
    assert migrated_doc == current_doc
    assert old_doc.get("modifiers") is None

    with pytest.raises(ValidationError):
        invalid = {**deepcopy(old_doc), "modifiers": []}
        migrate_document(old_manifest, invalid)


@pytest.mark.parametrize("factory,cls", [(_scene, SGEODocument), (_drawing, SDRWDocument)])
def test_real_v02_archive_opens_and_resaves_to_v03(factory, cls):
    doc, first, _ = factory()
    manifest = {**doc.manifest(), "format_version": "0.2.0", "schema": doc.FORMAT_ID + "/0.2"}
    payload = doc.as_dict()
    del payload["modifiers"]
    payload_bytes = canonical_json(payload)
    part = PAYLOADS[doc.FORMAT_ID]
    manifest["parts"] = [{"path": part, "size": len(payload_bytes), "sha256": digest(payload_bytes)}]
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("mimetype", MIMES[doc.FORMAT_ID])
        archive.writestr("manifest.json", canonical_json(manifest))
        archive.writestr(part, payload_bytes)
    legacy_bytes = output.getvalue()
    migrated = cls.from_bytes(legacy_bytes)
    assert migrated.source_version == "0.2.0"
    assert migrated.get(first) == doc.get(first)
    assert migrated.modifiers() == []
    assert migrated.validate("strict").ok
    upgraded = cls.from_bytes(migrated.dumps())
    assert upgraded.source_version == "0.3.0"
    assert upgraded.as_dict() == migrated.as_dict()


def test_invalid_reordering_is_rejected_without_untyped_errors():
    doc, first, _ = _scene()
    mid = doc.add_modifier(first)
    before = doc.dumps()
    with pytest.raises(ValidationError):
        doc.reorder_modifiers([{"id": mid}])
    assert doc.dumps() == before


@pytest.mark.parametrize("factory,cls", [(_scene, SGEODocument), (_drawing, SDRWDocument)])
def test_read_only_documents_can_inspect_but_not_edit_modifiers(factory, cls):
    doc, first, _ = factory()
    mid = doc.add_modifier(first)
    read_only = cls.from_bytes(doc.dumps(), mode="read")
    assert read_only.modifiers()[0]["id"] == mid
    assert read_only.evaluate_modifiers()[0]["status"] == "ok"
    with pytest.raises(ReadOnlyDocumentError):
        read_only.add_modifier(first)
    with pytest.raises(ReadOnlyDocumentError):
        read_only.remove_modifier(mid)


def test_invalid_policy_is_atomic():
    doc, item, _ = _scene()
    before = doc.dumps()
    with pytest.raises(ValidationError, match="max_error"):
        doc.add_modifier(item, evaluation_policy={"mode": "bounded_approximation"})
    with pytest.raises(ValidationError):
        doc.add_modifier(item, evaluation_policy={"mode": "exact", "absolute": float("nan")})
    assert doc.dumps() == before


@pytest.mark.parametrize("factory", [_scene, _drawing])
def test_transitive_cache_invalidation_and_topological_order(factory):
    doc, first, second = factory()
    one = doc.add_modifier(first)
    two = doc.add_modifier(second, depends_on=[one])
    assert doc.modifier_graph().order == (one, two)
    assert doc.modifier_graph().affected(item_ids=[first]) == (one, two)
    assert doc.modifier_graph().affected(item_ids=[second]) == (two,)
    assert all(not result["cache_hit"] for result in doc.evaluate_modifiers())
    assert all(result["cache_hit"] for result in doc.evaluate_modifiers())
    if isinstance(doc, SGEODocument):
        doc.set_geometry(first, {"kind": "box", "center": [0, 0, 0], "size": [3, 3, 3]})
    else:
        doc.update(first, geometry={"start": [0, 0], "end": [20, 0]})
    assert [result["cache_hit"] for result in doc.evaluate_modifiers()] == [False, False]
    assert all(result["cache_hit"] for result in doc.evaluate_modifiers())


@pytest.mark.parametrize("factory", [_scene, _drawing])
def test_disabled_and_invalid_topology_inputs(factory):
    doc, first, _ = factory()
    mid = doc.add_modifier(first, enabled=False)
    assert doc.evaluate_modifiers()[0]["status"] == "suppressed"
    doc.update_modifier(mid, enabled=True)
    assert doc.evaluate_modifiers()[0]["status"] == "ok"
    revision = doc.revision
    with pytest.raises(UnsupportedFeatureError, match="topology"):
        doc.update_modifier(mid, target={"document_id": doc.document_id, "item_id": first,
                                         "subelement": {"kind": "edge", "id": new_id()}})
    assert doc.revision == revision


def test_bounded_graph_depth_rejects_pathological_file():
    doc, first, _ = _scene()
    ids = [new_id() for _ in range(258)]
    spec = {"kind": "sgeo.identity", "version": 1,
            "target": {"document_id": doc.document_id, "item_id": first},
            "operands": [], "parameters": {}, "enabled": True,
            "evaluation_policy": {"mode": "exact"}}
    mods = [{**spec, "id": mid, "depends_on": [ids[i - 1]] if i else []}
            for i, mid in enumerate(ids)]
    with pytest.raises(ResourceLimitError):
        doc.apply([{"op": "add_modifier", "value": mod} for mod in mods])
    assert doc.modifiers() == []
