"""Named family types in SGEO: precedence, scope, atomic edits, persistence and capability gating (SPADRA P0-01/04)."""
import copy

import pytest

from src.utils.sgeo import SGEODocument
from src.utils.spatial_formats.common.errors import ValidationError, ConflictError
from src.utils.spatial_formats.sgeo.families import generate_geometry, resolve_parameters

DOOR = {
    "width": {"type": "length", "default": 0.9, "min": 0.6, "max": 1.8, "scope": "type"},
    "height": {"type": "length", "default": 2.1, "min": 1.8, "max": 2.6, "scope": "type"},
    "thickness": {"type": "length", "default": 0.04, "min": 0.02, "max": 0.1},               # instance scope (the default)
    "frame": {"type": "length", "default": 0.035},
    "clear_width": {"type": "length", "expression": "width - 2 * frame"},
}
TYPES = {"D-800": {"width": 0.8, "height": 2.1}, "D-900": {"width": 0.9, "height": 2.1}, "D-1000": {"width": 1.0, "height": 2.3}}


def door_family(doc, **kw):
    # a box generator needs width/depth/height: depth comes from the door's thickness for this test
    specs = copy.deepcopy(DOOR)
    specs["depth"] = {"type": "length", "expression": "thickness"}
    return doc.define_family("Single Leaf Door", specs, "builtin.box_v1", types=copy.deepcopy(TYPES), **kw)


def requires(data):
    import json
    import zipfile
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(data)) as z:
        return set(json.loads(z.read("manifest.json"))["requires"])


def size(doc, oid):
    return doc.get(oid)["geometry"]["size"]


def test_type_values_sit_between_defaults_and_instance_overrides():
    doc = SGEODocument.new("Doors")
    fid = door_family(doc)
    a = doc.instantiate(fid, family_type="D-1000")
    b = doc.instantiate(fid, {"thickness": 0.06}, family_type="D-800")
    c = doc.instantiate(fid)                                       # no type: family defaults
    assert size(doc, a) == [1.0, 0.04, 2.3] and size(doc, b) == [0.8, 0.06, 2.1] and size(doc, c) == [0.9, 0.04, 2.1]
    assert doc.parameter_values(a)["clear_width"] == pytest.approx(0.93)
    assert doc.get(a)["family_type"] == "D-1000" and "family_type" not in doc.get(c)
    # override beats type, and a formula follows whichever value wins
    doc.set_parameters(a, {"frame": 0.05})
    assert doc.parameter_values(a)["clear_width"] == pytest.approx(0.9)


def test_instance_cannot_override_a_type_scoped_parameter_or_a_formula():
    doc = SGEODocument.new("Doors")
    fid = door_family(doc)
    iid = doc.instantiate(fid, family_type="D-900")
    rev = doc.revision
    for bad in ({"width": 0.7}, {"clear_width": 0.5}, {"ghost": 1.0}):
        with pytest.raises(ValidationError):
            doc.set_parameters(iid, bad)
    with pytest.raises(ValidationError, match="type parameter"):
        doc.instantiate(fid, {"width": 0.7}, family_type="D-800")
    assert doc.revision == rev and size(doc, iid) == [0.9, 0.04, 2.1]        # nothing changed


def test_unknown_type_and_invalid_type_values_are_refused():
    doc = SGEODocument.new("Doors")
    fid = door_family(doc)
    with pytest.raises(ValidationError, match="unknown family type"):
        doc.instantiate(fid, family_type="D-9999")
    specs = copy.deepcopy(DOOR)
    specs["depth"] = {"type": "length", "expression": "thickness"}
    for bad_types, match in (({"X": {"width": 5.0}}, "outside bounds"), ({"X": {"clear_width": 0.5}}, "derived"),
                             ({"X": {"nope": 1}}, "overrides"), ({"a": {}, "A": {}}, "duplicate"), ({" x": {}}, "invalid family type name"),
                             ({"X": {"height": 2.3, "width": 0.0}}, "outside bounds")):
        with pytest.raises(ValidationError, match=match):
            doc.define_family("Bad", copy.deepcopy(specs), types=bad_types)
    assert len(doc.families()) == 1


def test_switching_type_regenerates_and_keeps_overrides():
    doc = SGEODocument.new("Doors")
    fid = door_family(doc)
    iid = doc.instantiate(fid, {"thickness": 0.05}, family_type="D-800")
    doc.set_family_type(iid, "D-1000")
    assert size(doc, iid) == [1.0, 0.05, 2.3]
    doc.set_family_type(iid, None)
    assert size(doc, iid) == [0.9, 0.05, 2.1] and "family_type" not in doc.get(iid)
    with pytest.raises(ValidationError):
        doc.set_family_type(iid, "missing")
    plain = doc.add_box(name="plain", size=(1, 1, 1))
    with pytest.raises(ValidationError, match="instantiated family"):
        doc.set_family_type(plain, "D-800")


def test_editing_a_type_propagates_to_every_instance_atomically():
    doc = SGEODocument.new("Doors")
    fid = door_family(doc)
    ids = [doc.instantiate(fid, family_type="D-900") for _ in range(3)] + [doc.instantiate(fid, family_type="D-800")]
    fam = doc.families()[0]
    fam.pop("instance_count")
    fam["types"]["D-900"] = {"width": 0.95, "height": 2.2}
    doc.update_family(fam)
    assert [size(doc, i)[0] for i in ids] == [0.95, 0.95, 0.95, 0.8]
    rev = doc.revision
    fam["types"]["D-900"] = {"width": 0.95, "height": 99.0}                       # exceeds the height bound
    with pytest.raises(ValidationError):
        doc.update_family(fam)
    assert doc.revision == rev and size(doc, ids[0]) == [0.95, 0.04, 2.2]


def test_removing_a_type_in_use_is_refused_and_unused_types_can_go():
    doc = SGEODocument.new("Doors")
    fid = door_family(doc)
    iid = doc.instantiate(fid, family_type="D-1000")
    fam = doc.families()[0]
    fam.pop("instance_count")
    del fam["types"]["D-1000"]
    rev = doc.revision
    with pytest.raises(ValidationError, match="cannot remove family type 'D-1000': 1 instance") as exc:
        doc.update_family(fam)
    assert exc.value.context["instances"] == [iid] and doc.revision == rev
    del fam["types"]["D-800"]
    fam["types"]["D-1000"] = TYPES["D-1000"]
    doc.update_family(fam)                                                          # D-800 is unused
    assert set(doc.families()[0]["types"]) == {"D-900", "D-1000"}


def test_renaming_a_type_repoints_its_instances_in_one_revision():
    doc = SGEODocument.new("Doors")
    fid = door_family(doc)
    iid = doc.instantiate(fid, family_type="D-800")
    fam = doc.families()[0]
    fam.pop("instance_count")
    fam["types"]["D-0800"] = fam["types"].pop("D-800")
    rev = doc.revision
    doc.update_family(fam, type_renames={"D-800": "D-0800"})
    assert doc.revision == rev + 1 and doc.get(iid)["family_type"] == "D-0800" and size(doc, iid)[0] == 0.8
    with pytest.raises(ValidationError):                                           # without the rename the type would dangle
        fam2 = copy.deepcopy(fam)
        fam2["types"]["D-0900"] = fam2["types"].pop("D-0800")
        doc.update_family(fam2)


def test_family_types_survive_save_and_reopen_and_are_declared_as_a_capability():
    doc = SGEODocument.new("Doors")
    fid = door_family(doc)
    iid = doc.instantiate(fid, {"thickness": 0.07}, family_type="D-1000")
    data = doc.dumps()
    again = SGEODocument.from_bytes(data)
    assert again.get(iid)["family_type"] == "D-1000" and again.families()[0]["types"] == TYPES
    assert size(again, iid) == size(doc, iid) and again.parameter_values(iid)["clear_width"] == pytest.approx(0.93)
    assert {"families", "family-types-v1", "parameter-types-v2"} <= requires(data)


def test_documents_that_do_not_use_the_new_features_keep_their_old_capability_list():
    doc = SGEODocument.new("Old")
    fid = doc.define_box_family("Block", width=1, depth=1, height=1)
    doc.instantiate(fid, {"width": 2})
    req = requires(doc.dumps())
    assert "families" in req and not ({"family-types-v1", "parameter-types-v2"} & req)


def test_new_parameter_types_declare_their_capability():
    doc = SGEODocument.new("Typed")
    fid = doc.define_family("Beam", {"width": {"type": "length", "default": 0.2}, "depth": {"type": "length", "default": 0.4},
                                      "height": {"type": "length", "default": 3.0}, "load": {"type": "force", "default": 5000.0},
                                      "grade": {"type": "enumeration", "choices": ["C24", "C30"], "default": "C24"}})
    iid = doc.instantiate(fid, {"grade": "C30"})
    again = SGEODocument.from_bytes(doc.dumps())
    assert again.parameter_values(iid)["load"] == 5000.0 and again.parameter_values(iid)["grade"] == "C30"
    assert "parameter-types-v2" in requires(doc.dumps()) and "family-types-v1" not in requires(doc.dumps())


def test_resolve_and_generate_accept_the_type_argument_directly():
    fam = {"id": "7f3c1c9e-3b1a-4c53-9a58-1f2d4d5e6f70", "name": "Block", "generator": {"kind": "builtin.box_v1"},
           "parameters": {"width": {"type": "length", "default": 1.0}, "depth": {"type": "length", "default": 1.0},
                          "height": {"type": "length", "default": 1.0}},
           "types": {"Tall": {"height": 3.0}}}
    assert generate_geometry(fam, {"width": 2.0}, "Tall")["size"] == [2.0, 1.0, 3.0]
    assert resolve_parameters(fam, None, "Tall")["height"] == 3.0 and resolve_parameters(fam)["height"] == 1.0
