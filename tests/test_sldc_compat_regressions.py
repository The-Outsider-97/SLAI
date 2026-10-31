"""SLDC forward-compatibility and safe-preview regression coverage.

The preview is deliberately data-only, bounded, integrity-checked and never
silently rewrites a future document. No Optimization dependencies.
"""
from __future__ import annotations

import copy
import io
import json
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

import pytest

from src.utils import sldc


def _modern():
    return sldc.new_package(
        "Safe preview", [
            sldc.make_block("heading", "Introduction", block_id="b1"),
            sldc.make_block("paragraph", "Known content", block_id="b2"),
        ], doc_id="doc-preview"
    )


def _reseal(data: bytes, *, version: str | None = None,
            requires: list[str] | None = None, mutate=None) -> bytes:
    with ZipFile(io.BytesIO(data), "r") as source:
        mimetype = source.read("mimetype")
        manifest = json.loads(source.read("manifest.json"))
        items = [(entry["name"], source.read(entry["name"]))
                 for entry in manifest["parts"]]
        if mutate is not None:
            new_items = []
            for name, raw in items:
                if name == "document.json":
                    document = json.loads(raw)
                    mutate(document)
                    raw = json.dumps(
                        document, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    # Keep the canonical content inventory internally consistent.
                    entry = next(e for e in manifest["parts"] if e["name"] == name)
                    import hashlib
                    entry["size"] = len(raw)
                    entry["sha256"] = hashlib.sha256(raw).hexdigest()
                new_items.append((name, raw))
            items = new_items
            import hashlib
            manifest["revision_id"] = sldc._revision_digest(
                (e["name"], e["size"], e["sha256"])
                for e in manifest["parts"] if e["role"] in ("content", "asset")
            )
        if version is not None:
            manifest["version"] = version
        if requires is not None:
            manifest["requires"] = requires
    out = io.BytesIO()
    with ZipFile(out, "w", allowZip64=False) as archive:
        archive.writestr("mimetype", mimetype, compress_type=ZIP_STORED)
        archive.writestr(
            "manifest.json", json.dumps(
                manifest, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode(), compress_type=ZIP_DEFLATED,
        )
        for name, raw in items:
            archive.writestr(name, raw, compress_type=ZIP_DEFLATED)
    return out.getvalue()


def test_known_package_preview_is_complete_and_plain_text():
    data = sldc.dumps_package(_modern())
    result = sldc.preview_package(data)
    assert result["complete"]
    assert "Introduction" in result["text"]
    assert "Known content" in result["text"]
    assert result["read_only"]


def test_unknown_required_feature_is_read_only_previewable():
    data = _reseal(
        sldc.dumps_package(_modern()), version="1.7.0",
        requires=["future-holograms"],
    )
    with pytest.raises(sldc.SLDCUnsupportedError):
        sldc.loads_package(data)
    result = sldc.preview_package(data)
    assert result["complete"] is False
    assert "future-holograms" in result["unsupported_requires"]
    assert result["newer_than_known"]


def test_hostile_future_blocks_are_reduced_to_plain_text():
    def mutate(document):
        document["title"] = ["not", "a", "string"]
        document["blocks"][0]["unknown_shape"] = {"executables": ["no"]}
        document["blocks"].append({"type": "hologram", "id": "../../bad", "text": "Future words"})
        document["blocks"].append("not a block")
        document["blocks"].append({"type": "paragraph", "text": 987})
        document["blocks"].append({
            "type": "table", "text": "Caption",
            "table": {"rows": [[{"text": "first"}, 11, {"text": "last"}], None]},
        })
    data = _reseal(sldc.dumps_package(_modern()), version="1.7.0", mutate=mutate)
    preview = sldc.preview_package(data)
    assert not preview["complete"]
    assert preview["skipped"] > 0
    assert "Future words" in preview["text"]
    assert "first" in preview["text"]
    assert "last" in preview["text"]
    assert all(set(b) == {"type", "text"} for b in preview["blocks"])
    assert not any("unknown_shape" in repr(b) for b in preview["blocks"])


def test_preview_to_package_rebuilds_without_unknown_original_fields():
    data = _reseal(
        sldc.dumps_package(_modern()), version="1.7.0",
        requires=["unknown-feature"],
    )
    preview = sldc.preview_package(data)
    new = sldc.preview_to_package(preview)
    assert new.document["id"] != "doc-preview"
    assert new.version == sldc.FORMAT_VERSION
    assert not new.assets and not new.comments and not new.signatures
    assert all(b["type"] == "paragraph" for b in new.document["blocks"])
    assert "Introduction" in sldc.extract_text(new)
    assert sldc.loads_package(sldc.dumps_package(new)).document == new.document


def test_preview_never_ignores_corrupt_source_container():
    raw = bytearray(sldc.dumps_package(_modern()))
    raw[-1] ^= 255
    # ZIP comments/EOCD bytes may not always detect a single bit flip.
    # Corrupt the actual document entry instead.
    with ZipFile(io.BytesIO(sldc.dumps_package(_modern()))) as archive:
        data = archive.read("document.json")
    original = sldc.dumps_package(_modern())
    with ZipFile(io.BytesIO(original)) as archive:
        parts = archive.namelist()
    # Intact manifest but rewritten document should fail SHA-256 validation.
    with ZipFile(io.BytesIO(original)) as archive:
        content = [(n, archive.read(n)) for n in parts]
    tampered = io.BytesIO()
    with ZipFile(tampered, "w") as archive:
        for name, payload in content:
            if name == "document.json":
                payload += b" "
            archive.writestr(name, payload)
    with pytest.raises(sldc.SLDCError):
        sldc.preview_package(tampered.getvalue())


def test_inspection_reports_newer_minor_without_rewriting():
    original = sldc.dumps_package(_modern())
    report = sldc.inspect_package(original)
    assert report["newer_than_known"] is False
    data = _reseal(original, version="1.7.2")
    report = sldc.inspect_package(data)
    assert report["newer_than_known"] is True
    loaded = sldc.loads_package(data)
    assert loaded.version == "1.7.2"
    with pytest.raises(sldc.SLDCUnsupportedError):
        sldc.dumps_package(loaded)


def test_newest_version_is_numeric_not_lexical():
    assert sldc.newest_version("1.9.0", "1.10.0", "1.2.11") == "1.10.0"
    with pytest.raises(sldc.SLDCValidationError):
        sldc.newest_version()


def test_merge_preserves_1_1_features_and_version():
    base = sldc.new_package(
        "Doc", [sldc.make_block("paragraph", "One", block_id="b1")],
        doc_id="doc-merge",
    )
    ours = base.copy()
    ours.document["blocks"].append(
        sldc.make_block("paragraph", "Ours", block_id="b2")
    )
    theirs = base.copy()
    theirs.version = "1.1.0"
    theirs.document["blocks"].append(
        sldc.make_block(
            "equation", "", block_id="b3",
            equation={"syntax": "latex", "source": "x^2"},
        )
    )
    merged, conflicts = sldc.merge_packages(base, ours, theirs)
    assert merged.version == "1.1.0"
    assert any(b["type"] == "equation" for b in merged.document["blocks"])
    sldc.validate_package(merged)



def test_future_image_alt_is_labeled_as_non_image_text():
    def mutate(document):
        document["blocks"][1] = {
            "id": "b2", "type": "figure", "text": "",
            "figure": {"asset": "unavailable", "alt": "red box"},
        }
    # Future-version preview does not claim that a missing image asset exists;
    # it only displays the textual alt field as an explicit image placeholder.
    data = _reseal(
        sldc.dumps_package(_modern()), version="1.7.0", mutate=mutate,
    )
    preview = sldc.preview_package(data)
    assert preview["blocks"][-1]["text"] == "[Image: red box]"
    restored = sldc.preview_to_package(preview)
    assert sldc.extract_text(restored).split("\n\n")[-1] == "[Image: red box]"
    assert restored.assets == {}


def test_future_unknown_types_with_text_do_not_inflate_skipped_block_count():
    def mutate(document):
        document["blocks"][1]["hologram"] = {"depth": 3}
        document["blocks"].append({
            "id": "future-1", "type": "hologram",
            "text": "future text", "future-key": 1,
        })
        document["blocks"].append("not a block")
        document["blocks"].append({
            "id": "future-2", "type": "paragraph", "text": 7,
        })
    preview = sldc.preview_package(
        _reseal(sldc.dumps_package(_modern()), version="1.7.0", mutate=mutate)
    )
    assert preview["skipped"] == 2
    assert preview["skipped_blocks"] == 2
    assert preview["blocks"][-1] == {"type": "paragraph", "text": "future text"}
    assert any("unknown" in note.lower() for note in preview["notes"])


def test_unknown_required_feature_reported_in_preview_notes():
    data = _reseal(
        sldc.dumps_package(_modern()), version="1.2.0", requires=["holo-layout"],
    )
    preview = sldc.preview_package(data)
    assert "holo-layout" in " ".join(preview["notes"])
    assert preview["unsupported_requires"] == ["holo-layout"]
    assert preview["complete"] is False


def test_invalid_future_title_falls_back_without_copying_unknown_data():
    def mutate(document):
        document["title"] = ["not", "text"]
        document["blocks"][1]["future_data"] = {"sensitive": "not copied"}
    preview = sldc.preview_package(
        _reseal(sldc.dumps_package(_modern()), version="1.7.0", mutate=mutate)
    )
    assert preview["title"] == "Untitled document"
    assert "not copied" not in repr(preview)
    assert any("title" in note.lower() for note in preview["notes"])
