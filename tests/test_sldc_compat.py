"""Forward compatibility of src.utils.sldc: reading newer files as a text-only preview, and combining packages of different 1.x versions.

A "future" file is made here by rewriting the manifest and document of a real package, the way a newer program would, and
then restoring the checksums, so the container checks pass and only the unknown content is left for the reader to deal with.
"""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

from src.utils import sldc as s


def _png(width: int = 40, height: int = 20) -> bytes:
    raw = b"".join(b"\x00" + bytes([200, 50, 50]) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _modern_package() -> s.SLDCPackage:
    asset = s.make_asset(_png())
    blocks = [s.make_block("heading", "Title", attributes={"level": 1}), s.make_block("paragraph", "Hello."),
              s.make_block("equation", "", equation={"syntax": "latex", "source": "E = mc^2"}),
              s.make_block("figure", "", figure={"asset": asset.id, "alt": "red box", "width_mm": 60, "rotation_deg": 90})]
    package = s.new_package("Modern", blocks, assets=[asset], sections=[{"id": "sec-1", "page": {"size": "A4", "header_mm": 14, "footer_mm": 6}}])
    package.version = "1.1.0"
    return package


def _rewrite(source: bytes, *, version: str, requires: tuple[str, ...] = (), edit=None) -> bytes:
    """What a newer program would have written: another version, extra required features, and content this module does not know."""
    import io
    archive = zipfile.ZipFile(io.BytesIO(source))
    parts = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(parts["manifest.json"])
    document = json.loads(parts["document.json"])
    if edit:
        edit(document)
    parts["document.json"] = json.dumps(document).encode()
    manifest["version"] = version
    manifest["requires"] = sorted(set(manifest["requires"]) | set(requires))
    for entry in manifest["parts"]:
        if entry["name"] == "document.json":
            entry["size"], entry["sha256"] = len(parts["document.json"]), hashlib.sha256(parts["document.json"]).hexdigest()
    manifest["revision_id"] = s._revision_digest([(e["name"], e["size"], e["sha256"]) for e in manifest["parts"] if e["role"] in ("content", "asset")])
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        target.writestr(info, parts["mimetype"])
        target.writestr("manifest.json", json.dumps(manifest))
        for entry in manifest["parts"]:
            target.writestr(entry["name"], parts[entry["name"]])
    return out.getvalue()


def _future(document_edit=None, requires=("holo-layout",), version="1.2.0") -> bytes:
    return _rewrite(s.dumps_package(_modern_package()), version=version, requires=requires, edit=document_edit)


class PreviewOfNewerFiles(unittest.TestCase):
    def test_a_file_this_module_reads_in_full_previews_as_complete(self) -> None:
        preview = s.preview_package(s.dumps_package(_modern_package()))
        self.assertTrue(preview["complete"])
        self.assertEqual([b["text"] for b in preview["blocks"]], ["Title", "Hello.", "E = mc^2", "[Image: red box]"])
        self.assertEqual(preview["blocks"][0]["level"], 1)

    def test_unknown_required_feature_is_refused_by_a_normal_read_but_can_be_previewed(self) -> None:
        data = _future()
        with self.assertRaises(s.SLDCUnsupportedError):
            s.loads_package(data)
        preview = s.preview_package(data)
        self.assertFalse(preview["complete"])
        self.assertEqual(preview["unsupported_requires"], ["holo-layout"])
        self.assertEqual(preview["version"], "1.2.0")
        self.assertIn("holo-layout", " ".join(preview["notes"]))
        self.assertEqual(preview["blocks"][1]["text"], "Hello.")

    def test_content_this_module_does_not_know_is_shown_as_text_or_counted_as_skipped(self) -> None:
        def edit(document: dict) -> None:
            document["blocks"][1]["hologram"] = {"depth": 3}                      # unknown key on a known block
            document["blocks"].append({"id": "bf1", "type": "hologram", "text": "future text", "holo": 1})   # unknown block type
            document["blocks"].append("not a block")
            document["blocks"].append({"id": "bf2", "type": "paragraph", "text": 7})
        data = _future(edit)
        with self.assertRaises(s.SLDCError):
            s.loads_package(data)
        preview = s.preview_package(data)
        self.assertEqual([b["text"] for b in preview["blocks"]][-1], "future text")
        self.assertEqual(preview["blocks"][-1]["type"], "paragraph")             # an unknown type is shown as plain text
        self.assertEqual(preview["skipped"], 2)
        self.assertFalse(preview["complete"])

    def test_the_container_is_still_checked(self) -> None:
        data = bytearray(_future())
        data[len(data) // 2] ^= 0xFF
        with self.assertRaises(s.SLDCError):
            s.preview_package(bytes(data))
        with self.assertRaises(s.SLDCUnsupportedError):                              # a new MAJOR version is not read at all
            s.preview_package(_future(version="2.0.0"))

    def test_preview_to_package_is_valid_text_only_and_never_carries_the_original(self) -> None:
        package = s.preview_to_package(s.preview_package(_future()))
        s.validate_package(package)
        self.assertEqual(package.assets, {})
        self.assertEqual(package.version, s.FORMAT_VERSION)
        self.assertEqual(s.extract_text(package).split("\n\n"), ["Title", "Hello.", "E = mc^2", "[Image: red box]"])

    def test_only_text_is_read_from_hostile_shapes(self) -> None:
        def edit(document: dict) -> None:
            document["title"] = ["not", "text"]
            document["blocks"][1] = {"id": "../x", "type": "heading", "text": "A", "attributes": {"level": True}}
            document["blocks"][2] = {"id": "b3", "type": "table", "text": "cap", "table": {"rows": [[{"text": "a"}, 5, {"text": "b"}], "x"]}}
        preview = s.preview_package(_future(edit))
        self.assertEqual(preview["title"], "Untitled document")
        self.assertIsNone(preview["blocks"][1]["id"])
        self.assertNotIn("level", preview["blocks"][1])                              # True is not a heading level
        self.assertEqual(preview["blocks"][2]["text"], "cap\na\tb\n")
        s.validate_package(s.preview_to_package(preview))


class InspectionTellsWhatToDo(unittest.TestCase):
    def test_inspect_flags_files_newer_than_this_module_knows(self) -> None:
        self.assertFalse(s.inspect_package(s.dumps_package(_modern_package()))["newer_than_known"])
        report = s.inspect_package(_future(requires=(), version="1.2.0"))
        self.assertTrue(report["newer_than_known"])
        self.assertEqual(report["unsupported_requires"], [])
        self.assertTrue(s.inspect_package(_future())["newer_than_known"])
        self.assertEqual(s.inspect_package(_future())["unsupported_requires"], ["holo-layout"])

    def test_a_newer_minor_that_adds_nothing_unknown_reads_normally(self) -> None:
        self.assertEqual(s.loads_package(_future(requires=(), version="1.2.0")).version, "1.2.0")


class VersionsAreCombinedUpwards(unittest.TestCase):
    def test_newest_version_compares_numbers_not_text(self) -> None:
        self.assertEqual(s.newest_version("1.0.0", "1.1.0"), "1.1.0")
        self.assertEqual(s.newest_version("1.9.0", "1.10.0", "1.2.3"), "1.10.0")
        with self.assertRaises(s.SLDCValidationError):
            s.newest_version("1.0.0", "latest")

    def test_merging_an_older_and_a_newer_copy_keeps_the_newer_features(self) -> None:
        base = s.new_package("Doc", [s.make_block("paragraph", "One", block_id="b1")], doc_id="doc-merge")
        ours = s.new_package("Doc", [s.make_block("paragraph", "One", block_id="b1"), s.make_block("paragraph", "Ours", block_id="b2")], doc_id="doc-merge")
        theirs = s.new_package("Doc", [s.make_block("paragraph", "One", block_id="b1"),
                                       s.make_block("equation", "", block_id="b3", equation={"syntax": "latex", "source": "x^2"})], doc_id="doc-merge")
        theirs.version = "1.1.0"
        merged, conflicts = s.merge_packages(base, ours, theirs)
        self.assertEqual(conflicts, [])
        self.assertEqual(merged.version, "1.1.0")
        s.validate_package(merged)
        self.assertIn("x^2", s.extract_text(merged))
        # and the other way round
        merged2, _ = s.merge_packages(base, theirs, ours)
        self.assertEqual(merged2.version, "1.1.0")


class RoundTripOfModernFeatures(unittest.TestCase):
    def test_a_modern_package_survives_write_and_read_with_its_version_and_page_bands(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.sldc"
            s.write_package(path, _modern_package())
            report = s.inspect_package(path)
            self.assertEqual(report["unsupported_requires"], [])
            self.assertIn("page-bands", report["requires"])
            again = s.read_package(path)
            self.assertEqual(again.version, "1.1.0")
            self.assertEqual(again.document["sections"][0]["page"]["header_mm"], 14)


if __name__ == "__main__":
    unittest.main()
