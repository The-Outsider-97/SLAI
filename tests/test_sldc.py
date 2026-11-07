"""Tests for ``src.utils.sldc``.

The first half covers SLDC 0.1.0, which is frozen: container (round-trip, determinism, atomic
writes), hostile or corrupted input (only ``SLDCError`` may ever escape the reader), validation, and
the SLAI-facing helpers (text/Markdown conversion, Reader bridge, digests, chunking). The second
half covers SLDC 1.0.0: the publishing profile, data-only guarantees, migration, collaboration,
signatures, encryption and the command line. Only temporary files are created; tests that need the
``cryptography`` package are skipped without it.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
import pickle
import random
import re
import struct
import sys
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from src.utils import sldc as sd


def sample_document(**overrides):
    document = sd.new_document(
        "Quarterly report",
        [
            sd.make_block("heading", "Summary", attributes={"level": 1}),
            sd.make_block("paragraph", "Revenue grew. Café ünïcödé ✓ 日本語"),
            sd.make_block("code", "print('hi')", attributes={"language": "python"}),
        ],
        language="en-GB",
        doc_id="doc-fixed",
        metadata={"owner": "finance", "nested": {"n": [1, 2.5, None, True]}},
        provenance=[sd.provenance_entry("test", timestamp="2026-01-01T00:00:00Z")],
    )
    document.update(overrides)
    return document


def build_raw(members, *, comment=b"") -> bytes:
    """Hand-build a ZIP from (name, bytes, compress_type) triples to forge hostile archives."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data, method in members:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = method
            archive.writestr(info, data)
        archive.comment = comment
    return buffer.getvalue()


def valid_members(document, *, payload=None, manifest_overrides=None):
    payload = sd._canonical_json(document) if payload is None else payload
    manifest = {
        "format": sd.FORMAT_ID, "version": sd.LEGACY_VERSION,
        "document_sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "document_size_bytes": len(payload),
    }
    manifest.update(manifest_overrides or {})
    return [
        ("mimetype", sd.MIMETYPE.encode("ascii"), zipfile.ZIP_STORED),
        ("manifest.json", sd._canonical_json(manifest), zipfile.ZIP_DEFLATED),
        ("document.json", payload, zipfile.ZIP_DEFLATED),
    ]


class _TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class ContainerTests(_TempDirTestCase):
    def test_file_round_trip_preserves_document(self) -> None:
        document = sample_document()
        path = sd.write_sldc(self.tmp / "a.sldc", document)
        self.assertEqual(sd.read_sldc(path), document)
        self.assertTrue(sd.is_sldc(path))
        self.assertTrue(sd.sniff_sldc(path))
        self.assertEqual(os.listdir(self.tmp), ["a.sldc"], "no temporary files may remain")

    def test_bytes_round_trip_and_read_ignores_suffix(self) -> None:
        document = sample_document()
        data = sd.dumps_sldc(document)
        self.assertEqual(sd.loads_sldc(data), document)
        self.assertEqual(sd.loads_sldc(memoryview(data)), document)
        renamed = self.tmp / "renamed.bin"
        renamed.write_bytes(data)
        self.assertEqual(sd.read_sldc(renamed), document)

    def test_output_is_deterministic_and_matches_digest(self) -> None:
        document = sample_document()
        first = sd.dumps_sldc(document)
        second = sd.dumps_sldc(dict(reversed(document.items())))  # different key order
        self.assertEqual(first, second)
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            self.assertEqual(archive.namelist(), list(sd.MEMBERS))
            manifest = __import__("json").loads(archive.read("manifest.json"))
        self.assertEqual(manifest["document_sha256"], sd.document_digest(document))

    def test_overwrite_false_never_replaces_existing_file(self) -> None:
        path = self.tmp / "a.sldc"
        path.write_bytes(b"precious")
        with self.assertRaises(FileExistsError):
            sd.write_sldc(path, sample_document())
        self.assertEqual(path.read_bytes(), b"precious")
        sd.write_sldc(path, sample_document(), overwrite=True)
        self.assertTrue(sd.is_sldc(path))
        self.assertEqual(os.listdir(self.tmp), ["a.sldc"])

    def test_filesystem_without_hard_links_still_works_and_still_refuses_overwrite(self) -> None:
        path = self.tmp / "a.sldc"
        with mock.patch("os.link", side_effect=OSError("hard links unsupported")):
            sd.write_sldc(path, sample_document())
            self.assertTrue(sd.is_sldc(path))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                sd.write_sldc(path, sample_document(title="Other"))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(os.listdir(self.tmp), ["a.sldc"])

    def test_failed_write_leaves_nothing_behind(self) -> None:
        bad = sample_document(blocks=[{"id": "x", "type": "paragraph", "text": 5}])
        with self.assertRaises(sd.SLDCValidationError):
            sd.write_sldc(self.tmp / "a.sldc", bad)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_wrong_suffix_and_missing_directory_are_rejected(self) -> None:
        with self.assertRaises(sd.SLDCValidationError):
            sd.write_sldc(self.tmp / "a.json", sample_document())
        with self.assertRaises(sd.SLDCValidationError):
            sd.write_sldc(self.tmp / "missing" / "a.sldc", sample_document())

    def test_missing_file_is_an_oserror_and_is_sldc_is_false(self) -> None:
        with self.assertRaises(FileNotFoundError):
            sd.read_sldc(self.tmp / "nope.sldc")
        self.assertFalse(sd.is_sldc(self.tmp / "nope.sldc"))
        self.assertFalse(sd.sniff_sldc(self.tmp / "nope.sldc"))
        with self.assertRaises(sd.SLDCValidationError):
            sd.read_sldc(self.tmp)  # a directory is not a regular file

    def test_unwritable_repetitive_document_fails_on_write_not_on_read(self) -> None:
        bomb = sd.new_document("Repetitive", [sd.make_block("paragraph", "a" * 3_000_000)])
        with self.assertRaises(sd.SLDCValidationError) as caught:
            sd.dumps_sldc(bomb)
        self.assertIn("compression-ratio", str(caught.exception))


class HostileInputTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.document = sample_document()
        self.good = sd.dumps_sldc(self.document)

    def test_non_sldc_inputs_raise_validation_error(self) -> None:
        for blob in (b"", b"hello", b"%PDF-1.7\n" + b"x" * 100, b"PK\x03\x04" + b"\0" * 60):
            with self.assertRaises(sd.SLDCValidationError):
                sd.loads_sldc(blob)

    def test_ordinary_zip_with_other_members_is_rejected(self) -> None:
        blob = build_raw([("a.txt", b"hi", zipfile.ZIP_DEFLATED)])
        with self.assertRaises(sd.SLDCError):
            sd.loads_sldc(blob)

    def test_prepended_data_trailing_comment_and_extra_entries_are_rejected(self) -> None:
        with self.assertRaises(sd.SLDCError):
            sd.loads_sldc(b"junk" + self.good)
        members = valid_members(self.document)
        with self.assertRaises(sd.SLDCError):
            sd.loads_sldc(build_raw(members, comment=b"hidden"))
        with self.assertRaises(sd.SLDCError):
            sd.loads_sldc(build_raw(members + [("extra.bin", b"x", zipfile.ZIP_STORED)]))
        with self.assertRaises(sd.SLDCError):
            sd.loads_sldc(build_raw([members[1], members[0], members[2]]))

    def test_compressed_mimetype_is_rejected(self) -> None:
        members = valid_members(self.document)
        members[0] = (members[0][0], members[0][1], zipfile.ZIP_DEFLATED)
        with self.assertRaises(sd.SLDCError):
            sd.loads_sldc(build_raw(members))

    def test_tampered_document_fails_integrity(self) -> None:
        members = valid_members(self.document)
        tampered = sd._canonical_json(sample_document(title="Tampered"))
        members[2] = ("document.json", tampered, zipfile.ZIP_DEFLATED)
        with self.assertRaises(sd.SLDCIntegrityError):
            sd.loads_sldc(build_raw(members))

    def test_wrong_size_hash_and_version_in_manifest(self) -> None:
        cases = [
            ({"document_size_bytes": 1}, sd.SLDCIntegrityError),
            ({"document_size_bytes": True}, sd.SLDCIntegrityError),
            ({"document_sha256": "0" * 64}, sd.SLDCIntegrityError),
            ({"version": "9.9.9"}, sd.SLDCValidationError),
            ({"format": "other"}, sd.SLDCValidationError),
        ]
        for override, expected in cases:
            blob = build_raw(valid_members(self.document, manifest_overrides=override))
            with self.subTest(override=override), self.assertRaises(expected):
                sd.loads_sldc(blob)

    def test_duplicate_keys_nan_and_non_object_payloads(self) -> None:
        for payload in (b'{"id":"a","id":"b"}', b'{"x":NaN}', b"[1,2]", b"\xff\xfe", b""):
            members = valid_members(self.document, payload=payload)
            with self.subTest(payload=payload), self.assertRaises(sd.SLDCValidationError):
                sd.loads_sldc(build_raw(members))

    def test_huge_integer_literal_never_leaks_a_bare_valueerror(self) -> None:
        payload = b'{"metadata":{"n":' + b"9" * 6000 + b"}}"
        try:
            sd.loads_sldc(build_raw(valid_members(self.document, payload=payload)))
        except sd.SLDCError:
            pass  # Python >= 3.11 limits integer digits; older versions fail field validation.

    def test_deeply_nested_json_is_a_validation_error(self) -> None:
        depth = 5000
        payload = b'{"metadata":' + b"[" * depth + b"]" * depth + b"}"
        with self.assertRaises(sd.SLDCValidationError):
            sd.loads_sldc(build_raw(valid_members(self.document, payload=payload)))
        nested: dict = {}
        for _ in range(sd.MAX_JSON_DEPTH + 1):
            nested = {"k": nested}
        with self.assertRaises(sd.SLDCValidationError):
            sd.validate_sldc(sample_document(metadata=nested))

    def test_corrupted_deflate_stream_raises_sldc_error_not_zlib_error(self) -> None:
        with zipfile.ZipFile(io.BytesIO(self.good)) as archive:
            info = archive.getinfo("document.json")
        start = info.header_offset + 30 + len(info.filename)
        damaged = bytearray(self.good)
        for offset in range(start, start + min(info.compress_size, 24)):
            damaged[offset] ^= 0xFF
        with self.assertRaises(sd.SLDCError):
            sd.loads_sldc(bytes(damaged))
        self.assertFalse(sd.is_sldc(self._write("damaged.sldc", bytes(damaged))))

    def test_truncated_archive_is_rejected_at_every_length(self) -> None:
        for length in range(0, len(self.good), 7):
            with self.subTest(length=length), self.assertRaises(sd.SLDCError):
                sd.loads_sldc(self.good[:length])

    def test_oversized_archive_is_rejected_before_parsing(self) -> None:
        with self.assertRaises(sd.SLDCValidationError):
            sd.loads_sldc(self.good + b"\0" * (sd.MAX_ARCHIVE_BYTES + 1))

    def test_fuzzed_archives_only_ever_raise_sldc_errors(self) -> None:
        rng = random.Random(20261020)
        for _ in range(400):
            blob = bytearray(self.good)
            for _ in range(rng.randint(1, 4)):
                blob[rng.randrange(len(blob))] = rng.randrange(256)
            if rng.random() < 0.3:
                del blob[rng.randrange(len(blob)):]
            try:
                result = sd.loads_sldc(bytes(blob))
            except sd.SLDCError:
                continue
            self.assertIsInstance(result, dict)  # A harmless flip (e.g. a timestamp byte).

    def _write(self, name: str, data: bytes) -> Path:
        path = self.tmp / name
        path.write_bytes(data)
        return path


class ValidationTests(unittest.TestCase):
    def check_rejects(self, **overrides) -> None:
        with self.assertRaises(sd.SLDCValidationError):
            sd.validate_sldc(sample_document(**overrides))

    def test_valid_document_and_read_only_mapping_pass(self) -> None:
        sd.validate_sldc(sample_document())
        sd.validate_sldc(types.MappingProxyType(sample_document()))
        sd.validate_sldc(sd.new_document("Empty"))

    def test_unhashable_block_type_is_a_validation_error_not_typeerror(self) -> None:
        for bad in ([], {}, None, 3):
            self.check_rejects(blocks=[{"id": "b1", "type": bad, "text": "x"}])

    def test_structural_errors_are_rejected(self) -> None:
        good = {"id": "b1", "type": "paragraph", "text": "x"}
        self.check_rejects(id="has space")
        self.check_rejects(title="   ")
        self.check_rejects(language="English language tag")
        self.check_rejects(blocks="nope")
        self.check_rejects(blocks=[good, dict(good)])
        self.check_rejects(blocks=[{**good, "surprise": 1}])
        self.check_rejects(blocks=[{**good, "annotations": [1]}])
        self.check_rejects(blocks=[{**good, "attributes": []}])
        self.check_rejects(blocks=[{**good, "source_ref": "x" * 5000}])
        self.check_rejects(blocks=[good] * 2)
        self.check_rejects(metadata=[])
        self.check_rejects(provenance=["x"])
        self.check_rejects(extra_key=1)

    def test_non_json_values_are_rejected_instead_of_silently_changed(self) -> None:
        self.check_rejects(metadata={1: "int key"})
        self.check_rejects(metadata={"nan": float("nan")})
        self.check_rejects(metadata={"inf": float("inf")})
        self.check_rejects(metadata={"tuple": (1, 2)})
        self.check_rejects(metadata={"set": {1}})
        self.check_rejects(metadata={"bytes": b"x"})
        self.check_rejects(title="lone surrogate \ud800")

    def test_too_many_blocks_and_too_much_text(self) -> None:
        blocks = [{"id": f"b{i}", "type": "paragraph", "text": ""}
                  for i in range(sd.MAX_BLOCKS + 1)]
        self.check_rejects(blocks=blocks)
        with mock.patch.object(sd, "MAX_TEXT_CHARS", 10):
            self.check_rejects(blocks=[{"id": "b1", "type": "paragraph", "text": "x" * 11}])

    def test_extract_text_joins_blocks_with_blank_lines(self) -> None:
        document = sd.new_document("T", [sd.make_block("paragraph", "one"),
                                         sd.make_block("quote", "two")])
        self.assertEqual(sd.extract_text(document), "one\n\ntwo")
        with self.assertRaises(sd.SLDCValidationError):
            sd.extract_text({"id": "x"})


class TextConversionTests(unittest.TestCase):
    def test_plain_text_paragraphs_and_line_endings(self) -> None:
        blocks = sd.blocks_from_text("First line\r\nsecond line\r\n\r\n\r\n  Next para  \n\n\n")
        self.assertEqual([b["text"] for b in blocks], ["First line\nsecond line", "Next para"])
        self.assertEqual([b["id"] for b in blocks], ["b00001", "b00002"])
        self.assertEqual(sd.blocks_from_text(""), [])
        self.assertEqual(sd.blocks_from_text("   \n\n  "), [])

    def test_markdown_blocks(self) -> None:
        markdown = (
            "# Title #\n\nIntro line one\nintro line two\n\n"
            "## Lists\n- alpha\n  continued\n  - nested\n1. first\n2) second\n\n"
            "> quoted\n> text\n\n"
            "```python\nx = 1\n\nprint(x)\n```\n\n"
            "~~~\nunclosed fence\n* * *\nAfter rule\n"
        )
        blocks = sd.blocks_from_text(markdown, markdown=True, source_ref="notes.md")
        summary = [(b["type"], b["text"], b.get("attributes")) for b in blocks]
        self.assertEqual(summary[0], ("heading", "Title", {"level": 1}))
        self.assertEqual(summary[1], ("paragraph", "Intro line one\nintro line two", None))
        self.assertEqual(summary[2], ("heading", "Lists", {"level": 2}))
        self.assertEqual(summary[3], ("list_item", "alpha\ncontinued", {"ordered": False}))
        self.assertEqual(summary[4], ("list_item", "nested", {"ordered": False, "depth": 1}))
        self.assertEqual(summary[5], ("list_item", "first", {"ordered": True}))
        self.assertEqual(summary[6], ("list_item", "second", {"ordered": True}))
        self.assertEqual(summary[7], ("quote", "quoted\ntext", None))
        self.assertEqual(summary[8], ("code", "x = 1\n\nprint(x)", {"language": "python"}))
        self.assertEqual(summary[9][0:2], ("code", "unclosed fence\n* * *\nAfter rule"))
        self.assertTrue(all(b["source_ref"] == "notes.md" for b in blocks))
        sd.validate_sldc(sd.new_document("T", blocks))

    def test_hash_in_text_is_not_a_heading_and_c_sharp_survives(self) -> None:
        blocks = sd.blocks_from_text("#hashtag\n\n# C#\n", markdown=True)
        self.assertEqual([(b["type"], b["text"]) for b in blocks],
                         [("paragraph", "#hashtag"), ("heading", "C#")])

    def test_markdown_round_trip_keeps_types_and_text(self) -> None:
        source = ("# Head\n\nPara text\n\n- one\n- two\n  - deep\n\n> quote\n\n"
                  "````\nfence ``` inside\n````\n")
        first = sd.new_document("T", sd.blocks_from_text(source, markdown=True))
        rendered = sd.to_markdown(first)
        second = sd.new_document("T", sd.blocks_from_text(rendered, markdown=True))
        project = lambda doc: [(b["type"], b["text"]) for b in doc["blocks"]]  # noqa: E731
        self.assertEqual(project(first), project(second))
        self.assertIn("````", rendered)

    def test_id_prefix_is_validated(self) -> None:
        with self.assertRaises(sd.SLDCValidationError):
            sd.blocks_from_text("x", id_prefix="bad prefix")
        with self.assertRaises(sd.SLDCValidationError):
            sd.blocks_from_text(b"bytes")  # type: ignore[arg-type]

    def test_new_document_assigns_ids_without_clashing(self) -> None:
        document = sd.new_document("T", [
            sd.make_block("paragraph", "a", block_id="b00001"),
            sd.make_block("paragraph", "b"),
            sd.make_block("paragraph", "c"),
        ])
        self.assertEqual([b["id"] for b in document["blocks"]], ["b00001", "b00002", "b00003"])
        sd.validate_sldc(document)


class ReaderBridgeTests(unittest.TestCase):
    RAW, CONTENT = "a" * 64, "b" * 64

    def reader_result(self, **overrides):
        result = {
            "status": "ok", "source": r"G:\Users\jean\secret\report.md", "filename": "report.md",
            "stem": "report", "extension": ".md", "content_type": "text/markdown",
            "parser": "text_parser", "content": "# Report\n\nBody text.",
            "metadata": {"sha256": self.RAW, "raw_sha256": self.RAW, "content_sha256": self.CONTENT,
                         "timing": object()},  # non-JSON values must not be copied
            "warnings": ["w1", "w2"],
        }
        result.update(overrides)
        return result

    def test_bridge_builds_valid_document_with_provenance_and_no_local_path(self) -> None:
        document = sd.document_from_reader_result(self.reader_result())
        sd.validate_sldc(document)
        self.assertEqual(document["title"], "report")
        self.assertEqual(document["blocks"][0]["type"], "heading")  # .md auto-detected
        self.assertEqual(document["blocks"][0]["source_ref"], "report.md")
        entry = document["provenance"][0]
        self.assertEqual(entry["activity"], "reader.parse")
        self.assertEqual((entry["raw_sha256"], entry["content_sha256"]), (self.RAW, self.CONTENT))
        self.assertEqual(entry["warning_count"], 2)
        self.assertNotIn("secret", sd._canonical_json(document).decode("utf-8"))
        sd.loads_sldc(sd.dumps_sldc(document))

    def test_bridge_handles_windows_source_without_filename_and_plain_text(self) -> None:
        result = self.reader_result(filename=None, stem=None, extension=None,
                                    content_type="text/plain", content="Just text.")
        document = sd.document_from_reader_result(result, title="Custom")
        self.assertEqual(document["title"], "Custom")
        self.assertEqual(document["blocks"][0]["source_ref"], "report.md")
        self.assertEqual(document["blocks"][0]["type"], "paragraph")

    def test_bridge_ignores_malformed_digests_and_rejects_bad_results(self) -> None:
        result = self.reader_result(metadata={"sha256": "not-a-digest"})
        entry = sd.document_from_reader_result(result)["provenance"][0]
        self.assertNotIn("raw_sha256", entry)
        for bad in (None, {}, {"content": 5}, "text"):
            with self.assertRaises(sd.SLDCValidationError):
                sd.document_from_reader_result(bad)  # type: ignore[arg-type]


class SlaiHelperTests(unittest.TestCase):
    def test_digests(self) -> None:
        document = sample_document()
        self.assertEqual(sd.document_digest(document), sd.document_digest(sample_document()))
        self.assertNotEqual(sd.document_digest(document),
                            sd.document_digest(sample_document(title="Other")))
        first = {"id": "a", "type": "paragraph", "text": "same"}
        moved = {"id": "z", "type": "paragraph", "text": "same", "source_ref": "elsewhere"}
        self.assertEqual(sd.block_digest(first), sd.block_digest(moved))
        self.assertNotEqual(sd.block_digest(first), sd.block_digest({**first, "type": "quote"}))

    def test_block_index_resolves_citations(self) -> None:
        document = sample_document()
        index = sd.block_index(document)
        self.assertEqual(index["b00002"]["type"], "paragraph")
        self.assertEqual(list(index), [b["id"] for b in document["blocks"]])

    def test_chunks_respect_limit_keep_block_ids_and_lose_no_text(self) -> None:
        blocks = [sd.make_block("paragraph", f"Block {i} " + "word " * (i % 7)) for i in range(60)]
        document = sd.new_document("Chunky", blocks)
        chunks = sd.chunk_document(document, max_chars=120)
        self.assertGreater(len(chunks), 5)
        self.assertEqual([c["index"] for c in chunks], list(range(len(chunks))))
        self.assertTrue(all(0 < len(c["text"]) <= 120 and not c["partial"] for c in chunks))
        flattened = [bid for c in chunks for bid in c["block_ids"]]
        self.assertEqual(flattened, [b["id"] for b in document["blocks"]])
        self.assertEqual("\n\n".join(c["text"] for c in chunks).split(),
                         sd.extract_text(document).split())

    def test_oversized_block_is_split_at_whitespace_and_flagged_partial(self) -> None:
        text = " ".join(f"w{i:03d}" for i in range(200))
        document = sd.new_document("Long", [sd.make_block("paragraph", "intro"),
                                             sd.make_block("paragraph", text)])
        chunks = sd.chunk_document(document, max_chars=100)
        self.assertEqual(chunks[0]["block_ids"], ["b00001"])
        pieces = [c for c in chunks if c["partial"]]
        self.assertTrue(pieces and all(c["block_ids"] == ["b00002"] for c in pieces))
        self.assertTrue(all(len(c["text"]) <= 100 for c in chunks))
        self.assertEqual(" ".join(c["text"] for c in pieces).split(), text.split())
        solid = sd.new_document("Hard", [sd.make_block("paragraph", "x" * 25)])
        hard = sd.chunk_document(solid, 10)
        self.assertEqual([len(c["text"]) for c in hard], [10, 10, 5])

    def test_chunk_arguments_are_checked(self) -> None:
        document = sd.new_document("T", [sd.make_block("paragraph", "x")])
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                sd.chunk_document(document, bad)  # type: ignore[arg-type]
        self.assertEqual(sd.chunk_document(sd.new_document("Empty")), [])

    def test_to_markdown_of_empty_document_and_odd_attributes(self) -> None:
        self.assertEqual(sd.to_markdown(sd.new_document("Empty")), "")
        document = sd.new_document("T", [
            sd.make_block("heading", "No level"),
            sd.make_block("heading", "Bad level", attributes={"level": 99}),
            sd.make_block("code", "x", attributes={"language": "has space"}),
        ])
        self.assertEqual(sd.to_markdown(document), "# No level\n\n# Bad level\n\n```\nx\n```\n")


class AuditRegressionTests(_TempDirTestCase):
    """Defects found in the SLDC integration audit (2026-10-20); each failed before the fix."""

    def test_small_highly_compressible_document_is_accepted(self) -> None:
        # Space-padded fixed-width text compresses far beyond 200:1 yet is tiny and harmless.
        padded = "\n\n".join(f"Item {i}" + " " * 4000 + "total" for i in range(60))
        document = sd.new_document("Padded report", [sd.make_block("paragraph", padded)])
        self.assertEqual(sd.loads_sldc(sd.dumps_sldc(document)), document)

    def test_large_expansion_is_still_rejected(self) -> None:
        bomb = sd.new_document("Bomb", [sd.make_block("paragraph", "a" * 3_000_000)])
        with self.assertRaises(sd.SLDCValidationError):
            sd.dumps_sldc(bomb)

    def test_reading_a_small_file_does_not_allocate_the_archive_limit(self) -> None:
        import tracemalloc
        path = sd.write_sldc(self.tmp / "tiny.sldc", sample_document())
        tracemalloc.start()
        sd.read_sldc(path)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 1_000_000, f"read_sldc peaked at {peak} bytes for a tiny file")

    def test_reader_bridge_is_deterministic_for_identical_input(self) -> None:
        result = {"content": "# T\n\nBody", "filename": "a.md", "stem": "a", "extension": ".md",
                  "metadata": {"parsed_at": "2026-10-20T10:00:00+00:00"}}
        first = sd.document_from_reader_result(result)
        second = sd.document_from_reader_result(dict(result))
        self.assertEqual(sd.document_digest(first), sd.document_digest(second))
        self.assertEqual(sd.dumps_sldc(first), sd.dumps_sldc(second))
        changed = sd.document_from_reader_result({**result, "content": "# T\n\nOther"})
        self.assertNotEqual(first["id"], changed["id"])
        self.assertEqual(first["provenance"][0]["timestamp"], "2026-10-20T10:00:00Z")

    def test_text_builder_ids_are_content_derived_unless_given(self) -> None:
        a = sd.document_from_text("x\n\ny", title="T")
        self.assertEqual(a["id"], sd.document_from_text("x\n\ny", title="T")["id"])
        self.assertNotEqual(a["id"], sd.document_from_text("x\n\nz", title="T")["id"])
        self.assertEqual(sd.document_from_text("x", title="T", doc_id="mine")["id"], "mine")
        self.assertTrue(sd.new_document("T")["id"].startswith("doc-"))  # generic builder unchanged


class CommandLineTests(_TempDirTestCase):
    def run_cli(self, *argv: str):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = sd.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_pack_check_and_text(self) -> None:
        source = self.tmp / "notes.md"
        source.write_text("# Notes\n\nHello wörld.\n", encoding="utf-8")
        target = self.tmp / "notes.sldc"
        code, out, _ = self.run_cli("pack", str(source), str(target))
        self.assertEqual(code, 0)
        package = sd.read_package(target)
        self.assertEqual(package.version, sd.FORMAT_VERSION)
        self.assertEqual(package.document["title"], "notes")
        self.assertEqual(package.document["provenance"][0]["activity"], "sldc.pack")
        code, out, _ = self.run_cli("check", str(target))
        self.assertEqual(code, 0)
        self.assertIn("OK", out)
        code, out, _ = self.run_cli("text", str(target))
        self.assertEqual((code, out), (0, "Notes\n\nHello wörld.\n"))
        code, out, _ = self.run_cli("text", "--markdown", str(target))
        self.assertEqual(out, "# Notes\n\nHello wörld.\n")

    def test_pack_refuses_to_overwrite_and_check_reports_failures(self) -> None:
        source = self.tmp / "a.txt"
        source.write_text("x", encoding="utf-8")
        target = self.tmp / "a.sldc"
        self.assertEqual(self.run_cli("pack", str(source), str(target))[0], 0)
        code, _, err = self.run_cli("pack", str(source), str(target))
        self.assertEqual(code, 1)
        self.assertIn("error", err)
        bogus = self.tmp / "bogus.sldc"
        bogus.write_bytes(b"nope")
        code, out, _ = self.run_cli("check", str(target), str(bogus), str(self.tmp / "gone"))
        self.assertEqual(code, 1)
        self.assertEqual(out.count("FAIL"), 2)
        self.assertEqual(out.count("OK"), 1)


# ----------------------------------------------------------------------------------------------
# SLDC 1.0 (publishing profile)
# ----------------------------------------------------------------------------------------------
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154"
    "78da63fcffff3f0300050001a5f645400000000049454e44ae426082")


def png_header(width: int, height: int) -> bytes:
    """A PNG signature + IHDR only: enough for the header parser, not decodable."""
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
            + struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00" + b"\x00" * 4)


def rich_package() -> sd.SLDCPackage:
    """One document that uses every 1.0 feature."""
    blocks = [
        sd.make_block("heading", "Annual report", block_id="h1", attributes={"level": 1},
                      style="Title"),
        sd.make_block("paragraph", "Revenue grew 12% (see Smith 2025).", block_id="p1",
                      section="body",
                      spans=[{"type": "format", "start": 0, "end": 7, "bold": True},
                             {"type": "ref", "start": 24, "end": 33, "ref": "smith25"},
                             {"type": "link", "start": 8, "end": 13, "href": "https://example.org/a"},
                             {"type": "style", "start": 14, "end": 16, "style": "Emphasis"},
                             {"type": "xref", "start": 17, "end": 20, "target": "t1"}]),
        sd.make_table([["Quarter", "Revenue"], ["Q1", "10"], [{"text": "Total", "colspan": 2}]],
                      header_rows=1, caption="Revenue by quarter", block_id="t1"),
        sd.make_figure("logo", "Company logo", caption="Figure 1", block_id="f1", width_mm=40,
                       align="center"),
        sd.make_block("embed", "Chart: revenue by quarter", block_id="e1", embed="chart1"),
        sd.make_block("paragraph", "Footnoted sentence.", block_id="p2", section="body",
                      spans=[{"type": "ref", "start": 0, "end": 18, "ref": "fn1"}]),
    ]
    return sd.new_package(
        "Annual report", blocks, language="en-GB", doc_id="doc-rich",
        publication={"subtitle": "FY2025", "authors": ["A. Writer"], "publisher": "SLAI Press",
                     "published": "2026-01-15", "keywords": ["finance"],
                     "identifier": {"scheme": "isbn", "value": "978-0-00-000000-2"},
                     "cover_asset": "logo"},
        metadata={"owner": "finance"},
        provenance=[sd.provenance_entry("test", timestamp="2026-01-01T00:00:00Z")],
        styles={"paragraph": {"Title": {"font_size_pt": 28, "bold": True, "align": "center"},
                              "Body": {"font_family": "Georgia", "line_height": 1.4,
                                       "color": "#222222"}},
                "character": {"Emphasis": {"italic": True, "color": "#aa0000"}}},
        sections=[{"id": "body", "page": {"size": "A4", "margins_mm": {"top": 20, "bottom": 20,
                                                                         "left": 25, "right": 25}},
                   "columns": 2, "footer": {"center": "{title} - page {page} of {pages}"},
                   "numbering": {"start": 1, "format": "decimal"}}],
        references=[{"id": "smith25", "kind": "citation", "title": "Finance in 2025",
                     "authors": ["Smith, J."], "year": 2025, "doi": "10.1234/abc.5678",
                     "url": "https://example.org/paper"},
                    {"id": "fn1", "kind": "footnote", "text": "Unaudited."}],
        embeds=[{"id": "chart1", "kind": "chart.bar", "schema_version": 1,
                 "params": {"series": [1, 2, 3], "note": "__import__('os').system('x')"},
                 "inputs": [{"ref": "t1", "sha256": "0" * 64}],
                 "output_sha256": "1" * 64, "capabilities": ["render"],
                 "policy_ref": "org-policy-7"}],
        assets=[sd.make_asset(PNG_1X1, asset_id="logo")],
        extensions={"acme.workflow": {"state": "draft"}})


def small_package(*texts: str) -> sd.SLDCPackage:
    texts = texts or ("one", "two", "three")
    return sd.new_package(
        "Small", [sd.make_block("paragraph", t, block_id=f"b{i}") for i, t in enumerate(texts, 1)],
        doc_id="doc-small",
        provenance=[sd.provenance_entry("test", timestamp="2026-01-01T00:00:00Z")])


def reseal(data: bytes, *, manifest=None, drop=(), replace=None, add=None) -> bytes:
    """Rebuild an SLDC archive with a modified manifest and/or parts, to forge hostile files."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        contents = {i.filename: archive.read(i) for i in archive.infolist()}
    manifest_data = json.loads(contents["manifest.json"])
    if callable(manifest):
        manifest(manifest_data)
    elif manifest:
        manifest_data.update(manifest)
    contents["manifest.json"] = sd._canonical_json(manifest_data)
    for name in drop:
        contents.pop(name)
    contents.update(replace or {})
    contents.update(add or {})
    order = ["mimetype", "manifest.json"] + [n for n in contents if n not in ("mimetype", "manifest.json")]
    return build_raw([(n, contents[n], zipfile.ZIP_STORED if n == "mimetype" else zipfile.ZIP_DEFLATED)
                      for n in order])


def refresh_part(data: bytes, name: str, raw: bytes) -> bytes:
    """Replace a part and keep the manifest's size, hash and revision consistent with it."""
    def fix(manifest):
        for entry in manifest["parts"]:
            if entry["name"] == name:
                entry.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        content = [(e["name"], e["size"], e["sha256"]) for e in manifest["parts"]
                   if e["role"] in ("content", "asset")]
        manifest["revision_id"] = sd._revision_digest(content)
    return reseal(data, manifest=fix, replace={name: raw})


class PackageFormatTests(_TempDirTestCase):
    def test_round_trip_is_deterministic_and_lossless(self) -> None:
        package = rich_package()
        data = sd.dumps_package(package)
        self.assertEqual(data, sd.dumps_package(package))
        back = sd.loads_package(data)
        self.assertEqual(back.version, sd.FORMAT_VERSION)
        self.assertEqual(back.document, package.document)
        self.assertEqual(back.assets["logo"].data, PNG_1X1)
        self.assertEqual(sd.dumps_package(back), data)
        self.assertEqual(sd.revision_id(back), sd.revision_id(package))
        self.assertEqual(sd.inspect_package(data)["requires"], ["embeds", "figures", "tables"])

    def test_mimetype_header_is_shared_with_every_version(self) -> None:
        data = sd.dumps_package(rich_package())
        self.assertTrue(data.startswith(b"PK\x03\x04"))
        self.assertIn(sd.MIMETYPE.encode(), data[:200])
        target = self.tmp / "rich.sldc"
        sd.write_package(target, rich_package())
        self.assertTrue(sd.sniff_sldc(target))
        self.assertEqual(sd.read_package(target).document["id"], "doc-rich")

    def test_write_package_never_overwrites_unless_asked(self) -> None:
        target = self.tmp / "p.sldc"
        sd.write_package(target, small_package())
        with self.assertRaises(FileExistsError):
            sd.write_package(target, small_package("changed"))
        self.assertEqual(sd.read_package(target).document["blocks"][0]["text"], "one")
        sd.write_package(target, small_package("changed"), overwrite=True)
        self.assertEqual(sd.read_package(target).document["blocks"][0]["text"], "changed")
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["p.sldc"])  # No temp files left.

    def test_legacy_api_refuses_1x_and_package_api_reads_0_1(self) -> None:
        data = sd.dumps_package(small_package())
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.loads_sldc(data)
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.write_sldc(self.tmp / "x.sldc", sd.loads_package(data).document)
        legacy = sd.dumps_sldc(sample_document())
        package = sd.loads_package(legacy)
        self.assertEqual((package.version, package.document), (sd.LEGACY_VERSION, sample_document()))
        self.assertEqual(sd.dumps_package(package), legacy)  # Byte-for-byte as before.
        self.assertTrue(sd.is_sldc(self.tmp / "none.sldc") is False)

    def test_legacy_package_cannot_carry_1x_content(self) -> None:
        package = sd.SLDCPackage(sample_document(), version=sd.LEGACY_VERSION)
        package.assets["a"] = sd.make_asset(PNG_1X1, asset_id="a")
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.dumps_package(package)

    def test_inspect_reads_the_manifest_only(self) -> None:
        data = sd.dumps_package(rich_package())
        report = sd.inspect_package(data)
        self.assertEqual((report["container"], report["version"], report["supported"]),
                         ("package", sd.FORMAT_VERSION, True))
        self.assertEqual(report["document_id"], "doc-rich")
        self.assertEqual(report["revision_id"], sd.revision_id(rich_package()))
        self.assertEqual([p["name"] for p in report["parts"]][:2], ["document.json", "assets/logo.png"])
        # A corrupted payload is invisible to inspect (it never reads parts) but not to the reader.
        broken = refresh_part(data, "document.json", b"{}")
        broken = reseal(data, replace={"document.json": b"{}"})
        self.assertEqual(sd.inspect_package(broken)["document_id"], "doc-rich")
        with self.assertRaises(sd.SLDCIntegrityError):
            sd.loads_package(broken)
        target = self.tmp / "r.sldc"
        target.write_bytes(data)
        self.assertEqual(sd.inspect_package(target)["revision_id"], report["revision_id"])
        self.assertEqual(sd.inspect_package(sd.dumps_sldc(sample_document()))["container"], "legacy")

    def test_unknown_required_feature_is_reported_and_refused_not_dropped(self) -> None:
        data = reseal(sd.dumps_package(small_package()), manifest={"requires": ["quantum-layout"]})
        report = sd.inspect_package(data)
        self.assertFalse(report["supported"])
        self.assertEqual(report["unsupported_requires"], ["quantum-layout"])
        with self.assertRaises(sd.SLDCUnsupportedError) as caught:
            sd.loads_package(data)
        self.assertIn("quantum-layout", str(caught.exception))

    def test_manifest_must_declare_features_the_document_uses(self) -> None:
        data = reseal(sd.dumps_package(rich_package()), manifest={"requires": []})
        with self.assertRaises(sd.SLDCValidationError):
            sd.loads_package(data)

    def test_versions_newer_minor_ok_newer_major_refused(self) -> None:
        data = sd.dumps_package(small_package())
        self.assertEqual(sd.loads_package(reseal(data, manifest={"version": "1.0.0"})).version,
                         sd.FORMAT_VERSION)
        for version in ("2.0.0", "0.2.0", "1.0", "x", "1.0.0-beta"):
            with self.subTest(version=version), self.assertRaises(sd.SLDCValidationError):
                sd.loads_package(reseal(data, manifest={"version": version}))

    def test_manifest_extra_keys_only_with_x_prefix(self) -> None:
        data = sd.dumps_package(small_package())
        self.assertEqual(sd.loads_package(reseal(data, manifest={"x-acme-note": {"a": 1}})).version,
                         sd.FORMAT_VERSION)
        with self.assertRaises(sd.SLDCValidationError):
            sd.loads_package(reseal(data, manifest={"acme": 1}))
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.loads_package(reseal(data, manifest={"profile": "slides"}))

    def test_corruption_and_structure_attacks(self) -> None:
        data = sd.dumps_package(rich_package())
        document = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("document.json"))
        cases = {
            "part edited": (lambda: reseal(data, replace={"document.json": b'{"x":1}'}),
                            sd.SLDCIntegrityError),
            "image edited": (lambda: reseal(data, replace={"assets/logo.png": PNG_1X1 + b"x"}),
                             sd.SLDCIntegrityError),
            "undeclared part": (lambda: reseal(data, add={"assets/evil.png": PNG_1X1}),
                                sd.SLDCValidationError),
            "missing part": (lambda: reseal(data, drop=["assets/logo.png"]), sd.SLDCValidationError),
            "script part": (lambda: reseal(data, add={"scripts/run.js": b"alert(1)"}),
                            sd.SLDCValidationError),
            "revision lies": (lambda: reseal(data, manifest={"revision_id": "0" * 64}),
                              sd.SLDCIntegrityError),
            "document id lies": (lambda: reseal(data, manifest={"document_id": "other"}),
                                 sd.SLDCIntegrityError),
            "path traversal": (lambda: reseal(data, manifest=lambda m: m["parts"][0].update(
                name="../document.json")), sd.SLDCValidationError),
            "wrong media type": (lambda: reseal(data, manifest=lambda m: m["parts"][1].update(
                media_type="image/svg+xml")), sd.SLDCValidationError),
            "parts unsorted": (lambda: reseal(data, manifest=lambda m: m["parts"].reverse()),
                               sd.SLDCValidationError),
            "renamed image type": (lambda: refresh_part(
                data, "assets/logo.png", b"<svg xmlns='http://www.w3.org/2000/svg'/>"),
                sd.SLDCValidationError),
        }
        for label, (forge, expected) in cases.items():
            with self.subTest(label), self.assertRaises(expected):
                sd.loads_package(forge())
        self.assertEqual(document["id"], "doc-rich")

    def test_revision_ignores_comments_and_signatures_but_not_content_or_assets(self) -> None:
        package = small_package()
        base = sd.revision_id(package)
        commented, _ = sd.add_comment(package, "b1", "Check this", author="Ann")
        self.assertEqual(sd.revision_id(commented), base)
        edited = package.copy()
        edited.document["blocks"][0]["text"] = "ONE"
        self.assertNotEqual(sd.revision_id(edited), base)
        self.assertNotEqual(sd.revision_id(rich_package()), sd.revision_id(small_package()))

    def test_assets_must_be_used_and_must_exist(self) -> None:
        package = small_package()
        package.assets["lonely"] = sd.make_asset(PNG_1X1, asset_id="lonely")
        with self.assertRaisesRegex(sd.SLDCValidationError, "no figure or cover"):
            sd.validate_package(package)
        self.assertEqual(sd.prune_assets(package).assets, {})
        figure = sd.new_package("T", [sd.make_figure("gone", "x")])
        with self.assertRaisesRegex(sd.SLDCValidationError, "missing asset"):
            sd.validate_package(figure)
        added = sd.add_asset(figure, sd.make_asset(PNG_1X1, asset_id="gone"))
        sd.validate_package(added)
        self.assertEqual(figure.assets, {})  # add_asset returned a copy.
        with self.assertRaises(sd.SLDCValidationError):
            sd.add_asset(added, sd.Asset("gone", "image/png", PNG_1X1 + b"x"))

    def test_package_copy_is_independent(self) -> None:
        package = rich_package()
        clone = package.copy()
        clone.document["blocks"][0]["text"] = "changed"
        clone.assets.clear()
        self.assertEqual(package.document["blocks"][0]["text"], "Annual report")
        self.assertIn("logo", package.assets)

    def test_functions_do_not_mutate_their_inputs(self) -> None:
        package = rich_package()
        before = copy.deepcopy(package.document)
        sd.next_revision(package)
        sd.add_comment(package, "p1", "hi", author="a")
        sd.export_legacy(package)
        sd.to_markdown(package)
        self.assertEqual(package.document, before)
        self.assertEqual(package.signatures, [])


class PublishingValidationTests(unittest.TestCase):
    def check(self, mutate, expected=sd.SLDCValidationError, *, package=None) -> None:
        package = package or rich_package()
        mutate(package)
        with self.assertRaises(expected):
            sd.validate_package(package)

    def test_rich_document_is_valid(self) -> None:
        sd.validate_package(rich_package())

    def test_tables(self) -> None:
        def table(package):
            return next(b for b in package.document["blocks"] if b["type"] == "table")["table"]
        bad = {
            "too many columns in a row": lambda p: table(p)["rows"][1].append({"text": "x"}),
            "short row": lambda p: table(p)["rows"][1].pop(),
            "colspan beyond columns": lambda p: table(p)["rows"][2][0].update(colspan=3),
            "rowspan past end": lambda p: table(p)["rows"][2][0].update(rowspan=2),
            "overlapping spans": lambda p: table(p)["rows"][1][0].update(rowspan=2),
            "header rows beyond rows": lambda p: table(p).update(header_rows=9),
            "cell text must be a string": lambda p: table(p)["rows"][1][0].update(text=5),
            "unknown cell key": lambda p: table(p)["rows"][1][0].update(onclick="x"),
            "no rows": lambda p: table(p).update(rows=[]),
            "table without table data": lambda p: p.document["blocks"][2].pop("table"),
            "table data on a paragraph": lambda p: p.document["blocks"][1].update(
                table={"columns": 1, "rows": [[{"text": "x"}]]}),
        }
        for label, mutate in bad.items():
            with self.subTest(label):
                self.check(mutate)

    def test_valid_rowspan_grid(self) -> None:
        rows = [[{"text": "a", "rowspan": 2}, {"text": "b"}], [{"text": "c"}]]
        sd.validate_package(sd.new_package("T", [sd.make_table(rows, columns=2)]))

    def test_figures_need_alt_text_or_an_explicit_decorative_flag(self) -> None:
        asset = sd.make_asset(PNG_1X1, asset_id="i")
        ok = sd.new_package("T", [sd.make_figure("i", "A chart")], assets=[asset])
        sd.validate_package(ok)
        sd.validate_package(sd.new_package(
            "T", [sd.make_figure("i", "", decorative=True)], assets=[asset]))
        for alt, decorative in (("", False), ("text", True), ("   ", False)):
            with self.subTest(alt=alt, decorative=decorative), self.assertRaises(sd.SLDCValidationError):
                sd.validate_package(sd.new_package(
                    "T", [sd.make_figure("i", alt, decorative=decorative)], assets=[asset]))

    def test_images_are_checked_by_header_and_never_decoded(self) -> None:
        svg = b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>"
        for label, data in {"svg": svg, "html": b"<html>", "empty": b"", "truncated png": PNG_1X1[:20],
                            "huge side": png_header(20_000, 10),
                            "too many pixels": png_header(16_000, 16_000),
                            "zero": png_header(0, 5)}.items():
            with self.subTest(label), self.assertRaises(sd.SLDCValidationError):
                sd.make_asset(data)
        self.assertEqual(sd.make_asset(PNG_1X1).media_type, "image/png")
        self.assertTrue(sd.make_asset(PNG_1X1).id.startswith("img-"))
        self.assertEqual(sd.make_asset(PNG_1X1).id, sd.make_asset(PNG_1X1).id)
        gif = b"GIF89a" + struct.pack("<HH", 3, 4) + b"\x00" * 10
        self.assertEqual(sd.make_asset(gif).media_type, "image/gif")
        webp = b"RIFF" + b"\x00" * 4 + b"WEBPVP8L" + b"\x00" * 4 + b"\x2f" + struct.pack("<I", 9 | (4 << 14)) + b"\x00" * 6
        self.assertEqual(sd.make_asset(webp).media_type, "image/webp")

    def test_asset_media_type_must_match_the_bytes(self) -> None:
        package = sd.new_package("T", [sd.make_figure("i", "x")])
        package.assets["i"] = sd.Asset("i", "image/jpeg", PNG_1X1)
        with self.assertRaisesRegex(sd.SLDCValidationError, "declared"):
            sd.validate_package(package)

    def test_total_image_budgets(self) -> None:
        blocks = [sd.make_figure(f"i{n}", "x") for n in range(5)]
        assets = [sd.Asset(f"i{n}", "image/png", png_header(16_000, 4_000)) for n in range(5)]
        package = sd.new_package("T", blocks, assets=assets)
        with self.assertRaisesRegex(sd.SLDCValidationError, "pixel budget"):
            sd.validate_package(package)

    def test_spans(self) -> None:
        def span(package, **changes):
            package.document["blocks"][1]["spans"][0].update(changes)
        bad = {
            "out of range": lambda p: span(p, end=999),
            "empty range": lambda p: span(p, start=3, end=3),
            "unknown type": lambda p: span(p, type="script"),
            "unknown format flag": lambda p: span(p, onclick=True),
            "bad colour": lambda p: span(p, color="red; background:url(x)"),
            "unsorted": lambda p: p.document["blocks"][1]["spans"].reverse(),
            "unknown reference": lambda p: p.document["blocks"][1]["spans"][1].update(ref="nope"),
            "unknown character style": lambda p: p.document["blocks"][1]["spans"][3].update(style="X"),
            "unknown xref target": lambda p: p.document["blocks"][1]["spans"][4].update(target="zz"),
            "span on non-list": lambda p: p.document["blocks"][1].update(spans={}),
        }
        for label, mutate in bad.items():
            with self.subTest(label):
                self.check(mutate)

    def test_links_are_inert_data_with_a_strict_scheme_list(self) -> None:
        for href, ok in (("https://example.org/x?y=1", True), ("http://example.org", True),
                         ("mailto:a@example.org", True), ("javascript:alert(1)", False),
                         ("data:text/html,<script>", False), ("file:///etc/passwd", False),
                         ("//example.org", False), ("/relative", False),
                         ("https://user:pw@example.org", False), ("https://exa mple.org", False),
                         ("https:///nohost", False), ("HTTPS://example.org", True),
                         ("https://example.org/" + "a" * 2100, False)):
            package = small_package("hello world")
            package.document["blocks"][0]["spans"] = [
                {"type": "link", "start": 0, "end": 5, "href": href}]
            with self.subTest(href=href[:40]):
                if ok:
                    sd.validate_package(package)
                else:
                    with self.assertRaises(sd.SLDCValidationError):
                        sd.validate_package(package)

    def test_style_properties_are_whitelisted_values_not_css(self) -> None:
        bad = [{"paragraph": {"X": {"font_family": "a; background:url(http://x)"}}},
               {"paragraph": {"X": {"color": "url(x)"}}},
               {"paragraph": {"X": {"font_size_pt": 9999}}},
               {"paragraph": {"X": {"css": "x"}}},
               {"paragraph": {"X": {"align": "left; x"}}},
               {"paragraph": {"X": {"bold": "yes"}}},
               {"paragraph": {"X": {"based_on": "X"}}},
               {"paragraph": {"A": {"based_on": "B"}, "B": {"based_on": "A"}}},
               {"paragraph": {"A": {"based_on": "Missing"}}},
               {"paragraph": {"bad name!": {}}},
               {"character": {"X": {"align": "left"}}},
               {"table": {}}]
        for styles in bad:
            with self.subTest(styles=styles), self.assertRaises(sd.SLDCValidationError):
                sd.validate_package(sd.new_package("T", [sd.make_block("paragraph", "x")], styles=styles))
        good = {"paragraph": {"A": {"bold": True}, "B": {"based_on": "A", "font_size_pt": 11}}}
        sd.validate_package(sd.new_package("T", [sd.make_block("paragraph", "x", style="B")], styles=good))
        with self.assertRaises(sd.SLDCValidationError):  # A block may only use a style that exists.
            sd.validate_package(sd.new_package("T", [sd.make_block("paragraph", "x", style="B")]))

    def test_sections_and_templates(self) -> None:
        base = {"id": "s1", "page": {"size": "Letter", "orientation": "landscape"}}
        sd.validate_package(sd.new_package("T", [], sections=[base]))
        bad = [{"id": "s1", "footer": {"center": "{evil}"}},
               {"id": "s1", "footer": {"center": "{page"}},
               {"id": "s1", "footer": {"center": "x}"}},
               {"id": "s1", "footer": {"style": "Missing"}},
               {"id": "s1", "page": {"size": "Poster"}},
               {"id": "s1", "page": {"size": "A4", "width_mm": 100}},
               {"id": "s1", "page": {"size": "A4", "margins_mm": {"left": 150, "right": 150}}},
               {"id": "s1", "columns": 99},
               {"id": "s1", "numbering": {"format": "wingdings"}},
               {"id": "s1", "onload": "x"}]
        for section in bad:
            with self.subTest(section=section), self.assertRaises(sd.SLDCValidationError):
                sd.validate_package(sd.new_package("T", [], sections=[section]))
        with self.assertRaises(sd.SLDCValidationError):
            sd.validate_package(sd.new_package("T", [], sections=[{"id": "a"}, {"id": "a"}]))
        sd.validate_package(sd.new_package(
            "T", [], sections=[{"id": "a", "footer": {"left": "{{literal}} {page}"}}]))

    def test_sections_are_contiguous_and_in_order(self) -> None:
        blocks = [sd.make_block("paragraph", "a", section="s1"),
                  sd.make_block("paragraph", "b", section="s2"),
                  sd.make_block("paragraph", "c", section="s1")]
        package = sd.new_package("T", blocks, sections=[{"id": "s1"}, {"id": "s2"}])
        with self.assertRaisesRegex(sd.SLDCValidationError, "earlier section"):
            sd.validate_package(package)

    def test_references(self) -> None:
        bad = [{"id": "r", "kind": "citation"},
               {"id": "r", "kind": "citation", "title": "T", "doi": "not-a-doi"},
               {"id": "r", "kind": "citation", "title": "T", "year": 0},
               {"id": "r", "kind": "citation", "title": "T", "url": "javascript:x"},
               {"id": "r", "kind": "citation", "title": "T", "authors": "me"},
               {"id": "r", "kind": "footnote"},
               {"id": "r", "kind": "endnote", "text": "x"},
               {"id": "r", "kind": "footnote", "text": "x", "extra": 1}]
        for reference in bad:
            with self.subTest(reference=reference), self.assertRaises(sd.SLDCValidationError):
                sd.validate_package(sd.new_package("T", [], references=[reference]))
        with self.assertRaises(sd.SLDCValidationError):
            sd.validate_package(sd.new_package("T", [], references=[
                {"id": "r", "kind": "footnote", "text": "a"}, {"id": "r", "kind": "footnote", "text": "b"}]))

    def test_publication_metadata(self) -> None:
        bad = [{"authors": "me"}, {"published": "tomorrow"}, {"published": "2026-02-31"},
               {"identifier": {"scheme": "ISBN", "value": "x"}}, {"cover_asset": "missing"},
               {"keywords": ["x"] * 99}, {"unknown": 1}]
        for publication in bad:
            with self.subTest(publication=publication), self.assertRaises(sd.SLDCValidationError):
                sd.validate_package(sd.new_package("T", [], publication=publication))

    def test_embeds_are_declarative_records(self) -> None:
        def embed(**changes):
            record = {"id": "e", "kind": "chart.bar", "schema_version": 1}
            record.update(changes)
            return sd.new_package("T", [sd.make_block("embed", "", embed="e")], embeds=[record])
        sd.validate_package(embed())
        bad = [{"kind": "Chart"}, {"kind": "x y"}, {"schema_version": 0}, {"schema_version": "1"},
               {"params": []}, {"params": {"x": "a" * 70_000}}, {"capabilities": ["net", "net"]},
               {"capabilities": ["Net!"]}, {"inputs": [{"ref": "x"}]},
               {"inputs": [{"ref": "x", "sha256": "abc"}]}, {"output_sha256": "abc"},
               {"command": "rm -rf /"}, {"url": "https://example.org/run.js"}]
        for changes in bad:
            with self.subTest(changes=list(changes)), self.assertRaises(sd.SLDCValidationError):
                sd.validate_package(embed(**changes))
        with self.assertRaises(sd.SLDCValidationError):  # An embed block must point at a record.
            sd.validate_package(sd.new_package("T", [sd.make_block("embed", "", embed="nope")]))

    def test_extensions_are_namespaced_and_inert(self) -> None:
        package = small_package()
        package.document["extensions"] = {"acme.flow": {"x": [1, 2]}}
        package.document["blocks"][0]["extensions"] = {"acme.note": "hi"}
        sd.validate_package(package)
        self.assertEqual(sd.loads_package(sd.dumps_package(package)).document["extensions"],
                         {"acme.flow": {"x": [1, 2]}})
        for key in ("flow", "ACME.x", "acme.", ".x", "a b.c"):
            package.document["extensions"] = {key: 1}
            with self.subTest(key=key), self.assertRaises(sd.SLDCValidationError):
                sd.validate_package(package)

    def test_unknown_keys_and_types_are_rejected_not_ignored(self) -> None:
        for mutate in (lambda d: d.update(script="x"), lambda d: d["blocks"][0].update(onload="x"),
                       lambda d: d["blocks"][0].update(type="iframe"),
                       lambda d: d["blocks"][0].update(annotations=[]),
                       lambda d: d.update(schema="slai.sldc/2"),
                       lambda d: d["blocks"][0].update(attributes={"level": 1}),
                       lambda d: d["blocks"][0].update(id=" bad id")):
            package = small_package()
            with self.subTest(), self.assertRaises(sd.SLDCValidationError):
                mutate(package.document)
                sd.validate_package(package)

    def test_attribute_rules(self) -> None:
        for block in (sd.make_block("heading", "x", attributes={"level": 9}),
                      sd.make_block("paragraph", "x", attributes={"level": 1}),
                      sd.make_block("code", "x", attributes={"language": "py thon"}),
                      sd.make_block("list_item", "x", attributes={"depth": -1}),
                      sd.make_block("paragraph", "x", attributes={"align": "middle"}),
                      sd.make_block("paragraph", "x", attributes={"page_break_before": "yes"})):
            with self.subTest(block=block), self.assertRaises(sd.SLDCValidationError):
                sd.validate_package(sd.new_package("T", [block]))
        sd.validate_package(sd.new_package("T", [
            sd.make_block("paragraph", "x", attributes={"align": "center", "page_break_before": True})]))

    def test_limits_apply_to_1x_documents(self) -> None:
        with mock.patch.object(sd, "MAX_BLOCKS", 2), self.assertRaises(sd.SLDCValidationError):
            sd.validate_package(small_package("a", "b", "c"))
        with mock.patch.object(sd, "MAX_TEXT_CHARS", 5), self.assertRaises(sd.SLDCValidationError):
            sd.validate_package(small_package("abc", "def"))
        with mock.patch.object(sd, "MAX_TABLE_CELLS", 3), self.assertRaises(sd.SLDCValidationError):
            sd.validate_package(sd.new_package("T", [sd.make_table([["a", "b"], ["c", "d"]])]))
        with mock.patch.object(sd, "MAX_SPANS_PER_BLOCK", 1), self.assertRaises(sd.SLDCValidationError):
            package = small_package("hello")
            package.document["blocks"][0]["spans"] = [
                {"type": "format", "start": 0, "end": 1, "bold": True},
                {"type": "format", "start": 1, "end": 2, "bold": True}]
            sd.validate_package(package)

    def test_unicode_offsets_count_characters_not_bytes(self) -> None:
        package = sd.new_package("T", [sd.make_block(
            "paragraph", "日本語 ✓", spans=[{"type": "format", "start": 4, "end": 5, "bold": True}])])
        sd.validate_package(package)
        package.document["blocks"][0]["spans"][0]["end"] = 6
        with self.assertRaises(sd.SLDCValidationError):
            sd.validate_package(package)


class DataOnlyTests(unittest.TestCase):
    """Opening a document must not execute or fetch anything it contains."""

    FORBIDDEN_MODULES = frozenset({
        "pickle", "marshal", "shelve", "subprocess", "importlib", "ctypes", "socket", "ssl", "http",
        "xmlrpc", "ftplib", "smtplib", "webbrowser", "runpy", "code", "codeop", "imp",
        "multiprocessing", "asyncio"})
    FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__", "input", "breakpoint"})
    FORBIDDEN_METHODS = frozenset({"system", "popen", "Popen", "urlopen", "startfile", "load"})

    def test_the_module_has_no_way_to_run_code_or_reach_the_network(self) -> None:
        source = Path(sd.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
                if node.module.startswith("urllib"):
                    self.assertEqual(node.module, "urllib.parse")  # Parsing only: no urllib.request.
        self.assertFalse(imported & self.FORBIDDEN_MODULES, imported & self.FORBIDDEN_MODULES)
        calls = {node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        methods = {node.func.attr for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & self.FORBIDDEN_CALLS, calls & self.FORBIDDEN_CALLS)
        self.assertFalse(methods & self.FORBIDDEN_METHODS, methods & self.FORBIDDEN_METHODS)

    def test_opening_a_document_with_links_embeds_and_code_strings_does_nothing(self) -> None:
        package = rich_package()
        package.document["blocks"][0]["text"] = "<script>alert(1)</script> {{7*7}} ${jndi:ldap://x}"
        data = sd.dumps_package(package)

        def refuse(*_args, **_kwargs):
            raise AssertionError("network access attempted")

        with mock.patch("socket.socket", refuse), mock.patch("socket.getaddrinfo", refuse), \
                mock.patch("builtins.exec", refuse), mock.patch("builtins.eval", refuse), \
                mock.patch("subprocess.Popen", refuse), mock.patch("os.system", refuse):
            back = sd.loads_package(data)
            sd.extract_text(back)
            sd.to_markdown(back)
            sd.chunk_document(back)
            sd.inspect_package(data)
            sd.export_legacy(back)
        self.assertEqual(back.document["embeds"][0]["params"]["note"], "__import__('os').system('x')")
        self.assertEqual(back.document["blocks"][0]["text"], package.document["blocks"][0]["text"])

    def test_markdown_export_never_emits_active_content(self) -> None:
        package = small_package("click here")
        package.document["blocks"][0]["spans"] = [
            {"type": "link", "start": 0, "end": 5, "href": "https://example.org/a)b"}]
        markdown = sd.to_markdown(package)
        self.assertIn("(https://example.org/a%29b)", markdown)
        self.assertNotIn("javascript:", markdown)


class MigrationTests(_TempDirTestCase):
    def legacy_document(self):
        document = sample_document()
        document["blocks"][1]["annotations"] = [{"kind": "note", "start": 0, "end": 3}]
        document["blocks"][1]["attributes"] = {"align": "left", "custom": [1, 2]}
        document["blocks"][1]["source_ref"] = "report.pdf#p3"
        return document

    def test_migration_keeps_ids_and_moves_what_it_cannot_map(self) -> None:
        legacy = sd.SLDCPackage(self.legacy_document(), version=sd.LEGACY_VERSION)
        package, report = sd.migrate_package(legacy, timestamp="2026-02-01T00:00:00Z")
        sd.validate_package(package)
        self.assertEqual(package.version, sd.FORMAT_VERSION)
        old, new = legacy.document, package.document
        self.assertEqual([b["id"] for b in new["blocks"]], [b["id"] for b in old["blocks"]])
        self.assertEqual([b["text"] for b in new["blocks"]], [b["text"] for b in old["blocks"]])
        self.assertEqual(new["id"], old["id"])
        self.assertEqual(new["metadata"], old["metadata"])
        second = new["blocks"][1]
        self.assertEqual(second["source_ref"], "report.pdf#p3")
        self.assertEqual(second["attributes"], {"align": "left"})
        self.assertNotIn("annotations", second)
        self.assertEqual(second["extensions"]["slai.sldc.legacy"],
                         {"annotations": old["blocks"][1]["annotations"],
                          "attributes": {"custom": [1, 2]}})
        self.assertEqual(new["provenance"][-1]["activity"], "sldc.migrate")
        self.assertEqual(new["provenance"][-1]["timestamp"], "2026-02-01T00:00:00Z")
        self.assertEqual(new["provenance"][:-1], old["provenance"])
        self.assertEqual((report["migrated"], report["lossy"], report["block_ids_preserved"]),
                         (True, False, True))
        self.assertEqual(report["moved_attributes"], [{"block": old["blocks"][1]["id"], "key": "custom"}])
        self.assertEqual(report["source_document_sha256"], sd.document_digest(old))
        json.dumps(report)  # Machine-readable.

    def test_migration_does_not_touch_the_input_and_is_idempotent(self) -> None:
        legacy = sd.SLDCPackage(self.legacy_document(), version=sd.LEGACY_VERSION)
        before = copy.deepcopy(legacy.document)
        package, _ = sd.migrate_package(legacy)
        self.assertEqual(legacy.document, before)
        again, report = sd.migrate_package(package)
        self.assertFalse(report["migrated"])
        self.assertEqual(again.document, package.document)
        migrated_from_mapping, _ = sd.migrate_package(self.legacy_document())
        self.assertEqual(migrated_from_mapping.version, sd.FORMAT_VERSION)

    def test_migrate_file_never_modifies_or_replaces_its_source(self) -> None:
        source, target = self.tmp / "old.sldc", self.tmp / "new.sldc"
        sd.write_sldc(source, self.legacy_document())
        original = source.read_bytes()
        report = sd.migrate_file(source, target)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(report["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(report["source_bytes"], len(original))
        self.assertEqual(sd.read_package(target).version, sd.FORMAT_VERSION)
        with self.assertRaises(FileExistsError):
            sd.migrate_file(source, target)
        with self.assertRaisesRegex(sd.SLDCValidationError, "overwrites its source"):
            sd.migrate_file(source, source, overwrite=True)
        with self.assertRaisesRegex(sd.SLDCUnsupportedError, "nothing to migrate"):
            sd.migrate_file(target, self.tmp / "again.sldc")
        self.assertEqual(source.read_bytes(), original)

    def test_export_legacy_is_lossy_labelled_and_reported(self) -> None:
        package = rich_package()
        legacy, report = sd.export_legacy(package, timestamp="2026-03-01T00:00:00Z")
        self.assertEqual(legacy.version, sd.LEGACY_VERSION)
        sd.validate_package(legacy)
        loaded = sd.loads_sldc(sd.dumps_package(legacy))  # A genuine, readable 0.1.0 file.
        label = loaded["metadata"]["sldc_export"]
        self.assertTrue(label["lossy"])
        self.assertEqual(label["source_revision_id"], sd.revision_id(package))
        self.assertEqual(label["dropped"], report["dropped"])
        self.assertEqual(label["publication"]["subtitle"], "FY2025")
        self.assertEqual(loaded["provenance"][-1]["activity"], "sldc.export_legacy")
        self.assertEqual(report["dropped"]["tables_flattened"], 1)
        self.assertEqual(report["dropped"]["figures_replaced"], 1)
        self.assertEqual(report["dropped"]["assets"], 1)
        self.assertGreaterEqual(report["dropped"]["spans"], 6)
        text = sd.extract_text(loaded)
        self.assertIn("Annual report", text)
        self.assertIn("Quarter\tRevenue", text)
        self.assertIn("[Figure] Company logo", text)
        self.assertIn("Chart: revenue by quarter", text)
        self.assertEqual([b["id"] for b in loaded["blocks"]], [b["id"] for b in package.document["blocks"]])
        self.assertEqual(loaded["blocks"][0]["attributes"], {"level": 1})
        self.assertEqual((report["blocks_in"], report["blocks_out"]), (6, 6))

    def test_export_drops_signatures_and_comments_and_says_so(self) -> None:
        package, _ = sd.add_comment(small_package(), "b1", "note", author="Ann")
        legacy, report = sd.export_legacy(package)
        self.assertEqual(report["dropped"]["comments"], 1)
        self.assertEqual((legacy.comments, legacy.signatures, legacy.assets), ([], [], {}))

    def test_legacy_to_1x_to_legacy_keeps_the_text(self) -> None:
        legacy = sd.SLDCPackage(sample_document(), version=sd.LEGACY_VERSION)
        package, _ = sd.migrate_package(legacy)
        back, _ = sd.export_legacy(package)
        self.assertEqual(sd.extract_text(back), sd.extract_text(legacy))
        self.assertEqual(sd.export_legacy(legacy)[1]["lossy"], False)


class CollaborationTests(_TempDirTestCase):
    def test_stable_block_ids_and_new_ids_do_not_collide(self) -> None:
        ids = {sd.new_block_id() for _ in range(2000)}
        self.assertEqual(len(ids), 2000)
        self.assertTrue(all(re.fullmatch(r"b[0-9a-f]{16}", i) for i in ids))
        self.assertEqual([b["id"] for b in sd.new_package("T", [sd.make_block("paragraph", "a"),
                                                                 sd.make_block("paragraph", "b")]).document["blocks"]],
                         ["b00001", "b00002"])

    def test_next_revision_records_lineage_and_drops_old_signatures(self) -> None:
        first = sd.next_revision(small_package(), author="Ann", note="draft", created="2026-01-01T00:00:00Z")
        self.assertEqual(first.document["revision"],
                         {"number": 1, "parents": [], "created": "2026-01-01T00:00:00Z",
                          "author": "Ann", "note": "draft"})
        second = sd.next_revision(first, created="2026-01-02T00:00:00Z")
        self.assertEqual(second.document["revision"]["number"], 2)
        self.assertEqual(second.document["revision"]["parents"], [sd.revision_id(first)])
        self.assertEqual(second.document["history"][0]["id"], sd.revision_id(first))
        self.assertEqual(second.document["history"][0]["number"], 1)
        third = sd.next_revision(second)
        self.assertEqual([h["number"] for h in third.document["history"]], [1, 2])
        sd.loads_package(sd.dumps_package(third))  # The lineage survives a round trip.
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.next_revision(sd.SLDCPackage(sample_document(), version=sd.LEGACY_VERSION))

    def test_diff(self) -> None:
        old = small_package("one", "two", "three", "four")
        new = old.copy()
        blocks = new.document["blocks"]
        blocks[1]["text"] = "TWO"
        removed = blocks.pop(2)
        blocks.append(blocks.pop(0))  # b1 moves to the end.
        blocks.insert(0, sd.make_block("paragraph", "fresh", block_id="n1"))
        new.document["title"] = "Renamed"
        diff = sd.diff_documents(old, new)
        self.assertEqual(diff["added"], ["n1"])
        self.assertEqual(diff["removed"], [removed["id"]])
        self.assertEqual(diff["modified"], ["b2"])
        self.assertEqual(diff["moved"], ["b1"])
        self.assertEqual(diff["unchanged"], 1)
        self.assertEqual(diff["changed_fields"], ["title"])
        same = sd.diff_documents(old, old.copy())
        self.assertEqual((same["added"], same["removed"], same["modified"], same["moved"]), ([], [], [], []))

    def test_diff_sees_changes_to_spans_and_styles_not_only_text(self) -> None:
        old = small_package("hello world")
        new = old.copy()
        new.document["blocks"][0]["spans"] = [{"type": "format", "start": 0, "end": 5, "bold": True}]
        self.assertEqual(sd.diff_documents(old, new)["modified"], ["b1"])

    def edit(self, package, index, text):
        copy_ = package.copy()
        copy_.document["blocks"][index]["text"] = text
        return copy_

    def test_merge_takes_independent_edits_from_both_sides(self) -> None:
        base = small_package("one", "two", "three")
        ours = self.edit(base, 0, "ONE")
        theirs = self.edit(base, 2, "THREE")
        theirs.document["blocks"].append(sd.make_block("paragraph", "their new", block_id="t9"))
        ours.document["title"] = "Better title"
        merged, conflicts = sd.merge_documents(base, ours, theirs)
        self.assertEqual(conflicts, [])
        self.assertEqual([b["text"] for b in merged["blocks"]], ["ONE", "two", "THREE", "their new"])
        self.assertEqual(merged["title"], "Better title")

    def test_merge_reports_conflicts_and_keeps_ours(self) -> None:
        base = small_package("one", "two", "three")
        ours = self.edit(base, 0, "ours")
        theirs = self.edit(base, 0, "theirs")
        del theirs.document["blocks"][1]  # They delete b2 ...
        ours = self.edit(ours, 1, "edited")  # ... which we edited.
        merged, conflicts = sd.merge_documents(base, ours, theirs)
        self.assertEqual({(c["kind"], c["block"]) for c in conflicts},
                         {("both_modified", "b1"), ("delete_modify", "b2")})
        self.assertEqual([b["text"] for b in merged["blocks"]], ["ours", "edited", "three"])

    def test_merge_identical_edits_and_one_sided_delete(self) -> None:
        base = small_package("one", "two", "three")
        ours = self.edit(base, 0, "same")
        theirs = self.edit(base, 0, "same")
        del theirs.document["blocks"][2]
        merged, conflicts = sd.merge_documents(base, ours, theirs)
        self.assertEqual(conflicts, [])
        self.assertEqual([b["text"] for b in merged["blocks"]], ["same", "two"])

    def test_merge_order(self) -> None:
        base = small_package("a", "b", "c", "d")
        ours = base.copy()
        ours.document["blocks"].insert(1, sd.make_block("paragraph", "ours-new", block_id="o1"))
        theirs = base.copy()
        theirs.document["blocks"].insert(1, sd.make_block("paragraph", "theirs-new", block_id="t1"))
        merged, conflicts = sd.merge_documents(base, ours, theirs)
        self.assertEqual(conflicts, [])
        self.assertEqual([b["text"] for b in merged["blocks"]],
                         ["a", "ours-new", "theirs-new", "b", "c", "d"])  # Ours first, both kept.
        moved = base.copy()
        moved.document["blocks"].append(moved.document["blocks"].pop(0))
        merged, conflicts = sd.merge_documents(base, moved, theirs)
        self.assertEqual(conflicts, [])
        self.assertEqual([b["id"] for b in merged["blocks"]], ["b2", "b3", "b4", "b1", "t1"])  # Our move, their insert.
        other = base.copy()
        other.document["blocks"].insert(0, other.document["blocks"].pop())
        merged, conflicts = sd.merge_documents(base, moved, other)
        self.assertEqual([c["kind"] for c in conflicts], ["order"])
        self.assertEqual([b["id"] for b in merged["blocks"]], [b["id"] for b in moved.document["blocks"]])

    def test_merge_both_added_the_same_id_differently(self) -> None:
        base = small_package("a")
        ours, theirs = base.copy(), base.copy()
        ours.document["blocks"].append(sd.make_block("paragraph", "x", block_id="dup"))
        theirs.document["blocks"].append(sd.make_block("paragraph", "y", block_id="dup"))
        merged, conflicts = sd.merge_documents(base, ours, theirs)
        self.assertEqual(conflicts, [{"kind": "both_added", "block": "dup"}])
        self.assertEqual(merged["blocks"][-1]["text"], "x")

    def test_merge_does_not_modify_its_inputs_and_needs_1x(self) -> None:
        base = small_package()
        ours = self.edit(base, 0, "x")
        snapshot = copy.deepcopy(ours.document)
        sd.merge_documents(base, ours, base)
        self.assertEqual(ours.document, snapshot)
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.merge_documents(sample_document(), sample_document(), sample_document())

    def test_merge_packages_builds_a_new_revision_and_unions_comments_and_assets(self) -> None:
        base = sd.next_revision(small_package(), created="2026-01-01T00:00:00Z")
        ours, thread_a = sd.add_comment(base, "b1", "ours says", author="Ann", created="2026-01-02T00:00:00Z")
        ours = self.edit(ours, 0, "ONE")
        theirs, thread_b = sd.add_comment(base, "b2", "theirs says", author="Bo", created="2026-01-03T00:00:00Z")
        theirs = sd.add_asset(theirs, sd.make_asset(PNG_1X1, asset_id="pic"))
        theirs.document["blocks"].append(sd.make_figure("pic", "pic", block_id="fig"))
        ours = sd.next_revision(ours, created="2026-01-04T00:00:00Z")
        theirs = sd.next_revision(theirs, created="2026-01-05T00:00:00Z")
        merged, conflicts = sd.merge_packages(base, ours, theirs, author="Cy", created="2026-01-06T00:00:00Z")
        self.assertEqual(conflicts, [])
        self.assertEqual({t["id"] for t in merged.comments}, {thread_a, thread_b})
        self.assertIn("pic", merged.assets)
        self.assertEqual(merged.document["revision"]["parents"],
                         [sd.revision_id(ours), sd.revision_id(theirs)])
        self.assertEqual(merged.document["revision"]["number"], 3)
        self.assertEqual(merged.signatures, [])
        ids = {h["id"] for h in merged.document["history"]}
        self.assertTrue({sd.revision_id(ours), sd.revision_id(theirs)} <= ids)
        sd.loads_package(sd.dumps_package(merged))
        clash = sd.add_asset(base, sd.make_asset(PNG_1X1, asset_id="pic"))
        clash.document["blocks"].append(sd.make_figure("pic", "pic", block_id="fig"))
        other = clash.copy()
        other.assets["pic"] = sd.make_asset(png_header(2, 2), asset_id="pic")
        _, conflicts = sd.merge_packages(base, clash, other)
        self.assertEqual(conflicts, [{"kind": "asset", "asset": "pic"}])


class CommentTests(_TempDirTestCase):
    def test_threads_replies_and_resolution(self) -> None:
        package = small_package("The quick brown fox")
        package, thread = sd.add_comment(package, "b1", "Which fox?", author="Ann",
                                         start=4, end=9, created="2026-01-01T10:00:00Z")
        package = sd.reply_to_thread(package, thread, "This one.", author="Bo", created="2026-01-01T11:00:00Z")
        package = sd.set_thread_status(package, thread, "resolved", by="Ann", at="2026-01-01T12:00:00Z")
        stored = package.comments[0]
        self.assertEqual(stored["status"], "resolved")
        self.assertEqual(stored["resolved"], {"by": "Ann", "at": "2026-01-01T12:00:00Z"})
        self.assertEqual([m["text"] for m in stored["messages"]], ["Which fox?", "This one."])
        self.assertEqual((stored["anchor"]["start"], stored["anchor"]["end"]), (4, 9))
        package = sd.reply_to_thread(package, thread, "Thanks", author="Ann")
        package = sd.set_thread_status(package, thread, "open", by="Ann")
        self.assertEqual(package.comments[0]["status"], "open")
        self.assertNotIn("resolved", package.comments[0])
        back = sd.loads_package(sd.dumps_package(package))
        self.assertEqual(back.comments, package.comments)
        self.assertEqual(len(back.comments[0]["messages"]), 3)

    def test_comments_never_change_the_document_revision_or_text(self) -> None:
        package = small_package("hello")
        commented, _ = sd.add_comment(package, "b1", "hmm", author="Ann")
        self.assertEqual(commented.document, package.document)
        self.assertEqual(sd.extract_text(commented), "hello")
        self.assertEqual(sd.revision_id(commented), sd.revision_id(package))
        self.assertNotEqual(sd.dumps_package(commented), sd.dumps_package(package))

    def test_anchor_status_tracks_edits(self) -> None:
        package = small_package("alpha", "beta")
        package, first = sd.add_comment(package, "b1", "a", author="Ann")
        package, second = sd.add_comment(package, "b2", "b", author="Ann")
        self.assertEqual(sd.comment_anchor_status(package), {first: "ok", second: "ok"})
        edited = package.copy()
        edited.document["blocks"][0]["text"] = "ALPHA"
        del edited.document["blocks"][1]
        self.assertEqual(sd.comment_anchor_status(edited), {first: "stale", second: "orphaned"})

    def test_invalid_comment_operations(self) -> None:
        package = small_package("hello")
        for call in (lambda: sd.add_comment(package, "nope", "x", author="a"),
                     lambda: sd.add_comment(package, "b1", "", author="a"),
                     lambda: sd.add_comment(package, "b1", "x", author=""),
                     lambda: sd.add_comment(package, "b1", "x", author="a", start=3, end=3),
                     lambda: sd.add_comment(package, "b1", "x", author="a", start=0, end=99),
                     lambda: sd.add_comment(package, "b1", "x", author="a", start=1),
                     lambda: sd.reply_to_thread(package, "missing", "x", author="a"),
                     lambda: sd.set_thread_status(package, "missing", "resolved", by="a"),
                     lambda: sd.add_comment(sd.SLDCPackage(sample_document(), version=sd.LEGACY_VERSION),
                                            "b00001", "x", author="a")):
            with self.assertRaises(sd.SLDCError):
                call()
        with_thread, thread = sd.add_comment(package, "b1", "x", author="a")
        with self.assertRaises(sd.SLDCValidationError):
            sd.set_thread_status(with_thread, thread, "closed", by="a")

    def test_hostile_comments_part_is_rejected(self) -> None:
        data = sd.dumps_package(sd.add_comment(small_package(), "b1", "x", author="a")[0])
        part = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("comments.json"))
        forged = []
        for mutate in (lambda p: p["threads"][0].update(status="deleted"),
                       lambda p: p["threads"][0].update(onclick="x"),
                       lambda p: p["threads"][0]["messages"].clear(),
                       lambda p: p["threads"].append(p["threads"][0]),
                       lambda p: p.update(schema="other/1"),
                       lambda p: p["threads"][0]["anchor"].update(start=5)):
            clone = copy.deepcopy(part)
            mutate(clone)
            forged.append(clone)
        for clone in forged:
            with self.subTest(clone=clone), self.assertRaises(sd.SLDCValidationError):
                sd.loads_package(refresh_part(data, "comments.json", sd._canonical_json(clone)))


HAVE_CRYPTO = importlib.util.find_spec("cryptography") is not None
needs_crypto = unittest.skipUnless(HAVE_CRYPTO, "the 'cryptography' package is not installed")
PASSWORD = "correct horse battery staple"


@needs_crypto
class SigningTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.alice = sd.Ed25519Signer.from_seed(bytes(range(32)))
        self.bob = sd.Ed25519Signer.from_seed(bytes(range(32, 64)))
        self.package = small_package()

    def sign(self, package=None, signer=None, **options):
        options.setdefault("signed_at", "2026-04-01T09:00:00Z")
        return sd.sign_package(package or self.package, signer or self.alice, **options)

    def state(self, package, **options):
        return [r["state"] for r in sd.verify_package(package, **options)]

    def test_sign_and_verify_report_each_fact_separately(self) -> None:
        signed = self.sign(statement="Approved for release")
        self.assertEqual(self.package.signatures, [])  # The input is untouched.
        (result,) = sd.verify_package(signed)
        self.assertEqual((result["signature_valid"], result["covers_current_content"]), (True, True))
        self.assertEqual((result["trust"], result["state"]), ("unknown_signer", "valid_unknown_signer"))
        self.assertEqual(result["signed_revision_id"], sd.revision_id(self.package))
        self.assertEqual(result["signed_at_claimed"], "2026-04-01T09:00:00Z")
        self.assertEqual((result["purpose"], result["statement"]), ("approval", "Approved for release"))
        self.assertFalse(any("verified" in key for key in result))
        self.assertEqual(self.state(signed, trusted_keys=[self.alice.key_id]), ["valid_trusted"])
        self.assertEqual(self.state(signed, trusted_keys={self.alice.key_id: "Alice"}), ["valid_trusted"])
        self.assertEqual(self.state(signed, trusted_keys=[self.alice.key_id],
                                    revoked_keys=[self.alice.key_id]), ["valid_revoked"])
        self.assertEqual(self.state(signed, trusted_keys=[self.bob.key_id]), ["valid_unknown_signer"])

    def test_signature_survives_serialisation_and_does_not_change_the_revision(self) -> None:
        signed = self.sign()
        self.assertEqual(sd.revision_id(signed), sd.revision_id(self.package))
        back = sd.loads_package(sd.dumps_package(signed))
        self.assertEqual(back.signatures, signed.signatures)
        self.assertEqual(self.state(back), ["valid_unknown_signer"])
        self.assertEqual(sd.dumps_package(back), sd.dumps_package(signed))
        names = [p["name"] for p in sd.inspect_package(sd.dumps_package(signed))["parts"]]
        self.assertEqual(names, ["document.json", f"signatures/{signed.signatures[0]['id']}.json"])

    def test_any_change_to_signed_content_is_reported_as_modified(self) -> None:
        signed = self.sign()
        edits = {
            "text": lambda p: p.document["blocks"][0].update(text="ONE"),
            "title": lambda p: p.document.update(title="Other"),
            "block added": lambda p: p.document["blocks"].append(sd.make_block("paragraph", "x", block_id="n")),
            "metadata": lambda p: p.document.update(metadata={"a": 1}),
        }
        for label, edit in edits.items():
            changed = signed.copy()
            edit(changed)
            with self.subTest(label):
                (result,) = sd.verify_package(changed)
                self.assertEqual(result["state"], "modified")
                self.assertTrue(result["signature_valid"])  # Genuine, but of an earlier revision.
                self.assertFalse(result["covers_current_content"])
                self.assertEqual(result["changed_parts"], ["document.json"])
        picture = sd.add_asset(sd.new_package("T", [sd.make_figure("i", "x")]), sd.make_asset(PNG_1X1, asset_id="i"))
        signed_picture = self.sign(picture)
        swapped = signed_picture.copy()
        swapped.assets["i"] = sd.make_asset(png_header(2, 2), asset_id="i")
        self.assertEqual(sd.verify_package(swapped)[0]["changed_parts"], ["assets/i.png"])

    def test_comments_are_outside_the_signature_unless_covered(self) -> None:
        signed = self.sign()
        commented, _ = sd.add_comment(signed, "b1", "fine", author="Ann")
        self.assertEqual(self.state(commented), ["valid_unknown_signer"])
        covered = self.sign(commented, cover_comments=True)
        self.assertEqual(self.state(covered), ["valid_unknown_signer"] * 2)
        more, _ = sd.add_comment(covered, "b2", "late", author="Bo")
        self.assertEqual(self.state(more), ["valid_unknown_signer", "modified"])
        self.assertEqual(sd.verify_package(more)[1]["changed_parts"], ["comments.json"])
        with self.assertRaises(sd.SLDCValidationError):
            self.sign(cover_comments=True)  # There is nothing to cover.

    def test_forgery_and_tampering_are_invalid(self) -> None:
        signed = self.sign()

        def tamper(mutate):
            package = signed.copy()
            mutate(package.signatures[0])
            return package
        flipped = lambda r: r.update(signature=sd._b64e(bytes(a ^ 1 for a in sd._b64d(r["signature"], "s"))))
        cases = {
            "signature bytes": flipped,
            "purpose": lambda r: r["payload"].update(purpose="authorship"),
            "signed_at": lambda r: r["payload"].update(signed_at="2020-01-01T00:00:00Z"),
            "statement": lambda r: r["payload"].update(statement="Different"),
            "covered hash": lambda r: r["payload"]["covered"][0].update(sha256="0" * 64),
            "other key, same id": lambda r: r.update(public_key=sd._b64e(self.bob.public_key)),
            "claims another signer": lambda r: (r.update(key_id=self.bob.key_id, public_key=sd._b64e(self.bob.public_key)),
                                                r["payload"].update(key_id=self.bob.key_id)),
        }
        for label, mutate in cases.items():
            with self.subTest(label):
                self.assertEqual(self.state(tamper(mutate)), ["invalid"])
        self.assertEqual(sd.verify_package(tamper(flipped))[0]["signature_valid"], False)

    def test_a_signature_cannot_be_grafted_onto_another_document(self) -> None:
        signed = self.sign()
        other = small_package("one", "two", "different")
        other.signatures = copy.deepcopy(signed.signatures)
        (result,) = sd.verify_package(other)
        self.assertEqual(result["state"], "modified")
        self.assertTrue(result["signature_valid"])
        elsewhere = sd.new_package("Small", [sd.make_block("paragraph", t, block_id=f"b{i}")
                                             for i, t in enumerate(("one", "two", "three"), 1)],
                                   doc_id="doc-other", provenance=self.package.document["provenance"])
        elsewhere.signatures = copy.deepcopy(signed.signatures)
        self.assertEqual(self.state(elsewhere), ["modified"])

    def test_several_signers_and_countersignatures(self) -> None:
        first = self.sign(purpose="authorship")
        second = self.sign(first, self.bob, purpose="review", statement="Reviewed")
        self.assertEqual(self.state(second, trusted_keys=[self.alice.key_id, self.bob.key_id]),
                         ["valid_trusted"] * 2)
        ids = [r["id"] for r in second.signatures]
        counter = self.sign(second, self.bob, purpose="countersignature", countersign=[ids[0]])
        self.assertEqual(self.state(counter), ["valid_unknown_signer"] * 3)
        broken = counter.copy()
        broken.signatures[0] = copy.deepcopy(second.signatures[0])
        broken.signatures[0]["signature"] = sd._b64e(b"\x01" * 64)
        self.assertEqual(self.state(broken)[0], "invalid")
        self.assertEqual(self.state(broken)[2], "invalid")  # The countersignature notices too.
        removed = counter.copy()
        del removed.signatures[0]
        self.assertEqual(self.state(removed)[1], "invalid")
        for kwargs in ({"purpose": "countersignature"}, {"countersign": [ids[0]]},
                       {"purpose": "countersignature", "countersign": ["nope"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(sd.SLDCValidationError):
                self.sign(second, self.bob, **kwargs)

    def test_signing_rules(self) -> None:
        signed = self.sign()
        with self.assertRaises(sd.SLDCValidationError):
            self.sign(signed, signature_id=signed.signatures[0]["id"])  # Duplicate id.
        for bad in ({"purpose": "love"}, {"signed_at": "yesterday"}, {"statement": "x" * 1001}):
            with self.subTest(bad=bad), self.assertRaises(sd.SLDCValidationError):
                self.sign(**bad)
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.sign_package(sd.SLDCPackage(sample_document(), version=sd.LEGACY_VERSION), self.alice)
        legacy_signed = types.SimpleNamespace(algorithm="rsa", key_id="0" * 64, public_key=b"")
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.sign_package(self.package, legacy_signed)
        with mock.patch.object(sd, "MAX_SIGNATURES", 1), self.assertRaises(sd.SLDCValidationError):
            self.sign(signed, self.bob)
        self.assertEqual(self.sign(signature_id="mine").signatures[0]["id"], "mine")

    def test_an_unknown_algorithm_is_indeterminate_not_valid_or_invalid(self) -> None:
        signed = self.sign()
        record = signed.signatures[0]
        record["algorithm"] = record["payload"]["algorithm"] = "ml-dsa-65"
        self.assertEqual(self.state(sd.loads_package(sd.dumps_package(signed))), ["indeterminate"])

    def test_a_malformed_signature_record_is_a_validation_error(self) -> None:
        signed = self.sign()
        for mutate in (lambda r: r.pop("signature"), lambda r: r.update(extra=1),
                       lambda r: r.update(schema="other/1"), lambda r: r.update(key_id="xyz"),
                       lambda r: r["payload"].update(purpose="x"), lambda r: r["payload"].pop("covered"),
                       lambda r: r["payload"].update(covered=[]), lambda r: r["payload"].update(version="x")):
            broken = signed.copy()
            mutate(broken.signatures[0])
            with self.assertRaises(sd.SLDCValidationError):
                sd.dumps_package(broken)

    def test_signing_survives_migration_and_edits_start_a_new_revision(self) -> None:
        legacy = sd.SLDCPackage(sample_document(), version=sd.LEGACY_VERSION)
        migrated, _ = sd.migrate_package(legacy)
        signed = self.sign(migrated)
        self.assertEqual(self.state(signed), ["valid_unknown_signer"])
        draft = sd.next_revision(signed)
        self.assertEqual(draft.signatures, [])  # Signatures belong to the revision they signed.
        draft.document["blocks"][0]["text"] = "Edited"
        resigned = self.sign(draft)
        self.assertNotEqual(sd.revision_id(resigned), sd.revision_id(signed))
        self.assertEqual(sd.verify_package(resigned)[0]["signed_revision_id"], sd.revision_id(resigned))
        self.assertEqual(sd.verify_package(signed)[0]["state"], "valid_unknown_signer")  # The old file still verifies.

    def test_signer_never_exposes_or_serialises_its_private_key(self) -> None:
        text = repr(self.alice) + str(self.alice.__dict__.keys())
        self.assertNotIn(bytes(range(32)).hex(), text)
        self.assertIn(self.alice.key_id[:16], repr(self.alice))
        with self.assertRaises(TypeError):
            pickle.dumps(self.alice)
        with self.assertRaises(sd.SLDCValidationError):
            sd.Ed25519Signer.from_seed(b"short")
        with self.assertRaises(sd.SLDCValidationError):
            sd.Ed25519Signer("not a key")
        self.assertEqual(sd.key_fingerprint(self.alice.public_key), self.alice.key_id)
        self.assertEqual(len(self.alice.public_key), 32)

    def test_known_answer_vectors_pin_canonical_encoding_and_signing(self) -> None:
        self.assertEqual(sd._canonical_json({"b": [1, "é", None], "a": {"z": True, "y": 1.5}}),
                         '{"a":{"y":1.5,"z":true},"b":[1,"é",null]}'.encode())
        package = small_package("one", "two")
        self.assertEqual(sd.revision_id(package), KNOWN_REVISION)
        signed = sd.sign_package(package, sd.Ed25519Signer.from_seed(bytes(32)),
                                 signed_at="2026-01-01T00:00:00Z", signature_id="kat")
        self.assertEqual(signed.signatures[0]["signature"], KNOWN_SIGNATURE)
        self.assertEqual(signed.signatures[0]["key_id"], KNOWN_KEY_ID)


KNOWN_REVISION = "5143e3605344525eb26c22820cda786f0234283d739f54df962561fd9b75e15c"
KNOWN_KEY_ID = "139e3940e64b5491722088d9a0d741628fc826e09475d341a780acde3c4b8070"
KNOWN_SIGNATURE = "mwZzaJGykyZia1dxOzJYAW2pWbrgbSLGN9FJHWMvjeZ84lUiWEzLaKJfBnI9ewXz_9A_HakgEtRuQNYH4c9mDQ"


class _EncryptionBase(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(sd, "MIN_SCRYPT_N", 16)  # Real settings are deliberately slow.
        patcher.start()
        self.addCleanup(patcher.stop)
        self.package = small_package()

    def password(self, value=PASSWORD, **options):
        options.setdefault("scrypt_n", 16)
        return sd.password_recipient(value, **options)


@needs_crypto
class EncryptionTests(_EncryptionBase):
    def encrypt(self, *recipients, **options):
        return sd.dumps_package(self.package, recipients=list(recipients) or [self.password()], **options)

    def test_password_round_trip_hides_the_content_and_keeps_signatures(self) -> None:
        signer = sd.Ed25519Signer.generate()
        signed = sd.sign_package(self.package, signer)
        data = sd.dumps_package(signed, recipients=[self.password(label="team")])
        self.assertTrue(data.startswith(b"PK\x03\x04"))
        for secret in (b"Small", b"doc-small", b"three", b"signatures/"):
            self.assertNotIn(secret, data)  # Not even in the compressed or listed form.
        back = sd.loads_package(data, password=PASSWORD)
        self.assertEqual(back.document, self.package.document)
        self.assertEqual([r["state"] for r in sd.verify_package(back, trusted_keys=[signer.key_id])],
                         ["valid_trusted"])
        report = sd.inspect_package(data)
        self.assertEqual(report["container"], "encrypted")
        self.assertEqual(report["encryption"]["recovery_policy"], "none")
        self.assertEqual(report["encryption"]["recipients"][0]["label"], "team")
        self.assertIsNone(report["document_id"])  # Not disclosed.
        self.assertNotIn(PASSWORD, json.dumps(report))

    def test_output_is_deliberately_not_deterministic(self) -> None:
        first, second = self.encrypt(), self.encrypt()
        self.assertNotEqual(first, second)
        for data in (first, second):
            self.assertEqual(sd.loads_package(data, password=PASSWORD).document, self.package.document)

    def test_wrong_missing_and_mismatched_credentials_fail_the_same_way(self) -> None:
        data = self.encrypt()
        priv, pub = sd.x25519_generate()
        messages = set()
        for options in ({"password": "wrong password here"}, {"private_key": priv}, {"password": ""},
                        {"password": PASSWORD.upper()}, {"private_key": b"short"}):
            with self.subTest(options=list(options)), self.assertRaises(sd.SLDCDecryptionError) as caught:
                sd.loads_package(data, **options)
            messages.add(str(caught.exception))
        with self.assertRaisesRegex(sd.SLDCDecryptionError, "supply a password"):
            sd.loads_package(data)
        self.assertEqual(len(messages), 1)  # No oracle: every failure reads alike.
        self.assertIn("file was modified", messages.pop())
        with self.assertRaises(sd.SLDCDecryptionError):
            sd.loads_package(self.encrypt(sd.x25519_recipient(pub)), password=PASSWORD)

    def test_modifying_anything_authenticated_is_detected(self) -> None:
        priv, pub = sd.x25519_generate()
        data = self.encrypt(self.password(), sd.x25519_recipient(pub, recovery=True),
                            recovery_policy="recovery_recipient")
        self.assertEqual(sd.loads_package(data, password=PASSWORD).document, self.package.document)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            payload = archive.read("payload.bin")
        flipped = bytearray(payload)
        flipped[len(flipped) // 2] ^= 1
        cases = {
            "ciphertext bit": reseal(data, replace={"payload.bin": bytes(flipped)}),
            "truncated tag": reseal(data, replace={"payload.bin": payload[:-1]}, manifest=lambda m: m[
                "encryption"].update(ciphertext_size=len(payload) - 1)),
            "recovery policy downgraded": reseal(data, manifest=lambda m: (
                m["encryption"].update(recovery_policy="none"),
                [r.update(role="primary") for r in m["encryption"]["recipients"]])),
            "nonce": reseal(data, manifest=lambda m: m["encryption"].update(nonce=sd._b64e(bytes(12)))),
            "version": reseal(data, manifest={"version": "1.4.0"}),
        }
        for label, forged in cases.items():
            with self.subTest(label), self.assertRaises(sd.SLDCDecryptionError):
                sd.loads_package(forged, password=PASSWORD)
        wrapped = reseal(data, manifest=lambda m: m["encryption"]["recipients"][0].update(
            wrapped=sd._b64e(bytes(48))))
        with self.assertRaises(sd.SLDCDecryptionError):
            sd.loads_package(wrapped, password=PASSWORD)
        self.assertEqual(sd.loads_package(wrapped, private_key=priv).document, self.package.document)

    def test_x25519_recipients(self) -> None:
        priv, pub = sd.x25519_generate()
        self.assertEqual((len(priv), len(pub)), (32, 32))
        data = self.encrypt(sd.x25519_recipient(pub, label="alice"))
        self.assertEqual(sd.loads_package(data, private_key=priv).document, self.package.document)
        from cryptography.hazmat.primitives.asymmetric import x25519
        key_object = x25519.X25519PrivateKey.from_private_bytes(priv)
        self.assertEqual(sd.loads_package(data, private_key=key_object).document, self.package.document)
        recipient = sd.inspect_package(data)["encryption"]["recipients"][0]
        self.assertEqual((recipient["kind"], recipient["key_id"]), ("x25519", sd.key_fingerprint(pub)))
        other_priv, _ = sd.x25519_generate()
        with self.assertRaises(sd.SLDCDecryptionError):
            sd.loads_package(data, private_key=other_priv)
        self.assertEqual(sd.x25519_recipient(key_object.public_key()).public_key, pub)
        with self.assertRaises(sd.SLDCValidationError):
            sd.x25519_recipient(b"short")
        with self.assertRaises(sd.SLDCValidationError):
            self.encrypt(sd.x25519_recipient(bytes(32)))  # A low-order point gives no shared secret.

    def test_several_recipients_each_open_it_alone(self) -> None:
        priv, pub = sd.x25519_generate()
        second = self.password("a second shared passphrase")
        data = self.encrypt(self.password(), second, sd.x25519_recipient(pub))
        for options in ({"password": PASSWORD}, {"password": "a second shared passphrase"}, {"private_key": priv}):
            self.assertEqual(sd.loads_package(data, **options).document, self.package.document)
        self.assertEqual(len(sd.inspect_package(data)["encryption"]["recipients"]), 3)
        self.assertEqual(sd.loads_package(data, password="a second shared passphrase", private_key=b"junk").document,
                         self.package.document)

    def test_recovery_policy_is_explicit_and_enforced(self) -> None:
        recovery = self.password("the escrowed recovery phrase", recovery=True)
        data = self.encrypt(self.password(), recovery, recovery_policy="recovery_recipient")
        self.assertEqual(sd.inspect_package(data)["encryption"]["recovery_policy"], "recovery_recipient")
        self.assertEqual(sd.loads_package(data, password="the escrowed recovery phrase").document,
                         self.package.document)
        for recipients, policy in (([recovery], "none"), ([self.password()], "recovery_recipient"),
                                   ([self.password()], "maybe")):
            with self.subTest(policy=policy), self.assertRaises(sd.SLDCValidationError):
                self.encrypt(*recipients, recovery_policy=policy)
        none = self.encrypt()
        self.assertEqual(sd.inspect_package(none)["encryption"]["recovery_policy"], "none")
        with self.assertRaises(sd.SLDCDecryptionError):  # Documented: no credential, no document.
            sd.loads_package(none, password="the escrowed recovery phrase")

    def test_recipient_limits_and_validation(self) -> None:
        for bad in ([], [self.password()] * 1 + ["nobody"]):
            with self.subTest(bad=bad), self.assertRaises(sd.SLDCValidationError):
                sd.dumps_package(self.package, recipients=bad)
        many = [self.password(f"password number {n}") for n in range(sd.MAX_RECIPIENTS + 1)]
        with self.assertRaises(sd.SLDCValidationError):
            sd.dumps_package(self.package, recipients=many)
        for bad_password in ("short", "", None, 12345678):
            with self.subTest(bad_password=bad_password), self.assertRaises(sd.SLDCValidationError):
                sd.password_recipient(bad_password)
        for n in (8, 100, 2 ** 30):
            with self.subTest(n=n), self.assertRaises(sd.SLDCValidationError):
                sd.password_recipient(PASSWORD, scrypt_n=n)
        with self.assertRaises(sd.SLDCValidationError):
            sd.password_recipient(PASSWORD, label="x" * 101, scrypt_n=16)
        _, pub = sd.x25519_generate()
        with self.assertRaises(sd.SLDCValidationError):
            self.encrypt(sd.x25519_recipient(pub), sd.x25519_recipient(pub))
        self.assertNotIn(PASSWORD, repr(self.password()))

    def test_passwords_are_unicode_normalised(self) -> None:
        composed, decomposed = "pässwörd-ünïcode", "pässwörd-ünïcode"
        data = self.encrypt(self.password(composed))
        self.assertEqual(sd.loads_package(data, password=decomposed).document, self.package.document)

    def test_add_recipient_keeps_the_ciphertext_byte_for_byte(self) -> None:
        priv, pub = sd.x25519_generate()
        data = self.encrypt()
        grown = sd.add_recipient(data, sd.x25519_recipient(pub, label="late"), password=PASSWORD)
        self.assertEqual(zipfile.ZipFile(io.BytesIO(grown)).read("payload.bin"),
                         zipfile.ZipFile(io.BytesIO(data)).read("payload.bin"))
        for options in ({"password": PASSWORD}, {"private_key": priv}):
            self.assertEqual(sd.loads_package(grown, **options).document, self.package.document)
        with self.assertRaises(sd.SLDCDecryptionError):
            sd.add_recipient(data, sd.x25519_recipient(pub), password="not a recipient pw")
        with self.assertRaisesRegex(sd.SLDCValidationError, "rekey"):
            sd.add_recipient(data, self.password("recovery phrase 123", recovery=True), password=PASSWORD)
        with self.assertRaises(sd.SLDCValidationError):
            sd.add_recipient(grown, sd.x25519_recipient(pub), password=PASSWORD)  # Same key twice.
        with self.assertRaises(sd.SLDCValidationError):
            sd.add_recipient(sd.dumps_package(self.package), self.password(), password=PASSWORD)

    def test_rekey_protects_the_new_copy_only(self) -> None:
        priv, pub = sd.x25519_generate()
        old = self.encrypt(self.password(), sd.x25519_recipient(pub))
        new = sd.rekey(old, [self.password("a brand new passphrase")], password=PASSWORD)
        self.assertEqual(sd.loads_package(new, password="a brand new passphrase").document, self.package.document)
        for credentials in ({"password": PASSWORD}, {"private_key": priv}):
            with self.assertRaises(sd.SLDCDecryptionError):
                sd.loads_package(new, **credentials)
        self.assertEqual(sd.loads_package(old, private_key=priv).document, self.package.document)  # The old copy is out there.
        self.assertNotEqual(zipfile.ZipFile(io.BytesIO(new)).read("payload.bin"),
                            zipfile.ZipFile(io.BytesIO(old)).read("payload.bin"))
        with self.assertRaises(sd.SLDCDecryptionError):
            sd.rekey(old, [self.password()], password="wrong wrong wrong")

    def test_hostile_envelopes_are_rejected_before_any_expensive_work(self) -> None:
        data = self.encrypt()

        def forge(change):
            def apply(manifest):
                change(manifest["encryption"])
            return reseal(data, manifest=apply)
        kdf = lambda e: e["recipients"][0]["kdf"]
        cases = {
            "scrypt n too large": lambda e: kdf(e).update(n=2 ** 28),
            "scrypt n too small": lambda e: kdf(e).update(n=2),
            "scrypt n not a power of two": lambda e: kdf(e).update(n=17),
            "scrypt r": lambda e: kdf(e).update(r=64),
            "scrypt p": lambda e: kdf(e).update(p=64),
            "kdf unknown": lambda e: kdf(e).update(name="md5"),
            "kdf extra key": lambda e: kdf(e).update(memory=1),
            "ciphertext size lies": lambda e: e.update(ciphertext_size=10 ** 9),
            "no recipients": lambda e: e.update(recipients=[]),
            "extra key": lambda e: e.update(note="hi"),
            "bad nonce": lambda e: e.update(nonce="AAAA"),
            "unknown kind": lambda e: e["recipients"][0].update(kind="rsa"),
            "duplicate recipient": lambda e: e["recipients"].append(copy.deepcopy(e["recipients"][0])),
            "wrapped length": lambda e: e["recipients"][0].update(wrapped=sd._b64e(b"x" * 10)),
            "unknown role": lambda e: e["recipients"][0].update(role="owner"),
        }
        for label, change in cases.items():
            with self.subTest(label), self.assertRaises(sd.SLDCValidationError):
                sd.inspect_package(forge(change))
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.inspect_package(forge(lambda e: e.update(scheme="slai.sldc.enc/9")))
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.inspect_package(forge(lambda e: e.update(cipher="aes-128-cbc")))
        with self.assertRaises(sd.SLDCValidationError):
            sd.inspect_package(reseal(data, manifest={"requires": []}))
        with self.assertRaises(sd.SLDCValidationError):
            sd.inspect_package(reseal(data, add={"document.json": b"{}"}))
        with self.assertRaises(sd.SLDCValidationError):
            sd.inspect_package(reseal(data, drop=["payload.bin"]))

    def test_nested_and_legacy_apis_refuse_encrypted_files(self) -> None:
        data = self.encrypt()
        with self.assertRaisesRegex(sd.SLDCUnsupportedError, "encrypted"):
            sd.loads_sldc(data)
        target = self.tmp / "enc.sldc"
        target.write_bytes(data)
        with self.assertRaises(sd.SLDCUnsupportedError):
            sd.read_sldc(target)
        self.assertFalse(sd.is_sldc(target))
        self.assertTrue(sd.is_sldc(target, strict=False))
        nested = sd._encrypt(data, [self.password()], "none")
        with self.assertRaisesRegex(sd.SLDCValidationError, "Nested"):
            sd.loads_package(nested, password=PASSWORD)
        self.assertEqual(sd.read_package(target, password=PASSWORD).document, self.package.document)
        with self.assertRaises(sd.SLDCValidationError):
            sd.rekey(sd.dumps_package(self.package), [self.password()], password=PASSWORD)

    def test_write_package_can_encrypt_atomically(self) -> None:
        target = self.tmp / "secret.sldc"
        sd.write_package(target, self.package, recipients=[self.password()])
        self.assertEqual(sd.read_package(target, password=PASSWORD).document, self.package.document)
        self.assertEqual(sd.inspect_package(target)["container"], "encrypted")

    def test_files_can_be_shared_without_the_password_in_the_file(self) -> None:
        data = self.encrypt()
        self.assertNotIn(PASSWORD.encode(), data)
        self.assertNotIn(sd._b64e(PASSWORD.encode()).encode(), data)

    def test_login_codes_are_not_document_keys(self) -> None:
        # A document password is never derived from anything else: the same password gives a
        # different wrapped key each time (random salt and nonce), so it cannot double as an account secret.
        first = sd.inspect_package(self.encrypt())
        second = sd.inspect_package(self.encrypt())
        self.assertNotEqual(first["encryption"]["recipients"][0]["id"], second["encryption"]["recipients"][0]["id"])


class OptionalDependencyTests(_EncryptionBase):
    """Reading and writing unsigned, unencrypted documents never needs ``cryptography``."""

    def blocked(self):
        names = [n for n in sys.modules if n == "cryptography" or n.startswith("cryptography.")]
        patches = {n: None for n in names}
        patches.update({"cryptography": None, "cryptography.exceptions": None,
                        "cryptography.hazmat.primitives": None})
        return mock.patch.dict(sys.modules, patches)

    def test_plain_documents_work_without_cryptography(self) -> None:
        with self.blocked():
            data = sd.dumps_package(rich_package())
            self.assertEqual(sd.loads_package(data).document, rich_package().document)
            sd.migrate_package(sd.SLDCPackage(sample_document(), version=sd.LEGACY_VERSION))
            sd.merge_packages(self.package, self.package, self.package)
            sd.add_comment(self.package, "b1", "x", author="a")

    def test_signing_and_encryption_say_what_is_missing(self) -> None:
        with self.blocked():
            for call in (lambda: sd.Ed25519Signer.generate(),
                         lambda: sd.dumps_package(self.package, recipients=[self.password()]),
                         lambda: sd.x25519_generate()):
                with self.assertRaisesRegex(sd.SLDCDependencyError, "cryptography"):
                    call()
            self.assertEqual(sd.verify_package(self.package), [])  # Nothing to verify, nothing needed.

    @needs_crypto
    def test_encrypted_and_signed_files_are_inspectable_but_never_falsely_verified(self) -> None:
        data = sd.dumps_package(self.package, recipients=[self.password()])
        signed = sd.sign_package(self.package, sd.Ed25519Signer.generate())
        with self.blocked():
            self.assertEqual(sd.inspect_package(data)["container"], "encrypted")
            self.assertEqual(sd.loads_package(sd.dumps_package(signed)).signatures, signed.signatures)
            with self.assertRaises(sd.SLDCDependencyError):
                sd.loads_package(data, password=PASSWORD)
            with self.assertRaises(sd.SLDCDependencyError):  # An error, never a quiet "valid".
                sd.verify_package(signed)


class CommandLinePackageTests(_TempDirTestCase):
    def run_cli(self, *argv: str, env=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.dict(os.environ, env or {}):
            code = sd.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def source(self, name="notes.md", text="# Notes\n\nHello wörld.\n"):
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_pack_writes_1x_by_default_and_0_1_on_request(self) -> None:
        source = self.source()
        modern, legacy = self.tmp / "modern.sldc", self.tmp / "legacy.sldc"
        self.assertEqual(self.run_cli("pack", str(source), str(modern))[0], 0)
        self.assertEqual(self.run_cli("pack", str(source), str(legacy), "--legacy")[0], 0)
        self.assertEqual(sd.read_package(modern).version, sd.FORMAT_VERSION)
        self.assertEqual(sd.read_sldc(legacy)["title"], "notes")
        self.assertEqual(sd.extract_text(sd.read_package(modern)), sd.extract_text(sd.read_sldc(legacy)))
        self.assertEqual(sd.read_package(modern).document["id"], sd.read_sldc(legacy)["id"])
        code, out, _ = self.run_cli("check", str(modern), str(legacy))
        self.assertEqual((code, out.count("OK"), "version=1.0.0" in out, "version=0.1.0" in out),
                         (0, 2, True, True))

    def test_info_is_json_about_the_manifest_only(self) -> None:
        path = self.tmp / "r.sldc"
        sd.write_package(path, rich_package())
        code, out, _ = self.run_cli("info", str(path))
        report = json.loads(out)
        self.assertEqual((code, report["container"], report["requires"]), (0, "package", ["embeds", "figures", "tables"]))
        self.assertEqual(self.run_cli("info", str(self.tmp / "missing.sldc"))[0], 1)

    def test_migrate_and_export_with_reports(self) -> None:
        old, new, back = self.tmp / "old.sldc", self.tmp / "new.sldc", self.tmp / "back.sldc"
        sd.write_sldc(old, sample_document())
        report_path = self.tmp / "migration.json"
        self.assertEqual(self.run_cli("migrate", str(old), str(new), "--report", str(report_path))[0], 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual((report["migrated"], report["lossy"]), (True, False))
        self.assertEqual(self.run_cli("migrate", str(old), str(new))[0], 1)  # Never overwrites.
        code, _, err = self.run_cli("migrate", str(new), str(self.tmp / "x.sldc"))
        self.assertEqual(code, 1)
        self.assertIn("nothing to migrate", err)
        export_report = self.tmp / "export.json"
        self.assertEqual(self.run_cli("export-legacy", str(new), str(back), "--report", str(export_report))[0], 0)
        self.assertEqual(sd.read_sldc(back)["metadata"]["sldc_export"]["lossy"], True)
        self.assertTrue(json.loads(export_report.read_text(encoding="utf-8"))["lossy"])

    @needs_crypto
    def test_passwords_come_from_the_environment_never_from_arguments(self) -> None:
        path = self.tmp / "secret.sldc"
        with mock.patch.object(sd, "MIN_SCRYPT_N", 16):
            sd.write_package(path, small_package("classified"),
                             recipients=[sd.password_recipient(PASSWORD, scrypt_n=16)])
            code, out, _ = self.run_cli("check", str(path))
            self.assertEqual((code, out.count("FAIL")), (1, 1))
            code, out, _ = self.run_cli("check", str(path), "--password-env", "SLDC_TEST_PW",
                                        env={"SLDC_TEST_PW": PASSWORD})
            self.assertEqual((code, out.count("OK")), (0, 1))
            code, out, _ = self.run_cli("text", str(path), "--password-env", "SLDC_TEST_PW",
                                        env={"SLDC_TEST_PW": PASSWORD})
            self.assertEqual((code, out), (0, "classified\n"))
            code, _, err = self.run_cli("check", str(path), "--password-env", "SLDC_TEST_UNSET")
            self.assertEqual(code, 1)
            self.assertIn("SLDC_TEST_UNSET", err)
            code, out, _ = self.run_cli("check", str(path), "--password-env", "SLDC_TEST_PW",
                                        env={"SLDC_TEST_PW": "wrong password here"})
            self.assertEqual((code, out.count("FAIL")), (1, 1))
            self.assertNotIn(PASSWORD, out)
        with self.assertRaises(SystemExit):
            self.run_cli("check", str(path), "--password", PASSWORD)  # No such argument exists.

    @needs_crypto
    def test_verify_exits_zero_only_for_trusted_valid_signatures(self) -> None:
        signer = sd.Ed25519Signer.generate()
        signed = self.tmp / "signed.sldc"
        unsigned = self.tmp / "unsigned.sldc"
        edited = self.tmp / "edited.sldc"
        package = small_package()
        sd.write_package(signed, sd.sign_package(package, signer, signed_at="2026-01-01T00:00:00Z"))
        sd.write_package(unsigned, package)
        forged = sd.sign_package(package, signer)
        forged.document["title"] = "Edited after signing"
        sd.write_package(edited, forged)
        code, out, _ = self.run_cli("verify", str(signed), "--trusted", signer.key_id)
        self.assertEqual((code, "valid_trusted" in out), (0, True))
        self.assertIn("claimed_time=2026-01-01T00:00:00Z", out)
        code, out, _ = self.run_cli("verify", str(signed))
        self.assertEqual((code, "valid_unknown_signer" in out), (1, True))
        self.assertEqual(self.run_cli("verify", str(signed), "--trusted", signer.key_id,
                                      "--revoked", signer.key_id)[0], 1)
        code, out, _ = self.run_cli("verify", str(unsigned))
        self.assertEqual((code, "no signatures" in out), (1, True))
        code, out, _ = self.run_cli("verify", str(edited), "--trusted", signer.key_id)
        self.assertEqual((code, "modified" in out), (1, True))
        self.assertIn("signed parts changed", out)

    def test_existing_commands_keep_their_behaviour(self) -> None:
        source = self.source("a.txt", "x")
        target = self.tmp / "a.sldc"
        self.assertEqual(self.run_cli("pack", str(source), str(target))[0], 0)
        code, _, err = self.run_cli("pack", str(source), str(target))
        self.assertEqual((code, "error" in err), (1, True))
        code, out, _ = self.run_cli("check", str(target), str(self.tmp / "gone"))
        self.assertEqual((code, out.count("FAIL"), out.count("OK")), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
