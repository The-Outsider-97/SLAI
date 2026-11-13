"""Tests for the Reader foundation and capability engines.

Foundations: DetectionEngine, IntegrityEngine, JobEngine. Capabilities: CompressionEngine, SalvageEngine.
Everything runs against the real engines and the real ReaderAgent (no mocks beyond one deliberate
removal of ``os.link``); only temporary files are created.

Run from the repository root:  python -m unittest tests.test_reader_engines -v
"""
from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import os
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

from src.agents.reader.compression_engine import CompressionEngine
from src.agents.reader.detection_engine import DetectionEngine
from src.agents.reader.integrity_engine import ExpansionGuard, ExpansionLimits, IntegrityEngine
from src.agents.reader.job_engine import JobEngine
from src.agents.reader.parser_engine import ParserEngine
from src.agents.reader.reader_memory import ReaderMemory
from src.agents.reader.salvage_engine import SalvageEngine
from src.agents.reader.utils.reader_error import (
    CompressionError,
    IntegrityError,
    JobNotFoundError,
    JobStateError,
    SalvageError,
)
from src.agents.reader.utils.reader_helpers import publish_file, sha256_file

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixture builders (standard library only)
# ---------------------------------------------------------------------------


def docx_bytes(paragraphs: int = 300, *, method: int = zipfile.ZIP_DEFLATED) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>Paragraph number {i} lorem ipsum dolor</w:t></w:r></w:p>" for i in range(paragraphs))
    document = f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", method) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def odt_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        archive.writestr("content.xml", "<c>" + "x" * 50_000 + "</c>")
    return buffer.getvalue()


def pdf_bytes(pages: int = 3) -> bytes:
    """A small, valid PDF with uncompressed text streams, built by hand so no PDF library is needed."""

    objects: list[bytes] = []
    page_ids = [4 + 2 * i for i in range(pages)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(("<< /Type /Pages /Kids [" + " ".join(f"{pid} 0 R" for pid in page_ids) + f"] /Count {pages} >>").encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index in range(pages):
        stream = f"BT /F1 12 Tf 50 700 Td (Page {index + 1} recoverable text) Tj ET".encode()
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * index} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class _TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def write(self, name: str, data) -> Path:
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class PublishFileTests(_TempDirTestCase):
    def test_never_overwrites_even_without_hard_links(self) -> None:
        """FAT/exFAT and some network shares have no hard links; publication must still be exclusive."""

        existing = self.write("out/report.txt", "first")
        staged = self.write("stage.tmp", "second")
        with mock.patch.object(os, "link", side_effect=OSError("hard links not supported")):
            published = publish_file(staged, existing)
        self.assertNotEqual(published, existing)
        self.assertEqual(existing.read_text(), "first")
        self.assertEqual(published.read_text(), "second")
        self.assertFalse(staged.exists())


# ---------------------------------------------------------------------------
# DetectionEngine
# ---------------------------------------------------------------------------


class DetectionEngineTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = DetectionEngine()

    def detect(self, path: Path) -> dict:
        return self.engine.detect(path)["payload"]

    def test_content_wins_over_a_misleading_extension(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 40
        payload = self.detect(self.write("photo.docx", png))
        self.assertEqual(payload["format_id"], "png")
        self.assertTrue(payload["extension_mismatch"])

    def test_ooxml_and_odf_are_told_apart_from_plain_zip(self) -> None:
        self.assertEqual(self.detect(self.write("a.docx", docx_bytes()))["format_id"], "docx")
        self.assertEqual(self.detect(self.write("a.odt", odt_bytes()))["format_id"], "odt")
        plain = io.BytesIO()
        with zipfile.ZipFile(plain, "w") as archive:
            archive.writestr("readme.txt", "hi")
        self.assertEqual(self.detect(self.write("a.zip", plain.getvalue()))["format_id"], "zip")

    def test_truncated_package_is_reported_damaged(self) -> None:
        good = docx_bytes()
        self.assertTrue(self.detect(self.write("cut.docx", good[: len(good) // 2]))["damaged"])

    def test_encrypted_office_file_is_not_called_a_mismatch(self) -> None:
        ole = bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 512 + "EncryptedPackage".encode("utf-16-le")
        payload = self.detect(self.write("secret.docx", ole))
        self.assertTrue(payload["encrypted"])
        self.assertFalse(payload["extension_mismatch"])

    def test_operations_are_derived_from_the_registry(self) -> None:
        self.assertTrue(self.engine.supports("pdf", "salvage"))
        self.assertFalse(self.engine.supports("png", "merge"))


# ---------------------------------------------------------------------------
# IntegrityEngine
# ---------------------------------------------------------------------------


class IntegrityEngineTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = IntegrityEngine()

    def validate(self, path: Path, **kwargs) -> dict:
        return self.engine.validate_file(path, **kwargs)["payload"]

    def test_expansion_guard_counts_real_bytes(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("big.bin", b"\x00" * (8 * 1024 * 1024))
        guard = ExpansionGuard(ExpansionLimits(max_member_bytes=1024 * 1024))
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            with self.assertRaises(IntegrityError):
                guard.inspect_zip(archive, source="bomb.zip")
            with self.assertRaises(IntegrityError):
                guard.read_member(archive, "big.bin", source="bomb.zip")

    def test_shallow_validation_reports_what_it_skipped_and_deep_finds_corruption(self) -> None:
        raw = bytearray(docx_bytes(method=zipfile.ZIP_STORED))
        marker = raw.find(b"Paragraph number 5 ")
        raw[marker] ^= 0xFF  # flip a byte inside stored content: the CRC no longer matches
        path = self.write("flip.docx", bytes(raw))
        shallow = self.validate(path, deep=False)
        deep = self.validate(path, deep=True)
        self.assertTrue(shallow["valid"])
        self.assertTrue(any(c["name"] == "zip_crc" and c["skipped"] for c in shallow["checks"]))
        self.assertFalse(deep["valid"])

    def test_corrupt_deflate_stream_is_a_finding_not_an_exception(self) -> None:
        raw = bytearray(docx_bytes())
        raw[len(raw) // 3] ^= 0xFF
        payload = self.validate(self.write("corrupt.docx", bytes(raw)), deep=True)  # must not raise zlib.error
        self.assertFalse(payload["valid"])

    def test_missing_required_member_marks_a_package_invalid(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
        payload = self.validate(self.write("hollow.docx", buffer.getvalue()), deep=True)
        self.assertFalse(payload["valid"])
        self.assertTrue(any(c["name"] == "required_members" and not c["ok"] for c in payload["checks"]))

    def test_unsafe_tar_member_and_xml_entities_are_rejected(self) -> None:
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as archive:
            info = tarfile.TarInfo("../evil.txt")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"hi"))
        self.assertFalse(self.validate(self.write("evil.tar", raw.getvalue()), deep=True)["valid"])
        entity = '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "y">]><a>&x;</a>'
        self.assertFalse(self.validate(self.write("entity.xml", entity))["valid"])

    def test_truncated_gzip_fails_only_in_deep_mode(self) -> None:
        data = gzip.compress(os.urandom(50_000))
        path = self.write("cut.gz", data[: len(data) // 2])
        self.assertTrue(self.validate(path, deep=False)["valid"])
        self.assertFalse(self.validate(path, deep=True)["valid"])

    def test_jsonl_and_csv_tails_are_flagged(self) -> None:
        self.assertFalse(self.validate(self.write("a.jsonl", '{"a":1}\n{"a":'))["valid"])
        csv_payload = self.validate(self.write("a.csv", 'id,name\n1,"unterminated'))
        self.assertTrue(any(c["name"] == "csv_strict_parse" and not c["ok"] for c in csv_payload["checks"]))

    def test_verify_artifact_detects_change(self) -> None:
        path = self.write("a.txt", "one")
        digest = sha256_file(path)
        self.assertEqual(self.engine.verify_artifact(path, expected_sha256=digest)["status"], "ok")
        path.write_text("two")
        self.assertEqual(self.engine.verify_artifact(path, expected_sha256=digest)["status"], "error")


# ---------------------------------------------------------------------------
# JobEngine
# ---------------------------------------------------------------------------


def _job_engine(root: Path) -> JobEngine:
    memory = ReaderMemory({
        "checkpoint_dir": str(root / "jobs"), "cache_dir": str(root / "cache"), "audit_dir": str(root / "audit"),
        "max_checkpoint_entries": 100_000, "write_indexes": False, "enable_audit_log": False, "checkpoint_prefix": "job",
    })
    return JobEngine(memory=memory)


class JobEngineTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = _job_engine(self.tmp)

    def outputs(self, count: int) -> list[Path]:
        return [self.write(f"o{i}.txt", f"result {i}") for i in range(count)]

    def test_idempotent_create_returns_the_same_job(self) -> None:
        first = self.engine.create_job("convert", ["a", "b"], idempotency_key=["x"])["payload"]
        second = self.engine.create_job("convert", ["c"], idempotency_key=["x"])["payload"]
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(first["job"]["job_id"], second["job"]["job_id"])

    def test_failed_item_makes_a_partial_job_and_resume_redoes_only_it(self) -> None:
        paths = self.outputs(3)
        job_id = self.engine.create_job("convert", [{"key": str(p)} for p in paths])["payload"]["job"]["job_id"]
        calls: list[str] = []

        def handler(item):
            calls.append(item["key"])
            if item["key"].endswith("o1.txt") and len(calls) == 2:
                raise RuntimeError("boom")
            return {"result": {"ok": True}, "artifacts": [item["key"]]}

        first = self.engine.run(job_id, handler)["payload"]
        self.assertEqual(first["job"]["status"], "partial")
        calls.clear()
        second = self.engine.run(job_id, handler, retry_failed=True)["payload"]
        self.assertEqual(second["job"]["status"], "completed")
        self.assertEqual((second["reused"], calls), (2, [str(paths[1])]))

    def test_changed_artifact_is_not_reused(self) -> None:
        paths = self.outputs(1)
        job_id = self.engine.create_job("convert", [{"key": str(paths[0])}])["payload"]["job"]["job_id"]
        self.engine.run(job_id, lambda item: {"artifacts": [item["key"]]})
        item_id = self.engine.get_job(job_id)["item_order"][0]
        self.assertIsNotNone(self.engine.reusable_item(job_id, item_id))
        paths[0].write_text("tampered")
        self.assertIsNone(self.engine.reusable_item(job_id, item_id))

    def test_lifecycle_rules(self) -> None:
        job_id = self.engine.create_job("convert", ["a"])["payload"]["job"]["job_id"]
        self.engine.run(job_id, lambda item: None)
        with self.assertRaises(JobStateError):
            self.engine.start_job(job_id)  # completed jobs are terminal
        with self.assertRaises(JobNotFoundError):
            self.engine.get_job("missing_job")

    def test_running_job_is_not_taken_over_unless_stale_or_forced(self) -> None:
        job_id = self.engine.create_job("convert", ["a"])["payload"]["job"]["job_id"]
        self.engine.start_job(job_id)
        with self.assertRaises(JobStateError):
            self.engine.start_job(job_id)
        self.engine.start_job(job_id, force=True)
        self.assertEqual(self.engine.get_job(job_id)["status"], "running")

    def test_records_are_stored_exactly_not_redacted(self) -> None:
        """Diagnostic checkpoints mask secrets and truncate long strings; job records must round-trip verbatim."""

        job_id = self.engine.create_job("convert", ["a"])["payload"]["job"]["job_id"]
        long_text = "x" * 5000
        self.engine.start_job(job_id)
        item_id = self.engine.get_job(job_id)["item_order"][0]
        self.engine.complete_item(job_id, item_id, {"token": "keep-me", "long": long_text})
        fresh = _job_engine(self.tmp).get_job(job_id, fresh=True)  # a second engine reads it from disk
        self.assertEqual(fresh["items"][item_id]["result"], {"token": "keep-me", "long": long_text})

    def test_job_survives_kill_9_and_resume_skips_finished_items(self) -> None:
        script = textwrap.dedent(
            f"""
            import sys, time
            from pathlib import Path
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from src.agents.reader.job_engine import JobEngine
            from src.agents.reader.reader_memory import ReaderMemory
            root = Path({str(self.tmp)!r})
            memory = ReaderMemory({{"checkpoint_dir": str(root / "jobs"), "cache_dir": str(root / "cache"),
                "audit_dir": str(root / "audit"), "max_checkpoint_entries": 100000, "write_indexes": False,
                "enable_audit_log": False, "checkpoint_prefix": "job"}})
            engine = JobEngine(memory=memory)
            engine.create_job("convert", [{{"key": "item%d" % i}} for i in range(6)], job_id="killme")
            def handler(item):
                out = root / (item["key"] + ".out")
                out.write_text("done " + item["key"])
                if item["key"] == "item2":
                    (root / "ready").write_text("1")
                    time.sleep(120)  # the parent kills this process here, mid-item
                return {{"artifacts": [out]}}
            engine.run("killme", handler)
            """
        )
        child = subprocess.Popen([sys.executable, "-I", "-c", script], cwd=str(self.tmp), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.time() + 60
            while not (self.tmp / "ready").exists():
                self.assertIsNone(child.poll(), "child exited before reaching the kill point")
                self.assertLess(time.time(), deadline, "child never reached the kill point")
                time.sleep(0.05)
        finally:
            child.kill()
            child.wait()

        engine = _job_engine(self.tmp)
        before = engine.get_job("killme", fresh=True)
        self.assertEqual(before["status"], "running")  # the journal says it was mid-run
        self.assertEqual(before["counts"]["done"], 2)

        handled: list[str] = []

        def finish(item):
            handled.append(item["key"])
            out = self.tmp / (item["key"] + ".out")
            out.write_text("done " + item["key"])
            return {"artifacts": [out]}

        outcome = engine.run("killme", finish, force=True)["payload"]
        self.assertEqual(outcome["job"]["status"], "completed")
        self.assertEqual(outcome["reused"], 2)
        self.assertEqual(handled, ["item2", "item3", "item4", "item5"])


# ---------------------------------------------------------------------------
# CompressionEngine
# ---------------------------------------------------------------------------


class CompressionEngineTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = CompressionEngine()
        self.out = self.tmp / "out"

    def test_package_formats_round_trip_and_are_deterministic(self) -> None:
        first = self.write("a.txt", "alpha " * 5000)
        second = self.write("b.txt", "beta " * 5000)
        for fmt in ("zip", "tar.gz", "tar.xz", "tar.bz2", "tar"):
            report = self.engine.package([first, second], self.out, package_format=fmt, name=f"p_{fmt.replace('.', '_')}")["payload"]["report"]
            self.assertTrue(report["verified"], fmt)
            self.assertEqual(report["verified_by"], "round_trip_sha256")
        one = self.engine.package([first, second], self.out, name="same")["payload"]["output_path"]
        two = self.engine.package([first, second], self.out, name="same")["payload"]["output_path"]
        self.assertNotEqual(one, two)  # never overwrites
        self.assertEqual(sha256_file(one), sha256_file(two))  # equal input, equal bytes

    def test_packaging_a_folder_does_not_swallow_its_own_output(self) -> None:
        self.write("folder/a.txt", "a")
        self.write("folder/sub/b.txt", "b")
        result = self.engine.package([self.tmp / "folder"], self.tmp / "folder" / "archives", name="f")["payload"]
        self.assertEqual(sorted(result["members"]), ["folder/a.txt", "folder/sub/b.txt"])

    def test_optimize_zip_package_is_lossless_and_keeps_odf_mimetype_first(self) -> None:
        stored = self.write("big.docx", docx_bytes(3000, method=zipfile.ZIP_STORED))
        result = self.engine.optimize(stored, self.out)["payload"]
        self.assertEqual(result["outcome"], "ok")
        self.assertEqual(result["report"]["verified_by"], "member_sha256")
        self.assertLess(result["report"]["output_bytes"], result["report"]["input_bytes"])
        odt = self.engine.optimize(self.write("a.odt", odt_bytes()), self.out)["payload"]
        with zipfile.ZipFile(odt["output_path"]) as archive:
            first = archive.infolist()[0]
            self.assertEqual((first.filename, first.compress_type), ("mimetype", zipfile.ZIP_STORED))

    def test_optimize_that_gains_nothing_keeps_the_original_and_writes_no_file(self) -> None:
        small = self.write("small.docx", docx_bytes(1))
        result = self.engine.optimize(small, self.out)["payload"]
        self.assertEqual(result["outcome"], "no_gain")
        self.assertIsNone(result["output_path"])
        self.assertEqual(list(self.out.glob("*")) if self.out.exists() else [], [])

    def test_unsupported_and_lossy_requests_are_explicit(self) -> None:
        with self.assertRaises(CompressionError):
            self.engine.optimize(self.write("a.txt", "hello"), self.out)

    def test_expand_skips_unsafe_members_and_never_writes_outside_the_target(self) -> None:
        evil = io.BytesIO()
        with zipfile.ZipFile(evil, "w") as archive:
            archive.writestr("../escape.txt", "x")
            archive.writestr("CON.txt", "reserved")
            archive.writestr("good.txt", "ok")
        result = self.engine.expand(self.write("evil.zip", evil.getvalue()), self.tmp / "x")["payload"]
        self.assertEqual(sorted(item["path"] for item in result["files"]), ["_CON.txt", "good.txt"])
        self.assertEqual(result["skipped"][0]["reason"], "unsafe_path")
        self.assertFalse((self.tmp / "escape.txt").exists())
        self.assertFalse((self.tmp / "x" / "escape.txt").exists())

    def test_expand_skips_tar_links(self) -> None:
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            info = tarfile.TarInfo("ok.txt")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"hi"))
            link = tarfile.TarInfo("lnk")
            link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(link)
        result = self.engine.expand(self.write("a.tar.gz", raw.getvalue()), self.tmp / "x")["payload"]
        self.assertEqual([item["path"] for item in result["files"]], ["ok.txt"])
        self.assertEqual(result["skipped"][0]["reason"], "link_or_special")

    def test_expand_bomb_is_stopped_on_real_bytes_and_leaves_nothing_behind(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("z.bin", b"\x00" * (20 * 1024 * 1024))
        engine = CompressionEngine()
        engine.guard.limits = ExpansionLimits(max_member_bytes=1024 * 1024, max_total_bytes=0, max_ratio=0, max_members=100)
        with self.assertRaises(IntegrityError):
            engine.expand(self.write("bomb.zip", buffer.getvalue()), self.tmp / "x")
        self.assertEqual(list((self.tmp / "x").rglob("*")), [])

    def test_stream_compression_round_trips(self) -> None:
        source = self.write("data.txt", "line\n" * 20_000)
        for method in ("gz", "bz2", "xz"):
            packed = self.engine.compress_file(source, self.out, method=method)["payload"]
            self.assertTrue(packed["report"]["verified"])
            expanded = self.engine.expand(packed["output_path"], self.tmp / f"x_{method}")["payload"]
            self.assertEqual(expanded["files"][0]["sha256"], sha256_file(source))


# ---------------------------------------------------------------------------
# SalvageEngine
# ---------------------------------------------------------------------------


class SalvageEngineTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.engine = SalvageEngine()
        self.parser = ParserEngine()
        self.out = self.tmp / "out"

    def test_truncated_docx_is_rebuilt_and_parses(self) -> None:
        good = docx_bytes()
        result = self.engine.salvage(self.write("cut.docx", good[: int(len(good) * 0.55)]), self.out)["payload"]
        self.assertEqual(result["report"]["level"], "partial")
        self.assertTrue(any(part["name"] == "word/document.xml" for part in result["report"]["lost_parts"]))
        parsed = self.parser.parse(result["repaired_path"])
        self.assertIn("Paragraph number 0 ", parsed["content"])
        self.assertLess(len(parsed["content"]), len(self.parser.parse(self.write("whole.docx", good))["content"]))

    def test_intact_file_needs_no_salvage(self) -> None:
        result = self.engine.salvage(self.write("ok.docx", docx_bytes()), self.out)["payload"]
        self.assertEqual((result["outcome"], result["repaired_path"]), ("not_needed", None))

    def test_unrecoverable_file_raises_with_the_report_attached(self) -> None:
        with self.assertRaises(SalvageError):
            self.engine.salvage(self.write("none.json", "not json at all"), self.out)
        with self.assertRaises(SalvageError):
            self.engine.salvage(self.write("garbage.docx", b"PK\x03\x04" + b"\x00" * 40), self.out)

    def test_original_file_is_never_modified(self) -> None:
        source = self.write("cut.docx", docx_bytes()[:3000])
        before = sha256_file(source)
        try:
            self.engine.salvage(source, self.out)
        except SalvageError:
            pass
        self.assertEqual(sha256_file(source), before)

    def test_truncated_json_drops_a_half_written_record_instead_of_returning_it_with_fields_missing(self) -> None:
        source = self.write("data.json", '{"items": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "be')
        result = self.engine.salvage(source, self.out)["payload"]
        self.assertEqual(json.loads(Path(result["repaired_path"]).read_text()), {"items": [{"id": 1, "name": "alpha"}]})
        self.assertEqual(result["report"]["level"], "partial")

    def test_json_repair_respects_escapes_and_does_not_trust_a_cut_number(self) -> None:
        escaped = self.write("e.json", '{"a": "x\\"y", "b": [1, 2, {"c": "d\\\\"}, 3')
        self.assertEqual(json.loads(Path(self.engine.salvage(escaped, self.out)["payload"]["repaired_path"]).read_text()),
                         {"a": 'x"y', "b": [1, 2, {"c": "d\\"}]})
        bare = self.write("n.json", "[1, 2, 3")
        self.assertEqual(json.loads(Path(self.engine.salvage(bare, self.out)["payload"]["repaired_path"]).read_text()), [1, 2])

    def test_torn_csv_jsonl_and_xml_tails(self) -> None:
        csv_result = self.engine.salvage(self.write("a.csv", "id,name,score\n1,ann,10\n2,bob,20\n3,cy"), self.out)["payload"]
        self.assertEqual(Path(csv_result["repaired_path"]).read_text(), "id,name,score\n1,ann,10\n2,bob,20\n")
        quote = self.engine.salvage(self.write("q.csv", 'id,name\n1,"multi\nline"\n2,"unterminated'), self.out)["payload"]
        self.assertEqual(Path(quote["repaired_path"]).read_text(), 'id,name\n1,"multi\nline"\n')
        lines = self.engine.salvage(self.write("a.jsonl", '{"a":1}\n{"a":2}\n{"a":'), self.out)["payload"]
        self.assertEqual(len(Path(lines["repaired_path"]).read_text().splitlines()), 2)
        xml = self.engine.salvage(self.write("a.xml", "<root><a>1</a><b><c>two</c><d>thr"), self.out)["payload"]
        self.assertIn("<c>two</c>", Path(xml["repaired_path"]).read_text())

    def test_truncated_pdf_yields_pages_or_text_but_never_claims_full(self) -> None:
        pdf = pdf_bytes(4)
        cut = pdf[: pdf.index(b"Page 3")]
        result = self.engine.salvage(self.write("cut.pdf", cut), self.out)["payload"]
        self.assertIn(result["report"]["level"], {"partial", "text_only"})
        repaired = Path(result["repaired_path"])
        text = repaired.read_text(errors="replace") if repaired.suffix == ".txt" else self.parser.parse(repaired)["content"]
        self.assertIn("Page 1 recoverable text", text)

    def test_truncated_compressed_streams_and_tar(self) -> None:
        payload = json.dumps({"rows": [{"id": i, "v": "x" * 20} for i in range(2000)]}).encode()
        packed = gzip.compress(payload)
        result = self.engine.salvage(self.write("d.json.gz", packed[: len(packed) // 2]), self.out)["payload"]
        self.assertEqual(result["report"]["level"], "partial")
        self.assertIsInstance(json.loads(Path(result["repaired_path"]).read_text()), dict)

        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            for index in range(8):
                data = os.urandom(20_000)
                info = tarfile.TarInfo(f"f{index}.bin")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        tar_result = self.engine.salvage(self.write("a.tar.gz", raw.getvalue()[: int(len(raw.getvalue()) * 0.55)]), self.out)["payload"]
        names = [part["name"] for part in tar_result["report"]["recovered_parts"]]
        self.assertTrue(0 < len(names) < 8)
        with tarfile.open(tar_result["repaired_path"]) as archive:
            self.assertEqual(sorted(archive.getnames()), sorted(names))


# ---------------------------------------------------------------------------
# ReaderAgent integration
# ---------------------------------------------------------------------------


class ReaderAgentEngineTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        from src.agents.agent_factory import AgentFactory
        from src.agents.collaborative.shared_memory import SharedMemory

        self.agent = AgentFactory().create("reader", shared_memory=SharedMemory())
        self.agent._job_engine = _job_engine(self.tmp)  # keep the job store inside this test's directory

    def run_task(self, **task):
        task.setdefault("output_dir", str(self.tmp / "out"))
        return self.agent.perform_task(task)

    def inputs(self, count: int = 3) -> list[str]:
        return [str(self.write(f"in/d{i}.txt", f"document {i}\n" * 20)) for i in range(count)]

    def test_new_modes_need_no_files_only_where_sensible(self) -> None:
        self.assertEqual(self.run_task(mode="jobs")["status"], "ok")
        self.assertEqual(self.run_task(mode="resume")["status"], "error")
        self.assertEqual(self.run_task(mode="detect")["status"], "error")

    def test_detect_validate_compress_decompress_modes(self) -> None:
        files = self.inputs(2)
        self.assertEqual(self.run_task(mode="detect", files=files)["payload"]["results"][0]["payload"]["format_id"], "txt")
        bad = str(self.write("bad.json", '{"a": '))
        validated = self.run_task(mode="validate", files=[files[0], bad], deep=True)
        self.assertEqual(validated["status"], "partial")
        self.assertEqual([item["payload"]["valid"] for item in validated["payload"]["results"]], [True, False])
        packed = self.run_task(mode="compress", files=files)
        archive = packed["payload"]["artifacts"][0]["output_path"]
        unpacked = self.run_task(mode="decompress", files=[archive], output_dir=str(self.tmp / "x"))
        self.assertEqual(sorted(f["path"] for f in unpacked["payload"]["results"][0]["payload"]["files"]), ["d0.txt", "d1.txt"])

    def test_salvage_flag_recovers_a_damaged_file_and_marks_the_run_partial(self) -> None:
        good = docx_bytes()
        damaged = str(self.write("in/cut.docx", good[: int(len(good) * 0.55)]))
        plain = self.run_task(files=[damaged], merge=False)
        self.assertEqual(plain["status"], "error")  # default behavior is unchanged
        recovered = self.run_task(files=[damaged], merge=False, salvage=True, include_documents=True)
        self.assertEqual(recovered["status"], "partial")
        self.assertIn("salvaged:partial", recovered["warnings"])
        self.assertEqual(recovered["payload"]["documents"][0]["source"], damaged)  # reported against the original file
        self.assertEqual(recovered["payload"]["salvage"][0]["level"], "partial")
        again = self.run_task(files=[damaged], merge=False)
        self.assertEqual(again["status"], "error")  # recovered content was not cached as a clean parse

    def test_same_stem_inputs_get_distinct_outputs(self) -> None:
        files = [str(self.write(f"in{i}/report.txt", f"unique-body-{i}\n")) for i in range(8)]
        result = self.run_task(files=files, merge=False, output_format="txt")
        paths = {artifact["output_path"] for artifact in result["payload"]["artifacts"]}
        self.assertEqual(len(paths), 8)
        self.assertEqual(len(list((self.tmp / "out").glob("*.txt"))), 8)

    def test_verify_and_detect_flags_annotate_results(self) -> None:
        result = self.run_task(files=self.inputs(1), merge=False, verify=True, detect=True, include_documents=True)
        self.assertEqual(result["status"], "ok", result.get("errors"))
        self.assertEqual(result["payload"]["artifacts"][0]["verification"]["integrity_status"], "ok")
        self.assertEqual(result["payload"]["documents"][0]["metadata"]["detection"]["format_id"], "txt")

    def test_durable_run_is_idempotent_and_resume_redoes_only_what_is_missing(self) -> None:
        files = self.inputs(3)
        first = self.run_task(files=files, merge=False, durable=True)
        self.assertEqual(first["payload"]["job"]["status"], "completed")
        again = self.run_task(files=files, merge=False, durable=True)
        self.assertTrue(again["metadata"]["reused_job"])
        self.assertEqual(len(again["payload"]["artifacts"]), 3)

        os.remove(first["payload"]["artifacts"][1]["output_path"])
        healed = self.run_task(files=files, merge=False, durable=True)
        self.assertEqual(healed["payload"]["job"]["reused_items"], 2)
        self.assertEqual(len(healed["payload"]["artifacts"]), 3)
        self.assertEqual(len(list((self.tmp / "out").glob("*.txt"))), 3)

    def test_partial_durable_job_resumes_after_the_input_appears(self) -> None:
        files = self.inputs(2)
        late = str(self.tmp / "in" / "late.txt")
        first = self.run_task(files=[*files, late], merge=False, durable=True)
        self.assertEqual((first["status"], first["payload"]["job"]["status"]), ("partial", "partial"))
        self.write("in/late.txt", "arrived\n")
        resumed = self.run_task(mode="resume", job_id=first["payload"]["job"]["job_id"])
        self.assertEqual(resumed["payload"]["job"]["status"], "completed")
        self.assertEqual((resumed["payload"]["job"]["reused_items"], len(resumed["payload"]["artifacts"])), (2, 3))

    def test_durable_merge_is_a_single_journaled_item(self) -> None:
        files = self.inputs(3)
        first = self.run_task(files=files, merge=True, durable=True, output_filename="all")
        self.assertEqual(first["payload"]["job"]["counts"]["total"], 1)
        again = self.run_task(files=files, merge=True, durable=True, output_filename="all")
        self.assertTrue(again["metadata"]["reused_job"])
        self.assertEqual(len(list((self.tmp / "out").glob("all*"))), 1)

    def test_capabilities_describe_every_engine_without_creating_the_job_store(self) -> None:
        from src.agents.agent_factory import AgentFactory
        from src.agents.collaborative.shared_memory import SharedMemory

        fresh = AgentFactory().create("reader", shared_memory=SharedMemory())
        capabilities = fresh.capabilities()
        for key in ("detection", "integrity", "compression", "salvage", "jobs"):
            self.assertIn(key, capabilities)
        for mode in ("detect", "validate", "compress", "decompress", "salvage", "jobs", "resume"):
            self.assertIn(mode, capabilities["modes"])
        self.assertIsNone(fresh._job_engine)  # describing the agent must not open the job store
        self.assertFalse((self.tmp / "tmp").exists())


if __name__ == "__main__":
    unittest.main()
