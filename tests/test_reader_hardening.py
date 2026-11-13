"""Regression tests for defects confirmed in the Reader subsystem audit (revision a854f17d).

Each test fails on the audited revision and passes with the accompanying fixes. They exercise the
real engines and the real ReaderAgent (no mocks); only temporary files are created.
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from src.agents.reader.conversion_engine import ConversionEngine
from src.agents.reader.parser_engine import ParserEngine
from src.agents.reader.recovery_engine import RecoveryEngine
from src.agents.reader.utils.reader_error import ParseFailureError

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
PLACEHOLDER = "[CORRUPTED_DATA]"


def _docx_bytes(body_xml: str) -> bytes:
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{W_NS}"><w:body>{body_xml}</w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


class _TempDirTestCase(unittest.TestCase):
    """Runs each test in its own working directory so Reader side effects never touch the repo."""

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
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path


class ParserEngineRegressionTests(_TempDirTestCase):
    def test_docx_preserves_tabs_and_line_breaks_but_not_tab_stops(self) -> None:
        body = (
            "<w:p><w:r><w:t>Alpha</w:t></w:r><w:r><w:tab/></w:r>"
            "<w:r><w:t>Beta</w:t><w:br/><w:t>Gamma</w:t></w:r></w:p>"
            '<w:p><w:pPr><w:tabs><w:tab w:val="left" w:pos="720"/></w:tabs></w:pPr>'
            "<w:r><w:t>Stop</w:t></w:r></w:p>"
        )
        result = ParserEngine().parse(self.write("fidelity.docx", _docx_bytes(body)))
        self.assertEqual(result["content"], "Alpha\tBeta\nGamma\nStop")

    def test_docx_with_oversized_uncompressed_xml_is_rejected(self) -> None:
        bomb = _docx_bytes("<w:p><w:r><w:t>x</w:t></w:r></w:p>" * 20000)  # ~700 KB expanded
        self.assertLess(len(bomb), 20000, "fixture must stay highly compressed")
        engine = ParserEngine()
        engine.docx_max_xml_bytes = 64 * 1024
        with self.assertRaises(ParseFailureError) as caught:
            engine.parse(self.write("bomb.docx", bomb))
        self.assertIn("docx_max_xml_bytes", str(caught.exception))

    def test_docx_within_limit_and_limit_disabled_still_parse(self) -> None:
        path = self.write("ok.docx", _docx_bytes("<w:p><w:r><w:t>fine</w:t></w:r></w:p>"))
        engine = ParserEngine()
        self.assertEqual(engine.parse(path)["content"], "fine")
        engine.docx_max_xml_bytes = 0
        self.assertEqual(engine.parse(path)["content"], "fine")

    def test_utf16_and_utf32_bom_text_decodes_correctly(self) -> None:
        engine = ParserEngine()
        for encoding in ("utf-16", "utf-32"):
            result = engine.parse(self.write(f"{encoding}.txt", "Wide text ✓\nline two\n".encode(encoding)))
            self.assertEqual(result["content"], "Wide text ✓\nline two\n", encoding)
            self.assertEqual(result["metadata"]["encoding"], encoding)
            self.assertNotIn("binary_signature_detected", result["warnings"], encoding)
            self.assertNotIn("null_bytes_detected", result["warnings"], encoding)

    def test_utf8_and_latin1_decoding_unchanged(self) -> None:
        engine = ParserEngine()
        self.assertEqual(engine.parse(self.write("u8.txt", "café ✓\n"))["content"], "café ✓\n")
        self.assertEqual(engine.parse(self.write("l1.txt", "café\n".encode("latin-1")))["content"], "café\n")

    def test_deeply_nested_json_does_not_leak_recursion_error(self) -> None:
        result = ParserEngine().parse(self.write("deep.json", "[" * 100000 + "]" * 100000))
        self.assertEqual(result["status"], "ok")
        self.assertIn("json_parse_warning", result["warnings"])

    def test_csv_dialect_fallback_does_not_mutate_stdlib_dialect(self) -> None:
        engine = ParserEngine()
        engine.csv_delimiter_fallback = ";"
        before = csv.excel.delimiter
        engine.parse(self.write("one_column.csv", "a\nb\nc\n"))
        self.assertEqual(csv.excel.delimiter, before)


class ConversionEngineRegressionTests(_TempDirTestCase):
    def _doc(self, content: str) -> dict:
        parsed = dict(ParserEngine().parse(self.write("src.txt", "seed\n")))
        parsed["content"] = content
        return parsed

    def test_xml_output_is_well_formed_when_content_has_control_characters(self) -> None:
        engine = ConversionEngine()
        doc = self._doc("a\x00b\x0bc\x1bd￾e")
        single = engine.convert(doc, "xml", str(self.tmp / "out"))
        root = ET.parse(single["output_path"]).getroot()
        self.assertEqual(root.findtext("content"), "abcde")
        merged = engine.merge([doc, self._doc("plain")], "xml", str(self.tmp / "out"), filename="merged")
        ET.parse(merged["output_path"])  # raises ParseError if not well-formed


class RecoveryEngineRegressionTests(_TempDirTestCase):
    def _doc(self, content: str) -> dict:
        parsed = dict(ParserEngine().parse(self.write("src.txt", "seed\n")))
        parsed["content"] = content
        return parsed

    def test_empty_documents_are_not_replaced_by_corruption_placeholder(self) -> None:
        for content in ("", "  \n\t "):
            result = RecoveryEngine().recover_document(self._doc(content))
            self.assertNotIn(PLACEHOLDER, result["content"], repr(content))
            self.assertNotEqual(result["status"], "unrecoverable", repr(content))

    def test_genuine_low_level_corruption_is_still_repaired(self) -> None:
        result = RecoveryEngine().recover_document(self._doc("line1\r\nline2\x00\x00\r\nline3"))
        self.assertNotIn("\x00", result["content"])
        self.assertIn("line2", result["content"])


class ReaderAgentRegressionTests(_TempDirTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._boot = tempfile.TemporaryDirectory()
        cls._cwd = os.getcwd()
        os.chdir(cls._boot.name)
        from src.agents.agent_factory import AgentFactory
        from src.agents.collaborative.shared_memory import SharedMemory

        cls.agent = AgentFactory().create("reader", shared_memory=SharedMemory())

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls._cwd)
        cls._boot.cleanup()

    def run_task(self, **task):
        task.setdefault("output_dir", str(self.tmp / "out"))
        return self.agent.perform_task(task)

    def test_empty_stage_batch_constructs(self) -> None:
        from src.agents.reader_agent import ReaderStageBatch

        batch = ReaderStageBatch.empty("recover")
        self.assertEqual((batch.stage, batch.failed_count, batch.success_count), ("recover", 0, 0))

    def test_parse_task_completes(self) -> None:
        result = self.run_task(operation="parse", files=[str(self.write("a.txt", "hello\n"))])
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertEqual(result["metadata"]["documents_processed"], 1)

    def test_pipeline_converts_or_merges_multiple_files(self) -> None:
        files = [str(self.write("a.txt", "one\n")), str(self.write("b.csv", "x,y\n1,2\n"))]
        separate = self.run_task(files=files, output_format="md", merge=False)
        self.assertEqual(separate["status"], "ok", separate.get("errors"))
        self.assertEqual(len(separate["payload"]["artifacts"]), 2)
        merged = self.run_task(files=files, output_format="md", merge=True, output_filename="both")
        self.assertEqual(merged["status"], "ok", merged.get("errors"))
        self.assertEqual(len(merged["payload"]["artifacts"]), 1)

    def test_empty_file_is_not_converted_to_corruption_placeholder(self) -> None:
        result = self.run_task(files=[str(self.write("empty.txt", b""))], output_format="txt")
        artifacts = result["payload"]["artifacts"]
        self.assertEqual(len(artifacts), 1)
        self.assertNotIn(PLACEHOLDER, Path(artifacts[0]["output_path"]).read_text(encoding="utf-8"))

    def test_conversion_without_any_artifact_reports_error_not_partial(self) -> None:
        blocker = self.write("blocker", b"i am a file, not a directory")
        result = self.run_task(
            files=[str(self.write("a.txt", "hello\n"))],
            output_dir=str(blocker / "sub"),
            output_format="txt",
        )
        self.assertEqual(result["payload"]["artifacts"], [])
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
