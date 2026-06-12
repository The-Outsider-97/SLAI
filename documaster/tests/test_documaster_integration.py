from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("werkzeug")
pytest.importorskip("bs4")
pytest.importorskip("docx")
pytest.importorskip("pypdf")

from documaster.services.document_ai_service import DocumentAIService
from documaster.services.document_extractor import DocumentExtractionError, DocumentExtractor
from documaster.slai_adapter import SLAIAdapter
from documaster.utils.documaster_utils import DocMasterFileService


class NoSLAIAdapter(SLAIAdapter):
    def initialize_runtime(self):
        self.state.available = False
        self.state.status = "test_unavailable"
        self.state.error = "SLAI disabled for test"
        return self.health()

    def read_path(self, *args, **kwargs):
        return None


def test_unsupported_file_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.exe"
    path.write_bytes(b"not a document")
    extractor = DocumentExtractor()
    with pytest.raises(DocumentExtractionError):
        extractor.extract_path(path)


def test_document_ai_service_returns_stable_error_for_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    service = DocumentAIService(adapter=NoSLAIAdapter())
    payload = service.run_path(task="summarize", path=path)
    assert payload["status"] == "error"
    assert payload["task"] == "summarize"
    assert payload["privacy"]["stored"] is False
    assert "issues" in payload["result"]


def test_document_ai_service_uses_local_fallback_when_reader_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("This is a compact test document. It explains DocuMaster integration clearly.", encoding="utf-8")
    service = DocumentAIService(adapter=NoSLAIAdapter())
    payload = service.run_path(task="summarize", path=path)
    assert payload["status"] in {"success", "partial"}
    assert payload["metadata"]["reader_agent_used"] is False
    assert payload["privacy"]["stored"] is False


def test_pdf_merge_plan_controls_page_selection(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    writer1 = pypdf.PdfWriter()
    writer1.add_blank_page(width=72, height=72)
    writer1.add_blank_page(width=72, height=72)
    pdf1 = tmp_path / "one.pdf"
    with pdf1.open("wb") as handle:
        writer1.write(handle)

    writer2 = pypdf.PdfWriter()
    writer2.add_blank_page(width=72, height=72)
    pdf2 = tmp_path / "two.pdf"
    with pdf2.open("wb") as handle:
        writer2.write(handle)

    service = DocMasterFileService()
    plan = service.build_pdf_merge_plan([pdf1, pdf2]).to_dict()
    assert len(plan["pages"]) == 3
    plan["pages"][1]["included"] = False
    plan["pages"][2]["user_order_index"] = 0
    plan["pages"][0]["user_order_index"] = 1
    output = tmp_path / "merged.pdf"
    service.merge_pdfs_with_plan(plan, output)
    reader = pypdf.PdfReader(str(output))
    assert len(reader.pages) == 2
