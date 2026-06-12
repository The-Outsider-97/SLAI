"""Privacy-safe document extraction for DocMaster.

The extractor reads uploaded/local files into memory, validates type and size, and
does not persist private document content. It is shared by the PyQt GUI and the
optional Flask API routes.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".xml", ".odt"}
DEFAULT_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TEXT_CHARS = 120_000


class DocumentExtractionError(ValueError):
    """Raised when a document cannot be safely read or validated."""


@dataclass(frozen=True)
class ExtractedDocument:
    filename: str
    detected_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class DocumentExtractor:
    """Extract readable text from supported document formats."""

    def __init__(
        self,
        *,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    ) -> None:
        self.max_file_size_bytes = int(max_file_size_bytes)
        self.max_text_chars = int(max_text_chars)

    def extract_upload(self, file: FileStorage) -> ExtractedDocument:
        filename = secure_filename(file.filename or "uploaded_document")
        raw = file.read()
        return self.extract_bytes(raw, filename=filename)

    def extract_path(self, path: str | Path) -> ExtractedDocument:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            raise DocumentExtractionError("Selected file does not exist or is not a file.")
        raw = file_path.read_bytes()
        return self.extract_bytes(raw, filename=file_path.name)

    def extract_bytes(self, raw: bytes, *, filename: str) -> ExtractedDocument:
        safe_name = secure_filename(filename or "document")
        ext = Path(safe_name).suffix.lower()
        warnings: List[str] = []

        if ext not in SUPPORTED_EXTENSIONS:
            raise DocumentExtractionError(
                f"Unsupported file type '{ext or 'unknown'}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )
        if len(raw) > self.max_file_size_bytes:
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            raise DocumentExtractionError(f"File is too large. Maximum upload size is {limit_mb:.0f} MB.")
        if not raw:
            raise DocumentExtractionError("The uploaded file is empty.")

        stream = io.BytesIO(raw)
        try:
            if ext == ".pdf":
                text, page_count, pdf_warnings = self._extract_pdf(stream)
                warnings.extend(pdf_warnings)
                metadata = {"page_count": page_count}
            elif ext == ".docx":
                text = self._extract_docx(stream)
                metadata = {"paragraph_count": text.count("\n") + 1 if text else 0}
            elif ext in {".html", ".htm", ".xml"}:
                text = self._extract_markup(raw)
                metadata = {}
            elif ext == ".odt":
                text = self._extract_odt(raw)
                metadata = {}
            else:
                text = self._decode_text(raw)
                metadata = {}
        except DocumentExtractionError:
            raise
        except PdfReadError as exc:
            raise DocumentExtractionError("PDF could not be read. It may be encrypted, corrupt, or image-only.") from exc
        except Exception as exc:  # noqa: BLE001
            raise DocumentExtractionError(f"Document could not be processed: {type(exc).__name__}.") from exc

        text = self._clean_text(text)
        if not text:
            raise DocumentExtractionError("No readable text was found. The file may be scanned, encrypted, or unsupported.")
        if len(text) > self.max_text_chars:
            text = text[: self.max_text_chars]
            warnings.append(f"Text was truncated to {self.max_text_chars:,} characters for safe processing.")

        metadata.update(
            {
                "word_count": len(text.split()),
                "char_count": len(text),
                "detected_type": ext.lstrip("."),
                "language": self._guess_language(text),
            }
        )
        return ExtractedDocument(safe_name, ext.lstrip("."), text, metadata, warnings)

    @staticmethod
    def _extract_pdf(stream: BinaryIO) -> tuple[str, int, List[str]]:
        reader = PdfReader(stream)
        if getattr(reader, "is_encrypted", False):
            raise DocumentExtractionError("Encrypted PDFs are not processed by default.")
        chunks: List[str] = []
        warnings: List[str] = []
        for index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                page_text = ""
                warnings.append(f"Page {index + 1} could not be extracted.")
            if page_text.strip():
                chunks.append(page_text)
        return "\n".join(chunks), len(reader.pages), warnings

    @staticmethod
    def _extract_docx(stream: BinaryIO) -> str:
        doc = Document(stream)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        table_cells: List[str] = []
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        table_cells.append(cell.text.strip())
        return "\n".join(paragraphs + table_cells)

    @staticmethod
    def _extract_odt(raw: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                content = archive.read("content.xml")
        except Exception as exc:  # noqa: BLE001
            raise DocumentExtractionError("ODT file could not be opened.") from exc
        soup = BeautifulSoup(content, "xml")
        return soup.get_text(" ")

    @staticmethod
    def _extract_markup(raw: bytes) -> str:
        decoded = DocumentExtractor._decode_text(raw)
        soup = BeautifulSoup(decoded, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(" ")

    @staticmethod
    def _decode_text(raw: bytes) -> str:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _guess_language(text: str) -> str:
        sample = text[:3000].lower()
        dutch_markers = {" de ", " het ", " een ", " voor ", " zijn ", " worden ", " niet "}
        english_markers = {" the ", " and ", " for ", " with ", " this ", " that ", " not "}
        nl = sum(1 for m in dutch_markers if m in f" {sample} ")
        en = sum(1 for m in english_markers if m in f" {sample} ")
        if nl > en:
            return "nl"
        if en > nl:
            return "en"
        return "unknown"
