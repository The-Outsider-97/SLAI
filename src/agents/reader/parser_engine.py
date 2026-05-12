from __future__ import annotations

"""Production-grade Parser Engine for the Reader subsystem.

The parser engine is responsible for turning supported files into deterministic
Reader document dictionaries. It intentionally delegates shared concerns to
``reader_helpers.py``: path validation, hashing, decoding, text quality,
redaction/JSON safety, chunk estimation, and common result helpers.

Design goals
------------
- Keep the public ``parse(file_path)`` contract compatible with the existing
  ReaderAgent pipeline.
- Avoid duplicated helper logic already centralized in ``reader_helpers.py``.
- Keep config loading simple: load once in ``__init__`` through the existing
  Reader config loader.
- Use typed Reader errors for all parser-domain failures.
- Prefer deterministic, conservative extraction. Do not fabricate content.
"""

import csv
import importlib
import io
import json
import re
import zipfile

from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from xml.etree import ElementTree as ET

from .utils.config_loader import get_config_section, load_reader_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from .reader_memory import ReaderMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Parser Engine")
printer = PrettyPrinter()


class _HTMLTextExtractor(HTMLParser):
    """Small deterministic HTML text extractor used by ParserEngine.

    This is intentionally narrow and dependency-free. It is not a browser, not a
    sanitizer, and not a renderer. The parser output remains a text
    representation for downstream Reader recovery/conversion.
    """

    _BLOCK_TAGS = {"address", "article", "aside", "blockquote", "br", "div", "dl", "fieldset",
                   "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
                   "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table",
                   "td", "th", "tr", "ul",}
    _SKIP_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        lowered = tag.lower()
        if lowered in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if lowered in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if lowered in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        lines: List[str] = []
        for line in "".join(self._parts).splitlines():
            normalized = re.sub(r"[ \t]+", " ", line).strip()
            if normalized:
                lines.append(normalized)
        return "\n".join(lines).strip()


class ParserEngine:
    """Normalize supported file inputs into deterministic Reader documents."""

    _TEXT_EXTENSIONS = {".txt", ".md", ".html", ".xml", ".json", ".csv"}
    _STRUCTURED_TEXT_EXTENSIONS = {".html", ".xml", ".json", ".csv"}
    _DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc"}
    _PARSER_VERSION = "2.2.0"

    def __init__(self, memory: Optional[ReaderMemory] = None) -> None:
        self.config = load_reader_config()
        self.reader_config = get_config_section("reader")
        self.parser_config = get_config_section("parser_engine")

        allowed = self.reader_config.get("allowed_input_extensions") or sorted(self._TEXT_EXTENSIONS)
        self.allowed_extensions = normalize_extension_set(allowed)
        self.max_file_size_bytes = coerce_int(
            self.reader_config.get("max_file_size_bytes", 50 * 1024 * 1024),
            50 * 1024 * 1024,
            minimum=0,
        )
        self.encoding_candidates: List[str] = [
            str(item) for item in self.reader_config.get("encoding_candidates", ["utf-8", "utf-8-sig", "latin-1"])
        ]

        self.text_extensions = normalize_extension_set(
            self.parser_config.get("text_extensions", sorted(self._TEXT_EXTENSIONS))
        )
        self.structured_text_extensions = normalize_extension_set(
            self.parser_config.get("structured_text_extensions", sorted(self._STRUCTURED_TEXT_EXTENSIONS))
        )
        self.document_extensions = normalize_extension_set(
            self.parser_config.get("document_extensions", sorted(self._DOCUMENT_EXTENSIONS))
        )

        self.allow_lossy_decode_fallback = coerce_bool(
            self.parser_config.get("allow_lossy_decode_fallback", True),
            True,
        )
        self.normalize_newlines_enabled = coerce_bool(self.parser_config.get("normalize_newlines", True), True)
        self.strip_bom = coerce_bool(self.parser_config.get("strip_bom", True), True)
        self.preserve_raw_content_for_text = coerce_bool(
            self.parser_config.get("preserve_raw_content_for_text", True),
            True,
        )
        self.fail_on_empty_content = coerce_bool(self.parser_config.get("fail_on_empty_content", False), False)
        self.max_parsed_chars = coerce_int(self.parser_config.get("max_parsed_chars", 0), 0, minimum=0)
        self.binary_signature_sample_size = coerce_int(
            self.parser_config.get("binary_signature_sample_size", 4096),
            4096,
            minimum=128,
        )

        self.extract_html_text = coerce_bool(self.parser_config.get("extract_html_text", True), True)
        self.extract_xml_text = coerce_bool(self.parser_config.get("extract_xml_text", True), True)
        self.json_pretty_print = coerce_bool(self.parser_config.get("json_pretty_print", True), True)
        self.json_indent = self.parser_config.get("json_indent", 2)
        self.csv_max_rows = coerce_int(self.parser_config.get("csv_max_rows", 50_000), 50_000, minimum=1)
        self.csv_max_columns = coerce_int(self.parser_config.get("csv_max_columns", 1_000), 1_000, minimum=1)
        self.csv_sniff_dialect = coerce_bool(self.parser_config.get("csv_sniff_dialect", True), True)
        self.csv_delimiter_fallback = str(self.parser_config.get("csv_delimiter_fallback", ",") or ",")

        self.enable_pdf = coerce_bool(self.parser_config.get("enable_pdf", True), True)
        self.pdf_backend = str(self.parser_config.get("pdf_backend", "pypdf") or "pypdf").strip().lower()
        self.pdf_max_pages = coerce_int(self.parser_config.get("pdf_max_pages", 0), 0, minimum=0)
        self.pdf_extract_page_breaks = coerce_bool(self.parser_config.get("pdf_extract_page_breaks", True), True)
        self.pdf_page_break_template = str(
            self.parser_config.get("pdf_page_break_template", "\n\n--- Page {page_number} ---\n\n")
        )

        self.enable_docx = coerce_bool(self.parser_config.get("enable_docx", True), True)
        self.enable_doc = coerce_bool(self.parser_config.get("enable_doc", False), False)
        self.legacy_doc_requires_external_backend = coerce_bool(
            self.parser_config.get("legacy_doc_requires_external_backend", True),
            True,
        )

        self.include_quality_report = coerce_bool(self.parser_config.get("include_quality_report", True), True)
        self.include_processing_cost = coerce_bool(self.parser_config.get("include_processing_cost", True), True)
        self.include_content_hash = coerce_bool(self.parser_config.get("include_content_hash", True), True)
        self.include_path_info = coerce_bool(self.parser_config.get("include_path_info", True), True)
        self.include_debug_preview = coerce_bool(self.parser_config.get("include_debug_preview", False), False)
        self.debug_preview_chars = coerce_int(self.parser_config.get("debug_preview_chars", 240), 240, minimum=0)

        self.memory = memory

        self._dispatch = {
            ".txt": self._parse_plain_text,
            ".md": self._parse_plain_text,
            ".html": self._parse_html,
            ".xml": self._parse_xml,
            ".json": self._parse_json,
            ".csv": self._parse_csv,
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".doc": self._parse_doc,
        }

        logger.info("Parser Engine initialized with %s allowed extensions", len(self.allowed_extensions))

    # ------------------------------------------------------------------
    # Validation and shared parse result construction
    # ------------------------------------------------------------------

    def _validate_input(self, file_path: str | Path) -> ReaderPathInfo:
        return validate_input_path(
            file_path,
            allowed_extensions=self.allowed_extensions,
            max_file_size_bytes=self.max_file_size_bytes,
            compute_hash=False,
        )

    def _read_bytes(self, path_info: ReaderPathInfo) -> bytes:
        try:
            return path_info.path.read_bytes()
        except PermissionError as exc:
            raise FileAccessDeniedError(str(path_info.path), operation="read", cause=exc) from exc
        except OSError as exc:
            raise ParseFailureError(str(path_info.path), f"Failed to read file: {exc}", cause=exc) from exc

    def _decode(self, raw_bytes: bytes, *, source: str) -> DecodedText:
        return decode_bytes(
            raw_bytes,
            source=source,
            encoding_candidates=self.encoding_candidates,
            allow_lossy_fallback=self.allow_lossy_decode_fallback,
        )

    def _normalize_content(self, content: str) -> str:
        output = str(content or "")
        if self.strip_bom:
            output = output.lstrip("\ufeff")
        if self.normalize_newlines_enabled:
            output = normalize_newlines(output)
        if self.max_parsed_chars and len(output) > self.max_parsed_chars:
            output = truncate_text(output, self.max_parsed_chars)
        return output

    def _detect_binary_signature(self, raw_bytes: bytes) -> bool:
        return is_probably_binary(raw_bytes, sample_size=self.binary_signature_sample_size)

    def _build_result(
        self,
        *,
        path_info: ReaderPathInfo,
        raw_bytes: bytes,
        content: str,
        parser_name: str,
        content_type: str,
        encoding: Optional[str] = None,
        had_decode_fallback: bool = False,
        warnings: Optional[Iterable[str]] = None,
        parser_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        content = self._normalize_content(content)
        quality = text_quality_report(content)
        binary_signature = self._detect_binary_signature(raw_bytes)

        warning_list = dedupe_preserve_order(
            list(warnings or [])
            + list(quality.warnings)
            + (["lossy_decode_fallback_used"] if had_decode_fallback else [])
            + (["binary_signature_detected"] if binary_signature and path_info.extension in self.text_extensions else [])
        )

        if self.fail_on_empty_content and not content.strip():
            raise EmptyContentError(str(path_info.path))

        content_sha256 = sha256_text(content) if self.include_content_hash else None
        raw_sha256 = sha256_bytes(raw_bytes)
        metadata: Dict[str, Any] = {
            "size": len(raw_bytes),
            "mtime": path_info.mtime,
            "encoding": encoding,
            "sha256": raw_sha256,
            "raw_sha256": raw_sha256,
            "content_sha256": content_sha256,
            "line_count": quality.line_count,
            "char_count": len(content),
            "byte_count": len(raw_bytes),
            "is_binary_signature": binary_signature,
            "parser": parser_name,
            "parser_engine_version": self._PARSER_VERSION,
            "content_type": content_type,
            "parsed_at": utc_now().isoformat(),
        }
        if self.include_path_info:
            metadata["path"] = path_info.to_dict()
        if parser_metadata:
            metadata["parser_metadata"] = json_safe(parser_metadata)
        if self.include_quality_report:
            metadata["quality"] = quality.to_dict()
        if self.include_processing_cost:
            metadata["processing_cost"] = {
                "char_count": len(content),
                "line_count": quality.line_count,
                "quality_score": quality.quality_score,
                "estimated_recovery_chunks": max(1, len(list(iter_text_chunks(content, 20_000)))) if content else 0,
                "requires_attention": quality.quality_score < 0.55 or bool(warning_list),
            }
        if self.include_debug_preview and self.debug_preview_chars:
            metadata["debug_preview"] = truncate_text(content, self.debug_preview_chars)

        return {
            "status": "ok",
            "source": str(path_info.path),
            "resolved_source": str(path_info.resolved_path),
            "filename": path_info.name,
            "stem": path_info.stem,
            "extension": path_info.extension,
            "content": content,
            "content_type": content_type,
            "parser": parser_name,
            "metadata": compact_none(metadata),
            "warnings": warning_list,
        }

    # ------------------------------------------------------------------
    # Text and structured text parsers
    # ------------------------------------------------------------------

    def _parse_plain_text(self, path_info: ReaderPathInfo, raw_bytes: bytes) -> Dict[str, Any]:
        decoded = self._decode(raw_bytes, source=str(path_info.path))
        return self._build_result(
            path_info=path_info,
            raw_bytes=raw_bytes,
            content=decoded.text,
            parser_name="plain_text",
            content_type="text/plain",
            encoding=decoded.encoding,
            had_decode_fallback=decoded.had_decode_fallback,
            warnings=decoded.warnings,
        )

    def _parse_html(self, path_info: ReaderPathInfo, raw_bytes: bytes) -> Dict[str, Any]:
        decoded = self._decode(raw_bytes, source=str(path_info.path))
        content = decoded.text
        parser_metadata: Dict[str, Any] = {"extracted_text": False}
        warnings = list(decoded.warnings)
        if self.extract_html_text:
            extractor = _HTMLTextExtractor()
            try:
                extractor.feed(decoded.text)
                extractor.close()
                extracted = extractor.get_text()
                if extracted:
                    content = extracted
                    parser_metadata["extracted_text"] = True
                else:
                    warnings.append("html_text_extraction_empty")
            except Exception as exc:
                warnings.append("html_text_extraction_failed")
                parser_metadata["html_extraction_error"] = type(exc).__name__
        return self._build_result(
            path_info=path_info,
            raw_bytes=raw_bytes,
            content=content,
            parser_name="html_text" if parser_metadata["extracted_text"] else "html_raw",
            content_type="text/html",
            encoding=decoded.encoding,
            had_decode_fallback=decoded.had_decode_fallback,
            warnings=warnings,
            parser_metadata=parser_metadata,
        )

    def _parse_xml(self, path_info: ReaderPathInfo, raw_bytes: bytes) -> Dict[str, Any]:
        decoded = self._decode(raw_bytes, source=str(path_info.path))
        content = decoded.text
        parser_metadata: Dict[str, Any] = {"extracted_text": False, "well_formed": False}
        warnings = list(decoded.warnings)
        if self.extract_xml_text:
            try:
                root = ET.fromstring(decoded.text)
                texts = [text.strip() for text in root.itertext() if text and text.strip()]
                if texts:
                    content = "\n".join(texts)
                    parser_metadata["extracted_text"] = True
                else:
                    warnings.append("xml_text_extraction_empty")
                parser_metadata["well_formed"] = True
                parser_metadata["root_tag"] = root.tag
            except ET.ParseError as exc:
                warnings.append("xml_parse_warning")
                parser_metadata["xml_parse_error"] = str(exc)
        return self._build_result(
            path_info=path_info,
            raw_bytes=raw_bytes,
            content=content,
            parser_name="xml_text" if parser_metadata["extracted_text"] else "xml_raw",
            content_type="application/xml",
            encoding=decoded.encoding,
            had_decode_fallback=decoded.had_decode_fallback,
            warnings=warnings,
            parser_metadata=parser_metadata,
        )

    def _parse_json(self, path_info: ReaderPathInfo, raw_bytes: bytes) -> Dict[str, Any]:
        decoded = self._decode(raw_bytes, source=str(path_info.path))
        content = decoded.text
        parser_metadata: Dict[str, Any] = {"valid_json": False}
        warnings = list(decoded.warnings)
        try:
            payload = json.loads(decoded.text)
            parser_metadata["valid_json"] = True
            parser_metadata["root_type"] = type(payload).__name__
            if isinstance(payload, Mapping):
                parser_metadata["top_level_keys"] = list(payload.keys())[:100]
            elif isinstance(payload, list):
                parser_metadata["top_level_items"] = len(payload)
            if self.json_pretty_print:
                content = json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True, indent=self.json_indent)
        except json.JSONDecodeError as exc:
            warnings.append("json_parse_warning")
            parser_metadata["json_parse_error"] = str(exc)
        return self._build_result(
            path_info=path_info,
            raw_bytes=raw_bytes,
            content=content,
            parser_name="json",
            content_type="application/json",
            encoding=decoded.encoding,
            had_decode_fallback=decoded.had_decode_fallback,
            warnings=warnings,
            parser_metadata=parser_metadata,
        )

    def _parse_csv(self, path_info: ReaderPathInfo, raw_bytes: bytes) -> Dict[str, Any]:
        decoded = self._decode(raw_bytes, source=str(path_info.path))
        warnings = list(decoded.warnings)
        parser_metadata: Dict[str, Any] = {"row_count": 0, "column_count": 0, "truncated_rows": False}
        text = decoded.text
        sample = text[:8192]
        delimiter = self.csv_delimiter_fallback
        try:
            dialect = csv.Sniffer().sniff(sample) if self.csv_sniff_dialect and sample.strip() else csv.excel
            delimiter = getattr(dialect, "delimiter", delimiter) or delimiter
        except csv.Error:
            dialect = csv.excel
            dialect.delimiter = delimiter  # type: ignore[attr-defined]
            warnings.append("csv_dialect_sniff_failed")

        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
        try:
            reader = csv.reader(io.StringIO(text), dialect)
            for row_index, row in enumerate(reader):
                if row_index >= self.csv_max_rows:
                    parser_metadata["truncated_rows"] = True
                    warnings.append("csv_row_limit_reached")
                    break
                trimmed = row[: self.csv_max_columns]
                if len(row) > self.csv_max_columns:
                    warnings.append("csv_column_limit_reached")
                parser_metadata["column_count"] = max(parser_metadata["column_count"], len(trimmed))
                writer.writerow(trimmed)
                parser_metadata["row_count"] += 1
        except csv.Error as exc:
            warnings.append("csv_parse_warning")
            parser_metadata["csv_parse_error"] = str(exc)
            output = io.StringIO()
            output.write(text)

        parser_metadata["delimiter"] = delimiter
        return self._build_result(
            path_info=path_info,
            raw_bytes=raw_bytes,
            content=output.getvalue(),
            parser_name="csv",
            content_type="text/csv",
            encoding=decoded.encoding,
            had_decode_fallback=decoded.had_decode_fallback,
            warnings=warnings,
            parser_metadata=parser_metadata,
        )

    # ------------------------------------------------------------------
    # Document parsers
    # ------------------------------------------------------------------

    def _parse_pdf(self, path_info: ReaderPathInfo, raw_bytes: bytes) -> Dict[str, Any]:
        if not self.enable_pdf:
            raise ParseFailureError(str(path_info.path), "PDF parsing is disabled in reader_config.yaml")
        if self.pdf_backend != "pypdf":
            raise ParseFailureError(
                str(path_info.path),
                f"Unsupported configured PDF backend: {self.pdf_backend}",
                context={"configured_backend": self.pdf_backend, "supported_backends": ["pypdf"]},
            )

        try:
            pypdf = importlib.import_module("pypdf")
        except ImportError as exc:
            raise ParseFailureError(
                str(path_info.path),
                "PDF parsing requires the optional 'pypdf' package",
                cause=exc,
                context={"backend": "pypdf"},
            ) from exc

        try:
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            if getattr(reader, "is_encrypted", False):
                raise ParseFailureError(str(path_info.path), "Encrypted PDF files are not supported by ParserEngine")
            page_count = len(reader.pages)
            max_pages = self.pdf_max_pages or page_count
            parts: List[str] = []
            extracted_pages = 0
            empty_pages = 0
            for page_number, page in enumerate(reader.pages[:max_pages], start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    empty_pages += 1
                if self.pdf_extract_page_breaks:
                    parts.append(self.pdf_page_break_template.format(page_number=page_number))
                parts.append(text)
                extracted_pages += 1
            warnings: List[str] = []
            if empty_pages:
                warnings.append("pdf_empty_pages_detected")
            if extracted_pages < page_count:
                warnings.append("pdf_page_limit_reached")
            parser_metadata = {
                "backend": "pypdf",
                "page_count": page_count,
                "pages_extracted": extracted_pages,
                "empty_pages": empty_pages,
                "truncated_pages": extracted_pages < page_count,
            }
            return self._build_result(
                path_info=path_info,
                raw_bytes=raw_bytes,
                content="\n".join(parts).strip(),
                parser_name="pdf_pypdf",
                content_type="application/pdf",
                encoding=None,
                warnings=warnings,
                parser_metadata=parser_metadata,
            )
        except ReaderError:
            raise
        except Exception as exc:
            raise ParseFailureError(str(path_info.path), f"Failed parsing PDF: {exc}", cause=exc) from exc

    def _parse_docx(self, path_info: ReaderPathInfo, raw_bytes: bytes) -> Dict[str, Any]:
        if not self.enable_docx:
            raise ParseFailureError(str(path_info.path), "DOCX parsing is disabled in reader_config.yaml")
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise ParseFailureError(str(path_info.path), "DOCX archive is missing word/document.xml")
                document_xml = archive.read("word/document.xml")
        except ReaderError:
            raise
        except zipfile.BadZipFile as exc:
            raise ParseFailureError(str(path_info.path), "DOCX file is not a valid ZIP-based document", cause=exc) from exc
        except OSError as exc:
            raise ParseFailureError(str(path_info.path), f"Failed reading DOCX archive: {exc}", cause=exc) from exc

        try:
            root = ET.fromstring(document_xml)
            namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs: List[str] = []
            for paragraph in root.findall(".//w:p", namespace):
                texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
                line = "".join(texts).strip()
                if line:
                    paragraphs.append(line)
            parser_metadata = {
                "paragraph_count": len(paragraphs),
                "source_xml_bytes": len(document_xml),
            }
            warnings = [] if paragraphs else ["docx_no_text_extracted"]
            return self._build_result(
                path_info=path_info,
                raw_bytes=raw_bytes,
                content="\n".join(paragraphs),
                parser_name="docx_xml",
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                encoding=None,
                warnings=warnings,
                parser_metadata=parser_metadata,
            )
        except ET.ParseError as exc:
            raise ParseFailureError(str(path_info.path), f"Failed parsing DOCX XML: {exc}", cause=exc) from exc

    def _parse_doc(self, path_info: ReaderPathInfo, raw_bytes: bytes) -> Dict[str, Any]:
        if not self.enable_doc:
            raise ParseFailureError(
                str(path_info.path),
                "Legacy .doc parsing is disabled. Convert the file to .docx, .pdf, or .txt before parsing.",
                context={"extension": ".doc", "configured_enable_doc": self.enable_doc},
            )
        if self.legacy_doc_requires_external_backend:
            raise ParseFailureError(
                str(path_info.path),
                "Legacy .doc parsing requires an external conversion backend that is not configured.",
                context={"extension": ".doc"},
            )
        decoded = self._decode(raw_bytes, source=str(path_info.path))
        warnings = list(decoded.warnings) + ["legacy_doc_lossy_text_decode"]
        return self._build_result(
            path_info=path_info,
            raw_bytes=raw_bytes,
            content=decoded.text,
            parser_name="doc_lossy_text",
            content_type="application/msword",
            encoding=decoded.encoding,
            had_decode_fallback=decoded.had_decode_fallback,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _config_snapshot(self) -> Dict[str, Any]:
        return {
            "allowed_extensions": sorted(self.allowed_extensions),
            "max_file_size_bytes": self.max_file_size_bytes,
            "encoding_candidates": self.encoding_candidates,
            "extract_html_text": self.extract_html_text,
            "extract_xml_text": self.extract_xml_text,
            "json_pretty_print": self.json_pretty_print,
            "pdf_enabled": self.enable_pdf,
            "pdf_backend": self.pdf_backend,
            "docx_enabled": self.enable_docx,
        }

    def parse(self, file_path: str | Path) -> Dict[str, Any]:
        """Parse one Reader input file into a deterministic document payload."""
    
        timer = OperationTimer("parse")
        path_info = self._validate_input(file_path)
        raw_bytes = self._read_bytes(path_info)
    
        cache_key_payload = None   # <-- define at the top
    
        # ---- Persistent cache lookup ----
        if self.memory is not None:
            cache_key_payload = {
                "action": "parse",
                "file": str(path_info.path),
                "size": path_info.size_bytes,
                "mtime": path_info.mtime,
                "parser_version": self._PARSER_VERSION,
                "config_snapshot": self._config_snapshot()
            }
            cached = self.memory.get_cache(cache_key_payload, namespace="parser")
            if cached is not None:
                cached_result = dict(cached)
                cached_result.setdefault("metadata", {})["cached"] = True
                return cached_result
    
        extension = path_info.extension
        parser = self._dispatch.get(extension)
        if parser is None:
            raise UnsupportedFormatError(str(path_info.path), extension, self.allowed_extensions)
    
        result = parser(path_info, raw_bytes)
        result["metadata"]["timing"] = timer.stop().to_dict()
    
        # ---- Persistent cache store ----
        if self.memory is not None and cache_key_payload is not None and result.get("status") == "ok":
            result_for_cache = dict(result)
            result_for_cache.pop("timing", None)
            self.memory.set_cache(cache_key_payload, result_for_cache, namespace="parser", ttl_seconds=None)
    
        return result

    def parse_many(self, file_paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
        """Parse multiple files sequentially and fail fast on Reader errors."""

        return [self.parse(file_path) for file_path in file_paths]

    def capabilities(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot of parser capabilities and settings."""

        return {
            "parser_version": self._PARSER_VERSION,
            "allowed_extensions": sorted(self.allowed_extensions),
            "text_extensions": sorted(self.text_extensions),
            "structured_text_extensions": sorted(self.structured_text_extensions),
            "document_extensions": sorted(self.document_extensions),
            "max_file_size_bytes": self.max_file_size_bytes,
            "encoding_candidates": list(self.encoding_candidates),
            "pdf_enabled": self.enable_pdf,
            "pdf_backend": self.pdf_backend,
            "docx_enabled": self.enable_docx,
            "doc_enabled": self.enable_doc,
        }


if __name__ == "__main__":
    print("\n=== Running Parser Engine ===\n")
    printer.status("TEST", "Parser Engine initialized", "info")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="reader_parser_test_") as tmp_dir:
        tmp = Path(tmp_dir)
        text_file = tmp / "sample.txt"
        md_file = tmp / "sample.md"
        json_file = tmp / "sample.json"
        csv_file = tmp / "sample.csv"
        html_file = tmp / "sample.html"
        xml_file = tmp / "sample.xml"

        text_file.write_text("Hello Reader\nThis is a parser test.\n", encoding="utf-8")
        md_file.write_text("# Title\n\nMarkdown body.\n", encoding="utf-8")
        json_file.write_text(json.dumps({"name": "reader", "items": [1, 2, 3]}), encoding="utf-8")
        csv_file.write_text("name,value\nalpha,1\nbeta,2\n", encoding="utf-8")
        html_file.write_text("<html><body><h1>Hello</h1><p>HTML body.</p></body></html>", encoding="utf-8")
        xml_file.write_text("<root><title>Hello</title><body>XML body.</body></root>", encoding="utf-8")

        engine = ParserEngine()
        printer.status("TEST", "Capabilities loaded", "info")
        assert ".txt" in engine.capabilities()["allowed_extensions"]

        parsed_text = engine.parse(text_file)
        assert parsed_text["status"] == "ok"
        assert parsed_text["extension"] == ".txt"
        assert parsed_text["metadata"]["encoding"] == "utf-8"
        assert "sha256" in parsed_text["metadata"]

        parsed_md = engine.parse(md_file)
        assert parsed_md["content"].startswith("# Title")

        parsed_json = engine.parse(json_file)
        assert parsed_json["metadata"]["parser_metadata"]["valid_json"] is True
        assert "reader" in parsed_json["content"]

        parsed_csv = engine.parse(csv_file)
        assert parsed_csv["metadata"]["parser_metadata"]["row_count"] == 3

        parsed_html = engine.parse(html_file)
        assert "HTML body" in parsed_html["content"]

        parsed_xml = engine.parse(xml_file)
        assert "XML body" in parsed_xml["content"]

        parsed_many = engine.parse_many([text_file, json_file])
        assert len(parsed_many) == 2
        assert all(doc["status"] == "ok" for doc in parsed_many)

        try:
            engine.parse(tmp / "missing.txt")
        except FileMissingError:
            printer.status("TEST", "Missing file error handled", "success")
        else:
            raise AssertionError("Expected FileMissingError for missing input")

    print("\n=== Test ran successfully ===\n")
