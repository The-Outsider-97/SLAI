"""Production-grade Detection Engine for the Reader subsystem.

The detection engine answers one question before any parsing work starts: *what is
this file really, and what can Reader do with it?* The answer comes from the file
content (signatures, ZIP member names, text heuristics), cross-checked against the
extension, and never from the extension alone.

Design goals
------------
- Own the single format registry: format ids, families, MIME types and extensions.
  Whether a format can be *read* is not duplicated here; it is asked of
  ``ParserEngine.supported_extensions()`` so a disabled handler or a missing
  optional backend is never advertised.
- Read as little as possible: a bounded header plus, for ZIP files, the member
  names. Members are never decompressed (the one-line ``mimetype`` entry of
  OpenDocument/EPUB packages is the only exception) and the ZIP entry count is
  checked before the central directory is loaded.
- Report disagreement instead of hiding it: an extension that contradicts the
  content yields ``extension_mismatch`` and a warning, and the content wins.
- Report damage as a signal for the salvage stage: a ZIP signature without a
  readable central directory is ``damaged``, not "unknown".
- Reuse ``reader_helpers.py`` for path handling, binary detection, results and
  config validation; use typed Reader errors for failures.
"""

from __future__ import annotations

import gzip
import io
import json
import struct
import zipfile
import zlib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .utils.config_loader import get_config_section, load_global_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Detection Engine")
printer = PrettyPrinter()

DETECTION_ENGINE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormatSpec:
    """One known file format. Registry entries describe formats, not Reader support."""

    format_id: str
    family: str
    mime: str
    extensions: Tuple[str, ...]
    text: bool = False
    container: Optional[str] = None
    optimizable: bool = False
    salvageable: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_id": self.format_id,
            "family": self.family,
            "mime": self.mime,
            "extensions": list(self.extensions),
            "text": self.text,
            "container": self.container,
            "optimizable": self.optimizable,
            "salvageable": self.salvageable,
            "description": self.description,
        }


def _spec(format_id: str, family: str, mime: str, extensions: Sequence[str], *, text: bool = False,
          container: Optional[str] = None, optimizable: bool = False, salvageable: bool = False,
          description: str = "") -> FormatSpec:
    return FormatSpec(format_id, family, mime, tuple(extensions), text, container, optimizable, salvageable,
                      description or format_id)


_OOXML = "application/vnd.openxmlformats-officedocument"
_ODF = "application/vnd.oasis.opendocument"

_FORMATS: Tuple[FormatSpec, ...] = (
    # Text and structured text
    _spec("txt", "text", "text/plain", (".txt", ".text"), text=True, salvageable=True),
    _spec("md", "text", "text/markdown", (".md", ".markdown"), text=True, salvageable=True),
    _spec("rst", "text", "text/x-rst", (".rst",), text=True, salvageable=True),
    _spec("log", "text", "text/plain", (".log",), text=True, salvageable=True),
    _spec("csv", "structured", "text/csv", (".csv",), text=True, salvageable=True),
    _spec("tsv", "structured", "text/tab-separated-values", (".tsv",), text=True, salvageable=True),
    _spec("json", "structured", "application/json", (".json",), text=True, salvageable=True),
    _spec("jsonl", "structured", "application/x-ndjson", (".jsonl", ".ndjson"), text=True, salvageable=True),
    _spec("yaml", "structured", "application/yaml", (".yaml", ".yml"), text=True),
    _spec("toml", "structured", "application/toml", (".toml",), text=True),
    _spec("ini", "structured", "text/plain", (".ini", ".cfg"), text=True),
    _spec("xml", "structured", "application/xml", (".xml",), text=True, salvageable=True),
    _spec("html", "structured", "text/html", (".html", ".htm", ".xhtml"), text=True),
    # Word processing and publishing
    _spec("docx", "word", f"{_OOXML}.wordprocessingml.document", (".docx",), container="zip", optimizable=True, salvageable=True),
    _spec("doc", "word", "application/msword", (".doc",), container="ole"),
    _spec("odt", "word", f"{_ODF}.text", (".odt",), container="zip", optimizable=True, salvageable=True),
    _spec("rtf", "word", "application/rtf", (".rtf",), text=True),
    _spec("pages", "word", "application/vnd.apple.pages", (".pages",), container="zip"),
    _spec("tex", "word", "application/x-tex", (".tex",), text=True),
    # Fixed layout
    _spec("pdf", "fixed_layout", "application/pdf", (".pdf",), optimizable=True, salvageable=True),
    _spec("xps", "fixed_layout", "application/vnd.ms-xpsdocument", (".xps",), container="zip"),
    _spec("oxps", "fixed_layout", "application/oxps", (".oxps",), container="zip"),
    _spec("ps", "fixed_layout", "application/postscript", (".ps",), text=True),
    _spec("eps", "fixed_layout", "application/postscript", (".eps",), text=True),
    _spec("djvu", "fixed_layout", "image/vnd.djvu", (".djvu", ".djv")),
    # Spreadsheets
    _spec("xlsx", "spreadsheet", f"{_OOXML}.spreadsheetml.sheet", (".xlsx",), container="zip", optimizable=True, salvageable=True),
    _spec("xls", "spreadsheet", "application/vnd.ms-excel", (".xls",), container="ole"),
    _spec("xlsm", "spreadsheet", "application/vnd.ms-excel.sheet.macroEnabled.12", (".xlsm",), container="zip", optimizable=True, salvageable=True),
    _spec("xlsb", "spreadsheet", "application/vnd.ms-excel.sheet.binary.macroEnabled.12", (".xlsb",), container="zip"),
    _spec("ods", "spreadsheet", f"{_ODF}.spreadsheet", (".ods",), container="zip", optimizable=True, salvageable=True),
    _spec("numbers", "spreadsheet", "application/vnd.apple.numbers", (".numbers",), container="zip"),
    # Presentations
    _spec("pptx", "presentation", f"{_OOXML}.presentationml.presentation", (".pptx",), container="zip", optimizable=True, salvageable=True),
    _spec("ppt", "presentation", "application/vnd.ms-powerpoint", (".ppt",), container="ole"),
    _spec("odp", "presentation", f"{_ODF}.presentation", (".odp",), container="zip", optimizable=True, salvageable=True),
    _spec("key", "presentation", "application/vnd.apple.keynote", (".key",), container="zip"),
    # E-books
    _spec("epub", "ebook", "application/epub+zip", (".epub",), container="zip", optimizable=True, salvageable=True),
    _spec("mobi", "ebook", "application/x-mobipocket-ebook", (".mobi",)),
    _spec("azw", "ebook", "application/vnd.amazon.ebook", (".azw", ".azw3")),
    _spec("fb2", "ebook", "application/x-fictionbook+xml", (".fb2",), text=True),
    # Email
    _spec("eml", "email", "message/rfc822", (".eml",), text=True),
    _spec("msg", "email", "application/vnd.ms-outlook", (".msg",), container="ole"),
    _spec("mbox", "email", "application/mbox", (".mbox",), text=True),
    # Images
    _spec("png", "image", "image/png", (".png",)),
    _spec("jpeg", "image", "image/jpeg", (".jpg", ".jpeg")),
    _spec("tiff", "image", "image/tiff", (".tif", ".tiff")),
    _spec("bmp", "image", "image/bmp", (".bmp",)),
    _spec("webp", "image", "image/webp", (".webp",)),
    _spec("gif", "image", "image/gif", (".gif",)),
    _spec("heic", "image", "image/heic", (".heic", ".heif")),
    _spec("svg", "image", "image/svg+xml", (".svg",), text=True),
    # Archives and compressed streams
    _spec("zip", "archive", "application/zip", (".zip",), container="zip"),
    _spec("tar", "archive", "application/x-tar", (".tar",), container="tar"),
    _spec("gzip", "archive", "application/gzip", (".gz",), container="gzip"),
    _spec("tar_gz", "archive", "application/gzip", (".tar.gz", ".tgz"), container="gzip"),
    _spec("bzip2", "archive", "application/x-bzip2", (".bz2",), container="bzip2"),
    _spec("xz", "archive", "application/x-xz", (".xz",), container="xz"),
    _spec("7z", "archive", "application/x-7z-compressed", (".7z",)),
    _spec("rar", "archive", "application/vnd.rar", (".rar",)),
    # Specialized
    _spec("ipynb", "specialized", "application/x-ipynb+json", (".ipynb",), text=True),
    _spec("bib", "specialized", "application/x-bibtex", (".bib",), text=True),
    _spec("ris", "specialized", "application/x-research-info-systems", (".ris",), text=True),
    _spec("ics", "specialized", "text/calendar", (".ics",), text=True),
    _spec("vcf", "specialized", "text/vcard", (".vcf",), text=True),
    _spec("sqlite", "specialized", "application/vnd.sqlite3", (".sqlite", ".sqlite3", ".db")),
)

_BY_ID: Dict[str, FormatSpec] = {spec.format_id: spec for spec in _FORMATS}
_BY_EXTENSION: Dict[str, FormatSpec] = {ext: spec for spec in _FORMATS for ext in spec.extensions}

# Formats that share a structure and may legitimately appear under each other's extension.
_COMPATIBLE_GROUPS: Tuple[frozenset, ...] = (
    frozenset({"xlsx", "xlsm", "xlsb"}),
    frozenset({"xml", "html", "svg", "fb2"}),
    frozenset({"json", "jsonl", "ipynb"}),
    frozenset({"gzip", "tar_gz"}),
    frozenset({"ps", "eps"}),
    frozenset({"pages", "numbers", "key"}),
    frozenset({"xps", "oxps"}),
)

_COMPRESSED_STREAMS = {"gzip", "bzip2", "xz"}
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_ENCRYPTED_PACKAGE_MARKER = "EncryptedPackage".encode("utf-16-le")
_OOXML_IDS = {"docx", "xlsx", "xlsm", "xlsb", "pptx"}
_HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis"}

# (magic, format id) checked at offset 0 in order. Container hints come from the registry.
_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"\x1f\x8b", "gzip"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"Rar!\x1a\x07", "rar"),
    (b"SQLite format 3\x00", "sqlite"),
    (b"AT&TFORM", "djvu"),
    (b"{\\rtf", "rtf"),
    (b"BEGIN:VCALENDAR", "ics"),
    (b"BEGIN:VCARD", "vcf"),
)

_KNOWN_CONFIG_KEYS = ("sniff_bytes", "ole_scan_bytes", "max_zip_entries", "max_zip_members_listed")


@dataclass(frozen=True)
class DetectionResult:
    """What a file is, how sure Reader is, and what is wrong with it."""

    name: str
    extension: str
    size_bytes: int
    format_id: str
    family: str
    mime: str
    detected_by: str
    confidence: float
    extension_format: Optional[str] = None
    content_format: Optional[str] = None
    extension_mismatch: bool = False
    container: Optional[str] = None
    encrypted: Optional[bool] = None
    damaged: bool = False
    is_text: bool = False
    members: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    capabilities: Mapping[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "format_id": self.format_id,
            "family": self.family,
            "mime": self.mime,
            "detected_by": self.detected_by,
            "confidence": round(self.confidence, 3),
            "extension_format": self.extension_format,
            "content_format": self.content_format,
            "extension_mismatch": self.extension_mismatch,
            "container": self.container,
            "encrypted": self.encrypted,
            "damaged": self.damaged,
            "is_text": self.is_text,
            "members": list(self.members),
            "warnings": list(self.warnings),
            "capabilities": dict(self.capabilities),
        }


class DetectionEngine:
    """Identifies file formats from content and reports what Reader can do with them."""

    OPERATIONS = ("read", "parse", "convert", "merge", "compress", "optimize", "expand", "salvage")

    def __init__(self, parser: Optional[Any] = None) -> None:
        self.config = load_global_config()
        self.reader_config = get_config_section("reader")
        self.de_config = get_config_section("detection_engine") or {}
        warn_unknown_config_keys("detection_engine", self.de_config, _KNOWN_CONFIG_KEYS)

        self.sniff_bytes = coerce_int(self.de_config.get("sniff_bytes"), 65536, minimum=512, maximum=4 * 1024 * 1024)
        self.ole_scan_bytes = coerce_int(self.de_config.get("ole_scan_bytes"), 1024 * 1024, minimum=0)
        self.max_zip_entries = coerce_int(self.de_config.get("max_zip_entries"), 10_000, minimum=1)
        self.max_zip_members_listed = coerce_int(self.de_config.get("max_zip_members_listed"), 20, minimum=0)

        # The parser is the single source of truth for what is readable. It is created lazily so
        # detection stays usable (and cheap) on its own.
        self._parser = parser

        self.settings = {
            "sniff_bytes": self.sniff_bytes,
            "ole_scan_bytes": self.ole_scan_bytes,
            "max_zip_entries": self.max_zip_entries,
            "max_zip_members_listed": self.max_zip_members_listed,
        }
        logger.info("Detection Engine initialized | formats=%s sniff_bytes=%s", len(_FORMATS), self.sniff_bytes)

    # ------------------------------------------------------------------
    # Registry and capability queries
    # ------------------------------------------------------------------

    @staticmethod
    def spec_for(key: str | None) -> Optional[FormatSpec]:
        """Look up a format by id or extension (``docx``, ``.docx`` and ``.DOCX`` all work)."""

        text = str(key or "").strip().lower()
        if not text:
            return None
        if text in _BY_ID:
            return _BY_ID[text]
        return _BY_EXTENSION.get(normalize_extension(text, include_dot=True))

    def _readable_extensions(self) -> Set[str]:
        if self._parser is None:
            try:
                from .parser_engine import ParserEngine

                self._parser = ParserEngine()
            except Exception as exc:
                logger.warning("Detection could not load ParserEngine; reporting no readable formats: %s", exc)
                return set()
        try:
            return set(self._parser.supported_extensions())
        except Exception as exc:
            logger.warning("ParserEngine.supported_extensions failed: %s", exc)
            return set()

    def operations_for(self, spec: Optional[FormatSpec], *, readable: Optional[Set[str]] = None) -> Dict[str, bool]:
        """Operations Reader can perform on a format right now."""

        if spec is None:
            return {operation: False for operation in self.OPERATIONS}
        readable_set = self._readable_extensions() if readable is None else readable
        can_read = any(ext in readable_set for ext in spec.extensions)
        return {
            "read": can_read,
            "parse": can_read,
            "convert": can_read,
            "merge": can_read,
            # Packaging works on any file; whether it saves space is reported at run time.
            "compress": True,
            "optimize": spec.optimizable,
            "expand": spec.container in {"zip", "gzip", "bzip2", "xz", "tar"} or spec.format_id in {"zip", "tar", "gzip", "tar_gz", "bzip2", "xz"},
            "salvage": spec.salvageable,
        }

    def supports(self, key: str, operation: str) -> bool:
        """True when ``operation`` is available for the format ``key`` (id or extension)."""

        op = str(operation or "").strip().lower()
        if op not in self.OPERATIONS:
            raise ReaderValidationError("Unknown Reader operation", {"operation": operation, "supported": list(self.OPERATIONS)})
        return bool(self.operations_for(self.spec_for(key)).get(op, False))

    def registry(self) -> List[Dict[str, Any]]:
        readable = self._readable_extensions()
        return [{**spec.to_dict(), "operations": self.operations_for(spec, readable=readable)} for spec in _FORMATS]

    def capabilities(self) -> Dict[str, Any]:
        readable = sorted(self._readable_extensions())
        return {
            "engine_version": DETECTION_ENGINE_VERSION,
            "format_count": len(_FORMATS),
            "families": sorted({spec.family for spec in _FORMATS}),
            "readable_extensions": readable,
            "operations": list(self.OPERATIONS),
            "settings": dict(self.settings),
        }

    def describe(self) -> Dict[str, Any]:
        """Return a safe operational description useful for health checks."""

        return {"status": "ok", **self.capabilities()}

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _extension_of(name: str) -> str:
        lowered = str(name or "").lower()
        if lowered.endswith(".tar.gz"):
            return ".tar.gz"
        return normalize_extension(Path(lowered).suffix, include_dot=True)

    @staticmethod
    def _decode_head(head: bytes) -> Tuple[Optional[str], bool]:
        """Decode a header as text. Returns (text or None, is_wide_encoding)."""

        for bom, encoding in ((b"\xff\xfe\x00\x00", "utf-32"), (b"\x00\x00\xfe\xff", "utf-32"),
                              (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
            if head.startswith(bom):
                try:
                    return head.decode(encoding, errors="ignore"), True
                except UnicodeError:
                    return None, True
        if is_probably_binary(head, sample_size=len(head) or 1):
            return None, False
        decoder = io.TextIOWrapper(io.BytesIO(head), encoding="utf-8", errors="replace")
        try:
            return decoder.read(), False
        finally:
            decoder.detach()

    @staticmethod
    def _zip_entry_count(source: Path | bytes, size: int) -> Optional[int]:
        """Entry count from the end-of-central-directory record, without opening the archive."""

        try:
            if isinstance(source, (bytes, bytearray)):
                tail = bytes(source[-65557:])
            else:
                with source.open("rb") as handle:
                    handle.seek(max(0, size - 65557))
                    tail = handle.read(65557)
        except OSError:
            return None
        position = tail.rfind(b"PK\x05\x06")
        if position < 0 or len(tail) - position < 22:
            return None
        total = struct.unpack_from("<H", tail, position + 10)[0]
        if total == 0xFFFF:
            return 1 << 32  # ZIP64: the real count is elsewhere; treat as "very many"
        return int(total)

    def _inspect_zip(self, source: Path | bytes, size: int) -> Dict[str, Any]:
        """Read ZIP member names (never contents) and classify OOXML/ODF/EPUB/XPS packages."""

        info: Dict[str, Any] = {"opened": False, "names": [], "mimetype": None, "encrypted": False,
                                "format_id": None, "error": None, "too_many": False}
        count = self._zip_entry_count(source, size)
        if count is not None and count > self.max_zip_entries:
            info["too_many"] = True
            info["error"] = "zip_entry_count_exceeds_limit"
            return info
        try:
            handle = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else source
            with zipfile.ZipFile(handle) as archive:
                members = archive.infolist()
                names = [member.filename for member in members]
                info["opened"] = True
                info["names"] = names
                info["encrypted"] = any(member.flag_bits & 0x1 for member in members)
                if "mimetype" in names:
                    entry = archive.getinfo("mimetype")
                    if entry.file_size <= 256 and not entry.flag_bits & 0x1:
                        info["mimetype"] = archive.read("mimetype").decode("ascii", errors="ignore").strip()
        except (zipfile.BadZipFile, OSError, EOFError, ValueError, NotImplementedError, RuntimeError) as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            return info

        names_set = set(info["names"])
        mimetype = str(info["mimetype"] or "")
        if "[Content_Types].xml" in names_set:
            if any(name.startswith("word/") for name in names_set):
                info["format_id"] = "docx"
            elif "xl/workbook.bin" in names_set:
                info["format_id"] = "xlsb"
            elif any(name.startswith("xl/") for name in names_set):
                info["format_id"] = "xlsm" if "xl/vbaProject.bin" in names_set else "xlsx"
            elif any(name.startswith("ppt/") for name in names_set):
                info["format_id"] = "pptx"
            elif any(name.startswith("Documents/") for name in names_set) or "FixedDocumentSequence.fdseq" in names_set:
                info["format_id"] = "oxps"
        elif "FixedDocumentSequence.fdseq" in names_set:
            info["format_id"] = "xps"
        elif mimetype == "application/epub+zip":
            info["format_id"] = "epub"
        elif mimetype.startswith(_ODF):
            info["format_id"] = {"text": "odt", "spreadsheet": "ods", "presentation": "odp"}.get(mimetype.rsplit(".", 1)[-1])
        elif "Index/Document.iwa" in names_set or "Index.zip" in names_set:
            info["format_id"] = "iwork"
        return info

    def _ole_encrypted(self, source: Path | bytes, size: int) -> Optional[bool]:
        """Encrypted Office packages are OLE files containing an ``EncryptedPackage`` stream."""

        if self.ole_scan_bytes <= 0:
            return None
        try:
            if isinstance(source, (bytes, bytearray)):
                window = bytes(source[: self.ole_scan_bytes])
            else:
                window = read_file_head(source, self.ole_scan_bytes)
        except ReaderError:
            return None
        if _ENCRYPTED_PACKAGE_MARKER in window:
            return True
        # Only a complete scan can rule encryption out.
        return False if size <= self.ole_scan_bytes else None

    def _signature_format(self, head: bytes) -> Tuple[Optional[str], Optional[str]]:
        """Return (format id or generic container, container) from the header, if any signature matches."""

        if head[:4] in _ZIP_SIGNATURES:
            return "zip", "zip"
        if head.startswith(_OLE_SIGNATURE):
            return "ole", "ole"
        if b"%PDF-" in head[:1024]:
            return "pdf", None
        if head.startswith(b"BZh") and head[3:4] in b"123456789" and len(head) > 3:
            return "bzip2", "bzip2"
        if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            return "webp", None
        if head[4:8] == b"ftyp" and head[8:12] in _HEIC_BRANDS:
            return "heic", None
        if head.startswith(b"BM") and len(head) >= 14 and head[6:10] == b"\x00\x00\x00\x00":
            return "bmp", None
        if head[257:262] == b"ustar":
            return "tar", "tar"
        if head.startswith(b"%!PS"):
            return ("eps" if b"EPSF" in head[:100] else "ps"), None
        for magic, format_id in _SIGNATURES:
            if head.startswith(magic):
                spec = _BY_ID.get(format_id)
                return format_id, (spec.container if spec else None)
        return None, None

    @staticmethod
    def _classify_text(text: str) -> Optional[str]:
        """Content type for text headers when there is a strong textual signature.

        ``None`` means "generic text": JSON, CSV, Markdown and friends are not guessed from
        content because plain prose can start with ``[`` or ``<`` too; the extension decides.
        """

        stripped = text.lstrip("\ufeff \t\r\n")
        lowered = stripped[:2048].lower()
        if lowered.startswith("<"):
            if "<svg" in lowered:
                return "svg"
            if lowered.startswith("<!doctype html") or "<html" in lowered:
                return "html"
            if "<fictionbook" in lowered:
                return "fb2"
            return "xml"
        if stripped.startswith("{") and '"nbformat"' in text[:65536] and '"cells"' in text[:65536]:
            return "ipynb"
        head_lines = stripped[:1024]
        if head_lines.startswith("From ") and "\n" in head_lines:
            return "mbox"
        header_names = ("received:", "from:", "subject:", "mime-version:", "message-id:", "date:", "to:")
        first_lines = [line.lower() for line in head_lines.splitlines()[:8]]
        if sum(1 for line in first_lines if line.startswith(header_names)) >= 3 and any(line.startswith(("subject:", "from:")) for line in first_lines):
            return "eml"
        return None

    @staticmethod
    def _compatible(content_id: str, extension_id: str) -> bool:
        if content_id == extension_id:
            return True
        return any(content_id in group and extension_id in group for group in _COMPATIBLE_GROUPS)

    def _detect_content(self, source: Path | bytes, head: bytes, size: int, extension: str) -> Dict[str, Any]:
        """Content-only identification. Returns a mutable analysis record."""

        analysis: Dict[str, Any] = {
            "content_id": None, "container": None, "encrypted": None, "damaged": False,
            "is_text": False, "members": [], "warnings": [], "detected_by": "none",
        }
        warnings: List[str] = analysis["warnings"]
        extension_spec = _BY_EXTENSION.get(extension)

        signature_id, container = self._signature_format(head)
        if signature_id == "zip":
            zip_info = self._inspect_zip(source, size)
            analysis["container"] = "zip"
            analysis["members"] = zip_info["names"][: self.max_zip_members_listed]
            if zip_info["encrypted"]:
                analysis["encrypted"] = True
                warnings.append("zip_members_encrypted")
            if zip_info["too_many"]:
                warnings.append("zip_entry_count_exceeds_limit")
                analysis["content_id"], analysis["detected_by"] = "zip", "signature"
            elif not zip_info["opened"]:
                analysis["damaged"] = True
                warnings.append("damaged_container")
                # A ZIP signature we cannot read: trust the extension only if it names a ZIP-based format.
                if extension_spec is not None and extension_spec.container == "zip":
                    analysis["content_id"], analysis["detected_by"] = extension_spec.format_id, "extension"
                else:
                    analysis["content_id"], analysis["detected_by"] = "zip", "signature"
            elif zip_info["format_id"] == "iwork":
                analysis["content_id"] = extension_spec.format_id if extension_spec and extension_spec.format_id in {"pages", "numbers", "key"} else "zip"
                analysis["detected_by"] = "container"
            elif zip_info["format_id"]:
                analysis["content_id"], analysis["detected_by"] = zip_info["format_id"], "container"
            else:
                analysis["content_id"], analysis["detected_by"] = "zip", "signature"
                if extension_spec is not None and extension_spec.container == "zip" and extension_spec.format_id != "zip":
                    warnings.append("zip_without_expected_package_structure")
            return analysis

        if signature_id == "ole":
            analysis["container"] = "ole"
            analysis["content_id"], analysis["detected_by"] = "ole", "signature"
            if extension_spec is not None and extension_spec.format_id in _OOXML_IDS:
                # An OOXML name on an OLE file is either an encrypted Office package or a legacy file
                # that was renamed; only the EncryptedPackage stream tells them apart.
                analysis["encrypted"] = self._ole_encrypted(source, size)
                if analysis["encrypted"] is True:
                    warnings.append("encrypted_ooxml_package")
                elif analysis["encrypted"] is None:
                    warnings.append("ole_wrapped_ooxml_possibly_encrypted")
                else:
                    warnings.append("ole_content_with_ooxml_extension")
            return analysis

        if signature_id == "gzip":
            analysis["container"] = "gzip"
            analysis["content_id"], analysis["detected_by"] = "gzip", "signature"
            try:
                inner = gzip.GzipFile(fileobj=io.BytesIO(head)).read(512)
            except (OSError, EOFError, zlib.error):
                inner = b""
            if inner[257:262] == b"ustar":
                analysis["content_id"] = "tar_gz"
            return analysis

        if signature_id is not None:
            spec = _BY_ID.get(signature_id)
            analysis["content_id"], analysis["detected_by"] = signature_id, "signature"
            analysis["container"] = container or (spec.container if spec else None)
            if spec is not None and spec.text:
                text, wide = self._decode_head(head)
                analysis["is_text"] = text is not None
            return analysis

        text, wide = self._decode_head(head)
        if text is not None:
            analysis["is_text"] = True
            classified = self._classify_text(text)
            analysis["content_id"] = classified
            analysis["detected_by"] = "text" if classified else "none"
            if wide:
                warnings.append("wide_text_encoding")
            if extension_spec is not None and extension_spec.format_id == "json" and size <= len(head):
                try:
                    json.loads(text)
                except ValueError:
                    warnings.append("content_not_valid_json")
            if extension_spec is not None and extension_spec.format_id == "xml" and classified is None and text.strip():
                warnings.append("content_not_xml")
            return analysis

        # Not text, no known signature.
        analysis["detected_by"] = "none"
        return analysis

    def _detect(self, source: Path | bytes, head: bytes, size: int, name: str) -> DetectionResult:
        extension = self._extension_of(name)
        extension_spec = _BY_EXTENSION.get(extension)
        analysis = self._detect_content(source, head, size, extension)
        warnings: List[str] = list(analysis["warnings"])
        content_id: Optional[str] = analysis["content_id"]
        content_spec = _BY_ID.get(content_id) if content_id else None
        mismatch = False
        confidence = 0.4
        detected_by = analysis["detected_by"]
        chosen: Optional[FormatSpec] = None

        if content_id == "ole":
            # OLE alone cannot name the application; the extension decides among OLE formats.
            if extension_spec is not None and (extension_spec.container == "ole" or extension_spec.format_id in _OOXML_IDS):
                chosen, confidence, detected_by = extension_spec, 0.8, "signature+extension"
                # An OOXML name on an OLE file is only expected when it is an encrypted package.
                mismatch = extension_spec.format_id in _OOXML_IDS and analysis["encrypted"] is not True
            else:
                chosen, confidence, detected_by = None, 0.5, "signature"
                mismatch = extension_spec is not None
                if mismatch:
                    warnings.append("extension_does_not_match_ole_content")
        elif content_id is not None and detected_by == "text":
            # A textual signature (markup, notebook, mail) is only a hint when the extension is
            # itself a text format: HTML inside a .txt or .md file is not a mismatch.
            if extension_spec is None:
                chosen, confidence = content_spec, 0.8
            elif extension_spec.text:
                chosen = extension_spec
                confidence = 0.9 if self._compatible(content_id, extension_spec.format_id) else 0.7
                if not self._compatible(content_id, extension_spec.format_id):
                    warnings.append(f"content_resembles:{content_id}")
            else:
                mismatch = True
                chosen, confidence = content_spec, 0.8
                warnings.append(f"extension_content_mismatch:{extension or 'none'}->{content_id}")
        elif content_id is not None and content_spec is not None:
            if extension_spec is None:
                chosen, confidence = content_spec, 0.9
            elif self._compatible(content_id, extension_spec.format_id):
                # Same format, or the extension names a compatible variant (xlsm for xlsx).
                chosen = extension_spec if extension_spec.format_id != content_id else content_spec
                confidence = 0.5 if detected_by == "extension" else 0.99
            else:
                mismatch = True
                chosen, confidence = content_spec, 0.9
                warnings.append(f"extension_content_mismatch:{extension or 'none'}->{content_id}")
        elif analysis["is_text"]:
            if extension_spec is not None and extension_spec.text:
                chosen, confidence, detected_by = extension_spec, 0.7, "text+extension"
            elif extension_spec is not None:
                mismatch = True
                chosen, confidence, detected_by = _BY_ID["txt"], 0.6, "text"
                warnings.append(f"text_content_with_binary_extension:{extension}")
            else:
                chosen, confidence, detected_by = _BY_ID["txt"], 0.6, "text"
        else:
            # Unknown binary content.
            if extension_spec is not None and extension_spec.text:
                mismatch = True
                warnings.append("binary_content_with_text_extension")
            elif extension_spec is not None:
                chosen, confidence, detected_by = extension_spec, 0.4, "extension"
                warnings.append("unverified_extension_only")

        # A damaged package keeps the extension's identity but is flagged for salvage.
        if analysis["damaged"] and chosen is None and extension_spec is not None:
            chosen = extension_spec

        if analysis["damaged"]:
            confidence = min(confidence, 0.5)

        format_id = chosen.format_id if chosen is not None else "unknown"
        return DetectionResult(
            name=name,
            extension=extension,
            size_bytes=size,
            format_id=format_id,
            family=chosen.family if chosen is not None else "unknown",
            mime=chosen.mime if chosen is not None else "application/octet-stream",
            detected_by=detected_by,
            confidence=confidence,
            extension_format=extension_spec.format_id if extension_spec is not None else None,
            content_format=content_id,
            extension_mismatch=mismatch,
            container=analysis["container"] or (chosen.container if chosen is not None else None),
            encrypted=analysis["encrypted"],
            damaged=bool(analysis["damaged"]),
            is_text=bool(analysis["is_text"]),
            members=tuple(analysis["members"]),
            warnings=tuple(dedupe_preserve_order(warnings)),
        )

    def detect(self, source: str | Path | bytes | bytearray, *, name: Optional[str] = None) -> Dict[str, Any]:
        """Identify one file (path) or buffer (bytes, with an optional ``name`` for the extension)."""

        timer = OperationTimer("detect")
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
            display_name = str(name or "")
            result = self._detect(data, data[: self.sniff_bytes], len(data), display_name)
        else:
            path = expand_path(source)
            if not path.exists():
                raise FileMissingError(str(path))
            if not path.is_file():
                raise FileNotRegularError(str(path))
            try:
                size = int(path.stat().st_size)
            except PermissionError as exc:
                raise FileAccessDeniedError(str(path), operation="stat", cause=exc) from exc
            except OSError as exc:
                raise ParseFailureError(str(path), f"Failed reading file metadata: {exc}", cause=exc) from exc
            head = read_file_head(path, self.sniff_bytes)
            result = self._detect(path, head, size, str(name or path.name))

        readable = self._readable_extensions()
        spec = _BY_ID.get(result.format_id)
        operations = self.operations_for(spec, readable=readable)
        result = DetectionResult(**{**result.__dict__, "capabilities": operations})
        for warning in result.warnings:
            logger.debug("Detection warning for %s: %s", result.name or "<buffer>", warning)
        return build_success_result(
            operation="detect",
            payload=result.to_dict(),
            metadata={"engine_version": DETECTION_ENGINE_VERSION},
            warnings=result.warnings,
            timer=timer,
        )

    def detect_many(self, sources: Iterable[str | Path]) -> List[Dict[str, Any]]:
        """Detect several files sequentially; failures become per-item error results."""

        results: List[Dict[str, Any]] = []
        for index, source in enumerate(sources):
            try:
                results.append(self.detect(source))
            except Exception as exc:
                error = normalize_reader_error(
                    exc,
                    message="Reader format detection failed",
                    context={"index": index, "source": str(source)},
                    stage=ReaderErrorStage.DETECTION,
                )
                results.append(build_error_result(error, operation="detect"))
        return results


if __name__ == "__main__":
    print("\n=== Running Detection Engine ===\n")
    printer.status("TEST", "Detection Engine initialized", "info")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="reader_detection_test_") as tmp_dir:
        tmp = Path(tmp_dir)
        engine = DetectionEngine()

        text_file = tmp / "note.txt"
        text_file.write_text("plain text\n", encoding="utf-8")
        found = engine.detect(text_file)["payload"]
        assert found["format_id"] == "txt" and not found["extension_mismatch"]

        fake_text = tmp / "image.txt"
        fake_text.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        found = engine.detect(fake_text)["payload"]
        assert found["format_id"] == "png" and found["extension_mismatch"]
        printer.status("TEST", "Extension/content mismatch detected", "success")

        broken_zip = tmp / "broken.docx"
        broken_zip.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
        found = engine.detect(broken_zip)["payload"]
        assert found["damaged"] and found["format_id"] == "docx"
        printer.status("TEST", "Damaged container flagged", "success")

        try:
            engine.detect(tmp / "missing.txt")
        except FileMissingError:
            printer.status("TEST", "Missing file error handled", "success")
        else:
            raise AssertionError("Expected FileMissingError for missing input")

    print("\n=== Test ran successfully ===\n")
